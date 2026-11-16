# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text autoregressive stack using the reviewed Ornith TP4 decoder unchanged."""

import json
import math
import os
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipCollectives, MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import layer_policy, load_policy

HF_MODEL = "ornith-ai/Ornith-1.5-9B"
HF_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


class Checkpoint:
    """Read only the requested tensors; never materialize the vision model."""

    def __init__(self, path=None):
        from huggingface_hub import snapshot_download

        # ORNITH_CHECKPOINT (opt-in) points at a local checkpoint directory, e.g. the weights
        # directory tt-inference-server exports through the model spec's model_path_env.
        path = path or os.environ.get("ORNITH_CHECKPOINT") or None
        self.path = Path(path or snapshot_download(HF_MODEL, revision=HF_REVISION, local_files_only=True))
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensors(self, prefix):
        from safetensors import safe_open

        result = {}
        for shard in sorted({v for k, v in self.index.items() if k.startswith(prefix)}):
            with safe_open(self.path / shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118 -- safetensors exposes keys(), not iteration
                    if name.startswith(prefix):
                        result[name[len(prefix) :]] = handle.get_tensor(name)
        return result

    def tensor(self, key):
        from safetensors import safe_open

        with safe_open(self.path / self.index[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


class OrnithModel:
    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=None,
        layer_indices=None,
        decode_head="dram",
        selective_inactive_restore=True,
        precision_config=None,
    ):
        from transformers import AutoConfig

        self.device = mesh_device
        self.precision_policy = load_policy(precision_config)
        policy = self.precision_policy
        if decode_head not in ("dram", "interleaved"):
            raise ValueError("decode_head must be dram or interleaved")
        self.decode_head_kind = decode_head
        if type(selective_inactive_restore) is not bool:
            raise TypeError("selective_inactive_restore must be boolean")
        self.selective_inactive_restore = selective_inactive_restore
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("The reviewed implementation requires a 1x4 mesh")
        self.checkpoint = Checkpoint(checkpoint)
        self.hf_config = AutoConfig.from_pretrained(self.checkpoint.path)
        self.config = self.hf_config.text_config
        if self.config.max_position_embeddings != policy["max_context"]:
            raise ValueError("Precision policy context must match the checkpoint capability")
        self.layer_indices = (
            list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        self.collectives = MultichipCollectives(mesh_device, dtype=getattr(ttnn, policy["ccl_dtype"]))
        self.layers = []
        for index in self.layer_indices:
            print(f"LOAD_LAYER {index}", flush=True)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    self.checkpoint.tensors(f"model.language_model.layers.{index}."),
                    hf_config=self.config,
                    layer_idx=index,
                    mesh_device=mesh_device,
                    collectives=self.collectives,
                    precision_policy=layer_policy(policy, index),
                )
            )
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy["compute_fidelities"]["lm_head"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy["terminal_norm_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        embedding = self.checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.embedding = self.upload(
            embedding, dim=1, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=getattr(ttnn, policy["embedding_dtype"])
        )
        head = embedding if self.config.tie_word_embeddings else self.checkpoint.tensor("lm_head.weight")
        self.head = self.upload(head.T, dim=1, dtype=getattr(ttnn, policy["weight_groups"]["lm_head"]))
        self.decode_head = None
        if decode_head == "dram":
            from tt_transformers.modules.lazy_weight import LazyWeight
            from tt_transformers.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig

            # Preserve the selected precision and vocab ownership. A separate
            # interleaved copy serves arbitrary-height prefill; this decode-only
            # copy is budgeted in the full-context memory evidence.
            banks = mesh_device.dram_grid_size().x
            width = self.config.vocab_size // 4
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (4096, math.ceil(width / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            weight = LazyWeight(
                source=head.T,
                device=mesh_device,
                dtype=self.head.dtype,
                _value=ttnn.to_memory_config(self.head, memory),
            )
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=2,
                per_core_M=1,
                per_core_N=math.ceil(width / (64 * 32)),
                num_workers_per_dram_bank=3,
                use_multicore_pipeline=True,
            )
            self.decode_head = LMHead1D.from_config(
                LMHead1DConfig(
                    output_weights=[weight],
                    mesh_device=mesh_device,
                    program_configs=[program],
                    compute_kernel_config=self.compute,
                    # This field controls output dtype; LazyWeight retains BF8.
                    lm_head_dtype=getattr(ttnn, policy["logits_dtype"]),
                    input_memcfg=self.collectives.ar_mem,
                    output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                    weights_memcfgs=[memory],
                )
            )
            self.decode_head.load_device_weights()
        # Qwen3.5 RMSNorm stores a delta from 1, including the terminal norm.
        norm = self.checkpoint.tensor("model.language_model.norm.weight").float() + 1
        self.norm = self.upload(norm.reshape(1, 1, 1, -1))
        self.vocab_size = self.config.vocab_size
        self.memory_after_weights = self.memory_view()
        print("MODEL_LOADED", self.memory_after_weights, flush=True)
        print("PRECISION_POLICY", json.dumps(self.precision_summary()), flush=True)

    def precision_summary(self):
        """Read loaded tensors and compute objects, not just the requested JSON."""
        return {
            "requested": self.precision_policy,
            "head_dtype": str(self.head.dtype),
            "head_fidelity": str(self.compute.math_fidelity),
            "embedding_dtype": str(self.embedding.dtype),
            "ccl_buffer_dtypes": [str(t.dtype) for t in self.collectives.ar_buffers],
            "layers": [
                {
                    "index": index,
                    "weights": {name: str(t.dtype) for name, t in layer.dram_weights.items()},
                    "prefill_weights": {name: str(t.dtype) for name, t in layer.weights.items()},
                    "compute": {name: str(layer.matmul_compute(name).math_fidelity) for name in layer.dram_weights},
                    "kv_dtype": str(layer.kv_dtype),
                    "decode_activation_dtype": str(layer.decode_activation_dtype),
                    "prefill_activation_dtype": str(layer.prefill_activation_dtype),
                    "prefill_l1_row_limit": layer.prefill_l1_limit,
                }
                for index, layer in zip(self.layer_indices, self.layers)
            ],
        }

    def upload(self, value, *, dim=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=True):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.device if device else None,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.device) if dim is None else ttnn.ShardTensorToMesh(self.device, dim)
            ),
        )

    def memory_view(self):
        view = ttnn.get_memory_view(self.device, ttnn.BufferType.DRAM)
        return {
            name: getattr(view, name)
            for name in dir(view)
            if not name.startswith("_") and isinstance(getattr(view, name), (int, float))
        }

    def allocate_cache(self, *, batch, context, global_pages=False, num_blocks=None):
        context = math.ceil(context / 32) * 32
        return [
            layer.allocate_batch_state(batch=batch, context=context, global_pages=global_pages, num_blocks=num_blocks)
            for layer in self.layers
        ]

    def embed(self, tokens, *, batch=None):
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, tokens.shape[-1], 1024))
        x = self.layers[0].gather(x)
        if batch is not None:
            x = x[:, :, :batch, :]
            x = ttnn.to_memory_config(x, self.collectives.ar_mem)
        return x

    def logits(self, hidden, *, decode=False):
        if decode:
            hidden = ttnn.rms_norm(
                hidden,
                weight=self.norm,
                epsilon=self.config.rms_norm_eps,
                program_config=self.layers[0].norm_program,
                compute_kernel_config=self.norm_compute,
                memory_config=self.collectives.ar_mem,
            )
            if self.decode_head is not None:
                return self.decode_head(hidden)
            hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden = ttnn.rms_norm(
                hidden, weight=self.norm, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.norm_compute
            )
        return ttnn.linear(hidden, self.head, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def prefill(self, tokens, *, cache, slot, start, page_table, last_only=False):
        hidden = self.embed(tokens)
        for layer, states in zip(self.layers, cache):
            hidden = layer.prefill_forward(hidden, state=states[slot], start_position=start, page_table=page_table)
        if last_only:
            hidden = hidden[:, :, -1:, :]
        return self.logits(hidden)

    def decode(self, tokens, *, cache, slots, positions, page_tables, batch_state=None, masked=False):
        """Decode using stable slots; masked traces preserve inactive pooled state.

        Allocate/warm BatchState before capture. A masked trace uses every slot
        in the complete pool, signed -1 positions for inactive rows, and its
        persistent active masks. Refresh masks only at scheduler transitions.
        All-active calls retain the original decoder operation sequence.
        """
        if masked:
            if batch_state is None or batch_state.model is not self or batch_state.cache is not cache:
                raise ValueError("Masked decode requires BatchState bound to this model and exact cache")
            if tuple(slots) != tuple(range(batch_state.capacity)):
                raise ValueError("Masked native decode must retain every fixed slot in pool order")
            if len(positions) != batch_state.capacity or len(page_tables) != batch_state.capacity:
                raise ValueError("Masked positions and page tables must cover the complete pool")
        hidden = self.embed(tokens, batch=len(slots))
        for index, (layer, states) in enumerate(zip(self.layers, cache)):
            if masked:
                batch_state.before_layer(index)
                hidden = layer.decode_batch_packed(
                    hidden,
                    states=states,
                    current_positions=positions,
                    page_tables=page_tables,
                    masked=True,
                )
                batch_state.after_layer(index)
            else:
                hidden = layer.decode_batch_packed(
                    hidden,
                    states=[states[i] for i in slots],
                    current_positions=positions,
                    page_tables=page_tables,
                )
        return self.logits(hidden, decode=True)

    def host_logits(self, logits):
        """Explicit diagnostic/host-sampling boundary; never called by token feedback."""
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=3)).float()
