# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full text model over the accepted TP4 decoder, with device-only forwards."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn

from .multichip_decoder import MultichipDecoder
from .precision import group_for_weight, layer_policy, load_precision

MODEL_ID = "Qwen/Qwen3.6-27B"
REVISION = "8f33b98937c6031bb49e713dee3e8e185aafd26f"


class Checkpoint:
    def __init__(self, path=None):
        self.path = Path(path or snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
        self.mapping = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = AutoConfig.from_pretrained(self.path).text_config

    def tensor(self, name):
        with safe_open(self.path / self.mapping[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def layer(self, index):
        prefix = f"model.language_model.layers.{index}."
        return {name[len(prefix) :]: self.tensor(name) for name in self.mapping if name.startswith(prefix)}


class Model:
    """Weights stay resident; residual shards cross every layer unchanged.

    Reduced layer_indices are diagnostic only. Defaults load all64 real layers.
    The only hidden gather added by this wrapper is at final normalization/head.
    """

    def __init__(self, mesh_device, *, checkpoint=None, layer_indices=None, precision_config=None):
        self.mesh = mesh_device
        self.precision = load_precision(precision_config)
        self.precision_observations = {}
        if tuple(self.mesh.shape) != (1, 4):
            raise ValueError("Full model requires the accepted 1x4 TP mesh")
        source = Checkpoint(checkpoint)
        self.checkpoint = source.path
        self.config = source.config
        if self.precision["supported_context"] != self.config.max_position_embeddings:
            raise ValueError("Selected precision and pinned HF context contracts differ")
        self.layer_indices = (
            list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        self.layers = []
        for i in self.layer_indices:
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    source.layer(i),
                    hf_config=self.config,
                    layer_idx=i,
                    mesh_device=self.mesh,
                    policy=layer_policy(self.precision, self.config.layer_types[i], i),
                )
            )
            print(f"MODEL_LOAD layer={i} count={len(self.layers)}", flush=True)
        self.io = self.layers[0]
        embedding = source.tensor("model.language_model.embed_tokens.weight")
        self.embedding = ttnn.from_torch(
            embedding,
            dtype=getattr(ttnn, self.precision["embedding_dtype"]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )
        head = embedding if self.config.tie_word_embeddings else source.tensor("lm_head.weight")
        self.lm_head = ttnn.from_torch(
            head.T.contiguous(),
            dtype=getattr(ttnn, self.precision["lm_head_dtype"]),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )
        final_norm = (1 + source.tensor("model.language_model.norm.weight").float()).reshape(1, 1, -1)
        self.io.weights["final_norm"] = self.io._upload(final_norm, getattr(ttnn, self.precision["norm_weight_dtype"]))
        self.io.shard_norms["final_norm"] = self.io._upload_shards(
            list(final_norm.chunk(4, -1)), getattr(ttnn, self.precision["norm_weight_dtype"])
        )
        # Shared tables, one copy per rank for the entire stack (text positions).
        rotary = Qwen3_5TextRotaryEmbedding(self.config)
        with torch.no_grad():
            cos, sin = rotary(
                torch.empty(1, 1, 1, dtype=torch.bfloat16), torch.arange(self.config.max_position_embeddings)[None]
            )
        self.rope_tables = tuple(self.io._upload(t[0], layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))
        self.loaded_memory = self.memory_view()
        self.head_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["lm_head_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        print("MODEL_MEMORY " + json.dumps(self.loaded_memory), flush=True)
        print("MODEL_PRECISION " + json.dumps(self.precision_summary()), flush=True)

    def precision_summary(self):
        layers = []
        for layer in self.layers:
            weights = {}
            for name, tensor in layer.weights.items():
                if len(tensor.shape) != 2:
                    continue
                group = group_for_weight(name)
                weights[name] = dict(dtype=str(tensor.dtype), fidelity=str(layer.matmul_configs[group].math_fidelity))
            layers.append(
                dict(
                    layer=layer.layer_idx,
                    kind=layer.kind,
                    weights=weights,
                    norm_dtypes={name: str(tensor.dtype) for name, tensor in layer.weights.items() if "norm" in name},
                    swiglu_dtype=str(layer.swiglu_weight.dtype),
                    swiglu_fidelity=str(layer.matmul_configs["mlp_gate_up"].math_fidelity),
                    policy=layer_policy(self.precision, layer.kind, layer.layer_idx),
                    boundary_observations=dict(layer.precision_observations),
                )
            )
        return dict(
            config=self.precision,
            runtime_contract_version=1,
            boundary_observations=dict(self.precision_observations),
            layers=layers,
            embedding_dtype=str(self.embedding.dtype),
            lm_head_dtype=str(self.lm_head.dtype),
            lm_head_fidelity=str(self.head_config.math_fidelity),
        )

    def memory_view(self):
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        fields = (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "total_bytes_free_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
        return {name: int(getattr(view, name)) for name in fields}

    def create_cache(self, page_table):
        return [
            layer.create_recurrent_state(page_table.shape[0])
            if layer.kind == "linear_attention"
            else layer.create_paged_state(page_table)
            for layer in self.layers
        ]

    def allocate_prefill(self, cache, page_table, *, prefill_batch_size=1):
        """Allocate bounded prefill state before any generator trace capture.

        The returned scheduler owns preparation and yields device logits chunks;
        the caller owns host diagnostic readback and logical output assembly.
        """
        from .bounded_prefill import BoundedPrefill

        return BoundedPrefill(self, cache, page_table, prefill_batch_size=prefill_batch_size)

    def _observe_precision(self, label, tensor, field):
        # Metadata only: no readback, allocation, cast, or device operation.
        expected = getattr(ttnn, self.precision[field])
        if tensor.dtype != expected:
            raise ValueError(f"{label}: {tensor.dtype} violates {field}={expected}")
        self.precision_observations.setdefault(
            label, dict(field=field, dtype=str(tensor.dtype), shape=list(tensor.shape), layout=str(tensor.layout))
        )
        return tensor

    def embeddings(self, tokens):
        value = ttnn.embedding(
            tokens,
            self.embedding,
            layout=ttnn.TILE_LAYOUT,
            dtype=getattr(ttnn, self.precision["activation_dtype"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._observe_precision("embedding_output", value, "activation_dtype")

    def rotations(self, positions, *, batch, length):
        return tuple(
            ttnn.reshape(
                ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                (batch, 1, length, 64),
            )
            for table in self.rope_tables
        )

    def terminal(self, hidden, *, decode=False):
        self._observe_precision("terminal_residual", hidden, "residual_dtype")
        normalized = self.io._norm(hidden, "final_norm", decode=decode)
        self._observe_precision("terminal_normalized", normalized, "activation_dtype")
        output = ttnn.matmul(
            normalized,
            self.lm_head,
            dtype=getattr(ttnn, self.precision["logits_dtype"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_config,
        )
        return self._observe_precision("terminal_logits", output, "logits_dtype")

    def prefill(self, tokens, positions, *, cache, start_pos=0, return_all_logits=True):
        batch, length = tokens.shape
        hidden = self.embeddings(tokens)
        rope = self.rotations(positions, batch=batch, length=length)
        for layer, state in zip(self.layers, cache):
            hidden = layer.prefill_forward(
                hidden, state=state, start_pos=start_pos, rope=rope if layer.kind == "full_attention" else None
            )
        return self.terminal(hidden if return_all_logits else hidden[:, -1:, :])

    def decode(self, tokens, positions, *, cache, batch, rope_positions=None, active_rows=None):
        # Sampling's stable [1,1,1,32] buffer is the sole token source.
        token_ids = ttnn.reshape(tokens, (1, 32))[:, :batch]
        hidden = self.embeddings(token_ids)
        ids = ttnn.reshape(
            ttnn.typecast(positions, ttnn.uint32) if rope_positions is None else rope_positions, (batch, 1)
        )
        rope = self.rotations(ids, batch=batch, length=1)
        for layer, state in zip(self.layers, cache):
            saved = None
            if active_rows is not None and layer.kind == "linear_attention":
                saved = (ttnn.clone(state.conv), ttnn.clone(state.recurrent))
            hidden = layer.decode_forward(
                hidden,
                state=state,
                current_pos=positions,
                rope=rope if layer.kind == "full_attention" else None,
                logical_batch=batch,
            )
            if saved is not None:
                # Attention's paged update skips position -1. GDN has no such
                # argument, so publish only active rows of its in-place state.
                for old, updated in zip(saved, (state.conv, state.recurrent)):
                    shape = (batch,) + (1,) * (len(updated.shape) - 1)
                    mask = ttnn.to_layout(ttnn.reshape(active_rows, shape), ttnn.TILE_LAYOUT)
                    merged = ttnn.where(
                        mask, ttnn.to_layout(updated, ttnn.TILE_LAYOUT), ttnn.to_layout(old, ttnn.TILE_LAYOUT)
                    )
                    ttnn.copy(ttnn.to_layout(merged, updated.layout), updated)
        logits = self.terminal(hidden, decode=True)
        logits = ttnn.reshape(logits, (1, 1, batch, self.config.vocab_size // 4))
        if batch < 32:
            logits = ttnn.pad(logits, ((0, 0), (0, 0), (0, 32 - batch), (0, 0)), value=0)
        return logits
