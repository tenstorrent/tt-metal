# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer text autoregressive model on the optimized QB2 TP4 decoder.

All checkpoint conversion is construction-time work. Runtime forwards consume
device tensors and explicit caller-owned caches. The inter-layer residual is
the unchanged optimized decoder contract; terminal tensors are vocabulary TP4.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig

import ttnn
from models.demos.muse_glimmer_30b_qb2.tt.multichip_decoder import MeshCCLManager, MultichipDecoder
from models.demos.muse_glimmer_30b_qb2.tt.precision import decoder_policy, load_precision_config

MODEL_ID = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
HF_SNAPSHOT = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--meta-models--Muse-Glimmer-30B/snapshots"
    / REVISION
)


def _mounted_snapshot():
    """Serving harnesses (TTI) mount the checkpoint and export HF_MODEL /
    MODEL_WEIGHTS_DIR as a local snapshot directory. Prefer it when it holds
    the safetensors index and config, so the container needs no hub access."""
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if (
            value
            and (Path(value) / "model.safetensors.index.json").is_file()
            and (Path(value) / "config.json").is_file()
        ):
            return Path(value)
    return None


SNAPSHOT = _mounted_snapshot() or HF_SNAPSHOT


class Checkpoint:
    def __init__(self, path=SNAPSHOT):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = SimpleNamespace(**json.loads((self.path / "config.json").read_text())["text_config"])

    def tensor(self, key):
        with safe_open(self.path / self.index[key], framework="pt") as handle:
            return handle.get_tensor(key)

    def layer(self, idx):
        prefix = f"model.language_model.layers.{idx}."
        return {key[len(prefix) :]: self.tensor(key) for key in self.index if key.startswith(prefix)}


class TerminalCCL:
    """Sampling1D adapter sharing the model's already-allocated semaphore pool."""

    def __init__(self, manager):
        self.manager = manager
        self.buffers = {}

    def get_num_links(self, axis=None):
        return self.manager.num_links

    def get_and_cycle_ag_semaphore_handles(self, axis=None):
        return self.manager.get_ag_ping_pong_semaphore()

    def get_and_cycle_barrier_semaphore_handle(self, axis=None):
        return self.manager.get_barrier_semaphore()

    def line_all_gather(self, x, *, dim, cluster_axis=None, memory_config=None, num_links=2, buffer_key=None):
        output = None
        # Sampling1D owns and explicitly deallocates its gathered candidates.
        # Those outputs are trace-local scratch, not reusable manager buffers.
        # The decode embedding output has model-owned persistent lifetime.
        if buffer_key is not None and not buffer_key.startswith("SAMPLING_"):
            key = (buffer_key, tuple(x.shape), x.dtype)
            if key not in self.buffers:
                shape = list(x.shape)
                shape[dim] *= 4
                self.buffers[key] = ttnn.empty(
                    shape,
                    device=self.manager.mesh_device,
                    dtype=x.dtype,
                    layout=x.layout,
                    memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                )
            output = self.buffers[key]
        return ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            cluster_axis=1,
            persistent_output_buffer=output,
            multi_device_global_semaphore=self.manager.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.manager.get_barrier_semaphore(),
            topology=self.manager.topology,
            num_links=min(num_links, self.manager.num_links),
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )


class MuseGlimmerModel:
    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=SNAPSHOT,
        layer_indices=None,
        policy=None,
        terminal_policy=None,
        precision_config=None,
    ):
        self.mesh = mesh_device
        self.checkpoint = Checkpoint(checkpoint)
        self.config = self.checkpoint.config
        self.precision_config = load_precision_config(precision_config)
        precision = self.precision_config
        if self.config.max_position_embeddings != precision["max_context"]:
            raise ValueError("Selected precision context differs from checkpoint")
        self.residual_dtype = getattr(ttnn, precision["residual_dtype"])
        self.logits_dtype = getattr(ttnn, precision["logits_dtype"])
        self.terminal_policy = dict(precision["terminal"])
        unknown = set(terminal_policy or {}) - self.terminal_policy.keys()
        if unknown:
            raise ValueError(f"Unknown terminal policy fields: {sorted(unknown)}")
        self.terminal_policy.update(terminal_policy or {})
        c = self.config
        self.layer_indices = list(range(c.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        self.ccl = MeshCCLManager(mesh_device, 2, ttnn.Topology.Ring)
        self.terminal_ccl = TerminalCCL(self.ccl)
        self.layers = []
        for idx in self.layer_indices:
            print(f"LOAD_LAYER {idx}", flush=True)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    self.checkpoint.layer(idx),
                    hf_config=c,
                    layer_idx=idx,
                    mesh_device=mesh_device,
                    ccl=self.ccl,
                    policy=dict(decoder_policy(precision, c.layer_types[idx], idx), **(policy or {})),
                )
            )
        # RoPE and absolute position tables are identical across sliding layers.
        # Reuse one backing allocation instead of retaining 39 duplicate tables.
        rotary = next(layer for layer in self.layers if layer.use_rope)
        for layer in self.layers:
            layer.positions = rotary.positions
            if layer.use_rope:
                layer.cos, layer.sin = rotary.cos, rotary.sin
        self.cos_table = ttnn.to_layout(
            ttnn.reshape(rotary.cos, (c.max_position_embeddings, c.head_dim)), ttnn.ROW_MAJOR_LAYOUT
        )
        self.sin_table = ttnn.to_layout(
            ttnn.reshape(rotary.sin, (c.max_position_embeddings, c.head_dim)), ttnn.ROW_MAJOR_LAYOUT
        )

        print("LOAD_EMBEDDING", flush=True)
        embedding = self.checkpoint.tensor("model.language_model.embed_tokens.weight")
        # Exact HF weightless RMSNorm folded into immutable embedding weights.
        normalized = torch.empty_like(embedding)
        for start in range(0, c.vocab_size, 4096):
            value = embedding[start : start + 4096].float()
            normalized[start : start + 4096] = (
                value * (value.square().mean(-1, keepdim=True) + c.rms_norm_eps).pow(-0.5)
            ).to(embedding.dtype)
        self.embedding_weight = self.upload(
            normalized, dtype=getattr(ttnn, precision["embedding_dtype"]), layout=ttnn.ROW_MAJOR_LAYOUT, shard_dim=1
        )
        del embedding, normalized
        self.final_norm = self.upload(
            self.checkpoint.tensor("model.language_model.norm.weight").reshape(1, 1, 208, 32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=getattr(ttnn, precision["norm_weight_dtype"]),
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, precision["final_norm_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=precision["final_norm_fp32"],
            packer_l1_acc=True,
        )
        self.vocab_size = c.vocab_size
        self.padded_vocab_size = math.ceil(c.vocab_size / 128) * 128
        print("LOAD_LM_HEAD", flush=True)
        head = self.checkpoint.tensor(
            "model.language_model.embed_tokens.weight" if c.tie_word_embeddings else "lm_head.weight"
        )
        head = torch.nn.functional.pad(head, (0, 0, 0, self.padded_vocab_size - c.vocab_size)).t().contiguous()
        helper = self.layers[0]
        self.final_norm_mem = helper._width_mem(c.hidden_size, exact=True)
        terminal = self.terminal_policy
        input_cores = terminal["input_cores"]
        if (c.hidden_size // 32) % input_cores:
            raise ValueError("LM-head input cores must divide hidden tiles")
        self.lm_input_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(input_cores, ttnn.CoreCoord(8, 8), row_wise=True),
                [32, c.hidden_size // input_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        local_width = self.padded_vocab_size // 4
        split_width = math.ceil(local_width / (32 * terminal["splits"])) * 32
        split_sizes, weights, memories, programs = [], [], [], []
        # Split within each rank's contiguous vocabulary ownership. Concatenating
        # the local outputs restores exactly the original vocabulary indices.
        for start in range(0, local_width, split_width):
            width = min(split_width, local_width - start)
            part = torch.cat(
                [head[:, rank * local_width + start : rank * local_width + start + width] for rank in range(4)], dim=1
            ).contiguous()
            memory = helper._weight_mem(c.hidden_size, width)
            split_sizes.append(width)
            memories.append(memory)
            programs.append(
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=terminal["block_w"],
                    per_core_M=1,
                    per_core_N=math.ceil(width / (32 * self.lm_input_mem.shard_spec.grid.num_cores())),
                    num_workers_per_dram_bank=terminal["readers"],
                )
            )
            weights.append(
                LazyWeight(
                    source=part,
                    device=mesh_device,
                    dtype=getattr(ttnn, terminal["weight_dtype"]),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory,
                    mesh_mapper_config=ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([4])
                    ),
                )
            )
        self.lm_program = programs[0]
        self.lm_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, terminal["fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=terminal["fp32"],
            packer_l1_acc=True,
        )
        self.lm_head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=weights,
                mesh_device=mesh_device,
                dim=c.hidden_size,
                max_batch_size=32,
                program_configs=programs,
                compute_kernel_config=self.lm_compute_config,
                output_split_sizes=split_sizes,
                lm_head_dtype=self.logits_dtype,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                input_memcfg=self.lm_input_mem,
                weights_memcfgs=memories,
            )
        )
        self.lm_head.load_device_weights()
        self.lm_decode = self.lm_head.output_weights[0]
        print("MODEL_LOADED", flush=True)

    def upload(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard_dim=None, memory=None):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh)
                if shard_dim is None
                else ttnn.ShardTensorToMesh(self.mesh, dim=shard_dim)
            ),
            memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
        )

    def allocate_cache(self, *, batch_size=1, num_pages=4096):
        return [layer.allocate_cache(batch_size=batch_size, num_pages=num_pages) for layer in self.layers]

    def embed(self, token_ids, *, decode=False):
        x = ttnn.embedding(token_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=self.residual_dtype)
        x = ttnn.reshape(x, (1, 1, token_ids.volume(), self.config.hidden_size // 4))
        x = self.terminal_ccl.line_all_gather(x, dim=3, buffer_key="decode_embedding" if decode else None)
        return ttnn.to_memory_config(x, self.layers[0].residual_mem) if decode else x

    def head(self, x, *, decode=False):
        if x.shape[-2] > 32:
            # All-logit diagnostics use the same measured terminal geometry.
            # Production prefill selects its final logical row before head().
            rows = x.shape[-2]
            parts = []
            for start in range(0, rows, 32):
                tile = ttnn.slice(x, [0, 0, start, 0], [1, 1, min(start + 32, rows), self.config.hidden_size])
                parts.append(self.head(tile, decode=True))
            return ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_memory_config(x, self.final_norm_mem)
        x = ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            weight=self.final_norm,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, 8], subblock_w=1, block_h=1, block_w=4, inplace=False
            ),
            compute_kernel_config=self.compute_config,
            memory_config=self.final_norm_mem,
        )
        x = ttnn.to_memory_config(x, self.lm_input_mem)
        logits = self.lm_head(x)
        logits = ttnn.multiply(logits, self.config.output_multiplier / self.config.final_logit_softcapping)
        return ttnn.multiply(ttnn.tanh(logits), self.config.final_logit_softcapping)

    def prefill_forward(self, token_ids, *, kv_cache, page_table, start_pos=0, slot=0, all_logits=False):
        x = self.embed(token_ids)
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.prefill_forward(x, kv_cache=cache, page_table=page_table, start_pos=start_pos, slots=[slot])
        if not all_logits:
            length = x.shape[-2]
            x = ttnn.slice(x, [0, 0, length - 1, 0], [1, 1, length, self.config.hidden_size])
        return self.head(x)

    def decode_forward(self, token_ids, *, kv_cache, page_table, current_pos, rope_indices):
        batch = token_ids.volume()
        x = self.embed(token_ids, decode=True)
        cos, sin = [
            ttnn.reshape(ttnn.embedding(rope_indices, table, layout=ttnn.TILE_LAYOUT), (1, batch, 1, 128))
            for table in (self.cos_table, self.sin_table)
        ]
        for layer_idx, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            x = layer.decode_forward(
                x,
                kv_cache=cache,
                page_table=page_table[layer_idx] if isinstance(page_table, (list, tuple)) else page_table,
                current_pos=current_pos,
                rope_cos=cos,
                rope_sin=sin,
            )
        return self.head(x, decode=True)

    def prefill_chunk(
        self,
        token_ids,
        *,
        kv_cache,
        page_table,
        fill_page_table,
        start_pos,
        rope_indices,
        tail_start=None,
        alignment=0,
    ):
        """Physical bucket with runtime position/table state, no request-key programs.

        The generator pads tokens, limits valid rows and owns cache pages. Causal
        masking makes padded tail rows invisible to every preceding logical row.
        Large continuation chunks start at multiples of 128, matching the
        inherited q64/k128 SDPA policy. Prepared 32-row chunks bridge other
        page-aligned starts using the inherited adjusted SDPA geometry.
        """
        x = self.embed(token_ids)
        size = token_ids.volume()
        if alignment not in (0, 32, 64, 96) or (alignment and size != 32):
            raise ValueError("Nonzero alignment hints require a 32-row append chunk")
        if size == 64 and tail_start is None:
            raise ValueError("The 64-row bucket requires the persistent absolute tail-start tensor")
        cos, sin = [
            ttnn.reshape(ttnn.embedding(rope_indices, table, layout=ttnn.TILE_LAYOUT), (1, 1, size, 128))
            for table in (self.cos_table, self.sin_table)
        ]
        for layer_idx, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            layer_table = page_table[layer_idx] if isinstance(page_table, (list, tuple)) else page_table
            layer_fill = fill_page_table[layer_idx] if isinstance(fill_page_table, (list, tuple)) else fill_page_table
            if size == 64:
                # Preserve PreparedPrefill's selected 32+32 decomposition,
                # including native AR and DRAM-sharded short projections.
                parts = []
                for offset, position in ((0, start_pos), (32, tail_start)):
                    hidden = ttnn.slice(x, [0, 0, offset, 0], [1, 1, offset + 32, self.config.hidden_size])
                    chunk_cos, chunk_sin = [
                        ttnn.slice(value, [0, 0, offset, 0], [1, 1, offset + 32, 128]) for value in (cos, sin)
                    ]
                    pages = ttnn.slice(layer_fill, [0, offset // 32], [1, offset // 32 + 1])
                    parts.append(
                        layer._prefill_chunk(
                            hidden,
                            start_pos=offset,
                            slot=0,
                            kv_cache=cache,
                            page_table=layer_table,
                            rope_cos=chunk_cos,
                            rope_sin=chunk_sin,
                            fill_page_table=pages,
                            chunk_start_tensor=position,
                        )
                    )
                x = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                x = layer._prefill_chunk(
                    x,
                    start_pos=alignment,
                    slot=0,
                    kv_cache=cache,
                    page_table=layer_table,
                    rope_cos=cos,
                    rope_sin=sin,
                    fill_page_table=layer_fill,
                    chunk_start_tensor=start_pos,
                )
        return x
