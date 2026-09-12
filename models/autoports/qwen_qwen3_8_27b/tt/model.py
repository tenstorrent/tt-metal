# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full text autoregressive path over the validated Blackhole TP4 decoder."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import DecoderState
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.common.modules.tt_ccl import TT_CCL

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def checkpoint_path():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))


class Checkpoint:
    """Load one layer at a time; safetensors stays memory mapped on the host."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def layer(self, index):
        prefix = f"model.language_model.layers.{index}."
        return {k.removeprefix(prefix): self.tensor(k) for k in self.index if k.startswith(prefix)}


@dataclass
class ModelCache:
    layers: list
    batch_size: int
    capacity: int
    num_pages: int


class QwenModel:
    def __init__(self, mesh_device, *, snapshot=None, layer_indices=None, head_strategy="dram"):
        self.mesh = mesh_device
        self.snapshot = Path(snapshot or checkpoint_path())
        self.config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True).text_config
        self.context = self.config.max_position_embeddings
        self.ccl = TT_CCL(mesh_device)
        self.checkpoint = Checkpoint(self.snapshot)
        self.layer_indices = (
            list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        self.layers = []
        for i in self.layer_indices:
            print(f"LOAD_LAYER {i}", flush=True)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    self.checkpoint.layer(i),
                    hf_config=self.config,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    ccl=self.ccl,
                )
            )
        self.embedding_weight = self.upload(
            self.checkpoint.tensor("model.language_model.embed_tokens.weight"),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard=-1,
        )
        self.norm_weight = self.upload(
            (self.checkpoint.tensor("model.language_model.norm.weight").float() + 1).reshape(1, 1, -1)
        )
        name = "model.language_model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        self.head_weight = self.upload(self.checkpoint.tensor(name).T.contiguous(), dtype=ttnn.bfloat8_b, shard=-1)
        self.head_strategy = head_strategy
        if head_strategy not in ("interleaved", "dram"):
            raise ValueError("Unknown LM-head program family")
        self.head_decode_weights = []
        if head_strategy == "dram":
            banks = mesh_device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            for start in range(0, self.config.vocab_size // 4, 8192):
                weight = self.head_weight[:, start : min(start + 8192, self.config.vocab_size // 4)]
                width = ((weight.shape[-1] + banks * 64 - 1) // (banks * 64)) * 64
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, [self.config.hidden_size, width], ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.head_decode_weights.append(ttnn.to_memory_config(weight, memory))
        self.head_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        rope = Qwen3_5TextRotaryEmbedding(self.config)
        cc, ss = rope(torch.empty(1, self.context, 1, dtype=torch.bfloat16), torch.arange(self.context)[None])
        self.cos_table, self.sin_table = [self.upload(v.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for v in (cc, ss)]
        self.rotary_width = cc.shape[-1]
        print("MODEL_LOADED", flush=True)

    def upload(self, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
        return ttnn.from_torch(
            tensor.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard)
            ),
        )

    def allocate_cache(self, *, batch_size, capacity):
        if not 1 <= batch_size <= 32 or not 1 <= capacity <= self.context:
            raise ValueError("Cache requires 1..32 slots and capacity within the HF context")
        pages = (capacity + 31) // 32
        return ModelCache(
            [layer.allocate_state(batch_size=batch_size, num_pages=batch_size * pages) for layer in self.layers],
            batch_size,
            capacity,
            batch_size * pages,
        )

    def reset_cache(self, cache):
        for state in cache.layers:
            for name in ("key", "value", "conv", "recurrent"):
                tensor = getattr(state, name)
                if tensor is not None:
                    ttnn.copy(ttnn.zeros_like(tensor), tensor)

    def embed(self, tokens, *, batch, length):
        out = ttnn.embedding(
            tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.reshape(out, [1, 1, batch * length, self.config.hidden_size // 4])
        out = ttnn.experimental.all_gather_async(
            out,
            dim=3,
            cluster_axis=1,
            mesh_device=self.mesh,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
        )
        return ttnn.reshape(out, [batch, length, self.config.hidden_size])

    def rope(self, positions, *, batch, length):
        return tuple(
            ttnn.reshape(
                ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                [batch, length, self.rotary_width],
            )
            for table in (self.cos_table, self.sin_table)
        )

    def logits(self, hidden, *, decode=False):
        if decode and self.head_strategy == "dram":
            batch = hidden.shape[0]
            memory = self.layers[0]._width_memory(40, 32, 128)
            hidden = ttnn.to_memory_config(ttnn.reshape(hidden, [1, 1, batch, 5120]), memory)
            hidden = ttnn.rms_norm(
                hidden,
                weight=self.norm_weight,
                epsilon=self.config.rms_norm_eps,
                memory_config=memory,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(10, 4), subblock_w=4, block_h=1, block_w=4, inplace=False
                ),
                compute_kernel_config=self.head_compute,
            )
            hidden = ttnn.to_memory_config(hidden, self.layers[0]._width_memory(8, 32, 640))
            return self._dram_logits(hidden)
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.rms_norm(
            hidden,
            weight=self.norm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_compute,
        )
        if decode:
            hidden = ttnn.reshape(
                ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [1, 1, hidden.shape[0], self.config.hidden_size]
            )
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
        return ttnn.linear(
            hidden,
            self.head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_compute,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def _dram_logits(self, hidden):
        parts = []
        for weight in self.head_decode_weights:
            config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=10,
                per_core_M=1,
                per_core_N=weight.memory_config().shard_spec.shape[1] // 32,
                num_workers_per_dram_bank=2,
                fused_activation=None,
            )
            out = ttnn.linear(
                hidden,
                weight,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.head_compute,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=config,
            )
            parts.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
        return ttnn.concat(parts, dim=-1)

    def prefill(self, tokens, *, cache, page_table, length, start_pos=0, slot=0, all_logits=False):
        """Single request, logical length; independent fixed-slot state is updated on device."""
        x = self.embed(tokens, batch=1, length=length)
        positions = self.upload(
            torch.arange(start_pos, start_pos + length, dtype=torch.int32).reshape(1, length),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cc, ss = self.rope(positions, batch=1, length=length)
        pos_matrix = ttnn.reshape(ttnn.typecast(positions, ttnn.int32), [length, 1])
        table = page_table[slot : slot + 1, :]
        for layer, state in zip(self.layers, cache.layers):
            if layer.kind == "linear_attention" and cache.batch_size > 1:
                local = DecoderState(conv=state.conv[slot : slot + 1], recurrent=state.recurrent[slot : slot + 1])
            else:
                local = state
            x = layer.prefill_forward(
                x, state=local, start_pos=start_pos, page_table=table, cos=cc, sin=ss, positions=pos_matrix
            )
            if local is not state:
                # Slice outputs do not alias their source. Scatter the request state back.
                for name in ("conv", "recurrent"):
                    before = getattr(state, name)
                    pieces = ([before[:slot]] if slot else []) + [getattr(local, name)]
                    if slot + 1 < cache.batch_size:
                        pieces.append(before[slot + 1 :])
                    ttnn.copy(ttnn.concat(pieces, dim=0), before)
        if not all_logits:
            x = x[:, length - 1 : length, :]
        return self.logits(x, decode=not all_logits)

    def decode(self, tokens, positions, *, cache, page_table, rope_indices=None, active_slots=None):
        b = cache.batch_size
        ids = ttnn.reshape(tokens, [1, 32])[:, :b]
        x = self.embed(ids, batch=b, length=1)
        indices = ttnn.reshape(
            rope_indices if rope_indices is not None else ttnn.typecast(positions, ttnn.uint32), [1, b]
        )
        cc, ss = self.rope(indices, batch=b, length=1)
        inactive = () if active_slots is None else tuple(i for i in range(b) if i not in active_slots)
        for layer, state in zip(self.layers, cache.layers):
            preserved = {}
            if inactive and layer.kind == "linear_attention":
                preserved = {
                    name: {i: getattr(state, name)[i : i + 1] for i in inactive} for name in ("conv", "recurrent")
                }
            x = layer.decode_forward(x, state=state, current_pos=positions, page_table=page_table, cos=cc, sin=ss)
            for name, rows in preserved.items():
                target = getattr(state, name)
                ttnn.copy(ttnn.concat([rows[i] if i in rows else target[i : i + 1] for i in range(b)], dim=0), target)
        return self.logits(x, decode=True)
