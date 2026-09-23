# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full text autoregressive path over the validated Blackhole TP4 decoder."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import DecoderState
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision
from models.common.modules.tt_ccl import TT_CCL

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def checkpoint_path():
    mounted_weights = os.getenv("MODEL_WEIGHTS_DIR")
    if mounted_weights:
        path = Path(mounted_weights).expanduser().resolve()
        for required in ("config.json", "model.safetensors.index.json"):
            if not (path / required).is_file():
                raise FileNotFoundError(f"MODEL_WEIGHTS_DIR is missing {required}: {path}")
        return path

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
    def __init__(self, mesh_device, *, snapshot=None, layer_indices=None, head_strategy="dram", precision_config=None):
        self.precision = load_precision(precision_config)
        self.mesh = mesh_device
        self.snapshot = Path(snapshot or checkpoint_path())
        self.config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True).text_config
        self.context = self.precision["max_context"]
        assert self.context == self.config.max_position_embeddings
        self.ccl = TT_CCL(mesh_device)
        self.checkpoint = Checkpoint(self.snapshot)
        self.layer_indices = (
            list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        self.layers = []
        # TILE [B,1,H] allocates a separate 32-row tile plane for every
        # request.  Keep multi-request decode residuals as [1,1,B,H] instead,
        # which has the same compact physical row layout as logical [B,H].
        # Set QWEN_COMPACT_DECODE_RESIDUAL=0 for an immediate rollback.
        self.compact_decode_residual = os.getenv("QWEN_COMPACT_DECODE_RESIDUAL", "1") == "1"
        self.decode_buckets = os.getenv("QWEN_DECODE_BUCKETS", "0") == "1"
        prefill_layout = os.getenv("QWEN_PREFILL_RESIDUAL_LAYOUT", "replicated")
        if prefill_layout not in ("replicated", "sharded", "sharded_replicated_norm"):
            raise ValueError("Unknown QWEN_PREFILL_RESIDUAL_LAYOUT")
        self.prefill_sharded_residual = prefill_layout != "replicated"
        self.prefill_batched_head = os.getenv("QWEN_PREFILL_BATCHED_HEAD", "0") == "1"
        row_parallel_norm = os.getenv("QWEN_PREFILL_ROW_PARALLEL_NORM", "0") == "1"
        if row_parallel_norm and prefill_layout != "sharded_replicated_norm":
            raise ValueError("QWEN_PREFILL_ROW_PARALLEL_NORM requires sharded_replicated_norm layout")
        prefill_policy = {}
        if self.prefill_sharded_residual:
            prefill_policy.update(
                prefill_sharded_residual=True,
                prefill_replicated_norm=prefill_layout == "sharded_replicated_norm",
                prefill_row_parallel_norm=row_parallel_norm,
            )
        if os.getenv("QWEN_PREFILL_PACKED_SWIGLU", "0") == "1":
            prefill_policy["prefill_packed_swiglu"] = True
        for i in self.layer_indices:
            print(f"LOAD_LAYER {i}", flush=True)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    self.checkpoint.layer(i),
                    hf_config=self.config,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    ccl=self.ccl,
                    policy={
                        **decoder_policy(self.precision, i),
                        **prefill_policy,
                        "compact_decode_residual": self.compact_decode_residual,
                        "compact_decode_mlp": os.getenv("QWEN_COMPACT_DECODE_MLP", "0") == "1",
                        "batched_decode_rope": os.getenv("QWEN_BATCHED_DECODE_ROPE", "0") == "1",
                        "compact_decode_attention": os.getenv("QWEN_COMPACT_DECODE_ATTENTION", "0") == "1",
                    },
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
        self.head_weight = self.upload(
            self.checkpoint.tensor(name).T.contiguous(),
            dtype=getattr(ttnn, self.precision["weight_groups"]["head"]),
            shard=-1,
        )
        self.head_strategy = head_strategy
        if head_strategy not in ("interleaved", "dram"):
            raise ValueError("Unknown LM-head program family")
        self.head_decode_weights = []
        if head_strategy == "dram":
            banks = mesh_device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            for start in range(0, self.config.vocab_size // 4, 16384):
                weight = self.head_weight[:, start : min(start + 16384, self.config.vocab_size // 4)]
                width = ((weight.shape[-1] + banks * 64 - 1) // (banks * 64)) * 64
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, [self.config.hidden_size, width], ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.head_decode_weights.append(ttnn.to_memory_config(weight, memory))
        self.head_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["compute_fidelities"]["head"]),
            math_approx_mode=False,
            fp32_dest_acc_en=self.precision["fp32_dest_acc_en"],
            packer_l1_acc=True,
        )
        self.norm_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["final_norm_compute_fidelity"]),
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

    def embed(self, tokens, *, batch, length, sharded=False, compact=False):
        if compact and (sharded or length != 1):
            raise ValueError("Compact embedding is supported only for replicated single-token decode")
        out = ttnn.embedding(
            tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.reshape(out, [1, 1, batch * length, self.config.hidden_size // 4])
        if sharded:
            return ttnn.reshape(out, [batch, length, self.config.hidden_size // 4])
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
        if compact:
            return out
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
        batch = hidden.shape[-2] if len(hidden.shape) == 4 else hidden.shape[0]
        if decode and self.head_strategy == "dram":
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
                compute_kernel_config=self.norm_compute,
            )
            hidden = ttnn.to_memory_config(hidden, self.layers[0]._width_memory(8, 32, 640))
            return self._dram_logits(hidden)
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.rms_norm(
            hidden,
            weight=self.norm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute,
        )
        if decode:
            hidden = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [1, 1, batch, self.config.hidden_size])
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
        return ttnn.linear(
            hidden,
            self.head_weight,
            dtype=getattr(ttnn, self.precision["logits_dtype"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_compute,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def _dram_logits(self, hidden):
        parts = []
        for weight in self.head_decode_weights:
            config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=5,
                per_core_M=1,
                per_core_N=weight.memory_config().shard_spec.shape[1] // 32,
                num_workers_per_dram_bank=2,
                fused_activation=None,
            )
            out = ttnn.linear(
                hidden,
                weight,
                dtype=getattr(ttnn, self.precision["logits_dtype"]),
                compute_kernel_config=self.head_compute,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=config,
            )
            parts.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
        return ttnn.concat(parts, dim=-1)

    def prefill(self, tokens, *, cache, page_table, length, start_pos=0, slot=0, all_logits=False, positions=None):
        """Single request, logical length; independent fixed-slot state is updated on device."""
        x = self.embed(tokens, batch=1, length=length)
        if positions is None:
            positions = self.upload(
                torch.arange(start_pos, start_pos + length, dtype=torch.int32).reshape(1, length),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        elif (
            not isinstance(positions, ttnn.Tensor)
            or tuple(positions.shape) != (1, length)
            or positions.dtype != ttnn.uint32
            or positions.layout != ttnn.ROW_MAJOR_LAYOUT
            or positions.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Prefill positions must be a UINT32 row-major DRAM tensor with shape [1, length]")
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

    def prefill_batch(self, tokens, *, cache, page_table, length, start_pos, slots, positions=None, return_logits=True):
        """Experimental equal-length, page-aligned prefill over consecutive slots."""
        batch = len(slots)
        if batch < 2 or slots != list(range(slots[0], slots[0] + batch)):
            raise ValueError("Batched prefill requires consecutive slots")
        if slots[0] < 0 or slots[-1] >= cache.batch_size or start_pos % 32:
            raise ValueError("Invalid slots or unaligned batched prefix")
        if length < 1 or start_pos < 0 or start_pos + length > cache.capacity:
            raise ValueError("Batched prefill exceeds cache capacity")
        first, end = slots[0], slots[-1] + 1
        sharded = getattr(self, "prefill_sharded_residual", False)
        x = self.embed(tokens, batch=batch, length=length, sharded=sharded)
        if positions is None:
            positions = self.upload(
                torch.arange(start_pos, start_pos + length, dtype=torch.int32).repeat(batch, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        elif (
            not isinstance(positions, ttnn.Tensor)
            or tuple(positions.shape) != (batch, length)
            or positions.dtype != ttnn.uint32
            or positions.layout != ttnn.ROW_MAJOR_LAYOUT
            or positions.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Batched prefill positions must be UINT32 row-major DRAM [batch, length]")
        cc, ss = self.rope(positions, batch=batch, length=length)
        table = page_table[first:end, :]
        for layer, state in zip(self.layers, cache.layers):
            local = state
            if layer.kind == "linear_attention" and batch != cache.batch_size:
                local = DecoderState(conv=state.conv[first:end], recurrent=state.recurrent[first:end])
            forward = layer.prefill_sharded_forward if sharded else layer.prefill_forward
            x = forward(x, state=local, start_pos=start_pos, page_table=table, cos=cc, sin=ss)
            if local is not state:
                for name in ("conv", "recurrent"):
                    before = getattr(state, name)
                    pieces = ([before[:first]] if first else []) + [getattr(local, name)]
                    if end < cache.batch_size:
                        pieces.append(before[end:])
                    ttnn.copy(ttnn.concat(pieces, dim=0), before)
        if not return_logits:
            return []
        # Public logits retain independent ownership, even with a shared head call.
        batched_head = getattr(self, "prefill_batched_head", False)
        if sharded or batched_head or getattr(self, "prefill_compact_head", False):
            x = x[:, length - 1 : length, :]
            length = 1
        if sharded:
            x = self.layers[-1]._gather(ttnn.reshape(x, [1, batch, 1, self.config.hidden_size // 4]))
            x = ttnn.reshape(x, [batch, 1, self.config.hidden_size])
        if batched_head:
            logits = self.logits(x, decode=True)
            return [ttnn.clone(logits[:, :, row : row + 1, :]) for row in range(batch)]
        return [self.logits(x[row : row + 1, length - 1 : length, :], decode=True) for row in range(batch)]

    def decode(self, tokens, positions, *, cache, page_table, rope_indices=None, active_slots=None):
        if getattr(self, "decode_buckets", False):
            slots = tuple(range(cache.batch_size)) if active_slots is None else tuple(active_slots)
            if not slots or len(set(slots)) != len(slots) or any(i < 0 or i >= cache.batch_size for i in slots):
                raise ValueError("Decode buckets require unique active slots inside the cache")
            if cache.batch_size not in (1, 8, 16):
                raise ValueError("Decode buckets require serving capacity 1, 8, or 16")
            bucket = next(b for b in (1, 8, 16) if b >= len(slots))
            if bucket != cache.batch_size or slots != tuple(range(cache.batch_size)):
                return self._decode_bucket(tokens, positions, cache, page_table, rope_indices, slots, bucket)
        return self._decode_fixed(
            tokens, positions, cache=cache, page_table=page_table, rope_indices=rope_indices, active_slots=active_slots
        )

    def _decode_bucket(self, tokens, positions, cache, page_table, rope_indices, slots, bucket):
        """Pack active rows on device, execute one fixed bucket, restore scheduler rows.

        Page IDs keep their original ownership in the shared KV pool. Linear
        state is gathered/scattered within the trace, so prefill, slot remapping,
        and warmup backups always see the authoritative full-capacity state.
        Sampling stays in scheduler order, preserving per-request RNG streams.
        """

        def pack(tensor, *, fill=0):
            contiguous = slots == tuple(range(slots[0], slots[0] + len(slots)))
            rows = [tensor[slots[0] : slots[-1] + 1]] if contiguous else [tensor[i : i + 1] for i in slots]
            packed = ttnn.concat(rows, dim=0) if len(rows) > 1 else rows[0]
            if len(slots) < bucket:
                # Device pad is traceable. full_like on row-major integers
                # uploads host data and is forbidden during trace capture.
                packed = ttnn.pad(packed, [(0, bucket - len(slots))] + [(0, 0)] * (len(packed.shape) - 1), value=fill)
            return packed

        ids = pack(ttnn.reshape(tokens, [32, 1]))
        ids = ttnn.reshape(ids, [1, 1, 1, bucket])
        ids = ttnn.pad(ids, [(0, 0), (0, 0), (0, 0), (0, 32 - bucket)], value=0)
        packed_positions = ttnn.reshape(pack(ttnn.reshape(positions, [cache.batch_size, 1]), fill=-1), [bucket])
        packed_rope = None
        if rope_indices is not None:
            packed_rope = ttnn.reshape(pack(ttnn.reshape(rope_indices, [cache.batch_size, 1])), [bucket])
        packed_table = pack(page_table)
        layers = []
        for state in cache.layers:
            layers.append(
                DecoderState(
                    key=state.key,
                    value=state.value,
                    conv=pack(state.conv) if state.conv is not None else None,
                    recurrent=pack(state.recurrent) if state.recurrent is not None else None,
                )
            )
        packed_cache = ModelCache(layers, bucket, cache.capacity, cache.num_pages)
        logits = self._decode_fixed(
            ids,
            packed_positions,
            cache=packed_cache,
            page_table=packed_table,
            rope_indices=packed_rope,
            # Padding state is disposable; only live rows are scattered below.
        )
        inverse = {slot: row for row, slot in enumerate(slots)}
        for original, packed in zip(cache.layers, layers):
            for name in ("conv", "recurrent"):
                target, source = getattr(original, name), getattr(packed, name)
                if target is None:
                    continue
                # Coalesce adjacent scheduler rows instead of copying each
                # inactive row separately (the usual mapping is a prefix).
                rows = []
                start = 0
                while start < cache.batch_size:
                    live = start in inverse
                    end = start + 1
                    while end < cache.batch_size and (end in inverse) == live:
                        if live and inverse[end] != inverse[start] + end - start:
                            break
                        end += 1
                    rows.append(source[inverse[start] : inverse[start] + end - start] if live else target[start:end])
                    start = end
                ttnn.copy(ttnn.concat(rows, dim=0) if len(rows) > 1 else rows[0], target)
        zero = ttnn.zeros_like(logits[:, :, :1, :])
        rows = [logits[:, :, inverse[i] : inverse[i] + 1, :] if i in inverse else zero for i in range(cache.batch_size)]
        return ttnn.concat(rows, dim=2) if len(rows) > 1 else rows[0]

    def _decode_fixed(self, tokens, positions, *, cache, page_table, rope_indices=None, active_slots=None):
        b = cache.batch_size
        ids = ttnn.reshape(tokens, [1, 32])[:, :b]
        x = self.embed(ids, batch=b, length=1, compact=self.compact_decode_residual and b > 1)
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
