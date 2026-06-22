# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Ministral3 attention: base TT Attention + Llama-4 post-RoPE Q scaling + legacy HF rope.

from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext

import ttnn
from models.experimental.devstral2_small.tt.tt_ministralrmsnorm import (
    ministral_prefill_block_shard_grid,
    ministral_prefill_block_shard_mem_cfg,
)
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode

_TILE = 32
_WO_1D_GRID_X = 10
_WO_1D_GRID_Y = 8


def ministral_qkv_bf16_activations_enabled() -> bool:
    """Default on: BF16 QKV activations with BFP8 weights (see forward_prefill)."""
    default = os.environ.get("TT_MINISTRAL3_QKV_BF16_ACT", "1")
    return default.strip().lower() not in ("0", "false", "no")


def _qkv_shard_n(args) -> int:
    return int(args.qkv_size) // int(args.num_devices)


def _qkv_block_sharding_enabled(args, seq_len: int) -> bool:
    """Block-sharded in0 for 128×5120×1536 QKV (norm out → matmul, no sharded_to_interleaved)."""
    return _qkv_linear_sweep_enabled(args, seq_len)


def _qkv_block_shard_program_config(args, seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    padded_seq = ((int(seq_len) + _TILE - 1) // _TILE) * _TILE
    qkv_n = _qkv_shard_n(args)
    grid = ministral_prefill_block_shard_grid(args, seq_len)
    grid_x, grid_y = int(grid.x), int(grid.y)
    per_core_m = (padded_seq // _TILE) // grid_y
    per_core_k = (int(args.dim) // _TILE) // grid_x
    per_core_n = (qkv_n // _TILE) // grid_x
    out_subblock_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= 4)
    in0_block_w = next(d for d in (4, 2, 1) if per_core_k % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _qkv_linear_sweep_enabled(args, seq_len: int) -> bool:
    """Sweep winner for 128×5120×1536 prefill QKV (test_linear_128x5120x1536_sweep)."""
    default = os.environ.get("TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM", "1")
    if os.environ.get("TT_MINISTRAL3_QKV_LINEAR_SWEEP", default).strip().lower() in ("0", "false", "no"):
        return False
    return (
        int(seq_len) <= 128
        and int(args.dim) == 5120
        and _qkv_shard_n(args) == 1536
        and not args.use_minimal_qkv_prefill_matmul(seq_len)
    )


def _qkv_linear_sweep_program_config() -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    # 2D l1/ds/dram 8x4 w8: sweep vs Tracy 128×5120×1536 ~55us (test_matmul_128x5120x1536_sweep).
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        out_block_h=1,
        out_block_w=6,
        per_core_M=1,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _prepare_qkv_linear_sweep_input(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.memory_config().buffer_type == ttnn.BufferType.L1:
        return x
    return ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)


def _wo_k_shard(args) -> int:
    return int(args.n_heads * args.head_dim) // int(args.num_devices)


def _use_1d_wo_dram_weights(args) -> bool:
    """1D mcast WO prefill sweep uses interleaved DRAM weights (not width-sharded decode WO)."""
    return (
        not args.use_fused_all_gather_matmul
        and not args.is_galaxy
        and int(args.dim) == 5120
        and _wo_k_shard(args) == 1024
    )


def _load_wo_prefill_sweep_interleaved(
    mesh_device,
    args,
    configuration,
    state_dict,
    weight_cache_path,
    layer_num,
    dtype,
) -> ttnn.Tensor:
    """Load WO from cache in DRAM interleaved layout (same pattern as MLP w2_prefill_sweep)."""
    layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
    wo_str = f"{layer_name}.wo"
    pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    if configuration.dummy_weights or weight_cache_path is None:
        cache_file = None
    else:
        cache_file = weight_cache_path / f"{layer_name}.wo_dram_il"
    mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    return ttnn.as_tensor(
        pt_wo,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        cache_file_name=cache_file,
    )


def _wo_linear_sweep_fits_device(mesh_device) -> bool:
    grid = mesh_device.compute_with_storage_grid_size()
    return int(grid.x) >= _WO_1D_GRID_X and int(grid.y) >= _WO_1D_GRID_Y


def _wo_linear_sweep_enabled(args, seq_len: int, full_seq_len: int, mesh_device) -> bool:
    """Sweep winner for 128×1024×5120 prefill WO (test_matmul_128x1024x5120_sweep)."""
    default = os.environ.get("TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM", "1")
    if os.environ.get("TT_MINISTRAL3_WO_LINEAR_SWEEP", default).strip().lower() in ("0", "false", "no"):
        return False
    if int(full_seq_len) != int(seq_len):
        return False
    if not _wo_linear_sweep_fits_device(mesh_device):
        return False
    return int(seq_len) <= 128 and int(args.dim) == 5120 and _wo_k_shard(args) == 1024


def _wo_linear_sweep_program_config() -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    # mcast1d_10x8_pcn2_ibw4 l1/dram/l1: ~20us vs Tracy 128×1024×5120 ~28us (test_matmul_128x1024x5120_sweep).
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(_WO_1D_GRID_X, _WO_1D_GRID_Y),
        in0_block_w=4,
        out_subblock_h=2,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=2,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _wo_linear_sweep_compute_kernel_config():
    # Match test_matmul_128x1024x5120_sweep (HiFi2, approx=True, no fp32 acc).
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _prepare_wo_linear_sweep_input(x: ttnn.Tensor) -> ttnn.Tensor:
    mc = x.memory_config()
    if mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return x
    return ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)


def _typecast_if_needed(x: ttnn.Tensor, dtype, memory_config=None) -> ttnn.Tensor:
    if x.dtype == dtype:
        return x
    kwargs = {"dtype": dtype}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.typecast(x, **kwargs)


@contextmanager
def _skip_identity_typecast():
    """Skip no-op typecasts; avoid deallocating a tensor still used as the cast output."""
    orig_typecast = ttnn.typecast
    orig_deallocate = ttnn.deallocate
    identity_src_ids: set[int] = set()

    def typecast(tensor, dtype=None, **kwargs):
        target = dtype if dtype is not None else kwargs.get("dtype")
        if target is not None and tensor.dtype == target:
            identity_src_ids.add(id(tensor))
            return tensor
        return orig_typecast(tensor, dtype=dtype, **kwargs)

    def deallocate(tensor):
        tid = id(tensor)
        if tid in identity_src_ids:
            identity_src_ids.discard(tid)
            return
        orig_deallocate(tensor)

    ttnn.typecast = typecast
    ttnn.deallocate = deallocate
    try:
        yield
    finally:
        ttnn.typecast = orig_typecast
        ttnn.deallocate = orig_deallocate


@contextmanager
def _qkv_linear_sweep_program_config_override(args, seq_len: int, *, block_sharded_in0: bool = False):
    if not _qkv_linear_sweep_enabled(args, seq_len) and not block_sharded_in0:
        yield
        return
    orig_get_pc = args.get_attn_qkv_program_config
    orig_get_mm = args.get_attn_qkv_mm_mem_config

    def get_pc_override(mode, sl=1, prefetcher=None):
        if mode == Mode.PREFILL and block_sharded_in0 and _qkv_block_sharding_enabled(args, sl):
            return _qkv_block_shard_program_config(args, sl)
        if mode == Mode.PREFILL and _qkv_linear_sweep_enabled(args, sl):
            return _qkv_linear_sweep_program_config()
        return orig_get_pc(mode, sl, prefetcher)

    def get_mm_override(mode, prefetcher=None):
        if mode == Mode.PREFILL and block_sharded_in0:
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
        return orig_get_mm(mode, prefetcher)

    args.get_attn_qkv_program_config = get_pc_override
    args.get_attn_qkv_mm_mem_config = get_mm_override
    try:
        yield
    finally:
        args.get_attn_qkv_program_config = orig_get_pc
        args.get_attn_qkv_mm_mem_config = orig_get_mm


@contextmanager
def _qkv_block_shard_linear_patch(attn: "TtMinistralAttention", seq_len: int):
    """DRAM QKV output after block-sharded matmul (parent all_reduce expects interleaved DRAM)."""
    orig_linear = ttnn.linear
    block_out = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    def linear(x, weight, **kwargs):
        if id(weight) != id(attn.wqkv) or not x.memory_config().is_sharded():
            return orig_linear(x, weight, **kwargs)
        kwargs = dict(kwargs)
        kwargs["memory_config"] = block_out
        kwargs["program_config"] = _qkv_block_shard_program_config(attn.args, seq_len)
        out = orig_linear(x, weight, **kwargs)
        if out.memory_config().is_sharded():
            return ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    ttnn.linear = linear
    try:
        yield
    finally:
        ttnn.linear = orig_linear


@contextmanager
def _wo_linear_sweep_runtime_patch(attn: "TtMinistralAttention"):
    """L1 WO in0 + 1D mcast matmul (l1/dram/l1); matches test_matmul_128x1024x5120_sweep best."""
    orig_linear = ttnn.linear

    wo_ids = {id(attn.wo), id(attn.wo_prefill_sweep)}

    def linear(x, weight, **kwargs):
        if id(weight) not in wo_ids:
            return orig_linear(x, weight, **kwargs)
        kwargs = dict(kwargs)
        kwargs["program_config"] = _wo_linear_sweep_program_config()
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
        kwargs["compute_kernel_config"] = _wo_linear_sweep_compute_kernel_config()
        x = _prepare_wo_linear_sweep_input(x)
        return orig_linear(x, attn.wo_prefill_sweep, **kwargs)

    ttnn.linear = linear
    try:
        yield
    finally:
        ttnn.linear = orig_linear


class TtMinistralAttention(Attention):
    """Ministral attention with Llama-4 Q scaling; legacy HF rope (not rotary_embedding_hf)."""

    def __init__(
        self,
        *args,
        llama_4_scaling_beta: float | None = None,
        original_max_position_embeddings: int | None = None,
        **kwargs,
    ):
        configuration = kwargs["configuration"]
        mesh_device = args[0]
        model_args = args[2]
        state_dict = args[3]
        weight_cache_path = args[4]
        layer_num = kwargs["layer_num"] if "layer_num" in kwargs else args[5]
        self.llama_4_scaling_beta = llama_4_scaling_beta
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(*args, **kwargs)
        # QKV prefill (128×5120×1536): LoFi BFP8 matmul; WO prefill stays HiFi2.
        self.li_qkv_prefill_compute_kernel_cfg = configuration.compute_kernel_config_lofi
        self.li_o_prefill_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        if self.wqkv.dtype != ttnn.bfloat8_b:
            self.wqkv = ttnn.typecast(self.wqkv, dtype=ttnn.bfloat8_b)
        if self.wo.dtype != ttnn.bfloat8_b:
            self.wo = ttnn.typecast(self.wo, dtype=ttnn.bfloat8_b)
        if _use_1d_wo_dram_weights(model_args):
            self.wo_prefill_sweep = _load_wo_prefill_sweep_interleaved(
                mesh_device,
                model_args,
                configuration,
                state_dict,
                weight_cache_path,
                layer_num,
                ttnn.bfloat8_b,
            )
        else:
            self.wo_prefill_sweep = self.wo

        if self.use_hf_rope:  # legacy rotary_embedding, not rotary_embedding_hf
            self.rotary_embedding_decode = self._hf_rope_decode_legacy
            self.rotary_embedding_prefill = self._hf_rope_prefill_legacy

        _decode_rope = self.rotary_embedding_decode
        _prefill_rope = self.rotary_embedding_prefill

        def rotary_embedding_decode_wrapped(q, k, rot_mats, current_pos):
            q, k = _decode_rope(q, k, rot_mats, current_pos)
            return self._apply_llama4_query_scale_decode(q, current_pos), k

        def rotary_embedding_prefill_wrapped(q, k, rot_mats):
            q, k = _prefill_rope(q, k, rot_mats)
            pos_tt = self._prefill_position_ids_for_llama4_scale
            return self._apply_llama4_query_scale_prefill(q, pos_tt), k

        self.rotary_embedding_decode = rotary_embedding_decode_wrapped
        self.rotary_embedding_prefill = rotary_embedding_prefill_wrapped
        self._prefill_position_ids_for_llama4_scale: ttnn.Tensor | None = None

    def get_prefill_qkv_input_mem_config(self, full_seq_len: int) -> ttnn.MemoryConfig | None:
        """BLOCK-sharded L1 in0 for QKV when norm and matmul share the same shard grid."""
        if not _qkv_block_sharding_enabled(self.args, int(full_seq_len)):
            return None
        return ministral_prefill_block_shard_mem_cfg(self.args, int(full_seq_len))

    def _prefill_kv_fill_input_mem_config(self) -> ttnn.MemoryConfig:
        """L1 interleaved K/V tiles for fill_cache (UpdateKVCache in0)."""
        if os.environ.get("TT_MINISTRAL3_KV_PREFILL_L1", "1").strip().lower() in ("0", "false", "no"):
            return ttnn.DRAM_MEMORY_CONFIG
        return ttnn.L1_MEMORY_CONFIG

    def _hf_rope_prefill_legacy(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        """Legacy HF prefill rope (full TILE cos/sin from TtMinistral3RotaryEmbedding)."""
        q_heads_1QSD_pre_rot = _typecast_if_needed(q_heads_1QSD_pre_rot, ttnn.bfloat16)
        k_heads_1KSD_pre_rot = _typecast_if_needed(k_heads_1KSD_pre_rot, ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding(q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1])
        k_heads_1KSD = ttnn.experimental.rotary_embedding(k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1])
        return q_heads_1QSD, k_heads_1KSD

    def _hf_rope_decode_legacy(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        """Legacy HF decode rope (per-batch slices from HfRotarySetupOld.get_rot_mats)."""
        cos, sin = rot_mats[0], rot_mats[1]
        B_iter = self.batch_size_per_device_group

        q_heads_pre_rot_1BQD = _typecast_if_needed(q_heads_pre_rot_1BQD, ttnn.bfloat16)
        k_heads_pre_rot_1BKD = _typecast_if_needed(k_heads_pre_rot_1BKD, ttnn.bfloat16)

        q_out_mem = q_heads_pre_rot_1BQD.memory_config()
        k_out_mem = k_heads_pre_rot_1BKD.memory_config()

        if B_iter == 1:
            q_b = q_heads_pre_rot_1BQD[:, 0:1, :, :]
            k_b = k_heads_pre_rot_1BKD[:, 0:1, :, :]
            cos_b = cos[:, :, 0:1, :]
            sin_b = sin[:, :, 0:1, :]
            q_heads_1BQD = ttnn.experimental.rotary_embedding(q_b, cos_b, sin_b, 0)
            k_heads_1BKD = ttnn.experimental.rotary_embedding(k_b, cos_b, sin_b, 0)
            if q_heads_1BQD.memory_config() != q_out_mem:
                q_repacked = ttnn.to_memory_config(q_heads_1BQD, q_out_mem)
                if q_repacked is not q_heads_1BQD:
                    ttnn.deallocate(q_heads_1BQD)
                q_heads_1BQD = q_repacked
            if k_heads_1BKD.memory_config() != k_out_mem:
                k_repacked = ttnn.to_memory_config(k_heads_1BKD, k_out_mem)
                if k_repacked is not k_heads_1BKD:
                    ttnn.deallocate(k_heads_1BKD)
                k_heads_1BKD = k_repacked
        else:
            q_il_parts = []
            k_il_parts = []
            for b in range(B_iter):
                q_b = q_heads_pre_rot_1BQD[:, b : b + 1, :, :]
                k_b = k_heads_pre_rot_1BKD[:, b : b + 1, :, :]
                cos_b = cos[:, :, b : b + 1, :]
                sin_b = sin[:, :, b : b + 1, :]
                q_rot = ttnn.experimental.rotary_embedding(q_b, cos_b, sin_b, 0)
                k_rot = ttnn.experimental.rotary_embedding(k_b, cos_b, sin_b, 0)
                q_il_parts.append(ttnn.to_memory_config(q_rot, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16))
                k_il_parts.append(ttnn.to_memory_config(k_rot, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16))
                ttnn.deallocate(q_rot)
                ttnn.deallocate(k_rot)

            q_merged_il = ttnn.concat(q_il_parts, dim=1)
            k_merged_il = ttnn.concat(k_il_parts, dim=1)
            for t in q_il_parts:
                ttnn.deallocate(t)
            for t in k_il_parts:
                ttnn.deallocate(t)

            q_heads_1BQD = ttnn.interleaved_to_sharded(q_merged_il, q_out_mem)
            k_heads_1BKD = ttnn.interleaved_to_sharded(k_merged_il, k_out_mem)
            ttnn.deallocate(q_merged_il)
            ttnn.deallocate(k_merged_il)

        q_heads_1BQD = ttnn.reshape(  # legacy rope pads heads to 32 tiles
            q_heads_1BQD,
            (1, self.batch_size_per_device_group, self.n_local_heads, self.head_dim),
            (1, self.batch_size_per_device_group, 32, self.head_dim),
        )
        k_heads_1BKD = ttnn.reshape(
            k_heads_1BKD,
            (1, self.batch_size_per_device_group, self.n_local_kv_heads, self.head_dim),
            (1, self.batch_size_per_device_group, 32, self.head_dim),
        )
        q_heads_1BQD = q_heads_1BQD[:, :, : self.n_local_heads]
        k_heads_1BKD = k_heads_1BKD[:, :, : self.n_local_kv_heads]
        return q_heads_1BQD, k_heads_1BKD

    def _llama4_scaling_enabled(self) -> bool:
        return self.llama_4_scaling_beta is not None and self.original_max_position_embeddings is not None

    def _llama4_scale_factor_from_positions_ttnn(self, pos_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Llama-4 scale factor in bf16 to avoid fp32 tilize/typecast overhead."""
        orig = float(self.original_max_position_embeddings)
        beta = float(self.llama_4_scaling_beta)
        pos_f = _typecast_if_needed(pos_tt, ttnn.bfloat16)
        ratio = ttnn.divide(pos_f, orig)
        floored = ttnn.floor(ratio)
        log_term = ttnn.log1p(floored)
        scaled = ttnn.mul(log_term, beta)
        return ttnn.add(scaled, 1.0, dtype=ttnn.bfloat16)

    def _reshape_decode_positions(self, current_pos: ttnn.Tensor, batch_dim: int) -> ttnn.Tensor:
        """Return positions with shape ``[1, batch_dim]`` for per-row scale (matches ``q`` batch axis)."""
        sh = tuple(current_pos.shape)
        numel = math.prod(sh) if sh else 0
        if numel == batch_dim:
            return ttnn.reshape(current_pos, (1, batch_dim))
        if numel > batch_dim:
            flat = ttnn.reshape(current_pos, (1, 1, 1, numel))
            sliced = flat[:, :, :, :batch_dim]
            return ttnn.reshape(sliced, (1, batch_dim))
        raise ValueError(f"Ministral Llama-4 scale: current_pos has {numel} elements but q_heads batch is {batch_dim}")

    def _apply_llama4_query_scale_decode(self, q_heads, current_pos):
        if not self._llama4_scaling_enabled():
            return q_heads
        b = int(q_heads.shape[1])
        pos_row = self._reshape_decode_positions(current_pos, b)
        scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_row)
        scale_4d = ttnn.reshape(scale_f, (1, b, 1, 1))
        mul_dtype = q_heads.dtype if q_heads.dtype in (ttnn.bfloat16, ttnn.bfloat8_b) else ttnn.bfloat16
        scale_tt = _typecast_if_needed(scale_4d, mul_dtype)
        out = ttnn.mul(q_heads, scale_tt, dtype=mul_dtype)
        if scale_tt is not scale_4d:
            ttnn.deallocate(scale_tt)
        return out

    def _apply_llama4_query_scale_prefill(self, q_heads, position_ids: ttnn.Tensor | None = None):
        if not self._llama4_scaling_enabled():
            return q_heads
        pos_tt = position_ids
        if pos_tt is None:
            return q_heads

        sh = tuple(q_heads.shape)
        if len(sh) != 4:
            return q_heads
        batch_dim, seq_dim = sh[0], sh[2]
        shp = tuple(pos_tt.shape)
        if len(shp) == 2 and shp[0] == batch_dim and shp[1] == seq_dim:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_tt)
        elif len(shp) == 1 and shp[0] == seq_dim and batch_dim == 1:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(ttnn.reshape(pos_tt, (1, seq_dim)))
        else:
            return q_heads

        scale_4d = ttnn.reshape(scale_f, (batch_dim, 1, seq_dim, 1))
        mul_dtype = q_heads.dtype if q_heads.dtype in (ttnn.bfloat16, ttnn.bfloat8_b) else ttnn.bfloat16
        scale_tt = _typecast_if_needed(scale_4d, mul_dtype)
        out = ttnn.mul(q_heads, scale_tt, dtype=mul_dtype)
        if scale_tt is not scale_4d:
            ttnn.deallocate(scale_tt)
        return out

    def _prefill_bf16_activations_enabled(self) -> bool:
        return ministral_qkv_bf16_activations_enabled()

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids: ttnn.Tensor | None = None,
    ):
        """Prefill with optional device position_ids for Llama-4 Q scaling."""
        x_prefill = x_11SH
        self._prefill_position_ids_for_llama4_scale = position_ids
        old_activation_dtype = self.activation_dtype
        old_ccl_dtype = self.ccl_dtype
        if self._prefill_bf16_activations_enabled():
            # BF16×BFP8 QKV; BF16 KV cache skips prefill K/V BF16=>BFP8 casts before fill_cache.
            self.activation_dtype = ttnn.bfloat16
            self.ccl_dtype = ttnn.bfloat16
        else:
            if x_prefill.dtype != ttnn.bfloat8_b:
                x_prefill = ttnn.typecast(x_prefill, dtype=ttnn.bfloat8_b)
            self.activation_dtype = ttnn.bfloat8_b

        batch_size = x_prefill.shape[0]
        qkv_seq_len = int(x_prefill.shape[-2])
        if batch_size > 1:
            qkv_seq_len *= int(x_prefill.shape[-3]) * batch_size
        qkv_block_in = (
            _qkv_block_sharding_enabled(self.args, qkv_seq_len)
            and x_prefill.memory_config().is_sharded()
            and x_prefill.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        use_qkv_sweep = _qkv_linear_sweep_enabled(self.args, qkv_seq_len) and not qkv_block_in
        if use_qkv_sweep:
            x_prefill = _prepare_qkv_linear_sweep_input(x_prefill)
        use_wo_sweep = _wo_linear_sweep_enabled(self.args, qkv_seq_len, qkv_seq_len, self.mesh_device)

        try:
            with (
                _qkv_linear_sweep_program_config_override(self.args, qkv_seq_len, block_sharded_in0=qkv_block_in),
                _qkv_block_shard_linear_patch(self, qkv_seq_len) if qkv_block_in else nullcontext(),
                _wo_linear_sweep_runtime_patch(self) if use_wo_sweep else nullcontext(),
                _skip_identity_typecast(),
            ):
                return super().forward_prefill(
                    x_prefill,
                    rot_mats,
                    user_id=user_id,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start_idx,
                    kv_cache=kv_cache,
                )
        finally:
            self.activation_dtype = old_activation_dtype
            self.ccl_dtype = old_ccl_dtype
            self._prefill_position_ids_for_llama4_scale = None

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids=None,
    ):
        if mode == Mode.PREFILL:
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        return super().forward(
            x,
            current_pos,
            rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )


# ModelArgs.get_state_dict_prefix(self.__class__.__name__, layer_num) expects "Attention"
# (checkpoint keys are layers.{i}.attention.*, not TtMinistralAttention).
TtMinistralAttention.__name__ = "Attention"
TtMinistralAttention.__qualname__ = "Attention"


__all__ = ["TtMinistralAttention"]
