# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standard pre-norm transformer decoder layer for GLM-4.7-REAP-218B.

Layers 0-2: dense MLP (intermediate_size=12288)
Layers 3-91: MoE (96 routed experts EP=32 + 1 shared expert TP=8)

Attention uses standard GQA (96Q/8KV heads, NOT MLA).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import time

from loguru import logger

import ttnn

from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.layer_weights import DecoderLayerTTWeights
from models.demos.glm4_moe.tt.moe_tt import (
    Glm4MoeMoERuntime,
    moe_sparse_experts_forward_tt,
    moe_topk_tt,
    shared_expert_forward_tt,
    _env_bool,
    _get_mesh_shape,
    _moe_sparse_tokens_multiple,
    _parse_math_fidelity,
)
from models.demos.glm4_moe.tt.attention_tt import Glm4MoeAttention, _simple_all_reduce, _simple_all_reduce_host


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tp_cluster_axis(device: Any) -> int | None:
    """Return the mesh axis used for TP sharding.

    Galaxy TG Mesh(8,4): TP=axis 0 (rows=8).
    T3K Mesh(1,8): TP=axis 1 (cols=8).
    """
    if device.__class__.__name__ != "MeshDevice":
        return None
    mesh_rows, mesh_cols = _get_mesh_shape(device)
    if mesh_rows > 1 and mesh_cols > 1:
        # 2D mesh (TG): TP is axis 0 (rows=8)
        return 0
    if mesh_cols > 1:
        return 1
    if mesh_rows > 1:
        return 0
    return None


_prefill_prog_cfg_cache: dict[str, object] = {}  # module-level cache for prefill program configs


def _make_prefill_matmul_program_config(seq_len: int, K: int, N: int, label: str = ""):
    """Build explicit MatmulMultiCoreReuseMultiCastProgramConfig for prefill on TG mesh.

    Without this, ttnn.linear auto-selects a program config that hangs on TG mesh.
    Pattern from tt_transformers/tt/model_config.py (Llama Galaxy).
    """
    cache_key = f"{label}_{seq_len}_{K}_{N}"
    if cache_key in _prefill_prog_cfg_cache:
        return _prefill_prog_cfg_cache[cache_key]

    tile_size = 32
    grid_rows = 8
    grid_cols = 8

    M_tiles = seq_len // tile_size
    N_tiles = math.ceil(N / tile_size)

    per_core_M = max(1, math.ceil(M_tiles / grid_rows))
    per_core_N = max(1, math.ceil(N_tiles / grid_cols))

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_cols, grid_rows),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    _prefill_prog_cfg_cache[cache_key] = cfg
    return cfg


def _sharded_rms_norm(x: ttnn.Tensor, norm_module: Any, hidden_size: int, num_cores: int = 8, worker_core_range: Any = None) -> ttnn.Tensor:
    """Width-sharded multi-core RMSNorm to avoid L1 overflow for large hidden sizes.

    For hidden_size=5120, single-core RMSNorm overflows L1 by ~13.6 KB (1.51 MB vs 1.43 MB limit).
    Spreading across 8 cores reduces per-core memory to ~100 KB — well within L1.

    For prefill (large sequence lengths), auto-scales num_cores up so each core's
    shard fits within L1. The rms_norm kernel needs ~6 bytes per element in CBs
    (input + output + intermediates).

    Uses ttnn.to_memory_config for sharding (works on TG mesh; interleaved_to_sharded hangs).

    Args:
        worker_core_range: Optional CoreRangeSet to use for sharding (e.g. worker grid
            when DRAM prefetcher is active). When provided, num_cores is derived from it
            and the core range is used directly instead of building CoreRange((0,0)..(N-1,0)).
    """
    input_shape = [int(d) for d in x.shape]  # save for shape restoration
    h_logical = int(x.shape[-2])  # logical height (may NOT be tile-padded)
    h = ((h_logical + 31) // 32) * 32  # round up to tile boundary (32)

    # When worker_core_range is explicitly provided, derive num_cores from it.
    if worker_core_range is not None:
        num_cores = worker_core_range.num_cores()

    # Auto-scale num_cores for large sequences to fit within L1.
    # L1 budget ~1.43 MB. rms_norm CBs ≈ 6 bytes per element (3x BF16 for in/out/scratch).
    # Per-core shard: h * (hidden/num_cores) * 6 bytes must fit in L1.
    L1_BUDGET = 1_400_000  # conservative (actual 1,499,136)
    BYTES_PER_ELEM = 6  # empirical: ~3x BF16 for rms_norm CBs
    tiles_w_total = hidden_size // 32
    while num_cores < 64:
        shard_w_candidate = hidden_size // num_cores
        per_core_bytes = h * shard_w_candidate * BYTES_PER_ELEM
        if per_core_bytes <= L1_BUDGET and tiles_w_total % num_cores == 0:
            break
        # Try doubling cores
        next_cores = num_cores * 2
        if tiles_w_total % next_cores != 0:
            # Find next valid divisor
            for c in range(num_cores + 1, 65):
                if tiles_w_total % c == 0:
                    next_cores = c
                    break
            else:
                break
        num_cores = next_cores

    tile_h = h // 32  # number of tile rows
    tiles_w = hidden_size // 32  # total tiles in width
    tiles_per_core = tiles_w // num_cores  # tiles per core

    # Subblock width: largest value <= 4 that evenly divides tiles_per_core
    subblock_w = 4
    while subblock_w > 0:
        if tiles_per_core % subblock_w == 0:
            break
        subblock_w -= 1

    # Width-sharded memory config: each core gets [h, hidden/num_cores] elements
    shard_w = hidden_size // num_cores
    if worker_core_range is not None:
        core_range = worker_core_range
    else:
        core_range = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))
        ])
    shard_spec = ttnn.ShardSpec(core_range, [h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
    )

    # Matching program config
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores, 1],
        subblock_w=subblock_w,
        block_h=tile_h,
        block_w=tiles_per_core,
        inplace=False,
    )

    # Shard input from DRAM interleaved to L1 width-sharded
    x_sharded = ttnn.to_memory_config(x, sharded_mem_cfg)

    # Run sharded RMSNorm
    result = ttnn.rms_norm(
        x_sharded,
        epsilon=norm_module.eps,
        weight=norm_module.weight,
        program_config=program_config,
        compute_kernel_config=norm_module.compute_kernel_config_hifi2,
    )
    ttnn.deallocate(x_sharded, force=False)

    # Convert back to interleaved DRAM
    result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)

    # Restore original logical shape — sharding pads batch to tile boundary (e.g. 1→32)
    # and sharded_to_interleaved does NOT restore the logical shape.
    if int(result.shape[-2]) != h_logical:
        result = ttnn.slice(result, starts=[0, 0, 0, 0], ends=input_shape)

    return result


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Glm4MoeDecoderLayer:
    """Single decoder layer for GLM-4.7-REAP-218B.

    Pre-norm architecture:
    1. RMSNorm -> GQA Attention -> residual
    2. RMSNorm -> MLP (dense or MoE) -> residual
    """

    def __init__(
        self,
        *,
        mesh_device: Any,
        tt_ccl: Any | None,
        hparams: Glm4MoeHParams,
        layer_weights: DecoderLayerTTWeights,
        configuration: Any | None = None,
        paged_attention_config: Any | None = None,
        moe_runtime: Glm4MoeMoERuntime | None = None,
    ):
        self.device = mesh_device
        self.tt_ccl = tt_ccl
        self.hparams = hparams
        self.layer_weights = layer_weights
        self.configuration = configuration
        self.paged_attention_config = paged_attention_config
        self.moe_runtime = moe_runtime
        self.layer_idx = int(layer_weights.layer_idx)

        self.is_dense_layer = self.layer_idx < int(hparams.first_k_dense_replace)
        self.has_moe = not self.is_dense_layer and layer_weights.moe is not None

        # Attention module (GQA).
        logger.info("  [DEBUG L{}] creating attention", layer_weights.layer_idx)
        self.attention = Glm4MoeAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            layer_weights=layer_weights,
            hparams=hparams,
            configuration=configuration,
            paged_attention_config=paged_attention_config,
        )
        logger.info("  [DEBUG L{}] attention created", layer_weights.layer_idx)

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        mode: str = "decode",
        active_batch: int | None = None,
        global_cb=None,
        sub_device_id=None,
        prefetch_qkv_pc=None,
        prefetch_oproj_pc=None,
        prefetch_qkv_in_mc=None,
        prefetch_qkv_out_mc=None,
        prefetch_oproj_in_mc=None,
        prefetch_oproj_out_mc=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ) -> ttnn.Tensor:
        """Forward pass for one decoder layer.

        Args:
            x: Hidden states [1,1,B,hidden] (decode) or [1,1,S,hidden] (prefill).
            current_pos: Position indices for KV cache update.
            rot_mats: RoPE rotation matrices (cos, sin, trans_mat or tuple).
            page_table: Paged KV cache page table.
            kv_cache: [cache_k, cache_v] for this layer.
            mode: "decode" or "prefill".
            active_batch: True logical batch size (avoids relying on x.shape which
                may be tile-padded by ttnn ops).

        Returns:
            Updated hidden states, same shape as input.
        """
        import sys as _sys
        _dbg_fwd = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
        _lid_fwd = getattr(self.layer_weights, 'layer_idx', '?')
        if _dbg_fwd:
            print(f"  [DEBUG DL{_lid_fwd}] forward entry ({mode}) sync ...", flush=True, file=_sys.stderr)
            ttnn.synchronize_device(self.device)
            print(f"  [DEBUG DL{_lid_fwd}] forward entry ({mode}) sync OK", flush=True, file=_sys.stderr)

        if mode == "decode":
            return self._forward_decode(x, current_pos, rot_mats, page_table, kv_cache,
                                        active_batch=active_batch, global_cb=global_cb, sub_device_id=sub_device_id,
                                        prefetch_qkv_pc=prefetch_qkv_pc, prefetch_oproj_pc=prefetch_oproj_pc,
                                        prefetch_qkv_in_mc=prefetch_qkv_in_mc, prefetch_qkv_out_mc=prefetch_qkv_out_mc,
                                        prefetch_oproj_in_mc=prefetch_oproj_in_mc, prefetch_oproj_out_mc=prefetch_oproj_out_mc)
        else:
            return self._forward_prefill(x, current_pos, rot_mats, page_table, kv_cache,
                                          chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx)

    def _forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        active_batch: int | None = None,
        global_cb=None,
        sub_device_id=None,
        prefetch_qkv_pc=None,
        prefetch_oproj_pc=None,
        prefetch_qkv_in_mc=None,
        prefetch_qkv_out_mc=None,
        prefetch_oproj_in_mc=None,
        prefetch_oproj_out_mc=None,
    ) -> ttnn.Tensor:
        """Decode forward: batch in dim=2, seq_len=1 per token."""
        w = self.layer_weights
        device = self.device
        hparams = self.hparams

        tp_axis = _tp_cluster_axis(device)
        tp_enabled = tp_axis is not None

        # Compute kernel config for MLP.
        mlp_fidelity = _parse_math_fidelity(
            os.environ.get("GLM4_MOE_MLP_FIDELITY", ""),
            default=ttnn.MathFidelity.LoFi,
        )
        mlp_approx = os.environ.get("GLM4_MOE_MLP_APPROX", "1").strip() != "0"
        _fp32_acc_decode = os.environ.get("GLM4_MOE_FP32_ACC", "0").strip() != "0"
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=mlp_fidelity,
            math_approx_mode=mlp_approx,
            fp32_dest_acc_en=_fp32_acc_decode,
            packer_l1_acc=True,
        )

        _COMP = os.environ.get("GLM4_MOE_PROFILE_COMPONENTS", "0").strip() not in ("0", "")
        _lid = getattr(w, 'layer_idx', '?')

        def _cs():
            if not _COMP: return None
            try:
                ttnn.synchronize_device(device)
            except RuntimeError:
                return None  # Inside trace capture — skip profiling
            return time.perf_counter_ns()

        def _ce(t0, comp):
            if t0 is None: return
            try:
                ttnn.synchronize_device(device)
            except RuntimeError:
                return  # Inside trace capture — skip profiling
            ms = (time.perf_counter_ns() - t0) / 1e6
            if _lid in (0, 45, 91):  # Sample 3 layers
                logger.info("TTCOMP L{} {} {:.1f}ms", _lid, comp, ms)

        # ---- Pre-attention norm ----
        # When prefetcher is active (global_cb set), use worker core range to avoid
        # overlapping with sender columns (cols 6-7).
        _norm_worker_cr = None
        _worker_scg = None  # sub_core_grids for eltwise ops (full worker grid)
        if global_cb is not None:
            # 5 cores for hidden=5120 (1024 elems/core), placed at cols 0-4 row 0
            # Must be within worker SubDevice (cols 0-5), avoiding sender cols 6,7.
            _norm_worker_cr = ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))
            ])
            # Full worker grid for eltwise ops (add, multiply, etc.)
            _worker_scg = ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))
            ])
        t0 = _cs()
        h = _sharded_rms_norm(x, w.input_layernorm, hparams.hidden_size, worker_core_range=_norm_worker_cr)
        _ce(t0, "pre_attn_norm")

        # ---- GQA Attention ----
        t0 = _cs()
        attn_out = self.attention.forward(
            h, current_pos, rot_mats, mode="decode", page_table=page_table, kv_cache=kv_cache,
            active_batch=active_batch, global_cb=global_cb, sub_device_id=sub_device_id,
            prefetch_qkv_pc=prefetch_qkv_pc, prefetch_oproj_pc=prefetch_oproj_pc,
            prefetch_qkv_in_mc=prefetch_qkv_in_mc, prefetch_qkv_out_mc=prefetch_qkv_out_mc,
            prefetch_oproj_in_mc=prefetch_oproj_in_mc, prefetch_oproj_out_mc=prefetch_oproj_out_mc,
        )
        _ce(t0, "attention")

        # ---- Residual ----
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=_worker_scg)
        ttnn.deallocate(attn_out, force=False)

        # ---- Pre-MLP norm ----
        residual = x
        t0 = _cs()
        h = _sharded_rms_norm(x, w.post_attention_layernorm, hparams.hidden_size, worker_core_range=_norm_worker_cr)
        _ce(t0, "pre_mlp_norm")

        # ---- MLP ----
        t0 = _cs()
        if self.is_dense_layer:
            mlp_out = self._dense_mlp_forward(
                h, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down,
                compute_kernel_config=mlp_compute_kernel_config,
            )
            ttnn.deallocate(h, force=False)
            if tp_enabled:
                mlp_out = _simple_all_reduce(
                    mlp_out, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl,
                    subdevice_id=sub_device_id,
                )
            _ce(t0, "dense_mlp")
        else:
            mlp_out = self._moe_forward(
                h, compute_kernel_config=mlp_compute_kernel_config,
                tp_axis=tp_axis, tp_enabled=tp_enabled,
                sub_device_id=sub_device_id,
                worker_scg=_worker_scg,
            )
            _ce(t0, "moe")

        # ---- Residual ----
        res_shape = [int(d) for d in residual.shape]
        mlp_shape = [int(d) for d in mlp_out.shape]
        if mlp_shape != res_shape:
            res_vol = 1
            mlp_vol = 1
            for d in res_shape:
                res_vol *= d
            for d in mlp_shape:
                mlp_vol *= d
            if mlp_vol == res_vol:
                mlp_out = ttnn.reshape(mlp_out, res_shape, mlp_shape)
            else:
                mlp_out = ttnn.slice(mlp_out, starts=[0] * len(res_shape), ends=res_shape)

        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=_worker_scg)
        ttnn.deallocate(mlp_out, force=False)
        ttnn.deallocate(residual, force=False)
        return x

    def _forward_prefill(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        chunk_page_table: ttnn.Tensor = None,
        chunk_start_idx: int = None,
    ) -> ttnn.Tensor:
        """Prefill forward: batch=1, seq_len in dim=2."""
        w = self.layer_weights
        device = self.device
        hparams = self.hparams

        import sys as _sys
        _dbg = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
        _dbg_prefill = os.environ.get("GLM4_MOE_DEBUG_PREFILL", "0") != "0"
        _lid = getattr(w, 'layer_idx', '?')
        def _dlsync(label):
            if _dbg:
                print(f"  [DEBUG DL{_lid} PREFILL] {label} ...", flush=True, file=_sys.stderr)
                ttnn.synchronize_device(device)
                print(f"  [DEBUG DL{_lid} PREFILL] {label} OK", flush=True, file=_sys.stderr)

        # Prefill timing debug (lighter than DEBUG_SYNC — no device sync, just wall-clock)
        _pf_times = {}
        def _pf_mark(label):
            if _dbg_prefill:
                _pf_times[label] = time.time()

        _pf_mark("start")

        tp_axis = _tp_cluster_axis(device)
        tp_enabled = tp_axis is not None

        # FP32 dest accumulation reduces BF8 rounding error through 92 layers
        _fp32_acc = os.environ.get("GLM4_MOE_FP32_ACC", "0").strip() != "0"
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=_fp32_acc,
            packer_l1_acc=True,
        )

        if _dbg_prefill:
            print(f"    [DL{_lid} PREFILL] input x.shape={list(x.shape)}, "
                  f"is_dense={self.is_dense_layer}", flush=True, file=_sys.stderr)

        _dlsync("before rms_norm")

        # ---- Pre-attention norm ----
        _pf_mark("pre_norm_start")
        h = ttnn.rms_norm(
            x,
            epsilon=w.input_layernorm.eps,
            weight=w.input_layernorm.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _pf_mark("pre_norm_end")

        _dlsync("after rms_norm, before attention")

        # ---- GQA Attention ----
        _pf_mark("attn_start")
        attn_out = self.attention.forward(
            h, current_pos, rot_mats, mode="prefill", page_table=page_table, kv_cache=kv_cache,
            chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
        )
        _pf_mark("attn_end")

        # ---- Residual ----
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, force=False)

        _dlsync("after attention + residual")

        # ---- Pre-MLP norm (DRAM-interleaved for prefill, same as pre-attn) ----
        residual = x
        _pf_mark("post_norm_start")
        h = ttnn.rms_norm(
            x,
            epsilon=w.post_attention_layernorm.eps,
            weight=w.post_attention_layernorm.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _pf_mark("post_norm_end")

        _dlsync("after post_attn_norm, before MLP")

        # ---- MLP ----
        _pf_mark("mlp_start")
        seq_len = x.shape[2]
        _is_tg = tp_axis is not None and tp_axis == 0  # TG mesh: TP on axis 0
        if self.is_dense_layer:
            # On TG, provide explicit program configs to avoid auto-config hang
            _gu_pc = None
            _dn_pc = None
            if _is_tg:
                _gu_pc = _make_prefill_matmul_program_config(
                    seq_len, int(w.w_mlp_gate.shape[2]), int(w.w_mlp_gate.shape[3]),
                    label=f"mlp_gate_up_L{_lid}",
                )
                _dn_pc = _make_prefill_matmul_program_config(
                    seq_len, int(w.w_mlp_down.shape[2]), int(w.w_mlp_down.shape[3]),
                    label=f"mlp_down_L{_lid}",
                )
            mlp_out = self._dense_mlp_forward(
                h, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down,
                compute_kernel_config=mlp_compute_kernel_config,
                gate_up_program_config=_gu_pc,
                down_program_config=_dn_pc,
            )
            ttnn.deallocate(h, force=False)
            if tp_enabled:
                mlp_out = _simple_all_reduce(
                    mlp_out, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl if not _is_tg else None,
                    impl="rs_ag" if _is_tg else None,
                )
        else:
            mlp_out = self._moe_forward(
                h, compute_kernel_config=mlp_compute_kernel_config,
                tp_axis=tp_axis, tp_enabled=tp_enabled,
                is_tg_prefill=_is_tg,
            )
        _pf_mark("mlp_end")

        _dlsync("after MLP")

        # ---- Residual (prefill path) ----
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out, force=False)
        ttnn.deallocate(residual, force=False)

        _pf_mark("end")

        if _dbg_prefill:
            def _dt(a, b):
                if a in _pf_times and b in _pf_times:
                    return f"{(_pf_times[b] - _pf_times[a])*1000:.1f}ms"
                return "N/A"
            total = _dt("start", "end")
            print(f"    [DL{_lid} PREFILL] output x.shape={list(x.shape)}, "
                  f"total={total}, pre_norm={_dt('pre_norm_start','pre_norm_end')}, "
                  f"attn={_dt('attn_start','attn_end')}, "
                  f"post_norm={_dt('post_norm_start','post_norm_end')}, "
                  f"mlp={_dt('mlp_start','mlp_end')}", flush=True, file=_sys.stderr)

        return x

    # -----------------------------------------------------------------------
    # Dense MLP (layers 0-2): gate_up_SiLU_down, TP=8 column/row parallel
    # -----------------------------------------------------------------------

    def _dense_mlp_forward(
        self,
        x: ttnn.Tensor,
        w_gate: ttnn.Tensor,
        w_up: ttnn.Tensor,
        w_down: ttnn.Tensor,
        compute_kernel_config: Any | None = None,
        gate_up_program_config: Any | None = None,
        down_program_config: Any | None = None,
    ) -> ttnn.Tensor:
        """Dense MLP: gate_proj * SiLU(up_proj) -> down_proj.

        gate/up are column-parallel (sharded output dim by TP=8).
        down is row-parallel (sharded input dim by TP=8, needs all_reduce).
        """
        kwargs: dict[str, Any] = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config

        gate_kwargs = {**kwargs}
        if gate_up_program_config is not None:
            gate_kwargs["program_config"] = gate_up_program_config
        gate = ttnn.linear(x, w_gate, **gate_kwargs)
        up = ttnn.linear(x, w_up, **gate_kwargs)
        x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate, force=False)
        ttnn.deallocate(up, force=False)
        down_kwargs = {**kwargs}
        if down_program_config is not None:
            down_kwargs["program_config"] = down_program_config
        out = ttnn.linear(x_ff, w_down, **down_kwargs)
        ttnn.deallocate(x_ff, force=False)
        return out

    # -----------------------------------------------------------------------
    # MoE (layers 3-91): routing + shared expert + routed experts
    # -----------------------------------------------------------------------

    def _moe_forward(
        self,
        x: ttnn.Tensor,
        compute_kernel_config: Any | None = None,
        tp_axis: int | None = None,
        tp_enabled: bool = False,
        sub_device_id: Any = None,
        is_tg_prefill: bool = False,
        worker_scg: Any = None,
    ) -> ttnn.Tensor:
        """MoE forward: shared expert + routed experts -> sum.

        Shared expert: TP=8 gate-up-SiLU-down MLP.
        Routed experts: top-8 routing + sparse_matmul across EP=32.
        """
        w = self.layer_weights
        device = self.device
        hparams = self.hparams
        moe_runtime = self.moe_runtime
        moe_w = w.moe

        tokens = int(x.shape[2])

        # Pad tokens for sparse expert alignment.
        pad_tokens = 0
        if moe_runtime is not None:
            sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
            pad_tokens = (-tokens) % sparse_multiple
            if pad_tokens:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)

        # On TG mesh prefill, build explicit program configs for matmuls
        _shared_gu_pc = None
        _shared_dn_pc = None
        _router_pc = None
        _lid = getattr(w, 'layer_idx', '?')
        if is_tg_prefill:
            _t = tokens + pad_tokens
            if w.w_mlp_gate_up is not None:
                _shared_gu_pc = _make_prefill_matmul_program_config(
                    _t, int(w.w_mlp_gate_up.shape[2]), int(w.w_mlp_gate_up.shape[3]),
                    label=f"shared_gate_up_L{_lid}",
                )
            elif w.w_mlp_gate is not None:
                _shared_gu_pc = _make_prefill_matmul_program_config(
                    _t, int(w.w_mlp_gate.shape[2]), int(w.w_mlp_gate.shape[3]),
                    label=f"shared_gate_L{_lid}",
                )
            if w.w_mlp_down is not None:
                _shared_dn_pc = _make_prefill_matmul_program_config(
                    _t, int(w.w_mlp_down.shape[2]), int(w.w_mlp_down.shape[3]),
                    label=f"shared_down_L{_lid}",
                )
            if moe_w is not None and moe_w.w_gate is not None:
                _router_pc = _make_prefill_matmul_program_config(
                    _t, int(moe_w.w_gate.shape[2]), int(moe_w.w_gate.shape[3]),
                    label=f"router_L{_lid}",
                )

        moe_decode_mc = getattr(moe_runtime, "decode_memory_config", ttnn.DRAM_MEMORY_CONFIG) if moe_runtime else ttnn.DRAM_MEMORY_CONFIG

        # ---- Shared expert (dense MLP, runs on all tokens) ----
        shared_out_partial = shared_expert_forward_tt(
            x=x,
            w_gate=w.w_mlp_gate,
            w_up=w.w_mlp_up,
            w_down=w.w_mlp_down,
            w_gate_up=w.w_mlp_gate_up,
            memory_config=moe_decode_mc,
            compute_kernel_config=compute_kernel_config,
            gate_up_program_config=_shared_gu_pc,
            down_program_config=_shared_dn_pc,
        )
        # NOTE: No TP reduce here — fused with EP reduce below

        # ---- Routed experts (skip EP reduce — will fuse with shared TP reduce) ----
        topk_weights, topk_indices = moe_topk_tt(
            x=x,
            moe_w=moe_w,
            hparams=hparams,
            compute_kernel_config=compute_kernel_config,
            router_program_config=_router_pc,
        )

        routed_out_partial = moe_sparse_experts_forward_tt(
            device=device,
            hidden_states=x,
            topk_expert_indices=topk_indices,
            topk_expert_weights=topk_weights,
            moe_w=moe_w,
            rt=moe_runtime,
            memory_config=moe_decode_mc,
            skip_final_reduce=True,
        )

        # Prefill on TG: use device-side rs_ag reduce (reduce_scatter + all_gather)
        _reduce_impl = "rs_ag" if is_tg_prefill else None

        # ---- Fused reduce: scale shared by 1/DP, combine, single reduce ----
        fuse_reduce = _env_bool("GLM4_MOE_FUSE_SHARED_EP_REDUCE", default=True)
        if fuse_reduce and tp_enabled and tp_axis is not None:
            mesh_shape = _get_mesh_shape(device)
            num_dp = mesh_shape[1]  # DP = cols = 4 for Galaxy TG
            if num_dp > 1:
                shared_out_partial = ttnn.mul(shared_out_partial, 1.0 / num_dp,
                                              memory_config=moe_decode_mc, sub_core_grids=worker_scg)

            combined = ttnn.add(shared_out_partial, routed_out_partial,
                                memory_config=moe_decode_mc, sub_core_grids=worker_scg)
            ttnn.deallocate(shared_out_partial, force=False)
            ttnn.deallocate(routed_out_partial, force=False)

            # EP reduce: 2-step device all_reduce or host-side fallback
            _ep_reduce_device = _env_bool("GLM4_MOE_EP_REDUCE_DEVICE", default=False)
            if _ep_reduce_device:
                # 2-step device-side: axis=0 (8-way TP) then axis=1 (4-way DP)
                # During prefill, use rs_ag (no CCL handle) instead of ccl-based reduce
                _fused_ccl = self.tt_ccl if not is_tg_prefill else None
                _fused_impl = _reduce_impl if is_tg_prefill else None
                mlp_out = _simple_all_reduce(
                    combined, device, cluster_axis=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=_fused_ccl,
                    impl=_fused_impl,
                    subdevice_id=sub_device_id,
                )
                mesh_shape2 = _get_mesh_shape(device)
                if mesh_shape2[1] > 1:
                    mlp_out = _simple_all_reduce(
                        mlp_out, device, cluster_axis=1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ccl=_fused_ccl,
                        impl=_fused_impl,
                        subdevice_id=sub_device_id,
                    )
            else:
                # Host-side 32-way sum fallback
                _orig_dtype = combined.dtype
                dev_tensors = ttnn.get_device_tensors(combined)
                host_sum = ttnn.to_torch(dev_tensors[0].cpu())
                for i in range(1, len(dev_tensors)):
                    host_sum = host_sum + ttnn.to_torch(dev_tensors[i].cpu())
                ttnn.deallocate(combined, force=False)
                mlp_out = ttnn.from_torch(
                    host_sum,
                    device=device,
                    dtype=_orig_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
        else:
            # Fallback: separate reduces (original behavior)
            if tp_enabled and tp_axis is not None:
                shared_out_partial = _simple_all_reduce(
                    shared_out_partial, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl if not is_tg_prefill else None,
                    impl=_reduce_impl,
                    subdevice_id=sub_device_id,
                )
            # EP reduce for routed experts
            _ep_reduce_device2 = _env_bool("GLM4_MOE_EP_REDUCE_DEVICE", default=False)
            if _ep_reduce_device2 and device.__class__.__name__ == "MeshDevice":
                # 2-step device-side: axis=0 (8-way TP) then axis=1 (4-way DP)
                # During prefill, use rs_ag (no CCL handle) instead of ccl-based reduce
                _ep_ccl = self.tt_ccl if not is_tg_prefill else None
                _ep_impl = _reduce_impl if is_tg_prefill else None
                routed_out_partial = _simple_all_reduce(
                    routed_out_partial, device, cluster_axis=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=_ep_ccl,
                    impl=_ep_impl,
                    subdevice_id=sub_device_id,
                )
                mesh_shape3 = _get_mesh_shape(device)
                if mesh_shape3[1] > 1:
                    routed_out_partial = _simple_all_reduce(
                        routed_out_partial, device, cluster_axis=1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ccl=_ep_ccl,
                        impl=_ep_impl,
                        subdevice_id=sub_device_id,
                    )
            elif device.__class__.__name__ == "MeshDevice":
                # Host-side 32-way sum fallback
                num_devices = device.get_num_devices()
                if num_devices > 1:
                    _orig_dtype2 = routed_out_partial.dtype
                    dev_tensors2 = ttnn.get_device_tensors(routed_out_partial)
                    host_sum2 = ttnn.to_torch(dev_tensors2[0].cpu())
                    for i in range(1, len(dev_tensors2)):
                        host_sum2 = host_sum2 + ttnn.to_torch(dev_tensors2[i].cpu())
                    ttnn.deallocate(routed_out_partial, force=False)
                    routed_out_partial = ttnn.from_torch(
                        host_sum2,
                        device=device,
                        dtype=_orig_dtype2,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                    )
            mlp_out = ttnn.add(shared_out_partial, routed_out_partial,
                               memory_config=moe_decode_mc, sub_core_grids=worker_scg)
            ttnn.deallocate(shared_out_partial, force=False)
            ttnn.deallocate(routed_out_partial, force=False)

        # Slice back to real token count if padded.
        if pad_tokens:
            mlp_out = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])

        return mlp_out
