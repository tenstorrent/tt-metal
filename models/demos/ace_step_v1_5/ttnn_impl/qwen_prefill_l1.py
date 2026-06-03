# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE Qwen3 prefill: route activations through L1 (LoFi + ``bfloat8_b`` via :mod:`math_perf_env`).

``tt_transformers`` defaults prefill residuals and matmul ``in0`` to DRAM. Tracy reports large
``MinimalMatmulDeviceOperation`` / ``MatmulDeviceOperation (in0:dram_interleaved)`` buckets for
conditioning; this module patches the loaded Qwen stack so prefill uses ``L1_MEMORY_CONFIG``
without editing upstream ``tt_transformers`` sources.
"""

from __future__ import annotations

import functools
import math
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import ttnn
from models.tt_transformers.tt.common import Mode

from .math_perf_env import (
    ace_step_build_prefill_block_sharded_norm_config,
    ace_step_encoder_matmul_program_config,
    ace_step_init_cond_rmsnorm_compute_kernel_config,
    ace_step_linear_l1_memory_config,
    ace_step_prefill_block_sharded_norm_enabled,
    ace_step_rms_norm_block_sharded,
)

# ModelArgs getters that return ``DRAM_MEMORY_CONFIG`` for ``Mode.PREFILL``.
_PREFILL_DRAM_MEMCFG_GETTERS: tuple[str, ...] = (
    "get_residual_mem_config",
    "get_mlp_input_mem_config",
    "get_mlp_ff1_3_mem_config",
    "get_mlp_ff2_mem_config",
    "get_mlp_output_mem_config",
    "get_attn_input_mem_config",
    "get_attn_qkv_mm_mem_config",
    "get_attn_qkv_all_reduce_output_mem_config",
    "get_attn_create_head_input_mem_config",
    "get_attn_create_head_output_mem_config",
    "get_attn_sdpa_output_mem_config",
    "get_attn_concat_heads_output_mem_config",
    "get_attn_all_gather_output_mem_config",
    "get_attn_wo_output_mem_config",
    "get_attn_dense_output_mem_config",
    "get_attn_all_reduce_output_mem_config",
    "get_attn_gather_users_mem_config",
)


def _is_dram_mc(mc: Any, dram_mc: Any) -> bool:
    return mc is not None and dram_mc is not None and mc == dram_mc


def _ensure_l1(ttnn_module: Any, tensor: Any, *, l1_mc: Any) -> Any:
    if tensor is None or l1_mc is None or not hasattr(ttnn_module, "to_memory_config"):
        return tensor
    if tensor.memory_config() == l1_mc:
        return tensor
    return ttnn_module.to_memory_config(tensor, l1_mc)


def _ensure_dram_interleaved(ttnn_module: Any, tensor: Any, *, dram_mc: Any) -> Any:
    """Mcast matmul sweeps use DRAM interleaved weights; ``create_dram_sharded_mem_config`` weights crash CB setup."""
    if tensor is None or dram_mc is None or not hasattr(ttnn_module, "to_memory_config"):
        return tensor
    if tensor.memory_config() == dram_mc:
        return tensor
    return ttnn_module.to_memory_config(tensor, dram_mc)


def _swap_dram_kwarg(kwargs: dict, *, dram_mc: Any, l1_mc: Any) -> None:
    mc = kwargs.get("memory_config")
    if _is_dram_mc(mc, dram_mc):
        kwargs["memory_config"] = l1_mc
    imc = kwargs.get("intermediate_memory_config")
    if _is_dram_mc(imc, dram_mc):
        kwargs["intermediate_memory_config"] = l1_mc


def _ensure_l1_arg(arg: Any, *, l1_mc: Any) -> Any:
    """Move a tensor or a list of tensors (e.g. ``ttnn.concat`` inputs) to L1."""
    if isinstance(arg, list):
        return [_ensure_l1(ttnn, t, l1_mc=l1_mc) for t in arg]
    return _ensure_l1(ttnn, arg, l1_mc=l1_mc)


def _ensure_l1_first_arg(args: tuple, *, l1_mc: Any) -> tuple:
    if not args:
        return args
    return (_ensure_l1_arg(args[0], l1_mc=l1_mc),) + args[1:]


def _ensure_matmul_in0(ttnn_module: Any, tensor: Any, *, l1_mc: Any) -> Any:
    """L1 interleaved for DRAM ``in0``; leave L1-sharded activations unchanged for 2D mcast matmul."""
    if tensor is None or l1_mc is None or not hasattr(ttnn_module, "to_memory_config"):
        return tensor
    try:
        if hasattr(tensor, "memory_config") and tensor.memory_config().is_sharded():
            return tensor
    except Exception:
        pass
    return _ensure_l1(ttnn_module, tensor, l1_mc=l1_mc)


def _patch_lru_cached_getter(model_args: Any, name: str, *, dram_mc: Any, l1_mc: Any) -> None:
    original: Callable = getattr(model_args, name)

    if hasattr(original, "cache_clear"):
        original.cache_clear()

    @functools.wraps(original)
    def patched(*args, **kwargs):
        mode = args[0] if args else kwargs.get("mode")
        out = original(*args, **kwargs)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def _patch_mlp_ff2_all_reduce_getter(model_args: Any, *, dram_mc: Any, l1_mc: Any) -> None:
    name = "get_mlp_ff2_all_reduce_mem_config"
    original: Callable = getattr(model_args, name)

    @functools.wraps(original)
    def patched(mode, tensor):
        out = original(mode, tensor)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    setattr(model_args, name, patched)


def ace_step_patch_model_args_prefill_l1(model_args: Any) -> None:
    """Force L1 interleaved for all ``ModelArgs`` prefill activation memory getters."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        return

    for name in _PREFILL_DRAM_MEMCFG_GETTERS:
        if hasattr(model_args, name):
            _patch_lru_cached_getter(model_args, name, dram_mc=dram_mc, l1_mc=l1_mc)

    if hasattr(model_args, "get_mlp_ff2_all_reduce_mem_config"):
        _patch_mlp_ff2_all_reduce_getter(model_args, dram_mc=dram_mc, l1_mc=l1_mc)


def _patch_distributed_norm_prefill_l1(
    norm_mod: Any,
    *,
    l1_mc: Any,
) -> None:
    """Prefill ``DistributedNorm``: L1 activations + BLOCK-sharded ``LayerNormDeviceOperation``."""
    orig_forward = norm_mod.forward

    def forward(x, mode: Mode, norm_config=None):
        if getattr(norm_mod, "TG", False) or mode != Mode.PREFILL:
            return orig_forward(x, mode, norm_config)

        input_mem_cfg = l1_mc
        inner = norm_mod.norm
        use_block = (
            ace_step_prefill_block_sharded_norm_enabled()
            and not norm_mod.args.is_distributed_norm(mode)
            and not (getattr(inner, "is_distributed", None) and inner.is_distributed(mode))
        )
        blk_cfg = (
            ace_step_build_prefill_block_sharded_norm_config(ttnn, x, norm_mod.args.mesh_device) if use_block else None
        )

        if norm_mod.args.is_multichip and not norm_mod.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=norm_mod.prefetcher.worker_sub_device_id if norm_mod.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        if blk_cfg is not None:
            gamma = inner.weight
            if getattr(inner, "is_distributed", None) and inner.is_distributed(mode):
                gamma = inner.weight_distributed
            ck = ace_step_init_cond_rmsnorm_compute_kernel_config(norm_mod.args.mesh_device)
            # Block-sharded norm input (I2S inside helper); S2I to L1 interleaved for 1D matmul in0.
            x = ace_step_rms_norm_block_sharded(
                ttnn,
                x,
                gamma,
                inner.eps,
                device=norm_mod.args.mesh_device,
                l1_mc=l1_mc,
                compute_kernel_config=ck,
                return_sharded=False,
            )
        else:
            x = norm_mod.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)

        if norm_mod.args.is_distributed_norm(mode) and norm_mod.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return x

    norm_mod.forward = forward  # type: ignore[method-assign]


def _patch_embd_l1_output(embd_module: Any, *, l1_mc: Any) -> None:
    """Prefill-only L1 embedding output; decode passes an explicit ``memory_config`` — leave it."""

    orig_forward = embd_module.forward

    def forward(x, memory_config=None):
        if memory_config is not None:
            return orig_forward(x, memory_config=memory_config)
        out = orig_forward(x, memory_config=l1_mc)
        return _ensure_l1(ttnn, out, l1_mc=l1_mc)

    embd_module.forward = forward  # type: ignore[method-assign]


def _patch_decoder_l1_residual(decoder_layer: Any, *, l1_mc: Any) -> None:
    orig_forward = decoder_layer.forward

    def forward(x, *args, mode="decode", **kwargs):
        if mode == Mode.PREFILL:
            x = _ensure_l1(ttnn, x, l1_mc=l1_mc)
        return orig_forward(x, *args, mode=mode, **kwargs)

    decoder_layer.forward = forward  # type: ignore[method-assign]


_ACE_STEP_QWEN_PREFILL_MM_SEQ_MAX = 256


def _ace_step_prefill_seq_active(seq_len: int) -> bool:
    return 128 < int(seq_len) <= _ACE_STEP_QWEN_PREFILL_MM_SEQ_MAX


def _ace_step_qkv_out_dim(model_args: Any) -> int:
    cluster = getattr(model_args, "cluster_shape", (1, 1))
    cols = int(cluster[1]) if len(cluster) > 1 else 1
    return int(model_args.qkv_size) // max(1, cols)


def _ace_step_is_mcast_matmul_program_config(pc: Any) -> bool:
    if pc is None:
        return False
    name = type(pc).__name__
    return name in (
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "MatmulMultiCoreReuseMultiCastProgramConfig",
    )


def _ace_step_prefill_qkv_program_config(model_args: Any, seq_len: int) -> Any:
    device = getattr(model_args, "mesh_device", None) or getattr(model_args, "device", None)
    if device is None:
        return None
    return ace_step_encoder_matmul_program_config(
        device,
        seq_len=int(seq_len),
        in_dim=int(model_args.dim),
        out_dim=_ace_step_qkv_out_dim(model_args),
    )


def _ace_step_prefill_ff2_program_config(model_args: Any, seq_len: int) -> Any:
    device = getattr(model_args, "mesh_device", None) or getattr(model_args, "device", None)
    if device is None:
        return None
    cluster = getattr(model_args, "cluster_shape", (1, 1))
    cols = int(cluster[1]) if len(cluster) > 1 else 1
    k = int(model_args.hidden_dim) // max(1, cols)
    return ace_step_encoder_matmul_program_config(
        device,
        seq_len=int(seq_len),
        in_dim=k,
        out_dim=int(model_args.dim),
    )


def _patch_prefill_matmul_program_getter(
    model_args: Any,
    name: str,
    *,
    builder: Callable[[Any, int], Any],
) -> None:
    original: Callable | None = getattr(model_args, name, None)
    if original is None:
        return
    if hasattr(original, "cache_clear"):
        original.cache_clear()

    @functools.wraps(original)
    def patched(mode, seq_len=1, *args, **kwargs):
        if mode == Mode.PREFILL and _ace_step_prefill_seq_active(seq_len):
            pc = builder(model_args, int(seq_len))
            if pc is not None:
                return pc
        return original(mode, seq_len, *args, **kwargs)

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]
    setattr(model_args, name, patched)


def _rehome_mcast_matmul_weights_to_dram_interleaved(tt_model: Any) -> None:
    """One-time: ``MatmulMultiCoreReuseMultiCast*`` expects DRAM-interleaved weights (sweep ``l1/dram/*``)."""
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if dram_mc is None:
        return
    for layer in getattr(tt_model, "layers", []):
        attn = getattr(layer, "attention", None)
        if attn is not None and getattr(attn, "wqkv", None) is not None:
            wqkv = _ensure_dram_interleaved(ttnn, attn.wqkv, dram_mc=dram_mc)
            if wqkv is not attn.wqkv:
                attn.wqkv = wqkv
        mlp = getattr(layer, "feed_forward", None)
        if mlp is not None and getattr(mlp, "w2", None) is not None:
            w2 = _ensure_dram_interleaved(ttnn, mlp.w2, dram_mc=dram_mc)
            if w2 is not mlp.w2:
                mlp.w2 = w2


def ace_step_apply_qwen_prefill_matmul_configs(model_args: Any) -> None:
    """Pin swept 1D/2D mcast matmul configs for Qwen3 prefill qkv + FF2 (replaces MinimalMatmul)."""
    _patch_prefill_matmul_program_getter(
        model_args,
        "get_attn_qkv_program_config",
        builder=_ace_step_prefill_qkv_program_config,
    )
    _patch_prefill_matmul_program_getter(
        model_args,
        "get_mlp_ff2_prg_config",
        builder=_ace_step_prefill_ff2_program_config,
    )


# --------------------------------------------------------------------------------------------------
# Fused MLP gate+up matmul (the two SLOW ``256 x 1024 x 3072`` ``MatmulDeviceOperation`` rows).
#
# Stock ``tt_transformers`` MLP.forward issues TWO separate ``ttnn.linear`` calls — gate (w1) and
# up (w3) — that share the SAME input ``x`` and K=1024, differing only in weights.  We pre-concat
# w1+w3 on the output axis into a single ``[dim, 2*hidden_dim]`` weight (=> N=6144) and run ONE
# matmul, then slice the gate/up halves back out for the existing SiLU*mul.  One op launch instead
# of two; ``x`` read once instead of twice.
#
# Program config: the fused N is so wide that the 2D block-mcast kernel blows the L1 CB budget
# (per_core_N=24..48).  The 1D mcast_in0 kernel keeps all M tiles per core and splits N: with
# Nt=192 and per_core_N=2 it fills exactly 96 cores (structural optimum on the 11x10=110-core BH
# grid).  Device sweep (test_matmul_256x1024x6144_gateup_sweep.py, LoFi BF16xBFP8=>BF16, fp32 acc,
# in-process kernel-duration) winner: 1D l1/dram/ws grid (11,10), per_core_N=2, ibw=8 = 23.34μs
# (vs 26.49μs ibw=8 L1 interleaved baseline; 1.13×).  WIDTH_SHARDED out breaks ``ttnn.slice``;
# keep L1 interleaved out here (sweep ws win not usable without a de-shard).
# Gated to 128 < seq_len <= 256 and the non-galaxy path.
_ACE_STEP_QWEN_GATEUP_GRID = (11, 10)
_ACE_STEP_QWEN_GATEUP_TARGET_CORES = 96
_ACE_STEP_QWEN_GATEUP_IBW = 8


def _ace_step_gateup_subblock(per_core_m: int, per_core_n: int, *, out_sharded: bool = False) -> tuple[int, int]:
    """Largest (out_subblock_h, out_subblock_w) with h*w<=4 (fp32 dest acc)."""
    if out_sharded:
        for w in (4, 3, 2, 1):
            if per_core_n % w == 0:
                return 1, w
        return 1, 1
    for h, w in ((2, 2), (4, 1), (1, 4), (2, 1), (1, 2), (1, 1)):
        if per_core_m % h == 0 and per_core_n % w == 0:
            return h, w
    return 1, 1


def _ace_step_gateup_1d_config(seq_len: int, k: int, fused_n: int, *, out_sharded: bool = True):
    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    coord = getattr(ttnn, "CoreCoord", None)
    if cfg_cls is None or coord is None:
        return None
    mt = math.ceil(seq_len / ttnn.TILE_SIZE)
    kt = math.ceil(k / ttnn.TILE_SIZE)
    nt = math.ceil(fused_n / ttnn.TILE_SIZE)
    gx, gy = _ACE_STEP_QWEN_GATEUP_GRID
    # in0_block_w must divide Kt; prefer the tuned 8, else the largest divisor <= 8.
    ibw = (
        _ACE_STEP_QWEN_GATEUP_IBW
        if kt % _ACE_STEP_QWEN_GATEUP_IBW == 0
        else next((d for d in range(min(8, kt), 0, -1) if kt % d == 0), 1)
    )
    # per_core_N sets cores_used = ceil(Nt/per_core_N); aim for ~96 cores, capped by the grid.
    per_core_n = max(1, math.ceil(nt / _ACE_STEP_QWEN_GATEUP_TARGET_CORES))
    if math.ceil(nt / per_core_n) > gx * gy:
        per_core_n = math.ceil(nt / (gx * gy))
    sh, sw = _ace_step_gateup_subblock(mt, per_core_n, out_sharded=out_sharded)
    return cfg_cls(
        compute_with_storage_grid_size=coord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=mt,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _ace_step_build_fused_gate_up_weight(mlp_mod: Any) -> Any:
    """Concat w1 (gate) + w3 (up) on the output axis into one DRAM-interleaved weight."""
    w1 = getattr(mlp_mod, "w1", None)
    w3 = getattr(mlp_mod, "w3", None)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if w1 is None or w3 is None or dram_mc is None:
        return None
    w1_i = ttnn.to_memory_config(w1, dram_mc)
    w3_i = ttnn.to_memory_config(w3, dram_mc)
    w13 = ttnn.concat([w1_i, w3_i], dim=-1, memory_config=dram_mc)
    ttnn.deallocate(w1_i)
    ttnn.deallocate(w3_i)
    return w13


def _patch_mlp_fused_gate_up(mlp_mod: Any) -> None:
    """Fuse the gate (w1) + up (w3) prefill matmuls of one MLP into a single N-wide matmul."""
    if getattr(mlp_mod, "args", None) is None or getattr(mlp_mod.args, "is_galaxy", False):
        return
    if getattr(mlp_mod, "prefetcher", None) is not None:
        return  # prefetcher path threads w1/w3 through the global CB separately; leave it stock
    fused_w13 = _ace_step_build_fused_gate_up_weight(mlp_mod)
    if fused_w13 is None:
        return
    mlp_mod._ace_fused_w13 = fused_w13

    from models.tt_transformers.tt.ccl import tt_all_reduce
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    orig_forward = mlp_mod.forward
    seq_max = _ACE_STEP_QWEN_PREFILL_MM_SEQ_MAX

    def forward(x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        if mode != Mode.PREFILL or not (128 < int(seq_len) <= seq_max):
            return orig_forward(x, mode)
        if seq_len >= mlp_mod.args.prefill_len_cutoff:
            return orig_forward(x, mode)  # reshaped multi-block prefill — defer to stock

        layer_num = max(mlp_mod.layer_num, 0)
        opt = mlp_mod.decoders_optimizations
        activation_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.ACTIVATION)
        ff1_3_ckc = opt.get_math_fidelity(decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=mlp_mod.args)
        ff2_ckc = opt.get_math_fidelity(decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=mlp_mod.args)
        pc_2 = mlp_mod.args.get_mlp_ff2_prg_config(mode, seq_len, mlp_mod.prefetcher)
        ff1_3_mc = mlp_mod.args.get_mlp_ff1_3_mem_config(mode, mlp_mod.prefetcher)

        hidden = mlp_mod.args.hidden_dim
        fused_pc = _ace_step_gateup_1d_config(int(seq_len), mlp_mod.args.dim, 2 * hidden, out_sharded=False)

        w13_out = ttnn.linear(
            x,
            mlp_mod._ace_fused_w13,
            dtype=activation_dtype or ttnn.bfloat16,
            compute_kernel_config=ff1_3_ckc,
            program_config=fused_pc,
            memory_config=ff1_3_mc,
        )
        ttnn.deallocate(x)

        rank = len(w13_out.shape)
        begins = [0] * rank
        gate_ends = list(w13_out.shape)
        gate_ends[-1] = hidden
        up_begins = [0] * rank
        up_begins[-1] = hidden
        up_ends = list(w13_out.shape)
        w1_out = ttnn.slice(w13_out, begins, gate_ends, memory_config=ff1_3_mc)
        w3_out = ttnn.slice(w13_out, up_begins, up_ends, memory_config=ff1_3_mc)
        ttnn.deallocate(w13_out)

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[mlp_mod.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if seq_len > 128:
            w2_out = ttnn.experimental.minimal_matmul(w2_in, mlp_mod.w2, compute_kernel_config=ff2_ckc, config=pc_2)
        else:
            w2_out = ttnn.linear(
                w2_in,
                mlp_mod.w2,
                compute_kernel_config=ff2_ckc,
                dtype=activation_dtype or ttnn.bfloat16,
                program_config=pc_2,
                memory_config=mlp_mod.args.get_mlp_ff2_mem_config(mode, mlp_mod.prefetcher),
            )
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            mlp_mod.mesh_device,
            mlp_mod.tt_ccl,
            cluster_axis=0,
            dim=3,
            sharded=False,
            memory_config=mlp_mod.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=mlp_mod.args.ccl_dtype,
            use_composite=mlp_mod.dim == 8192,
            topology=mlp_mod.args.ccl_topology(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            subdevice_id=None,
        )
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        return w2_out_reduced

    mlp_mod.forward = forward  # type: ignore[method-assign]


def ace_step_apply_qwen_prefill_gate_up_fusion(tt_model: Any) -> None:
    """Fuse gate+up matmuls in every decoder MLP (seq<=256 non-galaxy prefill)."""
    for layer in getattr(tt_model, "layers", []):
        mlp_mod = getattr(layer, "feed_forward", None)
        if mlp_mod is not None and hasattr(mlp_mod, "w1") and hasattr(mlp_mod, "w3"):
            _patch_mlp_fused_gate_up(mlp_mod)


def ace_step_apply_qwen_prefill_l1(tt_model: Any, model_args: Any) -> None:
    """Patch a loaded ``tt_transformers`` Qwen model for L1 prefill activations."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    if l1_mc is None:
        return

    ace_step_patch_model_args_prefill_l1(model_args)
    ace_step_apply_qwen_prefill_matmul_configs(model_args)
    _rehome_mcast_matmul_weights_to_dram_interleaved(tt_model)
    ace_step_apply_qwen_prefill_gate_up_fusion(tt_model)

    _patch_embd_l1_output(tt_model.embd, l1_mc=l1_mc)
    for layer in tt_model.layers:
        _patch_decoder_l1_residual(layer, l1_mc=l1_mc)
        attn_norm = getattr(layer, "attention_norm", None)
        if attn_norm is not None:
            _patch_distributed_norm_prefill_l1(attn_norm, l1_mc=l1_mc)
        for norm_attr in ("ff_norm", "pre_ff_norm", "post_ff_norm"):
            norm_mod = getattr(layer, norm_attr, None)
            if norm_mod is not None:
                _patch_distributed_norm_prefill_l1(norm_mod, l1_mc=l1_mc)

    if hasattr(tt_model, "norm"):
        _patch_distributed_norm_prefill_l1(tt_model.norm, l1_mc=l1_mc)


@contextmanager
def ace_step_qwen_prefill_l1_op_context() -> Iterator[None]:
    """During Qwen prefill, redirect hard-coded ``DRAM_MEMORY_CONFIG`` TTNN calls to L1."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        yield
        return

    saved: dict[str, Any] = {}
    experimental = getattr(ttnn, "experimental", None)
    tile_layout = getattr(ttnn, "TILE_LAYOUT", None)

    def _wrap_l1_compute(name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if name.endswith("minimal_matmul") and kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return fn(*args, **kwargs)

        return wrapper

    def _wrap_l1_unary(name: str, fn: Callable) -> Callable:
        """Matmul/SDPA/concat paths: L1 ``in0`` + L1 output when callers pass DRAM defaults."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if args:
                if isinstance(args[0], list):
                    # e.g. ttnn.concat([t0, t1, ...], dim=1)
                    args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
                else:
                    in0 = _ensure_matmul_in0(ttnn, args[0], l1_mc=l1_mc)
                    if in0 is not args[0]:
                        args = (in0,) + args[1:]
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            out = fn(*args, **kwargs)
            return _ensure_l1(ttnn, out, l1_mc=l1_mc)

        return wrapper

    def _wrap_to_layout(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(tensor, layout, *args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            out = fn(tensor, layout, *args, **kwargs)
            if tile_layout is not None and layout == tile_layout:
                out = _ensure_l1(ttnn, out, l1_mc=l1_mc)
            return out

        return wrapper

    # linear: move in0 (activation) AND output to L1 so all MatmulDeviceOperation calls
    # see L1-resident in0, eliminating the "in0:dram_interleaved" bucket entirely.
    if hasattr(ttnn, "linear"):
        saved["ttnn.linear"] = ttnn.linear
        ttnn.linear = _wrap_l1_unary("linear", ttnn.linear)

    for op_name in ("add", "mul", "multiply", "pad", "reshape", "permute", "typecast"):
        if hasattr(ttnn, op_name):
            saved[f"ttnn.{op_name}"] = getattr(ttnn, op_name)
            setattr(ttnn, op_name, _wrap_l1_compute(op_name, getattr(ttnn, op_name)))

    if hasattr(ttnn, "embedding"):
        _orig_embedding = ttnn.embedding
        saved["ttnn.embedding"] = _orig_embedding

        @functools.wraps(_orig_embedding)
        def _embedding(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return _orig_embedding(*args, **kwargs)

        ttnn.embedding = _embedding

    if hasattr(ttnn, "to_layout"):
        saved["ttnn.to_layout"] = ttnn.to_layout
        ttnn.to_layout = _wrap_to_layout(ttnn.to_layout)

    if hasattr(ttnn, "concat"):
        saved["ttnn.concat"] = ttnn.concat
        ttnn.concat = _wrap_l1_unary("concat", ttnn.concat)

    transformer = getattr(ttnn, "transformer", None)
    if transformer is not None:
        for op_name in ("scaled_dot_product_attention", "chunked_scaled_dot_product_attention"):
            if hasattr(transformer, op_name):
                key = f"transformer.{op_name}"
                saved[key] = getattr(transformer, op_name)
                setattr(transformer, op_name, _wrap_l1_unary(op_name, getattr(transformer, op_name)))

    if experimental is not None:
        for op_name in ("nlp_create_qkv_heads", "all_gather_async"):
            if hasattr(experimental, op_name):
                key = f"experimental.{op_name}"
                saved[key] = getattr(experimental, op_name)
                setattr(experimental, op_name, _wrap_l1_compute(op_name, getattr(experimental, op_name)))
        if hasattr(experimental, "minimal_matmul"):
            _orig_mm = experimental.minimal_matmul
            saved["experimental.minimal_matmul"] = _orig_mm

            @functools.wraps(_orig_mm)
            def _patched_minimal_matmul(*args, **kwargs):
                config = kwargs.get("config")
                if _ace_step_is_mcast_matmul_program_config(config) and len(args) >= 2:
                    in0_orig = args[0]
                    in0 = _ensure_matmul_in0(ttnn, in0_orig, l1_mc=l1_mc)
                    weight_orig = args[1]
                    weight = _ensure_dram_interleaved(ttnn, weight_orig, dram_mc=dram_mc)
                    if in0 is not in0_orig or weight is not weight_orig:
                        args = (in0, weight) + args[2:]
                    out_mc = kwargs.get("memory_config") or l1_mc
                    matmul_kw = {k: v for k, v in kwargs.items() if k not in ("config", "memory_config")}
                    return ttnn.matmul(
                        args[0],
                        args[1],
                        program_config=config,
                        memory_config=out_mc,
                        **matmul_kw,
                    )
                # Stock MinimalMatmul: L1 activations + L1 output; weights may stay in DRAM.
                if kwargs.get("memory_config") is None:
                    kwargs["memory_config"] = l1_mc
                if args:
                    in0_orig = args[0]
                    in0 = _ensure_matmul_in0(ttnn, in0_orig, l1_mc=l1_mc)
                    if in0 is not in0_orig:
                        args = (in0,) + args[1:]
                return _orig_mm(*args, **kwargs)

            experimental.minimal_matmul = _patched_minimal_matmul
        if hasattr(experimental, "nlp_concat_heads"):
            saved["experimental.nlp_concat_heads"] = experimental.nlp_concat_heads
            experimental.nlp_concat_heads = _wrap_l1_unary("nlp_concat_heads", experimental.nlp_concat_heads)

    tt_ccl_mod = None
    try:
        from models.tt_transformers.tt import ccl as tt_ccl_mod

        if hasattr(tt_ccl_mod, "tt_all_reduce"):
            saved["tt_all_reduce"] = tt_ccl_mod.tt_all_reduce
            tt_ccl_mod.tt_all_reduce = _wrap_l1_compute("tt_all_reduce", tt_ccl_mod.tt_all_reduce)
    except ImportError:
        tt_ccl_mod = None

    try:
        yield
    finally:
        for key, fn in saved.items():
            if key.startswith("ttnn."):
                setattr(ttnn, key.split(".", 1)[1], fn)
            elif key.startswith("experimental.") and experimental is not None:
                setattr(experimental, key.split(".", 1)[1], fn)
            elif key.startswith("transformer.") and transformer is not None:
                setattr(transformer, key.split(".", 1)[1], fn)
            elif key == "tt_all_reduce" and tt_ccl_mod is not None:
                tt_ccl_mod.tt_all_reduce = fn


__all__ = [
    "ace_step_apply_qwen_prefill_l1",
    "ace_step_apply_qwen_prefill_gate_up_fusion",
    "ace_step_patch_model_args_prefill_l1",
    "ace_step_qwen_prefill_l1_op_context",
]
