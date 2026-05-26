# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TTNN throughput helpers (alignment with ``tt-perf-report`` / ``perf*.txt`` stacks).

Stacked E2E summaries often show large DRAM-interleaved shares for:

- ``PermuteDeviceOperation`` (~26 %)
- ``ReshapeViewDeviceOperation`` (~22 %)

Both ``ttnn.reshape`` and ``ttnn.permute`` accept ``memory_config``; this module **always** requests
L1 outputs where supported so reshape/permute chains avoid unnecessary DRAM round-trips.

E2E Tracy ``BinaryNgDeviceOperation (in0:dram_interleaved)`` (~24% device time) was dominated by
**VAE Snake** in FP32 (``BF16→FP32`` typecast, ~784 μs FP32 ``multiply``/``add`` per layer).
``TtSnake1d`` now always uses BF16 compute to match conv activations.

Production defaults (PCC + perf + E2E — no env toggle):

- **LoFi** matmul / RMSNorm / SDPA via :func:`ace_step_init_dit_linear_compute_kernel_config`,
  :func:`ace_step_init_cond_linear_compute_kernel_config`, :func:`ace_step_qwen3_optimizations`
- **``bfloat8_b``** linear projection weights via :func:`ace_step_linear_weight_dtype`
  (embedding tables / norm scales / conv kernels stay BF16 unless noted)
- **L1 interleaved** activations via :func:`ace_step_linear_l1_memory_config` /
  :func:`ace_step_ensure_l1_activation` / :func:`ace_step_ensure_cond_activation`

Remaining **DRAM ``in0``** buckets in Tracy (``perf_dit_4`` / conditioning) may still appear for:

- Qwen **SDPA attn masks**, **KV paged cache** (``PagedFillCache``), and **weight** tensors (DRAM by design)
- **Embedding lookup** token indices (``EmbeddingsDeviceOperation in0:dram`` — uint32 ids stay DRAM)

Other expected DRAM in DiT/VAE traces:

- ``proj_in`` uses **L1 TILE linear** patch embed (not ``conv1d``) to avoid ``Tilize`` / ``Copy`` / im2col DRAM matmul.
- Denoise feeds **TILE BF16 L1** ``xt``/``ctx`` (not ROW_MAJOR DRAM) so Tracy drops front ``Tilize``/``CopyDevice``.
- Decoder **matmuls**: LoFi + ``bfloat8_b`` weights + L1 activations; **rms_norm**: LoFi + L1 (not default HiFi4).
- **SDPA attn masks**: DRAM-only (TTNN requirement).
- **RoPE / norm / linear weights**: DRAM storage; matmul reads weights from DRAM while ``in0`` activations are L1.
- Residual **BinaryNg (in0:dram)** (~0.3%): usually scalar-broadcast or slice outputs — call sites use :func:`ace_step_ensure_l1_activation` after ``ace_step_add_one``.

DiT linears are often DRAM-bound at HiFi4 without tuning (reference path only):

- ``256×1024×1024`` — attn ``q_proj`` / ``o_proj``
- ``256×2048×2048`` — fused attn ``wkv``
- ``256×3072×3072`` — MLP ``gate_proj`` / ``up_proj`` / ``down_proj``

VAE decode exposes large-M matmuls inside ``conv1d`` / ``conv_transpose2d`` im2col (e.g.
``1920×512×512``, ``30720×128×128``, ``61440×128×128``). Production VAE uses **LoFi** conv compute + **BF16** activations; **``bfloat8_b``** weights on
``k>1`` conv / conv-transpose im2col (DRAM BW). ``k=1`` projections use ``ttnn.conv1d`` with tuned
1D mcast matmul configs on large im2col ``M`` (``61440`` / ``30720`` / ``7680`` buckets via
:func:`ace_step_vae_conv1d_im2col_matmul_program_config`).
Opt-in **``bfloat8_b`` activation compute** (conv output dtype + Snake TILE chain) via
``ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=1``; inter-op buffers stay BF16 ``ROW_MAJOR`` (TTNN layout limit).

Memory policy (VAE):

- **L1 interleaved** on ``1×1`` conv + conv-transpose activations and outputs.
- **Snake eltwise chain** (multiply, sin, square, multiply, add + Tilize/Untilize) runs in L1
  (params in L1, intermediates in L1) — eliminates ~5 DRAM round-trips per call vs the
  original all-DRAM path.  Snake **output** is staged back to DRAM inside ``TtSnake1d``
  before returning so the caller always receives a DRAM tensor.
- **k>1 ``conv1d`` input/output** stays in DRAM (static CB region extends to ~180 KiB on
  Blackhole — any live L1 activation in that band fails program validation at compile time;
  k>1 output also exceeds per-bank L1 budget).

Condition encoder linears (lyric/timbre, ``hidden_size=2048``) are often DRAM-bound:

- ``32×2048×2048`` — attn ``q``/``k``/``v``/``o`` (short packed sequences)
- ``32×6144×6144`` — MLP ``gate``/``up``
- ``288×2048×2048`` — longer lyric windows
"""

from __future__ import annotations

import os
from typing import Any


def _l1_memory_kwargs(ttnn: Any) -> dict:
    mc = getattr(ttnn, "L1_MEMORY_CONFIG", None)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_reshape_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.reshape`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_permute_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.permute`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_to_layout_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for ``ttnn.to_layout`` so ``Tilize*`` outputs land in L1 (not DRAM)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_concat_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for ``ttnn.concat`` on activation tensors."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_lofi_bfloat8_enabled() -> bool:
    """ACE-Step production path: LoFi compute + ``bfloat8_b`` linear weights (always on, no env)."""
    return True


def ace_step_dit_lofi_bfloat8_enabled() -> bool:
    """Alias for :func:`ace_step_lofi_bfloat8_enabled` (DiT call sites)."""
    return ace_step_lofi_bfloat8_enabled()


def ace_step_linear_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """Weight storage dtype for **linear** projections (activations stay ``default_dtype``, usually BF16)."""
    if ace_step_lofi_bfloat8_enabled():
        return getattr(ttnn, "bfloat8_b", None) or default_dtype
    return default_dtype


def ace_step_dit_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """DiT linear weights — alias for :func:`ace_step_linear_weight_dtype`."""
    return ace_step_linear_weight_dtype(ttnn, default_dtype)


def ace_step_dit_conv_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """Legacy conv weight dtype helper (``proj_in`` is linear TILE now; kept for callers)."""
    return ace_step_dit_weight_dtype(ttnn, default_dtype)


def ace_step_dit_weight_layout(ttnn: Any, weight_dtype: Any, *, default_layout: Any) -> Any:
    """Layout for weight upload: ``bfloat8_b`` / ``bfloat4_b`` must be TILE (TT_FATAL otherwise)."""
    bf8 = getattr(ttnn, "bfloat8_b", None)
    bf4 = getattr(ttnn, "bfloat4_b", None)
    if weight_dtype in (bf8, bf4):
        return ttnn.TILE_LAYOUT
    return default_layout


def ace_step_is_tile_layout(ttnn: Any, tensor: Any) -> bool:
    tile = getattr(ttnn, "TILE_LAYOUT", None)
    if tile is None or tensor is None:
        return False
    layout = getattr(tensor, "layout", None)
    return layout == tile


def ace_step_ensure_tile_layout(
    ttnn: Any, tensor: Any, l1_mc: Any | None = None, *, out_memory_config: Any | None = None
) -> Any:
    """Return *tensor* unchanged when already TILE — avoids redundant ``TilizeDeviceOperation``.

    When a Tilize is required the output is placed in L1 (via ``L1_MEMORY_CONFIG``) so the
    caller gets an L1-resident tile instead of a DRAM-resident one.  Pass *out_memory_config*
    (e.g. ``DRAM_MEMORY_CONFIG`` for SDPA masks) to override the default L1 placement.
    """
    if tensor is None:
        return tensor
    out_mc = out_memory_config if out_memory_config is not None else l1_mc
    if out_mc is None and out_memory_config is None:
        out_mc = ace_step_linear_l1_memory_config(ttnn)
    if ace_step_is_tile_layout(ttnn, tensor):
        if out_mc is not None and hasattr(tensor, "memory_config") and hasattr(ttnn, "to_memory_config"):
            if tensor.memory_config() != out_mc:
                return ttnn.to_memory_config(tensor, out_mc)
        return tensor
    kw = {"memory_config": out_mc} if out_mc is not None else {}
    return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, **kw)


def ace_step_upload_f32_np_as_bf16_tile(
    ttnn: Any,
    host_f32: Any,
    *,
    device: Any,
    dtype: Any | None = None,
    memory_config: Any | None = None,
    mesh_mapper: Any | None = None,
) -> Any:
    """Upload host float32 bias/mask numpy as BF16 TILE without on-device FP32 tilize + typecast.

    ``ttnn.as_tensor(fp32_np, dtype=bfloat16, layout=TILE)`` runs Tilize FP32→FP32 then
    Typecast FP32→BF16 on device. Converting on host and uploading ROW_MAJOR BF16 needs one
    BF16 tilize only.
    """
    import numpy as np
    import torch

    bf16 = dtype if dtype is not None else getattr(ttnn, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("bfloat16 required for ace_step_upload_f32_np_as_bf16_tile")
    mc = memory_config if memory_config is not None else ace_step_sdpa_mask_memory_config(ttnn)
    host = np.ascontiguousarray(host_f32, dtype=np.float32)
    torch_bf16 = torch.from_numpy(host).to(torch.bfloat16)
    kw: dict = {}
    if mesh_mapper is not None:
        kw["mesh_mapper"] = mesh_mapper
    if mc is not None:
        kw["memory_config"] = mc
    tt_rm = ttnn.from_torch(
        torch_bf16,
        device=device,
        dtype=bf16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        **kw,
    )
    return ace_step_ensure_tile_layout(ttnn, tt_rm, out_memory_config=mc)


def ace_step_ensure_row_major_layout(ttnn: Any, tensor: Any) -> Any:
    """Return *tensor* unchanged when already ROW_MAJOR."""
    if tensor is None:
        return tensor
    rm = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if rm is not None and getattr(tensor, "layout", None) == rm:
        return tensor
    return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)


def ace_step_dit_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_cond_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_vae_conv_perf_enabled() -> bool:
    """Deprecated: VAE conv perf path is always enabled. Kept for call-site compatibility."""
    return True


def ace_step_init_vae_conv_compute_kernel_config(device: Any):
    """LoFi compute kernel for Oobleck VAE ``conv1d`` / ``conv_transpose2d`` im2col matmuls."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_vae_bfloat8_activations_enabled() -> bool:
    """Opt-in ``bfloat8_b`` for VAE conv im2col + Snake eltwise compute (``ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=1``).

    TTNN does not support ``bfloat8_b`` with ``ROW_MAJOR`` activations (conv1d / slice / residual
    trim). Inter-op buffers stay **BF16 ROW_MAJOR**; :func:`ace_step_vae_activation_compute_dtype`
    applies inside conv/Snake kernels only.
    """
    return os.environ.get("ACE_STEP_VAE_BFLOAT8_ACTIVATIONS", "0").lower() in ("1", "true", "yes", "on")


def ace_step_vae_activation_storage_dtype(ttnn: Any) -> Any:
    """VAE activation buffers between ops: **BF16** ``ROW_MAJOR`` (slice / residual / chunking)."""
    dtype = getattr(ttnn, "bfloat16", None)
    if dtype is None:
        raise RuntimeError("TTNN build missing bfloat16; VAE activation storage requires BF16")
    return dtype


def ace_step_vae_activation_compute_dtype(ttnn: Any) -> Any:
    """VAE conv im2col + Snake internal compute dtype (``bfloat8_b`` when opt-in env is set)."""
    if ace_step_vae_bfloat8_activations_enabled():
        bf8 = getattr(ttnn, "bfloat8_b", None)
        if bf8 is not None:
            return bf8
    return ace_step_vae_activation_storage_dtype(ttnn)


def ace_step_vae_activation_dtype(ttnn: Any) -> Any:
    """Alias for :func:`ace_step_vae_activation_compute_dtype` (Tracy / demo-facing dtype)."""
    return ace_step_vae_activation_compute_dtype(ttnn)


def ace_step_vae_host_weight_staging_dtype(ttnn: Any) -> Any:
    """Host ROW_MAJOR dtype before ``prepare_conv_weights`` (always BF16)."""
    return ace_step_vae_activation_storage_dtype(ttnn)


def ace_step_vae_normalize_activation_output(ttnn: Any, tensor: Any, *, storage_dtype: Any, compute_dtype: Any) -> Any:
    """Return a ``ROW_MAJOR`` tensor in *storage_dtype* (post-conv / post-Snake boundary contract)."""
    if tensor is None:
        return tensor
    if compute_dtype != storage_dtype and getattr(tensor, "dtype", None) != storage_dtype:
        dram = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        kw = {"memory_config": dram} if dram is not None else {}
        tensor = ttnn.typecast(tensor, storage_dtype, **kw)
    return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)


def ace_step_vae_activation_memory_config(ttnn: Any):
    """L1 interleaved activations for VAE Snake / ``1×1`` conv / conv-transpose glue ops."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_vae_conv1d_memory_config(ttnn: Any, *, kernel_size: int):
    """Memory config for ``TtConv1d``: L1 for ``kernel_size==1``, DRAM for ``k>1``.

    Wide ``k>1`` conv programs allocate multi-MB L1 output tensors (``conv2d_L1``) that exceed
    per-bank budget on Blackhole when activations are forced into L1 interleaved.
    """
    if int(kernel_size) == 1:
        return ace_step_vae_activation_memory_config(ttnn)
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_vae_conv_weight_dtype(ttnn: Any, default_dtype: Any, *, kernel_size: int) -> Any:
    """VAE conv weights: ``bfloat8_b`` for ``k>1`` DRAM im2col (halves weight BW); BF16 for ``k==1``."""
    if int(kernel_size) > 1:
        return ace_step_linear_weight_dtype(ttnn, default_dtype)
    return default_dtype


def ace_step_vae_typecast_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for VAE ``typecast`` (latent FP32 → BF16)."""
    mc = l1_mc if l1_mc is not None else ace_step_vae_activation_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_vae_eltwise_kwargs(ttnn: Any, *, device: Any = None, l1_mc: Any | None = None) -> dict:
    """Keyword args for VAE Snake ``multiply`` / ``add`` — L1 output only.

    ``ttnn.multiply`` / ``ttnn.add`` accept ``memory_config`` but not ``compute_kernel_config`` or
    ``fast_and_approximate_mode`` on all builds. LoFi fidelity is applied on conv im2col matmuls.
    """
    _ = device
    return ace_step_binary_kwargs(ttnn, l1_mc)


def ace_step_device_profiler_enabled() -> bool:
    return (
        os.environ.get("TTNN_OP_PROFILER", "").strip() == "1"
        or os.environ.get("TT_METAL_DEVICE_PROFILER", "").strip() == "1"
    )


def ace_step_profiler_flush_every_layer() -> int:
    """Drain device profiler every N VAE/DiT layers (0=off). Default ``1`` when profiling."""
    if not ace_step_device_profiler_enabled():
        return 0
    try:
        return max(0, int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY_LAYER", "1")))
    except ValueError:
        return 1


def ace_step_enable_tracy_profiler_env() -> None:
    """Match TTNN + metal profilers so Tracy host/device op IDs align in ``cpp_device_perf_report.csv``."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "").strip() == "1":
        os.environ.setdefault("TTNN_OP_PROFILER", "1")


def ace_step_flush_device_profiler(device) -> None:
    """Sync and drain per-RISC device profiler rings (avoids Tracy 'Device data missing' on merge)."""
    if not ace_step_device_profiler_enabled():
        return
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        return
    try:
        import ttnn

        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def ace_step_vae_conv1d_im2col_matmul_enabled() -> bool:
    """Whether ``1×1`` conv may bypass ``ttnn.conv1d`` with tuned ``ttnn.linear`` matmul configs.

    - ``ACE_STEP_VAE_LARGE_M_MATMUL=0`` — always off.
    - ``ACE_STEP_VAE_LARGE_M_MATMUL=1`` — always on.
    - Default — on. **Stays on under profilers** so large im2col ``M`` uses a clamped full-grid
      program config instead of ``ttnn.conv1d`` matmul probing 640 M-tiles as 640 cores.
    """
    flag = os.environ.get("ACE_STEP_VAE_LARGE_M_MATMUL")
    if flag is not None:
        return flag.lower() not in ("0", "false", "no", "off")
    return True


def ace_step_vae_max_per_core_m_tiles() -> int:
    """Max ``per_core_M`` (M tiles) for VAE im2col ``MultiCast1D`` before falling back to ``ttnn.conv1d``.

    Default ``8`` for normal perf. While profiling, default ``20`` so ``61440×128`` im2col
    (``per_core_M≈18`` on 110 cores) still uses tuned matmul. Override via env.
    """
    default = "20" if ace_step_device_profiler_enabled() else "8"
    return max(1, int(os.environ.get("ACE_STEP_VAE_MAX_PER_CORE_M_TILES", default)))


def ace_step_vae_conv1d_im2col_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Tuned 1D reuse config for VAE ``1×1`` ``ttnn.conv1d`` im2col matmuls.

    Applies to large-M buckets from Tracy (``61440×128``, ``30720×128``, ``7680×256``, ``1920×512``) using
    **tall-M** ``MultiCast1D`` (``mcast_in0=False``, ``M`` split across cores). Skips ``M < 7680`` only.
    """
    m = int(m_dim)
    k = int(k_dim)
    n = int(n_dim)
    if m < 7680:
        return None
    if not ace_step_vae_conv1d_im2col_matmul_enabled():
        # Large im2col still needs a clamped program config — ``ttnn.conv1d`` probes M-tiles as cores.
        pass
    return ace_step_vae_large_m_matmul_program_config(
        device,
        m_dim=m,
        k_dim=k,
        n_dim=n,
    )


def _mcast_1d_vae_im2col_tall_m_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    in0_block_w_cap: int = 2,
    out_subblock_h_cap: int = 4,
    out_subblock_w: int = 1,
):
    """1D reuse matmul for **tall** VAE ``1×1`` conv im2col (``M >> N``).

    Uses ``mcast_in0=False`` and splits ``M`` across the compute grid. ``mcast_in0=True`` would
    require every core to buffer the **full** ``M`` strip in L1 (see TT_FATAL in
    ``matmul_multicore_reuse_mcast_1d``) and overflows Blackhole for production audio lengths
    (``61440×128`` im2col).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    grid = device.compute_with_storage_grid_size()
    gx = max(1, int(grid.x))
    gy = max(1, int(grid.y))
    num_cores = gx * gy
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    m_tiles = max(1, (int(m_dim) + tile - 1) // tile)
    n_tiles = max(1, (int(n_dim) + tile - 1) // tile)
    k = max(tile, int(k_dim))
    k_tiles = max(1, k // tile)

    per_core_m = max(1, (m_tiles + num_cores - 1) // num_cores)
    if per_core_m > ace_step_vae_max_per_core_m_tiles() and not ace_step_device_profiler_enabled():
        return None
    num_blocks_m = (m_tiles + per_core_m - 1) // per_core_m
    if num_blocks_m > num_cores:
        return None

    in0_block_w = min(int(in0_block_w_cap), k_tiles)
    while k_tiles % in0_block_w != 0 and in0_block_w > 1:
        in0_block_w -= 1

    # Use the full device grid — a 1-row subgrid (e.g. 11×1) makes matmul output-spec probing
    # request hundreds of cores against only 11 available (TT_FATAL spam / Tracy 32K limit).
    per_core_n = max(1, (n_tiles + num_cores - 1) // num_cores)

    out_subblock_h = min(int(out_subblock_h_cap), per_core_m)
    while per_core_m % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1

    out_subblock_w_target = min(int(out_subblock_w), max(1, int(per_core_n)))
    if out_subblock_w_target > 1 and per_core_n % out_subblock_w_target != 0:
        per_core_n = ((per_core_n + out_subblock_w_target - 1) // out_subblock_w_target) * out_subblock_w_target

    out_subblock_w_eff = min(int(out_subblock_w), max(1, int(per_core_n)))
    while per_core_n % out_subblock_w_eff != 0 and out_subblock_w_eff > 1:
        out_subblock_w_eff -= 1

    return cfg_cls(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w_eff,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


def ace_step_vae_large_m_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Matmul program config for VAE conv im2col shapes (e.g. 1920×512, 30720×128, 61440×128)."""
    tile = 32
    m = max(1, int(m_dim))
    k = max(tile, int(k_dim))
    n = max(tile, int(n_dim))
    if m >= 61440:
        return _mcast_1d_vae_im2col_tall_m_program_config(
            device,
            m_dim=m,
            k_dim=k,
            n_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=2,
            out_subblock_w=2,
        )
    if m >= 7680:
        return _mcast_1d_vae_im2col_tall_m_program_config(
            device,
            m_dim=m,
            k_dim=k,
            n_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=1,
            out_subblock_w=4,
        )
    return _mcast_1d_vae_im2col_tall_m_program_config(
        device,
        m_dim=m,
        k_dim=k,
        n_dim=n,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_linear_l1_memory_config(ttnn: Any):
    """L1 interleaved buffer config for linear activations / outputs."""
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


def ace_step_dit_weight_memory_config(ttnn: Any):
    """DRAM interleaved for DiT **weights** (linears, norms, RoPE tables, conv kernels)."""
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_dit_linear_l1_memory_config(ttnn: Any):
    """Alias for :func:`ace_step_linear_l1_memory_config`."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_linear_kwargs_memory_config(
    program_config: Any | None,
    *,
    linear_out_l1: Any | None,
    dram: Any | None,
) -> Any | None:
    """L1 linear outputs only when the 1D-mcast program is active; DRAM matmul must not use L1."""
    if program_config is not None and linear_out_l1 is not None:
        return linear_out_l1
    return dram


def ace_step_sdpa_activation_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for fused SDPA so attention output stays in L1 (mask stays DRAM)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_safe_deallocate(ttnn: Any, *tensors: Any) -> None:
    """Best-effort ``ttnn.deallocate`` for optional / already-freed tensors."""
    for t in tensors:
        if t is None:
            continue
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def ace_step_add_one(ttnn: Any, tensor: Any, **kwargs: Any) -> Any:
    """``tensor + 1`` via ``ttnn.add(tensor, 1.0)`` — avoids per-call ``ones_like`` / ``full``."""
    return ttnn.add(tensor, 1.0, **kwargs)


def ace_step_nlp_concat_heads(ttnn: Any, ctx: Any, *, l1_mc: Any | None = None) -> Any:
    """Replace output permute+reshape with ``ttnn.experimental.nlp_concat_heads``.

    Converts ``[B, H, S, Dh]`` → ``[B, 1, S, H*Dh]`` in a single device kernel,
    eliminating one permute (~360 μs) and one non-view reshape (~405 μs) per attention block.

    SDPA may leave ``ctx`` in DRAM when ``memory_config`` is omitted; move to L1 first so Tracy
    does not bucket this op as ``in0:dram_interleaved`` (~10 ms in DiT perf traces).

    Falls back to the original permute+reshape path if the op is unavailable.
    """
    experimental = getattr(ttnn, "experimental", None)
    nlp_concat = getattr(experimental, "nlp_concat_heads", None) if experimental is not None else None
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    # SDPA without ``SDPAProgramConfig`` leaves interleaved L1; skip sharded→interleaved copy.
    if mc is not None and hasattr(ctx, "memory_config") and hasattr(ttnn, "to_memory_config"):
        if ctx.memory_config() != mc:
            ctx = ace_step_ensure_l1_activation(ttnn, ctx, mc)
    if nlp_concat is not None:
        kw = {"memory_config": mc} if mc is not None else {}
        return nlp_concat(ctx, **kw)
    # Fallback: original two-op path.
    B, H, S, Dh = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2]), int(ctx.shape[3])
    _kw = {"memory_config": mc} if mc is not None else {}
    ctx = ttnn.permute(ctx, (0, 2, 1, 3), **_kw)
    ctx = ttnn.reshape(ctx, (B, 1, S, H * Dh), **_kw)
    return ctx


def ace_step_eltwise_l1_memory_config(ttnn: Any):
    """L1 config for BinaryNg / ``add`` / ``multiply`` / ``softmax`` activations (Tracy ``in0`` bucket)."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_sdpa_mask_memory_config(ttnn: Any):
    """SDPA requires ``attn_mask`` buffers in DRAM (see ``sdpa_device_operation.cpp``)."""
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_ensure_l1_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """Move a device tensor to L1 interleaved so BinaryNg tags ``in0:l1_interleaved``."""
    if tensor is None:
        return tensor
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    if mc is None or not hasattr(ttnn, "to_memory_config"):
        return tensor
    if hasattr(tensor, "memory_config") and tensor.memory_config() == mc:
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def ace_step_ensure_dram_activation(ttnn: Any, tensor: Any, dram_mc: Any | None = None) -> Any:
    """Move a device tensor to DRAM so matmul programs do not clash with L1 activation buffers."""
    if tensor is None:
        return tensor
    mc = dram_mc if dram_mc is not None else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mc is None or not hasattr(ttnn, "to_memory_config"):
        return tensor
    if hasattr(tensor, "memory_config") and tensor.memory_config() == mc:
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def ace_step_dit_fused_m_tiles(*, batch_size: int, seq_len: int, tile: int = 32) -> int:
    """Fused batch×seq tile rows used by 1D-mcast matmul (``fuse_batch=True``)."""
    tile_size = max(1, int(tile))
    s_tiles = (max(1, int(seq_len)) + tile_size - 1) // tile_size
    return max(1, int(batch_size)) * s_tiles


def ace_step_dit_prefers_dram_activations(*, batch_size: int, seq_len: int, max_fused_m: int = 16) -> bool:
    """True when DiT should keep activations in DRAM (long clips; avoids L1/DRAM mixing noise)."""
    return ace_step_dit_fused_m_tiles(batch_size=int(batch_size), seq_len=int(seq_len)) > int(max_fused_m)


def ace_step_dit_body_trace_safe(*, batch_size: int, patch_seq_len: int, max_fused_m: int = 16) -> bool:
    """Return False when DiT body trace replay is known to drift vs eager (audible noise).

    Long clips fall back to DRAM matmul (``per_core_M`` > 16) with ``to_memory_config`` in the
    graph; body trace capture/replay is not bit-accurate in that regime (same class of issue as
    traced VAE tiles / ``DitCfgPrepTrace``).
    """
    return ace_step_dit_fused_m_tiles(batch_size=int(batch_size), seq_len=int(patch_seq_len)) <= int(max_fused_m)


def ace_step_matmul_activation(
    ttnn: Any,
    tensor: Any,
    linear_kwargs: dict,
    *,
    l1_fn,
    dram_mc: Any | None = None,
) -> Any:
    """Place matmul ``in0`` in L1 when a 1D-mcast program is active, else DRAM."""
    if "program_config" in linear_kwargs:
        return l1_fn(tensor)
    return ace_step_ensure_dram_activation(ttnn, tensor, dram_mc)


def ace_step_from_torch_activation(
    ttnn: Any,
    tensor: Any,
    *,
    device: Any,
    dtype: Any,
    layout: Any | None = None,
    l1_mc: Any | None = None,
) -> Any:
    """Upload a host tensor directly into L1 (avoids default DRAM ``from_torch``)."""
    tile = getattr(ttnn, "TILE_LAYOUT", None)
    use_layout = layout if layout is not None else tile
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    kw: dict = {"device": device, "dtype": dtype}
    if use_layout is not None:
        kw["layout"] = use_layout
    if mc is not None:
        kw["memory_config"] = mc
    return ttnn.from_torch(tensor, **kw)


def ace_step_pad_activation_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for ``ttnn.pad`` / fill-pad style ops on activations."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_binary_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for ``add`` / ``multiply`` / ``softmax`` with L1 output."""
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_init_hifi2_linear_compute_kernel_config(device: Any):
    """HiFi2 linear config for DRAM-bound projections (DiT + condition encoder)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_hifi4_linear_compute_kernel_config(device: Any):
    """HiFi4 linear config for FLOP-bound projections (condition encoder small-seq linears)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_lofi_linear_compute_kernel_config(device: Any):
    """LoFi linear config for DiT throughput (paired with ``bfloat8_b`` weights)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_dit_linear_compute_kernel_config(device: Any):
    """DiT linear compute kernel: LoFi (production default, matches perf traces)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_linear_compute_kernel_config(device: Any):
    """Condition encoder / lyric / timbre linears: LoFi (same default as DiT)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_sdpa_compute_kernel_config(device: Any):
    """SDPA compute kernel for condition lyric/timbre encoders: LoFi."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_qwen3_optimizations(model_args: Any):
    """``tt_transformers`` decoder config for ACE Qwen3 caption encoder: LoFi + ``bfloat8_b`` weights.

    Passed to :func:`models.tt_transformers.tt.common.create_tt_model` as ``optimizations=``.
    Sets per-op math fidelity and ``bfloat8_b`` projection dtypes (replacing the default
    ``accuracy`` path that uses HiFi4 on prefill attention). Prefill activations are moved to
    L1 via :mod:`qwen_prefill_l1` (``ace_step_apply_qwen_prefill_l1``).
    """
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
                OpGroup.LI_FF2: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.LOFI,
                OpGroup.SDPA_DECODE: MathFidelitySetting.LOFI,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.LOFI,
                OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.LOFI,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, decoder_conf=conf)


def ace_step_init_dit_rmsnorm_compute_kernel_config(device: Any):
    """RMSNorm compute kernel for DiT blocks: LoFi (default TTNN rmsnorm uses HiFi4)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_rmsnorm_compute_kernel_config(device: Any):
    """RMSNorm compute kernel for condition encoder blocks: LoFi."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_dit_rms_norm_kwargs(ttnn: Any, l1_mc: Any | None = None, *, device: Any = None) -> dict:
    """``memory_config`` + LoFi ``compute_kernel_config`` for DiT ``ttnn.rms_norm``."""
    return ace_step_cond_rms_norm_kwargs(ttnn, l1_mc, device=device)


def ace_step_cond_rms_norm_kwargs(ttnn: Any, l1_mc: Any | None = None, *, device: Any = None) -> dict:
    """``memory_config`` + LoFi ``compute_kernel_config`` for ``ttnn.rms_norm`` (keeps TILE/L1)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    kw: dict = {}
    if mc is not None:
        kw["memory_config"] = mc
    if device is not None:
        ck = ace_step_init_cond_rmsnorm_compute_kernel_config(device)
        if ck is not None:
            kw["compute_kernel_config"] = ck
    return kw


def ace_step_ensure_dit_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """TILE layout + L1 interleaved — DiT matmul/binary/rms_norm input contract."""
    return ace_step_ensure_cond_activation(ttnn, tensor, l1_mc)


def ace_step_ensure_cond_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """TILE layout + L1 interleaved — condition encoder matmul/binary/rms_norm contract."""
    if tensor is None:
        return tensor
    t = ace_step_ensure_tile_layout(ttnn, tensor)
    return ace_step_ensure_l1_activation(ttnn, t, l1_mc)


def ace_step_dit_activation_from_torch(
    ttnn: Any,
    host_tensor: Any,
    *,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    """Upload a host tensor as TILE BF16 in L1 — same contract as perf/E2E (no env toggle)."""
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

    use_dtype = dtype if dtype is not None else getattr(ttnn, "bfloat16", None)
    if use_dtype is None:
        raise RuntimeError("bfloat16 required for activations")
    if torch is not None and isinstance(host_tensor, torch.Tensor):
        arr = host_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        arr = np.asarray(host_tensor, dtype=np.float32)
    return ace_step_from_torch_activation(
        ttnn,
        arr,
        device=device,
        dtype=use_dtype,
        l1_mc=ace_step_linear_l1_memory_config(ttnn),
    )


def _mcast_1d_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    in0_block_w_cap: int = 2,
    out_subblock_h_cap: int = 4,
    out_subblock_w: int = 1,
    max_per_core_m: int = 16,
):
    """Shared 1D mcast matmul program config builder for ACE-Step linears.

    With ``fuse_batch=True``, TTNN fuses batch into the ``M`` dim (tile rows) for tensors
    shaped ``[B, 1, S, K]``. TILE layout pads ``S`` to the tile height, so runtime
    ``M`` (tiles) is ``B * ceil(seq_len / tile)`` once ``S`` is padded to TILE height.
    ``seq_len`` without batch—or ignoring TILE padding along ``S``—can under-report
    ``per_core_M``. For ``N`` (tiles across the output inner dim), derive ``per_core_N`` from
    ``ceil(out_dim / tile)`` spread across ``grid.x`` so ``num_blocks_x`` stays within core count.
    ``out_subblock_w`` is clipped so ``per_core_N % out_subblock_w == 0`` (default ``out_block_w``
    equals ``per_core_N``).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    grid = device.compute_with_storage_grid_size()
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    bsz = max(1, int(batch_size))
    seq = max(1, int(seq_len))
    # mcast_in0 requires num_blocks_y == 1: every core must hold all M tiles (per_core_M = total M).
    # Do NOT reduce per_core_M to split M across y-rows — that violates the constraint.
    # Instead, use extra y-rows as additional N-workers so the full grid participates.
    s_tiles = (seq + tile - 1) // tile
    per_core_m = max(1, bsz * s_tiles)

    # Guard against L1 CB overflow on WH cores (~1.5 MB each).
    # Static CBs grow ~28 KB per per_core_M tile row; at per_core_M=48 they reach 1339 KB,
    # leaving no room for L1 activation tensors (observed clash at 865 KB with duration=30).
    # per_core_M<=16 is safe for short clips (~10s, fused_M<=16). Longer clips use DRAM matmul.
    if per_core_m > int(max_per_core_m):
        return None

    k = max(tile, int(in_dim))

    k_tiles = max(1, k // tile)
    in0_block_w = min(int(in0_block_w_cap), k_tiles)

    n_width_tiles = max(1, (int(out_dim) + tile - 1) // tile)
    gx = max(1, int(grid.x))
    gy = max(1, int(grid.y))

    # Use y-rows as additional N-workers: with (gx, y_rows) grid, N is distributed across
    # gx*y_rows cores.  When N tiles fit within gx*gy we can set per_core_n=1 and just pick
    # enough rows; when N tiles exceed total cores we use the full grid with per_core_n>1.
    if n_width_tiles <= gx * gy:
        y_rows = min(gy, max(1, (n_width_tiles + gx - 1) // gx))
        per_core_n = max(1, (n_width_tiles + gx * y_rows - 1) // (gx * y_rows))
    else:
        y_rows = gy
        per_core_n = max(1, (n_width_tiles + gx * gy - 1) // (gx * gy))

    # When M=1 tile (e.g. seq_len=32), out_subblock_h is forced to 1.  TTNN recommends
    # out_subblock_h * out_subblock_w >= 2, so we need per_core_n >= 2 for out_subblock_w=2.
    # Reduce y_rows (fewer cores) until each core handles at least 2 N tiles.
    if per_core_m == 1 and per_core_n < 2 and n_width_tiles >= 2:
        for cand_y in range(y_rows - 1, 0, -1):
            cand_pcn = max(1, (n_width_tiles + gx * cand_y - 1) // (gx * cand_y))
            if cand_pcn >= 2:
                y_rows = cand_y
                per_core_n = cand_pcn
                break

    out_subblock_h = min(int(out_subblock_h_cap), per_core_m)
    while per_core_m % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1

    # Default ``out_block_w`` is ``per_core_N``; TTNN requires ``out_block_w % out_subblock_w == 0``.
    # Round per_core_N up to a multiple of the requested subblock width instead of shrinking the
    # subblock to 1×1 (e.g. 64 N-tiles / 10 cores → 7 → snap to 8 so 1×2 subblocks are valid).
    out_subblock_w_target = min(int(out_subblock_w), max(1, int(per_core_n)))
    if out_subblock_w_target > 1 and per_core_n % out_subblock_w_target != 0:
        per_core_n = ((per_core_n + out_subblock_w_target - 1) // out_subblock_w_target) * out_subblock_w_target

    out_subblock_w_eff = min(int(out_subblock_w), max(1, int(per_core_n)))
    while per_core_n % out_subblock_w_eff != 0 and out_subblock_w_eff > 1:
        out_subblock_w_eff -= 1

    return cfg_cls(
        compute_with_storage_grid_size=(gx, y_rows),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w_eff,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def ace_step_dit_attn_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for square DiT ``q`` / ``o`` (e.g. 256×1024×1024)."""
    pc = _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=2,
    )
    if pc is not None:
        return pc
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_cond_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """Program config for condition encoder linears (e.g. 32×2048×2048, 288×2048×2048)."""
    short = int(seq_len) <= 64
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_cond_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for condition MLP gate/up (e.g. 32×6144×6144).

    Lyric/timbre encoders use intermediate 6144×2048; ``in0_block_w=2`` plus L1-hosted
    activations can overrun per-core circular-buffer budget (static CB vs tensor L1).
    Use ``in0_block_w_cap=1`` on this wide path by default.
    """
    short = int(seq_len) <= 64
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=intermediate_size,
        batch_size=batch_size,
        in0_block_w_cap=1,
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_dit_fused_wkv_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    fused_kv_dim: int,
    batch_size: int = 1,
):
    """Program config for fused ``wkv`` (e.g. 256×2048×2048 when ``hidden_size=1024``, GQA ``fused_kv_dim=2048``)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=fused_kv_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``gate_proj`` / ``up_proj`` (e.g. 256×3072×3072)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=intermediate_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_down_proj_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    intermediate_size: int,
    hidden_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``down_proj`` (e.g. 256×3072×3072 when intermediate==hidden)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=intermediate_size,
        out_dim=hidden_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dense_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """General 1D mcast program config for dense linears not covered by a dedicated helper.

    Used for: condition_embedder (256×2048×1024), depatchify proj_out (256×1024×4096),
    text projector, and any other linear where shapes are not known at module init time.
    """
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )
