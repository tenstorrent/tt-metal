# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute-kernel defaults for BGE-M3.

- **Matmul + SDPA:** **HiFi4 + FP32 dest acc + L1 packer acc** on **Blackhole and Wormhole** so full-model
  PCC stays **> 0.94** at **S8192** (and ~0.99 on BH). Wormhole used to default to HiFi2 for stability,
  but HiFi2 drifts to ~0.85 PCC at long sequence; HiFi4 aligns WH with the BH numerics path.
- **LayerNorm:** HiFi4 + FP32 on all archs (matches ``ttnn`` layer_norm defaults in Metal).
- ``max_seq_len`` on kernel helpers is kept for API compatibility / future tuning; fidelity does not
  downgrade on Wormhole by sequence length.

NOTE: Do not combine smaller SDPA K tiles with disabled FP32 matmul dest acc (previously ~0.2 PCC).

**MLP:** the encoder MLP uses ``ttnn.linear(..., activation="gelu")`` on the Wi projection so GELU fuses
into the matmul (fewer device ops / less DRAM traffic than a separate ``ttnn.gelu``).

**Perf:** ``bge_m3_linear_activation_memory_config`` selects **L1 vs DRAM** for encoder activations (matmul,
SDPA, LayerNorm, MLP, attention WO) using ``max_seq_len`` and ``max_batch_size * max_seq_len`` (threshold
``BGE_M3_L1_LINEAR_MAX_SEQ_LEN``). **Do not** use L1 for large batch×seq on only some of these: mixed layouts
hit L1 bank OOM or ``validate_circular_buffer_region`` (``program.cpp:1145``) on Wormhole. **MLP Wi** may
force DRAM for mid-seq; see ``bge_m3_mlp_wi_output_memory_config``.

Encoder SDPA picks the largest **Q** and **K** chunk in ``(256, 128, 64, 32)`` that divides ``seq_len``
(capped at 256 to avoid oversized K tiles), with ``exp_approx_mode`` **False**, to reduce tiling
iterations without the softmax shortcuts that hurt PCC. Matmul linears use the default ``ttnn`` program config.

**S32 / small compile-time ``max_seq_len``:** when ``runtime sequence_length <= 32``, matmul ``core_grid``
row count is **capped at four** (full width) so a single-tile ``M`` does not schedule an entire 8-row
device grid that is mostly idle (a narrower single-row grid regressed wall time in practice). When
``max_seq_len <= 32``, Wormhole matmul / LN / SDPA compute configs also set ``packer_l1_acc=False`` to trim
packer staging on short activations; Blackhole and longer ``max_seq_len`` builds keep ``packer_l1_acc=True``.
"""

from __future__ import annotations

from ttnn.device import is_blackhole as ttnn_is_blackhole
from ttnn.device import is_wormhole_b0 as ttnn_is_wormhole_b0

import ttnn

# Encoder seq cap for L1 on short-seq paths and for ``batch * max_seq_len`` matmul envelope (see module doc).
BGE_M3_L1_LINEAR_MAX_SEQ_LEN = 512

# Wormhole-only: MLP Wi may use DRAM for ``max_seq_len`` in (MATMUL_SHORT, this] when still on L1 elsewhere.
BGE_M3_WORMHOLE_MLP_WI_DRAM_MAX_SEQ_LEN = 127

# Runtime ``seq_len`` at or below this triggers a **capped** matmul ``core_grid`` height (see
# ``bge_m3_matmul_core_grid``). Tuned vs a single-row grid which increased total device time.
BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN = 32
BGE_M3_MATMUL_SHORT_SEQ_CORE_ROWS_CAP = 4

# When compile-time ``max_seq_len`` is at or below this, Wormhole may disable L1 packer accumulation on
# matmul / layernorm / SDPA compute kernels for faster short-sequence runs (long-seq builds unchanged).
BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN = 32


def is_wormhole_family_device(mesh_device: ttnn.MeshDevice) -> bool:
    """True when running on Wormhole (not Blackhole).

    Pytest's ``CreateDevice`` fixture sometimes does not match ``is_wormhole_b0(device)`` the same
    way as mesh bring-up; fall back to ``ttnn.get_arch_name()`` so long-seq HiFi4 still applies.
    """
    if ttnn_is_blackhole(mesh_device):
        return False
    if ttnn_is_wormhole_b0(mesh_device):
        return True
    return "wormhole" in ttnn.get_arch_name().lower()


def max_qkv_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per QKV matmul shard (must divide seq_len when chunking)."""
    return 8192


def max_wo_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per attention output (WO) matmul shard."""
    return 8192


def bge_m3_weight_dram_memory_config() -> ttnn.MemoryConfig:
    """All parameters / embedding tables stay in DRAM."""
    return ttnn.DRAM_MEMORY_CONFIG


def bge_m3_linear_activation_memory_config(
    max_seq_len: int | None,
    max_batch_size: int | None = None,
) -> ttnn.MemoryConfig:
    """Encoder activations (matmul, SDPA, LayerNorm, WO, MLP): L1 when ``max_seq_len`` and ``batch*max_seq_len`` ≤512."""
    s = 0 if max_seq_len is None else int(max_seq_len)
    if s > BGE_M3_L1_LINEAR_MAX_SEQ_LEN:
        return ttnn.DRAM_MEMORY_CONFIG
    b = 1 if max_batch_size is None else max(1, int(max_batch_size))
    if b * s > BGE_M3_L1_LINEAR_MAX_SEQ_LEN:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def bge_m3_mlp_wi_output_memory_config(
    max_seq_len: int | None,
    max_batch_size: int | None,
    mesh_device: ttnn.MeshDevice | None,
) -> ttnn.MemoryConfig:
    """Output memory for MLP ``Wi`` (fused GELU). Uses DRAM when matmul envelope does, or WH mid-seq L1 clash band."""
    base = bge_m3_linear_activation_memory_config(max_seq_len, max_batch_size)
    if base == ttnn.DRAM_MEMORY_CONFIG:
        return base
    if (
        mesh_device is not None
        and is_wormhole_family_device(mesh_device)
        and max_seq_len is not None
        and max_seq_len > BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN
        and max_seq_len <= BGE_M3_WORMHOLE_MLP_WI_DRAM_MAX_SEQ_LEN
    ):
        return ttnn.DRAM_MEMORY_CONFIG
    return base


def bge_m3_matmul_core_grid(
    mesh_device: ttnn.MeshDevice | None,
    sequence_length: int | None = None,
) -> ttnn.CoreGrid:
    """Core grid for ``ttnn.linear(..., core_grid=...)``.

    Uses the device compute grid for moderate/long sequences. For ``sequence_length`` at or below
    ``BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN``, **row count is capped** at ``BGE_M3_MATMUL_SHORT_SEQ_CORE_ROWS_CAP``
    while keeping full width, matching the case where ``M`` is only one tile high (e.g. S32) so extra
    rows are pure scheduling overhead.
    """
    if mesh_device is None:
        gx, gy = 8, 8
    else:
        try:
            g = mesh_device.compute_with_storage_grid_size()
            gx, gy = int(g.x), int(g.y)
        except Exception:
            gx, gy = 8, 8

    if sequence_length is not None and sequence_length <= BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN:
        return ttnn.CoreGrid(y=min(BGE_M3_MATMUL_SHORT_SEQ_CORE_ROWS_CAP, gy), x=gx)
    return ttnn.CoreGrid(y=gy, x=gx)


def bge_m3_matmul_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return a compute kernel config for matmul operations in BGE-M3.

    **Blackhole** and **Wormhole** use **HiFi4** (required for PCC > 0.94 at S8192 on WH). Other archs
    fall back to HiFi2.

    For Wormhole only, when ``max_seq_len <= BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN``, sets
    ``packer_l1_acc=False`` to reduce packer work on very short compile-time envelopes; otherwise
    L1 packer accumulation stays enabled. FP32 destination accumulation is always enabled.
    """
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        fidelity = ttnn.MathFidelity.HiFi4
    else:
        fidelity = ttnn.MathFidelity.HiFi2
    packer_l1_acc = True
    if (
        not ttnn_is_blackhole(mesh_device)
        and is_wormhole_family_device(mesh_device)
        and max_seq_len is not None
        and max_seq_len <= BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN
    ):
        packer_l1_acc = False
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def bge_m3_sdpa_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return the compute kernel config for scaled dot-product attention (SDPA) in BGE-M3.

    **Blackhole** and **Wormhole** use **HiFi4**, matching matmul fidelity for end-to-end PCC.

    Wormhole uses the same ``packer_l1_acc`` policy as ``bge_m3_matmul_compute_kernel_config`` when
    ``max_seq_len`` is at or below ``BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN``.
    """
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        sdpa_fidelity = ttnn.MathFidelity.HiFi4
    else:
        sdpa_fidelity = ttnn.MathFidelity.HiFi2
    packer_l1_acc = True
    if (
        not ttnn_is_blackhole(mesh_device)
        and is_wormhole_family_device(mesh_device)
        and max_seq_len is not None
        and max_seq_len <= BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN
    ):
        packer_l1_acc = False
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=sdpa_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def bge_m3_layernorm_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return the compute kernel config for LayerNorm in BGE-M3 on the given mesh device.

    Matches ``ttnn`` layer_norm defaults in Metal (``layernorm.cpp``): HiFi4, FP32 dest acc.

    For Wormhole only, when ``max_seq_len <= BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN``, uses
    ``packer_l1_acc=False`` (same short-seq policy as matmul).
    """
    packer_l1_acc = True
    if (
        not ttnn_is_blackhole(mesh_device)
        and is_wormhole_family_device(mesh_device)
        and max_seq_len is not None
        and max_seq_len <= BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN
    ):
        packer_l1_acc = False
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )
