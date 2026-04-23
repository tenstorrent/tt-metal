# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute-kernel and activation-memory policy for the BGE-M3 encoder on TTNN.

- **Matmul / SDPA:** HiFi2 on archs that are not Blackhole or Wormhole; **HiFi4** with FP32 dest acc on
  Blackhole and Wormhole (Wormhole previously used HiFi2, which drifts on long sequence).
- **LayerNorm:** HiFi4 + FP32 on all archs (aligned with ttnn layer_norm in Metal).
- **Activations:** ``bge_m3_linear_activation_memory_config`` (and MLP ``Wi`` via
  ``bge_m3_mlp_wi_output_memory_config`` on Wormhole) pick L1 vs DRAM from ``max_seq_len`` and
  ``max_batch_size * max_seq_len``. Do not use mixed L1/DRAM envelopes across matmul, SDPA, LayerNorm, MLP,
  and attention WO, or Wormhole can OOM or fail circular-buffer validation. MLP fuses GELU with
  ``ttnn.linear(..., activation="gelu")`` on Wi.

- **Caveat:** Do not combine smaller SDPA K-tiles with FP32 matmul dest acc disabled (historically ~0.2 PCC).
- **Attention (encoder):** Q chunk 128, K from (256, 128) dividing ``seq_len``,
  ``exp_approx_mode`` for 128-aligned lengths; linears use ``bge_m3_matmul_*`` / core grid helpers.
- **Short ``sequence_length`` (e.g. 32):** ``bge_m3_matmul_core_grid`` may cap height to four rows only for
  single batch; multi-batch short runs use the full grid. Wormhole may set ``packer_l1_acc=False`` on
  matmul, SDPA, and layernorm for single-batch short ``max_seq_len`` (see ``_wormhole_use_fast_packer_offload``).

``max_qkv_mm_chunk_seq_len`` / ``max_wo_mm_chunk_seq_len`` return 8192 (chunk ceiling); ``max_seq_len`` on
the compute helpers is for API compatibility and Wormhole packer policy, not a fidelity downgrade.
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
# matmul / layernorm / SDPA compute kernels — only for **single-batch** short-seq (see packer helpers).
BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN = 32


def _wormhole_use_fast_packer_offload(
    max_seq_len: int | None,
    max_batch_size: int | None,
) -> bool:
    """True → ``packer_l1_acc=False`` on Wormhole. Single-batch S32 only; multi-batch matches S64+ policy."""
    if max_seq_len is None:
        return False
    if int(max_seq_len) > BGE_M3_FAST_PACKER_OFFLOAD_MAX_SEQ_LEN:
        return False
    b = 1 if max_batch_size is None else max(1, int(max_batch_size))
    return b <= 1


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
    batch_size: int | None = None,
) -> ttnn.CoreGrid:
    """Core grid for ``ttnn.linear(..., core_grid=...)``.

    For ``sequence_length`` at or below ``BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN``, row count is **capped** at
    ``BGE_M3_MATMUL_SHORT_SEQ_CORE_ROWS_CAP`` only when ``batch_size`` is omitted or ``<= 1`` (single-batch
    short-seq: one-tile ``M``). For ``batch_size > 1``, the full grid height is used so multi-batch short
    sequences (e.g. batch 25, S32) are not under-scheduled on cores.
    """
    if mesh_device is None:
        gx, gy = 8, 8
    else:
        try:
            g = mesh_device.compute_with_storage_grid_size()
            gx, gy = int(g.x), int(g.y)
        except Exception:
            gx, gy = 8, 8

    use_short_seq_row_cap = sequence_length is not None and sequence_length <= BGE_M3_MATMUL_SHORT_SEQ_MAX_LEN
    single_batch = batch_size is None or int(batch_size) <= 1
    if use_short_seq_row_cap and single_batch:
        return ttnn.CoreGrid(y=min(BGE_M3_MATMUL_SHORT_SEQ_CORE_ROWS_CAP, gy), x=gx)
    return ttnn.CoreGrid(y=gy, x=gx)


def bge_m3_matmul_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
    max_batch_size: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return a compute kernel config for matmul operations in BGE-M3.

    **Blackhole** and **Wormhole** use **HiFi4** (required for PCC > 0.94 at S8192 on WH). Other archs
    fall back to HiFi2.

    For Wormhole only, ``packer_l1_acc=False`` when ``_wormhole_use_fast_packer_offload`` (single-batch
    short ``max_seq_len``); multi-batch S32 keeps ``packer_l1_acc=True``. FP32 destination acc stays on.
    """
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        fidelity = ttnn.MathFidelity.HiFi4
    else:
        fidelity = ttnn.MathFidelity.HiFi2
    packer_l1_acc = True
    if not ttnn_is_blackhole(mesh_device) and is_wormhole_family_device(mesh_device):
        if _wormhole_use_fast_packer_offload(max_seq_len, max_batch_size):
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
    max_batch_size: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """SDPA compute config: same HiFi2/HiFi4 and ``packer_l1_acc`` rules as ``bge_m3_matmul_compute_kernel_config``."""
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        sdpa_fidelity = ttnn.MathFidelity.HiFi4
    else:
        sdpa_fidelity = ttnn.MathFidelity.HiFi2
    packer_l1_acc = True
    if not ttnn_is_blackhole(mesh_device) and is_wormhole_family_device(mesh_device):
        if _wormhole_use_fast_packer_offload(max_seq_len, max_batch_size):
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
    max_batch_size: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """LayerNorm: HiFi4 + FP32 dest acc; ``packer_l1_acc`` matches ``bge_m3_matmul_compute_kernel_config``."""
    packer_l1_acc = True
    if not ttnn_is_blackhole(mesh_device) and is_wormhole_family_device(mesh_device):
        if _wormhole_use_fast_packer_offload(max_seq_len, max_batch_size):
            packer_l1_acc = False
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )
