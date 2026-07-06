# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""External Conv3d blocking config for the ACE-Step Oobleck VAE decoder.

The VAE is ~99% of the pipeline's device-kernel time, and Conv3d is ~96% of the VAE (Tracy). Every
Oobleck conv otherwise hits the tt_dit conv3d FALLBACK blocking (C_out_block=32, T_out_block=1),
which is far from optimal for the very-long-T upsampled activations.

This module registers per-(in, out, kernel) blockings EXTERNALLY via tt_dit's public
`register_conv3d_configs` — no tt_dit edits, no per-conv hardcoding in the VAE forward. Blockings are
found by the kernel-time sweep (`perf/sweep_vae_conv3d.py`) and only change tiling, never the math
(PCC-verified vs the fallback). Applied once at pipeline build time (`build_vae_decoder`).

Keyed by (arch, dtype) so the model can ship different tuned tables across HW and precisions
(matches the tt_dit `_PRESETS` / `matmul.get_matmul_config` convention). A missing key => no
registration => tt_dit's own fallback (safe no-op).

The 1D-via-3D convs look up their config with kernel (k, 1, 1); we register under that exact key
(NOT _ntuple(k,3)=(k,k,k), which the convs never look up).
"""

from __future__ import annotations

import ttnn

# Blocking tuple order: (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).
# Empty tables => tt_dit fallback (the current, correct baseline). The sweep fills these in.
#
# Keyed: (arch_name, dtype_name) -> { (in_channels, out_channels, (k, 1, 1)): blocking }
#
# bf16 baseline: every VAE conv shape registered with (32,32,1) — the known L1-fitting blocking that
# holds PCC 0.9999 vs the fp32 fallback. Without this, the bf16 path falls to the NONE branch which
# sets C_in_block=in_channels (up to 2048) -> a 64x bigger L1 circular buffer -> OOM. This table lets
# bf16 FIT; the sweep then replaces individual entries with faster fitting blockings.
_VAE_BF16_BLACKHOLE = {
    # Swept (sweep_vae_bf16.py): T_out_block=32 wins on the compute-heavy convs (the old T_out=1 was
    # catastrophically overhead-bound). 99-649x per-shape vs (32,32,1) on the long-T convs.
    (64, 2048, (7, 1, 1)): (64, 256, 32, 1, 1),
    (128, 32, (7, 1, 1)): (128, 32, 8, 1, 1),
    (128, 128, (1, 1, 1)): (128, 128, 8, 1, 1),
    (128, 128, (4, 1, 1)): (128, 128, 32, 1, 1),
    (128, 128, (7, 1, 1)): (128, 128, 32, 1, 1),
    (256, 128, (8, 1, 1)): (256, 128, 32, 1, 1),
    (256, 256, (1, 1, 1)): (256, 128, 8, 1, 1),
    (256, 256, (7, 1, 1)): (256, 256, 32, 1, 1),
    (512, 256, (8, 1, 1)): (256, 128, 32, 1, 1),
    (512, 512, (1, 1, 1)): (256, 128, 8, 1, 1),
    (512, 512, (7, 1, 1)): (256, 128, 32, 1, 1),
    (1024, 512, (12, 1, 1)): (256, 128, 32, 1, 1),
    (1024, 1024, (1, 1, 1)): (256, 128, 8, 1, 1),
    (1024, 1024, (7, 1, 1)): (512, 64, 32, 1, 1),
    (2048, 1024, (20, 1, 1)): (32, 32, 1, 1, 1),  # k20 OOMs at big blocks; sweep separately (TODO)
}

_VAE_CONV3D_PRESETS: dict[tuple[str, str], dict[tuple[int, int, tuple[int, int, int]], tuple[int, ...]]] = {
    ("blackhole", "bfloat16"): _VAE_BF16_BLACKHOLE,
    # fp32 keeps tt_dit's own _FP32_BLLOCKINGS table (no external override needed).
}

# The VAE runs bf16 by default (no fp32 anywhere): halves conv bandwidth on the bandwidth-bound
# audio path. Overridable via the ACESTEP_VAE_DTYPE env / explicit dtype arg for A/B profiling.
_DEFAULT_VAE_DTYPE = ttnn.bfloat16


def vae_default_dtype():
    """Default VAE compute dtype (bf16). Env ACESTEP_VAE_DTYPE=float32 forces fp32 for A/B tests."""
    import os

    v = os.environ.get("ACESTEP_VAE_DTYPE", "").lower()
    if v == "float32":
        return ttnn.float32
    if v == "bfloat16":
        return ttnn.bfloat16
    return _DEFAULT_VAE_DTYPE


def _arch_name(mesh_device) -> str:
    try:
        return str(mesh_device.arch()).lower().split(".")[-1]
    except Exception:
        return "blackhole"


def _dtype_name(dtype) -> str:
    return "bfloat16" if dtype == ttnn.bfloat16 else "float32"


def apply_vae_conv3d_config(mesh_device, dtype) -> int:
    """Register the tuned VAE Conv3d blockings for (arch, dtype). Returns #entries registered.

    No-op (returns 0) when no preset exists for the (arch, dtype) pair — the VAE then uses tt_dit's
    own fallback blocking, i.e. the exact current baseline. Safe to call unconditionally at build.
    """
    from models.tt_dit.utils import conv3d as _c3d

    preset = _VAE_CONV3D_PRESETS.get((_arch_name(mesh_device), _dtype_name(dtype)))
    if not preset:
        return 0
    # Register under the exact (k, 1, 1) key the 1D-via-3D convs look up (bypass the (k,k,k) helper).
    for (c_in, c_out, ks), blk in preset.items():
        _c3d._DEFAULT_BLOCKINGS[(c_in, c_out, ks)] = tuple(blk)
    return len(preset)
