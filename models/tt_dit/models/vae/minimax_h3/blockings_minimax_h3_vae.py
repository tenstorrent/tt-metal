# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Swept matmul blockings for the MiniMax-H3 visual VAE decoder.

Without these, all four per-layer shapes miss ``_grid_config_lookup`` and
``get_matmul_config`` falls through to the generic ``8x8x8`` -- which it logs as
``No known best blocking for (M, K, N) = ...``. ``M = 1824`` is fixed by the architecture
rather than by the request: a work unit is always a ``(1, 24, 7, 16, 16)`` latent, so
1792 patches plus a tile-padded 5-token suffix, at every resolution and duration.

The decoder's forward runs no CCL, so these are plain single-device ``minimal_matmul``
calls on the ``11x10`` grid a BH Galaxy clamps to. ``to_out`` and ``ff2`` reach
``dit_minimal_matmul_addcmul_fused`` instead, which changes only the write-back epilogue;
the lookup is keyed on ``(M, K, N, grid)`` alone, so one entry serves both ops either way.

Sizing, so nobody re-runs the sweep expecting more (BH Galaxy, min-of-N device kernel
duration, subblock ``(2, 2)`` throughout because ``fp32_dest_acc_en`` halves DEST):

| shape | default | swept | delta |
|---|---|---|---|
| ``to_qkv`` (1824, 2048, 6144) | 348.1 us | 260.6 us | **-25.1 %** |
| ``ff1`` (1824, 2048, 16384) | 812.5 us | 727.5 us | -10.5 % |
| ``to_out`` (1824, 2048, 2048) | 100.2 us | 93.6 us | -6.6 % |
| ``ff2`` (1824, 8192, 2048) | 278.9 us | 275.9 us | -1.1 % |

**Net is 0.18 ms per layer, 6.6 ms per wave, ~0.14 s of a 4.65 s decode stage.** Free at
runtime, and far inside the +-8 % run-to-run band, so it is only ever visible at the op
level. It also bounds the lever: 1.54 ms of default blocking per layer against a measured
137.8 ms/wave decoder is 40 % matmul, and the swept optimum is 12 % away -- the decoder's
matmuls are near a hardware ceiling, not mis-blocked, and the other 60 % of the stage is
where the remaining headroom lives.

``get_matmul_config``'s catastrophic tail is real but is not what the default hits: the
``to_qkv`` sweep's median combo is **+1405 %** and its worst +8245 %, while ``8x8x8`` lands
at +33.6 %. Re-sweep with ``utils/sweep_mm_block_sizes.py`` (shape ids
``1824_<K>_<N>_11x10_mm_*``).
"""

from __future__ import annotations

from ....utils.matmul import register_matmul_configs

# (M, K, N) -> (M_block, K_block, N_block, (subblock_h, subblock_w))
_DECODER_BLOCKINGS_11X10 = {
    (1824, 2048, 6144): (14, 2, 10, (2, 2)),  # to_qkv, fused Q/K/V
    (1824, 2048, 2048): (6, 2, 6, (2, 2)),  # to_out
    (1824, 2048, 16384): (6, 4, 12, (2, 2)),  # ff1, N doubled by the packed [gate|up] weight
    (1824, 8192, 2048): (14, 4, 6, (2, 2)),  # ff2
}


def register_h3_vae_decoder_blockings() -> int:
    """Register the swept blockings. Returns the number of entries added. Idempotent."""
    register_matmul_configs({"11x10": _DECODER_BLOCKINGS_11X10})
    return len(_DECODER_BLOCKINGS_11X10)
