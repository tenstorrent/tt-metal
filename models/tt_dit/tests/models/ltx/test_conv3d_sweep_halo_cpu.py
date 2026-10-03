# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the halo-mode conv3d sweep helpers and the LTX-2.5 544x960 halo layer list."""

import pytest

from models.tt_dit.utils.conv3d import _BLOCKINGS

from ..wan2_2.bruteforce_conv3d_sweep import HaloSpec, build_all_blockings, halo_masks, halo_sticks
from .bruteforce_conv3d_sweep_ltx import _SWEEP_LAYERS_LTX25_544P_145F_HALO


@pytest.mark.parametrize(
    "T, H, W, sticks",
    # Halo buffer sizes of the s0 and s2 decoder convs in the t96 device profile.
    [(19, 9, 8, 722), (73, 36, 32, 10220)],
)
def test_halo_sticks_match_profiled_buffers(T, H, W, sticks):
    assert halo_sticks(T, H, W, 1, 1) == sticks


def test_halo_masks_only_when_shards_overhang():
    assert halo_masks(HaloSpec(2, 4, 68, 120), 36, 32) == (68, 120)
    assert halo_masks(HaloSpec(2, 4, 72, 128), 36, 32) == (0, 0)
    assert halo_masks(HaloSpec(2, 4), 36, 32) == (0, 0)


@pytest.mark.parametrize(
    "name, C_in, C_out, T, H, W, key, logical_hw",
    _SWEEP_LAYERS_LTX25_544P_145F_HALO,
    ids=[l[0] for l in _SWEEP_LAYERS_LTX25_544P_145F_HALO],
)
def test_ltx25_halo_layers_key_into_table(name, C_in, C_out, T, H, W, key, logical_hw):
    blk = _BLOCKINGS.get((4, 8, C_in, C_out, (3, 3, 3), *key))
    assert blk is not None, f"{name}: no _BLOCKINGS entry for {key}"
    # The shard overhangs the logical size on both axes, as in production.
    assert halo_masks(HaloSpec(2, 4, *logical_hw), H - 2, W - 2) == logical_hw
    combos = build_all_blockings(C_in, C_out, (3, 3, 3), H, W, T, max_t_block=8, hw_product=32)
    assert tuple(blk) in combos


# Halo-only reader winners (544x960/145f, mesh 4x8). C_in_block must stay 128: it sets the
# reduction order, so keeping it is what keeps the decode bit-identical to the old blocking.
@pytest.mark.parametrize(
    "key, blocking",
    [
        ((4, 8, 128, 128, (3, 3, 3), 147, 68, 60), (128, 64, 6, 4, 8)),  # s4_res
        ((4, 8, 512, 4096, (3, 3, 3), 39, 17, 15), (128, 64, 5, 2, 16)),  # s1_up
    ],
    ids=["s4_res", "s1_up"],
)
def test_ltx25_halo_winners_in_table(key, blocking):
    assert _BLOCKINGS[key] == blocking
