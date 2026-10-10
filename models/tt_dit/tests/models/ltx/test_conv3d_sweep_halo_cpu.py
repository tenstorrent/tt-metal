# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the halo-mode conv3d sweep helpers and the LTX-2.5 544x960 halo layer list."""

import pytest

from models.tt_dit.utils.conv3d import _BLOCKINGS

from ..wan2_2.bruteforce_conv3d_sweep import HaloSpec, build_all_blockings, halo_masks, halo_sticks, prefetch_shard_fits
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
    # The key is the exact shard; the swept shard is at least that, and masks only where it overhangs.
    assert key[1:] == (-(-logical_hw[0] // 2), -(-logical_hw[1] // 4))
    assert H - 2 >= key[1] and W - 2 >= key[2]
    masks = halo_masks(HaloSpec(2, 4, *logical_hw), H - 2, W - 2)
    assert masks == tuple(l if (s * f > l) else 0 for l, s, f in zip(logical_hw, (H - 2, W - 2), (2, 4)))
    combos = build_all_blockings(C_in, C_out, (3, 3, 3), H, W, T, max_t_block=8, hw_product=(16, 32, 64))
    assert tuple(blk) in combos


# Halo-only reader winners (544x960/145f, mesh 4x8). C_in_block must stay 128: it sets the
# reduction order, so keeping it is what keeps the decode bit-identical to the old blocking.
@pytest.mark.parametrize(
    "key, blocking",
    [
        ((4, 8, 128, 128, (3, 3, 3), 147, 68, 60), (128, 64, 6, 4, 8)),  # s4_res
        ((4, 8, 512, 4096, (3, 3, 3), 39, 17, 15), (128, 64, 5, 2, 16)),  # s1_up
        ((4, 8, 512, 512, (3, 3, 3), 39, 17, 15), (64, 256, 2, 2, 8)),  # s1_res
        ((4, 8, 512, 512, (3, 3, 3), 75, 34, 30), (64, 256, 2, 4, 4)),  # s2_res
        ((4, 8, 256, 256, (3, 3, 3), 147, 34, 30), (64, 256, 2, 4, 4)),  # s3_res
        ((4, 8, 256, 512, (3, 3, 3), 147, 34, 30), (64, 256, 2, 4, 4)),  # s3_chg
    ],
    ids=["s4_res", "s1_up", "s1_res", "s2_res", "s3_res", "s3_chg"],
)
def test_ltx25_halo_winners_in_table(key, blocking):
    assert _BLOCKINGS[key] == blocking


@pytest.mark.parametrize(
    "blocking, fits",
    # exact_s2_res (C_in=512) blockings from blx03 job 484 that the factory budget puts on either side.
    [
        ((64, 256, 1, 8, 4), True),
        ((64, 128, 6, 8, 2), True),
        ((64, 64, 3, 8, 8), True),
        ((64, 128, 7, 8, 2), True),
        ((64, 128, 6, 8, 8), False),
        ((64, 128, 6, 16, 4), False),
        ((64, 128, 7, 8, 8), False),
    ],
)
def test_prefetch_shard_fits_matches_factory_budget(blocking, fits):
    assert prefetch_shard_fits(*blocking, (3, 3, 3), 512) == fits


def test_ltx_table_blockings_without_prefetch_shard():
    # In halo mode conv3d rejects a blocking without an L1 prefetch shard (the direct reader drops the halo).
    no_shard = {
        k
        for k, v in _BLOCKINGS.items()
        if k[2] in (128, 256, 512, 1024) and k[4] == (3, 3, 3) and not prefetch_shard_fits(*v, k[4], k[2])
    }
    assert no_shard == set()
