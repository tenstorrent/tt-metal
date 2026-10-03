# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the halo-mode conv3d sweep helpers and the LTX-2.5 544x960 halo layer list."""

import pytest

from models.tt_dit.utils.conv3d import _BLOCKINGS

from ..wan2_2.bruteforce_conv3d_sweep import HaloSpec, build_all_blockings, halo_masks, halo_sticks, prefetch_shard_fits
from .bruteforce_conv3d_sweep_ltx import _SWEEP_LAYERS_LTX25_544P_145F_HALO, _SWEEP_LAYERS_LTX25_544P_145F_HALO_EXACT


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
    combos = build_all_blockings(C_in, C_out, (3, 3, 3), H, W, T, max_t_block=8, hw_product=(16, 32, 64))
    assert tuple(blk) in combos


@pytest.mark.parametrize(
    "name, C_in, C_out, T, H, W, key, logical_hw",
    _SWEEP_LAYERS_LTX25_544P_145F_HALO_EXACT,
    ids=[l[0] for l in _SWEEP_LAYERS_LTX25_544P_145F_HALO_EXACT],
)
def test_ltx25_halo_exact_layers(name, C_in, C_out, T, H, W, key, logical_hw):
    assert (4, 8, C_in, C_out, (3, 3, 3), *key) in _BLOCKINGS, f"{name}: no _BLOCKINGS entry for {key}"
    # Exact shards: the unpadded shard is the key's output dims and tiles the logical size, so no chip masks.
    assert (T, H - 2, W - 2) == key
    assert halo_masks(HaloSpec(2, 4, *logical_hw), H - 2, W - 2) == (0, 0)
    assert build_all_blockings(C_in, C_out, (3, 3, 3), H, W, T, max_t_block=8, hw_product=(16, 32, 64))


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
    # These LTX table blockings get no L1 prefetch shard, so in halo mode conv3d now rejects them
    # (before, the direct reader ran and dropped the halo). They need new blockings.
    no_shard = {
        k
        for k, v in _BLOCKINGS.items()
        if k[2] in (128, 256, 512, 1024) and k[4] == (3, 3, 3) and not prefetch_shard_fits(*v, k[4], k[2])
    }
    assert no_shard == {
        (4, 8, 1024, 1024, (3, 3, 3), 22, 10, 8),
        (4, 8, 1024, 1024, (3, 3, 3), 22, 5, 4),
        (4, 8, 128, 1024, (3, 3, 3), 22, 5, 4),
        (4, 8, 128, 1024, (3, 3, 3), 21, 5, 4),
        (2, 4, 128, 128, (3, 3, 3), 147, 136, 120),
    }
