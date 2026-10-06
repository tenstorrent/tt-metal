# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the halo-mode conv3d sweep helpers and the LTX-2.5 544x960 halo layer list."""

import pytest

from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, _FP32_BLOCKINGS

from ..wan2_2.bruteforce_conv3d_sweep import (
    HaloSpec,
    build_all_blockings,
    halo_masks,
    halo_sticks,
    prefetch_shard_fits,
    vol2col_cb_first_straddle,
    vol2col_chunks_fit,
    vol2col_rm_pages,
)
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


@pytest.mark.parametrize(
    "blocking, fits",
    # exact_s2_res (C_in=512) blockings from blx03 jobs 269-285: the four that hung and those that passed.
    [
        ((64, 128, 5, 4, 4), False),
        ((64, 128, 5, 8, 2), False),
        ((64, 128, 7, 4, 4), False),
        ((64, 128, 7, 8, 2), False),
        ((64, 64, 3, 8, 8), True),
        ((64, 64, 3, 16, 4), True),
        ((64, 32, 3, 4, 4), True),
        ((64, 32, 3, 8, 2), True),
        ((64, 128, 6, 8, 2), True),
    ],
)
def test_vol2col_chunks_fit_splits_job_484_bisect(blocking, fits):
    assert vol2col_chunks_fit(*blocking[2:]) == fits


@pytest.mark.parametrize("table", [_BLOCKINGS, _DEFAULT_BLOCKINGS, _FP32_BLOCKINGS], ids=["exact", "default", "fp32"])
def test_table_blockings_fit_vol2col_chunks(table):
    # The T-relaxed lookup clamps T_out_block down, so every smaller T must fit too.
    bad = {k: v for k, v in table.items() if not all(vol2col_chunks_fit(t, *v[3:]) for t in range(1, v[2] + 1))}
    assert bad == {}


def _old_vol2col_rm_pages(num_patches):
    return min(num_patches, 32) if num_patches % 32 == 0 else min(num_patches, 64)


@pytest.mark.parametrize(
    "blocking, hung",
    # blx03 jobs 273-285: the 64-page sizing straddles exactly for the four blockings that hung.
    [
        ((5, 4, 4), True),
        ((5, 8, 2), True),
        ((7, 4, 4), True),
        ((7, 8, 2), True),
        ((3, 8, 8), False),
        ((3, 16, 4), False),
        ((3, 4, 4), False),
        ((3, 8, 2), False),
        ((6, 8, 2), False),
    ],
)
def test_vol2col_cb_sizing_straddle_matches_device(blocking, hung):
    num_patches = blocking[0] * blocking[1] * blocking[2]
    assert (vol2col_cb_first_straddle(num_patches, _old_vol2col_rm_pages(num_patches)) is not None) == hung
    assert vol2col_cb_first_straddle(num_patches, vol2col_rm_pages(num_patches)) is None


def test_vol2col_rm_pages_never_straddle():
    bad = [n for n in range(1, 2049) if vol2col_cb_first_straddle(n, vol2col_rm_pages(n)) is not None]
    assert bad == []


def test_vol2col_rm_pages_unchanged_for_guarded_blockings():
    # Blockings the old guard accepted keep their CB size, so their L1 budget and perf do not move.
    changed = [
        n for n in range(1, 2049) if vol2col_chunks_fit(n, 1, 1) and vol2col_rm_pages(n) != _old_vol2col_rm_pages(n)
    ]
    assert changed == []
