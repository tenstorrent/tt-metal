# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

BUNDLE_TOKENS = 5120
TILE_TOKENS = 32


def _address(global_tile, *, sp, layer, num_layers, table):
    global_bundle_tiles = BUNDLE_TOKENS // TILE_TOKENS
    local_bundle_tiles = global_bundle_tiles // sp
    logical_bundle, tile_in_bundle = divmod(global_tile, global_bundle_tiles)
    owner, local_tile = divmod(tile_in_bundle, local_bundle_tiles)
    physical_bundle = table[logical_bundle]
    physical_batch = physical_bundle * num_layers + layer
    page = physical_batch * local_bundle_tiles * 4 + local_tile * 4
    return owner, page


def test_compact_bundle_table_resolves_natural_tiles_without_subpage_entries():
    sp = 4
    rng = random.Random(0x52)
    logical_bundles = 205
    permutation = list(range(logical_bundles))
    rng.shuffle(permutation)
    layer, num_layers = 7, 21

    seen = set()
    for global_tile in range(logical_bundles * (BUNDLE_TOKENS // TILE_TOKENS)):
        owner, page = _address(global_tile, sp=sp, layer=layer, num_layers=num_layers, table=permutation)
        assert 0 <= owner < sp
        assert (owner, page) not in seen
        seen.add((owner, page))

    # One table entry per 5120-token bundle, independent of SP and 32-token subpages.
    assert len(permutation) == logical_bundles
    assert len(seen) == logical_bundles * (BUNDLE_TOKENS // TILE_TOKENS)


def test_sp4_is_direct_four_owner_partition():
    table = [3]
    for owner in range(4):
        start = owner * 40
        actual = {_address(t, sp=4, layer=0, num_layers=1, table=table)[0] for t in range(start, start + 40)}
        assert actual == {owner}


@pytest.mark.parametrize("tile", [39, 40, 79, 159, 160])
def test_global_bundle_boundary_mapping(tile):
    sp = 4
    owner, page = _address(tile, sp=sp, layer=0, num_layers=1, table=[7, 11])
    global_bundle_tiles = 160
    local_bundle_tiles = global_bundle_tiles // sp
    logical_bundle, in_bundle = divmod(tile, global_bundle_tiles)
    expected_owner, expected_local = divmod(in_bundle, local_bundle_tiles)
    expected_page = [7, 11][logical_bundle] * local_bundle_tiles * 4 + expected_local * 4
    assert (owner, page) == (expected_owner, expected_page)


def test_bundle_major_layer_inner_page_formula():
    table = [9, 2]
    owner, page = _address(160 + 159, sp=4, layer=20, num_layers=21, table=table)
    local_bundle_tiles = 160 // 4
    expected_batch = 2 * 21 + 20
    assert owner == 3
    assert page == expected_batch * local_bundle_tiles * 4 + 39 * 4
