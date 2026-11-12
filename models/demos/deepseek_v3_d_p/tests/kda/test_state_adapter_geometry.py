# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the KDA state migration contract: segment counts, bytes and numbering."""

from dataclasses import replace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    assemble_convolution,
    assemble_recurrent,
)

LAYOUTS = [(1, 8), (2, 4), (4, 2), (8, 4)]
LAYOUT_IDS = [f"SP{sp}xTP{tp}" for sp, tp in LAYOUTS]


def k3_geometry(sp: int, tp: int) -> KdaContractGeometry:
    return KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(sp, tp), sp_axis=0, tp_axis=1)


@pytest.mark.parametrize("sp,tp", LAYOUTS, ids=LAYOUT_IDS)
def test_k3_contract_counts_and_bytes(sp: int, tp: int) -> None:
    geometry = k3_geometry(sp, tp)

    assert geometry.local_heads == 96 // tp
    assert geometry.recurrent_shape == (1, 96 // tp, 128, 128)
    # One 64-channel segment per page: 3 * local_heads * 128 / 64 pages of 3 * 64 elements.
    assert geometry.convolution_shape == (1, 3 * (96 // tp) * 128 // 64, 3 * 64)
    assert geometry.recurrent_segments_per_layer == 384
    assert geometry.convolution_segments_per_layer == 576
    assert geometry.recurrent_segment_bytes == 16_384
    assert geometry.convolution_segment_bytes == 384
    # Each segment is stored once on the mesh.
    assert geometry.recurrent_shards_per_layer * sp * tp == 384
    assert geometry.convolution_shards_per_layer * sp * tp == 576
    assert geometry.convolution_slab_row_width == 192
    # Per-device payload of one layer, the numbers the migration budget is quoted in.
    assert geometry.stored_heads == 96 // (sp * tp)
    assert (
        geometry.recurrent_shards_per_layer * geometry.recurrent_segment_bytes == geometry.stored_heads * 128 * 128 * 4
    )
    assert (
        geometry.convolution_shards_per_layer * geometry.convolution_segment_bytes
        == geometry.stored_heads * 3 * 3 * 128 * 2
    )


@pytest.mark.parametrize("sp,tp", LAYOUTS, ids=LAYOUT_IDS)
def test_recurrent_segments_are_a_bijection(sp: int, tp: int) -> None:
    geometry = k3_geometry(sp, tp)
    seen = {}
    for tp_col in range(tp):
        for h_local in range(geometry.local_heads):
            for band in range(geometry.bands):
                segment = geometry.recurrent_segment(tp_col, h_local, band)
                assert 0 <= segment < geometry.recurrent_segments_per_layer
                assert geometry.decompose_recurrent(segment) == (tp_col, h_local, band)
                seen[segment] = (tp_col, h_local, band)
    assert len(seen) == geometry.recurrent_segments_per_layer
    # Head-major then band: consecutive segments of one head are its four V-bands.
    assert geometry.recurrent_segment(0, 0, 3) + 1 == geometry.recurrent_segment(0, 1, 0)
    # Global head g = tp_col * local_heads + h_local, so column 1 starts right after column 0's heads.
    assert geometry.recurrent_segment(1, 0, 0) == geometry.local_heads * geometry.bands


@pytest.mark.parametrize("sp,tp", LAYOUTS, ids=LAYOUT_IDS)
def test_convolution_segments_follow_the_golden_branch_order(sp: int, tp: int) -> None:
    geometry = k3_geometry(sp, tp)
    seen = {}
    for branch in range(3):
        for tp_col in range(tp):
            for h_local in range(geometry.local_heads):
                for half in range(geometry.halves):
                    segment = geometry.convolution_segment(branch, tp_col, h_local, half)
                    assert 0 <= segment < geometry.convolution_segments_per_layer
                    assert geometry.decompose_convolution(segment) == (branch, tp_col, h_local, half)
                    seen[segment] = (branch, tp_col, h_local, half)
    assert len(seen) == geometry.convolution_segments_per_layer
    # [all q | all k | all v]: every q segment precedes every k segment, whatever the TP column.
    last_q = max(s for s, (branch, *_rest) in seen.items() if branch == 0)
    first_k = min(s for s, (branch, *_rest) in seen.items() if branch == 1)
    assert last_q < first_k


@pytest.mark.parametrize("sp,tp", LAYOUTS, ids=LAYOUT_IDS)
def test_local_shards_match_the_per_chip_channel_layout(sp: int, tp: int) -> None:
    """A chip's native convolution row is [q_local | k_local | v_local], D channels per head inside a branch; its
    store keeps that branch-major order over its own heads."""
    geometry = k3_geometry(sp, tp)
    columns = []
    for branch in range(3):
        for h_local in range(geometry.local_heads):
            for half in range(geometry.halves):
                column = geometry.convolution_native_page(branch, h_local, half)
                channel = branch * geometry.local_heads * 128 + h_local * 128 + half * 64
                assert column * 64 == channel
                columns.append(column)
    assert columns == list(range(3 * geometry.local_heads * geometry.halves))
    stored = [
        geometry.convolution_stored_page(branch, j, half)
        for branch in range(3)
        for j in range(geometry.stored_heads)
        for half in range(geometry.halves)
    ]
    assert stored == list(range(geometry.convolution_shards_per_layer))

    # Batches are contiguous shard runs, so batch 1 starts where batch 0's last shard ends.
    assert geometry.recurrent_local_shard(1, 0, 0) == geometry.recurrent_shards_per_layer
    assert geometry.convolution_local_shard(1, 0, 0, 0) == geometry.convolution_shards_per_layer
    last = geometry.recurrent_local_shard(0, geometry.stored_heads - 1, geometry.bands - 1)
    assert last + 1 == geometry.recurrent_shards_per_layer


def test_assembly_reconstructs_global_state_from_segments() -> None:
    geometry = k3_geometry(2, 4)
    torch.manual_seed(0)
    recurrent = torch.randn(96, 128, 128)
    convolution = torch.randn(3, 3 * 96 * 128).to(torch.bfloat16)

    recurrent_segments = {}
    for segment in range(geometry.recurrent_segments_per_layer):
        tp_col, h_local, band = geometry.decompose_recurrent(segment)
        head = tp_col * geometry.local_heads + h_local
        recurrent_segments[segment] = recurrent[head, :, 32 * band : 32 * band + 32]
    assert torch.equal(assemble_recurrent(recurrent_segments, geometry), recurrent)

    convolution_segments = {}
    for segment in range(geometry.convolution_segments_per_layer):
        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
        head = tp_col * geometry.local_heads + h_local
        start = (branch * 96 + head) * 128 + half * 64
        convolution_segments[segment] = convolution[:, start : start + 64]
    assert torch.equal(assemble_convolution(convolution_segments, geometry), convolution)


@pytest.mark.parametrize("sp,tp", [(1, 5), (2, 7), (1, 96 * 2)], ids=["tp5", "tp7", "tp192"])
def test_geometry_rejects_heads_not_divisible_by_tp(sp: int, tp: int, expect_error) -> None:
    with expect_error(ValueError, "cannot be divided"):
        k3_geometry(sp, tp)


def test_geometry_rejects_same_sp_and_tp_axis(expect_error) -> None:
    with expect_error(ValueError, "distinct 2D SP/TP axes"):
        KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(2, 4), sp_axis=1, tp_axis=1)


def _geometry_with_head_dim(head_dim: int) -> KdaContractGeometry:
    config = replace(kimi_k3_kda_config(), head_k_dim=head_dim, head_v_dim=head_dim)
    return KdaContractGeometry.from_kda_config(config, mesh_shape=(8, 4), sp_axis=0, tp_axis=1)


@pytest.mark.parametrize("head_dim,pages,bands", [(64, 1, 2), (128, 2, 4), (256, 4, 8)])
def test_geometry_accepts_head_dim_of_whole_convolution_pages(head_dim: int, pages: int, bands: int) -> None:
    geometry = _geometry_with_head_dim(head_dim)

    assert geometry.halves == pages
    assert geometry.bands == bands
    assert geometry.convolution_shape == (1, 3 * 24 * pages, 3 * 64)


@pytest.mark.parametrize("head_dim", [32, 96])
def test_geometry_rejects_head_dim_of_partial_convolution_pages(head_dim: int, expect_error) -> None:
    with expect_error(ValueError, f"head_dim {head_dim} must be a multiple of 64"):
        _geometry_with_head_dim(head_dim)


# --- SP-sharded storage: each head stored once on the mesh ---

# (SP, TP, global heads, H_store): Galaxy K3; the LoudBox / QuietBox perf layouts, which keep Galaxy's 24 heads per
# chip; the QuietBox SP2xTP2 report layout (48 heads).
SHARDED_LAYOUTS = [(8, 4, 96, 3), (8, 1, 24, 3), (4, 1, 24, 6), (2, 2, 48, 12)]
SHARDED_IDS = ["galaxy-SP8xTP4", "loudbox-SP8xTP1", "quietbox-SP4xTP1", "quietbox-SP2xTP2"]


def _geometry(sp: int, tp: int, heads: int) -> KdaContractGeometry:
    return KdaContractGeometry(
        num_heads=heads, head_dim=128, conv_history=3, sequence_parallel_size=sp, tensor_parallel_size=tp
    )


@pytest.mark.parametrize("sp,tp,heads,stored", SHARDED_LAYOUTS, ids=SHARDED_IDS)
def test_sharded_geometry_stores_each_head_once(sp: int, tp: int, heads: int, stored: int) -> None:
    geometry = _geometry(sp, tp, heads)

    assert geometry.stored_heads == stored == heads // (sp * tp)
    assert geometry.recurrent_shards_per_layer == stored * 4
    assert geometry.convolution_shards_per_layer == 3 * stored * 2
    assert geometry.stored_recurrent_shape == (1, stored, 128, 128)
    assert geometry.stored_convolution_shape == (1, 6 * stored, 192)
    # Per-chip bytes per layer and slot: 1/SP of the TP column's heads.
    assert geometry.recurrent_shards_per_layer * geometry.recurrent_segment_bytes == stored * 65_536
    assert geometry.convolution_shards_per_layer * geometry.convolution_segment_bytes == stored * 2_304
    # Every global segment is stored exactly once on the mesh.
    assert geometry.recurrent_shards_per_layer * sp * tp == geometry.recurrent_segments_per_layer
    assert geometry.convolution_shards_per_layer * sp * tp == geometry.convolution_segments_per_layer


# Hand-checked: segment -> (SP row, TP column, stored head j, shard within the layer). H_local = heads / TP.
#   recurrent   G = g * 4 + band,                g = tp * H_local + row * H_store + j, shard = j * 4 + band
#   convolution G = (branch * heads + g) * 2 + half, shard = (branch * H_store + j) * 2 + half
HAND_CHECKED = [
    # Galaxy, H_local 24, H_store 3. g 31 = column 1, local head 7 = row 2, j 1.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_recurrent", 126, (2, 1, 1, 6)),
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_recurrent", 0, (0, 0, 0, 0)),
    # v, g 95 = column 3, local head 23 = row 7, j 2, half 1: the last stored page.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_convolution", 575, (7, 3, 2, 17)),
    # k, g 26 = column 1, local head 2 = row 0, j 2, half 0.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_convolution", 244, (0, 1, 2, 10)),
    # LoudBox SP8xTP1, 24 heads: g 13 = row 4, j 1, band 3.
    ("loudbox-SP8xTP1", (8, 1, 24), "kda_recurrent", 55, (4, 0, 1, 7)),
    # q, g 23 = row 7, j 2, half 0.
    ("loudbox-SP8xTP1", (8, 1, 24), "kda_convolution", 46, (7, 0, 2, 4)),
    # QuietBox SP4xTP1, H_store 6: g 17 = row 2, j 5, band 1.
    ("quietbox-SP4xTP1", (4, 1, 24), "kda_recurrent", 69, (2, 0, 5, 21)),
    # k, g 7 = row 1, j 1, half 1.
    ("quietbox-SP4xTP1", (4, 1, 24), "kda_convolution", 63, (1, 0, 1, 15)),
    # SP2xTP2, 48 heads, H_local 24, H_store 12: g 37 = column 1, local head 13 = row 1, j 1.
    ("quietbox-SP2xTP2", (2, 2, 48), "kda_recurrent", 148, (1, 1, 1, 4)),
    # v, g 11 = column 0, row 0, j 11, half 1.
    ("quietbox-SP2xTP2", (2, 2, 48), "kda_convolution", 215, (0, 0, 11, 71)),
]


@pytest.mark.parametrize(
    "mesh,kind,segment,expected", [case[1:] for case in HAND_CHECKED], ids=[c[0] for c in HAND_CHECKED]
)
def test_sharded_segment_owner_is_hand_checked(mesh, kind, segment, expected) -> None:
    geometry = _geometry(*mesh)
    if kind == "kda_recurrent":
        tp_col, h_local, band = geometry.decompose_recurrent(segment)
        sp_row, j = geometry.stored_head(h_local)
        shard = geometry.recurrent_local_shard(0, j, band)
    else:
        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
        sp_row, j = geometry.stored_head(h_local)
        shard = geometry.convolution_stored_page(branch, j, half)
    assert (sp_row, tp_col, j, shard) == expected
    assert geometry.local_head(sp_row, j) == h_local


@pytest.mark.parametrize("sp,tp,heads,stored", SHARDED_LAYOUTS, ids=SHARDED_IDS)
def test_gather_over_sp_rebuilds_the_native_carry(sp: int, tp: int, heads: int, stored: int) -> None:
    """The layer's import gather, on host: concatenating the SP rows' stores on dim 1 is the
    native carry -- the recurrent store as is, the convolution store viewed as [3, H_store * halves, 192] -- and
    slicing the native carry the same way is the export partition. No permute either way."""
    geometry = _geometry(sp, tp, heads)
    local, halves, width = geometry.local_heads, geometry.halves, geometry.convolution_slab_row_width
    torch.manual_seed(1)
    native_recurrent = torch.randn(geometry.recurrent_shape)
    native_convolution = torch.randn(geometry.convolution_shape)

    stores_recurrent, stores_convolution = [], []
    for row in range(sp):
        recurrent = torch.empty(geometry.stored_recurrent_shape)
        convolution = torch.empty(geometry.stored_convolution_shape)
        for j in range(stored):
            h_local = geometry.local_head(row, j)
            recurrent[0, j] = native_recurrent[0, h_local]
            for branch in range(3):
                for half in range(halves):
                    convolution[0, geometry.convolution_stored_page(branch, j, half)] = native_convolution[
                        0, geometry.convolution_native_page(branch, h_local, half)
                    ]
        stores_recurrent.append(recurrent)
        stores_convolution.append(convolution)

    assert torch.equal(torch.cat(stores_recurrent, dim=1), native_recurrent)
    gathered = torch.cat([store.view(3, stored * halves, width) for store in stores_convolution], dim=1)
    assert torch.equal(gathered.view(geometry.convolution_shape), native_convolution)
    for row in range(sp):
        assert torch.equal(native_recurrent[:, row * stored : (row + 1) * stored], stores_recurrent[row])
        partition = native_convolution.view(3, local * halves, width)[
            :, row * stored * halves : (row + 1) * stored * halves
        ]
        assert torch.equal(partition.reshape(geometry.stored_convolution_shape), stores_convolution[row])


@pytest.mark.parametrize("sp,tp,heads", [(5, 4, 96), (5, 1, 24), (16, 4, 96)], ids=["sp5tp4", "sp5tp1", "sp16tp4"])
def test_sharding_rejects_heads_not_divisible_by_sp(sp: int, tp: int, heads: int, expect_error) -> None:
    with expect_error(ValueError, "cannot be sharded over SP"):
        _geometry(sp, tp, heads)


def test_sharded_store_indices_are_range_checked(expect_error) -> None:
    geometry = _geometry(8, 4, 96)
    with expect_error(ValueError, "j=3 out of range"):
        geometry.recurrent_local_shard(0, 3, 0)
    with expect_error(ValueError, "sp_row=8 out of range"):
        geometry.local_head(8, 0)
