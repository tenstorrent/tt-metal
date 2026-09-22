# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the KDA state migration contract: segment counts, bytes and numbering."""

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
    assert geometry.convolution_shape == (1, 3, 3 * (96 // tp) * 128)
    assert geometry.recurrent_segments_per_layer == 384
    assert geometry.convolution_segments_per_layer == 576
    assert geometry.recurrent_segment_bytes == 16_384
    assert geometry.convolution_segment_bytes == 384
    assert geometry.recurrent_shards_per_layer * tp == 384
    assert geometry.convolution_shards_per_layer * tp == 576
    assert geometry.convolution_slab_row_width == 192
    # Per-device payload of one layer, the numbers the migration budget is quoted in.
    assert (
        geometry.recurrent_shards_per_layer * geometry.recurrent_segment_bytes == geometry.local_heads * 128 * 128 * 4
    )
    assert (
        geometry.convolution_shards_per_layer * geometry.convolution_segment_bytes
        == geometry.local_heads * 3 * 3 * 128 * 2
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
    """A chip's convolution row is [q_local | k_local | v_local], D channels per head inside a branch."""
    geometry = k3_geometry(sp, tp)
    columns = []
    for branch in range(3):
        for h_local in range(geometry.local_heads):
            for half in range(geometry.halves):
                column = geometry.convolution_local_column(branch, h_local, half)
                channel = branch * geometry.local_heads * 128 + h_local * 128 + half * 64
                assert column * 64 == channel
                columns.append(column)
    assert columns == list(range(geometry.convolution_shards_per_layer))

    # Batches are contiguous shard runs, so batch 1 starts where batch 0's last shard ends.
    assert geometry.recurrent_local_shard(1, 0, 0) == geometry.recurrent_shards_per_layer
    assert geometry.convolution_local_shard(1, 0, 0, 0) == geometry.convolution_shards_per_layer
    last = geometry.recurrent_local_shard(0, geometry.local_heads - 1, geometry.bands - 1)
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
