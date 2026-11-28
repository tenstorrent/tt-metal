# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every value of the KNOBS dict (issuers_per_core: perf lamp L1; partial_tile_read: the
valid-row span, the two face-row spans of perf lamp L2, the whole tile; issuer_placement,
issuer_split and bank_stagger: the Refinement 2 issue-path levers) is bit exact on the width and
height placements, and the host geometry of the span and the issuer split matches op_design.md.

flux58: every tile row is partial (valid-row span). h33_width: tile row 0 full, tile row 1
partial, so both spans run in one block. flux58_640: 20 tiles per core. height_2x1x33x128_4cores:
one tile row per height shard, alternating full and partial across images. NaN in the input's
tile padding must not reach the logical values.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    check_output,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import describe_plan, g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor

H33_WIDTH = copy.deepcopy(FLUX_58)
H33_WIDTH["shape"] = [33, 5120]
H33_WIDTH["out_memory"]["shard_shape"] = [64, 160]

FLUX_58_640 = copy.deepcopy(FLUX_58)
FLUX_58_640["out_memory"]["grid"] = [[0, 0], [7, 0]]
FLUX_58_640["out_memory"]["shard_shape"] = [32, 640]

# 2 images of 2 tile rows (the second partial) over 4 height shards of one tile row each.
HEIGHT_4_CORES = copy.deepcopy(FLUX_58)
HEIGHT_4_CORES["shape"] = [2, 1, 33, 128]
HEIGHT_4_CORES["out_memory"].update(scheme="height_sharded", grid=[[0, 0], [3, 0]], shard_shape=[32, 128])

SCENARIOS = pytest.mark.parametrize(
    "scenario",
    [FLUX_58, H33_WIDTH, FLUX_58_640, HEIGHT_4_CORES],
    ids=["flux58", "h33_width", "flux58_640", "height_2x1x33x128_4cores"],
)


def _run_exact(scenario, device):
    torch.manual_seed(0)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=build_memory_config(scenario["in_memory"], device),
    )
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = ttnn.to_torch(ttnn_input).clone()
    output_config = build_memory_config(scenario["out_memory"], device)
    output = g6_interleaved_to_sharded(ttnn_input, output_config)
    check_output(
        output,
        expected,
        shape=scenario["shape"],
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=output_config,
        gate="exact",
    )
    return ttnn_input, output_config


@SCENARIOS
@pytest.mark.parametrize("partial_tile_read", descriptor.KNOB_VALUES["partial_tile_read"])
@pytest.mark.parametrize("issuers_per_core", descriptor.KNOB_VALUES["issuers_per_core"])
def test_knob_values_bit_exact(scenario, issuers_per_core, partial_tile_read, device, monkeypatch):
    monkeypatch.setitem(descriptor.KNOBS, "issuers_per_core", issuers_per_core)
    monkeypatch.setitem(descriptor.KNOBS, "partial_tile_read", partial_tile_read)
    ttnn_input, output_config = _run_exact(scenario, device)
    plan = describe_plan(ttnn_input, output_config)
    assert plan["issuers_per_core"] == issuers_per_core
    assert plan["partial_tile_read"] == partial_tile_read


@SCENARIOS
@pytest.mark.parametrize("bank_stagger", descriptor.KNOB_VALUES["bank_stagger"])
@pytest.mark.parametrize("issuer_split", descriptor.KNOB_VALUES["issuer_split"])
@pytest.mark.parametrize("issuer_placement", descriptor.KNOB_VALUES["issuer_placement"])
def test_issue_path_knobs_bit_exact(scenario, issuer_placement, issuer_split, bank_stagger, device, monkeypatch):
    monkeypatch.setitem(descriptor.KNOBS, "issuers_per_core", 2)
    monkeypatch.setitem(descriptor.KNOBS, "issuer_placement", issuer_placement)
    monkeypatch.setitem(descriptor.KNOBS, "issuer_split", issuer_split)
    monkeypatch.setitem(descriptor.KNOBS, "bank_stagger", bank_stagger)
    ttnn_input, output_config = _run_exact(scenario, device)
    plan = describe_plan(ttnn_input, output_config)
    assert (plan["issuer_placement"], plan["issuer_split"], plan["bank_stagger"]) == (
        issuer_placement,
        issuer_split,
        bank_stagger,
    )


def test_one_issuer_bank_stagger_bit_exact(device, monkeypatch):
    monkeypatch.setitem(descriptor.KNOBS, "issuers_per_core", 1)
    monkeypatch.setitem(descriptor.KNOBS, "bank_stagger", True)
    _run_exact(FLUX_58_640, device)


# h_rem -> (valid_rows spans, face_rows spans), as (offset, bytes) pairs; whole_tile is always [(0, 2048)].
SPANS = {
    0: ([(0, 2048)], [(0, 2048)]),
    1: ([(0, 544)], [(0, 32), (512, 32)]),
    16: ([(0, 1024)], [(0, 512), (512, 512)]),
    17: ([(0, 1568)], [(0, 1056), (1536, 32)]),
    31: ([(0, 2016)], [(0, 1504), (1536, 480)]),
}


def _spans(geom, mode):
    first, second = descriptor.partial_spans(geom, mode)
    return [first] + ([second] if second[1] else [])


@pytest.mark.parametrize("h_rem", sorted(SPANS))
def test_partial_spans(h_rem):
    geom = descriptor.Geometry(
        scheme=descriptor.WIDTH,
        num_images=1,
        image_tile_rows=1,
        tensor_tile_rows=1,
        tensor_tile_cols=1,
        h_rem=h_rem,
        tile_bytes=2048,
        face_h=16,
        face_bytes=512,
        face_row_bytes=32,
        faces_per_tile_row=2,
        shard_tile_rows=1,
        shard_tile_cols=1,
        num_shards=1,
        cores=((0, 0),),
    )
    valid_rows, face_rows = SPANS[h_rem]
    assert _spans(geom, "valid_rows") == valid_rows
    assert _spans(geom, "face_rows") == face_rows
    assert _spans(geom, "whole_tile") == [(0, 2048)]
    # Every logical byte of rows [0, h_rem) is inside a face_rows span.
    covered = set()
    for offset, size in face_rows:
        covered.update(range(offset, offset + size))
    for row in range(h_rem):
        face_row, row_in_face = divmod(row, 16)
        for face in (2 * face_row, 2 * face_row + 1):
            start = face * 512 + row_in_face * 32
            assert set(range(start, start + 32)) <= covered


@pytest.mark.parametrize(
    "rows,cols,issuers,expected",
    [
        (1, 5, 1, [(0, 0, 1, 5)]),
        (1, 5, 2, [(0, 0, 1, 3), (0, 3, 1, 2)]),
        (2, 4, 2, [(0, 0, 2, 2), (0, 2, 2, 2)]),
        (3, 1, 2, [(0, 0, 2, 1), (2, 0, 1, 1)]),
        (1, 1, 2, [(0, 0, 1, 1), (1, 0, 0, 1)]),
        (0, 0, 2, [(0, 0, 0, 0), (0, 0, 0, 0)]),
    ],
)
def test_issuer_blocks(rows, cols, issuers, expected):
    assert descriptor.issuer_blocks(rows, cols, issuers) == expected


@pytest.mark.parametrize(
    "rows,cols,weights,expected",
    [
        (1, 5, [2 / 3, 1 / 3], [(0, 0, 1, 3), (0, 3, 1, 2)]),  # p1: 8x4 owners
        (1, 20, [8 / 9, 1 / 9], [(0, 0, 1, 18), (0, 18, 1, 2)]),  # 8 owners in one row
        (1, 20, [1 / 9, 8 / 9], [(0, 0, 1, 2), (0, 2, 1, 18)]),  # 8 owners in one column
        (3, 1, [1 / 2, 1 / 2], [(0, 0, 2, 1), (2, 0, 1, 1)]),
        (1, 1, [0.9, 0.1], [(0, 0, 1, 1), (1, 0, 0, 1)]),
    ],
)
def test_issuer_blocks_weighted(rows, cols, weights, expected):
    assert descriptor.issuer_blocks(rows, cols, 2, weights) == expected


@pytest.mark.parametrize(
    "owners,nocs,expected",
    [
        ([(x, y) for y in range(4) for x in range(8)], [0, 1], [8 / 12, 4 / 12]),
        ([(x, 0) for x in range(8)], [0, 1], [8 / 9, 1 / 9]),
        ([(0, y) for y in range(8)], [1, 0], [8 / 9, 1 / 9]),
        ([(0, 0)], [0, 1], [1 / 2, 1 / 2]),
        ([(x, 0) for x in range(8)], [0], None),
    ],
)
def test_noc_weights(owners, nocs, expected):
    assert (
        descriptor.noc_weights(owners, nocs) == pytest.approx(expected)
        if expected
        else (descriptor.noc_weights(owners, nocs) is None)
    )
