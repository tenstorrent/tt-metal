# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Host-only tests for nd_core_map (no device needed):
#   pytest tests/didt/test_nd_core_map.py

import pytest
import torch

from tt_py_test_utils_common import nd_core_map

GRID = (12, 10)
BLOCK = (4, 3)
# Partial last row/column of blocks, like minimal_matmul's padded tile split
H, W = GRID[1] * BLOCK[0] - 2, GRID[0] * BLOCK[1] - 1


def corrupt(cores, block=BLOCK, shape=(H, W), transpose=False):
    ref = torch.zeros(shape)
    out = ref.clone()
    for x, y in cores:
        r, c = (x, y) if transpose else (y, x)
        out[..., r * block[0], c * block[1]] = 1.0
    return ref, out


def failing(ref, out, **kw):
    return nd_core_map.core_mismatches(ref, out, BLOCK, **kw)


def test_single_core():
    cores = failing(*corrupt([(5, 0)]))
    assert cores == {(5, 0): 1}
    assert nd_core_map.classify(cores, GRID) == "single core (5,0)"


def test_partial_last_block_is_counted():
    ref, out = corrupt([])
    out[H - 1, W - 1] = 1.0
    assert failing(ref, out) == {(11, 9): 1}


def test_counts_every_element_in_block():
    ref = torch.zeros(H, W)
    out = ref.clone()
    out[0:4, 15:18] = 1.0  # all of core (5,0)
    assert failing(ref, out) == {(5, 0): 12}


def test_row_run_reaching_edge():
    cores = failing(*corrupt([(x, 3) for x in range(7, 12)]))
    assert nd_core_map.classify(cores, GRID) == "row y=3: x=7..11, reaches edge x=11 (5 cores)"


def test_column_run_reaching_edge():
    cores = failing(*corrupt([(6, y) for y in range(0, 4)]))
    assert nd_core_map.classify(cores, GRID) == "column x=6: y=0..3, reaches edge y=0 (4 cores)"


def test_row_not_contiguous():
    cores = failing(*corrupt([(1, 2), (2, 2), (8, 2)]))
    assert nd_core_map.classify(cores, GRID) == "row y=2: x=1..2,8 (not contiguous) (3 cores)"


def test_rectangle_whole_grid_scattered_none():
    rect = [(x, y) for x in range(2, 4) for y in range(5, 7)]
    assert nd_core_map.classify(rect, GRID) == "rectangle x=2..3 y=5..6 (4 cores)"
    every = [(x, y) for x in range(GRID[0]) for y in range(GRID[1])]
    assert nd_core_map.classify(every, GRID) == "whole grid (120 cores)"
    assert nd_core_map.classify([(1, 1), (4, 7)], GRID).startswith("scattered (2 cores in 2 rows, 2 columns)")
    assert nd_core_map.classify([], GRID) == "none"


def test_classify_without_grid_has_no_edges():
    assert nd_core_map.classify([(x, 3) for x in range(7, 12)]) == "row y=3: x=7..11 (5 cores)"


def test_transpose():
    ref, out = corrupt([(5, 0)], transpose=True)
    assert failing(ref, out, transpose=True) == {(5, 0): 1}


def test_linear_x():
    # 1D: 6 blocks along N, assigned row-wise over a 4-wide grid -> block 5 is core (1, 1)
    ref = torch.zeros(2, 18)
    out = ref.clone()
    out[0, 5 * 3] = 1.0
    assert nd_core_map.core_mismatches(ref, out, (2, 3), linear_x=4) == {(1, 1): 1}


def test_batch_dims_are_summed():
    ref, out = corrupt([(5, 0)], shape=(1, 1, H, W))
    assert failing(ref, out) == {(5, 0): 1}


SB = (64, 96)  # structure tests: one core block of 2x3 tiles


def structure(mutate, dtype=torch.bfloat16):
    torch.manual_seed(0)
    ref = torch.randn(2 * SB[0], 2 * SB[1]).to(dtype)
    out = ref.clone()
    mutate(out[SB[0] :, SB[1] :])  # core (1,1)'s block
    assert nd_core_map.core_blocks(ref, out, SB) == {(1, 1): (1, 1)}
    return nd_core_map.block_structure(ref, out, 1, 1, SB)


def test_structure_row_strip():
    def bad_row(b):
        b[5, :] += 1.0

    s = structure(bad_row)
    assert s["label"] == "row strips"


def test_structure_column_strip():
    def bad_col(b):
        b[:, 40] += 1.0

    s = structure(bad_col)
    assert s["label"] == "column strips"


def test_structure_whole_tile():
    def bad_tile(b):
        b[32:64, 0:32] += 1.0

    s = structure(bad_tile)
    assert s["label"] == "whole tiles"


def test_structure_face_column_lane():
    # Face column 3 wrong in every face of every tile: columns 3, 19, 35, ... -> 1/16 of all columns
    def lane(b):
        b[:, 3::16] += 1.0

    s = structure(lane)
    assert s["label"] == "face column lane 3"
    assert s["col_lanes"][0] == (3, 1.0)


def test_structure_face_row_lane():
    def lane(b):
        b[7::16, :] += 1.0

    assert structure(lane)["label"] == "face row lane 7"


def test_structure_single_strip_is_not_a_lane():
    # One bad column sits on one face lane but only in one tile column: stays a column strip
    def bad_col(b):
        b[:, 40] += 1.0

    s = structure(bad_col)
    assert s["label"] == "column strips" and s["col_lanes"][0] == (8, 1.0)


def test_structure_scattered_bits():
    # Flip bit 13 of a few elements that all sit in face 1 (top-right 16x16) of different tiles
    def flip(b):
        v = b.view(torch.int16)
        for r, c in [(1, 20), (3, 50), (35, 90), (40, 18)]:
            v[r, c] ^= 1 << 13

    s = structure(flip)
    assert s["label"] == "scattered"
    assert s["elements"] == 4 and s["bits"] == [(13, 1.0)]


def test_structure_float32_bits():
    def flip(b):
        b.view(torch.int32)[2, 2] ^= 1 << 30

    s = structure(flip, dtype=torch.float32)
    assert s["bits"] == [(30, 1.0)]


def test_structure_partial_edge_block():
    ref = torch.zeros(100, 70)
    out = ref.clone()
    out[96:100, 64:70] = 1.0  # bottom-right partial block of a 32x32 block grid
    s = nd_core_map.block_structure(ref, out, 3, 2, (32, 32))
    assert s["elements"] == 24


def descriptor_owner(row_tile, col_tile, M_tiles, N_tiles, grid_x, grid_y):
    """Owner core per minimal_matmul_program_descriptor.cpp (M <= N, not transposed):
    M_start_tile = M_tiles_per_core * core.y, N_start_tile = N_tiles_per_core * core.x,
    tiles_per_core = round_up(tiles, cores) / cores."""
    m_tpc = -(-M_tiles // grid_y) * grid_y // grid_y
    n_tpc = -(-N_tiles // grid_x) * grid_x // grid_x
    return col_tile // n_tpc, row_tile // m_tpc


@pytest.mark.parametrize("grid", [(12, 10), (11, 10)], ids=["12x10", "11x10"])
def test_minimal_matmul_block_matches_descriptor(grid):
    # Same block formula as test_minimal_matmul.py, checked against the descriptor's tile ranges
    # at a few tiles, including the padded last row/column of cores
    M = N = 16384
    grid_x, grid_y = grid
    block = (32 * -(-(M // 32) // grid_y), 32 * -(-(N // 32) // grid_x))
    for row_tile, col_tile in [(0, 0), (51, 42), (52, 43), (300, 200), (511, 511), (468, 0), (0, 470)]:
        expected = descriptor_owner(row_tile, col_tile, M // 32, N // 32, grid_x, grid_y)
        got = nd_core_map.block_to_core((row_tile * 32) // block[0], (col_tile * 32) // block[1], blocks_x=grid_x)
        assert got == expected, (row_tile, col_tile)


def test_summarize_lane_across_checks():
    def lane(b):
        b[:, 3::16] += 1.0

    s = nd_core_map.summarize_structures([structure(lane), structure(lane), {"label": "none", "elements": 0}])
    assert s["checks"] == 2 and s["labels"] == {"face column lane 3": 2} and s["top_col_lane"] == (3, 2)
    line = nd_core_map.describe_summary(s)
    assert line.startswith("face column lane 3 in 2/2 checks;") and "most common face column" not in line


def test_summarize_mixed_labels_reports_top_lane():
    def lane(b):
        b[:, 7::16] += 1.0

    def scatter(b):
        b[0:64:9, 7] += 1.0  # mostly lane 7 but in one face group only, plus stray elements: scattered
        b[5, 0] += 1.0
        b[40, 50] += 1.0
        b[20, 33] += 1.0

    s = nd_core_map.summarize_structures([structure(lane), structure(scatter)])
    line = nd_core_map.describe_summary(s)
    assert s["top_col_lane"] == (7, 2)
    assert line.startswith("face column lane 7 in 1/2 checks (also ")
    assert nd_core_map.summarize_structures([]) is None and nd_core_map.describe_summary(None) == "no structure"


# --- per-tile owner tables (expected values: worked examples from the op program factories, and the shard specs
# recorded by graph capture on a p100, worker grid 11x10) ---------------------------------------------------------


def core_of(owner, cores, row, col):
    i = int(owner[row // 32, col // 32])
    return cores[i] if i >= 0 else None


def test_contiguous_owner_index_uneven():
    # 10 units over 4 cores: 3, 3, 2, 2 (the first 10 % 4 cores take one more)
    idx = nd_core_map.contiguous_owner_index(torch.arange(10), 10, 4)
    assert idx.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]


def test_contiguous_owner_index_fewer_units_than_cores():
    idx = nd_core_map.contiguous_owner_index(torch.arange(3), 3, 8)
    assert idx.tolist() == [0, 1, 2]


def test_owner_sdpa_13x10():
    # (1, 10, 9472, 128), q_chunk 256: 370 chunks over 130 cores; element (b=0, h=3, s=5000) -> g=130 -> core 43
    owner, cores = nd_core_map.owner_sdpa((1, 10, 9472, 128), 256, 13, 10)
    assert core_of(owner, cores, 3 * 9472 + 5000, 77) == (4, 3)


def test_owner_sdpa_11x10_extra_boundary():
    # 370 chunks over 110 cores: cores 0..39 take 4, the rest 3; g=159 is core 39's last, g=160 core 40's first
    owner, cores = nd_core_map.owner_sdpa((1, 10, 9472, 128), 256, 11, 10)
    g159, g160 = (4 * 37 + 11, 4 * 37 + 12)  # head 4, chunks 11 and 12
    assert core_of(owner, cores, 4 * 9472 + 11 * 256, 0) == (39 % 11, 39 // 11)
    assert core_of(owner, cores, 4 * 9472 + 12 * 256, 0) == (40 % 11, 40 // 11)
    assert (g159, g160) == (159, 160)


def test_owner_mla_sdpa_13x10():
    # (1, 32, 4096, 128), q_chunk 128: element (0, h=5, s=1000) -> g = 5*32 + 7 = 167 -> core 20 = (7, 1)
    owner, cores = nd_core_map.owner_sdpa((1, 32, 4096, 128), 128, 13, 10)
    assert core_of(owner, cores, 5 * 4096 + 1000, 64) == (7, 1)


def test_owner_contiguous_tiles_binary_mul_13x10():
    # [1, 1, 640*10, 576*13]: 234 tile columns, 360 tiles per core; tile 359 -> core 0, tile 360 -> core 1
    owner, cores = nd_core_map.owner_contiguous_tiles((1, 1, 6400, 7488), 13, 10)
    assert core_of(owner, cores, 1 * 32, 125 * 32) == (0, 0)  # tile 234 + 125 = 359
    assert core_of(owner, cores, 1 * 32, 126 * 32) == (1, 0)  # tile 360
    assert core_of(owner, cores, 6399, 7487) == (12, 9)  # last tile, last core


def test_owner_sharded_height_resnet_conv():
    # resnet conv on 11x10: one image (2560 rows) per core, row-major
    owner, cores = nd_core_map.owner_sharded((1, 1, 2560 * 110, 64), "HEIGHT", (2560, 64), [((0, 0), (10, 9))], True)
    assert core_of(owner, cores, 2560 * 13 + 5, 63) == (2, 1)


def test_owner_sharded_block_row_major():
    # recorded on bh-43: [1, 1, 1024, 1280] BLOCK [128, 128] on (0,0)-(9,7) ROW_MAJOR
    owner, cores = nd_core_map.owner_sharded((1, 1, 1024, 1280), "BLOCK", (128, 128), [((0, 0), (9, 7))], True)
    assert core_of(owner, cores, 300, 700) == (5, 2)


def test_owner_sharded_block_col_major():
    owner, cores = nd_core_map.owner_sharded((1, 1, 1024, 1280), "BLOCK", (128, 128), [((0, 0), (7, 9))], False)
    assert core_of(owner, cores, 300, 700) == (2, 5)


def test_owner_conv_slices_vae_upsample_256():
    # recorded on bh-43: 256x256 output, 2 width slices of 128, each BLOCK [3296, 64] on (0,0)-(7,9) ROW_MAJOR
    spec = ("BLOCK", (3296, 64), [((0, 0), (7, 9))], True)
    slices = [(0, 1, 0, 256, 0, 128) + spec, (0, 1, 0, 256, 128, 256) + spec]
    owner, cores = nd_core_map.owner_conv_slices((1, 1, 256 * 256, 512), 256, 256, slices)
    # h=10, w=130 (slice 2, local row 10*128 + 2 = 1282 -> shard row 0), c=100 -> core (1, 0)
    assert core_of(owner, cores, 10 * 256 + 128, 100) == (1, 0)
    # h=200, w=5 (slice 1, local row 200*128 + 5 = 25605 -> shard row 7), c=500 -> core (7, 7)
    assert core_of(owner, cores, 200 * 256 + 0, 500) == (7, 7)


def test_owner_core_mismatches_counts_and_unowned():
    owner = torch.tensor([[0, 1], [1, -1]])
    ref = torch.zeros(64, 64)
    out = ref.clone()
    out[0, 0] = 1  # tile (0, 0) -> core 0
    out[40, 3] = 1  # tile (1, 0) -> core 1
    out[40, 40] = 1  # tile (1, 1) -> nobody
    cores, unowned = nd_core_map.owner_core_mismatches(ref, out, owner, [(3, 4), (5, 6)])
    assert cores == {(3, 4): 1, (5, 6): 1} and unowned == 1


def test_conv_slices_from_graph_parses_capture():
    # argument text exactly as graph capture recorded it on bh-43 (trimmed); note the range list is not valid JSON
    mc = (
        "MemoryConfig(memory_layout=TensorMemoryLayout::BLOCK_SHARDED,buffer_type=BufferType::L1,"
        'shard_spec=ShardSpec{grid=[{"start":{"x":0,"y":0},"end":{"x":7,"y":9}], shape=[3296, 64], '
        "orientation=ShardOrientation::ROW_MAJOR},nd_shard_spec=std::nullopt)"
    )
    graph = [
        {"node_type": "function_start", "counter": 1, "params": {"name": "Conv2dDeviceOperation"}, "arguments": []},
        {
            "node_type": "function_start",
            "counter": 2,
            "params": {"name": "SliceWriteDeviceOperation"},
            "arguments": [
                "SliceWriteParams(Shape([0, 0, 128, 0]), Shape([1, 256, 256, 512]), Shape([1, 1, 1, 1]))",
                "{Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 32768, 512]),"
                f"tensor_layout=TensorLayout(memory_config={mc})))}}",
            ],
        },
    ]
    assert nd_core_map.conv_slices_from_graph(graph) == [
        (0, 1, 0, 256, 128, 256, "BLOCK", (3296, 64), [((0, 0), (7, 9))], True)
    ]


def test_owner_conv_slices_batch_two():
    # two 32x32 images in one piece, HEIGHT sharded 1024 rows per core: image 1 lands on the second core
    spec = ("HEIGHT", (1024, 64), [((0, 0), (1, 0))], True)
    owner, cores = nd_core_map.owner_conv_slices((1, 1, 2 * 32 * 32, 64), 32, 32, [(0, 2, 0, 32, 0, 32) + spec])
    assert core_of(owner, cores, 1024 + 5 * 32, 0) == (1, 0)


def test_owner_conv_slices_refuses_unaligned_slice(expect_error):
    spec = ("BLOCK", (960, 64), [((0, 0), (7, 9))], True)
    with expect_error(ValueError, "not tile aligned"):
        nd_core_map.owner_conv_slices((1, 1, 480 * 480, 512), 480, 480, [(0, 1, 0, 480, 0, 120) + spec])


def test_matmul_fuse_batch_folds_rows():
    # [1, 2, 64, 32] folded to [128, 32] with 64-row blocks: batch 1 belongs to block row 1, not row 0
    ref = torch.zeros(1, 2, 64, 32)
    out = ref.clone()
    out[0, 1, 3, 0] = 1
    folded = nd_core_map.core_mismatches(ref.reshape(-1, 32), out.reshape(-1, 32), (64, 32))
    assert folded == {(0, 1): 1}
