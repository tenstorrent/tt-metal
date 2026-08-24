# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
deepseek_top32 presorted-1024 vehicle (lane GK, 2026-08-24) — the
blaze-dstop32 SKIP_BLOCKED_VEHICLE follow-up (lane FD registration, lane EX
lift).  Runs sources/deepseek_top32_test.cpp.

Contract (blaze run_top32_llk_presorted_1024_opt):
- input: per row, num_chunks x 1024 bf16 elements, every consecutive
  32-element run sorted DESCENDING (the presorted-run precondition), plus a
  parallel uint16 index payload per element;
- each 1024 chunk is one 32x32 tile (row r of the tile = run r), ingested
  TRANSPOSED (the op's transpose_tile) so runs become columns;
- output: the row's top-32 [value | index] pairs, sorted descending, in even
  column 0 of faces F0/F1 of the packed value/index tiles (the op packs
  exactly those 32 cells; this vehicle packs the full tiles and the checker
  reads those cells).

The golden is an independent torch oracle (torch.topk on the flat row) — the
CX discipline: distinct exactly-representable bf16 stimuli make the top-32
index set unambiguous, and every returned index must point at the value
returned beside it.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, DS_TOP32

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024
FACE_DIM = 16
ELEMENTS_PER_FACE = FACE_DIM * FACE_DIM
RUN_LEN = 32
TOPK = 32
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

IMPL_LABELS = {0: "hand_original", 1: "semantic_lift"}


def _bitcast_float32(words: torch.Tensor) -> torch.Tensor:
    return words.to(torch.int32).view(torch.float32)


def _distinct_bf16_from_hi16(hi16: torch.Tensor) -> torch.Tensor:
    return _bitcast_float32(hi16.to(torch.int64) << 16)


def _make_row(search_len: int, seed: int, mode: str) -> torch.Tensor:
    """One row of `search_len` distinct, exactly-representable bf16 values as
    float32, with every consecutive 32-element run sorted descending (the
    presorted-run precondition)."""
    gen = torch.Generator().manual_seed(seed)
    if mode == "positive":
        hi16 = 0x3F80 + torch.randperm(search_len, generator=gen)
        vals = _distinct_bf16_from_hi16(hi16)
    elif mode == "signed":
        n_neg = search_len // 2
        pos = 0x3F80 + torch.arange(search_len - n_neg)  # >= +1.0
        neg = 0xBF80 + torch.arange(n_neg)  # <= -1.0
        hi16 = torch.cat([pos, neg])[torch.randperm(search_len, generator=gen)]
        vals = _distinct_bf16_from_hi16(hi16)
    else:
        raise ValueError(f"unknown mode {mode}")
    # Sort each 32-run descending IN PLACE (indices follow positions, so the
    # index payload is built after this).
    runs = vals.view(-1, RUN_LEN)
    runs, _ = torch.sort(runs, dim=1, descending=True)
    return runs.reshape(-1)


def _tilize_32x32(matrix: torch.Tensor) -> torch.Tensor:
    """Logical 32x32 row-major -> face-major flat (F0, F1, F2, F3)."""
    assert matrix.shape == (32, 32)
    f0 = matrix[0:16, 0:16].reshape(-1)
    f1 = matrix[0:16, 16:32].reshape(-1)
    f2 = matrix[16:32, 0:16].reshape(-1)
    f3 = matrix[16:32, 16:32].reshape(-1)
    return torch.cat([f0, f1, f2, f3])


def _build_input(num_chunks: int, num_rows: int, mode: str):
    """Build src_A ([values c0..cN-1][indices c0..cN-1] per row, tilized) and
    the per-row value tensor for the golden."""
    search_len = num_chunks * ELEMENTS_PER_TILE
    rows = torch.empty((num_rows, search_len), dtype=torch.float32)
    tiles = []
    for r in range(num_rows):
        rows[r] = _make_row(search_len, 100 + r, mode)
        value_tiles = []
        index_tiles = []
        for c in range(num_chunks):
            chunk = rows[r, c * ELEMENTS_PER_TILE : (c + 1) * ELEMENTS_PER_TILE]
            value_tiles.append(_tilize_32x32(chunk.view(32, 32)))
            idx = torch.arange(
                c * ELEMENTS_PER_TILE, (c + 1) * ELEMENTS_PER_TILE, dtype=torch.int32
            ).to(torch.uint16)
            # uint16 payload bit-cast into the bf16-typed stimulus buffer.
            idx_as_bf16 = idx.view(torch.bfloat16).to(torch.float32)
            index_tiles.append(_tilize_32x32(idx_as_bf16.view(32, 32)))
        tiles.extend(value_tiles)
        tiles.extend(index_tiles)
    src_A = torch.cat(tiles)
    return src_A, rows


def _winner_cells(tile_flat: torch.Tensor) -> torch.Tensor:
    """The 32 contractual output cells: even column 0 of F0 then F1, top-down
    (tile_flat is the packed face-major tile)."""
    f0 = tile_flat[0:ELEMENTS_PER_FACE].view(16, 16)[:, 0]
    f1 = tile_flat[ELEMENTS_PER_FACE : 2 * ELEMENTS_PER_FACE].view(16, 16)[:, 0]
    return torch.cat([f0, f1])


def _check(result, rows):
    num_rows = rows.shape[0]
    res = torch.tensor(result, dtype=format_dict[FORMATS.output_format])
    per_row = 2 * ELEMENTS_PER_TILE
    assert res.numel() == num_rows * per_row, f"result size {res.numel()}"

    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        val_cells = _winner_cells(block[:ELEMENTS_PER_TILE]).to(torch.float32)
        idx_cells = (
            _winner_cells(block[ELEMENTS_PER_TILE:])
            .to(torch.bfloat16)
            .view(torch.uint16)
            .to(torch.int64)
        )

        gold = torch.topk(rows[r], TOPK)
        gold_vals = gold.values  # sorted descending
        gold_idx = set(int(i) for i in gold.indices.tolist())

        # Sorted-descending top-32 values, exact (distinct representable bf16).
        assert torch.equal(val_cells, gold_vals), (
            f"row {r}: top-32 value cells != torch.topk (sorted desc)\n"
            f"  got  {val_cells[:8].tolist()} ...\n"
            f"  want {gold_vals[:8].tolist()} ..."
        )

        got_idx = [int(i) for i in idx_cells.tolist()]
        assert len(set(got_idx)) == TOPK, f"row {r}: returned indices not distinct"
        assert set(got_idx) == gold_idx, (
            f"row {r}: top-32 index set mismatch\n"
            f"  missing {sorted(gold_idx - set(got_idx))[:8]}\n"
            f"  extra   {sorted(set(got_idx) - gold_idx)[:8]}"
        )
        # Pairing: each index points at the value beside it (exact).
        for i, v in zip(got_idx, val_cells.tolist()):
            assert float(rows[r][i]) == float(
                v
            ), f"row {r}: index {i} pairs value {v} != input {float(rows[r][i])}"


def _variant(num_chunks, num_rows, impl, mode="positive"):
    src_A, rows = _build_input(num_chunks, num_rows, mode)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[FORMATS.input_format])
    config = TestConfig(
        test_name="sources/deepseek_top32_test.cpp",
        formats=FORMATS,
        templates=[
            DEST_SYNC(DestSync.Full),
            DS_TOP32(
                ds32_num_chunks=num_chunks, ds32_num_rows=num_rows, ds32_impl=impl
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=num_rows * 2 * num_chunks,
            tile_count_B=1,
            tile_count_res=num_rows * 2,
        ),
        dest_acc=DestAccumulation.No,  # 16-bit Dst: the lane-EX lift's contract
        unpack_to_dest=False,
    )
    return config, rows


@parametrize(
    num_chunks=[1, 2, 4],
    impl=list(IMPL_LABELS),
    mode=lambda num_chunks: ["positive", "signed"] if num_chunks == 2 else ["positive"],
)
def test_deepseek_top32(num_chunks, impl, mode):
    config, rows = _variant(num_chunks, num_rows=2, impl=impl, mode=mode)
    _check(config.run().result, rows)


# ---------------------------------------------------------------------------
# 32-bit arms (lane GK, X6 transpose-ingest payoff): impl 2 = hand in-tree
# math transpose LLK ingest of the Int32 index tiles; impl 3 = the same
# ingest on the typed sfpi_crosslane.h X6 surface.  Values bf16 -> fp32
# dest-acc Dst; phases = vendored original fp32 arm (is_fp32_dest_acc_en =
# true — the blaze sampling pipeline's production flag).
# ---------------------------------------------------------------------------
FORMATS32 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Int32)
IMPL32_LABELS = {2: "hand_transpose_ingest", 3: "x6_transpose_ingest"}


def _build_input_32(num_chunks: int, num_rows: int, mode: str):
    """Values (bf16, buffer_A) and Int32 index tiles (buffer_B), tilized."""
    search_len = num_chunks * ELEMENTS_PER_TILE
    rows = torch.empty((num_rows, search_len), dtype=torch.float32)
    value_tiles = []
    index_tiles = []
    for r in range(num_rows):
        rows[r] = _make_row(search_len, 100 + r, mode)
        for c in range(num_chunks):
            chunk = rows[r, c * ELEMENTS_PER_TILE : (c + 1) * ELEMENTS_PER_TILE]
            value_tiles.append(_tilize_32x32(chunk.view(32, 32)))
            idx = torch.arange(
                c * ELEMENTS_PER_TILE, (c + 1) * ELEMENTS_PER_TILE, dtype=torch.int32
            )
            index_tiles.append(_tilize_32x32(idx.view(32, 32)))
    return torch.cat(value_tiles), torch.cat(index_tiles), rows


def _check_32(result, rows):
    num_rows = rows.shape[0]
    res = torch.tensor(result, dtype=format_dict[FORMATS32.output_format])
    per_row = 2 * ELEMENTS_PER_TILE
    assert res.numel() == num_rows * per_row, f"result size {res.numel()}"

    for r in range(num_rows):
        block = res[r * per_row : (r + 1) * per_row]
        # Value tile packed Float32: bitcast the int32 read back to float.
        val_cells = (
            _winner_cells(block[:ELEMENTS_PER_TILE]).to(torch.int32).view(torch.float32)
        )
        idx_cells = _winner_cells(block[ELEMENTS_PER_TILE:]).to(torch.int64)

        gold = torch.topk(rows[r], TOPK)
        assert torch.equal(val_cells, gold.values), (
            f"row {r}: top-32 value cells != torch.topk (sorted desc)\n"
            f"  got  {val_cells[:8].tolist()} ...\n"
            f"  want {gold.values[:8].tolist()} ..."
        )
        got_idx = [int(i) for i in idx_cells.tolist()]
        gold_idx = set(int(i) for i in gold.indices.tolist())
        assert len(set(got_idx)) == TOPK, f"row {r}: returned indices not distinct"
        assert set(got_idx) == gold_idx, (
            f"row {r}: top-32 index set mismatch\n"
            f"  missing {sorted(gold_idx - set(got_idx))[:8]}\n"
            f"  extra   {sorted(set(got_idx) - gold_idx)[:8]}"
        )
        for i, v in zip(got_idx, val_cells.tolist()):
            assert float(rows[r][i]) == float(
                v
            ), f"row {r}: index {i} pairs value {v} != input {float(rows[r][i])}"


def _variant_32(num_chunks, num_rows, impl, mode="positive"):
    src_A, src_B, rows = _build_input_32(num_chunks, num_rows, mode)
    config = TestConfig(
        test_name="sources/deepseek_top32_test.cpp",
        formats=FORMATS32,
        templates=[
            DEST_SYNC(DestSync.Full),
            DS_TOP32(
                ds32_num_chunks=num_chunks, ds32_num_rows=num_rows, ds32_impl=impl
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS32.input_format,
            src_B,
            DataFormat.Int32,
            FORMATS32.output_format,
            tile_count_A=num_rows * num_chunks,
            tile_count_B=num_rows * num_chunks,
            tile_count_res=num_rows * 2,
        ),
        dest_acc=DestAccumulation.Yes,  # 32-bit Dst: fp32 values + Int32 indices
        unpack_to_dest=False,  # per-stage template args inside the kernel
    )
    return config, rows


@parametrize(
    num_chunks=[1, 2],
    impl=list(IMPL32_LABELS),
    mode=lambda num_chunks: ["positive", "signed"] if num_chunks == 2 else ["positive"],
)
def test_deepseek_top32_x6(num_chunks, impl, mode):
    config, rows = _variant_32(num_chunks, num_rows=2, impl=impl, mode=mode)
    _check_32(config.run().result, rows)


@pytest.mark.parametrize("label", ["hand_transpose_ingest", "x6_transpose_ingest"])
def test_deepseek_top32_x6_device_profile(perf_report, label: str):
    num_chunks, num_rows = 2, 2
    impl = {v: k for k, v in IMPL32_LABELS.items()}[label]
    src_A, src_B, _rows = _build_input_32(num_chunks, num_rows, "positive")

    configuration = PerfConfig(
        "sources/deepseek_top32_test.cpp",
        FORMATS32,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            DEST_SYNC(DestSync.Full),
            DS_TOP32(
                ds32_num_chunks=num_chunks, ds32_num_rows=num_rows, ds32_impl=impl
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS32.input_format,
            src_B,
            DataFormat.Int32,
            FORMATS32.output_format,
            tile_count_A=num_rows * num_chunks,
            tile_count_B=num_rows * num_chunks,
            tile_count_res=num_rows * 2,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=False,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    kernel_rows = rows[rows["marker"] == "KERNEL"]
    assert len(kernel_rows) >= 1, rows.to_string(index=False)
    cycles = float(kernel_rows.iloc[-1]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"DS_TOP32_X6_DEVICE_PROFILE impl={label} kernel_cycles={cycles:.2f}")


# Device profile (KERNEL zone, drain-inclusive: helpers/src/trisc.cpp wraps
# run_kernel + tensix_sync — the owner-ratified e2e verdict metric).  The
# profiled shape is the 2-chunk 2-row point (prep + combine + final + the
# per-row reset), one node per arm.
@pytest.mark.parametrize("label", ["hand_original", "semantic_lift"])
def test_deepseek_top32_device_profile(perf_report, label: str):
    num_chunks, num_rows = 2, 2
    impl = {v: k for k, v in IMPL_LABELS.items()}[label]
    src_A, _rows = _build_input(num_chunks, num_rows, "positive")
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[FORMATS.input_format])

    configuration = PerfConfig(
        "sources/deepseek_top32_test.cpp",
        FORMATS,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            DEST_SYNC(DestSync.Full),
            DS_TOP32(
                ds32_num_chunks=num_chunks, ds32_num_rows=num_rows, ds32_impl=impl
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=num_rows * 2 * num_chunks,
            tile_count_B=1,
            tile_count_res=num_rows * 2,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    kernel_rows = rows[rows["marker"] == "KERNEL"]
    assert len(kernel_rows) >= 1, rows.to_string(index=False)
    cycles = float(kernel_rows.iloc[-1]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"DS_TOP32_DEVICE_PROFILE impl={label} kernel_cycles={cycles:.2f}")
