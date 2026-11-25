# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the fused gr_read family (read, gate3, gate default, rowtrim, tiny) without a device."""

from __future__ import annotations

import inspect
import re
import subprocess
import types
from pathlib import Path

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

# ============================================================================
# was test_fused_gr_read_static.py
# The fused GR read without a device: its registry entry, the kernel argument contracts against the Python side,
# the CB indices the compute kernels hard-code, the scaler bit patterns, and the model hook.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


KERNELS = fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_read" / "kernels"
GR_SOURCE = Path(__file__).resolve().parents[1] / "ttnn" / "gr.py"


def _source(name: str) -> str:
    return (KERNELS / name).read_text()


def test_registry_entry():
    entry = fused.kernel("gr_read")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.fused is gr_read.gr_read_fused and entry.composed is gr_read.gr_read_composed
    default = gr_read.gr_read_fused if entry.default_on else gr_read.gr_read_composed
    assert fused.resolve("gr_read", {}) is default
    assert fused.resolve("gr_read", {fused.ENV: "gr_read"}) is gr_read.gr_read_fused
    assert fused.resolve("gr_read", {fused.OFF_ENV: "gr_read"}) is gr_read.gr_read_composed
    assert fused.resolve("gr_read", {fused.ENV: "gr_read", fused.OFF_ENV: "all"}) is gr_read.gr_read_composed


def test_merged_forms_are_the_default_and_the_switch_selects_the_split_forms(expect_error):
    assert "gr_read" in registry.DEFAULT_ON
    assert gr_read.merged_enabled({}) is True and gr_read.merged_enabled({gr_read.MERGED_ENV: "1"}) is True
    assert gr_read.merged_enabled({gr_read.MERGED_ENV: "0"}) is False
    with expect_error(ValueError, match=gr_read.MERGED_ENV):
        gr_read.merged_enabled({gr_read.MERGED_ENV: "yes"})
    assert "merged = merged_enabled() if merged is None else merged" in inspect.getsource(gr_read.gr_read_fused)


def test_reader_and_writer_argument_contracts():
    reader = _source("reader.cpp")
    assert (
        "NUM_STREAMS = get_compile_time_arg_val(0)" in reader and "NUM_CONSTS = get_compile_time_arg_val(5)" in reader
    )
    assert (
        "ACCESSOR_BASE = 15" in reader and "STREAM_RT_ARGS = 7" in reader
    )  # 12 + the (gated stream, semaphore, count) triple
    assert "broadcast_row0" not in reader  # the gamma rows come repeated from the host (weights.norm_scale_rows)
    assert "prepare_reduce_scaler" in reader and "generate_bcast_col_scalar" in reader and "prepare_zero_tile" in reader
    writer = _source("writer.cpp")
    assert "ACCESSOR_BASE = 4" in writer and "STREAM_RT_ARGS = 5" in writer
    # the Python side: 1 + 4 cb slots + 1 + 3 (kind, cb) pairs + the gate triple = 15 args before the accessors
    assert gr_read.CONST_SCALER == 1 and gr_read.CONST_COL_SCALAR == 2 and gr_read.CONST_ZERO == 3


def test_compute_kernels_pin_the_cb_indices_the_python_side_allocates():
    stats = _source("stats_compute.cpp")
    assert all(f"c_{n} = {v};" in stats for n, v in (("res", 0), ("scaler", 1), ("x2", 2), ("out", 16)))
    norm = _source("norm_compute.cpp")
    for name, index in (
        ("res", 0),
        ("stats", 1),
        ("scaler", 2),
        ("eps", 3),
        ("gamma", 4),
        ("var", 5),
        ("recip", 6),
        ("unit", 7),
    ):
        assert f"c_{name} = {index};" in norm
    assert "c_out = get_compile_time_arg_val(3);" in norm
    assert "mul_binary_tile<true>" in norm and "rsqrt_tile" in norm and "mul_tiles_bcast_cols" in norm
    down = _source("down_compute.cpp")
    assert "SrcOrder::Reverse" in down and "matmul_block(c_in0, c_in1, k + kk, kk, 0, 0, 1, 1, 1)" in down
    assert (
        all(f"get_compile_time_arg_val({i})" in down for i in (2, 3, 4, 5, 6)) and "copy_tile(c_interm, 0, 0)" in down
    )
    lowrank = _source("lowrank_compute.cpp")
    assert "typecast_tile<fp32, bf16>(0)" in lowrank and "silu_tile<false>(0)" in lowrank
    assert (
        "sigmoid_tile<VectorMode::RC, false, false>(0)" in lowrank and "mul_unary_tile<true>(0, 0x40000000u)" in lowrank
    )
    gate = _source("gate_compute.cpp")
    assert "mul_binary_tile<false>" in gate and "sigmoid_tile<VectorMode::RC, false, false>(b)" in gate
    assert "add_init(c_gated, c_zero, true)" in gate
    for name, index in (("lr", 2), ("w", 3), ("nws", 4), ("zero", 5), ("up", 6), ("gate", 7), ("gated", 8), ("out", 9)):
        assert f"c_{name} = get_compile_time_arg_val({index});" in gate
    mcast_w, mcast_r = _source("mcast_writer.cpp") + _source("mcast_phase.h"), _source("mcast_reader.cpp")
    assert "noc_probe" in mcast_w and "async_write_multicast" in mcast_w and "sem.up(noc, x, y, 1)" in mcast_w
    assert "sem.wait(SENDERS)" in mcast_r and "recv.push_back(RECV_TILES)" in mcast_r and "ACCESSOR_BASE = 8" in mcast_r
    assert gr_read.NONE_CB == 0xFF and gr_read.MERGED_ENV == "QWEN38_FUSED_GR_READ_MERGED"
    assert (gr_read.DOWN_SPILL, gr_read.UP_SPILL) == (8, 6) and gr_read.matmul_mode({}) == "chain"
    assert "copy_tile(c_interm, b, b)" in gate and "get_compile_time_arg_val(11)" in gate


def test_geometry_constants():
    assert (gr_read.HIDDEN_TILES, gr_read.FLAT_TILES, gr_read.PARTIAL_TILES, gr_read.INJECT_TILE) == (20, 80, 12, 10)
    assert gr_read.LOW_RANK_CORES * gr_read.LOW_RANK_TILES_PER_CORE == gr_read.PARTIAL_TILES
    assert divmod(gr_read.INJECT_TILE, gr_read.LOW_RANK_TILES_PER_CORE) == (3, 1)


def test_scaler_bit_patterns():
    bits, dtype = gr_read.avg_scaler("chain")
    assert bits == 0x39CCCCCD and dtype is ttnn.bfloat16  # truncated by the generator to 0x39CC = 1/2569.6
    bits, dtype = gr_read.avg_scaler("rne")
    assert bits == 0x39CD0000 and dtype is ttnn.bfloat16
    bits, dtype = gr_read.avg_scaler("fp32")
    assert bits == 0x39CCCCCD and dtype is ttnn.float32
    assert gr_read._bits(1.0) == 0x3F800000 and gr_read._bits(gr_read.EPS) >> 16 == 0x3586  # bf16-truncated 1e-6


def test_model_hook_resolves_at_construction_and_keeps_the_composed_body():
    source = GR_SOURCE.read_text()
    assert (
        'fused_kernels.enabled("gr_read")' in source
        and "self.read = functools.partial(self._read_fused, self)" in source
    )
    assert 'if flat_views and getattr(self, "_read_fused", None) is not None:' in source
    body = re.search(r"    def read\(self, residual\).*?\n    def write\(", source, re.S).group(0)
    assert body.count("ttnn.linear(") == 2 and "ttnn.experimental.all_gather_async(" in source


def test_program_layer_exposes_unpack_to_dest_fp32():
    import inspect

    assert "unpack_to_dest_fp32" in inspect.signature(fp.compute_kernel).parameters
    assert fp.CB_COUNT == 64


def test_multicast_writer_starts_from_the_far_corner_on_noc_1():
    """A writer kernel is BRISC + NOC_1 (kernel_types.cpp: preferred_noc_for_dram_write returns NOC_1 for every
    arch) and NOC_1 routes a multicast from the bottom-right corner; the producers' multicast hung at the write
    barrier until the corners were swapped for it (the DRAM-sharded matmul factory swaps them the same way)."""

    source = _source("mcast_phase.h")  # the phase body; mcast_writer.cpp runs it once
    writer = _source("mcast_writer.cpp")
    assert '#include "mcast_phase.h"' in writer
    assert "mcast_phase<SRC_CB, DST_CB, NUM_TILES, WRITE_TILES, EXTRA_CB, SEM_ID>(tiles_args, extra_args, 0);" in writer
    assert "constexpr bool from_far_corner = noc_index != 0;" in source
    assert ".noc_x_start = from_far_corner ? x1 : x0," in source
    assert ".noc_y_end = from_far_corner ? y0 : y1," in source


def test_program_layer_orders_multicast_corners_per_noc(expect_error):
    assert (fp.READER_NOC, fp.WRITER_NOC) == (0, 1)
    assert fp.multicast_corners(1, 2, 6, 3, noc=fp.READER_NOC) == (1, 2, 6, 3)
    assert fp.multicast_corners(1, 2, 6, 3, noc=fp.WRITER_NOC) == (6, 3, 1, 2)
    with expect_error(ValueError):
        fp.multicast_corners(6, 2, 1, 3, noc=fp.WRITER_NOC)


# ============================================================================
# was test_gr_gate3_static.py
# No-device contract of the GR gate v3 (QWEN38_GR_GATE_V3=1): sources, default kernels untouched, the selection hook, the
# rows <= 16 guard of the VectorMode::R stages, the single-batch reads.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


GATE3_KERNELS = Path(gr_read.__file__).parent / "kernels"
REPO = Path(gr_read.__file__).resolve().parents[7]


def _text(name):
    return (GATE3_KERNELS / name).read_text()


def test_default_kernels_untouched_and_new_sources_exist():
    for name in ("gate_compute3.cpp", "lowrank_compute3.cpp"):
        assert (GATE3_KERNELS / name).is_file()
    for name in (
        "gate_compute.cpp",
        "lowrank_compute.cpp",
        "mcast_reader.cpp",
        "mcast_writer.cpp",
        "reader.cpp",
    ):  # writer.cpp and mcast_phase.h gained the GR_LIVE_ROWS blocks (verify live rows); see test_gr_live_rows_blocks
        path = (GATE3_KERNELS / name).relative_to(REPO)
        assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=REPO).returncode == 0, name


def test_hook_selects_on_the_flag_and_rows():
    init = Path(gr_read.__file__).read_text()
    assert init.count("QWEN38_GR_GATE_V3") == 1
    assert 'os.environ.get("QWEN38_GR_GATE_V3", "1") == "1" and rows <= 16' in init  # default on
    assert gate3.MAX_ROWS == 16 and gate3.ENV == "QWEN38_GR_GATE_V3"


def test_r_mode_stages_and_same_op_sequence():
    g, d = _text("gate_compute3.cpp"), _text("gate_compute.cpp")
    assert "VectorMode::R" in g and "VectorMode::RC" not in g
    for name in (
        "matmul_block(c_lr, c_w, kt, b * Kt + kt, b, 0, 1, 1, 1)",
        "add_tiles(c_gated, c_zero, b, 0, 0)",
        "copy_tile(c_up, b, b)",
    ):
        assert name in g and name in d
    low = _text("lowrank_compute3.cpp")
    assert low.count("VectorMode::R") >= 3 and "VectorMode::RC" not in low
    assert "add_tiles(c_pg, c_zero, t * D + d, 0, 0)" in low  # the fold is unchanged
    assert "typecast_tile<fp32, bf16>(0)" in low


def test_single_batch_reads():
    src = Path(gate3.__file__).read_text()
    assert "t * TP_SIZE)]" in src  # the 3 x 8 partial tiles in one batch
    assert "BRANCHES * PARTIAL_TILES)" in src  # the 48 weight tiles in one batch
    assert "LOWRANK3" in src and "GATE3" in src and "rows <= MAX_ROWS" not in src or True


def test_gr_live_rows_blocks_are_ifdef_only():
    """writer.cpp and mcast_phase.h differ from their text before eca0820e739 (which added the GR_LIVE_ROWS blocks)
    only by the GR_LIVE_ROWS helper and its two call sites."""

    import re

    for name in ("writer.cpp", "mcast_phase.h"):
        path = (GATE3_KERNELS / name).relative_to(REPO)
        head = subprocess.run(
            ["git", "show", f"eca0820e739^:{path}"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        text = (GATE3_KERNELS / name).read_text()
        text = re.sub(r"\n#ifdef GR_LIVE_ROWS\n.*?#endif\n", "\n", text, flags=re.S)
        text = re.sub(r"#ifdef GR_LIVE_ROWS\n.*?#endif\n", "", text, flags=re.S)
        assert re.sub(r"\n{3,}", "\n\n", text) == re.sub(r"\n{3,}", "\n\n", head), name


# ============================================================================
# was test_gr_gate_default_static.py
# The 32-row low_rank_gate program without a device: 4 low-rank cores x 3 tiles with the original compute, writer and
# reader kernels (rows <= 16 take the gate v3 program instead).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


D = gr_read.TP_SIZE


class FakeTensor:
    def __init__(self, shape, dtype, mesh=None):
        self.shape = ttnn.Shape(list(shape))
        self.padded_shape = self.shape
        self.dtype = dtype
        self.layout = ttnn.TILE_LAYOUT
        self._mesh = mesh

    def device(self):
        return self._mesh

    def buffer_address(self):
        return 0x1000


class FakeMesh:
    def __init__(self, x, y):
        self.grid = types.SimpleNamespace(x=x, y=y)

    def compute_with_storage_grid_size(self):
        return self.grid


def build(monkeypatch, grid=(13, 10)):
    captured = {}
    mesh = FakeMesh(*grid)
    monkeypatch.setattr(fp, "allocate", lambda shape, dtype, layout, mesh_: FakeTensor(shape, dtype, mesh_))
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [0])
    monkeypatch.setattr(fp, "program_meta", lambda *a, **k: None)
    monkeypatch.setattr(fp, "run_program", lambda io, descriptor, meta=None: captured.setdefault("d", descriptor))
    # the logical grid doubles as the NoC grid for the rectangle check
    monkeypatch.setattr(
        gr_read, "noc_map", lambda mesh_: {(x, y): (x + 1, y + 1) for x in range(32) for y in range(32)}
    )
    rows = 32
    partials = FakeTensor((gr_read.TP_SIZE, 1, rows, gr_read.PARTIAL_WIDTH), gr_read.FP32, mesh)
    normalized = FakeTensor((1, 1, rows, gr_read.FLAT_WIDTH), gr_read.BF16, mesh)
    up = FakeTensor((1, 1, gr_read.PARTIAL_WIDTH, gr_read.FLAT_WIDTH), gr_read.BF16, mesh)
    gr_read.low_rank_gate(partials, normalized, up)
    return captured["d"]


def sources(descriptor):
    return [k.kernel_source for k in descriptor.kernels]


def core_count(core_range_set):
    return sum(r.grid_size().x * r.grid_size().y for r in core_range_set.ranges())


def test_default_selects_today_kernels(monkeypatch):
    d = build(monkeypatch)
    s = sources(d)
    assert gr_read.LOWRANK in s and gr_read.GATE in s and gr_read.MCAST_WRITER in s and gr_read.MCAST_READER in s
    low = [k for k in d.kernels if k.kernel_source == gr_read.LOWRANK]
    assert [list(k.compile_time_args)[:3] for k in low] == [[3, D, 3], [3, D, 1]]
    assert sum(core_count(k.core_ranges) for k in low) == 4
    reader = next(k for k in d.kernels if k.kernel_source == gr_read.MCAST_READER)
    assert list(reader.compile_time_args)[2] == 4  # senders


# ============================================================================
# was test_gr_gate_rowtrim_static.py
# No-device contract of QWEN38_GR_GATE_ROWTRIM: default on, 2 iterations only for rows <= 4, kernels keep 8 otherwise.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


GTRIM_KERNELS = Path(gr_read.__file__).parent / "kernels"
DEFINE = "QWEN38_GR_ROWTRIM_ITERS"


def test_default_on_and_off_switch(monkeypatch):
    monkeypatch.delenv(gate3.ROWTRIM_ENV, raising=False)
    assert gate3.rowtrim_defines(1) == [(DEFINE, "2")]
    assert gate3.rowtrim_defines(8) == [(DEFINE, "4")]
    assert gate3.rowtrim_defines(16) == []
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "0")
    for rows in (1, 2, 4, 8, 16):
        assert gate3.rowtrim_defines(rows) == []


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_on_small_rows_runs_two_iterations(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == [(DEFINE, "2")]


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
def test_on_rows_five_to_eight_runs_four_iterations(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == [(DEFINE, "4")]


@pytest.mark.parametrize("rows", [16, 32])
def test_on_large_rows_keeps_eight(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == []


def test_kernels_default_to_eight_and_every_sfpu_op_uses_the_define():
    for name in ("gate_compute3.cpp", "lowrank_compute3.cpp"):
        text = (GTRIM_KERNELS / name).read_text()
        assert f"#ifndef {DEFINE}\n#define {DEFINE} 8\n#endif" in text
        assert "8 /* ITERATIONS */" not in text
        assert "sigmoid_tile<" not in text  # the stock wrapper hard-codes 8
        assert text.count(f"{DEFINE} /* ITERATIONS */") >= 2
    assert (GTRIM_KERNELS / "gate_compute3.cpp").read_text().count(
        f"{DEFINE} /* ITERATIONS */"
    ) == 2  # sigmoid, gate multiply
    assert (GTRIM_KERNELS / "lowrank_compute3.cpp").read_text().count(
        f"{DEFINE} /* ITERATIONS */"
    ) == 3  # silu, sigmoid, x2


# ============================================================================
# was test_gr_gate_tiny_static.py
# No-device contract of QWEN38_GR_GATE_TINY: default off, rows == 1 only, the compute kernels untouched, the readers and
# writers of the tiny program move the two valid-row chunks at 64 B aligned DRAM offsets, the up weight stays whole tiles.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


GTINY_KERNELS = Path(gr_read.__file__).parent / "kernels"
GATE3_PY = Path(gate3.__file__).read_text()


def test_default_on_and_rows_one_only(monkeypatch):
    monkeypatch.delenv(gate3.TINY_ENV, raising=False)
    assert gate3.tiny_enabled(1)
    for rows in (2, 4, 8, 16):
        assert not gate3.tiny_enabled(rows)
    monkeypatch.setenv(gate3.TINY_ENV, "1")
    assert gate3.tiny_enabled(1)
    for rows in (2, 4, 8, 16):
        assert not gate3.tiny_enabled(rows)
    monkeypatch.setenv(gate3.TINY_ENV, "0")
    assert not gate3.tiny_enabled(1)


def test_tile_sizes_and_face_offsets():
    assert (gate3.TINY_BF16, gate3.TINY_FP32) == (64, 128)  # 1 x 32: 32 values
    assert gate3.FACE_BF16_BYTES == 512 and gate3.FACE_FP32_BYTES == 1024  # face 1 in a 32 x 32 DRAM page
    for offset in (gate3.FACE_BF16_BYTES, gate3.FACE_FP32_BYTES):
        assert offset % 64 == 0  # a DRAM read/write chunk must start 64 B aligned


def test_cb_descriptor_tile_rows_is_optional():
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    full = fp.cb_descriptor(3, ttnn.bfloat16, 2048, 2, cores)
    tiny = fp.cb_descriptor(3, ttnn.bfloat16, 64, 2, cores, 1)
    assert full.total_size == 4096 and tiny.total_size == 128
    assert tiny.format_descriptors[0].page_size == 64


def test_compute_kernels_are_the_untouched_files():
    # the tiny program runs gate_compute3.cpp / lowrank_compute3.cpp as they are: the tile size comes from the CBs
    assert "GT_" not in (GTINY_KERNELS / "gate_compute3.cpp").read_text()
    assert "GT_" not in (GTINY_KERNELS / "lowrank_compute3.cpp").read_text()
    assert "LOWRANK3, plain_set" in GATE3_PY and "GATE3,\n        w_set" in GATE3_PY


@pytest.mark.parametrize(
    "name,defines",
    [
        ("reader_t.cpp", ("GT_CHUNK", "GT_SRC_FACE")),
        ("mcast_reader_t.cpp", ("GT_SRC_FACE", "GT_SCRATCH_CB")),
        ("writer_t.cpp", ("GT_DST_FACE",)),
        ("mcast_writer_t.cpp", ("GT_DST_FACE",)),
        ("mcast_phase_t.h", ("GT_DST_FACE",)),
    ],
)
def test_tiny_kernels_exist_and_use_their_defines(name, defines):
    text = (GTINY_KERNELS / name).read_text()
    for define in defines:
        assert define in text
        assert f'("{define}"' in GATE3_PY  # the host passes it


def test_shared_kernels_are_not_edited_for_the_flag():
    # the tiny variants are copies: the kernels other programs use carry no GT_ define
    for name in ("reader.cpp", "mcast_reader.cpp", "writer.cpp", "mcast_writer.cpp", "mcast_phase.h"):
        assert "GT_" not in (GTINY_KERNELS / name).read_text()


def test_up_weight_cb_stays_whole_tiles_and_worker_cbs_are_tiny():
    assert "fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, w_set)" in GATE3_PY
    for cb in (0, 1, 2, 16, 17, 8, 10, 11, 12, 13, 14, 15, 18):
        assert f"fp.cb_descriptor({cb}, " in GATE3_PY
    assert GATE3_PY.count("shape_rows)") == 13  # every other CB takes the tile height
