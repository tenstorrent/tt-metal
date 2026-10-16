# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused GR read without a device: its registry entry, the kernel argument contracts against the Python side,
the CB indices the compute kernels hard-code, the scaler bit patterns, and the model hook."""

import inspect
import re
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry

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

    source = _source(
        "mcast_phase.h"
    )  # the phase body; mcast_writer.cpp runs it once, gr_fold's mcast_writer2.cpp twice
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
