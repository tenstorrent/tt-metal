# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused final mixer without a device: its registry entry, the low-rank kernels' argument contracts, the reduce
form (the accurate fp32 SFPU fold over the stacked device rows = the chain's `ttnn::sum` over a transposed-into-H
dim), the matmul
spills against the DRAM-sharded config rule, the reuse of F2's programs, and the hook pinned in ``final_mixer.py``."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import final_mixer as fm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read as gr
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

HERE = Path(__file__).resolve().parents[1]
MIXER_SOURCE = (HERE / "ttnn" / "final_mixer.py").read_text()
COMPUTE = (fp.REPO_ROOT / fm.LOWRANK_COMPUTE).read_text()
READER = (fp.REPO_ROOT / fm.LOWRANK_READER).read_text()
WRITER = (fp.REPO_ROOT / fm.LOWRANK_WRITER).read_text()


def _flat(source: str) -> str:
    """Source with runs of whitespace collapsed (black wraps long calls; the pins are on the arguments)."""

    return re.sub(r"\s+", " ", source)


def _in0_block_w(k: int, num_cores: int) -> int:
    """The rule of decode_matmul.dram_sharded_matmul_configs: the largest divisor <= 8 of the K tiles per core."""

    k_tiles_per_core = k // (32 * num_cores)
    return next(width for width in range(8, 0, -1) if k_tiles_per_core % width == 0)


def test_registered_bitwise_opt_in():
    entry = fused.kernel("final_mixer")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.default_on is ("final_mixer" in fused.DEFAULT_ON)  # opt-in until its timing pin; the list decides
    assert entry.fused is fm.final_mixer_fused and entry.composed is fm.final_mixer_composed
    assert fused.resolve("final_mixer", {}) is fm.final_mixer_composed
    assert fused.resolve("final_mixer", {fused.ENV: "final_mixer"}) is fm.final_mixer_fused


def test_spills_follow_the_dram_sharded_matmul_rule():
    assert fm.DOWN_SPILL == _in0_block_w(2560, 5) == gr.DOWN_SPILL == 8
    assert fm.UP_SPILL == _in0_block_w(320, 2) == 5 and gr.UP_SPILL == _in0_block_w(384, 2) == 6
    assert fm.RANK_TILES == 10 and fm.RANK == 320


def test_low_rank_kernels_are_the_chains_reduce_form():
    assert "ReduceDim::REDUCE_COL," in COMPUTE and "ReduceFp32Mode::Accurate>(" in COMPUTE  # the chain's SFPU fp32 fold
    assert "compute_kernel_lib::ReduceInputBlockShape::of(1, T, 1)," in COMPUTE
    assert '#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"' in COMPUTE
    assert "reduce_tile<" not in COMPUTE  # never the FPU reduce: it truncates fp32 to tf32 on the way into SrcA/SrcB
    assert "typecast_tile<fp32, bf16>(0);" in COMPUTE and "silu_tile<false>(0);" in COMPUTE
    assert COMPUTE.index("ReduceFp32Mode::Accurate>(") < COMPUTE.index("typecast_tile<") < COMPUTE.index("silu_tile<")
    assert "ckernel::ReduceDim::REDUCE_COL>(1.0f);" in READER  # the chain's fp32 scaler tile
    assert "noc.async_read_barrier();  // the zero copies land before the rows" in READER
    assert "const uint32_t page = d * W + first + t;" in READER  # this core's column window of the W-tile-wide row
    assert ".page_id = page, .offset_bytes = 0}" in READER and ".offset_bytes = FACE_BYTES}" in READER
    for source, n_args, tensors in ((COMPUTE, 1, 0), (READER, 3, 2), (WRITER, 1, 1)):
        used = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", source)})
        assert used == list(range(n_args)), used
        assert source.count("TensorAccessorArgs<") == tensors
    assert "TensorAccessorArgs<3>()" in READER
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", READER)}) == [0, 1, 2]
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", WRITER)}) == [0]


def test_python_side_matches_the_kernel_cbs():
    source = inspect.getsource(fm.low_rank)
    assert (
        "unpack_to_dest_fp32=(0, 2)" in source
    )  # the stacked tiles (the SFPU fold reads the dest) and the sum re-read
    assert "[RANK_TILES, TP_SIZE, RANK_TILES] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero)" in source
    assert "[(core, [gathered_partials.buffer_address(), zero.buffer_address(), 0])]" in source
    for index, dtype in ((0, "FP32"), (1, "FP32"), (2, "FP32"), (3, "BF16"), (16, "BF16")):
        assert f"fp.cb_descriptor({index}, {dtype}," in source, index
    assert "gr.DOWN" in inspect.getsource(fm.down) and "gr.GATE" in inspect.getsource(fm.gate)
    assert "[FLAT_TILES, blk, 0, 1, 16, DOWN_SPILL, 2]" in inspect.getsource(fm.down)
    assert "[RANK_TILES, BRANCHES, 0, 1, 2, 3, 4, 5, 6, 16, UP_SPILL, 7]" in inspect.getsource(fm.gate)
    fused_source = inspect.getsource(fm.final_mixer_fused)
    assert fused_source.count("ttnn.all_gather(") == 2 and "gr.stats(residual)" in fused_source
    assert "gr.normalize(residual, gathered_stats, module_norm_scale_rows(module))" in fused_source
    assert "normalize_down( residual, gathered_stats, module_norm_scale_rows(module), module.weights.down )" in _flat(
        fused_source
    )
    assert "block = low_rank_gate(gathered_partials, normalized, module.weights.up)" in fused_source
    assert fused_source.index("merged = merged_enabled() if merged is None else merged") < fused_source.index(
        "if merged:"
    )
    assert "gr.noc_map( residual.device() )" in _flat(fused_source)  # the multicast rectangles, once per mesh
    assert "except gr.NocMapMismatch" in fused_source  # split fallback
    noc = inspect.getsource(gr.noc_map)
    assert "for shard in ttnn.get_device_tensors(out):" in noc and "ttnn.to_torch(shard)" in noc  # per-device readback
    assert "raise NocMapMismatch(" in noc and "ttnn.to_torch(out)" not in noc


def test_merged_programs_are_f2s_multicast_programs_with_the_mixers_shapes():
    """normalize_down / low_rank_gate: F2's kernels (NORM, DOWN, GATE, the multicast writer/reader) and the mixer's
    own low-rank fold on the row cores; the grids hold one column tile per consumer core."""

    assert fm.merged_enabled({}) is True and fm.merged_enabled({fm.MERGED_ENV: "0"}) is False
    assert fm.merged_enabled({fm.MERGED_ENV: "1"}) is True
    assert fm.DOWN_GRID[0] * fm.DOWN_GRID[1] == fm.RANK_TILES == 10
    assert fm.GATE_GRID[0] * fm.GATE_GRID[1] == fm.HIDDEN_TILES == 20 and fm.GATE_GRID == (5, 4)  # F2's gate grid
    assert fm.LOW_RANK_CORES * fm.LOW_RANK_TILES_PER_CORE == fm.RANK_TILES and fm.LOW_RANK_CORES <= fm.GATE_GRID[0]
    assert fm.BRANCHES <= fm.DOWN_GRID[0]  # the producer row sits above the consumers' columns
    nd = _flat(inspect.getsource(fm.normalize_down))
    f2_nd = _flat(inspect.getsource(gr.normalize_down))
    for pin in (
        "fp32_dest=True, unpack_to_dest_fp32=(4, 7)",  # F2's NORM: gamma rows and x_normed unpacked to the fp32 dest
        "src_cb=16, dst_cb=8, tiles=HIDDEN_TILES, tiles_tensor=normalized,",
        "recv_cb=8, recv_tiles=FLAT_TILES, senders=BRANCHES,",
        "[FLAT_TILES, 8, 8, 9, 17, DOWN_SPILL",  # F2's DOWN on CBs 8 (row), 9 (weight), 17 (partial), the mixer's spill 8
        "semaphores=[fp.semaphore_descriptor(0, all_set)]",
    ):
        assert pin in nd and pin in f2_nd, pin
    assert "[FLAT_TILES, 8, 8, 9, 17, DOWN_SPILL, 10], fp32_dest=True)" in nd
    assert "gr.NORM" in nd and "gr.DOWN" in nd and "gr._rectangle(mesh, *DOWN_GRID, rows_above=BRANCHES)" in nd
    assert 'gr.avg_scaler("chain")' in nd and "[scaler_bits, gr._bits(gr.EPS)]" in nd
    lg = _flat(inspect.getsource(fm.low_rank_gate))
    f2_lg = _flat(inspect.getsource(gr.low_rank_gate))
    for pin in (
        "[RANK_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL, 15],",
        "fp32_dest=True, unpack_to_dest_fp32=(10, 12, 13),",
        "recv_cb=8, recv_tiles=RANK_TILES, senders=LOW_RANK_CORES, zero_cb=11,",
        "[t, TP_SIZE, RANK_TILES] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero)",
        "zero.buffer_address(), c * t])",
        "fp.compute_kernel(LOWRANK_COMPUTE, p_set, [t], fp32_dest=True, unpack_to_dest_fp32=(0, 2))",
        "gr._rectangle(mesh, *GATE_GRID, rows_above=LOW_RANK_CORES)",
    ):
        assert pin in lg, pin
    assert "[PARTIAL_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL if" in f2_lg  # the same GATE contract
    assert "unpack_to_dest_fp32=(10, 12, 13)," in f2_lg
    for index, dtype in ((0, "FP32"), (1, "FP32"), (2, "FP32"), (3, "BF16"), (16, "BF16")):
        assert f"fp.cb_descriptor({index}, {dtype}," in lg, index  # the fold's CBs as in low_rank


def test_hook_is_pinned_in_the_mixer():
    assert (
        "    _fused_forward = None  # QWEN38_FUSED=final_mixer binds ttnn/fused/final_mixer per instance"
        in MIXER_SOURCE
    )
    assert 'if fused_kernels.enabled("final_mixer"):' in MIXER_SOURCE
    assert 'self._fused_forward = functools.partial(fused_kernels.kernel("final_mixer").fused, self)' in MIXER_SOURCE
    call = MIXER_SOURCE[MIXER_SOURCE.index("    def __call__(self, residual):") :]
    branch = "        if self._fused_forward is not None:\n            return self._fused_forward(residual)\n"
    assert branch in call and call.index(branch) < call.index("normalized_ws = self._normalize(residual)")


def test_every_chain_attribute_the_fused_mixer_uses_exists():
    """Every ``module.<name>`` of the model-level glue is a member of Qwen38TTNNFinalMixer (methods, class attributes,
    ``self.<name> =`` assignments); the glue only runs on four chips, so a wrong name would surface in acceptance."""

    body = MIXER_SOURCE[re.search(r"\nclass Qwen38TTNNFinalMixer\b[(:]", MIXER_SOURCE).start() :]
    body = body[: body.find("\nclass ", 1) if body.find("\nclass ", 1) > 0 else len(body)]
    names = set(re.findall(r"\n    def ([A-Za-z_][A-Za-z_0-9]*)\(", body))
    names |= set(re.findall(r"\n    ([A-Za-z_][A-Za-z_0-9]*)\s*[:=]", body))
    names |= set(re.findall(r"self\.([A-Za-z_][A-Za-z_0-9]*)\s*=", body))
    used = set(re.findall(r"\bmodule\.([A-Za-z_][A-Za-z_0-9]*)", inspect.getsource(fm.final_mixer_fused)))
    used |= set(re.findall(r"\bmodule\.([A-Za-z_][A-Za-z_0-9]*)", inspect.getsource(fm.module_norm_scale_rows)))
    assert used and used <= names, sorted(used - names)


def test_manifest_lists_the_files():
    manifest_path = HERE / "tools" / "release" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("tools/release/manifest.json is not in this tree (the public tree ships without tools/release/)")
    manifest = json.loads(manifest_path.read_text())["public"]
    for path in (
        "tests/test_fused_final_mixer_static.py",
        "ttnn/fused/final_mixer/__init__.py",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_compute.cpp",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_reader.cpp",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_writer.cpp",
    ):
        assert path in manifest, path
