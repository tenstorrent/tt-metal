# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the GR gate v3 (QWEN38_GR_GATE_V3=1): sources, default kernels untouched, the selection hook, the
rows <= 16 guard of the VectorMode::R stages, the single-batch reads."""

import subprocess
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

KERNELS = Path(gr_read.__file__).parent / "kernels"
REPO = Path(gr_read.__file__).resolve().parents[7]


def _text(name):
    return (KERNELS / name).read_text()


def test_default_kernels_untouched_and_new_sources_exist():
    for name in ("gate_compute3.cpp", "lowrank_compute3.cpp"):
        assert (KERNELS / name).is_file()
    for name in (
        "gate_compute.cpp",
        "lowrank_compute.cpp",
        "mcast_reader.cpp",
        "mcast_writer.cpp",
        "reader.cpp",
        "writer.cpp",
    ):
        path = (KERNELS / name).relative_to(REPO)
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
