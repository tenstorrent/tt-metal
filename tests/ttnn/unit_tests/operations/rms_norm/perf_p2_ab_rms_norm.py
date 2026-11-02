# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 before/after probe (not collected by a directory run — no `test_` prefix).

Runs each cell twice in one session: `base` = the op's ProgramDescriptor + kernels from a checkout of the Perf 2
round-start commit (P2AB_BASE_DIR, a `git worktree` of d521ece3aed — ebe9c9d1ebc + probe-only ablation hooks),
`head` = the current op. Cells: perf_p2_rms_norm.CELLS + the guard set below (P2AB_CELLS selects).

    git worktree add /tmp/rmsp2/base_wt d521ece3aed
    P2AB_BASE_DIR=/tmp/rmsp2/base_wt scripts/run_safe_pytest.sh --profile <this file>
"""

import importlib.util
import os
import sys

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import run_rms_norm

sys.path.insert(0, os.path.dirname(__file__))
from perf_p2_rms_norm import CELLS, _cell, _perf, _res  # noqa: E402

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
GUARD = {
    # one representative per distinct kernel path x layout x placement not already in CELLS
    "int_decode_32768": _cell((1, 1, 32, 32768)),
    "int_mid_7168": _perf((1, 1, 640, 7168)),
    "int_perf_7168x2048": _perf((1, 1, 7168, 2048)),
    "int_perf_32x7168": _perf((1, 1, 32, 7168)),
    "int_rm_128x4096_gf32": _cell((1, 1, 128, 4096), layout=RM, glayout=RM, gdtype=ttnn.float32),
    "int_rm_160x11008": _res((1, 1, 160, 11008), ML.INTERLEAVED, RM),
    "wid_perf_32x5120": _perf((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),
    "wid_tile_2048x256": _cell((1, 1, 2048, 256), ml=ML.WIDTH_SHARDED),
    "wid_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    "wid_rm_3x1x736x5119": _res((3, 1, 736, 5119), ML.WIDTH_SHARDED, RM),
    "wid_rm_224x3072": _res((1, 1, 224, 3072), ML.WIDTH_SHARDED, RM),
    "hgt_tile_4096_g": _cell((1, 1, 128, 4096), ml=ML.HEIGHT_SHARDED),
    "hgt_tile_160x11008": _res((1, 1, 160, 11008), ML.HEIGHT_SHARDED),
    "hgt_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM),
    "hgt_rm_3104x4064": _res((3104, 4064), ML.HEIGHT_SHARDED, RM),
    "hgt_rm_f32_128x8192": _cell((128, 8192), dtype=ttnn.float32, layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM),
    "blk_perf_640x6144": _perf((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    "blk_perf_8192x1024": _perf((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    "blk_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
    "blk_rm_96x6144": _res((1, 1, 96, 6144), ML.BLOCK_SHARDED, RM),
    "blk_tile_13x777x1023": _res((13, 777, 1023), ML.BLOCK_SHARDED),
}
ALL = {**CELLS, **GUARD}
_sel = os.environ.get("P2AB_CELLS")
IDS = [c for c in ALL if not _sel or c in _sel.split(",")]
_BASE = os.environ.get("P2AB_BASE_DIR", "/tmp/rmsp2/base_wt")


def _base_create():
    path = os.path.join(_BASE, "ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py")
    spec = importlib.util.spec_from_file_location("rms_norm_pd_p2_round_start", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod.create_program_descriptor


VARIANTS = os.environ.get("P2AB_VARIANTS", "base,head").split(",")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cell", IDS)
def test_ab(device, cell, variant, monkeypatch):
    if variant == "base":
        monkeypatch.setattr(
            sys.modules["ttnn.operations.rms_norm.rms_norm"], "create_program_descriptor", _base_create()
        )
    run_rms_norm(device=device, **ALL[cell])
