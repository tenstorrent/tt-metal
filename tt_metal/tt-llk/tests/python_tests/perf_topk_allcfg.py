# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch, not for merge. MATH_ISOLATE cost of one TopK LLK call (sources/topk_allcfg_perf.cpp) at every argument
set of the compute API topk.h: local sort K 4..64 (end phase log K - 1), tile0_sorted (head only), merge and rebuild at
every K and m_iter form, both directions, every sort mode in 16-bit and 32-bit DEST, and the per-tile helpers.
TILE_LOOP is cycles per call. TK_PICK=<file of ids> narrows the list (one id per line)."""

import os
from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TemplateParameter

pytestmark = [skip_for_quasar, skip_for_wormhole]

OPS = ["sort", "merge", "rebuild", "fuse", "defuse", "stamp_local", "stamp_range", "negzero", "strip_tags",
       "finalize_u16", "u16_prepare", "u16_move_half"]
MODES = [("unstable", False), ("unstable", True), ("stable", False), ("stable", True), ("fused", True),
         ("rank_stamped", True)]


@dataclass
class TOPK_ALLCFG(TemplateParameter):
    tk_op: str = "sort"
    tk_k: int = 32
    tk_idir: int = 0
    tk_miter: int = 0
    tk_mode: str = "unstable"
    tk_skip: int = 1
    tk_tile0: bool = False

    def convert_to_cpp(self) -> str:
        logk = self.tk_k.bit_length() - 1
        return "\n".join([
            f"constexpr int TOPK_OP = {OPS.index(self.tk_op)};",
            f"constexpr int TOPK_K = {self.tk_k};",
            f"constexpr int TOPK_LOGK = {logk};",
            f"constexpr int TOPK_END_PHASE = {logk - 1};",
            f"constexpr int TOPK_IDIR = {self.tk_idir};",
            f"constexpr int TOPK_MITER = {self.tk_miter};",
            f"constexpr bool TOPK_STABLE_SORT = {str(self.tk_mode == 'stable').lower()};",
            f"constexpr bool TOPK_FUSED_STABLE = {str(self.tk_mode == 'fused').lower()};",
            f"constexpr bool TOPK_RANK_STAMPED = {str(self.tk_mode == 'rank_stamped').lower()};",
            f"constexpr int TOPK_REBUILD_SKIP_SECOND = {self.tk_skip};",
            f"#define TOPK_TILE0 {1 if self.tk_tile0 else 0}",
        ])


def _rows():
    rows = []
    for mode, fp32 in MODES:
        for idir in (0, 1):
            for k in (4, 8, 16, 32, 64):
                rows.append(("sort", k, idir, 0, mode, fp32, 1, False))
                for m in ((0, 1, 2, 3, 4) if k <= 16 else (0, 1, 2)):
                    rows.append(("merge", k, idir, m, mode, fp32, 1, False))
                for m in ((0, 1, 2, 3) if k <= 16 else (0, 1, 2)):
                    rows.append(("rebuild", k, idir, m, mode, fp32, 1, False))
            for k in (32, 64):
                rows.append(("rebuild", k, idir, 0, mode, fp32, 0, False))
                rows.append(("sort", k, idir, 0, mode, fp32, 1, True))
    for largest in (0, 1):
        idir = 0 if largest else 1
        rows += [("fuse", 32, idir, 0, "fused", True, 1, False), ("defuse", 32, idir, 0, "fused", True, 1, False),
                 ("stamp_local", 32, idir, 0, "rank_stamped", True, 1, False),
                 ("stamp_range", 32, idir, 0, "rank_stamped", True, 1, False)]
    rows += [("strip_tags", 32, 0, 0, "rank_stamped", True, 1, False), ("negzero", 32, 0, 0, "unstable", True, 1, False)]
    for fp32 in (False, True):
        rows += [(op, 32, 0, 0, "unstable", fp32, 1, False) for op in ("finalize_u16", "u16_prepare", "u16_move_half")]
    return rows


def _id(r):
    op, k, idir, m, mode, fp32, skip, t0 = r
    return (f"{op}-k{k}-d{idir}-m{m}-{mode}-{'fp32' if fp32 else 'fp16'}{'-s0' if skip == 0 else ''}"
            f"{'-tile0' if t0 else ''}")


ROWS = _rows()
if os.environ.get("TK_PICK"):
    _pick = set(open(os.environ["TK_PICK"]).read().split())
    ROWS = [r for r in ROWS if _id(r) in _pick]


@pytest.mark.perf
@pytest.mark.parametrize("row", ROWS, ids=[_id(r) for r in ROWS])
def test_perf_topk_allcfg(perf_report, row):
    op, k, idir, m, mode, fp32, skip, t0 = row
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    PerfConfig(
        "sources/topk_allcfg_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[TOPK_ALLCFG(tk_op=op, tk_k=k, tk_idir=idir, tk_miter=m, tk_mode=mode, tk_skip=skip, tk_tile0=t0)],
        runtimes=[TILE_COUNT(1), LOOP_FACTOR(16)],
        variant_stimuli=StimuliConfig(None, formats.input_format, None, formats.input_format, formats.output_format,
                                      tile_count_A=1, tile_count_B=1, tile_count_res=1),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.Yes if fp32 else DestAccumulation.No,
    ).run(perf_report)
