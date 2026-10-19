# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch sweep (not for merge): the plain tilize (unpack tilize, A2D tilize math, tilize pack) over every
configuration axis the Blackhole LLK takes. Kernel sources/tilize_allcfg_perf.cpp; the block form runs only where
the tree has _llk_unpack_tilize_block_ (#60672)."""
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TemplateParameter, generate_input_dim

F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes
_LLK = Path(os.environ.get("LLK_HOME", Path(__file__).resolve().parents[2]))
HAS_BLOCK = "_llk_unpack_tilize_block_" in (_LLK / "tt_llk_blackhole/llk_lib/llk_unpack_tilize.h").read_text()


@dataclass
class TA(TemplateParameter):
    ta_form: str = "tile"  # tile = one unpack call per tile, block = one per row (#60672)
    ta_face_r: int = 16
    ta_nf: int = 4
    ta_full: int = 0
    ta_sect: int = 0  # tiles per DEST section, 0 = as many as fit

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [f"#define UNPACK_BLOCK {1 if self.ta_form == 'block' else 0}", f"#define TA_FACE_R {self.ta_face_r}",
             f"#define TA_NF {self.ta_nf}", f"#define TA_FULL {self.ta_full}", f"#define TA_SECT {self.ta_sect}"])


def io(a, b):
    return InputOutputFormat(a, b)


FLOAT_IN = [F.Float16_b, F.Float16, F.Float32, F.Fp8_e4m3]
FLOAT_OUT = [F.Float16_b, F.Float16, F.Float32, F.Bfp8_b, F.Fp8_e4m3]
FMTS = [io(a, b) for a in FLOAT_IN for b in FLOAT_OUT] + [io(F.Float16_b, F.Bfp4_b), io(F.Float32, F.Bfp4_b)]
INTS = [io(F.Int32, F.Int32), io(F.UInt32, F.UInt32), io(F.UInt16, F.UInt16)]
BF16, FP32 = io(F.Float16_b, F.Float16_b), io(F.Float32, F.Float32)

CASES = []  # (name, fmt, acc, rt, ct, TA)
for fmt in FMTS + INTS:
    for acc in (NO, YES):
        for ct in (1, 2, 4, 8, 16):
            CASES.append(("fmt", fmt, acc, 1, ct, TA()))
for fmt, acc in ((BF16, NO), (FP32, YES), (io(F.Float32, F.Float16_b), NO)):
    for ct in (3, 5, 6, 7, 12, 32):
        CASES.append(("ct", fmt, acc, 1, ct, TA()))
    for ct in (1, 2, 4, 8, 16):
        CASES.append(("rt", fmt, acc, 4, ct, TA()))
for fmt, acc in ((BF16, NO), (FP32, YES)):
    for fr in (1, 2, 4, 8, 16):
        for nf in (2, 4):
            if (fr, nf) == (16, 4):
                continue
            for ct in (1, 4, 8):
                CASES.append(("tiny", fmt, acc, 1, ct, TA(ta_face_r=fr, ta_nf=nf)))
for fmt, acc in ((BF16, NO), (io(F.Float32, F.Float16_b), NO), (FP32, YES), (io(F.Int32, F.Int32), YES),
                 (io(F.Float16_b, F.Bfp8_b), NO)):
    for ct in (1, 2, 4, 8):
        CASES.append(("sect1", fmt, acc, 1, ct, TA(ta_sect=1)))
for fmt, acc in ((BF16, NO), (FP32, YES)):
    for ct in (4, 8, 16):
        CASES.append(("full", fmt, acc, 1, ct, TA(ta_full=1)))
for fmt, acc in ((BF16, NO), (io(F.Float16, F.Float16), NO), (io(F.Float32, F.Float16_b), NO), (FP32, YES),
                 (io(F.Int32, F.Int32), YES), (io(F.Float16_b, F.Bfp8_b), NO), (BF16, YES)):
    for ct in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
        CASES.append(("block", fmt, acc, 1, ct, TA(ta_form="block")))
    for ct in (2, 8):
        CASES.append(("block", fmt, acc, 4, ct, TA(ta_form="block")))
for fmt, acc in ((BF16, NO), (FP32, YES)):
    for ct in (4, 8, 16):
        CASES.append(("block", fmt, acc, 1, ct, TA(ta_form="block", ta_full=1)))


def _id(c):
    n, fmt, acc, rt, ct, ta = c
    return (f"{n}-{fmt.input_format.name}-{fmt.output_format.name}-{acc.name}-r{rt}c{ct}-f{ta.ta_face_r}n{ta.ta_nf}"
            f"-s{ta.ta_sect}-y{ta.ta_full}-{ta.ta_form}")


@pytest.mark.perf
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_perf_tilize_allcfg(perf_report, case):
    _, fmt, acc, rt, ct, ta = case
    if ta.ta_form == "block" and not HAS_BLOCK:
        pytest.skip("no block tilize in this tree")
    tiles = rt * ct
    rows = ta.ta_face_r * (2 if ta.ta_nf == 4 else 1)
    dims = [rt * rows, ct * 32]
    configuration = PerfConfig(
        "sources/tilize_allcfg_perf.cpp",
        fmt,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[ta],
        runtimes=[generate_input_dim(dims, dims, tile_dimensions=(rows, 32)), TILE_COUNT(tiles), LOOP_FACTOR(256 if tiles <= 8 else (128 if tiles <= 16 else 64))],
        variant_stimuli=StimuliConfig(None, fmt.input_format, None, fmt.input_format, fmt.output_format,
                                      tile_count_A=tiles, tile_count_B=tiles, tile_count_res=tiles),
        dest_acc=acc,
        unpack_to_dest=fmt.input_format in (F.Int32, F.UInt32),
    )
    configuration.run(perf_report)
