# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch sweep (not for merge): the reduce with a tilized operand A (pool2d, bilinear upsample) over its LLK axes.
Kernel sources/tilizeA_B_allcfg_perf.cpp: the per-tile form (main's llk_unpack_tilizeA_B_block) on every tree, the
block call where the tree has _llk_unpack_tilizeA_B_block_ (#59143)."""
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, MathOperation, PerfRunType, ReducePool
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (IN_TILE_DIMS, LOOP_FACTOR, MATH_FIDELITY, MATH_OP, NUM_BLOCKS, NUM_FACES,
                                             TILE_COUNT, TemplateParameter, generate_input_dim)

F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes
BF16, FP32 = InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Float32, F.Float32)
_LLK = Path(os.environ.get("LLK_HOME", Path(__file__).resolve().parents[2]))
HAS_BLOCK = "_llk_unpack_tilizeA_B_block_" in (_LLK / "tt_llk_blackhole/llk_lib/llk_unpack_tilize.h").read_text()


@dataclass
class TAB(TemplateParameter):
    tab_form: str = "tile"
    tab_reloadb: int = 1
    tab_zeroa: int = 0

    def convert_to_cpp(self) -> str:
        return (f"#define TAB_BLOCK {1 if self.tab_form == 'block' else 0}\n#define TAB_RELOADB {self.tab_reloadb}\n"
                f"#define TAB_ZEROA {self.tab_zeroa}")


V = []  # (pool, fmt, acc, r, nf, ct, nb, reloadb, zeroa)
for r, nf in ((1, 2), (2, 2), (4, 2), (8, 2), (9, 2), (16, 2), (16, 4), (4, 4), (9, 4)):
    for ct in (1, 2, 4, 8):
        V.append(("max", BF16, NO, r, nf, ct, 1, 1, 0))
for pool, fmt, acc in (("avg", BF16, NO), ("max", FP32, NO), ("max", BF16, YES), ("avg", FP32, YES)):
    for r, nf in ((1, 2), (4, 2), (9, 2), (16, 2), (16, 4)):
        for ct in (1, 8):
            V.append((pool, fmt, acc, r, nf, ct, 1, 1, 0))
for r, nf in ((16, 4), (9, 2)):
    V.append(("max", BF16, NO, r, nf, 8, 8, 1, 0))
V += [("max", BF16, NO, 9, 2, 8, 1, 0, 0), ("max", BF16, NO, 9, 2, 8, 1, 1, 1)]
CASES = [(form,) + v for form in ("tile", "block") for v in V]


def _id(c):
    form, pool, fmt, acc, r, nf, ct, nb, rb, za = c
    return f"{form}-{pool}-{fmt.input_format.name}-{acc.name}-r{r}n{nf}-c{ct}-b{nb}-rb{rb}-za{za}"


@pytest.mark.perf
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_perf_tilizeA_B_allcfg(perf_report, case):
    form, pool, fmt, acc, r, nf, ct, nb, rb, za = case
    if form == "block" and not HAS_BLOCK:
        pytest.skip("no block tilizeA_B in this tree")
    rows = r * (2 if nf == 4 else 1)
    cols = 16 if nf == 1 else 32
    configuration = PerfConfig(
        "sources/tilizeA_B_allcfg_perf.cpp",
        fmt,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=ReducePool.Max if pool == "max" else ReducePool.Average),
                   MATH_FIDELITY(MathFidelity.HiFi4), TAB(form, rb, za)],
        runtimes=[generate_input_dim((32, ct * 32), (32, ct * 32)), TILE_COUNT(ct), NUM_BLOCKS(nb), LOOP_FACTOR(32),
                  NUM_FACES(nf), IN_TILE_DIMS(in0_r_dim=rows, in0_c_dim=cols, in1_r_dim=rows, in1_c_dim=cols)],
        variant_stimuli=StimuliConfig(None, fmt.input_format, None, fmt.input_format, fmt.output_format,
                                      tile_count_A=ct, tile_count_B=1, tile_count_res=ct),
        dest_acc=acc,
        unpack_to_dest=False,
    )
    configuration.run(perf_report)
