# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58722 review, CI only): the block unpack (_llk_unpack_AB_block_) and the block pack
(_llk_pack_block_contiguous_) in the unified harness, at 2, 4, 8 and 2 x 8 tiles per DEST section: main (per face, per-tile
calls), per tile, per tile with the block unpack, and with the block pack as well."""
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType, DestAccumulation, DestSync, MathFidelity, MathOperation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.test_variant_parameters import TemplateParameter
from test_eltwise_binary import _run_eltwise_binary_test


@dataclass
class EB_BLOCK(TemplateParameter):
    eb_variant: str = "main"

    def convert_to_cpp(self) -> str:
        u = "true" if self.eb_variant in ("pt_bu", "pt_bu_bp") else "false"
        p = "true" if self.eb_variant == "pt_bu_bp" else "false"
        return f"#define EB_BLOCK_DEFINED 1\nconstexpr bool EB_BLOCK_UNPACK = {u};\nconstexpr bool EB_BLOCK_PACK = {p};"


VARIANTS = {"main": True, "pt": False, "pt_bu": False, "pt_bu_bp": False}
CASES = [
    ("bf16_add_lofi", InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b), MathOperation.Elwadd, MathFidelity.LoFi),
    ("bfp8_add_lofi", InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b), MathOperation.Elwadd, MathFidelity.LoFi),
    ("bfp8in_bf16out_add_lofi", InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b), MathOperation.Elwadd, MathFidelity.LoFi),
    ("bf16_mul_hifi4", InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b), MathOperation.Elwmul, MathFidelity.HiFi4),
    ("bf16_mul_hifi2", InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b), MathOperation.Elwmul, MathFidelity.HiFi2),
]


@pytest.mark.perf
@parametrize(
    case=[c[0] for c in CASES],
    variant=list(VARIANTS),
    input_dimensions=[[64, 32], [128, 32], [256, 32], [512, 32]],
    run_types=[ALL_PERF_RUN_TYPES],
)
def test_perf_eltwise_binary_blk(perf_report, case, variant, input_dimensions, run_types):
    _, formats, op, fid = next(c for c in CASES if c[0] == case)
    per_face = VARIANTS[variant]
    if variant == "pt_bu_bp" and formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("the block pack takes no per-tile exponent section")
    _run_eltwise_binary_test(
        DestAccumulation.No,
        DestSync.Half,
        False,
        formats,
        BroadcastType.None_,
        op,
        fid,
        Transpose.No,
        input_dimensions,
        [32, 32],
        False,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=32,
        per_face_handoff=per_face,
        extra_templates=(EB_BLOCK(variant),),
    )


@parametrize(
    case=[c[0] for c in CASES],
    variant=["pt_bu", "pt_bu_bp"],
    input_dimensions=[[64, 32], [256, 32], [512, 32]],
)
def test_eltwise_binary_blk_functional(case, variant, input_dimensions):
    _, formats, op, fid = next(c for c in CASES if c[0] == case)
    if variant == "pt_bu_bp" and formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("the block pack takes no per-tile exponent section")
    _run_eltwise_binary_test(
        DestAccumulation.No,
        DestSync.Half,
        False,
        formats,
        BroadcastType.None_,
        op,
        fid,
        Transpose.No,
        input_dimensions,
        [32, 32],
        False,
        per_face_handoff=False,
        extra_templates=(EB_BLOCK(variant),),
    )
