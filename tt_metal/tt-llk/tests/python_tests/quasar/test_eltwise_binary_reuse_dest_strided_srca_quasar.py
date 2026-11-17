# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for tt-metal#60241 item 3: a strided SrcA unpack followed by an eltwise binary with reuse_dest.
# See sources/quasar/eltwise_binary_reuse_dest_strided_srca_quasar_test.cpp for the poison modes.
from dataclasses import dataclass

import pytest

# Module import, not `from ... import test_*`, so pytest does not collect the base test here too.
import quasar.test_eltwise_binary_reuse_dest_quasar as base
from helpers.format_config import DataFormat
from helpers.llk_params import (
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
    quasar_dest_sync_modes,
    runtime,
)
from helpers.test_variant_parameters import TemplateParameter


@dataclass
class STRIDE_POISON(TemplateParameter):
    mode: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int STRIDE_POISON = {self.mode};"


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwadd],
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    dest_sync_mode=lambda: quasar_dest_sync_modes(is_perf=False),
    input_dimensions=runtime(base.INPUT_DIMENSIONS),
    output_dimensions=runtime(base.valid_output_dimensions),
    stride_poison=[0, 1, 2],
)
def test_eltwise_binary_reuse_dest_strided_srca_quasar(
    formats,
    mathop,
    reuse_dest_type,
    dest_sync_mode,
    input_dimensions,
    output_dimensions,
    stride_poison,
):
    base.test_eltwise_binary_reuse_dest_quasar(
        formats,
        mathop,
        reuse_dest_type,
        MathFidelity.LoFi,
        dest_sync_mode,
        input_dimensions,
        output_dimensions,
        [PerfRunType.L1_TO_L1],
        1,
        test_name="sources/quasar/eltwise_binary_reuse_dest_strided_srca_quasar_test.cpp",
        extra_templates=(STRIDE_POISON(stride_poison),),
    )
