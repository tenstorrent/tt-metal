# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-op SrcA Y-stride restore test for the SDPA `_llk_unpack_bcastA_B_` path.

This is Phase 3 + Phase 4 of the unpacker-stride coverage roadmap, covering the
two PR #45127 changes that touch the SrcA **Y-stride**:

* **C1 / Phase 4** — `_llk_unpack_bcastA_B_uninit_` now restores the canonical
  Y-stride (`canonical_unpA_y_stride(dst)` = 16 for bf16) instead of the hardcoded
  bcast value 32. NO existing test calls this function at all.
* **C2 / Phase 3** — `_llk_unpack_reconfig_data_format_srca_impl_` (FACE_ROW_MAJOR
  branch) now also (re)writes that canonical Y-stride. The matmul-tilize test only
  ever calls the reconfig with `p_dim_stride_target::IGNORE`, so the new write is
  otherwise unexercised.

`_llk_unpack_bcastA_B_init_` mutates the SrcA Y-stride to 32. The kernel runs a
bcast op (run 0), restores via the method under test (uninit OR reconfig), then
does a plain `_llk_unpack_A_` datacopy of the ORIGINAL operand-A tile (run 1) with
no other state reset. If the restore leaves Y-stride at 32 instead of the canonical
16, run 1 reads SrcA with the wrong row stride and diverges from the datacopy golden.

Only 16-bit formats (Float16_b / Float16) expose the bug: for those the canonical
Y-stride is 16, whereas the old hardcoded value was 32. (Float32's canonical stride
is already 32, so it would not reveal the regression.)
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    SRCA_REUSE_COUNT,
    TemplateParameter,
)
from helpers.utils import passed_test

# 1 SrcA tile reused for 1 SrcB tile — the minimal bcast that still sets Y-stride=32.
BCAST_SRCA_REUSE = 1
NUM_FACES = 4


@dataclass
class RESTORE_VIA_RECONFIG(TemplateParameter):
    """Selects the run-0 -> run-1 SrcA restore path in the C++ kernel.

    False -> `_llk_unpack_bcastA_B_uninit_` (C1, Phase 4)
    True  -> `_llk_unpack_reconfig_data_format_srca_impl_<.., FACE_ROW_MAJOR>` (C2, Phase 3)
    """

    via_reconfig: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool RESTORE_VIA_RECONFIG = {str(self.via_reconfig).lower()};"
        )


@skip_for_blackhole
@parametrize(
    # Same format for both ops so the run-1 datacopy needs no data-format reconfig —
    # isolating the Y-stride restore as the only state change that matters.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    # "uninit" -> Phase 4 (C1); "reconfig" -> Phase 3 (C2).
    restore_via_reconfig=[False, True],
)
def test_unpack_bcastA_B_uninit_restore(
    formats,
    dest_acc,
    restore_via_reconfig,
):
    torch_format = format_dict[formats.output_format]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Golden is the run-1 plain datacopy of the original operand-A tile; the run-0
    # bcast only pollutes the SrcA Y-stride. A correct restore yields an identity
    # copy of src_A in face layout.
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        NUM_FACES,
        input_dimensions,
        16,  # face_r_dim
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/unpack_bcastA_B_uninit_restore_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Elwadd),
            RESTORE_VIA_RECONFIG(via_reconfig=restore_via_reconfig),
        ],
        runtimes=[
            SRCA_REUSE_COUNT(BCAST_SRCA_REUSE),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        L1_to_L1_iterations=L1_to_L1_iterations,
    ), "Assert against golden failed"
