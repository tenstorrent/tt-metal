# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 pack's writes to config words shared with the math thread, while the math thread's next op runs.

pack_configure writes STACC_RELU ApplyRelu and ALU_FORMAT_SPEC_REG2 Dstacc after a wait for the packer only; the
bytes they sit in also hold fields the FPU and the SFPU read. The kernel (llk2_pack_shared_config_test.cpp) runs
ROUNDS ops back to back (a matmul, or a copy and a silu on the SFPU) over a Dest ring of two sections and packs
them with ReLU alternately off and on, so every pack's configure changes the shared byte while the next op runs.
Every round must come out as its own ReLU setting says.
"""

from dataclasses import dataclass

import torch
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    PackGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PackerReluType,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.utils import passed_test
from llk2_common import (
    DEST_HALF_TILES,
    TILE_DATUMS,
    TILE_DIM,
    tilize_operands,
    unwrap_case,
)
from llk2_paths import cpp_source, include_dirs

ROUNDS = 8  # ReLU off, on, off, on, ...: seven op boundaries, each a change of the shared byte
KT = 2  # a matmul of two K tiles per output tile: the FPU is busy for the whole of the previous pack's configure


@dataclass
class LLK2_PACK_SHARED_CONFIG(TemplateParameter):
    tiles: int
    kt_dim: int
    sfpu_op: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES = {self.tiles};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr std::uint32_t ROUNDS = {ROUNDS};\n"
            f"constexpr bool SFPU_OP = {'true' if self.sfpu_op else 'false'};"
        )


def _combinations():
    combos = []
    for sfpu_op in (False, True):
        for fmt in input_output_formats(
            [DataFormat.Float16_b, DataFormat.Float32], same=True
        ):
            for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
                if fmt.input_format == DataFormat.Float32 and (
                    dest_acc == DestAccumulation.No or sfpu_op
                ):
                    continue  # the SFPU case stays with inputs that go through SrcA
                combos.append((fmt, dest_acc, sfpu_op))
    return combos


@parametrize(case=_combinations())
def test_llk2_pack_shared_config(case):
    formats, dest_acc, sfpu_op = unwrap_case(case)
    tiles = DEST_HALF_TILES[dest_acc]
    torch_format = format_dict[formats.output_format]
    math_fidelity = MathFidelity.LoFi
    spec = StimuliSpec.uniform(low=-1.0, high=1.0)  # negatives for ReLU to clamp

    if sfpu_op:
        dims_A = dims_B = [TILE_DIM, tiles * TILE_DIM]
        kt = 1
    else:
        kt = KT
        dims_A, dims_B = [TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, tiles * TILE_DIM]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    if sfpu_op:
        golden = get_golden_generator(UnarySFPUGolden)(
            MathOperation.Silu,
            src_A,
            formats.output_format,
            dest_acc,
            formats.input_format,
            dims_A,
        )
        stimuli_A, stimuli_B = src_A, src_B
    else:
        golden = get_golden_generator(MatmulGolden)(
            src_A,
            src_B,
            formats.output_format,
            math_fidelity,
            input_A_dimensions=dims_A,
            input_B_dimensions=dims_B,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
        stimuli_A, stimuli_B = tilize_operands(
            src_A, dims_A, src_B, dims_B, formats.input_format
        )
    pack_src = infer_data_formats(
        input_format=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
    ).pack_src
    relu_config = PackGolden.generate_relu_config(
        PackerReluType.ZeroRelu, relu_threshold=0.0, intermediate_format=pack_src
    )
    plain = torch.as_tensor(golden).to(torch_format)
    clamped = torch.as_tensor(PackGolden.apply_relu(golden, relu_config, pack_src)).to(
        torch_format
    )

    configuration = TestConfig(
        cpp_source("llk2_pack_shared_config_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_PACK_SHARED_CONFIG(tiles, kt, sfpu_op),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            stimuli_A,
            formats.input_format,
            stimuli_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=ROUNDS * tiles,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == ROUNDS * len(
        plain
    ), "Result tensor and golden tensor are not of the same length"
    assert (plain < 0).sum() > len(
        plain
    ) // 8, "the op's result must have negatives for ReLU to show"
    res_tensor = torch.tensor(res, dtype=torch_format).view(ROUNDS, tiles * TILE_DATUMS)
    for r in range(ROUNDS):
        expected = clamped if r % 2 else plain
        if r % 2:
            assert (res_tensor[r] >= 0).all(), f"round {r}: ReLU left negative values"
        else:
            # Clearly negative datums must stay negative: nothing clamped a round packed without ReLU.
            clearly_negative = plain.to(torch.float32) < -0.05
            kept = (res_tensor[r].to(torch.float32) < 0)[clearly_negative]
            assert (
                kept.all()
            ), f"round {r} (no ReLU): {int((~kept).sum())} negative datums came out clamped"
        assert passed_test(
            expected, res_tensor[r], formats.output_format
        ), f"round {r}: assert against golden failed"
