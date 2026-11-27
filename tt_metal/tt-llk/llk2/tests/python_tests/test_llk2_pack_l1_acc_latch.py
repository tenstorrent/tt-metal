# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware property: the packer takes Pack_L1_Acc per PACR, when the PACR is issued.

llk_lib/pack.h switches Pack_L1_Acc between the PACRs of a sum pack, and pack_configure writes it behind the
previous pack's PACRs. Both wait for the packer first, but a compiler that tracks Tensix state deletes those waits
(its model: a PACR takes its config when it is issued), so the write then sits right behind PACRs the packer is
still working on. This test pins the property that makes that safe, on the device, and fails if it does not hold.

The kernel (llk2_pack_l1_acc_latch_test.cpp) packs a matmul's result Y once (L1 = Y), then PASSES more times tile
by tile with the bit switched from acc_from to acc_to behind PACR switch_at - 1 of every tile, with no wait for the
packer. A PACR packs 4 Dest rows = 64 datums, so if the bit is taken per PACR at issue, exactly the first
64 * switch_at datums of every tile are packed with acc_from and the rest with acc_to: datums packed without
accumulation end as Y, the others as Y added PASSES + 1 times. switch_at = 16 is the write right behind the tile's
last PACR (what the deleted wait leaves; use_mop: behind a MOP that issues the 16 PACRs back to back);
switch_at = 15 is its positive control, 0 the plain pack.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, PackGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
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

PASSES = 8
PACRS_PER_TILE = 16
DATUMS_PER_PACR = TILE_DATUMS // PACRS_PER_TILE  # 4 Dest rows of 16


@dataclass
class LLK2_PACK_L1_ACC_LATCH(TemplateParameter):
    tiles: int
    acc_from: int
    acc_to: int
    switch_at: int
    use_mop: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES = {self.tiles};\n"
            f"constexpr std::uint32_t PASSES = {PASSES};\n"
            f"constexpr std::uint32_t ACC_FROM = {self.acc_from};\n"
            f"constexpr std::uint32_t ACC_TO = {self.acc_to};\n"
            f"constexpr std::uint32_t SWITCH_AT = {self.switch_at};\n"
            f"constexpr bool USE_MOP = {'true' if self.use_mop else 'false'};"
        )


def _combinations():
    combos = []
    # L1 adds Float32, Float16_b and Float16 only.
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if (
                fmt.input_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for acc_from, acc_to in ((0, 1), (1, 0)):
                for switch_at, use_mop in (
                    (0, False),
                    (1, False),
                    (8, False),
                    (15, False),
                    (16, False),
                    (16, True),
                ):
                    combos.append((fmt, dest_acc, acc_from, acc_to, switch_at, use_mop))
    return combos


@parametrize(case=_combinations())
def test_llk2_pack_l1_acc_latch(case):
    formats, dest_acc, acc_from, acc_to, switch_at, use_mop = unwrap_case(case)
    tiles = DEST_HALF_TILES[dest_acc]
    dims_A, dims_B = [TILE_DIM, TILE_DIM], [TILE_DIM, tiles * TILE_DIM]
    torch_format = format_dict[formats.output_format]
    math_fidelity = MathFidelity.LoFi

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    y = get_golden_generator(MatmulGolden)(
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
    # Datums packed with the bit off hold Y (the last pack's); datums packed with it on hold Y added once per pack.
    y = torch.as_tensor(y).view(tiles, TILE_DATUMS)
    summed = torch.as_tensor(
        PackGolden.accumulate_l1([y.flatten()] * (PASSES + 1), formats.output_format)
    ).view(tiles, TILE_DATUMS)
    before_switch = torch.arange(TILE_DATUMS) < switch_at * DATUMS_PER_PACR
    with_acc = before_switch if acc_from else ~before_switch
    golden = torch.where(with_acc, summed, y.to(summed.dtype)).flatten()

    tilized_A, tilized_B = tilize_operands(
        src_A, dims_A, src_B, dims_B, formats.input_format
    )
    configuration = TestConfig(
        cpp_source("llk2_pack_l1_acc_latch_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_PACK_L1_ACC_LATCH(tiles, acc_from, acc_to, switch_at, use_mop),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tiles,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=torch_format).view(tiles, TILE_DATUMS)
    golden = golden.to(torch_format).view(tiles, TILE_DATUMS)
    # Which value of the bit each datum was packed with, from the result alone: (PASSES + 1) * Y is far from Y.
    # y > 0 everywhere but where both are 0, which says nothing and is left out.
    yv = y.to(torch.float32)
    informative = yv > 0.05
    saw_acc = (res_tensor.to(torch.float32) > 0.5 * (PASSES + 2) * yv) & informative
    expected_acc = with_acc.expand(tiles, TILE_DATUMS) & informative
    wrong = saw_acc != expected_acc
    assert not wrong.any(), (
        f"Pack_L1_Acc was not taken per PACR at issue: {int(wrong.sum())} datums were packed with the other value; "
        f"per tile, first and last wrong datum: "
        f"{[(int(t), int(w.nonzero()[0]), int(w.nonzero()[-1])) for t, w in enumerate(wrong) if w.any()]}"
    )
    assert passed_test(
        golden.flatten(), res_tensor.flatten(), formats.output_format
    ), "Assert against golden failed"
