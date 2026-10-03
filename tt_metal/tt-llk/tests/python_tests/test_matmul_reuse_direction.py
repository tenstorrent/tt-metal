# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.device import LLKAssertException
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

KT_DIM = 2


@dataclass
class MATMUL_INIT_BLOCK(TemplateParameter):
    rt_dim: int
    ct_dim: int
    pack_result: bool = True

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t INIT_RT_DIM = {self.rt_dim};",
                f"constexpr std::uint32_t INIT_CT_DIM = {self.ct_dim};",
                f"constexpr bool PACK_RESULT = {str(self.pack_result).lower()};",
            ]
        )


def _configuration(formats, init_block, block, pack_result):
    (init_rt, init_ct), (rt, ct) = init_block, block
    a_shape, b_shape = (rt * 32, KT_DIM * 32), (KT_DIM * 32, init_ct * 32)
    # Small binary fractions keep every product and sum exact at LoFi with a 16-bit DEST.
    generator = torch.Generator().manual_seed(7)
    a = torch.randint(-1, 2, a_shape, generator=generator).to(torch.bfloat16) / 8
    b = torch.randint(-1, 2, b_shape, generator=generator).to(torch.bfloat16) / 8

    def tiled(tensor, shape):
        return tilize_block(
            tensor, dimensions=shape, stimuli_format=formats.input_format
        ).flatten()

    configuration = TestConfig(
        "sources/matmul_reuse_direction_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            THROTTLE_LEVEL(0),
            MATMUL_INIT_BLOCK(init_rt, init_ct, pack_result),
        ],
        runtimes=[NUM_FACES(), TILE_COUNT(rt * ct), CRK_TILE_DIMM(ct, rt, KT_DIM)],
        variant_stimuli=StimuliConfig(
            tiled(a, a_shape),
            formats.input_format,
            tiled(b, b_shape),
            formats.input_format,
            formats.output_format,
            tile_count_A=rt * KT_DIM,
            tile_count_B=KT_DIM * init_ct,
            tile_count_res=rt * ct,
        ),
        dest_acc=DestAccumulation.No,
    )
    return configuration, a, b[:, : ct * 32]


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    # (init rt x ct, call rt x ct): the call narrows the block and keeps ct >= rt as it was at init
    blocks=[
        ((1, 4), (1, 1)),
        ((1, 4), (1, 3)),
        ((2, 4), (2, 2)),
        ((2, 4), (2, 3)),
        ((4, 1), (3, 1)),
        ((4, 2), (3, 2)),
    ],
)
def test_matmul_narrowed_block(formats, blocks):
    """Blocks narrower than the init's in the same reuse direction, as the DRAM sharded matmul's last sub block."""
    init_block, block = blocks
    configuration, a, b = _configuration(formats, init_block, block, True)
    rt, ct = block
    result = torch.tensor(configuration.run().result, dtype=torch.bfloat16)
    assert result.numel() == rt * ct * 1024
    expected = get_golden_generator(MatmulGolden)(
        a,
        b,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=(rt * 32, KT_DIM * 32),
        input_B_dimensions=(KT_DIM * 32, ct * 32),
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    assert passed_test(expected, result, formats.output_format)


@skip_for_wormhole
@skip_for_quasar
@pytest.mark.skipif(
    os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1", reason="needs LLK asserts"
)
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    # (init rt x ct, call rt x ct): ct >= rt at init and not in the call, and the reverse
    blocks=[((2, 2), (2, 1)), ((2, 1), (1, 1))],
)
def test_matmul_reuse_direction_flip_asserts(formats, blocks):
    """A call whose reuse direction differs from the init's stops at the LLK assert instead of desynchronising the threads."""
    init_block, block = blocks
    configuration, _, _ = _configuration(formats, init_block, block, False)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        LLKAssertException
    ):
        configuration.run()
