# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compile/smoke test for the Blackhole-only experimental hardware-teardown LLK
family "hw_cleanup" (compute_kernel_hw_cleanup.h ->
llk_{unpack,math,pack}_hw_cleanup.h + shared llk_hw_cleanup.h).

hw_cleanup is a TEARDOWN family with NO numeric output of its own. It drains the
three TRISCs, rendezvouses T0/T1/T2 through hardware mailboxes, and reprograms
both cfg banks to a canonical Float16_b 32x32 / four-face / 2048B geometry,
leaving cfg bank 0 selected (see the header docstrings).

The C++ source copies two tiles with the per-thread cleanup canonicals (the same
entry points compute_kernel_hw_cleanup() dispatches) between them. Each thread
finishes the first tile, the pack included, before the cleanup, which is the
API's precondition (MATH_PACK and UNPACK_SYNC drained); every thread re-inits
after it, as a following op must (the cleanup poisons the MOPs, the math
ADDR_MODs and the pack strides). The golden is the identity of both tiles.

Blackhole-only, and off the simulator, which cannot model the cfg-bank reprogram
(UnimplementedFunctionality: tensix_cfg_wr32 reg=281).
"""

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TILE_COUNT
from helpers.utils import passed_test

pytestmark = blackhole_only

# Single config: the canonical geometry cleanup itself restores (32x32 tiles,
# four faces, Float16_b in and out), one tile before the cleanup and one after.
NUM_FACES_VALUE = 4


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[64, 32]],
)
def test_hw_cleanup(formats, dest_acc, input_dimensions, request):
    if request.config.getoption("--run-simulator"):
        pytest.skip("ttsim cannot model the cfg-bank reprogram (reg 281)")

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Golden: identity datacopy of both tiles; the cleanup contributes no data.
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        NUM_FACES_VALUE,
        input_dimensions,
    )

    configuration = TestConfig(
        "sources/hw_cleanup_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(NUM_FACES_VALUE),
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
            num_faces=NUM_FACES_VALUE,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Every lane of both tiles is defined (full identity datacopy).
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
