# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import CausalConv1dSiluGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DISABLE_SRC_ZERO_FLAG
from helpers.utils import passed_test

# Tile-count layout matches sfpu_causal_conv1d_silu_test.cpp:
#   buffer_A tiles: [wa, wb, wc, wd]  (4 per-channel conv weights)
#   buffer_B tiles: [x, y, z, w]      (3 cache entries + 1 matmul-produced sample)
#   buffer_Res tiles: [new_cache, x, y, silu_out]
_NUM_WEIGHT_TILES = 4
_NUM_DATA_TILES = 4
_NUM_RES_TILES = 4


def _run_causal_conv1d_silu(formats, dest_acc):
    spec = StimuliSpec.uniform(low=-1.0, high=1.0)

    # 8 independent single-tile operands, generated one pair at a time so the
    # tile order inside the concatenated buffers is fully under our control
    # (avoids relying on generate_stimuli's internal multi-tile layout).
    wa, _, wb, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        stimuli_format_B=formats.input_format,
        spec_A=spec,
        spec_B=spec,
    )
    wc, _, wd, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        stimuli_format_B=formats.input_format,
        spec_A=spec,
        spec_B=spec,
    )
    x, _, y, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        stimuli_format_B=formats.input_format,
        spec_A=spec,
        spec_B=spec,
    )
    z, _, w, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        stimuli_format_B=formats.input_format,
        spec_A=spec,
        spec_B=spec,
    )

    buffer_A = torch.cat([t.flatten() for t in (wa, wb, wc, wd)])
    buffer_B = torch.cat([t.flatten() for t in (x, y, z, w)])

    golden_generator = get_golden_generator(CausalConv1dSiluGolden)
    new_cache_g, x_g, y_g, silu_g = golden_generator(
        wa, wb, wc, wd, x, y, z, w, formats.output_format
    )
    golden = torch.cat([new_cache_g, x_g, y_g, silu_g])

    configuration = TestConfig(
        "sources/sfpu_causal_conv1d_silu_test.cpp",
        formats,
        templates=[DISABLE_SRC_ZERO_FLAG(True)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            buffer_A,
            formats.input_format,
            buffer_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=_NUM_WEIGHT_TILES,
            tile_count_B=_NUM_DATA_TILES,
            tile_count_res=_NUM_RES_TILES,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_causal_conv1d_silu(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("Float16_b inputs with dest_acc=Yes are not required for this op")

    _run_causal_conv1d_silu(formats, dest_acc)
