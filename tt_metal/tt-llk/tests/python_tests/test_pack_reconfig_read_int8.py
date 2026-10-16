# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Packer data-format reconfig across an 8-bit integer output with a 16-bit Dest.

Two runs of one kernel pack the same Dest contents with ``dest_acc=No``. Run 0 configures the
packer directly for the output format. Run 1 configures the format on the other side of the
int8 / float boundary first (the format a preceding op would have left behind) and then
reconfigures to the output format with ``_llk_pack_reconfig_data_format_``. With a 16-bit Dest
the packer's Dest-read control decides how a datum is read: an Int8/UInt8 pack takes the raw
8-bit integer read, the float formats take the rounding read. The full configure programs that
control from the format, the reconfig has to program the same field set, so run 1 must pack the
bits run 0 packs.

Legs:
  Float16_b -> UInt8, previous format Float16_b: the math thread A2D-copies each tile into Dest
      and an in-place SFPU pass rounds every datum to an integer in 0..65535 and stores its low
      10 magnitude bits back as an "integer 8" Dest datum. Run 0 is also checked against
      ``value & 0xFF`` (the raw 8-bit read keeps the low byte of the magnitude).
  Float16_b -> Float16_b, previous format UInt8: plain datacopy of stimuli that carry bf16
      denormals and NaN. The rounding read the full configure selects flushes denormals and turns
      NaN into infinity; a raw read passes them through.
"""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CONFIGURE_TEST_RUN_IDX,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)


def _whole_numbers_0_1023(size, dtype, generator):
    # bf16 keeps 8 significant bits, so values above 256 are quantized; the golden works from
    # the quantized stimuli, so the exact set of values does not matter.
    return torch.randint(0, 1024, (size,), generator=generator).to(dtype)


def _normals_denormals_nan(size, dtype, generator):
    # Thirds: whole numbers in [1, 255], bf16 denormals (zero exponent, non-zero mantissa in
    # the top 7 fraction bits, so the fp32 -> bf16 conversion is exact), and NaN.
    values = torch.randint(1, 256, (size,), generator=generator).to(torch.float32)
    kind = torch.randint(0, 3, (size,), generator=generator)
    denormal_bits = (
        torch.randint(1, 128, (size,), generator=generator, dtype=torch.int32) << 16
    )
    values[kind == 1] = denormal_bits.view(torch.float32)[kind == 1]
    values[kind == 2] = float("nan")
    return values.to(dtype)


_UINT8_OUT = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt8)
_BF16_OUT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


@parametrize(
    formats=[_UINT8_OUT, _BF16_OUT],
    # 16-bit Dest is the point of the test: the 8-bit integer Dest read only exists there.
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[32, 32], [64, 64]],
)
def test_pack_reconfig_read_int8(formats, dest_acc, input_dimensions):
    int8_out = formats.output_format == DataFormat.UInt8
    spec = StimuliSpec(
        distribution=_whole_numbers_0_1023 if int8_out else _normals_denormals_nan,
        seed=0,
    )
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    def runtimes(run_idx):
        return [
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            CONFIGURE_TEST_RUN_IDX(run_idx),
        ]

    configuration = TestConfig(
        "sources/pack_reconfig_read_int8_test.cpp",
        formats,
        templates=[generate_input_dim(input_dimensions, input_dimensions)],
        runtimes=runtimes(0),
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
        unpack_to_dest=False,
    )
    if int8_out:
        # The SFPU pass leaves 8-bit integer datums in Dest, so the packer reads UInt8, not the
        # inferred Float16_b that a plain datacopy would leave there (same patch as the typecast
        # test).
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = DataFormat.UInt8

    torch_format = format_dict[formats.output_format]
    # Compare bit patterns: NaN never compares equal as a float.
    bits_format = torch.uint8 if int8_out else torch.int16

    # Run 0: full configure straight onto the output format.
    direct = torch.tensor(configuration.run().result, dtype=torch_format)
    assert (
        len(direct) == src_A.numel()
    ), f"result has {len(direct)} datums, input has {src_A.numel()}"
    if int8_out:
        golden_tensor = (
            src_A.to(torch.float32).round().clamp(0, 65535).to(torch.int64) & 0xFF
        ).to(torch.uint8)
        assert torch.equal(direct, golden_tensor), (
            "UInt8 pack from a 16-bit Dest after a full configure does not keep the low byte of "
            "the Dest datum magnitude"
        )

    # Run 1: full configure for the previous format, then reconfig onto the output format.
    # Only the runtime parameter changes between the runs.
    configuration.runtimes = runtimes(1)
    reconfigured = torch.tensor(configuration.run().result, dtype=torch_format)
    assert torch.equal(reconfigured.view(bits_format), direct.view(bits_format)), (
        f"{formats.output_format.name} pack from a 16-bit Dest after "
        "_llk_pack_reconfig_data_format_ differs from the same pack after a full configure"
    )
