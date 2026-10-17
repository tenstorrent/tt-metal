# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Packer data-format reconfig across an 8-bit integer output with a 16-bit Dest.

Two runs of one kernel pack the same Dest contents with a 16-bit Dest. Run 0 configures the
packer directly for the output format. Run 1 configures the format on the other side of the
int8 / float boundary first (the format a preceding op would have left behind) and then
reconfigures to the output format with ``_llk_pack_reconfig_data_format_``. With a 16-bit Dest
the packer's Dest-read control decides how a datum is read: an Int8/UInt8 pack takes the raw
8-bit integer read, the float formats take the rounding read. The full configure programs that
control from the format, the reconfig has to program the same field set, so run 1 must pack the
bytes run 0 packs. The gate is on the raw result bytes in L1.

Legs:
  Float16_b -> Float16_b, previous format UInt8: plain datacopy of stimuli that carry bf16
      denormals and NaN. The rounding read the full configure selects flushes denormals and turns
      NaN into infinity; a raw read passes them through. A reconfig that leaves the raw read behind
      fails this leg.
  Float16_b -> UInt8, previous format Float16_b: the math thread A2D-copies each tile into Dest
      and an in-place SFPU pass rounds every datum to an integer in 0..65535 and stores its low
      10 magnitude bits back as an "integer 8" Dest datum. Run 0 is also checked against
      ``value & 0xFF`` (the raw 8-bit read keeps the low byte of the magnitude). Smoke coverage
      of the reconfig onto an 8-bit integer output: on Wormhole the UInt8 pack produces these bytes
      whether the raw read is selected or not, so the register-level gate for this direction is
      ``z_state/reconfig/test_pack_reconfig.py`` with ``dest_acc=No``.
"""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
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

# 16-bit Dest is the point of the test: the 8-bit integer Dest read only exists there.
_DEST_ACC = DestAccumulation.No

# An "integer 8" Dest datum keeps the low 10 magnitude bits of the value SFPSTORE INT8 writes.
_INT8_DEST_MAGNITUDE_LIMIT = 1 << 10
# The SFPU pass rounds FP32 to UInt16, saturating at the unsigned 16-bit range.
_UINT16_MAX = (1 << 16) - 1
# A UInt8 pack keeps the low byte of the Dest datum magnitude.
_UINT8_MASK = (1 << 8) - 1
# bf16 keeps 8 significant bits: whole numbers below this are exact in bf16.
_BF16_EXACT_WHOLE_LIMIT = 1 << 8
# bf16 is the top 16 bits of fp32; its mantissa is the top 7 fraction bits.
_FP32_TO_BF16_SHIFT = 16
_BF16_MANTISSA_BITS = 7


def _whole_numbers_below_int8_dest_limit(size, dtype, generator):
    # bf16 quantizes whole numbers at and above _BF16_EXACT_WHOLE_LIMIT; the golden works from
    # the quantized stimuli, so the exact set of values does not matter.
    return torch.randint(
        0, _INT8_DEST_MAGNITUDE_LIMIT, (size,), generator=generator
    ).to(dtype)


def _normals_denormals_nan(size, dtype, generator):
    # Thirds: exact whole numbers, bf16 denormals (zero exponent, non-zero mantissa in the top 7
    # fraction bits, so the fp32 -> bf16 conversion is exact), and NaN.
    values = torch.randint(1, _BF16_EXACT_WHOLE_LIMIT, (size,), generator=generator).to(
        torch.float32
    )
    kind = torch.randint(0, 3, (size,), generator=generator)
    denormal_bits = (
        torch.randint(
            1, 1 << _BF16_MANTISSA_BITS, (size,), generator=generator, dtype=torch.int32
        )
        << _FP32_TO_BF16_SHIFT
    )
    values[kind == 1] = denormal_bits.view(torch.float32)[kind == 1]
    values[kind == 2] = float("nan")
    return values.to(dtype)


_UINT8_OUT = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt8)
_BF16_OUT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


@parametrize(
    formats=[_UINT8_OUT, _BF16_OUT],
    input_dimensions=[[32, 32]],
)
def test_pack_reconfig_read_int8(formats, input_dimensions):
    int8_out = formats.output_format == DataFormat.UInt8
    spec = StimuliSpec(
        distribution=(
            _whole_numbers_below_int8_dest_limit if int8_out else _normals_denormals_nan
        ),
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
        _DEST_ACC,
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
        dest_acc=_DEST_ACC,
        unpack_to_dest=False,
    )
    if int8_out:
        # The SFPU pass leaves 8-bit integer datums in Dest, so the packer reads UInt8, not the
        # inferred Float16_b that a plain datacopy would leave there (same patch as the typecast
        # test).
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = DataFormat.UInt8

    def run_and_collect_raw_bytes():
        # The decoded result canonicalizes NaN and re-encodes the datums; the bit-identity gate is
        # on the bytes the packer wrote to L1.
        configuration.run()
        return configuration.variant_stimuli.collect_raw_result_bytes(
            TestConfig.TENSIX_LOCATION
        )

    # Run 0: full configure straight onto the output format.
    direct = run_and_collect_raw_bytes()
    expected_bytes = formats.output_format.num_bytes_per_tile(src_A.numel())
    assert (
        len(direct) == expected_bytes
    ), f"result has {len(direct)} bytes, {expected_bytes} expected"
    if int8_out:
        golden_tensor = (
            src_A.to(torch.float32).round().clamp(0, _UINT16_MAX).to(torch.int64)
            & _UINT8_MASK
        ).to(torch.uint8)
        assert torch.equal(
            torch.frombuffer(bytearray(direct), dtype=torch.uint8), golden_tensor
        ), (
            "UInt8 pack from a 16-bit Dest after a full configure does not keep the low byte of "
            "the Dest datum magnitude"
        )

    # Run 1: full configure for the previous format, then reconfig onto the output format.
    # Only the runtime parameter changes between the runs.
    configuration.runtimes = runtimes(1)
    reconfigured = run_and_collect_raw_bytes()
    differing = sum(a != b for a, b in zip(direct, reconfigured))
    assert reconfigured == direct, (
        f"{formats.output_format.name} pack from a 16-bit Dest after "
        "_llk_pack_reconfig_data_format_ differs from the same pack after a full configure in "
        f"{differing} of {len(direct)} bytes"
    )
