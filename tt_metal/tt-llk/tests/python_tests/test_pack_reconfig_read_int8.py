# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Packer data-format reconfig from an 8-bit integer format with a 16-bit Dest.

Two runs of one kernel pack the same Dest contents. Run 0 configures the packer directly for
the Float16_b output. Run 1 configures UInt8 first (the format a preceding op would have left
behind) and reconfigures to Float16_b with ``_llk_pack_reconfig_data_format_``. With a 16-bit
Dest the packer's Dest-read control selects a raw read for the integer formats and a rounding
read for the float formats; the rounding read flushes denormals and turns NaN into infinity, so
a reconfig that leaves the raw read behind packs different bytes. The gate is on the raw result
bytes in L1. The math thread keeps the Src zero flag disabled for the copy, as copy_tile_init
does, so the bf16 denormals in the stimuli reach Dest instead of being flushed by the datacopy.
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
)

# 16-bit Dest is the point of the test: the 8-bit integer Dest read only exists there.
_DEST_ACC = DestAccumulation.No
# One tile: the defect does not depend on the tile count.
_INPUT_DIMENSIONS = [32, 32]

# bf16 keeps 8 significant bits: whole numbers below this are exact in bf16.
_BF16_EXACT_WHOLE_LIMIT = 1 << 8
# bf16 is the top 16 bits of fp32; its mantissa is the top 7 fraction bits.
_FP32_TO_BF16_SHIFT = 16
_BF16_MANTISSA_BITS = 7
# Stimulus kinds, drawn uniformly per datum.
_KIND_NORMAL = 0
_KIND_DENORMAL = 1
_KIND_NAN = 2
_NUM_KINDS = 3


def _normals_denormals_nan(size, dtype, generator):
    # Exact whole numbers, bf16 denormals (zero exponent, non-zero top-7 mantissa) and NaN.
    values = torch.randint(1, _BF16_EXACT_WHOLE_LIMIT, (size,), generator=generator).to(
        torch.float32
    )
    kind = torch.randint(_KIND_NORMAL, _NUM_KINDS, (size,), generator=generator)
    denormal_bits = (
        torch.randint(
            1, 1 << _BF16_MANTISSA_BITS, (size,), generator=generator, dtype=torch.int32
        )
        << _FP32_TO_BF16_SHIFT
    )
    values[kind == _KIND_DENORMAL] = denormal_bits.view(torch.float32)[
        kind == _KIND_DENORMAL
    ]
    values[kind == _KIND_NAN] = float("nan")
    return values.to(dtype)


_BF16_OUT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


@parametrize(formats=[_BF16_OUT])
def test_pack_reconfig_read_int8(formats):
    # A single parametrize axis arrives as a 1-tuple.
    (formats,) = formats
    input_dimensions = _INPUT_DIMENSIONS
    spec = StimuliSpec(distribution=_normals_denormals_nan, seed=0)
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
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            CONFIGURE_TEST_RUN_IDX(run_idx),
        ]

    configuration = TestConfig(
        "sources/pack_reconfig_read_int8_test.cpp",
        formats,
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

    def run_and_collect_raw_bytes():
        # The gate is on the bytes the packer wrote to L1; the decoded result canonicalizes NaN.
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

    # Run 1: full configure for UInt8, then reconfig onto the output format. The result buffer
    # is cleared first so a run 1 that packs less than run 0 cannot pass on run 0's bytes.
    configuration.runtimes = runtimes(1)
    configuration.variant_stimuli.clear_result_buffer(TestConfig.TENSIX_LOCATION)
    reconfigured = run_and_collect_raw_bytes()
    differing = sum(a != b for a, b in zip(direct, reconfigured))
    assert reconfigured == direct, (
        f"{formats.output_format.name} pack from a 16-bit Dest after "
        "_llk_pack_reconfig_data_format_ differs from the same pack after a full configure in "
        f"{differing} of {len(direct)} bytes"
    )
