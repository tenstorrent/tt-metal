# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import FACES_PER_TILE
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PackerReluType,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_API,
    PACK_SRC_FROM_OUTPUT,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from test_pack import test_pack as run_pack


def pack_api_config(
    api_arm, input_dimensions, dest_sync, api_opts=None, is_perf=False, run_types=None
):
    """test_pack's stimuli and golden on pack_api_test.cpp: the compute API's pack_block (pack_api_shim.h), compiled as
    production compiles it."""

    def make(test_config_kwargs, v):
        templates = [
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(),
            DEST_SYNC(dest_sync),
            PACK_API(api_arm, v["num_tiles_in_block"], **(api_opts or {})),
            PACK_SRC_FROM_OUTPUT(),
            MATH_FIDELITY(MathFidelity.LoFi),
        ]
        runtimes = [
            TILE_COUNT(v["tile_cnt"]),
            DEST_INDEX(v["dest_index"]),
            RELU_CONFIG(v["relu_config"]),
            NUM_FACES(num_faces=FACES_PER_TILE),
            NUM_BLOCKS(v["num_blocks"]),
            NUM_TILES_IN_BLOCK(v["num_tiles_in_block"]),
            LOOP_FACTOR(v["loop_factor"]),
        ]
        kwargs = dict(test_config_kwargs)
        kwargs["test_name"] = "sources/pack_api_test.cpp"
        kwargs["templates"] = templates
        kwargs["runtimes"] = runtimes
        kwargs["compile_time_formats"] = True
        kwargs["kernel_flags"] = ["-flto=1", "-flto-partition=one"]
        return create_test_or_perf_config(
            is_perf=is_perf,
            run_types=run_types or [],
            test_config_kwargs=kwargs,
        )

    return make


# The compute API's pack_block through pack_api_shim.h: the form tables (0), one form (T, B, W), main's per-tile loop (M).
PACK_API_SWEEP = dict(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[128, 64]],
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_index=[0],
    block_tiles=[2, 4],
    api_arm=["0", "T", "B", "W", "M"],
)


@skip_for_wormhole
@parametrize(**PACK_API_SWEEP)
def test_pack_api(
    formats,
    dest_acc,
    input_dimensions,
    relu_type,
    dest_sync,
    dest_index,
    block_tiles,
    api_arm,
):
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        block_tiles=block_tiles,
        make_config=pack_api_config(api_arm, input_dimensions, dest_sync),
    )


# Configurations paired with production ELFs instruction by instruction: bmm metal2 with pack_block<4> on its out DFB (2),
# in a kernel with the matmul and pack-thread SFPU API headers.
PACK_API_PRODUCTION = {
    "bmm_full_f16b_2x2": dict(
        dest_sync=DestSync.Full,
        block_tiles=4,
        opts=dict(api_out=2, api_matmul=True, api_sfpu=True, api_dst=0),
    ),
    "partial_face_compile": dict(
        dest_sync=DestSync.Half,
        block_tiles=4,
        opts=dict(
            api_out=2, api_matmul=True, api_sfpu=True, api_dst=0, api_partial_face=True
        ),
    ),
    "bmm_half_f16b_2x2": dict(
        dest_sync=DestSync.Half,
        block_tiles=4,
        opts=dict(api_out=2, api_matmul=True, api_sfpu=True, api_dst=0),
    ),
}


@skip_for_wormhole
@parametrize(cell=list(PACK_API_PRODUCTION), api_arm=["0", "M"])
def test_pack_api_production(cell, api_arm):
    c = PACK_API_PRODUCTION[cell]
    run_pack(
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        [128, 64],
        PackerReluType.NoRelu,
        c["dest_sync"],
        0,
        block_tiles=c["block_tiles"],
        make_config=pack_api_config(
            api_arm, [128, 64], c["dest_sync"], api_opts=c["opts"]
        ),
    )
