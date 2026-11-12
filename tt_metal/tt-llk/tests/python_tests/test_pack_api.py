# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PackerReluType
from helpers.param_config import parametrize
from test_pack import test_pack as run_pack

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
        api_arm=api_arm,
    )


# Configurations paired with production ELFs instruction by instruction: bmm metal2 with pack_block<4> on its out DFB (2),
# in a kernel with the matmul and pack-thread SFPU API headers.
PACK_API_PRODUCTION = {
    "bmm_full_f16b_2x2": dict(
        dest_sync=DestSync.Full,
        block_tiles=4,
        opts=dict(out=2, matmul=True, sfpu=True, dst=0),
    ),
    "bmm_half_f16b_2x2": dict(
        dest_sync=DestSync.Half,
        block_tiles=4,
        opts=dict(out=2, matmul=True, sfpu=True, dst=0),
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
        api_arm=api_arm,
        api_opts=c["opts"],
    )
