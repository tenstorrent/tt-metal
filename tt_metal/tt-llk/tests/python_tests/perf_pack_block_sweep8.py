"""Crossover sweep 8 (ad hoc): blocks of 9 to 16 tiles and the output formats sweep 2 skipped. Crossover sweep 2 (ad hoc): llk_pack_block's forms over its whole decision surface (see pack_block_sweep8_test.cpp).

Per cell (output format, tile shape, DEST width, sync, tiles per block) seven arms: state steady/init/reconfigure x form per-tile/block/W offset (steady only).
"""

from dataclasses import dataclass

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_SRC_FROM_OUTPUT,
    RELU_CONFIG,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TILIZE,
    TemplateParameter,
    generate_input_dim,
)
from helpers.pack_sweep import NEW_ARMS, SHIM_CONFIG, bind_pack_src_b, sweep_templates
from helpers.tile_constants import get_tile_params

F = DataFormat
BFP = [F.Bfp8_b, F.Bfp4_b, F.Bfp2_b, F.Bfp8, F.Bfp4, F.Bfp2]
BLOCKS = 6
ARMS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (4, 0),
    (5, 0),
    (6, 0),
    (8, 2),
] + NEW_ARMS  # pack_sweep.py: arm ids; pk-limit5: (8, 2) W under the cb0ac rule in a CB of two blocks (bit 15); (4|5|6, 0) main per tile in steady / init / reconfigure (bit 14)  # (state, form)
BYTES = {
    F.Float32: 4,
    F.Int32: 4,
    F.UInt32: 4,
    F.Tf32: 4,
    F.Float16_b: 2,
    F.Float16: 2,
    F.UInt16: 2,
    F.Int8: 1,
    F.UInt8: 1,
    F.Fp8_e4m3: 1,
    F.Lf8: 1,
}
BFP_DATA = {F.Bfp8_b: 16, F.Bfp8: 16, F.Bfp4_b: 8, F.Bfp4: 8, F.Bfp2_b: 4, F.Bfp2: 4}


def page_bytes(fmt, r, faces):
    if fmt in BFP_DATA:
        return faces * (r * BFP_DATA[fmt] + 16)
    return faces * r * 16 * BYTES[fmt]


@dataclass
class PACK_BLOCK_SWEEP2(TemplateParameter):
    """sweep_closed: the block form of the cell is the closed block (block float or a tile that is not one contiguous 32x32 page)"""

    block_sweep_closed: str

    def convert_to_cpp(self) -> str:
        a, b = self.block_sweep_closed.split("/")
        return f"constexpr bool sweep_closed = {str(a == 'closed').lower()};\nconstexpr bool sweep_b_closed = {str(b == 'closed').lower()};"


def _caps(acc, sync):
    if sync == DestSync.Half:
        return 4 if acc == DestAccumulation.Yes else 8
    return 8 if acc == DestAccumulation.Yes else 16


def _cases():
    out = []
    accs = (DestAccumulation.No, DestAccumulation.Yes)
    syncs = (DestSync.Half, DestSync.Full)

    def add(fam, fin, fout, dims, acc_list, lo=1):
        for acc in acc_list:
            for sync in syncs:
                for tiles in range(lo, _caps(acc, sync) + 1):
                    out.append((fam, fin, fout, dims, acc, sync, tiles))

    # blocks of 9 to 16 tiles (full sync, 16 bit DEST) of the formats sweep 2 measured
    for fmt in BFP:
        add("bfp", F.Float16_b, fmt, (32, 32), (DestAccumulation.No,), 9)
    add("wide", F.Float16_b, F.Float32, (32, 32), (DestAccumulation.No,), 9)
    add("f16b", F.Float16_b, F.Float16_b, (32, 32), (DestAccumulation.No,), 9)
    for dims in ((16, 32), (8, 32), (4, 32), (1, 32), (16, 16)):
        add("small", F.Float16_b, F.Float16_b, dims, (DestAccumulation.No,), 9)
    add("small", F.Float16_b, F.Float32, (16, 32), (DestAccumulation.No,), 9)
    add("small", F.Float16_b, F.Bfp8_b, (16, 32), (DestAccumulation.No,), 9)
    # output formats sweep 2 skipped, every tile count the DEST holds
    for fin, fout, acc_list in [
        (F.Float16, F.Float16, accs),
        (F.Float16_b, F.UInt16, accs),
        (F.Float16_b, F.Int8, accs),
        (F.Float16_b, F.UInt8, accs),
        (F.Float16_b, F.Fp8_e4m3, accs),
        (F.Float16_b, F.Lf8, accs),
        (F.Float16_b, F.Tf32, accs),
        (F.Float32, F.Float32, (DestAccumulation.Yes,)),
        (F.Float32, F.Int8, (DestAccumulation.Yes,)),
        (F.Int32, F.Int32, (DestAccumulation.Yes,)),
        (F.UInt32, F.UInt32, (DestAccumulation.Yes,)),
    ]:
        add("fmt", fin, fout, (32, 32), acc_list)
    return out


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    case=_cases(),
    arm=ARMS,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_block_sweep8(
    perf_report, case, arm, run_types, loop_factor, is_perf
):
    fam, fin, fout, dims_t, dest_acc, dest_sync, block_tiles = case
    formats = InputOutputFormat(fin, fout)
    face_r, nfr, nfc = get_tile_params(dims_t)
    faces = nfr * nfc
    tile_cnt = BLOCKS * block_tiles
    dims = (
        [dims_t[0] * tile_cnt, dims_t[1]] if dims_t != (32, 32) else [32, 32 * tile_cnt]
    )
    tdim = list(dims_t)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=fin,
        input_dimensions_A=dims,
        stimuli_format_B=fin,
        input_dimensions_B=dims,
        tile_dimensions=tdim,
    )
    contiguous = fout not in BFP_DATA and faces == 4 and face_r == 16
    pa, pb = page_bytes(fout, face_r, faces), page_bytes(F.Float16_b, face_r, faces)
    opsize = max(1024, ((max(pa, pb) + 1023) // 1024) * 1024)
    stim_kw = (
        dict(
            num_faces=faces,
            face_r_dim=face_r,
            tile_dimensions=tdim,
            use_dense_tile_dimensions=True,
        )
        if dims_t != (32, 32)
        else {}
    )
    kwargs = {
        "test_name": "sources/pack_block_sweep8_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(dims, dims, tile_dimensions=dims_t),
            TILIZE(),
            DEST_SYNC(dest_sync),
            PACK_BLOCK_SWEEP2(
                ("plain" if contiguous else "closed")
                + "/"
                + ("plain" if (faces == 4 and face_r == 16) else "closed")
            ),
            PACK_SRC_FROM_OUTPUT(),
            *sweep_templates(arm, block_tiles, faces, face_r, fidelity="LoFi"),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX((arm[0] << 12) | (arm[1] << 8)),
            RELU_CONFIG(0),
            NUM_FACES(num_faces=faces),
            TEST_FACE_DIMS(face_r),
            IN_TILE_DIMS(dims_t[0], dims_t[1]),
            NUM_BLOCKS(BLOCKS),
            NUM_TILES_IN_BLOCK(block_tiles),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            fin,
            src_B,
            fin,
            fout,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            operand_res_tile_size=opsize,
            **stim_kw,
        ),
        "dest_acc": dest_acc,
        **SHIM_CONFIG,
        "unpack_to_dest": fin.is_32_bit() and dest_acc == DestAccumulation.Yes,
    }
    create_test_or_perf_config(
        is_perf=True, run_types=run_types, test_config_kwargs=bind_pack_src_b(kwargs)
    ).run(perf_report)
