"""Context sweep (pk-limit3, ad hoc): llk_pack_block's forms behind the other unpack/math contexts of the Blackhole LLK API (see pack_ctx_test.cpp).

Contexts: eltwise add, eltwise mul (LoFi, HiFi4), unpack tilize, reduce row (LoFi, HiFi4), reduce col (HiFi4), unpack transpose, datacopy + unary SFPU (abs).
Per cell (context, output class, input, DEST width, sync, tiles per block) the nine arms of sweep 8 (state steady/init/reconfigure x form per-tile/block/W offset) and (0, 3), main's per-tile pack.
test_pack_ctx_equiv runs the same kernel once (L1_TO_L1) and compares the L1 bytes of every arm with the per-tile arm of its state.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
from helpers.pack_sweep import NEW_ARMS, SHIM_CONFIG, bind_pack_src_b, sweep_templates
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
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
from helpers.tile_constants import get_tile_params
from ttexalens.tt_exalens_lib import read_from_device, write_to_device

F = DataFormat
BLOCKS = 6
ARMS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (5, 0),
    (6, 0),
    (8, 2),
] + NEW_ARMS  # pack_sweep.py: arm ids; pk-limit5: (8, 2) W under the cb0ac rule in a CB of two blocks (bit 15); (5, 0), (6, 0) main per tile in init, reconfigure (DST_INDEX bit 14)  # (state, form); (0, 3) main per-tile pack (no byte-wise destination write)
BYTES = {F.Float32: 4, F.Float16_b: 2, F.Float16: 2}
BFP_DATA = {F.Bfp8_b: 16, F.Bfp4_b: 8, F.Bfp4: 8}
CTX = {
    "add": (0, None),
    "mul": (1, None),
    "tilize": (2, None),
    "rrow": (3, "REDUCE_ROW"),
    "trans": (4, None),
    "sfpu": (5, None),
    "rcol": (6, "REDUCE_COL"),
}
MEAS = [
    ("add", MathFidelity.LoFi),
    ("mul", MathFidelity.LoFi),
    ("mul", MathFidelity.HiFi4),
    ("sfpu", MathFidelity.LoFi),
    ("tilize", MathFidelity.LoFi),
    ("trans", MathFidelity.LoFi),
    ("rrow", MathFidelity.LoFi),
    ("rrow", MathFidelity.HiFi4),
    ("rcol", MathFidelity.HiFi4),
]
TINY_OK = {"add", "mul", "sfpu"}
CLASSES = [
    (F.Float16_b, (16, 16)),
    (F.Float16_b, (16, 32)),
    (F.Float16_b, (8, 32)),
    (F.Float16_b, (4, 32)),
    (F.Float16_b, (1, 32)),
    (F.Bfp8_b, (32, 32)),
    (F.Bfp4_b, (32, 32)),
    (F.Bfp4, (32, 32)),
    (F.Float32, (32, 32)),
    (F.Float16_b, (32, 32)),
]


def page_bytes(fmt, r, faces):
    if fmt in BFP_DATA:
        return faces * (r * BFP_DATA[fmt] + 16)
    return faces * r * 16 * BYTES[fmt]


@dataclass
class PACK_CTX(TemplateParameter):
    """context/closed A/closed B (the pack_block_sweep column)"""

    pack_block_sweep: str

    def convert_to_cpp(self) -> str:
        c, a, b = self.pack_block_sweep.split("/")
        cid, rdim = CTX[c]
        return "\n".join(
            [
                f"constexpr int sweep_ctx = {cid};",
                f"constexpr auto REDUCE_DIM = ckernel::ReduceDim::{rdim or 'REDUCE_ROW'};",
                "constexpr auto POOL_TYPE = ckernel::PoolType::SUM;",
                f"constexpr bool sweep_closed = {str(a == 'closed').lower()};",
                f"constexpr bool sweep_b_closed = {str(b == 'closed').lower()};",
            ]
        )


def _caps(acc, sync):
    if sync == DestSync.Half:
        return 4 if acc == DestAccumulation.Yes else 8
    return 8 if acc == DestAccumulation.Yes else 16


def _cases():
    out = []
    for ctx, fid in MEAS:
        for fout, dims in CLASSES:
            if dims != (32, 32) and ctx not in TINY_OK:
                continue
            for fin, acc in (
                (F.Float16_b, DestAccumulation.No),
                (F.Float16_b, DestAccumulation.Yes),
                (F.Float32, DestAccumulation.Yes),
            ):
                for sync in (DestSync.Half, DestSync.Full):
                    for tiles in range(1, _caps(acc, sync) + 1):
                        out.append((ctx, fid, fin, fout, dims, acc, sync, tiles))
    return out


def _kwargs(case, arm, loop_factor):
    ctx, fid, fin, fout, dims_t, dest_acc, dest_sync, block_tiles = case
    state, form = arm
    formats = InputOutputFormat(fin, fout)
    face_r, nfr, nfc = get_tile_params(dims_t)
    faces = nfr * nfc
    tile_cnt = BLOCKS * block_tiles
    dims = (
        [dims_t[0] * tile_cnt, dims_t[1]] if dims_t != (32, 32) else [32, 32 * tile_cnt]
    )
    tdim = list(dims_t)
    torch.manual_seed(1234)
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
    kind = (
        ctx
        + "/"
        + ("plain" if contiguous else "closed")
        + "/"
        + ("plain" if (faces == 4 and face_r == 16) else "closed")
    )
    kwargs = {
        "test_name": "sources/pack_ctx_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(dims, dims, tile_dimensions=dims_t),
            TILIZE(),
            DEST_SYNC(dest_sync),
            MATH_FIDELITY(fid),
            PACK_CTX(kind),
            PACK_SRC_FROM_OUTPUT(),
            *sweep_templates(arm, block_tiles, faces, face_r),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX((state << 12) | (form << 8)),
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
        "unpack_to_dest": ctx == "sfpu"
        and fin.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
    }
    return bind_pack_src_b(kwargs), (pa, pb, block_tiles, state)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    case=_cases(),
    arm=ARMS,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_ctx(perf_report, case, arm, run_types, loop_factor, is_perf):
    kwargs, _ = _kwargs(case, arm, loop_factor)
    create_test_or_perf_config(
        is_perf=True, run_types=run_types, test_config_kwargs=kwargs
    ).run(perf_report)


_EQ_REF = {}


@skip_for_wormhole
@parametrize(case=_cases(), arm=ARMS)
def test_pack_ctx_equiv(case, arm):
    kwargs, (pa, pb, n, state) = _kwargs(case, arm, 1)
    cfg = create_test_or_perf_config(
        is_perf=False, run_types=[PerfRunType.L1_TO_L1], test_config_kwargs=kwargs
    )
    base = cfg.variant_stimuli.buf_res_addr
    off_b = ((((BLOCKS + 1) // 2) * n * pa) + 1023) & ~1023
    regions = (
        [(base, BLOCKS * n * pa)]
        if (state & 3) != 2
        else [
            (base, ((BLOCKS + 1) // 2) * n * pa),
            (base + off_b, (BLOCKS // 2) * n * pb),
        ]
    )
    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        for a, sz in regions:
            write_to_device(TestConfig.TENSIX_LOCATION, a, [0xA5] * sz)
    cfg.run()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    raw = b"".join(
        bytes(read_from_device(TestConfig.TENSIX_LOCATION, a, num_bytes=sz))
        for a, sz in regions
    )
    key = (case, state)
    if arm[1] == 0:
        _EQ_REF[key] = raw
        return
    ref = _EQ_REF.get(key) if arm != (0, 3) else _EQ_REF.get((case, 0))
    if ref is None:
        pytest.skip("per-tile arm of this state did not run first")
    diff = [i for i in range(len(ref)) if ref[i] != raw[i]]
    assert (
        not diff
    ), f"arm {arm} differs from the per-tile arm in {len(diff)} of {len(ref)} bytes, first at {diff[0]}: ref {ref[diff[0]:diff[0]+16].hex()} arm {raw[diff[0]:diff[0]+16].hex()}"
