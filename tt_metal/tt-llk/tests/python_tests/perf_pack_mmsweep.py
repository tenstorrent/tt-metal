"""pk-ctx matmul + SFPU tail sweep on pack_mmsweep_test.cpp (pure matmul mmK, math-thread tails xm<op>K, pack-thread tails xp<op>K): llk_pack_block's forms behind matmul math, the pack side of pack_ctx_test.cpp.

Per cell (context, input, output, DEST width, sync, tiles per block) the arms of the states the cell's mask groups take W or the block in (CELLS), and pk-limit5's
main per tile after an init or a reconfigure and W in a CB of two blocks. farm_pairs() is the farm module's inline list, case for case.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
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
]  # (state, form); (0, 3) main per-tile pack (no byte-wise destination write)
BYTES = {F.Float32: 4, F.Float16_b: 2, F.Float16: 2}
BFP_DATA = {F.Bfp8_b: 16, F.Bfp4_b: 8, F.Bfp4: 8}
CTX = {"mm": 7, "xm": 12, "xp": 13}
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
        kt = int(c[-1])
        c = c[:-1]
        kind, op = (c[:2], c[2:]) if c.startswith("x") else ("mm", "abs")
        return "\n".join(
            [
                f"constexpr int mm_ctx = {CTX[kind]};",
                f"constexpr std::uint32_t mm_kt = {kt};",
                "constexpr bool mm_prefix = false;",
                f"#define PK_TAIL_OP {op}",
                "constexpr auto REDUCE_DIM = ckernel::ReduceDim::REDUCE_ROW;",
                "constexpr auto POOL_TYPE = ckernel::PoolType::SUM;",
                f"constexpr bool sweep_closed = {str(a == 'closed').lower()};",
                f"constexpr bool sweep_b_closed = {str(b == 'closed').lower()};",
            ]
        )


def _caps(acc, sync):
    if sync == DestSync.Half:
        return 4 if acc == DestAccumulation.Yes else 8
    return 8 if acc == DestAccumulation.Yes else 16


# The matmul cells some mask group packs as W or the block (pk-ctx sweep_cells.csv, gen_sweep.py): input, output, fp32 DEST, sync,
# then per tile count 1 to 16 the states measured as bits: 1 steady (per tile, block, W, main per tile), 2 init (per tile,
# block), 4 reconfigure (per tile, A block). The cases run cell by cell, every context of CTXS, the arms in order.
CELLS = [
    ("Float16_b", "Bfp4", "No", "Full", "0777777777777777"),
    ("Float16_b", "Bfp4", "No", "Half", "0577777700000000"),
    ("Float16_b", "Bfp4", "Yes", "Full", "0575755500000000"),
    ("Float16_b", "Bfp4", "Yes", "Half", "0577000000000000"),
    ("Float16_b", "Bfp4_b", "No", "Full", "0577777777777777"),
    ("Float16_b", "Bfp4_b", "No", "Half", "0577777700000000"),
    ("Float16_b", "Bfp4_b", "Yes", "Full", "0577777700000000"),
    ("Float16_b", "Bfp4_b", "Yes", "Half", "0577000000000000"),
    ("Float16_b", "Bfp8_b", "No", "Full", "0577777777777777"),
    ("Float16_b", "Bfp8_b", "No", "Half", "0577777700000000"),
    ("Float16_b", "Bfp8_b", "Yes", "Full", "0577777700000000"),
    ("Float16_b", "Bfp8_b", "Yes", "Half", "0577000000000000"),
    ("Float16_b", "Float16", "No", "Full", "7777777777777777"),
    ("Float16_b", "Float16", "No", "Half", "5777777700000000"),
    ("Float16_b", "Float16", "Yes", "Full", "5777777700000000"),
    ("Float16_b", "Float16", "Yes", "Half", "4777000000000000"),
    ("Float16_b", "Float16_b", "No", "Full", "7777777777777777"),
    ("Float16_b", "Float16_b", "No", "Half", "5777777700000000"),
    ("Float16_b", "Float16_b", "Yes", "Full", "5777777700000000"),
    ("Float16_b", "Float16_b", "Yes", "Half", "5777000000000000"),
    ("Float16_b", "Float32", "No", "Full", "7777777777777777"),
    ("Float16_b", "Float32", "No", "Half", "5777777700000000"),
    ("Float32", "Bfp4", "Yes", "Full", "0575755500000000"),
    ("Float32", "Bfp4", "Yes", "Half", "0577000000000000"),
    ("Float32", "Bfp4_b", "Yes", "Full", "0577777700000000"),
    ("Float32", "Bfp4_b", "Yes", "Half", "0577000000000000"),
    ("Float32", "Bfp8_b", "Yes", "Full", "0577777700000000"),
    ("Float32", "Bfp8_b", "Yes", "Half", "0571000000000000"),
    ("Float32", "Float16", "Yes", "Full", "5777777700000000"),
    ("Float32", "Float16", "Yes", "Half", "4777000000000000"),
    ("Float32", "Float16_b", "Yes", "Full", "5777777700000000"),
    ("Float32", "Float16_b", "Yes", "Half", "5777000000000000"),
    ("Float32", "Float32", "Yes", "Full", "5777777700000000"),
    ("Float32", "Float32", "Yes", "Half", "5777000000000000"),
]
CTXS = [("mm1", "LoFi"), ("mm4", "LoFi"), ("mm1", "HiFi2")]
CTXS += [
    (f"xm{op}1", f) for op in ("relu_max", "gelu", "silu") for f in ("LoFi", "HiFi2")
]
CTXS += [
    (f"xp{op}1", f)
    for op in ("abs", "relu_max", "gelu", "silu")
    for f in ("LoFi", "HiFi2")
]
STATE_ARMS = {
    0: [(0, 0), (0, 1), (0, 2), (0, 3)],
    1: [(1, 0), (1, 1)],
    2: [(2, 0), (2, 1)],
}


def farm_pairs():
    """The (case, arm) list of the pk-ctx farm module, in its order."""
    out = []
    for fin, fout, acc, sync, states in CELLS:
        for n, m in enumerate(states, 1):
            arms = [a for st in range(3) if (int(m) >> st) & 1 for a in STATE_ARMS[st]]
            for c, f in CTXS:
                case = (
                    c,
                    getattr(MathFidelity, f),
                    getattr(F, fin),
                    getattr(F, fout),
                    (32, 32),
                    getattr(DestAccumulation, acc),
                    getattr(DestSync, sync),
                    n,
                )
                out += [(case, a) for a in arms]
    return out


PAIRS = farm_pairs()


# pk-limit5: main per tile in the init / reconfigure state (DST_INDEX bit 14) for every case with a per-tile arm there
PAIRS += [(c, (a[0] + 4, 0)) for c, a in list(PAIRS) if a in ((1, 0), (2, 0))]
PAIRS += [
    (c, (8, 2)) for c, a in list(PAIRS) if a == (0, 2)
]  # pk-limit5: W under the cb0ac rule in a CB of two blocks (bit 15)


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
        "test_name": "sources/pack_mmsweep_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(dims, dims, tile_dimensions=dims_t),
            TILIZE(),
            DEST_SYNC(dest_sync),
            MATH_FIDELITY(fid),
            PACK_CTX(kind),
            PACK_SRC_FROM_OUTPUT(),
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
        "unpack_to_dest": False,
    }
    return kwargs, (pa, pb, block_tiles, state)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(ca=PAIRS, run_types=[ALL_PERF_RUN_TYPES], loop_factor=[32], is_perf=[True])
def test_perf_pack_mmsweep(perf_report, ca, run_types, loop_factor, is_perf):
    case, arm = ca[0] if len(ca) == 1 else ca
    kwargs, _ = _kwargs(case, arm, loop_factor)
    create_test_or_perf_config(
        is_perf=True, run_types=run_types, test_config_kwargs=kwargs
    ).run(perf_report)
