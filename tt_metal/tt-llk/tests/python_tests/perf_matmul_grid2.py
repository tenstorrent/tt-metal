# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul grid 2: what perf_matmul_grid leaves out of everything the ttnn matmul kernels reach on Blackhole.
Unpack walk: every out_subblock h x w of the DEST capacity at inner k blocks of 3, 8 and 32 tiles whose in0 walk (h-1)*k tiles of the SrcA format and w tiles of the SrcB format stay in the L1 below the
perf counters (MATMUL_WIDE_WALK reads on past the 16 tile ring; the data does not matter), the shapes the 16 tile ring of perf_matmul_grid cut. Several inner blocks: nb 2 and 4 with packer L1 accumulate
(MATMUL_NB: block 0 of each nb writes, the rest accumulate), the several-block keys on, one shape and k class per cell and nb. Tiny tiles: in0 height 1 to 16 with in1 width 32, in0 height 1 to 16 and 32
with in1 width 16, over every SrcA x SrcB x output of Float16_b, Bfp8_b, Bfp4_b and Float32 (Bfp4_b from height 4), one block per cell and tile shape, on the math matmul test with its own operand sizes.
SrcA is input_format (unpacker 0, matmul in1), SrcB is input_format_B (unpacker 1, matmul in0). Shapes drawn by a fixed hash of the cell name so the ids stay stable."""

from typing import Callable, List, Union

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import (
    DEST_RT_CT_BLOCKS,
    PERF_RING_TILES,
    generate_tile_dims,
    matmul_pack_form,
    mid_fill_rt_ct_pairs,
    unpack_matmul_fits_perf_ring,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_PACK_BLOCK,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    MATMUL_NB,
    MATMUL_OPERAND_SIZES,
    MATMUL_WIDE_WALK,
    IN_TILE_DIMS,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    DEST_INDEX,
    UNPACK_TRANS_WITHIN_FACE,
    MATMUL_IN1_ALT,
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.llk_params import format_tile_sizes

import json, os
from dataclasses import replace

from conftest import skip_for_wormhole
from helpers.format_config import InputOutputFormat
from helpers.llk_params import L1Accumulation
from helpers.matmul_sweep import FaceLayoutConfig, MatmulConfig
from helpers.llk_params import StochasticRounding
from helpers.test_variant_parameters import MATMUL_L1_ACC, PACK_SRC_FROM_OUTPUT
import perf_matmul_grid as _g

_F = _g._F
_FMTS = _g._FMTS
_FIDS = _g._FIDS
_CAP = _g._CAP
_hash = _g._hash
# L1 below the perf counters that the in0 walk may read (from PERF_INPUT_A) and the in1 walk (from PERF_INPUT_B); tests/helpers/include/perf.h
_WALK_A = 0x169000 - 0x21000
_WALK_B = 0x169000 - 0x31000
_KT_WIDE = (3, 8, 32)
_NB_KT = (1, 3, 8)
_TINY_SHAPES = [(r, 32) for r in (1, 2, 4, 8, 16)] + [(r, 16) for r in (1, 2, 4, 8, 16, 32)]
# (rt, ct, kt) of the tiny block; "full" is the one row of ct = the DEST capacity
_TINY_BLOCKS = ((1, 1, 1), (1, 4, 1), (1, 4, 4), (2, 2, 4), (1, 0, 1))


def _size(fmt):
    return format_tile_sizes[_F[fmt]]


def _fits_walk(a, b, rt, ct, kt):
    """In0 is strided by the SrcA format's tile size (matmul_test.cpp MATMUL_TILE_SIZE_IN0), in1 by the SrcB format's."""
    return ((rt - 1) * kt + 1) * _size(a) <= _WALK_A and ct * _size(b) <= _WALK_B


def _shapes(cap, kt, a, b):
    return [(h, w) for h in range(1, cap + 1) for w in range(1, cap + 1) if h * w <= cap and _fits_walk(a, b, h, w, kt)]


def _cells():
    for a in _FMTS:
        for b in _FMTS:
            for o in _FMTS:
                for acc in "NY":
                    for sy in "HF":
                        for fi in _FIDS:
                            yield (a, b, o, acc, sy, fi)


def _grid2():
    """Rows (cell, rt, ct, kt, throttle, l1_acc, nb)."""
    excl = set(json.load(open(os.path.join(os.path.dirname(__file__), "grid_in_sweep.json"))))
    have = set(",".join(map(str, r)) for r in _g._grid())
    rows = []
    for cell in _cells():
        a, b, o, acc, sy, fi = cell
        tag = "".join(cell)
        cap = _CAP[(sy, acc)]
        for kt in _KT_WIDE:
            rows += [(*cell, h, w, kt, 0, 0, 1) for h, w in _shapes(cap, kt, a, b)]
        for nb in (2, 4):
            kt = _NB_KT[_hash(tag + "B%d" % nb) % 3]
            sh = _shapes(cap, kt, a, b)
            rows.append((*cell, *sh[_hash(tag + "b%d" % nb) % len(sh)], kt, 0, 1, nb))
    rows = [r for r in rows if ",".join(map(str, r[:11])) not in excl and (r[11] > 1 or ",".join(map(str, r[:11])) not in have)]
    assert len(set(rows)) == len(rows)
    return rows


def _cases():
    out = []
    for a, b, o, acc, sy, fi, rt, ct, kt, thr, l1, nb in _grid2():
        fmt = InputOutputFormat(_F[a], _F[o], input_format_B=_F[b])
        out.append(
            (
                fmt,
                DestAccumulation.No if acc == "N" else DestAccumulation.Yes,
                DestSync.Half if sy == "H" else DestSync.Full,
                ([rt * TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, ct * TILE_DIM]),
                getattr(MathFidelity, fi),
                thr,
                L1Accumulation.Yes if l1 else L1Accumulation.No,
                nb,
            )
        )
    return out


def _tiny_rows():
    """Rows (cell, in0 height, in1 width, rt, ct, kt): one block per cell and tile shape; Bfp4_b takes face height 4 and up."""
    rows = []
    for cell in _cells():
        a, b, o, acc, sy, fi = cell
        tag = "".join(cell)
        cap = _CAP[(sy, acc)]
        for r0, c1 in _TINY_SHAPES:
            if "B4" in (a, b, o) and r0 < 4:
                continue
            if acc == "N" and is_dest_acc_needed(InputOutputFormat(_F[a], _F[o], input_format_B=_F[b])):
                continue
            blk = _TINY_BLOCKS[_hash(tag + "t%dx%d" % (r0, c1)) % len(_TINY_BLOCKS)]
            rt, ct, kt = blk[0], blk[1] or cap, blk[2]
            rows.append((*cell, r0, c1, rt, ct, kt))
    return rows


def tiny_config(a, b, o, acc, sy, r0, c1, rt, ct, kt):
    """MatmulConfig of a tiny tile block (the tile shapes of perf_math_matmul_cov): in0 r0 x 32 with 2 faces (4 at r0 = 32), in1 32 x c1 with 4 faces (2 at c1 = 16)."""
    fmt = InputOutputFormat(_F[a], _F[o], input_format_B=_F[b])
    td = _g.generate_tile_dims(([rt * 32, kt * 32], [kt * 32, ct * 32]), in0_tile_r_dim=r0)
    td = replace(td, in0_tile_c_dim=32, in1_tile_r_dim=32, in1_tile_c_dim=c1)
    f0 = 4 if r0 == 32 else 2
    f1 = 2 if c1 == 16 else 4
    fo = 4 if (r0, c1) == (32, 32) else (1 if (r0 <= 16 and c1 == 16) else 2)
    p0 = r0 < 32 and not (c1 == 16 and r0 == 16)
    fl = FaceLayoutConfig(
        unpack_transpose_faces=Transpose.No,
        unpack_transpose_within_face=Transpose.No,
        num_faces_in0=f0,
        num_faces_in1=f1,
        num_faces=fo,
        partial_face_in0=p0,
        partial_face_in1=False,
        partial_face_math=r0 < 16,
        partial_face_pack=r0 < 16,
    )
    return MatmulConfig(
        tile_dimensions=td,
        face_layout_config=fl,
        formats=fmt,
        stochastic_rnd=StochasticRounding.No,
        dst_index=0,
        dest_sync=DestSync.Half if sy == "H" else DestSync.Full,
        dest_acc=DestAccumulation.No if acc == "N" else DestAccumulation.Yes,
    )


def _tiny_cases():
    return [(a, b, o, acc, sy, getattr(MathFidelity, fi), r0, c1, rt, ct, kt) for a, b, o, acc, sy, fi, r0, c1, rt, ct, kt in _tiny_rows()]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(case=_cases())
def test_perf_matmul_grid2(
    perf_report,
    case,
):

    case = case[0] if len(case) == 1 else case
    formats, dest_acc, dest_sync, (matrix_a, matrix_b), math_fidelity, throttle, l1_acc, num_blocks = case

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    # the MOP walk over in0 leaves the 16 tile ring: it reads on through the L1 below the perf counters
    wide_walk = (dims.rt_dim - 1) * dims.kt_dim >= PERF_RING_TILES or dims.ct_dim - 1 >= PERF_RING_TILES
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in generate_dest_corner_combinations.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)


    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PACK_SRC_FROM_OUTPUT(),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, dest_acc, throttle),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(throttle),
            *MATMUL_STREAM_NARROW.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(formats, math_fidelity, dims.ct_dim, dims.rt_dim, dest_sync),
            *MATMUL_STREAM_Z.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
                kt_dim=dims.kt_dim,
                throttle=throttle,
            ),
            *MATMUL_UNPACK_TTSYNC.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim, dest_sync
            ),
            *MATMUL_K_LOOP.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim, throttle=throttle
            ),
            *MATMUL_ROW_END_FOLD.for_block(dims.ct_dim, dims.rt_dim),
            *MATMUL_ROW_MOP.for_block(
                formats, dest_acc, math_fidelity, dest_sync, dims.ct_dim, dims.rt_dim, dims.kt_dim, num_blocks, num_blocks > 1
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats, dims.kt_dim, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc, unfuse_one_tile=True, throttle=throttle,
                sections=num_blocks, several_blocks=num_blocks > 1,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    dims.rt_dim,
                    dims.ct_dim,
                    dims.kt_dim,
                    throttle=throttle,
                    l1_acc=l1_acc == L1Accumulation.Yes,
                )
            ),
            MATMUL_L1_ACC(l1_acc),
            MATMUL_NB(num_blocks),
            *([MATMUL_WIDE_WALK()] if wide_walk else []),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format_B,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
