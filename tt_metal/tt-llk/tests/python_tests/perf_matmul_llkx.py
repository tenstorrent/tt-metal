# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul LLK space beyond ttnn (ops/blackhole/matmul/llk_space_1029.md): what the LLK API takes on Blackhole and no ttnn matmul reaches, over the matmul format and mode cells.
New formats: Float16 and Bfp8 (5 bit exponent family), Bfp2_b, and Int8 / UInt8 with a 32 bit integer DEST, crossed over sync, dest accumulate and fidelity at the grid's k classes 1, 3 and 8 and four shapes per cell.
New modes over the grid's 1024 format cells: static throttle 2 and 3, in1 transpose, the FPU / packer / both stochastic rounding units, DEST base offsets, and the tiny and K-face tile shapes of the new formats.
Rows come in two bodies: the matmul test (32x32 tiles, packer L1 accumulate off) and the math matmul test (tile shapes and DEST offsets). Rows whose key the 34053-row sweep already runs are left out (llkx_in_sweep.json).
SrcA is input_format (unpacker 0, matmul in1), SrcB is input_format_B (unpacker 1, matmul in0). Shapes and representatives are drawn by a fixed hash of the cell name so the ids stay stable."""

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
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Cold start, functional-max inner dim, and reuse/bandwidth.
KT_DIMS = [1, 4, 32]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]


def dest_corner_mn(max_tiles: int) -> List[tuple]:
    """1×1, dest-fill vectors/square, dest occupancy blocks, and half-dest mid-fill when they fit."""
    square = int(max_tiles**0.5)
    corners = [(1, 1), (1, max_tiles), (max_tiles, 1), (square, square)]
    corners.extend((rt, ct) for rt, ct in DEST_RT_CT_BLOCKS if rt * ct <= max_tiles)
    corners.extend(mid_fill_rt_ct_pairs(max_tiles))
    return list(dict.fromkeys(corners))


def generate_dest_corner_combinations(max_tiles: int, kt_dims=KT_DIMS) -> List[tuple]:
    return [
        (
            [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
            [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        )
        for mt_dim, nt_dim in dest_corner_mn(max_tiles)
        for kt_dim in kt_dims
        if unpack_matmul_fits_perf_ring(mt_dim, nt_dim, kt_dim)
    ]


def matmul_combos(
    formats: List[FormatConfig],
    dest_acc: Union[
        List[DestAccumulation], Callable[[FormatConfig], List[DestAccumulation]]
    ],
):
    def _acc_modes(fmt: FormatConfig) -> List[DestAccumulation]:
        return dest_acc(fmt) if callable(dest_acc) else dest_acc

    def _dest_bank_max_tiles(
        format: FormatConfig, dest_acc: DestAccumulation, dest_sync: DestSync
    ):
        capacity_divisor = (
            2 if is_dest_acc_needed(format) or dest_acc == DestAccumulation.Yes else 1
        )
        return DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor

    unique_max_tiles = set(
        _dest_bank_max_tiles(fmt, acc, sync)
        for fmt in formats
        for acc in _acc_modes(fmt)
        for sync in DEST_SYNC_MODES
    )
    dimensions = {
        max_tiles: generate_dest_corner_combinations(max_tiles, kt_dims=KT_DIMS)
        for max_tiles in unique_max_tiles
    }

    return [
        (format, accumulation, dest_sync, dims)
        for format in formats
        for accumulation in _acc_modes(format)
        for dest_sync in DEST_SYNC_MODES
        for dims in dimensions[_dest_bank_max_tiles(format, accumulation, dest_sync)]
    ]



from itertools import chain, product
import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
)
from helpers.matmul_sweep import (
    DEST_RT_CT_BLOCKS,
    MatmulConfig,
    generate_face_layout_config_sweep,
    generate_tile_dims,
    matmul_pack_form,
    mid_fill_rt_ct_pairs,
    skip_matmul_combination,
    sweep_tiny_tiles_matmul,
)
from helpers.param_config import DEST_SYNC_TILE_LIMITS, input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_ROW_MOP,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_IN1_ALT,
    MATMUL_PACK_BLOCK,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from dataclasses import replace
from helpers.matmul_sweep import FaceLayoutConfig, MatmulConfig, generate_tile_dims
from helpers.llk_params import StochasticRounding

import json, os
from conftest import skip_for_wormhole
from helpers.format_config import InputOutputFormat
from helpers.llk_params import L1Accumulation
from helpers.test_variant_parameters import MATMUL_L1_ACC, PACK_SRC_FROM_OUTPUT, MATMUL_STOCH_RND

_F = {"F16b": DataFormat.Float16_b, "B8": DataFormat.Bfp8_b, "B4": DataFormat.Bfp4_b, "F32": DataFormat.Float32,
      "F16a": DataFormat.Float16, "B8a": DataFormat.Bfp8, "B2": DataFormat.Bfp2_b, "I8": DataFormat.Int8, "U8": DataFormat.UInt8, "I32": DataFormat.Int32}
_OLD = ("F16b", "B8", "B4", "F32")
_B_FAM = ("F16b", "B8", "B4", "B2", "F32")
_A_FAM = ("F16a", "B8a")
_FIDS = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
_CAP = {("H", "N"): 8, ("H", "Y"): 4, ("F", "N"): 16, ("F", "Y"): 8}
_KT = (1, 3, 8)
_SR = ("Fpu", "Pack", "All")
_SRCODE = {StochasticRounding.Fpu: 1, StochasticRounding.Pack: 2, StochasticRounding.All: 3}  # loop_factor 65, 66, 67 tells the rows apart in the csv (no column carries the mode)
_SRV = {"Fpu": StochasticRounding.Fpu, "Pack": StochasticRounding.Pack, "All": StochasticRounding.All}
_GEOM = [(r, 32) for r in (1, 2, 4, 8, 16)] + [(r, 16) for r in (1, 2, 4, 8, 16, 32)]
_FAMS1 = "NTR"
_FAMS2 = ""
_EXCL = set(json.load(open(os.path.join(os.path.dirname(__file__), "llkx_in_sweep.json"))))


def _hash(tag):
    return sum((i + 1) * ord(c) for i, c in enumerate(tag))


def _shapes(cap, kt):
    """(rt, ct) with rt*ct within cap and the last in0 tile ((rt-1)*kt) and last in1 tile (ct-1) inside the 16 tile perf ring."""
    return [(h, w) for h in range(1, cap + 1) for w in range(1, cap + 1) if h * w <= cap and (h - 1) * kt < 16 and w - 1 < 16]


def _four(cap, kt, tag):
    """four representatives of the shapes at k tile count kt: 1x1, the widest row, the tallest column, one hashed."""
    sh = _shapes(cap, kt)
    out = [(1, 1), (1, cap), max(sh, key=lambda s: (s[0], -s[1])), sh[_hash(tag + "s%d" % kt) % len(sh)]]
    return list(dict.fromkeys(out))


def _one(cap, kt, tag):
    sh = _shapes(cap, kt)
    return sh[_hash(tag + "o%d" % kt) % len(sh)]


def _new_cells():
    """(a, b, o, acc) of the new formats: A family operands, Bfp2_b operands and outputs, integer operands (32 bit DEST only), 16 bit outputs of block float and 32 bit operands."""
    cells = []
    for a in _A_FAM:
        for b in _A_FAM:
            for o in ("F16a", "B8a", "F32", "F16b"):
                cells += [(a, b, o, "N"), (a, b, o, "Y")]
    for a in _B_FAM:
        for b in _B_FAM:
            if "B2" not in (a, b):
                continue
            for o in _B_FAM:
                cells += [(a, b, o, "N"), (a, b, o, "Y")]
    for a in _OLD:
        for b in _OLD:
            cells += [(a, b, "B2", "N"), (a, b, "B2", "Y")]
    for x in _B_FAM:
        cells.append((x, x, "F16a", "Y"))
    for x in ("I8", "U8"):
        cells.append((x, x, "I32", "Y"))
    return cells


def _is_new(c):
    return c[0] in ("F16a", "B8a", "B2", "I8", "U8") or c[1] in ("F16a", "B8a", "B2", "I8", "U8") or c[2] in ("F16a", "B8a", "B2", "I32")


def _fn1_rows():
    """Rows (fam, a, b, o, acc, sy, fi, rt, ct, kt, thr, tr, sr) of the matmul test body."""
    rows = []
    for a, b, o, acc in _new_cells():
        for sy in "HF":
            for fi in _FIDS:
                cell = (a, b, o, acc, sy, fi)
                tag = "".join(cell)
                cap = _CAP[(sy, acc)]
                for kt in _KT:
                    sh = [_one(cap, kt, tag)] if "B2" in (a, b, o) and not (a in _A_FAM or o in ("F16a", "B8a")) else _four(cap, kt, tag)
                    rows += [("N", *cell, h, w, kt, 0, 0, "") for h, w in sh]
    for a in _OLD:
        for b in _OLD:
            for o in _OLD:
                for acc in "NY":
                    for sy in "HF":
                        for fi in _FIDS:
                            cell = (a, b, o, acc, sy, fi)
                            tag = "".join(cell)
                            cap = _CAP[(sy, acc)]
                            for thr in (2, 3):
                                kt = _KT[_hash(tag + "K%d" % thr) % 3]
                                rows.append(("T", *cell, *_one(cap, kt, tag + "T%d" % thr), kt, thr, 0, ""))
                            for kt in _KT:
                                rows.append(("R", *cell, *_one(cap, kt, tag + "R"), kt, 0, 1, ""))
                            for sr in _SR:
                                kt = _KT[_hash(tag + "S" + sr) % 3]
                                rows.append(("S", *cell, *_one(cap, kt, tag + "S" + sr), kt, 0, 0, sr))
    rows = [r for r in rows if r[0] in _FAMS1 and ",".join(map(str, r[1:])) not in _EXCL]
    return list(dict.fromkeys(rows))


def _cap_shapes_small(cap, tag):
    sh = [(h, w) for h in range(1, cap + 1) for w in range(1, cap + 1) if h * w <= cap // 2]
    return sh


def _fn2_rows():
    """Rows (fam, a, o, acc, sy, fi, r0, c1, rt, ct, kt, dst) of the math matmul body: DEST base offsets of the equal operand format cells, tile shapes of the new equal-format cells."""
    rows = []
    for a in _OLD:
        for o in _OLD:
            for acc in "NY":
                for sy in "HF":
                    for fi in _FIDS:
                        cell = (a, o, acc, sy, fi)
                        tag = "".join(cell)
                        cap = _CAP[(sy, acc)]
                        if is_dest_acc_needed(InputOutputFormat(_F[a], _F[o])) and acc == "N":
                            continue
                        sh = _cap_shapes_small(cap, tag)
                        kt = _KT[_hash(tag + "dk") % 3]
                        for k, which in enumerate(("mid", "top")):
                            h, w = sh[_hash(tag + "d" + which) % len(sh)]
                            dst = cap // 2 if which == "mid" else cap - h * w
                            rows.append(("D", *cell, 32, 32, h, w, kt, dst))
    for a, b, o, acc in _new_cells():
        if a != b or not _is_new((a, b, o)):
            continue
        for sy in "HF":
            for fi in _FIDS:
                cell = (a, o, acc, sy, fi)
                tag = "".join(cell)
                cap = _CAP[(sy, acc)]
                r0, c1 = _GEOM[_hash(tag + "g") % len(_GEOM)]
                if "B2" in (a, o) and r0 < 4:
                    r0 = 4
                blk = ((1, 1, 1), (1, 4, 1), (1, 4, 4), (2, 2, 4), (1, cap, 1))[_hash(tag + "gb") % 5]
                rows.append(("G", *cell, r0, c1, *blk, 0))
    rows = [r for r in rows if r[0] in _FAMS2 and ",".join(map(str, r[1:])) not in _EXCL]
    return list(dict.fromkeys(rows))


def _fmt(a, b, o):
    return InputOutputFormat(_F[a], _F[o], input_format_B=_F[b]) if a != b else InputOutputFormat(_F[a], _F[o], input_format_B=_F[a])


def _cases():
    out = []
    for fam, a, b, o, acc, sy, fi, rt, ct, kt, thr, tr, sr in _fn1_rows():
        out.append(
            (
                _fmt(a, b, o),
                DestAccumulation.No if acc == "N" else DestAccumulation.Yes,
                DestSync.Half if sy == "H" else DestSync.Full,
                ([rt * TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, ct * TILE_DIM]),
                getattr(MathFidelity, fi),
                thr,
                Transpose.Yes if tr else Transpose.No,
                _SRV.get(sr),
            )
        )
    return out


def _config(a, o, acc, sy, r0, c1, rt, ct, kt, dst):
    """MatmulConfig of a math matmul block (equal operand formats); r0 x 32 in0 tiles with 2 faces (4 at 32), 32 x c1 in1 tiles with 4 faces (2 at 16)."""
    fmt = InputOutputFormat(_F[a], _F[o])
    td = generate_tile_dims(([rt * 32, kt * 32], [kt * 32, ct * 32]), in0_tile_r_dim=r0)
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
        dst_index=dst,
        dest_sync=DestSync.Half if sy == "H" else DestSync.Full,
        dest_acc=DestAccumulation.No if acc == "N" else DestAccumulation.Yes,
    )


def _cases2():
    out, ids = [], []
    for fam, a, o, acc, sy, fi, r0, c1, rt, ct, kt, dst in _fn2_rows():
        out.append((getattr(MathFidelity, fi), _config(a, o, acc, sy, r0, c1, rt, ct, kt, dst), 0, 1))
        ids.append("llkx2-" + "-".join(map(str, (fam, a, o, acc, sy, fi, r0, c1, rt, ct, kt, dst))))
    return out, ids


_C2, _I2 = _cases2() if _FAMS2 else ([], [])


@pytest.mark.perf
@skip_for_wormhole
@parametrize(case=_cases())
def test_perf_matmul_llkx(
    perf_report,
    case,
):

    case = case[0] if len(case) == 1 else case
    formats, dest_acc, dest_sync, (matrix_a, matrix_b), math_fidelity, throttle, transpose, stoch = case

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
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in generate_dest_corner_combinations.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)


    # cells where the keyed forms read slower than main's keep main's form (llk_matmul_forms.h matmul_grid_main_cells), as in the grid modules
    from helpers.matmul_prod_keys import grid_main_cell

    main_form = grid_main_cell(
        math_fidelity, dest_sync == DestSync.Half, dest_acc == DestAccumulation.Yes, formats.input_format, formats.input_format_B, formats.output_format,
        dims.rt_dim, dims.ct_dim, dims.kt_dim, throttle, False, 1,
    )
    form_keys = [
        k.with_block(dims.ct_dim, dims.rt_dim, dims.kt_dim, dest_acc, throttle, False)
        for k in MATMUL_FORM_KEYS.for_block(
            formats, dims.kt_dim, dims.ct_dim, dims.rt_dim,
            math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc, unfuse_one_tile=True, throttle=throttle,
        )
    ]
    if main_form:
        for k in form_keys:
            k.keep_flip = False
            k.unfused_post = False
            k.force_form = ""
    pack_block = MATMUL_PACK_BLOCK(
        matmul_pack_form(formats, dest_sync, math_fidelity, dims.rt_dim, dims.ct_dim, dims.kt_dim, throttle=throttle)
    )
    main_templates = [
        MATH_FIDELITY(math_fidelity),
        PACK_SRC_FROM_OUTPUT(),
        DEST_SYNC(dest_sync),
        THROTTLE_LEVEL(throttle),
        *form_keys,
        pack_block,
        *([MATMUL_STOCH_RND(stoch)] if stoch is not None else []),
    ]

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=main_templates if main_form else [
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
                formats, dest_acc, math_fidelity, dest_sync, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *form_keys,
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    dims.rt_dim,
                    dims.ct_dim,
                    dims.kt_dim,
                    throttle=throttle,
                )
            ),
            *([MATMUL_STOCH_RND(stoch)] if stoch is not None else []),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose),
            NUM_FACES(),
            LOOP_FACTOR(64 + _SRCODE.get(stoch, 0)),
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
