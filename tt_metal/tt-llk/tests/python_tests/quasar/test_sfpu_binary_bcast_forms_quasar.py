# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dest-side SFPU broadcast forms beyond the full-tile sfpu_{add,sub,mul}_bcast_{col,row}
(test_eltwise_binary_sfpu_quasar.py covers those): the partial column broadcasts, the parked scalar
and the fused normalisations. Kept out of test_eltwise_binary_sfpu_quasar.py so the perf test built
on that module keeps its parameter schema."""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    VectorMode,
)
from helpers.param_config import InputOutputFormat, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_FACES,
    SFPU_BCAST_FORM,
    SFPU_BINARY_OP,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES, MAX_TILE_ELEMENTS
from helpers.utils import passed_test

# fp32 only (the Compute API contract of these forms), staged with unpack_to_dest into Dest tiles 0..3.
_FORMS_CPP_SOURCE = "sources/quasar/sfpu_binary_bcast_forms_quasar_test.cpp"
_FORMS_TILE_CNT = 4
# Dest tiles of the operand roles T0..T3 (see the C++ header): identity, and a permutation that
# puts the data tile above its statistics tiles.
_FORMS_TILE_LAYOUTS = [(0, 1, 2, 3), (2, 0, 3, 1)]
_FORMS_SPECIALS = (float("inf"), float("-inf"), float("nan"))
_FORMS_BINARY_OPS = [
    MathOperation.SfpuElwadd,
    MathOperation.SfpuElwsub,
    MathOperation.SfpuElwmul,
]


def _face_order_to_rows(flat):
    """One 32x32 tile in L1 face order (face-major, 16x16 faces) -> [32, 32] rows."""
    return flat.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)


def _rows_to_face_order(rows):
    """Inverse of :func:`_face_order_to_rows`."""
    return rows.reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(-1)


def _forms_uniform(gen, low, high):
    return torch.rand(32, 32, generator=gen, dtype=torch.float32) * (high - low) + low


def _forms_poison(rows, keep):
    """Inf / -Inf / NaN everywhere ``keep`` is False: the lanes a form must never read."""
    r = torch.arange(32).reshape(32, 1)
    c = torch.arange(32).reshape(1, 32)
    poison = torch.tensor(_FORMS_SPECIALS)[(r + c) % len(_FORMS_SPECIALS)]
    return torch.where(keep, rows, poison)


def _forms_col0_mask(rows=slice(0, 32)):
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[rows, 0] = True
    return mask


def _forms_scalar_mask():
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[0, 0] = True
    return mask


def _forms_bits(t):
    return t.contiguous().view(torch.int32)


def _run_sfpu_bcast_form(
    form,
    role_rows,
    golden_rows,
    touched,
    *,
    tile_layout,
    binary_op="ADD",
    data_band=0,
    col_band=0,
    scalar_iterations=8,
    vector_mode=VectorMode.RC,
    max_ulp=1,
):
    """Stage the four role tiles at ``tile_layout``, run ``form`` and check each output tile.

    ``golden_rows`` holds the expected output tiles (rows layout), ``touched`` the matching
    lane masks: touched lanes are held to ``max_ulp`` steps, every other lane must keep the
    data tile's bits.
    """
    formats = InputOutputFormat(
        input_format=DataFormat.Float32, output_format=DataFormat.Float32
    )
    tiles = [None] * _FORMS_TILE_CNT
    for role, dest_tile in enumerate(tile_layout):
        tiles[dest_tile] = _rows_to_face_order(role_rows[role])
    src_A = torch.cat(tiles)
    out_tiles = len(golden_rows)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": _FORMS_CPP_SOURCE,
            "formats": formats,
            "templates": [
                SFPU_BINARY_OP(binary_op),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
                DEST_SYNC(),
                SFPU_BCAST_FORM(
                    sfpu_bcast_form=form,
                    sfpu_bcast_tiles=tuple(tile_layout),
                    sfpu_bcast_data_band=data_band,
                    sfpu_bcast_col_band=col_band,
                    sfpu_bcast_scalar_iterations=scalar_iterations,
                    sfpu_bcast_vector_mode=vector_mode,
                ),
            ],
            "runtimes": [
                TILE_COUNT(_FORMS_TILE_CNT),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                torch.zeros_like(src_A),
                formats.input_format,
                formats.output_format,
                tile_count_A=_FORMS_TILE_CNT,
                tile_count_B=_FORMS_TILE_CNT,
                tile_count_res=out_tiles,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": True,
            "dest_acc": DestAccumulation.Yes,
        },
    )
    res = torch.tensor(configuration.run().result, dtype=torch.float32)
    assert len(res) == out_tiles * MAX_TILE_ELEMENTS
    for i, (golden, mask) in enumerate(zip(golden_rows, touched)):
        got = _face_order_to_rows(
            res[i * MAX_TILE_ELEMENTS : (i + 1) * MAX_TILE_ELEMENTS]
        )
        assert passed_test(
            golden[mask], got[mask], DataFormat.Float32, max_ulp=max_ulp
        ), f"{form}: output tile {i} differs on the lanes it computes"
        assert torch.equal(
            _forms_bits(got[~mask]), _forms_bits(golden[~mask])
        ), f"{form}: output tile {i} changed lanes outside the form"


# --- sfpu_{add,sub,mul}_bcast_col(VectorMode::R) and sfpu_{add,sub,mul}_bcast_col_band -----
_BCAST_COL_BAND_PAIRS = [(0, 3), (2, 1)]


def _bcast_col_partial_case(mathop, rows_r, col_rows):
    """data OP col[r][0] on data rows ``rows_r``, reading column-tile rows ``col_rows``."""
    gen = torch.Generator().manual_seed(1201)
    data = _forms_uniform(gen, -4.0, 4.0)
    col = _forms_uniform(gen, -2.0, 2.0)
    # -0.0 in the column (ADD keeps -0.0 + -0.0); every lane the form must not read poisoned.
    col[col_rows.start, 0] = -0.0
    col = _forms_poison(col, _forms_col0_mask(col_rows))
    bcast = col[col_rows, 0:1].expand(rows_r.stop - rows_r.start, 32)
    golden = data.clone()
    if mathop == MathOperation.SfpuElwadd:
        golden[rows_r] = data[rows_r] + bcast
    elif mathop == MathOperation.SfpuElwsub:
        golden[rows_r] = data[rows_r] - bcast
    else:
        golden[rows_r] = data[rows_r] * bcast
    touched = torch.zeros(32, 32, dtype=torch.bool)
    touched[rows_r] = True
    filler = _forms_uniform(gen, -1.0, 1.0)
    return [data, col, filler, filler.clone()], [golden], [touched]


@pytest.mark.quasar
@parametrize(
    mathop=_FORMS_BINARY_OPS,
    tile_layout=_FORMS_TILE_LAYOUTS,
)
def test_sfpu_bcast_col_rows_0_15_quasar(mathop, tile_layout):
    """sfpu_*_bcast_col(..., VectorMode::R): tile rows 0..15 only; rows 16..31 untouched and unread."""
    roles, golden, touched = _bcast_col_partial_case(mathop, slice(0, 16), slice(0, 16))
    _run_sfpu_bcast_form(
        "ROWS_0_15",
        roles,
        golden,
        touched,
        tile_layout=tile_layout,
        binary_op=mathop.cpp_enum_value,
    )


@pytest.mark.quasar
@parametrize(
    mathop=_FORMS_BINARY_OPS,
    bands=_BCAST_COL_BAND_PAIRS,
    tile_layout=_FORMS_TILE_LAYOUTS,
)
def test_sfpu_bcast_col_band_quasar(mathop, bands, tile_layout):
    """sfpu_*_bcast_col_band: data rows 4*data_band..+3 OP column rows 4*col_band..+3, nothing else."""
    data_band, col_band = bands
    roles, golden, touched = _bcast_col_partial_case(
        mathop,
        slice(4 * data_band, 4 * data_band + 4),
        slice(4 * col_band, 4 * col_band + 4),
    )
    _run_sfpu_bcast_form(
        "BAND",
        roles,
        golden,
        touched,
        tile_layout=tile_layout,
        binary_op=mathop.cpp_enum_value,
        data_band=data_band,
        col_band=col_band,
    )


# --- sfpu_bcast_scalar_load + sfpu_mul_bcast_scalar ---------------------------------------
def _mul_bcast_scalar_touched(iterations, vector_mode):
    """Iteration k of a face is face rows 4*(k//2)..+3 of one column parity (even k: even columns)."""
    face = torch.zeros(16, 16, dtype=torch.bool)
    for k in range(iterations):
        face[4 * (k // 2) : 4 * (k // 2) + 4, (k % 2) :: 2] = True
    faces = {
        VectorMode.RC: (0, 1, 2, 3),
        VectorMode.R: (0, 1),
        VectorMode.C: (0, 2),
    }[vector_mode]
    touched = torch.zeros(32, 32, dtype=torch.bool)
    for f in faces:
        touched[
            16 * (f // 2) : 16 * (f // 2) + 16, 16 * (f % 2) : 16 * (f % 2) + 16
        ] = face
    return touched


@pytest.mark.quasar
@parametrize(
    iterations=[2, 8],
    vector_mode=[VectorMode.RC, VectorMode.R, VectorMode.C],
    tile_layout=_FORMS_TILE_LAYOUTS,
)
def test_sfpu_mul_bcast_scalar_quasar(iterations, vector_mode, tile_layout):
    """One sfpu_bcast_scalar_load, then sfpu_mul_bcast_scalar<ITERATIONS>(tile, vector_mode) on two tiles."""
    gen = torch.Generator().manual_seed(1202)
    data0 = _forms_uniform(gen, -4.0, 4.0)
    data1 = _forms_uniform(gen, -4.0, 4.0)
    scalar = _forms_poison(_forms_uniform(gen, -3.0, 3.0), _forms_scalar_mask())
    s = scalar[0, 0]
    touched = _mul_bcast_scalar_touched(iterations, vector_mode)
    golden = [torch.where(touched, d * s, d) for d in (data0, data1)]
    _run_sfpu_bcast_form(
        "MUL_SCALAR",
        [data0, scalar, data1, _forms_uniform(gen, -1.0, 1.0)],
        golden,
        [touched, touched],
        tile_layout=tile_layout,
        scalar_iterations=iterations,
        vector_mode=vector_mode,
    )


# --- sfpu_normalize_bcast_col / _two_tiles / sfpu_residual_normalize_bcast_col / _scalar -----
_NORMALIZE_FORMS = [
    "NORM_COL",
    "NORM_COL_TWO_TILES",
    "RESIDUAL_NORM_COL",
    "NORM_SCALAR",
]


@pytest.mark.quasar
@parametrize(
    form=_NORMALIZE_FORMS,
    tile_layout=_FORMS_TILE_LAYOUTS,
)
def test_sfpu_normalize_bcast_quasar(form, tile_layout):
    """(data [+ residual] - mean) * inv_std, mean / inv_std from column 0 (col forms) or [0][0] (scalar)."""
    gen = torch.Generator().manual_seed(1203)
    data = _forms_uniform(gen, -4.0, 4.0)
    mean = _forms_uniform(gen, -1.0, 1.0)
    inv_std = _forms_uniform(gen, 0.5, 2.0)
    extra = _forms_uniform(gen, -4.0, 4.0)  # residual, or the second data tile
    keep = _forms_scalar_mask() if form == "NORM_SCALAR" else _forms_col0_mask()
    mean = _forms_poison(mean, keep)
    inv_std = _forms_poison(inv_std, keep)
    if form == "NORM_SCALAR":
        m, s = mean[0, 0], inv_std[0, 0]
    else:
        m, s = mean[:, 0:1], inv_std[:, 0:1]
    # The kernel adds the negated mean and then multiplies: fp32 sub and mul, each rounded.
    if form == "RESIDUAL_NORM_COL":
        golden = [((data + extra) - m) * s]
    elif form == "NORM_COL_TWO_TILES":
        golden = [(data - m) * s, (extra - m) * s]
    else:
        golden = [(data - m) * s]
    everything = torch.ones(32, 32, dtype=torch.bool)
    _run_sfpu_bcast_form(
        form,
        [data, mean, inv_std, extra],
        golden,
        [everything] * len(golden),
        tile_layout=tile_layout,
        max_ulp=2,
    )
