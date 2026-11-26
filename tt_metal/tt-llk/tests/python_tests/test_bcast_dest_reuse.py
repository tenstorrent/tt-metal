# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Coverage for the bcast dest-reuse family (Blackhole only):
``_llk_math_bcast_dest_reuse[_init]_`` and ``_llk_unpack_A_bcast_dest_reuse_init_``.

Covers both ``BroadcastType::SCALAR`` (element [0] broadcast across all elements) and
``BroadcastType::COL`` (Face 0 column 0 across 16 rows broadcast across all columns/faces),
driven from a single unpack call with SrcB sourced from DEST via ``MOVD2B`` under a
``WAIT_SFPU | MATH | SRCB_VLD`` stall.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.constraints import skip_if_32bit_input_without_dest_acc
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIM,
    EltwiseBinaryGolden,
    SrcFormatModel,
    round_to_dest_width,
)
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    BCAST_DEST_REUSE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024
ELEMENTS_PER_FACE = 256

OPERAND_LOW, OPERAND_HIGH = -4.0, 4.0

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

OPS = [MathOperation.Elwadd, MathOperation.Elwmul]

NUM_TILES = [1, 2, 3, 4]

_FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}


def _fidelities(mathop):
    if mathop == MathOperation.Elwmul:
        return list(ALL_MATH_FIDELITIES)
    return [MathFidelity.LoFi]


def _clear_dest_values(mathop):
    if mathop == MathOperation.Elwmul:
        return [True]
    return [False, True]


def _skip_unsupported(formats, dest_acc):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            "bf16 + dest_acc=Yes is not a distinct code path on this LLK; "
            "fp32 covers the fp32-DEST branches"
        )


def _run(
    formats,
    dest_acc,
    mathop,
    math_fidelity,
    num_tiles,
    *,
    num_faces=4,
    clear_dest=True,
    unpack_full_transpose=False,
    clobber_unpa_x_end=False,
    preserve_src_zero_flag=False,
    operands=None,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    if operands is None:
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=[num_tiles * TILE_DIM, TILE_DIM],
            stimuli_format_B=formats.input_format,
            input_dimensions_B=[TILE_DIM, TILE_DIM],
            spec_A=StimuliSpec.uniform(low=OPERAND_LOW, high=OPERAND_HIGH),
            spec_B=StimuliSpec.uniform(low=OPERAND_LOW, high=OPERAND_HIGH),
        )
    else:
        src_A, src_B = operands
        tile_cnt_A, tile_cnt_B = num_tiles, 1

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/bcast_dest_reuse_test.cpp",
            formats=formats,
            templates=[
                DEST_SYNC(DestSync.Half),
                MATH_OP(mathop=mathop),
                MATH_FIDELITY(math_fidelity),
                BCAST_DEST_REUSE(
                    bcast_dest_reuse_type=BroadcastType.Scalar,
                    bcast_dest_reuse_num_tiles=num_tiles,
                    bcast_dest_reuse_num_faces=num_faces,
                    clear_dest=clear_dest,
                    unpack_full_transpose=unpack_full_transpose,
                    bcast_clobber_unpa_x_end=clobber_unpa_x_end,
                    bcast_preserve_src_zero_flag=preserve_src_zero_flag,
                ),
            ],
            runtimes=[TILE_COUNT(num_tiles)],
            variant_stimuli=StimuliConfig(
                src_A.flatten(),
                formats.input_format,
                src_B.flatten(),
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=num_tiles,
            ),
            unpack_to_dest=False,
            dest_acc=dest_acc,
        ),
    )
    if is_perf:
        return None

    span = num_tiles * ELEMENTS_PER_TILE
    res_from_L1 = outcome.result[:span]
    torch_format = format_dict[formats.output_format]

    device = torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)
    seen_A = round_to_dest_width(src_A.flatten()[:span].to(torch.float32), dest_acc)
    seed = round_to_dest_width(
        src_B.flatten()[:ELEMENTS_PER_TILE].to(torch.float32), dest_acc
    )
    scalar = seed[:1].item()
    return device, seen_A, seed, scalar


def _tight_tolerances(output_format, *, mathop=MathOperation.Elwmul, dirty_dest=False):
    if output_format == DataFormat.Float32:
        if mathop == MathOperation.Elwmul:
            return dict(custom_atol=1e-5, custom_rtol=1e-5)
        # FPU ELWADD/ELWSUB aligns the smaller operand into the 10-bit TF32 adder
        # (up to 1 TF32 ULP of max(|A|, |B|) <= 2^-8 ~= 3.9e-3 for operands in [-4, 4]).
        return dict(custom_atol=5e-3, custom_rtol=2e-3)
    # BF16 DEST rounds each phase to 7-bit mantissa (1 ULP = 2^-7 ~= 7.8e-3); keep rtol
    # below the shift from a dropped phase while allowing 1-ULP rounding. When adding
    # onto a dirty DEST seeded in [-4, 4], allow 1-ULP cancellation (~0.03125).
    return dict(custom_atol=0.05 if dirty_dest else 2e-3, custom_rtol=8e-3)


def _to_src_reg(tensor, data_format):
    rounded = SrcFormatModel.from_src_format(
        data_format, SrcFormatModel.to_src_format(data_format, tensor)
    ).to(torch.float32)
    return torch.where(tensor == 0, torch.zeros_like(rounded), rounded)


def _expected(seen_A, scalar, mathop, math_fidelity, output_format, dest_prior=None):
    scalar_tensor = (
        scalar if isinstance(scalar, torch.Tensor) else torch.full_like(seen_A, scalar)
    )
    if mathop == MathOperation.Elwadd:
        return _to_src_reg(seen_A, output_format) + _to_src_reg(
            scalar_tensor, output_format
        )
    if mathop == MathOperation.Elwsub:
        return _to_src_reg(seen_A, output_format) - _to_src_reg(
            scalar_tensor, output_format
        )

    binary_golden = EltwiseBinaryGolden()

    product = None
    for fidelity_iteration in range(_FIDELITY_PHASES[math_fidelity]):
        a_masked, b_masked = binary_golden._apply_fidelity_masking(
            output_format, seen_A, scalar_tensor, fidelity_iteration
        )
        phase = a_masked.to(torch.float32) * b_masked.to(torch.float32)
        product = phase if product is None else product + phase

    if dest_prior is None:
        return product
    return _to_src_reg(dest_prior, output_format) + product


def _operands_with_full_mantissa_scalar(formats, num_tiles):
    """Build operands whose broadcast scalar populates the low SrcB fidelity bits."""
    src_A, _cnt_A, src_B, _cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[num_tiles * TILE_DIM, TILE_DIM],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[TILE_DIM, TILE_DIM],
        spec_A=StimuliSpec.uniform(low=OPERAND_LOW, high=OPERAND_HIGH),
        spec_B=StimuliSpec.uniform(low=OPERAND_LOW, high=OPERAND_HIGH),
    )
    # Maximize the low SrcB fidelity-bit ratio (1 + 15*2^-10 for TF32, 1 + 2^-7 for BF16)
    # so HiFi3/HiFi4 phases shift the product well past the tight tolerance bounds.
    low_bit_scalar = (
        1.0146484375 if formats.input_format == DataFormat.Float32 else 1.0078125
    )
    src_B = src_B.clone()
    src_B.view(-1)[0] = low_bit_scalar
    return src_A, src_B


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
    num_tiles=NUM_TILES,
    clear_dest=lambda mathop: _clear_dest_values(mathop),
)
def test_bcast_scalar_dest_reuse(
    formats, dest_acc, mathop, math_fidelity, num_tiles, clear_dest
):
    _skip_unsupported(formats, dest_acc)

    operands = (
        _operands_with_full_mantissa_scalar(formats, num_tiles)
        if mathop == MathOperation.Elwmul
        else None
    )
    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles,
        clear_dest=clear_dest,
        operands=operands,
    )
    golden = _expected(seen_A, scalar, mathop, math_fidelity, formats.output_format)

    assert passed_test(
        golden,
        device,
        formats.output_format,
        **_tight_tolerances(formats.output_format, mathop=mathop),
    ), (
        f"bcast-scalar dest reuse mismatch (op={mathop.name}, "
        f"fidelity={math_fidelity.name}, num_tiles={num_tiles}, "
        f"clear_dest={clear_dest}, scalar={scalar:.6f})"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=_fidelities(MathOperation.Elwmul),
)
def test_bcast_scalar_dest_reuse_mul_accumulates_into_dirty_dest(
    formats, dest_acc, math_fidelity
):
    """ELWMUL adds its product onto whatever DEST already held."""
    _skip_unsupported(formats, dest_acc)

    device, seen_A, seed, scalar = _run(
        formats,
        dest_acc,
        MathOperation.Elwmul,
        math_fidelity,
        num_tiles=1,
        clear_dest=False,
    )
    accumulating = _expected(
        seen_A,
        scalar,
        MathOperation.Elwmul,
        math_fidelity,
        formats.output_format,
        dest_prior=seed,
    )
    overwriting = _expected(
        seen_A, scalar, MathOperation.Elwmul, math_fidelity, formats.output_format
    )

    assert passed_test(
        accumulating,
        device,
        formats.output_format,
        **_tight_tolerances(formats.output_format, dirty_dest=True),
    ), (
        f"ELWMUL with clear_dest=False did not accumulate onto the seeded DEST "
        f"(fidelity={math_fidelity.name}, scalar={scalar:.6f})."
    )
    assert not passed_test(
        overwriting, device, formats.output_format, print_errors=False
    ), (
        "ELWMUL with clear_dest=False matched the overwriting golden, so the seeded DEST "
        "made no difference and this test is not measuring the accumulate path"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_tiles=NUM_TILES,
)
def test_bcast_scalar_dest_reuse_add_overwrites_dirty_dest(
    formats, dest_acc, num_tiles
):
    """Negative control for the test above: ELWADD overwrites dirty DEST."""
    _skip_unsupported(formats, dest_acc)

    dirty, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        MathOperation.Elwadd,
        MathFidelity.LoFi,
        num_tiles,
        clear_dest=False,
    )
    golden = _expected(
        seen_A, scalar, MathOperation.Elwadd, MathFidelity.LoFi, formats.output_format
    )

    assert passed_test(golden, dirty, formats.output_format), (
        f"ELWADD with clear_dest=False did not overwrite the seeded DEST "
        f"(num_tiles={num_tiles}, scalar={scalar:.6f})"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    num_faces=[1, 2],
    num_tiles=[1, 2],
    math_fidelity=lambda mathop: _fidelities(mathop),
)
def test_bcast_scalar_dest_reuse_partial_faces(
    formats, dest_acc, mathop, num_faces, num_tiles, math_fidelity
):
    """num_faces < 4: the MOP covers only the leading faces of each tile."""
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=num_tiles,
        num_faces=num_faces,
        clear_dest=True,
    )

    covered = num_faces * ELEMENTS_PER_FACE
    for tile in range(num_tiles):
        source = seen_A[tile * covered : (tile + 1) * covered]
        golden_covered = _expected(
            source, scalar, mathop, math_fidelity, formats.output_format
        )
        base = tile * ELEMENTS_PER_TILE

        assert passed_test(
            golden_covered,
            device[base : base + covered],
            formats.output_format,
            **_tight_tolerances(formats.output_format, mathop=mathop),
        ), (
            f"the {num_faces} face(s) the MOP covers in tile {tile} are wrong "
            f"(op={mathop.name}, fidelity={math_fidelity.name}, num_tiles={num_tiles}, "
            f"scalar={scalar:.6f})"
        )

        tail = device[base + covered : base + ELEMENTS_PER_TILE]
        nonzero = torch.count_nonzero(tail).item()
        assert nonzero == 0, (
            f"clear_dest=True left {nonzero} non-zero elements in the {4 - num_faces} "
            f"face(s) of tile {tile} the MOP does not cover"
        )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
)
def test_bcast_scalar_dest_reuse_unpack_full_transpose(
    formats, dest_acc, mathop, math_fidelity
):
    """Transpose-fold path (num_tiles == 1, num_faces == 4)."""
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=1,
        clear_dest=True,
        unpack_full_transpose=True,
    )

    faces = seen_A[:ELEMENTS_PER_TILE].reshape(2, 2, 16, 16)
    transposed = faces.permute(1, 0, 3, 2).reshape(-1)

    golden = _expected(transposed, scalar, mathop, math_fidelity, formats.output_format)

    assert passed_test(
        golden,
        device,
        formats.output_format,
        **_tight_tolerances(formats.output_format, mathop=mathop),
    ), (
        f"unpack_full_transpose mismatch (op={mathop.name}, "
        f"fidelity={math_fidelity.name}, scalar={scalar:.6f})"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    num_tiles=[1, 2],
)
def test_bcast_scalar_dest_reuse_reprograms_srca_x_span(
    formats, dest_acc, mathop, num_tiles
):
    """The unpack init must program the X span on unpacker A."""
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        MathFidelity.LoFi,
        num_tiles,
        clear_dest=True,
        clobber_unpa_x_end=True,
    )
    golden = _expected(seen_A, scalar, mathop, MathFidelity.LoFi, formats.output_format)

    assert passed_test(golden, device, formats.output_format), (
        f"bcast_dest_reuse result is wrong after unpacker A was left at a one-row X span "
        f"(op={mathop.name}, num_tiles={num_tiles}, scalar={scalar:.6f})"
    )


BF16_DENORMAL_MIN = 2.0**-133
BF16_DENORMAL_MANTISSAS = 1 << 7
DENORMAL_SRCA_SCALE = 2.0**100
DENORMAL_SRCA_LOW, DENORMAL_SRCA_HIGH = 1.0, 4.0
TINY_NORMAL_SCALAR = 2.0**-100


@pytest.mark.parametrize("math_fidelity", _fidelities(MathOperation.Elwmul))
@pytest.mark.parametrize("scalar_kind", ["denormal", "tiny_normal"])
def test_bcast_scalar_dest_reuse_flushes_denormal_scalar_after_keep_flag(
    math_fidelity, scalar_kind
):
    """The math init must restore the Src zero flag a preceding copy_init left at keep."""
    formats = FORMATS[0]
    assert formats.input_format == DataFormat.Float16_b
    dest_acc = DestAccumulation.No
    num_tiles = 1

    torch.manual_seed(0)
    src_A = (
        torch.rand(num_tiles * TILE_DIM, TILE_DIM)
        * (DENORMAL_SRCA_HIGH - DENORMAL_SRCA_LOW)
        + DENORMAL_SRCA_LOW
    ) * DENORMAL_SRCA_SCALE
    src_B = torch.rand(TILE_DIM, TILE_DIM) * (OPERAND_HIGH - OPERAND_LOW) + OPERAND_LOW
    if scalar_kind == "denormal":
        src_B[0, 0] = (
            BF16_DENORMAL_MIN * torch.randint(1, BF16_DENORMAL_MANTISSAS, (1,)).item()
        )
    else:
        src_B[0, 0] = TINY_NORMAL_SCALAR
    src_A = src_A.to(torch.bfloat16)
    src_B = src_B.to(torch.bfloat16)
    assert src_B[0, 0].item() != 0.0, "scalar did not survive the bf16 cast"

    device, seen_A, _seed, _scalar = _run(
        formats,
        dest_acc,
        MathOperation.Elwmul,
        math_fidelity,
        num_tiles,
        clear_dest=True,
        preserve_src_zero_flag=True,
        operands=(src_A, src_B),
    )

    if scalar_kind == "denormal":
        kept = torch.count_nonzero(device).item()
        assert kept == 0, (
            f"denormal broadcast scalar was not flushed (fidelity={math_fidelity.name}): "
            f"{kept} nonzero outputs, max |value| {device.abs().max().item():.3e}"
        )
        return

    golden = _expected(
        seen_A,
        TINY_NORMAL_SCALAR,
        MathOperation.Elwmul,
        math_fidelity,
        formats.output_format,
    )
    assert (
        torch.count_nonzero(device).item() == device.numel()
    ), f"tiny normal scalar 2^-100 produced zero outputs (fidelity={math_fidelity.name})"
    assert passed_test(
        golden, device, formats.output_format
    ), f"tiny normal scalar 2^-100 product mismatch (fidelity={math_fidelity.name})"


def test_bcast_scalar_dest_reuse_repeat_state():
    formats = FORMATS[0]
    device, seen_A, _seed, scalar = _run(
        formats,
        DestAccumulation.No,
        MathOperation.Elwmul,
        MathFidelity.LoFi,
        1,
        clear_dest=True,
        loop_factor=2,
    )
    golden = _expected(
        seen_A,
        scalar,
        MathOperation.Elwmul,
        MathFidelity.LoFi,
        formats.output_format,
    )
    assert passed_test(golden, device, formats.output_format)


@pytest.mark.parametrize(
    ("mathop", "math_fidelity", "num_tiles", "num_faces", "dest_sync", "dest_acc"),
    [
        (
            MathOperation.Elwmul,
            MathFidelity.LoFi,
            4,
            4,
            DestSync.Half,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.HiFi4,
            4,
            4,
            DestSync.Half,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.HiFi4,
            4,
            4,
            DestSync.Half,
            DestAccumulation.Yes,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.LoFi,
            4,
            2,
            DestSync.Half,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.HiFi4,
            4,
            2,
            DestSync.Half,
            DestAccumulation.Yes,
        ),
        (
            MathOperation.Elwadd,
            MathFidelity.LoFi,
            4,
            4,
            DestSync.Half,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwsub,
            MathFidelity.LoFi,
            4,
            4,
            DestSync.Half,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.LoFi,
            16,
            4,
            DestSync.Full,
            DestAccumulation.No,
        ),
        (
            MathOperation.Elwmul,
            MathFidelity.LoFi,
            16,
            2,
            DestSync.Full,
            DestAccumulation.No,
        ),
    ],
)
def test_bcast_col_dest_reuse(
    mathop,
    math_fidelity,
    num_tiles,
    num_faces,
    dest_sync,
    dest_acc,
):
    """BroadcastType::COL dest-reuse across ELWMUL/ELWADD/ELWSUB, 2 & 4 faces, SyncHalf & SyncFull."""
    torch.manual_seed(123)
    data_format = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )
    torch_dtype = torch.float32 if data_format == DataFormat.Float32 else torch.bfloat16
    active_h = 16 if num_faces == 2 else 32
    tile_dims = [active_h, 32]
    tile_elems = active_h * 32

    a_tiles = torch.randn(num_tiles, active_h, 32).to(torch_dtype)
    col_vec = torch.empty(16).uniform_(0.25, 2.0).to(torch_dtype)
    b_tile = torch.zeros(active_h, 32, dtype=torch_dtype)
    # Seed col 0 of Face 0 (rows 0..15, col 0); populate other columns with distinct noise.
    b_tile[:16, :] = -9.0
    b_tile[:16, 0] = col_vec

    src_a = torch.cat(
        [
            tilize_block(
                a_tiles[t].flatten(),
                tile_dims,
                stimuli_format=data_format,
                num_faces=num_faces,
                tile_dimensions=tile_dims,
                face_r_dim=16,
            ).flatten()
            for t in range(num_tiles)
        ]
    )
    src_b = tilize_block(
        b_tile.flatten(),
        tile_dims,
        stimuli_format=data_format,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=16,
    ).flatten()

    outcome = execute_llk(
        is_perf=False,
        perf_report=None,
        run_types=None,
        loop_factor=1,
        test_config_kwargs=dict(
            test_name="sources/bcast_dest_reuse_test.cpp",
            formats=input_output_formats([data_format], same=True)[0],
            templates=[
                DEST_SYNC(dest_sync),
                MATH_OP(mathop=mathop),
                MATH_FIDELITY(math_fidelity),
                BCAST_DEST_REUSE(
                    bcast_dest_reuse_type=BroadcastType.Column,
                    bcast_dest_reuse_num_tiles=num_tiles,
                    bcast_dest_reuse_num_faces=num_faces,
                    clear_dest=True,
                ),
            ],
            runtimes=[TILE_COUNT(num_tiles)],
            variant_stimuli=StimuliConfig(
                src_a,
                data_format,
                src_b,
                data_format,
                data_format,
                tile_count_A=num_tiles,
                tile_count_B=1,
                tile_count_res=num_tiles,
                num_faces=num_faces,
                face_r_dim=16,
                tile_dimensions=tile_dims,
                use_dense_tile_dimensions=True,
            ),
            dest_acc=dest_acc,
            unpack_to_dest=False,
        ),
    )

    raw = torch.tensor(outcome.result, dtype=torch_dtype)
    out = torch.stack(
        [
            untilize_block(
                raw[t * tile_elems : (t + 1) * tile_elems],
                data_format,
                tile_dims,
                num_faces=num_faces,
                tile_dimensions=tile_dims,
                face_r_dim=16,
            )
            .reshape(active_h, 32)
            .float()
            for t in range(num_tiles)
        ]
    )

    bcast = (
        col_vec.float()
        .repeat(active_h // 16)
        .view(1, active_h, 1)
        .expand(num_tiles, active_h, 32)
        .reshape(-1)
    )
    a_ref = a_tiles.float().reshape(-1)
    want = _expected(a_ref, bcast, mathop, math_fidelity, data_format).reshape(
        num_tiles, active_h, 32
    )

    if dest_acc == DestAccumulation.Yes:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 2e-3, 1.2e-2
    torch.testing.assert_close(out, want, atol=atol, rtol=rtol)
