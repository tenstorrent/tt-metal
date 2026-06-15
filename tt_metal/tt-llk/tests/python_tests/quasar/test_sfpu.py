# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generic, registry-driven SFPU test for Quasar.

A single C++ skeleton (``sources/quasar/sfpu_test.cpp``) dispatches SFPU ops
through preprocessor defines. Each op is described once as a spec module under
``quasar/sfpu/{unary,binary,ternary}/`` and auto-discovered into
``UNARY_OP_REGISTRY`` / ``BINARY_OP_REGISTRY``. This driver parametrizes over
those registries; adding a new op is just a new spec module, no new test code.

Golden references are derived from each spec's ``math_op`` via the existing
arity-specific golden generators (no per-op golden is stored on the spec).

The three arities are symmetric here: each is described once as a spec module and
swept by its driver. The ternary ``where`` op lives in ``quasar/sfpu/ternary/where.py``.
"""

from dataclasses import dataclass
from dataclasses import fields as dc_fields

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BinarySFPUGolden,
    TernarySFPUGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    InputOutputFormat,
    generate_sfpu_format_dest_acc_combinations,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    SFPU_DEFINES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test
from quasar.sfpu import (
    BINARY_OP_REGISTRY,
    TERNARY_OP_REGISTRY,
    UNARY_OP_REGISTRY,
)
from quasar.sfpu._utils import expand_extra_templates

_SOURCE = "sources/quasar/sfpu_test.cpp"
_NUM_FACES = 4
_ELEMENTS_PER_TILE = 1024  # 4 faces * 16 * 16


def _io(fmt) -> InputOutputFormat:
    """Adapt a spec's per-arity format entry to the TestConfig InputOutputFormat.

    Unary specs expose ``input_format``; binary specs expose ``input_a_format``
    (operands share a format in these tests)."""
    in_fmt = fmt.input_format if hasattr(fmt, "input_format") else fmt.input_a_format
    return InputOutputFormat(input_format=in_fmt, output_format=fmt.output_format)


def _formats(spec):
    """(io, dest_acc) sweep for a spec's declared formats."""
    return generate_sfpu_format_dest_acc_combinations([_io(f) for f in spec.formats])


def _prepare(spec, src, *, src_tiles):
    """Apply the op's optional stimulus post-processing hook (e.g. divide injects
    its NaN/Inf/1.0 special-case lanes); identity when the op declares none.
    ``src_tiles`` is the tuple of DEST tile indices the operands occupy this variant."""
    return spec.prepare(src, src_tiles=src_tiles) if spec.prepare is not None else src


def _implied_math_formats(io):
    """ImpliedMathFormat sweep for a format: MX formats require it enabled, so they
    run only ``Yes``; every other format runs both ``No`` and ``Yes``."""
    if io.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


@dataclass(frozen=True)
class _ExtraTemplates:
    """One resolved combination of a spec's extra build-time template params, named for
    the pytest variant id (``parametrize`` uses the ``.name`` attribute)."""

    templates: tuple

    @property
    def name(self) -> str:
        if not self.templates:
            return "default"
        return "+".join(
            ".".join(
                getattr(getattr(t, f.name), "name", str(getattr(t, f.name)))
                for f in dc_fields(t)
            )
            for t in self.templates
        )


def _extra_template_variants(spec):
    """Sweep a spec's ``extra_templates``: expand any list-valued TemplateParameter
    field into the cross product, one variant per combination. A spec with no extras
    (or only scalar fields) yields a single variant, so this never multiplies the
    common case."""
    return [
        _ExtraTemplates(combo) for combo in expand_extra_templates(spec.extra_templates)
    ]


def _golden(spec, default, *, src, io, dest_acc, dimensions):
    """Resolve the op's golden: its custom ``golden`` hook if declared, else the
    arity default. ``default`` is a zero-arg callable producing the standard
    math_op-keyed golden, so the common case pays nothing."""
    if spec.golden is not None:
        return spec.golden(src=src, io=io, dest_acc=dest_acc, dimensions=dimensions)
    return default()


def _supported_formats(spec):
    """(io, dest_acc) sweep dropping Float16 + DestAcc.Yes, which the multi-operand SFPU
    paths (binary, ternary) don't support. Used by the binary and ternary drivers;
    unary keeps the unfiltered ``_formats``."""
    return [
        (io, da)
        for io, da in _formats(spec)
        if not (io.input_format == DataFormat.Float16 and da == DestAccumulation.Yes)
    ]


def _run(
    *,
    arity_macro,
    spec,
    io,
    dest_acc,
    src_A,
    src_B,
    tile_cnt_A,
    tile_count_res,
    golden_tensor,
    unpack_to_dest,
    implied_math_format=ImpliedMathFormat.No,
    dest_sync=DestSync.Half,
    dst_index=0,
    operands=None,
    extra_templates=(),
):
    """Build the variant, run it through the SFPU skeleton, and assert vs golden.

    Everything that doesn't depend on op arity lives here; the per-arity test
    functions only differ in how they prepare stimuli and the golden tensor.
    ``operands`` is an optional (src0, src1, dst) tile triple emitted as
    SFPU_BINARY_OPERANDS for tile-index binary ops (None leaves the op's own operand
    placement intact); ``dst_index`` is the DEST tile the result is packed from.
    ``extra_templates`` is the op's resolved extra build-time template params for this
    variant (e.g. ternary's VECTOR_MODE), spliced into the build verbatim."""
    defines = dict(spec.sfpu_defines(io))
    if operands is not None:
        src0_idx, src1_idx, dst_idx = operands
        defines["SFPU_BINARY_OPERANDS"] = f"{src0_idx}u, {src1_idx}u, {dst_idx}u"

    configuration = TestConfig(
        _SOURCE,
        io,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            SFPU_DEFINES(arity_macro, spec.include_header, defines),
            *extra_templates,
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(dst_index),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            io.input_format,
            src_B,
            io.input_format,
            io.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_count_res,
            num_faces=_NUM_FACES,
            twos_complement=io.input_format.is_integer(),
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[io.output_format])
    assert passed_test(
        golden_tensor, res_tensor, io.output_format
    ), "Assert against golden failed"

    # Op-specific exactness checks the tolerance comparison can't express.
    if spec.verify is not None:
        spec.verify(res_tensor, io=io)


# ---------------------------------------------------------------------------
# Unary
# ---------------------------------------------------------------------------
@pytest.mark.quasar
@parametrize(
    spec=UNARY_OP_REGISTRY,
    format_dest_acc=_formats,
    implied_math_format=lambda format_dest_acc: _implied_math_formats(
        format_dest_acc[0]
    ),
    dest_sync=[DestSync.Half, DestSync.Full],
    input_dimensions=[[32, 32], [64, 64]],
)
def test_sfpu_unary(
    spec, format_dest_acc, implied_math_format, dest_sync, input_dimensions
):
    """Run a unary SFPU op (one input tile, transformed in place)."""
    io, dest_acc = format_dest_acc

    torch.manual_seed(0)  # reproducible stimuli
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=io.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=io.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = _prepare(spec, src_A, src_tiles=(0,))

    golden_tensor = _golden(
        spec,
        lambda: get_golden_generator(UnarySFPUGolden)(
            spec.math_op,
            src_A,
            io.output_format,
            dest_acc,
            io.input_format,
            input_dimensions,
        ),
        src=src_A,
        io=io,
        dest_acc=dest_acc,
        dimensions=input_dimensions,
    )

    _run(
        arity_macro=spec.arity_macro,
        spec=spec,
        io=io,
        dest_acc=dest_acc,
        src_A=src_A,
        src_B=src_B,
        tile_cnt_A=tile_cnt_A,
        tile_count_res=tile_cnt_A,
        golden_tensor=golden_tensor,
        unpack_to_dest=io.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes,
        implied_math_format=implied_math_format,
        dest_sync=dest_sync,
        extra_templates=tuple(spec.extra_templates),
    )


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------
def _tile_layouts(spec):
    """Tile-layout sweep for a binary spec: each (src0, src1, dst) the driver places
    operands at. ``None`` for offset-style ops that place operands themselves — they
    run the single default layout (operands intact, result packed from tile 0)."""
    return spec.tile_layouts if spec.tile_layouts is not None else [None]


@pytest.mark.quasar
@parametrize(
    spec=BINARY_OP_REGISTRY,
    format_dest_acc=_supported_formats,
    implied_math_format=lambda format_dest_acc: _implied_math_formats(
        format_dest_acc[0]
    ),
    tile_layout=_tile_layouts,
)
def test_sfpu_binary(spec, format_dest_acc, implied_math_format, tile_layout):
    """Run a binary SFPU op: two operand tiles in DEST, one result tile packed out.

    ``tile_layout`` is the (src0, src1, dst) DEST-tile placement for tile-index ops
    (swept to cover result-over-operand aliasing); ``None`` means the op places its
    own operands (offset-style int ops) and the result is packed from tile 0."""
    io, dest_acc = format_dest_acc
    data_format = io.input_format

    if tile_layout is None:
        # Offset-style op: operands at tiles 0/1, result to tile 0 (driver leaves the
        # op's own SFPU_BINARY_OPERANDS in place).
        src0_idx, src1_idx, dst_idx = 0, 1, 0
        operands = None
    else:
        src0_idx, src1_idx, dst_idx = tile_layout
        operands = tile_layout

    # Enough tiles to cover the highest operand/result index, concatenated along rows.
    num_tiles = max(src0_idx, src1_idx, dst_idx) + 1
    input_dimensions = [num_tiles * 32, 32]

    # Per-op stimulus range when supplied (e.g. divide bounds divisors away from
    # zero), else a sane default for the dtype.
    if spec.stimuli is not None:
        stim = spec.stimuli
    elif data_format.is_integer():
        iinfo = torch.iinfo(format_dict[data_format])
        stim = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    else:
        stim = StimuliSpec.uniform(low=-1.0, high=1.0)

    torch.manual_seed(0)  # reproducible stimuli
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
        spec_A=stim,
        spec_B=stim,
    )
    src_A = _prepare(spec, src_A, src_tiles=(src0_idx, src1_idx))

    dst_start = dst_idx * _ELEMENTS_PER_TILE
    golden_tensor = _golden(
        spec,
        lambda: get_golden_generator(BinarySFPUGolden)(
            spec.math_op,
            src_A,
            src0_idx,
            src1_idx,
            dst_idx,
            32,  # num_iterations: 32 rows = 1 full tile
            input_dimensions,
            data_format,
        ).flatten()[dst_start : dst_start + _ELEMENTS_PER_TILE],
        src=src_A,
        io=io,
        dest_acc=dest_acc,
        dimensions=input_dimensions,
    )

    _run(
        arity_macro=spec.arity_macro,
        spec=spec,
        io=io,
        dest_acc=dest_acc,
        src_A=src_A,
        src_B=src_B,
        tile_cnt_A=tile_cnt_A,
        tile_count_res=1,
        golden_tensor=golden_tensor,
        unpack_to_dest=True,
        implied_math_format=implied_math_format,
        dst_index=dst_idx,
        operands=operands,
        extra_templates=tuple(spec.extra_templates),
    )


# ---------------------------------------------------------------------------
# Ternary
# ---------------------------------------------------------------------------
@pytest.mark.quasar
@parametrize(
    spec=TERNARY_OP_REGISTRY,
    format_dest_acc=_supported_formats,
    implied_math_format=lambda format_dest_acc: _implied_math_formats(
        format_dest_acc[0]
    ),
    extra_template=_extra_template_variants,
)
def test_sfpu_ternary(spec, format_dest_acc, implied_math_format, extra_template):
    """Run a ternary SFPU op: three operands at DEST tiles 0/1/2, result to tile 0.

    ``extra_template`` is one resolved combination of the spec's extra build-time
    template params (e.g. its VECTOR_MODE face selector), swept when the spec declares
    a list-valued field."""
    io, dest_acc = format_dest_acc
    data_format = io.input_format
    # Three input tiles concatenated along rows -> tiles 0, 1, 2 in DEST.
    input_dimensions = [3 * 32, 32]

    torch.manual_seed(0)  # reproducible stimuli
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = _prepare(spec, src_A, src_tiles=(0, 1, 2))

    golden_tensor = _golden(
        spec,
        lambda: get_golden_generator(TernarySFPUGolden)(
            spec.math_op,
            src_A,
            0,  # src1 tile index
            1,  # src2 tile index
            2,  # src3 tile index
            0,  # dst tile index
            32,  # num_iterations: 32 rows = 1 full tile
            input_dimensions,
            data_format,
        ).flatten()[0:_ELEMENTS_PER_TILE],
        src=src_A,
        io=io,
        dest_acc=dest_acc,
        dimensions=input_dimensions,
    )

    _run(
        arity_macro=spec.arity_macro,
        spec=spec,
        io=io,
        dest_acc=dest_acc,
        src_A=src_A,
        src_B=torch.zeros_like(src_A),
        tile_cnt_A=tile_cnt_A,
        tile_count_res=1,
        golden_tensor=golden_tensor,
        unpack_to_dest=io.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes,
        implied_math_format=implied_math_format,
        # The op's resolved extra templates for this variant (its VECTOR_MODE face
        # selector by default). The default golden compares all faces (VectorMode.RC).
        extra_templates=extra_template.templates,
    )
