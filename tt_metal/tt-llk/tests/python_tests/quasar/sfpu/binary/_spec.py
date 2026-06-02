import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

from helpers.format_config import DataFormat

from .._spec import BaseOpSpec


@dataclass(frozen=True)
class BinaryInputOutputFormat:
    input_a_format: DataFormat
    input_b_format: DataFormat
    output_format: DataFormat


def binary_input_output_formats(
    inputs_a: List[DataFormat],
    inputs_b: List[DataFormat],
    outputs: List[DataFormat],
) -> List[BinaryInputOutputFormat]:
    return [
        BinaryInputOutputFormat(a, b, o)
        for a, b, o in itertools.product(inputs_a, inputs_b, outputs)
    ]


@dataclass(kw_only=True)
class BinaryOpSpec(BaseOpSpec):
    formats: List[BinaryInputOutputFormat]
    # DEST tile-layout sweep: each entry is (src0_tile, src1_tile, dst_tile) and
    # the driver places operands / packs the result accordingly (emitting
    # SFPU_BINARY_OPERANDS and DST_INDEX per layout). Use for ops that index by
    # tile (the sfpi float ops) to cover result-over-operand aliasing. Leave None
    # for offset-style ops (the Int32 ops) that place operands themselves via their
    # own SFPU_BINARY_OPERANDS define; those run a single layout (tiles 0/1 -> 0).
    tile_layouts: Optional[List[Tuple[int, int, int]]] = None
    arity_macro: str = "SFPU_BINARY_OP"


# Float formats shared by the sfpi binary ops.
_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]

# DEST tile layouts (src0, src1, dst) covering result-over-operand aliasing: result
# written back over the dividend tile, over the divisor tile, and to a separate tile
# with both operand offsets non-zero.
_FLOAT_TILE_LAYOUTS = [(0, 1, 0), (0, 1, 1), (2, 3, 0)]


def float_binary_op_spec(
    *,
    name,
    math_op,
    binop,
    stimuli,
    prepare=None,
    verify=None,
    tile_layouts=_FLOAT_TILE_LAYOUTS,
):
    """Build a BinaryOpSpec for an sfpi float binary op (``calculate_sfpu_binary``).

    ``binop`` is the ``ckernel::BinaryOp`` enum name ("MUL" / "DIV"). These ops
    index by tile, are templated on the build's ``is_fp32_dest_acc_en``, and need an
    op-specific init call. They sweep the full input x output float-format matrix.

    ``tile_layouts`` controls the DEST tile-layout sweep and defaults to the full
    aliasing set (``_FLOAT_TILE_LAYOUTS``). Ops that don't need the extra coverage can
    pass a shorter list or ``None`` (single default layout: operands at tiles 0/1,
    result to tile 0). ``prepare`` / ``verify`` are optional hooks (see BaseOpSpec)."""
    op = (
        f"ckernel::sfpu::calculate_sfpu_binary<false, ckernel::BinaryOp::{binop}, "
        "is_fp32_dest_acc_en>"
    )
    init = f"ckernel::sfpu::sfpu_binary_init<false, ckernel::BinaryOp::{binop}>();"
    return BinaryOpSpec(
        name=name,
        math_op=math_op,
        include_header="llk_sfpu/ckernel_sfpu_binary.h",
        # Tile-index ops: the driver supplies SFPU_BINARY_OPERANDS per tile layout,
        # so only the op call and its init are declared here.
        sfpu_defines=lambda formats: {
            "SFPU_OP_CALL": op,
            "SFPU_INIT": init,
        },
        # Operands share a format (the driver feeds io.input_format to both), so sweep
        # the full input x output matrix with input_a == input_b.
        formats=[
            BinaryInputOutputFormat(i, i, o)
            for i in _FLOAT_FORMATS
            for o in _FLOAT_FORMATS
        ],
        tile_layouts=tile_layouts,
        stimuli=stimuli,
        prepare=prepare,
        verify=verify,
    )
