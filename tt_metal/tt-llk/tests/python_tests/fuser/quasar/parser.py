# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar fuser config parser.

Supports: eltwise binary (Elwadd/Elwmul/Elwsub), datacopy, matmul, reduce,
unary SFPU, binary SFPU, eltwise broadcast (COL/ROW/SCALAR),
unary broadcast (COL/ROW/SCALAR).
Unsupported on Quasar: MatmulNoMop, ReduceBlockMax, ReduceBlockMaxRuntime,
SubBcastColCustom.
"""

from typing import Annotated, ClassVar, List, Union

from fuser.validator import (
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    PackSchema,
    UnarySfpuMathSchema,
    _tile_dims,
)
from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    EltwiseBinaryReuseDestType,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    Transpose,
    UnpackToDest,
)
from pydantic import Field

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.reduce import ReduceFpu
from .fpu.transpose_dest import TransposeDestFpu
from .fpu.unary_broadcast import UnaryBroadcastFpu
from .packer.matmul import MatmulPacker
from .packer.packer import Packer
from .packer.untilize import PackUntilize
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.reduce_tilize_a import UnpackReduceTilize
from .unpacker.tilize_a import UnpackerTilizeA
from .unpacker.transpose_dest import TransposeDestUnpacker
from .unpacker.unary_broadcast import UnaryBroadcastUnpacker
from .unpacker.unpack_a import UnpackerA
from .unpacker.unpack_ab import UnpackerAB

_no_broadcast = (
    lambda s, a, b: s.broadcast_type != BroadcastType.None_,
    "Quasar does not support broadcast in fuser",
)

_no_unpack_to_dest = (
    lambda s, a, b: s.unpack_to_dest == UnpackToDest.Yes,
    "unpack_to_dest is not supported for this kernel",
)

_broadcast_required = (
    lambda s, a, b: s.broadcast_type == BroadcastType.None_,
    "UnaryBroadcast requires a broadcast_type",
)

_no_transpose_unpack_to_dest = (
    lambda s, a, b: s.unpack_to_dest == UnpackToDest.Yes
    and (
        s.unpack_transpose_faces == Transpose.Yes
        or s.unpack_transpose_within_face == Transpose.Yes
    ),
    "Quasar does not support transpose with unpack_to_dest",
)

_no_transpose = (
    lambda s, a, b: s.unpack_transpose_faces == Transpose.Yes
    or s.unpack_transpose_within_face == Transpose.Yes,
    "Quasar does not support transpose for this unpacker",
)

_no_transpose_mismatch = (
    lambda s, a, b: s.unpack_transpose_faces != s.unpack_transpose_within_face,
    "Quasar requires both transpose_faces and transpose_within_face to have the same value",
)

_eltwise_unpacker_reuse = (
    lambda s, a, b: s.unpacker is not None
    and s.reuse_dest != EltwiseBinaryReuseDestType.NONE
    and s.unpacker != "UnpackerA",
    "Eltwise with reuse_dest: unpacker must be UnpackerA",
)

_eltwise_unpacker_default = (
    lambda s, a, b: s.unpacker is not None
    and s.reuse_dest == EltwiseBinaryReuseDestType.NONE
    and s.unpacker != "UnpackerAB",
    "Eltwise: unpacker must be UnpackerAB",
)

_no_broadcast_reuse_dest = (
    lambda s, a, b: s.broadcast_type != BroadcastType.None_
    and s.reuse_dest != EltwiseBinaryReuseDestType.NONE,
    "Quasar broadcast does not support reuse_dest",
)

_no_broadcast_acc_to_dest = (
    lambda s, a, b: s.broadcast_type != BroadcastType.None_
    and s.acc_to_dest == AccToDest.Yes,
    "Quasar broadcast does not support acc_to_dest",
)

_eltwise_checks = [
    _no_transpose_unpack_to_dest,
    _no_transpose_mismatch,
    _eltwise_unpacker_reuse,
    _eltwise_unpacker_default,
    _no_broadcast_reuse_dest,
    _no_broadcast_acc_to_dest,
]

_lofi_only = (
    lambda s, a, b: s.math_fidelity != MathFidelity.LoFi,
    "only LoFi math fidelity is supported for this operation",
)

_eltwise_lofi_checks = [*_eltwise_checks, _lofi_only]

_no_reuse_dest = (
    lambda s, a, b: s.reuse_dest != EltwiseBinaryReuseDestType.NONE,
    "reuse_dest is only supported for Eltwise operations",
)

_datacopy_unpacker = (
    lambda s, a, b: s.unpacker is not None
    and s.unpacker not in {"UnpackerA", "UnpackerTilizeA"},
    "Datacopy: unpacker must be UnpackerA or UnpackerTilizeA",
)

_block_full_width = (
    lambda s, a, b: s._block_size[1] != a.dimensions[1],
    "block width must be same as operand width",
)

_forced_unpacker = lambda name: (
    lambda s, a, b: s.unpacker is not None and s.unpacker != name,
    f"unpacker must be {name}",
)

_forced_unpackers = lambda names: (
    lambda s, a, b: s.unpacker is not None and s.unpacker not in names,
    f"unpacker must be one of: {', '.join(names)}",
)

_reduce_col_only = (
    lambda s, a, b: s.operation != "Reduce" or s.reduce_dim != ReduceDimension.Column,
    "unpacker can only be paired with a column reduce (operation: Reduce, reduce_dim: REDUCE_COL)",
)

_matmul_dim_check = (
    lambda s, a, b: a.dimensions[1] != b.dimensions[0],
    "Matmul: incompatible dimensions for src_a and src_b",
)

_matmul_inner_dims = (
    lambda s, a, b: a.tile_shape.total_col_dim() != b.tile_shape.total_row_dim(),
    "Matmul tile inner dimensions must match: in0 cols must equal in1 rows",
)

_reduce_params = (
    lambda s, a, b: s.reduce_pool is None or s.reduce_dim is None,
    "Reduce requires both reduce_pool and reduce_dim",
)

_only_32x32_tile = (
    lambda s, a, b: _tile_dims(a.tile_shape) != (32, 32),
    "Only (32, 32) tiles are supported for this operation",
)

UNPACKER_MAP = {
    "UnpackerA": (
        lambda s: UnpackerA(reuse_dest=s.reuse_dest),
        [_no_transpose_unpack_to_dest, _no_transpose_mismatch],
    ),
    "UnpackerTilizeA": (
        lambda s: UnpackerTilizeA(),
        [_no_transpose, _block_full_width, _no_unpack_to_dest],
    ),
    "UnpackerAB": (
        lambda s: UnpackerAB(),
        [_no_transpose],
    ),
    "MatmulUnpacker": (
        lambda s: MatmulUnpacker(),
        [_no_transpose],
    ),
    "ReduceUnpacker": (
        lambda s: ReduceUnpacker(s.reduce_dim, s.reduce_pool),
        [_no_transpose],
    ),
    "TransposeDestUnpacker": (
        lambda s: TransposeDestUnpacker(),
        None,
    ),
    "UnaryBroadcastUnpacker": (
        lambda s: UnaryBroadcastUnpacker(),
        [_broadcast_required, _no_transpose, _no_unpack_to_dest],
    ),
    "UnpackReduceTilize": (
        lambda s: UnpackReduceTilize(),
        [_no_transpose, _no_unpack_to_dest, _reduce_col_only, _only_32x32_tile],
    ),
}

FPU_MAP = {
    "Elwadd": (
        lambda s: EltwiseFpu(MathOperation.Elwadd),
        _eltwise_lofi_checks,
    ),
    "Elwmul": (
        lambda s: EltwiseFpu(MathOperation.Elwmul),
        _eltwise_checks,
    ),
    "Elwsub": (
        lambda s: EltwiseFpu(MathOperation.Elwsub),
        _eltwise_lofi_checks,
    ),
    "Datacopy": (
        lambda s: DatacopyFpu(),
        [
            _no_reuse_dest,
            _datacopy_unpacker,
            _no_broadcast,
            _no_transpose_unpack_to_dest,
            _no_transpose_mismatch,
        ],
    ),
    "Matmul": (
        lambda s: MatmulFpu(),
        [
            _no_reuse_dest,
            _no_broadcast,
            _matmul_dim_check,
            _forced_unpacker("MatmulUnpacker"),
            _matmul_inner_dims,
        ],
    ),
    "Reduce": (
        lambda s: ReduceFpu(s.reduce_dim, s.reduce_pool),
        [
            _no_reuse_dest,
            _no_broadcast,
            _reduce_params,
            _forced_unpackers(("ReduceUnpacker", "UnpackReduceTilize")),
        ],
    ),
    "TransposeDest": (
        lambda s: TransposeDestFpu(),
        [
            _no_reuse_dest,
            _no_broadcast,
            _no_transpose,
            _forced_unpacker("TransposeDestUnpacker"),
            _only_32x32_tile,
        ],
    ),
    "UnaryBroadcast": (
        lambda s: UnaryBroadcastFpu(),
        [
            _broadcast_required,
            _no_reuse_dest,
            _no_broadcast_acc_to_dest,
            _no_unpack_to_dest,
            _only_32x32_tile,
            _forced_unpacker("UnaryBroadcastUnpacker"),
        ],
    ),
}

_l1_acc_format = (
    lambda s, output: s.pack_l1_accumulation == L1Accumulation.Yes
    and not output.data_format.supports_l1_accumulation(),
    "Output data format does not support L1 accumulation",
)

_untilize_full_tile = (
    lambda s, output: output.tile_shape.total_num_faces() != 4,
    "PackUntilize supports only 32x32 output tiles, tiny tiles need strided pack",
)

_untilize_no_l1_acc = (
    lambda s, output: s.pack_l1_accumulation == L1Accumulation.Yes,
    "PackUntilize does not support L1 accumulation",
)

PACKER_MAP = {
    "Packer": (Packer, [_l1_acc_format]),
    "MatmulPacker": (MatmulPacker, [_l1_acc_format]),
    "PackUntilize": (PackUntilize, [_untilize_full_tile, _untilize_no_l1_acc]),
}

_eltwise_dims = lambda a, b: (min(a[0], b[0]), min(a[1], b[1]))
_matmul_dims = lambda a, b: (a[0], b[1])
_src_a_dims = lambda a, b: a
_src_b_dims = lambda a, b: b

OUTPUT_DIMS = {
    "Elwadd": _eltwise_dims,
    "Elwmul": _eltwise_dims,
    "Elwsub": _eltwise_dims,
    "Datacopy": _src_a_dims,
    "Matmul": _matmul_dims,
    "Reduce": _src_a_dims,
    "TransposeDest": _src_a_dims,
    "UnaryBroadcast": _src_b_dims,
}


UNARY_SFPU_OPS = {
    MathOperation.Abs,
    MathOperation.Exp,
    MathOperation.Gelu,
    MathOperation.Reciprocal,
    MathOperation.Relu,
    MathOperation.Rsqrt,
    MathOperation.Sigmoid,
    MathOperation.Silu,
    MathOperation.Sqrt,
    MathOperation.Square,
    MathOperation.Tanh,
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
}

BINARY_SFPU_OPS = {
    MathOperation.SfpuElwadd,
    MathOperation.SfpuElwmul,
    MathOperation.SfpuElwdiv,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuElwGe,
}


class FpuMathSchema(FpuMathSchemaBase):
    _fpu_map: ClassVar = FPU_MAP
    _unpacker_map: ClassVar = UNPACKER_MAP
    _output_dims: ClassVar = OUTPUT_DIMS


class QuasarUnarySfpuMathSchema(UnarySfpuMathSchema):
    _sfpu_cls: ClassVar = UnarySfpu
    _sfpu_ops: ClassVar = UNARY_SFPU_OPS


class QuasarBinarySfpuMathSchema(BinarySfpuMathSchema):
    _sfpu_cls: ClassVar = BinarySfpu
    _sfpu_ops: ClassVar = BINARY_SFPU_OPS


MathSchema = Annotated[
    Union[FpuMathSchema, QuasarUnarySfpuMathSchema, QuasarBinarySfpuMathSchema],
    Field(discriminator="type"),
]


class QuasarPackSchema(PackSchema):
    _packer_map: ClassVar = PACKER_MAP


PackEntrySchema = Union[
    QuasarUnarySfpuMathSchema, QuasarBinarySfpuMathSchema, QuasarPackSchema
]


class OperationSchema(OperationSchemaBase):
    math: List[MathSchema] = Field(..., min_length=1)
    pack: List[PackEntrySchema] = Field(..., min_length=1)

    def _arch_validate(self):
        if (
            self.math
            and isinstance(self.math[0], FpuMathSchema)
            and self.math[0].operation == "TransposeDest"
        ):
            raise ValueError(
                "TransposeDest cannot be the first math operation: Dst must already contain data"
            )
