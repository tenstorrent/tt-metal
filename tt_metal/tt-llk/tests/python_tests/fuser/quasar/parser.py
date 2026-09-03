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
    ELTWISE_DIMS,
    IN0_REQUIRED,
    IN1_REQUIRED,
    INT32_NEEDS_UNPACK_TO_DEST,
    L1_ACC_FORMAT_SUPPORTED,
    LOFI_ONLY,
    MATMUL_DIMS,
    MATMUL_INNER_TILE_DIMS,
    MATMUL_OPERAND_DIMS,
    NO_BROADCAST,
    NO_BROADCAST_ACC_TO_DEST,
    NO_BROADCAST_REUSE_DEST,
    NO_REUSE_DEST,
    NO_TRANSPOSE,
    NO_TRANSPOSE_UNPACK_TO_DEST,
    NO_UNPACK_TO_DEST,
    PACK_NO_L1_ACC,
    REDUCE_PARAMS_REQUIRED,
    SRC_A_DIMS,
    TRANSPOSE_WITHIN_FACE_REQUIRED,
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    PackSchema,
    UnarySfpuMathSchema,
    eltwise_unpacker_rules,
    forced_unpackers,
    reject,
    require_dest_tiles,
    require_src_a_tiles,
)
from helpers.llk_params import (
    BroadcastType,
    MathOperation,
    ReduceDimension,
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

_broadcast_required = reject(
    lambda s, a, b: s.broadcast_type == BroadcastType.None_,
    "UnaryBroadcast requires a broadcast_type",
)

_no_transpose_mismatch = reject(
    lambda s, a, b: s.transpose_faces != s.transpose_within_face,
    "requires both transpose_faces and transpose_within_face to have the same value",
)

_block_full_width = reject(
    lambda s, a, b: s._block_size[1] != a.dimensions[1],
    "block width must be same as operand width",
)

_reduce_col_only = reject(
    lambda s, a, b: s.operation != "Reduce" or s.reduce_dim != ReduceDimension.Column,
    "unpacker can only be paired with a column reduce (operation: Reduce, reduce_dim: REDUCE_COL)",
)

_eltwise_checks = [
    NO_UNPACK_TO_DEST,
    _no_transpose_mismatch,
    *eltwise_unpacker_rules,
    NO_BROADCAST_REUSE_DEST,
    NO_BROADCAST_ACC_TO_DEST,
]

_eltwise_lofi_checks = [*_eltwise_checks, LOFI_ONLY]

UNPACKER_MAP = {
    "UnpackerA": (
        lambda s: UnpackerA(reuse_dest=s.reuse_dest),
        [
            IN0_REQUIRED,
            INT32_NEEDS_UNPACK_TO_DEST,
            NO_TRANSPOSE_UNPACK_TO_DEST,
            _no_transpose_mismatch,
        ],
    ),
    "UnpackerTilizeA": (
        lambda s: UnpackerTilizeA(),
        [
            IN0_REQUIRED,
            NO_BROADCAST,
            NO_TRANSPOSE,
            _block_full_width,
            NO_UNPACK_TO_DEST,
        ],
    ),
    "UnpackerAB": (
        lambda s: UnpackerAB(),
        [IN0_REQUIRED, IN1_REQUIRED, NO_TRANSPOSE],
    ),
    "MatmulUnpacker": (
        lambda s: MatmulUnpacker(),
        [IN0_REQUIRED, IN1_REQUIRED, NO_TRANSPOSE],
    ),
    "ReduceUnpacker": (
        lambda s: ReduceUnpacker(s.reduce_dim, s.reduce_pool),
        [IN0_REQUIRED, IN1_REQUIRED, NO_TRANSPOSE],
    ),
    "TransposeDestUnpacker": (
        lambda s: TransposeDestUnpacker(),
        [],
    ),
    "UnaryBroadcastUnpacker": (
        lambda s: UnaryBroadcastUnpacker(),
        [IN0_REQUIRED, _broadcast_required, NO_TRANSPOSE, NO_UNPACK_TO_DEST],
    ),
    "UnpackReduceTilize": (
        lambda s: UnpackReduceTilize(s.reduce_dim, s.reduce_pool),
        [
            IN0_REQUIRED,
            IN1_REQUIRED,
            NO_TRANSPOSE,
            NO_UNPACK_TO_DEST,
            _reduce_col_only,
            require_src_a_tiles((32, 32)),
        ],
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
            NO_REUSE_DEST,
            forced_unpackers("UnpackerA", "UnpackerTilizeA"),
            NO_BROADCAST,
            NO_TRANSPOSE_UNPACK_TO_DEST,
            _no_transpose_mismatch,
        ],
    ),
    "Matmul": (
        lambda s: MatmulFpu(),
        [
            NO_REUSE_DEST,
            NO_BROADCAST,
            MATMUL_OPERAND_DIMS,
            forced_unpackers("MatmulUnpacker"),
            MATMUL_INNER_TILE_DIMS,
        ],
    ),
    "Reduce": (
        lambda s: ReduceFpu(s.reduce_dim, s.reduce_pool),
        [
            NO_REUSE_DEST,
            NO_BROADCAST,
            REDUCE_PARAMS_REQUIRED,
            forced_unpackers("ReduceUnpacker", "UnpackReduceTilize"),
        ],
    ),
    "TransposeDest": (
        lambda s: TransposeDestFpu(),
        [
            NO_REUSE_DEST,
            NO_BROADCAST,
            TRANSPOSE_WITHIN_FACE_REQUIRED,
            forced_unpackers("TransposeDestUnpacker"),
            require_dest_tiles((32, 32)),
        ],
    ),
    "UnaryBroadcast": (
        lambda s: UnaryBroadcastFpu(),
        [
            _broadcast_required,
            NO_REUSE_DEST,
            NO_BROADCAST_ACC_TO_DEST,
            NO_UNPACK_TO_DEST,
            require_src_a_tiles((32, 32)),
            forced_unpackers("UnaryBroadcastUnpacker"),
        ],
    ),
}

# Quasar has no strided pack path for tiny tiles, so the shared full-tile rule carries
# an arch-specific hint here.
_untilize_full_tile = reject(
    lambda s, output: output.tile_shape.total_num_faces() != 4,
    "PackUntilize supports only 32x32 output tiles, tiny tiles need strided pack",
)

PACKER_MAP = {
    "Packer": (Packer, [L1_ACC_FORMAT_SUPPORTED]),
    "MatmulPacker": (MatmulPacker, [L1_ACC_FORMAT_SUPPORTED]),
    "PackUntilize": (PackUntilize, [_untilize_full_tile, PACK_NO_L1_ACC]),
}

OUTPUT_DIMS = {
    "Elwadd": ELTWISE_DIMS,
    "Elwmul": ELTWISE_DIMS,
    "Elwsub": ELTWISE_DIMS,
    "Datacopy": SRC_A_DIMS,
    "Matmul": MATMUL_DIMS,
    "Reduce": SRC_A_DIMS,
    "TransposeDest": SRC_A_DIMS,
    "UnaryBroadcast": SRC_A_DIMS,
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
    _iteration_step: ClassVar[int] = 4


class QuasarBinarySfpuMathSchema(BinarySfpuMathSchema):
    _sfpu_cls: ClassVar = BinarySfpu
    _sfpu_ops: ClassVar = BINARY_SFPU_OPS
    _iteration_step: ClassVar[int] = 4


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
    dest_consuming_operations: ClassVar = frozenset({"TransposeDest"})

    math: List[MathSchema] = Field(..., min_length=1)
    pack: List[PackEntrySchema] = Field(..., min_length=1)
