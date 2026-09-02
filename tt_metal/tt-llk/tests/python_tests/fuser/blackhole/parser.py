# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole fuser config parser.

All the arch-specific configuration lives in plain dicts at the top of the file.
The shared base classes in fuser.validator read these dicts to run validation and
build runtime objects, so this file only needs to define the data and thin subclasses.

To add a new FPU op, add one entry to FPU_MAP with its factory and checks,
and one to OUTPUT_DIMS for output dimension computation.
"""

from typing import Annotated, ClassVar, List, Union

from fuser.validator import (
    DATACOPY_TILE_32X32_ONLY,
    DEST_TO_SRCA_NEEDS_ACC,
    ELTWISE_DIMS,
    INT32_NEEDS_UNPACK_TO_DEST,
    L1_ACC_FORMAT_SUPPORTED,
    LOFI_ONLY,
    MATMUL_DIMS,
    MATMUL_INNER_TILE_DIMS,
    MATMUL_OPERAND_DIMS,
    NO_BROADCAST,
    NO_BROADCAST_ACC_TO_DEST,
    NO_BROADCAST_REUSE_DEST,
    NO_COL_ROW_BCAST_32X16,
    NO_REUSE_DEST,
    NO_TRANSPOSE,
    NO_TRANSPOSE_FACES,
    NO_TRANSPOSE_UNPACK_TO_DEST,
    NO_UNPACK_TO_DEST,
    PACK_FULL_TILE_ONLY,
    PACK_NO_BLOCK_FLOAT,
    PACK_NO_L1_ACC,
    REDUCE_PARAMS_REQUIRED,
    SCALAR_BCAST_NO_TRANSPOSE_FACES,
    SRC_A_DIMS,
    SUB_BCAST_COL_REQUIRED,
    SUPPORTED_SRC_A_TILE,
    TRANSPOSE_NEEDS_FULL_TILE,
    TRANSPOSE_WITHIN_FACE_REQUIRED,
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    PackSchema,
    UnarySfpuMathSchema,
    eltwise_unpacker_rules,
    forced_unpackers,
    require_src_a_tiles,
)
from helpers.llk_params import (
    MathOperation,
    Tilize,
)
from pydantic import Field

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.matmul_no_mop import MatmulNoMopFpu
from .fpu.reduce import ReduceFpu
from .fpu.reduce_block_max import ReduceBlockMaxFpu
from .fpu.reduce_block_max_runtime import ReduceBlockMaxRuntimeFpu
from .fpu.sub_bcast_col_custom import SubBcastColCustomFpu
from .fpu.transpose_dest import TransposeDestFpu
from .packer.packer import Packer
from .packer.untilize import PackUntilize
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.reduce_block_max import ReduceBlockMaxUnpacker
from .unpacker.reduce_block_max_runtime import ReduceBlockMaxRuntimeUnpacker
from .unpacker.sub_bcast_col_custom import SubBcastColCustomUnpacker
from .unpacker.tilize_a import UnpackerTilizeA
from .unpacker.transpose_dest import TransposeDestUnpacker
from .unpacker.unpack_a import UnpackerA
from .unpacker.unpack_ab import UnpackerAB

UNPACKER_MAP = {
    "UnpackerA": (
        lambda s: UnpackerA(),
        [INT32_NEEDS_UNPACK_TO_DEST, NO_TRANSPOSE_UNPACK_TO_DEST],
    ),
    "UnpackerAB": (
        lambda s: UnpackerAB(),
        [SCALAR_BCAST_NO_TRANSPOSE_FACES],
    ),
    "UnpackerTilizeA": (
        lambda s: UnpackerTilizeA(),
        [NO_BROADCAST, NO_TRANSPOSE],
    ),
    "MatmulUnpacker": (
        lambda s: MatmulUnpacker(),
        [NO_TRANSPOSE_FACES],
    ),
    "ReduceUnpacker": (
        lambda s: ReduceUnpacker(s.reduce_dim, s.reduce_pool),
        None,
    ),
    "ReduceBlockMaxUnpacker": (
        lambda s: ReduceBlockMaxUnpacker(),
        None,
    ),
    "ReduceBlockMaxRuntimeUnpacker": (
        lambda s: ReduceBlockMaxRuntimeUnpacker(),
        None,
    ),
    "SubBcastColCustomUnpacker": (
        lambda s: SubBcastColCustomUnpacker(),
        None,
    ),
    "TransposeDestUnpacker": (
        lambda s: TransposeDestUnpacker(),
        None,
    ),
}

_eltwise_checks = [
    DEST_TO_SRCA_NEEDS_ACC,
    *eltwise_unpacker_rules,
    SUPPORTED_SRC_A_TILE,
    TRANSPOSE_NEEDS_FULL_TILE,
    NO_UNPACK_TO_DEST,
    NO_COL_ROW_BCAST_32X16,
    NO_BROADCAST_REUSE_DEST,
    NO_BROADCAST_ACC_TO_DEST,
]
_eltwise_lofi_checks = [*_eltwise_checks, LOFI_ONLY]

_matmul_checks = [
    NO_REUSE_DEST,
    MATMUL_OPERAND_DIMS,
    forced_unpackers("MatmulUnpacker"),
    MATMUL_INNER_TILE_DIMS,
]

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
            SUPPORTED_SRC_A_TILE,
            DATACOPY_TILE_32X32_ONLY,
        ],
    ),
    "Matmul": (
        lambda s: MatmulFpu(),
        _matmul_checks,
    ),
    "MatmulNoMop": (
        lambda s: MatmulNoMopFpu(),
        _matmul_checks,
    ),
    "Reduce": (
        lambda s: ReduceFpu(s.reduce_dim, s.reduce_pool),
        [
            NO_REUSE_DEST,
            REDUCE_PARAMS_REQUIRED,
            forced_unpackers("ReduceUnpacker"),
            SUPPORTED_SRC_A_TILE,
        ],
    ),
    "ReduceBlockMax": (
        lambda s: ReduceBlockMaxFpu(),
        [
            NO_REUSE_DEST,
            forced_unpackers("ReduceBlockMaxUnpacker"),
            require_src_a_tiles((32, 32), (16, 32)),
        ],
    ),
    "ReduceBlockMaxRuntime": (
        lambda s: ReduceBlockMaxRuntimeFpu(),
        [
            NO_REUSE_DEST,
            forced_unpackers("ReduceBlockMaxRuntimeUnpacker"),
            require_src_a_tiles((32, 32), (16, 32)),
        ],
    ),
    "SubBcastColCustom": (
        lambda s: SubBcastColCustomFpu(),
        [
            NO_REUSE_DEST,
            forced_unpackers("SubBcastColCustomUnpacker"),
            SUB_BCAST_COL_REQUIRED,
            require_src_a_tiles((32, 32)),
        ],
    ),
    "TransposeDest": (
        lambda s: TransposeDestFpu(),
        [
            NO_REUSE_DEST,
            forced_unpackers("TransposeDestUnpacker"),
            TRANSPOSE_WITHIN_FACE_REQUIRED,
            require_src_a_tiles((32, 32)),
        ],
    ),
}

PACKER_MAP = {
    "Packer": (Packer, [L1_ACC_FORMAT_SUPPORTED]),
    "PackUntilize": (
        PackUntilize,
        [PACK_FULL_TILE_ONLY, PACK_NO_BLOCK_FLOAT, PACK_NO_L1_ACC],
    ),
}

OUTPUT_DIMS = {
    "Elwadd": ELTWISE_DIMS,
    "Elwmul": ELTWISE_DIMS,
    "Elwsub": ELTWISE_DIMS,
    "Datacopy": SRC_A_DIMS,
    "Matmul": MATMUL_DIMS,
    "MatmulNoMop": MATMUL_DIMS,
    "Reduce": SRC_A_DIMS,
    "ReduceBlockMax": SRC_A_DIMS,
    "ReduceBlockMaxRuntime": SRC_A_DIMS,
    "SubBcastColCustom": SRC_A_DIMS,
    "TransposeDest": SRC_A_DIMS,
}

UNARY_SFPU_OPS = {
    MathOperation.Abs,
    MathOperation.Acosh,
    MathOperation.Asinh,
    MathOperation.Atanh,
    MathOperation.Celu,
    MathOperation.Cos,
    MathOperation.Elu,
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Fill,
    MathOperation.Gelu,
    MathOperation.Hardsigmoid,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Neg,
    MathOperation.Reciprocal,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
    MathOperation.Rsqrt,
    MathOperation.Silu,
    MathOperation.Sin,
    MathOperation.Sqrt,
    MathOperation.Square,
    MathOperation.Tanh,
    MathOperation.Threshold,
}

BINARY_SFPU_OPS = {
    MathOperation.SfpuElwadd,
    MathOperation.SfpuElwmul,
    MathOperation.SfpuElwsub,
    MathOperation.SfpuElwLeftShift,
    MathOperation.SfpuElwRightShift,
    MathOperation.SfpuElwLogicalRightShift,
    MathOperation.SfpuXlogy,
    MathOperation.SfpuAddTopRow,
}


class FpuMathSchema(FpuMathSchemaBase):
    _fpu_map: ClassVar = FPU_MAP
    _unpacker_map: ClassVar = UNPACKER_MAP
    _output_dims: ClassVar = OUTPUT_DIMS


class BlackholeUnarySfpuMathSchema(UnarySfpuMathSchema):
    _sfpu_cls: ClassVar = UnarySfpu
    _sfpu_ops: ClassVar = UNARY_SFPU_OPS


class BlackholeBinarySfpuMathSchema(BinarySfpuMathSchema):
    _sfpu_cls: ClassVar = BinarySfpu
    _sfpu_ops: ClassVar = BINARY_SFPU_OPS


MathSchema = Annotated[
    Union[FpuMathSchema, BlackholeUnarySfpuMathSchema, BlackholeBinarySfpuMathSchema],
    Field(discriminator="type"),
]


class BlackholePackSchema(PackSchema):
    _packer_map: ClassVar = PACKER_MAP


PackEntrySchema = Union[
    BlackholeUnarySfpuMathSchema, BlackholeBinarySfpuMathSchema, BlackholePackSchema
]


class OperationSchema(OperationSchemaBase):
    dest_consuming_operations: ClassVar = frozenset({"TransposeDest"})

    math: List[MathSchema] = Field(..., min_length=1)
    pack: List[PackEntrySchema] = Field(..., min_length=1)
    bh_tilize: Tilize = Tilize.No

    def _arch_validate(self):
        unique_unpackers = {
            m.unpacker
            for m in self.math
            if isinstance(m, FpuMathSchema) and m.unpacker is not None
        }

        if "UnpackerTilizeA" in unique_unpackers:
            self.bh_tilize = Tilize.Yes

        if len(unique_unpackers) > 1 and "UnpackerTilizeA" in unique_unpackers:
            raise ValueError(
                "UnpackerTilizeA cannot be combined with other unpackers on BH"
            )

    def _arch_kwargs(self) -> dict:
        return {"bh_tilize": self.bh_tilize}
