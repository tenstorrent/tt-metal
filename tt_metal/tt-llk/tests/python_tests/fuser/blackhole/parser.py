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
    SUPPORTED_TILE_SIZES,
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    UnarySfpuMathSchema,
    _has_transpose,
    _tile_dims,
)
from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    DataFormat,
    EltwiseBinaryReuseDestType,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    Tilize,
    UnpackToDest,
)
from pydantic import Field

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.reduce import ReduceFpu
from .fpu.reduce_block_max import ReduceBlockMaxFpu
from .fpu.reduce_block_max_runtime import ReduceBlockMaxRuntimeFpu
from .packer.packer import Packer
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.reduce_block_max import ReduceBlockMaxUnpacker
from .unpacker.reduce_block_max_runtime import ReduceBlockMaxRuntimeUnpacker
from .unpacker.tilize_a import UnpackerTilizeA
from .unpacker.unpack_a import UnpackerA
from .unpacker.unpack_ab import UnpackerAB

_int32_unpack_to_dest = (
    lambda s, a, b: a.data_format == DataFormat.Int32
    and s.unpack_to_dest != UnpackToDest.Yes,
    "Int32 src_a requires unpack_to_dest: Yes (SrcA/SrcB registers are 19-bit wide)",
)

_ab_checks = [
    (
        lambda s, a, b: s.broadcast_type == BroadcastType.Scalar
        and s.unpack_transpose_faces.value,
        "SrcA transpose is not supported with scalar broadcast",
    ),
]

_tilize_a_checks = [
    (
        lambda s, a, b: s.broadcast_type != BroadcastType.None_,
        "UnpackerTilizeA does not support broadcast",
    ),
    (
        lambda s, a, b: s.unpack_transpose_faces.value
        or s.unpack_transpose_within_face.value,
        "UnpackerTilizeA does not support transpose",
    ),
]

_matmul_checks = [
    (
        lambda s, a, b: s.unpack_transpose_within_face != s.unpack_transpose_faces,
        "MatmulUnpacker does not support different values for transpose_faces and transpose_within_face",
    ),
]

UNPACKER_MAP = {
    "UnpackerA": (
        lambda s: UnpackerA(),
        [_int32_unpack_to_dest],
    ),
    "UnpackerAB": (
        lambda s: UnpackerAB(),
        _ab_checks,
    ),
    "UnpackerTilizeA": (
        lambda s: UnpackerTilizeA(),
        _tilize_a_checks,
    ),
    "MatmulUnpacker": (
        lambda s: MatmulUnpacker(),
        _matmul_checks,
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
}

_no_reuse_dest = (
    lambda s, a, b: s.reuse_dest != EltwiseBinaryReuseDestType.NONE,
    "reuse_dest is only supported for Eltwise operations",
)

_dest_to_srca_needs_acc = (
    lambda s, a, b: s.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
    and s.acc_to_dest != AccToDest.Yes,
    "reuse_dest DEST_TO_SRCA requires acc_to_dest: true",
)

_lofi_only = (
    lambda s, a, b: s.math_fidelity != MathFidelity.LoFi,
    "only LoFi math fidelity is supported for this operation",
)

_matmul_dim_check = (
    lambda s, a, b: a.dimensions[1] != b.dimensions[0],
    "Matmul: incompatible dimensions for src_a and src_b",
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

_datacopy_unpacker = (
    lambda s, a, b: s.unpacker is not None
    and s.unpacker not in {"UnpackerA", "UnpackerTilizeA"},
    "Datacopy: unpacker must be UnpackerA or UnpackerTilizeA",
)

_forced_unpacker = lambda name: (
    lambda s, a, b: s.unpacker is not None and s.unpacker != name,
    f"unpacker must be {name}",
)

_reduce_params = (
    lambda s, a, b: s.reduce_pool is None or s.reduce_dim is None,
    "Reduce requires both reduce_pool and reduce_dim",
)

_unsupported_tile = (
    lambda s, a, b: _tile_dims(a.tile_shape) not in SUPPORTED_TILE_SIZES,
    "Unsupported src_a tile shape",
)

_eltwise_transpose_tile = (
    lambda s, a, b: _has_transpose(s) and _tile_dims(a.tile_shape) != (32, 32),
    "Only (32, 32) tiles are supported for eltwise with transpose",
)

_eltwise_bcast_32x16 = (
    lambda s, a, b: s.broadcast_type in (BroadcastType.Column, BroadcastType.Row)
    and _tile_dims(a.tile_shape) == (32, 16),
    "32x16 tiles are not supported for eltwise with column/row broadcast",
)

_datacopy_only_32x32 = (
    lambda s, a, b: _tile_dims(a.tile_shape) != (32, 32)
    and (
        _has_transpose(s)
        or s.broadcast_type in (BroadcastType.Column, BroadcastType.Row)
        or (
            s.unpack_to_dest == UnpackToDest.Yes
            and s.broadcast_type != BroadcastType.None_
        )
    ),
    "Only (32, 32) tiles are supported for Datacopy with transpose, col/row broadcast, or unpack-to-dest with broadcast",
)

_matmul_in0_16x16 = (
    lambda s, a, b: _tile_dims(a.tile_shape) == (16, 16),
    "Matmul in0 tile shape (16, 16) is not supported",
)

_matmul_in1_tile = (
    lambda s, a, b: _tile_dims(b.tile_shape) not in ((32, 32), (32, 16), (16, 32)),
    "Matmul in1 tile shape must be (32, 32), (32, 16), or (16, 32)",
)

_matmul_inner_dims = (
    lambda s, a, b: a.tile_shape.total_col_dim() != b.tile_shape.total_row_dim(),
    "Matmul tile inner dimensions must match: in0 cols must equal in1 rows",
)

_eltwise_checks = [
    _dest_to_srca_needs_acc,
    _eltwise_unpacker_reuse,
    _eltwise_unpacker_default,
    _unsupported_tile,
    _eltwise_transpose_tile,
    _eltwise_bcast_32x16,
]
_eltwise_lofi_checks = [*_eltwise_checks, _lofi_only]

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
        [_no_reuse_dest, _datacopy_unpacker, _unsupported_tile, _datacopy_only_32x32],
    ),
    "Matmul": (
        lambda s: MatmulFpu(),
        [
            _no_reuse_dest,
            _matmul_dim_check,
            _forced_unpacker("MatmulUnpacker"),
            _matmul_in0_16x16,
            _matmul_in1_tile,
            _matmul_inner_dims,
        ],
    ),
    "Reduce": (
        lambda s: ReduceFpu(s.reduce_dim, s.reduce_pool),
        [
            _no_reuse_dest,
            _reduce_params,
            _forced_unpacker("ReduceUnpacker"),
            _unsupported_tile,
        ],
    ),
    "ReduceBlockMax": (
        lambda s: ReduceBlockMaxFpu(),
        [_no_reuse_dest, _forced_unpacker("ReduceBlockMaxUnpacker"), _unsupported_tile],
    ),
    "ReduceBlockMaxRuntime": (
        lambda s: ReduceBlockMaxRuntimeFpu(),
        [
            _no_reuse_dest,
            _forced_unpacker("ReduceBlockMaxRuntimeUnpacker"),
            _unsupported_tile,
        ],
    ),
}

_l1_acc_format = (
    lambda s, output: s.pack_l1_accumulation == L1Accumulation.Yes
    and not output.data_format.supports_l1_accumulation(),
    "Output data format does not support L1 accumulation",
)

PACKER_MAP = {
    "Packer": (Packer, [_l1_acc_format]),
}

_eltwise_dims = lambda a, b: (min(a[0], b[0]), min(a[1], b[1]))
_matmul_dims = lambda a, b: (a[0], b[1])
_src_a_dims = lambda a, b: a

OUTPUT_DIMS = {
    "Elwadd": _eltwise_dims,
    "Elwmul": _eltwise_dims,
    "Elwsub": _eltwise_dims,
    "Datacopy": _src_a_dims,
    "Matmul": _matmul_dims,
    "Reduce": _src_a_dims,
    "ReduceBlockMax": _src_a_dims,
    "ReduceBlockMaxRuntime": _src_a_dims,
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


class OperationSchema(OperationSchemaBase):
    _packer_map: ClassVar = PACKER_MAP
    math: List[MathSchema] = Field(..., min_length=1)
    bh_tilize: Tilize = Tilize.No

    def _arch_validate(self):
        unpackers = [
            m.unpacker
            for m in self.math
            if isinstance(m, FpuMathSchema) and m.unpacker is not None
        ]

        unique_unpackers = set(unpackers)

        if "UnpackerTilizeA" in unique_unpackers:
            self.bh_tilize = Tilize.Yes

        if len(unique_unpackers) > 1 and "UnpackerTilizeA" in unique_unpackers:
            raise ValueError(
                "UnpackerTilizeA cannot be combined with other unpackers on BH"
            )

    def _arch_kwargs(self) -> dict:
        return {"bh_tilize": self.bh_tilize}
