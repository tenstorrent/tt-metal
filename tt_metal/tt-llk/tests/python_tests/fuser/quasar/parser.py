# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar fuser config parser.

Supports: eltwise binary (Elwadd/Elwmul/Elwsub), datacopy, matmul, reduce,
unary SFPU, binary SFPU.
Unsupported on Quasar: MatmulNoMop, ReduceBlockMax, ReduceBlockMaxRuntime,
SubBcastColCustom.
No broadcast, no transpose.
"""

from typing import Annotated, ClassVar, List, Union

from fuser.validator import (
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    PackSchema,
    UnarySfpuMathSchema,
)
from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    EltwiseBinaryReuseDestType,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    Transpose,
)
from pydantic import Field

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.reduce import ReduceFpu
from .packer.packer import Packer
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.unpack_a import UnpackerA
from .unpacker.unpack_ab import UnpackerAB

_no_broadcast = (
    lambda s, a, b: s.broadcast_type != BroadcastType.None_,
    "Quasar does not support broadcast in fuser",
)

_no_transpose = (
    lambda s, a, b: s.unpack_transpose_faces == Transpose.Yes
    or s.unpack_transpose_within_face == Transpose.Yes,
    "Quasar does not support transpose in unpack",
)

_dest_to_srca_needs_acc = (
    lambda s, a, b: s.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
    and s.acc_to_dest != AccToDest.Yes,
    "reuse_dest DEST_TO_SRCA requires acc_to_dest: true",
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

_eltwise_checks = [
    _no_broadcast,
    _no_transpose,
    _dest_to_srca_needs_acc,
    _eltwise_unpacker_reuse,
    _eltwise_unpacker_default,
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

_forced_unpacker = lambda name: (
    lambda s, a, b: s.unpacker is not None and s.unpacker != name,
    f"unpacker must be {name}",
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

UNPACKER_MAP = {
    "UnpackerA": (
        lambda s: UnpackerA(),
        [_no_broadcast, _no_transpose],
    ),
    "UnpackerAB": (
        lambda s: UnpackerAB(),
        [_no_broadcast, _no_transpose],
    ),
    "MatmulUnpacker": (
        lambda s: MatmulUnpacker(),
        [_no_transpose],
    ),
    "ReduceUnpacker": (
        lambda s: ReduceUnpacker(s.reduce_dim, s.reduce_pool),
        None,
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
        [_no_reuse_dest, _datacopy_unpacker, _no_broadcast, _no_transpose],
    ),
    "Matmul": (
        lambda s: MatmulFpu(),
        [
            _no_reuse_dest,
            _matmul_dim_check,
            _forced_unpacker("MatmulUnpacker"),
            _matmul_inner_dims,
        ],
    ),
    "Reduce": (
        lambda s: ReduceFpu(s.reduce_dim, s.reduce_pool),
        [
            _no_reuse_dest,
            _reduce_params,
            _forced_unpacker("ReduceUnpacker"),
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
        for m in self.math:
            if isinstance(m, FpuMathSchema):
                if m.broadcast_type != BroadcastType.None_:
                    raise ValueError("Quasar does not support broadcast in fuser")
                if (
                    m.unpack_transpose_faces == Transpose.Yes
                    or m.unpack_transpose_within_face == Transpose.Yes
                ):
                    raise ValueError("Quasar does not support transpose in unpack")
