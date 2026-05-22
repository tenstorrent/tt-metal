# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole fuser config parser.

All the arch-specific configuration lives in plain dicts at the top of the file.
The shared base classes in fuser.validator read these dicts to run validation and
build runtime objects, so this file only needs to define the data and thin subclasses.

To add a new FPU op, add one entry to FPU_MAP with its tags, one to OUTPUT_DIMS,
and one to UNPACKER_RULES if the op requires a specific unpacker.
"""

from typing import Annotated, List, Optional, Union

from fuser.validator import (
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    UnarySfpuMathSchema,
    build_compute_node,
    compute_output_dimensions,
    validate_fpu_math,
)
from helpers.llk_params import MathFidelity, MathOperation, ReduceDimension, Tilize
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.reduce import ReduceFpu
from .fpu.reduce_block_max import ReduceBlockMaxFpu
from .packer.packer import Packer
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.reduce_block_max import ReduceBlockMaxUnpacker
from .unpacker.tilize_a import UnpackerTilizeA
from .unpacker.unpack_a import UnpackerA
from .unpacker.unpack_ab import UnpackerAB

UNPACKER_MAP = {
    "UnpackerA": UnpackerA,
    "UnpackerAB": UnpackerAB,
    "UnpackerTilizeA": UnpackerTilizeA,
    "MatmulUnpacker": MatmulUnpacker,
    "ReduceUnpacker": ReduceUnpacker,
    "ReduceBlockMaxUnpacker": ReduceBlockMaxUnpacker,
}

PACKER_MAP = {
    "Packer": Packer,
}

FPU_MAP = {
    "Elwadd": (lambda: EltwiseFpu(MathOperation.Elwadd), {"eltwise"}),
    "Elwmul": (lambda: EltwiseFpu(MathOperation.Elwmul), {"eltwise"}),
    "Elwsub": (lambda: EltwiseFpu(MathOperation.Elwsub), {"eltwise"}),
    "Datacopy": (DatacopyFpu, set()),
    "Matmul": (MatmulFpu, {"matmul"}),
    "Reduce": (ReduceFpu, {"reduce"}),
    "ReduceBlockMax": (ReduceBlockMaxFpu, {"reduce"}),
}

_tagged = lambda tag: {op for op, (_, tags) in FPU_MAP.items() if tag in tags}

ELTWISE_OPS = _tagged("eltwise")
MATMUL_OPS = _tagged("matmul")
REDUCE_OPS = _tagged("reduce")

FORCED_REDUCE_DIM = {
    "ReduceBlockMax": ReduceDimension.Row,
}

SUPPORTED_FIDELITIES = {
    "Elwadd": {MathFidelity.LoFi},
    "Elwsub": {MathFidelity.LoFi},
}

UNPACKER_RULES = {
    "Datacopy": {"UnpackerA", "UnpackerTilizeA"},
    "Matmul": "MatmulUnpacker",
    "Reduce": "ReduceUnpacker",
    "ReduceBlockMax": "ReduceBlockMaxUnpacker",
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
    """Blackhole FPU math node (type="Fpu").

    Validates operation and unpacker strings against the keys of FPU_MAP and
    UNPACKER_MAP, then passes the Blackhole dicts to validate_fpu_math() for
    all cross-field checks.
    """

    operation: str
    unpacker: Optional[str] = None

    @field_validator("operation", mode="after")
    @classmethod
    def validate_operation(cls, v):
        if v not in FPU_MAP:
            raise ValueError(f"Unknown FPU operation: {v}")
        return v

    @field_validator("unpacker", mode="after")
    @classmethod
    def validate_unpacker(cls, v):
        if v is not None and v not in UNPACKER_MAP:
            raise ValueError(f"Unknown unpacker: {v}")
        return v

    @model_validator(mode="after")
    def validate_fpu_config(self) -> "FpuMathSchema":
        validate_fpu_math(
            self,
            ELTWISE_OPS,
            REDUCE_OPS,
            FORCED_REDUCE_DIM,
            SUPPORTED_FIDELITIES,
            UNPACKER_RULES,
        )
        return self

    def to_compute_node(self, operands, output):
        return build_compute_node(self, operands, FPU_MAP, MATMUL_OPS, UNPACKER_MAP)

    def get_output_dimensions(self, operands):
        return compute_output_dimensions(self, operands, OUTPUT_DIMS)


class BlackholeUnarySfpuMathSchema(UnarySfpuMathSchema):
    """Blackhole unary SFPU node, only allows operations listed in UNARY_SFPU_OPS."""

    @field_validator("operation", mode="after")
    @classmethod
    def validate_arch_operation(cls, v):
        if v not in UNARY_SFPU_OPS:
            raise ValueError(f"Unsupported unary SFPU operation: {v.name}")
        return v

    def _sfpu_class(self):
        return UnarySfpu


class BlackholeBinarySfpuMathSchema(BinarySfpuMathSchema):
    """Blackhole binary SFPU node, only allows operations listed in BINARY_SFPU_OPS."""

    @field_validator("operation", mode="after")
    @classmethod
    def validate_arch_operation(cls, v):
        if v not in BINARY_SFPU_OPS:
            raise ValueError(f"Unsupported binary SFPU operation: {v.name}")
        return v

    def _sfpu_class(self):
        return BinarySfpu


MathSchema = Annotated[
    Union[FpuMathSchema, BlackholeUnarySfpuMathSchema, BlackholeBinarySfpuMathSchema],
    Field(discriminator="type"),
]


class OperationSchema(OperationSchemaBase):
    """Blackhole fused operation with one output and a pipeline of math nodes.

    Extends the base with tilize handling. The _arch_validate() method looks at
    which unpackers are used and sets the bh_tilize flag when UnpackerTilizeA is
    present. Then _arch_kwargs() forwards that flag to FusedOperation.
    """

    math: List[MathSchema] = Field(..., min_length=1)
    packer: str = "Packer"
    bh_tilize: Tilize = Tilize.No

    @field_validator("packer", mode="after")
    @classmethod
    def validate_packer(cls, v):
        if v not in PACKER_MAP:
            raise ValueError(f"Unknown packer: {v}")
        return v

    def _get_packer_class(self):
        return PACKER_MAP[self.packer]

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationSchema":
        self._arch_validate()
        return self

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
