# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Annotated, List, Optional, Type, Union

from fuser.validator import (
    BinarySfpuMathSchema,
    FpuMathSchemaBase,
    OperationSchemaBase,
    UnarySfpuMathSchema,
    build_compute_node,
    compute_output_dimensions,
    validate_fpu_math,
)
from helpers.llk_params import MathOperation, Tilize
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


class UnpackerEnum(Enum):
    UnpackerA = UnpackerA
    UnpackerAB = UnpackerAB
    UnpackerTilizeA = UnpackerTilizeA
    MatmulUnpacker = MatmulUnpacker
    ReduceUnpacker = ReduceUnpacker
    ReduceBlockMaxUnpacker = ReduceBlockMaxUnpacker

    def to_runtime(self) -> Type:
        return self.value


class PackerEnum(Enum):
    Packer = Packer

    def to_runtime(self) -> Type:
        return self.value


class FpuOperationEnum(str, Enum):
    Elwadd = "Elwadd"
    Elwmul = "Elwmul"
    Elwsub = "Elwsub"
    Datacopy = "Datacopy"
    Matmul = "Matmul"
    Reduce = "Reduce"
    ReduceBlockMax = "ReduceBlockMax"

    def to_math_operation(self):
        return getattr(MathOperation, self.value)


FPU_MAP = {
    "Elwadd": lambda: EltwiseFpu(MathOperation.Elwadd),
    "Elwmul": lambda: EltwiseFpu(MathOperation.Elwmul),
    "Elwsub": lambda: EltwiseFpu(MathOperation.Elwsub),
    "Datacopy": DatacopyFpu,
    "Matmul": MatmulFpu,
    "Reduce": ReduceFpu,
    "ReduceBlockMax": ReduceBlockMaxFpu,
}

UNPACKER_RULES = {
    "Datacopy": {"UnpackerA", "UnpackerTilizeA"},
    "Matmul": "MatmulUnpacker",
    "Reduce": "ReduceUnpacker",
    "ReduceBlockMax": "ReduceBlockMaxUnpacker",
}

ELTWISE_OPS = {"Elwadd", "Elwmul", "Elwsub"}
MATMUL_OPS = {"Matmul"}
REDUCE_OPS = {"Reduce"}
FORCED_ROW_REDUCE_OPS = {"ReduceBlockMax"}
SRC_A_DIM_OPS = {"Datacopy", "Reduce", "ReduceBlockMax"}


class FpuMathSchema(FpuMathSchemaBase):
    operation: FpuOperationEnum
    unpacker: Optional[UnpackerEnum] = None

    @field_validator("unpacker", mode="before")
    @classmethod
    def parse_unpacker(cls, v):
        if isinstance(v, UnpackerEnum):
            return v
        if isinstance(v, str) and v in UnpackerEnum.__members__:
            return UnpackerEnum[v]
        return v

    @model_validator(mode="after")
    def validate_fpu_config(self) -> "FpuMathSchema":
        validate_fpu_math(
            self, ELTWISE_OPS, REDUCE_OPS, FORCED_ROW_REDUCE_OPS, UNPACKER_RULES
        )
        return self

    def to_compute_node(self, operands, output):
        return build_compute_node(self, operands, FPU_MAP, MATMUL_OPS)

    def get_output_dimensions(self, operands):
        return compute_output_dimensions(
            self, operands, ELTWISE_OPS, MATMUL_OPS, SRC_A_DIM_OPS
        )


class BlackholeUnarySfpuMathSchema(UnarySfpuMathSchema):
    def _sfpu_class(self):
        return UnarySfpu


class BlackholeBinarySfpuMathSchema(BinarySfpuMathSchema):
    def _sfpu_class(self):
        return BinarySfpu


MathSchema = Annotated[
    Union[FpuMathSchema, BlackholeUnarySfpuMathSchema, BlackholeBinarySfpuMathSchema],
    Field(discriminator="type"),
]


class OperationSchema(OperationSchemaBase):
    math: List[MathSchema] = Field(..., min_length=1)
    packer: PackerEnum = PackerEnum.Packer
    bh_tilize: Optional[Tilize] = None

    @field_validator("packer", mode="before")
    @classmethod
    def parse_packer(cls, v):
        if isinstance(v, PackerEnum):
            return v
        if isinstance(v, str) and v in PackerEnum.__members__:
            return PackerEnum[v]
        return v

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

        if UnpackerEnum.UnpackerTilizeA in unique_unpackers:
            self.bh_tilize = Tilize.Yes
        else:
            self.bh_tilize = Tilize.No

        if (
            len(unique_unpackers) > 1
            and UnpackerEnum.UnpackerTilizeA in unique_unpackers
        ):
            raise ValueError(
                "UnpackerTilizeA cannot be combined with other unpackers on BH"
            )

    def _arch_kwargs(self) -> dict:
        if self.bh_tilize:
            return {"bh_tilize": self.bh_tilize}
        return {}
