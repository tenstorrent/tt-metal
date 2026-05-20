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
from helpers.llk_params import MathOperation
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from .fpu.datacopy import DatacopyFpu
from .fpu.eltwise import EltwiseFpu
from .fpu.matmul import MatmulFpu
from .fpu.matmul_no_mop import MatmulNoMopFpu
from .fpu.reduce import ReduceFpu
from .fpu.reduce_block_max import ReduceBlockMaxFpu
from .fpu.reduce_block_max_runtime import ReduceBlockMaxRuntimeFpu
from .fpu.sub_bcast_col_custom import SubBcastColCustomFpu
from .packer.packer import Packer
from .sfpu.binary import BinarySfpu
from .sfpu.unary import UnarySfpu
from .unpacker.matmul import MatmulUnpacker
from .unpacker.reduce import ReduceUnpacker
from .unpacker.reduce_block_max import ReduceBlockMaxUnpacker
from .unpacker.reduce_block_max_runtime import ReduceBlockMaxRuntimeUnpacker
from .unpacker.sub_bcast_col_custom import SubBcastColCustomUnpacker
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
    ReduceBlockMaxRuntimeUnpacker = ReduceBlockMaxRuntimeUnpacker
    SubBcastColCustomUnpacker = SubBcastColCustomUnpacker

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
    MatmulNoMop = "MatmulNoMop"
    Reduce = "Reduce"
    ReduceBlockMax = "ReduceBlockMax"
    ReduceBlockMaxRuntime = "ReduceBlockMaxRuntime"
    SubBcastColCustom = "SubBcastColCustom"

    def to_math_operation(self):
        return getattr(MathOperation, self.value)


FPU_MAP = {
    "Elwadd": lambda: EltwiseFpu(MathOperation.Elwadd),
    "Elwmul": lambda: EltwiseFpu(MathOperation.Elwmul),
    "Elwsub": lambda: EltwiseFpu(MathOperation.Elwsub),
    "Datacopy": DatacopyFpu,
    "Matmul": MatmulFpu,
    "MatmulNoMop": MatmulNoMopFpu,
    "Reduce": ReduceFpu,
    "ReduceBlockMax": ReduceBlockMaxFpu,
    "ReduceBlockMaxRuntime": ReduceBlockMaxRuntimeFpu,
    "SubBcastColCustom": SubBcastColCustomFpu,
}

UNPACKER_RULES = {
    "Datacopy": {"UnpackerA", "UnpackerTilizeA"},
    "Matmul": "MatmulUnpacker",
    "MatmulNoMop": "MatmulUnpacker",
    "Reduce": "ReduceUnpacker",
    "ReduceBlockMax": "ReduceBlockMaxUnpacker",
    "ReduceBlockMaxRuntime": "ReduceBlockMaxRuntimeUnpacker",
    "SubBcastColCustom": "SubBcastColCustomUnpacker",
}

ELTWISE_OPS = {"Elwadd", "Elwmul", "Elwsub"}
MATMUL_OPS = {"Matmul", "MatmulNoMop"}
REDUCE_OPS = {"Reduce"}
FORCED_ROW_REDUCE_OPS = {"ReduceBlockMax", "ReduceBlockMaxRuntime"}
SRC_A_DIM_OPS = {
    "Datacopy",
    "Reduce",
    "ReduceBlockMax",
    "ReduceBlockMaxRuntime",
    "SubBcastColCustom",
}


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


class WormholeUnarySfpuMathSchema(UnarySfpuMathSchema):
    def _sfpu_class(self):
        return UnarySfpu


class WormholeBinarySfpuMathSchema(BinarySfpuMathSchema):
    def _sfpu_class(self):
        return BinarySfpu


MathSchema = Annotated[
    Union[FpuMathSchema, WormholeUnarySfpuMathSchema, WormholeBinarySfpuMathSchema],
    Field(discriminator="type"),
]


class OperationSchema(OperationSchemaBase):
    math: List[MathSchema] = Field(..., min_length=1)
    packer: PackerEnum = PackerEnum.Packer

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
