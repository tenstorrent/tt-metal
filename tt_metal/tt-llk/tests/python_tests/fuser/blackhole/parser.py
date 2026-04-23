# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Annotated, List, Literal, Optional, Type, Union

from fuser.fused_math import ComputeNode, ComputePipeline
from fuser.fused_operation import FusedOperation
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    BroadcastType,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    Tilize,
    Transpose,
)
from pydantic import (
    BaseModel,
    ConfigDict,
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

    def is_eltwise(self) -> bool:
        return self in {
            FpuOperationEnum.Elwadd,
            FpuOperationEnum.Elwmul,
            FpuOperationEnum.Elwsub,
        }

    def to_math_operation(self):
        return getattr(MathOperation, self.value)


class UnaryOperationEnum(str, Enum):
    Abs = "Abs"
    Acosh = "Acosh"
    Asinh = "Asinh"
    Atanh = "Atanh"
    Celu = "Celu"
    Cos = "Cos"
    Elu = "Elu"
    Exp = "Exp"
    Exp2 = "Exp2"
    Fill = "Fill"
    Gelu = "Gelu"
    Hardsigmoid = "Hardsigmoid"
    Log = "Log"
    Log1p = "Log1p"
    Neg = "Neg"
    Reciprocal = "Reciprocal"
    ReluMax = "ReluMax"
    ReluMin = "ReluMin"
    Rsqrt = "Rsqrt"
    Silu = "Silu"
    Sin = "Sin"
    Sqrt = "Sqrt"
    Square = "Square"
    Tanh = "Tanh"
    Threshold = "Threshold"

    def to_math_operation(self):
        return getattr(MathOperation, self.value)


class BinaryOperationEnum(str, Enum):
    SfpuElwadd = "SfpuElwadd"
    SfpuElwmul = "SfpuElwmul"
    SfpuElwsub = "SfpuElwsub"
    SfpuElwLeftShift = "SfpuElwLeftShift"
    SfpuElwRightShift = "SfpuElwRightShift"
    SfpuElwLogicalRightShift = "SfpuElwLogicalRightShift"
    SfpuXlogy = "SfpuXlogy"
    SfpuAddTopRow = "SfpuAddTopRow"

    def to_math_operation(self):
        return getattr(MathOperation, self.value)


class FpuMathSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Fpu"]
    operation: FpuOperationEnum
    unpacker: Optional[UnpackerEnum] = None
    broadcast_type: BroadcastType = BroadcastType.None_
    reuse_dest: Optional[EltwiseBinaryReuseDestType] = None
    reduce_pool: Optional[ReducePool] = None
    reduce_dim: Optional[ReduceDimension] = None
    unpack_transpose_within_face: Transpose = Transpose.No
    unpack_transpose_faces: Transpose = Transpose.No
    math_fidelity: MathFidelity = MathFidelity.LoFi

    @field_validator("unpacker", mode="before")
    @classmethod
    def parse_unpacker(cls, v):
        if isinstance(v, UnpackerEnum):
            return v
        if isinstance(v, str) and v in UnpackerEnum.__members__:
            return UnpackerEnum[v]
        return v

    @field_validator("math_fidelity", mode="before")
    @classmethod
    def parse_math_fidelity(cls, v):
        if isinstance(v, MathFidelity):
            return v
        if isinstance(v, str):
            try:
                return MathFidelity[v]
            except KeyError:
                pass
        return v

    @model_validator(mode="after")
    def validate_fpu_config(self) -> "FpuMathSchema":
        if self.operation == FpuOperationEnum.Reduce and self.reduce_pool is None:
            raise ValueError(f"Reduce operations require reduce_pool: {ReducePool}")

        if self.operation == FpuOperationEnum.Reduce and self.reduce_dim is None:
            raise ValueError(f"Reduce operations require reduce_dim: {ReduceDimension}")

        if self.operation == FpuOperationEnum.ReduceBlockMax:
            if self.reduce_dim is None:
                self.reduce_dim = ReduceDimension.Row
            elif self.reduce_dim != ReduceDimension.Row:
                raise ValueError(
                    f'Reduce operations require reduce_dim: "{ReduceDimension.Row.value}"'
                )

        if self.unpacker is not None:
            if self.operation == FpuOperationEnum.Datacopy:
                if self.unpacker not in {
                    UnpackerEnum.UnpackerA,
                    UnpackerEnum.UnpackerTilizeA,
                }:
                    raise ValueError(
                        f"Datacopy: unpacker must be UnpackerA or UnpackerTilizeA, got '{self.unpacker.value}'"
                    )
            elif self.operation == FpuOperationEnum.Matmul:
                if self.unpacker != UnpackerEnum.MatmulUnpacker:
                    raise ValueError(
                        f"Matmul: unpacker must be MatmulUnpacker, got '{self.unpacker.value}'"
                    )
            elif self.operation == FpuOperationEnum.Reduce:
                if self.unpacker != UnpackerEnum.ReduceUnpacker:
                    raise ValueError(
                        f"Reduce: unpacker must be ReduceUnpacker, got '{self.unpacker.value}'"
                    )
            elif self.operation.is_eltwise():
                if (
                    self.reuse_dest is not None
                    and self.reuse_dest != EltwiseBinaryReuseDestType.NONE
                ):
                    if self.unpacker != UnpackerEnum.UnpackerA:
                        raise ValueError(
                            f"Eltwise with reuse_dest: unpacker must be UnpackerA, got '{self.unpacker.value}'"
                        )
                elif self.unpacker != UnpackerEnum.UnpackerAB:
                    raise ValueError(
                        f"Eltwise: unpacker must be UnpackerAB, got '{self.unpacker.value}'"
                    )
            elif self.operation == FpuOperationEnum.ReduceBlockMax:
                if self.unpacker != UnpackerEnum.ReduceBlockMaxUnpacker:
                    raise ValueError(
                        f"ReduceBlockMax: unpacker must be ReduceBlockMaxUnpacker, got '{self.unpacker.value}'"
                    )

        if self.unpacker == UnpackerEnum.UnpackerTilizeA:
            if self.broadcast_type != BroadcastType.None_:
                raise ValueError("UnpackerTilizeA does not support broadcast")

            if (
                self.unpack_transpose_faces.value
                or self.unpack_transpose_within_face.value
            ):
                raise ValueError("UnpackerTilizeA does not support transpose")

        if self.unpacker == UnpackerEnum.MatmulUnpacker:
            if self.unpack_transpose_within_face != self.unpack_transpose_faces:
                raise ValueError(
                    "MatmulUnpacker does not support different values for transpose_faces and transpose_within_face"
                )

        if self.unpacker == UnpackerEnum.UnpackerAB:
            if (
                self.broadcast_type == BroadcastType.Scalar
                and self.unpack_transpose_faces.value
            ):
                raise ValueError(
                    "SrcA transpose is not supported with scalar broadcast"
                )

            if self.unpack_transpose_within_face != self.unpack_transpose_faces:
                raise ValueError(
                    "UnpackerAB does not support different values for transpose_faces and transpose_within_face"
                )

        # LLK contract: eltwise add/sub only support LoFi fidelity.
        if (
            self.operation in [FpuOperationEnum.Elwadd, FpuOperationEnum.Elwsub]
            and self.math_fidelity != MathFidelity.LoFi
        ):
            raise ValueError(f"{self.operation} does not support {self.math_fidelity}")

        if (
            self.reuse_dest is not None
            and self.reuse_dest != EltwiseBinaryReuseDestType.NONE
            and not self.operation.is_eltwise()
        ):
            raise ValueError(
                f"reuse_dest: only for Eltwise operations, not '{self.operation.value}'"
            )

        if (
            self.reuse_dest is not None
            and self.reuse_dest != EltwiseBinaryReuseDestType.NONE
            and not self.operation.is_eltwise()
        ):
            raise ValueError(
                f"reuse_dest: only for Eltwise operations, not '{self.operation.value}'"
            )

        return self

    def to_compute_node(self):
        if self.operation.is_eltwise():
            fpu = EltwiseFpu(self.operation.to_math_operation())
        elif self.operation == FpuOperationEnum.Reduce:
            fpu = ReduceFpu()
        elif self.operation == FpuOperationEnum.Datacopy:
            fpu = DatacopyFpu()
        elif self.operation == FpuOperationEnum.Matmul:
            fpu = MatmulFpu()
        elif self.operation == FpuOperationEnum.ReduceBlockMax:
            fpu = ReduceBlockMaxFpu()
        else:
            raise ValueError(f"Unknown FPU operation: {self.operation}")

        kwargs = {}
        if self.unpacker:
            kwargs["unpacker"] = self.unpacker.to_runtime()
        if self.unpack_transpose_within_face:
            kwargs["unpack_transpose_within_face"] = self.unpack_transpose_within_face
        if self.unpack_transpose_faces:
            kwargs["unpack_transpose_faces"] = self.unpack_transpose_faces
        if self.broadcast_type:
            kwargs["broadcast_type"] = self.broadcast_type
        if self.reuse_dest:
            kwargs["reuse_dest"] = self.reuse_dest
        if self.reduce_dim:
            kwargs["reduce_dim"] = self.reduce_dim
        if self.reduce_pool:
            kwargs["reduce_pool"] = self.reduce_pool
        if self.math_fidelity:
            kwargs["math_fidelity"] = self.math_fidelity

        return ComputeNode(fpu=fpu, sfpu=None, **kwargs)


class UnarySfpuMathSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["UnarySfpu"]
    operation: UnaryOperationEnum
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    dst_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    fill_const_value: float = 1.0

    def to_compute_node(self):

        sfpu = UnarySfpu(
            self.operation.to_math_operation(),
            self.approximation_mode,
            self.iterations,
            self.dst_dest_tile_index,
            self.fill_const_value,
        )
        return ComputeNode(unpacker=None, fpu=None, sfpu=sfpu)


class BinarySfpuMathSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["BinarySfpu"]
    operation: BinaryOperationEnum
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    src1_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    src2_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    dst_dest_tile_index: Annotated[int, Field(ge=0)] = 0

    def to_compute_node(self):

        sfpu = BinarySfpu(
            self.operation.to_math_operation(),
            self.approximation_mode,
            self.iterations,
            self.src1_dest_tile_index,
            self.src2_dest_tile_index,
            self.dst_dest_tile_index,
        )
        return ComputeNode(unpacker=None, fpu=None, sfpu=sfpu)


MathSchema = Annotated[
    Union[FpuMathSchema, UnarySfpuMathSchema, BinarySfpuMathSchema],
    Field(discriminator="type"),
]


class OperationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src_a: str = Field(..., min_length=1)
    src_b: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)

    src_a_dims: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    src_b_dims: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    output_dims: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]

    input_format: DataFormat = Field(default_factory=lambda: DataFormat.Float16_b)
    output_format: DataFormat = Field(default_factory=lambda: DataFormat.Float16_b)

    src_a_const_value: Optional[float] = None
    src_b_const_value: Optional[float] = None

    math: List[MathSchema] = Field(..., min_length=1)

    packer: PackerEnum = PackerEnum.Packer
    dest_sync: Optional[DestSync] = None
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    bh_tilize: Optional[Tilize] = None

    @field_validator("src_a_dims", "src_b_dims", "output_dims", "block_size")
    @classmethod
    def validate_dimensions(cls, v: List[int]) -> List[int]:
        for dim in v:
            if dim <= 0:
                raise ValueError(f"must be positive, got {dim}")
            if dim % 32 != 0:
                raise ValueError(f"must be multiple of 32, got {dim}")
        return v

    @field_validator("input_format", "output_format", mode="before")
    @classmethod
    def parse_data_format(cls, v):
        if isinstance(v, DataFormat):
            return v
        if isinstance(v, str):
            try:
                return DataFormat[v]
            except KeyError:
                pass
        return v

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
        has_matmul = any(
            isinstance(m, FpuMathSchema) and m.operation == FpuOperationEnum.Matmul
            for m in self.math
        )

        if has_matmul:
            if self.src_a_dims[1] != self.src_b_dims[0]:
                raise ValueError(
                    f"Matmul: src_a[1]={self.src_a_dims[1]} != src_b[0]={self.src_b_dims[0]}"
                )

        if (
            self.block_size[0] > self.output_dims[0]
            or self.block_size[1] > self.output_dims[1]
        ):
            raise ValueError(
                f"Block size {self.block_size} exceeds output dimensions {self.output_dims}"
            )

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

        return self

    def to_fused_operation(self, operands):
        operand_mapping = operands.create_mapping(
            src_a=self.src_a,
            src_b=self.src_b,
            output=self.output,
            src_a_dims=self.src_a_dims,
            src_b_dims=self.src_b_dims,
            output_dims=self.output_dims,
            input_format=self.input_format,
            output_format=self.output_format,
            src_a_const_value=self.src_a_const_value,
            src_b_const_value=self.src_b_const_value,
        )

        math_ops = [m.to_compute_node() for m in self.math]

        kwargs = {
            "bh_tilize": self.bh_tilize,
        }
        if self.dest_sync:
            kwargs["dest_sync"] = self.dest_sync
        if self.block_size:
            kwargs["block_size"] = self.block_size

        return FusedOperation(
            operand_mapping=operand_mapping,
            math=ComputePipeline(math_ops, self.packer.to_runtime()),
            **kwargs,
        )
