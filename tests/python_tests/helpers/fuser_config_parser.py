# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Type, Union

import yaml
from helpers.format_config import DataFormat
from helpers.fused_fpu import (
    DatacopyFpu,
    EltwiseFpu,
    MatmulFpu,
    ReduceBlockMaxFpu,
    ReduceFpu,
)
from helpers.fused_math import ComputeNode, ComputePipeline
from helpers.fused_operand import OperandRegistry
from helpers.fused_operation import FusedOperation
from helpers.fused_packer import Packer
from helpers.fused_sfpu import BinarySfpu, UnarySfpu
from helpers.fused_unpacker import (
    MatmulUnpacker,
    ReduceBlockMaxUnpacker,
    ReduceUnpacker,
    UnpackerA,
    UnpackerAB,
    UnpackerTilizeA,
)
from helpers.fuser_config import FuserConfig, GlobalConfig
from helpers.llk_params import (
    ApproximationMode,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    ReducePool,
    Transpose,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

FUSER_CONFIG_DIR = (
    Path(os.environ.get("LLK_HOME", ".")) / "tests" / "python_tests" / "fuser_config"
)


def format_validation_error(error: ValidationError) -> str:
    messages = []
    for err in error.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]

        if "Input should be" in msg:
            inp = err.get("input")
            valid_values = re.findall(r"'([^']+)'", msg)
            expected = ", ".join(valid_values) if valid_values else msg
            messages.append(f"'{loc}': got '{inp}', expected: {expected}")
        elif "Extra inputs are not permitted" in msg:
            messages.append(f"'{loc}': unknown field")
        elif "Field required" in msg:
            messages.append(f"'{loc}': required field missing")
        else:
            clean_msg = msg.removeprefix("Value error, ")
            messages.append(f"'{loc}': {clean_msg}")

    return "\n".join(messages)


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
    ReduceColumn = "ReduceColumn"
    ReduceRow = "ReduceRow"
    ReduceScalar = "ReduceScalar"
    ReduceBlockMax = "ReduceBlockMax"

    def is_eltwise(self) -> bool:
        return self in {
            FpuOperationEnum.Elwadd,
            FpuOperationEnum.Elwmul,
            FpuOperationEnum.Elwsub,
        }

    def is_reduce(self) -> bool:
        return self in {
            FpuOperationEnum.ReduceColumn,
            FpuOperationEnum.ReduceRow,
            FpuOperationEnum.ReduceScalar,
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
    broadcast_type: Optional[BroadcastType] = None
    reuse_dest: Optional[EltwiseBinaryReuseDestType] = None
    reduce_pool: Optional[ReducePool] = None
    unpack_transpose_within_face: Optional[Transpose] = None
    unpack_transpose_faces: Optional[Transpose] = None

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
        if self.operation.is_reduce() and self.reduce_pool is None:
            raise ValueError(f"Reduce operations require reduce_pool: {ReducePool}")

        if not self.operation.is_reduce() and self.reduce_pool is not None:
            raise ValueError(
                f"reduce_pool: only for Reduce*, not '{self.operation.value}'"
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
            elif self.operation.is_reduce():
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

        if (
            self.reuse_dest is not None
            and self.reuse_dest != EltwiseBinaryReuseDestType.NONE
        ):
            if not self.operation.is_eltwise():
                raise ValueError(
                    f"reuse_dest: only for Eltwise operations, not '{self.operation.value}'"
                )

        return self

    def to_compute_node(self):
        if self.operation.is_eltwise():
            fpu = EltwiseFpu(self.operation.to_math_operation())
        elif self.operation.is_reduce():
            fpu = ReduceFpu(self.operation.to_math_operation(), pool=self.reduce_pool)
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
    math_fidelity: MathFidelity = MathFidelity.LoFi
    dest_sync: Optional[DestSync] = None
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]

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
            "math_fidelity": self.math_fidelity,
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


class FuserConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dest_acc: DestAccumulation = DestAccumulation.No
    loop_factor: Annotated[int, Field(ge=1)] = 16
    operations: List[OperationSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_config(self) -> "FuserConfigSchema":
        outputs = set()
        for i, op in enumerate(self.operations):
            if op.output in outputs:
                raise ValueError(f"op[{i}].output='{op.output}' already defined")
            outputs.add(op.output)
        return self

    def to_fuser_config(self, test_name: str):
        operands = OperandRegistry()
        pipeline = [op.to_fused_operation(operands) for op in self.operations]

        return FuserConfig(
            pipeline=pipeline,
            global_config=GlobalConfig(
                dest_acc=self.dest_acc,
                test_name=test_name,
                loop_factor=self.loop_factor,
            ),
        )

    @classmethod
    def validate_file(cls, yaml_path: Union[str, Path]) -> "FuserConfigSchema":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        try:
            return cls.model_validate(config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed for {yaml_path.name}:\n{format_validation_error(e)}"
            ) from None

    @classmethod
    def validate_string(cls, yaml_content: str) -> "FuserConfigSchema":
        config_dict = yaml.safe_load(yaml_content)
        try:
            return cls.model_validate(config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed:\n{format_validation_error(e)}"
            ) from None

    @classmethod
    def load(cls, test_name: str):
        yaml_path = FUSER_CONFIG_DIR / f"{test_name}.yaml"
        schema = cls.validate_file(yaml_path)
        return schema.to_fuser_config(test_name)
