# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

from fuser.compute_node import ComputeNode
from fuser.fused_math import ComputePipeline
from fuser.fused_operation import FusedOperation
from helpers.llk_params import (
    AccToDest,
    ApproximationMode,
    BroadcastType,
    ClearFP32DstAcc,
    DataFormat,
    DestSync,
    EltwiseBinaryReuseDestType,
    EnforceFP32Accumulation,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    MathOpType,
    PackerReluType,
    ReduceDimension,
    ReducePool,
    Transpose,
    UnpackToDest,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class UnarySfpuMathSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["UnarySfpu"]
    operation: MathOperation
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    dst_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    fill_const_value: float = 1.0

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, v):
        if isinstance(v, MathOperation):
            if v.operation_type != MathOpType.SFPU_UNARY:
                raise ValueError(f"{v.name} is not a unary SFPU operation")
            return v
        if isinstance(v, str):
            try:
                op = MathOperation[v]
            except KeyError:
                raise ValueError(f"Unknown operation: {v}")
            if op.operation_type != MathOpType.SFPU_UNARY:
                raise ValueError(f"{v} is not a unary SFPU operation")
            return op
        raise ValueError(f"Invalid operation: {v}")

    def _sfpu_class(self):
        raise NotImplementedError

    def to_compute_node(self, operands, output):
        sfpu = self._sfpu_class()(
            self.operation,
            self.approximation_mode,
            self.iterations,
            self.dst_dest_tile_index,
            self.fill_const_value,
        )
        return ComputeNode(unpacker=None, fpu=None, sfpu=sfpu)

    def get_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        return None


class BinarySfpuMathSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["BinarySfpu"]
    operation: MathOperation
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    src1_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    src2_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    dst_dest_tile_index: Annotated[int, Field(ge=0)] = 0

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, v):
        if isinstance(v, MathOperation):
            if v.operation_type not in (
                MathOpType.SFPU_BINARY,
                MathOpType.SFPU_BINARY_INT,
            ):
                raise ValueError(f"{v.name} is not a binary SFPU operation")
            return v
        if isinstance(v, str):
            try:
                op = MathOperation[v]
            except KeyError:
                raise ValueError(f"Unknown operation: {v}")
            if op.operation_type not in (
                MathOpType.SFPU_BINARY,
                MathOpType.SFPU_BINARY_INT,
            ):
                raise ValueError(f"{v} is not a binary SFPU operation")
            return op
        raise ValueError(f"Invalid operation: {v}")

    def _sfpu_class(self):
        raise NotImplementedError

    def to_compute_node(self, operands, output):
        sfpu = self._sfpu_class()(
            self.operation,
            self.approximation_mode,
            self.iterations,
            self.src1_dest_tile_index,
            self.src2_dest_tile_index,
            self.dst_dest_tile_index,
        )
        return ComputeNode(unpacker=None, fpu=None, sfpu=sfpu)

    def get_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        return None


class FpuMathSchemaBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Fpu"]
    broadcast_type: BroadcastType = BroadcastType.None_
    reuse_dest: Optional[EltwiseBinaryReuseDestType] = None
    reduce_pool: Optional[ReducePool] = None
    reduce_dim: Optional[ReduceDimension] = None
    enforce_fp32_accumulation: Optional[EnforceFP32Accumulation] = None
    acc_to_dest: Optional[AccToDest] = None
    unpack_transpose_within_face: Transpose = Transpose.No
    unpack_transpose_faces: Transpose = Transpose.No
    math_fidelity: MathFidelity = MathFidelity.LoFi
    unpack_to_dest: UnpackToDest = UnpackToDest.No
    src_a: str = Field(..., min_length=1)
    src_b: str = Field(..., min_length=1)

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


def validate_fpu_math(
    schema,
    eltwise_ops: Set[str],
    reduce_ops: Set[str],
    forced_row_reduce_ops: Set[str],
    supported_fidelities: Dict[str, Set],
    unpacker_rules: Dict[str, Union[str, Set[str]]],
):
    op = schema.operation

    if op in reduce_ops and schema.reduce_pool is None:
        raise ValueError(f"Reduce operations require reduce_pool: {ReducePool}")

    if op in reduce_ops and schema.reduce_dim is None:
        raise ValueError(f"Reduce operations require reduce_dim: {ReduceDimension}")

    if op in forced_row_reduce_ops:
        if schema.reduce_dim is None:
            schema.reduce_dim = ReduceDimension.Row
        elif schema.reduce_dim != ReduceDimension.Row:
            raise ValueError(
                f'Reduce operations require reduce_dim: "{ReduceDimension.Row.value}"'
            )

    unpacker_name = schema.unpacker

    if schema.unpacker is not None:
        rule = unpacker_rules.get(op)
        if rule is not None:
            allowed = rule if isinstance(rule, set) else {rule}
            if unpacker_name not in allowed:
                expected = " or ".join(sorted(allowed))
                raise ValueError(
                    f"{op}: unpacker must be {expected}, got '{unpacker_name}'"
                )
        elif op in eltwise_ops:
            if (
                schema.reuse_dest is not None
                and schema.reuse_dest != EltwiseBinaryReuseDestType.NONE
            ):
                if unpacker_name != "UnpackerA":
                    raise ValueError(
                        f"Eltwise with reuse_dest: unpacker must be UnpackerA, got '{unpacker_name}'"
                    )
            elif unpacker_name != "UnpackerAB":
                raise ValueError(
                    f"Eltwise: unpacker must be UnpackerAB, got '{unpacker_name}'"
                )

    if unpacker_name == "UnpackerTilizeA":
        if schema.broadcast_type != BroadcastType.None_:
            raise ValueError("UnpackerTilizeA does not support broadcast")
        if (
            schema.unpack_transpose_faces.value
            or schema.unpack_transpose_within_face.value
        ):
            raise ValueError("UnpackerTilizeA does not support transpose")

    if unpacker_name == "MatmulUnpacker":
        if schema.unpack_transpose_within_face != schema.unpack_transpose_faces:
            raise ValueError(
                "MatmulUnpacker does not support different values for transpose_faces and transpose_within_face"
            )

    if unpacker_name == "UnpackerAB":
        if (
            schema.broadcast_type == BroadcastType.Scalar
            and schema.unpack_transpose_faces.value
        ):
            raise ValueError("SrcA transpose is not supported with scalar broadcast")

    allowed_fidelities = supported_fidelities.get(op)
    if (
        allowed_fidelities is not None
        and schema.math_fidelity not in allowed_fidelities
    ):
        raise ValueError(f"{schema.operation} does not support {schema.math_fidelity}")

    if (
        schema.reuse_dest is not None
        and schema.reuse_dest != EltwiseBinaryReuseDestType.NONE
        and op not in eltwise_ops
    ):
        raise ValueError(f"reuse_dest: only for Eltwise operations, not '{op}'")

    if (
        schema.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
        and schema.acc_to_dest != AccToDest.Yes
    ):
        raise ValueError(
            "reuse_dest DEST_TO_SRCA requires acc_to_dest: true. "
            "The LLK unpacker routes L1 data to srcB only when acc_to_dest is enabled; "
            "without it, L1 data goes to srcA and gets overwritten by dest, leaving srcB as zeros."
        )


def build_compute_node(
    schema,
    operands,
    fpu_map: Dict[str, Tuple[Callable, Set[str]]],
    matmul_ops: Set[str],
    unpacker_map: Dict[str, type],
):
    op = schema.operation
    src_a = operands.get(schema.src_a)
    src_b = operands.get(schema.src_b)

    if op in matmul_ops and src_a.dimensions[1] != src_b.dimensions[0]:
        raise ValueError("Matmul: incompatible dimensions for src_a and src_b")

    if (
        src_a.data_format == DataFormat.Int32
        and schema.unpack_to_dest != UnpackToDest.Yes
    ):
        raise ValueError(
            f"src_a format {src_a.data_format} requires unpack_to_dest: Yes. "
            f"SrcA/SrcB registers are 19-bit wide and cannot hold 32-bit integers; "
            f"they must be unpacked directly to DEST."
        )

    entry = fpu_map.get(op)
    if entry is None:
        raise ValueError(f"Unknown FPU operation: {schema.operation}")
    factory, _ = entry
    fpu = factory()

    clear_fp32_dst_acc = (
        ClearFP32DstAcc.Yes
        if schema.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
        or schema.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB
        else ClearFP32DstAcc.No
    )

    kwargs = {}
    if schema.unpacker:
        kwargs["unpacker"] = unpacker_map[schema.unpacker]
    if schema.unpack_transpose_within_face:
        kwargs["unpack_transpose_within_face"] = schema.unpack_transpose_within_face
    if schema.unpack_transpose_faces:
        kwargs["unpack_transpose_faces"] = schema.unpack_transpose_faces
    if schema.broadcast_type:
        kwargs["broadcast_type"] = schema.broadcast_type
    if schema.reuse_dest:
        kwargs["reuse_dest"] = schema.reuse_dest
    if schema.reduce_dim:
        kwargs["reduce_dim"] = schema.reduce_dim
    if schema.reduce_pool:
        kwargs["reduce_pool"] = schema.reduce_pool
    if schema.math_fidelity:
        kwargs["math_fidelity"] = schema.math_fidelity
    if schema.enforce_fp32_accumulation:
        kwargs["enforce_fp32_accumulation"] = schema.enforce_fp32_accumulation
    if clear_fp32_dst_acc:
        kwargs["clear_fp32_dst_acc"] = clear_fp32_dst_acc
    if schema.acc_to_dest:
        kwargs["acc_to_dest"] = schema.acc_to_dest
    if schema.unpack_to_dest:
        kwargs["unpack_to_dest"] = schema.unpack_to_dest

    return ComputeNode(fpu=fpu, src_a=src_a, src_b=src_b, sfpu=None, **kwargs)


def compute_output_dimensions(
    schema,
    operands,
    output_dims: Dict[str, Callable],
) -> Optional[Tuple[int, int]]:
    fn = output_dims.get(schema.operation)
    if fn is None:
        return None
    src_a = operands.get(schema.src_a).dimensions
    src_b = operands.get(schema.src_b).dimensions
    return fn(src_a, src_b)


class OperationSchemaBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: str = Field(..., min_length=1)
    dest_sync: Optional[DestSync] = None
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    pack_relu: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    pack_l1_accumulation: L1Accumulation = L1Accumulation.No

    def _arch_validate(self):
        pass

    def _arch_kwargs(self) -> dict:
        return {}

    def _get_packer_class(self):
        raise NotImplementedError

    def to_fused_operation(self, operands):
        output = operands.get(name=self.output)
        math_ops = [m.to_compute_node(operands, output) for m in self.math]
        output.is_output = True

        max_out_dims = self._calculate_max_output_dimensions(operands)
        resolved_max_out_dims = (
            max_out_dims if max_out_dims is not None else output.dimensions
        )

        if (
            self.block_size[0] > output.dimensions[0]
            or self.block_size[1] > output.dimensions[1]
        ):
            raise ValueError(
                f"Block size {self.block_size} exceeds output dimensions {output.dimensions}"
            )

        if (
            self.pack_l1_accumulation == L1Accumulation.Yes
            and not output.data_format.supports_l1_accumulation()
        ):
            raise ValueError(f"{output.data_format} does not support L1 accumulation")

        kwargs = {}
        if self.dest_sync:
            kwargs["dest_sync"] = self.dest_sync
        if self.block_size:
            kwargs["block_size"] = self.block_size
        if self.pack_relu:
            kwargs["pack_relu"] = self.pack_relu
        if self.relu_threshold:
            kwargs["relu_threshold"] = self.relu_threshold
        if self.pack_l1_accumulation:
            kwargs["pack_l1_accumulation"] = self.pack_l1_accumulation
        kwargs.update(self._arch_kwargs())

        return FusedOperation(
            math=ComputePipeline(math_ops, self._get_packer_class()),
            output=output,
            max_output_dimensions=resolved_max_out_dims,
            **kwargs,
        )

    def _calculate_max_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        dims = []
        for m in self.math:
            op_dims = m.get_output_dimensions(operands)
            if op_dims is not None:
                dims.append(op_dims)

        if not dims:
            return None

        max_r = min(d[0] for d in dims)
        max_c = min(d[1] for d in dims)
        return (max_r, max_c)
