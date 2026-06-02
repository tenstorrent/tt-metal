# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared base classes and validation for fuser config schemas.

Each architecture (wormhole/parser.py, blackhole/parser.py) inherits from the base
classes defined here and supplies plain dicts that control all validation and
construction. The dicts are:

    FPU_MAP              op name to (factory(schema), checks), set via _fpu_map class attr
    UNPACKER_MAP         unpacker name to (factory(schema), checks), set via _unpacker_map class attr
    PACKER_MAP           packer name to (class, checks), set via _packer_map class attr
    OUTPUT_DIMS          op name to lambda(src_a, src_b), set via _output_dims class attr
"""

from typing import Annotated, ClassVar, List, Literal, Optional, Tuple

from fuser.compute_node import ComputeNode
from fuser.fused_math import ComputePipeline
from fuser.fused_operation import FusedOperation
from helpers.llk_params import (
    AccToDest,
    ApproximationMode,
    BroadcastType,
    ClearFP32DstAcc,
    DestSync,
    EltwiseBinaryReuseDestType,
    EnforceFP32Accumulation,
    L1Accumulation,
    MathFidelity,
    MathOperation,
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
    model_validator,
)


class UnarySfpuMathSchema(BaseModel):
    """Base schema for unary SFPU math nodes (type="UnarySfpu").

    Each architecture subclass sets _sfpu_cls (runtime SFPU class).
    Subclasses can add field validators for arch-specific operation checks.
    """

    model_config = ConfigDict(extra="forbid")

    _sfpu_cls: ClassVar = None
    _sfpu_ops: ClassVar[set] = set()

    type: Literal["UnarySfpu"]
    operation: MathOperation
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    dst_dest_tile_index: Annotated[int, Field(ge=0)] = 0
    fill_const_value: float = 1.0

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, v):
        if isinstance(v, str):
            try:
                v = MathOperation[v]
            except KeyError:
                raise ValueError(f"Unknown operation: {v}")
        if not isinstance(v, MathOperation):
            raise ValueError(f"Invalid operation: {v}")
        if v not in cls._sfpu_ops:
            raise ValueError(f"{v.name} is not a supported unary SFPU operation")
        return v

    def to_compute_node(self, operands, output):
        sfpu = type(self)._sfpu_cls(
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
    """Base schema for binary SFPU math nodes (type="BinarySfpu").

    Each architecture subclass sets _sfpu_cls (runtime SFPU class).
    Subclasses can add field validators for arch-specific operation checks.
    """

    model_config = ConfigDict(extra="forbid")

    _sfpu_cls: ClassVar = None
    _sfpu_ops: ClassVar[set] = set()

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
        if isinstance(v, str):
            try:
                v = MathOperation[v]
            except KeyError:
                raise ValueError(f"Unknown operation: {v}")
        if not isinstance(v, MathOperation):
            raise ValueError(f"Invalid operation: {v}")
        if v not in cls._sfpu_ops:
            raise ValueError(f"{v.name} is not a supported binary SFPU operation")
        return v

    def to_compute_node(self, operands, output):
        sfpu = type(self)._sfpu_cls(
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
    """Base schema for FPU math nodes (type="Fpu").

    Each architecture subclass sets _fpu_map, _unpacker_map, and _output_dims
    to wire in its arch-specific dicts — no method overrides needed.
    """

    model_config = ConfigDict(extra="forbid")

    _fpu_map: ClassVar[dict] = {}
    _unpacker_map: ClassVar[dict] = {}
    _output_dims: ClassVar[dict] = {}

    type: Literal["Fpu"]
    operation: str
    unpacker: Optional[str] = None
    broadcast_type: BroadcastType = BroadcastType.None_
    reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE
    reduce_pool: Optional[ReducePool] = None
    reduce_dim: Optional[ReduceDimension] = None
    enforce_fp32_accumulation: EnforceFP32Accumulation = EnforceFP32Accumulation.No
    acc_to_dest: AccToDest = AccToDest.No
    unpack_transpose_within_face: Transpose = Transpose.No
    unpack_transpose_faces: Transpose = Transpose.No
    math_fidelity: MathFidelity = MathFidelity.LoFi
    unpack_to_dest: UnpackToDest = UnpackToDest.No
    src_a: str = Field(..., min_length=1)
    src_b: str = Field(..., min_length=1)

    @field_validator("operation", mode="after")
    @classmethod
    def validate_operation(cls, v):
        if v not in cls._fpu_map:
            raise ValueError(f"Unknown FPU operation: {v}")
        return v

    @field_validator("unpacker", mode="after")
    @classmethod
    def validate_unpacker(cls, v):
        if v is not None and v not in cls._unpacker_map:
            raise ValueError(f"Unknown unpacker: {v}")
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

    def to_compute_node(self, operands, output):
        src_a = operands.get(self.src_a)
        src_b = operands.get(self.src_b)

        factory, checks = type(self)._fpu_map[self.operation]

        if checks is not None:
            for check, error_msg in checks:
                if check(self, src_a, src_b):
                    raise ValueError(error_msg)

        if self.unpacker is not None:
            unpacker_factory, unpacker_checks = type(self)._unpacker_map[self.unpacker]
            if unpacker_checks is not None:
                for check, error_msg in unpacker_checks:
                    if check(self, src_a, src_b):
                        raise ValueError(error_msg)

        fpu = factory(self)

        clear_fp32_dst_acc = (
            ClearFP32DstAcc.Yes
            if self.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
            or self.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB
            else ClearFP32DstAcc.No
        )

        kwargs = {
            "unpack_transpose_within_face": self.unpack_transpose_within_face,
            "unpack_transpose_faces": self.unpack_transpose_faces,
            "broadcast_type": self.broadcast_type,
            "reuse_dest": self.reuse_dest,
            "math_fidelity": self.math_fidelity,
            "enforce_fp32_accumulation": self.enforce_fp32_accumulation,
            "clear_fp32_dst_acc": clear_fp32_dst_acc,
            "acc_to_dest": self.acc_to_dest,
            "unpack_to_dest": self.unpack_to_dest,
        }
        if self.unpacker is not None:
            kwargs["unpacker"] = unpacker_factory(self)

        return ComputeNode(fpu=fpu, src_a=src_a, src_b=src_b, sfpu=None, **kwargs)

    def get_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        fn = type(self)._output_dims.get(self.operation)
        if fn is None:
            return None
        src_a = operands.get(self.src_a).dimensions
        src_b = operands.get(self.src_b).dimensions
        return fn(src_a, src_b)


class OperationSchemaBase(BaseModel):
    """Base schema for a fused operation with one output and one or more math nodes.

    Each architecture subclass sets _packer_map and adds its own math list field.
    Blackhole also overrides _arch_validate() for tilize detection and _arch_kwargs()
    to forward the bh_tilize flag to FusedOperation.
    """

    model_config = ConfigDict(extra="forbid")

    _packer_map: ClassVar[dict] = {}

    output: str = Field(..., min_length=1)
    packer: str = "Packer"
    dest_sync: DestSync = DestSync.Half
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    pack_relu: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    pack_l1_accumulation: L1Accumulation = L1Accumulation.No

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationSchemaBase":
        if self.packer not in type(self)._packer_map:
            raise ValueError(f"Unknown packer: {self.packer}")
        self._arch_validate()
        return self

    def _arch_validate(self):
        pass

    def _arch_kwargs(self) -> dict:
        return {}

    def _get_packer_class(self):
        cls, _ = type(self)._packer_map[self.packer]
        return cls

    def to_fused_operation(self, operands):
        output = operands.get(name=self.output)
        math_ops = [m.to_compute_node(operands, output) for m in self.math]
        output.is_output = True

        max_out_dims = self._calculate_max_output_dimensions(operands)
        resolved_max_out_dims = (
            max_out_dims if max_out_dims is not None else output.dimensions
        )

        _, checks = type(self)._packer_map[self.packer]
        if checks is not None:
            for check, error_msg in checks:
                if check(self, output):
                    raise ValueError(error_msg)

        reduce_dim = None
        for node in math_ops:
            if node.fpu is not None and hasattr(node.fpu, "reduce_dim"):
                reduce_dim = node.fpu.reduce_dim
                break

        kwargs = {
            "block_size": self.block_size,
            "pack_relu": self.pack_relu,
            "relu_threshold": self.relu_threshold,
            "pack_l1_accumulation": self.pack_l1_accumulation,
            "dest_sync": self.dest_sync,
            "reduce_dim": reduce_dim,
        }
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

        bound_r = min(d[0] for d in dims)
        bound_c = min(d[1] for d in dims)
        return (bound_r, bound_c)
