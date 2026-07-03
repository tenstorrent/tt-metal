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

from fuser.fpu_node import FpuNode
from fuser.fused_math import ComputePipeline
from fuser.fused_operation import FusedOperation
from fuser.pack_node import PackNode
from fuser.sfpu_node import SfpuNode
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
from helpers.tile_shape import TileShape, construct_tile_shape
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SUPPORTED_TILE_SIZES = {
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
    (16, 32),
    (32, 32),
    (16, 16),
    (32, 16),
}


def _tile_dims(ts: TileShape) -> Tuple[int, int]:
    return (ts.total_row_dim(), ts.total_col_dim())


def _is_sfpu_tile(dims: Tuple[int, int]) -> bool:
    return dims in ((16, 32), (32, 32), (32, 16))


def _has_transpose(schema) -> bool:
    return (
        schema.unpack_transpose_faces == Transpose.Yes
        or schema.unpack_transpose_within_face == Transpose.Yes
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

    def to_node(self, operands):
        sfpu = type(self)._sfpu_cls(
            self.operation,
            self.approximation_mode,
            self.iterations,
            self.dst_dest_tile_index,
            self.fill_const_value,
        )
        return SfpuNode(sfpu=sfpu)

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

    def to_node(self, operands):
        sfpu = type(self)._sfpu_cls(
            self.operation,
            self.approximation_mode,
            self.iterations,
            self.src1_dest_tile_index,
            self.src2_dest_tile_index,
            self.dst_dest_tile_index,
        )
        return SfpuNode(sfpu=sfpu)

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
    reduce_to_tile: bool = False
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

    def to_node(self, operands):
        src_a = operands.get(self.src_a)
        src_b = operands.get(self.src_b)

        factory, checks = type(self)._fpu_map[self.operation]

        if checks is not None:
            for check, error_msg in checks:
                if check(self, src_a, src_b):
                    raise ValueError(error_msg)

        if self.unpacker is not None:
            _, checks = type(self)._unpacker_map[self.unpacker]
            if checks is not None:
                for check, error_msg in checks:
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
            "reduce_to_tile": self.reduce_to_tile,
        }
        if self.unpacker is not None:
            unpacker_factory, _ = type(self)._unpacker_map[self.unpacker]
            kwargs["unpacker"] = unpacker_factory(self)

        return FpuNode(fpu=fpu, src_a=src_a, src_b=src_b, **kwargs)

    def get_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        fn = type(self)._output_dims.get(self.operation)
        if fn is None:
            return None
        src_a = operands.get(self.src_a).dimensions
        src_b = operands.get(self.src_b).dimensions
        return fn(src_a, src_b)


class PackSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _packer_map: ClassVar[dict] = {}

    type: Literal["Pack"] = "Pack"
    output: str = Field(..., min_length=1)
    packer: str = "Packer"
    pack_relu: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    pack_l1_accumulation: L1Accumulation = L1Accumulation.No

    @field_validator("packer", mode="after")
    @classmethod
    def validate_packer(cls, v):
        if cls._packer_map and v not in cls._packer_map:
            raise ValueError(f"Unknown packer: {v}")
        return v

    def to_node(self, operands):
        output = operands.get(name=self.output)
        output.is_output = True

        packer_cls, checks = type(self)._packer_map[self.packer]
        if checks is not None:
            for check, error_msg in checks:
                if check(self, output):
                    raise ValueError(error_msg)

        return PackNode(
            packer=packer_cls(),
            output=output,
            pack_relu=self.pack_relu,
            relu_threshold=self.relu_threshold,
            pack_l1_accumulation=self.pack_l1_accumulation,
        )


class OperationSchemaBase(BaseModel):
    """Base schema for a fused operation with one output and one or more math nodes.

    Each architecture subclass adds its own math and pack list fields.
    Blackhole also overrides _arch_validate() for tilize detection and _arch_kwargs()
    to forward the bh_tilize flag to FusedOperation.
    """

    model_config = ConfigDict(extra="forbid")

    dest_sync: DestSync = DestSync.Half
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    pack: List[PackSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationSchemaBase":
        if not any(isinstance(e, PackSchema) for e in self.pack):
            raise ValueError("pack list must contain at least one Pack entry")
        if not isinstance(self.pack[-1], PackSchema):
            raise ValueError("pack list must end with a Pack entry")

        self._arch_validate()
        return self

    def _arch_validate(self):
        pass

    def _arch_kwargs(self) -> dict:
        return {}

    def _resolve_output_tile_shape(self, operands) -> TileShape:
        """Resolve the output/dest tile shape for this operation.

        For most ops, all operands share the same tile shape. For matmul,
        output tile shape derives from input tile shapes: out_rows = in0_rows,
        out_cols = in1_cols.
        """
        output_tile_shapes = []

        for m in self.math:
            if not hasattr(m, "src_a"):
                continue
            src_a_ts = operands.get(m.src_a).tile_shape
            src_b_ts = operands.get(m.src_b).tile_shape

            if m.operation in ("Matmul", "MatmulNoMop"):
                out_tile_dims = (
                    src_a_ts.total_row_dim(),
                    src_b_ts.total_col_dim(),
                )
                output_tile_shapes.append(construct_tile_shape(out_tile_dims))
            else:
                if _tile_dims(src_a_ts) != _tile_dims(src_b_ts):
                    raise ValueError(
                        f"src_a tile shape {_tile_dims(src_a_ts)} != src_b tile shape "
                        f"{_tile_dims(src_b_ts)} for {m.operation}"
                    )
                output_tile_shapes.append(src_a_ts)

        pack_schemas = [e for e in self.pack if isinstance(e, PackSchema)]

        if not output_tile_shapes:
            output_tile_shapes = [
                operands.get(e.output).tile_shape for e in pack_schemas
            ]

        first = output_tile_shapes[0]
        for ts in output_tile_shapes[1:]:
            if _tile_dims(ts) != _tile_dims(first):
                raise ValueError(
                    f"All math nodes must produce the same output tile shape. "
                    f"Got {_tile_dims(first)} and {_tile_dims(ts)}"
                )

        for entry in pack_schemas:
            pack_ts = operands.get(entry.output).tile_shape
            if _tile_dims(pack_ts) != _tile_dims(first):
                raise ValueError(
                    f"Pack output '{entry.output}' tile shape {_tile_dims(pack_ts)} "
                    f"does not match computed output tile shape {_tile_dims(first)}"
                )

        return first

    def to_fused_operation(self, operands, dest_acc=False):
        tile_shape = self._resolve_output_tile_shape(operands)

        tile_r = tile_shape.total_row_dim()
        tile_c = tile_shape.total_col_dim()
        block_r, block_c = self.block_size

        if block_r % tile_r != 0 or block_c % tile_c != 0:
            raise ValueError(
                f"Block size ({self.block_size}) must be a multiple of tile dimensions "
                f"({tile_r}, {tile_c})"
            )

        block_tiles = (block_r // tile_r) * (block_c // tile_c)
        dest_faces = 32 if self.dest_sync == DestSync.Half else 64
        if dest_acc:
            dest_faces //= 2
        dest_tile_capacity = dest_faces // tile_shape.total_num_faces()

        if block_tiles > dest_tile_capacity:
            raise ValueError(
                f"Block size {self.block_size} requires {block_tiles} tiles "
                f"({block_tiles * tile_shape.total_num_faces()} faces) but dest can hold "
                f"{dest_tile_capacity} tiles ({dest_faces} faces) with "
                f"dest_sync={self.dest_sync.name}, dest_acc={dest_acc}"
            )

        pack_nodes = [entry.to_node(operands) for entry in self.pack]

        math_ops = [m.to_node(operands) for m in self.math]

        has_sfpu = any(isinstance(node, SfpuNode) for node in math_ops)
        has_fpu = any(isinstance(node, FpuNode) for node in math_ops)
        if has_sfpu and not has_fpu:
            dims = _tile_dims(tile_shape)
            if not _is_sfpu_tile(dims):
                raise ValueError(
                    f"Tile shape {dims} is not supported for SFPU operations. "
                    f"Supported: [(16, 32), (32, 16), (32, 32)]"
                )

        max_out_dims = self._calculate_max_output_dimensions(operands)

        reduce_dim = None
        for node in math_ops:
            if isinstance(node, FpuNode) and hasattr(node.fpu, "reduce_dim"):
                reduce_dim = node.fpu.reduce_dim
                break

        kwargs = {
            "block_size": self.block_size,
            "tile_shape": tile_shape,
            "dest_sync": self.dest_sync,
            "reduce_dim": reduce_dim,
        }
        kwargs.update(self._arch_kwargs())

        return FusedOperation(
            math=ComputePipeline(math_ops, pack_nodes),
            max_output_dimensions=max_out_dims,
            **kwargs,
        )

    def _calculate_max_output_dimensions(self, operands) -> Tuple[int, int]:
        dims = []
        for m in self.math:
            op_dims = m.get_output_dimensions(operands)
            if op_dims is not None:
                dims.append(op_dims)

        if not dims:
            dims = [
                operands.get(e.output).dimensions
                for e in self.pack
                if isinstance(e, PackSchema)
            ]

        bound_r = min(d[0] for d in dims)
        bound_c = min(d[1] for d in dims)
        return (bound_r, bound_c)
