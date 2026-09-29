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
    OUTPUT_DIMS          op name to lambda(in0, in1), set via _output_dims class attr
    UNARY/BINARY_SFPU_OPS  set of supported MathOperation, set via _sfpu_ops class attr
"""

from typing import Annotated, ClassVar, List, Literal, Optional, Tuple

from fuser.compute_pipeline import ComputePipeline
from fuser.fpu_node import FpuNode
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode
from fuser.sfpu_node import SfpuNode
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

SFPU_TILE_SIZES = {
    (16, 32),
    (32, 16),
    (32, 32),
}


def reject(condition, message):
    return (condition, message)


def format_tile_dims(shapes) -> str:
    return ", ".join(str(tuple(shape)) for shape in shapes)


def require_src_a_tiles(*shapes):
    return reject(
        lambda s, a, b: a is not None and a.tile_shape.tile_dims not in shapes,
        f"Only {format_tile_dims(shapes)} tiles are supported for this operation",
    )


def require_dest_tiles(*shapes):
    return reject(
        lambda s, a, b: s._dest_tile_shape is None
        or s._dest_tile_shape.tile_dims not in shapes,
        f"Only {format_tile_dims(shapes)} dest tiles are supported for this operation",
    )


def forced_unpackers(*names, when=None, note=None):
    def condition(s, a, b):
        if when is not None and not when(s, a, b):
            return False
        return s.unpacker is not None and s.unpacker not in names

    allowed = names[0] if len(names) == 1 else f"one of: {', '.join(names)}"
    return reject(condition, f"unpacker must be {allowed}{f' {note}' if note else ''}")


reuses_dest = lambda s, a, b: s.reuse_dest != EltwiseBinaryReuseDestType.NONE

eltwise_unpacker_rules = [
    forced_unpackers("UnpackerA", when=reuses_dest, note="when reuse_dest is set"),
    forced_unpackers(
        "UnpackerAB",
        when=lambda s, a, b: not reuses_dest(s, a, b),
        note="unless reuse_dest is set",
    ),
]

NO_BROADCAST = reject(
    lambda s, a, b: s.broadcast_type != BroadcastType.None_,
    "broadcast is not supported for this kernel",
)

NO_TRANSPOSE = reject(
    lambda s, a, b: s.has_transpose,
    "transpose is not supported for this kernel",
)

NO_TRANSPOSE_FACES = reject(
    lambda s, a, b: s.transpose_faces == Transpose.Yes,
    "transpose_faces is not supported, only transpose_within_face",
)

NO_UNPACK_TO_DEST = reject(
    lambda s, a, b: s.unpack_to_dest == UnpackToDest.Yes,
    "unpack_to_dest is not supported for this kernel",
)

NO_TRANSPOSE_UNPACK_TO_DEST = reject(
    lambda s, a, b: s.unpack_to_dest == UnpackToDest.Yes and s.has_transpose,
    "does not support transpose with unpack_to_dest",
)

NO_BROADCAST_REUSE_DEST = reject(
    lambda s, a, b: s.broadcast_type != BroadcastType.None_
    and s.reuse_dest != EltwiseBinaryReuseDestType.NONE,
    "broadcast does not support reuse_dest",
)

NO_BROADCAST_ACC_TO_DEST = reject(
    lambda s, a, b: s.broadcast_type != BroadcastType.None_
    and s.acc_to_dest == AccToDest.Yes,
    "broadcast does not support acc_to_dest",
)

INT32_NEEDS_UNPACK_TO_DEST = reject(
    lambda s, a, b: a.data_format == DataFormat.Int32
    and s.unpack_to_dest != UnpackToDest.Yes,
    "Int32 in0 requires unpack_to_dest: Yes (SrcA/SrcB registers are 19-bit wide)",
)

IN0_REQUIRED = reject(
    lambda s, a, b: a is None,
    "in0 is required: this operation reads an operand from L1",
)

IN1_REQUIRED = reject(
    lambda s, a, b: b is None,
    "in1 is required: this operation reads a second operand",
)

NO_REUSE_DEST = reject(
    reuses_dest, "reuse_dest is only supported for Eltwise operations"
)

DEST_TO_SRCA_NEEDS_ACC = reject(
    lambda s, a, b: s.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
    and s.acc_to_dest != AccToDest.Yes,
    "reuse_dest DEST_TO_SRCA requires acc_to_dest: true",
)

LOFI_ONLY = reject(
    lambda s, a, b: s.math_fidelity != MathFidelity.LoFi,
    "only LoFi math fidelity is supported for this operation",
)

REDUCE_PARAMS_REQUIRED = reject(
    lambda s, a, b: s.reduce_pool is None or s.reduce_dim is None,
    "Reduce requires both reduce_pool and reduce_dim",
)

MATMUL_OPERAND_DIMS = reject(
    lambda s, a, b: a is not None
    and b is not None
    and a.dimensions[1] != b.dimensions[0],
    "Matmul: incompatible dimensions for in0 and in1",
)

MATMUL_INNER_TILE_DIMS = reject(
    lambda s, a, b: a is not None
    and b is not None
    and (
        a.tile_shape.total_col_dim() != b.tile_shape.total_row_dim()
        or a.tile_shape.tile_dims == (16, 16)
        or b.tile_shape.tile_dims not in ((32, 32), (32, 16), (16, 32))
    ),
    "Matmul tile inner dimensions must match: in0 cols must equal in1 rows; "
    "in0 tile shape (16, 16) is not supported; "
    "in1 tile shape must be (32, 32), (32, 16), or (16, 32)",
)

SUPPORTED_SRC_A_TILE = reject(
    lambda s, a, b: a is not None
    and a.tile_shape.tile_dims not in SUPPORTED_TILE_SIZES,
    "Unsupported in0 tile shape",
)

TRANSPOSE_NEEDS_FULL_TILE = reject(
    lambda s, a, b: a is not None
    and s.has_transpose
    and a.tile_shape.tile_dims != (32, 32),
    "Only (32, 32) tiles are supported with transpose",
)

TRANSPOSE_WITHIN_FACE_REQUIRED = reject(
    lambda s, a, b: s.transpose_within_face != Transpose.Yes,
    "TransposeDest requires transpose_within_face = Yes",
)

SCALAR_BCAST_NO_TRANSPOSE_FACES = reject(
    lambda s, a, b: s.broadcast_type == BroadcastType.Scalar
    and s.transpose_faces == Transpose.Yes,
    "SrcA transpose is not supported with scalar broadcast",
)

NO_COL_ROW_BCAST_32X16 = reject(
    lambda s, a, b: a is not None
    and s.broadcast_type in (BroadcastType.Column, BroadcastType.Row)
    and a.tile_shape.tile_dims == (32, 16),
    "32x16 tiles are not supported for eltwise with column/row broadcast",
)

DATACOPY_TILE_32X32_ONLY = reject(
    lambda s, a, b: a is not None
    and a.tile_shape.tile_dims != (32, 32)
    and (
        s.has_transpose
        or s.broadcast_type in (BroadcastType.Column, BroadcastType.Row)
        or (
            s.unpack_to_dest == UnpackToDest.Yes
            and s.broadcast_type != BroadcastType.None_
        )
    ),
    "Only (32, 32) tiles are supported for Datacopy with transpose, col/row broadcast, or unpack-to-dest with broadcast",
)

SUB_BCAST_COL_REQUIRED = reject(
    lambda s, a, b: s.broadcast_type != BroadcastType.Column,
    'SubBcastColCustom requires broadcast_type: "COL"',
)

L1_ACC_FORMAT_SUPPORTED = reject(
    lambda s, output: s.pack_l1_accumulation == L1Accumulation.Yes
    and not output.data_format.supports_l1_accumulation(),
    "Output data format does not support L1 accumulation",
)

PACK_FULL_TILE_ONLY = reject(
    lambda s, output: output.tile_shape.total_num_faces() != 4,
    "PackUntilize supports only 32x32 output tiles",
)

PACK_NO_BLOCK_FLOAT = reject(
    lambda s, output: output.data_format.is_block_float(),
    "PackUntilize does not support block float output formats",
)

PACK_NO_L1_ACC = reject(
    lambda s, output: s.pack_l1_accumulation == L1Accumulation.Yes,
    "PackUntilize does not support L1 accumulation",
)


ELTWISE_DIMS = lambda a, b: (min(a[0], b[0]), min(a[1], b[1]))
MATMUL_DIMS = lambda a, b: (a[0], b[1])
SRC_A_DIMS = lambda a, b: a


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
                valid_ops = sorted(op.name for op in cls._sfpu_ops)
                raise ValueError(
                    f"Unknown operation: {v}, expected one of: {', '.join(valid_ops)}"
                )
        if not isinstance(v, MathOperation):
            raise ValueError(f"Invalid operation: {v}")
        if v not in cls._sfpu_ops:
            valid_ops = sorted(op.name for op in cls._sfpu_ops)
            raise ValueError(
                f"{v.name} is not a supported unary SFPU operation, expected one of: {', '.join(valid_ops)}"
            )
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
                valid_ops = sorted(op.name for op in cls._sfpu_ops)
                raise ValueError(
                    f"Unknown operation: {v}, expected one of: {', '.join(valid_ops)}"
                )
        if not isinstance(v, MathOperation):
            raise ValueError(f"Invalid operation: {v}")
        if v not in cls._sfpu_ops:
            valid_ops = sorted(op.name for op in cls._sfpu_ops)
            raise ValueError(
                f"{v.name} is not a supported binary SFPU operation, expected one of: {', '.join(valid_ops)}"
            )
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
    _dest_tile_shape: Optional[TileShape] = None

    type: Literal["Fpu"]
    operation: str
    unpacker: Optional[str] = None
    broadcast_type: BroadcastType = BroadcastType.None_
    broadcast_tile: Optional[Annotated[int, Field(ge=0)]] = None
    reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE
    reduce_pool: Optional[ReducePool] = None
    reduce_dim: Optional[ReduceDimension] = None
    enforce_fp32_accumulation: EnforceFP32Accumulation = EnforceFP32Accumulation.No
    acc_to_dest: AccToDest = AccToDest.No
    transpose_within_face: Transpose = Transpose.No
    transpose_faces: Transpose = Transpose.No
    math_fidelity: MathFidelity = MathFidelity.LoFi
    unpack_to_dest: UnpackToDest = UnpackToDest.No
    reduce_to_tile: bool = False
    in0: Optional[str] = None
    in1: Optional[str] = None

    @property
    def has_transpose(self) -> bool:
        return (
            self.transpose_faces == Transpose.Yes
            or self.transpose_within_face == Transpose.Yes
        )

    @field_validator("operation", mode="after")
    @classmethod
    def validate_operation(cls, v):
        if v not in cls._fpu_map:
            valid_ops = sorted(cls._fpu_map.keys())
            raise ValueError(
                f"Unknown FPU operation: {v}, expected one of: {', '.join(valid_ops)}"
            )
        return v

    @model_validator(mode="after")
    def validate_broadcast_tile(self) -> "FpuMathSchemaBase":
        if (
            self.broadcast_tile is not None
            and self.broadcast_type == BroadcastType.None_
        ):
            raise ValueError("broadcast_tile requires a broadcast_type")
        return self

    @field_validator("unpacker", mode="after")
    @classmethod
    def validate_unpacker(cls, v):
        if v is not None and v not in cls._unpacker_map:
            valid_ops = sorted(cls._unpacker_map.keys())
            raise ValueError(
                f"Unknown unpacker: {v}, expected one of: {', '.join(valid_ops)}"
            )
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
        src_a = None
        if self.in0 is not None:
            src_a = operands.get(self.in0)
            src_a.is_input = True
        src_b = None
        if self.in1 is not None:
            src_b = operands.get(self.in1)
            src_b.is_input = True

        factory, checks = type(self)._fpu_map[self.operation]

        if self.unpacker is not None:
            _, unpacker_checks = type(self)._unpacker_map[self.unpacker]
            for check, error_msg in unpacker_checks:
                if check(self, src_a, src_b):
                    raise ValueError(error_msg)

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
            "transpose_within_face": self.transpose_within_face,
            "transpose_faces": self.transpose_faces,
            "broadcast_type": self.broadcast_type,
            "broadcast_tile": self.broadcast_tile,
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
        if fn is None or self.in0 is None:
            return None
        src_a = operands.get(self.in0).dimensions
        src_b = operands.get(self.in1).dimensions if self.in1 is not None else src_a
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
        if v not in cls._packer_map:
            valid_ops = sorted(cls._packer_map.keys())
            raise ValueError(
                f"Unknown packer: {v}, expected one of: {', '.join(valid_ops)}"
            )
        return v

    def to_node(self, operands):
        output = operands.get(name=self.output)
        output.is_output = True

        packer_cls, checks = type(self)._packer_map[self.packer]
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
    """Base schema for a fused operation: one or more math nodes and one or more packs.

    Each architecture subclass adds its own math and pack list fields. Blackhole also
    overrides _arch_validate() to reject mixed unpackers with UnpackerTilizeA, and
    _arch_kwargs() to forward the bh_tilize flag to L1Operation.
    """

    model_config = ConfigDict(extra="forbid")

    dest_consuming_operations: ClassVar[frozenset] = frozenset()

    dest_sync: DestSync = DestSync.Half
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    pack: List[PackSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationSchemaBase":
        if not self.pack_schemas:
            raise ValueError("pack list must contain at least one Pack entry")
        if not isinstance(self.pack[-1], PackSchema):
            raise ValueError("pack list must end with a Pack entry")

        self._validate_dest_consumers()
        self._arch_validate()
        return self

    def _validate_dest_consumers(self):
        first = self.math[0] if self.math else None
        operation = getattr(first, "operation", None)
        reuses_dest = (
            getattr(first, "reuse_dest", None) is not None
            and first.reuse_dest != EltwiseBinaryReuseDestType.NONE
        )
        if operation in type(self).dest_consuming_operations or reuses_dest:
            raise ValueError(
                f"{operation} cannot be the first math operation: "
                "Dst must already contain data"
            )

    @property
    def pack_schemas(self) -> List[PackSchema]:
        """The Pack entries of this operation, without the interleaved SFPU nodes."""
        return [entry for entry in self.pack if isinstance(entry, PackSchema)]

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
            if getattr(m, "in0", None) is None:
                continue
            src_a_ts = operands.get(m.in0).tile_shape
            src_b_ts = operands.get(m.in1).tile_shape if m.in1 is not None else src_a_ts

            if m.operation in ("Matmul", "MatmulNoMop"):
                out_tile_dims = (
                    src_a_ts.total_row_dim(),
                    src_b_ts.total_col_dim(),
                )
                output_tile_shapes.append(construct_tile_shape(out_tile_dims))
            else:
                if src_a_ts.tile_dims != src_b_ts.tile_dims:
                    raise ValueError(
                        f"in0 tile shape {src_a_ts.tile_dims} != in1 tile shape "
                        f"{src_b_ts.tile_dims} for {m.operation}"
                    )
                output_tile_shapes.append(src_a_ts)

        if not output_tile_shapes:
            output_tile_shapes = [
                operands.get(e.output).tile_shape for e in self.pack_schemas
            ]

        first = output_tile_shapes[0]
        for ts in output_tile_shapes[1:]:
            if ts.tile_dims != first.tile_dims:
                raise ValueError(
                    f"All math nodes must produce the same output tile shape. "
                    f"Got {first.tile_dims} and {ts.tile_dims}"
                )

        for entry in self.pack_schemas:
            pack_ts = operands.get(entry.output).tile_shape
            if pack_ts.tile_dims != first.tile_dims:
                raise ValueError(
                    f"Pack output '{entry.output}' tile shape {pack_ts.tile_dims} "
                    f"does not match computed output tile shape {first.tile_dims}"
                )

        return first

    def to_l1_operation(self, operands, dest_acc=False):
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

        for p in self.pack:
            p._block_size = self.block_size
        for m in self.math:
            m._block_size = self.block_size
            if isinstance(m, FpuMathSchemaBase):
                m._dest_tile_shape = tile_shape

        pack_nodes = []
        for i, p in enumerate(self.pack):
            try:
                pack_nodes.append(p.to_node(operands))
            except ValueError as e:
                raise ValueError(f"Pack entry {i + 1}\n    {e}") from None

        math_ops = []
        for i, m in enumerate(self.math):
            node_type = getattr(m, "type", type(m).__name__)
            try:
                math_ops.append(m.to_node(operands))
            except ValueError as e:
                raise ValueError(f"Math node {i + 1} ({node_type})\n    {e}") from None

        has_sfpu = any(isinstance(node, SfpuNode) for node in math_ops)
        has_fpu = any(isinstance(node, FpuNode) for node in math_ops)
        if has_sfpu and not has_fpu:
            dims = tile_shape.tile_dims
            if dims not in SFPU_TILE_SIZES:
                raise ValueError(
                    f"Tile shape {dims} is not supported for SFPU operations. "
                    f"Supported: {format_tile_dims(sorted(SFPU_TILE_SIZES))}"
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

        return L1Operation(
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
            dims = [operands.get(e.output).dimensions for e in self.pack_schemas]

        bound_r = min(d[0] for d in dims)
        bound_c = min(d[1] for d in dims)
        return (bound_r, bound_c)
