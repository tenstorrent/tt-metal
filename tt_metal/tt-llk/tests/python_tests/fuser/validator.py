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

import re
from typing import Annotated, ClassVar, List, Literal, Optional, Tuple

from fuser.compute_pipeline import ComputePipeline
from fuser.fpu_node import FpuNode
from fuser.isolate_sfpu_node import IsolateSfpuNode
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

# Reserved token used by isolate_sfpu nodes to reference the Dest register file
# instead of an L1 operand. An operand may not be named this.
DEST = "dest"

# A dest path token is 'dest' (tile 0) or 'dest[<tile index>]' — the tile index
# selects which dest tile within the block is read/written.
_DEST_PATH_RE = re.compile(r"^dest(?:\[(\d+)\])?$")


def is_dest_path(value) -> bool:
    return isinstance(value, str) and _DEST_PATH_RE.fullmatch(value) is not None


def dest_path_index(value) -> Optional[int]:
    """Return the dest tile index for a 'dest'/'dest[i]' token, else None."""
    if not isinstance(value, str):
        return None
    m = _DEST_PATH_RE.fullmatch(value)
    if m is None:
        return None
    return int(m.group(1)) if m.group(1) is not None else 0


def is_isolate_node(node) -> bool:
    """True if the math entry is a TRISC3 isolate Sfpu node."""
    return getattr(node, "type", None) in ("IsolateUnarySfpu", "IsolateBinarySfpu")


def reject(condition, message):
    return (condition, message)


def require_src_a_tiles(*shapes):
    return reject(
        lambda s, a, b: a.tile_shape.tile_dims not in shapes,
        f"Only {shapes} tiles are supported for this operation",
    )


def forced_unpackers(*names, when=None, note=None):
    def condition(s, a, b):
        if when is not None and not when(s, a, b):
            return False
        return s.unpacker is not None and s.unpacker not in names

    allowed = names[0] if len(names) == 1 else f"one of: {', '.join(names)}"
    return reject(condition, f"unpacker must be {allowed}{f' {note}' if note else ''}")


reuses_dest = lambda s, a, b: s.reuse_dest != EltwiseBinaryReuseDestType.NONE

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

INT32_NEEDS_UNPACK_TO_DEST = reject(
    lambda s, a, b: a.data_format == DataFormat.Int32
    and s.unpack_to_dest != UnpackToDest.Yes,
    "Int32 src_a requires unpack_to_dest: Yes (SrcA/SrcB registers are 19-bit wide)",
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
    lambda s, a, b: a.dimensions[1] != b.dimensions[0],
    "Matmul: incompatible dimensions for src_a and src_b",
)

MATMUL_INNER_TILE_DIMS = reject(
    lambda s, a, b: (
        a.tile_shape.total_col_dim() != b.tile_shape.total_row_dim()
        or a.tile_shape.tile_dims == (16, 16)
        or b.tile_shape.tile_dims not in ((32, 32), (32, 16), (16, 32))
    ),
    "Matmul tile inner dimensions must match: in0 cols must equal in1 rows; "
    "in0 tile shape (16, 16) is not supported; "
    "in1 tile shape must be (32, 32), (32, 16), or (16, 32)",
)

SUPPORTED_SRC_A_TILE = reject(
    lambda s, a, b: a.tile_shape.tile_dims not in SUPPORTED_TILE_SIZES,
    "Unsupported src_a tile shape",
)

TRANSPOSE_NEEDS_FULL_TILE = reject(
    lambda s, a, b: s.has_transpose and a.tile_shape.tile_dims != (32, 32),
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
    lambda s, a, b: s.broadcast_type in (BroadcastType.Column, BroadcastType.Row)
    and a.tile_shape.tile_dims == (32, 16),
    "32x16 tiles are not supported for eltwise with column/row broadcast",
)

DATACOPY_TILE_32X32_ONLY = reject(
    lambda s, a, b: a.tile_shape.tile_dims != (32, 32)
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
SRC_B_DIMS = lambda a, b: b


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


class _IsolateSfpuSchemaBase(BaseModel):
    """Shared schema for isolate_sfpu nodes (Quasar TRISC3 SrcS data path).

    src_a/src_b/output reference an L1 operand (SrcS path) or a dest token
    'dest' / 'dest[<tile index>]' (Dest path, tile index within the block).
    Each architecture subclass sets _sfpu_map, mapping each supported
    MathOperation to the IsolateSfpu class implementing it on the third
    thread's SrcS register file. Isolate nodes are entries of the ordered math
    list, so they interleave with Fpu nodes in pipeline order.
    """

    model_config = ConfigDict(extra="forbid")

    _sfpu_map: ClassVar[dict] = {}

    operation: MathOperation
    approximation_mode: ApproximationMode = ApproximationMode.No
    iterations: Annotated[int, Field(ge=1)] = 8
    src_a: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)

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
        if v not in cls._sfpu_map:
            supported = sorted(op.name for op in cls._sfpu_map)
            raise ValueError(
                f"{v.name} is not a supported isolate SFPU operation "
                f"(supported: {supported})"
            )
        return v

    @field_validator("src_a", "output", mode="after")
    @classmethod
    def validate_path(cls, v):
        if is_dest_path(v):
            return v
        if v.startswith("dest"):
            raise ValueError(
                f"invalid dest path '{v}': expected 'dest' or 'dest[<tile index>]'"
            )
        return v

    def _resolve_operand(self, name: str, operands, is_input: bool):
        if is_dest_path(name):
            return None
        operand = operands.get(name)
        if operand is None:
            raise ValueError(f"Operand '{name}' not found")
        if is_input:
            operand.is_input = True
        else:
            operand.is_output = True
        return operand

    def _resolve_path(self, name: str, operands, is_input: bool):
        return (self._resolve_operand(name, operands, is_input), dest_path_index(name))

    def to_node(self, operands):
        src_a, src_a_dest_idx = self._resolve_path(self.src_a, operands, is_input=True)
        output, output_dest_idx = self._resolve_path(
            self.output, operands, is_input=False
        )
        return self._build_node(
            src_a, src_a_dest_idx, output, output_dest_idx, operands
        )

    def get_output_dimensions(self, operands) -> Optional[Tuple[int, int]]:
        return None


class IsolateUnarySfpuMathSchema(_IsolateSfpuSchemaBase):
    """Base schema for unary isolate_sfpu nodes (type="IsolateUnarySfpu")."""

    type: Literal["IsolateUnarySfpu"]

    def _build_node(self, src_a, src_a_dest_idx, output, output_dest_idx, operands):
        sfpu = type(self)._sfpu_map[self.operation](
            self.operation,
            self.approximation_mode,
            self.iterations,
        )
        return IsolateSfpuNode(
            sfpu=sfpu,
            src_a=src_a,
            src_a_dest_index=src_a_dest_idx,
            output=output,
            output_dest_index=output_dest_idx,
        )


class IsolateBinarySfpuMathSchema(_IsolateSfpuSchemaBase):
    """Base schema for binary isolate_sfpu nodes (type="IsolateBinarySfpu")."""

    type: Literal["IsolateBinarySfpu"]
    src_b: str = Field(..., min_length=1)

    @field_validator("src_b", mode="after")
    @classmethod
    def validate_src_b_path(cls, v):
        return cls.validate_path(v)

    def _build_node(self, src_a, src_a_dest_idx, output, output_dest_idx, operands):
        src_b, src_b_dest_idx = self._resolve_path(self.src_b, operands, is_input=True)
        sfpu = type(self)._sfpu_map[self.operation](
            self.operation,
            self.approximation_mode,
            self.iterations,
        )
        return IsolateSfpuNode(
            sfpu=sfpu,
            src_a=src_a,
            src_a_dest_index=src_a_dest_idx,
            src_b=src_b,
            src_b_dest_index=src_b_dest_idx,
            output=output,
            output_dest_index=output_dest_idx,
        )


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
    transpose_within_face: Transpose = Transpose.No
    transpose_faces: Transpose = Transpose.No
    math_fidelity: MathFidelity = MathFidelity.LoFi
    unpack_to_dest: UnpackToDest = UnpackToDest.No
    reduce_to_tile: bool = False
    src_a: str = Field(..., min_length=1)
    src_b: str = Field(..., min_length=1)

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
        src_a.is_input = True
        src_b = operands.get(self.src_b)
        src_b.is_input = True

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
            "transpose_within_face": self.transpose_within_face,
            "transpose_faces": self.transpose_faces,
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
    to forward the bh_tilize flag to L1Operation.
    """

    model_config = ConfigDict(extra="forbid")

    dest_consuming_operations: ClassVar[frozenset] = frozenset()

    dest_sync: DestSync = DestSync.Half
    block_size: Annotated[List[int], Field(min_length=2, max_length=2)] = [32, 32]
    pack: List[PackSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> "OperationSchemaBase":
        if not self.math and not self.pack:
            raise ValueError("operation must have at least one math or pack entry")
        if self.pack:
            if not any(isinstance(e, PackSchema) for e in self.pack):
                raise ValueError("pack list must contain at least one Pack entry")
            if not isinstance(self.pack[-1], PackSchema):
                raise ValueError("pack list must end with a Pack entry")

        self._validate_dest_consumers()
        self._validate_isolate_sfpu()
        self._validate_dest_ordering()
        self._arch_validate()
        return self

    def _validate_dest_consumers(self):
        if not self.math:
            return
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

    @staticmethod
    def _is_isolate(node) -> bool:
        return is_isolate_node(node)

    def _isolate_nodes(self):
        """TRISC3 isolate entries of the ordered math list, in pipeline order."""
        return [m for m in self.math if type(self)._is_isolate(m)]

    def _math_thread_nodes(self):
        """MATH-thread entries of the ordered math list (Fpu / inline Sfpu)."""
        return [m for m in self.math if not type(self)._is_isolate(m)]

    @staticmethod
    def _is_inline_sfpu(node) -> bool:
        return getattr(node, "type", None) in ("UnarySfpu", "BinarySfpu")

    def _dest_access(self, node) -> Tuple[bool, bool]:
        """(reads_dest, writes_dest) for one math entry.

        Fpu nodes always write dest (their result goes to dest) and additionally
        read it when using unpack_to_dest, reuse_dest, or a dest-consuming
        operation. Inline Sfpu nodes read dest (SrcA) and write it back. Isolate
        nodes read/write dest only where a path token is a dest path.
        """
        t = getattr(node, "type", None)
        if t == "Fpu":
            reads = (
                node.unpack_to_dest == UnpackToDest.Yes
                or getattr(node, "reuse_dest", None)
                not in (None, EltwiseBinaryReuseDestType.NONE)
                or node.operation in type(self).dest_consuming_operations
            )
            return reads, True
        if t in ("UnarySfpu", "BinarySfpu"):
            return True, True
        if type(self)._is_isolate(node):
            reads = is_dest_path(node.src_a) or is_dest_path(
                getattr(node, "src_b", None)
            )
            writes = is_dest_path(node.output)
            return reads, writes
        return False, False

    def _validate_isolate_sfpu(self):
        """Cross-checks between TRISC3 isolate nodes and the rest of the pipeline.

        The shared SFPU unit has a single dest dvalid client slot, so an operation
        cannot mix MATH-thread inline Sfpu nodes with TRISC3 isolate nodes. The
        SrcS register file has two input slots but no SrcS+SrcS helper yet, so a
        binary isolate node needs at least one dest operand.
        """
        if not self._isolate_nodes():
            return

        math_types = {getattr(m, "type", None) for m in self.math}
        pack_types = {getattr(p, "type", None) for p in self.pack}
        if math_types & {"UnarySfpu", "BinarySfpu"} or pack_types & {
            "UnarySfpu",
            "BinarySfpu",
        }:
            raise ValueError(
                "cannot mix inline UnarySfpu/BinarySfpu (MATH or PACK thread) with "
                "isolate Sfpu nodes (TRISC3): both drive the shared SFPU unit"
            )

        for node in self._isolate_nodes():
            if node.type == "IsolateBinarySfpu" and not (
                is_dest_path(node.src_a) or is_dest_path(node.src_b)
            ):
                raise ValueError(
                    "binary isolate_sfpu with both operands from L1 (SrcS+SrcS) is "
                    "not supported yet: at least one of src_a/src_b must be a dest "
                    "path ('dest' or 'dest[<tile index>]')"
                )

    def _validate_dest_ordering(self):
        """Every dest read must follow a dest producer earlier in the pipeline.

        Walks the ordered math list tracking whether dest has been produced. A
        node may read dest if it was produced earlier, or if the node itself
        writes dest first (e.g. an unpack_to_dest Fpu node self-produces before
        its own read). The pack list reads dest, so it also requires a producer.
        """
        produced_dest = False
        for node in self.math:
            reads, writes = self._dest_access(node)
            if reads and not (produced_dest or writes):
                raise ValueError(
                    f"math node of type '{getattr(node, 'type', None)}' reads 'dest' "
                    "but nothing earlier in the math list has written dest"
                )
            if writes:
                produced_dest = True

        if self.pack and not produced_dest:
            raise ValueError(
                "pack list requires a dest producer: add a math node that writes "
                "dest (an Fpu node or an isolate_sfpu node writing 'dest')"
            )

        # The FPU_SFPU handshake issues one MATH->TRISC3 token per isolate dest
        # read, produced by the immediately preceding MATH-thread node. An isolate
        # dest read must therefore be adjacent to a MATH-thread producer; a run of
        # consecutive isolate dest reads would need an isolate->isolate handoff,
        # which is not supported yet.
        for i, node in enumerate(self.math):
            if not type(self)._is_isolate(node):
                continue
            reads_dest = is_dest_path(node.src_a) or is_dest_path(
                getattr(node, "src_b", None)
            )
            if not reads_dest:
                continue
            prev = self.math[i - 1] if i > 0 else None
            if prev is None or type(self)._is_isolate(prev):
                raise ValueError(
                    "isolate_sfpu node reading 'dest' must be immediately preceded "
                    "by a MATH-thread node (its dest producer); consecutive isolate "
                    "dest reads are not supported by the FPU_SFPU handshake yet"
                )

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

        for m in self._math_thread_nodes():
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
                if src_a_ts.tile_dims != src_b_ts.tile_dims:
                    raise ValueError(
                        f"src_a tile shape {src_a_ts.tile_dims} != src_b tile shape "
                        f"{src_b_ts.tile_dims} for {m.operation}"
                    )
                output_tile_shapes.append(src_a_ts)

        pack_schemas = [e for e in self.pack if isinstance(e, PackSchema)]

        if not output_tile_shapes:
            output_tile_shapes = [
                operands.get(e.output).tile_shape for e in pack_schemas
            ]

        for node in self._isolate_nodes():
            if not is_dest_path(node.output):
                output_tile_shapes.append(operands.get(node.output).tile_shape)

        if not output_tile_shapes:
            raise ValueError(
                "could not resolve an output tile shape: operation has no math, "
                "pack, or isolate_sfpu entries with an L1 output"
            )

        first = output_tile_shapes[0]
        for ts in output_tile_shapes[1:]:
            if ts.tile_dims != first.tile_dims:
                raise ValueError(
                    f"All math nodes must produce the same output tile shape. "
                    f"Got {first.tile_dims} and {ts.tile_dims}"
                )

        for entry in pack_schemas:
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
        uses_dest = bool(self.pack) or any(
            any(rw) for rw in (self._dest_access(m) for m in self.math)
        )
        if uses_dest:
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

        for node in self._isolate_nodes():
            for path_name, path in (
                ("src_a", node.src_a),
                ("src_b", getattr(node, "src_b", None)),
                ("output", node.output),
            ):
                idx = dest_path_index(path)
                if idx is not None and idx >= block_tiles:
                    raise ValueError(
                        f"isolate_sfpu {path_name} references dest tile index "
                        f"{idx} but block size {self.block_size} only has "
                        f"{block_tiles} tiles"
                    )

        for p in self.pack:
            p._block_size = self.block_size
        for m in self.math:
            m._block_size = self.block_size

        pack_nodes = [p.to_node(operands) for p in self.pack]

        # Stamp each built node with its position in the merged, order-preserving
        # math[] list so codegen can reconstruct the exact FPU<->isolate-SFPU
        # interleave (used for the FPU_SFPU/SFPU_FPU handshake placement).
        merged_order = {id(m): i for i, m in enumerate(self.math)}

        def build_ordered(schemas):
            nodes = []
            for m in schemas:
                node = m.to_node(operands)
                node.order_index = merged_order[id(m)]
                node._dest_access = self._dest_access(m)
                nodes.append(node)
            return nodes

        math_ops = build_ordered(self._math_thread_nodes())
        isolate_nodes = build_ordered(self._isolate_nodes())

        has_sfpu = any(isinstance(node, SfpuNode) for node in math_ops)
        has_fpu = any(isinstance(node, FpuNode) for node in math_ops)
        if has_sfpu and not has_fpu:
            dims = tile_shape.tile_dims
            if dims not in ((16, 32), (32, 32), (32, 16)):
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

        return L1Operation(
            math=ComputePipeline(
                math_ops, pack_nodes, isolate_sfpu_nodes=isolate_nodes
            ),
            max_output_dimensions=max_out_dims,
            **kwargs,
        )

    def _calculate_max_output_dimensions(self, operands) -> Tuple[int, int]:
        dims = []
        for m in self._math_thread_nodes():
            op_dims = m.get_output_dimensions(operands)
            if op_dims is not None:
                dims.append(op_dims)

        if not dims:
            dims = [
                operands.get(e.output).dimensions
                for e in self.pack
                if isinstance(e, PackSchema)
            ]

        if not dims:
            dims = [
                operands.get(n.output).dimensions
                for n in self._isolate_nodes()
                if not is_dest_path(n.output)
            ]

        if not dims:
            raise ValueError(
                "could not resolve output dimensions: operation has no math, pack, "
                "or isolate_sfpu entries with an L1 output"
            )

        bound_r = min(d[0] for d in dims)
        bound_c = min(d[1] for d in dims)
        return (bound_r, bound_c)
