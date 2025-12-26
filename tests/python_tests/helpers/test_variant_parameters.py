# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from ctypes import c_uint32
from dataclasses import dataclass
from typing import Optional, Tuple

from .llk_params import (
    FPU_BINARY_OPERATIONS,
    REDUCE_OPERATIONS,
    SFPU_BINARY_OPERATIONS,
    SFPU_UNARY_OPERATIONS,
    ApproximationMode,
    BroadcastType,
    DataCopyType,
    DestSync,
    EltwiseBinaryReuseDestType,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    NarrowTile,
    PerfRunType,
    ReducePool,
    StochasticRounding,
    Tilize,
    Transpose,
    UnpackerEngine,
)
from .matmul_sweep import validate_tile_dimensions

# Base parameter classes


@dataclass
class TemplateParameter(ABC):
    @abstractmethod
    def covert_to_cpp(self) -> str:
        pass


@dataclass
class RuntimeParameter:

    @abstractmethod
    def covert_to_cpp(self) -> str:
        pass

    @abstractmethod
    def convert_to_struct_fields(self) -> tuple[str, str]:
        pass


# === TEMPLATE PARAMETER IMPLEMENTATIONS ===


@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    level: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.level};"


@dataclass
class MATH_TRANSPOSE_FACES(TemplateParameter):
    do_or_not: Transpose

    def covert_to_cpp(self) -> str:
        return f"constexpr bool MATH_TRANSPOSE_FACES = {str(self.do_or_not.value).lower()};"


@dataclass
class STOCHASTIC_ROUNDING(TemplateParameter):
    type: StochasticRounding

    def covert_to_cpp(self) -> str:
        return f"constexpr auto STOCHASTIC_RND = ckernel::{self.type.value};"


@dataclass
class DATA_COPY_TYPE(TemplateParameter):
    type: DataCopyType

    def covert_to_cpp(self) -> str:
        return (
            f"constexpr auto DATA_COPY_TYPE = ckernel::DataCopyType::{self.type.value};"
        )


@dataclass
class BROADCAST_TYPE(TemplateParameter):
    type: BroadcastType

    def covert_to_cpp(self) -> str:
        return f"constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::{self.type.value};"


@dataclass
class ACC_TO_DEST(TemplateParameter):
    value: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool ACC_TO_DEST = {str(self.value).lower()};"


@dataclass
class REUSE_DEST_TYPE(TemplateParameter):
    type: EltwiseBinaryReuseDestType

    def covert_to_cpp(self) -> str:
        return f"constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::{self.type.name};"


def _generate_operation_constants(mathop: MathOperation) -> list[str]:
    """Generate the appropriate operation constants based on the math operation type."""
    constants = []

    if mathop in SFPU_UNARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{mathop.cpp_enum_value};"
        )
    elif mathop in SFPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::{mathop.cpp_enum_value};"
        )
    elif mathop in FPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::{mathop.cpp_enum_value};"
        )

    return constants


@dataclass
class MATH_OP(TemplateParameter):
    mathop: MathOperation = None
    unary_extra: MathOperation = None
    pool_type: ReducePool = None

    def covert_to_cpp(self) -> str:
        temp_header = []
        if self.mathop:
            temp_header.append("\n// Math operation configuration")
            temp_header.extend(_generate_operation_constants(self.mathop))

            # Handle reduce operations
            if self.mathop in REDUCE_OPERATIONS:
                temp_header.append(
                    f"constexpr auto REDUCE_DIM = ckernel::ReduceDim::{self.mathop.cpp_enum_value};"
                )
                if self.pool_type:
                    temp_header.append(
                        f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.pool_type.value};"
                    )

        # Optional extra unary operation (used when both a binary and unary op
        # need to be present in the same kernel, e.g. binary-eltwise followed by
        # SFPU unary).  If 'unary_op' exists, append its constant.
        # Only add if we haven't already added a unary operation from the main mathop
        if self.unary_extra and (
            self.mathop is None or self.mathop not in SFPU_UNARY_OPERATIONS
        ):
            temp_header.extend(
                [
                    "\n// Additional SFPU unary operation",
                    f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{self.unary_extra.cpp_enum_value};",
                ]
            )

        return "\n".join(temp_header)


@dataclass
class DISABLE_SRC_ZERO_FLAG(TemplateParameter):
    value: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool disable_src_zero_flag = {str(self.value).lower()};"


@dataclass
class MATH_FIDELITY(TemplateParameter):
    fidelity: MathFidelity

    def covert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t MATH_FIDELITY = {self.fidelity.value};"


@dataclass
class APPROX_MODE(TemplateParameter):
    mode: ApproximationMode = ApproximationMode.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool APPROX_MODE = {self.mode.value};"


@dataclass
class DEST_SYNC(TemplateParameter):
    mode: DestSync = DestSync.Half

    def covert_to_cpp(self) -> str:
        return f"constexpr auto dest_sync = ckernel::DstSync::Sync{self.mode.name};"


@dataclass
class TILIZE(TemplateParameter):
    choice: Tilize = Tilize.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool tilize_en = {str(self.choice.value).lower()};"


@dataclass
class IMPLIED_MATH_FORMAT(TemplateParameter):
    data: ImpliedMathFormat = ImpliedMathFormat.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool IMPLIED_MATH_FORMAT = {self.data.value};"


@dataclass
class UNPACKER_ENGINE_SEL(TemplateParameter):
    type: UnpackerEngine = UnpackerEngine.UnpA

    def covert_to_cpp(self) -> str:
        return f"constexpr uint UNPACKER_ENGINE_SEL = p_unpacr::{self.type.value};"


@dataclass
class PERF_RUN_TYPE(TemplateParameter):
    type: PerfRunType

    def covert_to_cpp(self) -> str:
        return f"\nconstexpr auto PERF_RUN_TYPE = PerfRunType::{self.type.name};"


@dataclass
class REDUCE_POOL_TYPE(TemplateParameter):
    type: ReducePool

    def covert_to_cpp(self) -> str:
        return f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.type.value};"


@dataclass
class INPUT_DIMENSIONS(TemplateParameter):
    srcA: Tuple[int, int]
    srcB: Tuple[int, int]
    block_ct_dim: Optional[int] = None
    block_rt_dim: Optional[int] = None

    def covert_to_cpp(self) -> str:
        num_rows, num_cols = 32, 32
        validate_tile_dimensions(self.srcA[0], num_rows)
        validate_tile_dimensions(self.srcA[1], num_cols)
        validate_tile_dimensions(self.srcB[0], num_rows)
        validate_tile_dimensions(self.srcB[1], num_cols)

        full_ct_dim = self.srcB[1] // num_cols
        full_rt_dim = self.srcA[0] // num_rows

        block_ct_dim = full_ct_dim if self.block_ct_dim is None else self.block_ct_dim
        block_rt_dim = full_rt_dim if self.block_rt_dim is None else self.block_rt_dim

        lines: list[str] = [
            f"constexpr uint32_t FULL_RT_DIM = {full_rt_dim};",
            f"constexpr uint32_t FULL_CT_DIM = {full_ct_dim};",
            f"constexpr uint32_t BLOCK_CT_DIM = {block_ct_dim};",  # RT + TP
            f"constexpr uint32_t BLOCK_RT_DIM = {block_rt_dim};",  # RT + TP
        ]
        return "\n".join(lines)


@dataclass
class ADD_TOP_ROW(TemplateParameter):
    do_or_dont: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool ADD_TOP_ROW = {str(self.do_or_dont).lower()};"


# === RUNTIME PARAMETER IMPLEMENTATIONS ===


@dataclass
class LOOP_FACTOR(RuntimeParameter):
    factor: int = 1

    def covert_to_cpp(self) -> str:
        return f"constexpr int LOOP_FACTOR = {self.factor};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int LOOP_FACTOR;", "i"


@dataclass
class UNPACK_TRANS_FACES(RuntimeParameter):
    do_or_not: Transpose = Transpose.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_FACES = {str(self.do_or_not.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_FACES;", "?"


@dataclass
class UNPACK_TRANS_WITHING_FACE(RuntimeParameter):
    do_or_not: Transpose = Transpose.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {str(self.do_or_not.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_WITHIN_FACE;", "?"


@dataclass
class NARROW_TILE(RuntimeParameter):
    is_or_isnt: NarrowTile = NarrowTile.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool NARROW_TILE = {str(self.is_or_isnt.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool NARROW_TILE;", "?"


@dataclass
class DEST_INDEX(RuntimeParameter):
    index: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int DST_INDEX = {self.index};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int DST_INDEX;", "i"


@dataclass
class TILE_COUNT(RuntimeParameter):
    count: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int TILE_CNT = {self.count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int TILE_CNT;", "i"


@dataclass
class SRCA_REUSE_COUNT(RuntimeParameter):
    count: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int SRCA_REUSE_COUNT = {self.count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int SRCA_REUSE_COUNT;", "i"


@dataclass
class PARTIAL_FACE(RuntimeParameter):
    partial_a: bool = False
    partial_face_pack: bool = False
    partial_b: bool = False
    partial_face_math: bool = False

    def covert_to_cpp(self) -> str:
        lines: list[str] = []

        if self.partial_a:
            lines.append(
                f"constexpr bool PARTIAL_FACE_A = {str(self.partial_a).lower()};"
            )
            lines.append(
                f"constexpr bool PARTIAL_FACE_PACK = {str(self.partial_a).lower()};"
            )

        if self.partial_b:
            lines.append(
                f"constexpr bool PARTIAL_FACE_B = {str(self.partial_b).lower()};"
            )
            lines.append(
                f"constexpr bool PARTIAL_FACE_MATH = {str(self.partial_b).lower()};"
            )

        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "bool PARTIAL_FACE_A;",
            "bool PARTIAL_FACE_PACK;",
            "bool PARTIAL_FACE_B;",
            "bool PARTIAL_FACE_MATH;",
        ]
        return "\n".join(lines), "????"


@dataclass
class CRK_TILE_DIMM(RuntimeParameter):
    c_dimm: c_uint32 = 0
    r_dimm: c_uint32 = 0
    k_dimm: c_uint32 = 0

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr uint32_t RT_DIM = {self.r_dimm};",
            f"constexpr uint32_t CT_DIM = {self.c_dimm};",
            f"constexpr uint32_t KT_DIM = {self.k_dimm};",
        ]

        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "uint32_t CT_DIM;",
            "uint32_t RT_DIM;",
            "uint32_t KT_DIM;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_FACES(RuntimeParameter):
    num_faces: int = 0  # Number of active faces for result matrix
    num_faces_A: int = 0  # Number of active faces for matrix A
    num_faces_B: int = 0  # Number of active faces for matrix B

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int num_faces = {self.num_faces};",
            (
                f"constexpr int num_faces_A = {self.num_faces_A};"
                if self.num_faces_A
                else ""
            ),
            (
                f"constexpr int num_faces_B = {self.num_faces_B};"
                if self.num_faces_B
                else ""
            ),
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = ["int num_faces;", "int num_faces_A;", "int num_faces_B;"]
        return "\n".join(lines), "iii"


@dataclass
class TEST_FACE_DIMS(RuntimeParameter):
    face_r_dim: int = 16
    face_c_dim: int = 16

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int TEST_FACE_R_DIM = {self.face_r_dim};",
            f"constexpr int TEST_FACE_C_DIM = {self.face_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int TEST_FACE_R_DIM;",
            "int TEST_FACE_C_DIM;",
        ]
        return "\n".join(lines), "ii"


@dataclass
class IN_TILE_DIMS(RuntimeParameter):
    in0_r_dim: int = 32
    in0_c_dim: int = 32
    in1_r_dim: int = 32
    in1_c_dim: int = 32

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int in0_tile_r_dim = {self.in0_r_dim};",
            f"constexpr int in0_tile_c_dim = {self.in0_c_dim};",
            f"constexpr int in1_tile_r_dim = {self.in1_r_dim};",
            f"constexpr int in1_tile_c_dim = {self.in1_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int in0_tile_r_dim;",
            "int in0_tile_c_dim;",
            "int in1_tile_r_dim;",
            "int in1_tile_c_dim;",
        ]
        return "\n".join(lines), "iiii"


@dataclass
class RELU_CONFIG(RuntimeParameter):
    config: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int RELU_CONFIG = {self.config};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int RELU_CONFIG;", "i"


@dataclass
class NUM_ROWS_TO_PACK(RuntimeParameter):
    num_rows: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr uint32_t NUM_ROWS_TO_PACK = {self.num_rows};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "uint32_t NUM_ROWS_TO_PACK;", "I"
