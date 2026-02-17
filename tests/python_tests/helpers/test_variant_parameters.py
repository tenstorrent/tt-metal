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
    FastMode,
    ImpliedMathFormat,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    NarrowTile,
    PerfRunType,
    ReducePool,
    StableSort,
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
    throttle_level: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"


@dataclass
class MATH_TRANSPOSE_FACES(TemplateParameter):
    math_transpose_faces: Transpose

    def covert_to_cpp(self) -> str:
        return f"constexpr bool MATH_TRANSPOSE_FACES = {str(self.math_transpose_faces.value).lower()};"


@dataclass
class STOCHASTIC_ROUNDING(TemplateParameter):
    stochastic_rounding: StochasticRounding

    def covert_to_cpp(self) -> str:
        return f"constexpr auto STOCHASTIC_RND = ckernel::{self.stochastic_rounding.value};"


@dataclass
class DATA_COPY_TYPE(TemplateParameter):
    data_copy_type: DataCopyType

    def covert_to_cpp(self) -> str:
        return f"constexpr auto DATA_COPY_TYPE = ckernel::DataCopyType::{self.data_copy_type.value};"


@dataclass
class BROADCAST_TYPE(TemplateParameter):
    broadcast_type: BroadcastType

    def covert_to_cpp(self) -> str:
        return f"constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::{self.broadcast_type.value};"


@dataclass
class ACC_TO_DEST(TemplateParameter):
    acc_to_dest: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool ACC_TO_DEST = {str(self.acc_to_dest).lower()};"


@dataclass
class REUSE_DEST_TYPE(TemplateParameter):
    reuse_dest_type: EltwiseBinaryReuseDestType

    def covert_to_cpp(self) -> str:
        return f"constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::{self.reuse_dest_type.name};"


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
    disable_src_zero_flag: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool disable_src_zero_flag = {str(self.disable_src_zero_flag).lower()};"


@dataclass
class MATH_FIDELITY(TemplateParameter):
    math_fidelity: MathFidelity

    def covert_to_cpp(self) -> str:
        return f"constexpr ckernel::MathFidelity MATH_FIDELITY = {self.math_fidelity.cpp_enum_value};"


@dataclass
class APPROX_MODE(TemplateParameter):
    approx_mode: ApproximationMode = ApproximationMode.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool APPROX_MODE = {self.approx_mode.value};"


@dataclass
class ITERATIONS(TemplateParameter):
    iterations: int = 8

    def covert_to_cpp(self) -> str:
        return f"constexpr int ITERATIONS = {self.iterations};"


@dataclass
class FAST_MODE(TemplateParameter):
    fast_mode: FastMode = FastMode.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool FAST_MODE = {str(self.fast_mode.value).lower()};"


@dataclass
class CLAMP_NEGATIVE(TemplateParameter):
    clamp_negative: bool = True

    def covert_to_cpp(self) -> str:
        return f"constexpr bool CLAMP_NEGATIVE = {str(self.clamp_negative).lower()};"


@dataclass
class STABLE_SORT(TemplateParameter):
    stable_sort: StableSort = StableSort.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool STABLE_SORT = {str(self.stable_sort.value).lower()};"


@dataclass
class DEST_SYNC(TemplateParameter):
    dest_sync: DestSync = DestSync.Half

    def covert_to_cpp(self) -> str:
        return (
            f"constexpr auto dest_sync = ckernel::DstSync::Sync{self.dest_sync.name};"
        )


@dataclass
class TILIZE(TemplateParameter):
    tilize: Tilize = Tilize.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool tilize_en = {str(self.tilize.value).lower()};"


@dataclass
class IMPLIED_MATH_FORMAT(TemplateParameter):
    implied_math_format: ImpliedMathFormat = ImpliedMathFormat.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool IMPLIED_MATH_FORMAT = {self.implied_math_format.value};"


@dataclass
class UNPACKER_ENGINE_SEL(TemplateParameter):
    unpacker_engine_sel: UnpackerEngine = UnpackerEngine.UnpA

    def covert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t UNPACKER_ENGINE_SEL = p_unpacr::{self.unpacker_engine_sel.value};"


@dataclass
class PERF_RUN_TYPE(TemplateParameter):
    perf_run_type: PerfRunType

    def covert_to_cpp(self) -> str:
        return (
            f"\nconstexpr auto PERF_RUN_TYPE = PerfRunType::{self.perf_run_type.name};"
        )


@dataclass
class REDUCE_POOL_TYPE(TemplateParameter):
    reduce_pool_type: ReducePool

    def covert_to_cpp(self) -> str:
        return f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.reduce_pool_type.value};"


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
            f"constexpr std::uint32_t FULL_RT_DIM = {full_rt_dim};",
            f"constexpr std::uint32_t FULL_CT_DIM = {full_ct_dim};",
            f"constexpr std::uint32_t BLOCK_CT_DIM = {block_ct_dim};",  # RT + TP
            f"constexpr std::uint32_t BLOCK_RT_DIM = {block_rt_dim};",  # RT + TP
        ]
        return "\n".join(lines)


@dataclass
class ADD_TOP_ROW(TemplateParameter):
    add_top_row: bool

    def covert_to_cpp(self) -> str:
        return f"constexpr bool ADD_TOP_ROW = {str(self.add_top_row).lower()};"


# === RUNTIME PARAMETER IMPLEMENTATIONS ===


@dataclass
class LOOP_FACTOR(RuntimeParameter):
    loop_factor: int = 1

    def covert_to_cpp(self) -> str:
        return f"constexpr int LOOP_FACTOR = {self.loop_factor};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int LOOP_FACTOR;", "i"


@dataclass
class UNPACK_TRANS_FACES(RuntimeParameter):
    unpack_transpose_faces: Transpose = Transpose.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_FACES = {str(self.unpack_transpose_faces.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_FACES;", "?"


@dataclass
class UNPACK_TRANS_WITHIN_FACE(RuntimeParameter):
    unpack_transpose_within_face: Transpose = Transpose.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {str(self.unpack_transpose_within_face.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_WITHIN_FACE;", "?"


@dataclass
class NARROW_TILE(RuntimeParameter):
    narrow_tile: NarrowTile = NarrowTile.No

    def covert_to_cpp(self) -> str:
        return f"constexpr bool NARROW_TILE = {str(self.narrow_tile.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool NARROW_TILE;", "?"


@dataclass
class DEST_INDEX(RuntimeParameter):
    dst_index: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int DST_INDEX = {self.dst_index};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int DST_INDEX;", "i"


@dataclass
class L1_ACC(RuntimeParameter):
    l1_acc: L1Accumulation = L1Accumulation.No

    def covert_to_cpp(self) -> str:
        return (
            f"constexpr int L1_ACC = {1 if self.l1_acc == L1Accumulation.Yes else 0};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int L1_ACC;", "i"


@dataclass
class TILE_COUNT(RuntimeParameter):
    tile_cnt: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int TILE_CNT = {self.tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int TILE_CNT;", "i"


@dataclass
class INPUT_TILE_CNT(RuntimeParameter):
    tile_cnt: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int INPUT_TILE_CNT = {self.tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int INPUT_TILE_CNT;", "i"


@dataclass
class OUTPUT_TILE_CNT(RuntimeParameter):
    tile_cnt: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int OUTPUT_TILE_CNT = {self.tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int OUTPUT_TILE_CNT;", "i"


@dataclass
class REDUCE_TO_ONE(RuntimeParameter):
    is_reduce_to_one: bool = False

    def covert_to_cpp(self) -> str:
        return (
            f"constexpr bool IS_REDUCE_TO_ONE = {str(self.is_reduce_to_one).lower()};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool IS_REDUCE_TO_ONE;", "?"


@dataclass
class NUM_TILES_IN_BLOCK(RuntimeParameter):
    num_tiles_in_block: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int NUM_TILES_IN_BLOCK = {self.num_tiles_in_block};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int NUM_TILES_IN_BLOCK;", "i"


@dataclass
class SRCA_REUSE_COUNT(RuntimeParameter):
    srca_reuse_count: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int SRCA_REUSE_COUNT = {self.srca_reuse_count};"

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
            f"constexpr std::uint32_t RT_DIM = {self.r_dimm};",
            f"constexpr std::uint32_t CT_DIM = {self.c_dimm};",
            f"constexpr std::uint32_t KT_DIM = {self.k_dimm};",
        ]

        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t CT_DIM;",
            "std::uint32_t RT_DIM;",
            "std::uint32_t KT_DIM;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_TILES_IN_BLOCK(RuntimeParameter):
    num_tiles_in_block: int = 1

    def covert_to_cpp(self) -> str:
        return f"constexpr int NUM_TILES_IN_BLOCK = {self.num_tiles_in_block};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int NUM_TILES_IN_BLOCK;", "i"


@dataclass
class NUM_BLOCKS(RuntimeParameter):
    num_blocks: int = 1

    def covert_to_cpp(self) -> str:
        return f"constexpr int NUM_BLOCKS = {self.num_blocks};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int NUM_BLOCKS;", "i"


@dataclass
class NUM_FACES(RuntimeParameter):
    num_faces: int = 4  # Number of active faces for result matrix
    num_faces_A: int = 4  # Number of active faces for matrix A
    num_faces_B: int = 4  # Number of active faces for matrix B

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
class NUM_FACES_R_DIM(RuntimeParameter):
    num_faces_r_dim_A: int = 2  # Number of faces in row dimension for matrix A
    num_faces_r_dim_B: int = 2  # Number of faces in row dimension for matrix B

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            (
                f"constexpr int num_faces_r_dim_A = {self.num_faces_r_dim_A};"
                if self.num_faces_r_dim_A
                else ""
            ),
            (
                f"constexpr int num_faces_r_dim_B = {self.num_faces_r_dim_B};"
                if self.num_faces_r_dim_B
                else ""
            ),
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int num_faces_r_dim_A;",
            "int num_faces_r_dim_B;",
        ]
        return "\n".join(lines), "ii"


@dataclass
class NUM_FACES_C_DIM(RuntimeParameter):
    num_faces_c_dim_A: int = 2  # Number of faces in column dimension for matrix A
    num_faces_c_dim_B: int = 2  # Number of faces in column dimension for matrix B

    def covert_to_cpp(self) -> str:
        lines: list[str] = [
            (
                f"constexpr int num_faces_c_dim_A = {self.num_faces_c_dim_A};"
                if self.num_faces_c_dim_A
                else ""
            ),
            (
                f"constexpr int num_faces_c_dim_B = {self.num_faces_c_dim_B};"
                if self.num_faces_c_dim_B
                else ""
            ),
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int num_faces_c_dim_A;",
            "int num_faces_c_dim_B;",
        ]
        return "\n".join(lines), "ii"


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
    relu_config: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr int RELU_CONFIG = {self.relu_config};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int RELU_CONFIG;", "i"


@dataclass
class NUM_ROWS_TO_PACK(RuntimeParameter):
    num_rows_to_pack: int = 0

    def covert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_ROWS_TO_PACK = {self.num_rows_to_pack};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t NUM_ROWS_TO_PACK;", "I"
