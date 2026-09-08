# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import struct
from abc import ABC, abstractmethod
from ctypes import c_uint32
from dataclasses import dataclass

from .format_config import DataFormat
from .golden_generators import TILE_DIMENSIONS
from .llk_params import (
    FPU_BINARY_OPERATIONS,
    REDUCE_OPERATIONS,
    SFPU_BINARY_OPERATIONS,
    SFPU_UNARY_OPERATIONS,
    ApproximationMode,
    BroadcastType,
    DataCopyType,
    DestSync,
    DstRoundingMode,
    EltwiseBinaryReuseDestType,
    FastMode,
    ImpliedMathFormat,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    NarrowTile,
    PerfRunType,
    ReducePool,
    SdpaFwOp,
    SdpaOp,
    StableSort,
    StochasticRounding,
    Tilize,
    TopKSortDirection,
    TopKXLChunkBaseMode,
    TopKXLIndexOp,
    TopKXLSortMode,
    Transpose,
    UnpackerEngine,
    VectorMode,
)
from .matmul_sweep import validate_tile_dimensions

# Base parameter classes


@dataclass
class TemplateParameter(ABC):
    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass


@dataclass
class RuntimeParameter(ABC):

    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass

    @abstractmethod
    def convert_to_struct_fields(self) -> tuple[str, str]:
        pass


# === TEMPLATE PARAMETER IMPLEMENTATIONS ===


@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    throttle_level: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"


@dataclass
class MATH_TRANSPOSE_FACES(TemplateParameter):
    math_transpose_faces: Transpose

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MATH_TRANSPOSE_FACES = {str(self.math_transpose_faces.value).lower()};"


@dataclass
class STOCHASTIC_ROUNDING(TemplateParameter):
    stochastic_rounding: StochasticRounding

    def convert_to_cpp(self) -> str:
        return f"constexpr auto STOCHASTIC_RND = ckernel::{self.stochastic_rounding.value};"


@dataclass
class DATA_COPY_TYPE(TemplateParameter):
    data_copy_type: DataCopyType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto DATA_COPY_TYPE = ckernel::DataCopyType::{self.data_copy_type.value};"


@dataclass
class BROADCAST_TYPE(TemplateParameter):
    broadcast_type: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::{self.broadcast_type.value};"


@dataclass
class ACC_TO_DEST(TemplateParameter):
    acc_to_dest: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ACC_TO_DEST = {str(self.acc_to_dest).lower()};"


@dataclass
class REUSE_DEST_TYPE(TemplateParameter):
    reuse_dest_type: EltwiseBinaryReuseDestType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::{self.reuse_dest_type.name};"


@dataclass
class EN_DEST_REUSE(TemplateParameter):
    def convert_to_cpp(self) -> str:
        return "#define EN_DEST_REUSE"


@dataclass
class SFPU_INT_OP(TemplateParameter):
    """Emit a #define to select the integer SFPU operation in a shared C++ test source.

    Supported values: "MUL", "GT", "LT", "LE", "GE".  When omitted the C++ source
    falls through to its default (add_int) path.
    """

    int_op: str = ""

    def convert_to_cpp(self) -> str:
        if self.int_op:
            return f"#define SFPU_INT_OP_{self.int_op.upper()}"
        return ""


@dataclass
class SFPU_BINARY_OP(TemplateParameter):
    """Select the consolidated Quasar binary-SFPU op at compile time.

    Emits ``constexpr ckernel::BinaryOp SFPU_BINARY_OP = ckernel::BinaryOp::<op>;``,
    consumed by ``sfpu_operations_quasar.h``. ``op`` is one of:
    ADD, MUL, DIV, GT, LT, LE, GE, MAX, MIN (reusing the LLK BinaryOp enum, like
    Blackhole — int vs float MUL is disambiguated by the math format in the cpp).
    """

    op: str = "ADD"

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::BinaryOp SFPU_BINARY_OP = ckernel::BinaryOp::{self.op};"


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

    def convert_to_cpp(self) -> str:
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
class SFPU_TERNARY_OP(TemplateParameter):
    """Select the ternary SFPU op at compile time.

    Emits ``constexpr auto SFPU_TERNARY_OPERATION = SfpuType::<op>;`` consumed by
    ``sfpu_operations.h``. ``ternary_mathop.cpp_enum_value`` must match the
    ``SfpuType`` enumerator name (e.g. ``addcmul``/``addcdiv``).
    """

    ternary_mathop: MathOperation = None

    def convert_to_cpp(self) -> str:
        return f"constexpr auto SFPU_TERNARY_OPERATION = SfpuType::{self.ternary_mathop.cpp_enum_value};"


@dataclass
class SFPU_TERNARY_SCALAR(TemplateParameter):
    """Scalar multiplier for addcmul/addcdiv, passed as a raw fp32 bit pattern.

    The ternary addc kernels take a ``std::uint32_t value`` reinterpreted as float in
    the SFPU. Emit the bit pattern so the C++ and torch golden agree exactly.
    """

    ternary_scalar_bits: int = 0x40000000  # 2.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_TERNARY_SCALAR = {self.ternary_scalar_bits}u;"


@dataclass
class SFPU_BINOP_MODE(TemplateParameter):
    """Select the float unary-with-scalar binop at compile time.

    Emits ``constexpr int SFPU_BINOP_MODE = <n>;`` consumed by
    ``sfpu_binop_scalar_{test,perf}.cpp``, matching the BINOP_MODE enum in
    ``ckernel_sfpu_binop_with_unary.h`` (ADD=0, SUB=1, MUL=2, DIV=3, RSUB=4).
    """

    # Maps MathOperation.cpp_enum_value -> the kernel's BINOP_MODE integer.
    _MODE = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3, "RSUB": 4}

    binop_mathop: MathOperation = None

    def convert_to_cpp(self) -> str:
        return f"constexpr int SFPU_BINOP_MODE = {self._MODE[self.binop_mathop.cpp_enum_value]};"


@dataclass
class SFPU_UNARY_SCALAR(TemplateParameter):
    """Scalar operand for the float unary-with-scalar binops, as raw fp32 bits.

    ``calculate_binop_with_scalar`` decodes it via ``Converter::as_float``; emit
    the bit pattern so the C++ and torch golden agree exactly. For DIV this is
    the host-inverted divisor (1/divisor), since the kernel multiplies.
    """

    value_bits: int = 0x40000000  # 2.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_UNARY_SCALAR = {self.value_bits}u;"


@dataclass
class SFPU_SHIFT_AMOUNT(TemplateParameter):
    """Shift amount for the *unary* shift ops (LeftShift / RightShift).

    Emitted as a macro rather than a constexpr because sfpu_operations.h selects on
    ``#ifdef SFPU_SHIFT_AMOUNT``: the header is shared by every unary test, and only the shift
    sweep sets this, so the others have to keep compiling without it. The binary shift ops take
    their amount as a second operand and need none of this.
    """

    shift_amount: int = 3

    def convert_to_cpp(self) -> str:
        return f"#define SFPU_SHIFT_AMOUNT {self.shift_amount}u"


@dataclass
class DISABLE_SRC_ZERO_FLAG(TemplateParameter):
    disable_src_zero_flag: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool disable_src_zero_flag = {str(self.disable_src_zero_flag).lower()};"


@dataclass
class MATH_FIDELITY(TemplateParameter):
    math_fidelity: MathFidelity

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::MathFidelity MATH_FIDELITY = {self.math_fidelity.cpp_enum_value};"


@dataclass
class APPROX_MODE(TemplateParameter):
    approx_mode: ApproximationMode = ApproximationMode.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool APPROX_MODE = {self.approx_mode.cpp_enum_value};"


@dataclass
class SFPU_FAST_APPROX(TemplateParameter):
    """The sqrt/rsqrt family's ``FAST_APPROX`` template argument.

    Emits ``constexpr bool SFPU_FAST_APPROX = <bool>;``. Distinct from
    :class:`APPROX_MODE`: ``APPROX_MODE`` selects which approximation *body* runs
    (SQRT_10-bits vs SQRT_23-bits), while this flag only drops the trailing
    ``v_if(x < 0) -> NaN`` guard in ``_calculate_sqrt_body_``. It is therefore
    unobservable unless the stimuli reach a negative argument.

    Surfaced as ``fast_and_approx`` on the compute API's ``add_rsqrt_tile``.
    """

    fast_approx: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_FAST_APPROX = {str(self.fast_approx).lower()};"


@dataclass
class CUSTOM_MM_UNINIT(TemplateParameter):
    """The custom_mm_block_uninit / compressed_custom_mm_block_uninit configuration.

    Emits the three compile-time switches ``custom_mm_uninit_restore_test.cpp`` reads:

    ``UNINIT_DENSE_PACKING``   the ``dense_packing`` template argument, applied to both
                               the run-0 init (W-stride -> 32 rows) and the uninit
                               (W-stride -> 64 rows). The two must agree: a block packed
                               dense and torn down non-dense is not a supported call.
    ``UNINIT_RESTORE_MOP``     the ``restore_tile_pack_mop`` template argument, i.e.
                               whether the uninit reinstalls the Default tile-pack MOP.
    ``UNINIT_SKIP``            negative control -- drop the uninit entirely. Not a
                               supported configuration; it exists to prove the restores
                               are load-bearing rather than incidentally redundant.
    ``BLOCK_MOP_NUM_FACES``    the tile geometry the run-0 block-contiguous MOP is
                               programmed with. The pack MOP bakes in tile geometry, so
                               this decides whether the MOP restore is observable at all:
                               at 4 (same geometry as the run-1 pack) restoring and not
                               restoring are indistinguishable, while at 2 (a 16x32 tiny
                               tile) the un-restored MOP packs the wrong face count --
                               the hazard the uninit's comment describes.
    """

    dense_packing: bool = False
    restore_mop: bool = False
    skip: bool = False
    block_mop_num_faces: int = 2

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool UNINIT_DENSE_PACKING = {str(self.dense_packing).lower()};",
                f"constexpr bool UNINIT_RESTORE_MOP = {str(self.restore_mop).lower()};",
                f"constexpr bool UNINIT_SKIP = {str(self.skip).lower()};",
                f"constexpr std::uint32_t BLOCK_MOP_NUM_FACES = {self.block_mop_num_faces}u;",
            ]
        )


@dataclass
class SAMPLING_PRGM0_HAZARD(TemplateParameter):
    """Cross-op vConstFloatPrgm0 hazard switches for ``sfpu_sampling_test.cpp``.

    Emits ``#define SAMPLING_POLLUTE_PRGM0`` and/or ``#define SAMPLING_SKIP_RECIP_INIT``.

    ``pollute``  run ``log_init`` first, standing in for an earlier op in the same kernel
                 that owns vConstFloatPrgm0 (log sets it to ~8.3e-8; the non-legacy
                 reciprocal needs 2.0f).
    ``skip_init`` drop ``sampling_recip_init``. Not a supported call -- it exists so the
                 test can show the init is load-bearing rather than merely present.
    """

    pollute: bool = False
    skip_init: bool = False

    def convert_to_cpp(self) -> str:
        lines = []
        if self.pollute:
            lines.append("#define SAMPLING_POLLUTE_PRGM0")
        if self.skip_init:
            lines.append("#define SAMPLING_SKIP_RECIP_INIT")
        return "\n".join(lines)


@dataclass
class DST_WRITE_ADDR_OFFSET(TemplateParameter):
    """Compile-time knobs for ``set_dst_write_addr_offset_test.cpp``.

    ``OFFSET_ENABLED``   whether the SFPU body calls
                         ``ckernel::sfpu::set_dst_write_addr_offset`` at all. False is the
                         control the helper is measured against: with it absent,
                         ``SFPU_DST_INDEX`` alone decides where the negate lands, so
                         ``helper(N * 64) at dst_index 0`` and ``no helper at dst_index N``
                         must agree bit for bit.
    ``OFFSET_ROWS``      the argument, in Dst ROWS. One 32x32 tile is 64 rows
                         (``DstTileSizeLog2[Tile32x32] == 6``), which is what makes the two
                         real call patterns land where they do: 64 is a whole-tile rebase
                         (deepseek_top32_rm's ``tile_offset``) and 2 is the column-group
                         flip (topk_xl's ``odd_col_offset``).
    ``SFPU_DST_INDEX``   the ``dst_index`` handed to
                         ``_llk_math_eltwise_unary_sfpu_params_``, i.e. where
                         ``_llk_math_eltwise_sfpu_start_`` points the write pointer before
                         the body runs -- and therefore what the helper overwrites.
    ``SFPU_ENABLED``     drop the SFPU op entirely, giving the datacopy-only baseline the
                         negated variants are diffed against.
    """

    offset_enabled: bool = True
    offset_rows: int = 0
    sfpu_dst_index: int = 0
    sfpu_enabled: bool = True

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool OFFSET_ENABLED = {str(self.offset_enabled).lower()};",
                f"constexpr std::uint32_t OFFSET_ROWS = {self.offset_rows}u;",
                f"constexpr std::uint32_t SFPU_DST_INDEX = {self.sfpu_dst_index}u;",
                f"constexpr bool SFPU_ENABLED = {str(self.sfpu_enabled).lower()};",
            ]
        )


@dataclass
class PACK_NUM_TILES(TemplateParameter):
    """Tile count for the block/per-tile pack drivers.

    Emits ``constexpr std::uint32_t PACK_NUM_TILES = <n>;``. Distinct from the runtime
    ``TILE_COUNT``: the pack loops here are compile-time bounded so the block MOP's
    outer-loop patching and the per-tile loop stay in step.
    """

    num_tiles: int = 4

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t PACK_NUM_TILES = {self.num_tiles}u;"


@dataclass
class RMSNORM_DEST_REUSE(TemplateParameter):
    """Compile-time knobs for ``rmsnorm_bcast_scalar_dest_reuse_test.cpp``.

    All four are template arguments (or a template-fixed runtime argument) on the LLK
    pair, so none of them can be a runtime parameter:

    ``rmsnorm_num_tiles``
        Outer-loop count of the math MOP *and* the unpack MOP -- one
        ``_llk_unpack_A_`` call walks this many tiles. Bounded by DEST half-sync capacity.
    ``rmsnorm_num_faces``
        Runtime argument to both ``_init_``s, but it sizes the MOP loops, so a variant
        must be built per value. Only 1, 2 and 4 are accepted (``LLK_ASSERT``).
    ``clear_dest``
        Template argument on the math execute; gates the ``ZEROACC`` between the MOVD2B
        and the MOP.
    ``unpack_full_transpose``
        Drives both ``transpose_of_faces`` and ``within_face_16x16_transpose`` on the
        unpack init. This axis exists only because blaze's version of the header won the
        reconciliation, so it is new reachable surface. Its replay-buffer path is
        restricted to one tile and four faces by ``LLK_ASSERT``.

    The two count fields carry the ``rmsnorm_`` prefix so they match the constants they
    emit and stay globally unique: ``test_perf_header_gate.py`` requires that no two
    parameter classes declare the same field name, because a test passing both would
    produce two perf-CSV columns with the same header. Bare ``num_tiles`` belongs to
    ``PACK_NUM_TILES`` and bare ``num_faces`` to ``NUM_FACES``.
    """

    rmsnorm_num_tiles: int = 1
    rmsnorm_num_faces: int = 4
    clear_dest: bool = False
    unpack_full_transpose: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t RMSNORM_NUM_TILES = {self.rmsnorm_num_tiles}u;",
                f"constexpr std::uint32_t RMSNORM_NUM_FACES = {self.rmsnorm_num_faces}u;",
                f"constexpr bool RMSNORM_CLEAR_DEST = {str(self.clear_dest).lower()};",
                "constexpr bool RMSNORM_UNPACK_FULL_TRANSPOSE = "
                f"{str(self.unpack_full_transpose).lower()};",
            ]
        )


@dataclass
class REDUCE_BLOCK_CT_DIM(TemplateParameter):
    """Compile-time block width (in tiles) for the block-based reduce_block_max_row LLKs.

    A standalone one-line constant, deliberately *not* routed through the matmul-centric
    ``INPUT_DIMENSIONS`` bundle: this pure block-reduce test has no use for that bundle's
    other fields (``FULL_RT_DIM`` / ``FULL_CT_DIM`` / ``BLOCK_RT_DIM``) or its
    ``generate_input_dim`` tile-shape validation, and it needs only a plain compile-time
    block width. The distinct name (``REDUCE_BLOCK_CT_DIM``, not ``BLOCK_CT_DIM``) also
    preempts a redefinition clash if ``INPUT_DIMENSIONS`` — the sole emitter of
    ``BLOCK_CT_DIM`` — is ever added to this test: the header generator concatenates every
    param's ``convert_to_cpp()`` with no de-dup, so two same-named ``constexpr`` lines would
    fail to compile. (This test does not currently emit ``BLOCK_CT_DIM``.)
    """

    reduce_block_ct_dim: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t REDUCE_BLOCK_CT_DIM = {self.reduce_block_ct_dim};"
        )


@dataclass
class USE_RUNTIME(TemplateParameter):
    """Selects the runtime (dynamic block_ct_dim) reduce_block_max_row LLK family."""

    use_runtime: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool USE_RUNTIME = {str(self.use_runtime).lower()};"


@dataclass
class REINIT_MODE(TemplateParameter):
    """Selects the reduce_block_max_row re-arm path: 0=none, 1=short (MOP + addrmods),
    2=minimal (ADDR_MOD_1/2/6), 3=addrmod-only reinit (ADDR_MOD_1/2/3/6)."""

    reinit_mode: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int REINIT_MODE = {self.reinit_mode};"


@dataclass
class CLOBBER_OP(TemplateParameter):
    """Op run between reduce init and reinit to overwrite the reduce MOP/addrmods
    (reconfig-escape guard for the reinit paths):
    0=none, 1=eltwise binary (all addrmods + MOP), 2=minimal_safe (ADDR_MOD_1/2/6 only),
    3=addrmod_all (ADDR_MOD_1/2/3/6).
    """

    clobber_op: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int CLOBBER_OP = {self.clobber_op};"


@dataclass
class RESPECT_TRIGGER(TemplateParameter):
    """Enable the reduce_block_max_row producer/consumer trigger handshake.

    When true, the unpack splits the block reduce into two half-width MOP runs
    separated by a HW semaphore wait (FPU_SFPU), so the reduce can start on the
    first half of the block before the second half is signalled. The test's PACK
    thread plays the producer (posts the tokens). Requires an even block_ct_dim
    (the unpack MOP outerloop is block_ct_dim / 2)."""

    respect_trigger: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool RESPECT_TRIGGER = {str(self.respect_trigger).lower()};"


@dataclass
class OVERLAP_FIRST_HALF(TemplateParameter):
    """Overlap the first-half reduce with the second-half pack (runtime family only).

    When true (and RESPECT_TRIGGER + USE_RUNTIME), the unpack's first half gates on
    the early UNPACK_MATH_DONE token instead of FPU_SFPU, so run()#1 overlaps the
    second-half pack. Ignored by the compile-time unpack family (no overlap path)."""

    overlap_first_half: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool OVERLAP_FIRST_HALF = {str(self.overlap_first_half).lower()};"


@dataclass
class ITERATIONS(TemplateParameter):
    iterations: int = 8

    def convert_to_cpp(self) -> str:
        return f"constexpr int ITERATIONS = {self.iterations};"


@dataclass
class FAST_MODE(TemplateParameter):
    fast_mode: FastMode = FastMode.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool FAST_MODE = {str(self.fast_mode.value).lower()};"


@dataclass
class CLAMP_NEGATIVE(TemplateParameter):
    clamp_negative: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool CLAMP_NEGATIVE = {str(self.clamp_negative).lower()};"


@dataclass
class STABLE_SORT(TemplateParameter):
    stable_sort: StableSort = StableSort.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool STABLE_SORT = {str(self.stable_sort.value).lower()};"


@dataclass
class DEST_SYNC(TemplateParameter):
    dest_sync: DestSync = DestSync.Half

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr auto dest_sync = ckernel::DstSync::Sync{self.dest_sync.name};"
        )


@dataclass
class TILIZE(TemplateParameter):
    tilize: Tilize = Tilize.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool tilize_en = {str(self.tilize.value).lower()};"


@dataclass
class IMPLIED_MATH_FORMAT(TemplateParameter):
    implied_math_format: ImpliedMathFormat = ImpliedMathFormat.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool IMPLIED_MATH_FORMAT = {self.implied_math_format.value};"


@dataclass
class ENABLE_2X_FORMAT(TemplateParameter):
    enable_2x_format: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool ENABLE_2X_FORMAT = {str(self.enable_2x_format).lower()};"
        )


@dataclass
class ENABLE_DIRECT_INDEXING(TemplateParameter):
    enable_direct_indexing: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ENABLE_DIRECT_INDEXING = {str(self.enable_direct_indexing).lower()};"


@dataclass
class UNPACKER_ENGINE_SEL(TemplateParameter):
    unpacker_engine_sel: UnpackerEngine = UnpackerEngine.UnpA

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t UNPACKER_ENGINE_SEL = p_unpacr::{self.unpacker_engine_sel.value};"


@dataclass
class VECTOR_MODE(TemplateParameter):
    vector_mode: VectorMode = VectorMode.RC

    def convert_to_cpp(self) -> str:
        return f"constexpr auto VECTOR_MODE = {self.vector_mode.cpp_enum_value};"


@dataclass
class PERF_RUN_TYPE(TemplateParameter):
    perf_run_type: PerfRunType

    def convert_to_cpp(self) -> str:
        return (
            f"\nconstexpr auto PERF_RUN_TYPE = PerfRunType::{self.perf_run_type.name};"
        )


@dataclass
class REDUCE_POOL_TYPE(TemplateParameter):
    reduce_pool_type: ReducePool

    def convert_to_cpp(self) -> str:
        return f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.reduce_pool_type.value};"


@dataclass
class SDPA_OP(TemplateParameter):
    sdpa_op: SdpaOp = SdpaOp.RecipLegacy

    def convert_to_cpp(self) -> str:
        return f"constexpr int SDPA_OP = {self.sdpa_op.value};"


@dataclass
class SDPA_EXP_SCALE(TemplateParameter):
    scale_bf16: int = 0x3F80  # bf16(1.0)

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint16_t EXP_SCALE_BF16 = {self.scale_bf16}u;"


@dataclass
class SDPA_SOFTPLUS_PARAMS(TemplateParameter):
    softplus_beta_bits: int = 0x3F800000  # 1.0f
    softplus_beta_reciprocal_bits: int = 0x3F800000  # 1.0f
    softplus_threshold_bits: int = 0x41A00000  # 20.0f

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t SOFTPLUS_BETA_BITS = {self.softplus_beta_bits}u;",
            f"constexpr std::uint32_t SOFTPLUS_BETA_RECIPROCAL_BITS = {self.softplus_beta_reciprocal_bits}u;",
            f"constexpr std::uint32_t SOFTPLUS_THRESHOLD_BITS = {self.softplus_threshold_bits}u;",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_FW_OP(TemplateParameter):
    sdpa_fw_op: SdpaFwOp = SdpaFwOp.Recip

    def convert_to_cpp(self) -> str:
        return f"constexpr int SDPA_FW_OP = {self.sdpa_fw_op.value};"


@dataclass
class TOPK(TemplateParameter):
    topk_k: int = 0
    topk_matrix_width: int = 0
    topk_sort_direction: TopKSortDirection = TopKSortDirection.Descending
    topk_stable_sort: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TOPK_K = {self.topk_k};",
            f"constexpr std::uint32_t TOPK_LOGK = {int(math.log2(self.topk_k))};",
            f"constexpr std::uint32_t TOPK_NUM_ITERATIONS = {int(math.log2(self.topk_matrix_width // TILE_DIMENSIONS[1] // 2))};",
            f"constexpr std::uint32_t TOPK_SORT_DIRECTION = {self.topk_sort_direction.value};",
            f"constexpr bool TOPK_STABLE_SORT = {str(self.topk_stable_sort).lower()};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t TOPK_K;",
            "std::uint32_t TOPK_LOGK;",
            "std::uint32_t TOPK_NUM_ITERATIONS;",
            "std::uint32_t TOPK_SORT_DIRECTION;",
            "bool TOPK_STABLE_SORT;",
        ]
        return "\n".join(lines), "IIII?"


@dataclass
class GENERALIZED_MOE_GATE(TemplateParameter):
    """Compile-time configuration for the generalized_moe_gate test.

    ``read_base``/``from_*``/``to_*`` are SFPU column offsets; a run occupies the pair ``{lo, hi}``.
    ``row_src``/``row_dst``/``srcb`` name copy4rows' 4-row DEST blocks and its SrcB scratch window.
    ``eps`` and ``scale`` are float bit patterns, as the LLK takes them.
    ``sections`` is how many DEST sections the kernel runs; 2 puts the second in the upper
    half under DstSync::Half and packs it to buffer_Res[4..7].
    ``sigmoid`` selects the op's enable_sigmoid front-end: transpose, sigmoid, then a RELOAD
    binary reading SrcA back out of DEST.
    """

    mode: int = 0
    sub_op: int = 0
    grouped: bool = False
    topk: int = 8
    softmax: bool = False
    produce_run: bool = False
    reload: bool = False
    eps: int = 0
    scale: int = 0
    read_base: int = 0
    from_lo: int = 0
    from_hi: int = 2
    to_lo: int = 0
    to_hi: int = 2
    field: int = 0
    idx_offset: int = 0
    row_src: int = 0
    row_dst: int = 4
    srcb: int = 16
    second_copy: bool = False
    pre_copy4rows: bool = False
    row_src_2: int = 0
    row_dst_2: int = 8
    srcb_2: int = 20
    d2b_dst: int = 0
    b2d_base: int = 0
    sections: int = 1
    sigmoid: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int GMG_MODE = {self.mode};",
            f"constexpr int GMG_SUB_OP = {self.sub_op};",
            f"constexpr bool GMG_GROUPED = {str(self.grouped).lower()};",
            f"constexpr std::uint32_t GMG_TOPK = {self.topk};",
            f"constexpr bool GMG_SOFTMAX = {str(self.softmax).lower()};",
            f"constexpr bool GMG_PRODUCE_RUN = {str(self.produce_run).lower()};",
            f"constexpr bool GMG_RELOAD = {str(self.reload).lower()};",
            f"constexpr std::uint32_t GMG_EPS = {self.eps};",
            f"constexpr std::uint32_t GMG_SCALE = {self.scale};",
            f"constexpr std::uint32_t GMG_READ_BASE = {self.read_base};",
            f"constexpr std::uint32_t GMG_FROM_LO = {self.from_lo};",
            f"constexpr std::uint32_t GMG_FROM_HI = {self.from_hi};",
            f"constexpr std::uint32_t GMG_TO_LO = {self.to_lo};",
            f"constexpr std::uint32_t GMG_TO_HI = {self.to_hi};",
            f"constexpr std::uint32_t GMG_FIELD = {self.field};",
            f"constexpr std::uint32_t GMG_IDX_OFFSET = {self.idx_offset};",
            f"constexpr std::uint32_t GMG_ROW_SRC = {self.row_src};",
            f"constexpr std::uint32_t GMG_ROW_DST = {self.row_dst};",
            f"constexpr std::uint32_t GMG_SRCB = {self.srcb};",
            f"constexpr bool GMG_SECOND_COPY = {str(self.second_copy).lower()};",
            f"constexpr bool GMG_PRE_COPY4ROWS = {str(self.pre_copy4rows).lower()};",
            f"constexpr std::uint32_t GMG_ROW_SRC_2 = {self.row_src_2};",
            f"constexpr std::uint32_t GMG_ROW_DST_2 = {self.row_dst_2};",
            f"constexpr std::uint32_t GMG_SRCB_2 = {self.srcb_2};",
            f"constexpr std::uint32_t GMG_D2B_DST = {self.d2b_dst};",
            f"constexpr std::uint32_t GMG_B2D_BASE = {self.b2d_base};",
            f"constexpr std::uint32_t GMG_SECTIONS = {self.sections};",
            f"constexpr bool GMG_SIGMOID = {str(self.sigmoid).lower()};",
        ]
        return "\n".join(lines)


@dataclass
class DEEPSEEK_MOE_GATE(TemplateParameter):
    """Compile-time configuration for the deepseek_moe_gate test."""

    dmg_mode: int = 0
    dmg_sub_op: int = 0
    dmg_sigmoid: bool = False
    dmg_reload: bool = False
    dmg_eps: int = 0
    dmg_scale: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int DMG_MODE = {self.dmg_mode};",
            f"constexpr int DMG_SUB_OP = {self.dmg_sub_op};",
            f"constexpr bool DMG_SIGMOID = {str(self.dmg_sigmoid).lower()};",
            f"constexpr bool DMG_RELOAD = {str(self.dmg_reload).lower()};",
            f"constexpr std::uint32_t DMG_EPS = {self.dmg_eps};",
            f"constexpr std::uint32_t DMG_SCALE = {self.dmg_scale};",
        ]
        return "\n".join(lines)


@dataclass
class HADAMARD(TemplateParameter):
    """Compile-time configuration for the H128 Hadamard test."""

    hadamard_normalize: bool = True
    h16_tile_index: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr bool HADAMARD_NORMALIZE = {str(self.hadamard_normalize).lower()};",
            f"constexpr std::uint32_t HADAMARD_H16_TILE_INDEX = {self.h16_tile_index};",
        ]
        return "\n".join(lines)


@dataclass
class ROPE(TemplateParameter):
    ht: int = 1
    wt: int = 1
    x_base: int = 0
    x_stride: int = 64
    cos_base: int = 64
    sin_base: int = 128
    cs_stride: int = 64
    has_scale: bool = False
    scale_fp32: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t ROPE_HT = {self.ht};",
            f"constexpr std::uint32_t ROPE_WT = {self.wt};",
            f"constexpr std::uint32_t ROPE_X_BASE = {self.x_base};",
            f"constexpr std::uint32_t ROPE_X_STRIDE = {self.x_stride};",
            f"constexpr std::uint32_t ROPE_COS_BASE = {self.cos_base};",
            f"constexpr std::uint32_t ROPE_SIN_BASE = {self.sin_base};",
            f"constexpr std::uint32_t ROPE_CS_STRIDE = {self.cs_stride};",
            f"constexpr bool ROPE_HAS_SCALE = {str(self.has_scale).lower()};",
            f"constexpr std::uint32_t ROPE_SCALE_FP32 = {hex(self.scale_fp32)};",
        ]
        return "\n".join(lines)


@dataclass
class TOPK_XL(TemplateParameter):
    k: int = 512
    num_chunks: int = 1
    tail_elements: int = 512
    num_rows: int = 1
    index_op: TopKXLIndexOp = TopKXLIndexOp.RowMajor
    group_id: int = 0
    group_shift: int = 16
    core_id: int = 0
    sort_direction: TopKSortDirection = TopKSortDirection.Descending
    fused_reduce: bool = False
    chunk_base_mode: TopKXLChunkBaseMode = TopKXLChunkBaseMode.Static
    chunk_base: int = 0
    fused_e2e: bool = False
    seg_base: int = 0
    sort_mode: TopKXLSortMode = TopKXLSortMode.Dispatch
    lsb_row_major: bool = False
    reinit_after_copy: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TOPK_XL_K = {self.k};",
            f"constexpr std::uint32_t TOPK_XL_NUM_CHUNKS = {self.num_chunks};",
            f"constexpr std::uint32_t TOPK_XL_TAIL_ELEMENTS = {self.tail_elements};",
            f"constexpr std::uint32_t TOPK_XL_NUM_ROWS = {self.num_rows};",
            f"constexpr std::uint32_t TOPK_XL_INDEX_OP = {self.index_op.value};",
            f"constexpr std::uint32_t TOPK_XL_GROUP_ID = {self.group_id};",
            f"constexpr std::uint32_t TOPK_XL_GROUP_SHIFT = {self.group_shift};",
            f"constexpr std::uint32_t TOPK_XL_CORE_ID = {self.core_id};",
            f"constexpr bool TOPK_XL_ASCENDING = {str(self.sort_direction == TopKSortDirection.Ascending).lower()};",
            f"constexpr bool TOPK_XL_FUSED_REDUCE = {str(self.fused_reduce).lower()};",
            f"constexpr std::uint32_t TOPK_XL_CHUNK_BASE_MODE = {self.chunk_base_mode.value};",
            f"constexpr std::uint32_t TOPK_XL_CHUNK_BASE = {self.chunk_base};",
            f"constexpr bool TOPK_XL_FUSED_E2E = {str(self.fused_e2e).lower()};",
            f"constexpr std::uint32_t TOPK_XL_SEG_BASE = {self.seg_base};",
            f"constexpr std::uint32_t TOPK_XL_SORT_MODE = {self.sort_mode.value};",
            f"constexpr bool TOPK_XL_LSB_ROW_MAJOR = {str(self.lsb_row_major).lower()};",
            f"constexpr bool TOPK_XL_REINIT_AFTER_COPY = {str(self.reinit_after_copy).lower()};",
        ]
        return "\n".join(lines)


@dataclass
class TOP32_RM(TemplateParameter):
    """Compile-time knobs for ``top32_rm_test.cpp`` (the DeepSeek top32_rm family).

    ``row_elements``  length of the row being reduced to its top 32. The driver walks it in
                      64-element chunks, so a value that is 32 (mod 64) is what reaches the
                      ``num_faces=2`` tail path -- the same shape the Metal dev test covers
                      with row=160.
    ``datum_bytes``   L1 datum size of both operands, which is the only thing the driver
                      needs to turn a chunk index into an address: one chunk is
                      ``64 * datum_bytes`` bytes, and unpacker addresses count 16-byte words.
                      Must match the format the stimuli are written in.
    ``top_min``       ``_bitonic_top32_merge_``'s template polarity. False (the consumer's
                      value) keeps the max half of each compare-exchange, i.e. a top-32.
    ``top32_mode``    0 walks the row in 64-element chunks through this family's own unpack
                      (``top32_rm_dev_compute.cpp``); 1 is the pre-sorted path, which
                      transposes whole 1024-element tiles into Dest and runs
                      ``_bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_``
                      (``top32_rm_dev_compute_v2.cpp``), then finishes any remainder in
                      64-element chunks the way that kernel does. Mode 1 requires the input to
                      be pre-sorted into descending runs of 32.
    ``via_wrappers``  route every SFPU call through the Metal wrapper layer
                      (``experimental/llk_sfpu/llk_math_deepseek_top32_rm.h``) instead of
                      calling the ``ckernel::sfpu::`` primitives directly. The wrappers are
                      thin -- each is the same ``_llk_math_eltwise_unary_sfpu_params_`` call --
                      so this is the same computation through one more layer, and the only thing
                      in the tree that calls those 7 entry points at all.
    """

    row_elements: int = 64
    datum_bytes: int = 2
    top_min: bool = False
    via_wrappers: bool = False
    # Named for the constant it emits rather than bare `mode`: field names have to be unique
    # across every parameter class, or two params in one test produce duplicate perf-CSV
    # columns (test_perf_header_gate.py). `mode` is already taken by GENERALIZED_MOE_GATE.
    top32_mode: int = 0

    def convert_to_cpp(self) -> str:
        if self.row_elements % 32 != 0:
            raise ValueError(
                f"row_elements must be a multiple of 32, got {self.row_elements}"
            )
        if self.top32_mode not in (0, 1):
            raise ValueError(
                f"top32_mode must be 0 (plain) or 1 (pre-sorted), got {self.top32_mode}"
            )
        if self.top32_mode == 1 and self.row_elements < 1024:
            raise ValueError(
                "the pre-sorted mode needs at least one whole 1024-element chunk, got "
                f"{self.row_elements}"
            )
        return "\n".join(
            [
                f"constexpr std::uint32_t TOP32_ROW_ELEMENTS = {self.row_elements}u;",
                f"constexpr std::uint32_t TOP32_DATUM_BYTES = {self.datum_bytes}u;",
                f"constexpr bool TOP32_TOP_MIN = {str(self.top_min).lower()};",
                f"constexpr std::uint32_t TOP32_MODE = {self.top32_mode}u;",
                f"#define TOP32_VIA_WRAPPERS {int(self.via_wrappers)}",
            ]
        )


@dataclass
class ADD_TOP_ROW(TemplateParameter):
    add_top_row: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ADD_TOP_ROW = {str(self.add_top_row).lower()};"


@dataclass
class TO_FROM_INT8(TemplateParameter):
    to_from_int8: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool TO_FROM_INT8 = {str(self.to_from_int8).lower()};"


@dataclass
class IS_MAX_OP(TemplateParameter):
    """Compile-time flag: true for element-wise max, false for min."""

    is_max_op: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool IS_MAX_OP = {str(self.is_max_op).lower()};"


@dataclass
class SFPU_SCALE_EN(TemplateParameter):
    """Compile-time SCALE_EN flag for SFPU kernels that optionally pre-scale their input.

    Pairs with ``SFPU_UNARY_SCALAR``, which carries the scale itself. Kernels in the
    exp family take the scale as a *bfloat16* bit pattern (e.g. 0x3F80 for 1.0f,
    ``p_sfpu::kCONST_1_FP16B``), not an fp32 one.
    """

    scale_en: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_SCALE_EN = {str(self.scale_en).lower()};"


@dataclass
class SOFTMAX_K(TemplateParameter):
    """Number of valid lanes ``k`` for the softmax_k SFPU entry (``_softmax_k_<k>``).

    ``k`` counts values per row inside face 0's 16 columns; columns >= k must be
    exactly 0.0 in DEST (the kernel's predication treats 0.0 as padding).
    """

    softmax_k: int = 16

    def convert_to_cpp(self) -> str:
        return f"constexpr int SOFTMAX_K = {self.softmax_k};"


@dataclass
class SAMPLING_OP(TemplateParameter):
    """Select which ckernel_sfpu_sampling.h entry point the sampling test drives.

    Emits ``#define SAMPLING_OP_<NAME>`` consumed by ``sfpu_sampling_test.cpp``.
    """

    sampling_op: str = "recip_scalar"

    def convert_to_cpp(self) -> str:
        return f"#define SAMPLING_OP_{self.sampling_op.upper()}"


@dataclass
class SAMPLING_LEGACY_COMPAT(TemplateParameter):
    """``legacy_compat`` template argument of ``calculate_sampling_recip_scalar``."""

    legacy_compat: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SAMPLING_LEGACY_COMPAT = {str(self.legacy_compat).lower()};"


@dataclass
class MOE_GATE_TOPK(TemplateParameter):
    """Compile-time configuration of the generic MoE-gate top-k SFPU entry.

    Mirrors the first five template parameters of
    ``ckernel::sfpu::_generic_moe_gate_topk_<normalize, num_selected_experts,
    num_total_experts, zero_tail, full_sort, generate_indices = true>``. The
    dataclass defaults are the compute-API wrapper's
    (api/compute/experimental/generic_moe_gate.h), not the template's -- the only
    template parameter carrying a C++ default is the sixth, ``generate_indices``.

    ``generate_indices`` is deliberately not modelled: the driver instantiates with
    five arguments, so it is pinned to true and the kernel always numbers the experts
    itself. The caller-supplied index-mapping path (generate_indices = false) is
    therefore untested.
    """

    num_selected_experts: int = 8
    num_total_experts: int = 256
    normalize: bool = False
    zero_tail: bool = False
    full_sort: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int MOE_GATE_NUM_SELECTED_EXPERTS = {self.num_selected_experts};",
            f"constexpr int MOE_GATE_NUM_TOTAL_EXPERTS = {self.num_total_experts};",
            f"constexpr bool MOE_GATE_NORMALIZE = {str(self.normalize).lower()};",
            f"constexpr bool MOE_GATE_ZERO_TAIL = {str(self.zero_tail).lower()};",
            f"constexpr bool MOE_GATE_FULL_SORT = {str(self.full_sort).lower()};",
        ]
        return "\n".join(lines)


@dataclass
class MOE_GATE_NORMALIZE_PARAMS(TemplateParameter):
    """``eps`` / ``scale`` for the MoE-gate normalize step, as raw fp32 bit patterns.

    Both are decoded on device via ``Converter::as_float``, so emitting the bit
    patterns keeps the kernel and the torch golden exactly aligned.
    """

    eps_bits: int = 0x00000000  # 0.0f
    scale_bits: int = 0x3F800000  # 1.0f
    extra_scale_bits: int = 0x3F800000  # 1.0f, identity for the do_extra_scale path

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t MOE_GATE_EPS_BITS = {self.eps_bits}u;",
            f"constexpr std::uint32_t MOE_GATE_SCALE_BITS = {self.scale_bits}u;",
            f"constexpr std::uint32_t MOE_GATE_EXTRA_SCALE_BITS = {self.extra_scale_bits}u;",
        ]
        return "\n".join(lines)


# === RUNTIME PARAMETER IMPLEMENTATIONS ===


def generate_input_dim(
    srcA: tuple[int],
    srcB: tuple[int],
    block_ct_dim: int = None,
    block_rt_dim: int = None,
    tile_dimensions: tuple[int, int] = (32, 32),
):
    num_rows, num_cols = tile_dimensions
    validate_tile_dimensions(srcA[0], num_rows)
    validate_tile_dimensions(srcA[1], num_cols)
    validate_tile_dimensions(srcB[0], num_rows)
    validate_tile_dimensions(srcB[1], num_cols)

    full_ct_dim = srcB[1] // num_cols
    full_rt_dim = srcA[0] // num_rows

    block_ct_dim = full_ct_dim if block_ct_dim is None else block_ct_dim
    block_rt_dim = full_rt_dim if block_rt_dim is None else block_rt_dim

    return INPUT_DIMENSIONS(full_rt_dim, full_ct_dim, block_ct_dim, block_rt_dim)


@dataclass
class INPUT_DIMENSIONS(RuntimeParameter):
    full_rt_dim: int = 0
    full_ct_dim: int = 0
    block_ct_dim: int = 0
    block_rt_dim: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t FULL_RT_DIM = {self.full_rt_dim};",
            f"constexpr std::uint32_t FULL_CT_DIM = {self.full_ct_dim};",
            f"constexpr std::uint32_t BLOCK_CT_DIM = {self.block_ct_dim};",
            f"constexpr std::uint32_t BLOCK_RT_DIM = {self.block_rt_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            f"std::uint32_t FULL_RT_DIM;",
            f"std::uint32_t FULL_CT_DIM;",
            f"std::uint32_t BLOCK_CT_DIM;",
            f"std::uint32_t BLOCK_RT_DIM;",
        ]
        return "\n".join(lines), "IIII"


@dataclass
class LOOP_FACTOR(RuntimeParameter):
    loop_factor: int = 1

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t LOOP_FACTOR = {self.loop_factor};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t LOOP_FACTOR;", "I"


@dataclass
class UNPACK_TRANS_FACES(RuntimeParameter):
    unpack_transpose_faces: Transpose = Transpose.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_FACES = {str(self.unpack_transpose_faces.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_FACES;", "?"


@dataclass
class UNPACK_TRANS_WITHIN_FACE(RuntimeParameter):
    unpack_transpose_within_face: Transpose = Transpose.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {str(self.unpack_transpose_within_face.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_WITHIN_FACE;", "?"


@dataclass
class NARROW_TILE(RuntimeParameter):
    narrow_tile: NarrowTile = NarrowTile.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool NARROW_TILE = {str(self.narrow_tile.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool NARROW_TILE;", "?"


@dataclass
class DEST_INDEX(RuntimeParameter):
    dst_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int DST_INDEX = {self.dst_index};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int DST_INDEX;", "i"


@dataclass
class SFPU_TILE_INDICES(RuntimeParameter):
    src0_tile_idx: int = 0
    src1_tile_idx: int = 1
    dst_tile_idx: int = 0

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr int SRC0_TILE_IDX = {self.src0_tile_idx};",
            f"constexpr int SRC1_TILE_IDX = {self.src1_tile_idx};",
            f"constexpr int DST_TILE_IDX = {self.dst_tile_idx};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "int SRC0_TILE_IDX;",
            "int SRC1_TILE_IDX;",
            "int DST_TILE_IDX;",
        ]
        return "\n".join(lines), "iii"


@dataclass
class ZERO_POINT(RuntimeParameter):
    """fp32 bit-pattern of the quant-family zero-point, passed to the binary SFPU
    init at runtime. DEQUANT expects the bits of -zero_point (the init negates the
    contract by loading these bits directly). Ignored by non-quant binary ops."""

    zero_point_bits: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t ZERO_POINT = {self.zero_point_bits}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t ZERO_POINT;", "I"


@dataclass
class SIGN_MAGNITUDE_FORMAT(TemplateParameter):
    """Quant-family SMAG32 datapath toggle; read only by the quant binary ops."""

    sign_magnitude: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool SFPU_SIGN_MAGNITUDE = {str(self.sign_magnitude).lower()};"
        )


@dataclass
class SFPU_DST_ROUNDING_MODE(TemplateParameter):
    """Selects the bf16 narrowing mode for binary SFPU ADD/SUB results."""

    dst_rounding: DstRoundingMode = DstRoundingMode.Default

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::DstRoundingMode SFPU_DST_ROUNDING_MODE = {self.dst_rounding.cpp_enum_value};"


@dataclass
class L1_ACC(RuntimeParameter):
    l1_acc: L1Accumulation = L1Accumulation.No

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int L1_ACC = {1 if self.l1_acc == L1Accumulation.Yes else 0};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int L1_ACC;", "i"


@dataclass
class TILE_COUNT(RuntimeParameter):
    tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TILE_CNT = {self.tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t TILE_CNT;", "I"


@dataclass
class NUM_GUARD_TILES(RuntimeParameter):
    count: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_GUARD_TILES = {self.count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t NUM_GUARD_TILES;", "I"


@dataclass
class INPUT_TILE_CNT(RuntimeParameter):
    input_tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int INPUT_TILE_CNT = {self.input_tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int INPUT_TILE_CNT;", "i"


@dataclass
class OUTPUT_TILE_CNT(RuntimeParameter):
    output_tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int OUTPUT_TILE_CNT = {self.output_tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int OUTPUT_TILE_CNT;", "i"


@dataclass
class REDUCE_TO_ONE(RuntimeParameter):
    is_reduce_to_one: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool IS_REDUCE_TO_ONE = {str(self.is_reduce_to_one).lower()};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool IS_REDUCE_TO_ONE;", "?"


@dataclass
class SRCA_REUSE_COUNT(RuntimeParameter):
    srca_reuse_count: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int SRCA_REUSE_COUNT = {self.srca_reuse_count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int SRCA_REUSE_COUNT;", "i"


@dataclass
class PARTIAL_FACE(RuntimeParameter):
    partial_a: bool = False
    partial_face_pack: bool = False
    partial_b: bool = False
    partial_face_math: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr bool PARTIAL_FACE_A = {str(self.partial_a).lower()};",
            f"constexpr bool PARTIAL_FACE_PACK = {str(self.partial_face_pack).lower()};",
            f"constexpr bool PARTIAL_FACE_B = {str(self.partial_b).lower()};",
            f"constexpr bool PARTIAL_FACE_MATH = {str(self.partial_face_math).lower()};",
        ]

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

    def convert_to_cpp(self) -> str:
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
    input_num_tiles_in_block: int = None
    output_num_tiles_in_block: int = None

    def __post_init__(self):
        if self.input_num_tiles_in_block is None:
            self.input_num_tiles_in_block = self.num_tiles_in_block
        if self.output_num_tiles_in_block is None:
            self.output_num_tiles_in_block = self.num_tiles_in_block

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t NUM_TILES_IN_BLOCK = {self.num_tiles_in_block};",
            f"constexpr std::uint32_t INPUT_NUM_TILES_IN_BLOCK = {self.input_num_tiles_in_block};",
            f"constexpr std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = {self.output_num_tiles_in_block};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "std::uint32_t NUM_TILES_IN_BLOCK;",
            "std::uint32_t INPUT_NUM_TILES_IN_BLOCK;",
            "std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_BLOCKS(RuntimeParameter):
    num_blocks: int = 1
    input_num_blocks: int = None
    output_num_blocks: int = None

    def __post_init__(self):
        if self.input_num_blocks is None:
            self.input_num_blocks = self.num_blocks
        if self.output_num_blocks is None:
            self.output_num_blocks = self.num_blocks

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr int NUM_BLOCKS = {self.num_blocks};",
            f"constexpr int INPUT_NUM_BLOCKS = {self.input_num_blocks};",
            f"constexpr int OUTPUT_NUM_BLOCKS = {self.output_num_blocks};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "int NUM_BLOCKS;",
            "int INPUT_NUM_BLOCKS;",
            "int OUTPUT_NUM_BLOCKS;",
        ]
        return "\n".join(lines), "iii"


@dataclass
class NUM_FACES(RuntimeParameter):
    num_faces: int = 4  # Number of active faces for result matrix
    num_faces_A: int = 4  # Number of active faces for matrix A
    num_faces_B: int = 4  # Number of active faces for matrix B

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t num_faces = {self.num_faces};",
            f"constexpr std::uint32_t num_faces_A = {self.num_faces_A};",
            f"constexpr std::uint32_t num_faces_B = {self.num_faces_B};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t num_faces;",
            "std::uint32_t num_faces_A;",
            "std::uint32_t num_faces_B;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_FACES_R_DIM(RuntimeParameter):
    num_faces_r_dim_A: int = 2  # Number of faces in row dimension for matrix A
    num_faces_r_dim_B: int = 2  # Number of faces in row dimension for matrix B

    def convert_to_cpp(self) -> str:
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

    def convert_to_cpp(self) -> str:
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


# NOTE: If IN_FACE_DIMS parameter is propagated throughout test-infra, it can replace
# other variables used to pass input face dimensions (eg. TEST_FACE_DIMS).
@dataclass
class IN_FACE_DIMS(RuntimeParameter):
    in0_face_r_dim: int = 16
    in0_face_c_dim: int = 16
    in1_face_r_dim: int = 16
    in1_face_c_dim: int = 16

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int in0_face_r_dim = {self.in0_face_r_dim};",
            f"constexpr int in0_face_c_dim = {self.in0_face_c_dim};",
            f"constexpr int in1_face_r_dim = {self.in1_face_r_dim};",
            f"constexpr int in1_face_c_dim = {self.in1_face_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int in0_face_r_dim;",
            "int in0_face_c_dim;",
            "int in1_face_r_dim;",
            "int in1_face_c_dim;",
        ]
        return "\n".join(lines), "iiii"


@dataclass
class TEST_FACE_DIMS(RuntimeParameter):
    face_r_dim: int = 16
    face_c_dim: int = 16

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TEST_FACE_R_DIM = {self.face_r_dim};",
            f"constexpr std::uint32_t TEST_FACE_C_DIM = {self.face_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t TEST_FACE_R_DIM;",
            "std::uint32_t TEST_FACE_C_DIM;",
        ]
        return "\n".join(lines), "II"


@dataclass
class IN_TILE_DIMS(RuntimeParameter):
    in0_r_dim: int = 32
    in0_c_dim: int = 32
    in1_r_dim: int = 32
    in1_c_dim: int = 32

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t in0_tile_r_dim = {self.in0_r_dim};",
            f"constexpr std::uint32_t in0_tile_c_dim = {self.in0_c_dim};",
            f"constexpr std::uint32_t in1_tile_r_dim = {self.in1_r_dim};",
            f"constexpr std::uint32_t in1_tile_c_dim = {self.in1_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t in0_tile_r_dim;",
            "std::uint32_t in0_tile_c_dim;",
            "std::uint32_t in1_tile_r_dim;",
            "std::uint32_t in1_tile_c_dim;",
        ]
        return "\n".join(lines), "IIII"


@dataclass
class RELU_CONFIG(RuntimeParameter):
    """Packer ReLU config: packed 32-bit value (mode in low 2 bits, threshold in bits 16–31)."""

    relu_config: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int RELU_CONFIG = {self.relu_config};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int RELU_CONFIG;", "i"


@dataclass
class NUM_ROWS_TO_PACK(RuntimeParameter):
    num_rows_to_pack: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_ROWS_TO_PACK = {self.num_rows_to_pack};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t NUM_ROWS_TO_PACK;", "I"


@dataclass
class EMA_ALPHA_BETA(TemplateParameter):
    """Alpha/beta smoothing weights for the EMA entry, as raw fp32 bit patterns.

    ``_load_alpha_beta_`` loads each as the fp32 representation into LREG5 (alpha)
    and LREG6 (beta); the kernel computes ``EMA_new = alpha*EMA_old + beta*input``.
    Emitting the bit patterns keeps the C++ and torch golden exactly aligned.
    """

    alpha_bits: int = 0x3E800000  # 0.25f
    beta_bits: int = 0x3F400000  # 0.75f

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t EMA_ALPHA_BITS = {self.alpha_bits}u;",
            f"constexpr std::uint32_t EMA_BETA_BITS = {self.beta_bits}u;",
        ]
        return "\n".join(lines)


@dataclass
class TILE_DST_CT_OFFSET(TemplateParameter):
    offset: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TILE_DST_CT_OFFSET = {self.offset};"


@dataclass
class CONFIGURE_TEST_RUN_IDX(RuntimeParameter):
    configure_test_run_idx: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CONFIGURE_TEST_RUN_IDX = {self.configure_test_run_idx};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t CONFIGURE_TEST_RUN_IDX;", "I"


@dataclass
class HOST_IS_STREAM_PRODUCER(RuntimeParameter):
    host_is_stream_producer: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool HOST_IS_STREAM_PRODUCER = {str(self.host_is_stream_producer).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool HOST_IS_STREAM_PRODUCER;", "?"


@dataclass
class HOST_IS_STREAM_CONSUMER(RuntimeParameter):
    host_is_stream_consumer: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool HOST_IS_STREAM_CONSUMER = {str(self.host_is_stream_consumer).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool HOST_IS_STREAM_CONSUMER;", "?"


@dataclass
class FILL_INT_FORMAT(TemplateParameter):
    data_format: DataFormat = DataFormat.Int32

    def convert_to_cpp(self) -> str:
        return f"constexpr auto FILL_INT_FORMAT = DataFormat::{self.data_format.name};"


@dataclass
class TYPECAST_FORMATS(TemplateParameter):
    """Compile-time config for the SFPU typecast test kernel.

    Emits the logical input/output ``DataFormat`` enum values consumed by
    ``typecast_tile<IN, OUT>`` (mirrored by the typecast dispatch in
    ``sfpu_operations.h``, reached via ``SfpuType::typecast``).
    """

    input_format: DataFormat = DataFormat.Float32
    output_format: DataFormat = DataFormat.Float16_b

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr auto TYPECAST_IN_FORMAT = DataFormat::{self.input_format.name};",
            f"constexpr auto TYPECAST_OUT_FORMAT = DataFormat::{self.output_format.name};",
        ]
        return "\n".join(lines)


@dataclass
class CUSTOM_MM_REUSE_CFG(TemplateParameter):
    """Compile-time chain geometry for the custom_mm_reuse_dest_srcb test.

    The reuse LLK is the second matmul of a fused chain: its in0 operand is moved
    out of DEST (where a preceding custom_mm<dense_packing> left its output) into
    SrcB via MOVD2B; only the weights are unpacked into SrcA. The C++ driver runs
    the whole chain, so these four constants size both the producer custom_mm and
    the reuse consumer:

    ``in0_tile_r_dim``  height of the in0 tile (1/2/4/8); selects the MOVD2B pattern.
    ``producer_kt``     producer inner dim in tiles (its kt_dim).
    ``reuse_kt``        consumer inner dim in tiles == producer ct_dim == number of
                        DEST-resident in0 tiles the reuse op reduces over.
    ``reuse_nt``        consumer output width in tiles (1 to 16).
    """

    in0_tile_r_dim: int = 8
    producer_kt: int = 2
    reuse_kt: int = 2
    reuse_nt: int = 1

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t IN0_TILE_R_DIM = {self.in0_tile_r_dim}u;",
            f"constexpr std::uint32_t PRODUCER_KT = {self.producer_kt}u;",
            f"constexpr std::uint32_t REUSE_KT = {self.reuse_kt}u;",
            f"constexpr std::uint32_t REUSE_NT = {self.reuse_nt}u;",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_CUSTOM_MM_FLAGS(TemplateParameter):
    """Compile-time knobs for the experimental sdpa_custom_mm block matmul test.

    Emitted as #defines (not constexpr) because the C++ driver both consumes
    SIGNAL_GRANULARITY as a non-type template argument and guards each name with an
    #ifndef default, so a standalone compile of the source still works.

    signal_granularity : FPU->SFPU post cadence (llk_math_sdpa_custom_mm template arg).
                         Must divide ct_dim. Purely a signalling cadence; does not change
                         the numeric result.
    read_transposed    : selects the transposed SrcA (in1) L1 walk in the unpack LLK.
    mm_transpose       : the `transpose` init flag threaded through the unpack/math inits
                         (addr_mod SrcA increment + Haloize_mode).
    """

    signal_granularity: int = 1
    read_transposed: bool = False
    mm_transpose: bool = False

    def convert_to_cpp(self) -> str:
        lines = [
            f"#define SIGNAL_GRANULARITY {self.signal_granularity}",
            f"#define READ_TRANSPOSED {str(self.read_transposed).lower()}",
            f"#define MM_TRANSPOSE {str(self.mm_transpose).lower()}",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_CUSTOM_MM_REUSE_DEST(TemplateParameter):
    """Compile-time dims for the experimental sdpa_custom_mm_reuse_dest_srcb OV matmul.

    kt_dim: number of K tiles (softmax-score / V K tiles). The unpack MOP requires an
            even kt_dim >= 2 (see the "kt_dim: even number from 2 to 256" note in
            llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h); kt_dim < 2 underflows the
            MOP iteration count and hangs the unpacker.
    nt_dim: number of V head-dim output tiles per K iteration (1..16).
    """

    kt_dim: int = 2
    nt_dim: int = 1

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};",
            f"constexpr std::uint32_t NT_DIM = {self.nt_dim};",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_REDUCE_ROW_POOL(TemplateParameter):
    """Select MAX vs SUM for the experimental sdpa_reduce_row op.

    Emits ``constexpr int SDPA_REDUCE_POOL = 0|1;`` consumed by
    sources/sdpa_reduce_row_test.cpp (0 == MAX -> calculate_sdpa_reduce_max_row,
    1 == SUM -> calculate_sdpa_reduce_sum_row). The op only supports MAX and SUM.
    """

    reduce_pool: ReducePool = ReducePool.Max

    def convert_to_cpp(self) -> str:
        pool_to_int = {ReducePool.Max: 0, ReducePool.Sum: 1}
        try:
            value = pool_to_int[self.reduce_pool]
        except KeyError:
            raise ValueError(
                f"sdpa_reduce_row supports only MAX and SUM pools, got {self.reduce_pool}"
            )
        return f"constexpr int SDPA_REDUCE_POOL = {value};"


@dataclass
class MUL_REDUCE_SCALAR_CHUNK_SIZE(RuntimeParameter):
    chunk_size: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CHUNK_SIZE = {self.chunk_size};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t CHUNK_SIZE;", "I"


@dataclass
class MULSCALARHIFI_HIFI_INIT(TemplateParameter):
    """Select the REVERTED HiFi general-init path in eltwise_mul_scalar_hifi_test.cpp.

    Emits ``#define HIFI_GENERAL_INIT`` when ``enabled`` so the C++ reproduces
    ``deepseek_binary_dest_reuse_tiles_init``'s HiFi branch verbatim: the general
    ``_llk_math_eltwise_binary_init_<ELWMUL, NONE, MATH_FIDELITY, DEST_TO_SRCA>``
    called with a hard-coded ``ckernel::DEFAULT_TENSOR_SHAPE`` instead of the
    kernel's real tile shape (api/compute/experimental/eltwise_mul_scalar.h:74-88).
    That mis-specialization hangs the device on silicon (tt-blaze #1760); the
    Python test is marked xfail. When disabled, the C++ falls through to the
    non-reverted control path (general init with the correct tensor_shape).
    """

    enabled: bool = True

    def convert_to_cpp(self) -> str:
        return "#define HIFI_GENERAL_INIT" if self.enabled else ""


@dataclass
class ZERO_PAD_ROWS(TemplateParameter):
    """SFPU row bounds for _zero_pad_tile_."""

    valid_rows: int = 0
    total_rows: int = 32

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int ZERO_PAD_VALID_ROWS = {self.valid_rows};\n"
            f"constexpr int ZERO_PAD_TOTAL_ROWS = {self.total_rows};"
        )


@dataclass
class SPARSE_K_CONFIG(TemplateParameter):
    sparse_k_iterations: int = 32
    bank_mask: int = 0x3F
    my_bank: int = 0
    global_bank_shift: int = 14
    within_bank_mask: int = 0x3FFF
    out_shift: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int SPARSE_K_ITERATIONS = {self.sparse_k_iterations};\n"
            f"constexpr std::uint32_t SPARSE_K_BANK_MASK = {self.bank_mask}u;\n"
            f"constexpr std::uint32_t SPARSE_K_MY_BANK = {self.my_bank}u;\n"
            f"constexpr std::uint32_t SPARSE_K_GLOBAL_BANK_SHIFT = {self.global_bank_shift}u;\n"
            f"constexpr std::uint32_t SPARSE_K_WITHIN_BANK_MASK = {self.within_bank_mask}u;\n"
            f"constexpr std::uint32_t SPARSE_K_OUT_SHIFT = {self.out_shift}u;"
        )


@dataclass
class CLAMPED_SILU_PARAMS(TemplateParameter):
    clamped_silu_op: str = "GATE"
    scalar0: float = 1.0
    scalar1: float = 1.0

    @staticmethod
    def _fp32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def convert_to_cpp(self) -> str:
        return (
            f"#define CLAMPED_SILU_OP_{self.clamped_silu_op}\n"
            f"constexpr std::uint32_t CLAMPED_SILU_SCALAR0 = {self._fp32_bits(self.scalar0)}u;\n"
            f"constexpr std::uint32_t CLAMPED_SILU_SCALAR1 = {self._fp32_bits(self.scalar1)}u;"
        )
