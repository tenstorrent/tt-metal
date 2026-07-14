# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from enum import Enum, auto

import torch

from .format_config import DataFormat

format_dict = {
    DataFormat.Tf32: torch.float32,
    DataFormat.Float32: torch.float32,
    DataFormat.Float16: torch.float16,
    DataFormat.Float16_b: torch.bfloat16,
    DataFormat.Bfp8_b: torch.bfloat16,  # BFP8 not native to PyTorch, is represented as bfloat16
    DataFormat.Bfp4_b: torch.bfloat16,  # BFP4 not native to PyTorch, is represented as bfloat16
    DataFormat.Bfp2_b: torch.bfloat16,  # BFP2 not native to PyTorch, is represented as bfloat16
    DataFormat.Int32: torch.int32,
    DataFormat.UInt32: torch.int64,
    DataFormat.Int16: torch.int16,
    DataFormat.UInt16: torch.int32,
    DataFormat.Int8: torch.int8,
    DataFormat.UInt8: torch.uint8,
    DataFormat.MxFp8R: torch.bfloat16,
    DataFormat.MxFp8P: torch.bfloat16,
    DataFormat.MxFp4: torch.bfloat16,
    DataFormat.MxInt8: torch.bfloat16,
    DataFormat.MxInt4: torch.bfloat16,
    DataFormat.MxInt2: torch.bfloat16,
    DataFormat.Fp8_e4m3: torch.bfloat16,
}


class MathOpType(Enum):
    """Enum for different types of math operations."""

    SFPU_UNARY = auto()
    SFPU_BINARY = auto()
    SFPU_BINARY_INT = auto()
    SFPU_TERNARY = auto()
    SFPU_BINOP_SCALAR = auto()

    FPU_BINARY = auto()
    REDUCE = auto()


# Operation specification
OpSpec = namedtuple("OpSpec", ["cpp_enum_value", "operation_type"])


class MathOperation(Enum):
    """
    An enumeration class that holds all the math operations supported by the LLKs.
    Each enum value is an OpSpec tuple containing (cpp_enum_value, operation_type).

    Operations are organized by type:
    - FPU_BINARY: Floating Point Unit binary operations
    - SFPU_UNARY: Special Function Processing Unit unary operations
    - SFPU_BINARY: Special Function Processing Unit binary operations
    - REDUCE: Reduction operations
    """

    # =============================================================================
    # FPU BINARY OPERATIONS
    # =============================================================================
    Elwadd = OpSpec("ELWADD", MathOpType.FPU_BINARY)
    Elwmul = OpSpec("ELWMUL", MathOpType.FPU_BINARY)
    Elwsub = OpSpec("ELWSUB", MathOpType.FPU_BINARY)

    # =============================================================================
    # SFPU UNARY OPERATIONS
    # =============================================================================
    Abs = OpSpec("abs", MathOpType.SFPU_UNARY)
    Atanh = OpSpec("atanh", MathOpType.SFPU_UNARY)
    Asinh = OpSpec("asinh", MathOpType.SFPU_UNARY)
    Acosh = OpSpec("acosh", MathOpType.SFPU_UNARY)
    Celu = OpSpec("celu", MathOpType.SFPU_UNARY)
    Cos = OpSpec("cosine", MathOpType.SFPU_UNARY)
    Elu = OpSpec("elu", MathOpType.SFPU_UNARY)
    Exp = OpSpec("exponential", MathOpType.SFPU_UNARY)
    Exp2 = OpSpec("exp2", MathOpType.SFPU_UNARY)
    Fill = OpSpec("fill", MathOpType.SFPU_UNARY)
    Gelu = OpSpec("gelu", MathOpType.SFPU_UNARY)
    GeluTanh = OpSpec("gelu_tanh", MathOpType.SFPU_UNARY)
    Hardsigmoid = OpSpec("hardsigmoid", MathOpType.SFPU_UNARY)
    Log = OpSpec("log", MathOpType.SFPU_UNARY)
    Log1p = OpSpec("log1p", MathOpType.SFPU_UNARY)
    Neg = OpSpec("negative", MathOpType.SFPU_UNARY)
    Reciprocal = OpSpec("reciprocal", MathOpType.SFPU_UNARY)
    Relu = OpSpec("relu", MathOpType.SFPU_UNARY)
    Rsqrt = OpSpec("rsqrt", MathOpType.SFPU_UNARY)
    Sigmoid = OpSpec("sigmoid", MathOpType.SFPU_UNARY)
    Sin = OpSpec("sine", MathOpType.SFPU_UNARY)
    Silu = OpSpec("silu", MathOpType.SFPU_UNARY)
    Signbit = OpSpec("signbit", MathOpType.SFPU_UNARY)
    Sqrt = OpSpec("sqrt", MathOpType.SFPU_UNARY)
    Square = OpSpec("square", MathOpType.SFPU_UNARY)
    Erfinv = OpSpec("erfinv", MathOpType.SFPU_UNARY)
    Heaviside = OpSpec("heaviside", MathOpType.SFPU_UNARY)
    Softshrink = OpSpec("softshrink", MathOpType.SFPU_UNARY)
    Softsign = OpSpec("softsign", MathOpType.SFPU_UNARY)
    Mish = OpSpec("mish", MathOpType.SFPU_UNARY)
    Selu = OpSpec("selu", MathOpType.SFPU_UNARY)
    I0 = OpSpec("i0", MathOpType.SFPU_UNARY)
    Rdiv = OpSpec("rdiv", MathOpType.SFPU_UNARY)
    Clamp = OpSpec("clamp", MathOpType.SFPU_UNARY)
    Hardtanh = OpSpec("hardtanh", MathOpType.SFPU_UNARY)
    Tanhshrink = OpSpec("tanhshrink", MathOpType.SFPU_UNARY)
    Floor = OpSpec("floor", MathOpType.SFPU_UNARY)
    Ceil = OpSpec("ceil", MathOpType.SFPU_UNARY)
    Trunc = OpSpec("trunc", MathOpType.SFPU_UNARY)
    Frac = OpSpec("frac", MathOpType.SFPU_UNARY)
    # Comparison-to-zero unary SFPU ops. cpp_enum_value must exactly match the
    # SfpuType enumerator name so SFPU_UNARY_OPERATION = SfpuType::{value} resolves.
    EqualZero = OpSpec("equal_zero", MathOpType.SFPU_UNARY)
    NotEqualZero = OpSpec("not_equal_zero", MathOpType.SFPU_UNARY)
    LessThanZero = OpSpec("less_than_zero", MathOpType.SFPU_UNARY)
    GreaterThanZero = OpSpec("greater_than_zero", MathOpType.SFPU_UNARY)
    LessThanEqualZero = OpSpec("less_than_equal_zero", MathOpType.SFPU_UNARY)
    GreaterThanEqualZero = OpSpec("greater_than_equal_zero", MathOpType.SFPU_UNARY)
    # Swiglu is technically a binary SFPU op (gate+up → out), but because
    # Quasar lacks the llk_math_eltwise_binary_sfpu_* dispatcher, its test
    # harness runs through the unary SFPU path. We therefore register it as
    # SFPU_UNARY for the test-dispatch constant (SfpuType::swiglu). The
    # actual binary semantics are implemented by the C++ test source which
    # unpacks two input tiles into Dest and calls _calculate_swiglu_ with
    # three offsets directly.
    SfpuSwiGLU = OpSpec("swiglu", MathOpType.SFPU_UNARY)
    Tanh = OpSpec("tanh", MathOpType.SFPU_UNARY)
    # Typecast is dispatched by the (input, output) DataFormat pair rather than a
    # single op, but it maps to SfpuType::typecast and runs through the shared
    # unary-SFPU dispatch (see call_unary_sfpu_operation in sfpu_operations.h).
    Typecast = OpSpec("typecast", MathOpType.SFPU_UNARY)
    Threshold = OpSpec("threshold", MathOpType.SFPU_UNARY)
    ReluMax = OpSpec("relu_max", MathOpType.SFPU_UNARY)
    ReluMin = OpSpec("relu_min", MathOpType.SFPU_UNARY)
    Lrelu = OpSpec("lrelu", MathOpType.SFPU_UNARY)
    Erf = OpSpec("erf", MathOpType.SFPU_UNARY)
    Erfc = OpSpec("erfc", MathOpType.SFPU_UNARY)
    Expm1 = OpSpec("expm1", MathOpType.SFPU_UNARY)
    Cbrt = OpSpec("cbrt", MathOpType.SFPU_UNARY)
    I1 = OpSpec("i1", MathOpType.SFPU_UNARY)
    Sign = OpSpec("sign", MathOpType.SFPU_UNARY)
    TanhDerivative = OpSpec("tanh_derivative", MathOpType.SFPU_UNARY)
    # Legacy LUT variant of tanh'(x): 1 - tanh(x)^2 with tanh from the piecewise
    # LUT (distinct kernel path from the accurate sech2 TanhDerivative above).
    TanhDerivativeLut = OpSpec("tanh_derivative_lut", MathOpType.SFPU_UNARY)
    # Legacy-compat rsqrt (reciprocal-root method); distinct kernel path from the
    # accurate Rsqrt (which uses legacy_compat=false).
    RsqrtCompat = OpSpec("rsqrt_compat", MathOpType.SFPU_UNARY)
    # Component-wise expm1 shared helper (used by ELU/CELU/SELU); distinct from the
    # standalone Expm1 kernel.
    Expm1Cw = OpSpec("expm1_cw", MathOpType.SFPU_UNARY)
    Hardmish = OpSpec("hardmish", MathOpType.SFPU_UNARY)
    Lgamma = OpSpec("lgamma", MathOpType.SFPU_UNARY)
    Digamma = OpSpec("digamma", MathOpType.SFPU_UNARY)
    Identity = OpSpec("identity", MathOpType.SFPU_UNARY)
    Prelu = OpSpec("prelu", MathOpType.SFPU_UNARY)
    Rpow = OpSpec("rpow", MathOpType.SFPU_UNARY)
    UnaryPower = OpSpec("power", MathOpType.SFPU_UNARY)
    Fmod = OpSpec("fmod", MathOpType.SFPU_UNARY)
    Remainder = OpSpec("remainder", MathOpType.SFPU_UNARY)
    UnaryGt = OpSpec("unary_gt", MathOpType.SFPU_UNARY)
    UnaryLt = OpSpec("unary_lt", MathOpType.SFPU_UNARY)
    UnaryGe = OpSpec("unary_ge", MathOpType.SFPU_UNARY)
    UnaryLe = OpSpec("unary_le", MathOpType.SFPU_UNARY)
    UnaryMax = OpSpec("unary_max", MathOpType.SFPU_UNARY)
    UnaryMin = OpSpec("unary_min", MathOpType.SFPU_UNARY)
    Polygamma = OpSpec("polygamma", MathOpType.SFPU_UNARY)
    Xielu = OpSpec("xielu", MathOpType.SFPU_UNARY)
    Hardshrink = OpSpec("hardshrink", MathOpType.SFPU_UNARY)
    Softplus = OpSpec("softplus", MathOpType.SFPU_UNARY)
    SigmoidAppx = OpSpec("sigmoid_appx", MathOpType.SFPU_UNARY)
    SqrtCustom = OpSpec("sqrt_custom", MathOpType.SFPU_UNARY)
    Add1 = OpSpec("add1", MathOpType.SFPU_UNARY)
    CastFp32ToFp16a = OpSpec("cast_fp32_to_fp16a", MathOpType.SFPU_UNARY)
    # isinf/isnan family: cpp_enum_value must match the SfpuType enumerator name
    # so SFPU_UNARY_OPERATION = SfpuType::{value} resolves.
    Isinf = OpSpec("isinf", MathOpType.SFPU_UNARY)
    Isposinf = OpSpec("isposinf", MathOpType.SFPU_UNARY)
    Isneginf = OpSpec("isneginf", MathOpType.SFPU_UNARY)
    Isnan = OpSpec("isnan", MathOpType.SFPU_UNARY)
    Isfinite = OpSpec("isfinite", MathOpType.SFPU_UNARY)
    AddInt32 = OpSpec("add_int32", MathOpType.SFPU_UNARY)
    SubInt32 = OpSpec("sub_int32", MathOpType.SFPU_UNARY)
    AbsInt32 = OpSpec("abs_int32", MathOpType.SFPU_UNARY)
    BitwiseNot = OpSpec("bitwise_not", MathOpType.SFPU_UNARY)
    # logical_not(x) = (x == 0) ? 1 : 0, exercised on the float (DEFAULT-layout)
    # path. cpp_enum_value must match the SfpuType enumerator name.
    # NOTE: main added `LogicalNot` with the same cpp value; keep both so
    # references to either name resolve (equal values alias in Python enums).
    LogicalNot = OpSpec("logical_not_unary", MathOpType.SFPU_UNARY)
    LogicalNotUnary = OpSpec("logical_not_unary", MathOpType.SFPU_UNARY)
    TopKLocalSort = OpSpec("topk_local_sort", MathOpType.SFPU_UNARY)
    TopKMerge = OpSpec("topk_merge", MathOpType.SFPU_UNARY)
    TopKRebuild = OpSpec("topk_rebuild", MathOpType.SFPU_UNARY)
    # =============================================================================
    # SFPU BINARY OPERATIONS
    # =============================================================================
    SfpuElwadd = OpSpec("ADD", MathOpType.SFPU_BINARY)
    SfpuElwLeftShift = OpSpec("LSHFT", MathOpType.SFPU_BINARY)
    SfpuElwLogicalRightShift = OpSpec("LOGICAL_RSHFT", MathOpType.SFPU_BINARY)
    SfpuElwmul = OpSpec("MUL", MathOpType.SFPU_BINARY)
    SfpuElwRightShift = OpSpec("RSHFT", MathOpType.SFPU_BINARY)
    SfpuElwsub = OpSpec("SUB", MathOpType.SFPU_BINARY)
    SfpuXlogy = OpSpec("XLOGY", MathOpType.SFPU_BINARY)
    SfpuAddTopRow = OpSpec("ADD_TOP_ROW", MathOpType.SFPU_BINARY)
    SfpuElwdiv = OpSpec("DIV", MathOpType.SFPU_BINARY)
    SfpuElwrsub = OpSpec("RSUB", MathOpType.SFPU_BINARY)
    SfpuElwpow = OpSpec("POW", MathOpType.SFPU_BINARY)
    SfpuElwmulInt = OpSpec("MUL", MathOpType.SFPU_BINARY_INT)
    SfpuGtInt = OpSpec("GT_INT", MathOpType.SFPU_BINARY_INT)
    SfpuLtInt = OpSpec("LT_INT", MathOpType.SFPU_BINARY_INT)
    SfpuLeInt = OpSpec("LE_INT", MathOpType.SFPU_BINARY_INT)
    SfpuGeInt = OpSpec("GE_INT", MathOpType.SFPU_BINARY_INT)
    SfpuElwLt = OpSpec("LT", MathOpType.SFPU_BINARY)
    SfpuElwGt = OpSpec("GT", MathOpType.SFPU_BINARY)
    SfpuElwLe = OpSpec("LE", MathOpType.SFPU_BINARY)
    SfpuElwGe = OpSpec("GE", MathOpType.SFPU_BINARY)
    SfpuElwEq = OpSpec("EQ", MathOpType.SFPU_BINARY)
    SfpuElwNe = OpSpec("NE", MathOpType.SFPU_BINARY)
    # Binary SFPU kernels wired up for functional coverage (no dedicated production
    # BinaryOp; the enum values live at the end of ckernel::BinaryOp).
    SfpuBinaryMax = OpSpec("MAX", MathOpType.SFPU_BINARY)
    SfpuBinaryMin = OpSpec("MIN", MathOpType.SFPU_BINARY)
    SfpuBinaryFmod = OpSpec("FMOD", MathOpType.SFPU_BINARY)
    SfpuBinaryRemainder = OpSpec("REMAINDER", MathOpType.SFPU_BINARY)
    SfpuBitwiseAnd = OpSpec("BITWISE_AND", MathOpType.SFPU_BINARY)
    SfpuBitwiseOr = OpSpec("BITWISE_OR", MathOpType.SFPU_BINARY)
    SfpuBitwiseXor = OpSpec("BITWISE_XOR", MathOpType.SFPU_BINARY)
    SfpuDivInt32 = OpSpec("DIV_INT32", MathOpType.SFPU_BINARY)
    SfpuDivInt32Floor = OpSpec("DIV_INT32_FLOOR", MathOpType.SFPU_BINARY)
    SfpuGcd = OpSpec("GCD", MathOpType.SFPU_BINARY)
    SfpuLcm = OpSpec("LCM", MathOpType.SFPU_BINARY)
    SfpuRsubInt32 = OpSpec("RSUB_INT32", MathOpType.SFPU_BINARY)
    SfpuMask = OpSpec("MASK", MathOpType.SFPU_BINARY)
    SfpuAtan2 = OpSpec("ATAN2", MathOpType.SFPU_BINARY)
    SfpuMulInt32 = OpSpec("MUL_INT32", MathOpType.SFPU_BINARY)
    SfpuIsclose = OpSpec("ISCLOSE", MathOpType.SFPU_BINARY)
    SfpuLogsigmoid = OpSpec("LOGSIGMOID", MathOpType.SFPU_BINARY)

    # =============================================================================
    # SFPU TERNARY OPERATIONS
    # =============================================================================
    SfpuWhere = OpSpec("WHERE", MathOpType.SFPU_TERNARY)
    TTNNWhere = SfpuWhere
    SfpuAddcmul = OpSpec("addcmul", MathOpType.SFPU_TERNARY)
    SfpuAddcdiv = OpSpec("addcdiv", MathOpType.SFPU_TERNARY)
    SfpuLerp = OpSpec("lerp", MathOpType.SFPU_TERNARY)
    SfpuSnakeBeta = OpSpec("snake_beta", MathOpType.SFPU_TERNARY)

    # =============================================================================
    # SFPU FLOAT UNARY-WITH-SCALAR BINOPS
    # =============================================================================
    ScalarAdd = OpSpec("ADD", MathOpType.SFPU_BINOP_SCALAR)
    ScalarSub = OpSpec("SUB", MathOpType.SFPU_BINOP_SCALAR)
    ScalarMul = OpSpec("MUL", MathOpType.SFPU_BINOP_SCALAR)
    ScalarDiv = OpSpec("DIV", MathOpType.SFPU_BINOP_SCALAR)
    ScalarRsub = OpSpec("RSUB", MathOpType.SFPU_BINOP_SCALAR)

    # =============================================================================
    # REDUCE OPERATIONS
    # =============================================================================
    ReduceColumn = OpSpec("REDUCE_COL", MathOpType.REDUCE)
    ReduceRow = OpSpec("REDUCE_ROW", MathOpType.REDUCE)
    ReduceScalar = OpSpec("REDUCE_SCALAR", MathOpType.REDUCE)

    # =============================================================================
    # PROPERTIES AND UTILITY METHODS
    # =============================================================================
    @property
    def cpp_enum_value(self):
        """Get the C++ enum value for this operation."""
        return self.value.cpp_enum_value

    @property
    def operation_type(self):
        """Get the operation type for this operation."""
        return self.value.operation_type

    @classmethod
    def _get_operations_by_type(cls, op_type: MathOpType):
        """Get all operations of a specific type."""
        return {op for op in cls if op.operation_type == op_type}

    @classmethod
    def get_fpu_binary_operations(cls):
        """Get all FPU binary operations."""
        return cls._get_operations_by_type(MathOpType.FPU_BINARY)

    @classmethod
    def get_sfpu_unary_operations(cls):
        """Get all SFPU unary operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_UNARY)

    @classmethod
    def get_sfpu_binary_operations(cls):
        """Get all SFPU binary operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_BINARY)

    @classmethod
    def get_sfpu_ternary_operations(cls):
        """Get all SFPU ternary operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_TERNARY)

    @classmethod
    def get_sfpu_binop_scalar_operations(cls):
        """Get all SFPU float unary-with-scalar binop operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_BINOP_SCALAR)

    @classmethod
    def get_reduce_operations(cls):
        """Get all reduce operations."""
        return cls._get_operations_by_type(MathOpType.REDUCE)


SFPU_UNARY_OPERATIONS = MathOperation.get_sfpu_unary_operations()
SFPU_BINARY_OPERATIONS = MathOperation.get_sfpu_binary_operations()
SFPU_TERNARY_OPERATIONS = MathOperation.get_sfpu_ternary_operations()
SFPU_BINOP_SCALAR_OPERATIONS = MathOperation.get_sfpu_binop_scalar_operations()
FPU_BINARY_OPERATIONS = MathOperation.get_fpu_binary_operations()
REDUCE_OPERATIONS = MathOperation.get_reduce_operations()


class ReduceDimension(Enum):
    Column = "REDUCE_COL"
    Row = "REDUCE_ROW"
    Scalar = "REDUCE_SCALAR"

    @property
    def cpp_enum_value(self):
        return f"ReduceDim::{self.value}"


class ReducePool(Enum):
    Max = "MAX"
    Min = "MIN"
    Sum = "SUM"
    Average = "AVG"

    @property
    def cpp_enum_value(self):
        return f"PoolType::{self.value}"


class DestAccumulation(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class L1Accumulation(Enum):
    Yes = 1
    No = 0

    @property
    def cpp_enum_value(self):
        return str(self.value)


class StochasticRounding(Enum):
    No = "StochRndType::None"
    Fpu = "StochRndType::Fpu"
    Pack = "StochRndType::Pack"
    All = "StochRndType::All"


class PackerReluType(Enum):
    """
    Relu activation function types for packer operations.
    """

    NoRelu = "NO_RELU"
    ZeroRelu = "ZERO_RELU"
    MinThresholdRelu = "MIN_THRESHOLD_RELU"
    MaxThresholdRelu = "MAX_THRESHOLD_RELU"

    @property
    def cpp_enum_value(self):
        return f"ReluType::{self.value}"

    @property
    def bits(self) -> int:
        return _PACKER_RELU_BITS[self.value]

    @classmethod
    def from_bits(cls, bits: int) -> "PackerReluType":
        return cls(_PACKER_RELU_BITS_INV[bits])


_PACKER_RELU_BITS = {
    "NO_RELU": 0,
    "ZERO_RELU": 1,
    "MIN_THRESHOLD_RELU": 2,
    "MAX_THRESHOLD_RELU": 3,
}
_PACKER_RELU_BITS_INV = {v: k for k, v in _PACKER_RELU_BITS.items()}


def pack_relu_config(mode: "PackerReluType", threshold_bits: int) -> int:
    """Pack ReLU mode (2 bits) and threshold (16 bits) into a 32-bit config word."""
    return (mode.bits & 0x3) | ((threshold_bits & 0xFFFF) << 16)


class Haloize(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class ApproximationMode(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class Transpose(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class MathFidelity(Enum):
    LoFi = 0
    HiFi2 = 2
    HiFi3 = 3
    HiFi4 = 4

    @property
    def cpp_enum_value(self):
        return f"ckernel::MathFidelity::{self.name}"


class DestSync(Enum):
    Half = "SyncHalf"
    Full = "SyncFull"

    @property
    def cpp_enum_value(self):
        return f"DstSync::{self.value}"


class NarrowTile(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class PartialFace(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class EnforceFP32Accumulation(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class ClearFP32DstAcc(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class AccToDest(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class UnpackToDest(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class Tilize(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()

    @property
    def pack_mode_value(self) -> str:
        return "PackMode::Tilize" if self == Tilize.Yes else "PackMode::Default"


class FastMode(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class StableSort(Enum):
    Yes = True
    No = False

    @property
    def cpp_enum_value(self):
        return str(self.value).lower()


class Mailboxes(Enum):
    Unpacker = 0x1FFB8
    Math = Unpacker + 4
    Packer = Unpacker + 8
    BriscCommand0 = Unpacker + 12
    BriscCommand1 = Unpacker + 16
    BriscCounter = Unpacker + 20
    BriscBread0 = Unpacker + 24
    BriscBread1 = Unpacker + 28


class MailboxesCoverage(Enum):
    Unpacker = 0x6DFB8
    Math = Unpacker + 4
    Packer = Unpacker + 8
    BriscCommand0 = Unpacker + 12
    BriscCommand1 = Unpacker + 16
    BriscCounter = Unpacker + 20
    BriscBread0 = Unpacker + 24
    BriscBread1 = Unpacker + 28


class MailboxesQuasar(Enum):
    Unpacker = 0x1FFB8
    Math = Unpacker + 4
    Packer = Unpacker + 8
    Sfpu = Unpacker + 12


class MailboxesCoverageQuasar(Enum):
    Unpacker = 0x6DFB8
    Math = Unpacker + 4
    Packer = Unpacker + 8
    Sfpu = Unpacker + 12


class BriscCmd(Enum):
    IDLE_STATE = 0
    START_TRISCS = 1
    RESET_TRISCS = 2
    UPDATE_START_ADDR_CACHE_AND_START = 3


format_tile_sizes = {
    DataFormat.Bfp8_b: 1088,
    DataFormat.Bfp4_b: 576,
    DataFormat.Bfp2_b: 320,
    DataFormat.Float16: 2048,
    DataFormat.Float16_b: 2048,
    DataFormat.Float32: 4096,
    DataFormat.Int32: 4096,
    DataFormat.Tf32: 3072,  # 3 bytes * 1024 elements
    DataFormat.UInt32: 4096,
    DataFormat.Int16: 2048,
    DataFormat.UInt16: 2048,
    DataFormat.Int8: 1024,  # 1 byte * 1024 elements
    DataFormat.UInt8: 1024,  # 1 byte * 1024 elements
    # MX formats: 1 byte per element + 1 scale (8 bits) per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 32 elements) = 1056 bytes
    DataFormat.MxFp8R: 1056,
    DataFormat.MxFp8P: 1056,
    # MXFp4 half byte per element + 1 scale (8 bits) per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 16 bytes of FP4 data) = 544 bytes
    DataFormat.MxFp4: 544,
    # MxInt8: 1 byte per element + 1 scale (8 bits) per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 32 bytes of INT8 data) = 1056 bytes
    DataFormat.MxInt8: 1056,
    # MxInt4: half byte per element (2 packed per byte) + 1 scale per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 16 bytes of INT4 data) = 544 bytes
    DataFormat.MxInt4: 544,
    # MxInt2: quarter byte per element (4 packed per byte) + 1 scale per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 8 bytes of INT2 data) = 288 bytes
    DataFormat.MxInt2: 288,
    DataFormat.Fp8_e4m3: 1024,  # 1 byte per element, no exponent section
}


class BroadcastType(Enum):
    """
    Enum for broadcast types in LLK kernels.
    """

    None_ = "NONE"
    Column = "COL"
    Row = "ROW"
    Scalar = "SCALAR"

    @property
    def cpp_enum_value(self):
        return f"BroadcastType::{self.value}"


class EltwiseBinaryReuseDestType(Enum):
    """
    Enum for destination reuse types in elementwise binary ops.
    """

    NONE = "NONE"
    DEST_TO_SRCA = "DEST_TO_SRCA"
    DEST_TO_SRCB = "DEST_TO_SRCB"

    @property
    def cpp_enum_value(self):
        return f"EltwiseBinaryReuseDestType::{self.value}"


class DataCopyType(Enum):
    A2D = "A2D"
    B2D = "B2D"

    @property
    def cpp_enum_value(self):
        return f"DataCopyType::{self.value}"


class BlocksCalculationAlgorithm(Enum):
    """
    Enum for block processing algorithms in LLK kernels.
    """

    Standard = "STANDARD"
    Tilize = "TILIZE"
    Untilize = "UNTILIZE"


class PerfRunType(Enum):
    L1_TO_L1 = 1
    UNPACK_ISOLATE = 2
    MATH_ISOLATE = 3
    PACK_ISOLATE = 4
    L1_CONGESTION = 5


# Single pytest case runs every PerfRunType so the module CSV has one
# homogeneous schema (mean/TEXT_SIZE columns for all modes in each row).
# Pass as a nested list so @parametrize yields one value: the full mode list.
PERF_RUN_TYPES_QUASAR = [
    [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ],
]


# ******** QUASAR specific ********
class ImpliedMathFormat(Enum):
    No = "false"
    Yes = "true"


class UnpackerEngine(Enum):
    """
    Enum for unpacker engine selection.
    """

    UnpA = "UNP_A"
    UnpB = "UNP_B"
    UnpS = "UNP_S"
    UnpDest = "UNP_DEST"


class ReluConfig(Enum):
    NoRelu = 0
    ZeroRelu = 1


class TopKSortDirection(Enum):
    Descending = 0
    Ascending = 1


class VectorMode(Enum):
    """Mirrors ckernel::VectorMode in tt_llk_quasar/llk_lib/llk_defs.h.

    Selects which faces an SFPU dispatch processes:
      * ``None_``: invoke the SFPU kernel once with no face advances (covers face 0 only).
      * ``R``: faces 0 and 1 (top face-row of the tile).
      * ``C``: faces 0 and 2 (left face-column of the tile).
      * ``RC``: all four faces — the default.
    """

    None_ = 0
    R = 1
    C = 2
    RC = 4

    @property
    def cpp_enum_value(self):
        return (
            f"ckernel::VectorMode::{'None' if self == VectorMode.None_ else self.name}"
        )


class GoldenType(Enum):
    L1_GOLDEN = "L1_GOLDEN"
    MASTER_GOLDEN = "MASTER_GOLDEN"


# *********************************
