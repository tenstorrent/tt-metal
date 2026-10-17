# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Threshold decoding of the tt-llk scalar compares (tt-llk#1701 item 16).

``_calculate_comp_unary_<APPROX, unary_*>(std::uint32_t value)`` in
``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h`` takes the threshold as fp32 bits, like the metal
``calculate_unary_*`` kernels and the other tt-llk ops with a uint32 scalar (relu, threshold, fill). It
used to build the vFloat with ``vFloat s = value``, a numeric uint32 -> float conversion: 0x3F000000
(0.5f) became 1056964608.0f, and no negative, infinite or NaN threshold could be expressed at all.

Inputs and thresholds stay away from NaN and from a zero threshold in the decode test, so it checks
only how the threshold is decoded. The golden is torch's compare and the check is exact: every lane
is 0.0 or 1.0.

Both tests run on Blackhole and Wormhole (the Wormhole copy of this header decodes the threshold the
same way). Wormhole's ordered compares (gt/lt/ge/le) get an infinite threshold against an infinite
datum wrong, a defect of those compares rather than of the decode, so exactly those four decode cells
are expected failures there (measured on n150; see ``_wormhole_inf_compare_defect``).
``test_sfpu_comp_unary_eq_ne_ieee`` checks that eq/ne follow IEEE equality: -0.0 equals +0.0 and a NaN
threshold equals nothing, including a datum with the same NaN bits. It also drives the nonzero
thresholds of the decode test, with the NaN and denormal datums added, so a threshold-decode
regression on either architecture shows up there too.
"""

import struct

import numpy as np
import pytest
import torch
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathOperation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    MATH_OP,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

ELEMENTS_PER_TILE = 1024


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# Inputs: each finite threshold below with its fp32 neighbours, the largest finite values next to the
# infinities, and values on both sides of every threshold.
INPUTS = (
    0.0,
    -0.0,
    0.5,
    _from_bits(0x3EFFFFFF),  # 0.5 - 1 ulp
    _from_bits(0x3F000001),  # 0.5 + 1 ulp
    1.0,
    _from_bits(0x3F7FFFFF),  # 1.0 - 1 ulp
    _from_bits(0x3F800001),  # 1.0 + 1 ulp
    -1.0,
    -2.5,
    _from_bits(0xC01FFFFF),  # -2.5 + 1 ulp
    _from_bits(0xC0200001),  # -2.5 - 1 ulp
    _from_bits(
        0x3DCCCCCD
    ),  # 0.1f: low mantissa bits set, so a decode that kept only the upper half fails
    _from_bits(0x3DCCCCCC),  # 0.1f - 1 ulp
    _from_bits(0x3DCCCCCE),  # 0.1f + 1 ulp
    0.099609375,  # 0.1f truncated to bf16 (0x3DCC0000)
    3.0,
    1.0e9,
    1.1e9,  # straddle 1056964608.0, the value 0x3F000000 used to decode to
    _from_bits(0x7F7FFFFF),  # FLT_MAX
    _from_bits(0xFF7FFFFF),  # -FLT_MAX
    float("inf"),
    float("-inf"),
)

THRESHOLDS = [0.5, 1.0, -2.5, _from_bits(0x3DCCCCCD), float("inf"), float("-inf")]

_TORCH_COMPARE = {
    MathOperation.UnaryGt: torch.gt,
    MathOperation.UnaryLt: torch.lt,
    MathOperation.UnaryGe: torch.ge,
    MathOperation.UnaryLe: torch.le,
    MathOperation.UnaryEq: torch.eq,
    MathOperation.UnaryNe: torch.ne,
}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


INPUT_BITS = tuple(_bits(x) for x in INPUTS)


class _Threshold:
    """A threshold given by its exact fp32 bits. A Python float cannot carry every pattern: the
    double round trip quiets a signalling NaN (0x7F800001 becomes 0x7FC00001)."""

    def __init__(self, bits: int, name: str):
        self.bits = bits
        self.name = name  # the pytest id

    def __repr__(self) -> str:
        return self.name


def _tensor_from_bits(bits) -> torch.Tensor:
    return torch.from_numpy(np.array(bits, dtype=np.uint32).view(np.float32))


def _run_threshold_compare(formats, mathop, threshold_bits, dest_sync, input_bits):
    reps = ELEMENTS_PER_TILE // len(input_bits) + 1
    lane_bits = (list(input_bits) * reps)[:ELEMENTS_PER_TILE]
    src_A = _tensor_from_bits(lane_bits)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)
    threshold = _from_bits(threshold_bits)

    configuration = TestConfig(
        "sources/sfpu_comp_unary_threshold_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop),
            SFPU_UNARY_SCALAR(threshold_bits),
            VECTOR_MODE(VectorMode.RC),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE],
        dtype=format_dict[formats.output_format],
    ).to(torch.float32)

    golden = _TORCH_COMPARE[mathop](
        src_A, torch.tensor(threshold, dtype=torch.float32)
    ).to(torch.float32)

    # Keyed on the lane's bits: NaN lanes dedupe (nan != nan as a float) and +0.0 / -0.0 stay apart.
    wrong = sorted(
        {
            (lane_bits[i], res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name}(x, {threshold:g}) with value=0x{threshold_bits:08X} disagrees with torch on "
        f"{len(wrong)} inputs:\n"
        + "\n".join(
            f"  x = {_from_bits(b):.9g} (0x{b:08X}): got {got}, expected {want}"
            for b, got, want in wrong
        )
    )


_ORDERED_COMPARES = (
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
)


def _wormhole_inf_compare_defect(mathop, threshold) -> bool:
    """The four decode cells Wormhole gets wrong for a reason outside the threshold decode: its
    unary_gt/lt/ge/le compare an infinite threshold against the infinite datum wrong (x = inf lane
    only: gt and lt return 1, ge and le return 0). Blackhole passes the same cells."""
    return (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and mathop in _ORDERED_COMPARES
        and threshold == float("inf")
    )


_WORMHOLE_INF_COMPARE_REASON = (
    "Wormhole unary_gt/lt/ge/le compare an infinite threshold against the infinite datum wrong "
    "(x = inf: gt/lt return 1, ge/le return 0), a defect of those compares, not of the threshold "
    "decode this test checks. Non-strict so a fix surfaces as XPASS."
)


@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=list(_TORCH_COMPARE),
    threshold=THRESHOLDS,
    dest_sync=[DestSync.Half],
)
def test_sfpu_comp_unary_threshold(request, formats, mathop, threshold, dest_sync):
    if _wormhole_inf_compare_defect(mathop, threshold):
        request.node.add_marker(
            pytest.mark.xfail(reason=_WORMHOLE_INF_COMPARE_REASON, strict=False)
        )
    _run_threshold_compare(formats, mathop, _bits(threshold), dest_sync, INPUT_BITS)


# IEEE equality: -0.0 == +0.0, and NaN != NaN even for identical bits. Every NaN threshold is also a
# datum (torch.eq -> 0, ne -> 1 on the identical-bits lane), and the nonzero thresholds are the
# decode test's, so a NaN datum also meets each of them.
_NAN_THRESHOLDS = (
    _Threshold(0x7FC00000, "nan"),
    _Threshold(0xFFC00000, "-nan"),
    _Threshold(0x7F800001, "snan"),  # signalling: not representable as a Python float
)
# fp32 denormals, as datums and as thresholds, with the smallest normals as the control. A denormal
# datum reaches the compare as a denormal (measured on both architectures through this kernel's
# unpack-to-dest fp32 path): eq(denorm, 0.0) is 0 and eq(denorm, the same denorm) is 1, exactly
# torch's answer, so the golden applies no flush. A flush anywhere on the path would fail the zero
# and denormal threshold cells here.
_DENORMAL_THRESHOLDS = (
    _Threshold(0x00000001, "denorm_min"),
    _Threshold(0x807FFFFF, "-denorm_max"),
)
_DENORMAL_BITS = (0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF)
_MIN_NORMAL_BITS = (0x00800000, 0x80800000)
IEEE_EQ_NE_INPUT_BITS = (
    INPUT_BITS
    + _DENORMAL_BITS
    + _MIN_NORMAL_BITS
    + tuple(t.bits for t in _NAN_THRESHOLDS)
)
IEEE_EQ_NE_THRESHOLDS = (
    [_Threshold(_bits(t), str(t)) for t in THRESHOLDS]
    + [_Threshold(0x00000000, "0.0"), _Threshold(0x80000000, "-0.0")]
    + list(_DENORMAL_THRESHOLDS)
    + list(_NAN_THRESHOLDS)
)


@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=[MathOperation.UnaryEq, MathOperation.UnaryNe],
    threshold=IEEE_EQ_NE_THRESHOLDS,
    dest_sync=[DestSync.Half],
)
def test_sfpu_comp_unary_eq_ne_ieee(formats, mathop, threshold, dest_sync):
    _run_threshold_compare(
        formats, mathop, threshold.bits, dest_sync, IEEE_EQ_NE_INPUT_BITS
    )
