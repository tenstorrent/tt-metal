# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lane FD blaze-registration vehicle (correctness + device-profile perf).

Races the vendored tt-blaze SFPU kernels (helpers/include/blaze_vendored/,
byte-exact from tt-blaze nkapre/sfpi @ 69b8782e2, manifest VENDORED.md there)
through sources/sfpu_blaze_test.cpp:

* ops 1-11 are blaze's TYPED already-semantic kernels — sem == orig source, so
  their sweep rows are causal-only (OFF vs ON of the review flag set);
* rope (op 12) and sdpa_reduce_row (op 13) are blaze RAW-TTI hand kernels
  raced against lane EW's typed semantic lifts via the BLAZE_IMPL axis
  (0 = byte-exact original, 2 = lift), i.e. genuine full2x2 rows.

Goldens are torch models of the kernels' documented math (SEMANTIC-LIFT.md is
the spec for the lifted pairs).  Cross-lane / cross-row structured goldens
(rope, sdpa_reduce_row, zero_pad) are computed through an explicit model of
the SFPU DEST addressing (`_vec_flat_indices`): address A covers face A//16,
face rows 4*((A%16)//4) + lane//8, column parity (A>>1)&1 with column
2*(lane%8)+parity — the reading the in-tree generic-moe-gate test pinned on
Blackhole silicon.  All rows are CORRECTNESS-UNVERIFIED-ON-SILICON until the
first weekly books them; the mapping is part of what that first run verifies.

Stimuli ride the raw (untilized) coverage-vehicle path: the flat stimuli
vector is the face-major tile, and the goldens transform the same flat frame.
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    format_dict,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BLAZE_IMPL,
    BLAZE_OP,
    BLAZE_PARAMS,
    BLAZE_SUBOP,
    TILE_COUNT,
)
from helpers.utils import passed_test


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


# Fixed op parameters — must match sources/sfpu_blaze_test.cpp.
_CSILU_LIMIT = 2.0
_CSILU_ALPHA = 1.702  # the GPT-OSS SwiGLU alpha
_SITU_BETA = 8.0
_SOFTCAP_CAP = 30.0  # Gemma-style final-logit cap
_SILU_SCALE = 0.5
_ADD_RSQRT_EPS = 0.5
_SKF_BANK_MASK = 0x3F
_SKF_MY_BANK = 5
_SKF_GLOBAL_BANK_SHIFT = 10
_SKF_WITHIN_BANK_MASK = 0x3FF
_SKF_OUT_SHIFT = 0
_ZERO_PAD_VALID_ROWS = 24

_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_INT32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

_INT_SPEC = StimuliSpec.uniform(low=0.0, high=float(2**16 - 1))

_IMPL_IDS = {
    0: "blaze_original",
    2: "semantic_lift",
    3: "semantic_lift_walk",
    4: "crosslane",
    5: "uniform",
    6: "uniform_walk8",
    7: "uniform_half",
    8: "uniform_seq",
}


# --------------------------------------------------------------------------
# SFPU DEST address model (silicon-pinned reading, see module docstring).
# --------------------------------------------------------------------------


def _vec_flat_indices(addr: int) -> torch.Tensor:
    """Flat (face-major) indices of the 32 lanes of the vector at DEST addr."""
    tile = addr // 64
    a = addr % 64
    face = a // 16
    band = ((a % 16) // 4) * 4
    parity = (a >> 1) & 1
    lanes = torch.arange(32)
    row = band + lanes // 8
    col = 2 * (lanes % 8) + parity
    return tile * 1024 + face * 256 + row * 16 + col


def _vec_load(flat: torch.Tensor, addr: int) -> torch.Tensor:
    return flat[_vec_flat_indices(addr)].clone()


def _vec_store(flat: torch.Tensor, addr: int, value: torch.Tensor) -> None:
    flat[_vec_flat_indices(addr)] = value


def _shr1(v: torch.Tensor) -> torch.Tensor:
    """SFPSHFT2 SUBVEC_SHFLSHR1: out[l] = in[l-1] for l%8 != 0, else 0."""
    out = torch.zeros_like(v)
    lanes = torch.arange(32)
    src = lanes - 1
    keep = (lanes % 8) != 0
    out[keep] = v[src[keep]]
    return out


def _ror1(v: torch.Tensor) -> torch.Tensor:
    """SFPSHFT2 SUBVEC_SHFLROR1: out[l] = in[l-1] for l%8 != 0, else in[l+7]."""
    lanes = torch.arange(32)
    src = torch.where((lanes % 8) != 0, lanes - 1, lanes + 7)
    return v[src].clone()


# --------------------------------------------------------------------------
# Shared runner
# --------------------------------------------------------------------------


def _run_blaze(
    op,
    golden_fn,
    formats=_BF16,
    dest_acc=DestAccumulation.No,
    impl=0,
    subop=0,
    param0_bits=0x3F800000,
    param1_bits=0x3F800000,
    spec_A=None,
    spec_B=None,
    src_B_override=None,
    atol=None,
    rtol=None,
    tile_cnt=1,
):
    """Drive one blaze-op variant and gate it against golden_fn(src_A, src_B).

    tile_cnt > 1 (lane FE multi-tile rows) drives the vehicle's runtime
    TILE_CNT loop: tile_cnt input tiles in buffer_A, the golden applied
    per tile (every golden here models exactly one tile's transform; for
    rope the single cos/sin tile in buffer_B is shared by every x tile).
    """
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32 * tile_cnt],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=spec_A,
        spec_B=spec_B,
    )
    assert tile_cnt_A == tile_cnt
    if src_B_override is not None:
        src_B = src_B_override.to(src_B.dtype).reshape(src_B.shape)

    src_A_flat = src_A.flatten()
    golden = torch.cat(
        [
            golden_fn(src_A_flat[t * 1024 : (t + 1) * 1024], src_B.flatten())
            for t in range(tile_cnt)
        ]
    )

    templates = [
        BLAZE_OP(op),
        BLAZE_SUBOP(subop),
        BLAZE_IMPL(impl),
        BLAZE_PARAMS(param0_bits=param0_bits, param1_bits=param1_bits),
        APPROX_MODE(ApproximationMode.No),
    ]

    configuration = TestConfig(
        "sources/sfpu_blaze_test.cpp",
        formats,
        templates=templates,
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: 1024 * tile_cnt]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = golden.to(torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=atol,
        custom_rtol=rtol,
    ), "Assert against golden failed"


# --------------------------------------------------------------------------
# Typed already-semantic blaze kernels (sem == orig; causal-only rows)
# --------------------------------------------------------------------------


def test_sfpu_blaze_clampedsilu_gate():
    """blaze clamped_silu gate: min(x, limit) * sigmoid(alpha * min(x, limit))."""

    def golden(a, _b):
        x = torch.clamp(a.to(torch.float32), max=_CSILU_LIMIT)
        return x * torch.sigmoid(_CSILU_ALPHA * x)

    _run_blaze(
        1,
        golden,
        param0_bits=_bits(_CSILU_LIMIT),
        param1_bits=_bits(_CSILU_ALPHA),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.02,
        rtol=0.05,
    )


def test_sfpu_blaze_clampedsilu_up():
    """blaze clamped_silu up: clamp(x, -limit, limit) + 1."""

    def golden(a, _b):
        return torch.clamp(a.to(torch.float32), -_CSILU_LIMIT, _CSILU_LIMIT) + 1.0

    _run_blaze(
        2,
        golden,
        param0_bits=_bits(_CSILU_LIMIT),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
    )


def test_sfpu_blaze_clampedsilu_clamped():
    """blaze clamped_silu clamp-only: clamp(x, -limit, limit).  Exact gate."""

    def golden(a, _b):
        return torch.clamp(a.to(torch.float32), -_CSILU_LIMIT, _CSILU_LIMIT)

    _run_blaze(
        3,
        golden,
        param0_bits=_bits(_CSILU_LIMIT),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.0,
        rtol=0.0,
    )


def test_sfpu_blaze_situ_gate():
    """blaze SiTU gate (Kimi K3): beta * tanh(x / beta) * sigmoid(x)."""

    def golden(a, _b):
        x = a.to(torch.float32)
        return _SITU_BETA * torch.tanh(x / _SITU_BETA) * torch.sigmoid(x)

    _run_blaze(
        4,
        golden,
        param0_bits=_bits(_SITU_BETA),
        param1_bits=_bits(1.0 / _SITU_BETA),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.03,
        rtol=0.05,
    )


def test_sfpu_blaze_scaledtanh():
    """blaze SiTU up transform: beta * tanh(x / beta)."""

    def golden(a, _b):
        x = a.to(torch.float32)
        return _SITU_BETA * torch.tanh(x / _SITU_BETA)

    _run_blaze(
        5,
        golden,
        param0_bits=_bits(_SITU_BETA),
        param1_bits=_bits(1.0 / _SITU_BETA),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.03,
        rtol=0.05,
    )


def test_sfpu_blaze_logitsoftcap():
    """blaze logit softcap (Gemma), via the MATH-gate twin: cap * tanh(x).

    The blaze original is #ifdef TRISC_PACK-only; the vendored semantic twin
    (blaze semantic/logit_softcap.hpp) is its byte-equivalent body under a
    MATH||PACK gate — gate-blocked, not body-blocked (lane EW proof)."""

    def golden(a, _b):
        return _SOFTCAP_CAP * torch.tanh(a.to(torch.float32))

    _run_blaze(
        6,
        golden,
        param0_bits=_bits(_SOFTCAP_CAP),
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.25,  # cap=30 scales the bf16 quantum; ~30*2^-8 + tanh residual
        rtol=0.05,
    )


def test_sfpu_blaze_siluscaled():
    """blaze silu_scaled with tail scale: s * silu(s * x)."""

    def golden(a, _b):
        x = _SILU_SCALE * a.to(torch.float32)
        return _SILU_SCALE * (x * torch.sigmoid(x))

    _run_blaze(
        7,
        golden,
        param0_bits=_bits(_SILU_SCALE),
        param1_bits=0,
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        atol=0.02,
        rtol=0.05,
    )


def test_sfpu_blaze_sparsekfilter():
    """blaze sparse_k_filter original (the tt-metal port is raced by lane EU):
    y = bank-hit ? (within-bank slot + 1) << shift : 0.  Exact integer gate."""

    def golden(a, _b):
        x = a.to(torch.int64)
        bank = (x >> _SKF_GLOBAL_BANK_SHIFT) & _SKF_BANK_MASK
        local = x & _SKF_WITHIN_BANK_MASK
        y = torch.where(
            bank == _SKF_MY_BANK, (local + 1) << _SKF_OUT_SHIFT, torch.zeros_like(x)
        )
        return y.to(torch.int32)

    _run_blaze(
        8,
        golden,
        formats=_INT32,
        dest_acc=DestAccumulation.Yes,
        spec_A=_INT_SPEC,
        atol=0.0,
        rtol=0.0,
    )


def test_sfpu_blaze_zeropad():
    """blaze zero_pad original: scrub SFPU rows [24, 32) to +0.0.  Exact gate.

    SFPU rows 24-31 = DEST addresses 48-62 = face 3 of the tile (the address
    model above), i.e. the last 256 cells of the face-major flat vector."""

    def golden(a, _b):
        out = a.clone()
        out[_ZERO_PAD_VALID_ROWS * 32 :] = 0.0
        return out

    _run_blaze(9, golden, atol=0.0, rtol=0.0)


def test_sfpu_blaze_addrsqrt():
    """blaze add_rsqrt (RMSNorm idiom): 1/sqrt(x + eps)."""

    def golden(a, _b):
        return torch.rsqrt(a.to(torch.float32) + _ADD_RSQRT_EPS)

    _run_blaze(
        10,
        golden,
        param0_bits=_bits(_ADD_RSQRT_EPS),
        spec_A=StimuliSpec.uniform(low=0.05, high=6.0),
        atol=0.05,
        rtol=0.05,
    )


def test_sfpu_blaze_sdpaexp():
    """blaze sdpa exp (accurate upper unclamped), driven per-row by a typed
    wrapper loop: exp(x) on the sdpa domain x <= 0."""

    def golden(a, _b):
        return torch.exp(a.to(torch.float32))

    _run_blaze(
        11,
        golden,
        spec_A=StimuliSpec.uniform(low=-8.0, high=0.0),
    )


# --------------------------------------------------------------------------
# RAW-TTI blaze kernels vs lane-EW typed lifts (full2x2 rows)
# --------------------------------------------------------------------------


def _rope_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """rope <Ht=1, Wt=1, x_base=0, x_stride=64, cos_base=64, sin_base=96>.

    x = Dst tile 0 (buffer_A); cos = tile 1 faces 0-1, sin = tile 1 faces 2-3
    (buffer_B top/bottom halves).  Per face f in {0, 1}: the single vector
    pair at x addresses (16f, 16f+2) is rotated by the even-parity cos/sin
    vectors at addresses (64+16f, 96+16f):
        x'_even = cos*x_even - sin*x_odd ;  x'_odd = sin*x_even + cos*x_odd
    Only face rows 0-3 of faces 0/1 are touched ([1,32] tiny-tile contract).
    """
    flat = torch.cat([a.to(torch.float32), b.to(torch.float32)])
    out = flat.clone()
    for f in (0, 1):
        xe = _vec_load(flat, 16 * f)
        xo = _vec_load(flat, 16 * f + 2)
        ce = _vec_load(flat, 64 + 16 * f)
        se = _vec_load(flat, 96 + 16 * f)
        _vec_store(out, 16 * f, _bf16(ce * xe - se * xo))
        _vec_store(out, 16 * f + 2, _bf16(se * xe + ce * xo))
    return out[:1024]


def _sdpa_reduce_row_golden(a: torch.Tensor, pool: str) -> torch.Tensor:
    """sdpa_reduce_row <block_width=4, skip_signalling>, src=dst=tile 0.

    Simulates the kernel's exact accumulation order and lane folds (the lift
    body is the executable spec; the original is instruction-identical per
    lane EW's argument):
      accA over vector rows {0,1,4,5} of each 8-row block, accB over
      {2,3,6,7}; 4 blocks; epilogue = SHFLSHR1 fold tree (4x/2x/1x) + final
      SHFLROR1; results stored to Dst addresses 0 and 4.
    """
    op = (lambda x, y: torch.maximum(x, y)) if pool == "max" else (lambda x, y: x + y)
    flat = a.to(torch.float32).clone()
    order = [0, 2, 1, 3, 4, 6, 5, 7]  # (A,B) interleave: A0,B0,A1,B1,...

    accA = accB = None
    for blk in range(4):
        base = 16 * blk
        vs = {k: _vec_load(flat, base + 2 * k) for k in range(8)}
        if blk == 0:
            accA, accB = vs[0], vs[2]
        else:
            accA, accB = op(accA, vs[0]), op(accB, vs[2])
        for k in order[2:]:
            if k in (1, 4, 5):
                accA = op(accA, vs[k])
            else:
                accB = op(accB, vs[k])

    def fold(acc):
        t = acc
        for _ in range(4):
            t = _shr1(t)
        acc = op(acc, t)
        t = _shr1(_shr1(acc))
        acc = op(acc, t)
        t = _shr1(acc)
        acc = op(acc, t)
        return _ror1(acc)

    out = flat.clone()
    _vec_store(out, 0, _bf16(fold(accA)))
    _vec_store(out, 4, _bf16(fold(accB)))
    return out


@pytest.mark.parametrize("impl", [0, 2], ids=lambda i: _IMPL_IDS[i])
def test_sfpu_blaze_rope(impl):
    """blaze RAW-TTI rope vs lane-EW typed lift (SEMANTIC-LIFT.md lift 1)."""

    def golden(a, b):
        return _rope_golden(a, b)

    # cos/sin tile: interleaved (Meta) layout — each angle duplicated across
    # its pair — cos in faces 0-1 (flat [0, 512)), sin in faces 2-3.
    torch.manual_seed(1)
    theta = torch.rand(256) * 6.28318530718
    cs = torch.zeros(1024)
    cs[:512] = torch.cos(theta).repeat_interleave(2)
    cs[512:] = torch.sin(theta).repeat_interleave(2)
    cs = _bf16(cs)

    _run_blaze(
        12,
        golden,
        impl=impl,
        spec_A=StimuliSpec.uniform(low=-1.0, high=1.0),
        src_B_override=cs,
    )


@pytest.mark.parametrize("impl", [0, 2, 3, 4, 5, 6, 7, 8], ids=lambda i: _IMPL_IDS[i])
@pytest.mark.parametrize("pool", ["max", "sum"], ids=["MAX", "SUM"])
def test_sfpu_blaze_sdpa_reduce_row(pool, impl):
    """blaze RAW-TTI sdpa_reduce_row vs lane-EW typed lift (lift 2).

    Positive stimuli keep the SHFLSHR1 zero-fill lanes deterministic for both
    pool types; the golden simulates the exact fold, so every cell of the
    result rows (including the partial-fold lanes) is checked, and this
    correctness leg doubles as lane S4's missing subvec_shflshr1/ror1
    lane-semantics silicon probe."""

    def golden(a, _b):
        return _sdpa_reduce_row_golden(a, pool)

    _run_blaze(
        13,
        golden,
        impl=impl,
        subop=0 if pool == "max" else 1,
        spec_A=StimuliSpec.uniform(low=0.1, high=2.0),
    )


# --------------------------------------------------------------------------
# Device-profile perf nodes (Lane BK recipe: MATH_ISOLATE, BLAZE_BODY zone)
# --------------------------------------------------------------------------

_PERF_POINTS = {
    "clampedsilu-gate": dict(op=1, p0=_bits(_CSILU_LIMIT), p1=_bits(_CSILU_ALPHA)),
    "clampedsilu-up": dict(op=2, p0=_bits(_CSILU_LIMIT)),
    "clampedsilu-clamped": dict(op=3, p0=_bits(_CSILU_LIMIT)),
    "situ-gate": dict(op=4, p0=_bits(_SITU_BETA), p1=_bits(1.0 / _SITU_BETA)),
    "scaledtanh": dict(op=5, p0=_bits(_SITU_BETA), p1=_bits(1.0 / _SITU_BETA)),
    "logitsoftcap": dict(op=6, p0=_bits(_SOFTCAP_CAP)),
    "siluscaled": dict(op=7, p0=_bits(_SILU_SCALE)),
    "sparsekfilter": dict(op=8, formats=_INT32, dest_acc=DestAccumulation.Yes),
    "zeropad": dict(op=9),
    "addrsqrt": dict(op=10, p0=_bits(_ADD_RSQRT_EPS)),
    "sdpaexp": dict(op=11),
    "rope-orig": dict(op=12, impl=0),
    "rope-lift": dict(op=12, impl=2),
    "sdpareducerow-max-orig": dict(op=13, impl=0),
    "sdpareducerow-max-lift": dict(op=13, impl=2),
    "sdpareducerow-max-walk": dict(op=13, impl=3),
    "sdpareducerow-sum-orig": dict(op=13, subop=1, impl=0),
    "sdpareducerow-sum-lift": dict(op=13, subop=1, impl=2),
    "sdpareducerow-sum-walk": dict(op=13, subop=1, impl=3),
    "sdpareducerow-max-cl": dict(op=13, impl=4),
    "sdpareducerow-sum-cl": dict(op=13, subop=1, impl=4),
    # Lane IE uniform-block twins (seeded accumulators + encodable walk;
    # helpers/include/blaze_twins/sdpa_reduce_row_uniform.hpp).
    "sdpareducerow-max-uni": dict(op=13, impl=5),
    "sdpareducerow-sum-uni": dict(op=13, subop=1, impl=5),
    "sdpareducerow-max-uni8": dict(op=13, impl=6),
    "sdpareducerow-sum-uni8": dict(op=13, subop=1, impl=6),
    "sdpareducerow-max-unih": dict(op=13, impl=7),
    "sdpareducerow-sum-unih": dict(op=13, subop=1, impl=7),
    "sdpareducerow-max-useq": dict(op=13, impl=8),
    "sdpareducerow-sum-useq": dict(op=13, subop=1, impl=8),
}


def _run_blaze_device_profile(perf_report, point, tile_cnt, tag):
    """Shared driver: one on-device MATH-zone sample of the BLAZE_BODY zone.

    The zone sits INSIDE the vehicle's tile loop, so mean(MATH_ISOLATE) is
    the mean per-tile body time at every tile_cnt — directly comparable
    across tile counts (lane FE multi-tile rows)."""
    cfg = _PERF_POINTS[point]
    formats = cfg.get("formats", _BF16)
    dest_acc = cfg.get("dest_acc", DestAccumulation.No)

    torch.manual_seed(0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32 * tile_cnt],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=(
            _INT_SPEC if formats is _INT32 else StimuliSpec.uniform(low=0.1, high=2.0)
        ),
    )

    configuration = PerfConfig(
        "sources/sfpu_blaze_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            BLAZE_OP(cfg["op"]),
            BLAZE_SUBOP(cfg.get("subop", 0)),
            BLAZE_IMPL(cfg.get("impl", 0)),
            BLAZE_PARAMS(
                param0_bits=cfg.get("p0", 0x3F800000),
                param1_bits=cfg.get("p1", 0x3F800000),
            ),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    configuration.run(perf_report, run_count=1)
    frame = perf_report.frame()
    rows = frame[frame["marker"] == "BLAZE_BODY"]
    assert len(rows) == 1, frame.to_string(index=False)
    cycles = float(rows.iloc[0]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"{tag} {point} tiles={tile_cnt} math_cycles_per_tile={int(cycles)}")


@pytest.mark.perf
@pytest.mark.parametrize("point", list(_PERF_POINTS), ids=lambda p: p)
def test_sfpu_blaze_device_profile(perf_report, point):
    """One on-device MATH-zone sample of the BLAZE_BODY zone per op/arm."""
    _run_blaze_device_profile(perf_report, point, 1, "BLAZE_DEVICE_PROFILE")


# --------------------------------------------------------------------------
# Lane FE multi-tile rows: the same op/arm points at tile_count 8 and 32.
# tiles=1 stays anchored by the nodes above; goldens/checks are per tile;
# the BLAZE_BODY diagnostic zone reads out per-tile at every count (mean
# across zone instances), the KERNEL zone gives the end-to-end verdict.
# --------------------------------------------------------------------------

_MT_TILE_COUNTS = [8, 32]


def _mt_rope_cs():
    """The rope test's interleaved cos/sin tile (same construction/seed)."""
    torch.manual_seed(1)
    theta = torch.rand(256) * 6.28318530718
    cs = torch.zeros(1024)
    cs[:512] = torch.cos(theta).repeat_interleave(2)
    cs[512:] = torch.sin(theta).repeat_interleave(2)
    return _bf16(cs)


def _g_csilu_gate(a, _b):
    x = torch.clamp(a.to(torch.float32), max=_CSILU_LIMIT)
    return x * torch.sigmoid(_CSILU_ALPHA * x)


def _g_csilu_up(a, _b):
    return torch.clamp(a.to(torch.float32), -_CSILU_LIMIT, _CSILU_LIMIT) + 1.0


def _g_csilu_clamped(a, _b):
    return torch.clamp(a.to(torch.float32), -_CSILU_LIMIT, _CSILU_LIMIT)


def _g_situ_gate(a, _b):
    x = a.to(torch.float32)
    return _SITU_BETA * torch.tanh(x / _SITU_BETA) * torch.sigmoid(x)


def _g_scaledtanh(a, _b):
    return _SITU_BETA * torch.tanh(a.to(torch.float32) / _SITU_BETA)


def _g_logitsoftcap(a, _b):
    return _SOFTCAP_CAP * torch.tanh(a.to(torch.float32))


def _g_siluscaled(a, _b):
    x = _SILU_SCALE * a.to(torch.float32)
    return _SILU_SCALE * (x * torch.sigmoid(x))


def _g_sparsekfilter(a, _b):
    x = a.to(torch.int64)
    bank = (x >> _SKF_GLOBAL_BANK_SHIFT) & _SKF_BANK_MASK
    local = x & _SKF_WITHIN_BANK_MASK
    y = torch.where(
        bank == _SKF_MY_BANK, (local + 1) << _SKF_OUT_SHIFT, torch.zeros_like(x)
    )
    return y.to(torch.int32)


def _g_zeropad(a, _b):
    out = a.clone()
    out[_ZERO_PAD_VALID_ROWS * 32 :] = 0.0
    return out


def _g_addrsqrt(a, _b):
    return torch.rsqrt(a.to(torch.float32) + _ADD_RSQRT_EPS)


def _g_sdpaexp(a, _b):
    return torch.exp(a.to(torch.float32))


_U44 = dict(spec_A=StimuliSpec.uniform(low=-4.0, high=4.0))

# Per-point correctness wiring for the multi-tile nodes: golden + stimuli +
# tolerances, mirroring each single-tile test above one-for-one (op / impl /
# subop / param bits come from _PERF_POINTS).
_MT_CORR = {
    "clampedsilu-gate": dict(golden=_g_csilu_gate, atol=0.02, rtol=0.05, **_U44),
    "clampedsilu-up": dict(golden=_g_csilu_up, **_U44),
    "clampedsilu-clamped": dict(golden=_g_csilu_clamped, atol=0.0, rtol=0.0, **_U44),
    "situ-gate": dict(golden=_g_situ_gate, atol=0.03, rtol=0.05, **_U44),
    "scaledtanh": dict(golden=_g_scaledtanh, atol=0.03, rtol=0.05, **_U44),
    "logitsoftcap": dict(golden=_g_logitsoftcap, atol=0.25, rtol=0.05, **_U44),
    "siluscaled": dict(golden=_g_siluscaled, atol=0.02, rtol=0.05, **_U44),
    "sparsekfilter": dict(golden=_g_sparsekfilter, atol=0.0, rtol=0.0),
    "zeropad": dict(golden=_g_zeropad, atol=0.0, rtol=0.0),
    "addrsqrt": dict(
        golden=_g_addrsqrt,
        atol=0.05,
        rtol=0.05,
        spec_A=StimuliSpec.uniform(low=0.05, high=6.0),
    ),
    "sdpaexp": dict(golden=_g_sdpaexp, spec_A=StimuliSpec.uniform(low=-8.0, high=0.0)),
    "rope-orig": dict(
        golden=_rope_golden, spec_A=StimuliSpec.uniform(low=-1.0, high=1.0), rope=True
    ),
    "rope-lift": dict(
        golden=_rope_golden, spec_A=StimuliSpec.uniform(low=-1.0, high=1.0), rope=True
    ),
    "sdpareducerow-max-orig": dict(pool="max"),
    "sdpareducerow-max-lift": dict(pool="max"),
    "sdpareducerow-max-walk": dict(pool="max"),
    "sdpareducerow-sum-orig": dict(pool="sum"),
    "sdpareducerow-sum-lift": dict(pool="sum"),
    "sdpareducerow-sum-walk": dict(pool="sum"),
    "sdpareducerow-max-cl": dict(pool="max"),
    "sdpareducerow-sum-cl": dict(pool="sum"),
    "sdpareducerow-max-uni": dict(pool="max"),
    "sdpareducerow-sum-uni": dict(pool="sum"),
    "sdpareducerow-max-uni8": dict(pool="max"),
    "sdpareducerow-sum-uni8": dict(pool="sum"),
    "sdpareducerow-max-unih": dict(pool="max"),
    "sdpareducerow-sum-unih": dict(pool="sum"),
    "sdpareducerow-max-useq": dict(pool="max"),
    "sdpareducerow-sum-useq": dict(pool="sum"),
}


@pytest.mark.parametrize("tile_cnt", _MT_TILE_COUNTS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("point", list(_MT_CORR), ids=lambda p: p)
def test_sfpu_blaze_multitile(point, tile_cnt):
    """Multi-tile correctness for every raced op/arm: goldens per tile."""
    cfg = _PERF_POINTS[point]
    spec = _MT_CORR[point]

    kwargs = dict(
        impl=cfg.get("impl", 0),
        subop=cfg.get("subop", 0),
        tile_cnt=tile_cnt,
    )
    if "formats" in cfg:
        kwargs["formats"] = cfg["formats"]
    if "dest_acc" in cfg:
        kwargs["dest_acc"] = cfg["dest_acc"]
    if "p0" in cfg:
        kwargs["param0_bits"] = cfg["p0"]
    if "p1" in cfg:
        kwargs["param1_bits"] = cfg["p1"]
    if point == "siluscaled":
        kwargs["param1_bits"] = 0  # the corr contract (perf's default 1.0f is inert)
    if "spec_A" in spec:
        kwargs["spec_A"] = spec["spec_A"]
    for k in ("atol", "rtol"):
        if k in spec:
            kwargs[k] = spec[k]
    if point == "sparsekfilter":
        kwargs["spec_A"] = _INT_SPEC

    if spec.get("rope"):
        kwargs["src_B_override"] = _mt_rope_cs()
        golden = lambda a, b: _rope_golden(a, b)  # noqa: E731
    elif "pool" in spec:
        pool = spec["pool"]
        kwargs["spec_A"] = StimuliSpec.uniform(low=0.1, high=2.0)
        golden = lambda a, _b: _sdpa_reduce_row_golden(a, pool)  # noqa: E731
    else:
        golden = spec["golden"]

    _run_blaze(cfg["op"], golden, **kwargs)


@pytest.mark.perf
@pytest.mark.parametrize("tile_cnt", _MT_TILE_COUNTS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("point", list(_PERF_POINTS), ids=lambda p: p)
def test_sfpu_blaze_device_profile_mt(perf_report, point, tile_cnt):
    """Multi-tile MATH-zone sample per op/arm (per-tile BLAZE_BODY readout)."""
    _run_blaze_device_profile(perf_report, point, tile_cnt, "BLAZE_DEVICE_PROFILE_MT")
