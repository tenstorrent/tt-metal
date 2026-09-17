# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
add_rsqrt SFPU functor test (Blackhole only).

Covers the single entry point of
hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h,
promoted out of the deepseek_v3_b1 demo tree by tt-metal #52709:

    calculate_add_rsqrt<APPROX, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>(addend)
        ->  dst = rsqrt(dst + addend)

The fused form exists for RMSNorm's rsqrt(variance + epsilon): the add happens inside
the SFPU slot, so the variance never round-trips through DEST at the dest width.

Three template axes, all reachable from the compute API (add_rsqrt_tile):

  APPROX        LUT-only SQRT_10-bits body vs the SQRT_23-bits Newton refinement.
                Swept in the main test, with a looser tolerance for the approx body.
  fp32_dest_acc When false the functor does its own convert<vFloat16b>(Nearest) before
                the store, so the golden must round to the dest width to match. Driven
                by the dest_acc axis.
  FAST_APPROX   Drops the trailing `v_if(x < 0) -> NaN` guard in _calculate_sqrt_body_.
                That guard is the ONLY thing the flag changes, so it is unobservable on
                a non-negative domain -- test_sfpu_add_rsqrt_fast_approx_negative_guard
                is the case that actually separates the two settings.

Domain. rsqrt is only defined for x + addend > 0, so the main sweep stays strictly
positive; the negative and zero arguments are their own tests with their own exact
expectations (NaN and +inf respectively) rather than tolerance comparisons, since
neither is a value passed_test can meaningfully bound.
"""

import struct

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import round_to_dest_width
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_FAST_APPROX,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
)
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Same-in-same-out: the functor never touches srcA, so a mixed pair would only re-test
# unpack/pack. Float32 with dest_acc=No is skipped as elsewhere in the SFPU suites.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

# 0.0        -> reduces to plain rsqrt, cross-checking against MathOperation.Rsqrt
#               in test_sfpu_unary.py
# 1e-6, 1e-5 -> the production RMSNorm epsilon range
# 1.0        -> an addend large enough to dominate the input, so a dropped or
#               mis-decoded addend cannot pass by being numerically small
ADDENDS = (0.0, 1e-6, 1e-5, 1.0)

# Tolerances. The shared default for every float format is atol=rtol=0.05, which is far
# too loose to notice a wrong rsqrt -- the whole positive sweep lands in [0.44, 3.2], so
# 5% would accept a result off by an entire bf16 ULP many times over. These are measured
# envelopes instead: the max relative error observed on BH p100a over the sweep below
# (uniform(0.1, 4.0), all four addends, both dest_acc settings), with rtol set ~2.5x
# above it so seed changes and the odd outlier lane stay inside.
#
#   body            output    measured max_rel   rtol used   vs 0.05 default
#   SQRT_23-bits    Float32         1.3e-7         1e-6        50000x tighter
#   SQRT_23-bits    Float16_b       3.9e-3         1.0e-2          5x tighter
#   SQRT_10-bits    Float32         8.8e-4         3e-3           17x tighter
#   SQRT_10-bits    Float16_b       7.6e-3         2.0e-2        2.5x tighter
#
# The two Float16_b rows are dominated by one bf16 ULP (2^-8 = 3.9e-3), not by the SFPU:
# with dest_acc=No the functor's own convert<vFloat16b>(Nearest) matches the golden's
# rounding exactly (measured 0.0 error), while with dest_acc=Yes the packer's fp32->bf16
# conversion and torch's differ by up to one step. Both are folded into one bf16 number
# rather than split by dest_acc, since a 1-ULP packer disagreement is not a defect.
#
# atol is 0 so the rtol column above is what actually gates each cell. passed_test feeds
# both into torch.isclose, which bounds |device - golden| by atol + rtol*|golden|, so any
# atol floor competes with the relative term rather than backing it up. The old 1e-3 floor
# made the tightest cell meaningless: at (ApproximationMode.No, Float32) the golden spans
# [0.45, 3.2], so rtol*|golden| tops out near 3.2e-6 -- three orders under the floor, which
# would therefore have accepted an error ~7700x larger than the 1.3e-7 actually measured.
# That is the only cell exercising the SQRT_23-bit Newton refinement at full fp32 output
# width, and the separate pcc > 0.99 gate is blind to per-element error at this scale.
#
# Dropping the floor is safe because the golden is bounded away from zero everywhere in
# this domain: the sweep is uniform(0.1, 4.0) with addends up to 1.0, so
# rsqrt(x + addend) stays inside [1/sqrt(5.0), 1/sqrt(0.1)] = [0.447, 3.163]. There are no
# near-zero results for an absolute floor to protect. (The zero and negative argument
# cases, whose results are +inf and NaN, are separate tests using exact predicates.)
_ATOL = 0.0
_RTOL = {
    # (approx, output is 32-bit)
    (ApproximationMode.No, True): 1e-6,
    (ApproximationMode.No, False): 1.0e-2,
    (ApproximationMode.Yes, True): 3e-3,
    (ApproximationMode.Yes, False): 2.0e-2,
}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _skip_unsupported(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")


def _build_add_rsqrt(
    formats,
    dest_acc,
    addend,
    approx=ApproximationMode.No,
    fast_approx=False,
    spec_A=None,
):
    """Build one variant without running it, returning (configuration, src_A).

    Split out of _run_add_rsqrt so a test that needs two variants can prepare()
    both before running either -- see test_sfpu_add_rsqrt_fast_approx_negative_guard.
    """
    torch.manual_seed(0)
    spec_a = StimuliSpec.uniform(low=0.1, high=4.0) if spec_A is None else spec_A

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=spec_a,
    )

    configuration = TestConfig(
        "sources/sfpu_add_rsqrt_test.cpp",
        formats,
        templates=[
            APPROX_MODE(approx),
            SFPU_FAST_APPROX(fast_approx),
            SFPU_UNARY_SCALAR(_bits(addend)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    return configuration, src_A


def _finish_add_rsqrt(configuration, src_A, formats, dest_acc):
    """Run a prepared variant, returning (device_tensor, input_tensor) as fp32.

    Deliberately returns raw tensors instead of asserting: the negative/zero-argument
    tests need exact NaN/inf predicates rather than a tolerance comparison.
    """
    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    device = torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)
    # The input as the device saw it: quantised to the input format, then to the dest
    # width the datacopy landed it at.
    seen = round_to_dest_width(
        src_A.flatten()[:ELEMENTS_PER_TILE].to(torch.float32), dest_acc
    )
    return device, seen


def _run_add_rsqrt(
    formats,
    dest_acc,
    addend,
    approx=ApproximationMode.No,
    fast_approx=False,
    spec_A=None,
):
    """Compile+run one variant, returning (device_tensor, input_tensor) as fp32."""
    configuration, src_A = _build_add_rsqrt(
        formats, dest_acc, addend, approx=approx, fast_approx=fast_approx, spec_A=spec_A
    )
    return _finish_add_rsqrt(configuration, src_A, formats, dest_acc)


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    addend=list(ADDENDS),
    approx=[ApproximationMode.No, ApproximationMode.Yes],
)
def test_sfpu_add_rsqrt(formats, dest_acc, addend, approx):
    """rsqrt(x + addend) over a strictly positive domain."""
    _skip_unsupported(formats, dest_acc)

    device, seen = _run_add_rsqrt(formats, dest_acc, addend, approx=approx)

    # Golden mirrors the functor: add in fp32, rsqrt, then the functor's own
    # convert<vFloat16b>(Nearest) when the dest is 16-bit.
    golden = round_to_dest_width(torch.rsqrt(seen + addend), dest_acc)

    rtol = _RTOL[(approx, formats.output_format.is_32_bit())]

    assert passed_test(
        golden,
        device,
        formats.output_format,
        custom_rtol=rtol,
        custom_atol=_ATOL,
    ), (
        f"add_rsqrt mismatch (addend={addend}, approx={approx.name}, "
        f"dest_acc={dest_acc.name}, rtol={rtol})"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    approx=[ApproximationMode.No, ApproximationMode.Yes],
)
def test_sfpu_add_rsqrt_fast_approx_negative_guard(formats, dest_acc, approx):
    """The one case that tells FAST_APPROX=true from FAST_APPROX=false.

    _calculate_sqrt_body_ ends with
    ``if constexpr (!FAST_APPROX) { v_if(x < 0.0f) { y = quiet_NaN(); } }``.
    A negative argument is the only way to observe the flag: on a non-negative
    domain the two settings are bit-identical, which is why the main sweep pins
    FAST_APPROX=false and does not sweep it.

    What is asserted, and what is deliberately not
    ---------------------------------------------
    x < 0 is outside rsqrt's domain, and the header states no result for it beyond
    the guard itself, so this pins only the guard's *observable effect* -- it must
    not be read as blessing any particular invalid value. Both assertions below are
    independent of the un-guarded value:

      1. FAST_APPROX=false leaves no negative lane. That follows from the guard
         itself, which replaces the result with quiet_NaN -- and on the bf16-input
         paths that arrives as +inf. Neither is negative. `isnan` specifically is
         NOT asserted: the NaN-vs-+inf split is format-dependent, so an isnan-based
         assertion would fail on Float16_b while the guard is working correctly.
      2. The two builds differ somewhere, which is what makes the flag observable
         at all and the reason this test exists.

    Assertion 2 replaces an earlier `negative_lanes > 0` check on the FAST_APPROX=true
    side. Measured on Blackhole p100a the un-guarded body does leave negative results
    (-inf on most lanes, plus a handful of NaN / finite lanes), but nothing specifies
    that: the sign falls out of `vConstIntPrgm0 - i` going negative for these exponents,
    i.e. from the current LUT seed rather than from an invariant. An accuracy-motivated
    retune of that seed or the Newton constants could make the un-guarded result
    non-negative while the guard still works, failing the old assertion for no real
    defect. Comparing the two builds pins the same intent without depending on a value
    the implementation does not define.
    """
    _skip_unsupported(formats, dest_acc)

    # Strictly negative even after the addend is applied. Both builds seed the same, so
    # the two runs see identical stimuli and can be compared lane by lane.
    variant = dict(
        addend=0.0,
        approx=approx,
        spec_A=StimuliSpec.uniform(low=-4.0, high=-0.1),
    )
    guarded_cfg, guarded_src = _build_add_rsqrt(
        formats, dest_acc, fast_approx=False, **variant
    )
    unguarded_cfg, unguarded_src = _build_add_rsqrt(
        formats, dest_acc, fast_approx=True, **variant
    )
    # Build both before running either: `prepare()` is the build half of `run()`, and
    # under --compile-producer `run()` skips as soon as the first variant is built, so
    # the second would otherwise never emit its ELF.
    guarded_cfg.prepare()
    unguarded_cfg.prepare()
    guarded, _ = _finish_add_rsqrt(guarded_cfg, guarded_src, formats, dest_acc)
    unguarded, _ = _finish_add_rsqrt(unguarded_cfg, unguarded_src, formats, dest_acc)

    negative_lanes = int((guarded < 0).sum())
    assert negative_lanes == 0, (
        "FAST_APPROX=false must replace every negative-input result with the "
        f"guard's invalid value, but {negative_lanes}/{guarded.numel()} lanes "
        "are negative -- the v_if(x < 0) guard looks dropped"
    )

    # NaN != NaN, so exclude lanes where both builds returned NaN before comparing.
    both_nan = guarded.isnan() & unguarded.isnan()
    differs = ~both_nan & (guarded != unguarded)
    assert bool(differs.any()), (
        "FAST_APPROX=true and FAST_APPROX=false returned identical results on a "
        "strictly negative domain, so the v_if(x < 0) guard the flag gates is no "
        "longer observable -- either it was dropped from both builds, or FAST_APPROX "
        "stopped reaching _calculate_sqrt_body_"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    approx=[ApproximationMode.No, ApproximationMode.Yes],
)
def test_sfpu_add_rsqrt_zero_argument(formats, dest_acc, approx):
    """x + addend == 0 -> +inf.

    Both sqrt bodies special-case this off the input bits (`x_bits != 0` in the
    RECIPROCAL branch, otherwise y = infinity_bits), so it is a defined result rather
    than an undefined-domain hole -- worth pinning, since the default positive sweep
    never reaches it and RMSNorm with eps=0 on an all-zero row does.
    """
    _skip_unsupported(formats, dest_acc)

    device, _ = _run_add_rsqrt(
        formats,
        dest_acc,
        addend=0.0,
        approx=approx,
        spec_A=StimuliSpec.constant(value=0.0),
    )

    assert torch.isinf(device).all() and (device > 0).all(), (
        "rsqrt(0) must be +inf on every lane, got "
        f"{device.unique(sorted=True)[:8].tolist()}"
    )
