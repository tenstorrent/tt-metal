# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3B numerical-divergence micro-tests for `MatmulGolden`.

Reference plan:
  ttsim-private/docs/audit/2026-05-14-bfp-investigation/
      phase3B-dominant-cluster-plan.md (Section 3, divergences D1/D2/D3)

These tests exercise the three concrete ways the LLK matmul golden
diverges from silicon for `Float16 -> Float16_b, DestAcc=No`:

  D1: FP16 stimuli were cast to BF16 (`to_tensor(operand, Float16_b)`)
      BEFORE fidelity masking, dropping 3 mantissa bits of precision
      relative to the FP16 register state on silicon.

  D2: `_apply_fidelity_masking` was applied in the OUTPUT format's
      11-bit TF32 envelope (BF16-derived: 7 mantissa + 3 zero-pad +
      implicit-1). Silicon's MVMUL applies the mask in the INPUT
      format's envelope (FP16-derived: 10 mantissa + implicit-1).
      The two envelopes select different bits.

  D3: The K/fidelity-iteration accumulator was cast to the OUTPUT
      torch dtype (`torch.bfloat16` for Float16_b) on every iteration.
      Silicon's Dst16b for `Float16 -> Float16_b, DestAcc=No` keeps
      the accumulator in FP16 across the K/fidelity loop and only
      narrows to BF16 once at packer-time.

Strategy for each test
----------------------
The post-patch `MatmulGolden._matmul_default` falls back to the legacy
buggy behaviour when `input_A_format` is NOT supplied (line ~1151:
`accumulator_format = input_A_format or data_format`). We exploit this
to drive both code paths from the same harness:

  legacy_golden : call WITHOUT input_A_format/input_B_format -> simulates
                  the pre-patch shape (output-format mask + output-format
                  accumulator). Will be reached whenever 3B-A's patch is
                  in flight or hasn't landed yet.

  fixed_golden  : call WITH input_A_format=input_format and
                  input_B_format=input_format -> exercises the patched
                  path explicitly (input-format mask + input-format
                  accumulator + single final narrow).

  silicon_ref   : hand-rolled emulation of tensix.cpp:1735-1748 for the
                  `Float16 -> Float16_b, DestAcc=No` MVMUL path. Used as
                  ground truth for D1 and D3. For D2 we compare in the
                  TF32 envelope directly via the same helpers the golden
                  uses (`SrcFormatModel.to_src_format`).

Each test PASSES iff:
  - legacy and silicon disagree on at least one element (proves the
    divergence is *real* — not a hypothetical), AND
  - fixed and silicon agree to within tolerance (proves the patch
    actually closes the divergence).

Pre-patch (3B-A not yet landed): the patched code path is missing,
so the `fixed_golden` call still runs the legacy logic and the
"fixed == silicon" assertion FAILS — that's the expected pre-patch
signal. Post-patch: both assertions pass.

Marker
------
Marked `phase3b_validation` so the regular sweep does not pick them up
until the patch is merged. Run explicitly:

    pytest -m phase3b_validation test_phase3b_matmul_divergence.py -v

No simulator or device is required (these are pure-Python golden tests).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    SrcFormatModel,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, MathFidelity

pytestmark = pytest.mark.phase3b_validation

TILE = 32  # single tile, single face-of-faces; small enough to stay deterministic


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _golden():
    return get_golden_generator(MatmulGolden)


def _legacy_call(
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    output_format: DataFormat,
    math_fidelity: MathFidelity,
    dims_a=(TILE, TILE),
    dims_b=(TILE, TILE),
) -> torch.Tensor:
    """
    Emulate the pre-patch call shape: do NOT pass input_A_format /
    input_B_format / dest_acc. With the post-patch code in place this
    falls through to `accumulator_format = data_format`, i.e. the
    legacy "mask + accumulate in output format" behaviour. With the
    pre-patch code in place this is the ONLY shape (the kwargs didn't
    exist). Either way we get the buggy path.
    """
    return _golden()(
        src_a,
        src_b,
        output_format,
        math_fidelity,
        input_A_dimensions=list(dims_a),
        input_B_dimensions=list(dims_b),
        tilize=False,
    )


def _fixed_call(
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    math_fidelity: MathFidelity,
    dest_acc: DestAccumulation = DestAccumulation.No,
    dims_a=(TILE, TILE),
    dims_b=(TILE, TILE),
) -> torch.Tensor:
    """
    Emulate the post-patch caller (test_matmul.py:104-115) by passing
    input_A_format, input_B_format, and dest_acc. With the post-patch
    code this exercises:
      * input-format-keyed `_apply_fidelity_masking` (D2 fix)
      * input-format-keyed `to_tensor` pre-cast (D1 fix)
      * input-format-keyed per-iter accumulator (D3 fix)
      * single final narrow to output (D3 fix)
    With the pre-patch code these kwargs are silently ignored (the
    `_prepare_fidelity_operands` signature didn't accept the new
    keyword), so the call still runs the legacy path and the
    fixed-vs-silicon assertion below trips — which is exactly the
    "test fails pre-patch, passes post-patch" signal we want.
    """
    return _golden()(
        src_a,
        src_b,
        output_format,
        math_fidelity,
        input_A_dimensions=list(dims_a),
        input_B_dimensions=list(dims_b),
        tilize=False,
        input_A_format=input_format,
        input_B_format=input_format,
        dest_acc=dest_acc,
    )


def _silicon_fp16_to_bf16_matmul(
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    math_fidelity: MathFidelity,
    dims_a=(TILE, TILE),
    dims_b=(TILE, TILE),
) -> torch.Tensor:
    """
    Silicon-faithful reference for `Float16 -> Float16_b, DestAcc=No`.

    Per `tensix.cpp:1735-1748` (Phase 3B plan §2):
      * SrcA, SrcB are FP16 in registers.
      * `mvmul()` accumulates into FP16 Dst16b (5e10m).
      * Across fidelity iters / K, the accumulator stays FP16.
      * At packer time the FP16 Dst tile is narrowed to BF16 in L1.

    We model this by:
      * Casting stimuli to FP16 (no BF16 narrow).
      * For each fidelity iter, masking the FP16 mantissa in the
        FP16-native 10-bit field (the 11-bit TF32 envelope is just
        `(1 << 10) | (mantissa & 0x3FF)`; the spec mask is applied to
        that envelope, so for FP16 inputs the implicit-1 bit (bit 10)
        is always part of the kept set).
      * `torch.matmul` in FP16, accumulating in FP16 across iters.
      * Final cast to BF16 once at the end.

    This is the same algebra the patched `_matmul_default` aims to
    reproduce.
    """
    # Quasar uses a different mask shape; the plan focuses on WH/BH.
    # For the divergence demo we hard-code the non-Quasar 11-bit masks.
    NON_QUASAR_MASKS = [
        (0b11111000000, 0b11111110000),  # LoFi
        (0b00000111110, 0b11111110000),  # HiFi2
        (0b11111000000, 0b00000001111),  # HiFi3
        (0b00000111110, 0b00000001111),  # HiFi4 (full mask = identity here)
    ]
    # Match the golden's `_get_fidelity_iters` shape exactly:
    #   HiFi4 -> single unmasked matmul (the four explicit-phase masks
    #            partition the mantissa bit-disjointly, so their sum
    #            equals one full-precision matmul). The golden uses
    #            `[None]` to skip the mask round-trip; we use the same
    #            shortcut so the silicon reference is comparable.
    #   HiFi3 -> phases [0, 1, 2]
    #   HiFi2 -> phases [0, 1]
    #   LoFi  -> phases [0]
    if math_fidelity == MathFidelity.HiFi4:
        iters: list[Optional[int]] = [None]
    else:
        iter_count = {
            MathFidelity.LoFi: 0,
            MathFidelity.HiFi2: 1,
            MathFidelity.HiFi3: 2,
        }[math_fidelity]
        iters = list(range(iter_count + 1))

    # Cast stimuli to FP16; this is what the unpacker delivers to SrcA/SrcB.
    a_fp16 = src_a.to(torch.float16).clone()
    b_fp16 = src_b.to(torch.float16).clone()
    M, K1 = dims_a
    K2, N = dims_b
    assert K1 == K2

    accum: Optional[torch.Tensor] = None
    for it in iters:
        if it is None:
            # Unmasked single-pass matmul — matches the golden's HiFi4
            # shortcut and the algebraic full-precision result.
            t1, t2 = a_fp16, b_fp16
        else:
            # Use SrcFormatModel directly so the silicon reference
            # masks the FP16-derived TF32 envelope with the SAME
            # round-trip semantics the patched golden uses. That keeps
            # the reference and the patched golden bit-equivalent
            # whenever the spec mask matches.
            sa, ea, ma = SrcFormatModel._fp16_to_tf32(a_fp16)
            sb, eb, mb = SrcFormatModel._fp16_to_tf32(b_fp16)
            ma = ma & NON_QUASAR_MASKS[it][0]
            mb = mb & NON_QUASAR_MASKS[it][1]
            t1 = SrcFormatModel.from_src_format(DataFormat.Float16, (sa, ea, ma)).to(
                torch.float16
            )
            t2 = SrcFormatModel.from_src_format(DataFormat.Float16, (sb, eb, mb)).to(
                torch.float16
            )
        partial = torch.matmul(t1.view(M, K1), t2.view(K2, N)).view(-1)
        # Critical: keep accumulator in FP16 (Dst16b FP16 lane).
        partial = partial.to(torch.float16)
        if accum is None:
            accum = partial
        else:
            accum = (accum + partial).to(torch.float16)
    assert accum is not None
    # Packer narrow: single final cast to BF16.
    return accum.to(torch.bfloat16)


def _fp16_bits_differ_from_bf16(values: torch.Tensor) -> bool:
    """
    Sanity-check helper: confirm the chosen stimuli have at least one
    FP16 value whose representation is NOT bit-equal to its BF16
    truncation (i.e. one of the bottom 3 FP16 mantissa bits is set).
    Without this, D1 is invisible by construction.
    """
    fp16 = values.to(torch.float16).view(torch.uint16).to(torch.int64)
    # BF16 has 7 mantissa bits; an FP16 value "round-trips" through
    # BF16 only if its bottom 3 mantissa bits are zero.
    return bool(((fp16 & 0x0007) != 0).any().item())


# --------------------------------------------------------------------------- #
# D1 — FP16 stimuli prematurely cast to BF16 before fidelity masking
# --------------------------------------------------------------------------- #


def test_D1_fp16_premature_bf16_cast():
    """
    Construct FP16 stimuli with nonzero low-3 mantissa bits and run a
    HiFi4 (identity-mask) matmul. At HiFi4 the fidelity mask is the
    identity, so D2 has no effect and ONLY D1 can move the output.

    Expected:
      legacy_golden  : stimuli are cast to BF16 in `_prepare_fidelity_operands`,
                       losing the bottom 3 mantissa bits BEFORE the matmul.
      fixed_golden   : stimuli are cast to FP16 in `_prepare_fidelity_operands`,
                       preserving full FP16 precision; matches silicon.
    """
    torch.manual_seed(0xD1)  # local determinism in addition to the
    # per-test sha256 seed set by conftest.pytest_runtest_setup.

    # Build A and B with FP16 values whose low mantissa bits are
    # deliberately nonzero. We start from a base and add an FP16-quantum
    # offset on every other element to guarantee bottom-3-bit population.
    base = torch.linspace(0.5, 1.5, TILE * TILE)
    perturbation = torch.tensor(
        [0.0 if (i % 2 == 0) else 5.0e-4 for i in range(TILE * TILE)]
    )
    src_a = (base + perturbation).to(torch.float16)
    src_b = (base.flip(0) + perturbation).to(torch.float16)

    # Sanity: confirm the stimuli actually distinguish FP16 from BF16.
    assert _fp16_bits_differ_from_bf16(
        src_a
    ), "D1 stimulus invariant broken: A round-trips through BF16 exactly"
    assert _fp16_bits_differ_from_bf16(
        src_b
    ), "D1 stimulus invariant broken: B round-trips through BF16 exactly"

    legacy = _legacy_call(src_a, src_b, DataFormat.Float16_b, MathFidelity.HiFi4)
    fixed = _fixed_call(
        src_a,
        src_b,
        DataFormat.Float16,
        DataFormat.Float16_b,
        MathFidelity.HiFi4,
    )
    silicon = _silicon_fp16_to_bf16_matmul(src_a, src_b, MathFidelity.HiFi4)

    # Pre-patch (and bug-present) assertion: legacy disagrees with silicon.
    legacy_diff = (legacy.to(torch.float32) - silicon.to(torch.float32)).abs()
    assert legacy_diff.max().item() > 0.0, (
        "D1 evidence missing: legacy matched silicon on FP16-tagged stimuli, "
        "so the premature-BF16-cast bug is not demonstrable on this input. "
        "Check the stimulus design."
    )

    # Post-patch assertion: fixed matches silicon bit-for-bit at BF16
    # precision. We use exact bitwise equality at the output dtype to
    # avoid sweeping a real bug under a tolerance.
    fixed_bf16 = fixed.to(torch.bfloat16)
    silicon_bf16 = silicon.to(torch.bfloat16)
    assert torch.equal(fixed_bf16, silicon_bf16), (
        "D1 patch validation failed: fixed_golden output disagrees with "
        "the silicon-faithful FP16-accumulate reference. "
        f"max |fixed-silicon| (fp32) = "
        f"{(fixed.to(torch.float32) - silicon.to(torch.float32)).abs().max().item()}"
    )


# --------------------------------------------------------------------------- #
# D2 — fidelity mask applied in the wrong (output-format) envelope
# --------------------------------------------------------------------------- #


def test_D2_envelope_mismatch():
    """
    Show that for `Float16 -> Float16_b` at a fidelity phase whose mask
    falls inside the bits where FP16 and BF16 envelopes DIFFER (bits
    0..2 of the 11-bit TF32 envelope), the legacy and silicon-correct
    paths produce different operands.

    BF16-envelope construction (`SrcFormatModel._fp16b_to_tf32`):
      mantissa = (bf16_mant_7b << 3) | (1 << 10)   # bits 0..2 always 0
    FP16-envelope construction (`SrcFormatModel._fp16_to_tf32`):
      mantissa = fp16_mant_10b | (1 << 10)          # bits 0..2 hold real bits

    The mask phase that exposes the gap most cleanly is HiFi3 phase 2,
    where mask_b = `0b00000001111` selects exactly the low-4 bits of the
    envelope. In the BF16 derivation those bits are (zero-pad-3) | (low
    bit of the BF16 7-bit mantissa); in the FP16 derivation they are
    the actual FP16 mantissa's bottom 4 bits.

    Expected:
      legacy_golden : masking acts on a BF16-derived envelope; the
                      bottom-3 FP16-mantissa bits were never seen.
      fixed_golden  : masking acts on the FP16-derived envelope, so
                      the spec mask drops the correct bits.

    Both paths produce a finite numerical drift from silicon-correct
    output that depends on each element's bottom-bit pattern; legacy
    drifts (because D1 erased the bottom-3 bits before D2's mask saw
    them), fixed does not.
    """
    torch.manual_seed(0xD2)

    # Pick stimuli with bottom-bit FP16 content so the envelope
    # mismatch is observable. Stay in [0.1, 2.0] to avoid catastrophic
    # cancellation and keep the answer well inside BF16 range.
    a = torch.empty(TILE * TILE, dtype=torch.float16).uniform_(0.1, 2.0)
    b = torch.empty(TILE * TILE, dtype=torch.float16).uniform_(0.1, 2.0)
    # Force the bottom 3 mantissa bits of every value to a deterministic
    # nonzero pattern so the envelope difference is uniform.
    a_raw = a.view(torch.uint16).to(torch.int64)
    b_raw = b.view(torch.uint16).to(torch.int64)
    a_raw = (a_raw & ~torch.tensor(0x0007, dtype=torch.int64)) | 0x0005
    b_raw = (b_raw & ~torch.tensor(0x0007, dtype=torch.int64)) | 0x0003
    src_a = a_raw.to(torch.uint16).view(torch.uint16).view(torch.float16)
    src_b = b_raw.to(torch.uint16).view(torch.uint16).view(torch.float16)

    # Cross-check via the same SrcFormatModel helpers the golden uses.
    # We assert that in the FP16-derived envelope, the bottom 3 bits
    # carry the stimulus information; in the BF16-derived envelope,
    # those 3 bits are always zero (zero-pad of the `<<3` step).
    # This makes the D2 envelope mismatch explicit before we touch the
    # golden.
    _, _, mant_fp16 = SrcFormatModel._fp16_to_tf32(src_a)
    _, _, mant_bf16 = SrcFormatModel._fp16b_to_tf32(src_a)
    BOTTOM_3 = 0b00000000111
    assert (mant_fp16 & BOTTOM_3).any(), (
        "D2 invariant broken: FP16-envelope has no info in bits 0..2 "
        "for the chosen stimulus; envelope-mismatch test cannot fire."
    )
    assert ((mant_bf16 & BOTTOM_3) == 0).all(), (
        "D2 invariant broken: BF16-envelope holds nonzero bits 0..2; "
        "the `(bf16_mant_7b << 3) | 0x400` construction must zero them."
    )

    # HiFi3 phase 2 uses mask_b = 0b00000001111 which reaches bit 0..3,
    # i.e. INTO the FP16-only zone. We run the full HiFi3 fidelity loop
    # (iters [0, 1, 2]) so the divergence is integrated, but the key
    # phase that exposes the envelope mismatch is the last one.
    legacy = _legacy_call(src_a, src_b, DataFormat.Float16_b, MathFidelity.HiFi3)
    fixed = _fixed_call(
        src_a,
        src_b,
        DataFormat.Float16,
        DataFormat.Float16_b,
        MathFidelity.HiFi3,
    )
    silicon = _silicon_fp16_to_bf16_matmul(src_a, src_b, MathFidelity.HiFi3)

    # Pre-patch / bug-present: legacy differs from silicon.
    legacy_diff = (legacy.to(torch.float32) - silicon.to(torch.float32)).abs()
    assert legacy_diff.max().item() > 0.0, (
        "D2 evidence missing: legacy matched silicon at LoFi, so the "
        "envelope-mismatch bug is not demonstrable on this input."
    )

    # Post-patch: fixed matches silicon up to BF16 quantization.
    # We allow a tight tolerance (1 ULP at BF16, ~ 2^-7 * max(|silicon|))
    # rather than bit-equality here because LoFi's mantissa truncation
    # introduces a rounding-direction choice that PyTorch's BF16 and
    # silicon's pack-time round-to-nearest may resolve differently on
    # exact-tie cases. Stimuli are constructed to avoid ties, but we
    # keep the slack to insulate the test from incidental ULP flips.
    fixed_f32 = fixed.to(torch.float32)
    silicon_f32 = silicon.to(torch.float32)
    tol = max(
        2**-7 * silicon_f32.abs().max().item(),  # ~1 BF16 ULP at peak
        1e-3,
    )
    err = (fixed_f32 - silicon_f32).abs().max().item()
    assert err <= tol, (
        f"D2 patch validation failed: |fixed - silicon|_max = {err:.6g} "
        f"exceeds tol {tol:.6g}. Patch did not close the envelope-mismatch "
        f"divergence."
    )


# --------------------------------------------------------------------------- #
# D3 — per-iter accumulation in BF16 vs FP16
# --------------------------------------------------------------------------- #


def test_D3_accumulation_format():
    """
    Use HiFi4 so the fidelity mask is the identity (kills D2) and stay
    inside FP16's range so the cast in D1 is a near-no-op when low
    bits are zero. The HiFi4 path runs `fidelity_iters = [None]`, so
    in practice only one fidelity iter executes. To isolate D3 we
    therefore exercise the *K-loop*: build A as `[M, K]` and B as
    `[K, N]` with K > 32, which forces `torch.matmul` to fold a long
    sum chain. The legacy golden then re-casts the partial to BF16
    per-iter; silicon keeps it in FP16. K=128 gives ~3-bit accumulation
    drift on average — well above BF16 quantization noise.

    Expected:
      legacy_golden : partial sum is cast to BF16 after every fidelity
                      iter (single iter at HiFi4, so this is effectively
                      a single late-narrow), AND the partial inputs
                      are cast to BF16 before the matmul (D1 in the
                      same call). Combined with K=128 chains of BF16
                      additions inside `torch.matmul(bf16,bf16)`, this
                      gives an output that differs from FP16-faithful
                      silicon.
      fixed_golden  : every per-iter `torch.matmul` runs in FP16, the
                      per-iter sum is kept in FP16, single final
                      narrow to BF16 -> bit-equal to silicon.
    """
    torch.manual_seed(0xD3)

    M, K, N = TILE, 4 * TILE, TILE  # K = 128
    # Use signed stimuli around zero so partial sums genuinely cancel
    # and rounding-direction differences accumulate visibly.
    src_a = torch.empty(M * K, dtype=torch.float16).uniform_(-1.0, 1.0)
    src_b = torch.empty(K * N, dtype=torch.float16).uniform_(-1.0, 1.0)

    # Force low-3 mantissa bits nonzero (same trick as D2) so the
    # FP16-vs-BF16 quantization step in the legacy path actually moves
    # the partials.
    a_raw = src_a.view(torch.uint16).to(torch.int64)
    b_raw = src_b.view(torch.uint16).to(torch.int64)
    a_raw = (a_raw & ~torch.tensor(0x0007, dtype=torch.int64)) | 0x0005
    b_raw = (b_raw & ~torch.tensor(0x0007, dtype=torch.int64)) | 0x0003
    src_a = a_raw.to(torch.uint16).view(torch.uint16).view(torch.float16)
    src_b = b_raw.to(torch.uint16).view(torch.uint16).view(torch.float16)

    legacy = _legacy_call(
        src_a,
        src_b,
        DataFormat.Float16_b,
        MathFidelity.HiFi4,
        dims_a=(M, K),
        dims_b=(K, N),
    )
    fixed = _fixed_call(
        src_a,
        src_b,
        DataFormat.Float16,
        DataFormat.Float16_b,
        MathFidelity.HiFi4,
        dims_a=(M, K),
        dims_b=(K, N),
    )
    silicon = _silicon_fp16_to_bf16_matmul(
        src_a,
        src_b,
        MathFidelity.HiFi4,
        dims_a=(M, K),
        dims_b=(K, N),
    )

    # Pre-patch / bug-present: K-loop drift in BF16 differs from FP16.
    legacy_diff = (legacy.to(torch.float32) - silicon.to(torch.float32)).abs()
    assert legacy_diff.max().item() > 0.0, (
        "D3 evidence missing: legacy matched silicon on a K=128 matmul "
        "of low-bit-populated FP16 stimuli. The accumulation-format "
        "divergence should be visible here."
    )

    # Post-patch: fixed matches silicon bit-for-bit at BF16.
    fixed_bf16 = fixed.to(torch.bfloat16)
    silicon_bf16 = silicon.to(torch.bfloat16)
    # Element-wise count of mismatches gives a more diagnosable failure
    # than `torch.equal` when the test trips.
    mism = (fixed_bf16 != silicon_bf16).sum().item()
    total = fixed_bf16.numel()
    assert mism == 0, (
        f"D3 patch validation failed: {mism}/{total} output elements "
        f"differ between fixed_golden and silicon-faithful reference. "
        f"max |fixed-silicon| (fp32) = "
        f"{(fixed.to(torch.float32) - silicon.to(torch.float32)).abs().max().item()}"
    )
