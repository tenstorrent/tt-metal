# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Small, fail-closed theorems about instantiated target input lattices.

These certificates concern representable target inputs, not fitted functions or
host-library arithmetic.  Keeping them independent of activation identities
lets Program verification consume them only when the data-flow shape preserves
the certified lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .program import TargetPolicy


class TargetLatticeProofError(ValueError):
    """The requested theorem is not established for this target policy."""


@dataclass(frozen=True)
class BhBf16RawClassPredicateCertificate:
    """Exhaustive semantics of one Blackhole raw-DST BF16 class predicate.

    Blackhole stores a BF16 lane in the low 16 bits of DST as
    ``sign[15] | mantissa[6:0]<<8 | exponent[7:0]``.  Consumers may use a
    schedule appropriate to their live range, but the schedule must implement
    this closed predicate exactly::

        (raw & equal_mask) == equal_value
            and (raw & nonzero_mask) != 0

    The certificate deliberately contains no instruction-selection claim.
    That keeps raw-class semantics shared between independent lowering paths
    while requiring each path to certify its own schedule and liveness.
    """

    target_sha256: str
    physical_encoding: str
    equal_mask_u16: int
    equal_value_u16: int
    nonzero_mask_u16: int
    raw_pattern_count: int
    true_pattern_count: int
    truth_sha256: str
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_bf16_raw_class_predicate_v1",
            "target_sha256": self.target_sha256,
            "physical_encoding": self.physical_encoding,
            "equal_mask_u16": f"0x{self.equal_mask_u16:04x}",
            "equal_value_u16": f"0x{self.equal_value_u16:04x}",
            "nonzero_mask_u16": f"0x{self.nonzero_mask_u16:04x}",
            "raw_pattern_count": self.raw_pattern_count,
            "true_pattern_count": self.true_pattern_count,
            "truth_sha256": self.truth_sha256,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BhBf16RawClassExcludedWordCertificate:
    """Exhaustive lowering of a raw class predicate by excluding one word.

    This proves that ``(raw & mask) == value && raw != excluded`` is exactly
    equivalent to a shared mask/value/nonzero predicate over all BF16 words.
    Resource bounds travel with the descriptor so a consumer cannot silently
    select a semantically right schedule that does not fit its live range.
    """

    target_sha256: str
    physical_encoding: str
    equal_mask_u16: int
    equal_value_u16: int
    excluded_word_u16: int
    reference_nonzero_mask_u16: int
    raw_pattern_count: int
    true_pattern_count: int
    predicate_issue_slots: int
    predicate_peak_live: int
    truth_sha256: str
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_bf16_raw_class_excluded_word_v1",
            "target_sha256": self.target_sha256,
            "physical_encoding": self.physical_encoding,
            "equal_mask_u16": f"0x{self.equal_mask_u16:04x}",
            "equal_value_u16": f"0x{self.equal_value_u16:04x}",
            "excluded_word_u16": f"0x{self.excluded_word_u16:04x}",
            "reference_nonzero_mask_u16": f"0x{self.reference_nonzero_mask_u16:04x}",
            "raw_pattern_count": self.raw_pattern_count,
            "true_pattern_count": self.true_pattern_count,
            "predicate_issue_slots": self.predicate_issue_slots,
            "predicate_peak_live": self.predicate_peak_live,
            "truth_sha256": self.truth_sha256,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BhBf16OriginTangentReconstructionCertificate:
    """Exact first-normal-binade reconstruction for DAZ-erased inputs.

    A nonzero BF16 subnormal has the mathematical form ``sign*m*2^-133``
    for integer ``m`` in ``[1, 127]``. If a declared zero-origin function is
    tangent to ``s*x`` on each side of zero, its target-rounded result can be
    reconstructed without ever creating an FP32 subnormal: round ``m*s`` to
    an integer ``q`` first, then apply one exponent adjustment by ``-133``.
    Blackhole v4.1 flushes ``q < 128`` and produces the exact first-normal
    BF16 word for ``q >= 128``.

    This certificate owns only target-lattice arithmetic. A lowering must
    separately prove typed slope provenance, raw-word liveness, and schedule
    capacity.
    """

    target_sha256: str
    positive_slope_bits: int
    negative_slope_bits: int
    raw_pattern_count: int
    reconstructed_pattern_count: int
    nonzero_normal_output_count: int
    positive_first_normal_input_bits: int
    negative_first_normal_input_bits: int
    output_truth_sha256: str
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_bf16_origin_tangent_reconstruction_v1",
            "target_sha256": self.target_sha256,
            "positive_slope_bits": f"0x{self.positive_slope_bits:08x}",
            "negative_slope_bits": f"0x{self.negative_slope_bits:08x}",
            "raw_pattern_count": self.raw_pattern_count,
            "reconstructed_pattern_count": self.reconstructed_pattern_count,
            "nonzero_normal_output_count": self.nonzero_normal_output_count,
            "positive_first_normal_input_bits": (f"0x{self.positive_first_normal_input_bits:04x}"),
            "negative_first_normal_input_bits": (f"0x{self.negative_first_normal_input_bits:04x}"),
            "output_truth_sha256": self.output_truth_sha256,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BhBf16OriginHalfTieCertificate:
    """Exact output-FTZ boundary for an anonymous half-slope origin.

    The theorem owns only the BF16 input lattice through the first normal
    binade.  With exact tangent ``x/2``, target BF16 RNE followed by output FTZ
    is zero everywhere in that envelope except at the largest positive and
    negative values of the first normal binade.  Those two exact ties round to
    the even minimum-normal result.  A compiler still has to prove that its
    typed graph reduces to this tangent over the owned envelope and that its
    raw discriminator/materializer fits the replay.
    """

    target_sha256: str
    slope_bits: int
    raw_pattern_count: int
    owned_pattern_count: int
    corrected_pattern_count: int
    positive_input_bits: int
    negative_input_bits: int
    positive_output_bits: int
    negative_output_bits: int
    output_truth_sha256: str
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_bf16_origin_half_tie_v1",
            "target_sha256": self.target_sha256,
            "slope_bits": f"0x{self.slope_bits:08x}",
            "raw_pattern_count": self.raw_pattern_count,
            "owned_pattern_count": self.owned_pattern_count,
            "corrected_pattern_count": self.corrected_pattern_count,
            "positive_input_bits": f"0x{self.positive_input_bits:04x}",
            "negative_input_bits": f"0x{self.negative_input_bits:04x}",
            "positive_output_bits": f"0x{self.positive_output_bits:04x}",
            "negative_output_bits": f"0x{self.negative_output_bits:04x}",
            "output_truth_sha256": self.output_truth_sha256,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BhBf16OriginLinearFtzRneCellCertificate:
    """Coefficient-derived BF16 cell lost by an FP32-FTZ linear multiply.

    The theorem enumerates the complete BF16 input lattice for one anonymous
    FP32 slope.  It is intentionally independent of a function name: a fitter
    must separately prove that its emitted form reduces to this linear term
    on the owned cell and that the mathematical target rounds identically.
    """

    target_sha256: str
    slope_bits: int
    positive_input_bits: int
    negative_input_bits: int
    positive_output_bits: int
    negative_output_bits: int
    comparison_exponent_shift: int
    scaled_magnitude_bits: int
    corrected_pattern_count: int
    truth_sha256: str
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_bf16_origin_linear_ftz_rne_cell_v1",
            "target_sha256": self.target_sha256,
            "slope_bits": f"0x{self.slope_bits:08x}",
            "positive_input_bits": f"0x{self.positive_input_bits:04x}",
            "negative_input_bits": f"0x{self.negative_input_bits:04x}",
            "positive_output_bits": f"0x{self.positive_output_bits:04x}",
            "negative_output_bits": f"0x{self.negative_output_bits:04x}",
            "comparison_exponent_shift": self.comparison_exponent_shift,
            "scaled_magnitude_bits": f"0x{self.scaled_magnitude_bits:08x}",
            "corrected_pattern_count": self.corrected_pattern_count,
            "truth_sha256": self.truth_sha256,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


def bh_bf16_ieee_to_dst_u16(raw_bf16_bits: np.ndarray) -> np.ndarray:
    """Convert IEEE-field-order BF16 words to Blackhole DST U16 encoding."""
    raw = np.asarray(raw_bf16_bits, dtype=np.uint16)
    sign = raw & np.uint16(0x8000)
    exponent = (raw >> np.uint16(7)) & np.uint16(0x00FF)
    mantissa = (raw & np.uint16(0x007F)) << np.uint16(8)
    return (sign | mantissa | exponent).astype(np.uint16, copy=False)


def bh_bf16_raw_class_predicate_bits(
    raw_bf16_bits: np.ndarray,
    *,
    equal_mask_u16: int,
    equal_value_u16: int,
    nonzero_mask_u16: int,
) -> np.ndarray:
    """Evaluate the shared mask/value/nonzero predicate on IEEE BF16 words."""
    for name, value in (
        ("equal_mask_u16", equal_mask_u16),
        ("equal_value_u16", equal_value_u16),
        ("nonzero_mask_u16", nonzero_mask_u16),
    ):
        if type(value) is not int or not 0 <= value <= 0xFFFF:
            raise TargetLatticeProofError(f"{name} must be a uint16")
    if equal_value_u16 & ~equal_mask_u16:
        raise TargetLatticeProofError("raw-class equal value must be a subset of its equality mask")
    if equal_mask_u16 & nonzero_mask_u16:
        raise TargetLatticeProofError("raw-class equality and nonzero masks must be disjoint")
    if nonzero_mask_u16 == 0:
        raise TargetLatticeProofError("raw-class nonzero mask must be nonzero")
    dst = bh_bf16_ieee_to_dst_u16(raw_bf16_bits)
    return ((dst & np.uint16(equal_mask_u16)) == np.uint16(equal_value_u16)) & (
        (dst & np.uint16(nonzero_mask_u16)) != np.uint16(0)
    )


def certify_bh_bf16_raw_class_predicate(
    target: TargetPolicy,
    *,
    equal_mask_u16: int,
    equal_value_u16: int,
    nonzero_mask_u16: int,
) -> BhBf16RawClassPredicateCertificate:
    """Enumerate one raw-DST predicate over all 65,536 BF16 encodings."""
    if not isinstance(target, TargetPolicy):
        raise TypeError("raw-class predicate proof requires a TargetPolicy")
    if not (
        target.precision == "bf16"
        and target.architecture == "bh"
        and target.intermediate_precision == "fp32"
        and target.input_daz
        and target.rounding_order == "post_round"
    ):
        raise TargetLatticeProofError(
            "raw-class predicate theorem requires BH BF16/FP32-intermediate " "DAZ post-round target semantics"
        )
    raw = np.arange(1 << 16, dtype=np.uint16)
    truth = bh_bf16_raw_class_predicate_bits(
        raw,
        equal_mask_u16=equal_mask_u16,
        equal_value_u16=equal_value_u16,
        nonzero_mask_u16=nonzero_mask_u16,
    )
    evidence = hashlib.sha256()
    evidence.update(raw.astype("<u2", copy=False).tobytes())
    evidence.update(truth.astype(np.uint8, copy=False).tobytes())
    return BhBf16RawClassPredicateCertificate(
        target_sha256=target.sha256,
        physical_encoding="sign15_mantissa14_8_exponent7_0",
        equal_mask_u16=equal_mask_u16,
        equal_value_u16=equal_value_u16,
        nonzero_mask_u16=nonzero_mask_u16,
        raw_pattern_count=1 << 16,
        true_pattern_count=int(np.count_nonzero(truth)),
        truth_sha256=evidence.hexdigest(),
        proof_rules=(
            "blackhole_float16_b_dst_u16_physical_encoding",
            "equality_and_nonzero_masks_are_disjoint",
            "equal_value_is_subset_of_equality_mask",
            "all_65536_ieee_bf16_words_exhaustively_encoded_and_tested",
            "instruction_schedule_and_liveness_not_claimed",
        ),
    )


def certify_bh_bf16_raw_class_excluded_word(
    target: TargetPolicy,
    *,
    equal_mask_u16: int,
    equal_value_u16: int,
    excluded_word_u16: int,
    reference_nonzero_mask_u16: int,
) -> BhBf16RawClassExcludedWordCertificate:
    """Prove a one-AND excluded-word schedule against the shared theorem."""
    reference = certify_bh_bf16_raw_class_predicate(
        target,
        equal_mask_u16=equal_mask_u16,
        equal_value_u16=equal_value_u16,
        nonzero_mask_u16=reference_nonzero_mask_u16,
    )
    if type(excluded_word_u16) is not int or not 0 <= excluded_word_u16 <= 0xFFFF:
        raise TargetLatticeProofError("excluded_word_u16 must be a uint16")
    if (excluded_word_u16 & equal_mask_u16) != equal_value_u16:
        raise TargetLatticeProofError("excluded raw word must satisfy the equality prefix")

    ieee = np.arange(1 << 16, dtype=np.uint16)
    dst = bh_bf16_ieee_to_dst_u16(ieee)
    shortened = ((dst & np.uint16(equal_mask_u16)) == np.uint16(equal_value_u16)) & (
        dst != np.uint16(excluded_word_u16)
    )
    canonical = bh_bf16_raw_class_predicate_bits(
        ieee,
        equal_mask_u16=equal_mask_u16,
        equal_value_u16=equal_value_u16,
        nonzero_mask_u16=reference_nonzero_mask_u16,
    )
    if not np.array_equal(shortened, canonical):
        raise TargetLatticeProofError("excluded-word schedule is not equivalent to the shared raw-class predicate")
    evidence = hashlib.sha256()
    evidence.update(ieee.astype("<u2", copy=False).tobytes())
    evidence.update(shortened.astype(np.uint8, copy=False).tobytes())
    return BhBf16RawClassExcludedWordCertificate(
        target_sha256=reference.target_sha256,
        physical_encoding=reference.physical_encoding,
        equal_mask_u16=equal_mask_u16,
        equal_value_u16=equal_value_u16,
        excluded_word_u16=excluded_word_u16,
        reference_nonzero_mask_u16=reference_nonzero_mask_u16,
        raw_pattern_count=1 << 16,
        true_pattern_count=int(np.count_nonzero(shortened)),
        predicate_issue_slots=3,
        predicate_peak_live=2,
        truth_sha256=evidence.hexdigest(),
        proof_rules=(
            "canonical_mask_value_nonzero_predicate_certified",
            "excluded_word_satisfies_equality_prefix",
            "all_65536_physical_words_exhaustively_equivalent",
            "one_mask_and_two_comparisons",
            "predicate_peak_live_two",
        ),
    )


def _fp32_bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def bh_bf16_origin_tangent_output_bits(
    raw_bf16_bits: np.ndarray,
    *,
    positive_slope: float,
    negative_slope: float,
) -> np.ndarray:
    """Model integer-first tangent reconstruction for zero/subnormal words."""
    raw = np.asarray(raw_bf16_bits, dtype=np.uint16)
    positive = np.float32(positive_slope)
    negative = np.float32(negative_slope)
    if not (
        np.isfinite(positive) and np.isfinite(negative) and positive > np.float32(0.0) and negative > np.float32(0.0)
    ):
        raise TargetLatticeProofError("origin tangent slopes must be positive FP32")

    owned = (raw & np.uint16(0x7F80)) == np.uint16(0)
    magnitude = (raw & np.uint16(0x007F)).astype(np.float32)
    slope = np.where((raw & np.uint16(0x8000)) != np.uint16(0), negative, positive).astype(np.float32)
    magic = np.float32(1.5 * (1 << 23))
    rounded = ((magnitude * slope + magic).astype(np.float32) - magic).astype(np.float32)
    integer = rounded.astype(np.uint16)
    if bool(np.any(integer[owned] > np.uint16(0x00FF))):
        raise TargetLatticeProofError("origin tangent reconstruction escapes the first BF16 normal binade")
    emitted_magnitude = np.where(integer >= np.uint16(0x0080), integer, np.uint16(0)).astype(np.uint16)
    output = (raw & np.uint16(0x8000)) | emitted_magnitude
    return np.where(owned, output, np.uint16(0)).astype(np.uint16)


def certify_bh_bf16_origin_tangent_reconstruction(
    target: TargetPolicy,
    *,
    positive_slope: float,
    negative_slope: float,
) -> BhBf16OriginTangentReconstructionCertificate:
    """Exhaustively certify integer-first reconstruction over BF16 raw words."""
    if not isinstance(target, TargetPolicy):
        raise TypeError("origin tangent proof requires a TargetPolicy")
    if not (
        target.precision == "bf16"
        and target.architecture == "bh"
        and target.intermediate_precision == "fp32"
        and target.input_daz
        and target.output_ftz
        and target.rounding_order == "post_round"
    ):
        raise TargetLatticeProofError("origin tangent reconstruction requires BH BF16 DAZ/FTZ post-round")

    positive = np.float32(positive_slope)
    negative = np.float32(negative_slope)
    raw = np.arange(1 << 16, dtype=np.uint16)
    emitted = bh_bf16_origin_tangent_output_bits(raw, positive_slope=float(positive), negative_slope=float(negative))
    exponent = raw & np.uint16(0x7F80)
    mantissa = raw & np.uint16(0x007F)
    owned = exponent == np.uint16(0)
    reconstructed = owned & (mantissa != np.uint16(0))

    # Independent target construction: exact BF16 input times the rounded
    # slope, BF16 RNE, then target output FTZ.
    inputs = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    slopes = np.where(np.signbit(inputs), negative, positive).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        tangent = (inputs * slopes).astype(np.float32)
    bits = tangent.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded_bits = bits + np.uint32(0x7FFF) + lsb
    rounded_bf16 = (rounded_bits >> np.uint32(16)).astype(np.uint16)
    rounded_exponent = rounded_bf16 & np.uint16(0x7F80)
    rounded_mantissa = rounded_bf16 & np.uint16(0x007F)
    rounded_subnormal = (rounded_exponent == 0) & (rounded_mantissa != 0)
    rounded_bf16[rounded_subnormal] &= np.uint16(0x8000)
    if not np.array_equal(emitted[owned], rounded_bf16[owned]):
        raise TargetLatticeProofError("integer-first origin reconstruction disagrees with BF16 tangent RNE")

    nonzero_normal = reconstructed & ((emitted & np.uint16(0x7FFF)) >= np.uint16(0x0080))
    positive_normal = np.flatnonzero(nonzero_normal & ((raw & 0x8000) == 0))
    negative_normal = np.flatnonzero(nonzero_normal & ((raw & 0x8000) != 0))
    if len(positive_normal) == 0 or len(negative_normal) == 0:
        raise TargetLatticeProofError("origin tangent reconstruction must reach the first normal binade")
    truth = hashlib.sha256()
    truth.update(raw.astype("<u2", copy=False).tobytes())
    truth.update(owned.astype(np.uint8, copy=False).tobytes())
    truth.update(emitted.astype("<u2", copy=False).tobytes())
    return BhBf16OriginTangentReconstructionCertificate(
        target_sha256=target.sha256,
        positive_slope_bits=_fp32_bits(float(positive)),
        negative_slope_bits=_fp32_bits(float(negative)),
        raw_pattern_count=1 << 16,
        reconstructed_pattern_count=int(np.count_nonzero(reconstructed)),
        nonzero_normal_output_count=int(np.count_nonzero(nonzero_normal)),
        positive_first_normal_input_bits=int(raw[positive_normal[0]]),
        negative_first_normal_input_bits=int(raw[negative_normal[0]]),
        output_truth_sha256=truth.hexdigest(),
        proof_rules=(
            "all_65536_bf16_raw_words_enumerated",
            "signed_bf16_subnormal_is_integer_m_times_2_pow_minus_133",
            "magic_rne_integer_precedes_exponent_adjustment",
            "blackhole_v4_1_addexp_flushes_q_below_128",
            "q_at_least_128_materializes_exact_first_normal_binade",
            "integer_first_output_matches_independent_bf16_tangent_rne",
            "instruction_schedule_and_program_declaration_require_separate_proof",
        ),
    )


def bh_bf16_origin_half_tie_output_bits(
    raw_bf16_bits: np.ndarray,
) -> np.ndarray:
    """Return the two exact nonzero outputs in the half-origin envelope."""
    raw = np.asarray(raw_bf16_bits, dtype=np.uint16)
    magnitude = raw & np.uint16(0x7FFF)
    owned = magnitude <= np.uint16(0x00FF)
    tie = magnitude == np.uint16(0x00FF)
    emitted = (raw & np.uint16(0x8000)) | np.uint16(0x0080)
    return np.where(owned & tie, emitted, np.uint16(0)).astype(np.uint16)


def certify_bh_bf16_origin_half_tie(target: TargetPolicy, *, slope: float) -> BhBf16OriginHalfTieCertificate:
    """Exhaustively certify the BF16 RNE/FTZ transition for ``x*slope``."""
    if not isinstance(target, TargetPolicy):
        raise TypeError("origin-half proof requires a TargetPolicy")
    if not (
        target.precision == "bf16"
        and target.architecture == "bh"
        and target.intermediate_precision == "fp32"
        and target.input_daz
        and target.output_ftz
        and target.rounding_order == "post_round"
    ):
        raise TargetLatticeProofError("origin-half tie theorem requires BH BF16 DAZ/FTZ post-round")
    rounded_slope = np.float32(slope)
    if _fp32_bits(float(rounded_slope)) != 0x3F000000:
        raise TargetLatticeProofError("origin-half tie theorem requires the exact FP32 slope one-half")

    raw = np.arange(1 << 16, dtype=np.uint16)
    magnitude = raw & np.uint16(0x7FFF)
    owned = magnitude <= np.uint16(0x00FF)
    emitted = bh_bf16_origin_half_tie_output_bits(raw)

    # Independent construction from the exact integer significand.  In the
    # first normal binade the input is ``(128 + mantissa) * 2^-133``;
    # division by two followed by RNE is an integer divide with an odd-quotient
    # tie increment.  Only q >= 128 survives target output FTZ.  The subnormal
    # input binade is computed by the same integer rule without the implicit
    # leading 128 and therefore cannot survive.
    exponent = (magnitude >> np.uint16(7)) & np.uint16(0x00FF)
    mantissa = magnitude & np.uint16(0x007F)
    significand = mantissa.astype(np.uint16)
    significand[exponent == np.uint16(1)] += np.uint16(128)
    quotient = significand >> np.uint16(1)
    quotient += (significand & np.uint16(3) == np.uint16(3)).astype(np.uint16)
    independent = np.where(
        owned & (quotient >= np.uint16(128)),
        (raw & np.uint16(0x8000)) | quotient,
        np.uint16(0),
    ).astype(np.uint16)
    if not np.array_equal(emitted[owned], independent[owned]):
        raise TargetLatticeProofError("origin-half raw construction disagrees with BF16 RNE/FTZ")
    corrected = owned & (emitted != np.uint16(0))
    if np.flatnonzero(corrected).tolist() != [0x00FF, 0x80FF]:
        raise TargetLatticeProofError("origin-half correction population drifted")

    evidence = hashlib.sha256()
    evidence.update(raw.astype("<u2", copy=False).tobytes())
    evidence.update(owned.astype(np.uint8, copy=False).tobytes())
    evidence.update(emitted.astype("<u2", copy=False).tobytes())
    return BhBf16OriginHalfTieCertificate(
        target_sha256=target.sha256,
        slope_bits=0x3F000000,
        raw_pattern_count=1 << 16,
        owned_pattern_count=int(np.count_nonzero(owned)),
        corrected_pattern_count=int(np.count_nonzero(corrected)),
        positive_input_bits=0x00FF,
        negative_input_bits=0x80FF,
        positive_output_bits=0x0080,
        negative_output_bits=0x8080,
        output_truth_sha256=evidence.hexdigest(),
        proof_rules=(
            "all_65536_bf16_raw_words_enumerated",
            "owned_lattice_is_signed_zero_subnormal_and_first_normal_binade",
            "exact_half_scale_has_one_rne_even_minimum_normal_tie_per_sign",
            "all_smaller_rounded_subnormal_outputs_are_target_ftz_zero",
            "typed_graph_provenance_and_replay_schedule_require_separate_proof",
        ),
    )


def certify_bh_bf16_origin_linear_ftz_rne_cell(
    target: TargetPolicy, *, slope: float
) -> BhBf16OriginLinearFtzRneCellCertificate:
    """Certify one signed BF16 rounding cell lost by multiply-first FTZ.

    ``ideal`` rounds the exact product to BF16 and then applies output FTZ.
    ``physical`` first rounds the product to FP32, applies Blackhole FP32 FTZ,
    and only then performs BF16 RNE.  The supported compact terminal has one
    differing magnitude (two signs); wider correction sets fail closed.
    """

    from ttpoly.precision import bf16 as _bf16

    if not isinstance(target, TargetPolicy):
        raise TypeError("origin-linear proof requires a TargetPolicy")
    if not (
        target.precision == "bf16"
        and target.architecture == "bh"
        and target.intermediate_precision == "fp32"
        and target.input_daz
        and target.output_ftz
        and target.rounding_order == "post_round"
    ):
        raise TargetLatticeProofError("origin-linear FTZ/RNE theorem requires BH BF16 DAZ/FTZ post-round")
    rounded_slope = np.float32(slope)
    if not (np.isfinite(rounded_slope) and 0.0 < rounded_slope < 1.0):
        raise TargetLatticeProofError("origin-linear FTZ/RNE theorem requires a positive sub-unit FP32 slope")

    raw = np.arange(1 << 16, dtype=np.uint16)
    magnitude = raw & np.uint16(0x7FFF)
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    subnormal = (magnitude & np.uint16(0x7F80)) == np.uint16(0)
    effective = values.copy()
    effective[subnormal] = np.copysign(np.float32(0.0), effective[subnormal])

    with np.errstate(invalid="ignore", over="ignore"):
        ideal = _bf16.to_bf16(
            effective.astype(np.float64) * float(rounded_slope),
            flush_to_zero=True,
            rounding_mode="rne",
        )
        physical = np.asarray(effective * rounded_slope, dtype=np.float32)
    physical_bits = physical.view(np.uint32)
    physical_subnormal = ((physical_bits & np.uint32(0x7F800000)) == 0) & ((physical_bits & np.uint32(0x7FFFFFFF)) != 0)
    physical_bits[physical_subnormal] &= np.uint32(0x80000000)
    physical = _bf16.to_bf16(
        physical_bits.view(np.float32),
        flush_to_zero=True,
        rounding_mode="rne",
    )
    ideal_bits = (np.asarray(ideal, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    physical_output_bits = (np.asarray(physical, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    differing = (ideal_bits & np.uint16(0x7FFF)) != (physical_output_bits & np.uint16(0x7FFF))
    indices = np.flatnonzero(differing).tolist()
    if len(indices) != 2 or indices[1] != (indices[0] | 0x8000):
        raise TargetLatticeProofError("origin-linear FTZ/RNE correction is not one symmetric BF16 cell")
    positive_input = int(indices[0])
    negative_input = int(indices[1])
    positive_output = int(ideal_bits[positive_input])
    negative_output = int(ideal_bits[negative_input])
    if not (positive_output == 0x0080 and negative_output == 0x8080 and positive_input < 0x8000):
        raise TargetLatticeProofError("origin-linear FTZ/RNE cell does not materialize signed MIN_NORMAL")

    input_exponent = (positive_input >> 7) & 0xFF
    comparison_shift = 127 - input_exponent
    scaled = np.ldexp(np.float32(values[positive_input]), comparison_shift)
    scaled_bits = _fp32_bits(float(scaled))
    if not (0x3F800000 <= scaled_bits < 0x40000000):
        raise TargetLatticeProofError("origin-linear comparison did not normalize into [1, 2)")

    evidence = hashlib.sha256()
    evidence.update(raw.astype("<u2", copy=False).tobytes())
    evidence.update(ideal_bits.astype("<u2", copy=False).tobytes())
    evidence.update(physical_output_bits.astype("<u2", copy=False).tobytes())
    evidence.update(differing.astype(np.uint8, copy=False).tobytes())
    return BhBf16OriginLinearFtzRneCellCertificate(
        target_sha256=target.sha256,
        slope_bits=_fp32_bits(float(rounded_slope)),
        positive_input_bits=positive_input,
        negative_input_bits=negative_input,
        positive_output_bits=positive_output,
        negative_output_bits=negative_output,
        comparison_exponent_shift=comparison_shift,
        scaled_magnitude_bits=scaled_bits,
        corrected_pattern_count=2,
        truth_sha256=evidence.hexdigest(),
        proof_rules=(
            "all_65536_bf16_raw_words_enumerated",
            "input_daz_precedes_anonymous_linear_term",
            "ideal_bf16_rne_precedes_output_ftz",
            "blackhole_fp32_ftz_precedes_destination_bf16_rne",
            "exactly_one_symmetric_input_magnitude_differs",
            "comparison_constant_is_normalized_numeric_magnitude",
            "typed_form_and_mathematical_target_match_require_fitter_proof",
        ),
    )


@dataclass(frozen=True)
class PiecewiseBoundaryThresholdCertificate:
    """Target-lattice equivalence for one physical ``x >= threshold`` test."""

    target_sha256: str
    mathematical_breakpoint: float
    boundary_owner: str
    physical_threshold: float
    finite_input_encodings_proved: int
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_piecewise_boundary_threshold_v1",
            "target_sha256": self.target_sha256,
            "mathematical_breakpoint": self.mathematical_breakpoint.hex(),
            "boundary_owner": self.boundary_owner,
            "physical_threshold": self.physical_threshold.hex(),
            "finite_input_encodings_proved": self.finite_input_encodings_proved,
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


def _bf16_successor(value: float) -> float:
    fp32 = np.asarray([value], dtype=np.float32)
    word = int(fp32.view(np.uint32)[0])
    if word & 0xFFFF:
        raise TargetLatticeProofError("piecewise breakpoint is not exactly representable as BF16")
    raw = word >> 16
    if value == 0.0:
        # Numeric successor is the smallest positive subnormal for either
        # signed-zero encoding.  Incrementing raw -0 (0x8000) would move to a
        # negative subnormal, which is ordered below zero.
        raw = 0x0001
    else:
        raw = raw + 1 if value > 0.0 else raw - 1
    successor = np.asarray([raw << 16], dtype=np.uint32).view(np.float32)[0]
    if not np.isfinite(successor):
        raise TargetLatticeProofError("piecewise BF16 successor is not a finite physical threshold")
    return float(successor)


def _piecewise_physical_threshold(target: TargetPolicy, breakpoint: float, owner: str) -> float:
    if owner == "right":
        return breakpoint
    if breakpoint == 0.0 and target.input_daz:
        return float(np.finfo(np.float32).tiny)
    if target.precision == "bf16":
        successor = _bf16_successor(breakpoint)
        if target.input_daz and 0.0 < successor < np.finfo(np.float32).tiny:
            successor = float(np.finfo(np.float32).tiny)
        return successor
    return float(np.nextafter(np.float32(breakpoint), np.float32(np.inf), dtype=np.float32))


def certify_piecewise_boundary_threshold(
    target: TargetPolicy, *, breakpoint: float, owner: str
) -> PiecewiseBoundaryThresholdCertificate:
    """Certify a target-specific ``>=`` threshold for left/right ownership.

    Special inputs are deliberately outside this theorem.  BF16 is enumerated
    exhaustively over every finite raw encoding: desired ownership is computed
    on the mathematical raw value and compared with the emitted predicate on
    the target-effective DAZ coordinate. FP32 uses the adjacency theorem of
    ``nextafter`` and is additionally exercised by stratified tests at every
    exponent. A zero boundary under DAZ is refused because raw subnormals need
    a pre-DAZ discriminator rather than a decoded floating threshold.
    """
    if not isinstance(target, TargetPolicy):
        raise TypeError("piecewise boundary proof requires a TargetPolicy")
    if target.architecture != "bh" or target.precision not in {"bf16", "fp32"}:
        raise TargetLatticeProofError("piecewise boundary threshold theorem requires BH BF16 or FP32")
    if target.intermediate_precision != "fp32":
        raise TargetLatticeProofError("piecewise boundary threshold theorem requires FP32 comparisons")
    if owner not in {"left", "right"}:
        raise TargetLatticeProofError("piecewise boundary owner must be 'left' or 'right'")
    if isinstance(breakpoint, bool) or not isinstance(breakpoint, (int, float)):
        raise TargetLatticeProofError("piecewise breakpoint must be finite")
    mathematical = float(breakpoint)
    if not math.isfinite(mathematical):
        raise TargetLatticeProofError("piecewise breakpoint must be finite")
    if target.input_daz and mathematical != 0.0 and abs(mathematical) < float(np.finfo(np.float32).tiny):
        raise TargetLatticeProofError("nonzero subnormal piecewise breakpoint is not in the DAZ effective lattice")
    if target.input_daz and mathematical == 0.0:
        raise TargetLatticeProofError("raw-input boundary ownership at zero requires a pre-DAZ discriminator")
    rounded = (
        float(np.asarray([mathematical], dtype=np.float32)[0])
        if target.precision == "fp32"
        else float(
            np.asarray(
                [int(np.asarray([mathematical], dtype=np.float32).view(np.uint32)[0]) & 0xFFFF0000],
                dtype=np.uint32,
            ).view(np.float32)[0]
        )
    )
    if rounded != mathematical:
        raise TargetLatticeProofError(f"piecewise breakpoint is not exactly representable as {target.precision}")

    physical = _piecewise_physical_threshold(target, mathematical, owner)
    if target.precision == "bf16":
        raw = np.arange(1 << 16, dtype=np.uint32)
        values = (raw << np.uint32(16)).view(np.float32)
        finite = np.isfinite(values)
        effective = values[finite].copy()
        if target.input_daz:
            subnormal = (np.abs(effective) < np.finfo(np.float32).tiny) & (effective != 0.0)
            effective[subnormal] = np.float32(0.0)
        raw_finite = values[finite]
        desired_right = raw_finite >= mathematical if owner == "right" else raw_finite > mathematical
        emitted_right = effective >= physical
        if not np.array_equal(desired_right, emitted_right):
            raise TargetLatticeProofError("physical threshold is not exhaustive-equivalent on BF16")
        population = int(np.count_nonzero(finite))
        population_rule = "all_finite_bf16_encodings_exhausted_after_target_daz"
    else:
        population = (1 << 32) - (2 * ((1 << 23) - 1)) - 2
        population_rule = "fp32_nextafter_adjacency_excludes_intervening_input"

    owner_rule = (
        "right_owner_uses_exact_mathematical_breakpoint"
        if owner == "right"
        else "left_owner_uses_next_effective_target_input"
    )
    return PiecewiseBoundaryThresholdCertificate(
        target_sha256=target.sha256,
        mathematical_breakpoint=mathematical,
        boundary_owner=owner,
        physical_threshold=physical,
        finite_input_encodings_proved=population,
        proof_rules=(
            "mathematical_breakpoint_exactly_representable_on_target",
            owner_rule,
            "physical_greater_equal_matches_declared_boundary_owner",
            population_rule,
            "nonfinite_inputs_owned_by_separate_special_policy",
        ),
    )


@dataclass(frozen=True)
class RoundEvenResidualLatticeCertificate:
    """Proof facts for ``x - round_even(x)`` on a DAZ input lattice.

    The Blackhole primitive returns an integral FP32 value.  For ``|x| < 0.5``
    it returns zero, so a nonzero residual is the original DAZ-normal input.
    From ``0.5`` through ``2**23``, FP32 spacing is at least ``2**-24`` and
    Sterbenz subtraction is exact.  At and above ``2**23`` every finite FP32
    value is already integral.  BF16 is a subset of that FP32 lattice.
    Therefore every nonzero residual is an ordinary FP32 normal value.
    """

    target_sha256: str
    input_precision: str
    intermediate_precision: str
    minimum_nonzero_magnitude: float
    maximum_magnitude: float
    fractional_spacing_floor: float
    integral_magnitude_threshold: float
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_bh_round_even_residual_lattice_v1",
            "target_sha256": self.target_sha256,
            "input_precision": self.input_precision,
            "intermediate_precision": self.intermediate_precision,
            "minimum_nonzero_magnitude": self.minimum_nonzero_magnitude.hex(),
            "maximum_magnitude": self.maximum_magnitude.hex(),
            "fractional_spacing_floor": self.fractional_spacing_floor.hex(),
            "integral_magnitude_threshold": (self.integral_magnitude_threshold.hex()),
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CodyWaiteLn2CoordinateCertificate:
    """A structural bound for an emitted one- or two-FMA ln(2) reduction.

    This is deliberately a coordinate certificate, not a statement about an
    activation or an approximation.  It covers the typed data flow

    ``s = fma(x, inv_ln2, 0); k = rint(s)``
    ``r = fma(k, neg_ln2_lo, fma(k, neg_ln2_hi, x))`` when the split low
    part is nonzero, or ``r = fma(k, neg_ln2_hi, x)`` when it is zero.

    BH/WH truncate the product before the add by at most one eighth of a
    binary32 product ULP and round the subsequent add to nearest-even.  The
    bound below keeps those errors, the rounded ``inv_ln2`` mismatch, and the
    rounded split-ln(2) mismatch explicit.  In particular it does *not*
    pretend that the emitted coordinate is the real-arithmetic interval
    ``[-ln(2)/2, ln(2)/2]``.
    """

    target_sha256: str
    source_lower: float
    source_upper: float
    maximum_integer_magnitude: int
    quotient_error_bound: float
    inverse_split_mismatch: float
    coordinate_lower: float
    coordinate_upper: float
    proof_rules: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "ttpoly_cody_waite_ln2_coordinate_v1",
            "target_sha256": self.target_sha256,
            "source_lower": self.source_lower.hex(),
            "source_upper": self.source_upper.hex(),
            "maximum_integer_magnitude": self.maximum_integer_magnitude,
            "quotient_error_bound": self.quotient_error_bound.hex(),
            "inverse_split_mismatch": self.inverse_split_mismatch.hex(),
            "coordinate_lower": self.coordinate_lower.hex(),
            "coordinate_upper": self.coordinate_upper.hex(),
            "proof_rules": list(self.proof_rules),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


def _fp32_ulp_upper(magnitude: float) -> float:
    """Largest binary32 ULP at a finite nonnegative magnitude."""
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise TargetLatticeProofError("FP32 ULP bound requires a finite magnitude")
    if magnitude < float(np.finfo(np.float32).tiny):
        return math.ldexp(1.0, -149)
    return math.ldexp(1.0, math.floor(math.log2(magnitude)) - 23)


def _fp32_upward(value: float) -> float:
    """Round a positive finite proof bound outward to binary32."""
    candidate = np.float32(value)
    if not np.isfinite(candidate):
        raise TargetLatticeProofError("coordinate proof bound overflowed FP32")
    if float(candidate) < value:
        candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    # One more representable neighbor covers the host-double evaluation of the
    # closed-form error expression; the theorem is not relying on that host
    # expression rounding inward.
    candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    return float(candidate)


def certify_bh_wh_cody_waite_ln2_coordinate(
    target: TargetPolicy,
    *,
    source_lower: float,
    source_upper: float,
    inverse_ln2: float,
    negative_ln2_hi: float,
    negative_ln2_lo: float,
) -> CodyWaiteLn2CoordinateCertificate:
    """Prove the coordinate envelope of a typed two-FMA ln(2) reducer.

    The currently proved magnitude is the production small-range reducer
    contract.  Larger inputs need a different reduction theorem rather than a
    silently wider approximation leaf.
    """
    if not isinstance(target, TargetPolicy):
        raise TypeError("Cody-Waite proof requires a TargetPolicy")
    if target.architecture not in {"bh", "wh"}:
        raise TargetLatticeProofError("two-FMA Cody-Waite coordinate theorem requires BH or WH")
    if target.intermediate_precision != "fp32":
        raise TargetLatticeProofError("two-FMA Cody-Waite coordinate theorem requires FP32 intermediates")
    values = (
        source_lower,
        source_upper,
        inverse_ln2,
        negative_ln2_hi,
        negative_ln2_lo,
    )
    if not all(math.isfinite(item) for item in values):
        raise TargetLatticeProofError("two-FMA Cody-Waite coordinate theorem requires finite operands")
    if source_lower > source_upper or inverse_ln2 <= 0.0:
        raise TargetLatticeProofError("two-FMA Cody-Waite coordinate theorem has invalid bounds/constants")

    maximum_source = max(abs(source_lower), abs(source_upper))
    if maximum_source > 128.0:
        raise TargetLatticeProofError("two-FMA Cody-Waite coordinate theorem requires |source| <= 128")

    quotient_magnitude = maximum_source * inverse_ln2
    quotient_product_error = 0.125 * _fp32_ulp_upper(quotient_magnitude)
    quotient_add_error = 0.5 * _fp32_ulp_upper(quotient_magnitude + quotient_product_error)
    quotient_error = quotient_product_error + quotient_add_error
    maximum_k = math.ceil(quotient_magnitude + quotient_error + 0.5)

    split_ln2 = -(negative_ln2_hi + negative_ln2_lo)
    inverse_split_mismatch = abs((1.0 / inverse_ln2) - split_ln2)
    ideal_coordinate = 0.5 / inverse_ln2 + quotient_error / inverse_ln2 + maximum_k * inverse_split_mismatch

    high_product_error = 0.125 * _fp32_ulp_upper(maximum_k * abs(negative_ln2_hi))
    high_sum_magnitude = ideal_coordinate + maximum_k * abs(negative_ln2_lo) + high_product_error
    high_add_error = 0.5 * _fp32_ulp_upper(high_sum_magnitude)
    high_error = high_product_error + high_add_error

    if negative_ln2_lo == 0.0:
        low_product_error = 0.0
        low_add_error = 0.0
    else:
        low_product_error = 0.125 * _fp32_ulp_upper(maximum_k * abs(negative_ln2_lo))
        low_sum_magnitude = ideal_coordinate + high_error + low_product_error
        low_add_error = 0.5 * _fp32_ulp_upper(low_sum_magnitude)
    coordinate_bound = _fp32_upward(ideal_coordinate + high_error + low_product_error + low_add_error)

    return CodyWaiteLn2CoordinateCertificate(
        target_sha256=target.sha256,
        source_lower=float(source_lower),
        source_upper=float(source_upper),
        maximum_integer_magnitude=maximum_k,
        quotient_error_bound=quotient_error,
        inverse_split_mismatch=inverse_split_mismatch,
        coordinate_lower=-coordinate_bound,
        coordinate_upper=coordinate_bound,
        proof_rules=(
            ("typed_cody_waite_ln2_unsplit_dataflow" if negative_ln2_lo == 0.0 else "typed_cody_waite_ln2_dataflow"),
            "bh_wh_product_preadd_error_at_most_eighth_ulp",
            "bh_wh_add_round_to_nearest_even",
            "round_nearest_residual_at_most_half",
            "rounded_inverse_ln2_mismatch_bounded",
            ("rounded_ln2_mismatch_bounded" if negative_ln2_lo == 0.0 else "rounded_split_ln2_mismatch_bounded"),
            "cody_waite_coordinate_outward_fp32",
        ),
    )


def certify_bh_round_even_residual_lattice(
    target: TargetPolicy,
) -> RoundEvenResidualLatticeCertificate:
    """Certify nonzero-normal round-even residuals, or fail closed.

    The theorem deliberately accepts only the currently emitted Blackhole
    primitive over DAZ BF16/FP32 inputs with FP32 intermediates.  A caller must
    separately establish that its operand is the target input or a
    lattice-preserving transform such as ``abs(input)``.
    """
    if not isinstance(target, TargetPolicy):
        raise TypeError("round-even lattice proof requires a TargetPolicy")
    if target.architecture != "bh":
        raise TargetLatticeProofError("round-even residual lattice theorem is Blackhole-only")
    if target.precision not in {"bf16", "fp32"}:
        raise TargetLatticeProofError("round-even residual lattice theorem requires BF16 or FP32 input")
    if target.intermediate_precision != "fp32":
        raise TargetLatticeProofError("round-even residual lattice theorem requires FP32 intermediates")
    if not target.input_daz:
        raise TargetLatticeProofError("round-even residual lattice theorem requires input DAZ ownership")
    if target.rounding_order != "post_round":
        raise TargetLatticeProofError("round-even residual lattice theorem requires post-round target policy")

    rules = [
        "bh_round_even_full_finite_fp32",
        "daz_excludes_nonzero_subnormal_inputs",
        "round_even_abs_lt_half_returns_zero",
        "round_even_sterbenz_subtraction_exact",
        "fp32_fractional_spacing_at_least_2^-24",
        "fp32_abs_ge_2^23_is_integral",
        "round_even_residual_nonzero_is_fp32_normal",
    ]
    if target.precision == "bf16":
        rules.append("bf16_input_lattice_is_fp32_subset")
    return RoundEvenResidualLatticeCertificate(
        target_sha256=target.sha256,
        input_precision=target.precision,
        intermediate_precision=target.intermediate_precision,
        minimum_nonzero_magnitude=float(np.finfo(np.float32).tiny),
        maximum_magnitude=0.5,
        fractional_spacing_floor=math.ldexp(1.0, -24),
        integral_magnitude_threshold=float(1 << 23),
        proof_rules=tuple(rules),
    )


@dataclass(frozen=True)
class PureUlpAcceptanceIntervals:
    """Fitter-side target-output intervals with an exact metric predicate.

    The numeric endpoints are solver hints.  The authoritative acceptance
    decision calls :func:`ttpoly.spec.units.ulp_error_pure`, so this module does
    not fork rounding, FTZ, or tester arithmetic from their existing owner.
    """

    golden: np.ndarray
    inputs: np.ndarray | None
    precision: str
    max_pure_ulp: float
    flush_order: str
    lower: np.ndarray
    upper: np.ndarray

    def errors(self, candidate) -> np.ndarray:
        from . import units

        return units.ulp_error_pure(
            self.golden,
            candidate,
            self.precision,
            inputs=self.inputs,
            flush_order=self.flush_order,
        )

    def accepts(self, candidate, *, strict: bool = False) -> np.ndarray:
        error = self.errors(candidate)
        comparison = error < self.max_pure_ulp if strict else error <= self.max_pure_ulp
        return np.isfinite(error) & comparison

    def summary(self, candidate, *, strict: bool = False) -> dict:
        error = self.errors(candidate)
        accepted = self.accepts(candidate, strict=strict)
        finite = error[np.isfinite(error)]
        accepted_count = int(np.count_nonzero(accepted))
        return {
            "points": int(error.size),
            "accepted": accepted_count,
            "violations": int(error.size - accepted_count),
            "max_pure_ulp": float(np.max(finite)) if finite.size else float("inf"),
            "mean_pure_ulp": (float(np.mean(finite)) if finite.size else float("inf")),
        }


def pure_ulp_acceptance_intervals(
    golden,
    *,
    precision: str,
    inputs=None,
    max_pure_ulp: float = 0.5,
    flush_order: str = "post_round",
) -> PureUlpAcceptanceIntervals:
    """Build pure-ULP output constraints without weakening final acceptance."""
    from . import units

    precision = str(precision).lower()
    units.Precision.coerce(precision)
    limit = float(max_pure_ulp)
    if not math.isfinite(limit) or limit < 0.0:
        raise TargetLatticeProofError("max_pure_ulp must be a finite non-negative value")
    truth = np.atleast_1d(np.asarray(golden, dtype=np.float64))
    input_array = None
    if inputs is not None:
        input_array = np.atleast_1d(np.asarray(inputs, dtype=np.float32))
        if input_array.shape != truth.shape:
            raise TargetLatticeProofError("inputs and golden must have identical shapes")

    # These are linear-solver hints only.  The final predicate above remains
    # the tester-exact FP32 quotient from units.ulp_error_pure.
    rounded = units._downcast_ftz(truth, precision, True).astype(np.float64)
    spacing = units.ulp_spacing_numpy(np.abs(rounded).astype(np.float32), precision).astype(np.float64)
    center = truth.copy()
    if flush_order == "post_round":
        center[np.isfinite(center) & (rounded == 0.0)] = 0.0
    else:
        center[np.isfinite(center) & (np.abs(center) < units.MIN_NORMAL)] = 0.0
    radius = spacing * limit
    lower = center - radius
    upper = center + radius
    undefined = ~np.isfinite(truth) | ~np.isfinite(spacing) | (spacing <= 0.0)
    lower[undefined] = np.nan
    upper[undefined] = np.nan
    return PureUlpAcceptanceIntervals(
        golden=truth,
        inputs=input_array,
        precision=precision,
        max_pure_ulp=limit,
        flush_order=str(flush_order),
        lower=lower,
        upper=upper,
    )


__all__ = [
    "BhBf16OriginHalfTieCertificate",
    "BhBf16OriginLinearFtzRneCellCertificate",
    "BhBf16OriginTangentReconstructionCertificate",
    "BhBf16RawClassPredicateCertificate",
    "BhBf16RawClassExcludedWordCertificate",
    "CodyWaiteLn2CoordinateCertificate",
    "RoundEvenResidualLatticeCertificate",
    "PureUlpAcceptanceIntervals",
    "TargetLatticeProofError",
    "bh_bf16_ieee_to_dst_u16",
    "bh_bf16_origin_half_tie_output_bits",
    "bh_bf16_origin_tangent_output_bits",
    "bh_bf16_raw_class_predicate_bits",
    "certify_bh_bf16_raw_class_predicate",
    "certify_bh_bf16_raw_class_excluded_word",
    "certify_bh_bf16_origin_half_tie",
    "certify_bh_bf16_origin_linear_ftz_rne_cell",
    "certify_bh_bf16_origin_tangent_reconstruction",
    "certify_bh_wh_cody_waite_ln2_coordinate",
    "certify_bh_round_even_residual_lattice",
    "pure_ulp_acceptance_intervals",
]
