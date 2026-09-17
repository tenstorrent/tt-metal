# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy contract an SFPU op declares, and the registry that holds them.

An op's accuracy budget is a *property of the op*, next to its other metadata, not a
magic number in a test body. That is what makes it reviewable, and what makes a kernel
improvement visible: you lower the number in the same PR that improves the kernel, and
the diff says so. It is the shape ttnn already uses — a ``GoldenComparisonConfig``
attached by the op's golden function, honoured by the comparator — expressed in this
harness's idiom as a table.

This lives in its own module rather than in ``sfpu_domains`` deliberately. The domain
registry is already 2500 lines and this is a separate concern with a different review
audience: a domain entry says what an op may be *fed*, a budget says how closely its
output must match. Nothing in ``sfpu_domains`` imports this; the drivers do.

**The budget is keyed on more than the op.** Approximation mode moves the error by orders
of magnitude, and the output format decides what is even visible: sub-ULP downward steps
at a LUT segment join are invisible in bfloat16 and large in float32, so a single number
across formats would be set by the bf16 measurement and never gate the interesting path.
The **input** format is in the key for the same reason from the other end: what reaches
the SFPU is already quantized by the unpack, and a ``Bfp8_b`` input carries 7 magnitude
bits behind a shared block exponent where an fp32 input carries 23. Measured on WH, 36 of
44 functional-sweep failures under a first cut of this table were ``Bfp8_b``-input
variants judged by a budget derived from fp32/fp16/bf16 inputs only. It also changes what
the numbers say: tanh fp32->fp32 at dest_acc=Yes agrees to 22 mantissa bits, which a cell
mixing all three input formats reported as 8.
Dest accumulation and the architecture are in the key for the same reason — a 16-bit Dest
truncates before the output rounding, and WH and BH SFPUs differ in available instructions
and therefore in kernel. Every dimension is optional: a key that leaves one unset matches
any value of it, and the most specific matching key wins.

**Enrolment is per op and per format, and incremental.** An op with no entry gets
:data:`TOLERANCE_CONTRACT` — today's ``atol``/``rtol`` plus PCC — and so does any request
for a format with no per-element ULP, which is how the block floats below ``Bfp8_b`` and
the MX formats keep their block-aware lattice compares while an enrolled op is gated on
the formats where a step count means something.

Numbers here are **measured**, not guessed. A budget derived from nothing is either so
loose it gates nothing or so tight it flakes; each entry carries the measurement it came
from in a trailing comment, with the architecture and date, so the next person can tell a
deliberate budget from a hopeful one.

**A budget binds every call site of the driver, and the driver has five stimulus
sources.** The recorded numbers come from the ``(op, format, dest_acc, approx_mode,
dimensions)`` sweep in ``test_eltwise_unary_sfpu`` -- not from
``accuracy/accuracy_harness.py``, which only ever runs the transcendentals plus
``Exp``/``Reciprocal``, omits ``Bfp8_b`` from its formats and has no dimensions axis, so
it cannot have produced them. That driver builds its stimulus from
``exclude_undefined(mathop, for_op_pipeline(...).spec_A)``.

The other four sources are the reason enrolling an op is not just "read the sweep":

* ``test_eltwise_unary_sfpu_edges`` passes its own ``edge_spec(...)`` with the plus/minus
  inf, NaN and signed-zero probes and the format extremes;
* ``_signbit``, ``_isinf_isnan`` and ``_threshold`` each build a hand-made spec and run
  ULP-gateable ``Float16_b``/``Float32``. None of the ops below reaches them today, but
  the ``ReluMin``/``ReluMax`` entries parked "for a later pass" *are* what the threshold
  sweep drives, with every third lane on the tie.

The ULP arm returns before both the tolerance gate and PCC, so on any of these a
zero-headroom budget has no backstop. Enrolling an op means measuring every sweep that
reaches it.

The edge sweep was run for the nine ops below and every variant it *generates* holds at
its enrolled budget. It is parametrized over ``input_output_formats([Float16_b,
Float32])``, so it never produces a ``Bfp8_b`` output -- and ``Floor``/``Ceil``/``Trunc``
carry a live ``max_ulp=0`` there through an unrestricted ``DEFAULT``, unlike
``Abs``/``Neg``/``Square``, which have a ``Bfp8_b`` carve-out. That variant has no edge
measurement and no backstop; extending the edge sweep to ``Bfp8_b`` would close it.
  wh: edges 42 passed / 6 skipped for Abs/Neg/Identity/Floor/Ceil/Trunc and
      7 passed / 17 skipped for Square/SigmoidAppx/GeluAppx, 2026-09-17
      (Float16_b and Float32 outputs only -- see above)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import has_ulp_gate

#: The architecture every measured budget in this table came from. An op resolves to the
#: tolerance metric anywhere else until the sweep has been re-run there.
MEASURED_ARCH = ChipArchitecture.WORMHOLE


class Metric(Enum):
    """Which gate a contract is written against.

    A closed two-member set, so there is no third gate to fall through to and no string
    spelling of one: a contract either replaces the tolerance check with a step count or
    it does not. It does *not* make a wrong value unrepresentable on its own -- Python
    does not enforce the annotation, so ``metric="ulp"`` still constructs -- which is
    what :meth:`AccuracyContract.__post_init__` is for. Every dimension of
    :class:`BudgetKey` is a real enum for the same reason, and guarded the same way.
    """

    ULP = "ulp"
    TOLERANCE = "tolerance"


@dataclass(frozen=True)
class AccuracyContract:
    """How closely one op's output must match its golden, and by which metric.

    ``metric=Metric.ULP`` means the verdict is "every element within *max_ulp*
    representable steps", with the tolerance and PCC checks skipped.
    ``metric=Metric.TOLERANCE`` is the harness's historical gate. The two sets of fields
    are mutually exclusive and :meth:`__post_init__` enforces that, so a half-converted
    entry cannot sit in the table looking plausible.

    The enum members, not the strings ``"ulp"``/``"tolerance"``: ``Metric`` is a bare
    ``Enum``, so ``"ulp" != Metric.ULP`` and an entry written with the string would have
    fallen through to the tolerance arm.
    """

    metric: Metric = Metric.ULP
    max_ulp: Optional[int] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    near_zero_atol: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            # The annotation is not a check: Python does not enforce enum membership, so
            # `metric="ulp"` -- or any typo -- previously took the `else` branch below and
            # became a *tolerance* contract, silently switching off the intended gate.
            raise ValueError(
                f"metric must be a Metric member, got {self.metric!r}; use "
                f"{Metric.ULP} or {Metric.TOLERANCE}"
            )
        for name in ("atol", "rtol", "near_zero_atol"):
            value = getattr(self, name)
            if value is not None and value < 0:
                # passed_test applies an override only `if custom_atol is not None and
                # custom_atol >= 0`, so a negative atol/rtol read in the table as a
                # declared tolerance and then silently ran against the per-format
                # default. A negative near_zero_atol is a silently inert floor.
                raise ValueError(f"{name} must not be negative, got {value}")
        if self.metric == Metric.ULP:
            if self.max_ulp is None:
                raise ValueError("a ulp contract needs max_ulp")
            if self.max_ulp < 0:
                raise ValueError(f"max_ulp must not be negative, got {self.max_ulp}")
            if self.atol is not None or self.rtol is not None:
                raise ValueError(
                    "a ulp contract replaces the tolerance gate, so atol/rtol would be "
                    "silently ignored; use near_zero_atol for the near-zero floor"
                )
        else:
            if self.max_ulp is not None or self.near_zero_atol is not None:
                raise ValueError(
                    "max_ulp and near_zero_atol belong to the ulp metric; set "
                    f"metric={Metric.ULP} to use them"
                )

    def passed_test_kwargs(self) -> Dict[str, Any]:
        """The contract as keyword arguments for ``passed_test``.

        Keeps the driver's call site to one ``**`` expansion, so switching an op between
        metrics is a registry edit and never a driver edit.
        """
        if self.metric == Metric.ULP:
            return {"max_ulp": self.max_ulp, "near_zero_atol": self.near_zero_atol}
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}


#: What an unenrolled op, or an enrolled op on a format with no per-element ULP, is
#: judged by: the per-format ``atol``/``rtol`` defaults plus ``PCC > 0.99``, exactly as
#: before. ``atol=None``/``rtol=None`` let ``passed_test`` use its own table.
TOLERANCE_CONTRACT = AccuracyContract(metric=Metric.TOLERANCE)


#: The enum each :class:`BudgetKey` dimension must be a member of. Kept beside the
#: dataclass rather than read off ``__annotations__``, which would hand back
#: ``Optional[...]`` and need unwrapping; :func:`test_every_budget_key_field_is_guarded`
#: asserts the two stay in step.
_BUDGET_KEY_TYPES: Dict[str, type] = {
    "approx_mode": ApproximationMode,
    "output_format": DataFormat,
    "dest_acc": DestAccumulation,
    "arch": ChipArchitecture,
}


@dataclass(frozen=True)
class BudgetKey:
    """Which variants of an op a contract applies to. Unset field == any value.

    A key with more fields set is more specific and wins over a broader one. Two keys that
    match the same variant with equal specificity are an authoring error, not a
    tie-break: :func:`validate_registry` rejects them rather than letting the table's
    iteration order decide a budget.
    """

    approx_mode: Optional[ApproximationMode] = None
    input_format: Optional[DataFormat] = None
    output_format: Optional[DataFormat] = None
    dest_acc: Optional[DestAccumulation] = None
    arch: Optional[ChipArchitecture] = None

    def __post_init__(self) -> None:
        # The annotation is not a check, exactly as for `AccuracyContract.metric`. All
        # four of these are bare `Enum`s, so `DestAccumulation.No.value is False` and
        # `ChipArchitecture.WORMHOLE.value == "wormhole"` never compare equal to their
        # members -- which makes `BudgetKey(dest_acc=True)` or `BudgetKey(arch="wormhole")`
        # *inert*: counted as set by `specificity`, matched by nothing in `matches()`.
        # The budget it declares then gates nothing, `budget_table()` sees no duplicate,
        # `validate_registry()` sees no tie, and `describe()` renders it identically to
        # the correct key, because `ChipArchitecture.__str__` returns `.value`. It fails
        # looser than declared, which is the drift this registry exists to stop.
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            expected = _BUDGET_KEY_TYPES[f.name]
            if not isinstance(value, expected):
                raise ValueError(
                    f"BudgetKey.{f.name} must be a {expected.__name__} member or None, "
                    f"got {value!r}; a non-member is counted as set and matches nothing"
                )

    @property
    def specificity(self) -> int:
        return sum(getattr(self, f.name) is not None for f in fields(self))

    def matches(
        self,
        *,
        approx_mode: Optional[ApproximationMode],
        input_format: Optional[DataFormat],
        output_format: Optional[DataFormat],
        dest_acc: Optional[DestAccumulation],
        arch: Optional[ChipArchitecture],
    ) -> bool:
        """Whether this key covers the given variant.

        A dimension the *caller* leaves unset can only match a wildcard: if a key is
        specific about ``dest_acc`` and the caller does not know it, the key does not
        apply. Guessing would hand back a budget measured for the other setting.
        """
        query = {
            "approx_mode": approx_mode,
            "input_format": input_format,
            "output_format": output_format,
            "dest_acc": dest_acc,
            "arch": arch,
        }
        for name, asked in query.items():
            wanted = getattr(self, name)
            if wanted is not None and wanted != asked:
                return False
        return True

    def describe(self) -> str:
        set_fields = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"BudgetKey({', '.join(set_fields)})" if set_fields else "DEFAULT"


#: Matches every variant of an op. The right key for a budget that does not yet vary.
DEFAULT = BudgetKey()


# ─────────────────────────────────────────────────────────────────────────────
# The registry
#
# Enrolled first: the ops whose results are exact by construction. They are the flakiness
# canaries for the whole metric — a 0-step budget on Abs cannot be wrong about the kernel,
# so if one of these fails, the golden or the datapath moved, not the SFPU. Getting them
# stable across a week of nightlies is what earns the right to enrol a transcendental.
#
# Every number below was measured on the sweep the harness already runs, not chosen: 967
# (op, format, dest_acc, approx_mode, dimensions) variants on Wormhole, 2026-09-16, with
# the budget temporarily raised past anything reachable so each variant reported instead
# of failing. Each entry carries what it came from. Re-measure on Blackhole before
# trusting these there; the keys have an `arch` dimension for the day they diverge.
#
# MathOperation.Relu is absent because _NON_SFPU_UNARY_OPS subtracts it from
# sfpu_unary_ops(): it is packer-applied via STACC_RELU and is not a member of SfpuType,
# so it will not compile through the unary driver at all. It *does* have an
# _OP_DOMAIN_REGISTRY entry, so that is not the blocker. ReluMax/ReluMin take a threshold
# operand and are left for a later pass.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Bfp8_b: gated in bfloat16 step space by helpers.ulp, and measured to be dominated by
# block quantization rather than by the op.
#
# The bf16 proxy is a real per-element criterion only while the block exponent is the one
# bf16 would have used. It is not, on this stimulus: Abs and Neg -- which cannot be wrong,
# they clear and flip a sign bit -- measure a maximum of 15616 steps, and the worst lane
# is `result 0.0 vs golden 0.062`. That is a small element in a block whose maximum is
# large, quantized to zero by the shared exponent, exactly as designed. 15616 steps is
# also two orders of magnitude past bf16's 128-step meaningful ceiling.
#
# So a step budget cannot gate these ops on Bfp8_b, and raising it until they pass would
# gate nothing at all. A near_zero_atol floor would absorb it -- those are near-zero lanes
# by construction -- but the floor would then be doing all the work below the top two
# decades, which is the flat-tolerance gate again under a new name. Recorded here and
# parked on the tolerance metric, which keeps the existing block-aware lattice compare in
# utils.py: already the stronger, block-aware criterion.
#
# Floor/Ceil/Trunc are the exception and are enrolled on Bfp8_b: their results are
# integers, which a shared exponent represents exactly, and they measure 0 steps.
#   wh: Abs/Neg max 15616 ULP, Square max 17664, Floor/Ceil/Trunc max 0, 2026-09-16
# ─────────────────────────────────────────────────────────────────────────────
_BFP8_B_QUANTIZATION_DOMINATES = AccuracyContract(metric=Metric.TOLERANCE)

#: The two coarse 3-segment LUT approximations, which share one number because they share
#: one cause. Named once so a retune cannot move SigmoidAppx and leave GeluAppx behind.
_COARSE_LUT_TOLERANCE = AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05)

#: The domain that makes a 0-step Bfp8_b budget legitimate for the integer-valued ops:
#: every block maximum stays below 2**7, so the shared exponent represents their results
#: exactly. Asserted against _OP_DOMAIN_REGISTRY by the host tests, because it is a
#: property of the *stimulus*, not of the format.
BFP8_B_EXACT_INTEGER_DOMAIN = 128.0


def budget_table(
    *entries: Tuple[BudgetKey, AccuracyContract]
) -> Dict[BudgetKey, AccuracyContract]:
    """One op's table, built from pairs so a repeated key is an error.

    ``BudgetKey`` is frozen, so two identical keys in a dict literal are equal and
    hash-equal and Python silently keeps the later contract -- which means
    :func:`validate_registry` received an already-deduplicated table and the tie-raise in
    :func:`resolve_contract` could never fire for the duplicate that the
    :class:`BudgetKey` docstring promises to reject. It saw one entry, not two. A copy-pasted
    ``BudgetKey(output_format=Float16_b)`` replacing a measured budget with a broader one
    would have failed nothing.
    """
    table: Dict[BudgetKey, AccuracyContract] = {}
    for key, contract in entries:
        if key in table:
            raise ValueError(
                f"duplicate budget key {key.describe()}: a dict literal would have kept "
                "only the later contract, and no guard downstream can see the first one"
            )
        table[key] = contract
    return table


def registry(
    *entries: Tuple[MathOperation, Dict[BudgetKey, AccuracyContract]]
) -> Dict[MathOperation, Dict[BudgetKey, AccuracyContract]]:
    """The whole table, built from pairs so a repeated op is an error.

    The same hazard as :func:`budget_table`, one level up and harder to see: the ops sit
    10-20 lines apart across three comment-delimited sections, so a re-added
    ``MathOperation.Square:`` drops the earlier entry silently. Nothing downstream can
    catch it -- :func:`validate_registry` iterates the already-deduplicated dict, the
    ``len(set(ops)) == len(ops)`` check in the host tests is tautological over dict keys,
    and pylint's ``duplicate-key`` is off (pre-commit runs it with ``--disable=all``).
    """
    table: Dict[MathOperation, Dict[BudgetKey, AccuracyContract]] = {}
    for op, contracts in entries:
        if op in table:
            raise ValueError(
                f"duplicate registry entry for {op.name}: a dict literal would have kept "
                "only the later table, and no guard downstream can see the first one"
            )
        table[op] = contracts
    return table


def _exact_everywhere() -> Dict[BudgetKey, AccuracyContract]:
    """A fresh 0-step table, for the ops measured exact on every output format.

    A factory rather than one dict literal aliased three ways, so a retune of one op
    cannot silently move the others.
    """
    return budget_table((DEFAULT, AccuracyContract(max_ulp=0)))


_SFPU_ACCURACY_BUDGET: Dict[MathOperation, Dict[BudgetKey, AccuracyContract]] = registry(
    # ── Exact everywhere, including Bfp8_b ───────────────────────────────────
    # Floor/Ceil/Trunc land on a representable integer below the format's integer limit
    # and are the identity above it.
    #
    # They are also the only ops enrolled on Bfp8_b, and the reason is narrower than
    # "integers are exact in a block float". A shared exponent does not represent
    # integers exactly in general: the in-block step scales with the block maximum, so an
    # integer is exact only while every block maximum stays below 2**7 = 128. That holds
    # here because _OP_DOMAIN_REGISTRY bounds these three to uniform(-10, 10) --
    # BFP8_B_EXACT_INTEGER_DOMAIN below pins it -- and not because of anything about the
    # format. An integer-valued op with a wider domain would fail, the same mechanism
    # that takes Abs and Neg to 15616 steps in the note above.
    #   wh: 0 ULP, 156 variants each, all four output formats x both dest_acc, 2026-09-16
    (MathOperation.Floor, _exact_everywhere()),
    (MathOperation.Ceil, _exact_everywhere()),
    (MathOperation.Trunc, _exact_everywhere()),
    # ── Sign-bit and select: exact in fp32, one step in the 16-bit formats ───
    # Abs clears the sign bit, Neg flips it, Identity copies; there is no arithmetic to
    # round. fp32 out is bit-exact. The 16-bit outputs are one step off on part of the
    # sweep, and it is the *pack* path rather than the op: the same single step shows up
    # for all three of these ops, and mostly at dest_acc=Yes, where the value is rounded
    # fp32 -> 16-bit once at pack instead of being truncated in a 16-bit Dest first.
    # A step budget is what makes that visible at all; atol=0.05 is ~6 bf16 steps.
    #
    # The budget is the measured maximum with no headroom added, deliberately. These
    # results are exact by construction, so any movement is a real signal and should
    # fail rather than be absorbed. They are the flakiness canaries for the metric: if
    # one of these starts failing, the golden or the datapath moved, not the kernel.
    # Each recorded number is a maximum over the *input* pipelines too: BudgetKey has no
    # input_format axis, and input_output_formats() is a full cross product, so the "80
    # variants" below is 2 outputs x 5 inputs x 2 approx x 2 dest_acc x 2 dimensions. The
    # 1-step pack allowance therefore also binds Float16_b->Float16_b and Bfp8_b->Float16_b,
    # which are bit-exact by construction for a sign-bit op -- so "zero headroom" holds for
    # the pipeline that set the maximum and is slack for the others. The axis is excluded
    # here because no enrolled cell has a *tighter* per-input number worth keying on yet;
    # P3 adds input_format to BudgetKey for the transcendentals, where the input pipeline
    # does move the measurement.
    #   wh: Abs/Neg max 0 ULP on Float32 (32 variants), 1 ULP on Float16/Float16_b
    #       (80 variants); Identity max 0 on Float32, 1 on Float16_b (4), 2026-09-16
    (
        MathOperation.Abs,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=1)),
            (BudgetKey(output_format=DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (BudgetKey(output_format=DataFormat.Bfp8_b), _BFP8_B_QUANTIZATION_DOMINATES),
        ),
    ),
    (
        MathOperation.Neg,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=1)),
            (BudgetKey(output_format=DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (BudgetKey(output_format=DataFormat.Bfp8_b), _BFP8_B_QUANTIZATION_DOMINATES),
        ),
    ),
    # Identity is keyed per format rather than through a DEFAULT, because unlike Abs/Neg
    # it was never in BROAD_SWEEP_OPS: only BROAD_FORMATS/FORMATS_BFP4_B reach a Float16
    # output, so fp16 was never measured for it at all. Square measured 1 step on
    # Float16_b and 4 on Float16, so fp16 is not safely interpolated from bf16 here -- an
    # unmeasured format falls back to the tolerance metric instead.
    (
        MathOperation.Identity,
        budget_table(
            (BudgetKey(output_format=DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (BudgetKey(output_format=DataFormat.Float16_b), AccuracyContract(max_ulp=1)),
        ),
    ),
    # ── One multiply, and one open question ─────────────────────────────────
    # x*x has rounding slack the sign-bit ops do not: the golden evaluates in float64 and
    # applies the Dest and output roundings, the hardware rounds in the datapath, and the
    # two can differ where the exact product falls on a tie. 1 step in bf16 and 4 in fp16
    # are consistent with that.
    #
    # Float32 is NOT enrolled, and the measurement is why: 65536 steps at dest_acc=No,
    # 32768 at dest_acc=Yes, on almost every element rather than a few. 65536 is exactly
    # 2**16 -- one step of a 16-bit Dest lattice expressed in fp32 units -- so the
    # dest_acc=No number is a single step of the *real* output lattice and says the golden
    # and the hardware round that step differently. 32768 = 2**15 at dest_acc=Yes has no
    # such explanation: with an fp32 Dest the product agrees to only ~8 mantissa bits.
    # Neither is something a budget should paper over, and attributing them (kernel, or
    # the golden's rounding model) is its own change. Parked on the tolerance metric so
    # the number is recorded rather than blessed.
    #   wh: Float16_b max 1 ULP (40 variants), Float16 max 4 (40),
    #       Float32 max 65536 @ dest_acc=No / 32768 @ dest_acc=Yes (32), 2026-09-16
    (
        MathOperation.Square,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=4)),
            (BudgetKey(output_format=DataFormat.Float16_b), AccuracyContract(max_ulp=1)),
            (
                BudgetKey(output_format=DataFormat.Float32),
                AccuracyContract(metric=Metric.TOLERANCE),
            ),
            (BudgetKey(output_format=DataFormat.Bfp8_b), _BFP8_B_QUANTIZATION_DOMINATES),
        ),
    ),
    # ── Still on the tolerance metric, moved here from the test body ─────────
    # These were CUSTOM_TOLERANCES in test_eltwise_unary_sfpu: a coarse 3-segment LUT
    # whose absolute error peaks near the knees, carrying atol=0.13 so the sweep passes.
    # They are the clearest argument for this whole mechanism -- that number makes the
    # test blind to a 10x regression anywhere else in the domain, and equally blind to the
    # improvement a LUT retune produces. They keep the tolerance metric until there is a
    # measured step budget to replace it with, but the number now sits next to the op
    # instead of in a dict in a test file.
    (MathOperation.SigmoidAppx, budget_table((DEFAULT, _COARSE_LUT_TOLERANCE))),
    (MathOperation.GeluAppx, budget_table((DEFAULT, _COARSE_LUT_TOLERANCE))),
    # ── Transcendentals, enrolled from the accuracy sweep (P3) ──────────────
    #
    # Emitted by accuracy/emit_budget.py from 1,142,784 sweep points on Wormhole across
    # 19 ops x 3 input x 3 output formats x approx x fast x dest_acc, deterministic
    # `ramp`, seed 0. Regenerate with:
    #
    #     ../.venv/bin/python -m accuracy.emit_budget --arch wh --formats fp32,bf16
    #
    # The `~N mantissa bits` figure is what makes these comparable across formats: one
    # bfloat16 step and 65536 float32 steps are the same physical accuracy, because
    # float32 counts finer steps. A five-figure float32 budget is a narrow result in a
    # wide container, not a sloppy gate, and it drops by a power of two per bit gained if
    # a kernel ever starts using the resolution it was given.
    #
    # Every key names **both** formats, and that is deliberate even where the paths agree.
    # The functional suite runs Bfp8_b inputs and this sweep does not, so a key without an
    # input format would extend a budget to a much coarser path that was never measured;
    # and this run emits fp32 and bf16 outputs only, so a key without an output format
    # would match the unmeasured Float16 and Bfp8_b outputs. Those variants fall back to
    # the tolerance metric on their own, which is what makes enrolment incremental.
    #
    # A variant left on the tolerance metric is scoped to the variant that was measured to
    # need it, not to the whole format pair -- see _NOT_PREDICTED_BY_SWEEP in the emitter.
    # Gelu fp32->fp32 diverges from the sweep at dest_acc=Yes and Log fp32->fp32 at
    # dest_acc=No, so each keeps its ULP budget on the other setting instead of merging
    # every approximation and destination variant of the pair into one wildcard.
    #
    # A budget of 1 against a comment reading "max 0 ULP" is not a typo: a measured zero
    # is floored to MIN_MEASURED_BUDGET, because "no error observed on this stimulus" is
    # not "no error possible" and a finite sample cannot assert exactness. Each such entry
    # now says so in its own comment too. An op that is exact by *construction* is a
    # different claim, and those budgets are hand-written above.
    #
    # near_zero_atol appears where the sweep found the hardware returning exactly 0
    # against a small non-zero reference -- gelu(-4.18), erfinv(0.0005) and approximate
    # exp(-9.68) all do it -- a five-figure step count that describes nothing about the
    # kernel's accuracy elsewhere. The floor holds those lanes so the budget stays tight.
    #
    # The floors are small, and deliberately so: the gate bounds its near-zero band
    # absolutely as well as relatively (at near_zero_atol / NEAR_ZERO_FRACTION), so a
    # floor only ever covers lanes whose magnitude is within 100x of the error being
    # forgiven. The emitter models that same split when it derives these numbers -- it
    # has to, or it would derive a floor from a lane the gate then refuses to rescue,
    # which is what made hardsigmoid fp32->fp32 dest_acc=Yes emit max_ulp=10 and then
    # fail the functional suite at 128 steps. It is now max_ulp=1024 with no floor,
    # which is the honest reading of the same measurement under the bounded band.
    (
        MathOperation.Acosh,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 76% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 88% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 1.0, 76% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 75% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32598 ULP, p99.9 32526.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40658),
            #   wh: max 1 ULP, p99.9 1.0, 75% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 40960 ULP, p99.9 40960.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=51200),
            #   wh: max 4094 ULP, p99.9 4093.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5117),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Asinh,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 85% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 1.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 81% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32706 ULP, p99.9 32703.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40879),
            #   wh: max 1 ULP, p99.9 1.0, 81% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 40960 ULP, p99.9 40960.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=51200),
            #   wh: max 4087 ULP, p99.9 4065.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5082),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Atanh,
        {
            #   wh: max 131072 ULP, p99.9 131072.0, 71% exact, ~6 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=163840),
            #   wh: max 2 ULP, p99.9 2.0, 77% exact, ~22 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=3),
            #   wh: max 2 ULP, p99.9 2.0, 71% exact, ~6 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=3),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 131072 ULP, p99.9 131072.0, 72% exact, ~6 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=163840),
            #   wh: max 32673 ULP, p99.9 32673.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40842),
            #   wh: max 2 ULP, p99.9 2.0, 72% exact, ~6 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=3),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: not enrolled -- functional draw reaches 57344 steps, ramp-derived 51200 (4096 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 93% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Celu,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 9 ULP, p99.9 9.0, 90% exact, ~20 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=12),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32752 ULP, p99.9 32501.7, 50% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40628),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32768 ULP, p99.9 32768.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4085 ULP, p99.9 4069.9, 50% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5088),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Cos,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 72% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 72% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=1),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32740 ULP, p99.9 32740.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40925),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32740 ULP, p99.9 32740.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40925),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32768 ULP, p99.9 32768.0, 15% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4093 ULP, p99.9 4092.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5115),
            #   wh: max 32768 ULP, p99.9 32768.0, 15% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4093 ULP, p99.9 4092.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5115),
            #   wh: max 1 ULP, p99.9 1.0, 93% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Elu,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 9 ULP, p99.9 9.0, 90% exact, ~20 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=12),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32752 ULP, p99.9 32501.7, 50% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40628),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32768 ULP, p99.9 32768.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4085 ULP, p99.9 4069.9, 50% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5088),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Erfinv,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 56% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962592769 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920, near_zero_atol=0.000534),
            #   wh: max 26321 ULP, p99.9 25227.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962639709 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=31534, near_zero_atol=0.000536),
            #   wh: max 1 ULP, p99.9 1.0, 56% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 14689 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2, near_zero_atol=0.000534),
            #   wh: max 1 ULP, p99.9 1.0, 96% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 14690 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2, near_zero_atol=0.000536),
            #   wh: max 65536 ULP, p99.9 65536.0, 56% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962658305 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920, near_zero_atol=0.000536),
            #   wh: max 50188 ULP, p99.9 50188.0, 0% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962658305 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=62735, near_zero_atol=0.000536),
            #   wh: max 1 ULP, p99.9 1.0, 56% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; 84 near-zero pts reach 14690 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2, near_zero_atol=0.000536),
            #   wh: max 24576 ULP, p99.9 24576.0, 56% exact, ~8 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962641921 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=30720, near_zero_atol=0.000536),
            #   wh: max 27027 ULP, p99.9 24417.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17; 42 near-zero pts reach 962641921 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=30522, near_zero_atol=0.000536),
            #   wh: max 1 ULP, p99.9 1.0, 53% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; 84 near-zero pts reach 14690 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2, near_zero_atol=0.000536),
        },
    ),
    (
        MathOperation.Exp,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 81% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 393216 ULP, p99.9 393216.0, 11% exact, ~4 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 81% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 6 ULP, p99.9 6.0, 16% exact, ~4 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 65536 ULP, p99.9 65536.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32767 ULP, p99.9 32752.0, 7% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40940),
            #   wh: max 393216 ULP, p99.9 393216.0, 14% exact, ~4 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 6 ULP, p99.9 6.0, 16% exact, ~4 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 57344 ULP, p99.9 57344.0, 14% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=71680),
            #   wh: max 8181 ULP, p99.9 7049.7, 0% exact, ~10 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=8813),
            #   wh: max 939728897 ULP, p99.9 939594744.8, 1% exact, ~-7 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 79% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 14340 ULP, p99.9 14338.0, 6% exact, ~-7 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
        },
    ),
    (
        MathOperation.Exp2,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 86% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 1.0, 82% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 83% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32393 ULP, p99.9 31806.9, 29% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=39759),
            #   wh: max 1 ULP, p99.9 1.0, 83% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 57344 ULP, p99.9 57344.0, 16% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=71680),
            #   wh: max 4064 ULP, p99.9 4051.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5064),
            #   wh: max 1 ULP, p99.9 1.0, 80% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Gelu,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 94% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: not enrolled -- near-zero tail: functional reaches 19,474,047 steps (2048 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: not enrolled -- near-zero tail: functional reaches 19,474,047 steps (4096 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: not enrolled -- near-zero tail: functional abs error 7.49e-07 exceeds the ramp-derived 7.08e-07 floor, 148 steps against a 2-step budget (2048 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: not enrolled -- near-zero tail: functional abs error 7.49e-07 exceeds the ramp-derived 7.08e-07 floor, 148 steps against a 2-step budget (4096 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 65536 ULP, p99.9 65536.0, 94% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 57344 ULP, p99.9 57344.0, 0% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 164 near-zero pts reach 3932160 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=71680, near_zero_atol=5.59e-07),
            #   wh: max 1936392194 ULP, p99.9 1936392194.0, 20% exact, ~-8 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 1.0, 99% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 164 near-zero pts reach 60 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2, near_zero_atol=5.59e-07),
            #   wh: max 29549 ULP, p99.9 29549.0, 34% exact, ~-8 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 57344 ULP, p99.9 57344.0, 13% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 462 near-zero pts reach 939524097 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=71680, near_zero_atol=7.63e-05),
            #   wh: max 46656 ULP, p99.9 37126.2, 0% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 176 near-zero pts reach 3932160 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=46656, near_zero_atol=7.08e-07),
            #   wh: max 1956462594 ULP, p99.9 1941721278.4, 16% exact, ~-8 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 91% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 462 near-zero pts reach 14337 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2, near_zero_atol=7.63e-05),
            #   wh: max 1 ULP, p99.9 1.0, 93% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; 176 near-zero pts reach 60 steps and are held by the atol floor instead
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2, near_zero_atol=7.08e-07),
            #   wh: max 29855 ULP, p99.9 29630.2, 33% exact, ~-8 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
        },
    ),
    (
        MathOperation.Hardsigmoid,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 69% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1024 ULP, p99.9 62.5, 88% exact, ~13 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1024),
            #   wh: max 1 ULP, p99.9 1.0, 69% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 69% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32768 ULP, p99.9 32768.0, 42% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 1 ULP, p99.9 1.0, 69% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 8192 ULP, p99.9 8192.0, 69% exact, ~10 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=10240),
            #   wh: max 4096 ULP, p99.9 4096.0, 41% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5120),
            #   wh: max 1 ULP, p99.9 1.0, 61% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Log,
        {
            #   wh: not enrolled -- functional draw reaches 65536 steps where the ramp sees 1 (2048 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 96% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: not enrolled -- functional draw reaches 65536 steps where the ramp sees 1 (2048 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 32693 ULP, p99.9 32693.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40867),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: not enrolled -- functional draw reaches 57344 steps, ramp-derived 40960 (4096 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 94% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Log1p,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 97% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 93% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 1.0, 97% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 97% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32743 ULP, p99.9 32714.4, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40893),
            #   wh: max 1 ULP, p99.9 1.0, 97% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 49152 ULP, p99.9 48767.0, 12% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=60959),
            #   wh: not enrolled -- near-zero tail: functional reaches 938,672,129 steps (2048 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: not enrolled -- near-zero tail: functional reaches 14324 steps (4096 pts, 2026-09-17)
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(metric=Metric.TOLERANCE),
        },
    ),
    (
        MathOperation.Reciprocal,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 99% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1 ULP, p99.9 1.0, 90% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 65536 ULP, p99.9 65536.0, 59% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 1650 ULP, p99.9 1648.0, 1% exact, ~12 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2060),
            #   wh: max 1 ULP, p99.9 1.0, 99% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 1 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 59% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 65536 ULP, p99.9 65536.0, 98% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32639 ULP, p99.9 32639.0, 1% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40799),
            #   wh: max 65536 ULP, p99.9 65536.0, 57% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 33278 ULP, p99.9 33278.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=41598),
            #   wh: max 1 ULP, p99.9 1.0, 98% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 57% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 32768 ULP, p99.9 32768.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4094 ULP, p99.9 4091.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5114),
            #   wh: max 8192 ULP, p99.9 8192.0, 45% exact, ~10 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=10240),
            #   wh: max 5319 ULP, p99.9 5319.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=6649),
            #   wh: max 1 ULP, p99.9 1.0, 48% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Rsqrt,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 2 ULP, p99.9 2.0, 67% exact, ~22 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=3),
            #   wh: max 65536 ULP, p99.9 65536.0, 91% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 14745 ULP, p99.9 14726.6, 0% exact, ~9 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=18409),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 88% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32723 ULP, p99.9 32723.0, 1% exact, ~8 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40904),
            #   wh: max 65536 ULP, p99.9 65536.0, 92% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 43254 ULP, p99.9 43254.0, 0% exact, ~8 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=54068),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 92% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 32768 ULP, p99.9 32768.0, 12% exact, ~8 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4093 ULP, p99.9 4087.9, 0% exact, ~11 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5110),
            #   wh: max 49152 ULP, p99.9 40960.0, 12% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=51200),
            #   wh: max 18072 ULP, p99.9 17563.0, 0% exact, ~9 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=21954),
            #   wh: max 1 ULP, p99.9 1.0, 88% exact, ~7 mantissa bits, 16384 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Silu,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 90% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 2 ULP, p99.9 2.0, 76% exact, ~22 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=3),
            #   wh: max 1 ULP, p99.9 1.0, 90% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 65536 ULP, p99.9 65536.0, 91% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32764 ULP, p99.9 32763.7, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40955),
            #   wh: max 1 ULP, p99.9 1.0, 91% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 57344 ULP, p99.9 57344.0, 11% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=71680),
            #   wh: max 4095 ULP, p99.9 4087.8, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5110),
            #   wh: max 1 ULP, p99.9 1.0, 88% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Sin,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 81% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 81% exact, ~23 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=1),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32737 ULP, p99.9 32708.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40885),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32737 ULP, p99.9 32708.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40885),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32768 ULP, p99.9 32768.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4093 ULP, p99.9 4092.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5115),
            #   wh: max 32768 ULP, p99.9 32768.0, 13% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4093 ULP, p99.9 4092.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5115),
            #   wh: max 1 ULP, p99.9 1.0, 95% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Sqrt,
        {
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 89% exact, ~23 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 65536 ULP, p99.9 65536.0, 85% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 14746 ULP, p99.9 14731.6, 0% exact, ~9 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=18415),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 85% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~23 mantissa bits, 4096 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 32736 ULP, p99.9 32736.0, 3% exact, ~8 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40920),
            #   wh: max 65536 ULP, p99.9 65536.0, 85% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 47467 ULP, p99.9 47467.0, 0% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=59334),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 8192 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 1 ULP, p99.9 1.0, 85% exact, ~7 mantissa bits, 8192 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 32768 ULP, p99.9 32768.0, 14% exact, ~8 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=40960),
            #   wh: max 4095 ULP, p99.9 4092.0, 1% exact, ~11 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5115),
            #   wh: max 49152 ULP, p99.9 40960.0, 14% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=51200),
            #   wh: max 18828 ULP, p99.9 18713.0, 0% exact, ~9 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=23392),
            #   wh: max 1 ULP, p99.9 1.0, 87% exact, ~7 mantissa bits, 16384 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16, output_format=DataFormat.Float16_b
            ): AccuracyContract(max_ulp=2),
        },
    ),
    (
        MathOperation.Tanh,
        {
            #   wh: max 65536 ULP, p99.9 65536.0, 86% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 2 ULP, p99.9 1.0, 83% exact, ~22 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 2424832 ULP, p99.9 2405616.0, 0% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 86% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 37 ULP, p99.9 37.0, 31% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 65536 ULP, p99.9 65536.0, 86% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=81920),
            #   wh: max 32666 ULP, p99.9 32540.0, 0% exact, ~8 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=40675),
            #   wh: max 2424832 ULP, p99.9 2371584.0, 31% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 86% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 0 ULP, p99.9 0.0, 100% exact, ~7 mantissa bits, 2048 pts, 2026-09-17; measured 0, floored to 1 (a finite sample cannot assert exactness)
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=1),
            #   wh: max 37 ULP, p99.9 36.0, 31% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 49152 ULP, p99.9 49152.0, 19% exact, ~7 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
            ): AccuracyContract(max_ulp=61440),
            #   wh: max 4093 ULP, p99.9 4088.0, 0% exact, ~11 mantissa bits, 2048 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.Yes,
            ): AccuracyContract(max_ulp=5110),
            #   wh: max 2418176 ULP, p99.9 2406400.0, 10% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 419430-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float32,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
            #   wh: max 1 ULP, p99.9 1.0, 88% exact, ~7 mantissa bits, 4096 pts, 2026-09-17
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.No,
            ): AccuracyContract(max_ulp=2),
            #   wh: max 37 ULP, p99.9 37.0, 32% exact, ~2 mantissa bits, 4096 pts, 2026-09-17; past the 6-step point where a budget stops being tighter than the tolerance it replaces, so tolerance
            BudgetKey(
                input_format=DataFormat.Float16,
                output_format=DataFormat.Float16_b,
                approx_mode=ApproximationMode.Yes,
            ): AccuracyContract(metric=Metric.TOLERANCE),
        },
    ),
)


def accuracy_contract(
    op: MathOperation,
    *,
    output_format: DataFormat,
    arch: ChipArchitecture,
    input_format: Optional[DataFormat] = None,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
) -> AccuracyContract:
    """The contract for one op variant, or :data:`TOLERANCE_CONTRACT` if it has none.

    Falling back rather than raising is what makes enrolment incremental: an unenrolled
    op, and an enrolled op asked about a format with no per-element ULP, both keep the
    behaviour they have today.

    *arch* is required, unlike the other three dimensions. Those may be left unset and
    then match only a wildcard key, which is the rule :meth:`BudgetKey.matches` documents:
    guessing would hand back a budget measured for the other setting. Architecture is the
    one dimension where the numbers explicitly do not transfer, so defaulting it to
    ``None`` would have resolved an unknown chip straight against the Wormhole table --
    the inverse of that rule, in the dimension that can least afford it. A caller that
    forgets the keyword now fails at the call rather than silently reinstating the
    Blackhole problem this gate exists to close.
    """
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if table is None:
        return TOLERANCE_CONTRACT

    contract = resolve_contract(
        table,
        label=op.name,
        approx_mode=approx_mode,
        input_format=input_format,
        output_format=output_format,
        dest_acc=dest_acc,
        arch=arch,
    )
    # Resolve first, then downgrade only a *ULP* contract. Both gates below are about
    # whether a step count is measurable and trustworthy here, and neither says anything
    # about a declared tolerance: gating before the lookup dropped SigmoidAppx's and
    # GeluAppx's atol=0.13 on every architecture but Wormhole, and would drop any future
    # tolerance contract on a block float, in both cases back to the default atol=0.05
    # those numbers exist to widen.
    if contract.metric != Metric.ULP:
        return contract
    if not has_ulp_gate(output_format):
        # The coarse block floats and the MX formats have block-aware lattice compares in
        # utils.py that are already the stronger criterion; a per-element step count
        # against their bf16 view is not a property of the element.
        return TOLERANCE_CONTRACT
    if arch != MEASURED_ARCH and not _key_names_arch(
        op, arch, output_format, approx_mode, dest_acc
    ):
        # Every *unkeyed* number in the table was measured on Wormhole with no headroom
        # added, so letting it bind on an architecture that was never swept would make the
        # "re-measure on Blackhole first" caveat unenforceable -- WH and BH SFPUs differ
        # in available instructions and therefore in kernel.
        #
        # A contract whose winning key names `arch` explicitly is exempt: that is a
        # measurement someone took *on* that architecture, and downgrading it made the
        # advertised arch dimension impossible to use for enrolling Blackhole or Quasar.
        # Adding arch=WORMHOLE to the shared keys instead would tie specificity with the
        # per-format keys and make validate_registry() raise, which is why the default is
        # a gate here rather than a key there.
        return TOLERANCE_CONTRACT
    return contract


def _key_names_arch(
    op: MathOperation,
    arch: ChipArchitecture,
    output_format: DataFormat,
    approx_mode: Optional[ApproximationMode],
    dest_acc: Optional[DestAccumulation],
) -> bool:
    """Whether the key that wins for this variant pins ``arch`` itself.

    Asked separately rather than returned from :func:`resolve_contract`, so that
    function keeps its single-purpose signature and can go on being tested against small
    purpose-built tables.
    """
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if not table:
        return False
    matched = [
        key
        for key in table
        if key.arch is not None
        and key.matches(
            approx_mode=approx_mode,
            output_format=output_format,
            dest_acc=dest_acc,
            arch=arch,
        )
    ]
    if not matched:
        return False
    # Only if an arch-pinned key is among the most specific matches -- otherwise a
    # broader arch-pinned key would exempt a narrower shared one.
    best = max(
        key.specificity
        for key in table
        if key.matches(
            approx_mode=approx_mode,
            output_format=output_format,
            dest_acc=dest_acc,
            arch=arch,
        )
    )
    return any(key.specificity == best for key in matched)


def resolve_contract(
    table: Dict[BudgetKey, AccuracyContract],
    *,
    label: str,
    output_format: DataFormat,
    input_format: Optional[DataFormat] = None,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
    arch: Optional[ChipArchitecture] = None,
) -> AccuracyContract:
    """Pick the most specific contract in *table* covering one variant.

    Split out from :func:`accuracy_contract` so the resolution rules can be tested
    against small purpose-built tables instead of against the live registry, which would
    make those tests fail every time a budget is enrolled.
    """
    matched = [
        (key, contract)
        for key, contract in table.items()
        if key.matches(
            approx_mode=approx_mode,
            input_format=input_format,
            output_format=output_format,
            dest_acc=dest_acc,
            arch=arch,
        )
    ]
    if not matched:
        return TOLERANCE_CONTRACT

    best = max(key.specificity for key, _ in matched)
    winners = [(key, contract) for key, contract in matched if key.specificity == best]
    if len(winners) > 1:
        raise ValueError(
            f"{label} has {len(winners)} equally specific budget keys matching "
            f"output_format={output_format.name}, input_format={input_format}, "
            f"approx_mode={approx_mode}, dest_acc={dest_acc}, arch={arch}: "
            f"{', '.join(key.describe() for key, _ in winners)}. Make one of them more "
            "specific; the table's order must not decide a budget."
        )
    return winners[0][1]


def enrolled_ops() -> Tuple[MathOperation, ...]:
    """Every op with a declared contract, in name order. For reporting and tests."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


def validate_registry() -> None:
    """Raise if any op in the table can resolve ambiguously, or has an empty entry.

    Exhaustive over the variant space rather than a review convention: it is small
    (approximation mode × ULP-capable input format × output format × Dest accumulation ×
    architecture) and an ambiguity that shows up for one format only is exactly what a
    reader misses.
    """
    from .ulp import _ULP_PROXY_DTYPES, ULP_FORMATS

    # Every output format a driver may pass, not only the ULP-capable ones:
    # accuracy_contract() calls resolve_contract() *before* the has_ulp_gate downgrade, so
    # the tie check runs for formats that end up on the tolerance metric -- Bfp4_b among
    # them, which FORMATS_BFP4_B reaches for six enrolled ops. And None on the two axes
    # that default to it, since matches() treats an unset caller dimension as a
    # wildcard-only subset: a tie between BudgetKey(output_format=Bfp4_b) and
    # BudgetKey(dest_acc=Yes) would otherwise surface mid device run.
    gateable = list(ULP_FORMATS) + list(_ULP_PROXY_DTYPES)
    formats = gateable + [f for f in DataFormat if f not in gateable]
    approx_modes = list(ApproximationMode) + [None]
    dest_accs = list(DestAccumulation) + [None]
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        if not table:
            raise ValueError(
                f"{op.name} has an empty budget entry; remove it so the op falls back to "
                "the tolerance metric explicitly"
            )
        for approx_mode in approx_modes:
            for output_format in formats:
                for input_format in formats + [None]:
                    for dest_acc in dest_accs:
                        for arch in ChipArchitecture:
                            accuracy_contract(
                                op,
                                output_format=output_format,
                                input_format=input_format,
                                approx_mode=approx_mode,
                                dest_acc=dest_acc,
                                arch=arch,
                            )
