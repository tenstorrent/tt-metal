# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy contract an SFPU op declares, and the registry that holds them.

An op's budget is a *property of the op*, not a magic number in a test body. That is what
makes it reviewable, and a kernel improvement visible: the number drops in the same PR
that improves the kernel. It is the shape ttnn already uses (``GoldenComparisonConfig``,
attached by the golden and honoured by the comparator), as a table. Its own module rather
than ``sfpu_domains`` because a domain entry says what an op may be *fed* and a budget
says how closely its output must match.

**Keyed on more than the op.** Approximation mode moves the error by orders of magnitude,
and the output format decides what is even visible: a sub-ULP downward step at a LUT
segment join is invisible in bfloat16 and large in float32, so one number across formats
would be set by the bf16 measurement and never gate the interesting path. Dest
accumulation and the architecture are in the key for the same reason. Unset dimensions
match any value, and the most specific key wins.

**Enrolment is per op, per format, and incremental.** An op with no entry gets
:data:`TOLERANCE_CONTRACT`, and so does any request for a format with no per-element ULP
-- which is how the coarse block floats and the MX formats keep their block-aware lattice
compares while an enrolled op is gated where a step count means something.

**Numbers here are measured, not guessed**, and each carries the architecture it came
from. Measured against the right thing, too: the ULP arm returns before both the
tolerance gate and PCC, so a zero-headroom budget has no backstop, and enrolling an op
means measuring *every* sweep that reaches the driver -- the ramp sweep, its ``_edges``
variant with the inf/NaN/signed-zero probes, and the hand-built ``_signbit``,
``_isinf_isnan`` and ``_threshold`` specs. One gap is open: the edge sweep produces only
``Float16_b`` and ``Float32`` outputs, so the ``Bfp8_b`` variant that Floor/Ceil/Trunc
carry through an unrestricted ``DEFAULT`` has no edge measurement behind it.
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
    """Which gate a contract is written against: a closed two-member set, so there is no
    third to fall through to. It does not make a wrong value unrepresentable on its own
    -- ``metric="ulp"`` still constructs -- which is what ``__post_init__`` is for."""

    ULP = "ulp"
    TOLERANCE = "tolerance"


@dataclass(frozen=True)
class AccuracyContract:
    """How closely one op's output must match its golden, and by which metric.

    ``Metric.ULP`` means "every element within *max_ulp* steps", with the tolerance and
    PCC checks skipped; ``Metric.TOLERANCE`` is the historical gate. The two sets of
    fields are mutually exclusive, enforced below, so a half-converted entry cannot sit
    in the table looking plausible.
    """

    metric: Metric = Metric.ULP
    max_ulp: Optional[int] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    near_zero_atol: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            # The annotation is not a check: `metric="ulp"` would take the `else` branch
            # below and become a *tolerance* contract, switching off the intended gate.
            raise ValueError(
                f"metric must be a Metric member, got {self.metric!r}; use "
                f"{Metric.ULP} or {Metric.TOLERANCE}"
            )
        for name in ("atol", "rtol", "near_zero_atol"):
            value = getattr(self, name)
            if value is not None and value < 0:
                # passed_test applies an override only when it is `>= 0`, so a negative
                # atol/rtol reads here as a declared tolerance and then runs against the
                # per-format default. A negative near_zero_atol is an inert floor.
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
        """The contract as ``passed_test`` keyword arguments, so the driver's call site
        is one ``**`` expansion and switching metrics is a registry edit."""
        if self.metric == Metric.ULP:
            return {"max_ulp": self.max_ulp, "near_zero_atol": self.near_zero_atol}
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}


#: What an unenrolled op -- or an enrolled one on a format with no per-element ULP -- is
#: judged by: the per-format defaults plus ``PCC > 0.99``, exactly as before.
TOLERANCE_CONTRACT = AccuracyContract(metric=Metric.TOLERANCE)


#: The enum each :class:`BudgetKey` dimension must be a member of. Hand-maintained rather
#: than read off ``__annotations__``, which would hand back ``Optional[...]``; a host test
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

    More fields set means more specific, and the most specific match wins. Two keys that
    match one variant with equal specificity are an authoring error, not a tie-break:
    :func:`validate_registry` rejects them rather than letting iteration order decide.
    """

    approx_mode: Optional[ApproximationMode] = None
    output_format: Optional[DataFormat] = None
    dest_acc: Optional[DestAccumulation] = None
    arch: Optional[ChipArchitecture] = None

    def __post_init__(self) -> None:
        # The annotation is not a check, as for `AccuracyContract.metric`. All four are
        # bare `Enum`s, so `DestAccumulation.No.value is False` never equals its member
        # -- making `BudgetKey(dest_acc=True)` *inert*: counted as set by `specificity`,
        # matched by nothing, and rendered identically by `describe()`. It would fail
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
        output_format: Optional[DataFormat],
        dest_acc: Optional[DestAccumulation],
        arch: Optional[ChipArchitecture],
    ) -> bool:
        """Whether this key covers the given variant.

        A dimension the *caller* leaves unset matches only a wildcard: guessing would
        hand back a budget measured for the other setting.
        """
        query = {
            "approx_mode": approx_mode,
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


def _out(output_format: DataFormat) -> BudgetKey:
    """The commonest key shape: a budget that varies only by output format."""
    return BudgetKey(output_format=output_format)


#: One op's keyed budgets. Aliased because the full spelling pushes every signature
#: and the registry header past the line limit.
_BudgetTable = Dict[BudgetKey, AccuracyContract]


# ── The registry ────────────────────────────────────────────────────────────
#
# Enrolled first are the ops exact by construction: they are the flakiness canaries, since
# a 0-step budget on Abs cannot be wrong about the kernel -- if one fails, the golden or
# the datapath moved. Re-measure on Blackhole before trusting any of it there.
#
# Relu is absent because it is packer-applied via STACC_RELU and is not a SfpuType member,
# so it will not compile through the unary driver; ReluMax/ReluMin take a threshold
# operand and are left for a later pass.

# ── Bfp8_b: dominated by block quantization, not by the op ──────────────────
#
# The bf16 proxy is a real per-element criterion only while the block exponent is the one
# bf16 would have used, and on this stimulus it is not: Abs and Neg -- which only clear
# and flip a sign bit -- measure 15616 steps, the worst lane `result 0.0 vs golden 0.062`.
# That is a small element quantized to zero by a shared exponent, exactly as designed, and
# two orders of magnitude past bf16's 128-step ceiling. So a budget cannot gate these
# here, and raising it until they pass gates nothing; a near_zero_atol floor would absorb
# it but would then be the flat-tolerance gate under a new name. Parked on tolerance,
# whose lattice compare is already the stronger criterion. Floor/Ceil/Trunc are the
# exception: integer results, which a shared exponent represents exactly.
#   wh: Abs/Neg max 15616 ULP, Square max 17664, Floor/Ceil/Trunc max 0, 2026-09-16
_BFP8_QUANTIZED = AccuracyContract(metric=Metric.TOLERANCE)

#: The two coarse 3-segment LUT approximations, which share one number because they share
#: one cause. Named once so a retune cannot move SigmoidAppx and leave GeluAppx behind.
_COARSE_LUT = AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05)

#: What makes a 0-step Bfp8_b budget legitimate for the integer-valued ops: every block
#: maximum stays below 2**7, so the shared exponent is exact. A property of the
#: *stimulus*, not the format, so the host tests assert it against _OP_DOMAIN_REGISTRY.
BFP8_B_EXACT_INTEGER_DOMAIN = 128.0


def budget_table(*entries: Tuple[BudgetKey, AccuracyContract]) -> _BudgetTable:
    """One op's table, built from pairs so a repeated key is an error.

    ``BudgetKey`` is frozen, so two identical keys in a dict literal are hash-equal and
    Python keeps the later contract -- leaving every downstream guard, including the
    tie-raise in :func:`resolve_contract`, looking at one entry rather than two.
    """
    table: _BudgetTable = {}
    for key, contract in entries:
        if key in table:
            raise ValueError(
                f"duplicate budget key {key.describe()}: a dict literal would have kept "
                "only the later contract, and no guard downstream can see the first one"
            )
        table[key] = contract
    return table


def registry(
    *entries: Tuple[MathOperation, _BudgetTable]
) -> Dict[MathOperation, _BudgetTable]:
    """The whole table, built from pairs so a repeated op is an error -- the same hazard
    as :func:`budget_table` one level up, where the ops sit far apart and nothing
    downstream sees the dropped one."""
    table: Dict[MathOperation, _BudgetTable] = {}
    for op, contracts in entries:
        if op in table:
            raise ValueError(
                f"duplicate registry entry for {op.name}: a dict literal would have kept "
                "only the later table, and no guard downstream can see the first one"
            )
        table[op] = contracts
    return table


def _exact_everywhere() -> _BudgetTable:
    """A fresh 0-step table for the ops measured exact on every output format. A factory
    rather than one literal aliased three ways, so a retune cannot move the others."""
    return budget_table((DEFAULT, AccuracyContract(max_ulp=0)))


_SFPU_ACCURACY_BUDGET: Dict[MathOperation, _BudgetTable] = registry(
    # ── Exact everywhere, including Bfp8_b ──────────────────────────────
    # The only ops enrolled on Bfp8_b, for a narrower reason than "integers are exact in
    # a block float": a shared exponent is exact only while every block maximum stays
    # below 2**7, which holds only because _OP_DOMAIN_REGISTRY bounds these three to
    # uniform(-10, 10). A wider domain fails, by what takes Abs/Neg to 15616 steps.
    #   wh: 0 ULP, 156 variants each, all four output formats x both dest_acc, 2026-09-16
    (MathOperation.Floor, _exact_everywhere()),
    (MathOperation.Ceil, _exact_everywhere()),
    (MathOperation.Trunc, _exact_everywhere()),
    # ── Sign-bit and select: exact in fp32, one step in the 16-bit formats ──
    # No arithmetic to round, so fp32 is bit-exact. The single step on the 16-bit outputs
    # is the *pack* path, not the op -- it shows up for all three, mostly at dest_acc=Yes
    # where the value is rounded at pack rather than truncated in Dest first, and a step
    # budget is what makes it visible at all (atol=0.05 is ~6 bf16 steps). No headroom,
    # deliberately: these are exact by construction, so any movement is real signal.
    #   wh: Abs/Neg max 0 ULP on Float32 (32 variants), 1 ULP on Float16/Float16_b
    #       (80 variants); Identity max 0 on Float32, 1 on Float16_b (4), 2026-09-16
    (
        MathOperation.Abs,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=1)),
            (_out(DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (_out(DataFormat.Bfp8_b), _BFP8_QUANTIZED),
        ),
    ),
    (
        MathOperation.Neg,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=1)),
            (_out(DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (_out(DataFormat.Bfp8_b), _BFP8_QUANTIZED),
        ),
    ),
    # Identity is keyed per format, not through a DEFAULT: it was never in
    # BROAD_SWEEP_OPS, so fp16 was never measured for it -- and Square measured 1 step on
    # bf16 against 4 on fp16, so fp16 is not safely interpolated from bf16.
    (
        MathOperation.Identity,
        budget_table(
            (_out(DataFormat.Float32), AccuracyContract(max_ulp=0)),
            (_out(DataFormat.Float16_b), AccuracyContract(max_ulp=1)),
        ),
    ),
    # ── One multiply, and one open question ─────────────────────────────
    # x*x has rounding slack the sign-bit ops do not: the golden rounds in float64, the
    # hardware in the datapath, and ties can differ. Float32 is NOT enrolled, and the
    # measurement is why -- 65536 steps at dest_acc=No is 2**16, one step of a 16-bit
    # Dest lattice in fp32 units, so the two round the same step differently; 32768 at
    # dest_acc=Yes has no such explanation, the product agreeing to only ~8 mantissa
    # bits. Attributing either is its own change, so it is recorded, not blessed.
    #   wh: Float16_b max 1 ULP (40 variants), Float16 max 4 (40),
    #       Float32 max 65536 @ dest_acc=No / 32768 @ dest_acc=Yes (32), 2026-09-16
    (
        MathOperation.Square,
        budget_table(
            (DEFAULT, AccuracyContract(max_ulp=4)),
            (_out(DataFormat.Float16_b), AccuracyContract(max_ulp=1)),
            (_out(DataFormat.Float32), TOLERANCE_CONTRACT),
            (_out(DataFormat.Bfp8_b), _BFP8_QUANTIZED),
        ),
    ),
    # ── Still on the tolerance metric, moved here from the test body ────
    # CUSTOM_TOLERANCES in test_eltwise_unary_sfpu: a coarse 3-segment LUT carrying
    # atol=0.13 so the sweep passes. The clearest argument for this mechanism -- that
    # number makes the test blind to a 10x regression anywhere else in the domain, and
    # equally blind to the improvement a retune would produce.
    (MathOperation.SigmoidAppx, budget_table((DEFAULT, _COARSE_LUT))),
    (MathOperation.GeluAppx, budget_table((DEFAULT, _COARSE_LUT))),
)


def accuracy_contract(
    op: MathOperation,
    *,
    output_format: DataFormat,
    arch: ChipArchitecture,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
) -> AccuracyContract:
    """The contract for one op variant, or :data:`TOLERANCE_CONTRACT` if it has none.

    Falling back rather than raising is what makes enrolment incremental. *arch* is
    required, unlike the other three: it is the one dimension where the numbers
    explicitly do not transfer, so defaulting it would resolve an unknown chip straight
    against the Wormhole table.
    """
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if table is None:
        return TOLERANCE_CONTRACT

    contract = resolve_contract(
        table,
        label=op.name,
        approx_mode=approx_mode,
        output_format=output_format,
        dest_acc=dest_acc,
        arch=arch,
    )
    # Resolve first, then downgrade only a *ULP* contract: both gates below ask whether a
    # step count is trustworthy here, and neither says anything about a declared
    # tolerance. Gating before the lookup dropped SigmoidAppx's and GeluAppx's atol=0.13
    # on every arch but Wormhole, back to the default those numbers exist to widen.
    if contract.metric != Metric.ULP:
        return contract
    if not has_ulp_gate(output_format):
        # Their block-aware lattice compares are already the stronger criterion.
        return TOLERANCE_CONTRACT
    if arch != MEASURED_ARCH and not _key_names_arch(
        op, arch, output_format, approx_mode, dest_acc
    ):
        # Every *unkeyed* number was measured on Wormhole with no headroom, so letting it
        # bind on an unswept architecture would make the "re-measure first" caveat
        # unenforceable. A key that names `arch` itself is exempt -- that is a measurement
        # taken there. Adding arch=WORMHOLE to the shared keys instead would tie
        # specificity with the per-format ones and make validate_registry() raise, which
        # is why this is a gate here rather than a key there.
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

    Asked separately so :func:`resolve_contract` keeps its single-purpose signature.
    Only a *winning* arch-pinned key counts -- a broader one must not exempt a narrower
    shared key that beats it.
    """
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if not table:
        return False
    matched = [
        key
        for key in table
        if key.matches(
            approx_mode=approx_mode,
            output_format=output_format,
            dest_acc=dest_acc,
            arch=arch,
        )
    ]
    if not matched:
        return False
    best = max(key.specificity for key in matched)
    return any(key.arch is not None and key.specificity == best for key in matched)


def resolve_contract(
    table: _BudgetTable,
    *,
    label: str,
    output_format: DataFormat,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
    arch: Optional[ChipArchitecture] = None,
) -> AccuracyContract:
    """Pick the most specific contract in *table* covering one variant. Split out so the
    resolution rules can be tested against small purpose-built tables rather than the
    live registry, which would fail every time a budget is enrolled."""
    matched = [
        (key, contract)
        for key, contract in table.items()
        if key.matches(
            approx_mode=approx_mode,
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
            f"output_format={output_format.name}, approx_mode={approx_mode}, "
            f"dest_acc={dest_acc}, arch={arch}: "
            f"{', '.join(key.describe() for key, _ in winners)}. Make one of them more "
            "specific; the table's order must not decide a budget."
        )
    return winners[0][1]


def enrolled_ops() -> Tuple[MathOperation, ...]:
    """Every op with a declared contract, in name order. For reporting and tests."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


def validate_registry() -> None:
    """Raise if any op can resolve ambiguously, or has an empty entry.

    Exhaustive over the variant space rather than a review convention: it is small, and
    an ambiguity that shows up for one format only is exactly what a reader misses.
    """
    from .ulp import _ULP_PROXY_DTYPES, ULP_FORMATS

    # Every output format a driver may pass, not only the gateable ones: the tie check
    # runs before the has_ulp_gate downgrade, so it covers formats that end up on
    # tolerance. And `None` on the axes that default to it, since an unset caller
    # dimension matches only a wildcard -- otherwise a tie would surface mid device run.
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
                for dest_acc in dest_accs:
                    for arch in ChipArchitecture:
                        accuracy_contract(
                            op,
                            output_format=output_format,
                            approx_mode=approx_mode,
                            dest_acc=dest_acc,
                            arch=arch,
                        )
