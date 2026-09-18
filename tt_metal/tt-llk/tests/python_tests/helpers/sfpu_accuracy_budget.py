# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy contract an SFPU op declares, and the registry that holds them.

An op's budget is a *property of the op*, not a magic number in a test body. That is what
makes it reviewable, and a kernel improvement visible: the number drops in the same PR
that improves the kernel. It is the shape ttnn already uses (``GoldenComparisonConfig``,
attached by the golden and honoured by the comparator), as a table. Its own module rather
than ``sfpu_domains`` because a domain entry says what an op may be *fed* and a budget
says how closely its output must match.

**The table is data and lives in** ``sfpu_accuracy_budget.yaml``; this module is the
interface to it. Hundreds of measured rows have no business being Python -- as YAML they
diff one row at a time, regenerate wholesale from the sweep, and cannot smuggle in logic.
Everything the rows are checked against is here: the enums a field may name, the contract
invariants, and the matching rule.

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

import io
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import MANTISSA_BITS_FOR_ULP, MAX_MEANINGFUL_ULP, has_ulp_gate, ulp_dtype

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
    "input_format": DataFormat,
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
    input_format: Optional[DataFormat] = None
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
        input_format: Optional[DataFormat],
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

#: One op's keyed budgets.
_BudgetTable = Dict[BudgetKey, AccuracyContract]

#: What makes a 0-step Bfp8_b budget legitimate for the integer-valued ops: every block
#: maximum stays below 2**7, so the shared exponent is exact. A property of the
#: *stimulus*, not the format, so the host tests assert it against _OP_DOMAIN_REGISTRY.
BFP8_B_EXACT_INTEGER_DOMAIN = 128.0


# ── Loading the table ───────────────────────────────────────────────────────

#: The table itself. Data, not code: 263 rows of measured numbers have no business being
#: Python, and as YAML they diff one row at a time and can be regenerated wholesale.
_TABLE_PATH = Path(__file__).with_name("sfpu_accuracy_budget.yaml")

#: YAML row field -> :class:`BudgetKey` field. The short spellings keep a row on one line.
_KEY_FIELDS: Dict[str, str] = {
    "in": "input_format",
    "out": "output_format",
    "approx": "approx_mode",
    "dest": "dest_acc",
    "arch": "arch",
}
_CONTRACT_FIELDS = frozenset({"metric", "max_ulp", "near_zero_atol", "atol", "rtol"})


class _StrictLoader(yaml.SafeLoader):
    """A loader that refuses a duplicate mapping key instead of keeping the last one."""


def _no_duplicate_keys(loader: _StrictLoader, node: yaml.MappingNode) -> Dict[Any, Any]:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ValueError(
                f"{_TABLE_PATH.name}: duplicate entry for {key!r}. YAML keeps only the "
                "last, so the earlier budget would vanish with nothing to catch it."
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def _member(enum_cls: type, value: Any, where: str) -> Any:
    """One YAML scalar as an enum member, by name or by value.

    Both, because YAML 1.1 reads a bare ``No`` as ``False`` and ``ApproximationMode.No``
    is spelled ``False`` too -- so the quoted and unquoted forms should not disagree.
    """
    try:
        return enum_cls[value] if isinstance(value, str) else enum_cls(value)
    except (KeyError, ValueError):
        raise ValueError(
            f"{where}: {value!r} is not a {enum_cls.__name__}; expected one of "
            f"{', '.join(m.name for m in enum_cls)}"
        ) from None


def _row_to_entry(
    op_name: str, row: Dict[str, Any]
) -> Tuple[BudgetKey, AccuracyContract]:
    where = f"{_TABLE_PATH.name}: {op_name}"
    unknown = set(row) - set(_KEY_FIELDS) - _CONTRACT_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")

    types = {
        "in": DataFormat,
        "out": DataFormat,
        "approx": ApproximationMode,
        "dest": DestAccumulation,
        "arch": ChipArchitecture,
    }
    key = BudgetKey(
        **{
            _KEY_FIELDS[name]: _member(types[name], row[name], where)
            for name in _KEY_FIELDS
            if name in row
        }
    )
    contract = {field: row[field] for field in _CONTRACT_FIELDS if field in row}
    metric = contract.pop("metric", "ulp")
    if metric not in ("ulp", "tolerance"):
        raise ValueError(
            f"{where}: metric must be 'ulp' or 'tolerance', got {metric!r}"
        )
    # AccuracyContract.__post_init__ owns the rest of the validation, so a row that is
    # half-converted between the two metrics is refused there rather than here.
    return key, AccuracyContract(metric=Metric(metric), **contract)


def _load_table(path: Path = _TABLE_PATH) -> Dict[MathOperation, _BudgetTable]:
    """The YAML table as the registry the resolver walks.

    Every failure is the author's, so each one names the op it came from: an unknown op,
    an unknown field, a scalar that is not an enum member, a duplicate row, or a contract
    that :class:`AccuracyContract` refuses.
    """
    with io.open(path, encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_StrictLoader) or {}

    table: Dict[MathOperation, _BudgetTable] = {}
    for op_name, rows in raw.items():
        try:
            op = MathOperation[op_name]
        except KeyError:
            raise ValueError(
                f"{path.name}: {op_name!r} is not a MathOperation"
            ) from None
        if not rows:
            raise ValueError(
                f"{path.name}: {op_name} has no rows; remove it so the op falls back to "
                "the tolerance metric explicitly"
            )
        entries: _BudgetTable = {}
        for row in rows:
            key, contract = _row_to_entry(op_name, row)
            if key in entries:
                raise ValueError(
                    f"{path.name}: {op_name} repeats {key.describe()}; the later row "
                    "would silently replace the earlier budget"
                )
            entries[key] = contract
        table[op] = entries
    return table


_SFPU_ACCURACY_BUDGET: Dict[MathOperation, _BudgetTable] = _load_table()


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

    Falling back rather than raising is what makes enrolment incremental. *arch* is
    required, unlike the other four: it is the one dimension where the numbers
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
        input_format=input_format,
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
        op, arch, input_format, output_format, approx_mode, dest_acc
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
    input_format: Optional[DataFormat],
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
            input_format=input_format,
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
    input_format: Optional[DataFormat] = None,
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
            f"input_format={input_format}, output_format={output_format.name}, "
            f"approx_mode={approx_mode}, "
            f"dest_acc={dest_acc}, arch={arch}: "
            f"{', '.join(key.describe() for key, _ in winners)}. Make one of them more "
            "specific; the table's order must not decide a budget."
        )
    return winners[0][1]


def usable_budget_ceiling(output_format: DataFormat) -> float:
    """The largest budget that is still *stronger* than the gate it replaces.

    ``MAX_MEANINGFUL_ULP`` is the wrong bound: ``2**mantissa_bits`` is roughly 100%
    relative error, so it admits budgets that gate nothing -- and since ``passed_test``
    returns on the ULP verdict and skips both ``isclose`` and PCC, such a budget *is* the
    whole gate. Measured, approximate tanh on an fp32 output reached 2,949,120 steps,
    about 35% relative error, on an op bounded in (-1, 1).

    The real bound is the ``rtol`` half of the ``isclose`` this replaces, itself a step
    budget at large magnitude: about 419,430 steps for fp32, 51 for fp16, 6 for bf16.
    ``passed_test`` warns on the same line at runtime; no row here may cross it.
    """
    from .utils import tolerances

    dtype = ulp_dtype(output_format)
    by_rtol = tolerances[output_format].rtol * (1 << MANTISSA_BITS_FOR_ULP[dtype])
    return min(by_rtol, float(MAX_MEANINGFUL_ULP[dtype]))


def enrolled_ops() -> Tuple[MathOperation, ...]:
    """Every op with a declared contract, in name order. For reporting and tests."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


def validate_registry() -> None:
    """Raise if any op can resolve ambiguously.

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
    # The *input* axis comes from the table, not from the enum: a format no row pins
    # reproduces the `None` iteration exactly, since an unset key matches any value. Four
    # values today rather than 23, and it widens itself the moment a row pins a new one.
    input_formats = sorted(
        {key.input_format for table in _SFPU_ACCURACY_BUDGET.values() for key in table}
        - {None},
        key=lambda fmt: fmt.name,
    ) + [None]
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for approx_mode in approx_modes:
            for input_format in input_formats:
                for output_format in formats:
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
