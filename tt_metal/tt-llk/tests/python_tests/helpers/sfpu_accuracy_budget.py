# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy contract an SFPU op declares, and the registry that holds them.

An op's budget is a *property of the op*, not a magic number in a test body: that makes
it reviewable, and a kernel improvement visible as the number dropping in the same PR.
The table is data in ``sfpu_accuracy_budget.yaml``; this module loads it, checks it,
and resolves one variant of one op to its contract.

**Keyed on more than the op.** Approximation mode moves the error by orders of magnitude,
and the output format decides what is even visible: a sub-ULP downward step at a LUT
segment join is invisible in bfloat16 and large in float32, so one number across formats
would be set by the bf16 measurement and never gate the interesting path. Dest
accumulation and the architecture are in the key for the same reason. Unset dimensions
match any value, the most specific key wins, and two equally specific matches are an
authoring error rather than a tie-break.

**Enrolment is per op and incremental.** An op with no entry gets
:data:`TOLERANCE_CONTRACT`, and so does any request for a format with no per-element
ULP -- which is how the block floats keep their block-aware lattice compares.

**Numbers are measured, not guessed**, and every unkeyed number came from
:data:`MEASURED_ARCH`; a ULP row binds elsewhere only if its own key names that
architecture. Only declared tolerances live in the table so far. ``max_ulp`` is
accepted and validated, but a step budget is enrolled once the exhaustive sweep exists
to measure it, not declared against nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import yaml

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import MANTISSA_BITS_FOR_ULP, MAX_MEANINGFUL_ULP, has_ulp_gate, ulp_dtype

#: The architecture every unkeyed budget was measured on. Anywhere else an op resolves
#: to the tolerance metric until the sweep has been re-run there.
MEASURED_ARCH = ChipArchitecture.WORMHOLE


class Metric(Enum):
    """Which gate a contract is written against: a closed two-member set, so there is no
    third to fall through to. ``AccuracyContract`` refuses a non-member such as
    ``"ulp"``, which the annotation alone would let through."""

    ULP = "ulp"
    TOLERANCE = "tolerance"


def _check_number(name: str, value: Any, *, integer: bool = False) -> None:
    """Refuse what the annotations cannot.

    YAML 1.1 reads ``yes`` as a bool, and bool is an int, so ``atol: yes`` would apply
    as 1.0 and ``max_ulp: true`` as a 1-step budget. ``.nan`` is silently ignored by
    ``passed_test`` and ``.inf`` makes its gate unconditional. A negative override is
    treated as unset there and runs against the per-format default.
    """
    if isinstance(value, bool) or not isinstance(
        value, int if integer else (int, float)
    ):
        what = "an int step count" if integer else "a number"
        raise ValueError(f"{name} must be {what}, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")


@dataclass(frozen=True)
class AccuracyContract:
    """How closely one op's output must match its golden, and by which metric.

    ``Metric.ULP`` means "every element within *max_ulp* steps", with the tolerance and
    PCC checks skipped; ``Metric.TOLERANCE`` is the historical gate. The two sets of
    fields are mutually exclusive, so a half-converted entry cannot sit in the table
    looking plausible.
    """

    metric: Metric = Metric.ULP
    max_ulp: Optional[int] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    near_zero_atol: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            raise ValueError(
                f"metric must be a Metric member, got {self.metric!r}; use "
                f"{Metric.ULP} or {Metric.TOLERANCE}"
            )
        for name in ("atol", "rtol", "near_zero_atol"):
            if getattr(self, name) is not None:
                _check_number(name, getattr(self, name))
        if self.metric is Metric.ULP:
            if self.max_ulp is None:
                raise ValueError("a ulp contract needs max_ulp")
            _check_number("max_ulp", self.max_ulp, integer=True)
            if self.atol is not None or self.rtol is not None:
                raise ValueError(
                    "a ulp contract replaces the tolerance gate, so atol/rtol would be "
                    "silently ignored; use near_zero_atol for the near-zero floor"
                )
        elif self.max_ulp is not None or self.near_zero_atol is not None:
            raise ValueError(
                "max_ulp and near_zero_atol belong to the ulp metric; set "
                f"metric={Metric.ULP} to use them"
            )

    def tolerance_kwargs(self) -> Dict[str, Any]:
        """The contract as ``passed_test`` arguments for a *tolerance-only* caller.

        The functional drivers gate on tolerance and PCC. A step budget is measured by
        the exhaustive sweep over every value the format has, so it is far wider than
        the few thousand values a driver samples warrant, and feeding it back would
        loosen the driver's gate rather than tighten it. An op on the ULP metric
        therefore keeps today's per-format tolerance here.
        """
        if self.metric is Metric.ULP:
            return {}
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}

    def passed_test_kwargs(self) -> Dict[str, Any]:
        """The contract as ``passed_test`` keyword arguments, whichever metric it is on,
        so a call site is one ``**`` expansion and switching metrics is a table edit."""
        if self.metric is Metric.ULP:
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
    """Which variants of an op a contract applies to -- and, on the query side, which
    variant is being asked about. Unset field == any value.

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
        # The annotation is not a check. All four are bare `Enum`s, so a bool never
        # equals `DestAccumulation.No`: such a key would count as set and match nothing,
        # and such a query would silently fall through to a broader row.
        for name, expected in _BUDGET_KEY_TYPES.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise ValueError(
                    f"BudgetKey.{name} must be a {expected.__name__} member or None, "
                    f"got {value!r}; a non-member is counted as set and matches nothing"
                )
        # Cached once, in _BUDGET_KEY_TYPES order on both the key and the query side.
        # `matches` runs millions of times under validate_registry, and reading the
        # fields back through `dataclasses.fields` on every call made it 4x slower.
        object.__setattr__(
            self, "_values", tuple(getattr(self, name) for name in _BUDGET_KEY_TYPES)
        )

    @property
    def specificity(self) -> int:
        return sum(value is not None for value in self._values)

    def matches(self, query: BudgetKey) -> bool:
        """Whether this key covers *query*. A dimension the query leaves unset matches
        only a wildcard: guessing would hand back a budget measured for the other
        setting."""
        return all(
            wanted is None or wanted == asked
            for wanted, asked in zip(self._values, query._values)
        )

    def describe(self) -> str:
        set_fields = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"BudgetKey({', '.join(set_fields)})" if set_fields else "DEFAULT"


E = TypeVar("E", bound=Enum)

#: Matches every variant of an op. The right key for a budget that does not yet vary.
DEFAULT = BudgetKey()

#: One op's keyed budgets.
_BudgetTable = Dict[BudgetKey, AccuracyContract]

#: What makes a 0-step Bfp8_b budget legitimate for the integer-valued ops: every block
#: maximum stays below 2**7, so the shared exponent is exact. A property of the
#: *stimulus*, not the format, so the host tests assert it against _OP_DOMAIN_REGISTRY.
BFP8_B_EXACT_INTEGER_DOMAIN = 128.0


# ── Loading the table ───────────────────────────────────────────────────────

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


def _enum_member(enum_cls: Type[E], value: Any, where: str) -> E:
    """One YAML scalar as an enum member, by name or by value.

    By name *and* by value because YAML 1.1 reads a bare ``No`` as ``False``, and
    ``ApproximationMode.No`` is spelled ``False`` too, so the quoted and bare spellings
    must land on the same member. By value only when the scalar has the member value's
    own type: ``True == 1`` in Python, so ``ApproximationMode(1)`` and even
    ``ApproximationMode(1.0)`` would otherwise resolve to ``Yes``.
    """
    try:
        if isinstance(value, str) and value in enum_cls.__members__:
            return enum_cls[value]
        member = enum_cls(value)
        if type(member.value) is not type(value):
            raise ValueError(value)
        return member
    except (KeyError, ValueError):
        raise ValueError(
            f"{where}: {value!r} is not a {enum_cls.__name__}; expected one of "
            f"{', '.join(m.name for m in enum_cls)}"
        ) from None


def _refuse_duplicate_keys(node: yaml.Node, where: str) -> None:
    """PyYAML keeps the last of two identical mapping keys without a word, so a
    copy-pasted op name would drop the earlier op's whole table. Checked on the node
    tree the standard ``SafeLoader`` composes, before it constructs anything. ``<<``
    merge keys are skipped: overriding a merged field is what they are for."""
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                continue
            if key.value in seen:
                raise ValueError(
                    f"{where}: duplicate entry for {key.value!r}. YAML keeps only the "
                    "last, so the earlier one would vanish with nothing to catch it."
                )
            seen.add(key.value)
            _refuse_duplicate_keys(value, where)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _refuse_duplicate_keys(item, where)


def _read_yaml(path: Path) -> Dict[str, Any]:
    """*path* through PyYAML's ``SafeLoader``, refusing duplicate keys."""
    with open(path, encoding="utf-8") as handle:
        loader = yaml.SafeLoader(handle)
        try:
            node = loader.get_single_node()
            if node is None:
                return {}
            _refuse_duplicate_keys(node, path.name)
            loaded = loader.construct_document(node)
        finally:
            loader.dispose()
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{path.name}: expected a mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def _row_to_entry(
    where: str, row: Dict[str, Any]
) -> Tuple[BudgetKey, AccuracyContract]:
    unknown = set(row) - set(_KEY_FIELDS) - _CONTRACT_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
    key = BudgetKey(
        **{
            field: _enum_member(_BUDGET_KEY_TYPES[field], row[short], where)
            for short, field in _KEY_FIELDS.items()
            if short in row
        }
    )
    contract = {field: row[field] for field in _CONTRACT_FIELDS if field in row}
    metric = _enum_member(Metric, contract.pop("metric", "ulp"), where)
    # AccuracyContract.__post_init__ owns the rest of the validation, so a row that is
    # half-converted between the two metrics is refused there rather than here.
    return key, AccuracyContract(metric=metric, **contract)


def _load_table(path: Path = _TABLE_PATH) -> Dict[MathOperation, _BudgetTable]:
    """The YAML table as the registry the resolver walks. Every failure is the author's,
    so each one names the op it came from."""
    table: Dict[MathOperation, _BudgetTable] = {}
    for op_name, rows in _read_yaml(path).items():
        where = f"{path.name}: {op_name}"
        try:
            op = MathOperation[op_name]
        except KeyError:
            raise ValueError(
                f"{path.name}: {op_name!r} is not a MathOperation"
            ) from None
        if not rows:
            raise ValueError(
                f"{where} has no rows; remove it so the op falls back to the tolerance "
                "metric explicitly"
            )
        entries: _BudgetTable = {}
        for row in rows:
            key, contract = _row_to_entry(where, row)
            if key in entries:
                raise ValueError(
                    f"{where} repeats {key.describe()}; the later row would silently "
                    "replace the earlier budget"
                )
            entries[key] = contract
        table[op] = entries
    return table


_SFPU_ACCURACY_BUDGET: Dict[MathOperation, _BudgetTable] = _load_table()


# ── Resolving one variant ───────────────────────────────────────────────────


def _winner(
    table: _BudgetTable, query: BudgetKey, label: str
) -> Optional[Tuple[BudgetKey, AccuracyContract]]:
    """The single most specific entry of *table* covering *query*, or ``None``."""
    matched = [(key, contract) for key, contract in table.items() if key.matches(query)]
    if not matched:
        return None
    best = max(key.specificity for key, _ in matched)
    winners = [(key, contract) for key, contract in matched if key.specificity == best]
    if len(winners) > 1:
        raise ValueError(
            f"{label} has {len(winners)} equally specific budget keys matching "
            f"{query.describe()}: {', '.join(key.describe() for key, _ in winners)}. "
            "Make one of them more specific; the table's order must not decide a budget."
        )
    return winners[0]


def resolve_contract(
    table: _BudgetTable, query: BudgetKey, *, label: str = "table"
) -> AccuracyContract:
    """The most specific contract in *table* covering *query*, or
    :data:`TOLERANCE_CONTRACT`. Public so the resolution rules can be tested against small
    purpose-built tables rather than the live registry, which would fail every time a
    budget is enrolled."""
    found = _winner(table, query, label)
    return TOLERANCE_CONTRACT if found is None else found[1]


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
    required, unlike the other dimensions: it is the one where the numbers explicitly do
    not transfer, so defaulting it would resolve an unknown chip straight against the
    :data:`MEASURED_ARCH` table.
    """
    # Built before the enrolment fallback, so a miswired driver -- a string arch, a bool
    # dest_acc -- fails on every op, not only once its op is enrolled.
    query = BudgetKey(
        approx_mode=approx_mode,
        input_format=input_format,
        output_format=output_format,
        dest_acc=dest_acc,
        arch=arch,
    )
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if table is None:
        return TOLERANCE_CONTRACT
    found = _winner(table, query, op.name)
    if found is None:
        return TOLERANCE_CONTRACT
    key, contract = found
    if contract.metric is not Metric.ULP:
        return contract
    # A step count is trustworthy only where the format has a per-element ULP -- the
    # block floats' lattice compares are the stronger criterion -- and only on the
    # architecture it was measured on, which for an unkeyed row is MEASURED_ARCH. A row
    # whose own key names `arch` was measured there and is exempt.
    if has_ulp_gate(output_format) and (arch == MEASURED_ARCH or key.arch is not None):
        return contract
    # Downgrade onto the op's *own* tolerance row where it has one, not the global
    # default: a ULP row that wins on specificity must not shadow a broader declared atol.
    tolerance_rows = {
        key: contract
        for key, contract in table.items()
        if contract.metric is not Metric.ULP
    }
    return resolve_contract(tolerance_rows, query, label=op.name)


def enrolled_ops() -> Tuple[MathOperation, ...]:
    """Every op with a declared contract, in name order. For reporting and tests."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


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


def validate_registry() -> None:
    """Raise if any op can resolve ambiguously, for any variant a driver may ask about.

    Exhaustive over the variant space rather than a review convention: it is small, and
    an ambiguity that shows up for one format only is exactly what a reader misses.
    ``None`` is included on the axes a caller may leave unset, since an unset query
    dimension matches only a wildcard.
    """
    # The *input* axis comes from the table, not from the enum: a format no row pins
    # reproduces the `None` iteration exactly, since an unset key matches any value.
    input_formats = sorted(
        {key.input_format for table in _SFPU_ACCURACY_BUDGET.values() for key in table}
        - {None},
        key=lambda fmt: fmt.name,
    ) + [None]
    variants = product(
        [*ApproximationMode, None],
        input_formats,
        DataFormat,
        [*DestAccumulation, None],
        ChipArchitecture,
    )
    for op, (approx_mode, input_format, output_format, dest_acc, arch) in product(
        _SFPU_ACCURACY_BUDGET, variants
    ):
        accuracy_contract(
            op,
            output_format=output_format,
            input_format=input_format,
            approx_mode=approx_mode,
            dest_acc=dest_acc,
            arch=arch,
        )
