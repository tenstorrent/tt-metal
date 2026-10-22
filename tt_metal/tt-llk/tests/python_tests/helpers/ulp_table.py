# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SFPU accuracy table as typed rows, and the measurement records the sweep writes.

``sfpu_accuracy_budget.yaml`` is data and nothing else: every op maps to a list of
:class:`Row`, each written by :func:`dump_table` as one flow mapping. The YAML keys are
the :class:`Row` field names and every value is an enum member's name, a number or a
:class:`~helpers.ulp_domains.Domain` list, so :func:`from_plain` reads a row with no
per-field code, and anything else is refused with a typed
:class:`~helpers.ulp_domains.TableError`. ``--ulp-measure`` records are
:class:`Measurement` objects through the same two functions, one JSON object per line.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import ULP_FORMATS
from .ulp_domains import (
    Domain,
    Problem,
    TableError,
    check_domains,
    contract_of,
    usable_budget_ceiling,
)

TABLE_PATH = Path(__file__).with_name("sfpu_accuracy_budget.yaml")


class Metric(Enum):
    """Which gate a contract is written against."""

    ULP = "ulp"
    TOLERANCE = "tolerance"


def _check_number(value: Any, *, integer: bool = False, least: int = 0) -> None:
    """Refuse what the annotations cannot: a bool (YAML reads ``yes`` as one, and bool is
    an int), a non-finite value, or one below *least*."""
    kinds = int if integer else (int, float)
    if (
        isinstance(value, bool)
        or not isinstance(value, kinds)
        or not math.isfinite(value)
        or value < least
    ):
        raise TableError(Problem.BAD_NUMBER, value)


@dataclass(frozen=True)
class AccuracyContract:
    """How closely one op's output must match its golden, and by which metric.

    ``Metric.ULP`` means "every element within *max_ulp* steps", with the tolerance and
    PCC checks skipped; ``Metric.TOLERANCE`` is the historical gate. A *near_zero_atol*
    rides with a ULP contract, or with a list (*domains*), whose entries were measured
    outside it.
    """

    metric: Metric = Metric.ULP
    max_ulp: Optional[int] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    near_zero_atol: Optional[float] = None
    domains: Tuple[Domain, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            raise TableError(Problem.NOT_A_MEMBER, self.metric)
        for value in (self.atol, self.rtol, self.near_zero_atol):
            if value is not None:
                _check_number(value)
        if self.metric is Metric.ULP:
            if self.max_ulp is None:
                raise TableError(Problem.MISSING_FIGURE, self)
            _check_number(self.max_ulp, integer=True)
            if self.atol is not None or self.rtol is not None:
                raise TableError(Problem.MIXED_METRIC, self)
        elif self.max_ulp is not None or (
            self.near_zero_atol is not None and not self.domains
        ):
            raise TableError(Problem.MIXED_METRIC, self)

    @property
    def declares_tolerance(self) -> bool:
        """A tolerance contract that names an ``atol`` or ``rtol`` -- as opposed to a
        numberless tolerance row, which only opts its cell out of the ULP metric and must
        not retract a number the op declares on a broader row."""
        return self.metric is Metric.TOLERANCE and (
            self.atol is not None or self.rtol is not None
        )

    def tolerance_kwargs(self) -> Dict[str, Any]:
        """``passed_test`` arguments for a *tolerance-only* caller: the unary functional
        driver samples far fewer values than a sweep budget describes, so a ULP contract
        keeps today's per-format tolerance there."""
        if self.metric is Metric.ULP:
            return {}
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}

    def passed_test_kwargs(
        self, flush_subnormals: Optional[bool] = None
    ) -> Dict[str, Any]:
        """The contract as ``passed_test`` keyword arguments, whichever metric it is on.
        *flush_subnormals* overrides the ULP arm's per-dtype default; the tolerance arm
        has no such notion."""
        if self.metric is Metric.ULP:
            kwargs = {"max_ulp": self.max_ulp, "near_zero_atol": self.near_zero_atol}
            if flush_subnormals is not None:
                kwargs["flush_subnormals"] = flush_subnormals
            return kwargs
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}


#: What an unenrolled op -- or an enrolled one on a format the SFPU does not compute in
#: -- is judged by: the per-format defaults plus ``PCC > 0.99``.
TOLERANCE_CONTRACT = AccuracyContract(metric=Metric.TOLERANCE)


@dataclass(frozen=True)
class BudgetKey:
    """Which variants of an op a contract applies to -- and, on the query side, which
    variant is being asked about. Unset field == any value. More fields set is more
    specific, and the most specific match wins; two equally specific matches are an
    authoring error, not a tie-break."""

    approx_mode: Optional[ApproximationMode] = None
    input_format: Optional[DataFormat] = None
    output_format: Optional[DataFormat] = None
    dest_acc: Optional[DestAccumulation] = None
    arch: Optional[ChipArchitecture] = None

    def __post_init__(self) -> None:
        values = tuple(getattr(self, f.name) for f in fields(self))
        for value, expected in zip(values, _KEY_TYPES):
            # A bool never equals `DestAccumulation.No`: such a key would count as set
            # and match nothing.
            if value is not None and not isinstance(value, expected):
                raise TableError(Problem.NOT_A_MEMBER, value)
        # Cached: `matches` is validate_registry's hot loop.
        object.__setattr__(self, "_values", values)

    @property
    def specificity(self) -> int:
        return sum(value is not None for value in self._values)

    def matches(self, query: BudgetKey) -> bool:
        """Whether this key covers *query*. A dimension the query leaves unset matches
        only a wildcard."""
        a, i, o, d, r = self._values
        qa, qi, qo, qd, qr = query._values
        return (
            (a is None or a == qa)
            and (i is None or i == qi)
            and (o is None or o == qo)
            and (d is None or d == qd)
            and (r is None or r == qr)
        )

    def describe(self) -> str:
        set_fields = [
            f"{f.name}={getattr(self, f.name).name}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"BudgetKey({', '.join(set_fields)})" if set_fields else "DEFAULT"


#: The enum each :class:`BudgetKey` field must be a member of, in field order.
_KEY_TYPES = (
    ApproximationMode,
    DataFormat,
    DataFormat,
    DestAccumulation,
    ChipArchitecture,
)

#: Matches every variant of an op.
DEFAULT = BudgetKey()


@dataclass(frozen=True)
class Row:
    """One row of the table. The key fields say which variants it covers; the rest is
    its contract and the measurement behind it.

    A ULP row's gate is its ``measured`` figure. A tolerance row may record the
    ``measured`` figure and the ``nonfinite`` lane count the headroom report holds the
    cell to, and may declare ``atol``/``rtol``. A row the sweep writes for a cell instead
    carries its ``domains`` list, and no ``metric``, figure or bound beside it: the list
    is the whole statement. *note* is prose for people; no code reads it."""

    input: Optional[DataFormat] = None
    output: Optional[DataFormat] = None
    approx: Optional[ApproximationMode] = None
    dest: Optional[DestAccumulation] = None
    arch: Optional[ChipArchitecture] = None
    metric: Optional[Metric] = None
    measured: Optional[int] = None
    nonfinite: Optional[int] = None
    near_zero_atol: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    domains: Tuple[Domain, ...] = ()
    note: Optional[str] = None

    def __post_init__(self) -> None:
        for fmt in (self.input, self.output):
            if fmt is not None and fmt not in ULP_FORMATS:
                # A block float's error is the unpacker's and packer's, not the op's.
                raise TableError(Problem.NOT_A_COMPUTE_FORMAT, fmt)
        if self.measured is not None:
            _check_number(self.measured, integer=True)
        if self.nonfinite is not None:
            _check_number(self.nonfinite, integer=True, least=1)
        if self.domains:
            if (
                self.metric is not None
                or self.measured is not None
                or self.nonfinite is not None
                or self.atol is not None
                or self.rtol is not None
            ):
                raise TableError(Problem.BESIDE_DOMAINS, self)
            if self.input is None or self.output is None:
                raise TableError(Problem.UNPINNED_DOMAINS, self)
            check_domains(self.domains)
            if self.near_zero_atol is not None:
                _check_number(self.near_zero_atol)
            return
        if self.metric is not Metric.TOLERANCE:
            if self.measured is None:
                raise TableError(Problem.MISSING_FIGURE, self)
            if self.nonfinite is not None:
                raise TableError(Problem.NONFINITE_ON_BUDGET, self)
        # Constructed once, so a half-converted row is refused here, not on first use.
        object.__setattr__(self, "_contract", self._derive_contract())

    @property
    def key(self) -> BudgetKey:
        return BudgetKey(
            approx_mode=self.approx,
            input_format=self.input,
            output_format=self.output,
            dest_acc=self.dest,
            arch=self.arch,
        )

    @property
    def contract(self) -> AccuracyContract:
        if self.domains:
            gated, figure = contract_of(
                self.domains, usable_budget_ceiling(self.output)
            )
            return AccuracyContract(
                metric=Metric.ULP if gated else Metric.TOLERANCE,
                max_ulp=figure if gated else None,
                near_zero_atol=self.near_zero_atol,
                domains=self.domains,
            )
        return self._contract

    def _derive_contract(self) -> AccuracyContract:
        if self.metric is Metric.TOLERANCE:
            return AccuracyContract(
                metric=Metric.TOLERANCE,
                atol=self.atol,
                rtol=self.rtol,
                near_zero_atol=self.near_zero_atol,
            )
        return AccuracyContract(
            max_ulp=self.measured,
            atol=self.atol,
            rtol=self.rtol,
            near_zero_atol=self.near_zero_atol,
        )

    @property
    def unmeasurable(self) -> bool:
        """Whether lanes disagreeing with the golden about being finite held the cell
        on tolerance."""
        return self.nonfinite is not None


#: The table: each op's rows, in the order :func:`dump_table` writes them.
Table = Dict[MathOperation, Tuple[Row, ...]]


# ── Plain values: the one conversion between the dataclasses and YAML/JSON ─────


def to_plain(value: Any) -> Any:
    """*value* as YAML/JSON can hold it: a dataclass as a mapping of its fields that
    differ from their defaults, an enum member by name, a tuple as a list."""
    if is_dataclass(value):
        return {
            f.name: to_plain(getattr(value, f.name))
            for f in fields(value)
            if getattr(value, f.name) != f.default
        }
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (tuple, list)):
        return [to_plain(item) for item in value]
    return value


def from_plain(cls: type, data: Any) -> Any:
    """The dataclass *cls* from :func:`to_plain`'s output, every field converted by its
    annotation; anything that does not fit is a :class:`TableError`."""
    if not isinstance(data, dict):
        raise TableError(Problem.WRONG_SHAPE, data)
    hints = get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise TableError(Problem.UNKNOWN_FIELD, sorted(map(str, unknown)))
    return cls(**{name: _convert(hints[name], value) for name, value in data.items()})


def _convert(hint: Any, value: Any) -> Any:
    origin = get_origin(hint)
    if origin is Union:
        (inner,) = [a for a in get_args(hint) if a is not type(None)]
        return None if value is None else _convert(inner, value)
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(value, (list, tuple)):
            raise TableError(Problem.WRONG_SHAPE, value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value)
        if len(args) != len(value):
            raise TableError(Problem.WRONG_SHAPE, value)
        return tuple(_convert(a, item) for a, item in zip(args, value))
    if is_dataclass(hint):
        return from_plain(hint, value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        if not isinstance(value, str) or value not in hint.__members__:
            raise TableError(Problem.NOT_A_MEMBER, value)
        return hint[value]
    if hint is int:
        _check_number(value, integer=True, least=-(2**63))
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TableError(Problem.BAD_NUMBER, value)
        return float(value)
    if not isinstance(value, hint):
        raise TableError(Problem.WRONG_SHAPE, value)
    return value


# ── The table file ───────────────────────────────────────────────────────────


class _Loader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """PyYAML's safe loader, refusing a repeated mapping key: it would keep the last."""


def _construct_mapping(loader: _Loader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise TableError(
                Problem.DUPLICATE_YAML_KEY, f"line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def read_table(text: str) -> Table:
    """*text* as a :data:`Table`; a :class:`TableError` names the op and row it
    refuses."""
    loaded = yaml.load(text, Loader=_Loader) or {}
    if not isinstance(loaded, dict):
        raise TableError(Problem.WRONG_SHAPE, "the top level")
    table: Table = {}
    for name, rows in loaded.items():
        if not isinstance(name, str) or name not in MathOperation.__members__:
            raise TableError(Problem.UNKNOWN_OP, name)
        if not isinstance(rows, list) or not rows:
            raise TableError(Problem.NO_ROWS, name)
        typed = []
        for number, row in enumerate(rows, 1):
            try:
                typed.append(from_plain(Row, row))
            except TableError as error:
                raise TableError(error.problem, f"{name} row {number}: {error}")
        keys = [row.key for row in typed]
        if len(set(keys)) != len(keys):
            raise TableError(Problem.REPEATED_KEY, name)
        table[MathOperation[name]] = tuple(typed)
    return table


def load_table(path: Path = TABLE_PATH) -> Table:
    return read_table(Path(path).read_text(encoding="utf-8"))


_HEADER = """\
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The accuracy contract each SFPU op declares (helpers/ulp_table.py, tests/SFPU_ULP.md).
# Written by `test_unary_sfpu_ulp.py --ulp-emit` and helpers/ulp_table.dump_table.

"""


class _Dumper(yaml.SafeDumper):
    """One flow mapping per row, enum members by name, no line wrapping."""


_Dumper.add_representer(
    Row,
    lambda dumper, row: dumper.represent_mapping(
        "tag:yaml.org,2002:map", to_plain(row), flow_style=True
    ),
)


def _row_order(row: Row) -> Tuple:
    return tuple(
        (value is not None, value.name if value is not None else "")
        for value in (row.input, row.output, row.approx, row.dest, row.arch)
    )


def dump_table(table: Table) -> str:
    """*table* as the file holds it: ops by name, each op's rows by key."""
    data = {
        op.name: sorted(table[op], key=_row_order)
        for op in sorted(table, key=lambda op: op.name)
    }
    return _HEADER + yaml.dump(
        data,
        Dumper=_Dumper,
        sort_keys=False,
        width=math.inf,
        allow_unicode=True,
    )


def write_table_file(table: Table, path: Path = TABLE_PATH) -> None:
    """Dump *table* to *path*, after reading the dump back: a table that would not load
    is never written."""
    text = dump_table(table)
    read_table(text)
    Path(path).write_text(text, encoding="utf-8")


# ── Measurement records (--ulp-measure) ──────────────────────────────────────


@dataclass(frozen=True)
class Reading:
    """What the sweep measured over one entry of a cell's list: the worst lane outside
    the floor (``None`` where none is ranked) and how many lanes are ranked there."""

    low: float
    high: float
    worst: Optional[int]
    lanes: int


@dataclass(frozen=True)
class Measurement:
    """One comparison the recorder filed under the variant it ran: the worst ranked lane
    and how many lanes were ranked and unmeasurable, the lanes disagreeing with the
    golden about being finite, and for a sweep cell its per-binade profile (``bins``:
    ``(binade, present, ranked, worst)``) and its readings over the table's list."""

    op: MathOperation
    key: BudgetKey
    worst: int
    lanes: int
    unmeasurable: int = 0
    nonfinite: int = 0
    test: str = ""
    bins: Tuple[Tuple[int, int, int, int], ...] = ()
    readings: Tuple[Reading, ...] = ()

    def to_json(self) -> str:
        return json.dumps(to_plain(self))

    @classmethod
    def from_json(cls, line: str) -> "Measurement":
        return from_plain(cls, json.loads(line))


def read_measurements(path: Path) -> List[Measurement]:
    """The records in *path*; none for a missing file, which a sweep that died before
    its first cell leaves."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        Measurement.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
