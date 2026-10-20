# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data model and transitive effect resolution for the Blackhole state audit.

* :class:`Resource` - one canonical resource (or resource class) of the taxonomy.
* :class:`Effect`   - one read or write of one canonical resource, with value,
  condition, evidence (``path:line`` of the deepest sink), thread and the call
  chain (``via``) that reaches it from the reporting function.
* :class:`Pattern`  - a declarative recognizer families export; the effect DB
  turns every match into Effects.
* :class:`CallEdge` / :class:`FunctionModel` - the call graph node with its role
  and direct effects.
* :class:`CallGraph` - context-sensitive transitive resolution: callee effects
  are re-expressed in the caller's scope by substituting the call's (simple)
  argument expressions for the callee's parameters, conditions that fold to
  ``false`` are dropped, and resource names that depended on a parameter are
  re-normalized (e.g. ``addr_mod_t::set(mod_index)`` called with ``ADDR_MOD_3``
  yields ``ADDR_MOD.3``). Calls through callable parameters (``sfpu_func(..)``)
  resolve through the bound argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

ACCESS = ("read", "write")
VALUE_KINDS = ("const", "param", "expr", "unknown")
THREAD_VALUES = ("unpack", "math", "pack", "any")
ROLES = ("init", "uninit", "execute", "reconfig", "config", "helper", "hw_startup")
CONFIDENCE = ("high", "medium", "low")

_IDENT = re.compile(r"(?<![\w.$])(?<!->)(?<!::)([A-Za-z_]\w*)\b(?!\s*::)")
_ANY_IDENT = re.compile(r"[A-Za-z_]\w*")
_MAX_TEXT = 300
_MAX_EXPR = 4000
_MAX_DEPTH = 24


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Resource:
    """A canonical resource name (``cls`` is the prefix before the first dot)."""

    name: str
    cls: str
    scope: str = "shared"  # shared | per_thread | thread_named
    description: str = ""
    example: str = ""

    def to_json(self) -> dict:
        return {
            "class": self.cls,
            "description": self.description,
            "example": self.example,
            "name": self.name,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class Effect:
    """One access of one canonical resource.

    ``params``: parameters of the *reporting* function whose value flows into the
    resource, value or condition. ``via``: ``((callee_key, call_evidence), ...)``
    from the reporting function down to the function holding the sink; empty for
    a direct effect. ``evidence`` is always the sink's ``path:line``.
    """

    resource: str
    access: str  # read | write
    value_kind: str  # const | param | expr | unknown
    value: str
    condition: str  # "" when unconditional
    evidence: str  # repo-relative path:line of the sink
    thread: str  # unpack | math | pack | any
    family: str = "core"
    pattern: str = ""
    granularity: str = "field"
    detail: str = ""
    resolved: bool = True
    confidence: str = "high"
    params: tuple[str, ...] = ()
    via: tuple[tuple[str, str], ...] = ()
    resource_expr: str = ""
    normalizer: str = "literal"
    norm_index: int = 0

    @property
    def resource_class(self) -> str:
        return (
            "param"
            if self.resource.startswith("param:")
            else self.resource.split(".", 1)[0]
        )

    @property
    def is_direct(self) -> bool:
        return not self.via

    @property
    def sink_function(self) -> str | None:
        return self.via[-1][0] if self.via else None

    def identity(self) -> tuple:
        return (
            self.resource,
            self.access,
            self.value,
            self.condition,
            self.evidence,
            self.thread,
            self.granularity,
            self.detail,
            self.pattern,
            self.via,
        )

    def sort_key(self) -> tuple:
        return (
            self.evidence_sort(),
            self.resource,
            self.access,
            self.pattern,
            self.value,
            self.condition,
            self.thread,
            self.detail,
            self.via,
        )

    def evidence_sort(self) -> tuple:
        path, _, line = self.evidence.rpartition(":")
        return (path, int(line) if line.isdigit() else 0)

    def to_json(self) -> dict:
        return {
            "access": self.access,
            "condition": cap(self.condition),
            "confidence": self.confidence,
            "detail": self.detail,
            "evidence": self.evidence,
            "family": self.family,
            "granularity": self.granularity,
            "params": list(self.params),
            "pattern": self.pattern,
            "resolved": self.resolved,
            "resource": self.resource,
            "thread": self.thread,
            "value": self.value,
            "value_kind": self.value_kind,
            "via": [list(step) for step in self.via],
        }


@dataclass(frozen=True)
class PatternMatch:
    """What a pattern sees for one match (call site, regex or assignment)."""

    name: str  # callee final identifier, regex match text, or assigned base name
    callee: str = ""
    template_args: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    groups: tuple[tuple[str, str], ...] = ()  # named regex groups
    index: str = ""  # assign: subscript expression
    rhs: str = ""  # assign: right-hand side ("op= rhs" for compound)
    op: str = ""  # assign: operator
    receiver: str = ""
    receiver_type: str = ""
    text: str = ""
    line: int = 0
    offset: int = 0
    thread: str | None = None
    condition: str | None = None
    via_macro: str | None = None

    def group(self, name: str) -> str | None:
        for key, value in self.groups:
            if key == name:
                return value
        return None


Spec = "str | Callable[[PatternMatch], str | None] | None"


@dataclass(frozen=True)
class Pattern:
    """Declarative effect recognizer exported by a family (see SCHEMA.md).

    kind:
      ``call``   - ``match`` is a regex full-matched against the final identifier of
                   every call site / macro invocation (``r"TTI?_SETC16"``);
      ``member`` - like ``call`` but only member calls; ``receiver_type`` (regex)
                   must full-match the inferred receiver class;
      ``ctor``   - ``Type var(args)`` declarations; ``match`` is the type name;
      ``regex``  - ``match`` is searched in the masked body and in every macro
                   expansion; named groups are available to specs;
      ``assign`` - ``match`` is a regex for the assigned base name (``cfg``,
                   ``regfile``, ``vConstFloatPrgm\\d``); matches ``NAME[idx] op= rhs``
                   and ``NAME op= rhs``.
    resource/value/detail specs: ``"arg:N"``, ``"targ:N"``, ``"args:A,B"``,
    ``"targs"``, ``"allargs"``, ``"group:G"``, ``"index"``, ``"rhs"``, ``"name"``,
    ``"receiver"``, ``"literal:TEXT"``, ``"fmt:...{arg0}..{targ1}..{index}..{group:G}.."``
    or a callable ``(PatternMatch) -> str | None``.
    normalizer: a name registered in ``resources.ResourceTables.normalizers``
    (``cfg_rmw``, ``cfg_addr32``, ``gpr``, ``sfpconfig_dest``, ...) or a callable
    ``(expr, thread) -> list[Normalized] | str | None``.
    """

    id: str
    kind: str
    match: str
    resource: Any
    normalizer: Any = "literal"
    access: str = "write"
    value: Any = None
    detail: Any = None
    thread: str = "inherit"  # inherit | unpack | math | pack | any
    granularity: str | None = None  # override the normalizer's granularity
    receiver_type: str | None = None
    qualifier: str | None = None  # regex the call qualifier must full-match
    when: Callable[[PatternMatch], bool] | None = None
    stop: bool = True  # for call/member/ctor: do not descend into a scanned callee
    confidence: str = "high"
    note: str = ""


@dataclass(frozen=True)
class CallEdge:
    """A call site of a function with its statically resolved callee keys."""

    name: str
    callee: str
    callee_keys: tuple[str, ...]
    resolution: (
        str  # exact | overload | ambiguous | unresolved | param_callable | stopped
    )
    line: int
    evidence: str
    thread: str | None
    condition: str | None
    template_args: tuple[str, ...]
    args: tuple[str, ...]
    via_macro: str | None = None
    is_member: bool = False
    receiver_type: str | None = None
    is_ctor: bool = False
    stop: bool = False  # consumed by a stop pattern (no descent)
    patterns: tuple[str, ...] = ()  # ids of patterns that matched this call
    qualifier: str = ""

    def to_json(self) -> dict:
        return {
            "args": list(self.args),
            "callee": self.callee,
            "callee_keys": list(self.callee_keys),
            "condition": self.condition,
            "evidence": self.evidence,
            "is_ctor": self.is_ctor,
            "is_member": self.is_member,
            "name": self.name,
            "patterns": list(self.patterns),
            "receiver_type": self.receiver_type,
            "resolution": self.resolution,
            "stop": self.stop,
            "template_args": list(self.template_args),
            "thread": self.thread,
            "via_macro": self.via_macro,
        }


@dataclass
class FunctionModel:
    key: str
    name: str
    qualified_name: str
    path: str
    line: int
    end_line: int
    layer: str  # compute_api | llk_api | llk_lib | common | sfpu
    role: str  # init | uninit | execute | reconfig | config | helper | hw_startup
    thread: str  # unpack | math | pack | any
    params: tuple  # scan.Param (runtime)
    template_params: tuple  # scan.Param (template)
    signature: str
    effects: tuple[Effect, ...] = ()  # direct effects
    calls: tuple[CallEdge, ...] = ()
    attributes: tuple[str, ...] = ()
    pp_condition: str | None = None
    fingerprint: str = ""
    owner: str = ""
    local_constants: tuple[
        tuple[str, str], ...
    ] = ()  # constexpr/const single-assignment locals
    definition: Any = None  # scan.FunctionDef (not serialized)

    @property
    def all_params(self) -> tuple:
        return tuple(self.template_params) + tuple(self.params)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.all_params if p.name)

    @property
    def loc(self) -> str:
        return f"{self.path}:{self.line}"

    def to_json(self, transitive: Sequence[Effect] | None = None) -> dict:
        data = {
            "attributes": list(self.attributes),
            "calls": [edge.to_json() for edge in self.calls],
            "effects": [effect.to_json() for effect in self.effects],
            "end_line": self.end_line,
            "fingerprint": self.fingerprint,
            "key": self.key,
            "layer": self.layer,
            "line": self.line,
            "name": self.name,
            "owner": self.owner,
            "params": [p.to_json() for p in self.params],
            "path": self.path,
            "pp_condition": self.pp_condition,
            "qualified_name": self.qualified_name,
            "role": self.role,
            "signature": self.signature,
            "template_params": [p.to_json() for p in self.template_params],
            "thread": self.thread,
        }
        if transitive is not None:
            data["transitive_summary"] = summarize_effects(transitive)
        return data


def summarize_effects(effects: Sequence[Effect], *, keep: int = 3) -> list[dict]:
    """Compact JSON view of transitive effects: one row per (resource, access, thread).

    ``count`` effects, ``conditional`` true when every one is conditional, the first
    ``keep`` distinct evidences / values / conditions, the params that flow in, and
    the shortest ``via`` chain (callee keys only). Full detail: ``EffectDB.effects``.
    """
    groups: dict[tuple[str, str, str], list[Effect]] = {}
    for effect in effects:
        groups.setdefault((effect.resource, effect.access, effect.thread), []).append(
            effect
        )
    rows = []
    for (resource, access, thread), items in sorted(groups.items()):
        items = sorted(items, key=lambda e: e.sort_key())

        def first(values):
            seen: list[str] = []
            for value in values:
                if value and value not in seen:
                    seen.append(value)
                if len(seen) >= keep:
                    break
            return seen

        shortest = min(items, key=lambda e: (len(e.via), e.sort_key()))
        rows.append(
            {
                "access": access,
                "conditional": all(bool(e.condition) for e in items),
                "conditions": first(cap(e.condition) for e in items),
                "count": len(items),
                "evidence": first(e.evidence for e in items),
                "families": sorted({e.family for e in items}),
                "params": sorted({p for e in items for p in e.params}),
                "resolved": all(e.resolved for e in items),
                "resource": resource,
                "thread": thread,
                "values": first(e.value for e in items),
                "via": [step[0] for step in shortest.via],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def compact(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def identifiers(text: str | None) -> set[str]:
    """Bare identifiers (not member names, not namespace qualifiers)."""
    if not text:
        return set()
    return set(_IDENT.findall(text))


def simplify_arg(arg: str) -> str:
    """Strip forwarding wrappers: ``std::forward<T>(x)``, ``std::move(x)``, casts to ``T&&``."""
    text = compact(arg)
    for _ in range(4):
        match = re.fullmatch(
            r"(?:::)?std::(?:forward\s*<[^()]*>|move)\s*\((.*)\)", text
        )
        if match:
            text = compact(match.group(1))
            continue
        match = re.fullmatch(r"static_cast\s*<[^()]*&&\s*>\s*\((.*)\)", text)
        if match:
            text = compact(match.group(1))
            continue
        break
    return text


def substitute(text: str, bindings: dict[str, str]) -> str:
    """Replace bound parameter identifiers in ``text`` with their argument text."""
    if not text or not bindings:
        return text

    def replacement(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in bindings:
            return match.group(0)
        value = bindings[name]
        if match.start() == 0 and match.end() == len(text):
            return value
        if re.fullmatch(r"[\w:.<>, ]+|\(.*\)|-?\d[\w.]*", value):
            return value
        return f"({value})"

    result = compact(_IDENT.sub(replacement, text))
    if len(result) > _MAX_EXPR:
        result = result[: _MAX_EXPR - 3] + "..."
    return result


def cap(text: str, limit: int = _MAX_TEXT) -> str:
    """Display cap for value/condition/detail text (applied after evaluation)."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


#: Words that are literals, never parameters of a caller scope.
_LITERAL_WORDS = frozenset({"true", "false", "nullptr"})

_CONST_TOKEN = re.compile(
    r"^(?:true|false|nullptr|[A-Z][A-Z0-9_]*|to_underlying|static_cast|reinterpret_cast|uint\w*|int\w*|std|sizeof)$"
)


def classify_value(
    value: str, params: Iterable[str], constants: set[str] | None = None
) -> str:
    """const | param | expr | unknown, relative to the reporting function."""
    text = compact(value)
    if not text:
        return "unknown"
    params = set(params)
    if text in params:
        return "param"
    if "?" in text:
        return "unknown" if text.startswith("?") and text[1:].isidentifier() else "expr"
    if re.match(r"GPR\(", text):
        return "expr"  # the content of a GPR, not a constant (WRCFG/cfg_rmw_gpr values)
    names = identifiers(text)
    if names & params:
        return "expr"
    constants = constants or set()
    for name in names:
        if _CONST_TOKEN.match(name) or name in constants:
            continue
        return "expr"
    return "const"


def _low_precedence(text: str) -> bool:
    """Top-level ``||`` or ``?:`` (outside brackets): needs parentheses under ``&&``."""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and (char == "?" or text.startswith("||", index)):
            return True
        index += 1
    return False


def _deferred_part(detail: str) -> bool:
    return any(
        part.strip().split(":", 1)[0] == "deferred"
        for part in (detail or "").split(";")
    )


def merge_detail(normalizer_detail: str, pattern_detail: str) -> str:
    """A normalizer's detail (``LO_16``, ``mask=..``) plus the pattern's (``deferred``), deduplicated."""
    parts = [p for p in (normalizer_detail or "").split(";") if p]
    for part in (pattern_detail or "").split(";"):
        if part and part not in parts:
            parts.append(part)
    return ";".join(parts)


def keep_deferred(new_detail: str, old_detail: str) -> str:
    """``new_detail`` (re-normalized) keeping the ``deferred`` marker of ``old_detail``."""
    if _deferred_part(old_detail) and not _deferred_part(new_detail):
        return f"{new_detail};deferred" if new_detail else "deferred"
    return new_detail


def combine_conditions(*parts: str | None) -> str:
    """``&&``-join conditions (deduplicated), parenthesizing ``||`` / ``?:`` parts."""
    unique: list[str] = []
    for part in parts:
        item = compact(part)
        if not item:
            continue
        if _low_precedence(item):
            item = f"({item})"
        if item not in unique:
            unique.append(item)
    return " && ".join(unique)


# --------------------------------------------------------------------------- #
# Transitive resolution
# --------------------------------------------------------------------------- #
class CallGraph:
    """Resolves calls and computes context-sensitive transitive effects.

    ``functions``: key -> FunctionModel. ``normalize(kind, expr, thread)`` comes
    from ``resources.ResourceTables.normalize``. ``truth(text)`` folds conditions
    (``scan.truth``). ``arg_expanders`` rewrite argument lists before binding
    (``resources.expand_rmw_args``).
    """

    def __init__(
        self,
        functions: dict[str, FunctionModel],
        *,
        normalize: Callable[[str, str, str | None], list],
        truth: Callable[[str], bool | None],
        arg_expanders: Sequence[Callable[[Iterable[str]], tuple[str, ...]]] = (),
        constants: set[str] | None = None,
        custom_normalizers: dict[str, Callable] | None = None,
        simplify: Callable[[str], tuple[bool | None, str]] | None = None,
    ):
        self.functions = functions
        self.normalize = normalize
        self.truth = truth
        self.simplify = simplify
        self.arg_expanders = tuple(arg_expanders)
        self.constants = constants or set()
        self.custom_normalizers = custom_normalizers or {}
        self._by_name: dict[str, list[FunctionModel]] = {}
        for key in sorted(functions):
            fn = functions[key]
            self._by_name.setdefault(fn.name, []).append(fn)
        self._memo: dict[tuple, tuple[Effect, ...]] = {}
        self._relevant: dict[str, frozenset[str]] = {}
        self._stack: list[str] = []

    # ---- call resolution -------------------------------------------------- #
    def by_name(self, name: str) -> list[FunctionModel]:
        return self._by_name.get(name, [])

    def candidates(
        self,
        name: str,
        *,
        qualifier: str = "",
        template_args: Sequence[str] = (),
        args: Sequence[str] = (),
        is_member: bool = False,
        receiver_type: str | None = None,
        is_ctor: bool = False,
        caller: FunctionModel | None = None,
    ) -> tuple[list[FunctionModel], str]:
        """Return (candidates, resolution) for a call."""
        pool = self.by_name(name)
        if not pool:
            return [], "unresolved"
        if is_ctor:
            pool = [
                fn
                for fn in pool
                if fn.owner.rsplit("::", 1)[-1] == name
                and fn.definition is not None
                and fn.definition.is_ctor
            ]
        elif is_member:
            methods = [fn for fn in pool if fn.owner]
            if receiver_type:
                pool = [
                    fn
                    for fn in methods
                    if fn.owner.rsplit("::", 1)[-1] == receiver_type
                ]
            else:
                owners = {fn.owner for fn in methods}
                pool = methods if 0 < len(owners) <= 3 else []
        else:
            caller_owner = caller.owner if caller is not None else ""
            free = [
                fn
                for fn in pool
                if not fn.owner or (caller_owner and fn.owner == caller_owner)
            ]
            if qualifier:
                parts = [part for part in qualifier.split("::") if part]
                qualified = [
                    fn
                    for fn in pool
                    if fn.qualified_name.endswith(qualifier.lstrip(":") + "::" + name)
                    or (parts and parts[-1] in (fn.qualified_name.split("::")[:-1]))
                ]
                pool = qualified or free
            else:
                visible = [fn for fn in free if _visible(fn, caller)]
                pool = visible or free
        expanded_targs = self._expand(template_args)
        expanded_args = self._expand(args)
        packs = sum(1 for a in expanded_args if compact(a).endswith("..."))
        tpacks = sum(1 for a in expanded_targs if compact(a).endswith("..."))
        compatible = [
            fn
            for fn in pool
            if _arity_ok(
                fn,
                len(expanded_targs) - tpacks,
                len(expanded_args) - packs,
                flexible=bool(packs),
                tflexible=bool(tpacks),
            )
        ]
        if not compatible:
            return [], "unresolved"
        resolution = "exact" if len(compatible) == 1 else "overload"
        if len(compatible) > 1:
            spec_targs = tuple(compact(a).replace(" ", "") for a in template_args)
            specialized = [
                fn
                for fn in compatible
                if fn.definition is not None
                and fn.definition.specialization_args
                and tuple(
                    compact(a).replace(" ", "")
                    for a in fn.definition.specialization_args
                )
                == spec_targs
            ]
            if specialized:
                compatible = specialized
            else:
                primary = [
                    fn
                    for fn in compatible
                    if fn.definition is None or not fn.definition.specialization_args
                ]
                compatible = primary or compatible
        if len(compatible) > 1:
            deducible = [fn for fn in compatible if _deducible(fn, len(expanded_targs))]
            if deducible:
                compatible = deducible
        if len(compatible) > 1 and expanded_targs:
            scored = [(_targ_score(fn, expanded_targs), fn) for fn in compatible]
            best = max(score for score, _ in scored)
            compatible = [fn for score, fn in scored if score == best]
        if len(compatible) > 1 and caller is not None:
            scored = [(_type_score(fn, expanded_args, caller), fn) for fn in compatible]
            best = max(score for score, _ in scored)
            if best > 0:
                compatible = [fn for score, fn in scored if score == best]
        if len(compatible) > 1 and caller is not None:
            same_file = [fn for fn in compatible if fn.path == caller.path]
            if same_file:
                compatible = same_file
        if len(compatible) > 1:
            unconditional = [fn for fn in compatible if not fn.pp_condition]
            if unconditional and len(unconditional) < len(compatible):
                compatible = unconditional
        if len(compatible) > 1:
            resolution = "ambiguous"
        compatible.sort(key=lambda fn: fn.key)
        return compatible, resolution

    def _expand(self, values: Sequence[str]) -> tuple[str, ...]:
        result = tuple(values)
        for expander in self.arg_expanders:
            result = tuple(expander(result))
        return result

    def bind(
        self, callee: FunctionModel, template_args: Sequence[str], args: Sequence[str]
    ) -> dict[str, str]:
        """Map callee parameter names to argument texts (defaults when omitted)."""
        bindings: dict[str, str] = {}
        for params, values in (
            (callee.template_params, self._expand(template_args)),
            (callee.params, self._expand(args)),
        ):
            values = [simplify_arg(v) for v in values]
            if any(v.endswith("...") for v in values) and not any(
                p.variadic for p in params
            ):
                values = [v for v in values if not v.endswith("...")]
            position = 0
            for param in params:
                if not param.name:
                    position += 1
                    continue
                if param.variadic:
                    bindings[param.name] = ", ".join(values[position:])
                    position = len(values)
                    continue
                if position < len(values):
                    bindings[param.name] = values[position]
                elif param.default is not None:
                    bindings[param.name] = substitute(param.default, bindings)
                else:
                    bindings[param.name] = f"?{param.name}"
                position += 1
        return bindings

    # ---- relevance -------------------------------------------------------- #
    def relevant_params(self, fn: FunctionModel) -> frozenset[str]:
        cached = self._relevant.get(fn.key)
        if cached is not None:
            return cached
        names: set[str] = set()
        for effect in fn.effects:
            names |= (
                identifiers(effect.resource_expr)
                | identifiers(effect.value)
                | identifiers(effect.condition)
                | identifiers(effect.detail)
            )
        for edge in fn.calls:
            names.add(edge.name)
            for text in (*edge.args, *edge.template_args, edge.condition or ""):
                names |= identifiers(text)
        for _, expr in fn.local_constants:
            names |= identifiers(expr)
        result = frozenset(names & set(fn.param_names))
        self._relevant[fn.key] = result
        return result

    @staticmethod
    def frame_bindings(fn: FunctionModel, bound: dict[str, str]) -> dict[str, str]:
        """``bound`` plus the function's local constants (themselves substituted)."""
        if not fn.local_constants:
            return bound
        frame = dict(bound)
        for name, expr in fn.local_constants:
            if name in frame:
                continue
            frame[name] = substitute(expr, frame)
        return frame

    # ---- effects ---------------------------------------------------------- #
    def effects(
        self, key: str, bindings: dict[str, str] | None = None
    ) -> tuple[Effect, ...]:
        """Transitive effects of ``key`` in its own scope (``bindings`` empty) or,
        with bindings, re-expressed in the scope that supplied them.

        ``Effect.params`` lists the identifiers of the supplying scope that flow
        in (for an unbound call: the function's own parameter names) and
        ``value_kind`` is computed relative to them.
        """
        fn = self.functions.get(key)
        if fn is None:
            return ()
        bindings = bindings or {}
        result = self._resolve(fn, bindings, 0)
        own = set(fn.param_names)
        finished = []
        for effect in result:
            params = (
                tuple(sorted(effect.params))
                if bindings
                else tuple(sorted(set(effect.params) & own))
            )
            finished.append(
                replace(
                    effect,
                    params=params,
                    value_kind=classify_value(effect.value, params, self.constants),
                )
            )
        finished = _dedupe(finished)
        return tuple(finished)

    def _resolve(
        self, fn: FunctionModel, bindings: dict[str, str], depth: int
    ) -> tuple[Effect, ...]:
        relevant = self.relevant_params(fn)
        bound = {
            name: compact(value) for name, value in bindings.items() if name in relevant
        }
        memo_key = (fn.key, tuple(sorted(bound.items())))
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached
        if fn.key in self._stack or depth > _MAX_DEPTH:
            return ()
        self._stack.append(fn.key)
        try:
            own_params = set(fn.param_names)
            local: list[Effect] = list(fn.effects)
            for edge in fn.calls:
                if edge.stop:
                    continue
                child = self._edge_effects(fn, edge, bound, depth)
                if not child:
                    continue
                for effect in child:
                    params = tuple(sorted(set(effect.params) & own_params))
                    thread = effect.thread
                    if thread == "any":
                        thread = edge.thread or (
                            fn.thread if fn.thread != "any" else "any"
                        )
                    local.append(
                        replace(
                            effect,
                            params=params,
                            thread=thread,
                            condition=combine_conditions(
                                edge.condition, effect.condition
                            ),
                        )
                    )
            # apply bindings (caller scope) and local constants to every effect of this frame
            frame = self.frame_bindings(fn, bound)
            out: list[Effect] = []
            for effect in local:
                substituted = self._apply(effect, frame, own_params)
                if substituted is not None:
                    out.extend(substituted)
            result = tuple(_dedupe(out))
        finally:
            self._stack.pop()
        self._memo[memo_key] = result
        return result

    def _edge_effects(
        self, fn: FunctionModel, edge: CallEdge, bound: dict[str, str], depth: int
    ) -> list[Effect]:
        """Effects of one call edge, expressed in ``fn``'s scope (unbound params)."""
        callees: list[FunctionModel]
        resolution = edge.resolution
        variadic = {p.name for p in fn.all_params if p.variadic and p.name}
        template_args = (
            expand_packs(edge.template_args, bound, variadic)
            if variadic
            else edge.template_args
        )
        args = expand_packs(edge.args, bound, variadic) if variadic else edge.args
        if edge.resolution == "param_callable":
            target = bound.get(edge.name)
            if not target or target.startswith("?"):
                return []
            parsed = _parse_callable(target)
            if parsed is None:
                return []
            name, qualifier, targs = parsed
            callees, resolution = self.candidates(
                name, qualifier=qualifier, template_args=targs, args=args, caller=fn
            )
            template_args = targs
        elif (
            edge.resolution in ("ambiguous", "overload")
            and bound
            and any(identifiers(text) & set(bound) for text in template_args)
        ):
            substituted = tuple(substitute(text, bound) for text in template_args)
            callees, resolution = self.candidates(
                edge.name,
                qualifier=edge.qualifier,
                template_args=substituted,
                args=args,
                is_member=edge.is_member,
                receiver_type=edge.receiver_type,
                is_ctor=edge.is_ctor,
                caller=fn,
            )
            allowed = set(edge.callee_keys)
            callees = [c for c in callees if c.key in allowed] or [
                self.functions[k] for k in edge.callee_keys if k in self.functions
            ]
        else:
            callees = [
                self.functions[k] for k in edge.callee_keys if k in self.functions
            ]
            if args != edge.args or template_args != edge.template_args:
                packs = sum(1 for a in args if compact(a).endswith("..."))
                callees = [
                    c
                    for c in callees
                    if _arity_ok(
                        c,
                        len(self._expand(template_args)),
                        len(self._expand(args)) - packs,
                        flexible=bool(packs),
                    )
                ] or callees
        if not callees:
            return []
        effects: list[Effect] = []
        for callee in callees:
            child_bindings = self.bind(callee, template_args, args)
            # the binding texts are in fn's scope; if fn itself is bound, they will be
            # substituted again by _resolve(fn) through _apply.
            child = self._resolve(callee, child_bindings, depth + 1)
            for effect in child:
                confidence = effect.confidence
                if resolution == "ambiguous" or not effect.resolved:
                    confidence = "low"
                elif confidence == "high":
                    confidence = "medium"
                effects.append(
                    replace(
                        effect,
                        confidence=confidence,
                        via=((callee.key, edge.evidence),) + effect.via,
                    )
                )
        return effects

    def _apply(
        self, effect: Effect, bound: dict[str, str], own_params: set[str]
    ) -> list[Effect] | None:
        """Substitute ``bound`` into an effect expressed in a frame's scope."""
        params_in_effect = set(effect.params) | (
            (
                identifiers(effect.resource_expr)
                | identifiers(effect.value)
                | identifiers(effect.condition)
                | identifiers(effect.detail)
            )
            & own_params
        )
        if not bound:
            return [replace(effect, params=tuple(sorted(params_in_effect)))]
        condition = substitute(effect.condition, bound) if effect.condition else ""
        if condition:
            # the full condition is kept (display capping happens in to_json): a capped
            # text no longer folds one level up
            if self.simplify is not None:
                verdict, condition = self.simplify(condition)
            else:
                verdict = self.truth(condition)
            if verdict is False:
                return None
            if verdict is True:
                condition = ""
        value = cap(substitute(effect.value, bound)) if effect.value else ""
        detail = cap(substitute(effect.detail, bound)) if effect.detail else ""
        new_params: set[str] = set()
        for name in params_in_effect:
            if name in bound:
                new_params |= identifiers(bound[name]) - _LITERAL_WORDS
            else:
                new_params.add(name)
        resource_expr = (
            substitute(effect.resource_expr, bound)
            if effect.resource_expr
            else effect.resource_expr
        )
        base = replace(
            effect,
            condition=condition,
            value=value,
            detail=detail,
            params=tuple(sorted(new_params)),
        )
        if resource_expr == effect.resource_expr:
            return [base]
        normalized = self._normalize(effect.normalizer, resource_expr, effect.thread)
        if not normalized:
            return [replace(base, resource_expr=resource_expr)]
        if not effect.resolved:
            picks = list(enumerate(normalized))
        else:
            index = min(effect.norm_index, len(normalized) - 1)
            picks = [(index, normalized[index])]
        result = []
        for index, item in picks:
            result.append(
                replace(
                    base,
                    resource=item.name,
                    granularity=item.granularity,
                    detail=keep_deferred(item.detail or detail, detail),
                    resolved=item.resolved,
                    resource_expr=resource_expr,
                    norm_index=index,
                )
            )
        return result

    def _normalize(self, normalizer: str, expr: str, thread: str):
        function = self.custom_normalizers.get(normalizer)
        try:
            if function is not None:
                result = function(expr, thread=thread)
                if result is None:
                    return []
                return [result] if not isinstance(result, list) else result
            return self.normalize(normalizer, expr, thread)
        except KeyError:
            return []

    def reachable(
        self, key: str, *, include_stopped: bool = False
    ) -> list[tuple[str, int, str]]:
        """(callee key, depth, call evidence) reachable from ``key`` (static edges)."""
        seen: dict[str, tuple[int, str]] = {}
        frontier = [(key, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            fn = self.functions.get(current)
            if fn is None:
                continue
            for edge in fn.calls:
                if edge.stop and not include_stopped:
                    continue
                for callee in edge.callee_keys:
                    if callee not in seen and callee != key:
                        seen[callee] = (depth + 1, edge.evidence)
                        frontier.append((callee, depth + 1))
        return sorted(
            ((k, d, e) for k, (d, e) in seen.items()),
            key=lambda item: (item[1], item[0]),
        )


_TYPE_NOISE = {
    "const",
    "volatile",
    "std",
    "ckernel",
    "experimental",
    "typename",
    "class",
    "struct",
    "unsigned",
    "signed",
}


def _type_tokens(text: str) -> set[str]:
    return {token for token in _ANY_IDENT.findall(text or "")} - _TYPE_NOISE


def _deducible(fn: FunctionModel, n_targs: int) -> bool:
    """Template params not given explicitly need a default or must appear in a runtime param type."""
    if fn.definition is not None and fn.definition.specialization_args:
        return True
    runtime_types = " ".join(p.type for p in fn.params)
    runtime_tokens = _type_tokens(runtime_types)
    for param in fn.template_params[n_targs:]:
        if param.variadic or param.default is not None or not param.name:
            continue
        if param.name not in runtime_tokens:
            return False
    return True


_INTEGRAL = {
    "bool",
    "int",
    "uint",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "size_t",
    "unsigned",
    "char",
    "short",
    "long",
}


def _targ_score(fn: FunctionModel, targs: Sequence[str]) -> int:
    """Score explicit template arguments against template parameter kinds."""
    score = 0
    for param, arg in zip(fn.template_params, targs):
        text = compact(arg)
        tokens = _type_tokens(param.type)
        if not tokens or tokens <= {"typename", "class"}:
            continue
        scoped = re.match(
            r"^(?:::)?((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)::[A-Za-z_]\w*$", text
        )
        if scoped:
            scope = scoped.group(1).split("::")[-1]
            if scope in tokens:
                score += 2
            elif tokens & _INTEGRAL:
                score += 0
            else:
                score -= 2
        elif text in ("true", "false"):
            score += 1 if "bool" in tokens else (-2 if not tokens & _INTEGRAL else 0)
        elif re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|\d+)[uUlL]*", text):
            score += 1 if tokens & _INTEGRAL else -2
    return score


def _type_score(fn: FunctionModel, args: Sequence[str], caller: FunctionModel) -> int:
    """Prefer overloads whose parameter types match caller-parameter / scoped-enum arguments."""
    caller_types = {p.name: p.type for p in caller.all_params if p.name}
    score = 0
    for param, arg in zip(fn.params, args):
        text = compact(arg)
        if text in caller_types:
            caller_tokens = _type_tokens(caller_types[text])
            callee_tokens = _type_tokens(param.type)
            if caller_tokens & callee_tokens:
                score += 2
            elif caller_tokens and callee_tokens:
                score -= 1
        scoped = re.match(r"^((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)::[A-Za-z_]\w*$", text)
        if scoped and scoped.group(1).split("::")[-1] in _type_tokens(param.type):
            score += 1
    return score


def _namespace_of(fn: FunctionModel) -> str:
    if fn.definition is not None:
        return fn.definition.namespace
    parts = fn.qualified_name.split("::")[:-1]
    return "::".join(parts)


def _visible(candidate: FunctionModel, caller: FunctionModel | None) -> bool:
    """Unqualified-lookup visibility: enclosing namespaces plus using-directives."""
    namespace = _namespace_of(candidate)
    if not namespace or caller is None:
        return True
    caller_namespace = (
        _namespace_of(caller)
        if caller.definition is not None or "::" in caller.qualified_name
        else ""
    )
    if caller_namespace == namespace or caller_namespace.startswith(namespace + "::"):
        return True
    using = caller.definition.using if caller.definition is not None else ()
    return namespace in using


def _parse_callable(text: str) -> tuple[str, str, tuple[str, ...]] | None:
    """``::ns::fn<a, b>`` -> ("fn", "ns", ("a", "b"))."""
    from . import scan

    text = simplify_arg(text)
    text = re.sub(r"^&\s*", "", text)
    match = re.match(r"^((?:::)?(?:[A-Za-z_]\w*\s*::\s*)*)([A-Za-z_]\w*)\s*", text)
    if match is None:
        return None
    qualifier = compact(match.group(1)).replace(" ", "").strip(":")
    name = match.group(2)
    rest = text[match.end() :]
    targs: tuple[str, ...] = ()
    if rest.startswith("<"):
        close = scan.match_angle(rest, 0)
        if close is None:
            return None
        targs = scan.split_args(rest[1:close])
        rest = rest[close + 1 :].strip()
    if rest:
        return None
    return name, qualifier, targs


def _arity_ok(
    fn: FunctionModel,
    n_targs: int,
    n_args: int,
    *,
    flexible: bool = False,
    tflexible: bool = False,
) -> bool:
    """Argument-count compatibility; ``flexible`` = a pack expansion may supply 0..n more."""
    template_params = fn.template_params
    if fn.definition is not None and fn.definition.specialization_args:
        n_spec = len(fn.definition.specialization_args)
        if n_targs and n_targs != n_spec and not tflexible:
            return False
    else:
        t_variadic = any(p.variadic for p in template_params)
        if n_targs > len(template_params) and not t_variadic:
            return False
    runtime = fn.params
    r_variadic = any(p.variadic for p in runtime)
    required = sum(1 for p in runtime if p.default is None and not p.variadic)
    if n_args < required and not flexible:
        return False
    if n_args > len(runtime) and not r_variadic:
        return False
    return True


def expand_packs(
    args: Sequence[str], bound: dict[str, str], variadic: set[str]
) -> tuple[str, ...]:
    """Expand ``expr(pack)...`` arguments whose pack parameter is bound to a list."""
    from . import scan

    result: list[str] = []
    for arg in args:
        text = compact(arg)
        if not text.endswith("..."):
            result.append(arg)
            continue
        base = text[:-3].rstrip()
        names = sorted(identifiers(base) & variadic)
        # prefer a pack that is bound to something (a deduced template pack binds to "")
        name = next((n for n in names if n in bound and bound[n].strip()), None)
        if name is None:
            name = next((n for n in names if n in bound), None)
        if name is None:
            result.append(arg)
            continue
        for value in scan.split_args(bound[name]):
            result.append(substitute(base, {name: value}))
    return tuple(result)


def _dedupe(effects: Iterable[Effect]) -> list[Effect]:
    seen: set[tuple] = set()
    unique: list[Effect] = []
    for effect in effects:
        identity = effect.identity()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(effect)
    unique.sort(key=lambda effect: effect.sort_key())
    return unique


def dedupe_effects(effects: Iterable[Effect]) -> list[Effect]:
    """Public: drop identical effects and sort deterministically."""
    return _dedupe(effects)
