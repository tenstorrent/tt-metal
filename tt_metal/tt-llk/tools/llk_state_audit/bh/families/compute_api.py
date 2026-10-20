# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Family ``compute_api``: roles of the compute API, argument mapping, the FOLD.

Scope: ``tt_metal/hw/inc/api/compute/**`` (layer ``compute_api``) and, for argument
mapping only, kernel sources added later with ``EffectDB.add_sources`` (layer ``kernel``).

Every effect this family emits uses the ``param`` resource class (``param:<...>``). They are
facts about the compute-API layer, not hardware storage, so they never overlap a CFG, GPR,
SFPU or other hardware resource. Hardware effects of the callees keep coming from the core
and the other families through the call graph.

Effect vocabulary (``pattern`` -> resource, access, value, detail):

``compute_api.bind`` / ``compute_api.fold_bind``
    ``param:<callee>.<callee_param>``. One effect per named parameter of a statically
    resolved callee of a compute-API or kernel function. ``value`` is the argument text in the
    caller's scope, or the callee's declared default. ``detail`` is ``explicit`` when the call
    passes the argument, or ``default`` when the callee default applies. ``access`` is
    ``write`` when the callee role is init, uninit, reconfig, config or hw_startup. It is
    ``read`` when the callee is an execute. Helper callees and ``call_line`` parameters are
    skipped. ``fold_bind`` marks the bindings of a FOLDED edge: an execute compute-API
    function calling an init/uninit/reconfig/config/hw_startup callee. The value is substituted
    transitively, so a kernel calling ``exp_tile<true>(i)`` sees
    ``param:exp_tile_init.scale = 0x3F800000`` (via ``exp_tile``) next to its own explicit
    ``exp_tile_init<true, S>()`` binding ``param:exp_tile_init.scale = S``.
``compute_api.fold``
    ``param:fold.<execute>.<callee>``, write. One marker per folded edge. ``value`` is ``.``
    plus the callee role. ``detail`` lists the callee parameters as
    ``.fwd=.a,.b .pin=.c .dflt=.d``: forwarded from the execute's parameters (or locals
    computed from them), pinned to a constant, or left at the callee default.
``compute_api.define``
    ``param:define.<MACRO>``, read. A kernel #define consumed by a compute-API header, either
    as a preprocessor gate (``value`` ``.gate``) or as an argument value of a call
    (``value`` ``.<callee>.<param>``).
``compute_api.sw_read``
    ``SW.<global>``, read. Software-shadow globals declared in compute-API headers (e.g.
    ``throttled_mop_status``), read in a compute-API body. The core records the writes.
``compute_api.order``
    ``param:order.<resource>``, write. An ORDERING fact for a RISC-side store (``cfg[]`` /
    ``TT_CFG_STORE`` / ``cfg_write`` MMIO, ``regfile[]`` GPR store, ``mop_cfg[]``) reached from
    a compute-API function. The function body is flattened through its statically resolved
    callees (textual order, conditions kept). The store is paired with the next Tensix
    instruction on the same thread that can consume it: data instructions for CFG
    (:data:`CONFIG_ONLY_OPS` excluded), :data:`GPR_CONSUMER_OPS` for GPRs, and ``MOP`` for MOP
    config. Instructions recorded into the replay buffer (``load_replay_buf<NoExec>``
    callables, the next ``len`` after ``lltt::record<NoExec>``), ``TT_OP_*`` words, and calls
    inside asserts and sanitizer hooks are not issued, so they are skipped. Fences are
    STALLWAIT with TRISC_CFG in the wait mask, ``tensix_sync``, ``sync_regfile_write`` (GPR),
    and, for MOP config, a ``mop_sync`` between the previous MOP and the store. A fence
    protects a consumer when its conditions are implied by the store's and the consumer's
    conditions. Scanning stops at the first consumer on the store's own path. ``value`` is
    ``.ordered`` (every consumer up to that one is protected), ``.unordered`` (none is), or
    ``.conditional`` (some are). ``evidence`` is the store. ``detail`` is
    ``.fence=<loc> .use=<loc> .at=<loc>``: the protecting fence, the first unprotected
    consumer (or the consumer for ``.ordered``), and the call in this function that reaches the
    store. Locations are encoded with ``/`` -> ``::`` and ``-`` -> ``$`` so that call-graph
    substitution never rewrites them (decode with :func:`decode_location`). A fact is emitted in
    the lowest compute-API function that holds both the store and the consumer. Blackhole never
    enables the TENSIX_TRISC_SYNC hardware tracking (``set_ttsync_enables`` is Quasar-only), so
    only software fences count.

Tokens inside values and details start with ``.`` or are C++ keywords
(``explicit``/``default``). Transitive substitution therefore leaves them untouched.

Post-build helpers for the checker (pure functions of an ``EffectDB``):
:func:`folded_executes`, :func:`outside_init_state` and :func:`ordering_facts`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .. import scan
from ..model import (
    CallGraph,
    Effect,
    FunctionModel,
    combine_conditions,
    compact,
    identifiers,
    simplify_arg,
)
from ..resources import Normalized, expand_rmw_args

#: Unique family name. Every pattern id and normalizer name starts with "compute_api.".
FAMILY = "compute_api"
ENABLED = True

#: Callee roles that set persistent state. A call from an execute to one of these is a FOLD.
STATE_ROLES = ("init", "uninit", "reconfig", "config", "hw_startup")
#: Layers whose functions get argument-mapping (binding) effects.
BINDING_CALLER_LAYERS = ("compute_api", "kernel")
#: Callee parameters that never carry hardware state (sentinel call-site bookkeeping).
SKIP_PARAMS = frozenset({"call_line"})
#: Kernel-wide defines generated by the JIT (tt_metal/jit_build/genfiles.cpp) or by the kernel
#: build (MM_THROTTLE, REDUCE_OP, REDUCE_DIM) and consumed as values by compute-API headers.
VALUE_DEFINES = frozenset(
    {
        "APPROX",
        "DST_ACCUM_MODE",
        "DST_SYNC_MODE",
        "MATH_FIDELITY",
        "MM_THROTTLE",
        "REDUCE_DIM",
        "REDUCE_OP",
        "UnpackToDestEn",
    }
)
#: Preprocessor symbols that are not kernel configuration (arch, thread, sentinel, tooling).
_GATE_EXCLUDE = re.compile(
    r"^(?:ARCH_\w+|TRISC_\w+|UCK_CHLKC_\w+|COMPILE_FOR_\w+|TT_METAL_COMPUTE_KERNEL_SENTINEL\w*|"
    r"__\w+|defined|DEBUG_\w+|WATCHER_\w+|LLK_\w+|PROFILE_\w+|DEVICE_PRINT\w*|ENABLE_\w+)$"
)

PATTERNS: list = []
DISABLE_CORE: tuple[str, ...] = ()
GROUPS = [
    ("compute_api.binding", r"^param:(?!fold\.|define\.|order\.)[^.]+\.[^.]+$"),
    ("compute_api.fold", r"^param:fold\."),
    ("compute_api.define", r"^param:define\."),
    ("compute_api.order", r"^param:order\."),
]


def _param_normalizer(expr: str | None, thread: str | None = None):
    """``.<a>.<b>...`` (substitution-safe) -> ``param:<a>.<b>...``."""
    text = compact(expr or "")
    if not text.startswith(".") or len(text) < 2:
        return None
    return Normalized("param:" + text[1:], "whole")


NORMALIZERS = {"compute_api.param": _param_normalizer}


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #
#: Compute-API functions whose name-based default role is wrong (evidence: their bodies).
ROLE_OVERRIDES = {
    # configures the packer L1 accumulate bit, not a data-format reconfig
    "pack_reconfig_l1_acc": "config",
    # enable/disable fp32 dest accumulation (same callees as enable_fp32_dest_acc)
    "restore_fp32_dest_acc": "config",
    # compute_kernel_hw_startup + copy_init
    "init_sfpu": "hw_startup",
    # semaphore seeding + compute_kernel_hw_startup + llk_math_reconfig_remap
    "deepseek_compute_kernel_init": "hw_startup",
    # end-of-kernel canonical restore of unpack/math/pack state
    "compute_kernel_hw_cleanup": "uninit",
    # sdpa dummy-valid / srcb-reuse setup and teardown around a tile loop
    "sdpa_bcast_col_reuse_preamble": "init",
    "sdpa_bcast_col_srca_srcb_reuse_preamble": "init",
    "sdpa_bcast_col_reuse_postamble": "uninit",
    # clear persistent SFPU accumulator state between rows
    "welford_clear": "config",
    "ema_clear_previous_output": "config",
    "topk_xl_separate_indices_row_major_advance_chunk_base": "config",
    # SEMINIT of the compute semaphore
    "set": "config",
}
#: Pure compute-API helpers (no Tensix state, no cross-thread effect).
HELPER_NAMES = frozenset(
    {
        "bitwise_instr_mode",
        "deepseek_first_present_cb",
        "dim_stride_of",
        "round_up_l1_words",
        "same_tile_shape",
        "sdpa_fidelity_from_sel",
        "sem_l1_offset",
        "sfpu_reduce_format_supported",
        "should_use_blocked_pack_width",
        "tile_address",
        "tile_stride_words",
    }
)
_HELPER_RE = re.compile(
    r"^(?:is|should)_|^get_(?:arg|common_arg|absolute_logical|relative_logical)_"
)


def compute_role(fn, default: str) -> str | None:
    """Role of a compute-API function (``None`` for every other layer)."""
    if getattr(fn, "layer", None) != "compute_api":
        return None
    name = fn.name
    if getattr(fn, "is_ctor", False) or "/sentinel/" in fn.path:
        return "helper"
    if name in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[name]
    if "reinit" in name.lower():
        return "init"
    if name in HELPER_NAMES or _HELPER_RE.match(name):
        return "helper"
    return default


def role(fn, default: str) -> str | None:
    """Every compute-API function gets an explicit role from this family."""
    return compute_role(fn, default)


# --------------------------------------------------------------------------- #
# Local call-graph view (the plugin interface passes only one FunctionDef)
# --------------------------------------------------------------------------- #
def _model_of(fd: scan.FunctionDef) -> FunctionModel:
    return FunctionModel(
        key=fd.key,
        name=fd.name,
        qualified_name=fd.qualified_name,
        path=fd.path,
        line=fd.line,
        end_line=fd.end_line,
        layer=fd.layer,
        role="",
        thread=fd.thread_hint or "any",
        params=fd.params,
        template_params=fd.template_params,
        signature=fd.signature,
        owner=fd.owner,
        pp_condition=fd.pp_condition,
        definition=fd,
    )


class _Local:
    """Scan index of the tree plus a resolution-only CallGraph (no effects)."""

    def __init__(self, index: scan.ScanIndex):
        self.index = index
        self.models = {fd.key: _model_of(fd) for fd in index.functions}
        self.graph = CallGraph(
            self.models,
            normalize=lambda kind, expr, thread=None: [],
            truth=scan.truth,
            arg_expanders=(expand_rmw_args,),
            constants=set(index.constants),
        )
        self._roles: dict[str, str] = {}
        self._hooks = None
        self._events: dict[str, tuple] = {}
        self._stack: list[str] = []
        self.api_globals = tuple(
            sorted(
                {
                    name
                    for source in index.files
                    if source.layer == "compute_api"
                    for name in source.globals
                }
                - {"rta_count", "crta_count"}
            )
        )

    # ---- roles ------------------------------------------------------------ #
    def role_of(self, model: FunctionModel) -> str:
        cached = self._roles.get(model.key)
        if cached is not None:
            return cached
        from .. import effects_db

        fd = model.definition
        value = effects_db.default_role(fd)
        if self._hooks is None:
            try:
                self._hooks = [
                    family.role
                    for family in effects_db.load_families()
                    if family.role is not None
                ]
            except Exception:  # a sibling family failing to import must not hide roles
                self._hooks = [role]
        for hook in self._hooks:
            override = hook(fd, value)
            if override is not None:
                value = override
                break
        self._roles[model.key] = value
        return value

    # ---- resolution ------------------------------------------------------- #
    def resolve(self, call: scan.CallSite, caller: FunctionModel):
        if call.is_macro or call.is_ctor:
            return [], "unresolved"
        if (
            call.name in caller.param_names
            and not call.is_member
            and "::" not in call.callee
        ):
            return [], "param_callable"
        if call.is_member and not call.receiver_type:
            return [], "unresolved"
        candidates, resolution = self.graph.candidates(
            call.name,
            qualifier=call.qualifier,
            template_args=call.template_args,
            args=call.args,
            is_member=call.is_member,
            receiver_type=call.receiver_type,
            is_ctor=call.is_ctor,
            caller=caller,
        )
        candidates = [c for c in candidates if c.key != caller.key]
        return candidates, resolution


_CACHE: list[tuple[object, _Local]] = []


def _local_for(ctx) -> _Local:
    """The local view for the tree ``ctx`` belongs to (cached per ResourceTables object)."""
    tables = ctx.tables
    for owner, local in _CACHE:
        if owner is tables:
            return local
    index = getattr(ctx, "index", None)
    if index is None:
        index = scan.scan_tree(Path(tables.repo_root))
    local = _Local(index)
    _CACHE.insert(0, (tables, local))
    del _CACHE[4:]
    return local


# --------------------------------------------------------------------------- #
# Argument mapping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Binding:
    """One callee parameter bound at a call site."""

    param: str
    kind: str  # template | runtime
    position: int
    value: str
    source: str  # explicit | default


def bindings(
    graph: CallGraph, callee: FunctionModel, template_args, args
) -> list[Binding]:
    """Callee parameter -> argument text (or declared default), mirroring ``CallGraph.bind``."""
    bound = graph.bind(callee, template_args, args)
    result: list[Binding] = []
    for params, values in (
        (callee.template_params, graph._expand(template_args)),
        (callee.params, graph._expand(args)),
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
                given = values[position:]
                position = len(values)
                if given:
                    result.append(
                        Binding(
                            param.name,
                            param.kind,
                            param.position,
                            ", ".join(given),
                            "explicit",
                        )
                    )
                continue
            value = bound.get(param.name, "")
            if position < len(values):
                source = "explicit"
            elif param.default is not None:
                source = "default"
            else:
                source = ""
            position += 1
            if not source or not value or value.startswith("?"):
                continue
            result.append(
                Binding(param.name, param.kind, param.position, value, source)
            )
    return result


_LOCAL_ASSIGN = re.compile(
    r"(?<![\w.>:])([A-Za-z_]\w*)(?:\s*\[[^\]]*\])?\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)"
)


def _param_flow(fd: scan.FunctionDef) -> dict[str, set[str]]:
    """local -> parameters it is computed from (two passes, textual)."""
    params = set(fd.param_names)
    flow: dict[str, set[str]] = {}
    text = fd.masked_body
    for _ in range(2):
        for match in _LOCAL_ASSIGN.finditer(text):
            name = match.group(1)
            if name in params or name in scan._KEYWORDS:
                continue
            end = scan.statement_end(text, match.end())
            sources: set[str] = set()
            for ident in identifiers(text[match.end() : end]):
                if ident in params:
                    sources.add(ident)
                elif ident in flow:
                    sources |= flow[ident]
            if sources:
                flow.setdefault(name, set()).update(sources)
    return flow


def _depends(value: str, params: set[str], flow: dict[str, set[str]]) -> bool:
    names = identifiers(value)
    return bool(names & params) or any(flow.get(n) for n in names)


_GLOBAL_QUALIFIER = re.compile(r"(?<![\w:])::(?=[A-Za-z_])")


def canonical_value(text: str) -> str:
    """Spelling-insensitive value text: no leading global ``::``, no padding inside ``< >``.

    Macro expansion writes ``::SfpuType::exponential`` and ``exp_init< a, b >`` where a direct
    call writes ``SfpuType::exponential`` and ``exp_init<a, b>``. The same value should compare
    equal. Only a ``<`` glued to an identifier opens a template, and only its matching ``>`` is
    tightened, so comparisons such as ``a < b`` and ``x > y`` keep their spacing.
    """
    value = _GLOBAL_QUALIFIER.sub("", compact(text))
    out: list[str] = []
    opened: list[int] = []  # paren depth at each open template '<'
    parens = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char in "([":
            parens += 1
        elif char in ")]":
            parens -= 1
        elif (
            char == "<"
            and out
            and (out[-1].isalnum() or out[-1] in "_>")
            and value[index + 1 : index + 2] not in ("<", "=")
        ):
            opened.append(parens)
            out.append(char)
            index += 1
            while index < len(value) and value[index] == " ":
                index += 1
            continue
        elif (
            char == ">"
            and opened
            and opened[-1] == parens
            and value[index + 1 : index + 2] != "="
        ):
            opened.pop()
            while out and out[-1] == " ":
                out.pop()
        out.append(char)
        index += 1
    return "".join(out)


def _call_thread(fd: scan.FunctionDef, call: scan.CallSite) -> str | None:
    return call.thread or fd.thread_hint


def _binding_effects(fd: scan.FunctionDef, ctx, local: _Local) -> list[Effect]:
    caller = local.models.get(fd.key) or _model_of(fd)
    caller_role = ""
    if fd.layer == "compute_api":
        from .. import effects_db

        caller_role = compute_role(fd, effects_db.default_role(fd)) or ""
    params = set(fd.param_names)
    flow = _param_flow(fd) if fd.layer == "compute_api" else {}
    out: list[Effect] = []
    dead = unevaluated_spans(fd.masked_body)
    for call in fd.calls:
        if _inside(dead, call.offset):
            continue
        candidates, resolution = local.resolve(call, caller)
        if not candidates or len(candidates) > 3:
            continue
        confidence = "high" if len(candidates) == 1 else "low"
        for callee in candidates:
            callee_role = local.role_of(callee)
            if callee_role == "helper":
                continue
            folded = (
                fd.layer == "compute_api"
                and caller_role == "execute"
                and callee_role in STATE_ROLES
            )
            access = "write" if callee_role in STATE_ROLES else "read"
            pattern = "compute_api.fold_bind" if folded else "compute_api.bind"
            thread = _call_thread(fd, call)
            condition = call.condition or ""
            items = [
                replace(b, value=canonical_value(b.value))
                for b in bindings(local.graph, callee, call.template_args, call.args)
                if b.param not in SKIP_PARAMS
            ]
            for item in items:
                out.extend(
                    ctx.effects(
                        fd,
                        pattern_id=pattern,
                        resource_expr=f".{callee.name}.{item.param}",
                        normalizer="compute_api.param",
                        access=access,
                        value=item.value,
                        line=call.line,
                        condition=condition,
                        thread=thread,
                        detail=item.source,
                        granularity="whole",
                        confidence=confidence,
                    )
                )
                if fd.layer == "compute_api":
                    for define in sorted(identifiers(item.value) & VALUE_DEFINES):
                        out.extend(
                            ctx.effects(
                                fd,
                                pattern_id="compute_api.define",
                                resource_expr=f".define.{define}",
                                normalizer="compute_api.param",
                                access="read",
                                value=f".{callee.name}.{item.param}",
                                line=call.line,
                                condition=condition,
                                thread=thread,
                                detail=item.source,
                                granularity="whole",
                                confidence=confidence,
                            )
                        )
            if folded:
                groups: dict[str, list[str]] = {"fwd": [], "pin": [], "dflt": []}
                for item in items:
                    if item.source == "default":
                        groups["dflt"].append(item.param)
                    elif _depends(item.value, params, flow):
                        groups["fwd"].append(item.param)
                    else:
                        groups["pin"].append(item.param)
                detail = " ".join(
                    f".{key}=" + ",".join(f".{name}" for name in groups[key])
                    for key in ("fwd", "pin", "dflt")
                )
                out.extend(
                    ctx.effects(
                        fd,
                        pattern_id="compute_api.fold",
                        resource_expr=f".fold.{fd.name}.{callee.name}",
                        normalizer="compute_api.param",
                        access="write",
                        value=f".{callee_role}",
                        line=call.line,
                        condition=condition,
                        thread=thread,
                        detail=detail,
                        granularity="whole",
                        confidence=confidence,
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# Kernel #defines consumed as preprocessor gates
# --------------------------------------------------------------------------- #
_PP_NAME = re.compile(r"\b([A-Z_][A-Za-z0-9_]*)\b")


def _gate_defines(text: str | None) -> list[str]:
    if not text:
        return []
    names = []
    for name in _PP_NAME.findall(text):
        if _GATE_EXCLUDE.match(name) or not re.search(r"[A-Z]", name):
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
            continue
        if name not in names:
            names.append(name)
    return names


def _define_gate_effects(fd: scan.FunctionDef, ctx) -> list[Effect]:
    seen: set[tuple[str, str]] = set()
    out: list[Effect] = []

    def emit(define: str, pp: str, line: int, thread):
        key = (define, pp)
        if key in seen:
            return
        seen.add(key)
        out.extend(
            ctx.effects(
                fd,
                pattern_id="compute_api.define",
                resource_expr=f".define.{define}",
                normalizer="compute_api.param",
                access="read",
                value=".gate",
                line=line,
                condition=pp,
                thread=thread,
                granularity="whole",
            )
        )

    for define in _gate_defines(fd.pp_condition):
        emit(define, fd.pp_condition or "", fd.line, fd.thread_hint)
    for call in fd.calls:
        _, pp = fd.pp_info_at_line(call.line)
        if not pp or pp == fd.pp_condition:
            continue
        for define in _gate_defines(pp):
            emit(define, pp, call.line, _call_thread(fd, call))
    return out


# --------------------------------------------------------------------------- #
# Software-shadow globals of the compute API (reads)
# --------------------------------------------------------------------------- #
def _sw_read_effects(fd: scan.FunctionDef, ctx, local: _Local) -> list[Effect]:
    out: list[Effect] = []
    if not local.api_globals:
        return out
    names = "|".join(re.escape(n) for n in local.api_globals)
    regex = re.compile(rf"(?<![\w.>:])({names})\b(?!\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=))")
    spans = scan.thread_spans(fd.masked_body)
    seen: set[tuple[str, int]] = set()
    for match in regex.finditer(fd.masked_body):
        line = fd.line_at(match.start())
        key = (match.group(1), line)
        if key in seen:
            continue
        seen.add(key)
        thread = scan._thread_at(spans, match.start()) or fd.pp_info_at_line(line)[0]
        out.extend(
            ctx.effects(
                fd,
                pattern_id="compute_api.sw_read",
                resource_expr=match.group(1),
                normalizer="sw",
                access="read",
                value="",
                offset=match.start(),
                thread=thread,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Ordering of RISC-side stores
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Event:
    """One step of a flattened instruction/store sequence."""

    kind: str  # store | fence | use
    sub: str  # store: cfg|gpr|mop ; fence: cfg|gpr|all|mop ; use: instruction name
    target: str  # canonical resource of a store ("" when unknown)
    evidence: str  # sink path:line
    condition: str
    thread: str | None
    top: int  # index of the call site in the analysed function (-1: its own body)


_INSN = re.compile(r"TTI?_([A-Z][A-Z0-9_]*)")
_NO_USE = frozenset({"STALLWAIT", "NOP", "DMANOP"})
#: Tensix instructions that only write configuration/counters or synchronize: they do not
#: consume THCON/ALU/PACK configuration written from the RISC.
CONFIG_ONLY_OPS = frozenset(
    {
        "ADDDMAREG",
        "ADDRCRXY",
        "ADDRCRZW",
        "ATGETM",
        "ATRELM",
        "BITWOPDMAREG",
        "CFGSHIFTMASK",
        "CMPDMAREG",
        "DMANOP",
        "INCADCXY",
        "INCADCZW",
        "INCRWC",
        "MOP_CFG",
        "MULDMAREG",
        "NOP",
        "RMWCIB0",
        "RMWCIB1",
        "RMWCIB2",
        "RMWCIB3",
        "SEMGET",
        "SEMINIT",
        "SEMPOST",
        "SEMWAIT",
        "SETADC",
        "SETADCXX",
        "SETADCXY",
        "SETADCZW",
        "SETC16",
        "SETDMAREG",
        "SETRWC",
        "SHIFTDMAREG",
        "STALLWAIT",
        "SUBDMAREG",
        "WRCFG",
    }
)
#: Tensix instructions that read GPRs written from the RISC (DMA-register ALU, config moves,
#: indirect loads/stores, atomics) or move data with GPR-held addresses/headers.
GPR_CONSUMER_OPS = frozenset(
    {
        "ADDDMAREG",
        "ATCAS",
        "ATINCGET",
        "ATINCGETPTR",
        "ATSWAP",
        "BITWOPDMAREG",
        "CMPDMAREG",
        "FLUSHDMA",
        "LOADIND",
        "MOP",
        "MULDMAREG",
        "PACR",
        "REG2FLOP",
        "REPLAY",
        "SHIFTDMAREG",
        "STOREIND",
        "STOREREG",
        "SUBDMAREG",
        "UNPACR",
        "UNPACR_NOP",
        "WRCFG",
    }
)
#: Macros whose arguments are not evaluated in production builds (asserts, sanitizer hooks).
UNEVALUATED_MACROS = (
    "LLK_ASSERT_BLOCK",
    "LLK_ASSERT",
    "LLK_PANIC",
    "ASSERT",
    "SAN_HOOK",
    "static_assert",
    "sizeof",
    "decltype",
)
_UNEVALUATED = re.compile(r"(?<![\w.>])(?:" + "|".join(UNEVALUATED_MACROS) + r")\s*\(")


def unevaluated_spans(masked: str) -> list[tuple[int, int]]:
    """(start, end) offsets of the argument lists of :data:`UNEVALUATED_MACROS`."""
    spans = []
    for match in _UNEVALUATED.finditer(masked):
        close = scan.match_close(masked, match.end() - 1, "()")
        if close is not None:
            spans.append((match.end() - 1, close))
    return spans


def _inside(spans: list[tuple[int, int]], offset: int) -> bool:
    return any(start < offset < end for start, end in spans)


def _enclosing_block_end(masked: str, offset: int) -> int:
    depth = 0
    for position in range(offset - 1, -1, -1):
        char = masked[position]
        if char == "}":
            depth += 1
        elif char == "{":
            if depth == 0:
                close = scan.match_close(masked, position, "{}")
                return close if close is not None else len(masked)
            depth -= 1
    return len(masked)


def _executes_while_loading(call: scan.CallSite) -> bool:
    tokens = {compact(t).split("::")[-1] for t in call.template_args}
    return bool(tokens & {"Exec", "true"})


def replay_deferred(fd: scan.FunctionDef) -> set[int]:
    """Indices of ``fd.calls`` recorded into the replay buffer instead of being issued.

    Everything inside the argument list of ``load_replay_buf<NoExec>(..)`` (its callable), and
    the next ``len`` instruction macros after ``lltt::record<NoExec>(start, len)`` (the rest of
    the enclosing block when ``len`` is not a constant).
    """
    masked = fd.masked_body
    spans: list[tuple[int, int]] = []
    deferred: set[int] = set()
    calls = fd.calls
    for index, call in enumerate(calls):
        if call.name == "load_replay_buf" and not _executes_while_loading(call):
            opening = masked.find("(", call.offset)
            close = scan.match_close(masked, opening, "()") if opening >= 0 else None
            if close is not None:
                spans.append((opening, close))
        elif (
            call.name == "record"
            and call.qualifier.lstrip(":") == "lltt"
            and not _executes_while_loading(call)
            and not _inside(spans, call.offset)
        ):
            length = scan.evaluate(call.args[1]) if len(call.args) >= 2 else None
            remaining = (
                length
                if isinstance(length, int) and not isinstance(length, bool)
                else None
            )
            end = _enclosing_block_end(masked, call.offset)
            for later in range(index + 1, len(calls)):
                other = calls[later]
                if other.offset > end or (remaining is not None and remaining <= 0):
                    break
                if _INSN.fullmatch(other.name) and not other.name.startswith("TT_OP_"):
                    deferred.add(later)
                    if remaining is not None:
                        remaining -= 1
    for index, call in enumerate(calls):
        if _inside(spans, call.offset):
            deferred.add(index)
    return deferred


#: Unpacker config an ``UNPACR_NOP`` never reads: it moves no data from L1 (it only sets data
#: valid or fills a source register), so L1 base addresses, strides and tile geometry are unused.
_UNPACR_NOP_UNUSED = re.compile(
    r"THCON_SEC[01]_REG3_|UNP[01]_ADDR_CTRL_|TileDescriptor|Tile_x_dim|TILE_DESC"
)


def consumes(store_sub: str, op: str, target: str = "") -> bool:
    """Does Tensix instruction ``op`` consume state of a RISC store of kind ``store_sub``
    (to resource ``target``, when known)?"""
    if store_sub == "mop":
        return op == "MOP"
    if store_sub == "gpr":
        return op in GPR_CONSUMER_OPS
    if op == "UNPACR_NOP" and target and _UNPACR_NOP_UNUSED.search(target):
        return False
    return op not in CONFIG_ONLY_OPS


_STORE_ASSIGN = re.compile(
    r"(?<![\w.>])(?:\(\s*)?(?P<name>cfg|cfg_regs|cfg_ptr|regfile|mop_cfg)(?:\s*\))?\s*\[(?P<index>[^\]=;]*)\]\s*=(?!=)"
)
_MAX_EVENTS = 4000
_MAX_DEPTH = 14


def _stallwait_orders_cfg(args: tuple[str, ...]) -> bool:
    if len(args) < 2:
        return False
    wait = args[1]
    if "TRISC_CFG" in wait:
        return True
    value = scan.evaluate(wait)
    return (
        isinstance(value, int) and not isinstance(value, bool) and bool(value & 0x400)
    )


def _store_target(ctx, kind: str, index: str, thread: str | None) -> str:
    try:
        if kind == "cfg":
            found = ctx.tables.normalize("cfg_addr32", index, thread)
        elif kind == "gpr":
            found = ctx.tables.normalize("gpr", index, thread)
        else:
            return "MOP"
    except KeyError:
        return ""
    return found[0].name if found else ""


def _own_items(fd: scan.FunctionDef, ctx) -> list[tuple[int, int, object]]:
    """(offset, order, item) of one body; item is an Event (top=-1) or a CallSite."""
    items: list[tuple[int, int, object]] = []
    spans = scan.thread_spans(fd.masked_body)
    dead = unevaluated_spans(fd.masked_body)
    recorded = replay_deferred(fd)
    for order, call in enumerate(fd.calls):
        if _inside(dead, call.offset) or order in recorded:
            continue
        thread = call.thread or fd.thread_hint
        evidence = f"{fd.path}:{call.line}"
        condition = call.condition or ""
        name = call.name
        if name == "TT_CFG_STORE" and len(call.args) >= 2:
            items.append(
                (
                    call.offset,
                    order,
                    Event(
                        "store",
                        "cfg",
                        _store_target(ctx, "cfg", call.args[1], thread),
                        evidence,
                        condition,
                        thread,
                        -1,
                    ),
                )
            )
            continue
        if (
            name == "__builtin_rvtt_cfg_store"
            and call.via_macro != "TT_CFG_STORE"
            and len(call.args) >= 2
        ):
            items.append(
                (
                    call.offset,
                    order,
                    Event(
                        "store",
                        "cfg",
                        _store_target(ctx, "cfg", call.args[1], thread),
                        evidence,
                        condition,
                        thread,
                        -1,
                    ),
                )
            )
            continue
        if name == "cfg_write" and call.args:
            items.append(
                (
                    call.offset,
                    order,
                    Event(
                        "store",
                        "cfg",
                        _store_target(ctx, "cfg", call.args[0], thread),
                        evidence,
                        condition,
                        thread,
                        -1,
                    ),
                )
            )
            continue
        if name in ("tensix_sync", "__builtin_rvtt_store_blocking"):
            items.append(
                (
                    call.offset,
                    order,
                    Event("fence", "all", "", evidence, condition, thread, -1),
                )
            )
            continue
        if name == "mop_sync":
            items.append(
                (
                    call.offset,
                    order,
                    Event("fence", "mop", "", evidence, condition, thread, -1),
                )
            )
            continue
        if name == "sync_regfile_write":
            items.append(
                (
                    call.offset,
                    order,
                    Event("fence", "gpr", "", evidence, condition, thread, -1),
                )
            )
            continue
        insn = _INSN.fullmatch(name)
        if insn is not None and not name.startswith("TT_OP_"):
            op = insn.group(1)
            if op == "STALLWAIT":
                if _stallwait_orders_cfg(call.args):
                    items.append(
                        (
                            call.offset,
                            order,
                            Event("fence", "cfg", "", evidence, condition, thread, -1),
                        )
                    )
                continue
            if op in _NO_USE:
                continue
            items.append(
                (
                    call.offset,
                    order,
                    Event("use", op, "", evidence, condition, thread, -1),
                )
            )
            continue
        if call.is_macro:
            continue
        items.append((call.offset, order, call))
    base = len(fd.calls)
    for text, expansion in [(fd.masked_body, None)] + [
        (e.text, e) for e in fd.expansions
    ]:
        for match in _STORE_ASSIGN.finditer(text):
            name = match.group("name")
            kind = (
                "gpr" if name == "regfile" else ("mop" if name == "mop_cfg" else "cfg")
            )
            if expansion is None:
                offset = match.start()
                if _inside(dead, offset):
                    continue
                line = fd.line_at(offset)
                condition = fd.condition_at(offset) or ""
                thread = (
                    scan._thread_at(spans, offset)
                    or fd.pp_info_at_line(line)[0]
                    or fd.thread_hint
                )
            else:
                offset = expansion.offset
                line = expansion.line
                condition = expansion.condition or ""
                thread = expansion.thread or fd.thread_hint
            target = _store_target(ctx, kind, match.group("index"), thread)
            items.append(
                (
                    offset,
                    base,
                    Event(
                        "store",
                        kind,
                        target,
                        f"{fd.path}:{line}",
                        condition,
                        thread,
                        -1,
                    ),
                )
            )
            base += 1
    items.sort(key=lambda item: (item[0], item[1]))
    return items


def _flatten(
    local: _Local, fd: scan.FunctionDef, ctx, depth: int = 0
) -> tuple[Event, ...]:
    cached = local._events.get(fd.key)
    if cached is not None:
        return cached
    if fd.key in local._stack or depth > _MAX_DEPTH:
        return ()
    local._stack.append(fd.key)
    try:
        caller = local.models.get(fd.key) or _model_of(fd)
        events: list[Event] = []
        for _, order, item in _own_items(fd, ctx):
            if isinstance(item, Event):
                events.append(item)
                continue
            candidates, resolution = local.resolve(item, caller)
            if not candidates or len(candidates) > 2:
                continue
            thread = item.thread or fd.thread_hint
            for callee in candidates:
                for event in _flatten(local, callee.definition, ctx, depth + 1):
                    events.append(
                        replace(
                            event,
                            condition=combine_conditions(
                                item.condition, event.condition
                            ),
                            thread=event.thread or thread,
                            top=order,
                        )
                    )
            if len(events) > _MAX_EVENTS:
                break
        result = tuple(events[:_MAX_EVENTS])
    finally:
        local._stack.pop()
    local._events[fd.key] = result
    return result


def _same_thread(a: str | None, b: str | None) -> bool:
    return a in (None, "any") or b in (None, "any") or a == b


def _fence_matches(store: Event, fence: Event) -> bool:
    if fence.sub == "all":
        return True
    if store.sub == "cfg":
        return fence.sub == "cfg"
    if store.sub == "gpr":
        return fence.sub == "gpr"
    return False


def encode_location(evidence: str) -> str:
    """``path:line`` -> substitution-safe token (``/`` -> ``::``, ``-`` -> ``$``)."""
    return evidence.replace("/", "::").replace("-", "$")


def decode_location(token: str) -> str:
    """Inverse of :func:`encode_location`."""
    return token.replace("::", "/").replace("$", "-")


@dataclass(frozen=True)
class OrderFact:
    target: str
    verdict: str  # ordered | conditional | unordered
    store: Event
    fence: Event | None
    use: Event
    at: str  # evidence of the call in the analysed function that reaches the store


def order_facts(
    fd: scan.FunctionDef,
    events: Iterable[Event],
    owned_tops: frozenset[int] = frozenset(),
) -> list[OrderFact]:
    """Pair every RISC-side store with the next consuming Tensix instruction on its thread.

    ``owned_tops``: call-site indices of ``fd`` whose callee reports its own facts (another
    compute-API function); pairs entirely inside such a call are skipped here.
    """
    events = list(events)
    facts: list[OrderFact] = []
    seen: set[tuple] = set()
    calls = fd.calls
    for index, store in enumerate(events):
        if store.kind != "store":
            continue
        own = _parts(store.condition)
        # (consumer, protecting fence or None); stop at the first consumer on the store's path
        consumers: list[tuple[Event, Event | None]] = []
        if store.sub == "mop":
            fence = None
            for back in range(index - 1, -1, -1):
                prior = events[back]
                if not _same_thread(prior.thread, store.thread):
                    continue
                if prior.kind == "use" and consumes("mop", prior.sub):
                    break
                if prior.kind == "fence" and prior.sub in ("mop", "all"):
                    if _parts(prior.condition) <= own:
                        fence = prior
                        break
            for after in events[index + 1 :]:
                if (
                    after.kind == "use"
                    and _same_thread(after.thread, store.thread)
                    and consumes("mop", after.sub)
                ):
                    consumers.append((after, fence))
                    break
        else:
            fences: list[Event] = []
            for after in events[index + 1 :]:
                if not _same_thread(after.thread, store.thread):
                    continue
                if after.kind == "fence" and _fence_matches(store, after):
                    fences.append(after)
                    continue
                if after.kind != "use" or not consumes(
                    store.sub, after.sub, store.target or ""
                ):
                    continue
                path = own | _parts(after.condition)
                guard = next((f for f in fences if _parts(f.condition) <= path), None)
                consumers.append((after, guard))
                if _parts(after.condition) <= own:
                    break  # this consumer runs on every path that runs the store
        if not consumers:
            continue
        unprotected = [c for c, guard in consumers if guard is None]
        if not unprotected:
            verdict, use, fence = "ordered", consumers[-1][0], consumers[-1][1]
        elif (
            len(unprotected) == len(consumers)
            and _parts(consumers[-1][0].condition) <= own
        ):
            verdict, use, fence = "unordered", unprotected[0], None
        else:
            protected = [guard for _, guard in consumers if guard is not None]
            verdict, use, fence = (
                "conditional",
                unprotected[0],
                (protected[0] if protected else None),
            )
        if store.top >= 0 and store.top == use.top and store.top in owned_tops:
            continue  # store and use inside one compute-API callee: that callee reports it
        at = (
            f"{fd.path}:{calls[store.top].line}"
            if 0 <= store.top < len(calls)
            else store.evidence
        )
        fact = OrderFact(store.target or "?", verdict, store, fence, use, at)
        key = (
            fact.target,
            verdict,
            store.top,
            use.evidence,
            fence.evidence if fence else "",
        )
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
    return facts


def _parts(condition: str) -> frozenset[str]:
    return frozenset(
        part.strip() for part in (condition or "").split(" && ") if part.strip()
    )


def _order_effects(fd: scan.FunctionDef, ctx, local: _Local) -> list[Effect]:
    events = _flatten(local, fd, ctx)
    if not any(e.kind == "store" for e in events):
        return []
    caller = local.models.get(fd.key) or _model_of(fd)
    owned: set[int] = set()
    for order, call in enumerate(fd.calls):
        if call.is_macro:
            continue
        candidates, _ = local.resolve(call, caller)
        if candidates and all(c.layer == "compute_api" for c in candidates):
            owned.add(order)
    out: list[Effect] = []
    for fact in order_facts(fd, events, frozenset(owned)):
        detail = " ".join(
            (
                ".fence="
                + (encode_location(fact.fence.evidence) if fact.fence else ".none"),
                ".use=" + encode_location(fact.use.evidence),
                ".at=" + encode_location(fact.at),
            )
        )
        for effect in ctx.effects(
            fd,
            pattern_id="compute_api.order",
            resource_expr=f".order.{fact.target}",
            normalizer="compute_api.param",
            access="write",
            value=f".{fact.verdict}",
            line=fd.line,
            condition=fact.store.condition,
            thread=fact.store.thread,
            detail=detail,
            granularity="whole",
            confidence="medium" if fact.store.top >= 0 else "high",
        ):
            out.append(replace(effect, evidence=fact.store.evidence))
    return out


# --------------------------------------------------------------------------- #
# Plugin hook
# --------------------------------------------------------------------------- #
def extract(fn, ctx):
    """Bindings + fold markers (compute API and kernels); defines, SW reads, ordering (compute API)."""
    if fn.layer not in BINDING_CALLER_LAYERS:
        return []
    local = _local_for(ctx)
    found = _binding_effects(fn, ctx, local)
    if fn.layer == "compute_api":
        found.extend(_define_gate_effects(fn, ctx))
        found.extend(_sw_read_effects(fn, ctx, local))
        found.extend(_order_effects(fn, ctx, local))
    return found


# --------------------------------------------------------------------------- #
# Post-build helpers (checker API)
# --------------------------------------------------------------------------- #
def folded_executes(db) -> list[dict]:
    """Every compute-API execute that runs an init/uninit/reconfig/config internally.

    One row per (execute, folded callee call site) with the callee parameter split into
    forwarded / pinned / defaulted, plus the binding values. Sorted, JSON-ready.
    """
    rows = []
    for key in db.keys():
        fn = db.function(key)
        if fn.layer != "compute_api" or fn.role != "execute":
            continue
        markers = [e for e in fn.effects if e.pattern == "compute_api.fold"]
        if not markers:
            continue
        binds = [e for e in fn.effects if e.pattern == "compute_api.fold_bind"]
        for marker in markers:
            callee = marker.resource.split(".", 2)[2]
            groups = {}
            for part in marker.detail.split():
                name, _, values = part.partition("=")
                groups[name.lstrip(".")] = [
                    v.lstrip(".") for v in values.split(",") if v
                ]
            values = {
                e.resource.split(".", 1)[1]: {"value": e.value, "source": e.detail}
                for e in binds
                if e.evidence == marker.evidence
                and e.resource.startswith(f"param:{callee}.")
                and e.condition == marker.condition
            }
            rows.append(
                {
                    "callee": callee,
                    "callee_role": marker.value.lstrip("."),
                    "condition": marker.condition,
                    "defaulted": groups.get("dflt", []),
                    "evidence": marker.evidence,
                    "execute": fn.name,
                    "execute_key": key,
                    "forwarded": groups.get("fwd", []),
                    "path": fn.path,
                    "pinned": groups.get("pin", []),
                    "thread": marker.thread,
                    "values": dict(sorted(values.items())),
                }
            )
    rows.sort(key=lambda r: (r["path"], r["execute_key"], r["evidence"], r["callee"]))
    return rows


def _hardware(effect: Effect) -> bool:
    return effect.resource_class != "param" and effect.resolved


def _through_setup(db, effect: Effect) -> bool:
    """The write is reached through a hw_startup/config callee (an embedded hw_configure)."""
    for callee_key, _ in effect.via:
        callee = db.function(callee_key)
        if callee is not None and callee.role in ("hw_startup", "config"):
            return True
    return False


def outside_init_state(db) -> list[dict]:
    """Persistent state set only outside init, and the folded executes that now set it.

    A hardware resource qualifies when compute-API config/hw_startup functions write it and no
    compute-API init writes it except through an embedded hw_startup/config callee (for example
    ``llk_pack_hw_configure`` inside ``custom_mm_block_init``). Each row lists:

    * ``writers``: the config/hw_startup writers;
    * ``init_writers_via_setup``: the inits that write it only through such a callee;
    * ``folded_executes``: the executes that write it through a folded call, so the fold turned
      it into execute-time state.

    Kernel #defines consumed by compute-API headers follow as ``param:define.*`` rows.
    """
    init_proper: set[str] = set()
    init_setup: dict[str, set[str]] = {}
    cfg_writers: dict[str, set[str]] = {}
    fold_writers: dict[str, set[str]] = {}
    defines: dict[str, set[str]] = {}
    for key in db.keys():
        fn = db.function(key)
        if fn.layer != "compute_api":
            continue
        for effect in fn.effects:
            if effect.pattern == "compute_api.define":
                defines.setdefault(effect.resource, set()).add(fn.name)
        if fn.role not in ("init", "config", "hw_startup", "execute"):
            continue
        writes = [e for e in db.effects(key) if e.access == "write" and _hardware(e)]
        if fn.role == "init":
            for e in writes:
                if _through_setup(db, e):
                    init_setup.setdefault(e.resource, set()).add(fn.name)
                else:
                    init_proper.add(e.resource)
        elif fn.role in ("config", "hw_startup"):
            for e in writes:
                cfg_writers.setdefault(e.resource, set()).add(fn.name)
        else:
            folded = {
                e.resource.split(".", 2)[2]
                for e in fn.effects
                if e.pattern == "compute_api.fold"
            }
            for e in writes:
                first = db.function(e.via[0][0]) if e.via else None
                if first is not None and first.name in folded:
                    fold_writers.setdefault(e.resource, set()).add(fn.name)
    rows = []
    for resource in sorted(set(cfg_writers) - init_proper):
        rows.append(
            {
                "folded_executes": sorted(fold_writers.get(resource, ())),
                "init_writers_via_setup": sorted(init_setup.get(resource, ())),
                "resource": resource,
                "writers": sorted(cfg_writers[resource]),
            }
        )
    for resource in sorted(defines):
        rows.append(
            {
                "folded_executes": [],
                "init_writers_via_setup": [],
                "resource": resource,
                "writers": sorted(defines[resource]),
            }
        )
    return rows


def ordering_facts(db) -> list[dict]:
    """All ``compute_api.order`` facts of the compute API, decoded and sorted."""
    rows = []
    for key in db.keys():
        fn = db.function(key)
        if fn.layer != "compute_api":
            continue
        for e in fn.effects:
            if e.pattern != "compute_api.order":
                continue
            parts = dict(
                p.lstrip(".").split("=", 1) for p in e.detail.split() if "=" in p
            )
            rows.append(
                {
                    "at": decode_location(parts.get("at", "")),
                    "condition": e.condition,
                    "fence": (
                        ""
                        if parts.get("fence") == ".none"
                        else decode_location(parts.get("fence", ""))
                    ),
                    "function": fn.name,
                    "key": key,
                    "resource": e.resource.split(".", 1)[1],
                    "store": e.evidence,
                    "thread": e.thread,
                    "use": decode_location(parts.get("use", "")),
                    "verdict": e.value.lstrip("."),
                }
            )
    rows.sort(
        key=lambda r: (r["key"], r["store"], r["use"], r["resource"], r["verdict"])
    )
    return rows
