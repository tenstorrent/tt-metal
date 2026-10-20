# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole state-hazard rules over the effect DB and the compute-kernel scan.

``run(db, repo_root)`` returns ``{"findings": [...], "tree_fingerprint": ...}`` (SCHEMA.md
section 7; ``effects_db.normalize_findings`` validates and numbers them). Every finding is
backed by concrete writes, reads or calls of the model; nothing fires on names alone.

Kernel call effects are the callee's own-scope transitive effects (``db.effects(key)``) with the
kernel call's bound arguments substituted and ``if constexpr`` conditions folded. A write is
*fold-path* when it reaches the hardware through an init / reconfig / config / uninit that runs
inside an execute (the FOLD); explicit kernel init/config calls are the pre-fold contract.

``resource_conflict`` (one finding per resource group and writer/thread aggregate, over kernels)
    * dependent reader: an execute reads a persistent resource that it never writes, and the last
      write it sees (program order, loop back edges wrapped) is another op's fold-path write whose
      value differs from what the kernel's own setup (explicit init/config/hw_startup) left there.
      Typical: a folded reduce init/uninit rewriting the packer edge mask under ``pack_tile``.
    * cross-thread: fold-path writes of one *shared* resource from two TRISC threads with different
      values (in one loop, or one is read after the other): after the fold both happen per tile and
      interleave unsynchronized (e.g. PACK-thread ``exp_packthread_tile`` vs MATH-thread SFPU ops).
    * SFPU macro/constant state and replay ranges re-programmed by two ops in one loop body of an
      SDPA/attention kernel (same thread: thrash and blocked hoisting; replay: an SFPU recording
      overlapping an FPU/packer recording of the split buffer).
    Per-operand format/geometry state (tile descriptors, formats, zero flags) derived from the CB
    being processed is low confidence and kept only for attention-named ops.

``fold_drop`` (one finding per folded execute and parameter)
    A folded execute calls its init/config directly with a default or constant for parameter P
    while the most recent setter of P before it (explicit init/config call, define-gated call or
    hw_startup; loops wrapped) passed a different value, and P reaches hardware state (a write
    effect, or an instruction operand) below the folded call. Named ``param:<execute>.<P>``, where
    P is the execute's own defaulted parameter when the pinned value comes from one, else the
    folded init's parameter (e.g. ``param:copy_tile.transpose`` for ``copy_init``'s transpose).

``ordering``
    * ``compute_api.ordering_facts`` (compute_api family): RISC-side config/GPR stores reached from
      a compute API whose consuming Tensix instruction has no (or only a conditional) fence.
      Without that family, a body-local scan of ``regfile[]``/``cfg[]``/``TT_CFG_STORE`` stores and
      their consumers is used instead.
    * an SFPU execute that leaves ``SFPLOADMACRO`` sequences in flight (fewer than three
      ``SFPNOP`` and no ``WAIT_SFPU`` after its last LOADMACRO) followed, before any SFPU drain, by
      another op whose folded init rewrites state those sequences use with a different value.

Ranking: concrete clobbers (fold drops, dependent readers, cross-thread writers) above the
replay / ordering mechanisms, known values above unknown, SDPA/attention kernels and helpers
above the rest within a class, same-loop above kernel-wide, more kernels first. Resources
ignored by ``resource_conflict`` (every init canonically resets them and every execute consumes
them only as transient state) are listed in :data:`TRANSIENT_RESOURCES`.

Model refinements applied to every rule:

* the thread of a kernel-call effect is the compile-time thread of its path (the kernel call's
  ``UNPACK/MATH/PACK`` wrapper, the wrappers and ``#ifdef TRISC_*`` regions on ``via`` and at the
  sink); name hints (``_llk_math_*`` SFPU helpers run on the pack thread under ``PACK((..))``) do
  not override it, and an effect under two different compile-time threads is compiled out;
* a setter is part of the FOLD only in an execute's prologue (before its first execute call) or
  epilogue (after its last one); a setter between two executes is the op's own multi-phase
  sequencing (``generalized_moe_gate``, ``mul_reduce_scalar_tile``) and predates the fold;
* fold_drop matches the kernel's setter up to :data:`FOLD_DEPTH` levels below both the execute
  and the setter, names the state after the shallowest fold parameter driving the same sinks,
  counts only sinks reachable under the init's own bindings (Blackhole ELWMUL ignores
  ``acc_to_dest``), and requires the setter's value to actually write that state.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

from . import kernels as kmod
from . import resources, scan
from .model import classify_value, compact, identifiers, substitute

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
#: (regex on canonical resource name, why it is not a conflict hazard).
TRANSIENT_RESOURCES: tuple[tuple[str, str], ...] = (
    (
        r"^RWC\.",
        "math src/dest read-write counters: every math/SFPU execute resets them",
    ),
    (
        r"^ADC\.",
        "unpacker/packer address counters: every unpack/pack execute sets them per tile",
    ),
    (
        r"^MOP$",
        "MOP expander config: every init programs its own MOP; its execute runs it right away",
    ),
    (
        r"^THDCFG\.DEST_TARGET_REG_CFG_",
        "dest read/write offset: every execute sets it per tile (set_dst_write_addr)",
    ),
    (
        r"^CFG\.THCON_SEC[01]_REG3_Base",
        "unpacker L1 base addresses: written per tile by every unpack execute",
    ),
    (
        r"^CFG\.THCON_SEC[01]_REG1_L1_Dest_addr",
        "packer L1 destination: written per tile by every pack execute",
    ),
    (
        r"^CFG\.SCRATCH_SEC",
        "CFGSHIFTMASK scratch operands: written and consumed inside one execute",
    ),
    (
        r"^GPR\.\w+\.(?:TMP\w*|OUTPUT_ADDR\w*|ZERO|ONE|PERF_\w*|R\d+)$",
        "scratch/constant GPRs: staged and consumed inside one call",
    ),
    (
        r"^(?:SEM|MAILBOX|SW)\.",
        "synchronization objects and software bookkeeping, not config state",
    ),
    (r"^param:", "fold parameters are handled by the fold_drop rule"),
)
_TRANSIENT = [re.compile(rx) for rx, _ in TRANSIENT_RESOURCES]

#: Calls that never carry op state (synchronization / CB bookkeeping / getters).
SKIP_CALLS = re.compile(
    r"^(?:tile_regs_\w+|acquire_dst|release_dst|cb_\w+|get_\w+|read_tile_value|"
    r"llk_(?:math|pack|unpack)_(?:wait|dest_section|pack_wait|release|get)\w*|"
    r"llk_packer_wait_for_math_done|llk_push_\w+|llk_pop_\w+|llk_wait_\w+|"
    r"zero|none|is_\w+|mailbox_\w+|DPRINT\w*|UNPACK|MATH|PACK)$"
)
#: Roles that contribute op effects.
OP_ROLES = (
    "execute",
    "init",
    "reconfig",
    "uninit",
    "config",
    "hw_startup",
    "primitive",
)
#: Roles of an explicit configuration call (fold_drop setters).
SETTER_ROLES = ("init", "reconfig", "config", "uninit", "hw_startup")
#: Parameters that never denote state (call-site bookkeeping).
IGNORED_PARAMS = re.compile(r"^(?:call_line|line|file|operand_id|loc)$")

#: Resource-group priority (most specific first) used to merge related fields.
GROUP_PRIORITY = (
    "pack.relu",
    "pack.edge_mask",
    "pack.l1_acc",
    "pack.strides",
    "pack.counters",
    "pack.format",
    "unpack.tilize",
    "unpack.address",
    "unpack.format",
    "alu.format",
    "thread.dest_offset",
    "thread.srca",
    "thread.cfg_state",
    "sfpu.constants",
    "sfpu.loadmacro",
    "sfpu.lreg",
    "sfpu.control",
)
#: Per-operand format/geometry state: every reconfig derives it from the CB being processed, so a
#: conflict only matters when the CBs really differ in format (unknown statically).
FORMAT_GROUPS = ("unpack.format", "alu.format", "pack.format", "unpack.tilize")
FORMAT_STATE = re.compile(
    r"TILE_DESC|TileDescriptor|Tile_x_dim|LF8_4b_exp|INT8_math|Dstacc|Zero_Flag|_data_format|Fp32_enabled|"
    r"Out_data_format|In_data_format|SrcA|SrcB|Dest_cntx"
)
SFPU_PAIR_STATE = re.compile(r"^SFPU\.(?:LOADMACRO\.|LREG1[1-4]$|CTRL$)")
SFPU_MACRO_STATE = re.compile(r"^SFPU\.(?:LOADMACRO\.|LREG1[1-4]$|CTRL$)|^ADDR_MOD\.7$")

MAX_EVIDENCE = 24
MAX_OPS = 8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_transient(resource: str) -> bool:
    return any(rx.search(resource) for rx in _TRANSIENT)


def is_deferred(detail: str | None) -> bool:
    """Effect recorded into a MOP / replay program rather than issued at its sink.

    Families spell it ``deferred`` (``TT_OP_*`` words), ``deferred:replay`` (sfpu replay
    recording) or as a ``;deferred`` part of a ``start=..;len=..`` replay detail (math).
    """
    return any(
        part.strip().split(":", 1)[0] == "deferred"
        for part in (detail or "").split(";")
    )


_CAST = re.compile(r"\b(?:static_cast|reinterpret_cast)\s*<[^<>]*>")
_RELU_PARENS = re.compile(r"\(\s*((?:ckernel::)?ReluConfig::\w+\(\))\s*\)")
_RELU = (
    (
        re.compile(
            r"(?:ckernel::)?ReluConfig::none\(\)\s*\.\s*get_(?:mode|threshold)\(\)"
        ),
        "0",
    ),
    (re.compile(r"(?:ckernel::)?ReluConfig::zero\(\)\s*\.\s*get_threshold\(\)"), "0"),
    (re.compile(r"(?:ckernel::)?ReluConfig::zero\(\)\s*\.\s*get_mode\(\)"), "1"),
)
_INT_CACHE: dict[str, int | None] = {}


def _int(db, text: str) -> int | None:
    text = compact(text or "")
    if text in _INT_CACHE:
        return _INT_CACHE[text]
    value = _int_uncached(db, text)
    _INT_CACHE[text] = value
    return value


def _int_uncached(db, text: str) -> int | None:
    if text in ("true", "false"):
        return 1 if text == "true" else 0
    if "cast" in text or "ReluConfig" in text:
        text = _CAST.sub("", text)
        text = _RELU_PARENS.sub(r"\1", text)
        for regex, replacement in _RELU:
            text = regex.sub(replacement, text)
    try:
        value = scan.evaluate(text)
    except Exception:  # pragma: no cover - defensive
        value = None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    ternary = _split_ternary(text)
    if ternary is not None:
        # ``cond ? a : b`` whose condition decides through its conjuncts (scan.truth) even when
        # an operand of the condition is opaque
        verdict = truth(ternary[0])
        if verdict is not None:
            return _int(db, ternary[1] if verdict else ternary[2])
    try:
        return db.tables.int_value(text)
    except Exception:  # pragma: no cover - defensive
        return None


def _split_ternary(text: str):
    """(condition, if-true, if-false) of a top-level ``c ? a : b`` (outer parentheses removed)."""
    text = compact(text)
    while text.startswith("(") and scan.match_close(text, 0, "()") == len(text) - 1:
        text = compact(text[1:-1])
    depth = 0
    question = None
    nested = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            if question is None:
                question = index
            else:
                nested += 1
        elif depth == 0 and char == ":" and question is not None:
            if text[index - 1 : index + 1] == "::" or text[index : index + 2] == "::":
                continue
            if nested:
                nested -= 1
                continue
            return (
                compact(text[:question]),
                compact(text[question + 1 : index]),
                compact(text[index + 1 :]),
            )
    return None


_UNBOUND = re.compile(r"(?:^|[\s(,<\[])\?[A-Za-z_]")


def has_unbound(text: str) -> bool:
    """Whether a substituted text still holds an unbound parameter (``?name``)."""
    return bool(_UNBOUND.search(text or ""))


_OPAQUE_SFPCONFIG = re.compile(r"^imm=(?P<imm>[^;]*);mod=(?P<mod>.*)$")


def value_identity(db, effect) -> tuple:
    """A hashable identity of a written value; unknown values are unique per sink."""
    text = compact(effect.value or "")
    if effect.value_kind == "unknown" or not text or has_unbound(text):
        return ("u", effect.evidence, effect.resource, text)
    match = _OPAQUE_SFPCONFIG.match(text)
    if match:
        mod = _int(db, match.group("mod"))
        if (
            mod is None or mod % 2 == 0
        ):  # SFPCONFIG copies LREG0: the value is not the immediate
            return ("u", effect.evidence, effect.resource, text)
    number = _int(db, text)
    if number is not None:
        return ("i", number)
    return ("t", text)


def values_differ(db, a, b) -> bool:
    return value_identity(db, a) != value_identity(db, b)


def _replay_range(db, effect) -> tuple[int, int] | None:
    detail = effect.detail or ""
    start = re.search(r"start=([^;]+)", detail)
    length = re.search(r"len=([^;]+)", detail)
    if not start or not length:
        return None
    s, n = _int(db, start.group(1)), _int(db, length.group(1))
    if s is None or n is None:
        return None
    return (s, s + max(n, 1))


def _replay_bounds(db, effect) -> tuple[int, float] | None:
    """(start, end) of a replay range; end is +inf when only the start is known."""
    detail = effect.detail or ""
    start = re.search(r"start=([^;]+)", detail)
    if not start:
        return None
    s = _int(db, start.group(1))
    if s is None:
        return None
    length = re.search(r"len=([^;]+)", detail)
    n = _int(db, length.group(1)) if length else None
    return (s, s + max(n, 1) if n is not None else math.inf)


def _sfpu_sink(effect) -> bool:
    path = effect.evidence.rpartition(":")[0]
    return "/sfpu/" in path or path.rsplit("/", 1)[-1].startswith("ckernel_sfpu")


def replay_overlap_known(db, a, b) -> bool:
    """Both ranges have a known start and provably overlap."""
    ra, rb = _replay_bounds(db, a), _replay_bounds(db, b)
    if ra is None or rb is None or (math.isinf(ra[1]) and math.isinf(rb[1])):
        return False
    return ra[0] < rb[1] and rb[0] < ra[1]


def _replay_overlap(db, a, b) -> bool:
    ra, rb = _replay_range(db, a), _replay_range(db, b)
    if ra is None or rb is None:
        return True
    return ra[0] < rb[1] and rb[0] < ra[1]


def _thread_ok(resource: str, ta: str, tb: str) -> bool:
    if resources.scope_of(resource) == "per_thread":
        return resources.same_thread(ta, tb)
    return True


_ATTENTION_OP = re.compile(r"(?i)sdpa|attention|attn|flash|mla")


FORMAT_VALUE = re.compile(
    r"(?i)format|operand|num_faces|face_r_dim|tile_size|tile_shape|get_output|\bi?o?cb\b|_cb\b|\w_dim\b"
)


def format_derived(db, resource: str) -> bool:
    return bool(FORMAT_STATE.search(resource)) or bool(
        set(db.tables.groups_of(resource)) & set(FORMAT_GROUPS)
    )


#: Functions whose writes are derived from the processed operand's data format / geometry.
FORMAT_WRITER = re.compile(r"data_format|reconfig_df|full_operand|tile_shape|tile_dim")


def format_value(effect) -> bool:
    """A written value derived from the processed operand's format / geometry: the value text
    names a format/operand, or the write is made by a data-format reconfig function."""
    if FORMAT_VALUE.search(effect.value or "") or effect.value_kind == "unknown":
        return True
    for key, _ in effect.via[1:]:
        name = key.split("(", 1)[0].rsplit("::", 1)[-1].split("<", 1)[0]
        if FORMAT_WRITER.search(name):
            return True
    return False


SFPU_STATE_GROUPS = ("sfpu.constants", "sfpu.loadmacro", "sfpu.control")


def group_key(db, resource: str) -> str:
    groups = set(db.tables.groups_of(resource))
    for group in GROUP_PRIORITY:
        if group in groups:
            return "sfpu.state" if group in SFPU_STATE_GROUPS else group
    if resource.startswith("ADDR_MOD_PACK."):
        return "addr_mod_pack"
    if resource.startswith("ADDR_MOD."):
        slot = resource.split(".", 1)[1]
        return "addr_mod.sfpu" if slot in ("6", "7") else "addr_mod.fpu"
    storage = (
        db.tables.storage(resource)
        if resource.startswith(("CFG.", "THDCFG."))
        else None
    )
    if storage is not None:
        return f"cfgword:{storage[0]}:{storage[1]}"
    return resource


def _short(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _sink_function(db, effect) -> str:
    """Name of the function that holds the sink (deepest ``via`` step)."""
    if effect.via:
        fn = db.function(effect.via[-1][0])
        if fn is not None:
            return fn.name
        return effect.via[-1][0].split("(", 1)[0].rsplit("::", 1)[-1].split("<", 1)[0]
    return ""


def _condition_parts(call: kmod.KCall) -> str:
    parts = list(call.ifdefs)
    if call.condition:
        parts.append(call.condition)
    return " && ".join(f"({p})" if "||" in p or "?" in p else p for p in parts)


def _exclusive(a: kmod.KCall, b: kmod.KCall) -> bool:
    return kmod.exclusive_conditions(_condition_parts(a), _condition_parts(b))


def _apart(a, b) -> bool:
    """Two accesses that can never be live in the same compiled kernel (calls or sinks exclusive)."""
    if not (a.call.condition or a.call.ifdefs or a.effect.condition) or not (
        b.call.condition or b.call.ifdefs or b.effect.condition
    ):
        return False
    left = scan.join_conditions(_condition_parts(a.call), a.effect.condition)
    right = scan.join_conditions(_condition_parts(b.call), b.effect.condition)
    return kmod.exclusive_conditions(left, right)


# --------------------------------------------------------------------------- #
# Context: effects of kernel calls (cached)
# --------------------------------------------------------------------------- #
@dataclass
class _Access:
    call: kmod.KCall
    effect: object
    restore: bool = (
        False  # writes back a value the same call saved from config (RDCFG -> GPR)
    )


_GPR_VALUE = re.compile(r"\bGPR\(\s*(?:[\w:]*::)?(\w+)\s*\)")


def _saved_gprs(effects) -> set[str]:
    """GPR names a call loads from config (``RDCFG``): the save half of a save/restore pair."""
    saved: set[str] = set()
    for effect in effects:
        if (
            effect.access == "write"
            and effect.resource.startswith("GPR.")
            and (
                (effect.pattern or "").startswith("core.rdcfg")
                or (effect.value or "").startswith("CFG(")
            )
        ):
            saved.add(effect.resource.rsplit(".", 1)[-1])
    return saved


def _is_restore(effect, saved: set[str]) -> bool:
    if not saved or effect.access != "write":
        return False
    names = set(_GPR_VALUE.findall(effect.value or ""))
    return bool(names) and names <= saved


_TRUTH: dict[str, object] = {}


def truth(text: str):
    """Memoized three-valued ``scan.truth``."""
    if text not in _TRUTH:
        _TRUTH[text] = scan.truth(text)
    return _TRUTH[text]


class Context:
    """Kernel calls plus their effects (the callee's own-scope effects with the call's
    arguments substituted and its ``if constexpr`` conditions folded; cached)."""

    def __init__(self, db, kernels: Sequence[kmod.Kernel]):
        self.db = db
        self.kernels = list(kernels)
        self._effects: dict[tuple, tuple] = {}
        self._own: dict[str, tuple] = {}
        self._reach: dict[str, frozenset[str]] = {}
        self._edge_threads: dict[tuple, str | None] = {}
        self._sink_threads: dict[tuple, str | None] = {}

    # ---- compiled thread of an effect --------------------------------------- #
    def _edge_thread(self, parent: str, child: str, evidence: str) -> str | None:
        key = (parent, child, evidence)
        if key in self._edge_threads:
            return self._edge_threads[key]
        thread = None
        fn = self.db.function(parent)
        if fn is not None and evidence:
            edges = [e for e in fn.calls if e.evidence == evidence]
            exact = [e for e in edges if child in e.callee_keys]
            for edge in exact or edges:
                if edge.thread:
                    thread = edge.thread
                    break
        self._edge_threads[key] = thread
        return thread

    def _sink_thread(self, holder: str, evidence: str) -> str | None:
        key = (holder, evidence)
        if key in self._sink_threads:
            return self._sink_threads[key]
        thread = None
        fn = self.db.function(holder)
        definition = fn.definition if fn is not None else None
        path, _, line = evidence.rpartition(":")
        if definition is not None and line.isdigit() and path == definition.path:
            number = int(line)
            for site in definition.calls:
                if site.line == number and site.thread:
                    thread = site.thread
                    break
            if thread is None:
                thread = definition.pp_info_at_line(number)[0] or definition.pp_thread
        self._sink_threads[key] = thread
        return thread

    def compiled_threads(self, call: kmod.KCall, via, evidence: str) -> list[str]:
        """Threads fixed at compile time along the path of an effect: the kernel call's
        ``UNPACK/MATH/PACK`` wrapper, the wrapper or ``#ifdef TRISC_*`` region of every call on
        ``via``, and of the sink. Thread hints taken from function names are not compile-time
        facts (``_llk_math_*`` SFPU helpers also run on the pack thread)."""
        threads: list[str] = []
        if call.thread:
            threads.append(call.thread)
        for position in range(1, len(via)):
            thread = self._edge_thread(
                via[position - 1][0], via[position][0], via[position][1]
            )
            if thread:
                threads.append(thread)
        if via:
            thread = self._sink_thread(via[-1][0], evidence)
            if thread:
                threads.append(thread)
        return threads

    # ---- prologue / epilogue of an execute ---------------------------------- #
    def phase(self, parent: str, evidence: str) -> str:
        """Where a call sits in a compute-API execute body: ``prologue`` (before the first call
        of an execute), ``epilogue`` (after the last one) or ``body`` (between executes: a
        multi-phase op re-configuring itself mid-sequence)."""
        return setter_phase(self.db, parent, evidence)

    def _own_effects(self, key: str) -> tuple:
        cached = self._own.get(key)
        if cached is None:
            try:
                effects = tuple(self.db.effects(key))
            except Exception:  # pragma: no cover - defensive
                effects = ()
            names: set[str] = set()
            for effect in effects:
                names.update(effect.params)
            cached = (effects, frozenset(names))
            self._own[key] = cached
        return cached

    def effects(self, call: kmod.KCall) -> tuple:
        if not call.keys:
            return self._call_effects(call)
        if len(call.keys) == 1:
            return self._key_effects(call, call.keys[0], low=False)
        merged = []
        for key in call.keys:
            merged.extend(self._key_effects(call, key, low=True))
        return tuple(merged)

    def _key_effects(self, call: kmod.KCall, key: str, *, low: bool) -> tuple:
        fn = self.db.function(key)
        if fn is None:
            return ()
        effects, relevant = self._own_effects(key)
        try:
            binding = self.db.graph.bind(fn, call.template_args, call.args)
        except Exception:  # pragma: no cover - defensive
            binding = {}
        bound = {
            name: compact(binding[name]) for name in sorted(relevant) if name in binding
        }
        cache_key = (key, tuple(sorted(bound.items())), call.thread, low)
        cached = self._effects.get(cache_key)
        if cached is not None:
            return cached
        out = []
        evidence = call.evidence
        for effect in effects:
            changes: dict = {}
            via = ((key, ""),) + effect.via
            compiled = self.compiled_threads(call, via, effect.evidence)
            if len(set(compiled)) > 1:
                continue  # compiled out: inside another thread's wrapper / TRISC region
            if compiled and compiled[0] != effect.thread:
                changes["thread"] = compiled[0]
            uses = set(effect.params) & set(bound)
            if uses:
                value = substitute(effect.value, bound) if effect.value else ""
                condition = (
                    substitute(effect.condition, bound) if effect.condition else ""
                )
                if condition:
                    verdict = truth(condition)
                    if verdict is False:
                        continue
                    if verdict is True:
                        condition = ""
                changes["value"] = value
                changes["condition"] = condition
                if value != effect.value:
                    changes["value_kind"] = classify_value(
                        value, (), self.db.graph.constants
                    )
                if effect.detail:
                    changes["detail"] = substitute(effect.detail, bound)
            if effect.thread == "any" and call.thread and not compiled:
                changes["thread"] = call.thread
            changes["via"] = via
            if low:
                changes["confidence"] = "low"
            out.append(replace(effect, **changes))
        result = tuple(out)
        self._effects[cache_key] = result
        return result

    def _call_effects(self, call: kmod.KCall) -> tuple:
        key = (
            "call",
            call.name,
            call.qualifier,
            call.template_args,
            call.args,
            call.thread,
            call.is_member,
            call.receiver_type,
        )
        cached = self._effects.get(key)
        if cached is None:
            try:
                cached = self.db.call_effects(
                    call.name,
                    qualifier=call.qualifier,
                    template_args=call.template_args,
                    args=call.args,
                    thread=call.thread,
                    evidence=call.evidence,
                    caller_path=call.path,
                    is_member=call.is_member,
                    receiver_type=call.receiver_type,
                )
            except Exception:  # pragma: no cover - defensive
                cached = ()
            self._effects[key] = cached
        return cached

    def reach(self, key: str) -> frozenset[str]:
        cached = self._reach.get(key)
        if cached is None:
            try:
                found = {k for k, _, _ in self.db.reachable(key, include_stopped=True)}
            except Exception:  # pragma: no cover - defensive
                found = set()
            found.add(key)
            cached = frozenset(found)
            self._reach[key] = cached
        return cached


def _execute_span(db, key: str) -> tuple:
    """(first, last) line of the calls of ``key`` that reach an execute (LLK, compute API or SFPU
    execute); ``(None, None)`` when the body calls none."""
    fn = db.function(key)
    lines: list[int] = []
    if fn is not None:
        for edge in fn.calls:
            callees = [db.function(k) for k in edge.callee_keys]
            if any(c is not None and c.role == "execute" for c in callees):
                lines.append(_evidence_key(edge.evidence)[1])
    if not lines:
        return (None, None)
    return (min(lines), max(lines))


_SPANS: dict[tuple, tuple] = {}


def setter_phase(db, parent: str, evidence: str) -> str:
    """``prologue`` / ``epilogue`` / ``body`` position of a call inside an execute (see
    :meth:`Context.phase`). The FOLD puts an op's own reconfig/init before its executes and its
    uninit after them; a setter between two executes is the op's own multi-phase sequencing
    (``generalized_moe_gate`` re-initializing for its sigmoid phase, ``mul_reduce_scalar_tile``
    arming the packer mask for the pack that follows), which predates the fold."""
    key = (id(db), parent)
    span = _SPANS.get(key)
    if span is None:
        span = _SPANS[key] = _execute_span(db, parent)
    first, last = span
    line = _evidence_key(evidence)[1]
    if first is None or line < first:
        return "prologue"
    if line > last:
        return "epilogue"
    return "body"


def op_calls(kernel: kmod.Kernel) -> list[kmod.KCall]:
    return [
        call
        for call in kernel.calls
        if call.role in OP_ROLES and not SKIP_CALLS.match(call.name)
    ]


# --------------------------------------------------------------------------- #
# Finding accumulation
# --------------------------------------------------------------------------- #
@dataclass
class _Agg:
    kind: str
    resource_group: str
    ops: tuple[str, ...]
    resources: dict = field(default_factory=dict)  # resource -> kernel count
    kernels: set = field(default_factory=set)
    attention_kernels: set = field(default_factory=set)
    evidence: set = field(default_factory=set)
    flags: dict = field(default_factory=dict)  # flag -> set(kernels)
    values: dict = field(default_factory=dict)  # op -> set(value texts)
    threads: set = field(default_factory=set)
    details: set = field(default_factory=set)
    llk: dict = field(default_factory=dict)  # op -> set(sink function names)

    def flag(self, name: str, kernel: str) -> None:
        self.flags.setdefault(name, set()).add(kernel)

    def has(self, name: str) -> bool:
        return bool(self.flags.get(name))


def _add_evidence(agg: _Agg, *items: str) -> None:
    for item in items:
        if item:
            agg.evidence.add(item)


def _cap_evidence(evidence: Iterable[str], priority: Sequence[str] = ()) -> list[str]:
    ordered = [e for e in priority if e]
    rest = sorted(set(evidence) - set(ordered), key=_evidence_key)
    out: list[str] = []
    for item in ordered + rest:
        if item not in out:
            out.append(item)
        if len(out) >= MAX_EVIDENCE:
            break
    return out


def _evidence_key(text: str) -> tuple:
    path, _, line = text.rpartition(":")
    return (path, int(line) if line.isdigit() else 0)


# --------------------------------------------------------------------------- #
# Rule (a): resource_conflict
# --------------------------------------------------------------------------- #
def _collect_accesses(ctx: Context, kernel: kmod.Kernel):
    writes: dict[str, dict[str, list[_Access]]] = {}
    reads: dict[str, dict[str, list[_Access]]] = {}
    unresolved: dict[str, set[str]] = {}
    for call in op_calls(kernel):
        effects = ctx.effects(call)
        used_instr = _used_macro_templates(effects)
        saved = _saved_gprs(effects)
        for effect in effects:
            resource = effect.resource
            if (
                used_instr
                and effect.access == "read"
                and resource.startswith("SFPU.LOADMACRO.INSTR")
                and resource.rsplit(".", 1)[1] not in used_instr
            ):
                continue  # the op's own sequence words do not use this template
            if effect.access == "write" and (not effect.resolved or "?" in resource):
                unresolved.setdefault(call.name, set()).add(
                    resources.resource_class(resource)
                )
                continue
            if not effect.resolved or "?" in resource or is_transient(resource):
                continue
            if effect.access == "write" and is_deferred(effect.detail):
                continue
            target = writes if effect.access == "write" else reads
            target.setdefault(resource, {}).setdefault(call.name, []).append(
                _Access(call, effect, restore=_is_restore(effect, saved))
            )
    # several sinks describe one write (word-level core pattern + field-level family pattern):
    # keep, per call and sink, the write whose value is best known
    for by_op in writes.values():
        for op, accesses in by_op.items():
            best: dict[tuple, _Access] = {}
            for access in accesses:
                key = (
                    access.call.index,
                    access.effect.evidence,
                    access.effect.condition,
                )
                if key not in best or _known_rank(ctx.db, access.effect) > _known_rank(
                    ctx.db, best[key].effect
                ):
                    best[key] = access
            by_op[op] = sorted(
                best.values(),
                key=lambda a: (a.call.index, a.effect.evidence, a.effect.value),
            )
    return writes, reads, unresolved


def _known_rank(db, effect) -> int:
    identity = value_identity(db, effect)
    return {"i": 3, "t": 2, "u": 0}.get(identity[0], 1)


_USES = re.compile(r"uses=([^;]*)")


def _used_macro_templates(effects) -> set[str]:
    """INSTRn templates named by the sequence words an op programs (``uses=`` detail)."""
    used: set[str] = set()
    for effect in effects:
        if effect.access != "write" or not effect.resource.startswith(
            "SFPU.LOADMACRO.SEQ"
        ):
            continue
        match = _USES.search(effect.detail or "")
        if match:
            used.update(re.findall(r"INSTR\d", match.group(1)))
    return used


def _first_conflict(
    db,
    resource: str,
    xs: list[_Access],
    ys: list[_Access],
    *,
    require_loop: bool = False,
):
    """First (x, y) write pair with different values, compatible threads, live together."""
    for x in xs:
        for y in ys:
            if x.call.index == y.call.index:
                continue
            if require_loop and not kmod.same_loop(x.call, y.call):
                continue
            if _apart(x, y):
                continue
            if not _thread_ok(resource, x.effect.thread, y.effect.thread):
                continue
            if resource.startswith("REPLAY") and not _replay_overlap(
                db, x.effect, y.effect
            ):
                continue
            if not values_differ(db, x.effect, y.effect):
                continue
            return x, y
    return None


def _ordered_dependency(
    reads_r: list[_Access], writes_other: list[_Access]
) -> tuple | None:
    """A read that happens after (or in the same loop as) a write of another op."""
    for read in reads_r:
        for write in writes_other:
            if _apart(read, write):
                continue
            if write.call.index < read.call.index or kmod.same_loop(
                read.call, write.call
            ):
                return read, write
    return None


FOLD_ROLES = ("init", "reconfig", "config", "uninit", "hw_startup")


#: Layers of the init/reconfig/config/uninit entry points the fold moves into executes.
FOLD_LAYERS = ("compute_api", "llk_api")


def fold_path(db, access: _Access) -> bool:
    """Whether a write comes from an init/config/uninit that now runs inside an execute (the fold).

    The fold is a compute-API execute calling an API-level (compute API / llk_api) init,
    reconfig, config or uninit. Explicit kernel init/config calls are the pre-fold contract,
    and config helpers an LLK execute runs internally (``_configure_src_zero_flag_`` inside
    ``_llk_math_reduce_``) are that execute's own per-call contract; neither is counted.
    """
    if access.call.role != "execute":
        return False
    via = access.effect.via
    for position in range(1, len(via)):
        fn = db.function(via[position][0])
        if fn is None or fn.role not in FOLD_ROLES or fn.layer not in FOLD_LAYERS:
            continue
        parent = db.function(via[position - 1][0])
        if (
            parent is not None
            and parent.role == "execute"
            and parent.layer == "compute_api"
            and setter_phase(db, parent.key, via[position][1]) != "body"
        ):
            return True
    return False


def rule_conflicts(ctx: Context) -> list[dict]:
    db = ctx.db
    aggs: dict[tuple, _Agg] = {}
    for kernel in ctx.kernels:
        writes, reads, unresolved = _collect_accesses(ctx, kernel)
        order_cache: dict = {}
        for resource in sorted(writes):
            by_op = writes[resource]
            readers = reads.get(resource, {})
            writer_ops = sorted(by_op)
            group = group_key(db, resource)
            # ---- cross-thread writers ------------------------------------------ #
            # Two ops that both write a resource re-establish it themselves after the fold
            # (same thread: canonical reset, only thrash). Writes of shared state from two
            # threads now happen per tile and interleave without synchronization.
            if resources.scope_of(resource) == "shared" and group != "sfpu.lreg":
                _cross_thread(db, aggs, kernel, resource, group, by_op, readers)
            # ---- replay buffer ranges re-recorded in one loop body -------------- #
            # Every op re-records its own replay program at the start of its region (canonical,
            # like MOP). The hazard is an SFPU recording and an FPU/packer recording that
            # overlap: the buffer is split between them (replay_buf_offset), so an SFPU program
            # longer than its half, or an FPU program in the SFPU half, clobbers the other.
            if resource == "REPLAY" and kernel.attention:
                sfpu_ops = [
                    op
                    for op in writer_ops
                    if any(_sfpu_sink(a.effect) for a in by_op[op])
                ]
                other_ops = [op for op in writer_ops if op not in sfpu_ops]
                for op_a in sfpu_ops:
                    for op_b in other_ops:
                        pair = None
                        for x in by_op[op_a]:
                            if not _sfpu_sink(x.effect) or not fold_path(db, x):
                                continue
                            for y in by_op[op_b]:
                                if _sfpu_sink(y.effect) or not fold_path(db, y):
                                    continue
                                if not resources.same_thread(
                                    x.effect.thread, y.effect.thread
                                ):
                                    continue
                                if _apart(x, y) or not kmod.same_loop(x.call, y.call):
                                    continue
                                if replay_overlap_known(db, x.effect, y.effect):
                                    pair = (x, y)
                                    break
                            if pair:
                                break
                        if pair is None:
                            continue
                        x, y = pair
                        # one finding per SFPU recording (the init that records it) and thread:
                        # exp_packthread_tile and exp_tile_first_column both reach exp_init's
                        # recording, so they are one mechanism
                        recorder = _sink_function(db, x.effect) or op_a
                        key = (
                            "resource_conflict",
                            "replay_loop",
                            group,
                            (recorder, x.effect.thread),
                        )
                        agg = aggs.get(key)
                        if agg is None:
                            agg = aggs[key] = _Agg("replay_loop", group, (op_a,))
                        agg.values.setdefault("sfpu_ops", set()).add(op_a)
                        agg.values.setdefault("others", set()).add(op_b)
                        _record(
                            db,
                            agg,
                            kernel,
                            resource,
                            x,
                            y,
                            kmod.same_loop(x.call, y.call),
                            None,
                        )
                        agg.values.setdefault("ranges", set()).update(
                            {
                                f"{x.call.name}:{x.effect.detail[:24]}",
                                f"{y.call.name}:{y.effect.detail[:24]}",
                            }
                        )
            # ---- SFPU macro / constant state in one loop body ------------------- #
            if SFPU_PAIR_STATE.search(resource) and kernel.attention:
                for i, op_a in enumerate(writer_ops):
                    for op_b in writer_ops[i + 1 :]:
                        xs = [
                            a
                            for a in by_op[op_a]
                            if fold_path(db, a) and a.call.role != "hw_startup"
                        ]
                        ys = [
                            a
                            for a in by_op[op_b]
                            if fold_path(db, a) and a.call.role != "hw_startup"
                        ]
                        if not xs or not ys:
                            continue
                        pair = _first_conflict(db, resource, xs, ys, require_loop=True)
                        if pair is None:
                            continue
                        x, y = pair
                        if x.effect.thread != y.effect.thread and "any" not in (
                            x.effect.thread,
                            y.effect.thread,
                        ):
                            continue  # reported by the cross-thread check
                        key = ("resource_conflict", "sfpu_loop", group)
                        agg = aggs.get(key)
                        if agg is None:
                            agg = aggs[key] = _Agg("sfpu_loop", group, ())
                        agg.values.setdefault("pairs", set()).add(f"{op_a}+{op_b}")
                        _record(
                            db,
                            agg,
                            kernel,
                            resource,
                            x,
                            y,
                            kmod.same_loop(x.call, y.call),
                            None,
                        )
            # ---- dependent readers -------------------------------------------- #
            # An execute that reads the resource but never writes it sees whoever wrote it
            # last. A hazard when that last write (in program order, loops wrapped) is another
            # op's folded init/config inside an execute, and it differs from the value the
            # kernel's own setup (explicit init/config/hw_startup) left there.
            for reader_op in sorted(readers):
                if reader_op in by_op:
                    continue
                if resources.resource_class(resource) in unresolved.get(
                    reader_op, set()
                ):
                    continue  # the reader may write it through an unresolved name
                for read in readers[reader_op]:
                    if read.call.role != "execute":
                        continue
                    hit = _last_writes(db, kernel, order_cache, resource, read, by_op)
                    if hit is None:
                        continue
                    last, baseline = hit
                    key = ("resource_conflict", "dependent", group, (last.call.name,))
                    agg = aggs.get(key)
                    if agg is None:
                        agg = aggs[key] = _Agg("dependent", group, (last.call.name,))
                    other = baseline if baseline is not None else last
                    _record(
                        db,
                        agg,
                        kernel,
                        resource,
                        last,
                        other,
                        kmod.same_loop(last.call, read.call),
                        (read, last),
                        reader=read,
                    )
                    agg.values.setdefault("readers", set()).add(reader_op)
                    if baseline is not None:
                        agg.values.setdefault("baseline", set()).add(baseline.call.name)
                    break
    findings = [
        _conflict_finding(db, agg)
        for _, agg in sorted(aggs.items(), key=lambda kv: kv[0])
    ]
    # per-operand format state outside the attention kernels: no static evidence the CBs differ
    return [f for f in findings if not f.pop("_drop", False)]


_THREADS = ("unpack", "math", "pack")


def _loop_ranges(kernel: kmod.Kernel) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for call in kernel.calls:
        for loop in call.loops:
            lo, hi = ranges.get(loop.instance, (call.index, call.index))
            ranges[loop.instance] = (min(lo, call.index), max(hi, call.index))
    return ranges


def preceding(kernel: kmod.Kernel, call: kmod.KCall, cache: dict) -> list[int]:
    """Call indices in the order they last executed before ``call`` (loop back edges wrapped)."""
    key = call.index
    cached = cache.get(key)
    if cached is not None:
        return cached
    ranges = cache.get("__ranges__")
    if ranges is None:
        ranges = cache["__ranges__"] = _loop_ranges(kernel)
    order: list[int] = []
    seen: set[int] = {call.index}
    pos = call.index
    for loop in reversed(call.loops):
        lo, hi = ranges.get(loop.instance, (call.index, call.index))
        for index in list(range(pos - 1, lo - 1, -1)) + list(range(hi, call.index, -1)):
            if index not in seen:
                seen.add(index)
                order.append(index)
        pos = min(pos, lo)
    for index in range(pos - 1, -1, -1):
        if index not in seen:
            seen.add(index)
            order.append(index)
    cache[key] = order
    return order


def program_key(effect) -> tuple:
    """Program order of an effect inside the call that reaches it: the line of every call on
    its ``via`` chain (each in the body of the previous function), then the sink line. Two
    effects of one call compare in program order at their first differing level (an init's
    write before the uninit's write of the same execute)."""
    lines = []
    for _, evidence in effect.via:
        if evidence:
            lines.append(_evidence_key(evidence)[1])
    lines.append(_evidence_key(effect.evidence)[1])
    return tuple(lines)


def _last_writes(db, kernel, cache, resource, read: _Access, by_op):
    """(last write, kernel baseline write) seen by ``read`` when the last one is another op's
    folded write that differs from the baseline; else None."""
    by_index: dict[int, list[_Access]] = cache.setdefault(("writes", resource), {})
    if not by_index:
        for accesses in by_op.values():
            for access in accesses:
                by_index.setdefault(access.call.index, []).append(access)
        for accesses in by_index.values():
            # within one call, the write executed last comes first
            accesses.sort(key=lambda a: program_key(a.effect), reverse=True)
    last = None
    for index in preceding(kernel, read.call, cache):
        for access in by_index.get(index, ()):
            if not _thread_ok(resource, access.effect.thread, read.effect.thread):
                continue
            if _apart(access, read):
                continue
            if resource.startswith("REPLAY") and (
                _replay_range(db, read.effect) is None
                or not _replay_overlap(db, access.effect, read.effect)
            ):
                continue
            if access.restore:
                break  # the call's last write restores what it saved: the call is transparent
            if last is None:
                if not fold_path(db, access) or access.call.name == read.call.name:
                    return None  # the kernel's own setup (or the reader's op) wrote it last
                last = access
                break
            if fold_path(db, access):
                break  # still the fold's state; keep looking for the kernel's setup
            if values_differ(db, access.effect, last.effect):
                return last, access
            return None
    if last is None:
        return None
    # no explicit setup before: the resource holds its hardware default. A folded uninit
    # writes that default back (``_llk_pack_reduce_mask_clear_``), so it is not a hazard;
    # otherwise it is one only if another write of a different value exists
    if _via_role(db, last, "uninit"):
        return None
    for accesses in by_op.values():
        for access in accesses:
            if access.restore:
                continue
            if access.call.name != last.call.name and values_differ(
                db, access.effect, last.effect
            ):
                return last, None
    return None


def _via_role(db, access: _Access, role: str) -> bool:
    for key, _ in access.effect.via[1:]:
        fn = db.function(key)
        if fn is not None and fn.role == role:
            return True
    return False


def _cross_thread(db, aggs, kernel, resource, group, by_op, readers) -> None:
    """Fold-path writes of one shared resource from two threads with different values."""
    by_thread: dict[str, dict[str, list[_Access]]] = {}
    for op, accesses in by_op.items():
        for access in accesses:
            if access.call.role == "hw_startup" or not fold_path(db, access):
                continue
            thread = access.effect.thread
            if thread in _THREADS:
                by_thread.setdefault(thread, {}).setdefault(op, []).append(access)
    threads = sorted(by_thread)
    if len(threads) < 2:
        return
    all_reads = [a for accesses in readers.values() for a in accesses]
    for i, t1 in enumerate(threads):
        for t2 in threads[i + 1 :]:
            found = None
            for op_a in sorted(by_thread[t1]):
                for op_b in sorted(by_thread[t2]):
                    if op_a == op_b:
                        continue  # one op configuring its own threads
                    for x in by_thread[t1][op_a]:
                        for y in by_thread[t2][op_b]:
                            if _apart(x, y) or not values_differ(
                                db, x.effect, y.effect
                            ):
                                continue
                            loop = kmod.same_loop(x.call, y.call)
                            dep = (
                                _ordered_dependency(all_reads, [x, y])
                                if not loop
                                else None
                            )
                            if loop or dep:
                                found = (x, y, loop, dep)
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                continue
            # one finding per resource group and thread pair (t1 < t2); which side is the
            # minority is decided over all kernels when the finding is built
            key = ("resource_conflict", "cross", group, (t1, t2))
            agg = aggs.get(key)
            if agg is None:
                agg = aggs[key] = _Agg("cross", group, ())
            agg.values.setdefault(f"ops:{t1}", set()).update(by_thread[t1])
            agg.values.setdefault(f"ops:{t2}", set()).update(by_thread[t2])
            x, y, loop, dep = found
            _record(db, agg, kernel, resource, x, y, loop, dep)


def _record(db, agg: _Agg, kernel, resource, x, y, loop, dep, reader=None) -> None:
    agg.kernels.add(kernel.path)
    agg.resources[resource] = agg.resources.get(resource, 0) + 1
    attention = kernel.attention and (
        x.call.in_attention
        or y.call.in_attention
        or (reader and reader.call.in_attention)
    )
    if attention:
        agg.attention_kernels.add(kernel.path)
    if loop:
        agg.flag("same_loop", kernel.path)
    if dep:
        agg.flag("ordered", kernel.path)
    if reader is not None:
        agg.flag("dependent", kernel.path)
        writer = dep[1] if dep else None
        if writer is not None and any(
            (db.function(k) is not None and db.function(k).role == "uninit")
            for k, _ in writer.effect.via[1:]
        ):
            agg.flag("uninit", kernel.path)
    tx, ty = x.effect.thread, y.effect.thread
    if resources.scope_of(resource) == "shared" and tx != ty and "any" not in (tx, ty):
        agg.flag("cross_thread", kernel.path)
    if format_derived(db, resource) and (
        format_value(x.effect) or format_value(y.effect)
    ):
        agg.flag("format_pair", kernel.path)
    else:
        agg.flag("plain_pair", kernel.path)
    known = all(value_identity(db, e.effect)[0] != "u" for e in (x, y))
    if known:
        agg.flag("known_values", kernel.path)
    if x.effect.confidence == "low" or y.effect.confidence == "low":
        agg.flag("low", kernel.path)
    agg.threads.update({tx, ty})
    for access in (x, y):
        agg.values.setdefault(access.call.name, set()).add(
            compact(access.effect.value)[:80] or "?"
        )
        sink = _sink_function(db, access.effect)
        if sink:
            agg.llk.setdefault(access.call.name, set()).add(sink)
    if len(agg.evidence) < 4 * MAX_EVIDENCE:
        _add_evidence(
            agg, x.effect.evidence, y.effect.evidence, x.call.evidence, y.call.evidence
        )
        if reader is not None:
            _add_evidence(agg, reader.effect.evidence, reader.call.evidence)
            sink = _sink_function(db, reader.effect)
            if sink:
                agg.llk.setdefault(reader.call.name, set()).add(sink)


def _representative(agg: _Agg) -> str:
    return sorted(agg.resources.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _conflict_finding(db, agg: _Agg) -> dict:
    resource = _representative(agg)
    also = sorted(r for r in agg.resources if r != resource)
    llk: list[str] = []
    if agg.kind == "dependent":
        fold_writers = list(agg.ops)
        readers = sorted(agg.values.get("readers", set()))
        baseline = sorted(agg.values.get("baseline", set()))
        ops = fold_writers + readers
        for op in fold_writers + readers:
            llk.extend(sorted(agg.llk.get(op, set())))
        uninit = " and folded uninit" if agg.has("uninit") else ""
        setup = f"the {', '.join(baseline)} setup" if baseline else "the kernel's setup"
        verb = "rewrites" if len(fold_writers) == 1 else "rewrite"
        summary = (
            f"{', '.join(fold_writers)} now {verb} {resource} inside the execute (folded init/config{uninit}), so "
            f"{', '.join(readers[:6])}{' ...' if len(readers) > 6 else ''}, which read it without writing it, "
            f"{'sees' if len(readers) == 1 else 'see'} that value instead of {setup}"
            + (" in the same loop" if agg.has("same_loop") else "")
        )
    elif agg.kind == "cross":
        sides = sorted(
            (
                (len(ops), key.split(":", 1)[1], sorted(ops))
                for key, ops in agg.values.items()
                if key.startswith("ops:")
            ),
            key=lambda side: (side[0], side[1]),
        )
        (_, t_min, minority), (_, t_maj, majority_all) = sides[0], sides[-1]
        majority = sorted(set(majority_all) - set(minority))
        ops = minority + majority
        for op in minority + majority:
            llk.extend(sorted(agg.llk.get(op, set())))
        summary = (
            f"{', '.join(minority)} {'writes' if len(minority) == 1 else 'write'} {resource} from the {t_min} thread "
            f"while {', '.join(majority[:5])}{' ...' if len(majority) > 5 else ''} "
            f"{'writes' if len(majority) == 1 else 'write'} it with a different value from the "
            f"{t_maj} thread; after the fold both writes run per tile inside executes, so the threads interleave "
            f"unsynchronized writes of shared config"
        )
    elif agg.kind == "replay_loop":
        sfpu_ops = sorted(agg.values.get("sfpu_ops", set()) or {agg.ops[0]})
        op_a = ", ".join(sfpu_ops)
        others = sorted(set(agg.values.get("others", set())) - set(sfpu_ops))
        op_b = ", ".join(others[:4]) + (" ..." if len(others) > 4 else "")
        ops = sfpu_ops + others
        llk = []
        for op in sfpu_ops:
            llk.extend(sorted(agg.llk.get(op, set())))
        for other in others[:3]:
            llk.extend(sorted(agg.llk.get(other, set())))
        threads = "/".join(sorted(t for t in agg.threads if t and t != "any"))
        ranges = "; ".join(sorted(agg.values.get("ranges", set()))[:4])
        summary = (
            f"{op_a} and {op_b} re-record overlapping replay-buffer ranges ({ranges}) on the {threads or 'same'} "
            f"thread in the same loop body; after the fold each execute re-records its replay program every call, "
            f"so a hoisted or deduplicated recording replays the other op's instructions"
        )
    else:
        pairs = sorted(agg.values.get("pairs", set()))
        counts: dict[str, int] = {}
        for pair in pairs:
            for op in pair.split("+"):
                counts[op] = counts.get(op, 0) + 1
        ranked = sorted(counts, key=lambda op: (-counts[op], op))
        op_a, op_b = (ranked + ["?", "?"])[:2]
        ops = list(ranked)
        llk = []
        for op in ranked[:3]:
            llk.extend(sorted(agg.llk.get(op, set())))
        values = "; ".join(
            f"{op}={'|'.join(sorted(agg.values.get(op, {'?'}))[:2])}"
            for op in (op_a, op_b)
        )
        summary = (
            f"{op_a} and {op_b} both reprogram {resource} with different values ({values}) in the same loop "
            f"body{' (also ' + ', '.join(pairs[1:4]) + ')' if len(pairs) > 1 else ''}; after the fold each "
            f"execute re-runs its SFPU init every tile, so the pairs thrash the SFPU state and block hoisting, "
            f"and a hoisted or overlapping init corrupts the other op"
        )
    if also:
        summary += f"; also {', '.join(also[:4])}{' ...' if len(also) > 4 else ''}"
    for name in llk:
        if name and name not in ops:
            ops.append(name)
    ops = ops[:MAX_OPS]
    confidence = "medium"
    if agg.kind in ("dependent", "cross") and agg.has("known_values"):
        confidence = "high"
    if agg.has("low") and not agg.has("known_values"):
        confidence = "low"
    # ranking: a concrete clobber (dependent reader, cross-thread writers) above the replay /
    # ordering mechanisms that need a hoisting compiler or hardware timing to bite; the fold's
    # SDPA/attention targets get a smaller boost on top
    score = {"dependent": 6.5, "cross": 6.0, "replay_loop": 3.0, "sfpu_loop": 0.0}.get(
        agg.kind, 0.0
    )
    formats = agg.has("format_pair") and not agg.has("plain_pair")
    if formats:
        confidence = "low"
        score = 1.5
    if agg.attention_kernels:
        score += 2.0 if agg.kind in ("dependent", "cross") else 1.0
    elif any(kmod.ATTENTION_PATH.search(k) for k in agg.kernels):
        score += 0.5
    if agg.has("same_loop"):
        score += 0.5
    if agg.has("known_values"):
        score += 1.0
    if confidence == "low":
        score -= 3.0
    score += min(1.5, math.log10(1 + len(agg.kernels)))
    if agg.kind == "sfpu_loop":
        # same-thread thrash is correct at runtime: always below the other kinds
        score = min(score, 1.0) + min(0.5, math.log10(1 + len(agg.kernels)) / 4)
    return {
        "_score": score,
        # per-operand format/geometry state is re-derived from the CB by every reconfig: two ops
        # that leave different formats only conflict when their CBs differ in format, and a
        # pre-fold kernel that interleaved them already needed matching formats (no evidence)
        # same-thread SFPU re-programming by executes that each re-run their own init right
        # before their SFPU body is runtime-correct after the fold (thrash, not a hazard); an
        # execute that relies on another op's init is the dependent-reader case instead
        "_drop": formats or agg.kind == "sfpu_loop",
        "_debug": {k: sorted(v)[:6] for k, v in sorted(agg.values.items())},
        "confidence": confidence,
        "evidence": _cap_evidence(agg.evidence),
        "kernel_count": len(agg.kernels),
        "kernels": sorted(agg.kernels),
        "kind": "resource_conflict",
        "ops": ops,
        "resource": resource,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Rule (b): fold_drop
# --------------------------------------------------------------------------- #
_MARK = "__FOLD_P__"


@dataclass
class _Node:
    key: str
    name: str
    role: str
    params: dict  # param name -> bound text (kernel scope or marker scope)
    depth: int
    evidence: str
    entry: str = (
        ""  # call site in the walked root that leads to this node (depth-1 call)
    )


#: How deep below a folded execute (and below an explicit kernel setter) an init/config is
#: matched: ``mul_tiles -> mul_init -> binary_tiles_init`` against a kernel's ``mul_tiles_init ->
#: binary_tiles_init``.
FOLD_DEPTH = 3


class _Chain:
    """Static call chain of a function with substituted bindings (init/config callees)."""

    def __init__(self, db):
        self.db = db
        self._cache: dict[tuple, tuple] = {}

    def walk(
        self, key: str, bindings: dict, *, max_depth: int = 6, include_self: bool = True
    ) -> tuple:
        cache_key = (key, tuple(sorted(bindings.items())), max_depth, include_self)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        nodes: list[_Node] = []
        fn = self.db.function(key)
        if fn is not None:
            self._walk(
                fn, dict(bindings), 0, max_depth, nodes, include_self, set(), fn.loc, ""
            )
        result = tuple(nodes)
        self._cache[cache_key] = result
        return result

    def _walk(
        self,
        fn,
        bindings,
        depth,
        max_depth,
        nodes,
        include_self,
        stack,
        evidence,
        entry,
    ):
        if fn.key in stack or depth > max_depth:
            return
        if include_self or depth > 0:
            nodes.append(
                _Node(fn.key, fn.name, fn.role, dict(bindings), depth, evidence, entry)
            )
        stack = stack | {fn.key}
        frame = self.db.graph.frame_bindings(fn, bindings)
        for edge in fn.calls:
            if edge.stop and not edge.callee_keys:
                continue
            if edge.condition:
                verdict = truth(substitute(edge.condition, frame))
                if verdict is False:
                    continue
            targs = tuple(substitute(t, frame) for t in edge.template_args)
            args = tuple(substitute(a, frame) for a in edge.args)
            callees = [self.db.function(k) for k in edge.callee_keys]
            callees = [c for c in callees if c is not None]
            if edge.resolution == "param_callable":
                target = frame.get(edge.name)
                parsed = _parse_callable(target) if target else None
                if parsed:
                    name, qualifier, ctargs = parsed
                    try:
                        found, _ = self.db.graph.candidates(
                            name,
                            qualifier=qualifier,
                            template_args=ctargs,
                            args=args,
                            caller=fn,
                        )
                    except Exception:  # pragma: no cover - defensive
                        found = []
                    callees = found[:2]
                    targs = ctargs
            for callee in callees[:2]:
                if callee.role == "execute" and depth > 0:
                    continue
                try:
                    child = self.db.graph.bind(callee, targs, args)
                except Exception:  # pragma: no cover - defensive
                    continue
                self._walk(
                    callee,
                    child,
                    depth + 1,
                    max_depth,
                    nodes,
                    include_self,
                    stack,
                    edge.evidence,
                    entry or edge.evidence,
                )


def _parse_callable(text: str):
    from .model import _parse_callable as parse

    try:
        return parse(text)
    except Exception:  # pragma: no cover - defensive
        return None


def _explicit_params(fn, call: kmod.KCall) -> set[str]:
    names: set[str] = set()
    tparams = list(fn.template_params)
    for i, _ in enumerate(call.template_args):
        if i < len(tparams) and tparams[i].name:
            if tparams[i].variadic:
                names.update(p.name for p in tparams[i:] if p.name)
                break
            names.add(tparams[i].name)
    params = list(fn.params)
    for i, _ in enumerate(call.args):
        if i < len(params) and params[i].name:
            if params[i].variadic:
                names.update(p.name for p in params[i:] if p.name)
                break
            names.add(params[i].name)
    return names


def _is_folded(db, fn) -> bool:
    if fn is None or fn.role != "execute" or fn.layer != "compute_api":
        return False
    for edge in fn.calls:
        if setter_phase(db, fn.key, edge.evidence) == "body":
            continue
        for key in edge.callee_keys:
            callee = db.function(key)
            if (
                callee is not None
                and callee.role in ("init", "reconfig", "config", "uninit")
                and callee.layer in FOLD_LAYERS
            ):
                return True
    return False


_FORMAT_RECONFIG = re.compile(
    r"data_format|reconfig_df|full_operand|tile_shape|tile_dim"
)
_NAMESPACE = re.compile(r"(?<![\w:])(?:::)?(?:ckernel|sfpu|sfpi)::")
_DEFINE_ONLY = re.compile(r"^!?\(?[A-Z][A-Z0-9_]*\)?$")


def _norm_value(db, text: str):
    number = _int(db, text)
    if number is not None:
        return ("i", number)
    text = _NAMESPACE.sub("", compact(text))
    text = _NAMESPACE.sub("", text)
    return ("t", text)


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _comparable(db, a: str, b: str) -> bool:
    """False when a side is a bare compile-time define (APPROX, DST_ACCUM_MODE, ...), or the two
    texts differ only by such defines on one side (``f<X, MathFidelity::LoFi>`` vs
    ``f<X, MATH_FIDELITY>``): the values are unknown, not different."""
    for text in (a, b):
        if _int(db, text) is None and _DEFINE_ONLY.match(compact(text)):
            return False
    if _int(db, a) is None or _int(db, b) is None:
        ids_a, ids_b = set(_IDENT.findall(a)), set(_IDENT.findall(b))
        for only in (ids_a - ids_b, ids_b - ids_a):
            if only and all(
                _DEFINE_ONLY.match(name) and "_" in name or name.isupper()
                for name in only
            ):
                return False
    return True


@dataclass
class _Candidate:
    node_key: str
    node_name: str
    param: str
    origin: str  # F parameter name or "" (constant inside F)
    value: str  # kernel-scope value F uses
    sinks: tuple[str, ...]
    depth: int
    others: dict  # other fixed params of the node -> value
    canonical: str = (
        ""  # name of the shallowest fold parameter that drives the same sinks
    )


class _FoldModel:
    """Per folded execute: which fixed parameters reach hardware state."""

    def __init__(self, db, chain: _Chain):
        self.db = db
        self.chain = chain
        self._cache: dict[tuple, tuple] = {}
        self._writes: dict[tuple, frozenset] = {}

    def candidates(self, call: kmod.KCall) -> tuple:
        keys = call.keys
        if len(keys) != 1:
            return ()
        fn = self.db.function(keys[0])
        if not _is_folded(self.db, fn):
            return ()
        cache_key = (fn.key, call.template_args, len(call.args))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        explicit = _explicit_params(fn, call)
        kernel_bind = self.db.graph.bind(fn, call.template_args, call.args)
        scope_nodes = self.chain.walk(
            fn.key, {}, include_self=False, max_depth=FOLD_DEPTH
        )
        result: list[_Candidate] = []
        f_params = set(fn.param_names)
        # an init the fold reaches twice with different values for a parameter (tilize_init's
        # llk_pack_init<Tilize> and tilize_uninit's llk_pack_init<Default>) is set and reset by
        # the execute itself: no single value of it is "the fold's"
        seen_values: dict[tuple, set] = {}
        for node in self.chain.walk(
            fn.key, {}, include_self=False, max_depth=FOLD_DEPTH + 2
        ):
            for name, text in node.params.items():
                seen_values.setdefault((node.key, name), set()).add(
                    compact(substitute(text, kernel_bind))
                )
        for node in scope_nodes:
            if node.role not in SETTER_ROLES or node.depth > FOLD_DEPTH:
                continue  # inits/configs the folded call reaches
            if setter_phase(self.db, fn.key, node.entry) == "body":
                continue  # the op's own mid-sequence re-init, not the fold's prologue
            if _FORMAT_RECONFIG.search(node.name):
                continue  # per-operand format reconfig: every op sets it for its own CBs
            nfn = self.db.function(node.key)
            if nfn is None or nfn.layer not in (
                "compute_api",
                "llk_api",
                "llk_lib",
                "sfpu",
            ):
                continue
            fixed: dict[str, tuple[str, str]] = {}
            for param in nfn.all_params:
                name = param.name
                if not name or IGNORED_PARAMS.match(name) or name not in node.params:
                    continue
                scoped = node.params[name]
                if param.default and "__builtin" in param.default:
                    continue
                deps = identifiers(scoped) & f_params
                if deps & explicit:
                    continue
                value = compact(substitute(scoped, kernel_bind))
                if not value or has_unbound(value) or "__builtin" in value:
                    continue
                origin = sorted(deps)[0] if len(deps) == 1 else ""
                fixed[name] = (value, origin)
            for name, (value, origin) in sorted(fixed.items()):
                if len(seen_values.get((node.key, name), ())) > 1:
                    continue
                sinks = self._sinks(fn, call, kernel_bind, node, name, origin)
                if not sinks:
                    continue
                result.append(
                    _Candidate(
                        node_key=node.key,
                        node_name=node.name,
                        param=name,
                        origin=origin,
                        value=value,
                        sinks=sinks,
                        depth=node.depth,
                        others={k: v[0] for k, v in fixed.items() if k != name},
                    )
                )
        # one name per piece of state: a deeper init parameter fed by a shallower one (copy_init's
        # transpose_within_16x16_face -> llk_unpack_A_init's within_face_16x16_transpose) is
        # reported under the shallow name, whichever level the kernel's setter matched
        named: list[_Candidate] = []
        for cand in result:
            best = cand
            for other in result:
                if other.depth < best.depth and set(cand.sinks) <= set(other.sinks):
                    best = other
            named.append(replace(cand, canonical=best.origin or best.param))
        out = tuple(named)
        self._cache[cache_key] = out
        return out

    def setter_writes(self, node, cand) -> bool:
        """Whether the kernel setter, through its own binding of the shared init ``node``,
        writes at least one sink of the candidate parameter (or reaches an instruction sink).
        """
        if node is None:
            return False
        bound = {k: compact(v) for k, v in node.params.items()}
        key = (node.key, tuple(sorted(bound.items())))
        written = self._writes.get(key)
        if written is None:
            try:
                effects = self.db.graph.effects(node.key, bound)
            except Exception:  # pragma: no cover - defensive
                effects = ()
            written = frozenset(e.evidence for e in effects if e.access == "write")
            self._writes[key] = written
        sinks = set(cand.sinks)
        if sinks & written:
            return True
        # instruction-operand sinks (no write effect at that line): keep the candidate
        return not any(s in self._write_sinks(cand) for s in sinks)

    def _write_sinks(self, cand) -> frozenset:
        key = (cand.node_key,)
        cached = self._writes.get(key)
        if cached is None:
            try:
                effects = self.db.graph.effects(cand.node_key, {})
            except Exception:  # pragma: no cover - defensive
                effects = ()
            cached = frozenset(e.evidence for e in effects if e.access == "write")
            self._writes[key] = cached
        return cached

    def _sinks(self, fn, call, kernel_bind, node, name, origin) -> tuple[str, ...]:
        """Evidence that the parameter reaches hardware state below ``node``."""
        bound = {k: compact(substitute(v, kernel_bind)) for k, v in node.params.items()}
        marked = {name: _MARK}
        # the node's other bindings decide the branches the parameter can reach (acc_to_dest
        # feeds the ELWADD/ELWSUB MOP only; the ELWMUL MOP hard-codes 0 on Blackhole)
        feasible = {k: v for k, v in bound.items() if k != name and not has_unbound(v)}
        feasible[name] = _MARK
        sinks: set[str] = set()
        try:
            effects = self.db.graph.effects(node.key, feasible)
        except Exception:  # pragma: no cover - defensive
            effects = ()
        for effect in effects:
            if effect.access != "write" or effect.resource.startswith("param:"):
                continue
            in_value = (
                _MARK in (effect.value or "")
                or _MARK in (effect.resource_expr or "")
                or _MARK in (effect.detail or "")
            )
            in_condition = _MARK in (effect.condition or "")
            if not (in_value or in_condition):
                continue
            if effect.condition:
                actual = substitute(
                    (
                        (effect.condition or "").replace(
                            _MARK, f"({bound.get(name, '')})"
                        )
                        if not in_value
                        else (effect.condition or "")
                    ),
                    bound,
                )
                if _MARK not in actual and truth(actual) is False:
                    continue
            sinks.add(effect.evidence)
        if not sinks:
            # instruction operands: the parameter flows into a TTI_/TT_ macro below the node
            for sub in self.chain.walk(node.key, feasible, max_depth=5):
                sfn = self.db.function(sub.key)
                if sfn is None or sfn.definition is None:
                    continue
                flowing = {p for p, v in sub.params.items() if _MARK in (v or "")}
                if not flowing:
                    continue
                for site in sfn.definition.calls:
                    if not site.is_macro or not site.name.startswith("TT"):
                        continue
                    texts = " ".join(site.args + site.template_args)
                    if identifiers(texts) & flowing:
                        sinks.add(f"{sfn.path}:{site.line}")
        return tuple(sorted(sinks, key=_evidence_key))


def rule_fold_drop(ctx: Context) -> list[dict]:
    db = ctx.db
    chain = _Chain(db)
    fold = _FoldModel(db, chain)
    setter_cache: dict[tuple, dict] = {}

    def setter_nodes(call: kmod.KCall) -> dict:
        key = (call.keys, call.template_args, call.args)
        cached = setter_cache.get(key)
        if cached is not None:
            return cached
        nodes: dict[str, _Node] = {}
        for fkey in call.keys[:1]:
            fn = db.function(fkey)
            if fn is None:
                continue
            bind = db.graph.bind(fn, call.template_args, call.args)
            for node in chain.walk(fkey, bind, include_self=True, max_depth=FOLD_DEPTH):
                if node.key not in nodes or node.depth < nodes[node.key].depth:
                    nodes[node.key] = node
        setter_cache[key] = nodes
        return nodes

    aggs: dict[tuple, _Agg] = {}
    for kernel in ctx.kernels:
        calls = op_calls(kernel)
        for position, call in enumerate(calls):
            if call.role != "execute":
                continue
            candidates = fold.candidates(call)
            if not candidates:
                continue
            reported: set[tuple] = set()
            for cand in sorted(
                candidates, key=lambda c: (c.depth, c.node_key, c.param)
            ):
                setter = _last_setter(db, calls, position, cand, setter_nodes, fold)
                if setter is None:
                    continue
                prev, prev_value = setter
                if _norm_value(db, prev_value) == _norm_value(db, cand.value):
                    continue
                if not _comparable(db, prev_value, cand.value):
                    continue
                if not fold.setter_writes(setter_nodes(prev).get(cand.node_key), cand):
                    continue  # with its value the setter never wrote that state (a skip/gate flag)
                # a setter that differs from F in several other fixed params is a different op variant
                prev_node = setter_nodes(prev).get(cand.node_key)
                differing = 0
                if prev_node is not None:
                    for other, value in cand.others.items():
                        pv = prev_node.params.get(other)
                        if pv is not None and _norm_value(
                            db, compact(pv)
                        ) != _norm_value(db, value):
                            differing += 1
                # below the folded call itself the shared LLK init must be the same op variant
                limit = (
                    1
                    if cand.depth <= 1 and (prev_node is None or prev_node.depth <= 1)
                    else 0
                )
                if differing > limit:
                    continue
                dedupe = (
                    _norm_value(db, prev_value),
                    _norm_value(db, cand.value),
                    cand.canonical or cand.origin or cand.param,
                )
                sink_key = (frozenset(cand.sinks), dedupe[0], dedupe[1])
                if dedupe in reported or sink_key in reported:
                    continue
                reported.add(dedupe)
                reported.add(sink_key)
                param = cand.canonical or cand.origin or cand.param
                key = ("fold_drop", call.name, param)
                agg = aggs.get(key)
                if agg is None:
                    agg = aggs[key] = _Agg("fold_drop", call.name, (call.name, param))
                agg.kernels.add(kernel.path)
                if kernel.attention and (call.in_attention or prev.in_attention):
                    agg.attention_kernels.add(kernel.path)
                agg.values.setdefault("setter", set()).add(prev.name)
                agg.values.setdefault("node", set()).add(cand.node_name)
                agg.values.setdefault("kernel_value", set()).add(compact(prev_value))
                agg.values.setdefault("fold_value", set()).add(compact(cand.value))
                if prev.ifdefs:
                    agg.values.setdefault("defines", set()).update(
                        kmod._define_names(prev.ifdefs)
                    )
                if prev.role == "hw_startup":
                    agg.flag("hw_startup", kernel.path)
                if kmod.same_loop(prev, call):
                    agg.flag("same_loop", kernel.path)
                if (
                    _int(db, prev_value) is not None
                    and _int(db, cand.value) is not None
                ):
                    agg.flag("known_values", kernel.path)
                if len(agg.evidence) < 4 * MAX_EVIDENCE:
                    _add_evidence(agg, call.evidence, prev.evidence, *cand.sinks[:2])
                agg.details.update(cand.sinks[:2])
    findings = []
    for key in sorted(aggs):
        agg = aggs[key]
        api, param = agg.ops
        setters = sorted(agg.values.get("setter", set()))
        nodes = sorted(agg.values.get("node", set()))
        kv = sorted(
            {_clip(_display(db, v)) for v in agg.values.get("kernel_value", set())}
        )
        fv = sorted(
            {_clip(_display(db, v)) for v in agg.values.get("fold_value", set())}
        )
        if len(kv) > 4:
            kv = kv[:4] + [f"... ({len(kv) - 4} more)"]
        defines = sorted(agg.values.get("defines", set()))
        summary = (
            f"{api} now re-runs {', '.join(nodes)} with {param}={'|'.join(fv)}, overriding the "
            f"{'|'.join(kv)} the kernel configured earlier through {', '.join(setters)}"
            + (f" (under {', '.join(defines[:4])})" if defines else "")
            + ", so the folded execute silently drops the caller's configuration"
        )
        ops = [api] + [n for n in nodes if n != api] + [s for s in setters if s != api]
        score = 6.0 + (2.0 if agg.attention_kernels else 0.0)
        if agg.has("same_loop"):
            score += 0.5
        if agg.has("known_values"):
            score += 1.5
        score += min(1.5, math.log10(1 + len(agg.kernels)))
        confidence = "high" if agg.has("known_values") else "medium"
        findings.append(
            {
                "_score": score,
                "confidence": confidence,
                "evidence": _cap_evidence(
                    agg.evidence, sorted(agg.details, key=_evidence_key)[:4]
                ),
                "kernel_count": len(agg.kernels),
                "kernels": sorted(agg.kernels),
                "kind": "fold_drop",
                "ops": ops[:MAX_OPS],
                "resource": f"param:{api}.{param}",
                "summary": summary,
            }
        )
    return findings


_STEM = re.compile(r"_(?:re)?(?:init|uninit|config|configure|reconfig)(?:_\w*)?$")


def _setter_stem(name: str) -> str:
    """``pack_untilize_init`` -> ``pack_untilize``: the op a compute-API setter is named after."""
    stem = _STEM.sub("", name)
    return stem if stem != name and len(stem) >= 4 else ""


def _clip(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _display(db, text: str) -> str:
    number = _int(db, text)
    if number is None:
        return compact(text)
    return f"0x{number:X}" if number > 255 else str(number)


def _last_setter(db, calls, position, cand, setter_nodes, fold):
    """Most recent config/init call before ``calls[position]`` that sets the candidate.

    Returns (call, value) or None. A more recent execute that sets the same
    parameter itself (another folded call) ends the search: the state it leaves
    is its own, not the kernel's configuration.
    """
    target = calls[position]
    order = list(range(position - 1, -1, -1))
    # loop back edge: calls after the execute inside a loop it shares run before its next iteration
    tail = [
        i
        for i in range(len(calls) - 1, position, -1)
        if kmod.same_loop(calls[i], target)
    ]
    passed: list[str] = []  # executes between the candidate setter and the target
    for index in order + tail:
        prev = calls[index]
        if _exclusive(prev, target):
            continue
        if prev.role == "execute":
            passed.append(prev.name)
            if prev.name == target.name or any(
                c.node_key == cand.node_key and c.param == cand.param
                for c in fold.candidates(prev)
            ):
                return None
            # another folded execute that re-runs the same LLK init (at any depth of its own
            # folded chain, e.g. copy_tile -> copy_init -> llk_math_eltwise_unary_datacopy_init)
            # already replaced the state: an explicit setter before it configured that op
            reached = setter_nodes(prev).get(cand.node_key)
            if (
                reached is not None
                and reached.depth >= 1
                and cand.param in reached.params
                and _is_folded(
                    db, db.function(prev.keys[0]) if len(prev.keys) == 1 else None
                )
            ):
                return None
            continue
        if prev.role not in SETTER_ROLES:
            continue
        node = setter_nodes(prev).get(cand.node_key)
        if node is None or node.depth > FOLD_DEPTH or cand.param not in node.params:
            continue
        stem = _setter_stem(prev.name)
        if stem and any(
            name != target.name and name.startswith(stem) for name in passed
        ):
            return None  # the setter configured its own op (X_init -> X_...), run in between
        return prev, compact(node.params[cand.param])
    return None


# --------------------------------------------------------------------------- #
# Rule (c): ordering
# --------------------------------------------------------------------------- #
_SYNC_CALLS = re.compile(r"^(?:sync_regfile_write|tensix_sync|mop_sync)$")
_STALL = re.compile(r"^TTI?_STALLWAIT$|^__builtin_rvtt_stallwait\w*$")
_CONSUMERS = re.compile(r"^TT(?:I)?_(?:UNPACR\w*|PACR\w*|MOP|CFGSHIFTMASK|REPLAY)$")
_RISC_GPR = ("core.regfile_assign",)
_RISC_CFG = ("core.cfg_assign", "core.cfg_store", "core.cfg_write")
_TENSIX_GPR_READ = re.compile(
    r"^core\.(?:wrcfg|rdcfg|dmareg_alu|setdmareg|cfg_rmw_gpr)"
)
_CFG_CONSUMED = re.compile(
    r"^CFG\.(?:THCON_SEC[01]_REG[0-9]_|UNP[01]_|SCRATCH_SEC|PCK|PACK_|TILE_ROW_SET)"
)


def _stream(db, fn):
    """Ordered (line, rank, kind, data) items of one function body."""
    items: list[tuple] = []
    definition = fn.definition
    if definition is None:
        return items
    for effect in fn.effects:
        line = _evidence_key(effect.evidence)[1]
        if effect.pattern in _RISC_GPR and effect.access == "write":
            items.append((line, 1, "gpr_store", effect))
        elif effect.pattern in _RISC_CFG and effect.access == "write":
            items.append((line, 1, "cfg_store", effect))
        elif (
            effect.access == "read"
            and effect.resource.startswith("GPR.")
            and _TENSIX_GPR_READ.match(effect.pattern or "")
        ):
            items.append((line, 2, "gpr_read", effect))
    for site in definition.calls:
        if _SYNC_CALLS.match(site.name):
            items.append((site.line, 0, "sync_all", site))
        elif _STALL.match(site.name) and any("TRISC_CFG" in a for a in site.args):
            items.append((site.line, 0, "sync_cfg", site))
        elif _CONSUMERS.match(site.name):
            items.append((site.line, 3, "consumer", site))
        elif site.name in ("run", "program_and_run") and site.is_member:
            items.append((site.line, 3, "consumer", site))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return items


def rule_ordering(ctx: Context) -> list[dict]:
    db = ctx.db
    findings: list[dict] = []
    facts = _family_order_facts(db)
    if facts is not None:
        findings.extend(_fact_findings(ctx, facts))
    else:
        findings.extend(_stream_ordering(ctx))
    findings.extend(_loadmacro_findings(ctx))
    return findings


def _family_order_facts(db):
    """``compute_api.ordering_facts`` rows when that family is loaded, else None."""
    if not any(
        getattr(f, "name", "") == "compute_api" for f in getattr(db, "families", ())
    ):
        return None
    try:
        from .families import compute_api
    except Exception:  # pragma: no cover - family missing
        return None
    helper = getattr(compute_api, "ordering_facts", None)
    if helper is None:
        return None
    try:
        return helper(db)
    except Exception:  # pragma: no cover - defensive
        return None


class _FunctionIndex:
    def __init__(self, db):
        self.by_path: dict[str, list] = {}
        for fn in db.functions.values():
            self.by_path.setdefault(fn.path, []).append(fn)
        for fns in self.by_path.values():
            fns.sort(key=lambda f: (f.line, f.end_line, f.key))

    def at(self, evidence: str):
        path, _, line = evidence.rpartition(":")
        if not line.isdigit():
            return None
        number = int(line)
        best = None
        for fn in self.by_path.get(path, ()):
            if fn.line <= number <= fn.end_line and (
                best is None or fn.end_line - fn.line < best.end_line - best.line
            ):
                best = fn
        return best


def _fact_findings(ctx: Context, facts: list[dict]) -> list[dict]:
    db = ctx.db
    index = _FunctionIndex(db)
    grouped: dict[tuple, list[dict]] = {}
    for row in facts:
        if row.get("verdict") not in ("unordered", "conditional"):
            continue
        grouped.setdefault((row["key"], row["use"], row["verdict"]), []).append(row)
    findings = []
    for (key, use, verdict), rows in sorted(grouped.items()):
        fn = db.function(key)
        if fn is None:
            continue
        kernels, attention = _kernels_reaching(ctx, key)
        if not kernels:
            continue
        rows.sort(key=lambda r: (_evidence_key(r["store"]), r["resource"]))
        resources_ = []
        for row in rows:
            if row["resource"] not in resources_:
                resources_.append(row["resource"])
        store_fns, use_fn = [], index.at(use)
        for row in rows:
            holder = index.at(row["store"])
            if holder is not None and holder.name not in store_fns:
                store_fns.append(holder.name)
        ops = [fn.name] + [n for n in store_fns if n != fn.name]
        if use_fn is not None and use_fn.name not in ops:
            ops.append(use_fn.name)
        evidence = (
            [row["store"] for row in rows]
            + [use]
            + [row["at"] for row in rows if row.get("at")]
        )
        evidence += [row["fence"] for row in rows if row.get("fence")]
        resolved = [r for r in resources_ if "?" not in r]
        resource = resolved[0] if resolved else resources_[0]
        stores = ", ".join(resources_[:3]) + (" ..." if len(resources_) > 3 else "")
        condition = rows[0].get("condition") or ""
        summary = (
            f"{fn.name} reaches RISC-side config stores of {stores} ({', '.join(store_fns[:2]) or 'MMIO'}) that are "
            f"consumed by the Tensix instruction at {_short(use)} with "
            f"{'only a conditional' if verdict == 'conditional' else 'no'} STALLWAIT(TRISC_CFG)/tensix_sync fence in "
            f"between, so with intrinsic-issued instructions the consumer can run on the old value"
            + (f" (when {condition[:80]})" if condition else "")
        )
        score = (3.0 if verdict == "unordered" else 1.5) + (1.0 if attention else 0.0)
        score += min(1.5, math.log10(1 + len(kernels)))
        findings.append(
            {
                "_score": score,
                "confidence": (
                    "high" if verdict == "unordered" and not condition else "medium"
                ),
                "evidence": _cap_evidence(evidence, [use]),
                "kernel_count": len(kernels),
                "kernels": sorted(kernels),
                "kind": "ordering",
                "ops": ops[:MAX_OPS],
                "resource": resource,
                "summary": summary,
            }
        )
    return findings


def _stream_ordering(ctx: Context) -> list[dict]:
    """Fallback when the compute_api family is not loaded: body-local store/use scan."""
    db = ctx.db
    raw: list[tuple] = (
        []
    )  # (kind, resource, function key, store evidence, consumer evidence, detail)
    for key in db.keys():
        fn = db.function(key)
        if fn is None or fn.layer not in scan.LAYERS:
            continue
        items = _stream(db, fn)
        if not items:
            continue
        pending_gpr: dict[str, object] = {}
        pending_cfg: dict[str, object] = {}
        for line, _, kind, data in items:
            if kind in ("sync_all", "sync_cfg"):
                pending_gpr.clear()
                pending_cfg.clear()
            elif kind == "gpr_store":
                pending_gpr[data.resource] = data
            elif kind == "cfg_store":
                if _CFG_CONSUMED.match(data.resource):
                    pending_cfg[data.resource] = data
            elif kind == "gpr_read":
                store = pending_gpr.pop(data.resource, None)
                if store is not None:
                    raw.append(
                        ("gpr", data.resource, key, store.evidence, data.evidence, "")
                    )
            elif kind == "consumer":
                for resource, store in sorted(pending_cfg.items()):
                    raw.append(
                        (
                            "cfg",
                            resource,
                            key,
                            store.evidence,
                            f"{fn.path}:{line}",
                            data.name,
                        )
                    )
                pending_cfg.clear()
    return _ordering_findings(ctx, raw) if raw else []


def _kernels_reaching(ctx: Context, key: str) -> tuple[set, set]:
    kernels: set[str] = set()
    attention: set[str] = set()
    for kernel in ctx.kernels:
        for call in kernel.calls:
            if any(key in ctx.reach(k) for k in call.keys):
                kernels.add(kernel.path)
                if kernel.attention and call.in_attention:
                    attention.add(kernel.path)
                break
    return kernels, attention


def _ordering_findings(ctx: Context, raw: list[tuple]) -> list[dict]:
    db = ctx.db
    grouped: dict[tuple, list] = {}
    for kind, resource, key, store, consumer, detail in raw:
        grouped.setdefault((kind, key), []).append((resource, store, consumer, detail))
    findings = []
    for (kind, key), rows in sorted(grouped.items()):
        fn = db.function(key)
        kernels, attention = _kernels_reaching(ctx, key)
        if not kernels:
            continue
        rows.sort()
        resource = rows[0][0]
        evidence = []
        for _, store, consumer, _ in rows[:6]:
            evidence.extend([store, consumer])
        if kind == "gpr":
            summary = (
                f"{fn.name} stores {resource} from the RISC (regfile[]) and a later Tensix instruction reads it "
                f"with no sync_regfile_write/tensix_sync/STALLWAIT(TRISC_CFG) in between, so the instruction can "
                f"see the stale GPR once intrinsics let the compiler reorder or hoist the store"
            )
        else:
            consumers = sorted({row[3] for row in rows})
            summary = (
                f"{fn.name} writes {', '.join(sorted({r[0] for r in rows})[:3])} through a RISC-side config store "
                f"and then issues {', '.join(consumers)} with no STALLWAIT(.., TRISC_CFG) in between, so the "
                f"consumer can run before the store lands (or the compiler can sink the store past it)"
            )
        score = (
            2.0 + (1.0 if attention else 0.0) + min(1.5, math.log10(1 + len(kernels)))
        )
        findings.append(
            {
                "_score": score,
                "confidence": "medium",
                "evidence": _cap_evidence(evidence),
                "kernel_count": len(kernels),
                "kernels": sorted(kernels),
                "kind": "ordering",
                "ops": [fn.name],
                "resource": resource,
                "summary": summary,
            }
        )
    return findings


_LOADMACRO_LINE = re.compile(r"\bTTI?_SFPLOADMACRO\b")
_SFPNOP_LINE = re.compile(r"\bTTI_SFPNOP\b")
_WAIT_SFPU = re.compile(r"WAIT_SFPU|STALL_SFPU")


def _pending_macro(ctx: Context, evidence: str, cache: dict) -> bool | None:
    """True when the LOADMACRO at ``evidence`` is the last of its function with < 3 SFPNOP after it."""
    if evidence in cache:
        return cache[evidence]
    path, _, line = evidence.rpartition(":")
    result = None
    fns = [
        fn
        for fn in ctx.db.functions.values()
        if fn.path == path
        and fn.line <= int(line or 0) <= fn.end_line
        and fn.definition is not None
    ]
    if fns and line.isdigit():
        fn = sorted(fns, key=lambda f: f.end_line - f.line)[0]
        definition = fn.definition
        masked = definition.masked_body
        lines = masked.split("\n")
        rel = int(line) - definition.body_line
        tail = "\n".join(lines[rel + 1 :]) if 0 <= rel < len(lines) else ""
        if _LOADMACRO_LINE.search(tail):
            result = False  # not the last macro of the body
        else:
            nops = len(_SFPNOP_LINE.findall(tail))
            result = nops < 3 and not _WAIT_SFPU.search(tail)
    cache[evidence] = result
    return result


def _drains_sfpu(ctx: Context, call: kmod.KCall, cache: dict) -> bool:
    key = call.keys
    if key in cache:
        return cache[key]
    result = False
    for k in call.keys:
        for reached in ctx.reach(k):
            fn = ctx.db.function(reached)
            if fn is None or fn.definition is None:
                continue
            for site in fn.definition.calls:
                if (
                    _STALL.match(site.name) or site.name.startswith("t6_semaphore")
                ) and any("WAIT_SFPU" in a for a in site.args + site.template_args):
                    result = True
                    break
            if result:
                break
        if result:
            break
    cache[key] = result
    return result


def _loadmacro_findings(ctx: Context) -> list[dict]:
    db = ctx.db
    pending_cache: dict = {}
    drain_cache: dict = {}
    aggs: dict[tuple, _Agg] = {}
    for kernel in ctx.kernels:
        calls = op_calls(kernel)
        for position, call in enumerate(calls):
            macro_reads = [
                e
                for e in ctx.effects(call)
                if e.access == "read"
                and e.resource.startswith("SFPU.LOADMACRO")
                and (
                    e.pattern == "core.sfploadmacro"
                    or (e.pattern or "").startswith(("sfpu.loadmacro", "sfpu.replay"))
                )
                and not is_deferred(e.detail)
            ]
            if not macro_reads:
                continue
            last = sorted(macro_reads, key=lambda e: _evidence_key(e.evidence))[-1]
            # the macro must be issued by the execute path, not by an init
            if any(
                (db.function(step[0]) or None) is not None
                and db.function(step[0]).role == "init"
                for step in last.via[1:]
            ):
                continue
            if not _pending_macro(ctx, last.evidence, pending_cache):
                continue
            used = {
                e.resource
                for e in ctx.effects(call)
                if e.access == "read" and SFPU_MACRO_STATE.search(e.resource)
            }
            used |= {
                "SFPU.LOADMACRO.SEQ0",
                "SFPU.LOADMACRO.SEQ1",
                "SFPU.LOADMACRO.CTRL",
            } & {e.resource for e in ctx.effects(call) if e.access == "write"}
            # next SFPU-state writer on the same thread before any SFPU drain
            following = list(range(position + 1, len(calls)))
            following += [
                i for i in range(0, position + 1) if kmod.same_loop(calls[i], call)
            ]
            for index in following:
                nxt = calls[index]
                if _exclusive(nxt, call):
                    continue
                if _drains_sfpu(ctx, nxt, drain_cache) and not _writes_sfpu_first(
                    ctx, nxt
                ):
                    break
                if nxt.name == call.name:
                    break  # the op's own re-init rewrites the same state
                own = {}
                for e in ctx.effects(call):
                    if e.access == "write" and SFPU_MACRO_STATE.search(e.resource):
                        own.setdefault(e.resource, []).append(e)
                touching = [
                    e
                    for e in ctx.effects(nxt)
                    if e.access == "write"
                    and SFPU_MACRO_STATE.search(e.resource)
                    and resources.same_thread(e.thread, last.thread)
                    and e.resource in used
                    and not is_deferred(e.detail)
                ]
                if touching and nxt.role != "execute":
                    break  # an explicit kernel init rewrites it first: the pre-fold contract
                writes = [
                    e
                    for e in touching
                    if fold_path(db, _Access(nxt, e))
                    and all(
                        values_differ(db, e, mine) for mine in own.get(e.resource, ())
                    )
                ]
                if not writes:
                    continue
                if index <= position and not kmod.same_loop(nxt, call):
                    break
                write = sorted(
                    writes, key=lambda e: (e.resource not in used, e.resource)
                )[0]
                key = ("ordering", "loadmacro", call.name, nxt.name)
                agg = aggs.get(key)
                if agg is None:
                    agg = aggs[key] = _Agg(
                        "loadmacro", write.resource, (call.name, nxt.name)
                    )
                agg.kernels.add(kernel.path)
                agg.resources[write.resource] = agg.resources.get(write.resource, 0) + 1
                if kernel.attention and (call.in_attention or nxt.in_attention):
                    agg.attention_kernels.add(kernel.path)
                if kmod.same_loop(nxt, call):
                    agg.flag("same_loop", kernel.path)
                agg.llk.setdefault("macro", set()).add(_sink_function(db, last))
                agg.llk.setdefault("writer", set()).add(_sink_function(db, write))
                if len(agg.evidence) < 4 * MAX_EVIDENCE:
                    _add_evidence(
                        agg, last.evidence, write.evidence, call.evidence, nxt.evidence
                    )
                break
    findings = []
    for key in sorted(aggs):
        agg = aggs[key]
        op_a, op_b = agg.ops
        resource = _representative(agg)
        same = op_a == op_b
        summary = (
            f"{op_a} leaves SFPLOADMACRO sequences in flight (fewer than 3 SFPNOP and no WAIT_SFPU after the "
            f"last LOADMACRO) and {'its own next call' if same else op_b}{' in the same loop' if agg.has('same_loop') else ''} "
            f"re-runs an init that rewrites {resource} before any SFPU drain, so pending macros can execute with "
            f"the overwritten state"
        )
        ops = [op_a] + ([op_b] if not same else [])
        for name in sorted(agg.llk.get("macro", set())) + sorted(
            agg.llk.get("writer", set())
        ):
            if name and name not in ops:
                ops.append(name)
        score = (
            3.0
            + (1.0 if agg.attention_kernels else 0.0)
            + (0.5 if agg.has("same_loop") else 0.0)
        )
        score += min(1.5, math.log10(1 + len(agg.kernels)))
        findings.append(
            {
                "_score": score,
                "confidence": "medium" if not same else "low",
                "evidence": _cap_evidence(agg.evidence),
                "kernel_count": len(agg.kernels),
                "kernels": sorted(agg.kernels),
                "kind": "ordering",
                "ops": ops[:MAX_OPS],
                "resource": resource,
                "summary": summary,
            }
        )
    return findings


def _writes_sfpu_first(ctx: Context, call: kmod.KCall) -> bool:
    """Whether the call's first SFPU-state write precedes its SFPU drain (init before execute)."""
    for effect in ctx.effects(call):
        if effect.access == "write" and SFPU_MACRO_STATE.search(effect.resource):
            fn = ctx.db.function(effect.via[1][0]) if len(effect.via) > 1 else None
            if fn is not None and fn.role == "init":
                return True
    return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def merge_duplicates(findings: list[dict]) -> list[dict]:
    """One finding per (kind, resource, ops): rows from overloads / variants of one API
    (e.g. two ``custom_mm_block_init`` keys) are merged, keeping the best-scored text and
    the union of kernels and evidence."""
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for finding in findings:
        key = (finding["kind"], finding["resource"], tuple(finding["ops"]))
        current = merged.get(key)
        if current is None:
            merged[key] = dict(finding)
            order.append(key)
            continue
        best, other = (
            (finding, current)
            if (round(finding.get("_score", 0.0), 6), finding["summary"])
            > (round(current.get("_score", 0.0), 6), current["summary"])
            else (current, finding)
        )
        combined = dict(best)
        kernels = sorted(set(best["kernels"]) | set(other["kernels"]))
        combined["kernels"] = kernels
        combined["kernel_count"] = max(
            len(kernels), best["kernel_count"], other["kernel_count"]
        )
        combined["evidence"] = _cap_evidence(
            set(best["evidence"]) | set(other["evidence"]), best["evidence"]
        )
        order_rank = {"high": 0, "medium": 1, "low": 2}
        combined["confidence"] = min(
            (best["confidence"], other["confidence"]),
            key=lambda c: order_rank.get(c, 3),
        )
        merged[key] = combined
    return [merged[key] for key in order]


def rank(findings: list[dict]) -> list[dict]:
    """Deterministic total order: score desc, kernel_count desc, kind, resource, ops."""
    ordered = sorted(
        findings,
        key=lambda f: (
            -round(f.get("_score", 0.0), 6),
            -f.get("kernel_count", 0),
            f["kind"],
            f["resource"],
            f["ops"],
            f["summary"],
        ),
    )
    out = []
    for position, finding in enumerate(ordered, 1):
        item = {k: v for k, v in finding.items() if k != "_score"}
        item["rank"] = position
        out.append(item)
    return out


def run(
    db, repo_root: Path | str, *, kernel_paths: Sequence[str] | None = None
) -> dict:
    """All findings for ``repo_root`` (see module docstring)."""
    kernels = kmod.extract(db, repo_root, paths=kernel_paths)
    ctx = Context(db, kernels)
    findings: list[dict] = []
    findings.extend(rule_conflicts(ctx))
    findings.extend(rule_fold_drop(ctx))
    findings.extend(rule_ordering(ctx))
    # a finding must name a canonical resource: an unresolved name (``CFG.?expr``) is a
    # modelling gap, not a hazard
    findings = [f for f in findings if "?" not in f["resource"]]
    findings = merge_duplicates(findings)
    fingerprint = db.tree_fingerprint
    if kernels:  # fold the analysed kernel sources into the library fingerprint
        digest = hashlib.sha256()
        digest.update(db.tree_fingerprint.encode("ascii"))
        digest.update(b"\0" + kmod.fingerprint(kernels).encode("ascii"))
        fingerprint = digest.hexdigest()
    return {"findings": rank(findings), "tree_fingerprint": fingerprint}


__all__ = [
    "Context",
    "TRANSIENT_RESOURCES",
    "is_transient",
    "rank",
    "rule_conflicts",
    "rule_fold_drop",
    "rule_ordering",
    "run",
    "value_identity",
    "values_differ",
]
