# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scanner of every Blackhole compute kernel in a tt-metal tree.

For each compute kernel the scanner produces the ordered sequence of
compute-API / LLK calls it makes, with:

* the call's template and runtime argument expressions, re-expressed in kernel
  scope (helper parameters substituted by the helper call's arguments, and
  ``constexpr`` locals substituted by their initializers);
* the loop nesting of every call (``for``/``while``/``do`` bodies, including
  loops inside inlined helpers), as loop *instances* so two calls in the same
  loop of the same helper invocation can be told apart from two invocations;
* the ``#if``/``#ifdef`` context (undecidable preprocessor conditions, which keep
  the define names) and the runtime ``if``/``else`` context; ``if constexpr``
  branches that fold to false under the substituted bindings are dropped;
* the thread (``UNPACK((..))``/``MATH((..))``/``PACK((..))`` wrapper or
  ``#ifdef TRISC_*`` region), inherited through helpers;
* the chain of kernel-local helpers the call was reached through.

Kernel-local helpers (``compute_common.hpp``, ``compute_streaming.hpp``,
``kernel_lib/*.inl``, ...) are found by resolving the kernel's local
``#include`` directives and are inlined, so a kernel's call sequence includes
its helpers' calls. Library functions (compute API, llk_api, tt-llk) are not
inlined; they are looked up in the :class:`~.effects_db.EffectDB` and recorded
as calls.

Roots (repo-relative), see :data:`KERNEL_ROOTS`:

* ``ttnn/**/kernels*/**`` and ``ttnn/**/kernel/**`` (compute kernels only),
* ``ttnn/cpp/ttnn/kernel_lib/**``,
* ``models/**/kernels*/**`` and ``models/**/unified_kernels/**``,
* ``tests/tt_metal/tt_metal/test_kernels/compute/**``.

Quasar-only kernel directories (``/quasar/``, ``*_qsr``) are skipped: this
scanner models the Blackhole build.

The scanner needs no git and no compiler. Output order is deterministic
(kernels by path, calls in program order).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import scan
from .model import compact, substitute

SOURCE_SUFFIXES = (".cpp", ".cc", ".hpp", ".h", ".inl", ".hh", ".inc")
KERNEL_ROOTS = (
    "ttnn",
    "models",
    "tests/tt_metal/tt_metal/test_kernels/compute",
)
KERNEL_LIB = "ttnn/cpp/ttnn/kernel_lib/"
LOCAL_PREFIXES = ("ttnn/", "models/", "tests/")
ENTRY_NAMES = ("kernel_main", "MAIN")
MAX_DEPTH = 16
MAX_ARG_TEXT = 400

#: Include texts that mark a translation unit as a compute kernel.
_COMPUTE_INCLUDE = re.compile(
    r"(?:^|/)(?:api/compute/|compute_kernel_api|kernel_lib/(?:tilize|untilize|reduce|dest|dfb)_helpers|"
    r"llk_math_|llk_unpack_|llk_pack_|ckernel\.h|kernel_lib/eltwise)"
)
#: Path fragments of the SDPA / attention kernels and helpers the fold targeted.
ATTENTION_PATH = re.compile(
    r"(?i)(?:sdpa|attention|attn|flash_mla|flash_decode|/mla/|softmax_attn)"
)
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">]+)[">]', re.M)
_LOOP_KEYWORD = re.compile(r"(?<![\w.>:$])(for|while|do)\b")
_OBJECT_MACRO = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(?![\w(])[ \t]*(.*)$", re.M
)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Loop:
    """One loop instance enclosing a call."""

    id: str  # repo-relative path:line of the loop keyword
    instance: str  # id prefixed by the helper call chain that reached it
    kind: str  # for | while | do
    header: str  # loop header text (capped)


@dataclass(frozen=True)
class KCall:
    """One library call made by a kernel, in kernel scope."""

    index: int  # program order within the kernel
    name: str
    callee: str  # as written
    qualifier: str
    template_args: tuple[str, ...]  # kernel scope (helper params substituted)
    args: tuple[str, ...]
    raw_template_args: tuple[str, ...]  # as written at the call site
    raw_args: tuple[str, ...]
    path: str
    line: int
    thread: str | None  # unpack | math | pack | None (all threads)
    condition: str  # runtime / constexpr if-else conditions (kernel scope), "" if none
    ifdefs: tuple[str, ...]  # undecidable #if conditions (outermost first)
    loops: tuple[Loop, ...]  # outermost first
    stack: tuple[
        tuple[str, str], ...
    ]  # (helper name, call evidence) from the entry down
    via_macro: str | None
    is_member: bool
    receiver_type: str | None
    keys: tuple[str, ...]  # resolved EffectDB keys ("" entries never appear)
    resolution: str  # exact | overload | ambiguous | pattern
    role: str  # role of the resolved function, "primitive" for pattern-only calls
    layer: str  # layer of the resolved function ("" for primitives)
    in_attention: bool  # reached through an SDPA/attention source

    @property
    def evidence(self) -> str:
        return f"{self.path}:{self.line}"

    @property
    def loop_instances(self) -> tuple[str, ...]:
        return tuple(loop.instance for loop in self.loops)

    def to_json(self) -> dict:
        return {
            "args": list(self.args),
            "condition": self.condition,
            "evidence": self.evidence,
            "ifdefs": list(self.ifdefs),
            "in_attention": self.in_attention,
            "index": self.index,
            "keys": list(self.keys),
            "layer": self.layer,
            "loops": [loop.id for loop in self.loops],
            "name": self.name,
            "raw_args": list(self.raw_args),
            "raw_template_args": list(self.raw_template_args),
            "resolution": self.resolution,
            "role": self.role,
            "stack": [list(step) for step in self.stack],
            "template_args": list(self.template_args),
            "thread": self.thread,
            "via_macro": self.via_macro,
        }


@dataclass
class Kernel:
    """One compute kernel translation unit and its call sequence."""

    path: str
    entry: str
    sources: tuple[str, ...]  # the kernel plus its resolved local includes, sorted
    calls: tuple[KCall, ...]
    config: tuple[tuple[str, str], ...]  # entry constexpr locals + object-like #defines
    defines: tuple[str, ...]  # define names tested by undecidable #if around calls
    attention: bool
    fingerprint: str
    extra_roots: tuple[str, ...] = ()  # local functions analysed as extra roots

    def to_json(self) -> dict:
        return {
            "attention": self.attention,
            "calls": [call.to_json() for call in self.calls],
            "config": [list(item) for item in self.config],
            "defines": list(self.defines),
            "entry": self.entry,
            "extra_roots": list(self.extra_roots),
            "fingerprint": self.fingerprint,
            "path": self.path,
            "sources": list(self.sources),
        }


# --------------------------------------------------------------------------- #
# Discovery and include resolution
# --------------------------------------------------------------------------- #
def _skip_dir(rel_dir: str) -> bool:
    parts = rel_dir.split("/")
    return any(
        part == "quasar"
        or part.endswith("_qsr")
        or part.startswith(".")
        or part in ("__pycache__", "build", "built")
        for part in parts
    )


def in_kernel_roots(rel: str) -> bool:
    """Whether a repo-relative file path lies in one of the kernel roots."""
    if not rel.endswith(SOURCE_SUFFIXES):
        return False
    parts = rel.split("/")
    directories = parts[:-1]
    if _skip_dir("/".join(directories)):
        return False
    if rel.startswith("tests/tt_metal/tt_metal/test_kernels/compute/"):
        return True
    if rel.startswith(KERNEL_LIB):
        return True
    if rel.startswith("ttnn/"):
        return any(p.startswith("kernels") or p == "kernel" for p in directories)
    if rel.startswith("models/"):
        return any(
            p.startswith("kernels") or p == "unified_kernels" for p in directories
        )
    return False


def discover(repo_root: Path | str) -> list[str]:
    """Sorted repo-relative source paths under the kernel roots."""
    root = Path(repo_root)
    found: set[str] = set()
    for base in KERNEL_ROOTS:
        top = root / base
        if not top.is_dir():
            continue
        for directory, subdirs, files in os.walk(top):
            rel_dir = Path(directory).relative_to(root).as_posix()
            subdirs[:] = sorted(d for d in subdirs if not _skip_dir(f"{rel_dir}/{d}"))
            for name in files:
                rel = f"{rel_dir}/{name}"
                if in_kernel_roots(rel):
                    found.add(rel)
    return sorted(found)


def _normalize_rel(path: str) -> str | None:
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


class SourceCache:
    """Reads, masks and scans kernel-side sources once each."""

    def __init__(self, repo_root: Path | str, index: scan.ScanIndex | None):
        self.root = Path(repo_root)
        self.index = index
        self._text: dict[str, str | None] = {}
        self._includes: dict[str, tuple[tuple[str, str | None], ...]] = {}
        self._closure: dict[str, tuple[str, ...]] = {}
        self._scanned: dict[str, scan.SourceFile] = {}
        self._exists: dict[str, bool] = {}

    def exists(self, rel: str) -> bool:
        cached = self._exists.get(rel)
        if cached is None:
            cached = (self.root / rel).is_file()
            self._exists[rel] = cached
        return cached

    def text(self, rel: str) -> str | None:
        if rel not in self._text:
            path = self.root / rel
            try:
                self._text[rel] = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                self._text[rel] = None
        return self._text[rel]

    def resolve_include(self, includer: str, target: str) -> str | None:
        """Repo-relative path of a *local* include (ttnn/, models/, tests/), else None."""
        directory = includer.rsplit("/", 1)[0] if "/" in includer else ""
        candidates = [
            f"{directory}/{target}" if directory else target,
            target,
            f"ttnn/{target}",
            f"ttnn/cpp/{target}",
            f"ttnn/cpp/ttnn/{target}",
        ]
        for candidate in candidates:
            rel = _normalize_rel(candidate)
            if rel is None or not rel.startswith(LOCAL_PREFIXES):
                continue
            if self.exists(rel):
                return rel
        return None

    def includes(self, rel: str) -> tuple[tuple[str, str | None], ...]:
        """(include text, resolved local path or None) in file order."""
        cached = self._includes.get(rel)
        if cached is not None:
            return cached
        masked = _mask_comments(self.text(rel) or "")
        result = tuple(
            (match.group(1), self.resolve_include(rel, match.group(1)))
            for match in _INCLUDE.finditer(masked)
        )
        self._includes[rel] = result
        return result

    def closure(self, rel: str) -> tuple[str, ...]:
        """``rel`` plus every transitively included local file (sorted, deterministic)."""
        cached = self._closure.get(rel)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [rel]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for _, resolved in self.includes(current):
                if resolved and resolved not in seen:
                    stack.append(resolved)
        result = tuple(sorted(seen))
        self._closure[rel] = result
        return result

    def include_texts(self, rels: Iterable[str]) -> set[str]:
        texts: set[str] = set()
        for rel in rels:
            for target, _ in self.includes(rel):
                texts.add(target)
        return texts

    def scanned(self, rel: str) -> scan.SourceFile:
        cached = self._scanned.get(rel)
        if cached is not None:
            return cached
        text = self.text(rel) or ""
        macros = dict(self.index.macros) if self.index is not None else {}
        # macros of the file's own local include closure (helpers defined in headers)
        for dep in self.closure(rel):
            if dep == rel:
                continue
            dep_text = self.text(dep) or ""
            if "#define" not in dep_text:
                continue
            for macro in scan.scan_text(dep_text, dep, "kernel").macros:
                macros.setdefault(macro.name, macro)
        shim = _IndexShim(
            macros, self.index.functions if self.index is not None else []
        )
        source = scan.scan_source(text, rel, "kernel", index=shim)
        self._scanned[rel] = source
        return source


@dataclass
class _IndexShim:
    macros: dict
    functions: list


def _mask_comments(text: str) -> str:
    """Blank comments only (keeps string literals, so include texts survive)."""
    chars = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                chars[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if chars[k] != "\n":
                    chars[k] = " "
            i = j
        elif text[i] in "\"'":
            quote = text[i]
            j = i + 1
            while j < n and text[j] != quote and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Loops and conditions inside a body
# --------------------------------------------------------------------------- #
def loop_spans(masked: str) -> list[tuple[int, int, str, int, str]]:
    """(body_start, body_end, kind, keyword_offset, header) of every loop in a body."""
    spans: list[tuple[int, int, str, int, str]] = []
    length = len(masked)
    for match in _LOOP_KEYWORD.finditer(masked):
        kind = match.group(1)
        position = match.end()
        while position < length and masked[position].isspace():
            position += 1
        if kind == "do":
            if position < length and masked[position] == "{":
                close = scan.match_close(masked, position, "{}")
                if close is not None:
                    spans.append((position + 1, close, "do", match.start(), ""))
            continue
        if position >= length or masked[position] != "(":
            continue
        close_paren = scan.match_close(masked, position, "()")
        if close_paren is None:
            continue
        header = compact(masked[position + 1 : close_paren])
        after = close_paren + 1
        while after < length and masked[after].isspace():
            after += 1
        if kind == "while" and after < length and masked[after] == ";":
            continue  # tail of a do-while
        if after < length and masked[after] == "{":
            close = scan.match_close(masked, after, "{}")
            if close is None:
                continue
            spans.append((after + 1, close, kind, match.start(), header[:120]))
        else:
            end = scan.statement_end(masked, close_paren + 1)
            spans.append((close_paren + 1, end + 1, kind, match.start(), header[:120]))
    spans.sort(key=lambda item: (item[0], -item[1]))
    return spans


def _literal(part: str) -> tuple[str, bool]:
    """(positive text, negated) of one condition conjunct, outer parentheses removed."""
    text = _strip_parens(compact(part))
    negated = False
    while text.startswith("!"):
        inner = text[1:].strip()
        if inner.startswith("(") and scan.match_close(inner, 0, "()") == len(inner) - 1:
            inner = inner[1:-1].strip()
        elif not re.fullmatch(r"[\w:]+(?:\s*\([^()]*\))?", inner):
            break
        text = _strip_parens(inner)
        negated = not negated
    return text, negated


def _strip_parens(text: str) -> str:
    while text.startswith("(") and scan.match_close(text, 0, "()") == len(text) - 1:
        text = text[1:-1].strip()
    return text


@functools.lru_cache(maxsize=None)
def _literals(text: str) -> frozenset:
    return frozenset(_literal(p) for p in _split_condition(text))


def _exclusive(cond_a: str, cond_b: str) -> bool:
    """True when two condition texts cannot both hold: one has a conjunct and the other its
    negation, or one has ``!(Y)`` where every conjunct of ``Y`` is a conjunct of the other
    (``A && x && y`` vs ``A && !(x && y)``: the two arms of an ``if constexpr``)."""
    if not cond_a or not cond_b:
        return False
    lits_a = _literals(cond_a)
    lits_b = _literals(cond_b)
    for text, negated in lits_a:
        if (text, not negated) in lits_b:
            return True
    for mine, other in ((lits_a, lits_b), (lits_b, lits_a)):
        for text, negated in mine:
            if not negated:
                continue
            inner = _literals(text)
            if len(inner) > 1 and inner <= other:
                return True
    return False


def _split_condition(text: str) -> list[str]:
    """Split on top-level ``&&`` (parentheses respected)."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text.startswith("&&", i):
            parts.append(compact(text[start:i]))
            start = i + 2
            i += 1
        i += 1
    tail = compact(text[start:])
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


exclusive_conditions = _exclusive
split_condition = _split_condition


# --------------------------------------------------------------------------- #
# Inlining
# --------------------------------------------------------------------------- #
class _Inliner:
    def __init__(
        self,
        db,
        kernel_path: str,
        functions: Sequence[scan.FunctionDef],
        *,
        allow_experimental: bool = True,
    ):
        self.db = db
        self.kernel_path = kernel_path
        self.allow_experimental = allow_experimental
        self.by_name: dict[str, list[scan.FunctionDef]] = {}
        for fn in sorted(functions, key=lambda f: (f.path, f.line, f.key)):
            self.by_name.setdefault(fn.name, []).append(fn)
        self.calls: list[KCall] = []
        self.reached: set[str] = set()
        self._stack: list[str] = []
        self._loops_cache: dict[str, list] = {}
        self._blocks_cache: dict[str, list] = {}
        self._consts_cache: dict[str, tuple] = {}
        self._lib_cache: dict[tuple, tuple] = {}

    # ---- helpers --------------------------------------------------------- #
    def _fid(self, fn: scan.FunctionDef) -> str:
        return f"{fn.path}:{fn.line}:{fn.key}"

    def _loops(self, fn):
        fid = self._fid(fn)
        if fid not in self._loops_cache:
            self._loops_cache[fid] = loop_spans(fn.masked_body)
        return self._loops_cache[fid]

    def _blocks(self, fn):
        fid = self._fid(fn)
        if fid not in self._blocks_cache:
            self._blocks_cache[fid] = scan.control_blocks(fn.masked_body, fn.body)
        return self._blocks_cache[fid]

    def _local_constants(self, fn) -> tuple[tuple[str, str], ...]:
        fid = self._fid(fn)
        if fid not in self._consts_cache:
            from .effects_db import _local_constants

            try:
                self._consts_cache[fid] = _local_constants(fn)
            except Exception:  # pragma: no cover - defensive
                self._consts_cache[fid] = ()
        return self._consts_cache[fid]

    def local_candidates(self, call: scan.CallSite) -> list[scan.FunctionDef]:
        pool = self.by_name.get(call.name, [])
        if not pool:
            return []
        if call.is_member and call.receiver_type:
            typed = [
                fn for fn in pool if fn.owner.rsplit("::", 1)[-1] == call.receiver_type
            ]
            pool = typed or pool
        qualifier = call.qualifier
        if qualifier:
            last = qualifier.rsplit("::", 1)[-1]
            qualified = [
                fn
                for fn in pool
                if fn.qualified_name.endswith(f"{qualifier}::{call.name}")
                or last in fn.qualified_name.split("::")[:-1]
            ]
            pool = qualified or pool
        n_args = len(call.args)
        n_targs = len(call.template_args)

        def arity_ok(fn: scan.FunctionDef) -> bool:
            params = [p for p in fn.params if p.name or p.type]
            variadic = any(p.variadic for p in params)
            required = sum(1 for p in params if p.default is None and not p.variadic)
            if n_args < required or (n_args > len(params) and not variadic):
                return False
            tvariadic = any(p.variadic for p in fn.template_params)
            if n_targs > len(fn.template_params) and not tvariadic:
                return False
            return True

        compatible = [fn for fn in pool if arity_ok(fn)] or []
        return compatible

    def _library(self, call: scan.CallSite, targs, args, caller_path: str):
        """(keys, resolution, role, layer) for a library call, or None."""
        cache_key = (
            call.name,
            call.qualifier,
            tuple(targs),
            len(args),
            call.is_member,
            call.receiver_type,
            caller_path,
        )
        if cache_key in self._lib_cache:
            return self._lib_cache[cache_key]
        result = None
        try:
            candidates, resolution = self.db.resolve(
                call.name,
                qualifier=call.qualifier,
                template_args=targs,
                args=args,
                is_member=call.is_member,
                receiver_type=call.receiver_type,
                caller_path=caller_path,
            )
        except Exception:  # pragma: no cover - defensive
            candidates, resolution = [], "unresolved"
        if (
            candidates
            and not self.allow_experimental
            and "experimental" not in call.qualifier
        ):
            # the ckernel::experimental API is only visible to kernels that name it
            candidates = [
                c
                for c in candidates
                if "::experimental::" not in f"::{c.qualified_name}"
            ]
            if not candidates:
                self._lib_cache[cache_key] = None
                return None
        if candidates:
            roles = sorted({c.role for c in candidates})
            layers = sorted({c.layer for c in candidates})
            result = (
                tuple(sorted(c.key for c in candidates)),
                resolution,
                roles[0] if len(roles) == 1 else _pick_role(roles),
                layers[0],
            )
        elif _is_pattern_call(self.db, call):
            result = ((), "pattern", "primitive", "")
        self._lib_cache[cache_key] = result
        return result

    # ---- walk -------------------------------------------------------------- #
    def walk(
        self,
        fn: scan.FunctionDef,
        bindings: dict[str, str],
        *,
        loops: tuple[Loop, ...] = (),
        ifdefs: tuple[str, ...] = (),
        conditions: tuple[str, ...] = (),
        thread: str | None = None,
        stack: tuple[tuple[str, str], ...] = (),
        chain: str = "",
        depth: int = 0,
        attention: bool = False,
    ) -> None:
        fid = self._fid(fn)
        if fid in self._stack or depth > MAX_DEPTH:
            return
        self.reached.add(fid)
        self._stack.append(fid)
        try:
            self._walk_body(
                fn,
                bindings,
                loops,
                ifdefs,
                conditions,
                thread,
                stack,
                chain,
                depth,
                attention or bool(ATTENTION_PATH.search(fn.path)),
            )
        finally:
            self._stack.pop()

    def _walk_body(
        self,
        fn,
        bindings,
        loops,
        ifdefs,
        conditions,
        thread,
        stack,
        chain,
        depth,
        attention,
    ):
        frame = dict(bindings)
        for name, expr in self._local_constants(fn):
            if name in frame:
                continue
            frame[name] = _cap(substitute(expr, frame))
        fn_ifdefs = ifdefs
        if fn.pp_condition and fn.pp_condition not in fn_ifdefs:
            fn_ifdefs = fn_ifdefs + (fn.pp_condition,)
        spans = self._loops(fn)
        blocks = self._blocks(fn)
        for call in fn.calls:
            # control-flow context
            dead = False
            parts: list[str] = list(conditions)
            for block in blocks:
                if not (block["start"] <= call.offset < block["end"]):
                    continue
                text = _cap(substitute(block["condition"], frame))
                verdict = scan.truth(text)
                if verdict is False:
                    dead = True
                    break
                if verdict is True:
                    continue
                parts.append(text)
            if dead:
                continue
            _, pp = fn.pp_info_at_line(call.line)
            call_ifdefs = fn_ifdefs
            if pp and pp != fn.pp_condition and pp not in call_ifdefs:
                call_ifdefs = call_ifdefs + (pp,)
            call_loops = list(loops)
            for start, end, kind, keyword, header in spans:
                if start <= call.offset < end:
                    loop_id = f"{fn.path}:{fn.line_at(keyword)}"
                    call_loops.append(
                        Loop(
                            id=loop_id,
                            instance=f"{chain}>{loop_id}" if chain else loop_id,
                            kind=kind,
                            header=_cap(substitute(header, frame), 120),
                        )
                    )
            call_thread = call.thread or thread
            targs = tuple(_cap(substitute(a, frame)) for a in call.template_args)
            args = tuple(_cap(substitute(a, frame)) for a in call.args)
            evidence = f"{fn.path}:{call.line}"
            local = [] if call.is_macro else self.local_candidates(call)
            if local:
                callee = _choose(local, fn.path, self.kernel_path)
                child = _bind(self.db, callee, targs, args)
                self.walk(
                    callee,
                    child,
                    loops=tuple(call_loops),
                    ifdefs=call_ifdefs,
                    conditions=tuple(parts),
                    thread=call_thread,
                    stack=stack + ((callee.name, evidence),),
                    chain=f"{chain}>{evidence}" if chain else evidence,
                    depth=depth + 1,
                    attention=attention,
                )
                continue
            library = self._library(call, targs, args, fn.path)
            if library is None:
                continue
            keys, resolution, role, layer = library
            self.calls.append(
                KCall(
                    index=len(self.calls),
                    name=call.name,
                    callee=call.callee,
                    qualifier=call.qualifier,
                    template_args=targs,
                    args=args,
                    raw_template_args=tuple(call.template_args),
                    raw_args=tuple(call.args),
                    path=fn.path,
                    line=call.line,
                    thread=call_thread,
                    condition=scan.join_conditions(*parts),
                    ifdefs=call_ifdefs,
                    loops=tuple(call_loops),
                    stack=stack,
                    via_macro=call.via_macro,
                    is_member=call.is_member,
                    receiver_type=call.receiver_type,
                    keys=keys,
                    resolution=resolution,
                    role=role,
                    layer=layer,
                    in_attention=attention,
                )
            )


_ROLE_PRIORITY = (
    "execute",
    "init",
    "reconfig",
    "uninit",
    "config",
    "hw_startup",
    "helper",
)


def _pick_role(roles: Sequence[str]) -> str:
    for role in _ROLE_PRIORITY:
        if role in roles:
            return role
    return sorted(roles)[0]


def _cap(text: str, limit: int = MAX_ARG_TEXT) -> str:
    text = compact(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _choose(
    candidates: list[scan.FunctionDef], caller_path: str, kernel_path: str
) -> scan.FunctionDef:
    for preferred in (caller_path, kernel_path):
        same = [fn for fn in candidates if fn.path == preferred]
        if same:
            return same[0]
    return candidates[0]


def _bind(db, callee: scan.FunctionDef, targs, args) -> dict[str, str]:
    try:
        return db.graph.bind(callee, targs, args)
    except Exception:  # pragma: no cover - defensive
        return {}


_PATTERN_NAMES: dict[int, re.Pattern[str]] = {}


def _is_pattern_call(db, call: scan.CallSite) -> bool:
    """Whether a call/member pattern of the DB recognizes the call (instruction macros etc.)."""
    key = id(db)
    regex = _PATTERN_NAMES.get(key)
    if regex is None:
        alternatives = sorted(
            {
                f"(?:{pattern.match})"
                for _, pattern in getattr(db, "patterns", ())
                if pattern.kind == "call"
            }
        )
        regex = re.compile("|".join(alternatives) if alternatives else r"(?!x)x")
        _PATTERN_NAMES[key] = regex
    if call.is_member:
        return False
    return bool(regex.fullmatch(call.name))


# --------------------------------------------------------------------------- #
# Kernel extraction
# --------------------------------------------------------------------------- #
def _entry_functions(source: scan.SourceFile) -> list[scan.FunctionDef]:
    return sorted(
        (fn for fn in source.functions if fn.name in ENTRY_NAMES and not fn.owner),
        key=lambda fn: (fn.line, fn.key),
    )


def is_compute_closure(cache: SourceCache, closure: Sequence[str]) -> bool:
    for target in cache.include_texts(closure):
        if _COMPUTE_INCLUDE.search(target):
            return True
    return False


def _config_values(
    source: scan.SourceFile, entry: scan.FunctionDef, inliner: _Inliner
) -> tuple:
    items: dict[str, str] = {}
    for macro in source.macros:
        if macro.params is None and macro.body.strip():
            items.setdefault(f"#define {macro.name}", _cap(macro.body, 160))
    for name, value in sorted(source.constants.items()):
        items.setdefault(f"constexpr {name}", _cap(value, 160))
    for name, expr in inliner._local_constants(entry):
        items.setdefault(f"local {name}", _cap(expr, 160))
    return tuple(sorted(items.items()))


_DEFINE_NAME = re.compile(
    r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)|\b([A-Z][A-Z0-9_]{2,})\b"
)


def _define_names(texts: Iterable[str]) -> tuple[str, ...]:
    names: set[str] = set()
    for text in texts:
        for match in _DEFINE_NAME.finditer(text or ""):
            name = match.group(1) or match.group(2) or match.group(3)
            if name and name not in ("ARCH_BLACKHOLE",):
                names.add(name)
    return tuple(sorted(names))


_TEMPLATE_HEADER = re.compile(r"(?<![\w.])template\s*<")


@functools.lru_cache(maxsize=None)
def _prototypes_in(masked: str) -> tuple[tuple[str, tuple, tuple], ...]:
    """(name, template params, runtime params) of every ``template <..> R name(..);`` declaration."""
    found = []
    for match in _TEMPLATE_HEADER.finditer(masked):
        opening = match.end() - 1
        close = scan.match_angle(masked, opening)
        if close is None:
            continue
        position, depth, end = close + 1, 0, None
        while position < len(masked):
            char = masked[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and char in ";{}":
                end = position if char == ";" else None
                break
            position += 1
        if end is None:
            continue
        declaration = masked[close + 1 : end]
        head = re.search(r"([A-Za-z_]\w*)\s*\(", declaration)
        if head is None or head.group(1) in (
            "decltype",
            "sizeof",
            "alignas",
            "requires",
        ):
            continue
        paren = head.end() - 1
        paren_close = scan.match_close(declaration, paren, "()")
        if paren_close is None:
            continue
        tail = compact(declaration[paren_close + 1 :])
        if tail and not re.fullmatch(r"(?:const|noexcept|override|final|\s)*", tail):
            continue
        try:
            template_params = scan.parse_params(masked[opening + 1 : close], "template")
            params = scan.parse_params(declaration[paren + 1 : paren_close], "runtime")
        except Exception:  # pragma: no cover - defensive
            continue
        found.append((head.group(1), template_params, params))
    return tuple(found)


def _with_prototype_defaults(functions, cache: SourceCache, closure) -> list:
    """Definitions whose parameter defaults live on a separate declaration (``.hpp`` declares
    ``template <.., Mode m = Mode::X> void f(..);``, the ``.inl`` defines it without defaults)
    get those defaults, so unbound template parameters of kernel-library helpers resolve.
    """
    prototypes: dict[str, list] = {}
    for dep in closure:
        for name, template_params, params in _prototypes_in(
            _mask_comments(cache.text(dep) or "")
        ):
            prototypes.setdefault(name, []).append((template_params, params))
    if not prototypes:
        return list(functions)
    result = []
    for fn in functions:
        merged = fn
        for template_params, params in prototypes.get(fn.name, ()):
            if len(template_params) != len(fn.template_params) or len(params) != len(
                fn.params
            ):
                continue
            pairs = list(zip(fn.template_params, template_params)) + list(
                zip(fn.params, params)
            )
            if not any(
                mine.default is None and theirs.default is not None
                for mine, theirs in pairs
            ):
                continue
            merged = copy.copy(fn)
            merged.template_params = tuple(
                (
                    dataclasses.replace(mine, default=theirs.default)
                    if mine.default is None and theirs.default is not None
                    else mine
                )
                for mine, theirs in zip(fn.template_params, template_params)
            )
            merged.params = tuple(
                (
                    dataclasses.replace(mine, default=theirs.default)
                    if mine.default is None and theirs.default is not None
                    else mine
                )
                for mine, theirs in zip(fn.params, params)
            )
            break
        result.append(merged)
    return result


def extract_kernel(db, cache: SourceCache, rel: str) -> Kernel | None:
    """Analyse one kernel translation unit (``None`` when it is not a compute kernel)."""
    source = cache.scanned(rel)
    entries = _entry_functions(source)
    if not entries:
        return None
    closure = cache.closure(rel)
    if not is_compute_closure(cache, closure):
        return None
    functions: list[scan.FunctionDef] = []
    for dep in closure:
        functions.extend(cache.scanned(dep).functions)
    functions = _with_prototype_defaults(functions, cache, closure)
    allow_experimental = any(
        re.search(r"\bexperimental::|namespace\s+experimental\b", cache.text(dep) or "")
        for dep in closure
    )
    inliner = _Inliner(db, rel, functions, allow_experimental=allow_experimental)
    entry = entries[0]
    inliner.walk(entry, {}, attention=bool(ATTENTION_PATH.search(rel)))
    # Local functions with compute calls that the entry never reached (unified kernels whose
    # ops are reached through objects / operator()) are analysed as extra roots, after main.
    extra: list[str] = []
    if any(dep != rel for dep in closure):
        for fn in sorted(functions, key=lambda f: (f.path, f.line, f.key)):
            fid = inliner._fid(fn)
            if fid in inliner.reached or fn.name in ENTRY_NAMES:
                continue
            if not (fn.path.startswith("models/") and "unified_kernels" in fn.path):
                continue
            before = len(inliner.calls)
            inliner.walk(
                fn,
                {},
                stack=((fn.name, fn.loc),),
                chain=fn.loc,
                attention=bool(ATTENTION_PATH.search(fn.path)),
            )
            if len(inliner.calls) > before:
                extra.append(fn.loc)
    calls = tuple(inliner.calls)
    if not any(call.layer == "compute_api" for call in calls):
        return None
    digest = hashlib.sha256()
    for dep in closure:
        digest.update(
            dep.encode("utf-8")
            + b"\0"
            + (cache.text(dep) or "").encode("utf-8")
            + b"\0"
        )
    attention = bool(ATTENTION_PATH.search(rel)) or any(
        ATTENTION_PATH.search(dep)
        for dep in closure
        if dep != rel and not dep.startswith(KERNEL_LIB)
    )
    return Kernel(
        path=rel,
        entry=entry.name,
        sources=closure,
        calls=calls,
        config=_config_values(source, entry, inliner),
        defines=_define_names(part for call in calls for part in call.ifdefs),
        attention=attention,
        fingerprint=digest.hexdigest(),
        extra_roots=tuple(extra),
    )


def extract(
    db, repo_root: Path | str, *, paths: Sequence[str] | None = None
) -> list[Kernel]:
    """Every compute kernel under the kernel roots, sorted by path."""
    root = Path(repo_root)
    cache = SourceCache(root, getattr(db, "scan", None))
    candidates = list(paths) if paths is not None else discover(root)
    kernels: list[Kernel] = []
    for rel in sorted(candidates):
        if not rel.endswith((".cpp", ".cc")):
            continue
        text = cache.text(rel) or ""
        if not re.search(r"\b(?:kernel_main|MAIN)\b", text):
            continue
        kernel = extract_kernel(db, cache, rel)
        if kernel is not None:
            kernels.append(kernel)
    return kernels


def fingerprint(kernels: Sequence[Kernel]) -> str:
    """sha256 over the (path, fingerprint) of every analysed kernel."""
    digest = hashlib.sha256()
    for kernel in sorted(kernels, key=lambda k: k.path):
        digest.update(
            kernel.path.encode("utf-8")
            + b"\0"
            + kernel.fingerprint.encode("ascii")
            + b"\0"
        )
    return digest.hexdigest()


def same_loop(a: KCall, b: KCall) -> str | None:
    """The innermost loop instance both calls share, or ``None``."""
    shared = set(a.loop_instances) & set(b.loop_instances)
    if not shared:
        return None
    for loop in reversed(a.loops):
        if loop.instance in shared:
            return loop.instance
    return None


__all__ = [
    "ATTENTION_PATH",
    "KCall",
    "Kernel",
    "Loop",
    "SourceCache",
    "discover",
    "exclusive_conditions",
    "extract",
    "extract_kernel",
    "fingerprint",
    "in_kernel_roots",
    "loop_spans",
    "same_loop",
    "split_condition",
]
