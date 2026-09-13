# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-synchronised timing spans and the call-stack-shaped tree they record into.

``span`` is the one timing primitive: synchronise the device, open a node, run the body, synchronise
again, close the node with the elapsed milliseconds. ``timed`` is the same as a decorator, for a
method whose whole body is one span. Spans nest by a thread-local stack, so attribution needs no
argument threaded down the call chain: a stage contains its blocks, a block its attention, an
attention its collectives, and the tree is that stack remembered. A layer and a model both record
into it by importing this module alone.

Three gates, read once at import so one process holds one setting:

* ``TT_DIT_STAGE_TIMING`` -- ``ENABLED``. Nothing here does anything without it; every span is
  a pass-through, so instrumented code costs nothing in a normal run.
* ``TT_DIT_BLOCK_PROF`` -- ``DEEP``. Spans opened with ``deep=True`` record only under this. They
  sit inside per-block regions, so they are numerous and each costs two device syncs; a deep run's
  totals are NOT comparable with a plain timing run.
* ``TT_DIT_STAGE_LOG`` -- ``LIVE``. One stdout line as each span opens and one as it closes, so a
  run reports its own progress and a device-side hang reads as the last ``>`` with no ``<``. Does
  not touch the tree.

Absolute totals are inflated by one ``synchronize_device`` per span open and close; the split
between siblings is what the tree is for. Not valid under trace capture: there the spans time the
capture, not the execution.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
import time
from collections import OrderedDict, deque

import ttnn

#: One gate for the whole instrumentation, held here so no caller reads the environment itself:
#: import-time constants that can disagree would give a tree with partial data, and every "other"
#: remainder would then quietly lie about where time went.
ENABLED = os.environ.get("TT_DIT_STAGE_TIMING", "") not in ("", "0")

#: ``deep=True`` spans record only under this. They live inside per-block regions -- dozens more
#: device syncs per decode -- so the mode is opt-in and its totals are not comparable with ENABLED alone.
DEEP = ENABLED and os.environ.get("TT_DIT_BLOCK_PROF", "") not in ("", "0")

#: One stdout line as each span opens and one as it closes. A device-side hang then reads as the last
#: "> label" line with no matching "<", which names the culprit, and the lines keep a silence-reaping
#: job runner fed. Costs nothing beyond the syncs the spans already pay and does not touch the tree.
LIVE = ENABLED and os.environ.get("TT_DIT_STAGE_LOG", "") not in ("", "0")

ATTENTION, SDPA, ALLGATHER, MLP = "attention", "sdpa", "allgather", "mlp"
CONTEXT_INJECT, RESHAPE, UPSAMPLE = "context-inject", "reshape+permute", "upsample"
PROJ, NORM_ROPE = "projection", "norm+rope"
HOST_XFER, HOST_COMPUTE, SETUP = "host-transfer", "host-compute", "setup"

_MAX_DEPTH = 64
_WIDTH = 104
_LABEL_W = 56
_CAT_W = 16
_ROOTS: deque = deque(maxlen=8)
_LOCK = threading.Lock()
_local = threading.local()


class Node:
    __slots__ = ("label", "category", "incl_ms", "children", "parent", "flags")

    def __init__(self, label, category, parent):
        self.label, self.category, self.parent = label, category, parent
        self.incl_ms, self.children, self.flags = 0.0, [], set()

    @property
    def self_ms(self) -> float:
        """Time here that is not in a child. kv-allgather runs inside attention, so charging both in
        full would count those ms twice."""
        return max(self.incl_ms - sum(c.incl_ms for c in self.children), 0.0)

    def __repr__(self):  # debugging a live stack is the main use
        return f"<Node {self.label!r} {self.incl_ms:.1f}ms kids={len(self.children)}>"


class Span:
    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node


def _stack() -> list:
    st = getattr(_local, "stack", None)
    if st is None:
        st = _local.stack = []
    return st


def _live(mark: str, depth: int, label: str, tail: str = "") -> None:
    """One progress line: wall-clock stamp, depth as indent, ``>`` opening / ``<`` closing / ``!``
    aborted. Written straight to stdout and flushed, so it streams through ``pytest -s`` and a
    broker log alike instead of waiting on a buffer."""
    stamp = time.strftime("%H:%M:%S")
    sys.stdout.write(f"[stage {stamp}] {'  ' * depth}{mark} {label}{tail}\n")
    sys.stdout.flush()


def open_span(label, *, category=None, root=False):
    """Push a span. ``label`` is required: a node names itself from birth, so a span that never
    closes still says which one it was -- anonymous orphans are useless exactly when a leak needs
    finding. Returns ``None`` when disabled, and every close/abort accepts ``None``, so no call site
    needs a guard."""
    if not ENABLED:
        return None
    st = _stack()
    if root:
        del st[:]
    if len(st) >= _MAX_DEPTH:
        return None  # runaway guard: record nothing rather than grow the tree without bound
    parent = st[-1].node if st else None
    node = Node(label, category, parent)
    if parent is not None:
        parent.children.append(node)
    span = Span(node)
    if LIVE:
        _live(">", len(st), label)
    st.append(span)
    return span


def close_span(span, ms) -> None:
    if span is None or not ENABLED:
        return
    st = _stack()
    if not any(s is span for s in st):
        return  # double close: drop it rather than corrupt a live parent
    while st[-1] is not span:  # something opened below us never closed -- and it can name itself
        orphan = st.pop().node
        orphan.flags.add("unclosed")  # incl_ms stays 0, so it lands in the parent's remainder
    st.pop()
    span.node.incl_ms = ms  # a node closes exactly once: assignment, not accumulation
    if LIVE:
        _live("<", len(st), span.node.label, f"  {ms:.1f} ms")
    if span.node.parent is None:
        _finish(span.node, st)


def abort_span(span) -> None:
    """Exception path: keep the partial node but mark it, and never leave the stack deeper than it
    started."""
    if span is None or not ENABLED:
        return
    st = _stack()
    if not any(s is span for s in st):
        return
    while st[-1] is not span:
        st.pop().node.flags.add("unclosed")
    st.pop()
    span.node.flags.add("aborted")
    if LIVE:
        _live("!", len(st), span.node.label, "  aborted")
    if span.node.parent is None:
        _finish(span.node, st)


def _finish(root: Node, st: list) -> None:
    for leftover in st:  # a raise inside decode can leave these open
        leftover.node.flags.add("unclosed")
    del st[:]
    with _LOCK:
        _ROOTS.append(root)


# ---------------------------------------------------------------------------------- timing spans


@contextlib.contextmanager
def span(device, label: str, *, category: str | None = None, root: bool = False, deep: bool = False):
    """Time the body as one node of the tree.

    Synchronises ``device`` before the clock starts and again before it stops, so the measurement is
    the body's device work and not whatever was still queued from before it. ``root=True`` starts a
    new tree; ``deep=True`` records only under ``DEEP``. Inert -- no sync, no node -- when its gate
    is off, so a span costs nothing in an untimed run.

    The label is fixed at open rather than on the way out because a node names itself from birth:
    a span that never closes is then still identifiable, which is when the name matters most. On an
    exception the node is kept, marked ``aborted`` and popped, so no later span nests under a dead
    parent; the exception propagates.
    """
    if not ENABLED or (deep and not DEEP):
        yield
        return
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    node = open_span(label, category=category, root=root)
    try:
        yield
    except BaseException:
        abort_span(node)
        raise
    ttnn.synchronize_device(device)
    close_span(node, (time.perf_counter() - t0) * 1000)


def timed(label, *, category: str | None = None, root: bool = False, deep: bool = False, device="mesh_device"):
    """:func:`span` as a decorator, for a function whose whole body is one span.

    ``device`` is the name of the attribute of the first argument (``self``) that holds the mesh
    device, or a callable of the call's ``(*args, **kwargs)`` returning it. ``label`` is a string, or
    a callable of the same arguments returning one, for a label that depends on the call::

        @timed(lambda self, stage, *a, **k: f"stage {stage}", category=SETUP)
        def _setup(self, stage, x): ...

    When the gate is off the wrapper calls straight through without resolving either, so a
    decorated method costs the same as an undecorated one in an untimed run.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not ENABLED or (deep and not DEEP):
                return fn(*args, **kwargs)
            dev = device(*args, **kwargs) if callable(device) else getattr(args[0], device)
            name = label(*args, **kwargs) if callable(label) else label
            with span(dev, name, category=category, root=root):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def root_count() -> int:
    with _LOCK:
        return len(_ROOTS)


def roots() -> list:
    with _LOCK:
        return list(_ROOTS)


def reset() -> None:
    """Drop recorded roots and any half-open stack. For tests; not used by the decode path."""
    del _stack()[:]
    with _LOCK:
        _ROOTS.clear()


# --------------------------------------------------------------------------------------- reporting


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def _pool(nodes):
    """Group siblings by exact label. A pure view -- the tree is never mutated, and nothing is merged
    for merely containing a number. ``attention`` fires once per band, so its row pools those spans;
    ``stage5 block 0..7`` keep distinct labels and stay distinct rows."""
    groups = OrderedDict()
    for n in nodes:
        groups.setdefault(n.label, []).append(n)
    return groups


def _rows(label, nodes, root_ms, parent_ms, prefix="", is_last=True, depth=0, max_depth=8, out=None):
    out = [] if out is None else out
    incl = sum(n.incl_ms for n in nodes)
    kids = [c for n in nodes for c in n.children]
    kids_ms = sum(c.incl_ms for c in kids)
    marks = " !" if kids_ms > incl * 1.01 else ""  # broken pairing: show it, never clamp it away
    if any("unclosed" in n.flags for n in nodes):
        marks += "  (never closed)"  # named, because the label was set at open
    connector = "" if depth == 0 else ("└─ " if is_last else "├─ ")
    # The category is what the roll-up charges this node's SELF time to; "-" means uncategorised, i.e.
    # it lands in the roll-up's "other" row. Pooled siblings share a call site, so the first node's
    # category speaks for all of them.
    cat = nodes[0].category or "-"
    out.append(
        (f"{prefix}{connector}{label}{marks}", incl, _pct(incl, parent_ms), _pct(incl, root_ms), len(nodes), cat)
    )
    if depth >= max_depth:
        return out
    self_ms = max(incl - kids_ms, 0.0)
    groups = [
        (lbl, ns)
        for lbl, ns in _pool(kids).items()
        if sum(n.incl_ms for n in ns) >= 0.5 or _pct(sum(n.incl_ms for n in ns), root_ms) >= 0.1
    ]
    child_prefix = prefix + ("" if depth == 0 else ("   " if is_last else "│  "))
    for i, (lbl, ns) in enumerate(groups):
        last = i == len(groups) - 1 and self_ms < 0.5
        _rows(lbl, ns, root_ms, incl, child_prefix, last, depth + 1, max_depth, out)
    if groups and self_ms >= 0.5:
        # The remainder is the parent's own self time, so it carries the parent's category -- this row
        # is exactly what that category's roll-up entry is made of.
        out.append(
            (f"{child_prefix}└─ · other (unattributed)", self_ms, _pct(self_ms, incl), _pct(self_ms, root_ms), 0, cat)
        )
    return out


def category_totals(root: Node):
    """Exclusive self-time per category, and the span count behind each. Sums to the root by
    construction: every ms belongs to exactly one node's self-time, so unlike a flat ledger this
    needs no ``untracked`` estimate to reconcile."""
    totals, spans = {}, {}

    def walk(n):
        key = n.category or "other (uncategorized)"
        totals[key] = totals.get(key, 0.0) + n.self_ms
        spans[key] = spans.get(key, 0) + 1
        for c in n.children:
            walk(c)

    walk(root)
    return OrderedDict(sorted(totals.items(), key=lambda kv: -kv[1])), spans


def render_tree(root: Node, *, title: str, measured_ms: float | None = None) -> str:
    max_depth = int(os.environ.get("TT_DIT_TREE_DEPTH", 8))
    flags = f"  [{' '.join(sorted(root.flags))}]" if root.flags else ""
    head = f"root {root.incl_ms:.1f} ms{flags}"
    if measured_ms is not None:
        head += f"  ·  test-measured {measured_ms:.0f} ms (Δ {root.incl_ms - measured_ms:+.1f})"
    out = [
        "=" * _WIDTH,
        f"DECODE TREE · {title}",
        head,
        "absolute totals inflated by one synchronize_device per span open/close",
        "-" * _WIDTH,
        f"{'label':<{_LABEL_W}}{'ms':>10}{'%par':>8}{'%tot':>8}{'n':>5}  {'category':<{_CAT_W}}",
    ]
    for lbl, ms, par, tot, n, cat in _rows(root.label, [root], root.incl_ms, root.incl_ms, max_depth=max_depth):
        out.append(f"{lbl:<{_LABEL_W}}{ms:>10.1f}{par:>7.1f}%{tot:>7.1f}%{(n or ''):>5}  {cat:<{_CAT_W}}".rstrip())
    return "\n".join(out)


def render_categories(totals, spans, total_ms: float, *, title: str = "CATEGORY ROLL-UP") -> str:
    out = [f"{title} (exclusive self time)", f"{'category':<44}{'ms':>10}{'%tot':>8}{'spans':>7}"]
    for cat, ms in totals.items():
        out.append(f"{cat:<44}{ms:>10.1f}{_pct(ms, total_ms):>7.1f}%{spans[cat]:>7}")
    tracked = sum(totals.values())
    out.append(f"{'TOTAL':<44}{tracked:>10.1f}{_pct(tracked, total_ms):>7.1f}%")
    # Self-times partition the root exactly, so anything but 100% means some parent measured shorter
    # than the children it contains and its self-time clamped at zero. Say which, rather than let a
    # reader wonder why the column does not add up.
    if abs(tracked - total_ms) > max(0.5, total_ms * 0.0005):
        out.append(
            f"{'':<44}{'':>10}  ({tracked - total_ms:+.1f} ms vs root: a parent measured shorter than its children)"
        )
    return "\n".join(out)


def render(root: Node, *, title: str, measured_ms: float | None = None) -> str:
    """One string, printed with a single print(). Not per-line logger.info: under pytest -s every
    loguru line carries a ~40-char prefix that would shift rows and destroy the column alignment."""
    totals, spans = category_totals(root)
    return "\n".join(
        [
            render_tree(root, title=title, measured_ms=measured_ms),
            "-" * _WIDTH,
            render_categories(totals, spans, root.incl_ms),
            "=" * _WIDTH,
        ]
    )
