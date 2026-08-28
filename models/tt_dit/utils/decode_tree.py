# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Call-stack-shaped ledger for DiffVAE decode timings.

The timing helpers (``stage_timer``, ``block_prof``, ``na3d._sp_w_prof``) are already nested the way
a reader wants them: a stage contains its blocks, a
block contains its attention, an attention contains its collectives. Each one just threw its
measurement into a flat sink -- a log line or a module-global dict -- and forgot the stack it was on.
This module is that stack, and the tree falls out of it.

It measures nothing and syncs nothing: callers hand it durations they have already taken. That is
what lets ``layers/na3d.py`` import it without a layer importing a model -- attribution happens
through a thread-local stack rather than an argument threaded down the call chain.

Not valid under trace capture: there the spans time the capture, not the execution.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict, deque

#: One gate for the whole instrumentation. ``diffvae_ltx_stage5`` and ``na3d`` import THIS rather
#: than reading the env themselves: three import-time constants that can disagree would give a tree
#: with partial data, and every "other" remainder would then quietly lie about where time went.
ENABLED = os.environ.get("DIFFVAE_STAGE_TIMING", "") not in ("", "0")

#: Deep mode adds spans inside the 16 deterministic NABlocks -- 64 more device syncs per decode, so
#: it is opt-in and its numbers are NOT comparable with a plain DIFFVAE_STAGE_TIMING run.
DEEP = ENABLED and os.environ.get("DIFFVAE_BLOCK_PROF", "") not in ("", "0")

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
        full would count those ms twice -- which is what the flat [block-prof] table this replaced
        did, printing a span and the span inside it as peers."""
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
    if span.node.parent is None:
        _finish(span.node, st)


def _finish(root: Node, st: list) -> None:
    for leftover in st:  # a raise inside decode can leave these open
        leftover.node.flags.add("unclosed")
    del st[:]
    with _LOCK:
        _ROOTS.append(root)


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
    max_depth = int(os.environ.get("DIFFVAE_TREE_DEPTH", 8))
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
