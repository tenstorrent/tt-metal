# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Debug tracing for the distributed dispatch layer.

Usage:
    from ttml.distributed.debug import DispatchTracer, dispatch_trace

    # Option 1: context manager
    with DispatchTracer() as tracer:
        y = model(x)
    for entry in tracer.entries:
        print(entry)
    print(dispatch_trace.format_entries_tree(tracer.entries))

    # Option 2: global toggle
    dispatch_trace.enable()
    y = model(x)
    dispatch_trace.disable()
    print(dispatch_trace.format_entries_tree(dispatch_trace.entries))

    # Module scope (optional; AbstractModuleBase pushes automatically when tracing)
    with dispatch_trace.module_scope("my_block"):
        z = op(x)
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .layout import Layout

logger = logging.getLogger("ttml.distributed.dispatch")


def _thread_local() -> threading.local:
    return threading.local()


@dataclass
class TraceEntry:
    """Single dispatch event."""

    op_name: str
    input_layouts: List[Optional[Layout]]
    rule_name: Optional[str]
    plan: Any  # ShardingPlan or None
    pre_collectives: List[Dict[str, Any]]
    redistributions: List[Dict[str, Any]]
    post_collectives: List[Dict[str, Any]]
    output_layout: Optional[Layout]
    depth: int = 0
    module_stack: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = [f"op={self.op_name}"]
        parts.append(f"inputs={self.input_layouts}")
        if self.rule_name:
            parts.append(f"rule={self.rule_name}")
        if self.pre_collectives:
            parts.append(f"pre_ccl={self.pre_collectives}")
        if self.redistributions:
            parts.append(f"redist={self.redistributions}")
        if self.post_collectives:
            parts.append(f"post_ccl={self.post_collectives}")
        parts.append(f"output={self.output_layout}")
        return f"TraceEntry({', '.join(parts)})"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""

        def layout_repr(layout: Optional[Layout]) -> Optional[str]:
            return repr(layout) if layout is not None else None

        def json_safe(obj: Any) -> Any:
            """Recursively convert Layout and other non-JSON types to strings."""
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, Layout):
                return repr(obj)
            if isinstance(obj, (list, tuple)):
                return [json_safe(x) for x in obj]
            if isinstance(obj, dict):
                return {k: json_safe(v) for k, v in obj.items()}
            return repr(obj)

        return {
            "op_name": self.op_name,
            "rule_name": self.rule_name,
            "depth": self.depth,
            "module_stack": list(self.module_stack),
            "input_layouts": [layout_repr(l) for l in self.input_layouts],
            "output_layout": layout_repr(self.output_layout),
            "pre_collectives": json_safe(self.pre_collectives),
            "redistributions": json_safe(self.redistributions),
            "post_collectives": json_safe(self.post_collectives),
        }


class DispatchTrace:
    """In-memory trace buffer for dispatch events with optional module scope and nesting."""

    def __init__(self):
        self._enabled: bool = False
        self._entries: List[TraceEntry] = []
        self._local = _thread_local()

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def _set_depth(self, value: int) -> None:
        self._local.depth = value

    def _module_stack(self) -> List[str]:
        return getattr(self._local, "module_stack", [])

    def _set_module_stack(self, value: List[str]) -> None:
        self._local.module_stack = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entries(self) -> List[TraceEntry]:
        return list(self._entries)

    @property
    def current_depth(self) -> int:
        return self._depth()

    @property
    def current_module_stack(self) -> List[str]:
        return list(self._module_stack())

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._entries.clear()
        self._set_depth(0)
        self._set_module_stack([])

    def enter_dispatch(self) -> None:
        self._set_depth(self._depth() + 1)

    def exit_dispatch(self) -> None:
        self._set_depth(max(0, self._depth() - 1))

    def push_module(self, name: str) -> None:
        stack = list(self._module_stack())
        stack.append(name)
        self._set_module_stack(stack)

    def pop_module(self) -> None:
        stack = list(self._module_stack())
        if stack:
            stack.pop()
        self._set_module_stack(stack)

    @contextmanager
    def module_scope(self, name: str):
        """Context manager to push/pop a module name for the duration of the block."""
        self.push_module(name)
        try:
            yield
        finally:
            self.pop_module()

    def record(self, entry: TraceEntry) -> None:
        if self._enabled:
            entry.depth = self._depth()
            entry.module_stack = list(self._module_stack())
            self._entries.append(entry)
            logger.debug("%s", entry)

    def format_entries_tree(
        self,
        entries: Optional[List[TraceEntry]] = None,
        max_entries: Optional[int] = None,
        indent_str: str = "  ",
        branch: str = "├── ",
        last_branch: str = "└── ",
        bar: str = "│   ",
    ) -> str:
        """Format trace entries as a nested tree (module path + ops)."""
        if entries is None:
            entries = self.entries
        if not entries:
            return "(no trace entries)"
        if max_entries is not None:
            entries = entries[:max_entries]

        # Group entries by module_stack: path -> list of ops at that path
        by_path: Dict[tuple, List[TraceEntry]] = {}
        for e in entries:
            key = tuple(e.module_stack)
            by_path.setdefault(key, []).append(e)

        sorted_paths = sorted(by_path.keys(), key=lambda p: (len(p), p))

        def children_of(path: tuple) -> List[tuple]:
            n = len(path)
            return [p for p in sorted_paths if len(p) == n + 1 and p[:n] == path]

        def short_summary(entry: TraceEntry) -> str:
            parts = [f"op={entry.op_name}"]
            if entry.rule_name:
                parts.append(f"rule={entry.rule_name}")
            if entry.pre_collectives:
                parts.append("pre_ccl")
            if entry.redistributions:
                parts.append("redist")
            if entry.post_collectives:
                parts.append("post_ccl")
            return "  [" + ", ".join(parts) + "]"

        lines: List[str] = []

        def recurse(path: tuple, prefix: str, is_last_sibling: bool) -> None:
            segment = path[-1] if path else "root"
            path_entries = by_path.get(path, [])
            kids = children_of(path)
            has_children = len(kids) > 0
            node_char = last_branch if is_last_sibling else branch
            lines.append(prefix + node_char + segment)
            child_prefix = prefix + (indent_str if is_last_sibling else bar)
            for i, ent in enumerate(path_entries):
                op_char = (
                    last_branch if (i == len(path_entries) - 1 and not kids) else branch
                )
                lines.append(child_prefix + op_char + ent.op_name + short_summary(ent))
            for i, child_path in enumerate(kids):
                recurse(child_path, child_prefix, i == len(kids) - 1)

        root_entries = by_path.get((), [])
        root_kids = children_of(())
        # Root-level ops (no module)
        for i, ent in enumerate(root_entries):
            lines.append(
                (
                    last_branch
                    if i == len(root_entries) - 1 and not root_kids
                    else branch
                )
                + ent.op_name
                + short_summary(ent)
            )
        for i, child_path in enumerate(root_kids):
            recurse(child_path, "", i == len(root_kids) - 1)

        return "\n".join(lines) if lines else "(no entries)"

    def format_entries_html(
        self,
        entries: Optional[List[TraceEntry]] = None,
        title: str = "Dispatch trace",
    ) -> str:
        """Return a self-contained HTML page with a collapsible tree of trace entries."""
        if entries is None:
            entries = self.entries
        if not entries:
            return _HTML_TEMPLATE.format(
                title=title,
                total="0",
                body="<p>(no trace entries)</p>",
            )

        by_path: Dict[tuple, List[TraceEntry]] = {}
        all_paths: set = set()
        for e in entries:
            key = tuple(e.module_stack)
            by_path.setdefault(key, []).append(e)
            # Collect every prefix so we can show intermediate nodes (even with no direct ops)
            for i in range(len(key) + 1):
                all_paths.add(key[:i])
        sorted_paths = sorted(all_paths, key=lambda p: (len(p), p))

        def children_of(path: tuple) -> List[tuple]:
            n = len(path)
            return [p for p in sorted_paths if len(p) == n + 1 and p[:n] == path]

        def escape(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        def layout_str(x: Any) -> str:
            return escape(repr(x)) if x is not None else ""

        def ccls_html(ent: TraceEntry) -> str:
            """Render pre_collectives, redistributions and post_collectives as visible CCL lines."""
            lines: List[str] = []
            for c in ent.pre_collectives:
                ctype = c.get("type", "?")
                axis = c.get("mesh_axis", "")
                lines.append(
                    f'<span class="ccl pre">↑ {escape(str(ctype))}(axis={axis})</span>'
                )
            for r in ent.redistributions:
                arg = r.get("arg_idx", "")
                fr = r.get("from")
                to = r.get("to")
                lines.append(
                    f'<span class="ccl redist">redistribute(arg{arg}): {layout_str(fr)} → {layout_str(to)}</span>'
                )
            for c in ent.post_collectives:
                ctype = c.get("type", "?")
                axis = c.get("mesh_axis", "")
                noop = c.get("noop_backward", "")
                extra = f", noop_backward={noop}" if noop else ""
                lines.append(
                    f'<span class="ccl post">↳ {escape(str(ctype))}(axis={axis}{extra})</span>'
                )
            if not lines:
                return ""
            return '<div class="ccls">' + "".join(lines) + "</div>"

        def entry_summary(ent: TraceEntry) -> str:
            parts = [f"op={escape(ent.op_name)}"]
            if ent.rule_name:
                parts.append(f"rule={escape(ent.rule_name)}")
            if ent.pre_collectives:
                parts.append("pre_ccl")
            if ent.redistributions:
                parts.append("redist")
            if ent.post_collectives:
                parts.append("post_ccl")
            return " [" + ", ".join(parts) + "]"

        def op_li(ent: TraceEntry) -> str:
            ccl_block = ccls_html(ent)
            return (
                f'<li class="op"><span class="op-name">{escape(ent.op_name)}</span>'
                f'<span class="op-summary">{entry_summary(ent)}</span>{ccl_block}</li>'
            )

        def recurse(path: tuple) -> str:
            segment = path[-1] if path else "root"
            path_entries = by_path.get(path, [])
            kids = children_of(path)
            seg_esc = escape(segment)
            if not kids and not path_entries:
                return f'<li class="node"><span class="name">{seg_esc}</span></li>'
            inner: List[str] = []
            for ent in path_entries:
                inner.append(op_li(ent))
            for child_path in kids:
                inner.append(recurse(child_path))
            return (
                f'<li class="node"><details open><summary class="name">{seg_esc}</summary>'
                f"<ul>{chr(10).join(inner)}</ul></details></li>"
            )

        # Flame-graph style tree: weight = op count in subtree; one rect per module path
        def weight(path: tuple) -> int:
            return len(by_path.get(path, [])) + sum(
                weight(c) for c in children_of(path)
            )

        total_weight = max(1, weight(()))
        max_depth = max(len(p) for p in all_paths) if all_paths else 0
        bar_height = 12
        flame_height = (max_depth + 1) * bar_height

        # Collect (path, depth, weight) in tree order, then group by depth for layout
        nodes_in_order: List[tuple] = []

        def collect(path: tuple, depth: int) -> None:
            w = weight(path)
            nodes_in_order.append((path, depth, w))
            for c in children_of(path):
                collect(c, depth + 1)

        collect((), 0)

        # Group by depth so we can assign x from sibling order (cumulative width at same level)
        by_depth: Dict[int, List[tuple]] = {}
        for path, depth, w in nodes_in_order:
            by_depth.setdefault(depth, []).append((path, w))

        # Build SVG: root at bottom, depth upward; at each depth x = cumulative % of weights
        # Use <g> so <title> tooltip applies to the bar; add visible <text> label
        flame_rects: List[str] = []
        for depth in range(max_depth + 1):
            level_nodes = by_depth.get(depth, [])
            x_cur = 0.0
            for path, w in level_nodes:
                width_pct = 100.0 * w / total_weight
                if width_pct < 0.25:
                    continue
                segment = path[-1] if path else "root"
                seg_esc = escape(segment)
                full_label_esc = escape(f"{segment} ({w})")
                # Compact truncation: short labels for narrow bars (flame-graph style)
                max_chars = max(3, int(width_pct / 3.5))
                label = (
                    (segment[: max_chars - 1] + "..")
                    if len(segment) > max_chars
                    else segment
                )
                if width_pct >= 8:
                    label = f"{label} ({w})"
                label_esc = escape(label)
                y = (max_depth - depth) * bar_height
                hue = 200 + depth * 12
                text_x = x_cur + 0.2
                text_y = y + (bar_height - 1) / 2 + 2.2
                flame_rects.append(
                    f'<g class="flame-bar" data-path="{seg_esc}">'
                    f"<title>{full_label_esc} ops</title>"
                    f'<rect class="flame-rect" x="{x_cur:.2f}%" y="{y}" '
                    f'width="{width_pct:.2f}%" height="{bar_height - 1}" '
                    f'style="fill:hsl({hue},45%,28%);stroke:#555;stroke-width:0.3"/>'
                    f'<text class="flame-text" x="{text_x:.2f}%" y="{text_y}" '
                    f'font-size="9" fill="#e0e0e0">{label_esc}</text>'
                    f"</g>"
                )
                x_cur += width_pct

        flame_svg = (
            f'<svg class="flame-svg" viewBox="0 0 100 {flame_height}" '
            f'preserveAspectRatio="none" width="100%">'
            + "".join(flame_rects)
            + "</svg>"
        )
        flame_section = (
            '<section><h2 class="section-title">Flame graph (modules + ops)</h2>'
            '<p class="flame-meta">Width = op count in subtree. Hover for name and count.</p>'
            f'<div class="flame-container">{flame_svg}</div></section>\n  '
        )

        # Execution order section: same entries in record order with CCLs visible
        exec_lines: List[str] = []
        for i, ent in enumerate(entries):
            path_str = escape(" / ".join(ent.module_stack)) if ent.module_stack else "—"
            exec_lines.append(
                f'<div class="exec-row" style="padding-left: {ent.depth * 1.2}rem">'
                f'<span class="exec-idx">{i + 1}</span> '
                f'<span class="exec-path">{path_str}</span> '
                f'<span class="op-name">{escape(ent.op_name)}</span>'
                f'<span class="op-summary">{entry_summary(ent)}</span>'
                f"{ccls_html(ent)}</div>"
            )
        execution_body = "\n".join(exec_lines)

        body = (
            flame_section
            + '<section><h2 class="section-title">By execution order</h2><div class="exec-list">'
            + execution_body
            + "</div></section>"
        )
        return _HTML_TEMPLATE.format(
            title=title,
            total=str(len(entries)),
            body=body,
        )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: ui-monospace, monospace; font-size: 14px; margin: 1rem 2rem; background: #1e1e1e; color: #d4d4d4; }}
    h1 {{ font-size: 1.2rem; color: #9cdcfe; }}
    .tree {{ list-style: none; padding-left: 0; }}
    .tree ul {{ padding-left: 1.2rem; border-left: 1px solid #444; margin: 0.1rem 0; }}
    .node summary {{ cursor: pointer; color: #dcdcaa; }}
    .node summary:hover {{ color: #fff; }}
    .op {{ color: #ce9178; }}
    .op-name {{ font-weight: bold; color: #4ec9b0; }}
    .op-summary {{ color: #808080; font-size: 0.9em; }}
    .meta {{ color: #6a9955; margin-bottom: 0.5rem; }}
    .section-title {{ font-size: 1rem; color: #9cdcfe; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
    .exec-list {{ margin-top: 0.5rem; }}
    .exec-row {{ margin: 0.25rem 0; padding: 0.2rem 0; border-bottom: 1px solid #333; }}
    .exec-idx {{ color: #858585; margin-right: 0.5rem; min-width: 2.5rem; display: inline-block; }}
    .exec-path {{ color: #dcdcaa; margin-right: 0.5rem; }}
    .ccls {{ margin-top: 0.35rem; margin-left: 1rem; padding: 0.35rem 0.5rem; display: block; border-left: 3px solid #569cd6; background: #252526; border-radius: 0 4px 4px 0; }}
    .ccl {{ display: block; font-size: 0.9em; }}
    .ccl.pre {{ color: #4ec9b0; }}
    .ccl.redist {{ color: #ce9178; }}
    .ccl.post {{ color: #569cd6; }}
    .flame-meta {{ color: #858585; margin-bottom: 0.5rem; font-size: 0.9em; }}
    .flame-container {{ margin-top: 0.5rem; border: 1px solid #444; border-radius: 4px; overflow: hidden; min-height: 72px; max-height: 420px; }}
    .flame-svg {{ display: block; vertical-align: bottom; }}
    .flame-bar {{ cursor: pointer; }}
    .flame-bar:hover .flame-rect {{ filter: brightness(1.2); }}
    .flame-text {{ fill: #e0e0e0; font-size: 9px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Total events: {total}</p>
  {body}
</body>
</html>
"""


# Global singleton
dispatch_trace = DispatchTrace()


class DispatchTracer:
    """Context manager that enables tracing for a scope and collects entries."""

    def __init__(self):
        self._entries_before: int = 0

    def __enter__(self) -> "DispatchTracer":
        self._entries_before = len(dispatch_trace._entries)
        dispatch_trace.enable()
        return self

    def __exit__(self, *exc):
        dispatch_trace.disable()
        return False

    @property
    def entries(self) -> List[TraceEntry]:
        return dispatch_trace._entries[self._entries_before :]
