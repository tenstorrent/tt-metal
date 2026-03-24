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

import json
import logging
import os
import subprocess
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ttml.trainers import SFTTrainer

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
    op_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"op={self.op_name}"]
        parts.append(f"inputs={self.input_layouts}")
        if self.rule_name:
            parts.append(f"rule={self.rule_name}")
        if self.op_kwargs:
            parts.append(f"kwargs={self.op_kwargs}")
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
            "op_kwargs": json_safe(self.op_kwargs),
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

    def _folded_frame(self, name: str) -> str:
        """Sanitize a frame name for folded format (no semicolons)."""
        return str(name).replace(";", "_")

    def to_folded(
        self,
        entries: Optional[List[TraceEntry]] = None,
        time_order: bool = True,
        include_ccls: bool = True,
    ) -> str:
        """Return trace in folded stack format (semicolon-separated frames, space, count).
        Compatible with Brendan Gregg's flamegraph.pl and speedscope.

        When time_order=True (default): one line per event in execution order, count=1.
        X axis = time (execution order). Includes op name and, if include_ccls=True,
        pre_collectives (broadcast), redistributions, and post_collectives (all_reduce) as frames.

        When time_order=False: one line per distinct module+op stack, count = number of ops
        (classic aggregate view). CCLs are not included in this mode.
        """
        if entries is None:
            entries = self._entries
        if not entries:
            return ""

        if time_order:
            # Emit one line per "event" in execution order so X = time
            lines = []
            for e in entries:
                path = list(e.module_stack) if e.module_stack else ["root"]
                path_s = ";".join(self._folded_frame(s) for s in path)

                if include_ccls and e.pre_collectives:
                    for c in e.pre_collectives:
                        ctype = c.get("type", "pre_ccl")
                        axis = c.get("mesh_axis", "")
                        frame = f"{ctype}(axis={axis})" if axis else ctype
                        lines.append(f"{path_s};{self._folded_frame(frame)} 1")

                lines.append(f"{path_s};{self._folded_frame(e.op_name)} 1")

                if include_ccls and e.redistributions:
                    for r in e.redistributions:
                        arg = r.get("arg_idx", "")
                        lines.append(
                            f"{path_s};{self._folded_frame(e.op_name)};redistribute(arg{arg}) 1"
                        )

                if include_ccls and e.post_collectives:
                    for c in e.post_collectives:
                        ctype = c.get("type", "post_ccl")
                        axis = c.get("mesh_axis", "")
                        frame = f"{ctype}(axis={axis})" if axis else ctype
                        lines.append(
                            f"{path_s};{self._folded_frame(e.op_name)};{self._folded_frame(frame)} 1"
                        )

            return "\n".join(lines) + "\n"
        else:
            stacks = []
            for e in entries:
                path = list(e.module_stack) if e.module_stack else ["root"]
                path.append(e.op_name)
                stack = ";".join(self._folded_frame(s) for s in path)
                stacks.append(stack)
            lines = [
                f"{stack} {count}" for stack, count in sorted(Counter(stacks).items())
            ]
            return "\n".join(lines) + "\n" if lines else ""

    def export_folded(
        self,
        path: str,
        entries: Optional[List[TraceEntry]] = None,
        time_order: bool = True,
        include_ccls: bool = True,
    ) -> None:
        """Write folded format to a file. Use with: flamegraph.pl path.folded > path.svg
        By default uses time_order=True, include_ccls=True (X=time, all events including CCLs).
        """
        folded = self.to_folded(
            entries, time_order=time_order, include_ccls=include_ccls
        )
        with open(path, "w") as f:
            f.write(folded)

    def build_flamegraph_svg(
        self,
        entries: Optional[List[TraceEntry]] = None,
        out_path: Optional[str] = None,
        flamegraph_pl: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a classic flame graph SVG from the trace.
        Requires flamegraph.pl (Brendan Gregg's FlameGraph) on PATH or pass path.
        Clone: https://github.com/brendangregg/FlameGraph
        Returns path to generated SVG, or None if flamegraph.pl not found/failed.
        """
        if entries is None:
            entries = self._entries
        if not entries:
            return None
        script = flamegraph_pl or os.environ.get("FLAMEGRAPH_PL", "flamegraph.pl")
        try:
            folded = self.to_folded(entries)
            out_file = out_path or os.path.join(
                tempfile.gettempdir(), "dispatch_flame.svg"
            )
            result = subprocess.run(
                [script, "-t", "modules+ops"],
                input=folded,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                with open(out_file, "w") as f:
                    f.write(result.stdout)
                return out_file
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            logger.debug("flamegraph.pl not used: %s", e)
            return None

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
            if entry.op_kwargs:
                parts.append(f"kwargs={entry.op_kwargs}")
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
            if ent.op_kwargs:
                parts.append(f"kwargs={escape(repr(ent.op_kwargs))}")
            if ent.pre_collectives:
                parts.append("pre_ccl")
            if ent.redistributions:
                parts.append("redist")
            if ent.post_collectives:
                parts.append("post_ccl")
            return " [" + ", ".join(parts) + "]"

        # Execution order: entries in record order with CCLs visible
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
            '<section><h2 class="section-title">By execution order</h2><div class="exec-list">'
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


class DispatchTraceCallback:
    """Trainer callback for dispatch trace logging.

    Args:
        max_entries_to_print: Maximum entries to print at train end (default 20).
        first_step_only: If True, disable tracing after the first step completes.
        dump_path: Optional file path to dump entries. Use .html for an interactive
            HTML report (also writes .folded and _flame.svg); otherwise JSONL.
    """

    def __init__(
        self,
        max_entries_to_print: int = 20,
        first_step_only: bool = False,
        dump_path: Optional[str] = None,
    ):
        self.max_entries_to_print = max_entries_to_print
        self.first_step_only = first_step_only
        self.dump_path = dump_path
        self._first_step_done = False

    def on_train_begin(self, trainer: "SFTTrainer") -> None:
        dispatch_trace.clear()
        dispatch_trace.enable()
        self._first_step_done = False

    def on_step_end(
        self, trainer: "SFTTrainer", step: int, loss: float, lr: float
    ) -> None:
        if self.first_step_only and not self._first_step_done:
            self._first_step_done = True
            dispatch_trace.disable()
            self._dump_and_print("first step")

    def on_train_end(self, trainer: "SFTTrainer") -> None:
        dispatch_trace.disable()
        if not (self.first_step_only and self._first_step_done):
            self._dump_and_print("train end")

    def _dump_and_print(self, label: str) -> None:
        entries = dispatch_trace.entries
        print(f"\nDispatch trace ({label}): {len(entries)} recorded events")
        tree = dispatch_trace.format_entries_tree(
            entries=entries,
            max_entries=self.max_entries_to_print,
        )
        print(tree)
        if len(entries) > self.max_entries_to_print:
            print(
                f"  ... and {len(entries) - self.max_entries_to_print} more (use --debug_dispatch_dump for full trace)"
            )

        if self.dump_path:
            if self.dump_path.endswith(".html"):
                html = dispatch_trace.format_entries_html(
                    entries=entries, title="Dispatch trace"
                )
                with open(self.dump_path, "w") as f:
                    f.write(html)
                print(f"  Wrote HTML trace to {self.dump_path} (open in a browser)")
                base = self.dump_path[:-5]
                folded_path = base + ".folded"
                dispatch_trace.export_folded(folded_path, entries=entries)
                print(
                    f"  Wrote folded stacks to {folded_path} (use flamegraph.pl for classic SVG)"
                )
                svg_path = base + "_flame.svg"
                if dispatch_trace.build_flamegraph_svg(
                    entries=entries, out_path=svg_path
                ):
                    print(f"  Wrote flame graph to {svg_path} (open in browser)")
            else:
                with open(self.dump_path, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry.to_dict()) + "\n")
                print(f"  Dumped {len(entries)} entries to {self.dump_path} (JSONL)")
