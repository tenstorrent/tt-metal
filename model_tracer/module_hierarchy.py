# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture the ``torch.nn.Module`` call stack for every traced ttnn op.

Without this module, ``ttnn_operations_master.json`` knows *which* ops ran
but not *which submodule they came from* — so the perf report can show
"ttnn.matmul ran 96 times" but not "ttnn.matmul ran 32 times in
``decoder.layers.0.self_attn.q_proj``". Per-module attribution is the
substrate for the cytoscape graph view, the reference-model comparison,
and the suggestion engine.

How it works:

1. We patch ``torch.nn.Module.__init__`` to give every constructed Module
   a sequential ``_mt_id`` and install ``register_forward_pre_hook`` /
   ``register_forward_hook`` callbacks that push/pop a per-thread
   module-id stack around each ``forward()`` call. Hooks are the
   documented PyTorch extension surface, so this does not depend on
   private internals.

2. We register a ``_POST_TRACE_OBSERVERS`` callback into
   ``ttnn.operation_tracer``. After every successful op trace it fires
   with ``(op_counter, op_name)``; we snapshot the current module stack
   into an in-memory log keyed by op counter.

3. At session end we walk every root-level Module (one whose own
   ``_mt_id`` is the smallest in its named-modules subtree) via
   ``named_modules()`` and build a ``{mt_id: full_attribute_path}``
   mapping. Combined with the in-memory log this gives us a sidecar
   JSON file ``ttnn_module_hierarchy.json`` that the perf join layer
   reads. Each entry: ``{op_counter, op_name, module_stack:
   [{class_name, attribute_path}]}``.

This module is model-agnostic. It works on any PyTorch model that uses
``nn.Module`` (i.e., everything). It assumes only that ``ttnn`` ops are
issued from inside ``forward()`` calls, which is true for every demo we
care about.

The capture is opt-in: nothing happens until
``install_module_hierarchy_capture()`` is called. The pytest plugin
calls it when ``--trace-params`` is enabled.
"""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# State (per-process; we don't allow more than one active capture)
# ---------------------------------------------------------------------------


_INSTALLED = False
_STATE_LOCK = threading.Lock()

# Sequential id assigned to every nn.Module instance at construction time.
# Stored as the attribute `_mt_id` on the Module so weak-ref collection
# is unnecessary for the id itself.
_NEXT_MT_ID = [0]

# Weak references to every Module instance we've seen, indexed by mt_id.
# Used at session-end to walk named_modules and resolve attribute paths.
_MODULE_REFS: Dict[int, "weakref.ReferenceType[Any]"] = {}

# Per-thread call stack of (mt_id, class_name).
_TLS = threading.local()


def _stack() -> List[Tuple[int, str]]:
    s = getattr(_TLS, "stack", None)
    if s is None:
        s = []
        _TLS.stack = s
    return s


# In-memory log: list of CaptureRecord. We don't lock individual writes
# because traced runs are single-threaded in practice (the GIL plus the
# fact that PyTorch demos rarely use threads); if that changes we can
# add a lock around `_LOG.append`.
@dataclass
class CaptureRecord:
    """One entry per traced ttnn op."""

    op_counter: int
    op_name: str
    # Stack snapshot at the time the op fired, outer-most first.
    # Each entry: (mt_id, class_name). The attribute_path is resolved at
    # session-end from `_MODULE_REFS` so we don't pay that cost per op.
    stack: List[Tuple[int, str]] = field(default_factory=list)


_LOG: List[CaptureRecord] = []


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _pre_hook(module: Any, _input: Any) -> None:
    mt_id = getattr(module, "_mt_id", -1)
    cls = type(module).__name__
    _stack().append((mt_id, cls))


def _post_hook(module: Any, _input: Any, _output: Any) -> None:
    s = _stack()
    if not s:
        return
    # Pop the entry that matches this module's id, to be defensive against
    # the (rare) case where a forward() raises and we never see the
    # post-hook for some intermediate module.
    mt_id = getattr(module, "_mt_id", -1)
    for i in range(len(s) - 1, -1, -1):
        if s[i][0] == mt_id:
            del s[i:]
            return
    # Fall through: nothing matched; pop the top to keep the stack
    # bounded over a long run.
    s.pop()


def _on_op_traced(op_counter: int, op_name: str) -> None:
    # We copy the stack list so a later push/pop doesn't mutate this
    # snapshot. Tuples are immutable, so element-level copy is unneeded.
    _LOG.append(CaptureRecord(op_counter=op_counter, op_name=op_name, stack=list(_stack())))


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------


_ORIGINAL_INIT = None


def install_module_hierarchy_capture() -> None:
    """Idempotently install the module-hierarchy capture.

    Safe to call multiple times — subsequent calls are no-ops. Designed
    to be called once from ``pytest_configure`` when ``--trace-params``
    is set.
    """
    global _INSTALLED, _ORIGINAL_INIT

    with _STATE_LOCK:
        if _INSTALLED:
            return

        import torch.nn as nn

        _ORIGINAL_INIT = nn.Module.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _ORIGINAL_INIT(self, *args, **kwargs)
            mt_id = _NEXT_MT_ID[0]
            _NEXT_MT_ID[0] = mt_id + 1
            # Use setattr instead of self._mt_id = ... so any class that
            # has overridden __setattr__ (some custom nn.Modules do) goes
            # through its normal path.
            try:
                object.__setattr__(self, "_mt_id", mt_id)
            except Exception:
                # Very paranoid fallback for frozen/slotted modules.
                return
            _MODULE_REFS[mt_id] = weakref.ref(self)
            try:
                self.register_forward_pre_hook(_pre_hook)
                self.register_forward_hook(_post_hook)
            except Exception as exc:
                logger.debug(
                    "module_hierarchy: failed to register hooks on {} ({}): {}",
                    type(self).__name__,
                    mt_id,
                    exc,
                )

        nn.Module.__init__ = patched_init  # type: ignore[assignment]

        # Register the op-trace observer. Late import to avoid hard
        # dependency on ttnn being importable at module-load time (we
        # only need it when capture is actually requested).
        try:
            import ttnn.operation_tracer as ot

            ot.register_post_trace_observer(_on_op_traced)
        except ImportError as exc:
            logger.warning(
                "module_hierarchy: ttnn.operation_tracer not importable; " "per-module attribution disabled ({})",
                exc,
            )

        _INSTALLED = True
        logger.debug("module_hierarchy: capture installed")


# ---------------------------------------------------------------------------
# Path resolution + sidecar writer
# ---------------------------------------------------------------------------


def _resolve_attribute_paths() -> Dict[int, str]:
    """For every alive Module we've recorded, compute its attribute path
    relative to the highest-up ancestor we also have a ref to.

    Strategy: from each module ref, walk up via `_modules` parents — but
    PyTorch nn.Module doesn't keep a back-pointer to parents. So instead
    we iterate over all live modules and, for each that has any tracked
    child, do a single `named_modules()` walk to bulk-assign paths. We
    pick the module with the largest tracked-descendant set as the
    "root" for that name-space; everything reachable from it gets a
    relative attribute path. Modules not reached from any root get
    their class name as a fallback path.
    """
    alive: Dict[int, Any] = {}
    for mid, ref in _MODULE_REFS.items():
        m = ref()
        if m is not None:
            alive[mid] = m

    # Build a set of tracked ids that appear as descendants of each
    # candidate root. We approximate this by `id()` (Python id) instead
    # of mt_id since named_modules yields the same instance.
    py_id_to_mt: Dict[int, int] = {id(m): mid for mid, m in alive.items()}

    # Score each module by how many of our tracked modules are reachable
    # via its named_modules walk. The roots are the top-N by score.
    coverage: Dict[int, set] = {}
    for mid, m in alive.items():
        try:
            covered: set = set()
            for _name, sub in m.named_modules():
                sub_mt = py_id_to_mt.get(id(sub))
                if sub_mt is not None:
                    covered.add(sub_mt)
            coverage[mid] = covered
        except Exception:
            coverage[mid] = {mid}

    # Greedy cover: pick the module with the largest uncovered set; mark
    # its descendants assigned; repeat. This handles models that contain
    # multiple top-level Modules (rare but happens — e.g. encoder +
    # decoder + lm_head).
    paths: Dict[int, str] = {}
    remaining = set(alive.keys())
    while remaining:
        best_mid = None
        best_cov = set()
        for mid in remaining:
            cov = coverage.get(mid, set()) & remaining
            if len(cov) > len(best_cov):
                best_cov = cov
                best_mid = mid
        if best_mid is None or not best_cov:
            # No useful coverage left; assign the leftovers their class names.
            for mid in remaining:
                m = alive.get(mid)
                paths[mid] = type(m).__name__ if m is not None else f"<gone:{mid}>"
            break

        root = alive[best_mid]
        try:
            for name, sub in root.named_modules():
                sub_mt = py_id_to_mt.get(id(sub))
                if sub_mt is None:
                    continue
                # Top-level (root itself) gets the class name; everything
                # else gets the relative attribute path.
                attr_path = name if name else type(root).__name__
                paths[sub_mt] = attr_path
        except Exception as exc:
            logger.debug("module_hierarchy: named_modules walk failed: {}", exc)
            paths[best_mid] = type(root).__name__

        remaining -= best_cov
    return paths


def export_module_hierarchy(output_path: Path) -> Optional[Path]:
    """Write ``ttnn_module_hierarchy.json`` next to the tracer master.

    Returns the path written (or None if nothing was captured). Safe to
    call when capture wasn't installed — just writes an empty payload.

    The output schema:

      {
        "schema_version": 1,
        "module_paths": {
          "<mt_id>": "decoder.layers.0.self_attn.q_proj",
          ...
        },
        "module_classes": {
          "<mt_id>": "Linear",
          ...
        },
        "op_module_log": [
          {
            "op_counter": 1,
            "op_name": "ttnn.matmul",
            "stack_mt_ids": [3, 7, 12, 18]
          },
          ...
        ]
      }

    The perf join layer expands `stack_mt_ids` against `module_paths`
    when it constructs `JoinedRow.module_path`.
    """
    paths = _resolve_attribute_paths()
    classes = {}
    for mid, ref in _MODULE_REFS.items():
        m = ref()
        if m is not None:
            classes[str(mid)] = type(m).__name__

    payload = {
        "schema_version": 1,
        "module_paths": {str(mid): p for mid, p in paths.items()},
        "module_classes": classes,
        "op_module_log": [
            {
                "op_counter": rec.op_counter,
                "op_name": rec.op_name,
                "stack_mt_ids": [e[0] for e in rec.stack],
            }
            for rec in _LOG
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "module_hierarchy: wrote {} ({} ops, {} modules)",
        output_path,
        len(_LOG),
        len(paths),
    )
    return output_path


def reset_state_for_tests() -> None:
    """Forget all captured state. Used by tests only."""
    global _INSTALLED, _ORIGINAL_INIT
    with _STATE_LOCK:
        _MODULE_REFS.clear()
        _LOG.clear()
        _NEXT_MT_ID[0] = 0
        _TLS.stack = []
        if _ORIGINAL_INIT is not None:
            import torch.nn as nn

            nn.Module.__init__ = _ORIGINAL_INIT  # type: ignore[assignment]
            _ORIGINAL_INIT = None
        _INSTALLED = False
        try:
            import ttnn.operation_tracer as ot

            ot.unregister_post_trace_observer(_on_op_traced)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Convenience: standalone offline parser
# ---------------------------------------------------------------------------


def load_module_hierarchy(path: Path) -> Dict[str, Any]:
    """Read a previously-written sidecar JSON.

    Returns the parsed dict. Returns an empty payload if the file is
    missing or malformed (so callers can use the same code path for
    "no per-module data available").
    """
    if not path.exists():
        return {"schema_version": 1, "module_paths": {}, "module_classes": {}, "op_module_log": []}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("module_hierarchy: failed to parse {}: {}", path, exc)
        return {"schema_version": 1, "module_paths": {}, "module_classes": {}, "op_module_log": []}
