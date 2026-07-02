"""Generic TT-side per-submodule probe.

This module is loaded INSIDE the demo subprocess (not from the cli). It
walks whatever TT-side model object the demo built and, for each
sub-object that looks like a callable layer, wraps its ``__call__`` to
capture per-call output statistics. The captured stats are dumped to a
JSON sidecar that the cli reads after the demo exits.

Why this is generic
-------------------
The probe knows nothing about ``tt_transformers``, ``Whisper``, ``SAM2``,
or any other specific backbone. It works by reflection on whatever
attribute structure the TT model exposes:

* Anything whose ``type(obj).__name__`` does not contain the word
  ``Tensor`` (filter out ttnn tensor attributes) AND that is callable
  (``hasattr(obj, "__call__")``) AND that has at least one identifiable
  sub-attribute (heuristic for "this is a module not a primitive op")
  is a candidate.
* For each candidate we record ``id(obj)``, the access path
  (``model.layers[0].attention``), the class name (e.g. ``Attention``),
  and install a wrapper around ``__call__``.
* The wrapper captures the FIRST positional or keyword tensor in the
  output and reduces it to scalar stats (mean, std, l2, abs_max).

What gets dumped
----------------
A list of records, one per call::

    [
      {
        "qualified_name": "model.layers.0",
        "class_name": "TransformerBlock",
        "step": 0,
        "shape": [1, 1, 4096],
        "dtype": "bfloat16",
        "mean": 1.23e-3,
        "std": 4.56e-1,
        "l2": 1.4e+2,
        "abs_max": 7.8e+0
      },
      ...
    ]

Activation
----------
The cli activates this probe by setting two env vars before invoking the
demo subprocess:

* ``TT_PLANNER_PROBE_OUTPUT`` -- absolute path to the JSON sidecar to
  write. Empty / unset = probe disabled.
* ``TT_PLANNER_PROBE_DEPTH`` -- (optional) max depth of attribute walks.
  Defaults to 4. Deeper walks hook more modules but slow demo
  startup. 4 covers ``model.layers[i].attention.q_proj`` which is
  typically enough.

The probe is opt-in and fail-soft: if anything goes wrong during
install the demo runs unmodified, the JSON sidecar is empty, and the
cli falls back to HF-only signal (which is what the legacy loop has
today).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


_RECORDS: List[Dict[str, Any]] = []
_STEP_COUNTERS: Dict[str, int] = {}
_VISITED_IDS: Set[int] = set()
_OUTPUT_PATH: Optional[str] = None
_VERBOSE: bool = False


_PROBE_DEVICE: Any = None


_INSTALLED_ONCE: bool = False


_TRACE_ACTIVE: bool = False
_TRACE_PATCHED: bool = False


def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"[tt_probe] {msg}", file=sys.stderr, flush=True)


def _install_trace_guards() -> None:
    """Wrap ``ttnn.begin_trace_capture`` / ``ttnn.end_trace_capture`` so we
    can observe when the demo is inside a trace-capture region. Idempotent;
    no-op if ttnn isn't importable. The wraps PRESERVE original behavior;
    they only toggle a module-level flag that the per-layer wrapper reads.
    """
    global _TRACE_PATCHED
    if _TRACE_PATCHED:
        return
    try:
        import ttnn
    except Exception as e:
        _log(f"trace guard install skipped: ttnn import failed ({e})")
        return

    orig_begin = getattr(ttnn, "begin_trace_capture", None)
    orig_end = getattr(ttnn, "end_trace_capture", None)
    if orig_begin is None or orig_end is None:
        _log("trace guard install skipped: ttnn lacks begin/end_trace_capture")
        return

    def _wrapped_begin(*args, **kwargs):
        global _TRACE_ACTIVE
        _TRACE_ACTIVE = True
        _log("trace capture: ENTER (probe will skip device reads)")
        return orig_begin(*args, **kwargs)

    def _wrapped_end(*args, **kwargs):
        global _TRACE_ACTIVE
        try:
            return orig_end(*args, **kwargs)
        finally:
            _TRACE_ACTIVE = False
            _log("trace capture: EXIT (probe stat-collection re-enabled)")

    try:
        ttnn.begin_trace_capture = _wrapped_begin
        ttnn.end_trace_capture = _wrapped_end
        _TRACE_PATCHED = True
        _log("trace guards installed on ttnn.begin/end_trace_capture")
    except Exception as e:
        _log(f"trace guard install failed: {type(e).__name__}: {e}")


def is_trace_active() -> bool:
    """Public accessor used by tests / the smoke check."""
    return _TRACE_ACTIVE


def _tensor_stats(t: Any) -> Optional[Dict[str, Any]]:
    """Reduce any torch/ttnn tensor (or tensor-likable) to scalar stats.
    Delegates to :func:`activation_diff._tensor_stats` after converting
    ttnn->torch on a best-effort basis."""
    if t is None:
        return None
    try:
        from scripts.tt_hw_planner.activation_diff import (
            _tensor_stats as _base_stats,
            _to_torch_best_effort,
        )
    except Exception:
        return None
    tt = _to_torch_best_effort(t)
    if tt is None:
        return None
    d = _base_stats(tt)
    if d is None:
        return None

    d2 = dict(d)
    if "shape" in d2:
        d2["shape"] = list(d2["shape"])
    return d2


def _extract_first_tensor(out: Any) -> Any:
    """Pull the first tensor out of a (possibly nested) layer output.
    Thin wrapper around :func:`activation_diff._to_torch_best_effort`
    which already handles tuple/list/dict/ttnn/torch."""
    try:
        from scripts.tt_hw_planner.activation_diff import _to_torch_best_effort
    except Exception:
        return None
    return _to_torch_best_effort(out, device=_PROBE_DEVICE)


def _make_wrapper(
    qualified_name: str,
    class_name: str,
    original_call: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a callable so each invocation records output stats."""

    def _wrapped(*args, **kwargs):
        out = original_call(*args, **kwargs)

        if _TRACE_ACTIVE:
            return out
        try:
            step = _STEP_COUNTERS.get(qualified_name, 0)
            _STEP_COUNTERS[qualified_name] = step + 1
            t = _extract_first_tensor(out)
            stats = _tensor_stats(t)
            if stats is not None:
                rec = {
                    "qualified_name": qualified_name,
                    "class_name": class_name,
                    "step": step,
                    **stats,
                }
                _RECORDS.append(rec)
        except Exception:
            pass
        return out

    return _wrapped


def _layerish_suffixes() -> Tuple[str, ...]:
    """Lazy import so this module can be loaded inside the demo
    subprocess even if scripts.* isn't on sys.path yet at load time."""
    try:
        from scripts.tt_hw_planner.module_tree import _HIGH_LEVEL_SUFFIXES

        return _HIGH_LEVEL_SUFFIXES
    except Exception:
        return (
            "Block",
            "Layer",
            "Attention",
            "MLP",
            "FeedForward",
            "Encoder",
            "Decoder",
            "Transformer",
            "Embedding",
            "Embeddings",
            "PatchEmbed",
            "Head",
            "Norm",
            "Stage",
            "Mixer",
        )


def _looks_layerish(class_name: str) -> bool:
    """True if the class name carries a layer-like suffix. Delegates
    to :func:`module_tree._looks_high_level` so the HF probe, TT probe,
    and module-tree decomposition all share one definition of
    "compute layer"."""
    try:
        from scripts.tt_hw_planner.module_tree import _looks_high_level

        return _looks_high_level(class_name)
    except Exception:
        return any(class_name.endswith(s) for s in _layerish_suffixes())


def _walk_and_wrap(
    obj: Any,
    qualified_name: str,
    depth: int,
    max_depth: int,
    *,
    on_wrap: Callable[[str, str], None],
) -> None:
    """Recurse into ``obj`` up to ``max_depth`` levels, wrapping every
    sub-object whose class name looks layer-like.

    The recursion is breadth-first by attribute order; we visit each
    Python object at most once (``_VISITED_IDS``) so a backbone that
    cross-references modules doesn't double-wrap."""
    if depth > max_depth:
        return
    if id(obj) in _VISITED_IDS:
        return
    _VISITED_IDS.add(id(obj))

    cls_name = type(obj).__name__

    if _looks_layerish(cls_name) and depth > 0:
        _ENTRY_METHODS = (
            "forward",
            "ttnn_prefill_forward",
            "ttnn_decode_forward",
            "prefill_forward",
            "prefill_forward_single_user_text",
            "_apply_norm_and_lm_head",
            "transform_and_embed_prefill_inputs_device",
            "process_output_prefill",
            "process_hidden_states_after_prefill_trace",
        )
        for _mname in _ENTRY_METHODS:
            _m = getattr(obj, _mname, None)
            if _m is None or not callable(_m):
                continue
            try:
                wrapped_m = _make_wrapper(qualified_name, cls_name, _m)
                try:
                    object.__setattr__(obj, _mname, wrapped_m)
                    on_wrap(f"{qualified_name}.{_mname}" if _mname != "forward" else qualified_name, cls_name)
                except (AttributeError, TypeError):
                    if _mname == "forward":
                        try:
                            cls = type(obj)
                            orig_class_call = cls.__call__
                            wrapped_class_call = _make_wrapper(qualified_name, cls_name, orig_class_call)
                            cls.__call__ = wrapped_class_call
                            on_wrap(qualified_name, cls_name)
                        except Exception:
                            pass
            except Exception:
                continue

    try:
        attrs = list(vars(obj).items())
    except TypeError:
        attrs = []
    for name, child in attrs:
        if name.startswith("_"):
            continue
        if id(child) in _VISITED_IDS:
            continue
        child_qn = f"{qualified_name}.{name}" if qualified_name else name
        child_cls = type(child).__name__

        if isinstance(child, (int, float, str, bool, bytes, type(None))):
            continue

        if isinstance(child, (list, tuple)):
            for idx, sub in enumerate(child):
                if isinstance(sub, (int, float, str, bool, bytes)):
                    continue
                sub_qn = f"{child_qn}.{idx}"
                _walk_and_wrap(sub, sub_qn, depth + 1, max_depth, on_wrap=on_wrap)
            continue
        if isinstance(child, dict):
            for k, sub in child.items():
                if isinstance(sub, (int, float, str, bool, bytes)):
                    continue
                sub_qn = f"{child_qn}.{k}"
                _walk_and_wrap(sub, sub_qn, depth + 1, max_depth, on_wrap=on_wrap)
            continue

        _walk_and_wrap(child, child_qn, depth + 1, max_depth, on_wrap=on_wrap)


def install_probe(
    tt_model: Any,
    *,
    output_path: Optional[str] = None,
    max_depth: int = 4,
    verbose: bool = False,
) -> int:
    """Install per-module wrappers on ``tt_model``. Returns the number
    of modules wrapped.

    Idempotent: calling twice resets the records and re-wraps from
    scratch."""
    global _RECORDS, _STEP_COUNTERS, _VISITED_IDS, _OUTPUT_PATH, _VERBOSE, _PROBE_DEVICE, _INSTALLED_ONCE
    if _INSTALLED_ONCE:
        _log("install_probe skipped (already installed in this process)")
        return 0
    _INSTALLED_ONCE = True
    _RECORDS = []
    _STEP_COUNTERS = {}
    _VISITED_IDS = set()
    _OUTPUT_PATH = output_path or os.environ.get("TT_PLANNER_PROBE_OUTPUT") or None
    _VERBOSE = verbose or bool(os.environ.get("TT_PLANNER_PROBE_VERBOSE"))

    _PROBE_DEVICE = (
        getattr(tt_model, "mesh_device", None)
        or getattr(tt_model, "device", None)
        or getattr(tt_model, "_mesh_device", None)
    )

    _install_trace_guards()

    wrapped_count = [0]

    def _on_wrap(qn: str, cls: str) -> None:
        wrapped_count[0] += 1
        _log(f"wrap  {qn}  ({cls})")

    try:
        _walk_and_wrap(
            tt_model,
            qualified_name="model",
            depth=0,
            max_depth=max_depth,
            on_wrap=_on_wrap,
        )
    except Exception as exc:
        _log(f"install_probe failed: {type(exc).__name__}: {exc}\n" + traceback.format_exc())
    _log(f"wrapped {wrapped_count[0]} modules; output -> {_OUTPUT_PATH}")
    return wrapped_count[0]


def flush_probe() -> Optional[str]:
    """Dump accumulated records to the configured output path. Returns
    the path written or None on failure / no path configured."""
    if not _OUTPUT_PATH:
        return None
    try:
        meta = {
            "version": 1,
            "captured_at": time.time(),
            "count": len(_RECORDS),
            "records": _RECORDS,
        }
        with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        _log(f"flushed {len(_RECORDS)} records -> {_OUTPUT_PATH}")
        return _OUTPUT_PATH
    except Exception as exc:
        _log(f"flush failed: {type(exc).__name__}: {exc}")
        return None


def maybe_install_global(tt_model: Any) -> Optional[int]:
    """Convenience entry point used from inside the demo (after the TT
    model is constructed). Returns the wrap count if the probe was
    activated by the cli, else None.

    Caller is expected to register :func:`flush_probe` with ``atexit``
    so records are written even if the demo crashes."""
    out = os.environ.get("TT_PLANNER_PROBE_OUTPUT")
    if not out:
        return None
    try:
        depth = int(os.environ.get("TT_PLANNER_PROBE_DEPTH", "4"))
    except ValueError:
        depth = 4
    n = install_probe(
        tt_model,
        output_path=out,
        max_depth=depth,
        verbose=bool(os.environ.get("TT_PLANNER_PROBE_VERBOSE")),
    )
    try:
        import atexit

        atexit.register(flush_probe)
    except Exception:
        pass
    return n


__all__ = [
    "flush_probe",
    "install_probe",
    "maybe_install_global",
]
