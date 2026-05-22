# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime capture instrumentation for the dots.ocr TTNN pipeline.

Captures every TTNN op invocation + every ``TTNNModule`` call during a real
run of ``test_dots_ocr_text`` / ``test_dots_ocr_vision`` so the bottom-up unit
test matrix can be auto-generated.

Activation is **strictly opt-in** via ``DOTS_OCR_CAPTURE_SHAPES``. When the
env var is unset (or ``"0"`` / ``"false"``), :func:`install_capture` is a
no-op and zero wrappers are installed — production performance is not
perturbed.

Three-layer capture (matches Plan §2.1):

* **Layer A** — monkey-patch ``TTNNModule.__call__`` (``core/module.py``)
  to log module entry/exit with input/output tensor metadata.
* **Layer B** — monkey-patch ``DispatchManager.record_timing``
  (``core/run_config.py``) to attach shape/dtype info for ops dispatched
  through the symbiote ``__torch_dispatch__`` path.
* **Layer C** — wrap an allowlist of direct ``ttnn.*`` callables (e.g.
  ``ttnn.linear``, ``ttnn.matmul``, ``ttnn.experimental.rotary_embedding``)
  with recording proxies that read tensor metadata pre/post call.

Output JSON files (written on :func:`uninstall_capture`):

* ``{out_dir}/{phase_tag}_ops.json``                — every op record
* ``{out_dir}/{phase_tag}_ops_dedup.json``          — deduped per Plan §3.1
* ``{out_dir}/{phase_tag}_modules.json``            — every module record
* ``{out_dir}/{phase_tag}_modules_dedup.json``      — deduped per Plan §3.2

Safety:

* Wrappers never call ``ttnn.to_torch`` — only inspect ``shape`` /
  ``dtype`` / ``layout`` / ``memory_config``. Reading host data inside a
  wrapper would corrupt timing and break under trace.
* A module-level lock guards the shared record buffers.
* All wrappers degrade gracefully (record ``"<unserializable: ...>"``)
  rather than raising — a capture run must never crash the real test.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import traceback
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()

_ACTIVE: bool = False
_OUT_DIR: Optional[str] = None
_PHASE_TAG: str = "warmup"

# Buffers
_OP_RECORDS: List[Dict[str, Any]] = []
_MODULE_RECORDS: List[Dict[str, Any]] = []

# Counter for stable call_ids across both layers
_CALL_COUNTER: int = 0

# Originals saved here so we can restore on uninstall.
_ORIGINAL_TTNN_MODULE_CALL = None
_ORIGINAL_RECORD_TIMING = None
_ORIGINAL_TTNN_OPS: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
# Map module_name -> path string (qualified module hierarchy)
_MODULE_PATH_STACK: List[str] = []


def is_active() -> bool:
    """Return True if capture is currently installed."""
    return _ACTIVE


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def _next_call_id() -> int:
    global _CALL_COUNTER
    with _LOCK:
        _CALL_COUNTER += 1
        return _CALL_COUNTER


# ---------------------------------------------------------------------------
# Layer C — direct ttnn op allowlist (Plan §2.1)
# ---------------------------------------------------------------------------

# (module_attribute_path, attr_name) pairs. We resolve module_attribute_path
# against the ``ttnn`` package at install time.
_TTNN_OP_ALLOWLIST: Tuple[Tuple[str, str], ...] = (
    ("", "linear"),
    ("", "matmul"),
    ("", "rms_norm"),
    ("", "rms_norm_pre_all_gather"),
    ("", "rms_norm_post_all_gather"),
    ("", "layer_norm"),
    ("", "all_gather"),
    ("", "reduce_scatter"),
    ("", "embedding"),
    ("transformer", "scaled_dot_product_attention"),
    ("experimental", "rotary_embedding"),
    ("experimental", "nlp_create_qkv_heads"),
    ("experimental", "nlp_concat_heads"),
    ("experimental", "nlp_concat_heads_decode"),
    ("", "argmax"),
    ("", "add"),
    ("", "mul"),
    ("", "where"),
    ("", "typecast"),
    ("", "concat"),
    ("", "slice"),
    ("", "pad"),
)


# ---------------------------------------------------------------------------
# Tensor metadata serialization helpers
# ---------------------------------------------------------------------------


def _safe_repr(x: Any) -> str:
    try:
        s = repr(x)
        return s if len(s) < 1024 else s[:1024] + "..."
    except Exception as exc:  # pragma: no cover - defensive
        return f"<unserializable: {type(x).__name__}: {exc}>"


def _public_attrs(obj: Any) -> Dict[str, Any]:
    """Return non-private, non-callable public attrs of obj as a dict.

    All values are stringified via ``_safe_repr`` so the result is JSON-safe.
    """
    out: Dict[str, Any] = {}
    try:
        names = [n for n in dir(obj) if not n.startswith("_")]
    except Exception as exc:  # pragma: no cover - defensive
        return {"<dir_failed>": str(exc)}
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if callable(v):
            continue
        out[n] = _safe_repr(v)
    return out


def _shape_list(t: Any) -> Optional[List[int]]:
    try:
        sh = t.shape
        # ttnn.Shape is iterable; torch.Size is iterable.
        return [int(d) for d in sh]
    except Exception:
        return None


def _dtype_str(t: Any) -> Optional[str]:
    try:
        d = t.dtype
        return str(d)
    except Exception:
        return None


def _layout_str(t: Any) -> Optional[str]:
    try:
        return str(t.layout)
    except Exception:
        return None


def _memory_config_dict(t: Any) -> Dict[str, Any]:
    """Best-effort serialization of a ttnn.Tensor's memory_config.

    If the tensor is on host or memory_config() raises, return a stub with
    ``"buffer_type": "HOST"`` (or the failure reason).
    """
    try:
        mc = t.memory_config()
    except Exception as exc:
        return {"buffer_type": "HOST", "error": f"{type(exc).__name__}: {exc}"}
    if mc is None:
        return {"buffer_type": "HOST"}
    out: Dict[str, Any] = {"repr": _safe_repr(mc)}
    # Best-effort public attrs of MemoryConfig
    for attr in ("buffer_type", "memory_layout"):
        try:
            out[attr] = _safe_repr(getattr(mc, attr))
        except Exception:
            pass
    try:
        ss = getattr(mc, "shard_spec", None)
        if ss is not None:
            out["shard_spec"] = _safe_repr(ss)
    except Exception:
        pass
    return out


def _mesh_info(t: Any) -> Dict[str, Any]:
    """Best-effort extraction of mesh / multi-device metadata."""
    info: Dict[str, Any] = {}
    try:
        dev = t.device()
    except Exception:
        dev = None
    if dev is not None:
        for attr in ("shape", "num_devices"):
            try:
                v = getattr(dev, attr, None)
                if callable(v):
                    v = v()
                if v is not None:
                    info[f"device_{attr}"] = _safe_repr(v)
            except Exception:
                pass
    try:
        storage_type = t.storage_type()
        info["storage_type"] = _safe_repr(storage_type)
    except Exception:
        pass
    return info


def _tensor_metadata(t: Any) -> Dict[str, Any]:
    """Serialize a tensor (ttnn.Tensor or torch.Tensor) to JSON-safe metadata.

    NEVER reads host data — only descriptive attributes.
    """
    # Avoid importing ttnn at top-level (lazy) to keep this module
    # importable in non-ttnn contexts; but install_capture imports it anyway.
    try:
        import ttnn as _ttnn  # noqa: WPS433  (intentional lazy import)
    except Exception:
        _ttnn = None  # type: ignore[assignment]

    try:
        import torch as _torch  # noqa: WPS433
    except Exception:
        _torch = None  # type: ignore[assignment]

    # torch.Tensor branch (e.g. embedding inputs are still torch)
    if _torch is not None and isinstance(t, _torch.Tensor):
        return {
            "kind": "torch.Tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
        }

    if _ttnn is not None and isinstance(t, _ttnn.Tensor):
        out: Dict[str, Any] = {
            "kind": "ttnn.Tensor",
            "shape": _shape_list(t),
            "dtype": _dtype_str(t),
            "layout": _layout_str(t),
            "memory_config": _memory_config_dict(t),
        }
        out.update(_mesh_info(t))
        return out

    # Anything else (numbers, dicts, etc.)
    return {"kind": type(t).__name__, "repr": _safe_repr(t)}


def _walk_args(args: Any) -> List[Any]:
    """Yield tensor-like leaves from nested args. Non-tensors are stringified."""

    out: List[Any] = []

    def _walk(x: Any) -> None:
        try:
            import ttnn as _ttnn  # noqa: WPS433
        except Exception:
            _ttnn = None  # type: ignore[assignment]
        try:
            import torch as _torch  # noqa: WPS433
        except Exception:
            _torch = None  # type: ignore[assignment]

        is_tensor = False
        if _ttnn is not None and isinstance(x, _ttnn.Tensor):
            is_tensor = True
        elif _torch is not None and isinstance(x, _torch.Tensor):
            is_tensor = True

        if is_tensor:
            out.append(_tensor_metadata(x))
        elif isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
        elif isinstance(x, dict):
            for item in x.values():
                _walk(item)
        else:
            # Skip scalars/None silently; we record kwargs separately.
            pass

    _walk(args)
    return out


def _serialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            import ttnn as _ttnn  # noqa: WPS433
        except Exception:
            _ttnn = None  # type: ignore[assignment]
        try:
            import torch as _torch  # noqa: WPS433
        except Exception:
            _torch = None  # type: ignore[assignment]

        if _ttnn is not None and isinstance(v, _ttnn.Tensor):
            out[k] = _tensor_metadata(v)
            continue
        if _torch is not None and isinstance(v, _torch.Tensor):
            out[k] = _tensor_metadata(v)
            continue
        # Detect program_config / compute_kernel_config style structs
        if v is not None and not isinstance(v, (int, float, bool, str, list, tuple, dict)):
            try:
                kind = type(v).__name__
                out[k] = {"kind": kind, "fields": _public_attrs(v), "repr": _safe_repr(v)}
                continue
            except Exception:
                pass
        # Primitive
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = _safe_repr(v)
    return out


# ---------------------------------------------------------------------------
# Layer C — ttnn op recording proxy
# ---------------------------------------------------------------------------


def _recording_proxy(qualified_name: str, original: Callable) -> Callable:
    """Wrap a single ttnn callable. Records pre-call args, post-call output."""

    def _proxy(*args: Any, **kwargs: Any) -> Any:
        call_id = _next_call_id()
        # Capture inputs BEFORE the call — output tensors don't exist yet.
        in_records: List[Any] = []
        kw_records: Dict[str, Any] = {}
        module_path: Optional[str] = None
        try:
            in_records = _walk_args(args)
            kw_records = _serialize_kwargs(kwargs)
            with _LOCK:
                module_path = _MODULE_PATH_STACK[-1] if _MODULE_PATH_STACK else None
        except Exception as exc:  # never let capture crash the run
            in_records = [{"<capture_error>": f"{type(exc).__name__}: {exc}"}]

        result = original(*args, **kwargs)

        try:
            out_records = _walk_args(result if isinstance(result, (list, tuple)) else (result,))
            record = {
                "call_id": call_id,
                "phase": _PHASE_TAG,
                "op": qualified_name,
                "module_path": module_path,
                "source": "direct",
                "inputs": in_records,
                "kwargs": kw_records,
                "output": out_records,
            }
            with _LOCK:
                _OP_RECORDS.append(record)
        except Exception as exc:  # never let capture crash the run
            with _LOCK:
                _OP_RECORDS.append(
                    {
                        "call_id": call_id,
                        "phase": _PHASE_TAG,
                        "op": qualified_name,
                        "module_path": module_path,
                        "source": "direct",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return result

    _proxy.__wrapped__ = original  # type: ignore[attr-defined]
    _proxy.__name__ = f"_capture_proxy_{qualified_name.replace('.', '_')}"
    return _proxy


def _resolve_ttnn_attr(ttnn_mod: Any, sub_path: str, name: str) -> Tuple[Any, str]:
    """Return ``(parent_obj, qualified_name)`` for ``ttnn[.sub_path].name``."""
    parent = ttnn_mod
    parts = []
    if sub_path:
        for p in sub_path.split("."):
            parent = getattr(parent, p)
            parts.append(p)
    parts.append(name)
    return parent, "ttnn." + ".".join(parts)


# ---------------------------------------------------------------------------
# Layer A — TTNNModule.__call__ wrapper
# ---------------------------------------------------------------------------


def _make_module_call_wrapper(original_call: Callable) -> Callable:
    def _wrapped_call(self, *args: Any, **kwds: Any) -> Any:  # noqa: ANN001
        # Resolve module identity defensively (avoid raising before we even
        # call the wrapped function).
        try:
            module_name = self.module_name
        except Exception:
            module_name = f"{type(self).__name__}_{id(self)}"
        module_class = type(self).__name__
        call_id = _next_call_id()

        with _LOCK:
            parent_path = _MODULE_PATH_STACK[-1] if _MODULE_PATH_STACK else None
            qualified_path = (
                f"{parent_path}/{module_class}[{module_name}]"
                if parent_path is not None
                else f"{module_class}[{module_name}]"
            )
            _MODULE_PATH_STACK.append(qualified_path)

        in_records: List[Any] = []
        try:
            in_records = _walk_args(args)
            in_records.extend(_walk_args(kwds))
        except Exception:
            pass

        try:
            result = original_call(self, *args, **kwds)
        finally:
            with _LOCK:
                if _MODULE_PATH_STACK and _MODULE_PATH_STACK[-1] == qualified_path:
                    _MODULE_PATH_STACK.pop()

        try:
            out_records = _walk_args(result if isinstance(result, (list, tuple)) else (result,))
            record = {
                "call_id": call_id,
                "phase": _PHASE_TAG,
                "module_class": module_class,
                "module_name": module_name,
                "module_path": qualified_path,
                "inputs": in_records,
                "output": out_records,
            }
            with _LOCK:
                _MODULE_RECORDS.append(record)
        except Exception as exc:  # pragma: no cover
            with _LOCK:
                _MODULE_RECORDS.append(
                    {
                        "call_id": call_id,
                        "phase": _PHASE_TAG,
                        "module_class": module_class,
                        "module_name": module_name,
                        "module_path": qualified_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return result

    return _wrapped_call


# ---------------------------------------------------------------------------
# Layer B — DispatchManager.record_timing wrapper
# ---------------------------------------------------------------------------


def _make_record_timing_wrapper(original_record_timing: Callable) -> Callable:
    def _wrapped(backend: str, module_name: str, func_name: str, attrs: dict, duration: float) -> None:
        # Always forward to the original first so existing timing behavior is
        # unaffected.
        try:
            original_record_timing(backend, module_name, func_name, attrs, duration)
        except Exception:
            raise

        if backend != "TTNN":
            return
        # ``__call__`` recording artifacts use the suffixes below; skip them
        # to avoid noisy duplicates of every module call from Layer B.
        if (
            func_name.endswith("_preprocess_weights")
            or func_name.endswith("_move_weights_to_device")
            or func_name.endswith("_forward")
        ):
            return
        try:
            with _LOCK:
                module_path = _MODULE_PATH_STACK[-1] if _MODULE_PATH_STACK else None
                _OP_RECORDS.append(
                    {
                        "call_id": _next_call_id(),
                        "phase": _PHASE_TAG,
                        "op": func_name,
                        "module_path": module_path,
                        "dispatcher_module_name": module_name,
                        "source": "dispatch_to_ttnn_wrapper",
                        # No tensor shapes available here — the dispatcher
                        # already lost that info by the time it calls
                        # record_timing. Layer C covers shape capture for
                        # direct ttnn.* calls; this row exists so the
                        # `__torch_dispatch__` path is at least visible.
                        "attrs": _serialize_kwargs(attrs or {}),
                    }
                )
        except Exception:
            pass  # never crash the run

    return _wrapped


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_capture(out_dir: str, *, phase_tag: str = "warmup") -> None:
    """Activate capture. No-op if ``DOTS_OCR_CAPTURE_SHAPES`` is not truthy."""
    global _ACTIVE, _OUT_DIR, _PHASE_TAG
    global _ORIGINAL_TTNN_MODULE_CALL, _ORIGINAL_RECORD_TIMING

    if not _truthy(os.environ.get("DOTS_OCR_CAPTURE_SHAPES")):
        return

    if _ACTIVE:
        # Idempotent — re-installing is a no-op (but update phase tag).
        _PHASE_TAG = phase_tag
        return

    _OUT_DIR = out_dir
    _PHASE_TAG = phase_tag

    # Ensure out_dir exists early so failures fail fast.
    os.makedirs(out_dir, exist_ok=True)

    # Layer A — patch TTNNModule.__call__
    from models.experimental.tt_symbiote.core.module import TTNNModule

    _ORIGINAL_TTNN_MODULE_CALL = TTNNModule.__call__
    TTNNModule.__call__ = _make_module_call_wrapper(_ORIGINAL_TTNN_MODULE_CALL)

    # Layer B — patch DispatchManager.record_timing
    from models.experimental.tt_symbiote.core.run_config import DispatchManager

    _ORIGINAL_RECORD_TIMING = DispatchManager.record_timing
    DispatchManager.record_timing = staticmethod(_make_record_timing_wrapper(_ORIGINAL_RECORD_TIMING))

    # Layer C — wrap allowlisted ttnn ops
    import ttnn as _ttnn

    _ORIGINAL_TTNN_OPS.clear()
    for sub_path, name in _TTNN_OP_ALLOWLIST:
        try:
            parent, qualified = _resolve_ttnn_attr(_ttnn, sub_path, name)
            original = getattr(parent, name)
        except AttributeError:
            continue
        _ORIGINAL_TTNN_OPS[(sub_path, name)] = (parent, original)
        try:
            setattr(parent, name, _recording_proxy(qualified, original))
        except Exception as exc:  # some namespaces may forbid setattr
            print(f"[dots_ocr capture] WARN: could not wrap {qualified}: {exc}")
            _ORIGINAL_TTNN_OPS.pop((sub_path, name), None)

    _ACTIVE = True
    print(
        f"[dots_ocr capture] installed: phase_tag={phase_tag!r} out_dir={out_dir!r} "
        f"ttnn_ops_wrapped={len(_ORIGINAL_TTNN_OPS)}"
    )


def uninstall_capture() -> None:
    """Restore originals and flush JSON records to ``out_dir``."""
    global _ACTIVE, _ORIGINAL_TTNN_MODULE_CALL, _ORIGINAL_RECORD_TIMING

    if not _ACTIVE:
        return

    # Restore Layer C
    for (_sub_path, _name), (parent, original) in _ORIGINAL_TTNN_OPS.items():
        try:
            setattr(parent, _name, original)
        except Exception as exc:  # pragma: no cover
            print(f"[dots_ocr capture] WARN: failed to restore ttnn.{_sub_path}.{_name}: {exc}")
    _ORIGINAL_TTNN_OPS.clear()

    # Restore Layer B
    if _ORIGINAL_RECORD_TIMING is not None:
        try:
            from models.experimental.tt_symbiote.core.run_config import DispatchManager

            DispatchManager.record_timing = staticmethod(_ORIGINAL_RECORD_TIMING)
        except Exception as exc:  # pragma: no cover
            print(f"[dots_ocr capture] WARN: failed to restore record_timing: {exc}")
        _ORIGINAL_RECORD_TIMING = None

    # Restore Layer A
    if _ORIGINAL_TTNN_MODULE_CALL is not None:
        try:
            from models.experimental.tt_symbiote.core.module import TTNNModule

            TTNNModule.__call__ = _ORIGINAL_TTNN_MODULE_CALL
        except Exception as exc:  # pragma: no cover
            print(f"[dots_ocr capture] WARN: failed to restore TTNNModule.__call__: {exc}")
        _ORIGINAL_TTNN_MODULE_CALL = None

    _ACTIVE = False

    # Flush JSON
    try:
        _flush_records()
    except Exception as exc:  # pragma: no cover
        print(f"[dots_ocr capture] ERROR flushing JSON: {exc}\n{traceback.format_exc()}")


@contextlib.contextmanager
def set_phase(name: str):  # noqa: D401 - dual-use API
    """Update the phase tag attached to subsequent records.

    Usable both as a plain setter (``set_phase("decode_warmup")``) and as a
    context manager (``with set_phase("decode_warmup"): ...``).
    """
    global _PHASE_TAG
    prev = _PHASE_TAG
    _PHASE_TAG = name
    try:
        yield name
    finally:
        _PHASE_TAG = prev


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _dedup_key_for_op(rec: Dict[str, Any]) -> Tuple:
    """Per Plan §3.1. Excludes program_config_fields and compute_kernel
    fields from the key so they don't blow up the matrix.
    """

    def _tensor_key(t: Dict[str, Any]) -> Tuple:
        if not isinstance(t, dict):
            return (_safe_repr(t),)
        mc = t.get("memory_config") or {}
        return (
            tuple(t.get("shape") or ()),
            t.get("dtype"),
            t.get("layout"),
            mc.get("buffer_type"),
            mc.get("memory_layout"),
        )

    inputs = rec.get("inputs") or []
    input_keys = tuple(_tensor_key(t) for t in inputs)
    kwargs = rec.get("kwargs") or {}
    # Extract program_config / compute_kernel_config kinds only (not the
    # nested fields, per Plan §3.1).
    pc = kwargs.get("program_config")
    ckc = kwargs.get("compute_kernel_config")
    pc_kind = pc.get("kind") if isinstance(pc, dict) else None
    ckc_kind = ckc.get("kind") if isinstance(ckc, dict) else None
    math_fidelity = None
    if isinstance(ckc, dict):
        fields = ckc.get("fields") or {}
        math_fidelity = fields.get("math_fidelity")
    # Mesh shape — pull from first input if present.
    mesh_shape = None
    num_devices = None
    if inputs and isinstance(inputs[0], dict):
        mesh_shape = inputs[0].get("device_shape")
        num_devices = inputs[0].get("device_num_devices")
    return (
        rec.get("op"),
        input_keys,
        pc_kind,
        ckc_kind,
        math_fidelity,
        num_devices,
        mesh_shape,
    )


def _dedup_key_for_module(rec: Dict[str, Any]) -> Tuple:
    """Per Plan §3.2. Drops ``module_name`` (qualified path) so all
    instances of the same class with the same input shape collapse.
    """

    def _tensor_key(t: Dict[str, Any]) -> Tuple:
        if not isinstance(t, dict):
            return (_safe_repr(t),)
        return (tuple(t.get("shape") or ()), t.get("dtype"))

    inputs = rec.get("inputs") or []
    input_keys = tuple(_tensor_key(t) for t in inputs)
    return (rec.get("module_class"), input_keys, rec.get("phase"))


def _dedup(records: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], Tuple]) -> List[Dict[str, Any]]:
    groups: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()
    for r in records:
        try:
            k = key_fn(r)
        except Exception:
            k = ("<unhashable>", _next_call_id())
        if k not in groups:
            groups[k] = {**r, "count": 0, "representative_call_ids": []}
        groups[k]["count"] += 1
        if len(groups[k]["representative_call_ids"]) < 5:
            groups[k]["representative_call_ids"].append(r.get("call_id"))
    return list(groups.values())


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    return _safe_repr(o)


def _flush_records() -> None:
    assert _OUT_DIR is not None
    out_dir = _OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    tag = _PHASE_TAG

    ops_path = os.path.join(out_dir, f"{tag}_ops.json")
    modules_path = os.path.join(out_dir, f"{tag}_modules.json")
    ops_dedup_path = os.path.join(out_dir, f"{tag}_ops_dedup.json")
    modules_dedup_path = os.path.join(out_dir, f"{tag}_modules_dedup.json")

    with open(ops_path, "w") as f:
        json.dump(_OP_RECORDS, f, indent=2, default=_json_default)
    with open(modules_path, "w") as f:
        json.dump(_MODULE_RECORDS, f, indent=2, default=_json_default)

    ops_dedup = _dedup(_OP_RECORDS, _dedup_key_for_op)
    modules_dedup = _dedup(_MODULE_RECORDS, _dedup_key_for_module)
    with open(ops_dedup_path, "w") as f:
        json.dump(ops_dedup, f, indent=2, default=_json_default)
    with open(modules_dedup_path, "w") as f:
        json.dump(modules_dedup, f, indent=2, default=_json_default)

    print(
        f"[dots_ocr capture] flushed: ops={len(_OP_RECORDS)} (dedup={len(ops_dedup)}) "
        f"modules={len(_MODULE_RECORDS)} (dedup={len(modules_dedup)}) -> {out_dir}"
    )
