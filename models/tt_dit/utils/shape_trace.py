# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight JSONL shape tracing for TTNN ops.
# Controlled by environment variables:
# - TT_SHAPE_TRACE: "1" to enable (default "0")
# - TT_SHAPE_TRACE_PATH: output file path (default: "shape_trace.jsonl")
# - TT_SHAPE_TRACE_TAG: optional context tag (e.g., mesh/topology)
#
# Usage:
#   from models.tt_dit.utils.shape_trace import log_shape
#   log_shape("matmul", {"a": a_tensor, "b": b_tensor}, extra={"where": "linear.forward"})
#
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import ttnn  # type: ignore

_lock = threading.Lock()


def _is_enabled() -> bool:
    return os.environ.get("TT_SHAPE_TRACE", "0") in ("1", "true", "True")


def _outfile() -> str:
    return os.environ.get("TT_SHAPE_TRACE_PATH", "shape_trace.jsonl")


def _now_iso() -> str:
    # Avoid timezone complications; ISO without TZ is fine for logs
    return datetime.utcnow().isoformat(timespec="milliseconds")


def _tensor_info(t: ttnn.Tensor) -> Dict[str, Any]:
    try:
        # Prefer logical shape; also capture padded for program config discovery
        shape = tuple(getattr(t, "shape", ()))
        padded = tuple(getattr(t, "padded_shape", ()))
        dtype = getattr(t, "get_dtype", lambda: None)()
        # Mesh hints if available
        device = getattr(t, "device", lambda: None)()
        mesh_shape = None
        if device is not None and hasattr(device, "shape"):
            try:
                mesh_shape = tuple(device.shape)
            except Exception:
                mesh_shape = None
        return {
            "shape": tuple(int(x) for x in shape) if shape else None,
            "padded_shape": tuple(int(x) for x in padded) if padded else None,
            "dtype": str(dtype) if dtype is not None else None,
            "mesh_shape": mesh_shape,
        }
    except Exception as e:
        return {"error": f"tensor_info_failed: {e.__class__.__name__}: {e}"}


def _serialize(data: Dict[str, Any]) -> str:
    def _default(o: Any):
        try:
            return str(o)
        except Exception:
            return "<unserializable>"

    return json.dumps(data, default=_default, sort_keys=False)


def log_shape(op: str, tensors: Dict[str, ttnn.Tensor], *, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Append a JSONL record describing an op's input tensor shapes.
    No-ops when TT_SHAPE_TRACE is disabled.
    """
    if not _is_enabled():
        return

    try:
        record: Dict[str, Any] = {
            "ts": _now_iso(),
            "op": op,
            "tag": os.environ.get("TT_SHAPE_TRACE_TAG"),
            "inputs": {name: _tensor_info(t) for name, t in tensors.items()},
        }
        if extra:
            record["extra"] = extra
        text = _serialize(record)
        with _lock:
            with open(_outfile(), "a", encoding="utf-8") as f:
                f.write(text + "\n")
    except Exception:
        # Best-effort tracing; never fail the model/test
        return
