# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Optional JSONL logging of tensor fingerprints for MoE prefill determinism debugging.

Set ``DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG`` to a file path (JSONL). Each enabled
checkpoint appends one JSON object per line. Run the same workload twice and
compare logs with ``models/demos/deepseek_v3/tools/compare_determinism_logs.py``.

Notes:
- Full-tensor SHA-256 of host tensor bytes can be expensive on large activations;
  use short prompts / few MoE layers while tracing.
- ``DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_MAX_RECORDS`` (integer): after this many
  tensor records, logging becomes a no-op. Default is a safety cap of ``256``.
  Set ``0`` (or a negative value) for unlimited logs.
- ``DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_PROGRESS_EVERY`` (integer): print
  periodic progress every N records (default ``25``). Set ``0`` to disable.
- By default, only rank 0 logs (`TT_MESH_HOST_RANK` / MPI rank 0) to keep
  multihost runs responsive. Override with
  ``DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_RANK=all`` (or a specific rank id).
- In multi-host jobs, point each rank at a distinct path (for example include
  ``RANK`` or ``TT_MESH_HOST_RANK`` in the filename) so logs are not corrupted
  by concurrent writers.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import MeshDeviceStub

_LOG_ENV = "DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG"
_CONTEXT_ENV = "DEEPSEEK_MOE_PREFILL_DETERMINISM_CONTEXT"
_MAX_RECORDS_ENV = "DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_MAX_RECORDS"
_LOG_RANK_ENV = "DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_RANK"
_PROGRESS_EVERY_ENV = "DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG_PROGRESS_EVERY"
_DEFAULT_MAX_RECORDS = 256
_DEFAULT_PROGRESS_EVERY = 25

_counter = 0
_counter_lock = threading.Lock()
_file_lock = threading.Lock()
_tensor_records_logged = 0
_tensor_records_lock = threading.Lock()
_status_print_lock = threading.Lock()
_status_printed = False
_cap_notice_printed = False


def log_path() -> str | None:
    path = os.getenv(_LOG_ENV, "").strip()
    return path or None


def context_label() -> str | None:
    label = os.getenv(_CONTEXT_ENV, "").strip()
    return label or None


def _max_tensor_records() -> int | None:
    raw = os.getenv(_MAX_RECORDS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_RECORDS
    try:
        n = int(raw, 10)
    except ValueError:
        return _DEFAULT_MAX_RECORDS
    if n <= 0:
        return None
    return n


def _progress_every() -> int:
    raw = os.getenv(_PROGRESS_EVERY_ENV, "").strip()
    if not raw:
        return _DEFAULT_PROGRESS_EVERY
    try:
        n = int(raw, 10)
    except ValueError:
        return _DEFAULT_PROGRESS_EVERY
    return max(0, n)


def _current_rank() -> int:
    for rank_env in ("TT_MESH_HOST_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"):
        raw = os.getenv(rank_env)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _should_log_on_this_rank() -> bool:
    target = os.getenv(_LOG_RANK_ENV, "0").strip().lower()
    if target in ("", "0"):
        return _current_rank() == 0
    if target in ("all", "*"):
        return True
    try:
        return _current_rank() == int(target, 10)
    except ValueError:
        # Invalid env value should not disable logging unexpectedly.
        return True


def _resolved_log_path(path: str) -> str:
    rank = _current_rank()
    if "{rank}" in path:
        return path.replace("{rank}", str(rank))
    target = os.getenv(_LOG_RANK_ENV, "0").strip().lower()
    if target in ("all", "*"):
        root, ext = os.path.splitext(path)
        if ext:
            return f"{root}_r{rank}{ext}"
        return f"{path}_r{rank}"
    return path


def _next_seq() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _resolve_mesh_device(cfg: dict[str, Any]) -> ttnn.MeshDevice | None:
    ccl = cfg.get("ccl")
    mesh = getattr(ccl, "mesh_device", None) if ccl is not None else None
    if mesh is not None and not isinstance(mesh, MeshDeviceStub):
        return mesh
    md = cfg.get("mesh_device")
    if md is not None and not isinstance(md, MeshDeviceStub):
        return md  # type: ignore[return-value]
    return None


def _tensor_to_torch_host(tt: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    return ttnn.to_torch(
        tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(-2, -1),
            mesh_shape=mesh_device.shape,
        ),
    )


def _sha256_record(tag: str, torch_tensor: torch.Tensor) -> dict[str, Any]:
    t = torch_tensor.detach().contiguous().cpu()
    # Hash raw bytes from a dtype-agnostic uint8 reinterpretation.
    # This avoids numpy conversion failures for unsupported dtypes such as bfloat16.
    payload = t.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "tag": tag,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "sha256": digest,
    }


def maybe_log_tensor(cfg: dict[str, Any], tag: str, tensor: ttnn.Tensor | None) -> None:
    """If ``DEEPSEEK_MOE_PREFILL_DETERMINISM_LOG`` is set, append one JSON line for ``tensor``."""
    path = log_path()
    if path is None or tensor is None or not _should_log_on_this_rank():
        return

    global _cap_notice_printed
    global _status_printed
    global _tensor_records_logged

    max_records = _max_tensor_records()
    progress_every = _progress_every()
    current_rank = _current_rank()
    resolved_path = _resolved_log_path(path)

    with _status_print_lock:
        if not _status_printed:
            print(
                "[moe_prefill_determinism] enabled "
                f"rank={current_rank} path={resolved_path} "
                f"max_records={max_records if max_records is not None else 'unlimited'} "
                f"progress_every={progress_every}",
                flush=True,
            )
            _status_printed = True

    with _tensor_records_lock:
        if max_records is not None and _tensor_records_logged >= max_records:
            if not _cap_notice_printed:
                print(
                    "[moe_prefill_determinism] reached " f"max_records={max_records}; skipping further tensor logs",
                    flush=True,
                )
                _cap_notice_printed = True
            return
        _tensor_records_logged += 1
        record_idx = _tensor_records_logged

    mesh_device = _resolve_mesh_device(cfg)
    if mesh_device is None:
        record = {
            "seq": _next_seq(),
            "tag": tag,
            "error": "no_mesh_device_in_cfg_for_determinism_log",
        }
    else:
        try:
            th = _tensor_to_torch_host(tensor, mesh_device)
            record = {"seq": _next_seq(), **_sha256_record(tag, th)}
            del th
        except Exception as exc:  # noqa: BLE001 — debug path; must not break model
            record = {"seq": _next_seq(), "tag": tag, "error": f"{type(exc).__name__}: {exc}"}

    ctx = context_label()
    if ctx is not None:
        record["context"] = ctx

    line = json.dumps(record, sort_keys=True) + "\n"
    parent = os.path.dirname(resolved_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _file_lock:
        with open(resolved_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    if progress_every > 0 and record_idx % progress_every == 0:
        print(
            "[moe_prefill_determinism] progress " f"records={record_idx} tag={tag}",
            flush=True,
        )
