# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Optional JSONL logging of TTNN operation outputs for model-wide determinism.

Set ``DEEPSEEK_MODEL_OP_DETERMINISM_LOG`` to a JSONL path to enable logging.
Each logged record contains:
    - operation name
    - output index (for tuple/list outputs)
    - tensor shape / dtype / numel
    - SHA-256 hash of raw tensor bytes on host

Environment variables:
- ``DEEPSEEK_MODEL_OP_DETERMINISM_LOG``: target JSONL path.
- ``DEEPSEEK_MODEL_OP_DETERMINISM_CONTEXT``: optional context tag (for runA/runB).
- ``DEEPSEEK_MODEL_OP_DETERMINISM_LOG_MAX_RECORDS``: cap number of records
  written per rank (default: 512). Set <=0 for unlimited.
- ``DEEPSEEK_MODEL_OP_DETERMINISM_LOG_RANK``: ``0`` by default. Use ``all`` to
  enable on all ranks.
- ``DEEPSEEK_MODEL_OP_DETERMINISM_LOG_PROGRESS_EVERY``: print progress every N
  records (default: 50). Set <=0 to disable progress logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch

import ttnn

_LOG_ENV = "DEEPSEEK_MODEL_OP_DETERMINISM_LOG"
_CONTEXT_ENV = "DEEPSEEK_MODEL_OP_DETERMINISM_CONTEXT"
_MAX_RECORDS_ENV = "DEEPSEEK_MODEL_OP_DETERMINISM_LOG_MAX_RECORDS"
_LOG_RANK_ENV = "DEEPSEEK_MODEL_OP_DETERMINISM_LOG_RANK"
_PROGRESS_EVERY_ENV = "DEEPSEEK_MODEL_OP_DETERMINISM_LOG_PROGRESS_EVERY"

_DEFAULT_MAX_RECORDS = 512
_DEFAULT_PROGRESS_EVERY = 50

_seq = 0
_seq_lock = threading.Lock()
_file_lock = threading.Lock()
_record_count = 0
_record_count_lock = threading.Lock()
_status_lock = threading.Lock()
_status_printed = False
_cap_notice_printed = False


def _log_path() -> str | None:
    path = os.getenv(_LOG_ENV, "").strip()
    return path or None


def _context_label() -> str | None:
    label = os.getenv(_CONTEXT_ENV, "").strip()
    return label or None


def _max_records() -> int | None:
    raw = os.getenv(_MAX_RECORDS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_RECORDS
    try:
        value = int(raw, 10)
    except ValueError:
        return _DEFAULT_MAX_RECORDS
    if value <= 0:
        return None
    return value


def _progress_every() -> int:
    raw = os.getenv(_PROGRESS_EVERY_ENV, "").strip()
    if not raw:
        return _DEFAULT_PROGRESS_EVERY
    try:
        value = int(raw, 10)
    except ValueError:
        return _DEFAULT_PROGRESS_EVERY
    return max(0, value)


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
        # Invalid value should not silently disable logging.
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
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


def _reserve_record_slot(max_records: int | None) -> int | None:
    global _record_count
    global _cap_notice_printed

    with _record_count_lock:
        if max_records is not None and _record_count >= max_records:
            if not _cap_notice_printed:
                print(
                    "[model_op_determinism] reached " f"max_records={max_records}; skipping further logs",
                    flush=True,
                )
                _cap_notice_printed = True
            return None
        _record_count += 1
        return _record_count


def _is_ttnn_tensor_like(value: Any) -> bool:
    if isinstance(value, ttnn.Tensor):
        return True
    value_type = type(value)
    return value_type.__name__ == "Tensor" and value_type.__module__.startswith("ttnn")


def _iter_ttnn_tensors(value: Any) -> Iterator[ttnn.Tensor]:
    if _is_ttnn_tensor_like(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for element in value:
            yield from _iter_ttnn_tensors(element)
    elif isinstance(value, dict):
        for element in value.values():
            yield from _iter_ttnn_tensors(element)


def _tensor_to_torch_host(tt_tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice | None) -> torch.Tensor:
    try:
        return ttnn.to_torch(tt_tensor)
    except Exception as first_exc:
        if mesh_device is not None:
            return ttnn.to_torch(
                tt_tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(-2, -1),
                    mesh_shape=mesh_device.shape,
                ),
            )
        raise first_exc


def _sha256_record(torch_tensor: torch.Tensor) -> dict[str, Any]:
    host = torch_tensor.detach().contiguous().cpu()
    payload = host.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "numel": int(host.numel()),
        "sha256": digest,
    }


@contextmanager
def register_model_op_determinism_hook(mesh_device: ttnn.MeshDevice | None):
    """Register TTNN post-op hash logging hook when env is enabled."""

    path = _log_path()
    if path is None or not _should_log_on_this_rank():
        yield
        return

    max_records = _max_records()
    progress_every = _progress_every()
    current_rank = _current_rank()
    resolved_path = _resolved_log_path(path)

    global _status_printed

    with _status_lock:
        if not _status_printed:
            print(
                "[model_op_determinism] enabled "
                f"rank={current_rank} path={resolved_path} "
                f"max_records={max_records if max_records is not None else 'unlimited'} "
                f"progress_every={progress_every}",
                flush=True,
            )
            _status_printed = True

    context = _context_label()

    def _post_hook(operation: Any, _args: Any, _kwargs: Any, output: Any) -> None:
        op_name = getattr(operation, "python_fully_qualified_name", None) or str(operation)
        for output_index, tt_tensor in enumerate(_iter_ttnn_tensors(output)):
            record_idx = _reserve_record_slot(max_records)
            if record_idx is None:
                return

            seq = _next_seq()
            try:
                host_tensor = _tensor_to_torch_host(tt_tensor, mesh_device)
                record = {
                    "seq": seq,
                    "op": op_name,
                    "output_index": output_index,
                    **_sha256_record(host_tensor),
                }
                del host_tensor
            except Exception as exc:  # noqa: BLE001 - debug path must not break model
                record = {
                    "seq": seq,
                    "op": op_name,
                    "output_index": output_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            if context is not None:
                record["context"] = context

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
                    "[model_op_determinism] progress " f"records={record_idx} op={op_name} output_index={output_index}",
                    flush=True,
                )

    # Fast runtime mode can bypass the Python operation path where hooks are invoked.
    # Force it off while determinism capture is active so every op is observable.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        with ttnn.register_post_operation_hook(_post_hook):
            yield
