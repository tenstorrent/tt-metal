# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model descriptor tt-d-gen's prefill worker reads at ``device.descriptor_path``.

The contract is tt-d-gen's ``docs/schemas/model_descriptor.schema.json``. The runner's master rank writes it once the
runner is set up, at ``PREFILL_MODEL_DESCRIPTOR_PATH``, replacing the previous run's file through a rename.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
PATH_ENV = "PREFILL_MODEL_DESCRIPTOR_PATH"


def descriptor_path() -> str:
    return os.environ.get(PATH_ENV, "")


def process_started_at(pid: Optional[int] = None) -> float:
    """When the process started, in unix seconds: /proc/<pid>/stat's start ticks after /proc/stat's btime."""
    with open(f"/proc/{pid or os.getpid()}/stat") as fh:
        ticks = int(fh.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/stat") as fh:
        btime = next(int(line.split()[1]) for line in fh if line.startswith("btime"))
    return btime + ticks / os.sysconf("SC_CLK_TCK")


def metal_version() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short=10", "HEAD"],
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def prefill_descriptor(
    *,
    model_id: str,
    max_context_len: int,
    max_slots: int,
    kv_depth: int,
    num_layers: int,
    page_tokens: int,
    chunk_size: int,
    sp_factor: int,
    max_chunks_per_batch: int,
    num_mtp_tokens: int,
    service_id: str,
    ack_shm_name: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "prefill",
        "producer": {
            "name": "tt-metal-prefill",
            "version": metal_version(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": process_started_at(),
        },
        "model": {"id": model_id, "max_context_len": max_context_len},
        "capacity": {"max_slots": max_slots, "kv_depth": kv_depth},
        "kv": {"num_layers": num_layers, "layer_begin": 0, "layer_end": num_layers, "page_tokens": page_tokens},
        "prefill": {
            "chunk_size": chunk_size,
            "sp_factor": sp_factor,
            "max_chunks_per_batch": max_chunks_per_batch,
            "num_mtp_tokens": num_mtp_tokens,
        },
        "io": {"service_id": service_id, "ack_shm_name": ack_shm_name},
    }


def discard(path: str) -> None:
    """Remove an earlier run's descriptor, so a worker started before this run writes its own never reads it."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def write(path: str, doc: dict) -> None:
    """Replace ``path`` with ``doc`` in one rename, so a reader never sees a partial file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)
