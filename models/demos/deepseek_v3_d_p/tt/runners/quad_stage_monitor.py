# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Per-stage check for the quad prefill-runner screen.

Each pipeline rank reads back the activation it received and the one it produced, one tensor per device, and
compares them with what it saw on the first iteration (or with a known pattern). It records progress, the first
mismatch with its chip coordinates, and per-device hashes into files under ``results_dir``; the gwdt side
(``scripts/gwdt_quad_verdict.py``) turns those into a galaxy or link verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch

MAX_MISMATCHES = 20
INJECT_ENV = "PREFILL_QUAD_INJECT"


def device_shards(t) -> list[torch.Tensor]:
    """One host tensor per device, in mesh row-major order."""
    import ttnn

    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _hash(t: torch.Tensor) -> str:
    return hashlib.sha1(t.contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _parse_inject(rank: int) -> tuple[str, int] | None:
    """``<rank>:<input|output>[:<iteration>]``: flip one bit of device 0's copy on that rank, to prove the verdict
    names the right stage. The device data is untouched."""
    spec = os.environ.get(INJECT_ENV, "")
    if not spec:
        return None
    parts = spec.split(":")
    if int(parts[0]) != rank:
        return None
    return parts[1], int(parts[2]) if len(parts) > 2 else 1


class StageMonitor:
    def __init__(self, rank: int, mode: str, results_dir, mesh_cols: int, fabric: str, host: str | None = None):
        self.rank, self.mode, self.mesh_cols = rank, mode, mesh_cols
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.iteration = -1
        self.baseline: dict[tuple[str, int], list[torch.Tensor]] = {}
        self.pending_in: dict[int, list[str]] = {}
        self.inject = _parse_inject(rank)
        self.doc = {
            "rank": rank,
            "host": host or socket.gethostname().split(".")[0],
            "mode": mode,
            "fabric": fabric,
            "chunks_done": 0,
            "iterations": 0,
            "last_in": None,
            "last_out": None,
            "last_progress": time.time(),
            "first_bad": None,
            "mismatches": [],
            "error": None,
            "done": False,
        }
        self.flush()

    def check(self, what: str, chunk: int, shards: list[torch.Tensor], expected: list[torch.Tensor] | None = None):
        if what == "input" and chunk == 0:
            self.iteration += 1
            self.doc["iterations"] = self.iteration + 1
        it = max(self.iteration, 0)
        if self.inject == (what, it):
            shards = [shards[0].clone(), *shards[1:]]
            shards[0].view(-1).view(torch.int16)[0] ^= 1

        key = (what, chunk)
        ref = expected if expected is not None else self.baseline.get(key)
        if ref is None:
            self._set_baseline(key, it, shards)
        else:
            self._compare(what, chunk, it, shards, ref)

        hashes = [_hash(s) for s in shards]
        pos = [it, chunk]
        if what == "input":
            self.doc["last_in"] = pos
            self.pending_in[chunk] = hashes
        else:
            self.doc["last_out"] = pos
            self.doc["chunks_done"] += 1
            line = {"iter": it, "chunk": chunk, "in": self.pending_in.pop(chunk, []), "out": hashes}
            with open(self.dir / f"rank{self.rank}.hashes.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
        self.doc["last_progress"] = time.time()
        self.flush()

    def _set_baseline(self, key, it: int, shards: list[torch.Tensor]) -> None:
        # Bit-identity against an all-zero or non-finite baseline proves nothing.
        floats = [s for s in shards if s.is_floating_point()]
        if floats and not all(torch.isfinite(s).all() for s in floats):
            self.fail(f"iteration {it} {key[0]} chunk {key[1]} is not finite")
        elif not any(s.any() for s in shards):
            self.fail(f"iteration {it} {key[0]} chunk {key[1]} is all zeros")
        self.baseline[key] = [s.clone() for s in shards]

    def _compare(self, what: str, chunk: int, it: int, shards, ref) -> None:
        bad, n_diff = [], 0
        for i, (got, want) in enumerate(zip(shards, ref)):
            if not torch.equal(got, want):
                bad.append([i // self.mesh_cols, i % self.mesh_cols])
                n_diff += int((got != want).sum())
        if not bad:
            return
        rec = {"iter": it, "chunk": chunk, "what": what, "devices": bad, "n_diff": n_diff}
        if self.doc["first_bad"] is None:
            self.doc["first_bad"] = rec
        if len(self.doc["mismatches"]) < MAX_MISMATCHES:
            self.doc["mismatches"].append(rec)

    def fail(self, error: str) -> None:
        if self.doc["error"] is None:
            self.doc["error"] = error
        self.flush()

    def finish(self) -> None:
        self.doc["done"] = True
        self.flush()

    def flush(self) -> None:
        path = self.dir / f"rank{self.rank}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc))
        os.replace(tmp, path)
