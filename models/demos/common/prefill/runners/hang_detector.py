# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefill side of the stage hang watcher: build this rank's probe, hand it off, spawn the watcher.

Records where chip (0,0)'s chunk counters live and how far each moves per chunk, for hang_watcher.py to
poll out of process. Arming also makes SIGUSR1 dump the rank's Python stacks instead of killing it.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import tempfile

import ttnn

ENABLED_ENV = "TT_PREFILL_STAGE_HANG_DETECTION_ENABLED"
TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_TIMEOUT_SECONDS"
FIRST_CHUNK_TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_FIRST_CHUNK_TIMEOUT_SECONDS"
POLL_ENV = "TT_PREFILL_STAGE_HANG_POLL_SECONDS"
COMMAND_ENV = "TT_PREFILL_STAGE_HANG_COMMAND_TO_EXECUTE"
# CI's tt-triage hook; prefill legs keep metal's own watchdog off.
TRIAGE_COMMAND_ENV = "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"
DEFAULT_TIMEOUT_S = 30.0
# Covers trace capture and cold first-chunk compiles.
DEFAULT_FIRST_CHUNK_TIMEOUT_S = 600.0
DEFAULT_POLL_S = 1.0

BYTES_SENT_OFFSET = 0
_SENDER_MD_WORDS = 7  # bytes_acked_array[] follows, at the next L1-aligned offset


def bytes_acked_offset() -> int:
    alignment = int(ttnn.get_l1_alignment())
    return -(-_SENDER_MD_WORDS * 4 // alignment) * alignment


def _num_cores(core_range) -> int:
    grid = core_range.grid_size()
    return grid.x * grid.y


class _Builder:
    def __init__(self, mesh_device) -> None:
        self._mesh = mesh_device

    def chip(self, coord) -> int:
        fnid = self._mesh.get_fabric_node_id(coord)
        return int(
            ttnn.cluster.get_chip_unique_id_from_fabric_node_id(self._mesh, int(fnid.mesh_id), int(fnid.chip_id))
        )

    def word(self, coord, core, addr: int, unit: int) -> dict:
        return {"chip": self.chip(coord), "x": int(core.x), "y": int(core.y), "addr": int(addr), "unit": int(unit)}


def build_probe(mesh_device, *, h2d, d2d_in, d2d_out, d2h) -> tuple[dict, list[dict]]:
    """Chip (0,0)'s words, and the consumed counter of every chip for the verdict-time check."""
    b = _Builder(mesh_device)
    coord = ttnn.MeshCoordinate(0, 0)
    inbound = d2d_in if d2d_in is not None else h2d
    words = {}

    if d2d_in is not None:
        # Each upstream sender lane bumps bytes_sent once per landed transfer.
        cfg = d2d_in.get_socket_config_addr()
        unit = d2d_in.get_num_lanes(coord)
        words["A"] = b.word(coord, d2d_in.get_service_core(coord), cfg + BYTES_SENT_OFFSET, unit)
    else:
        # Bytes: a one-row data page and a one-row metadata page per chunk (see build_h2d_service).
        spec = h2d.get_per_shard_spec()
        shape = list(spec.shape)
        assert math.prod(shape[:-1]) == 1, f"H2D shard {shape} is not one row; the probe's byte unit assumes it is"
        unit = 2 * shape[-1] * ttnn.element_size(spec.dtype)
        cfg = h2d.get_socket_config_addr(coord)
        words["A"] = b.word(coord, h2d.get_service_core(coord), cfg + BYTES_SENT_OFFSET, unit)

    # One increment per worker core: 64 on the D2D grid, 1 on the H2D sync core.
    consumed_unit = _num_cores(inbound.get_worker_cores())
    words["C"] = b.word(coord, inbound.get_service_core(coord), inbound.get_consumed_counter_addr(coord), consumed_unit)

    if d2d_out is not None:
        service = d2d_out.get_service_core(coord)
        workers = d2d_out.get_worker_cores()
        words["P"] = b.word(coord, service, d2d_out.get_data_ready_counter_addr(coord), _num_cores(workers))
        # Multicast to every worker by our sender once it has handed the chunk to fabric.
        words["pushed"] = b.word(coord, workers.start, d2d_out.get_consumed_sem_addr(), 1)
        # Bumped by the next rank's receiver once its workers copied the chunk out.
        words["F"] = b.word(coord, service, d2d_out.get_socket_config_addr() + bytes_acked_offset(), 1)

    # One per layer ack: a heartbeat inside a chunk, and the last rank's only output.
    words["acks"] = b.word(coord, d2h.get_service_core(coord), d2h.get_write_ack_counter_addr(coord), 1)

    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    every_chip = []
    for r in range(rows):
        for c in range(cols):
            mc = ttnn.MeshCoordinate(r, c)
            addr = inbound.get_consumed_counter_addr(mc)
            every_chip.append({**b.word(mc, inbound.get_service_core(mc), addr, consumed_unit), "coord": [r, c]})
    return words, every_chip


class HangDetector:
    """Handle on this rank's watcher process."""

    def __init__(self, proc: subprocess.Popen, probe_path: str) -> None:
        self._proc = proc
        self._probe_path = probe_path

    @classmethod
    def arm(
        cls,
        mesh_device,
        rank: int,
        num_ranks: int,
        *,
        h2d,
        d2d_in,
        d2d_out,
        d2h,
        acks_per_chunk: int,
        ack_layers: list[int],
        warmup_transfers: int,
        warmup_acks: int,
        layers: str,
    ) -> "HangDetector | None":
        """warmup_transfers: warm-up records down each D2D boundary; warmup_acks: acks the capture's warm pass fires."""
        if os.environ.get(ENABLED_ENV, "0") != "1":
            return None

        words, every_chip = build_probe(mesh_device, h2d=h2d, d2d_in=d2d_in, d2d_out=d2d_out, d2h=d2h)
        kind = "single" if num_ranks == 1 else "first" if rank == 0 else "last" if rank == num_ranks - 1 else "middle"
        triage_dir = os.environ.get("TRIAGE_OUTPUT_DIR", "")
        handoff = {
            "rank": rank,
            "kind": kind,
            "pid": os.getpid(),
            "layers": layers,
            "acks_per_chunk": acks_per_chunk,
            "ack_layers": ack_layers,
            "warmup_transfers": warmup_transfers if num_ranks > 1 else 0,
            "warmup_acks": warmup_acks,
            "timeout_s": float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)),
            "first_chunk_timeout_s": float(os.environ.get(FIRST_CHUNK_TIMEOUT_ENV, DEFAULT_FIRST_CHUNK_TIMEOUT_S)),
            "poll_s": float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)),
            "command": os.environ.get(COMMAND_ENV) or os.environ.get(TRIAGE_COMMAND_ENV, ""),
            # CI uploads it, and the launcher stops the producer once a verdict lands there.
            "verdict_path": os.path.join(triage_dir, f"prefill_stage_hang_rank{rank}.txt") if triage_dir else "",
            "words": words,
            "consumed_on_every_chip": every_chip,
        }
        fd, probe_path = tempfile.mkstemp(prefix=f"prefill_hang_rank{rank}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(handoff, f, indent=2)

        # The watcher's SIGUSR1 then dumps stacks instead of killing the rank.
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        watcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hang_watcher.py")
        # Own session: an MPI teardown of the rank's process group must not kill it mid-triage.
        proc = subprocess.Popen([sys.executable, watcher, probe_path], start_new_session=True)
        return cls(proc, probe_path)

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        with contextlib.suppress(OSError):
            os.unlink(self._probe_path)


_active: HangDetector | None = None


def arm(*args, **kwargs) -> None:
    """Arm this rank's watcher (see HangDetector.arm). Module-level, so the request loop can stop it."""
    global _active
    stop()
    try:
        _active = HangDetector.arm(*args, **kwargs)
    except Exception as e:
        # Fail open: a raise here leaves the rank in MPI teardown, a hang nothing would triage.
        print(f"PREFILL HANG WATCHER: not armed ({type(e).__name__}: {e}); not watching", file=sys.stderr, flush=True)


def stop() -> None:
    """Stop the watcher before anything it reads goes away. Idempotent."""
    global _active
    if _active is not None:
        _active.stop()
        _active = None
