# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefill side of the stage hang watcher: describe this rank's counters, hand them off, spawn the watcher.

Lists, for every chip of the stage, the counters that follow a chunk through it: where each lives (core and
L1 address) and how far it moves per chunk. hang_watcher.py polls them out of process; nothing here touches
them after arming. The hand-off is one line of JSON. Spread out, with one chip of a GLM-5.3 middle rank
(addresses made up):

    {"stage_idx": 1, "pid": 4242, "layers": "20-39", "dummy": 0, "timeout_s": 30.0,
     "first_chunk_timeout_s": 600.0, "poll_s": 1.0, "command": "",
     "chips": [
       {"chip_uid": 81985529, "r": 0, "c": 0,
        "inputs": [{"name": "activation",
                    "A": {"x": 1, "y": 9, "addr": 1024, "unit": 4},
                    "C": {"x": 1, "y": 9, "addr": 1088, "unit": 8}},
                   {"name": "indices", "A": {...}, "C": {...}}],
        "outputs": [{"name": "activation", "P": {...}, "pushed": {...}, "F": {...}},
                    {"name": "indices", "P": {...}, "pushed": {...}, "F": {...}}],
        "acks": {"x": 3, "y": 9, "addr": 2048, "unit": 1}},
       ...]}

hang_watcher.py says what A, C, P, pushed and F count. Inputs are listed in the order the run loop receives
them and outputs in the order it sends them, the activation first: two of the watcher's rules depend on it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

import ttnn

ENABLED_ENV = "TT_PREFILL_STAGE_HANG_DETECTION_ENABLED"
TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_TIMEOUT_SECONDS"
FIRST_CHUNK_TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_FIRST_CHUNK_TIMEOUT_SECONDS"
POLL_ENV = "TT_PREFILL_STAGE_HANG_POLL_SECONDS"
COMMAND_ENV = "TT_PREFILL_STAGE_HANG_COMMAND_TO_EXECUTE"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_FIRST_CHUNK_TIMEOUT_S = 600.0  # covers trace capture and cold first-chunk compiles
DEFAULT_POLL_S = 1.0

BYTES_SENT_OFFSET = 0
_SENDER_MD_WORDS = 7  # bytes_acked_array[] follows, at the next L1-aligned offset


class HangDetector:
    """Handle on this rank's watcher process."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    @classmethod
    def arm(
        cls,
        mesh_device: ttnn.MeshDevice,
        rank: int,
        *,
        h2d: ttnn.H2DStreamService | None,
        d2d_in: ttnn.D2DStreamServiceReceiver | None,
        d2d_out: ttnn.D2DStreamServiceSender | None,
        d2h: ttnn.D2HStreamService | None,
        idx_in: ttnn.D2DStreamServiceReceiver | None,
        idx_out: ttnn.D2DStreamServiceSender | None,
        acks_per_chunk: int,
        dummy: int,
        layers: str,
    ) -> HangDetector | None:
        """Start this rank's watcher when TT_PREFILL_STAGE_HANG_DETECTION_ENABLED=1.

        None when it is off, or when the rank has neither a D2D output nor layer acks: then nothing says a
        chunk finished.

        dummy: output chunks this rank sends without having received them, 1 on a traced run: rank 0's warm-up
        record, or a single rank's capture acks.
        """
        if os.environ.get(ENABLED_ENV, "0") != "1" or (d2d_out is None and d2h is None):
            return None
        chips = []
        for r in range(mesh_device.shape[0]):
            for c in range(mesh_device.shape[1]):
                coord = ttnn.MeshCoordinate(r, c)
                node = mesh_device.get_fabric_node_id(coord)
                inputs = [cls._h2d_input(coord, h2d)] if h2d is not None else []
                for name, service in (("activation", d2d_in), ("indices", idx_in)):
                    if service is not None:
                        inputs.append(cls._d2d_input(name, coord, service))
                outputs = []
                for name, service in (("activation", d2d_out), ("indices", idx_out)):
                    if service is not None:
                        outputs.append(cls._d2d_output(name, coord, service))
                acks = None
                if d2h is not None:
                    # One per layer ack: the last and single rank's finished-chunk word, a heartbeat elsewhere.
                    acks = cls._word(d2h.get_service_core(coord), d2h.get_write_ack_counter_addr(coord), 1)
                if not outputs:
                    # Last and single rank: a chunk is out once all its layer acks are in, with no sender to blame.
                    done = {**acks, "unit": acks_per_chunk}
                    outputs.append({"name": "layer acks", "P": done, "pushed": done, "F": done})
                chip_uid = ttnn.cluster.get_chip_unique_id_from_fabric_node_id(
                    mesh_device, int(node.mesh_id), int(node.chip_id)
                )
                chips.append({"chip_uid": chip_uid, "r": r, "c": c, "inputs": inputs, "outputs": outputs, "acks": acks})
        handoff = {
            "stage_idx": rank,
            "pid": os.getpid(),
            "layers": layers,
            # Other ranks take in a dummy for each they send, so theirs cancels.
            "dummy": dummy if rank == 0 else 0,
            "timeout_s": float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)),
            "first_chunk_timeout_s": float(os.environ.get(FIRST_CHUNK_TIMEOUT_ENV, DEFAULT_FIRST_CHUNK_TIMEOUT_S)),
            "poll_s": float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)),
            "command": os.environ.get(COMMAND_ENV, ""),
            "chips": chips,
        }
        fd, probe_path = tempfile.mkstemp(prefix=f"prefill_hang_stage{rank}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(handoff, f, separators=(",", ":"))
        watcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hang_watcher.py")
        return cls(subprocess.Popen([sys.executable, watcher, probe_path]))

    def stop(self) -> None:
        """Stop the watcher before anything it reads goes away. Safe to call twice."""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    @classmethod
    def _h2d_input(cls, coord: ttnn.MeshCoordinate, h2d: ttnn.H2DStreamService) -> dict:
        # Bytes: a one-row data page and a one-row metadata page per chunk (see build_h2d_service).
        spec = h2d.get_per_shard_spec()
        assert math.prod(spec.shape[:-1]) == 1, f"H2D shard {list(spec.shape)} is not one row"
        core, unit = h2d.get_service_core(coord), 2 * spec.shape[-1] * ttnn.element_size(spec.dtype)
        return {
            "name": "h2d",
            "A": cls._word(core, h2d.get_socket_config_addr(coord) + BYTES_SENT_OFFSET, unit),
            "C": cls._word(core, h2d.get_consumed_counter_addr(coord), cls._cores(h2d)),
        }

    @classmethod
    def _d2d_input(cls, name: str, coord: ttnn.MeshCoordinate, service: ttnn.D2DStreamServiceReceiver) -> dict:
        # Each upstream sender lane bumps bytes_sent once per landed transfer; each of our workers bumps consumed.
        core = service.get_service_core(coord)
        return {
            "name": name,
            "A": cls._word(core, service.get_socket_config_addr() + BYTES_SENT_OFFSET, service.get_num_lanes(coord)),
            "C": cls._word(core, service.get_consumed_counter_addr(coord), cls._cores(service)),
        }

    @classmethod
    def _d2d_output(cls, name: str, coord: ttnn.MeshCoordinate, service: ttnn.D2DStreamServiceSender) -> dict:
        core = service.get_service_core(coord)
        return {
            "name": name,
            "P": cls._word(core, service.get_data_ready_counter_addr(coord), cls._cores(service)),
            # Multicast to every worker by our sender once it has handed the chunk to fabric.
            "pushed": cls._word(service.get_worker_cores().start, service.get_consumed_sem_addr(), 1),
            # Bumped by the next rank's receiver once its workers copied the chunk out.
            "F": cls._word(core, service.get_socket_config_addr() + cls._bytes_acked_offset(), 1),
        }

    @staticmethod
    def _word(core: ttnn.CoreCoord, addr: int, unit: int) -> dict:
        """A 32-bit counter at core and addr, and how far it moves per chunk."""
        return {"x": core.x, "y": core.y, "addr": addr, "unit": unit}

    @staticmethod
    def _cores(service) -> int:
        grid = service.get_worker_cores().grid_size()
        return grid.x * grid.y

    @staticmethod
    def _bytes_acked_offset() -> int:
        alignment = int(ttnn.get_l1_alignment())
        return -(-_SENDER_MD_WORDS * 4 // alignment) * alignment
