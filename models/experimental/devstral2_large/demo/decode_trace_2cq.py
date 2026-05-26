# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-command-queue (2CQ) host→device staging for traced Devstral2 decode.

CQ1 performs ``copy_host_to_device_tensor`` for ``(token, current_pos)`` while CQ0 runs
the decode forward / ``execute_trace``. Events keep ordering: CQ1 waits until CQ0 has
finished the previous trace before overwriting the persistent input buffers.

Pattern mirrors vision performant runners (e.g. SwinV2), adapted for autoregressive
decode where trace binds directly to ``decode_tok_dev`` / ``decode_pos_dev`` (no separate
DRAM→L1 reshard), so ``op_event`` is recorded **after** each CQ0 forward/trace completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


def _env_flag_enabled(name: str, *, default: str = "1") -> bool:
    """True unless the env var is explicitly 0 / false / no."""
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def decode_trace_2cq_enabled() -> bool:
    """True when ``DEVSTRAL2_DECODE_TRACE_2CQ`` is unset or truthy (default on)."""
    return _env_flag_enabled("DEVSTRAL2_DECODE_TRACE_2CQ")


def num_command_queues_for_decode() -> int:
    return 2 if decode_trace_2cq_enabled() else 1


def stage_decode_inputs(
    pipe: Optional["DecodeTrace2CQ"],
    mesh_device: ttnn.MeshDevice,
    decode_tok_dev: ttnn.Tensor,
    decode_pos_dev: ttnn.Tensor,
    token_id: int,
    pos: int,
) -> None:
    """Host→device staging for one decode step (2CQ pipeline or default CQ)."""
    if pipe is not None:
        pipe.write_inputs_cq1(token_id, pos)
        pipe.wait_inputs_ready_cq0()
        return
    from models.experimental.devstral2_large.demo.text_demo import (
        _current_pos_host,
        _input_ids_host,
    )

    ttnn.copy_host_to_device_tensor(
        _input_ids_host(torch.tensor([[token_id]], dtype=torch.long), mesh_device),
        decode_tok_dev,
    )
    ttnn.copy_host_to_device_tensor(
        _current_pos_host(torch.tensor([pos], dtype=torch.long), mesh_device),
        decode_pos_dev,
    )


def signal_decode_step_done(pipe: Optional["DecodeTrace2CQ"]) -> None:
    if pipe is not None:
        pipe.signal_trace_done_cq0()


@dataclass
class DecodeTrace2CQ:
    """Event-synced CQ1 input writes + CQ0 decode/trace on persistent device buffers."""

    mesh_device: ttnn.MeshDevice
    decode_tok_dev: ttnn.Tensor
    decode_pos_dev: ttnn.Tensor
    op_event: object = None
    write_event: object = None

    @classmethod
    def create(
        cls,
        mesh_device: ttnn.MeshDevice,
        decode_tok_dev: ttnn.Tensor,
        decode_pos_dev: ttnn.Tensor,
    ) -> DecodeTrace2CQ:
        state = cls(mesh_device, decode_tok_dev, decode_pos_dev)
        # Dummy op event on CQ0 so the first CQ1 write does not block forever.
        state.op_event = ttnn.record_event(mesh_device, 0)
        return state

    def write_inputs_cq1(self, token_id: int, pos: int) -> None:
        """Stage the next decode step inputs on CQ1 (after CQ0 released the buffers)."""
        from models.experimental.devstral2_large.demo.text_demo import (
            _current_pos_host,
            _input_ids_host,
        )

        ttnn.wait_for_event(1, self.op_event)
        tok_host = _input_ids_host(torch.tensor([[token_id]], dtype=torch.long), self.mesh_device)
        pos_host = _current_pos_host(torch.tensor([pos], dtype=torch.long), self.mesh_device)
        ttnn.copy_host_to_device_tensor(tok_host, self.decode_tok_dev, 1)
        ttnn.copy_host_to_device_tensor(pos_host, self.decode_pos_dev, 1)
        self.write_event = ttnn.record_event(self.mesh_device, 1)

    def wait_inputs_ready_cq0(self) -> None:
        ttnn.wait_for_event(0, self.write_event)

    def signal_trace_done_cq0(self) -> None:
        """CQ0: previous forward/trace finished; CQ1 may overwrite input buffers."""
        self.op_event = ttnn.record_event(self.mesh_device, 0)
