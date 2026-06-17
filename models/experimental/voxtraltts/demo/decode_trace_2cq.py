# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two-command-queue (2CQ) + trace staging for the Voxtral TTS text-decode step (Blackhole).

Why this exists
---------------
Voxtral decode is **host-dispatch-gap bound**: each of the ~hundreds of tiny ttnn ops per
token finishes on-device in microseconds, then waits ~100us for the host to enqueue the next
op. Capturing the 26-layer text-decode step (+ on-device RoPE gather) into a **trace** and
replaying it with ``execute_trace`` removes those gaps (host leaves the loop). A **second
command queue (CQ1)** stages the next step's inputs (embedding / position / rope-index) into
persistent device buffers while CQ0 replays the trace — events keep the ordering.

What is traced
--------------
The text-decode step only — ``HfRotarySetup.get_rot_mats(rot_idxs_dev)`` (device-side cos/sin
gather) followed by ``VoxtralTTTextModel.decode_step_from_embeds_tt``. The acoustic flow-matching
forward and the CPU code→embedding lookup stay outside the trace (they have per-step host data),
so each AR step is:

    execute_trace(text-decode)            # CQ0, bound to persistent (x_embed, pos, rot_idxs)
      -> hidden (device)
    acoustic.forward(hidden, noise)       # CQ0 (untraced; own per-step noise)
      -> codes -> host
    end-audio check + F.embedding(codes)  # host
    stage next (x_embed, pos, rot_idxs)   # CQ1, overlapped with the next execute_trace

Trace binds to the persistent buffer *objects*; we only ever rewrite their *contents*
(``copy_host_to_device_tensor``), never reallocate — that is what makes replay valid.

Persistent buffers bound by the decode trace (batch = 1 for TTS):
  - ``x_embed_dev``  [1, 1, 1, dim]  bf16   MM embedding (audio-codes → embedding, host F.embedding)
  - ``pos_dev``      [1]             int32  attention / KV-cache position
  - ``rot_idxs_dev`` [1, 1]          int32  rope cos/sin gather index (= position)

Enable/disable with ``VOXTRAL_DECODE_TRACE_2CQ`` (default on). With 2CQ off, a single CQ is used
and staging falls back to plain ``copy_host_to_device_tensor`` on CQ0 (still trace-replayed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

import ttnn


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
def _env_flag_enabled(name: str, *, default: str = "1") -> bool:
    """True unless the env var is explicitly 0 / false / no."""
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def decode_trace_enabled() -> bool:
    """True when ``VOXTRAL_DECODE_TRACE`` is truthy. Default OFF during bring-up; flip to "1"
    (or set the env var) once validated on-device."""
    return _env_flag_enabled("VOXTRAL_DECODE_TRACE", default="0")


def decode_trace_2cq_enabled() -> bool:
    """True when 2CQ input staging is on. Independent of ``VOXTRAL_DECODE_TRACE`` — production
    decode always trace-replays; this flag only toggles CQ1 overlap vs single-CQ staging."""
    return _env_flag_enabled("VOXTRAL_DECODE_TRACE_2CQ", default="1")


def num_command_queues_for_decode() -> int:
    """Open the device with this many CQs (pass to ``ttnn.open_mesh_device(..., num_command_queues=)``)."""
    return 2 if decode_trace_2cq_enabled() else 1


# ---------------------------------------------------------------------------
# Host staging tensors (no device=) — ready for copy_host_to_device_tensor
# ---------------------------------------------------------------------------
def embed_host(x_embed_4d: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Host ttnn.Tensor for the MM embedding ``[1, 1, 1, dim]`` (bf16, replicated)."""
    return ttnn.from_torch(
        x_embed_4d.reshape(1, 1, 1, -1).to(torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def pos_host(pos: int, mesh_device, cluster_shape) -> ttnn.Tensor:
    """Host ttnn.Tensor for the attention position ``[1]`` (int32, replicated)."""
    return ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )


def rot_idxs_host(pos: int, rope_setup) -> ttnn.Tensor:
    """Host ttnn.Tensor for the rope gather index, built by the rope setup itself.

    ``HfRotarySetup.get_rot_idxs`` produces a ``[1, nearest_32(batch)]`` **uint32 ROW_MAJOR** tensor
    (the layout/dtype ``ttnn.embedding`` requires inside ``get_rot_mats``). Use it directly so the
    persistent buffer and per-step writes always match the kernel's expectations.
    """
    return rope_setup.get_rot_idxs(torch.tensor([pos], dtype=torch.int32), on_host=True)


@dataclass
class VoxtralDecodeBuffers:
    """The persistent device buffers the decode trace binds to (rewritten in place each step)."""

    x_embed_dev: ttnn.Tensor
    pos_dev: ttnn.Tensor
    rot_idxs_dev: ttnn.Tensor
    rope_setup: Any  # used to build rot_idxs host stagers in the exact kernel-expected format

    @classmethod
    def create(cls, mesh_device, dim: int, cluster_shape, rope_setup, start_pos: int, init_embed_4d: torch.Tensor):
        """Allocate the persistent buffers once, seeded with the first step's inputs."""
        x_embed_dev = ttnn.to_device(embed_host(init_embed_4d, mesh_device), mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        pos_dev = ttnn.to_device(pos_host(start_pos, mesh_device, cluster_shape), mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        rot_idxs_dev = ttnn.to_device(rot_idxs_host(start_pos, rope_setup), mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        return cls(x_embed_dev, pos_dev, rot_idxs_dev, rope_setup)

    def deallocate(self) -> None:
        for t in (self.x_embed_dev, self.pos_dev, self.rot_idxs_dev):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)


# ---------------------------------------------------------------------------
# 2CQ input staging (event-synced CQ1 writes vs CQ0 trace replay)
# ---------------------------------------------------------------------------
@dataclass
class VoxtralDecodeTrace2CQ:
    """Event-synced CQ1 input writes + CQ0 decode-trace replay on persistent buffers.

    CQ1 performs ``copy_host_to_device_tensor`` for ``(x_embed, pos, rot_idxs)`` while CQ0 runs the
    traced text decode. ``op_event`` (recorded on CQ0 after each replay) gates CQ1 so it cannot
    overwrite the buffers until CQ0 has finished consuming them; ``write_event`` (recorded on CQ1
    after the writes) gates CQ0 so the next replay cannot start until the new inputs have landed.
    """

    mesh_device: Any
    bufs: VoxtralDecodeBuffers
    cluster_shape: Any
    op_event: object = None
    write_event: object = None

    @classmethod
    def create(cls, mesh_device, bufs: VoxtralDecodeBuffers, cluster_shape) -> "VoxtralDecodeTrace2CQ":
        state = cls(mesh_device, bufs, cluster_shape)
        # Dummy op event on CQ0 so the first CQ1 write does not block forever.
        state.op_event = ttnn.record_event(mesh_device, 0)
        return state

    def write_inputs_cq1(self, x_embed_4d: torch.Tensor, pos: int) -> None:
        """Stage the next step's (embed, pos, rot_idxs) on CQ1 once CQ0 has released the buffers."""
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(embed_host(x_embed_4d, self.mesh_device), self.bufs.x_embed_dev, 1)
        ttnn.copy_host_to_device_tensor(pos_host(pos, self.mesh_device, self.cluster_shape), self.bufs.pos_dev, 1)
        ttnn.copy_host_to_device_tensor(rot_idxs_host(pos, self.bufs.rope_setup), self.bufs.rot_idxs_dev, 1)
        self.write_event = ttnn.record_event(self.mesh_device, 1)

    def wait_inputs_ready_cq0(self) -> None:
        """CQ0: block the trace replay until CQ1's input writes have completed."""
        ttnn.wait_for_event(0, self.write_event)

    def signal_trace_done_cq0(self) -> None:
        """CQ0: previous replay finished consuming the buffers; CQ1 may overwrite them."""
        self.op_event = ttnn.record_event(self.mesh_device, 0)


# Short public alias so callers can import ``DecodeTrace2CQ``.
DecodeTrace2CQ = VoxtralDecodeTrace2CQ


def stage_decode_inputs(
    pipe: Optional[VoxtralDecodeTrace2CQ],
    bufs: VoxtralDecodeBuffers,
    mesh_device,
    cluster_shape,
    x_embed_4d: torch.Tensor,
    pos: int,
) -> None:
    """Host→device staging for one decode step (2CQ pipeline, or plain CQ0 when 2CQ is off)."""
    if pipe is not None:
        pipe.write_inputs_cq1(x_embed_4d, pos)
        pipe.wait_inputs_ready_cq0()
        return
    ttnn.copy_host_to_device_tensor(embed_host(x_embed_4d, mesh_device), bufs.x_embed_dev)
    ttnn.copy_host_to_device_tensor(pos_host(pos, mesh_device, cluster_shape), bufs.pos_dev)
    ttnn.copy_host_to_device_tensor(rot_idxs_host(pos, bufs.rope_setup), bufs.rot_idxs_dev)


def signal_decode_step_done(pipe: Optional[VoxtralDecodeTrace2CQ]) -> None:
    if pipe is not None:
        pipe.signal_trace_done_cq0()


# ---------------------------------------------------------------------------
# Trace capture / replay of the text-decode step (rope gather + 26 layers)
# ---------------------------------------------------------------------------
@dataclass
class TracedTextDecode:
    """Captures and replays the Voxtral text-decode step bound to ``VoxtralDecodeBuffers``.

    The traced region is: ``rope = rope_setup.get_rot_mats(rot_idxs_dev)`` (device cos/sin gather)
    then ``text.decode_step_from_embeds_tt(x_embed_dev, pos_dev, rope, ..., page_table)``. The output
    hidden lands in a persistent ``hidden_dev`` buffer that replay overwrites in place, so callers
    read it after each ``execute()`` (e.g. feed the acoustic model) without reallocation.
    """

    text: Any  # VoxtralTTTextModel
    bufs: VoxtralDecodeBuffers
    page_table: ttnn.Tensor
    mesh_device: Any
    trace_id: object = None
    hidden_dev: ttnn.Tensor = None

    def _run_forward(self) -> ttnn.Tensor:
        inner = self.text.inner
        rope_global = inner.rope_setup.get_rot_mats(self.bufs.rot_idxs_dev)
        rope_local = (
            inner.rope_local_setup.get_rot_mats(self.bufs.rot_idxs_dev)
            if getattr(inner, "rope_local_setup", None) is not None
            else None
        )
        return self.text.decode_step_from_embeds_tt(
            self.bufs.x_embed_dev,
            self.bufs.pos_dev,
            rope_global,
            rope_local,
            self.page_table,
        )

    def compile(self) -> ttnn.Tensor:
        """Untraced warm-up pass (populates the program/kernel cache before capture)."""
        out = self._run_forward()
        ttnn.synchronize_device(self.mesh_device)
        return out

    def capture(self) -> None:
        """Capture the trace bound to the persistent buffers. Call after ``compile()``."""
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.hidden_dev = self._run_forward()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def execute(self, *, blocking: bool = False) -> ttnn.Tensor:
        """Replay the captured decode trace on CQ0. Returns the persistent ``hidden_dev`` buffer."""
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=blocking)
        return self.hidden_dev

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None


# ---------------------------------------------------------------------------
# Trace capture / replay of the acoustic flow-matching (FM) Euler core
# ---------------------------------------------------------------------------
@dataclass
class AcousticFMBuffers:
    """Persistent device buffers bound by the acoustic FM trace (rewritten each step)."""

    llm_dev: ttnn.Tensor  # [bsz, 1, dim] bf16 TILE — the text hidden tile (host-staged via 4D->3D reshape)
    noise_dev: ttnn.Tensor  # [bsz, 1, n_acoustic] — FM initial noise (host-staged, seeded torch RNG)

    @classmethod
    def create(cls, mesh_device, acoustic, bsz: int, dim: int) -> "AcousticFMBuffers":
        llm0 = ttnn.from_torch(
            torch.zeros(bsz, 1, dim, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        llm_dev = ttnn.to_device(llm0, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        noise_dev = ttnn.to_device(acoustic.fm_noise_host_tt(bsz, 0), mesh_device, acoustic._fm_dram_mem_config)
        return cls(llm_dev, noise_dev)

    def deallocate(self) -> None:
        for t in (self.llm_dev, self.noise_dev):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)


@dataclass
class TracedAcousticFM:
    """Captures/replays the acoustic Euler-FM core (``acoustic.fm_decode_codes_tt``) — the ~66us/step,
    78%-of-decode bottleneck. Semantic argmax + end-audio mask + concat (``codes_from_fm``) stay outside
    (the per-frame host ``is_end`` branch isn't traceable). ``out_dev`` is the persistent FM acoustic
    tensor; the caller consumes it via ``codes_from_fm`` before the next replay."""

    acoustic: Any
    bufs: AcousticFMBuffers
    cfg_scalar: float
    mesh_device: Any
    trace_id: object = None
    out_dev: ttnn.Tensor = None

    def _fwd(self) -> ttnn.Tensor:
        return self.acoustic.fm_decode_codes_tt(self.bufs.llm_dev, self.bufs.noise_dev, self.cfg_scalar)

    def compile(self) -> ttnn.Tensor:
        out = self._fwd()
        ttnn.synchronize_device(self.mesh_device)
        return out

    def capture(self) -> None:
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.out_dev = self._fwd()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def execute(self, *, blocking: bool = False) -> ttnn.Tensor:
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=blocking)
        return self.out_dev

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None


def stage_acoustic_inputs(pipe, acoustic, bufs: AcousticFMBuffers, last_hidden_tt, seed: int) -> None:
    """Host→device staging for the acoustic FM trace (CQ0): hidden tile + seeded FM noise."""
    bsz = int(bufs.llm_dev.shape[0])
    llm_host = pipe._acoustic_hidden_host_torch(last_hidden_tt)  # [bsz, 1, dim] bf16
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(llm_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), bufs.llm_dev
    )
    ttnn.copy_host_to_device_tensor(acoustic.fm_noise_host_tt(bsz, seed), bufs.noise_dev)
