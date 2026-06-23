# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2-CQ staging for Voxtral text-decode on Blackhole."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

import ttnn


# ---------------------------------------------------------------------------
# Decode trace settings (default on; configure via code, not environment)
# ---------------------------------------------------------------------------
@dataclass
class DecodeTraceConfig:
    """Runtime flags for traced text-decode replay and optional 2-CQ input staging."""

    decode_trace: bool = True
    decode_trace_2cq: bool = True  # 2 CQs for overlapped input staging (independent of trace on/off)


_decode_trace_config = DecodeTraceConfig()


def get_decode_trace_config() -> DecodeTraceConfig:
    return _decode_trace_config


def configure_decode_trace(
    *,
    decode_trace: bool | None = None,
    decode_trace_2cq: bool | None = None,
) -> DecodeTraceConfig:
    """Set decode-trace flags before opening the device or running the pipeline."""
    if decode_trace is not None:
        _decode_trace_config.decode_trace = decode_trace
    if decode_trace_2cq is not None:
        _decode_trace_config.decode_trace_2cq = decode_trace_2cq
    return _decode_trace_config


def reset_decode_trace_config() -> None:
    global _decode_trace_config
    _decode_trace_config = DecodeTraceConfig()


def decode_trace_enabled() -> bool:
    """True when traced text-decode replay is enabled (default on)."""
    return _decode_trace_config.decode_trace


def decode_trace_2cq_enabled() -> bool:
    """True when the device should use 2 command queues for overlapped decode input staging."""
    return _decode_trace_config.decode_trace_2cq


def num_command_queues_for_decode() -> int:
    """Open the device with this many CQs (pass to ``ttnn.open_mesh_device(..., num_command_queues=)``)."""
    return 2 if decode_trace_2cq_enabled() else 1


# ---------------------------------------------------------------------------
# Host staging tensors (no device=) — ready for copy_host_to_device_tensor
# ---------------------------------------------------------------------------
def embed_host(x_embed_4d: torch.Tensor, mesh_device, cluster_shape=None) -> ttnn.Tensor:
    """Host ttnn.Tensor for decode embed ``[1, 1, 1, dim]`` (bf16; TP column-shards last dim)."""
    from models.experimental.voxtraltts.utils.mesh import (
        voxtral_replicate_mesh_mapper,
        voxtral_tp_shard_last_dim_mapper,
    )

    mapper = (
        voxtral_tp_shard_last_dim_mapper(mesh_device, cluster_shape)
        if cluster_shape is not None
        else voxtral_replicate_mesh_mapper(mesh_device)
    )
    return ttnn.from_torch(
        x_embed_4d.reshape(1, 1, 1, -1).to(torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
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
        x_embed_dev = ttnn.to_device(
            embed_host(init_embed_4d, mesh_device, cluster_shape), mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
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
        ttnn.copy_host_to_device_tensor(
            embed_host(x_embed_4d, self.mesh_device, self.cluster_shape), self.bufs.x_embed_dev, 1
        )
        _stage_pos_rot_tt(self, self.bufs, self.mesh_device, self.cluster_shape, pos, cq_id=1)
        self.write_event = ttnn.record_event(self.mesh_device, 1)

    def write_inputs_cq1_tt(self, x_embed_tt: ttnn.Tensor, pos: int) -> None:
        """CQ1 device embed staging (no host torch tensor)."""
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy(x_embed_tt, self.bufs.x_embed_dev)
        _stage_pos_rot_tt(self, self.bufs, self.mesh_device, self.cluster_shape, pos, cq_id=1)
        self.write_event = ttnn.record_event(self.mesh_device, 1)

    def wait_inputs_ready_cq0(self) -> None:
        """CQ0: block the trace replay until CQ1's input writes have completed."""
        ttnn.wait_for_event(0, self.write_event)

    def signal_trace_done_cq0(self) -> None:
        """CQ0: previous replay finished consuming the buffers; CQ1 may overwrite them."""
        self.op_event = ttnn.record_event(self.mesh_device, 0)


# Short public alias so callers can import ``DecodeTrace2CQ``.
DecodeTrace2CQ = VoxtralDecodeTrace2CQ


def embed_row_tile_tt(embeds_tt: ttnn.Tensor, index: int, dim: int) -> ttnn.Tensor:
    """One prompt/MM row ``[1, 1, 1, local_dim]`` TILE DRAM from ``[S, 1, 1, local_dim]`` device embeds."""
    local_dim = int(embeds_tt.shape[-1])
    row = ttnn.slice(embeds_tt, [index, 0, 0, 0], [index + 1, 1, 1, local_dim])
    row_4d = ttnn.reshape(row, (1, 1, 1, local_dim))
    if row_4d.layout != ttnn.TILE_LAYOUT:
        row_tile = ttnn.to_layout(row_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if row_4d.is_allocated():
            ttnn.deallocate(row_4d)
        return row_tile
    return ttnn.to_memory_config(row_4d, ttnn.DRAM_MEMORY_CONFIG)


def _stage_pos_rot_tt(
    pipe: Optional["VoxtralDecodeTrace2CQ"],
    bufs: VoxtralDecodeBuffers,
    mesh_device,
    cluster_shape,
    pos: int,
    *,
    cq_id: int,
) -> None:
    if pipe is not None:
        ttnn.copy_host_to_device_tensor(pos_host(pos, mesh_device, cluster_shape), bufs.pos_dev, cq_id)
        ttnn.copy_host_to_device_tensor(rot_idxs_host(pos, bufs.rope_setup), bufs.rot_idxs_dev, cq_id)
        return
    ttnn.copy_host_to_device_tensor(pos_host(pos, mesh_device, cluster_shape), bufs.pos_dev)
    ttnn.copy_host_to_device_tensor(rot_idxs_host(pos, bufs.rope_setup), bufs.rot_idxs_dev)


def stage_decode_inputs_tt(
    pipe: Optional["VoxtralDecodeTrace2CQ"],
    bufs: VoxtralDecodeBuffers,
    mesh_device,
    cluster_shape,
    x_embed_tt: ttnn.Tensor,
    pos: int,
) -> None:
    """Device→persistent-buffer staging for one decode step (no torch compute)."""
    if pipe is not None:
        pipe.write_inputs_cq1_tt(x_embed_tt, pos)
        pipe.wait_inputs_ready_cq0()
        return
    ttnn.copy(x_embed_tt, bufs.x_embed_dev)
    _stage_pos_rot_tt(pipe, bufs, mesh_device, cluster_shape, pos, cq_id=0)


def stage_prompt_embed_tt(
    pipe: Optional["VoxtralDecodeTrace2CQ"],
    bufs: VoxtralDecodeBuffers,
    mesh_device,
    cluster_shape,
    embeds_tt: ttnn.Tensor,
    token_index: int,
    pos: int,
    dim: int,
) -> None:
    """Slice one prompt embedding row on device and stage it into the decode trace buffers."""
    x_tile = embed_row_tile_tt(embeds_tt, token_index, dim)
    stage_decode_inputs_tt(pipe, bufs, mesh_device, cluster_shape, x_tile, pos)
    if x_tile.is_allocated():
        ttnn.deallocate(x_tile)


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
    ttnn.copy_host_to_device_tensor(embed_host(x_embed_4d, mesh_device, cluster_shape), bufs.x_embed_dev)
    _stage_pos_rot_tt(pipe, bufs, mesh_device, cluster_shape, pos, cq_id=0)


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
        from models.experimental.voxtraltts.utils.mesh import voxtral_from_torch

        # Persistent buffers must match the replicated topology of the acoustic activations
        # they are ``ttnn.copy``-written from each step (single shard on 1x1, replicated on a mesh).
        llm_dev = voxtral_from_torch(
            torch.zeros(bsz, 1, dim, dtype=torch.bfloat16),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        noise_dev = voxtral_from_torch(
            acoustic.fm_noise_host_torch(bsz, 0),
            mesh_device,
            dtype=acoustic.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=acoustic._fm_dram_mem_config,
        )
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


def stage_acoustic_inputs(
    pipe, acoustic, bufs: AcousticFMBuffers, last_hidden_tt, seed: int, *, noise_rng: str = "ttnn"
) -> None:
    """Stage acoustic FM trace inputs (CQ0): hidden tile + FM noise.

    1×1 (P150) keeps the proven host round-trip staging (host hidden + ``fm_noise_host_tt``) so the
    frame count / audio stay bit-for-bit identical to the reference path. Multi-device (1×4 TP) uses
    the device-resident copy (no host round-trip) which is required for the gathered TP hidden.
    """
    from models.experimental.voxtraltts.utils.mesh import voxtral_is_multi_device_mesh

    bsz = int(bufs.llm_dev.shape[0])
    if voxtral_is_multi_device_mesh(pipe.mesh_device):
        # TP hidden must stay device-resident; FM noise matches the proven P150 host-torch path.
        pipe._stage_acoustic_hidden_to_fm_buffer(last_hidden_tt, bufs.llm_dev)
        if noise_rng == "ttnn":
            noise_tt = acoustic.fm_noise_tt(bsz, seed, rng=noise_rng)
            ttnn.copy(noise_tt, bufs.noise_dev)
            if noise_tt.is_allocated():
                ttnn.deallocate(noise_tt)
        else:
            ttnn.copy_host_to_device_tensor(acoustic.fm_noise_host_tt(bsz, seed), bufs.noise_dev)
        return
    # 1×1 P150 path — host-staged hidden + host FM noise (matches the 392-frame reference numerics).
    llm_host = pipe._acoustic_hidden_host_torch(last_hidden_tt)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(llm_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), bufs.llm_dev
    )
    ttnn.copy_host_to_device_tensor(acoustic.fm_noise_host_tt(bsz, seed), bufs.noise_dev)
