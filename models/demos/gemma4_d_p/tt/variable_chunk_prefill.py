# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced Gemma4 CP prefill with one captured trace per chunk-width bucket.

A request picks its chunk width at admission from its prompt length, then keeps it for its
whole prefill. Each width needs its own captured trace (the graph shapes differ) and its own
pinned staging tensors, but everything else is shared: one set of weights, one KV cache, one
set of ring-gather scratch buffers, one ``PrefillMetadata``.

Why the width must not change mid-request: the ring KV cache is block-cyclic with period C
(local row ``chunk*L + j`` on rank ``r`` holds global token ``chunk*C + r*L + j``, ``L = C/cp``),
and ring_joint SDPA reconstructs a cached row's global position from the *current* chunk's
width. Writing a prefix at one width and reading it at another places the prefix at the wrong
positions -- silently, with no shape error. ``test_variable_chunk_prefill`` pins this with a
negative control.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.gemma4_d_p.tt.chunk_buckets import select_chunk_size, validate_chunk_sizes


def cp_shard_mapper(mesh_config, seq_dim=-1):
    """Shard ``seq_dim`` across the CP axis, replicating along TP."""
    mesh_dims = [None, None]
    mesh_dims[mesh_config.cp_axis] = seq_dim
    return mesh_config.shard_mapper(mesh_dims=mesh_dims)


def cp_gather_torch(tensor, mesh_config):
    """Read a CP-sharded mesh tensor back as one torch tensor in position order."""
    shards = ttnn.get_device_tensors(tensor)
    cp = mesh_config.cp_degree
    if cp <= 1:
        return ttnn.to_torch(shards[0]).float()
    cp_stride = mesh_config.tp_degree if mesh_config.cp_axis == 0 else 1
    return torch.cat([ttnn.to_torch(shards[r * cp_stride]).float() for r in range(cp)], dim=-2)


@dataclass
class ChunkBucket:
    """Per-width device state: pinned trace inputs and the captured trace."""

    chunk_size: int
    tokens: ttnn.Tensor
    positions: ttnn.Tensor
    trace_id: int | None = None
    output: ttnn.Tensor | None = None


@dataclass
class PrefillResult:
    """One request's outcome. ``device_s`` excludes host staging, as the demos report it."""

    prompt_len: int
    chunk_size: int
    num_chunks: int
    padded_len: int
    device_s: float
    stage_s: float
    per_chunk_s: list = field(default_factory=list)

    @property
    def padding_waste(self):
        """Padded tokens per real token. 1.0 means the prompt filled its chunks exactly."""
        return self.padded_len / self.prompt_len

    def describe(self):
        return (
            f"prompt={self.prompt_len} chunk={self.chunk_size} n_chunks={self.num_chunks} "
            f"padded={self.padded_len} ({self.padding_waste:.2f}x) "
            f"device={self.device_s * 1000:.1f}ms stage={self.stage_s * 1000:.1f}ms"
        )


class VariableChunkPrefill:
    """Drive a multi-width ``Gemma4Model``: capture a trace per width, prefill per request."""

    def __init__(self, model, mesh_config, chunk_sizes=None, pad_token_id=0):
        self.model = model
        self.mesh_config = mesh_config
        self.mesh_device = mesh_config.device
        self.pad_token_id = pad_token_id
        chunk_sizes = model.prefill_chunk_sizes if chunk_sizes is None else chunk_sizes
        self.chunk_sizes = validate_chunk_sizes(chunk_sizes, mesh_config.cp_degree, model.max_seq_len)
        missing = [c for c in self.chunk_sizes if c not in model.prefill_chunk_sizes]
        if missing:
            raise ValueError(
                f"widths {missing} have no RoPE table on this model (built for "
                f"{list(model.prefill_chunk_sizes)}); rebuild it with every width you intend to serve"
            )
        self.buckets = {c: self._make_bucket(c) for c in self.chunk_sizes}
        # The model must not refresh ring metadata itself: the driver does it outside the trace.
        model._prefill_metadata_external = True

    # ── setup ────────────────────────────────────────────────────────────────

    def _make_bucket(self, chunk_size):
        """Allocate the pinned, CP-sharded trace inputs for one width."""

        def _upload(values):
            return ttnn.to_device(
                ttnn.from_torch(
                    values.unsqueeze(0),
                    device=None,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=cp_shard_mapper(self.mesh_config, seq_dim=-1),
                ),
                device=self.mesh_device,
            )

        return ChunkBucket(
            chunk_size=chunk_size,
            tokens=_upload(torch.zeros(chunk_size, dtype=torch.int32)),
            positions=_upload(torch.arange(0, chunk_size, dtype=torch.int32)),
        )

    def _forward(self, bucket, chunk_start):
        self.model.set_prefill_rope_positions(bucket.positions)
        embeds = self.model.transform_and_embed_prefill_inputs_device(bucket.tokens)
        return self.model(hidden_states=embeds, chunk_start_idx=chunk_start, user_id=0)

    def capture(self, logger=None):
        """Compile and capture one trace per width. Call once, before any prefill."""
        for chunk_size, bucket in self.buckets.items():
            self._stage(bucket, chunk_idx=0, tokens_padded=None)
            t0 = time.time()
            warmup = self._forward(bucket, 0)
            ttnn.synchronize_device(self.mesh_device)
            warmup.deallocate(True)
            compile_s = time.time() - t0

            t0 = time.time()
            bucket.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            bucket.output = self._forward(bucket, 0)
            ttnn.end_trace_capture(self.mesh_device, bucket.trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            if logger is not None:
                logger.info(
                    f"[variable_chunk] chunk={chunk_size} compile={compile_s:.1f}s " f"capture={time.time() - t0:.1f}s"
                )
        return self

    def release(self):
        for bucket in self.buckets.values():
            if bucket.trace_id is not None:
                ttnn.release_trace(self.mesh_device, bucket.trace_id)
                bucket.trace_id = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
        return False

    # ── per-request ──────────────────────────────────────────────────────────

    def select_chunk_size(self, prompt_len):
        """The width this driver would admit ``prompt_len`` at."""
        return select_chunk_size(prompt_len, self.chunk_sizes)

    def _stage(self, bucket, chunk_idx, tokens_padded):
        """Refresh everything that varies per chunk. Never inside a trace."""
        chunk_size = bucket.chunk_size
        chunk_start = chunk_idx * chunk_size
        if tokens_padded is not None:
            staged = ttnn.from_torch(
                tokens_padded[chunk_start : chunk_start + chunk_size].unsqueeze(0).contiguous(),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=cp_shard_mapper(self.mesh_config, seq_dim=-1),
            )
            ttnn.copy_host_to_device_tensor(staged, bucket.tokens)
        positions = ttnn.from_torch(
            torch.arange(chunk_start, chunk_start + chunk_size, dtype=torch.int32).unsqueeze(0),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=cp_shard_mapper(self.mesh_config, seq_dim=-1),
        )
        ttnn.copy_host_to_device_tensor(positions, bucket.positions)
        self.model.prefill_metadata.update(slot_idx=0, kv_actual_global=chunk_start)
        return chunk_start

    def prefill(self, tokens, chunk_size=None, on_chunk=None):
        """Prefill one request, returning its ``PrefillResult``.

        ``tokens`` is a 1-D or ``[1, n]`` int tensor. A prompt that does not fill whole chunks
        is padded up to ``n_chunks * chunk_size``; causality makes the real prefix's outputs
        independent of the padding, and the padded tokens are exactly the wasted work a
        narrower bucket avoids.
        """
        tokens = tokens.reshape(-1)
        prompt_len = int(tokens.shape[0])
        chunk_size = self.select_chunk_size(prompt_len) if chunk_size is None else chunk_size
        if chunk_size not in self.buckets:
            raise ValueError(f"no captured trace for chunk {chunk_size}; have {sorted(self.buckets)}")
        bucket = self.buckets[chunk_size]
        if bucket.trace_id is None:
            raise RuntimeError("call capture() before prefill()")

        num_chunks = math.ceil(prompt_len / chunk_size)
        padded_len = num_chunks * chunk_size
        if padded_len > self.model.max_seq_len:
            raise ValueError(
                f"prompt {prompt_len} padded to {padded_len} at chunk {chunk_size} exceeds "
                f"max_seq_len {self.model.max_seq_len}"
            )
        tokens_padded = torch.full((padded_len,), self.pad_token_id, dtype=torch.int32)
        tokens_padded[:prompt_len] = tokens.to(torch.int32)

        per_chunk, stage_s = [], 0.0
        for chunk_idx in range(num_chunks):
            device_s, staged_s = self.run_chunk(chunk_size, chunk_idx, tokens_padded)
            per_chunk.append(device_s)
            stage_s += staged_s
            if on_chunk is not None:
                on_chunk(chunk_idx, bucket.output)

        return PrefillResult(
            prompt_len=prompt_len,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            padded_len=padded_len,
            device_s=sum(per_chunk),
            stage_s=stage_s,
            per_chunk_s=per_chunk,
        )

    def run_chunk(self, chunk_size, chunk_idx, tokens_padded):
        """Stage and replay exactly one chunk at ``chunk_start = chunk_idx * chunk_size``.

        The unit ``prefill`` is built from, exposed because it is also what a negative control
        needs: continuing a prefix at a width other than the one that wrote it.
        Returns ``(device_s, stage_s)``.
        """
        bucket = self.buckets[chunk_size]
        if bucket.trace_id is None:
            raise RuntimeError("call capture() before running a chunk")
        t0 = time.time()
        self._stage(bucket, chunk_idx, tokens_padded)
        stage_s = time.time() - t0
        t0 = time.time()
        ttnn.execute_trace(self.mesh_device, bucket.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        return time.time() - t0, stage_s

    def last_hidden_state(self, chunk_size):
        """Host copy of the last replayed chunk's hidden states, in position order.

        Returns ``[chunk_size, hidden]``; the row for global position ``p`` is
        ``p - chunk_idx * chunk_size``. The final real prompt token's row is what seeds decode.
        """
        hidden = cp_gather_torch(self.buckets[chunk_size].output, self.mesh_config)
        return hidden.reshape(-1, hidden.shape[-1])
