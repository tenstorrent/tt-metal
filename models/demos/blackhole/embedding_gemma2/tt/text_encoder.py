# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TextEncoder: host-side orchestration around TextModel (bucketing, input upload, trace, readback).

Host work lives here and only here: tokenized ids/masks are padded to a physical bucket, uploaded,
the device forward runs (eager or as a trace replay), and outputs are read back. The device
forward itself (`TextModel.forward`) issues no host ops.

Trace policy: one trace per (batch, bucket), all captured by one `capture_all(shapes)` call that
first uploads every bucket constant, allocates every persistent input and compiles every shape
(see its docstring for why). `encode(..., traced=True)` refreshes the persistent inputs with
`copy_host_to_device_tensor` and replays. A new logical length in an already captured bucket
reuses that trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import TextModelConfig, bucket_for, default_weight_cache_dir
from models.demos.blackhole.embedding_gemma2.tt.text_model import TextModel
from models.demos.blackhole.embedding_gemma2.tt.weights import (
    load_text_config_dict,
    load_text_state_dict,
    resolve_snapshot,
)


@dataclass
class HostInputs:
    input_ids: ttnn.Tensor  # host [B, S] uint32 ROW_MAJOR
    keep: ttnn.Tensor  # host [B, 1, 1, S] bf16 TILE
    batch: int
    seq_len: int  # physical bucket
    lengths: list[int]  # logical lengths (count of keep==1 per row)
    logical_seq_len: int


@dataclass
class _Trace:
    trace_id: int
    input_ids: ttnn.Tensor
    keep: ttnn.Tensor
    last_hidden: ttnn.Tensor
    embedding: ttnn.Tensor


class TextEncoder(LightweightModule):
    def __init__(self, model: TextModel, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config = model.config
        self._traces: dict[tuple[int, int], _Trace] = {}
        self._compiled: set[tuple[int, int]] = set()

    @classmethod
    def from_pretrained(
        cls,
        device,
        model_id_or_path: str = "google/embeddinggemma-2",
        *,
        max_seq_len: int = 8192,
        cache_dir: Path | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "TextEncoder":
        snapshot = resolve_snapshot(model_id_or_path)
        config = TextModelConfig.from_hf(load_text_config_dict(snapshot), max_seq_len=max_seq_len)
        if state_dict is None:
            state_dict = load_text_state_dict(snapshot)
        cache_dir = Path(cache_dir) if cache_dir is not None else default_weight_cache_dir() / snapshot.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = TextModel.from_state_dict(state_dict, config, device, cache_dir=cache_dir)
        return cls(model, device)

    # ---------------------------------------------------------------- host side
    def prepare_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> HostInputs:
        """Pad [B, L] ids / mask to the bucket and build host ttnn tensors (no device work)."""
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        B, L = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        S = bucket_for(L, self.config.max_seq_len)
        ids = torch.full((B, S), self.config.pad_token_id, dtype=torch.int32)
        ids[:, :L] = input_ids.to(torch.int32)
        keep = torch.zeros((B, 1, 1, S), dtype=torch.float32)
        keep[:, 0, 0, :L] = attention_mask.to(torch.float32)
        return HostInputs(
            input_ids=ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            keep=ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            batch=B,
            seq_len=S,
            lengths=attention_mask.sum(-1).tolist(),
            logical_seq_len=L,
        )

    def _to_device(self, host: HostInputs) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        ids = ttnn.to_device(host.input_ids, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        keep = ttnn.to_device(host.keep, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ids, keep

    @staticmethod
    def read_embedding(emb: ttnn.Tensor, batch: int) -> torch.Tensor:
        return ttnn.to_torch(emb).float().reshape(batch, -1)[:, :]

    @staticmethod
    def read_last_hidden(last: ttnn.Tensor, batch: int, logical_seq_len: int) -> torch.Tensor:
        t = ttnn.to_torch(last).float()
        return t.reshape(batch, t.shape[-2], t.shape[-1])[:, :logical_seq_len, :]

    # ---------------------------------------------------------------- eager
    def forward(self, input_ids: ttnn.Tensor, keep: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Device-only forward on already uploaded inputs (the measured unit)."""
        return self.model(input_ids, keep)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        traced: bool = False,
        return_last_hidden: bool = False,
    ):
        host = self.prepare_inputs(input_ids, attention_mask)
        if traced:
            last, emb = self.replay(host)
        else:
            key = (host.batch, host.seq_len)
            if self._traces and key not in self._compiled:
                raise RuntimeError(
                    f"eager shape {key} was not compiled before the traces were captured; its new "
                    "program-cache buffers could overlap trace buffers. Include it in capture_all()."
                )
            self.model.prepare(host.seq_len)
            ids, keep = self._to_device(host)
            last, emb = self.forward(ids, keep)
            self._compiled.add(key)
        out = self.read_embedding(emb, host.batch)
        if return_last_hidden:
            return out, self.read_last_hidden(last, host.batch, host.logical_seq_len)
        return out

    # ---------------------------------------------------------------- trace
    def capture_all(self, shapes) -> None:
        """Capture one trace per (batch, bucket) in `shapes`, in three phases.

        Any device allocation that outlives a capture and is made after it can sit in that trace's
        intermediate buffers and be overwritten on replay. This includes the program-cache buffers
        a new program allocates on its first run (measured with TT_METAL_TRACE_ALLOC_TRACKING=1).
        So, before the first capture: (1) upload every bucket constant, (2) allocate every
        persistent input, (3) compile every shape with one eager pass. Only then capture.
        Trace outputs are rewritten by their own replay; read them before replaying another trace.
        Call once per session; eager runs afterwards are allowed only for shapes listed here.
        """
        if self._traces:
            raise RuntimeError("capture_all() may run once, before any other trace is captured")
        shapes = sorted(set((int(b), int(s)) for b, s in shapes))
        for _, s in shapes:
            self.model.prepare(s)
        inputs = {}
        for b, s in shapes:
            dummy = self.prepare_inputs(torch.full((b, s), 2, dtype=torch.int64), torch.ones((b, s), dtype=torch.int64))
            inputs[(b, s)] = self._to_device(dummy)
        for key in shapes:
            last, emb = self.forward(*inputs[key])
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(last)
            ttnn.deallocate(emb)
            self._compiled.add(key)
        from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

        for key in shapes:
            ids, keep = inputs[key]
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            last, emb = self.forward(ids, keep)
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            # outputs are refreshed by every replay of their own trace (see docstring)
            acknowledge_corruptible(last)
            acknowledge_corruptible(emb)
            self._traces[key] = _Trace(trace_id, ids, keep, last, emb)

    def replay(self, host: HostInputs, *, blocking: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (host.batch, host.seq_len)
        if key not in self._traces:
            raise KeyError(f"no trace for (batch, bucket) {key}; captured: {sorted(self._traces)}")
        tr = self._traces[key]
        ttnn.copy_host_to_device_tensor(host.input_ids, tr.input_ids)
        ttnn.copy_host_to_device_tensor(host.keep, tr.keep)
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.last_hidden, tr.embedding

    def execute_captured(self, batch: int, seq_len: int, *, blocking: bool = False):
        """Replay a captured trace on whatever its persistent inputs hold (perf loop helper)."""
        tr = self._traces[(batch, seq_len)]
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.last_hidden, tr.embedding

    def release_traces(self) -> None:
        for tr in self._traces.values():
            ttnn.release_trace(self.device, tr.trace_id)
        self._traces.clear()
