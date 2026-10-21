# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident multi-layer GLM-5.3-Flash decode model on a mesh: embedding -> N decoder layers -> head, one
traced decode step for ``rows * users_per_row`` users (design section 5: TP over the columns, one user group per
row, hybrid experts). Every layer is a stage-1 ``FunctionalDecoder`` built on the mesh; the embedding table and
the head: embedding replicated on every chip, the head vocab split over the TP columns (all-gathered logits).

Memory per chip at 8x4: experts 42 layers x 7.7 GB / 32 ~ 10 GB, non-expert weights / 4 ~ 2.2 GB, embedding
1.27 GB, head 0.63 GB (bfp8), caches and states per layer.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import DecodePositions, FunctionalDecoder, PrefillTraceCall
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.trace_prefill import (
    DEFAULT_LENGTHS,
    PrefillSlot,
    PrefillTraceInputs,
    PrefillTraceResult,
    pick_prefill_length,
    supported_prefill_lengths,
)


class GLM53MeshModel:
    def __init__(
        self,
        mesh_device,
        cfg: GLM53FlashConfig,
        layers: list[int],
        *,
        max_batch: int,
        max_seq_len: int,
        snapshot: Path | None = None,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        build_head: bool = True,
    ):
        self.device = mesh_device
        self.cfg = cfg
        self.mesh = GLMMesh.from_device(mesh_device)
        self.layer_ids = list(layers)
        self.max_batch = max_batch  # per row
        self.total_users = max_batch * self.mesh.rows
        self.max_seq_len = max_seq_len
        snapshot = snapshot or default_snapshot_dir()
        headers = shard_headers(snapshot)
        t0 = time.time()
        self.embedding = TtEmbedding.from_checkpoint(mesh_device, cfg, snapshot, headers)
        self.head = (
            TtModelHead.from_checkpoint(mesh_device, cfg, snapshot, headers, mesh=self.mesh) if build_head else None
        )
        logger.info(f"mesh model: embedding + head built in {time.time() - t0:.1f} s")
        self.layers: list[FunctionalDecoder] = []
        self.build_times: dict[int, float] = {}
        for i in self.layer_ids:
            t0 = time.time()
            raw = load_layer_raw(snapshot, i, headers)
            t1 = time.time()
            dec = FunctionalDecoder.from_state_dict(
                raw,
                hf_config=cfg,
                layer_idx=i,
                mesh_device=mesh_device,
                max_batch=max_batch,
                max_seq_len=max_seq_len,
                weight_cache_path=weight_cache_path / f"layer{i}" if weight_cache_path else None,
                weight_dtype=weight_dtype,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
            )
            del raw
            self.layers.append(dec)
            self.build_times[i] = time.time() - t0
            logger.info(f"mesh model: layer {i} built in {self.build_times[i]:.1f} s (host load {t1 - t0:.1f} s)")
        self.positions: DecodePositions | None = None
        self.ids_dev: ttnn.Tensor | None = None
        # the traced prefill's device-side user slot (tt/trace_prefill.py): allocated here, before any trace
        # capture, because a persistent tensor allocated after a capture may alias that trace's transient buffers
        self.prefill_slot: PrefillSlot | None = PrefillSlot.allocate(self.mesh, self.layers) if self.layers else None
        self.prefill_traces: dict[int, PrefillTraceInputs] = {}

    # ------------------------------------------------------------------ inputs
    def ids_to_device(self, ids: torch.Tensor) -> ttnn.Tensor:
        """``[rows*B]`` token ids (row r's users first) -> the persistent ``[1,1,1,rows*B]`` uint32 input,
        row-sharded on a multi-row mesh (each row embeds its own B tokens)."""
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        if self.mesh.rows > 1:
            return self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def load_ids(self, ids: torch.Tensor):
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        mapper = self.mesh.shard_rows(3) if self.mesh.rows > 1 else None
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), self.ids_dev
        )

    def set_positions(self, host_positions: list[int]):
        if self.positions is None:
            self.positions = self.layers[0].make_decode_positions(host_positions)
        else:
            self.layers[0].update_decode_positions(self.positions, host_positions)

    # ------------------------------------------------------------------ step
    def decode_step(self, ids_dev: ttnn.Tensor, positions: DecodePositions, keep_layer_outputs: bool = False):
        """ids [1,1,1,B] (per row) -> (streams after the last layer [1,1,B,16384], logits, greedy ids).
        ``keep_layer_outputs`` (eager diagnostics only) keeps every layer's output in ``self.layer_outputs``."""
        streams = self.embedding(ids_dev)
        self.layer_outputs = []
        last = len(self.layers) - 1
        for k, dec in enumerate(self.layers):
            # the last layer feeds the head: its FFN wrap emits the mean over the streams directly
            collapse = k == last and self.head is not None and not keep_layer_outputs and dec.supports_collapse
            out = dec.decode_forward(streams, positions=positions, collapse=collapse)
            if keep_layer_outputs:
                self.layer_outputs.append(out)
            else:
                ttnn.deallocate(streams)
            streams = out
        if self.head is None:
            return streams, None, None
        logits, ids = self.head.forward_greedy(streams)
        return streams, logits, ids

    # ------------------------------------------------------------------ prefill (TTFT)
    def prefill(self, prompt_ids: torch.Tensor, user: int = 0, start: int = 0):
        """One prompt (or one chunk of it: ``start`` > 0 continues the user's prompt, the
        ``FunctionalDecoder`` chunked-prefill contract) through every layer (sequence-parallel over the
        rows on a multi-row mesh) and the head on its last token. Returns ``(streams_local, logits_last
        [1,1,1,V] per chip, owner_row)``: the row holding the last token is ``owner_row`` (every row
        computes the head on its own local row ``(S-1) % T_local``; only the owner's result is the
        prompt's)."""
        S = int(prompt_ids.numel())
        unit = 32 * self.mesh.rows
        T_pad = -(-S // unit) * unit
        host = self.embedding.tokens_to_host(prompt_ids, pad_to=T_pad)  # [1,1,1,T_pad]
        if self.mesh.rows > 1:
            ids_dev = self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        streams = self.embedding(ids_dev)  # [1,1,T_pad/rows,16384] per row (its slab)
        ttnn.deallocate(ids_dev)
        for dec in self.layers:
            out = dec.prefill_forward(streams, user=user, seq_len=S, start=start)
            ttnn.deallocate(streams)
            streams = out
        T_loc = T_pad // self.mesh.rows
        owner_row, local_last = divmod(S - 1, T_loc)
        logits = self.head.last_token_logits(streams, local_last + 1) if self.head is not None else None
        return streams, logits, owner_row

    def prefill_chunked(self, prompt_ids: torch.Tensor, user: int = 0, chunk: int = 32768, on_chunk=None):
        """A long prompt through ``prefill`` in consecutive chunks (every chunk but the last is a multiple
        of ``32 * rows`` tokens, so the row slabs stay tile-aligned). ``on_chunk(start, end, seconds)`` is
        called after every chunk (synchronised). Returns the LAST chunk's ``(streams, logits_last,
        owner_row)``: its ``owner_row`` / local row index the prompt's last token."""
        S = int(prompt_ids.numel())
        unit = 32 * self.mesh.rows
        chunk = max(unit, (chunk // unit) * unit)
        out = None
        for s0 in range(0, S, chunk):
            e = min(S, s0 + chunk)
            if out is not None:
                ttnn.deallocate(out[0])
                if out[1] is not None:
                    ttnn.deallocate(out[1])
            t0 = time.perf_counter()
            out = self.prefill(prompt_ids[s0:e], user=user, start=s0)
            if on_chunk is not None:
                ttnn.synchronize_device(self.device)
                on_chunk(s0, e, time.perf_counter() - t0)
        return out

    def first_token(self, logits_or_ids, owner_row: int = 0, row: int = 0) -> int:
        """The prompt's first greedy token from the owner row's chip: ``logits_or_ids`` is the eager
        ``prefill``'s last-token logits (argmax taken here), a greedy-id tensor (``row`` of it), or a
        ``PrefillTraceResult`` (its own tile row)."""
        if isinstance(logits_or_ids, PrefillTraceResult):
            res = logits_or_ids
            logits_or_ids, owner_row, row = res.ids, res.owner_row, res.row
        ids = logits_or_ids if logits_or_ids.dtype == ttnn.uint32 else self.head.greedy(logits_or_ids)
        shard = self.mesh.device_tensors(ids)[owner_row * self.mesh.cols]
        return int(ttnn.to_torch(shard).view(torch.int32).reshape(-1)[row])

    # ------------------------------------------------------------- traced prefill
    def trace_prefill_lengths(self, lengths=DEFAULT_LENGTHS) -> list[int]:
        """The traced chunk lengths this model supports (``trace_prefill.supported_prefill_lengths``)."""
        max_rows = -(-self.max_seq_len // 32) * 32
        return supported_prefill_lengths(self.mesh.rows, max_rows, lengths)

    def has_prefill_trace(self, S: int) -> bool:
        return pick_prefill_length(S, self.prefill_traces) is not None

    def _prefill_graph(self, inp: PrefillTraceInputs):
        """The traced prefill graph on the persistent inputs ``inp`` + ``self.prefill_slot`` (no host value
        of the prompt inside): embedding -> every layer (device-slot form) -> head on the selected row."""
        call = PrefillTraceCall(slot=self.prefill_slot, end=inp.end, ring_sel=inp.ring_sel)
        streams = self.embedding(inp.ids)  # [1,1,T_loc,16384] per row
        for dec in self.layers:
            out = dec.prefill_forward(streams, user=self.prefill_slot.user, seq_len=inp.T, start=0, trace=call)
            ttnn.deallocate(streams)
            streams = out
        if self.head is None:
            return streams
        logits = self.head.tile_logits(streams, inp.tile_start, inp.tile_end)  # [1,1,32,V]: the last token's tile
        ids = self.head.greedy(logits)
        # the tile outputs go to buffers that predate every capture (they survive other traces' replays and the
        # capture-created tensors die inside the trace); the streams stay the capture's transient output
        ttnn.copy(logits, inp.out_logits)
        ttnn.copy(ids, inp.out_ids)
        ttnn.deallocate(logits)
        ttnn.deallocate(ids)
        return streams

    def capture_prefill(self, lengths=None, cq_id: int = 0, warmup_user: int = 0) -> list[int]:
        """Allocate the persistent inputs of every requested length (default ``trace_prefill_lengths``),
        compile each graph eagerly (a dummy prompt of the padded length into slot ``warmup_user``), then
        capture one trace per length. Every allocation precedes the first capture; the warm-ups and the
        captures leave dummy data in slot ``warmup_user`` (the next prefill of that slot overwrites it).
        Call it before ``capture`` (the decode trace) or after ``warmup_decode``, so the decode step's
        persistent inputs exist before any capture too. Returns the lengths captured now."""
        if self.prefill_slot is None:
            raise RuntimeError("no layers")
        todo = [
            T for T in (self.trace_prefill_lengths() if lengths is None else lengths) if T not in self.prefill_traces
        ]
        if not todo:
            return []
        vocab = self.cfg.vocab_size if self.head is not None else None
        new = {T: PrefillTraceInputs.allocate(self.mesh, T, self.cfg.hidden_size, vocab) for T in todo}
        self.prefill_slot.set_user(warmup_user)
        for T, inp in new.items():  # program-cache warm-up with the exact shapes and argument values
            t0 = time.perf_counter()
            dummy = torch.arange(T, dtype=torch.long) % 1000 + 1000
            inp.set_prompt(self.mesh, self.embedding.tokens_to_host(dummy, pad_to=T), T)
            ttnn.deallocate(self._prefill_graph(inp))
            ttnn.synchronize_device(self.device)
            logger.info(f"traced prefill: T={T} compiled in {time.perf_counter() - t0:.1f} s")
        for T, inp in new.items():
            t0 = time.perf_counter()
            inp.trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
            inp.streams = self._prefill_graph(inp)
            ttnn.end_trace_capture(self.device, inp.trace_id, cq_id=cq_id)
            ttnn.synchronize_device(self.device)
            self.prefill_traces[T] = inp
            logger.info(f"traced prefill: T={T} captured in {time.perf_counter() - t0:.1f} s")
        return todo

    def prefill_traced(self, prompt_ids: torch.Tensor, user: int = 0, cq_id: int = 0) -> PrefillTraceResult:
        """One prompt through the captured trace of the smallest supported length >= its token count
        (``ValueError`` when none fits: use ``prefill``). Refreshes the slot / per-length inputs, replays,
        and returns a ``PrefillTraceResult``: the last layer's ``streams`` (``[1,1,T_loc,16384]`` per row,
        transient: read before another replay), the ``logits`` ``[1,1,32,V]`` / greedy ``ids`` ``[1,1,32,1]``
        of the last token's tile (persistent buffers) and ``owner_row`` / ``row`` locating the prompt's
        token (``first_token(result)``)."""
        S = int(prompt_ids.numel())
        T = pick_prefill_length(S, self.prefill_traces)
        if T is None:
            raise ValueError(f"no prefill trace for {S} tokens (captured: {sorted(self.prefill_traces)})")
        inp = self.prefill_traces[T]
        self.prefill_slot.set_user(user)
        inp.set_prompt(self.mesh, self.embedding.tokens_to_host(prompt_ids, pad_to=T), S)
        ttnn.execute_trace(self.device, inp.trace_id, cq_id=cq_id, blocking=True)
        for dec in self.layers:
            dec.next_start[user] = S
        return PrefillTraceResult(inp.streams, inp.out_logits, inp.out_ids, inp.owner_row(S), inp.last_row(S), S, T)

    def release_prefill_traces(self):
        for inp in self.prefill_traces.values():
            ttnn.release_trace(self.device, inp.trace_id)
        self.prefill_traces = {}

    def ids_to_host(self, ids: ttnn.Tensor) -> torch.Tensor:
        """greedy ids [1,1,B,1] uint32 per chip -> [rows*B] int64."""
        t = self.mesh.to_host(ids, rows=2 if self.mesh.rows > 1 else None)
        return t.view(torch.int32).reshape(-1).long()

    # ------------------------------------------------------------------ trace
    def warmup_decode(self, ids: torch.Tensor, host_positions: list[int]):
        """Allocate the persistent decode inputs (once) and run one eager decode step (compiles the
        programs, creates every lazily built constant) so that ``capture_prefill`` and ``capture`` can
        follow in either order with nothing allocated after a capture. Consumes the token."""
        if self.ids_dev is None:
            self.ids_dev = self.ids_to_device(ids)
        else:
            self.load_ids(ids)
        self.set_positions(host_positions)
        for t in self.decode_step(self.ids_dev, self.positions):
            if t is not None:
                ttnn.deallocate(t)
        ttnn.synchronize_device(self.device)

    def capture(self, ids: torch.Tensor, host_positions: list[int], cq_id: int = 0):
        """Warm-up (compile) + capture one decode step on persistent inputs; returns the trace id and the
        captured output tensors. The warm-up and the capture both consume the token (state advances twice)."""
        self.warmup_decode(ids, host_positions)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        outs = self.decode_step(self.ids_dev, self.positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        return trace_id, outs

    def replay(self, trace_id, cq_id: int = 0, blocking: bool = True):
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)

    def reset_kda_states(self):
        for dec in self.layers:
            if dec.kda_batched is None:
                continue
            for t in (dec.kda_batched.recurrent, dec.kda_batched.convolution):
                z = ttnn.zeros(
                    tuple(t.shape),
                    dtype=t.dtype,
                    layout=t.layout,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(z, t)
                ttnn.deallocate(z)
