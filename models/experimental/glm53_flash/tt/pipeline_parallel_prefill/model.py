# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel GLM-5.3-Flash prefill (after models/experimental/deepseek_v41_flash/.../pipeline_parallel_prefill).

The Galaxy's 8 x 4 mesh is split into its 1 x 4 rows; each row is one pipeline stage holding a contiguous run of
layers, tensor parallel inside the stage (see ``decoder_layer.py``). The embedding lives on the first stage and the
head (stream mean -> final norm -> vocab-sharded lm_head) on the last.

Activations are ROW_MAJOR everywhere they leave a sublayer: the residual streams travel as bf16 RM
``[1, 1, T, hc * D]`` between layers and between stages (HF's ``[B, S, hc, D]`` memory order, so a stage hand-off is a
plain copy of one replica). A prompt is prefilled in chunks; with ``pipelined=True`` every stage runs in its own host
thread and chunk ``c`` on stage ``s`` overlaps chunk ``c - 1`` on stage ``s + 1``.

Decode reuses the chunk path: a step is a chunk of ``DECODE_ROWS`` rows starting at the session position with one real
row. The per-layer caches (KDA conv tail + recurrent state, MLA latent + indexer pools) live in a ``Session`` and are
trimmed to the real tokens, so prefill and decode steps can be interleaved freely.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

import ttnn

from .common import DRAM, RM, ChunkInfo, TPContext, WeightLoader, pad_rows, to_tile
from .decoder_layer import DecoderLayer
from .weights import ModelSource

SEQ_ALIGN = 128  # prefill chunk rows are padded to the SDPA chunk size
DECODE_ROWS = 32  # rows of a decode step (one real)


@dataclass
class Session:
    """One sequence's caches: ``states[s][i]`` is layer ``i`` of stage ``s``; ``pos`` tokens have been consumed."""

    states: list
    pos: int = 0
    tokens: list = field(default_factory=list)


def plan_layer_placement(n_layers: int, n_stages: int) -> list[list[int]]:
    """Contiguous, as-even-as-possible split (earlier stages take the remainder)."""
    n_stages = min(n_stages, n_layers)
    base, extra = divmod(n_layers, n_stages)
    out, i = [], 0
    for s in range(n_stages):
        n = base + (1 if s < extra else 0)
        out.append(list(range(i, i + n)))
        i += n
    return out


def streams_from_host(ctx: TPContext, x: torch.Tensor, rows: int) -> ttnn.Tensor:
    """[n, hc * D] host streams -> bf16 RM [1, 1, rows, hc * D] replicated on the stage (zero padded rows)."""
    n, w = x.shape
    if n < rows:
        x = torch.cat([x, torch.zeros(rows - n, w, dtype=x.dtype)])
    return ctx.upload(x.to(torch.bfloat16).reshape(1, 1, rows, w), ttnn.bfloat16, RM)


def streams_to_host(ctx: TPContext, streams: ttnn.Tensor) -> torch.Tensor:
    """bf16 RM [1, 1, T, hc * D] (replicated) -> [T, hc * D] host bf16 (chip 0's replica)."""
    t = ctx.download(streams)
    return t.reshape(t.shape[-2], t.shape[-1])


class Stage:
    def __init__(
        self, device, cfg, layer_ids: Sequence[int], source: ModelSource, *, first: bool, last: bool, cache_dir=None
    ):
        self.ctx = ctx = TPContext(device)
        self.cfg = cfg
        self.layer_ids = list(layer_ids)
        self.first, self.last = first, last
        self.hc, self.hidden = cfg.hc_mult, cfg.hidden_size
        self.layers = []
        for idx in self.layer_ids:
            src = source.layer(idx)
            self.layers.append(DecoderLayer(ctx, cfg, idx, src, cache_dir=cache_dir))
            src.release()
        load = WeightLoader(ctx, cache_dir, "model")
        if first:
            self.embed = load("embed", source.embedding, dtype=ttnn.bfloat16, layout=RM)
        if last:
            self.final_norm = load("norm", lambda: source.final_norm().float().reshape(1, 1, 1, -1))
            self.vocab = cfg.vocab_size
            assert self.vocab % (ctx.tp * 32) == 0
            self.lm_head = load("lm_head", lambda: source.lm_head().t(), dtype=ttnn.bfloat16, shard_dim=-1)

    def new_state(self):
        return [layer.new_state() for layer in self.layers]

    def embed_tokens(self, ids: torch.Tensor, rows: int) -> ttnn.Tensor:
        """ids [n] -> bf16 RM streams [1, 1, rows, hc * D] (every stream starts as the token embedding)."""
        ctx = self.ctx
        padded = torch.zeros(1, rows, dtype=torch.int32)
        padded[0, : ids.numel()] = ids.to(torch.int32)
        tok = ttnn.from_torch(
            padded, dtype=ttnn.uint32, layout=RM, device=ctx.device, memory_config=DRAM, mesh_mapper=ctx.replicate()
        )
        e = ttnn.embedding(tok, self.embed, layout=RM, dtype=ttnn.bfloat16, memory_config=DRAM)  # [1, rows, D]
        e = ttnn.reshape(e, (1, 1, rows, self.hidden))
        return ttnn.concat([e] * self.hc, dim=-1)

    def run_layers(self, streams: ttnn.Tensor, states: list, info: ChunkInfo):
        new_states = []
        for layer, st in zip(self.layers, states):
            streams, st = layer.forward(streams, st, info)
            new_states.append(st)
        return streams, new_states

    def logits(self, streams: ttnn.Tensor, row: int) -> torch.Tensor:
        """Head on one row of the final streams -> fp32 logits [vocab] on the host."""
        ctx = self.ctx
        w = self.hc * self.hidden
        x = to_tile(ttnn.slice(streams, (0, 0, row, 0), (1, 1, row + 1, w)), ttnn.float32)
        acc = None
        for i in range(self.hc):
            s = ttnn.slice(x, (0, 0, 0, i * self.hidden), (1, 1, 1, (i + 1) * self.hidden))
            acc = s if acc is None else ttnn.add(acc, s)
        h = ttnn.typecast(ttnn.multiply(acc, 1.0 / self.hc), ttnn.bfloat16)
        h = ttnn.rms_norm(h, weight=self.final_norm, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=ctx.hifi4)
        lg = ttnn.matmul(h, self.lm_head, dtype=ttnn.float32, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        return ctx.download_sharded(lg, dim=-1).float().reshape(-1)[: self.vocab]


class GLMPipelinePrefill:
    """``prefill(ids)`` -> last-token logits; ``decode(token, session)`` / ``generate(ids, n)`` continue a sequence.
    ``stages`` are 1 x tp submeshes in pipeline order."""

    def __init__(
        self,
        stage_devices: Sequence,
        cfg,
        source: Optional[ModelSource] = None,
        *,
        n_layers: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        parallel_build: bool = True,
    ):
        self.cfg = cfg
        self.source = source or ModelSource()
        n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
        self.placement = plan_layer_placement(n_layers, len(stage_devices))
        n = len(self.placement)
        self.hc, self.hidden = cfg.hc_mult, cfg.hidden_size
        _ = self.source.headers  # read the shard index once, before the build threads

        def build(s):
            return Stage(
                stage_devices[s],
                cfg,
                self.placement[s],
                self.source,
                first=s == 0,
                last=s == n - 1,
                cache_dir=cache_dir,
            )

        if parallel_build and n > 1:
            with ThreadPoolExecutor(max_workers=n) as pool:
                self.stages = list(pool.map(build, range(n)))
        else:
            self.stages = [build(s) for s in range(n)]

    # ---------------------------------------------------------------------------------------------------------
    def new_session(self) -> Session:
        return Session(states=[st.new_state() for st in self.stages])

    def _chunks(self, n_tokens: int, chunk: int, offset: int = 0):
        assert chunk % SEQ_ALIGN == 0
        out = []
        for start in range(0, n_tokens, chunk):
            n = min(chunk, n_tokens - start)
            out.append((offset + start, pad_rows(n, SEQ_ALIGN), n))
        return out

    def _stage_step(self, s: int, payload, states, start, rows, n_valid, want_logits: bool):
        """One chunk through stage ``s``. payload: token ids (first stage) or host streams [n_valid, hc * D]."""
        stage = self.stages[s]
        info = ChunkInfo(stage.ctx, start, rows, n_valid, self.cfg.index_kpool)
        if stage.first:
            streams = stage.embed_tokens(payload, rows)
        else:
            streams = streams_from_host(stage.ctx, payload, rows)
        streams, states = stage.run_layers(streams, states, info)
        if stage.last:
            out = stage.logits(streams, n_valid - 1) if want_logits else None
        else:
            out = streams_to_host(stage.ctx, streams)[:n_valid]
        ttnn.deallocate(streams)
        return out, states

    def prefill(
        self,
        ids,
        *,
        chunk: int = 2048,
        pipelined: bool = True,
        all_chunk_logits: bool = False,
        session: Optional[Session] = None,
    ):
        """ids: 1-D token ids -> fp32 logits [vocab] of the last token (or one per chunk with ``all_chunk_logits``).
        With ``session`` the tokens continue that sequence and its caches are advanced in place."""
        ids = torch.as_tensor(ids).reshape(-1)
        offset = session.pos if session is not None else 0
        chunks = [(start, rows, n, start - offset) for start, rows, n in self._chunks(ids.numel(), chunk, offset)]
        states = session.states if session is not None else [st.new_state() for st in self.stages]
        result = self._prefill_chunks(ids, chunks, states, pipelined, all_chunk_logits)
        if session is not None:
            session.pos += ids.numel()
            session.tokens.extend(ids.tolist())
        return result

    def _prefill_chunks(self, ids, chunks, states, pipelined, all_chunk_logits):
        """Run ``chunks`` (start, rows, n_valid, offset into ids) through the stages, advancing ``states`` in place."""
        n_stages = len(self.stages)
        logits = [None] * len(chunks)

        def want(c):
            return all_chunk_logits or c == len(chunks) - 1

        if not pipelined or n_stages == 1:
            for c, (start, rows, n, i0) in enumerate(chunks):
                payload = ids[i0 : i0 + n]
                for s in range(n_stages):
                    payload, states[s] = self._stage_step(s, payload, states[s], start, rows, n, want(c))
                logits[c] = payload
            return logits if all_chunk_logits else logits[-1]

        queues = [queue.Queue() for _ in range(n_stages + 1)]
        errors = []

        def worker(s):
            try:
                for c, (start, rows, n, _) in enumerate(chunks):
                    item = queues[s].get()
                    if item is None:
                        break
                    out, states[s] = self._stage_step(s, item, states[s], start, rows, n, want(c))
                    queues[s + 1].put(out if out is not None else False)
            except BaseException as e:  # surface the stage error and unblock the rest of the pipeline
                errors.append((s, e))
                for q in queues[s + 1 :]:
                    q.put(None)

        threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(n_stages)]
        for t in threads:
            t.start()
        for start, rows, n, i0 in chunks:
            queues[0].put(ids[i0 : i0 + n])
        for c in range(len(chunks)):
            item = queues[n_stages].get()
            if item is None:
                break
            logits[c] = item if item is not False else None
        for t in threads:
            t.join()
        if errors:
            s, e = errors[0]
            raise RuntimeError(f"pipeline stage {s} failed") from e
        return logits if all_chunk_logits else logits[-1]

    # ---------------------------------------------------------------------------------------------------------
    def decode(self, token: int, session: Session) -> torch.Tensor:
        """Append one token to the session -> fp32 logits [vocab] of the next token."""
        payload = torch.tensor([int(token)])
        for s in range(len(self.stages)):
            payload, session.states[s] = self._stage_step(
                s, payload, session.states[s], session.pos, DECODE_ROWS, 1, True
            )
        session.pos += 1
        session.tokens.append(int(token))
        return payload

    def generate(self, ids, max_new_tokens: int, *, chunk: int = 2048, eos_token_id=None, session=None):
        """Greedy decoding: prefill ``ids`` then decode -> (new token ids, session). Every new token but the last is
        fed back, so the session ends holding the prompt and all new tokens except the final one."""
        session = session or self.new_session()
        logits = self.prefill(ids, chunk=chunk, session=session)
        out = []
        for i in range(max_new_tokens):
            tok = int(torch.argmax(logits))
            out.append(tok)
            if i == max_new_tokens - 1 or (eos_token_id is not None and tok == eos_token_id):
                break
            logits = self.decode(tok, session)
        return out, session


GLMPipeline = GLMPipelinePrefill
