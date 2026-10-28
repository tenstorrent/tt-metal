# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving wrapper for google/embeddinggemma-2 text embeddings (tt-inference-server).

Target path in tt-metal: models/demos/embedding_gemma2/demo/generator_vllm.py

tt-inference-server loads this class the same way it loads BGEForEmbedding, BgeM3ForEmbedding and
Qwen3ForEmbedding (cpp_server `EmbeddingGemma2Impl` in embedding_runner.cpp, Python
`EmbeddingGemma2Runner` in tt_model_runners/embedding_runner.py):

    model = EmbeddingGemma2ForEmbedding(device=mesh_device, max_batch_size=32, max_seq_len=8192,
                                        model_name="google/embeddinggemma-2")
    emb = model.forward(input_ids, attention_mask=attention_mask)   # torch.float32 [B, 768], unit L2
    ttnn.synchronize_device(mesh_device)

`input_ids` / `attention_mask` are right-padded [B, L] from
`AutoTokenizer.from_pretrained(model, revision=914f7f89...)(texts, padding=True, truncation=True,
max_length=max_seq_len)`. For a text input these are the ids sentence-transformers passes to the
model (BOS ... EOS; `chat_template.jinja` renders a lone text as the text itself), so `forward`
returns `SentenceTransformer("google/embeddinggemma-2").encode(text)`. Prompts
("task: search result | query: ", ...) are the client's job, as with Qwen3-Embedding instructions.

Serving shapes. A batch runs at the smallest text bucket >= its longest row (TextEncoder buckets,
32 ... max_seq_len) and at a batch size from a fixed ladder (default 1, 8, 32), so the set of
(batch, bucket) shapes is finite and every one is captured as a trace in the constructor (the
server's warmup). A shape is served when batch == 1 or batch * bucket <= max_tokens (default
8192); larger request batches are split into chunks of the largest ladder batch that fits, and a
chunk is padded up to the next ladder batch with copies of its first row (dropped on readback).

Environment overrides (all optional):
    EMBEDDING_GEMMA2_SERVE_TRACE       "0" runs eager (no traces); default "1"
    EMBEDDING_GEMMA2_SERVE_BATCHES     comma list, default "1,8,32" (clipped to max_batch_size)
    EMBEDDING_GEMMA2_SERVE_MAX_TOKENS  batch * bucket limit for batch > 1, default 8192
    EMBEDDING_GEMMA2_PRECISION         precision config file (tt/precision.py)
    EMBEDDING_GEMMA2_REVISION          HF revision (tt/weights.py), default 914f7f89...
    EMBEDDING_GEMMA2_CACHE             ttnn weight cache root (tt/config.py)
"""

from __future__ import annotations

import os

import torch

DEFAULT_SERVE_BATCHES = (1, 8, 32)
DEFAULT_SERVE_MAX_TOKENS = 8192
EMBEDDING_DIM = 768


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v not in ("0", "false", "False", "no")


def batch_ladder(max_batch_size: int, batches=DEFAULT_SERVE_BATCHES) -> tuple[int, ...]:
    """Ladder batch sizes <= max_batch_size, always including 1 and max_batch_size."""
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    return tuple(sorted({1, int(max_batch_size)} | {int(b) for b in batches if 1 <= int(b) <= max_batch_size}))


def shape_fits(batch: int, bucket: int, max_tokens: int) -> bool:
    return batch == 1 or batch * bucket <= max_tokens


def serving_shapes(ladder, buckets, max_seq_len: int, max_tokens: int) -> list[tuple[int, int]]:
    """Every (batch, bucket) the server can run; captured as traces at construction."""
    return [(b, s) for b in ladder for s in buckets if s <= max_seq_len and shape_fits(b, s, max_tokens)]


def plan_chunks(num_rows: int, bucket: int, ladder, max_tokens: int) -> list[tuple[int, int, int]]:
    """Split num_rows rows at one bucket into (start, end, padded_batch) chunks of served shapes."""
    if num_rows < 1:
        return []
    cap = max(b for b in ladder if shape_fits(b, bucket, max_tokens))
    chunks = []
    for start in range(0, num_rows, cap):
        end = min(num_rows, start + cap)
        chunks.append((start, end, min(b for b in ladder if b >= end - start)))
    return chunks


def pad_rows(t: torch.Tensor, batch: int) -> torch.Tensor:
    """Pad dim 0 of [n, L] to `batch` rows with copies of row 0 (finite, discarded on readback)."""
    n = t.shape[0]
    if n == batch:
        return t
    return torch.cat([t, t[:1].expand(batch - n, *t.shape[1:])], dim=0)


class EmbeddingGemma2ForEmbedding:
    """Text-only embeddinggemma-2 for the embedding server: tokenized ids in, [B, 768] out."""

    def __init__(
        self,
        device=None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        model_name: str = "google/embeddinggemma-2",
        *,
        traced: bool | None = None,
        serve_batches=None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        del kwargs  # dtype / model_location_generator style kwargs other wrappers take; precision is file-driven
        if device is None:
            raise ValueError("device is required")
        from models.demos.embedding_gemma2.tt.config import SEQ_BUCKETS
        from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder

        self.device = device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.model_name = model_name
        if serve_batches is None:
            env = os.environ.get("EMBEDDING_GEMMA2_SERVE_BATCHES")
            serve_batches = [int(x) for x in env.split(",") if x.strip()] if env else DEFAULT_SERVE_BATCHES
        self.ladder = batch_ladder(self.max_batch_size, serve_batches)
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else os.environ.get("EMBEDDING_GEMMA2_SERVE_MAX_TOKENS", DEFAULT_SERVE_MAX_TOKENS)
        )
        self.traced = _env_bool("EMBEDDING_GEMMA2_SERVE_TRACE", True) if traced is None else bool(traced)

        self.encoder = TextEncoder.from_pretrained(device, model_name, max_seq_len=self.max_seq_len)
        self.shapes = serving_shapes(self.ladder, SEQ_BUCKETS, self.max_seq_len, self.max_tokens)
        if self.traced:
            self.encoder.capture_all(self.shapes)

    def bucket(self, seq_len: int) -> int:
        from models.demos.embedding_gemma2.tt.config import bucket_for

        return bucket_for(seq_len, self.max_seq_len)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """[B, L] right-padded ids (+ mask) -> float32 [B, 768] unit-norm embeddings, row i for input i."""
        del kwargs
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        B, L = input_ids.shape
        if B > self.max_batch_size:
            raise ValueError(f"batch {B} exceeds max_batch_size {self.max_batch_size}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        bucket = self.bucket(L)
        out = torch.empty((B, EMBEDDING_DIM), dtype=torch.float32)
        for start, end, padded in plan_chunks(B, bucket, self.ladder, self.max_tokens):
            ids = pad_rows(input_ids[start:end], padded)
            mask = pad_rows(attention_mask[start:end], padded)
            emb = self.encoder.encode(ids, mask, traced=self.traced)
            out[start:end] = emb[: end - start]
        return out

    __call__ = forward

    def release(self) -> None:
        self.encoder.release_traces()
