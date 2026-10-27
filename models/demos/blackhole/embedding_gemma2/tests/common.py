# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers: deterministic natural-text inputs, padding, upload, PCC/cosine, reference cache."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.embedding_gemma2.tt.config import bucket_for

PCC_LAYER = 0.995  # per-layer / per-module bar
COSINE_EMBEDDING = 0.99  # pooled + normalized embedding bar (per prompt)

# Logical lengths around tile (32), window (512 radius -> 1025 span) and bucket boundaries.
SEQ_LENS = (1, 7, 31, 33, 511, 512, 513, 1025)
LONG_SEQ_LEN = 8192

PADDED_BATCHES = ((1025, 513, 31), (512, 33))
# batch-1 lengths + long length + padded batches (each row a different logical length)
CASES = tuple((L,) for L in SEQ_LENS) + ((LONG_SEQ_LEN,),) + PADDED_BATCHES


def case_id(lengths) -> str:
    return f"b{len(lengths)}_" + "_".join(f"s{n}" for n in lengths)


CASE_IDS = tuple(case_id(c) for c in CASES)

_TT_METAL_ROOT = Path(__file__).resolve().parents[5]


@functools.lru_cache(maxsize=1)
def _corpus_ids(snapshot: str) -> tuple[int, ...]:
    """Token ids of a fixed natural-language corpus (model card + an in-repo tech report)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot)
    text = (
        (Path(snapshot) / "README.md").read_text() + "\n" + (_TT_METAL_ROOT / "tech_reports/LLMs/llms.md").read_text()
    )
    ids = tok(text, add_special_tokens=True)["input_ids"]
    return tuple(ids)


def make_batch(snapshot, lengths: list[int], *, offsets: list[int] | None = None):
    """Right-padded [B, max(lengths)] ids + attention mask from the corpus (row b starts at offsets[b])."""
    corpus = _corpus_ids(str(snapshot))
    B, L = len(lengths), max(lengths)
    offsets = offsets or [b * 997 for b in range(B)]
    ids = torch.zeros((B, L), dtype=torch.int64)
    mask = torch.zeros((B, L), dtype=torch.int64)
    for b, (n, off) in enumerate(zip(lengths, offsets)):
        reps = (off + n) // len(corpus) + 1
        src = (corpus * reps)[off : off + n]
        ids[b, :n] = torch.tensor(src)
        ids[b, 0] = 2  # BOS
        mask[b, :n] = 1
    return ids, mask


def upload_inputs(device, ids: torch.Tensor, mask: torch.Tensor, max_seq_len: int):
    """Pad to the bucket and upload: returns (ids_dev [B,S] uint32 RM, keep_dev [B,1,1,S] bf16 TILE, S)."""
    B, L = ids.shape
    S = bucket_for(L, max_seq_len)
    ids_p = torch.zeros((B, S), dtype=torch.int32)
    ids_p[:, :L] = ids.to(torch.int32)
    keep = torch.zeros((B, 1, 1, S), dtype=torch.float32)
    keep[:, 0, 0, :L] = mask.float()
    ids_dev = ttnn.from_torch(ids_p, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    keep_dev = ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return ids_dev, keep_dev, S


def upload_activation(device, x: torch.Tensor, S: int) -> ttnn.Tensor:
    """[B, L, D] torch -> [B, 1, S, D] bf16 TILE device tensor, zero padded to S rows."""
    B, L, D = x.shape
    xp = torch.zeros((B, 1, S, D), dtype=torch.float32)
    xp[:, 0, :L] = x.float()
    return ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def download(t: ttnn.Tensor, B: int, L: int) -> torch.Tensor:
    """[B, 1, S, D] device -> [B, L, D] float32."""
    x = ttnn.to_torch(t).float()
    return x.reshape(B, x.shape[-2], x.shape[-1])[:, :L]


def valid_rows(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate the valid (mask == 1) token rows of [B, L, D] into [N, D]."""
    return x[mask.bool()]


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    _, value = comp_pcc(a.float(), b.float(), 0.0)
    return float(value)


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def log_result(name: str, value: float, bar: float, **fields):
    """Log a metric and, when EMBEDDING_GEMMA2_RESULTS names a file, append it there as JSON."""
    logger.info(f"{name}: {value:.6f} (bar {bar}) {fields}")
    path = os.environ.get("EMBEDDING_GEMMA2_RESULTS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"metric": name, "value": value, "bar": bar, **fields}) + "\n")
    return value >= bar
