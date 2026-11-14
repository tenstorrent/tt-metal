# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Routing captures: per-token selected experts for every MoE layer of one sequence.

On-disk format (``routing.npz``), independent of where the routing came from:
  expert_ids      uint16/int32 [layers, tokens, top_k]   global logical expert ids
  expert_weights  float16      [layers, tokens, top_k]   optional
  layers          int16        [layers]                  global layer ids

Adapters produce it from the device (``compact_torch_dumps`` reads the TtMoe ``PREFILL_DUMP_ROUTING_DIR``
format); ``import_routing`` accepts any other source (a reference model, vLLM dumps, ...) as arrays.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def validate(expert_ids: np.ndarray, num_experts: int, top_k: int, tokens: int | None = None) -> None:
    if expert_ids.ndim != 3 or expert_ids.shape[2] != top_k:
        raise ValueError(f"expert_ids must be [layers, tokens, {top_k}], got {expert_ids.shape}")
    if tokens is not None and expert_ids.shape[1] != tokens:
        raise ValueError(f"expected {tokens} tokens, got {expert_ids.shape[1]}")
    if expert_ids.min() < 0 or expert_ids.max() >= num_experts:
        raise ValueError(f"expert ids outside [0, {num_experts})")
    s = np.sort(expert_ids, axis=2)
    if (np.diff(s, axis=2) == 0).any():
        raise ValueError("a token selects the same expert twice")


def save(path: Path, expert_ids: np.ndarray, layers, expert_weights: np.ndarray | None = None) -> str:
    dtype = np.uint16 if expert_ids.max() < 65536 else np.int32
    arrays = dict(expert_ids=expert_ids.astype(dtype), layers=np.asarray(layers, dtype=np.int16))
    if expert_weights is not None:
        arrays["expert_weights"] = expert_weights.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> tuple[np.ndarray, list[int]]:
    z = np.load(path)
    return z["expert_ids"], z["layers"].tolist()


def compact_torch_dumps(
    dump_dir: Path, out: Path, num_experts: int, top_k: int, chunk_tokens: int, n_chunks: int
) -> dict:
    """TtMoe routing dumps (one .pt per layer and chunk: layer_idx, actual_start, expert_ids, expert_weights)
    -> routing.npz. Every layer must have every chunk exactly once."""
    import torch

    files = sorted(Path(dump_dir).glob("*.pt"))
    by = {}
    for f in files:
        s = torch.load(f, weights_only=True)
        key = (int(s["layer_idx"]), int(s["actual_start"]))
        if key in by or key[1] % chunk_tokens:
            raise ValueError(f"duplicate or misaligned routing dump {f.name}: {key}")
        ids, w = s["expert_ids"], s["expert_weights"]
        if tuple(ids.shape) != (chunk_tokens, top_k) or not torch.isfinite(w).all():
            raise ValueError(f"{f.name}: expert_ids {tuple(ids.shape)}, expected {(chunk_tokens, top_k)}")
        by[key] = (ids.numpy(), w.float().numpy())
    layers = sorted({k[0] for k in by})
    starts = list(range(0, n_chunks * chunk_tokens, chunk_tokens))
    if set(by) != {(l, s) for l in layers for s in starts}:
        raise ValueError(f"incomplete routing dump: {len(by)} files for {len(layers)} layers x {len(starts)} chunks")
    ids = np.stack([np.concatenate([by[l, s][0] for s in starts]) for l in layers])
    w = np.stack([np.concatenate([by[l, s][1] for s in starts]) for l in layers])
    validate(ids, num_experts, top_k, n_chunks * chunk_tokens)
    digest = save(out, ids, layers, w)
    return dict(layers=layers, raw_files=len(files), routing_sha256=digest)


def import_routing(out: Path, expert_ids, layers, num_experts: int, top_k: int, expert_weights=None) -> dict:
    ids = np.asarray(expert_ids)
    validate(ids, num_experts, top_k)
    return dict(
        layers=list(layers),
        routing_sha256=save(out, ids, layers, None if expert_weights is None else np.asarray(expert_weights)),
    )


def chunk_counts(ids: np.ndarray, rows: int, num_experts: int) -> np.ndarray:
    """ids [layers, chunk_tokens, top_k] -> counts [layers, rows, num_experts]: tokens each source row (its
    contiguous 1/rows of the chunk) routes to each expert."""
    layers, tokens, k = ids.shape
    x = ids.reshape(layers, rows, tokens // rows * k).astype(np.int64)
    off = (np.arange(layers)[:, None, None] * rows + np.arange(rows)[None, :, None]) * num_experts
    return (
        np.bincount((x + off).ravel(), minlength=layers * rows * num_experts)
        .reshape(layers, rows, num_experts)
        .astype(np.int32)
    )
