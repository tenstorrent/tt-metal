# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden runner + on-disk reference cache for Llama 3.1 8B.

Runs the torch reference (:mod:`..reference.model`) and dumps the per-layer inputs/outputs and the
per-layer K/V the device is PCC'd against. A full-model CPU forward at real dims is expensive, so
the whole-model trace is cached; per-module goldens are cheap and deliberately are not.

Two rules the cache embodies (see the bring-up recipe §3):

* the cache is keyed on **every** field that changes the output — :class:`ReferenceCacheKey` is
  frozen, so a changed field yields a different filename and a stale result is never silently
  reused;
* an expensive CPU run **asserts** rather than recomputes when ``require_hit`` is set, so CI fails
  loudly on a miss instead of burning an hour.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger

from .config import LlamaConfig
from .model import LlamaModel

CACHE_ENV = "TT_LLAMA31_8B_PREFILL_REF_CACHE"
DEFAULT_CACHE_DIR = "/tmp/llama3_1_8b_d_p_ref_cache"


@dataclass(frozen=True)
class ReferenceCacheKey:
    """Every parameter that affects the reference output.

    Frozen and complete on purpose: a field that changes the result but is missing here makes the
    cache return a stale trace that looks valid. Adding a field is always safe (it only forces a
    miss); omitting one is the bug this class exists to prevent.
    """

    weight_type: str  # "random" or "pretrained"
    seed: int  # RNG seed for random weights (ignored, but recorded, for pretrained)
    input_source: str  # "random_ids", "abc", or a prompt-file name
    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_chunks: int  # 1 = one-shot; >1 = chunked reference run
    dtype: str

    def __str__(self) -> str:
        return (
            f"{self.weight_type}_seed{self.seed}_{self.input_source}"
            f"_isl{self.seq_len}_L{self.num_layers}_h{self.hidden_size}"
            f"_i{self.intermediate_size}_v{self.vocab_size}"
            f"_chunks{self.num_chunks}_{self.dtype}"
        )

    @property
    def filename(self) -> str:
        return f"{self}.pt"


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE_DIR))


def cache_path(key: ReferenceCacheKey) -> Path:
    return cache_dir() / key.filename


def save_reference_cache(key: ReferenceCacheKey, payload: dict) -> Path:
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store the key alongside the payload so a file found by name can still be self-verified.
    torch.save({"key": asdict(key), "payload": payload}, path)
    logger.info(f"Saved reference cache: {path}")
    return path


def load_reference_cache(key: ReferenceCacheKey) -> dict | None:
    path = cache_path(key)
    if not path.exists():
        return None
    blob = torch.load(path, weights_only=False)
    stored = blob.get("key")
    if stored != asdict(key):
        # Name collision across key schema versions: treat as a miss rather than trust the file.
        logger.warning(f"Reference cache key mismatch at {path}; treating as a miss")
        return None
    logger.info(f"Loaded reference cache: {path}")
    return blob["payload"]


def build_reference_model(config: LlamaConfig, *, seed: int = 0, dtype=torch.float32) -> LlamaModel:
    """Reference model with deterministic random weights. Identical seed => identical weights."""
    torch.manual_seed(seed)
    model = LlamaModel(config)
    return model.to(dtype).eval()


def make_input_ids(source: str, seq_len: int, vocab_size: int, *, seed: int = 0) -> torch.Tensor:
    if source == "random_ids":
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab_size, (1, seq_len), generator=g)
    if source == "abc":
        # Deterministic, structured, and independent of vocab content: token i = i % vocab.
        return (torch.arange(seq_len) % vocab_size)[None]
    raise ValueError(f"unknown input_source {source!r}")


def run_golden(
    config: LlamaConfig,
    key: ReferenceCacheKey,
    *,
    require_hit: bool = False,
    use_cache: bool = True,
) -> dict:
    """Run (or load) the whole-model reference trace for ``key``.

    Returns a dict with:
      ``input_ids``   [1, S]
      ``logits``      [1, S, vocab]
      ``k`` / ``v``   lists of [1, n_kv, S, head_dim], one per layer — post-RoPE K, raw V. This is
                      the per-layer KV golden the device cache is PCC'd against.
      ``hidden``      list of [1, S, hidden], the residual stream after each layer.

    ``require_hit=True`` turns a cache miss into a failure — for CI, where recomputing a full-dims
    forward is not acceptable.
    """
    if use_cache:
        cached = load_reference_cache(key)
        if cached is not None:
            return cached
        if require_hit:
            raise RuntimeError(
                f"reference cache miss for {key} at {cache_path(key)}; generate it first "
                f"(scripts/generate_golden_trace.py) instead of recomputing a full CPU forward here"
            )

    assert key.num_layers == config.num_hidden_layers, "key/config disagree on num_layers"
    assert key.vocab_size == config.vocab_size, "key/config disagree on vocab_size"

    model = build_reference_model(config, seed=key.seed)
    input_ids = make_input_ids(key.input_source, key.seq_len, config.vocab_size, seed=key.seed)

    with torch.no_grad():
        if key.num_chunks == 1:
            logits, kvs, hidden = model(input_ids, return_hidden_states=True)
        else:
            assert key.seq_len % key.num_chunks == 0, "seq_len must divide evenly into num_chunks"
            chunk = key.seq_len // key.num_chunks
            past = None
            k_acc = [[] for _ in range(config.num_hidden_layers)]
            v_acc = [[] for _ in range(config.num_hidden_layers)]
            for c in range(key.num_chunks):
                ids_c = input_ids[:, c * chunk : (c + 1) * chunk]
                logits, kvs, hidden = model(ids_c, past_kvs=past, kv_offset=c * chunk, return_hidden_states=True)
                for i, (k, v) in enumerate(kvs):
                    k_acc[i].append(k)
                    v_acc[i].append(v)
                past = [(torch.cat(k_acc[i], dim=2), torch.cat(v_acc[i], dim=2)) for i in range(len(k_acc))]
            kvs = past

    payload = {
        "input_ids": input_ids,
        "logits": logits,
        "k": [k for k, _ in kvs],
        "v": [v for _, v in kvs],
        "hidden": hidden,
    }
    if use_cache:
        save_reference_cache(key, payload)
    return payload
