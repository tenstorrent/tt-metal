# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test-side glue around the sparse-MLA CPU reference package (`reference.cpu_deepseek_v32`).

The package owns the model + weights API (see its `API_SPEC.md`): a single canonical weights dict
(HF naming) constructs both `ttMLA` and `SparseMLAReference`, so device and truth are bit-identical
and PCC is meaningful. This module only adds what's specific to the test harness: deterministic input
generation, the per-variant disk cache for reference outputs, and a small weight-source helper.
"""

import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import (
    CANONICAL_WEIGHT_NAMES,
    SparseMLAReference,
    Weights,
    pretrained_mla_weights,
    random_mla_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import INDEXER_WEIGHT_NAMES

# Boundary check: the package's canonical indexer weight names must match the ttnn contract
# (TtIndexer.WEIGHT_NAMES), so a device-side rename surfaces here instead of drifting silently.
_CONTRACT_INDEXER_KEYS = frozenset(f"{n}.weight" for n in INDEXER_WEIGHT_NAMES)
_EMITTED_INDEXER_KEYS = frozenset(k for k in CANONICAL_WEIGHT_NAMES if k.startswith("indexer."))
assert _EMITTED_INDEXER_KEYS == _CONTRACT_INDEXER_KEYS, (
    f"canonical indexer keys drifted from TtIndexer.WEIGHT_NAMES: "
    f"{sorted(_EMITTED_INDEXER_KEYS)} != {sorted(_CONTRACT_INDEXER_KEYS)}"
)


def cpu_ref_cache_dir(variant) -> Path:
    """Disk cache dir for this variant's CPU reference outputs (env override, else /tmp)."""
    env = variant.mla_ref_cache_env or "DEEPSEEK_V3_MLA_REF_CACHE"
    return Path(os.environ.get(env, f"/tmp/{variant.name}_mla_ref_cache"))


def build_weights(variant, config, *, seed=42, layer=None, checkpoint_path=None, repo=None) -> tuple[Weights, str]:
    """Canonical weights + a source tag (so random/pretrained ref caches never collide).

    Random by default; pretrained layer `layer` from a local `checkpoint_path` or HF `repo` (defaulting
    to the variant's own `hf_repo_id` — the test owns the repo, per the package API_SPEC). The same dict
    feeds both the device (`ttMLA`) and the CPU truth (`SparseMLAReference`).
    """
    if checkpoint_path is not None:
        return pretrained_mla_weights(config, layer=layer or 0, checkpoint_path=checkpoint_path), f"ckptL{layer or 0}"
    if layer is not None:
        return pretrained_mla_weights(config, layer=layer, repo=repo or variant.hf_repo_id), f"layer{layer}"
    return random_mla_weights(config, seed=seed), f"random_seed{seed}"


def make_hidden(seq_len, hidden_size, seed=42, input_path=None):
    """MLA/indexer input [1, seq, hidden] bf16: from `input_path` (.pt, sliced/checked) or randn(seed)."""
    if input_path:
        t = torch.load(input_path, weights_only=True)
        t = t["hidden_states"] if isinstance(t, dict) else t
        t = t.reshape(-1, t.shape[-1])  # [.., hidden] -> [tokens, hidden]
        assert (
            t.shape[0] >= seq_len and t.shape[-1] == hidden_size
        ), f"input {tuple(t.shape)} can't supply [{seq_len}, {hidden_size}]"
        return t[:seq_len].reshape(1, seq_len, hidden_size).to(torch.bfloat16)
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)


def run_cpu_reference(config, weights, hidden_states, seq_len, cache_dir, cache_tag):
    """Single-shot sparse-MLA CPU truth (indexer active). Disk-cached (output + KVPE + index) under `cache_dir`.

    Returns (ref_output [1, seq, dim], ref_kvpe [1, 1, seq, kv_lora_rank + rope],
    ref_index [1, seq, index_head_dim]).
    """
    cache_path = Path(cache_dir) / f"{cache_tag}_seq{seq_len}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached CPU reference from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        if "ref_index" in cached:  # pre-index caches lack it — fall through and recompute
            return cached["ref_output"], cached["ref_kvpe"], cached["ref_index"]
        logger.info(f"Cached reference at {cache_path} predates the index cache; recomputing")

    ref = SparseMLAReference(config, weights, seq_len=seq_len)
    ref_output = ref.forward(hidden_states)
    ref_kvpe = ref.kvpe_cache  # device layout [1, 1, seq, kv_lora_rank + rope]
    ref_index = ref.index_cache  # [1, seq, index_head_dim]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe, "ref_index": ref_index}, cache_path)
    logger.info(f"Saved CPU reference to {cache_path}")
    return ref_output, ref_kvpe, ref_index


def run_cpu_reference_chunked(config, weights, hidden_states, seq_len, chunk, cache_dir, cache_tag):
    """Chunk-loop sparse-MLA CPU truth (decode branch via actual_start). Disk-cached.

    Returns (ref_output [1, seq, dim], ref_kvpe [1, seq, kv_lora_rank + rope],
    ref_index [1, seq, index_head_dim]).
    """
    cache_path = Path(cache_dir) / f"chunked_{cache_tag}_seq{seq_len}_c{chunk}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached chunked CPU reference from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        if "ref_index" in cached:  # pre-index caches lack it — fall through and recompute
            return cached["ref_output"], cached["ref_kvpe"], cached["ref_index"]
        logger.info(f"Cached chunked reference at {cache_path} predates the index cache; recomputing")

    ref = SparseMLAReference(config, weights, seq_len=seq_len)
    outs = [ref.forward(hidden_states[:, s : s + chunk], actual_start=s) for s in range(0, seq_len, chunk)]
    ref_output = torch.cat(outs, dim=1)
    ref_kvpe = ref.kvpe_cache.squeeze(1)  # [1, seq, kv_lora_rank + rope] (chunked KVPE comparison layout)
    ref_index = ref.index_cache  # [1, seq, index_head_dim]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe, "ref_index": ref_index}, cache_path)
    logger.info(f"Saved chunked CPU reference to {cache_path}")
    return ref_output, ref_kvpe, ref_index
