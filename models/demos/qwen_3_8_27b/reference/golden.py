# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden runners (torch only).

* ``stream_forward`` — the reference forward, one layer at a time: build a layer, load its weights
  (real checkpoint or a seeded random init), run it, free it. Peak memory is one layer, so the full
  64-layer model runs on the host without materialising 27B parameters.
* ``cached_forward`` — the compute-once cache for expensive end-to-end goldens, keyed on every field
  that changes the output via the shared frozen ``ReferenceCacheKey`` (a changed field is a different
  file, never a silently stale hit). Per-module goldens are cheap and deliberately not cached.

The graded artifact (P1/P2) is the separate full-depth golden trace named by ``PREFILL_TRACE_DIR``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    check_reference_cache_exists,
    load_reference_cache,
    save_reference_cache,
)
from models.demos.qwen_3_8_27b.config import Qwen38Config
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref

VARIANT = SimpleNamespace(name="qwen_3_8_27b", ref_cache_env="QWEN38_REF_CACHE")


def trace_token_ids(trace_dir) -> torch.Tensor:
    """The golden trace's input token ids ``[T]`` (int64), from either trace layout."""
    trace_dir = Path(trace_dir)
    meta = json.loads((trace_dir / "metadata.json").read_text())
    if "token_ids" in meta:
        return torch.tensor(meta["token_ids"], dtype=torch.int64)
    # agentic-prefill-goldens: one shared token cache per model; a trace is its first n_tokens
    with safe_open(str(trace_dir / meta["token_cache"]), "pt") as f:
        ids = f.get_tensor("token_ids")
    n = meta["n_tokens"]
    assert ids.numel() >= n, f"token cache holds {ids.numel()} ids; the trace needs {n}"
    return ids[:n].to(torch.int64)


def _load_layer(layer: torch.nn.Module, sd: dict):
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected}"
    assert not missing, f"missing keys {missing}"


def stream_forward(
    cfg: Qwen38Config,
    token_ids: torch.Tensor,
    *,
    reader=None,
    seed: int | None = None,
    num_layers: int | None = None,
    dtype=torch.float32,
    states=None,
    start_pos: int = 0,
    on_layer=None,
):
    """Reference forward over ``token_ids [1, T]``, layer by layer.

    Weights: ``reader`` (a CheckpointReader, real weights) or ``seed`` (per-layer seeded random init).
    Returns ``(final_hidden [1,T,H], states)``; attention states hold the accumulated K/V.
    """
    assert (reader is None) != (seed is None), "pass exactly one of reader / seed"
    n = num_layers or cfg.num_hidden_layers
    T = token_ids.shape[1]
    states = states if states is not None else [None] * n
    with torch.no_grad():
        if reader is not None:
            emb = reader.text("embed_tokens.weight")
            x = emb[token_ids[0]].to(dtype)[None]
            del emb
        else:
            g = torch.Generator().manual_seed(seed)
            emb = torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g).to(torch.bfloat16)
            x = emb[token_ids[0]].to(dtype)[None]
        cos, sin = ref.rope_cos_sin(cfg, torch.arange(start_pos, start_pos + T), dtype=dtype)
        new_states = []
        for i in range(n):
            layer = ref.DecoderLayer(cfg, i)
            if reader is not None:
                _load_layer(layer, reader.layer(i))
            else:
                ref.init_random_(layer, seed=seed * 1000 + i + 1)
                for p in layer.parameters():  # both sides see bf16-representable weights
                    p.data = p.data.to(torch.bfloat16)
            layer = layer.to(dtype)
            st = states[i]
            x, ns = layer(x, cos, sin, st)
            if layer.is_full and st is not None:
                ns = {"k": torch.cat([st["k"], ns["k"]], 2), "v": torch.cat([st["v"], ns["v"]], 2)}
            new_states.append(ns)
            if on_layer is not None:
                on_layer(i, x)
            del layer
        norm = ref.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if reader is not None:
            norm.weight.data = reader.text("norm.weight").float()
        else:
            ref.init_random_(norm, seed=seed * 1000)
            norm.weight.data = norm.weight.data.to(torch.bfloat16).float()
        x = norm.to(dtype)(x)
    return x, new_states


def random_weight_state_dicts(cfg: Qwen38Config, seed: int, num_layers: int):
    """The exact random weights ``stream_forward(seed=...)`` uses, as HF-named state dicts (for the device)."""
    g = torch.Generator().manual_seed(seed)
    out = {"embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g).to(torch.bfloat16)}
    for i in range(num_layers):
        layer = ref.init_random_(ref.DecoderLayer(cfg, i), seed=seed * 1000 + i + 1)
        out.update({f"layers.{i}.{k}": v.to(torch.bfloat16) for k, v in layer.state_dict().items()})
    norm = ref.init_random_(ref.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), seed=seed * 1000)
    out["norm.weight"] = norm.weight.data.to(torch.bfloat16)
    return out


def cache_key(weight_type: str, input_source: str, isl: int, num_layers: int) -> ReferenceCacheKey:
    # n_routed_experts is 0 (dense model); padding_side fixed (no padding in prefill goldens)
    return ReferenceCacheKey(weight_type, input_source, isl, num_layers, 0, "right")


def cached_forward(key: ReferenceCacheKey, compute, *, require_cached: bool = False):
    """Return ``(snapshots, states_flat)`` from the cache, or compute (``compute() -> (list, list)``) and save.

    ``require_cached`` asserts instead of recomputing (for CI, where a miss would burn an hour).
    """
    if check_reference_cache_exists(VARIANT, key):
        return load_reference_cache(VARIANT, key)
    if require_cached or os.environ.get("QWEN38_REQUIRE_REF_CACHE") == "1":
        raise FileNotFoundError(f"reference cache miss for {key} (QWEN38_REQUIRE_REF_CACHE=1)")
    snaps, flat = compute()
    save_reference_cache(VARIANT, key, snaps, flat)
    return snaps, flat


def flatten_states(states) -> list[torch.Tensor]:
    out = []
    for st in states:
        out.extend([st["k"], st["v"]] if "k" in st else [st["recurrent_state"], st["conv_state"]])
    return out
