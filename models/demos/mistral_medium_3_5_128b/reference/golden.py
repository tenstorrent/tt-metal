# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden runner: run the torch reference once and cache per-layer outputs + K/V on disk.

Reuses the shared golden-cache helpers (``deepseek_v3_d_p/utils/transformer_helpers.py``):
``ReferenceCacheKey`` is frozen, so a changed field yields a different filename and a stale result is
never reused silently. ``MistralReferenceCacheKey`` adds the two fields that change this model's output
and the base key does not carry: a digest of the (possibly reduced) model config and the weight seed.

Cache dir: ``$MISTRAL_REF_CACHE`` (default ``/tmp/mistral_medium_3_5_128b_transformer_ref_cache``).
``MISTRAL_REF_CACHE_REQUIRED=1`` turns a cache miss into an error instead of a CPU recompute.
"""

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    _ref_cache_dir,
    load_reference_cache,
    save_reference_cache,
)

from .model import build_reference_model, random_state_dict

GOLDEN_VARIANT = SimpleNamespace(name="mistral_medium_3_5_128b", ref_cache_env="MISTRAL_REF_CACHE")


@dataclass(frozen=True)
class MistralReferenceCacheKey(ReferenceCacheKey):
    config_digest: str = ""
    seed: int = 0

    def __str__(self) -> str:
        return f"{super().__str__()}_cfg{self.config_digest}_seed{self.seed}"


def config_digest(cfg) -> str:
    blob = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def random_input_ids(cfg, seq_len, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, cfg.vocab_size, (1, seq_len), generator=g)


def make_key(cfg, seq_len, seed=0, weight_type="random", input_source="random"):
    return MistralReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=seq_len,
        num_layers=cfg.num_hidden_layers,
        n_routed_experts=0,
        padding_side="right",
        config_digest=config_digest(cfg),
        seed=seed,
    )


def cache_path(key):
    return _ref_cache_dir(GOLDEN_VARIANT) / f"{key}.pt"


LOGITS_TAIL = 32  # logits are kept for the last tile of positions only


@torch.no_grad()
def run_reference(cfg, state_dict, input_ids):
    """Whole-model reference. Returns (snapshots, kv): snapshots = [embedding, layer_0 .. layer_{N-1},
    final_norm, logits of the last LOGITS_TAIL positions]; kv = one ``stack([k_rot, v])``
    [2, 1, Hkv, S, D] tensor per layer."""
    model = build_reference_model(cfg, state_dict)
    snapshots = []
    logits, final, kvs = model(input_ids, hiddens=snapshots, logits_last=LOGITS_TAIL)
    snapshots += [final, logits]
    return snapshots, [torch.stack([k, v]) for k, v in kvs]


def golden_forward(cfg, seq_len, seed=0):
    """Random-weight whole-model golden for ``cfg`` at ``seq_len`` tokens: load from the cache, else
    compute once and save. Returns (key, input_ids, snapshots, kv)."""
    key = make_key(cfg, seq_len, seed)
    input_ids = random_input_ids(cfg, seq_len, seed)
    try:
        snapshots, kv = load_reference_cache(GOLDEN_VARIANT, key)
        return key, input_ids, snapshots, kv
    except FileNotFoundError:
        if os.environ.get("MISTRAL_REF_CACHE_REQUIRED") == "1":
            raise
    logger.info(f"golden cache miss for {key}; computing the CPU reference once")
    snapshots, kv = run_reference(cfg, random_state_dict(cfg, seed), input_ids)
    save_reference_cache(GOLDEN_VARIANT, key, snapshots, kv)
    return key, input_ids, snapshots, kv
