# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The only writer of the package-local golden cache.

Recomputation is a deliberate operator action, never a side effect of a test run
(see :func:`~models.demos.mistral_medium_3_5_128b.reference.golden.load_golden`):

    python -m models.demos.mistral_medium_3_5_128b.reference.golden --regenerate

Each entry pairs a :class:`GoldenCacheKey` with the function that produces it. The key must name
every input the function reads — that invariant is what makes a stale cache impossible rather than
merely unlikely, and ``tests/unit/test_golden_cache.py`` checks it by perturbing config fields and
asserting the digest moves.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.demos.mistral_medium_3_5_128b.reference.golden import (
    GoldenCacheKey,
    run_reference_layer,
    run_reference_model,
    save_golden,
)
from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig, host_reduced_config
from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, LayerWeights, ModelWeights

HOST_TEST_TOKENS = 256  # a multiple of TILE_SIZE*sp, so the same length is usable on device
WEIGHT_SEED = 0
INPUT_SEED = 1


def random_hidden(cfg: MistralMediumConfig, n_tokens: int, seed: int = INPUT_SEED) -> torch.Tensor:
    """``[1, n_tokens, hidden]`` bf16 activations — the standard host-test input."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_tokens, cfg.hidden_size, generator=g, dtype=torch.float32).to(REF_DTYPE)


def decoder_layer_key(cfg: MistralMediumConfig, n_tokens: int = HOST_TEST_TOKENS) -> GoldenCacheKey:
    return GoldenCacheKey.for_config(
        cfg,
        kind="decoder_layer",
        weight_source=f"random:{WEIGHT_SEED}",
        input_source=f"randn:{INPUT_SEED}",
        n_tokens=n_tokens,
        num_layers=1,
    )


def compute_decoder_layer(cfg: MistralMediumConfig, n_tokens: int = HOST_TEST_TOKENS) -> dict[str, torch.Tensor]:
    """One reduced-config decoder layer over ``n_tokens``, with its input kept alongside the outputs.

    Storing the input too means a consumer never has to reproduce the RNG to use the entry; if the
    generator ever changed, the stored input and outputs would still be consistent with each other.
    """
    weights = LayerWeights.random(cfg, seed=WEIGHT_SEED)
    hidden = random_hidden(cfg, n_tokens)
    out, k, v = run_reference_layer(cfg, weights, hidden)
    return {"input": hidden, "output": out, "k": k, "v": v}


def model_key(cfg: MistralMediumConfig, n_tokens: int = HOST_TEST_TOKENS) -> GoldenCacheKey:
    return GoldenCacheKey.for_config(
        cfg,
        kind="model",
        weight_source=f"random:{WEIGHT_SEED}",
        input_source=f"randn:{INPUT_SEED}",
        n_tokens=n_tokens,
    )


def random_token_ids(cfg: MistralMediumConfig, n_tokens: int, seed: int = INPUT_SEED) -> torch.Tensor:
    """``[1, n_tokens]`` int64 token ids in range — the standard whole-model host-test input."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (1, n_tokens), generator=g, dtype=torch.int64)


def compute_model(cfg: MistralMediumConfig, n_tokens: int = HOST_TEST_TOKENS) -> dict[str, torch.Tensor]:
    """The whole reduced-config model over ``n_tokens`` random ids, with the ids stored alongside.

    Reduced config only: the real model is 128 B parameters and a host forward of it is not a thing
    that happens. Full-depth ground truth is the prepared golden trace.
    """
    weights = ModelWeights.random(cfg, seed=WEIGHT_SEED)
    ids = random_token_ids(cfg, n_tokens)
    out = run_reference_model(cfg, weights, ids)
    out["input_ids"] = ids
    return out


def regenerate_all() -> list[Path]:
    """Recompute and write every cache entry. Returns the paths written."""
    cfg = host_reduced_config()
    return [
        save_golden(decoder_layer_key(cfg), compute_decoder_layer(cfg)),
        save_golden(model_key(cfg), compute_model(cfg)),
    ]
