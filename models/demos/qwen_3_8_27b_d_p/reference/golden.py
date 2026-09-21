# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-once golden runner for the torch reference.

Two granularities, and the recipe treats them differently:

* **Per-module goldens** (D1/D2/M2) are cheap — a norm or an MLP at one chunk of tokens — so they
  are recomputed in-test and deliberately not cached.
* **The whole-model forward** is not. A full-depth CPU forward of 27B parameters is an hours-long
  run, so it goes through the shared reference cache
  (``deepseek_v3_d_p/utils/transformer_helpers``), keyed on :class:`GoldenCacheKey`.

Two rules the shared cache embodies and this module keeps:
  1. the key carries **every** field that changes the output, and is frozen, so a changed field
     yields a different filename rather than a silently stale hit;
  2. where recomputation is expensive, **assert** on a miss instead of recomputing — tests pass
     ``require_cached=True`` so CI fails loudly in seconds instead of burning an hour.

Imports torch plus the shared cache helpers only; no ttnn (see the reference purity contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    check_reference_cache_exists,
    load_reference_cache,
    save_reference_cache,
)

from .config import Qwen35TextConfig
from .modeling import AttentionCapture, GdnCapture, Qwen35TextModel, init_random_weights

# The shared helpers take a `variant` for two things only: the cache-dir env var and a name.
# Duck-typed here rather than registered in the DeepSeek variant table, which this model is not in.
CACHE_VARIANT = SimpleNamespace(name="qwen_3_8_27b", ref_cache_env="QWEN35_REF_CACHE")


@dataclass(frozen=True)
class GoldenCacheKey:
    """Everything that changes the reference's output. Frozen: a changed field is a new filename.

    ``weight_type`` distinguishes a real checkpoint from the seeded random init, ``weight_seed``
    separates two random draws, and ``chunking`` separates a one-shot forward from a chunked one
    (they should agree, which is precisely why their goldens must not share a file).
    """

    weight_type: str  # "pretrained" | "random"
    weight_seed: int
    input_source: str  # "random_ids" | a prompt name
    input_seed: int
    isl_total: int
    num_layers: int
    hidden_size: int
    vocab_size: int
    chunking: str  # "one_shot" | f"chunked{chunk_size}"
    dtype: str

    def __str__(self) -> str:
        return (
            f"{self.weight_type}{self.weight_seed}_{self.input_source}{self.input_seed}"
            f"_isl{self.isl_total}_layers{self.num_layers}_h{self.hidden_size}"
            f"_v{self.vocab_size}_{self.chunking}_{self.dtype}"
        )


@dataclass
class GoldenForward:
    """One reference forward: the per-layer output hidden states and each layer's carried state."""

    hidden_per_layer: list[torch.Tensor]
    states: list[AttentionCapture | GdnCapture]
    final_hidden: torch.Tensor

    def layer_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(K, V) for a full-attention layer — the golden-trace artifact P1/P2 PCC against."""
        state = self.states[layer_idx]
        assert isinstance(state, AttentionCapture), f"layer {layer_idx} is a Gated DeltaNet layer"
        return state.key, state.value

    def layer_gdn_state(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(conv_state, recurrent_state) for a linear-attention layer — the GDN analogue of K/V."""
        state = self.states[layer_idx]
        assert isinstance(state, GdnCapture), f"layer {layer_idx} is a full-attention layer"
        return state.conv_state, state.recurrent_state


def build_random_reference(cfg: Qwen35TextConfig, seed: int = 0) -> Qwen35TextModel:
    """A reference on seeded random weights — the D2/D3/M2/M3 oracle (recipe: all PCC tests up to
    P1 run on random weights, identical on both sides)."""
    model = Qwen35TextModel(cfg).eval()
    init_random_weights(model, seed=seed)
    return model


def run_reference(
    model: Qwen35TextModel,
    input_ids: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> GoldenForward:
    """One-shot (``chunk_size=None``) or chunked reference forward, capturing every layer."""
    hidden_per_layer: list[torch.Tensor] = []
    states: Optional[list] = None
    outputs: list[torch.Tensor] = []

    seq = input_ids.shape[1]
    step = chunk_size or seq
    assert seq % step == 0, f"isl {seq} is not a whole number of {step}-token chunks"

    for start in range(0, seq, step):
        captured: list[torch.Tensor] = []

        def on_layer(_idx: int, _state, hidden: torch.Tensor) -> None:
            captured.append(hidden.clone())

        with torch.no_grad():
            out, states = model(
                input_ids=input_ids[:, start : start + step],
                start_pos=start,
                states=states,
                skip_lm_head=True,
                on_layer=on_layer,
            )
        outputs.append(out)
        if not hidden_per_layer:
            hidden_per_layer = captured
        else:
            hidden_per_layer = [torch.cat([a, b], dim=1) for a, b in zip(hidden_per_layer, captured)]

    return GoldenForward(
        hidden_per_layer=hidden_per_layer,
        states=list(states),
        final_hidden=torch.cat(outputs, dim=1),
    )


def cached_reference(
    cfg: Qwen35TextConfig,
    input_ids: torch.Tensor,
    key: GoldenCacheKey,
    *,
    build: Callable[[], Qwen35TextModel],
    chunk_size: Optional[int] = None,
    require_cached: bool = False,
) -> GoldenForward:
    """Load the whole-model reference from the shared cache, or compute and store it.

    ``require_cached=True`` turns a miss into an assertion instead of an hour of CPU — the pattern
    ``deepseek_v3_d_p/tests/test_mla.py`` uses so CI never silently recomputes a reference.
    """
    if check_reference_cache_exists(CACHE_VARIANT, key):
        snapshots, extras = load_reference_cache(CACHE_VARIANT, key)
        return GoldenForward(
            hidden_per_layer=snapshots,
            states=[_state_from_dict(d) for d in extras[:-1]],
            final_hidden=extras[-1],
        )

    assert not require_cached, (
        f"no cached reference for {key} under ${CACHE_VARIANT.ref_cache_env}; generate it with "
        f"scripts/generate_golden_trace.py rather than recomputing a full-depth CPU forward here"
    )
    logger.warning(f"reference cache MISS for {key} — computing on CPU")
    golden = run_reference(build(), input_ids, chunk_size=chunk_size)
    # The shared loader reads with weights_only=True, so only tensors and plain containers survive
    # a round trip — the captures go to disk as str->tensor dicts and are rebuilt on load.
    save_reference_cache(
        CACHE_VARIANT,
        key,
        golden.hidden_per_layer,
        [_state_to_dict(s) for s in golden.states] + [golden.final_hidden],
    )
    return golden


def _state_to_dict(state: AttentionCapture | GdnCapture) -> dict[str, torch.Tensor]:
    if isinstance(state, AttentionCapture):
        return {"key": state.key, "value": state.value}
    return {"conv_state": state.conv_state, "recurrent_state": state.recurrent_state}


def _state_from_dict(d: dict[str, torch.Tensor]) -> AttentionCapture | GdnCapture:
    if "key" in d:
        return AttentionCapture(key=d["key"], value=d["value"])
    return GdnCapture(conv_state=d["conv_state"], recurrent_state=d["recurrent_state"])
