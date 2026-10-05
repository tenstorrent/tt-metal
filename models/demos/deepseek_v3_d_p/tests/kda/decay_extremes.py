# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA layer cases that drive the decay gate into its extremes, with cached FP64 CPU references.

Each case runs a few KDA heads (K = V = 128) on one device with the production Kimi-K3 recurrence numerics and
chains ``num_calls`` layer calls of ``chunk_tokens`` tokens. The hidden states are crafted so that every selected
head sees a prescribed decay gate and beta at every token:

    x_t = c + n_t,   [f_b_h f_a; b_proj_h] c = targets,   n_t ⟂ rowspace([f_b_h f_a; b_proj_h])

``c`` is the minimum-norm solution of the gate and beta targets, and ``n_t`` is unit-RMS Gaussian noise projected
out of that row space, so q, k, v and the output gate vary per token while the gate logits stay on target. Bands:

Only one head is crafted per case (the gate is rank head_k_dim across all heads).

* ``control``: per-token g ~ -1.3 (|G_last| ~ 43) and beta ~ 0.5; calibrates the device error floor.
* ``strong``: per-token g ~ -4.7 with fractional BF16 values (|G_last| ~ 150).
* ``strong-saturated``: g = -5 exactly (|G_last| = 160, the K3/GLM bound; measured on real K3 text, g1b.4.10).
* ``weak``: per-token g ~ -4.7e-5 (|G_last| ~ 1.5e-3 < 2^-9) and beta ~ 3.4e-4, so decay, not the delta-rule
  erase, sets the memory length (~670 chunks).
* ``weak-beta``: the weak gate with beta ~ 0.5 (the delta-rule erase dominates).
* ``text``: not crafted; the real-text layer input of tests/kda/text_input.py (Pride and Prejudice), so the heads
  see the decay and beta that the trained gate produces on prose. Exact layer input for GLM layer 0; for K3
  layers.1 it is input_layernorm(embedding), a proxy without layer 0's residual contribution. Cases select the heads
  whose measured per-chunk |G_last| on that text is most extreme.

The CPU reference is ``kda_forward_reference`` in FP64. Prepare references without a device:

    python -m models.demos.deepseek_v3_d_p.tests.kda.decay_extremes [--case NAME ...]

Real weights need ``KIMI_K3_CKPT`` / ``GLM_5_3_FLASH_CKPT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.glm_5_3_flash_config import glm_5_3_flash_kda_config
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    GLM_5_3_FLASH_FIRST_KDA_LAYER,
    GLM_5_3_FLASH_LAYER_0_SHA256,
    KIMI_K3_FIRST_KDA_LAYER,
    KIMI_K3_LAYER_1_SHA256,
    load_kda_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.kda.head_slice import kda_head_slice_config, slice_kda_heads
from models.demos.deepseek_v3_d_p.tests.kda.text_input import build_text_input, load_text_input, text_input_cache_path
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights

# Covers the stored payload: case construction (weights, crafted hidden) and the FP64 reference. Bump when either
# changes the stored tensors.
_CACHE_VERSION = 2
_GATE_LOWER_BOUND = -5.0
# Largest RMS of the crafted constant hidden component (unit-RMS noise is added on top).
_MAX_CENTER_RMS = 4.0
_SYNTHETIC_CONFIG = KDAConfig(
    hidden_size=1024,
    num_heads=1,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
    norm_eps=1e-5,
    gate_lower_bound=_GATE_LOWER_BOUND,
)
# Model name of each real weight source in tests/kda/text_input.py.
_TEXT_INPUT_MODELS = {"k3": "kimi_k3", "glm": "glm_5_3_flash"}
_TEXT_BAND = "text"
_WEIGHT_SOURCES = {
    # name: (checkpoint env var, layer index, pinned layer digest, config factory)
    "k3": ("KIMI_K3_CKPT", KIMI_K3_FIRST_KDA_LAYER, KIMI_K3_LAYER_1_SHA256, kimi_k3_kda_config),
    "glm": (
        "GLM_5_3_FLASH_CKPT",
        GLM_5_3_FLASH_FIRST_KDA_LAYER,
        GLM_5_3_FLASH_LAYER_0_SHA256,
        glm_5_3_flash_kda_config,
    ),
}
# Gate logit a = A (raw + dt_bias) per band; g = lower_bound * sigmoid(a). Beta logit per band.
_BANDS = {
    "control": (-1.0, 0.0),
    "strong": (2.75, 0.0),
    "strong-saturated": (30.0, 0.0),
    "weak": (math.log(4.7e-5 / 5.0), -8.0),
    "weak-beta": (math.log(4.7e-5 / 5.0), 0.0),
}


@dataclass(frozen=True)
class DecayExtremeCase:
    weights: str  # "synthetic", "k3" or "glm"
    band: str
    head_start: int = 0
    # The decay gate is low rank (f_b f_a through a head_k_dim bottleneck), so only one head's gate can be set
    # independently for every channel.
    num_heads: int = 1
    chunk_tokens: int = 1280
    num_calls: int = 1
    seed: int = 7

    def __post_init__(self) -> None:
        if self.weights not in ("synthetic", *_WEIGHT_SOURCES):
            raise ValueError(f"unknown weight source {self.weights!r}")
        if self.band not in _BANDS and self.band != _TEXT_BAND:
            raise ValueError(f"unknown band {self.band!r}")
        if self.band == _TEXT_BAND and self.weights not in _TEXT_INPUT_MODELS:
            raise ValueError(f"a real-text input needs real weights, got {self.weights!r}")

    @property
    def name(self) -> str:
        heads = f"h{self.head_start}-{self.head_start + self.num_heads}"
        return f"{self.weights}-{self.band}-{heads}-T{self.chunk_tokens}x{self.num_calls}"

    @property
    def weight_identity(self) -> str:
        if self.weights == "synthetic":
            return f"synthetic-{json.dumps(asdict(_SYNTHETIC_CONFIG), sort_keys=True)}"
        return _WEIGHT_SOURCES[self.weights][2]

    @property
    def tokens(self) -> int:
        return self.chunk_tokens * self.num_calls


def case_config(case: DecayExtremeCase) -> KDAConfig:
    if case.weights == "synthetic":
        return _SYNTHETIC_CONFIG
    return kda_head_slice_config(_WEIGHT_SOURCES[case.weights][3](), case.num_heads)


def case_weights(case: DecayExtremeCase) -> dict[str, torch.Tensor]:
    """Host weights of the case's heads (synthetic or a pinned real-checkpoint head slice)."""
    if case.weights == "synthetic":
        # Synthetic weights carry the band in dt_bias, so the crafted hidden only has to set beta.
        weights = random_weights(_SYNTHETIC_CONFIG)
        weights["dt_bias"] = _band_logits(case, _SYNTHETIC_CONFIG).reshape(-1).float() / weights["A_log"].reshape(
            -1, 1
        ).exp().repeat_interleave(_SYNTHETIC_CONFIG.head_k_dim, dim=0).reshape(-1)
        return weights
    env, layer, _, config_factory = _WEIGHT_SOURCES[case.weights]
    checkpoint = os.environ.get(env)
    if not checkpoint:
        raise RuntimeError(f"{case.name} needs {env}")
    config = config_factory()
    state_dict = load_kda_layer_state_dict(Path(checkpoint), layer, config)
    return slice_kda_heads(state_dict, config, num_heads=case.num_heads, head_start=case.head_start)


def _band_logits(case: DecayExtremeCase, config: KDAConfig) -> torch.Tensor:
    """Per-channel gate logits [H, K]: the band's logit with a small deterministic per-channel spread."""
    generator = torch.Generator().manual_seed(case.seed)
    spread = 0.1 * torch.randn(config.num_heads, config.head_k_dim, generator=generator, dtype=torch.float64)
    return _BANDS[case.band][0] + spread


def _text_input_path(case: DecayExtremeCase) -> Path:
    return text_input_cache_path(_TEXT_INPUT_MODELS[case.weights], _WEIGHT_SOURCES[case.weights][1], case.tokens)


def _text_hidden(case: DecayExtremeCase) -> torch.Tensor:
    """The real-text layer input [1, tokens, hidden] (built from the checkpoint when not cached)."""
    model, layer = _TEXT_INPUT_MODELS[case.weights], _WEIGHT_SOURCES[case.weights][1]
    hidden = load_text_input(model, layer, case.tokens)
    if hidden is None:
        hidden = build_text_input(model, layer, case.tokens, Path(os.environ[_WEIGHT_SOURCES[case.weights][0]]))
    return hidden


def crafted_hidden(case: DecayExtremeCase, weights: dict[str, torch.Tensor], config: KDAConfig) -> torch.Tensor:
    """BF16 hidden states [1, num_calls * chunk_tokens, hidden] that hold the band's gate and beta at every token."""
    _, beta_logit = _BANDS[case.band]
    heads, key_dim, hidden = config.num_heads, config.head_k_dim, config.hidden_size
    generator = torch.Generator().manual_seed(case.seed + 1)
    a = weights["A_log"].double().reshape(-1)[:heads].exp()
    logits = _band_logits(case, config)
    raw_target = (logits / a[:, None] - weights["dt_bias"].double().reshape(heads, key_dim)).reshape(-1)
    gate_rows = weights["f_b_proj.weight"].double() @ weights["f_a_proj.weight"].double()  # [H*K, hidden]
    rows = torch.cat((gate_rows, weights["b_proj.weight"].double()), dim=0)
    targets = torch.cat((raw_target, torch.full((heads,), beta_logit, dtype=torch.float64)))
    # Minimum-norm solution, ridge-limited to a realistic hidden magnitude when the gate rows are ill conditioned
    # (Kimi-K3's f_b f_a has condition ~1e5): channels then land near, not exactly on, the target.
    left, singular, right_t = torch.linalg.svd(rows, full_matrices=False)
    keep = singular > singular[0] * 1e-10
    left, singular, right_t = left[:, keep], singular[keep], right_t[keep]
    projected = left.T @ targets

    def solve(ridge: float) -> torch.Tensor:
        return right_t.T @ (singular / (singular.square() + ridge) * projected)

    center, ridge = solve(0.0), 0.0
    if float(center.square().mean().sqrt()) > _MAX_CENTER_RMS:
        low, high = 0.0, float(singular[0] ** 2)
        for _ in range(100):
            ridge = 0.5 * (low + high)
            low, high = (ridge, high) if float(solve(ridge).square().mean().sqrt()) > _MAX_CENTER_RMS else (low, ridge)
        center = solve(high)
    basis = right_t.T  # orthonormal basis of the row space
    noise = torch.randn(case.tokens, hidden, generator=generator, dtype=torch.float64)
    noise = noise - (noise @ basis) @ basis.T
    noise = noise / noise.square().mean(dim=-1, keepdim=True).sqrt()
    return (center + noise).unsqueeze(0).to(torch.bfloat16)


@dataclass(frozen=True)
class DecayExtremeReference:
    """Prepared case: host weights of the case's heads, crafted hidden, FP64 reference output and state per call."""

    weights: dict[str, torch.Tensor]
    hidden: torch.Tensor
    outputs: tuple[torch.Tensor, ...]
    states: tuple[torch.Tensor, ...]
    metadata: dict


def _cache_path(case: DecayExtremeCase) -> Path:
    import ttnn

    payload = {"version": _CACHE_VERSION, "case": asdict(case), "weights": case.weight_identity}
    if case.band == _TEXT_BAND:
        payload["inputs"] = _text_input_path(case).name  # the text input's own producer identity
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return Path(ttnn.CONFIG.model_cache_path) / "kda_decay_extremes" / f"{case.name}_{digest}.pt"


def _compute(case: DecayExtremeCase) -> DecayExtremeReference:
    config = case_config(case)
    weights = case_weights(case)
    text = case.band == _TEXT_BAND
    hidden = _text_hidden(case) if text else crafted_hidden(case, weights, config)
    state = None
    outputs, states = [], []
    for call in range(case.num_calls):
        chunk = hidden[:, call * case.chunk_tokens : (call + 1) * case.chunk_tokens]
        output, state = kda_forward_reference(chunk, weights, config, state, dtype=torch.float64)
        outputs.append(output)
        states.append(state.recurrent)
    gate = _reference_gate(weights, config, hidden)
    chunk_decay = -gate[0].reshape(-1, 32, config.num_heads, config.head_k_dim).sum(1)
    beta = torch.sigmoid(hidden.double()[0] @ weights["b_proj.weight"].double().T)
    metadata = {
        "center_rms": float(hidden.double()[0].mean(dim=0).square().mean().sqrt()),
        "hidden_rms": float(hidden.double().square().mean().sqrt()),
        "chunk_decay_min": float(chunk_decay.min()),
        "chunk_decay_median": float(chunk_decay.median()),
        "chunk_decay_max": float(chunk_decay.max()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
    }
    if text:
        chunk, remainder = divmod(int(chunk_decay.argmax()), config.num_heads * config.head_k_dim)
        metadata |= {
            "chunk_decay_max_at_chunk_channel": (chunk, remainder % config.head_k_dim),
            "weak_chunk_fraction": float((chunk_decay < 2.0**-9).double().mean()),
            "channels_weak_in_most_chunks": int(((chunk_decay < 2.0**-9).double().mean(0) > 0.5).sum()),
            "beta_median": float(beta.median()),
        }
    return DecayExtremeReference(weights, hidden, tuple(outputs), tuple(states), metadata)


def _reference_gate(weights, config, hidden):
    from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_gate_reference

    x = hidden.double()
    raw = (x @ weights["f_a_proj.weight"].double().T) @ weights["f_b_proj.weight"].double().T
    raw = raw.reshape(1, x.shape[1], config.num_heads, config.head_k_dim)
    return kda_gate_reference(raw, weights["A_log"], weights["dt_bias"], config.gate_lower_bound, dtype=torch.float64)


def decay_extreme_reference(case: DecayExtremeCase, *, compute_missing: bool) -> DecayExtremeReference:
    """Load the prepared reference; a miss fails fast unless compute_missing."""
    from models.demos.deepseek_v3_d_p.tests.kda.cases import prepared_cache_miss

    path = _cache_path(case)
    start = time.perf_counter()
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"{case.name}: reference cache hit {path} ({time.perf_counter() - start:.2f} s)")
        return DecayExtremeReference(**payload)
    if not compute_missing:
        raise prepared_cache_miss(case.name, "FP64 decay-extreme reference", path)
    reference = _compute(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(asdict(reference), temporary)
    temporary.replace(path)
    logger.info(
        f"{case.name}: reference computed in {time.perf_counter() - start:.1f} s -> {path}; {reference.metadata}"
    )
    return reference


# Registered cases (device tests select from these; preparation fills their references).
DECAY_EXTREME_CASES = {
    case.name: case
    for case in (
        DecayExtremeCase("synthetic", "control"),
        DecayExtremeCase("synthetic", "strong"),
        DecayExtremeCase("synthetic", "strong-saturated"),
        DecayExtremeCase("synthetic", "weak", num_calls=8),
        DecayExtremeCase("synthetic", "weak-beta", num_calls=8),
        DecayExtremeCase("k3", "control", head_start=48),
        DecayExtremeCase("k3", "strong", head_start=48),
        DecayExtremeCase("k3", "strong-saturated", head_start=48),
        DecayExtremeCase("k3", "weak", head_start=24, num_calls=8),
        DecayExtremeCase("glm", "control", head_start=50),
        DecayExtremeCase("glm", "strong", head_start=50),
        DecayExtremeCase("glm", "strong-saturated", head_start=50),
        DecayExtremeCase("glm", "weak", head_start=10, num_calls=8),
        # Real text (tt_metal_tracker-g1b.7.2). Heads chosen by the per-head per-chunk |G_last| of each model's
        # text input over 10240 tokens. K3 layers.1 never reaches the strong band on this text (max 79.7, h28): the
        # |G_last| = 160 window at tokens [776, 808) (aligned chunk 24 here) was measured with layers.0 gates
        # (heads 17/48); layers.1 h48 reaches 7.4 there.
        DecayExtremeCase("k3", _TEXT_BAND, head_start=28, num_calls=4),  # strongest: 73.9 within 5120 tokens
        DecayExtremeCase("k3", _TEXT_BAND, head_start=48),  # the layers.0 strong head, at the [776, 808) window
        DecayExtremeCase("k3", _TEXT_BAND, head_start=1, num_calls=8),  # LB-B real-text failure head
        DecayExtremeCase("k3", _TEXT_BAND, head_start=24, num_calls=8),  # most weak channels (107, 84% weak)
        DecayExtremeCase("k3", _TEXT_BAND, head_start=36, num_calls=8),  # weak channels (64) with low beta (0.11)
        DecayExtremeCase("glm", _TEXT_BAND, head_start=18),  # 156.9 in the first 1280 tokens (h18 c28)
        DecayExtremeCase("glm", _TEXT_BAND, head_start=32, num_calls=4),  # 157.3, the text maximum (chunk 77)
        DecayExtremeCase("glm", _TEXT_BAND, head_start=50, num_calls=4),  # crafted strong head; 147.5 on text
        DecayExtremeCase("glm", _TEXT_BAND, head_start=10, num_calls=8),  # LB-B real-text failure head (weak)
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FP64 references for KDA decay-extreme cases (no device).")
    parser.add_argument("--case", action="append", default=[], help="case name (default: all available)")
    arguments = parser.parse_args()
    names = arguments.case or list(DECAY_EXTREME_CASES)
    for name in names:
        case = DECAY_EXTREME_CASES[name]
        if case.weights != "synthetic" and not os.environ.get(_WEIGHT_SOURCES[case.weights][0]):
            logger.warning(f"{name}: skipped, {_WEIGHT_SOURCES[case.weights][0]} not set")
            continue
        logger.info(f"prepare {name} start")
        reference = decay_extreme_reference(case, compute_missing=True)
        logger.info(f"prepare {name} done: {reference.metadata}")


if __name__ == "__main__":
    main()
