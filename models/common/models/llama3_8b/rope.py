# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RoPE table construction for the TTTv2 Llama-3.1-8B path."""

import math
from dataclasses import dataclass

import torch
from loguru import logger


@dataclass(frozen=True)
class RopeScaling:
    rope_type: str
    factor: float | None = None
    original_max_position_embeddings: int | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0


def rope_scaling_model_factory(
    rope_scaling_params: dict | None, original_max_context_len: int | None = None
) -> RopeScaling | None:
    if rope_scaling_params is None:
        return None

    rope_type = rope_scaling_params.get("rope_type") or rope_scaling_params.get("type")
    if rope_type in ("default", "mrope"):
        logger.warning(
            f"Rope scaling type was set to {rope_type}, defaulting to no rope scaling as this rope type is not supported yet by TTTv2"
        )
        return None
    if rope_type not in ("linear", "llama3"):
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_type}")

    return RopeScaling(
        rope_type=rope_type,
        factor=rope_scaling_params.get("factor"),
        original_max_position_embeddings=rope_scaling_params.get(
            "original_max_position_embeddings", original_max_context_len
        ),
        low_freq_factor=rope_scaling_params.get("low_freq_factor", 1.0),
        high_freq_factor=rope_scaling_params.get("high_freq_factor", 4.0),
    )


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _gather_cos_sin(position_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _llama3_scaled_inv_freq(freqs: torch.Tensor, scaling: RopeScaling) -> torch.Tensor:
    assert scaling.factor is not None
    assert scaling.original_max_position_embeddings is not None

    low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor
    high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling.factor)
        else:
            smooth = (scaling.original_max_position_embeddings / wavelen - scaling.low_freq_factor) / (
                scaling.high_freq_factor - scaling.low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scaling.factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def compute_gather_cos_sin(
    dhead: int, end: int, theta: float, rope_scaling: RopeScaling | None
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = end // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dhead, 2).float() / dhead))

    if rope_scaling is None:
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return _permute_to_meta_format(emb.cos(), emb.sin())

    if rope_scaling.rope_type == "linear":
        assert rope_scaling.factor is not None
        inv_freq = inv_freq / rope_scaling.factor
    elif rope_scaling.rope_type == "llama3":
        inv_freq = _llama3_scaled_inv_freq(inv_freq, rope_scaling)
    else:
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_scaling.rope_type}")

    t = torch.arange(seq_len * 2.0)
    freqs = torch.outer(t, inv_freq).float()
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return _gather_cos_sin(torch.arange(seq_len), cos, sin)
