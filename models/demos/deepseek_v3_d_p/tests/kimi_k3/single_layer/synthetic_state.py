# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The AttnRes state a layer inherits, synthesized: a live stream plus ``ceil(L/12)`` sealed planes.

Sealed planes are BLOCK-LOCAL sums (sealing clears the live stream), so each plane is drawn on its
own rather than as a running prefix. The distribution is a stress distribution, not a model of real
activations: per-plane RMS spread over a decade so the AttnRes softmax has candidates of different
weight, per-token scale variation, and a few heavy-tailed rows -- the rows where a norm or a
softmax shift goes wrong first. Values are rounded to bf16 here, since that is what the device
holds and the reference should start from the same numbers.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config

H = KimiK3Config.EMB_SIZE
HEAVY_ROW_FRACTION = 0.01
HEAVY_ROW_SCALE = 20.0


def sealed_planes(layer: int) -> int:
    return -(-layer // KimiK3Config.ATTN_RES_BLOCK_SIZE)


def inherited_state(layer: int, tokens: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(live [T, d], sealed [T, S, d])`` in bf16-representable fp32, natural token order."""
    g = torch.Generator().manual_seed(seed * 7919 + layer)
    planes = sealed_planes(layer) + 1
    plane_rms = torch.logspace(-0.5, 0.5, planes)[torch.randperm(planes, generator=g)]
    token_scale = torch.exp(0.3 * torch.randn(tokens, 1, 1, generator=g))
    heavy = torch.rand(tokens, generator=g) < HEAVY_ROW_FRACTION
    token_scale[heavy] *= HEAVY_ROW_SCALE
    values = torch.randn(tokens, planes, H, generator=g) * plane_rms.view(1, planes, 1) * token_scale
    values = values.bfloat16().float()
    return values[:, 0].contiguous(), values[:, 1:].contiguous()


def request_tokens(tokens: int, *, seed: int) -> torch.Tensor:
    """Token ids uniform over the vocabulary (layer 0's input)."""
    g = torch.Generator().manual_seed(seed * 104729 + 1)
    return torch.randint(0, KimiK3Config.VOCAB_SIZE, (tokens,), generator=g)
