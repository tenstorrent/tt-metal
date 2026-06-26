# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic device canvas sampling tests for DiffusionGemma W4 (#47472)."""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.sampling_params import (
    canvas_sample_from_params,
    canvas_sampling_config_from_params,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _structured_logits(length: int, vocab_size: int):
    logits = torch.full((1, length, vocab_size), -2.0, dtype=torch.float32)
    base_ids = torch.arange(length) % vocab_size
    alt_ids = (base_ids + 17) % vocab_size
    logits[0, torch.arange(length), base_ids] = torch.linspace(0.5, 4.0, length)
    logits[0, torch.arange(length), alt_ids] = torch.linspace(0.25, 2.0, length)
    logits += torch.randn_like(logits) * 1.0e-3
    return logits


def test_canvas_sample_matches_injected_gumbel_reference(device):
    torch.manual_seed(23)
    length = 256
    vocab_size = 512
    temperature = S.temperature_at_step(step=5, num_steps=48, t_start=0.8, t_end=0.4)
    logits = _structured_logits(length, vocab_size)
    noise = S.sample_gumbel_noise(logits.shape, generator=torch.Generator().manual_seed(29))

    ref = S.gumbel_max_sample(logits, temperature, noise=noise)
    out = TS.canvas_sample(
        _to_device(device, logits),
        temperature,
        _to_device(device, noise),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(-1).to(torch.long), ref)


def test_canvas_sample_from_params_matches_injected_gumbel_reference(device):
    torch.manual_seed(37)
    length = 256
    vocab_size = 512
    temperature = S.temperature_at_step(step=11, num_steps=48, t_start=0.8, t_end=0.4)
    logits = _structured_logits(length, vocab_size)
    noise = S.sample_gumbel_noise(logits.shape, generator=torch.Generator().manual_seed(41))
    sampling_params = {"temperature": temperature, "top_k": 64, "top_p": 0.95, "seed": 41}
    config = canvas_sampling_config_from_params(sampling_params, default_temperature=0.8)
    assert config.top_k == 64
    assert config.top_p == 0.95
    assert config.top_k_top_p_supported is False

    ref = S.gumbel_max_sample(logits, temperature, noise=noise)
    out = canvas_sample_from_params(
        _to_device(device, logits),
        sampling_params,
        default_temperature=0.8,
        gumbel_noise=_to_device(device, noise),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(-1).to(torch.long), ref)


def test_temperature_scale_matches_reference(device):
    torch.manual_seed(31)
    logits = torch.randn(1, 256, 512, dtype=torch.float32)
    temperature = S.temperature_at_step(step=17, num_steps=48, t_start=0.8, t_end=0.4)
    out = TS.temperature_scale(_to_device(device, logits), temperature)

    passing, message = assert_with_pcc(logits / temperature, ttnn.to_torch(out), 0.9999)
    assert passing, message
