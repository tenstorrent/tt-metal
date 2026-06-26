# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the assembled DiffusionGemma denoise loop step (#47463)."""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.denoise_loop import (
    denoise_block,
    denoise_step,
    renoise,
    temperature_at_step,
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
    """Logits with stable argmax and well-separated entropy ordering."""
    logits = torch.full((1, length, vocab_size), -4.0, dtype=torch.float32)
    token_ids = torch.arange(length) % vocab_size
    sharpness = torch.linspace(0.25, 2.0, length)
    logits[0, torch.arange(length), token_ids] = sharpness
    logits += torch.randn_like(logits) * 1.0e-3
    return logits


def _budget_for_accept_count(entropy: torch.Tensor, count: int):
    sorted_entropy = torch.sort(entropy, dim=-1).values
    exclusive = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    return float((exclusive[0, count - 1] + exclusive[0, count]) / 2)


def test_single_denoise_step_matches_reference(device):
    torch.manual_seed(11)
    length = 256
    vocab_size = 256
    max_steps = 48
    step = 3
    temperature = temperature_at_step(step, max_steps, 0.8, 0.4)

    logits = _structured_logits(length, vocab_size)
    gumbel_noise = torch.zeros_like(logits)
    noise_tokens = torch.randint(0, vocab_size, (1, length), dtype=torch.long)
    ref_entropy = S.token_entropy(logits, temperature=temperature)
    accept_count = 96
    budget = _budget_for_accept_count(ref_entropy, accept_count)
    ref = S.denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=budget,
        vocab_size=vocab_size,
        sampler=S.SAMPLER_GUMBEL,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        min_accept=0,
    )

    tt = denoise_step(
        _to_device(device, logits.unsqueeze(1)),
        temperature=temperature,
        entropy_budget=budget,
        gumbel_noise=_to_device(device, gumbel_noise.unsqueeze(1)),
        noise_tokens=_to_device(device, noise_tokens.view(1, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
    )

    out_entropy = ttnn.to_torch(tt.entropy).squeeze(1).squeeze(-1).float()
    out_accept = ttnn.to_torch(tt.accept_mask).squeeze(1).squeeze(1) > 0.5
    out_sampled = ttnn.to_torch(tt.sampled).squeeze(1).squeeze(-1).to(torch.long)
    out_argmax = ttnn.to_torch(tt.argmax).squeeze(1).squeeze(-1).to(torch.long)
    out_canvas = ttnn.to_torch(tt.canvas).squeeze(1).squeeze(-1).to(torch.long)

    passing, message = assert_with_pcc(ref.entropy.float(), out_entropy.float(), 0.99)
    assert passing, message
    assert torch.equal(out_accept, ref.accept_mask)
    assert torch.equal(out_sampled, ref.sampled)
    assert torch.equal(out_argmax, ref.argmax)
    assert torch.equal(out_canvas, ref.canvas)
    assert int(out_accept.sum()) == accept_count


def test_uint32_renoise_preserves_full_vocab_token_ids(device):
    sampled = torch.tensor([0, 1, 65535, 65536, 131071, 131072, 262143, 17], dtype=torch.int32).view(1, 1, 8, 1)
    noise_tokens = torch.tensor([262143, 131072, 131071, 65536, 65535, 1, 0, 2048], dtype=torch.int32).view(1, 1, 8, 1)
    accept = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32).view(1, 1, 8, 1)
    ref = torch.where(accept.bool(), sampled, noise_tokens).view(1, 8).to(torch.long)

    out = renoise(
        _to_device(device, accept, dtype=ttnn.bfloat16),
        _to_device(device, sampled, dtype=ttnn.uint32),
        _to_device(device, noise_tokens, dtype=ttnn.uint32),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(1).squeeze(-1).to(torch.long), ref)


def test_multi_step_denoise_control_flow_smoke_matches_reference(device):
    """Synthetic controller smoke; real canvas->W2 logits cycling is covered in the integration suite."""
    torch.manual_seed(17)
    batch = 1
    length = 256
    vocab_size = 256
    max_steps = 4

    logits = _structured_logits(length, vocab_size)
    step0_temperature = temperature_at_step(0, max_steps, 0.8, 0.4)
    ref_entropy = S.token_entropy(logits, temperature=step0_temperature)
    budget = _budget_for_accept_count(ref_entropy, 96)
    cfg = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=10.0,
        stable_steps_to_halt=1,
        entropy_budget=budget,
    )
    init_canvas = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    gumbel_noise = [torch.zeros_like(logits) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (batch, length), dtype=torch.long) for _ in range(max_steps)]

    ref = ref_denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        cfg,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_logits = _to_device(device, logits.unsqueeze(1))
    tt_gumbel_noise = [_to_device(device, noise.unsqueeze(1)) for noise in gumbel_noise]
    tt_noise_tokens = [
        _to_device(device, noise.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32) for noise in noise_tokens
    ]
    tt = denoise_block(
        lambda canvas, step: tt_logits,
        _to_device(device, init_canvas.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
        cfg,
        gumbel_noise_fn=lambda step: tt_gumbel_noise[step],
        noise_tokens_fn=lambda step: tt_noise_tokens[step],
    )

    comparison = compare_trajectories(ref, tt, max_entropy_abs_err_threshold=0.2)
    accept_flips = sum(int((ra.accept_mask != rb.accept_mask).sum()) for ra, rb in zip(ref.per_step, tt.per_step))
    assert comparison.passed, comparison
    assert ref.halted and tt.halted
    assert ref.num_steps == tt.num_steps == 2
    assert accept_flips == 0
