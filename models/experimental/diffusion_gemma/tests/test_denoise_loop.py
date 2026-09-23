# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical reference trajectory, drift, halt, and device-parity denoise gates."""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block, denoise_step, temperature_at_step
from tests.ttnn.utils_for_testing import assert_with_pcc


def _gen(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _cfg(**kwargs):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kwargs)


def _peaked_logits(batch, length, vocab, target):
    logits = torch.full((batch, length, vocab), -1e4)
    logits[..., target] = 1e4
    return logits


def test_halts_on_stable_low_entropy():
    batch, length, vocab, target = 1, 8, 32, 7
    peaked = _peaked_logits(batch, length, vocab, target)
    init = S.random_canvas((batch, length), vocab, generator=_gen(1))
    trajectory = ref_denoise_block(lambda canvas, step: peaked, init, _cfg(), vocab)
    assert trajectory.halted
    assert trajectory.num_steps <= 3
    assert torch.equal(trajectory.committed, torch.full((batch, length), target))
    assert len(trajectory.per_step) == trajectory.num_steps


def test_runs_to_cap_when_never_converges():
    batch, length, vocab = 1, 8, 64
    flat = torch.zeros(batch, length, vocab)
    trajectory = ref_denoise_block(
        lambda canvas, step: flat,
        S.random_canvas((batch, length), vocab, generator=_gen(2)),
        _cfg(),
        vocab,
    )
    assert not trajectory.halted
    assert trajectory.num_steps == 8
    assert all(record.entropy_mean > 1.0 for record in trajectory.per_step)


def _random_traj(seed):
    batch, length, vocab = 1, 12, 40

    def logits_fn(canvas, step):
        return torch.randn(batch, length, vocab, generator=_gen(seed * 1000 + step))

    init = S.random_canvas((batch, length), vocab, generator=_gen(seed))
    return ref_denoise_block(logits_fn, init, _cfg(), vocab)


def test_entropy_abs_gate_catches_affine_error_pcc_misses():
    reference = _random_traj(seed=7)
    shifted_steps = [record._replace(entropy=record.entropy + 0.5) for record in reference.per_step]
    candidate = reference._replace(per_step=shifted_steps)
    comparison = compare_trajectories(reference, candidate)
    assert min(comparison.per_step_entropy_pcc) > 0.99
    assert comparison.max_entropy_abs_err >= 0.49
    assert not comparison.passed
    assert comparison.min_argmax_agreement == 1.0
    assert comparison.min_canvas_agreement == 1.0
    assert comparison.min_accept_iou == 1.0


def test_decision_level_fields_distinguish_drifted_trajectories():
    comparison = compare_trajectories(_random_traj(seed=11), _random_traj(seed=42))
    assert not comparison.passed
    assert comparison.min_sampled_agreement < 1.0
    assert comparison.min_accept_iou < 1.0
    assert comparison.min_canvas_agreement < 1.0


requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _structured_logits(length: int, vocab_size: int):
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


class _ResettableStaticLogits:
    def __init__(self, logits):
        self.logits = logits
        self.reset_calls = 0

    def __call__(self, canvas, step):
        return self.logits

    def reset(self):
        self.reset_calls += 1
        if self.logits is not None:
            self.logits.deallocate(True)
            self.logits = None


@requires_device
@pytest.mark.use_module_device
def test_single_denoise_step_matches_reference(device):
    torch.manual_seed(11)
    length, vocab_size, max_steps, step = 256, 256, 48, 3
    temperature = temperature_at_step(step, max_steps, 0.8, 0.4)
    logits = _structured_logits(length, vocab_size)
    gumbel_noise = torch.zeros_like(logits)
    noise_tokens = torch.randint(0, vocab_size, (1, length), dtype=torch.long)
    reference_entropy = S.token_entropy(logits, temperature=temperature)
    accept_count = 96
    budget = _budget_for_accept_count(reference_entropy, accept_count)
    reference = S.denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=budget,
        vocab_size=vocab_size,
        sampler=S.SAMPLER_GUMBEL,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        min_accept=0,
    )

    result = denoise_step(
        _to_device(device, logits.unsqueeze(1)),
        temperature=temperature,
        entropy_budget=budget,
        gumbel_noise=_to_device(device, gumbel_noise.unsqueeze(1)),
        noise_tokens=_to_device(
            device,
            noise_tokens.view(1, 1, length, 1).to(torch.int32),
            dtype=ttnn.uint32,
        ),
    )

    out_entropy = ttnn.to_torch(result.entropy).squeeze(1).squeeze(-1).float()
    out_accept = ttnn.to_torch(result.accept_mask).squeeze(1).squeeze(1) > 0.5
    out_sampled = ttnn.to_torch(result.sampled).squeeze(1).squeeze(-1).to(torch.long)
    out_argmax = ttnn.to_torch(result.argmax).squeeze(1).squeeze(-1).to(torch.long)
    out_canvas = ttnn.to_torch(result.canvas).squeeze(1).squeeze(-1).to(torch.long)

    passing, message = assert_with_pcc(reference.entropy.float(), out_entropy.float(), 0.99)
    assert passing, message
    assert torch.equal(out_accept, reference.accept_mask)
    assert torch.equal(out_sampled, reference.sampled)
    assert torch.equal(out_argmax, reference.argmax)
    assert torch.equal(out_canvas, reference.canvas)
    assert int(out_accept.sum()) == accept_count


@requires_device
@pytest.mark.use_module_device
def test_multi_step_denoise_control_flow_smoke_matches_reference(device):
    torch.manual_seed(17)
    batch, length, vocab_size, max_steps = 1, 256, 256, 4
    logits = _structured_logits(length, vocab_size)
    reference_entropy = S.token_entropy(logits, temperature=temperature_at_step(0, max_steps, 0.8, 0.4))
    budget = _budget_for_accept_count(reference_entropy, 96)
    config = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=10.0,
        stable_steps_to_halt=1,
        entropy_budget=budget,
    )
    init_canvas = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    gumbel_noise = [torch.zeros_like(logits) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (batch, length), dtype=torch.long) for _ in range(max_steps)]

    reference = ref_denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        config,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_logits = _ResettableStaticLogits(_to_device(device, logits.unsqueeze(1)))
    tt_gumbel_noise = [_to_device(device, noise.unsqueeze(1)) for noise in gumbel_noise]
    tt_noise_tokens = [
        _to_device(
            device,
            noise.view(batch, 1, length, 1).to(torch.int32),
            dtype=ttnn.uint32,
        )
        for noise in noise_tokens
    ]
    result = denoise_block(
        tt_logits,
        _to_device(
            device,
            init_canvas.view(batch, 1, length, 1).to(torch.int32),
            dtype=ttnn.uint32,
        ),
        config,
        gumbel_noise_fn=lambda step: tt_gumbel_noise[step],
        noise_tokens_fn=lambda step: tt_noise_tokens[step],
    )

    comparison = compare_trajectories(reference, result, max_entropy_abs_err_threshold=0.2)
    accept_flips = sum(
        int((ref_step.accept_mask != tt_step.accept_mask).sum())
        for ref_step, tt_step in zip(reference.per_step, result.per_step)
    )
    assert comparison.passed, comparison
    assert reference.halted and result.halted
    assert reference.num_steps == result.num_steps == 2
    assert accept_flips == 0
    assert tt_logits.reset_calls == 1
