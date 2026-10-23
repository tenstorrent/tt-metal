# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical host oracles and device regressions for diffusion sampling decisions."""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.denoise_loop import entropy_budget_accept
from models.experimental.diffusion_gemma.tt.sampling import argmax_last_dim
from models.experimental.diffusion_gemma.tt.sampling_params import (
    canvas_sample_from_params,
    canvas_sampling_config_from_params,
)


requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
requires_device_sfpi = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
module_device = pytest.mark.use_module_device

SEQUENCE_LENGTH = 256


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


# Minimal host oracles for each decision-critical behavior.
def test_token_entropy_uniform_and_peaked():
    vocab_size = 64
    uniform = torch.zeros(1, 4, vocab_size)
    expected = torch.log(torch.tensor(float(vocab_size)))
    assert torch.allclose(
        S.token_entropy(uniform),
        torch.full((1, 4), float(expected)),
        atol=1e-5,
    )

    peaked = torch.full((1, 4, vocab_size), -1e4)
    peaked[..., 0] = 1e4
    assert torch.all(S.token_entropy(peaked) < 1e-3)


def test_gumbel_max_zero_noise_is_argmax():
    logits = torch.randn(2, 8, 50, generator=_generator(1))
    out = S.gumbel_max_sample(
        logits,
        temperature=0.7,
        noise=torch.zeros_like(logits),
    )
    assert torch.equal(out, logits.argmax(dim=-1))


def test_acceptance_scatter_back_inverse_permutation():
    entropy = torch.rand(3, 17, generator=_generator(4))
    budget = 0.9
    accepted = S.entropy_budget_accept(entropy, budget=budget, min_accept=1)

    sorted_entropy, indices = torch.sort(entropy, dim=-1)
    exclusive_prefix = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    accepted_sorted = exclusive_prefix <= budget
    reference = torch.zeros_like(entropy, dtype=torch.bool)
    reference.scatter_(-1, indices, accepted_sorted)

    assert torch.equal(accepted, reference)


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(
        value,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _release(*tensors):
    for tensor in tensors:
        tensor.deallocate(True)


def _structured_logits_jittered(length, vocab_size):
    logits = torch.full((1, length, vocab_size), -2.0, dtype=torch.float32)
    base_ids = torch.arange(length) % vocab_size
    alternative_ids = (base_ids + 17) % vocab_size
    logits[0, torch.arange(length), base_ids] = torch.linspace(0.5, 4.0, length)
    logits[0, torch.arange(length), alternative_ids] = torch.linspace(
        0.25,
        2.0,
        length,
    )
    return logits + torch.randn_like(logits) * 1.0e-3


@requires_device
@module_device
def test_canvas_sampling_matches_injected_gumbel_reference(device):
    """Cover the direct op and serving wrapper with one shared host oracle."""
    torch.manual_seed(37)
    length = 256
    vocab_size = 512
    temperature = S.temperature_at_step(
        step=11,
        num_steps=48,
        t_start=0.8,
        t_end=0.4,
    )
    logits = _structured_logits_jittered(length, vocab_size)
    noise = S.sample_gumbel_noise(
        logits.shape,
        generator=torch.Generator().manual_seed(41),
    )
    sampling_params = {
        "temperature": temperature,
        "top_k": 64,
        "top_p": 0.95,
        "seed": 41,
    }
    config = canvas_sampling_config_from_params(
        sampling_params,
        default_temperature=0.8,
    )
    assert config.top_k == 64
    assert config.top_p == 0.95
    assert config.top_k_top_p_supported is False

    reference = S.gumbel_max_sample(logits, temperature, noise=noise)
    tt_logits = _to_device(device, logits)
    tt_noise = _to_device(device, noise)
    direct = TS.canvas_sample(tt_logits, temperature, tt_noise)
    configured = canvas_sample_from_params(
        tt_logits,
        sampling_params,
        default_temperature=0.8,
        gumbel_noise=tt_noise,
    )

    assert torch.equal(
        ttnn.to_torch(direct).squeeze(-1).to(torch.long),
        reference,
    )
    assert torch.equal(
        ttnn.to_torch(configured).squeeze(-1).to(torch.long),
        reference,
    )


def _structured_logits_repeated(num_samples, length, vocab_size):
    logits = torch.full((1, length, vocab_size), -1.5, dtype=torch.float32)
    top_ids = torch.arange(length) % vocab_size
    alternative_ids = (top_ids + 5) % vocab_size
    logits[0, torch.arange(length), top_ids] = torch.linspace(0.75, 1.25, length)
    logits[0, torch.arange(length), alternative_ids] = torch.linspace(
        0.25,
        0.75,
        length,
    )
    return logits.expand(num_samples, -1, -1).contiguous()


@requires_device
@module_device
def test_canvas_sample_matches_torch_argmax_with_readback_device_noise(device):
    num_samples, length, vocab_size = 64, 32, 32
    temperature = 0.7
    logits = _structured_logits_repeated(num_samples, length, vocab_size)

    tt_logits = _to_device(device, logits)
    device_noise = TS.sample_gumbel_noise(
        logits.shape,
        device=device,
        seed=47472,
    )
    samples = TS.canvas_sample(tt_logits, temperature, device_noise)
    host_noise = ttnn.to_torch(device_noise).float()
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)
    _release(tt_logits, device_noise, samples)

    reference = torch.argmax(logits / temperature + host_noise, dim=-1)
    assert torch.equal(sample_ids, reference)


CANVAS_LENGTH = 256
PROBE_VOCAB = 16384


def _draw_gumbel(arm, shape, *, device, seed):
    assert arm == "device"
    noise = TS.sample_gumbel_noise(shape, device=device, seed=seed)
    host = ttnn.to_torch(noise).float()
    noise.deallocate(True)
    return host.reshape(-1, shape[-1])[: shape[-2], :]


def _independence_metrics(noise):
    vocab_size = noise.shape[1]
    correlation = torch.nan_to_num(torch.corrcoef(noise.double()), nan=0.0)
    off_diagonal = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    magnitudes = correlation[off_diagonal].abs()
    sigma = 1.0 / (vocab_size**0.5)
    winners = noise.argmax(dim=-1)
    counts = torch.bincount(winners)
    return {
        "max_r_in_sigmas": float(magnitudes.max()) / sigma,
        "frac_pairs_over_5sigma": float((magnitudes > 5.0 * sigma).float().mean()),
        "distinct_winners": int((counts > 0).sum()),
        "max_multiplicity": int(counts.max()),
    }


def _measure_independence(arm, *, device):
    noise = _draw_gumbel(
        arm,
        (1, 1, CANVAS_LENGTH, PROBE_VOCAB),
        device=device,
        seed=48291,
    )
    return _independence_metrics(noise)


@requires_device
@module_device
def test_shipped_plain_gumbel_is_independent_across_canvas_positions(device):
    stats = _measure_independence("device", device=device)
    assert stats["max_multiplicity"] <= 4
    assert stats["frac_pairs_over_5sigma"] < 0.01


VOCAB_SIZE = 2048


def _varied_logits(seed=1):
    base = torch.randn(
        1,
        SEQUENCE_LENGTH,
        VOCAB_SIZE,
        generator=_generator(seed),
    )
    scales = torch.linspace(0.2, 6.0, SEQUENCE_LENGTH).view(
        1,
        SEQUENCE_LENGTH,
        1,
    )
    return base * scales


@requires_device_sfpi
@module_device
def test_token_entropy_bf16_accurate_and_bfp8_degrades(device):
    temperature = 0.6
    logits = _varied_logits()
    reference = S.token_entropy(logits, temperature=temperature)

    def error(dtype):
        out = ttnn.to_torch(
            TS.token_entropy(
                _to_device(device, logits, dtype=dtype),
                temperature=temperature,
            )
        ).squeeze(-1)
        assert torch.isfinite(out).all()
        return (reference - out).abs(), comp_pcc(reference, out, 0.0)[1]

    bf16_delta, bf16_pcc = error(ttnn.bfloat16)
    bfp8_delta, _ = error(ttnn.bfloat8_b)

    assert bf16_delta.mean() < 0.5
    assert bf16_pcc >= 0.99
    assert bfp8_delta.mean() > 2.0 * bf16_delta.mean()


@requires_device_sfpi
@module_device
def test_gumbel_max_argmax_agreement(device):
    temperature = 0.6
    logits = _varied_logits(seed=2)
    noise = S.sample_gumbel_noise(
        (1, SEQUENCE_LENGTH, VOCAB_SIZE),
        generator=_generator(3),
    )
    reference = S.gumbel_max_sample(logits, temperature, noise=noise)
    out = (
        ttnn.to_torch(
            TS.gumbel_max(
                _to_device(device, logits, dtype=ttnn.bfloat16),
                temperature,
                _to_device(device, noise, dtype=ttnn.bfloat16),
            )
        )
        .squeeze(-1)
        .to(torch.long)
    )

    agreement = float((out == reference).float().mean())
    assert agreement >= 0.95


def _budget_for_fraction(entropy, fraction):
    sorted_cumulative = torch.cumsum(
        torch.sort(entropy, dim=-1).values,
        dim=-1,
    )
    index = int(fraction * entropy.shape[-1])
    return float((sorted_cumulative[0, index - 1] + sorted_cumulative[0, index]) / 2)


@requires_device_sfpi
@module_device
def test_production_entropy_budget_accept_guards_device_sort_at_canvas_256(device):
    torch.manual_seed(47463)
    entropy = torch.rand(1, SEQUENCE_LENGTH) + torch.arange(SEQUENCE_LENGTH).float() * 1e-4
    budget = _budget_for_fraction(entropy, 0.375)
    reference = S.entropy_budget_accept(entropy, budget, min_accept=0)
    tt_entropy = ttnn.from_torch(
        entropy.float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    accepted = entropy_budget_accept(tt_entropy, budget)
    got = ttnn.to_torch(accepted) > 0.5

    try:
        assert torch.equal(got, reference)
    finally:
        accepted.deallocate(True)
        tt_entropy.deallocate(True)


def _served_logits(width, seed=13):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(
        1,
        1,
        SEQUENCE_LENGTH,
        width,
        generator=generator,
    ) * torch.linspace(0.5, 4.0, SEQUENCE_LENGTH).view(
        1,
        1,
        SEQUENCE_LENGTH,
        1,
    )
    winners = torch.randint(
        0,
        width,
        (SEQUENCE_LENGTH,),
        generator=generator,
    )
    for row, column in enumerate(winners.tolist()):
        logits[0, 0, row, column] = logits[0, 0, row].max().item() + 1.0
    return logits, winners


@requires_device
@module_device
def test_argmax_last_dim_matches_torch_at_served_width(device):
    width = 65536
    logits, expected = _served_logits(width)
    tt_logits = ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    try:
        got = ttnn.to_torch(argmax_last_dim(tt_logits)).reshape(-1)[:SEQUENCE_LENGTH].to(torch.int64)
    finally:
        tt_logits.deallocate(True)

    assert torch.equal(got, expected)
