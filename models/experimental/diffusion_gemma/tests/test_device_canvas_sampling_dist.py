# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Distributional device canvas sampling test for DiffusionGemma W4 (#47472)."""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.sampling_params import canvas_sample_from_params

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _structured_logits(num_samples: int, length: int, vocab_size: int):
    logits = torch.full((1, length, vocab_size), -1.5, dtype=torch.float32)
    top_ids = torch.arange(length) % vocab_size
    alt_ids = (top_ids + 5) % vocab_size
    logits[0, torch.arange(length), top_ids] = torch.linspace(0.75, 1.25, length)
    logits[0, torch.arange(length), alt_ids] = torch.linspace(0.25, 0.75, length)
    return logits.expand(num_samples, -1, -1).contiguous()


def _distribution_metrics(sample_ids, expected_probs):
    vocab_size = expected_probs.shape[-1]
    empirical = F.one_hot(sample_ids, num_classes=vocab_size).float().mean(dim=0)
    top_ids = expected_probs.argmax(dim=-1)
    top_expected = expected_probs.gather(-1, top_ids[:, None]).squeeze(-1)
    top_empirical = empirical.gather(-1, top_ids[:, None]).squeeze(-1)
    max_top1_freq_error = float((top_empirical - top_expected).abs().max())

    eps = 1.0e-4
    kl = (expected_probs * (expected_probs.clamp_min(eps).log() - empirical.clamp_min(eps).log())).sum(dim=-1)
    return max_top1_freq_error, float(kl.mean())


def test_canvas_sample_consumes_regenerated_device_noise(device):
    num_samples = 64
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)

    device_noise = TS.sample_gumbel_noise(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(_to_device(device, logits), temperature, device_noise)

    host_noise = ttnn.to_torch(device_noise).float()
    ref = torch.argmax(logits / temperature + host_noise, dim=-1)
    assert torch.equal(ttnn.to_torch(samples).squeeze(-1).to(torch.long), ref)


def test_canvas_sample_chunked_regenerated_noise_distribution(device):
    # N=4096 keeps 32-way categorical frequency error below 0.05 with fixed
    # device seeds. The vocab-chunked path is intentionally slow and diagnostic:
    # it proves the sampler distribution when the regenerated noise is iid enough.
    num_samples = 4096
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)

    device_noise = TS.sample_gumbel_noise_by_vocab_chunks(
        logits.shape,
        device=device,
        seed=47472,
        vocab_chunk_size=1,
    )
    samples = TS.canvas_sample(_to_device(device, logits), temperature, device_noise)
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling chunked dist] N={num_samples} max_top1_freq_error={max_top1_freq_error:.4f} "
        f"mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05


def test_canvas_sample_permuted_vocab_regenerated_noise_distribution(device):
    # One ttnn.rand call is still used, but vocab is generated as the outer axis
    # and permuted back to avoid the known innermost-vocab correlation.
    num_samples = 4096
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)

    device_noise = TS.sample_gumbel_noise_with_permuted_vocab(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(_to_device(device, logits), temperature, device_noise)
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling permuted-vocab dist] N={num_samples} "
        f"max_top1_freq_error={max_top1_freq_error:.4f} mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05


def test_canvas_sample_from_params_chunked_regenerated_noise_distribution(device):
    # Covers the vLLM seam's seed-regenerated path without using the known-biased
    # single-call ttnn.rand noise over the whole vocab axis.
    num_samples = 4096
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)
    sampling_params = {"temperature": temperature, "top_k": 64, "top_p": 0.95, "seed": 47472}

    samples = canvas_sample_from_params(
        _to_device(device, logits),
        sampling_params,
        default_temperature=0.8,
        use_vocab_chunked_noise=True,
        use_vocab_permuted_noise=False,
        vocab_chunk_size=1,
    )
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling params chunked dist] N={num_samples} "
        f"max_top1_freq_error={max_top1_freq_error:.4f} mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05


def test_canvas_sample_from_params_permuted_vocab_regenerated_noise_distribution(device):
    num_samples = 4096
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)
    sampling_params = {"temperature": temperature, "top_k": 64, "top_p": 0.95, "seed": 47472}

    samples = canvas_sample_from_params(
        _to_device(device, logits),
        sampling_params,
        default_temperature=0.8,
        use_vocab_permuted_noise=True,
    )
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling params permuted-vocab dist] N={num_samples} "
        f"max_top1_freq_error={max_top1_freq_error:.4f} mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05


@pytest.mark.xfail(
    reason=(
        "QB2 ttnn.rand regenerated noise is currently not iid enough for W4 distributional canvas sampling: "
        "uniform-logit argmax histograms show empty/high buckets while torch-injected Gumbel noise is exact."
    ),
    strict=True,
)
def test_canvas_sample_regenerated_noise_distribution(device):
    num_samples = 4096
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)

    device_noise = TS.sample_gumbel_noise(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(_to_device(device, logits), temperature, device_noise)
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling dist] N={num_samples} max_top1_freq_error={max_top1_freq_error:.4f} "
        f"mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05
