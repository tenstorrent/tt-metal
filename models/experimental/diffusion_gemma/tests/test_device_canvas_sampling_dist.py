# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Distributional device canvas sampling test for DiffusionGemma W4 (#47472)."""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.diffusion_gemma.tt import sampling as TS

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

    empirical = F.one_hot(sample_ids, num_classes=vocab_size).float().mean(dim=0)
    top_ids = expected_probs.argmax(dim=-1)
    top_expected = expected_probs.gather(-1, top_ids[:, None]).squeeze(-1)
    top_empirical = empirical.gather(-1, top_ids[:, None]).squeeze(-1)
    max_top1_freq_error = float((top_empirical - top_expected).abs().max())

    eps = 1.0e-4
    kl = (expected_probs * (expected_probs.clamp_min(eps).log() - empirical.clamp_min(eps).log())).sum(dim=-1)
    mean_kl = float(kl.mean())

    print(
        f"\n[canvas sampling dist] N={num_samples} max_top1_freq_error={max_top1_freq_error:.4f} "
        f"mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < 0.05
    assert mean_kl < 0.05
