# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity of our reference/ oracle vs the REAL installed transformers diffusion_gemma (#47468).

``transformers`` ships ``diffusion_gemma`` since 5.12, so the authoritative drift
guard is to test our purpose-built reference primitives against the **actual
installed** classes — not the vendored copies in ``reference/_upstream.py`` (those
remain only as a fallback for envs without diffusion_gemma, e.g. old CI). This file
supersedes ``test_upstream_parity.py`` whenever transformers >= 5.12 is present;
the two together mean the reference is pinned to the real model in every env.

Standalone-importable (no 51 GB checkpoint): the primitives instantiate from config
alone.
"""

import importlib.util

import pytest
import torch

from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.self_conditioning import DiffusionGemmaRMSNorm, SelfConditioning

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.diffusion_gemma") is None,
    reason="transformers.models.diffusion_gemma not installed (ships since transformers 5.12)",
)


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_entropy_accept_matches_real_EntropyBoundSampler():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )

    for seed, bound in [(2, 0.05), (3, 0.1), (4, 0.5), (5, 2.0)]:
        logits = torch.randn(1, 64, 128, generator=_gen(seed))
        sampler = EntropyBoundSampler(
            EntropyBoundSamplerConfig(entropy_bound=bound), canvas_length=64, vocab_size=128, max_denoising_steps=48
        )
        # accept_canvas sets sampler.accepted_token_mask (the scatter-back accept mask)
        sampler.accept_canvas(torch.zeros(1, 64, dtype=torch.long), torch.ones(1, 64, dtype=torch.long), logits, 1)
        real_mask = sampler.accepted_token_mask.bool()
        ours = S.entropy_budget_accept(S.token_entropy(logits), bound, min_accept=0)
        assert torch.equal(ours, real_mask), f"seed {seed} bound {bound}: accept mask differs from real sampler"


def test_temperature_matches_real_LinearSchedule():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        LinearTemperatureScheduleLogitsProcessor,
    )

    N, t_min, t_max = 48, 0.4, 0.8
    proc = LinearTemperatureScheduleLogitsProcessor(t_min=t_min, t_max=t_max, max_denoising_steps=N)
    scores = torch.ones(1, 10)
    for cur_step in range(1, N + 1):
        out = proc(None, scores.clone(), cur_step)  # scores / temperature
        real_temp = float(scores[0, 0] / out[0, 0])
        ours_temp = S.temperature_at_step(N - cur_step, N, t_max, t_min)  # forward index = N - cur_step
        assert abs(real_temp - ours_temp) < 1e-5, f"cur_step {cur_step}: temp {ours_temp} != {real_temp}"


def test_stopping_confidence_matches_real_criterion():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import StableAndConfidentStoppingCriteria

    logits = torch.randn(4, 32, 200, generator=_gen(6))
    thresh = float(S.token_entropy(logits).mean(dim=-1).median())
    # stability_threshold=0 -> stable is always True, isolating the confidence comparison
    real = StableAndConfidentStoppingCriteria(stability_threshold=0, confidence_threshold=thresh)
    real_out = real(logits.argmax(dim=-1), logits)
    ours = S.token_entropy(logits).mean(dim=-1) < thresh
    assert torch.equal(ours, real_out)


def test_rmsnorm_matches_real():
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaRMSNorm as RealRMS

    x = torch.randn(2, 3, 16, generator=_gen(9))
    for with_scale in [True, False]:
        real = RealRMS(16, eps=1e-6, with_scale=with_scale)
        ours = DiffusionGemmaRMSNorm(16, eps=1e-6, with_scale=with_scale)
        if with_scale:
            with torch.no_grad():
                real.weight.copy_(torch.randn(16, generator=_gen(3)))
                ours.weight.copy_(real.weight)
        assert torch.allclose(ours(x), real(x), atol=1e-6), f"with_scale={with_scale}"


def test_self_conditioning_matches_real():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaSelfConditioning

    hidden, inter = 16, 40
    cfg = DiffusionGemmaTextConfig(
        hidden_size=hidden, intermediate_size=inter, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh"
    )
    real = DiffusionGemmaSelfConditioning(cfg).eval()
    ours = SelfConditioning(hidden, intermediate_size=inter, eps=1e-6, activation="gelu_pytorch_tanh").eval()
    with torch.no_grad():
        ours.pre_norm.weight.copy_(real.pre_norm.weight)
        ours.gate_proj.weight.copy_(real.gate_proj.weight)
        ours.up_proj.weight.copy_(real.up_proj.weight)
        ours.down_proj.weight.copy_(real.down_proj.weight)
    emb = torch.randn(2, 5, hidden, generator=_gen(8))
    sig = torch.randn(2, 5, hidden, generator=_gen(9))
    assert torch.allclose(ours(emb, sig), real(emb, sig), atol=1e-5)
