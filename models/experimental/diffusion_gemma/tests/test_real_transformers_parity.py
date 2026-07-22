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
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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


def test_real_denoising_step_uses_temperature_processed_logits_for_decisions():
    """HF routes processed logits to accept/stop/self-conditioning, not raw logits."""

    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationMixin,
        LinearTemperatureScheduleLogitsProcessor,
    )

    class _DecoderOutput:
        def __init__(self, logits):
            self.logits = logits

    class _Sampler:
        def __init__(self):
            self.accept_logits = None

        def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
            self.accept_logits = logits.detach().clone()
            return denoiser_canvas

        def renoise_canvas(self, accepted_canvas, cur_step):
            return accepted_canvas

    class _Stopping:
        def __init__(self):
            self.argmax_canvas = None
            self.logits = None

        def __call__(self, argmax_canvas, logits):
            self.argmax_canvas = argmax_canvas.detach().clone()
            self.logits = logits.detach().clone()
            return torch.zeros(argmax_canvas.shape[0], dtype=torch.bool)

    batch, length, vocab = 1, 3, 7
    raw_logits = torch.randn(batch, length, vocab, generator=_gen(10))
    cur_step = 2
    processor = LogitsProcessorList(
        [LinearTemperatureScheduleLogitsProcessor(t_min=0.4, t_max=0.8, max_denoising_steps=4)]
    )
    expected_processed = processor(None, raw_logits, cur_step=torch.tensor(cur_step, dtype=torch.int32))

    fake_self = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab)),
        model=SimpleNamespace(
            decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)))
        ),
    )
    sampler = _Sampler()
    stopping = _Stopping()

    _, _, self_conditioning_logits, _ = DiffusionGemmaGenerationMixin._denoising_step(
        fake_self,
        lambda **kwargs: _DecoderOutput(raw_logits),
        current_canvas=torch.zeros(batch, length, dtype=torch.long),
        argmax_canvas=torch.zeros(batch, length, dtype=torch.long),
        input_ids=torch.zeros(batch, length, dtype=torch.long),
        decoder_position_ids=torch.arange(length).unsqueeze(0),
        self_conditioning_logits=None,
        mask_mapping={},
        past_key_values=None,
        finished_denoising=torch.zeros(batch, dtype=torch.bool),
        cur_step=cur_step,
        sampler=sampler,
        logits_processor=processor,
        diffusion_stopping_criteria=stopping,
    )

    assert torch.allclose(sampler.accept_logits, expected_processed)
    assert torch.allclose(stopping.logits, expected_processed)
    assert torch.equal(stopping.argmax_canvas, expected_processed.argmax(dim=-1))
    assert torch.allclose(self_conditioning_logits.float(), expected_processed.to(torch.bfloat16).float())


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


def test_decoder_soft_embedding_matches_reference_scale_and_mask():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderModel

    class _CaptureSelfConditioning(nn.Module):
        def __init__(self):
            super().__init__()
            self.signal = None

        def forward(self, inputs_embeds, self_conditioning_signal):
            self.signal = self_conditioning_signal.detach().clone()
            return inputs_embeds

    batch, length, vocab, hidden = 2, 4, 11, 4
    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        layer_types=[],
        rms_norm_eps=1e-6,
    )
    real = DiffusionGemmaDecoderModel(DiffusionGemmaConfig(text_config=text_cfg)).eval()
    capture = _CaptureSelfConditioning()
    real.self_conditioning = capture

    logits = torch.randn(batch, length, vocab, generator=_gen(12))
    mask = torch.tensor([True, False])
    real(
        decoder_input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        self_conditioning_logits=logits,
        self_conditioning_mask=mask,
        decoder_attention_mask={},
        decoder_position_ids=torch.arange(length).unsqueeze(0).expand(batch, -1),
        past_key_values=None,
    )

    ours = SelfConditioning.soft_embedding(logits, real.embed_tokens.weight, mask=mask)
    assert torch.allclose(capture.signal, ours, atol=1e-6)
    assert torch.all(capture.signal[1] == 0)
