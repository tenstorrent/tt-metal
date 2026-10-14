# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical real-HF parity, stage-gate, and trajectory-harness drift guards."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.hf_reference import run_reference_trajectory
from models.experimental.diffusion_gemma.reference.replay_hf_tt import (
    _validate_stage_gate_args,
    build_arg_parser,
)
from models.experimental.diffusion_gemma.reference.self_conditioning import DiffusionGemmaRMSNorm, SelfConditioning
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories

_HAS_HF_DIFFUSION_GEMMA = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("transformers.models.diffusion_gemma") is not None
)
_requires_hf_diffusion_gemma = pytest.mark.skipif(
    not _HAS_HF_DIFFUSION_GEMMA,
    reason="transformers.models.diffusion_gemma not installed (ships since transformers 5.12)",
)


def _gen(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _cfg(**kwargs):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kwargs)


class _MockCanvasModel:
    def __init__(self, batch, length, vocab, seed, drift=0.0):
        self._logits = torch.randn(batch, length, vocab, generator=_gen(seed))
        self._drift = drift

    def __call__(self, canvas, **kwargs):
        output = self._logits.clone()
        if self._drift:
            output[..., 0] += self._drift
        return output


def test_harness_rejects_drifted_candidate():
    batch, length, vocab = 1, 8, 32
    init = S.random_canvas((batch, length), vocab, generator=_gen(3))

    def gumbel_fn(step):
        return S.sample_gumbel_noise((batch, length, vocab), generator=_gen(500 + step))

    def noise_fn(step):
        return torch.randint(0, vocab, (batch, length), generator=_gen(600 + step))

    def run(drift):
        return run_reference_trajectory(
            _MockCanvasModel(batch, length, vocab, seed=5, drift=drift),
            init.clone(),
            _cfg(),
            vocab,
            gumbel_noise_fn=gumbel_fn,
            noise_tokens_fn=noise_fn,
        )

    reference = run(0.0)
    assert compare_trajectories(reference, run(0.0)).passed
    comparison = compare_trajectories(reference, run(5.0))
    assert not comparison.passed
    assert comparison.min_argmax_agreement < 1.0


@_requires_hf_diffusion_gemma
def test_entropy_accept_matches_real_EntropyBoundSampler():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )

    for seed, bound in [(2, 0.05), (3, 0.1), (4, 0.5), (5, 2.0)]:
        logits = torch.randn(1, 64, 128, generator=_gen(seed))
        sampler = EntropyBoundSampler(
            EntropyBoundSamplerConfig(entropy_bound=bound),
            canvas_length=64,
            vocab_size=128,
            max_denoising_steps=48,
        )
        sampler.accept_canvas(
            torch.zeros(1, 64, dtype=torch.long),
            torch.ones(1, 64, dtype=torch.long),
            logits,
            1,
        )
        ours = S.entropy_budget_accept(S.token_entropy(logits), bound, min_accept=0)
        assert torch.equal(ours, sampler.accepted_token_mask.bool())


@_requires_hf_diffusion_gemma
def test_temperature_matches_real_LinearSchedule():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        LinearTemperatureScheduleLogitsProcessor,
    )

    steps, t_min, t_max = 48, 0.4, 0.8
    processor = LinearTemperatureScheduleLogitsProcessor(
        t_min=t_min,
        t_max=t_max,
        max_denoising_steps=steps,
    )
    scores = torch.ones(1, 10)
    for current_step in range(1, steps + 1):
        output = processor(None, scores.clone(), current_step)
        real_temperature = float(scores[0, 0] / output[0, 0])
        ours = S.temperature_at_step(steps - current_step, steps, t_max, t_min)
        assert abs(real_temperature - ours) < 1e-5


@_requires_hf_diffusion_gemma
def test_stopping_confidence_matches_real_criterion():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import StableAndConfidentStoppingCriteria

    logits = torch.randn(4, 32, 200, generator=_gen(6))
    threshold = float(S.token_entropy(logits).mean(dim=-1).median())
    real = StableAndConfidentStoppingCriteria(stability_threshold=0, confidence_threshold=threshold)
    assert torch.equal(S.token_entropy(logits).mean(dim=-1) < threshold, real(logits.argmax(dim=-1), logits))


@_requires_hf_diffusion_gemma
def test_real_denoising_step_uses_temperature_processed_logits_for_decisions():
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

        def accept_canvas(self, current_canvas, denoiser_canvas, logits, current_step):
            self.accept_logits = logits.detach().clone()
            return denoiser_canvas

        def renoise_canvas(self, accepted_canvas, current_step):
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
    current_step = 2
    processor = LogitsProcessorList(
        [LinearTemperatureScheduleLogitsProcessor(t_min=0.4, t_max=0.8, max_denoising_steps=4)]
    )
    expected = processor(None, raw_logits, cur_step=torch.tensor(current_step, dtype=torch.int32))
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
        cur_step=current_step,
        sampler=sampler,
        logits_processor=processor,
        diffusion_stopping_criteria=stopping,
    )

    assert torch.allclose(sampler.accept_logits, expected)
    assert torch.allclose(stopping.logits, expected)
    assert torch.equal(stopping.argmax_canvas, expected.argmax(dim=-1))
    assert torch.allclose(self_conditioning_logits.float(), expected.to(torch.bfloat16).float())


@_requires_hf_diffusion_gemma
def test_rmsnorm_matches_real():
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaRMSNorm as RealRMS

    inputs = torch.randn(2, 3, 16, generator=_gen(9))
    for with_scale in (True, False):
        real = RealRMS(16, eps=1e-6, with_scale=with_scale)
        ours = DiffusionGemmaRMSNorm(16, eps=1e-6, with_scale=with_scale)
        if with_scale:
            with torch.no_grad():
                real.weight.copy_(torch.randn(16, generator=_gen(3)))
                ours.weight.copy_(real.weight)
        assert torch.allclose(ours(inputs), real(inputs), atol=1e-6)


@_requires_hf_diffusion_gemma
def test_self_conditioning_matches_real():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaSelfConditioning

    hidden, intermediate = 16, 40
    config = DiffusionGemmaTextConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
    )
    real = DiffusionGemmaSelfConditioning(config).eval()
    ours = SelfConditioning(
        hidden,
        intermediate_size=intermediate,
        eps=1e-6,
        activation="gelu_pytorch_tanh",
    ).eval()
    with torch.no_grad():
        ours.pre_norm.weight.copy_(real.pre_norm.weight)
        ours.gate_proj.weight.copy_(real.gate_proj.weight)
        ours.up_proj.weight.copy_(real.up_proj.weight)
        ours.down_proj.weight.copy_(real.down_proj.weight)
    embeddings = torch.randn(2, 5, hidden, generator=_gen(8))
    signal = torch.randn(2, 5, hidden, generator=_gen(9))
    assert torch.allclose(ours(embeddings, signal), real(embeddings, signal), atol=1e-5)


@_requires_hf_diffusion_gemma
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
    text_config = DiffusionGemmaTextConfig(
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
    real = DiffusionGemmaDecoderModel(DiffusionGemmaConfig(text_config=text_config)).eval()
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


def test_stage_gate_requires_canonical_production_replay(expect_error):
    args = build_arg_parser().parse_args(["--stage-gate", "--noise-mode", "seeded", "--max-denoising-steps", "8"])
    _validate_stage_gate_args(args)
    args.max_denoising_steps = 1
    with expect_error(ValueError, match="max-denoising-steps 8"):
        _validate_stage_gate_args(args)
