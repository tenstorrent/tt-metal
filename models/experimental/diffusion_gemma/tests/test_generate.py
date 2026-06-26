# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the block-autoregressive multi-canvas loop (#47464)."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import generate as RG
from models.experimental.diffusion_gemma.reference.generate import generate_blocks, make_replay_canvas_init_fn


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(canvas_length):
    return DiffusionConfig(
        canvas_length=canvas_length, max_denoise_steps=6, entropy_stop_threshold=0.1, stable_steps_to_halt=1
    )


class _PrefixDependentModel:
    """Peaked logits whose target token depends on how long the prefix is.

    target = prefix_len // canvas_len, so each committed block is predictable and
    we can verify the prefix actually grows (commit-append) between blocks.
    """

    def __init__(self, batch, canvas_len, vocab):
        self.batch, self.canvas_len, self.vocab = batch, canvas_len, vocab

    def __call__(self, prefix, canvas, step):
        target = (prefix.shape[1] // self.canvas_len) % self.vocab
        logits = torch.full((self.batch, self.canvas_len, self.vocab), -1e4)
        logits[..., target] = 1e4
        return logits


class _ToyCache:
    is_compileable = False
    max_cache_len = 0

    def get_seq_length(self):
        return 0


class _ToyEncoder:
    def __init__(self):
        self.total_encoded = 0

    def create_masks_for_generate(self, **kwargs):
        return {}

    def __call__(self, input_ids, **kwargs):
        self.total_encoded += input_ids.shape[1]
        return SimpleNamespace(past_key_values=kwargs["past_key_values"])


class _ToyDecoder:
    def __init__(self):
        self.embed_tokens = SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))

    def create_diffusion_decoder_attention_mask(self, **kwargs):
        return {}


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.diffusion_gemma") is None,
    reason="transformers.models.diffusion_gemma not installed (ships since transformers 5.12)",
)
def test_reference_generate_blocks_matches_hf_generate_outer_loop():
    """CPU #47464 acceptance: HF generate() and our reference commit the same blocks.

    The fake model keeps HF's generation mixin, sampler, denoise step, and
    commit-append loop, but replaces the 26B backbone with peaked logits whose
    target depends on the encoded prefix length. If HF or our reference stop
    advancing the prefix between blocks, the second and third committed blocks
    diverge.
    """

    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig,
        DiffusionGemmaGenerationMixin,
        EntropyBoundSamplerConfig,
    )

    class _ToyHFGenerateModel(DiffusionGemmaGenerationMixin):
        def __init__(self, canvas_len, vocab):
            self.dtype = torch.float32
            self.config = SimpleNamespace(
                canvas_length=canvas_len,
                text_config=SimpleNamespace(vocab_size=vocab),
                image_token_id=-1,
            )
            self.generation_config = DiffusionGemmaGenerationConfig(
                max_new_tokens=canvas_len,
                max_denoising_steps=1,
                sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
                t_min=0.4,
                t_max=0.8,
                stability_threshold=1,
                confidence_threshold=0.005,
                pad_token_id=None,
                eos_token_id=None,
                cache_implementation=None,
            )
            self.model = SimpleNamespace(encoder=_ToyEncoder(), decoder=_ToyDecoder())

        def forward(self, decoder_input_ids, **kwargs):
            batch, canvas_len = decoder_input_ids.shape
            target = (self.model.encoder.total_encoded // canvas_len) % self.config.text_config.vocab_size
            logits = torch.full((batch, canvas_len, self.config.text_config.vocab_size), -1e4)
            logits[..., target] = 1e4
            return SimpleNamespace(logits=logits)

    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)

    ref = generate_blocks(
        _PrefixDependentModel(batch, canvas_len, vocab),
        prompt,
        blocks,
        DiffusionConfig(
            canvas_length=canvas_len,
            max_denoise_steps=1,
            entropy_stop_threshold=0.005,
            stable_steps_to_halt=1,
        ),
        vocab,
        generator=_gen(4),
    )
    hf = _ToyHFGenerateModel(canvas_len, vocab)
    hf_out = hf.generate(prompt, past_key_values=_ToyCache(), max_new_tokens=blocks * canvas_len)
    hf_generated = hf_out.sequences[:, prompt.shape[1] :]

    assert torch.equal(hf_generated, ref.generated)
    for b in range(blocks):
        block_tokens = hf_generated[:, b * canvas_len : (b + 1) * canvas_len]
        assert torch.all(block_tokens == 2 + b)


def test_generates_num_blocks_times_canvas_tokens():
    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)  # prompt_len = 2*canvas_len
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    out = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(1))

    assert out.generated.shape == (batch, blocks * canvas_len)
    assert out.prompt_len == 2 * canvas_len
    assert len(out.trajectories) == blocks


def test_prefix_grows_so_committed_targets_advance():
    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)  # prompt_len//canvas_len == 2
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    out = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(2))

    # block b sees prefix_len = (2 + b) * canvas_len -> target token = 2 + b
    for b in range(blocks):
        block_tokens = out.generated[:, b * canvas_len : (b + 1) * canvas_len]
        assert torch.all(block_tokens == 2 + b)


def test_generation_is_deterministic_under_fixed_generator():
    batch, canvas_len, vocab, blocks = 2, 4, 16, 2
    prompt = torch.zeros(batch, canvas_len, dtype=torch.long)
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    a = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(3))
    b = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(3))
    assert torch.equal(a.generated, b.generated)


@pytest.mark.parametrize(
    "prompt,num_blocks,config,vocab,match",
    [
        (torch.zeros(4, dtype=torch.long), 1, _cfg(4), 16, "prompt_tokens must have shape"),
        (torch.zeros(1, 4, dtype=torch.long), -1, _cfg(4), 16, "num_blocks must be non-negative"),
        (torch.zeros(1, 4, dtype=torch.long), 1, _cfg(0), 16, "canvas_length must be positive"),
        (torch.zeros(1, 4, dtype=torch.long), 1, _cfg(4), 0, "vocab_size must be positive"),
    ],
)
def test_generate_blocks_rejects_bad_arguments(prompt, num_blocks, config, vocab, match):
    model = _PrefixDependentModel(1, max(config.canvas_length, 1), max(vocab, 1))

    with pytest.raises(ValueError, match=match):
        generate_blocks(model, prompt, num_blocks, config, vocab, generator=_gen(3))


def test_can_replay_fixed_initial_canvases(monkeypatch):
    batch, canvas_len, vocab, blocks = 1, 4, 16, 2
    prompt = torch.zeros(batch, canvas_len, dtype=torch.long)
    model = _PrefixDependentModel(batch, canvas_len, vocab)
    canvases = [
        torch.full((batch, canvas_len), 7, dtype=torch.long),
        torch.full((batch, canvas_len), 8, dtype=torch.long),
    ]
    calls = []

    def fail_random_canvas(*args, **kwargs):
        raise AssertionError("fixed init_canvas_fn should bypass random canvas generation")

    def init_canvas_fn(block_idx, prefix_tokens):
        calls.append((block_idx, prefix_tokens.clone()))
        return canvases[block_idx].clone()

    monkeypatch.setattr(RG.S, "random_canvas", fail_random_canvas)

    out = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, init_canvas_fn=init_canvas_fn)

    assert out.prompt_len == canvas_len
    assert len(out.trajectories) == blocks
    assert [call[0] for call in calls] == [0, 1]
    assert torch.equal(calls[0][1], prompt)
    assert torch.equal(calls[1][1], torch.cat([prompt, out.trajectories[0].committed], dim=1))


def test_replay_canvas_init_fn_clones_fixed_canvases():
    canvases = [torch.tensor([[4, 5]], dtype=torch.long), torch.tensor([[6, 7]], dtype=torch.long)]
    init_canvas_fn = make_replay_canvas_init_fn(canvases)
    canvases[0][0, 0] = 99

    first = init_canvas_fn(0, torch.zeros(1, 2, dtype=torch.long))
    first[0, 1] = 88
    second_read = init_canvas_fn(0, torch.zeros(1, 2, dtype=torch.long))

    assert torch.equal(first, torch.tensor([[4, 88]], dtype=torch.long))
    assert torch.equal(second_read, torch.tensor([[4, 5]], dtype=torch.long))
    assert torch.equal(init_canvas_fn(1, torch.zeros(1, 2, dtype=torch.long)), torch.tensor([[6, 7]]))


@pytest.mark.parametrize(
    "bad_canvas",
    [
        torch.zeros(4, dtype=torch.long),
        torch.zeros(1, 5, dtype=torch.long),
    ],
)
def test_replayed_initial_canvas_shape_is_validated(bad_canvas):
    batch, canvas_len, vocab, blocks = 1, 4, 16, 1
    prompt = torch.zeros(batch, canvas_len, dtype=torch.long)
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    with pytest.raises(ValueError, match="init_canvas_fn must return shape"):
        generate_blocks(
            model,
            prompt,
            blocks,
            _cfg(canvas_len),
            vocab,
            init_canvas_fn=lambda block_idx, prefix_tokens: bad_canvas,
        )
