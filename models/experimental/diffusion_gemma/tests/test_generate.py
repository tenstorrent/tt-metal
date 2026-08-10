# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the reference and device block-autoregressive generation loops (#47464)."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import generate as RG
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.reference.generate import (
    generate_blocks as reference_generate_blocks,
    make_replay_canvas_init_fn,
    make_replay_noise_fn,
)
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import (
    GeneratedBlock,
    PromptPrefill,
    commit_canvas_tokens,
    decode_generation,
    denoise_and_commit_block,
    generate_blocks,
    generate_from_prompt_tokens,
    generate_text,
    generate_text_from_checkpoint_state,
    generation_sequences,
    generation_token_ids,
    host_canvas_to_device,
    host_gumbel_noise_to_device,
    host_tokens_to_device,
    make_host_gumbel_noise_fn,
    make_host_noise_tokens_fn,
    make_seeded_gumbel_noise_fn,
    make_host_canvas_init_fn,
    make_seeded_host_canvas_init_fn,
    make_seeded_host_noise_tokens_fn,
    prefill_prompt_tokens,
    tokenize_prompt,
)

INT32_MAX = torch.iinfo(torch.int32).max


def _boom(*args, **kwargs):
    raise AssertionError("arguments must be validated before any tokenizer, model or device work")


# --- reference block loop -------------------------------------------------------------------


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

    ref = reference_generate_blocks(
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

    out = reference_generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(1))

    assert out.generated.shape == (batch, blocks * canvas_len)
    assert out.prompt_len == 2 * canvas_len
    assert len(out.trajectories) == blocks


def test_prefix_grows_so_committed_targets_advance():
    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)  # prompt_len//canvas_len == 2
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    out = reference_generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(2))

    # block b sees prefix_len = (2 + b) * canvas_len -> target token = 2 + b
    for b in range(blocks):
        block_tokens = out.generated[:, b * canvas_len : (b + 1) * canvas_len]
        assert torch.all(block_tokens == 2 + b)


def _ref_blocks(*, prompt=None, num_blocks=1, config=None, vocab=16, **kwargs):
    config = _cfg(4) if config is None else config
    prompt = torch.zeros(1, 4, dtype=torch.long) if prompt is None else prompt
    model = _PrefixDependentModel(1, max(config.canvas_length, 1), max(vocab, 1))
    return reference_generate_blocks(model, prompt, num_blocks, config, vocab, generator=_gen(3), **kwargs)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda: _ref_blocks(prompt=torch.zeros(4, dtype=torch.long)),
            "prompt_tokens must have shape",
            id="prompt-not-2d",
        ),
        pytest.param(lambda: _ref_blocks(num_blocks=-1), "num_blocks must be non-negative", id="negative-num-blocks"),
        pytest.param(lambda: _ref_blocks(config=_cfg(0)), "canvas_length must be positive", id="empty-canvas"),
        pytest.param(lambda: _ref_blocks(vocab=0), "vocab_size must be positive", id="empty-vocab"),
        pytest.param(
            lambda: _ref_blocks(init_canvas_fn=lambda block_idx, prefix_tokens: torch.zeros(4, dtype=torch.long)),
            "init_canvas_fn must return shape",
            id="init-canvas-not-2d",
        ),
        pytest.param(
            lambda: _ref_blocks(init_canvas_fn=lambda block_idx, prefix_tokens: torch.zeros(1, 5, dtype=torch.long)),
            "init_canvas_fn must return shape",
            id="init-canvas-wrong-length",
        ),
    ],
)
def test_reference_generate_blocks_rejects_bad_arguments(call, message, expect_error):
    with expect_error(ValueError, match=message):
        call()


# --- reference replay hooks -----------------------------------------------------------------


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

    out = reference_generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, init_canvas_fn=init_canvas_fn)

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


def test_replay_noise_fn_clones_fixed_block_step_noise():
    noise = [
        [torch.tensor([[1, 2]], dtype=torch.long), torch.tensor([[3, 4]], dtype=torch.long)],
        [torch.tensor([[5, 6]], dtype=torch.long)],
    ]
    noise_fn = make_replay_noise_fn(noise)
    noise[0][0][0, 0] = 99

    first = noise_fn(0)(0)
    first[0, 1] = 88

    assert torch.equal(first, torch.tensor([[1, 88]], dtype=torch.long))
    assert torch.equal(noise_fn(0)(0), torch.tensor([[1, 2]], dtype=torch.long))
    assert torch.equal(noise_fn(0)(1), torch.tensor([[3, 4]], dtype=torch.long))
    assert torch.equal(noise_fn(1)(0), torch.tensor([[5, 6]], dtype=torch.long))


def test_reference_generate_blocks_replays_fixed_denoise_noise():
    batch, canvas_len, vocab, blocks = 1, 3, 5, 1
    prompt = torch.zeros(batch, canvas_len, dtype=torch.long)
    init_canvas_fn = make_replay_canvas_init_fn([torch.tensor([[4, 4, 4]], dtype=torch.long)])
    gumbel_noise = [
        [
            torch.tensor([[[0.0, 9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 0.0, 7.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 6.0], [5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0]]]),
        ]
    ]
    noise_tokens = [[torch.tensor([[9, 8, 7]], dtype=torch.long), torch.tensor([[6, 5, 4]], dtype=torch.long)]]

    out = reference_generate_blocks(
        lambda prefix, canvas, step: torch.zeros(batch, canvas_len, vocab),
        prompt,
        blocks,
        DiffusionConfig(
            canvas_length=canvas_len,
            max_denoise_steps=2,
            entropy_budget=-1.0,
            entropy_stop_threshold=-1.0,
            stable_steps_to_halt=1,
        ),
        vocab,
        sampler=RG.S.SAMPLER_GUMBEL,
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=make_replay_noise_fn(gumbel_noise),
        noise_tokens_fn=make_replay_noise_fn(noise_tokens),
    )

    steps = out.trajectories[0].per_step
    assert torch.equal(steps[0].sampled, torch.tensor([[1, 2, 3]]))
    assert torch.equal(steps[0].canvas, torch.tensor([[9, 8, 7]]))
    assert torch.equal(steps[1].sampled, torch.tensor([[4, 0, 1]]))
    assert torch.equal(steps[1].canvas, torch.tensor([[6, 5, 4]]))


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        pytest.param(
            lambda: make_replay_canvas_init_fn([torch.tensor([[4, 5]], dtype=torch.long)])(
                1, torch.zeros(1, 2, dtype=torch.long)
            ),
            IndexError,
            "replay canvas block index 1 out of range",
            id="canvas-replay-block-out-of-range",
        ),
        pytest.param(
            lambda: make_replay_canvas_init_fn(
                [
                    torch.tensor([[4, 5]], dtype=torch.long),
                    torch.tensor([[6, 7, 8]], dtype=torch.long),
                ]
            ),
            ValueError,
            "replay canvas must all have shape",
            id="canvas-replay-mismatched-shapes",
        ),
        pytest.param(
            lambda: make_replay_noise_fn([[torch.tensor([[1, 2]], dtype=torch.long)]])(1),
            IndexError,
            "replay noise block index 1 out of range",
            id="noise-replay-block-out-of-range",
        ),
        pytest.param(
            lambda: make_replay_noise_fn([[torch.tensor([[1, 2]], dtype=torch.long)]])(0)(1),
            IndexError,
            "replay noise step index 1 out of range",
            id="noise-replay-step-out-of-range",
        ),
        pytest.param(
            lambda: make_replay_noise_fn(
                [[torch.tensor([[1, 2]], dtype=torch.long), torch.tensor([[3, 4, 5]], dtype=torch.long)]]
            ),
            ValueError,
            "replay noise must all have shape",
            id="noise-replay-ragged-within-block",
        ),
        pytest.param(
            lambda: make_replay_noise_fn(
                [[torch.tensor([[1, 2]], dtype=torch.long)], [torch.tensor([[3, 4, 5]], dtype=torch.long)]]
            ),
            ValueError,
            "replay noise must all have shape",
            id="noise-replay-ragged-across-blocks",
        ),
    ],
)
def test_reference_replay_hook_factories_reject_bad_inputs(call, error, message, expect_error):
    with expect_error(error, match=message):
        call()


# --- device denoise-and-commit block --------------------------------------------------------


class _FakeLogitsFn:
    q_rope_offset = None


def test_denoise_and_commit_block_threads_position_and_commits():
    calls = {}
    committed = torch.tensor([[7, 8, 9]], dtype=torch.long)
    trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])

    def fake_denoise_block(logits_fn, init_canvas, config, *, gumbel_noise_fn=None, noise_tokens_fn=None):
        calls["denoise"] = (logits_fn, init_canvas, config, gumbel_noise_fn, noise_tokens_fn)
        return trajectory

    def fake_commit(tt_model, canvas_tokens, *, start_pos, page_table=None, page_tables_per_layer=None):
        calls["commit"] = (tt_model, canvas_tokens, start_pos, page_table, page_tables_per_layer)

    logits_fn = _FakeLogitsFn()
    logits_fn.advance_prefix_after_commit = lambda next_pos: calls.setdefault("advance_prefix", next_pos)
    config = DiffusionConfig(canvas_length=3)
    gumbel_noise_fn = object()
    noise_tokens_fn = object()
    page_tables_per_layer = ["pages"]
    timings = {}

    out = denoise_and_commit_block(
        "model",
        logits_fn,
        "init-canvas",
        config,
        start_pos=32 + 2 * 256,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_tables_per_layer=page_tables_per_layer,
        denoise_block_fn=fake_denoise_block,
        commit_fn=fake_commit,
        timings=timings,
    )

    assert logits_fn.q_rope_offset == 544
    assert calls["denoise"] == (logits_fn, "init-canvas", config, gumbel_noise_fn, noise_tokens_fn)
    assert calls["commit"] == ("model", committed, 544, None, page_tables_per_layer)
    assert calls["advance_prefix"] == 547
    assert out.committed is committed
    assert out.next_pos == 547
    assert out.trajectory is trajectory
    assert timings.keys() == {"denoise_s", "commit_s"}
    assert timings["denoise_s"] >= 0
    assert timings["commit_s"] >= 0


@pytest.mark.parametrize(
    ("committed", "error", "message"),
    [
        (None, RuntimeError, "did not produce committed"),
        (torch.tensor([[7, 8]], dtype=torch.long), ValueError, "block.committed"),
    ],
)
def test_denoise_and_commit_block_rejects_bad_trajectory(committed, error, message, expect_error):
    trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])

    with expect_error(error, match=message):
        denoise_and_commit_block(
            object(),
            object(),
            object(),
            DiffusionConfig(canvas_length=3),
            start_pos=0,
            denoise_block_fn=lambda *args, **kwargs: trajectory,
            commit_fn=_boom,
        )


def test_commit_canvas_tokens_uses_diffusion_local_commit_decode(monkeypatch):
    class _FakeLogits:
        deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _Model:
        def __init__(self):
            self.prepared = []

        def prepare_inputs_decode(self, token, position, page_table=None):
            self.prepared.append((token.clone(), position.clone(), page_table))
            return ("x", "current-pos", "rot-idxs", "page-table")

        def ttnn_decode_forward(self, *args, **kwargs):
            raise AssertionError("commit should not call shared Gemma4 decode")

    calls = []
    logits = []

    def fake_commit_decode_forward(tt_model, *args, page_tables_per_layer=None, **kwargs):
        logit = _FakeLogits()
        logits.append(logit)
        calls.append((tt_model, args, page_tables_per_layer, kwargs))
        return logit, None

    monkeypatch.setattr(G, "commit_decode_forward", fake_commit_decode_forward)
    model = _Model()
    page_tables_per_layer = ["layer-pages"]

    commit_canvas_tokens(
        model,
        torch.tensor([[7, 8]], dtype=torch.long),
        start_pos=5,
        page_table="page",
        page_tables_per_layer=page_tables_per_layer,
    )

    assert [token.item() for token, _, _ in model.prepared] == [7, 8]
    assert [position.item() for _, position, _ in model.prepared] == [5, 6]
    assert [page_table for _, _, page_table in model.prepared] == ["page", "page"]
    assert [call[0] for call in calls] == [model, model]
    assert [call[1] for call in calls] == [
        ("x", "current-pos", "rot-idxs", "page-table"),
        ("x", "current-pos", "rot-idxs", "page-table"),
    ]
    assert [call[2] for call in calls] == [page_tables_per_layer, page_tables_per_layer]
    assert all(logit.deallocated for logit in logits)


# --- device generate_blocks -----------------------------------------------------------------


def test_generate_blocks_advances_position_and_concatenates_commits():
    calls = []

    def init_canvas_fn(block_idx, start_pos):
        calls.append(("init", block_idx, start_pos))
        return f"canvas-{block_idx}"

    def noise_factory(kind):
        def outer(block_idx):
            calls.append((kind, block_idx))
            return f"{kind}-{block_idx}"

        return outer

    def fake_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        block_idx = len([call for call in calls if call[0] == "block"])
        committed = torch.full((1, config.canvas_length), block_idx, dtype=torch.long)
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        calls.append(("block", init_canvas, kwargs["start_pos"], kwargs["gumbel_noise_fn"], kwargs["noise_tokens_fn"]))
        return GeneratedBlock(
            committed=committed, next_pos=kwargs["start_pos"] + config.canvas_length, trajectory=trajectory
        )

    out = generate_blocks(
        "model",
        "logits",
        prompt_len=32,
        num_blocks=3,
        config=DiffusionConfig(canvas_length=3),
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=noise_factory("gumbel"),
        noise_tokens_fn=noise_factory("noise"),
        block_fn=fake_block,
    )

    assert out.prompt_len == 32
    assert out.next_pos == 41
    assert torch.equal(out.generated, torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]]))
    assert len(out.trajectories) == 3
    assert [call for call in calls if call[0] == "init"] == [
        ("init", 0, 32),
        ("init", 1, 35),
        ("init", 2, 38),
    ]
    assert [call for call in calls if call[0] == "block"] == [
        ("block", "canvas-0", 32, "gumbel-0", "noise-0"),
        ("block", "canvas-1", 35, "gumbel-1", "noise-1"),
        ("block", "canvas-2", 38, "gumbel-2", "noise-2"),
    ]


def test_generate_blocks_deallocates_init_canvas_if_block_fails(expect_error):
    class _FakeDeviceCanvas:
        deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    init_canvas = _FakeDeviceCanvas()

    def failing_block(*args, **kwargs):
        raise RuntimeError("device op failed")

    with expect_error(RuntimeError, match="device op failed"):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=3),
            init_canvas_fn=lambda *args: init_canvas,
            block_fn=failing_block,
        )

    assert init_canvas.deallocated is True


def test_generate_blocks_failure_releases_upfront_controller_and_adapter_state(expect_error):
    events = []

    def fail(name):
        def _raise():
            events.append(name)
            raise RuntimeError(f"injected {name} cleanup failure")

        return _raise

    class _Logits:
        def __init__(self):
            self._upfront_traced_denoise_controller = SimpleNamespace(release=fail("controller"))

        def reset(self):
            events.append("adapter-reset")

    logits = _Logits()

    with expect_error(RuntimeError, match="injected replay failure"):
        generate_blocks(
            "model",
            logits,
            prompt_len=32,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=3),
            init_canvas_fn=lambda *args: "canvas",
            block_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected replay failure")),
        )

    assert events == ["controller", "adapter-reset"]
    assert not hasattr(logits, "_upfront_traced_denoise_controller")


def test_generate_blocks_allows_zero_blocks_without_init_canvas():
    out = generate_blocks(
        "model",
        "logits",
        prompt_len=32,
        num_blocks=0,
        config=DiffusionConfig(canvas_length=4),
        batch_size=2,
    )

    assert out.prompt_len == 32
    assert out.next_pos == 32
    assert out.trajectories == []
    assert torch.equal(out.generated, torch.empty((2, 0), dtype=torch.long))


def test_generate_blocks_stops_after_committed_stop_token():
    calls = []

    def init_canvas_fn(block_idx, start_pos):
        calls.append(("init", block_idx, start_pos))
        return f"canvas-{block_idx}"

    def fake_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        block_idx = len([call for call in calls if call[0] == "block"])
        committed = torch.tensor([[block_idx, 9 if block_idx == 1 else block_idx]], dtype=torch.long)
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        calls.append(("block", init_canvas, kwargs["start_pos"]))
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + committed.shape[1],
            trajectory=trajectory,
        )

    out = generate_blocks(
        "model",
        "logits",
        prompt_len=4,
        num_blocks=4,
        config=DiffusionConfig(canvas_length=2),
        init_canvas_fn=init_canvas_fn,
        stop_token_ids=9,
        block_fn=fake_block,
    )

    assert torch.equal(out.generated, torch.tensor([[0, 0, 1, 9]], dtype=torch.long))
    assert out.next_pos == 8
    assert len(out.trajectories) == 2
    assert [call for call in calls if call[0] == "init"] == [("init", 0, 4), ("init", 1, 6)]


@pytest.mark.parametrize(
    ("committed", "next_pos_delta", "message"),
    [
        (torch.tensor([7, 7, 7, 7], dtype=torch.long), 0, "block.committed"),
        (torch.tensor([[7, 7, 7]], dtype=torch.long), 0, "block.committed"),
        (torch.tensor([[7, 7, 7, 7], [8, 8, 8, 8]], dtype=torch.long), 0, "block.committed"),
        (torch.tensor([[1.5, 2.0, 3.0, 4.0]], dtype=torch.float32), 0, "integer token ids"),
        (torch.tensor([[1, -2, 3, 4]], dtype=torch.long), 0, "non-negative"),
        (torch.tensor([[1, INT32_MAX + 1, 3, 4]], dtype=torch.long), 0, "fit int32"),
        (torch.full((1, 4), 7, dtype=torch.long), 1, "block.next_pos"),
    ],
)
def test_generate_blocks_rejects_malformed_block_output(committed, next_pos_delta, message, expect_error):
    def bad_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + config.canvas_length + next_pos_delta,
            trajectory=trajectory,
        )

    with expect_error(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn=lambda *args: "canvas",
            block_fn=bad_block,
        )


@pytest.mark.parametrize(
    ("stop_token_ids", "message"),
    [
        ("9", "stop_token_ids"),
        ([9, -1], "non-negative"),
        ([9, INT32_MAX + 1], "fit int32"),
    ],
)
def test_generate_blocks_rejects_invalid_stop_token_ids(stop_token_ids, message, expect_error):
    committed = torch.tensor([[1, 2]], dtype=torch.long)

    def fake_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + committed.shape[1],
            trajectory=trajectory,
        )

    with expect_error(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=4,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn=lambda *args: "canvas",
            stop_token_ids=stop_token_ids,
            block_fn=fake_block,
        )


# --- generate_from_prompt_tokens ------------------------------------------------------------


def test_generate_from_prompt_tokens_prefills_then_runs_blocks():
    calls = []
    prompt_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    config = DiffusionConfig(canvas_length=3)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[7, 8, 9]], dtype=torch.long),
        prompt_len=prompt_tokens.shape[1],
        next_pos=prompt_tokens.shape[1] + config.canvas_length,
        trajectories=[],
    )

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        calls.append(("prefill", tt_model, tokens, page_table, page_tables_per_layer))
        return tokens.shape[1]

    def fake_blocks(tt_model, logits_fn, **kwargs):
        calls.append(("blocks", tt_model, logits_fn, kwargs))
        return generation

    init_canvas_fn = object()
    gumbel_noise_fn = object()
    noise_tokens_fn = object()
    out = generate_from_prompt_tokens(
        "model",
        "logits",
        prompt_tokens,
        num_blocks=2,
        config=config,
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_table="page-table",
        page_tables_per_layer=["layer-pages"],
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out is generation
    assert calls[0] == ("prefill", "model", prompt_tokens, "page-table", ["layer-pages"])
    assert calls[1][0:3] == ("blocks", "model", "logits")
    kwargs = calls[1][3]
    assert kwargs["prompt_len"] == prompt_tokens.shape[1]
    assert kwargs["num_blocks"] == 2
    assert kwargs["config"] is config
    assert kwargs["init_canvas_fn"] is init_canvas_fn
    assert kwargs["gumbel_noise_fn"] is gumbel_noise_fn
    assert kwargs["noise_tokens_fn"] is noise_tokens_fn
    assert kwargs["page_table"] == "page-table"
    assert kwargs["page_tables_per_layer"] == ["layer-pages"]
    assert kwargs["stop_token_ids"] is None


def test_generate_from_prompt_tokens_can_build_logits_after_prefill():
    calls = []
    prompt_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[7, 8]], dtype=torch.long),
        prompt_len=4,
        next_pos=6,
        trajectories=[],
    )

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        calls.append(("prefill", tt_model, tokens.clone(), page_table, page_tables_per_layer))
        return tokens.shape[1]

    def fake_builder(tt_model, **kwargs):
        calls.append(("builder", tt_model, kwargs))
        return "built-logits"

    def fake_blocks(tt_model, logits_fn, **kwargs):
        calls.append(("blocks", tt_model, logits_fn, kwargs))
        return generation

    out = generate_from_prompt_tokens(
        "model",
        None,
        prompt_tokens,
        num_blocks=1,
        config=DiffusionConfig(canvas_length=2),
        init_canvas_fn="init",
        page_table="page-table",
        page_tables_per_layer=["layer-pages"],
        logits_fn_builder=fake_builder,
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out is generation
    assert calls[0][0] == "prefill"
    assert calls[1][0:2] == ("builder", "model")
    builder_kwargs = calls[1][2]
    assert builder_kwargs["prompt_tokens"] is prompt_tokens
    assert builder_kwargs["prompt_len"] == prompt_tokens.shape[1]
    assert builder_kwargs["page_table"] == "page-table"
    assert builder_kwargs["page_tables_per_layer"] == ["layer-pages"]
    assert calls[2][0:3] == ("blocks", "model", "built-logits")


def test_generate_from_prompt_tokens_threads_aligned_prefill_cache_len():
    calls = []
    prompt_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[7, 8]], dtype=torch.long),
        prompt_len=32,
        next_pos=34,
        trajectories=[],
    )

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        calls.append(("prefill", tt_model, tokens.clone(), page_table, page_tables_per_layer))
        return PromptPrefill(prompt_len=tokens.shape[1], cache_len=32)

    def fake_builder(tt_model, **kwargs):
        calls.append(("builder", tt_model, kwargs))
        return "built-logits"

    def fake_blocks(tt_model, logits_fn, **kwargs):
        calls.append(("blocks", tt_model, logits_fn, kwargs))
        return generation

    out = generate_from_prompt_tokens(
        "model",
        None,
        prompt_tokens,
        num_blocks=1,
        config=DiffusionConfig(canvas_length=2),
        init_canvas_fn="init",
        logits_fn_builder=fake_builder,
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out is generation
    assert calls[1][2]["prompt_len"] == 32
    assert calls[2][3]["prompt_len"] == 32
    assert torch.equal(generation_sequences(prompt_tokens, out), torch.tensor([[1, 2, 3, 4, 5, 7, 8]]))


def test_generate_from_prompt_tokens_allows_exact_256k_context_boundary():
    class _Model:
        max_seq_len = 262144

    prompt_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.zeros((1, 32), dtype=torch.long),
        prompt_len=262112,
        next_pos=262144,
        trajectories=[],
    )

    out = generate_from_prompt_tokens(
        _Model(),
        "logits",
        prompt_tokens,
        num_blocks=1,
        config=DiffusionConfig(canvas_length=32),
        init_canvas_fn="init",
        prefill_fn=lambda *args, **kwargs: PromptPrefill(prompt_len=4, cache_len=262112),
        blocks_fn=lambda *args, **kwargs: generation,
    )

    assert out is generation


def test_generate_from_prompt_tokens_allows_zero_blocks_without_logits():
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        raise AssertionError("prefill should not run for zero generated blocks")

    def fake_blocks(tt_model, logits_fn, **kwargs):
        raise AssertionError("blocks should not run for zero generated blocks")

    out = generate_from_prompt_tokens(
        "model",
        None,
        prompt_tokens,
        num_blocks=0,
        config=DiffusionConfig(canvas_length=4),
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out.prompt_len == 3
    assert out.next_pos == 3
    assert out.trajectories == []
    assert torch.equal(out.generated, torch.empty((1, 0), dtype=torch.long))


def test_generate_from_prompt_tokens_zero_blocks_preserves_prompt_batch():
    """The zero-block fast path takes its batch from prompt_tokens, not from batch_size."""
    prompt_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    def fail_prefill(*args, **kwargs):
        raise AssertionError("prefill should not run")

    out = generate_from_prompt_tokens(
        "model",
        None,
        prompt_tokens,
        num_blocks=0,
        config=DiffusionConfig(canvas_length=4),
        prefill_fn=fail_prefill,
    )

    assert out.prompt_len == 3
    assert out.next_pos == 3
    assert torch.equal(out.generated, torch.empty((2, 0), dtype=torch.long))


# --- generate_text --------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def batch_decode(self, token_ids, **kwargs):
        self.calls.append((token_ids, kwargs))
        return [" ".join(str(token) for token in row) for row in token_ids]


class _FakeChatTokenizer(_FakeTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        self.calls.append((messages, add_generation_prompt, tokenize))
        return [len(messages), int(add_generation_prompt), 99]


class _FakeCallableTokenizer(_FakeTokenizer):
    def __call__(self, prompt, *, return_tensors):
        self.calls.append((prompt, return_tensors))
        return {"input_ids": torch.tensor([[7, 8, 9]], dtype=torch.int32)}


def test_generate_text_tokenizes_generates_and_decodes():
    calls = []
    tokenizer = _FakeChatTokenizer()
    config = DiffusionConfig(canvas_length=4)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5, 9, 6]], dtype=torch.long),
        prompt_len=3,
        next_pos=7,
        trajectories=[],
    )

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        calls.append(("prefill", tt_model, tokens.clone(), page_table, page_tables_per_layer))
        return tokens.shape[1]

    def fake_blocks(tt_model, logits_fn, **kwargs):
        calls.append(("blocks", tt_model, logits_fn, kwargs))
        return generation

    init_canvas_fn = object()
    out = generate_text(
        "model",
        "logits",
        tokenizer,
        "hello",
        num_blocks=1,
        config=config,
        init_canvas_fn=init_canvas_fn,
        system_prompt="be helpful",
        max_new_tokens=4,
        eos_token_id=9,
        decode_kwargs={"skip_special_tokens": True},
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert torch.equal(out.prompt_tokens, torch.tensor([[2, 1, 99]], dtype=torch.long))
    assert torch.equal(out.sequences, torch.tensor([[2, 1, 99, 4, 5, 9, 6]], dtype=torch.long))
    assert out.generation is generation
    assert out.text == ["4 5 9"]
    assert tokenizer.calls[0] == (
        [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hello"}],
        True,
        True,
    )
    assert tokenizer.calls[1] == ([[4, 5, 9]], {"skip_special_tokens": True})
    assert calls[0][0:2] == ("prefill", "model")
    assert torch.equal(calls[0][2], torch.tensor([[2, 1, 99]], dtype=torch.long))
    assert calls[0][3:] == (None, None)
    assert calls[1][0:3] == ("blocks", "model", "logits")
    assert calls[1][3]["init_canvas_fn"] is init_canvas_fn
    assert calls[1][3]["stop_token_ids"] == 9


# --- generate_text_from_checkpoint_state ----------------------------------------------------


class _EosTokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _VocabTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _LenTokenizer:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def test_generate_text_from_checkpoint_state_builds_logits_and_delegates():
    calls = {}
    result = object()

    def fake_builder_factory(dg_state_dict, **kwargs):
        calls["builder"] = (dg_state_dict, kwargs)
        return "builder"

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = (tt_model, logits_fn, tokenizer, prompt, kwargs)
        return result

    out = generate_text_from_checkpoint_state(
        "model",
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=2,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        adapter_kwargs={"adapter": "kwarg"},
        max_new_tokens=8,
        logits_fn_builder_factory=fake_builder_factory,
        generate_text_fn=fake_generate_text,
    )

    assert out is result
    # generate_text_from_checkpoint_state threads the config's denoise-step budget + temperature
    # schedule into adapter_kwargs via setdefault (DiffusionConfig defaults: 48 / 0.8 -> 0.4).
    assert calls["builder"] == (
        {"raw": "state"},
        {"adapter": "kwarg", "max_denoise_steps": 48, "temperature_start": 0.8, "temperature_end": 0.4},
    )
    assert calls["generate"][0:4] == ("model", None, "tokenizer", "hello")
    kwargs = calls["generate"][4]
    assert kwargs["num_blocks"] == 2
    assert kwargs["config"].canvas_length == 4
    assert kwargs["init_canvas_fn"] == "init"
    assert kwargs["logits_fn_builder"] == "builder"
    assert kwargs["max_new_tokens"] == 8


def test_generate_text_from_checkpoint_state_uses_model_config_for_adapter():
    calls = {}

    class _Model:
        hf_config = "model-config"

    def fake_builder_factory(dg_state_dict, **kwargs):
        calls["builder"] = (dg_state_dict, kwargs)
        return "builder"

    out = generate_text_from_checkpoint_state(
        _Model(),
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        logits_fn_builder_factory=fake_builder_factory,
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert out == "result"
    assert calls["builder"] == (
        {"raw": "state"},
        {"config": "model-config", "max_denoise_steps": 48, "temperature_start": 0.8, "temperature_end": 0.4},
    )


def test_generate_text_from_checkpoint_state_derives_num_blocks_from_max_new_tokens():
    calls = {}

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    out = generate_text_from_checkpoint_state(
        "model",
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        max_new_tokens=9,
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["num_blocks"] == 3
    assert calls["generate"]["max_new_tokens"] == 9


def test_generate_text_from_checkpoint_state_can_create_seeded_canvas_init(monkeypatch):
    calls = {}
    result = object()

    class _Model:
        mesh_device = "mesh"

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = (mesh_device, kwargs)
        return "init"

    def fake_noise_tokens_fn(mesh_device, **kwargs):
        calls["noise_tokens"] = (mesh_device, kwargs)
        return "noise"

    def fake_gumbel_noise_fn(mesh_device, **kwargs):
        calls["gumbel_noise"] = (mesh_device, kwargs)
        return "gumbel"

    def fake_builder_factory(dg_state_dict, **kwargs):
        calls["builder"] = (dg_state_dict, kwargs)
        return "builder"

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = (tt_model, logits_fn, tokenizer, prompt, kwargs)
        return result

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)
    monkeypatch.setattr(G, "make_seeded_host_noise_tokens_fn", fake_noise_tokens_fn)
    monkeypatch.setattr(G, "make_seeded_gumbel_noise_fn", fake_gumbel_noise_fn)

    out = generate_text_from_checkpoint_state(
        _Model(),
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        vocab_size=99,
        seed=123,
        gumbel_seed=321,
        noise_seed=456,
        batch=2,
        logits_fn_builder_factory=fake_builder_factory,
        generate_text_fn=fake_generate_text,
    )

    assert out is result
    assert calls["canvas_init"] == (
        "mesh",
        {
            "batch": 2,
            "canvas_len": 4,
            "vocab_size": 99,
            "seed": 123,
        },
    )
    assert calls["noise_tokens"] == (
        "mesh",
        {
            "batch": 2,
            "canvas_len": 4,
            "vocab_size": 99,
            "seed": 456,
        },
    )
    assert calls["gumbel_noise"] == (
        "mesh",
        {
            "batch": 2,
            "canvas_len": 4,
            "vocab_size": 99,
            "seed": 321,
        },
    )
    assert calls["generate"][4]["init_canvas_fn"] == "init"
    assert calls["generate"][4]["gumbel_noise_fn"] == "gumbel"
    assert calls["generate"][4]["noise_tokens_fn"] == "noise"
    assert calls["generate"][4]["logits_fn_builder"] == "builder"


@pytest.mark.parametrize(
    ("tokenizer", "model_attrs", "expected"),
    [
        pytest.param(_VocabTokenizer(99), {}, 99, id="tokenizer.vocab_size"),
        pytest.param(_LenTokenizer(101), {}, 101, id="len(tokenizer)"),
        pytest.param(object(), {"vocab_size": 103}, 103, id="model.vocab_size"),
        pytest.param(object(), {"hf_config": SimpleNamespace(vocab_size=105)}, 105, id="model.hf_config.vocab_size"),
    ],
)
def test_generate_text_from_checkpoint_state_infers_vocab_size_for_seeded_hooks(
    tokenizer, model_attrs, expected, monkeypatch
):
    calls = {}

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = kwargs
        return "init"

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)

    generate_text_from_checkpoint_state(
        SimpleNamespace(mesh_device="mesh", **model_attrs),
        tokenizer,
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        seed=123,
        noise_tokens_fn="noise",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["canvas_init"]["vocab_size"] == expected


def test_generate_text_from_checkpoint_state_defaults_eos_and_decode_kwargs_from_tokenizer():
    calls = {}

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    generate_text_from_checkpoint_state(
        "model",
        _EosTokenizer(9),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert calls["generate"]["eos_token_id"] == 9
    assert calls["generate"]["decode_kwargs"] == {"skip_special_tokens": True}


@pytest.mark.parametrize(
    ("kwarg", "value", "replaced_factory"),
    [
        pytest.param("noise_tokens_fn", "explicit-noise", "make_seeded_host_noise_tokens_fn", id="noise_tokens_fn"),
        pytest.param("gumbel_noise_fn", "explicit-gumbel", "make_seeded_gumbel_noise_fn", id="gumbel_noise_fn"),
        pytest.param("eos_token_id", None, None, id="eos_token_id"),
        pytest.param("decode_kwargs", {"clean_up_tokenization_spaces": False}, None, id="decode_kwargs"),
    ],
)
def test_generate_text_from_checkpoint_state_preserves_explicit_hooks(kwarg, value, replaced_factory, monkeypatch):
    """An explicitly supplied hook is threaded through untouched -- the seeded default must not
    silently replace it."""
    calls = {}

    if replaced_factory is not None:
        monkeypatch.setattr(G, replaced_factory, _boom)

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    generate_text_from_checkpoint_state(
        SimpleNamespace(mesh_device="mesh"),
        _EosTokenizer(9),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        vocab_size=99,
        seed=123,
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
        **{kwarg: value},
    )

    assert calls["generate"][kwarg] == value


# --- device entry-point argument validation -------------------------------------------------


class _ContextModel:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len


class _BoomTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("the prompt must not be tokenized before the arguments are validated")


def _blocks(**overrides):
    kwargs = dict(
        prompt_len=32,
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn=_boom,
        block_fn=_boom,
    )
    kwargs.update(overrides)
    return generate_blocks(kwargs.pop("tt_model", "model"), "logits", **kwargs)


def _from_prompt(**overrides):
    kwargs = dict(
        num_blocks=1,
        config=DiffusionConfig(canvas_length=2),
        init_canvas_fn="init",
        prefill_fn=lambda *args, **kwargs: PromptPrefill(prompt_len=4, cache_len=32),
        blocks_fn=_boom,
    )
    kwargs.update(overrides)
    return generate_from_prompt_tokens(
        kwargs.pop("tt_model", "model"),
        kwargs.pop("logits_fn", "logits"),
        kwargs.pop("prompt_tokens", torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        **kwargs,
    )


def _text(**overrides):
    kwargs = dict(
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        blocks_fn=_boom,
    )
    kwargs.update(overrides)
    return generate_text("model", "logits", _BoomTokenizer(), "hello", **kwargs)


def _from_state(**overrides):
    kwargs = dict(
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=_boom,
    )
    kwargs.update(overrides)
    return generate_text_from_checkpoint_state(
        kwargs.pop("tt_model", "model"), kwargs.pop("tokenizer", "tokenizer"), "hello", **kwargs
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda: _blocks(num_blocks=-1), "num_blocks must be non-negative", id="blocks-negative-num-blocks"
        ),
        pytest.param(lambda: _blocks(num_blocks=1.5), "integer", id="blocks-non-integer-num-blocks"),
        pytest.param(
            lambda: _blocks(config=DiffusionConfig(canvas_length=0)),
            "canvas_length must be positive",
            id="blocks-empty-canvas",
        ),
        pytest.param(lambda: _blocks(prompt_len=-1), "prompt_len", id="blocks-negative-prompt-len"),
        pytest.param(lambda: _blocks(prompt_len=INT32_MAX), "fit int32", id="blocks-prompt-len-overflow"),
        pytest.param(lambda: _blocks(batch_size=1.5, num_blocks=0), "integer", id="blocks-non-integer-batch"),
        pytest.param(
            lambda: _blocks(tt_model=_ContextModel(37), num_blocks=2, config=DiffusionConfig(canvas_length=3)),
            "context window",
            id="blocks-past-context-window",
        ),
        pytest.param(
            lambda: _from_prompt(num_blocks=0, prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.long)),
            "prompt_tokens must have shape",
            id="prompt-not-2d",
        ),
        pytest.param(
            lambda: _from_prompt(num_blocks=0, prompt_tokens=torch.empty((1, 0), dtype=torch.long)),
            "length",
            id="prompt-empty",
        ),
        pytest.param(
            lambda: _from_prompt(num_blocks=0, prompt_tokens=torch.tensor([[1, -2]], dtype=torch.long)),
            "non-negative",
            id="prompt-token-out-of-range",
        ),
        pytest.param(
            lambda: _from_prompt(prefill_fn=lambda *args, **kwargs: 3),
            "prefill prompt_len",
            id="prefill-length-mismatch",
        ),
        pytest.param(
            lambda: _from_prompt(prefill_fn=lambda *args, **kwargs: PromptPrefill(prompt_len=4, cache_len=3)),
            "cache_len",
            id="prefill-cache-shorter-than-prompt",
        ),
        pytest.param(
            lambda: _from_prompt(
                tt_model=_ContextModel(5),
                logits_fn=None,
                logits_fn_builder=_boom,
                prefill_fn=lambda *args, **kwargs: PromptPrefill(prompt_len=4, cache_len=4),
            ),
            "context window",
            id="prompt-past-context-window",
        ),
        pytest.param(
            lambda: _from_prompt(logits_fn_builder=lambda *args, **kwargs: "built"),
            "either logits_fn or logits_fn_builder",
            id="prompt-logits-and-builder",
        ),
        pytest.param(lambda: _text(num_blocks=-1), "num_blocks must be non-negative", id="text-negative-num-blocks"),
        pytest.param(lambda: _text(max_new_tokens=5), "num_blocks is too small", id="text-budget-exceeds-blocks"),
        pytest.param(lambda: _from_state(init_canvas_fn=None), "init_canvas_fn", id="state-no-canvas-source"),
        pytest.param(lambda: _from_state(num_blocks=None), "num_blocks", id="state-no-length-budget"),
        pytest.param(
            lambda: _from_state(max_new_tokens=5), "num_blocks is too small", id="state-budget-exceeds-blocks"
        ),
        pytest.param(lambda: _from_state(batch=0), "batch_size must be positive", id="state-empty-batch"),
        pytest.param(lambda: _from_state(vocab_size=0), "vocab_size must be positive", id="state-empty-vocab"),
        pytest.param(
            lambda: _from_state(tokenizer=_VocabTokenizer("99")),
            "vocab_size must be an integer",
            id="state-non-integer-inferred-vocab",
        ),
        pytest.param(
            lambda: _from_state(tt_model=object(), tokenizer=object(), seed=123, gumbel_noise_fn="gumbel"),
            "noise_tokens_fn requires vocab_size",
            id="state-seeded-noise-without-vocab",
        ),
        pytest.param(
            lambda: _from_state(tt_model=object(), tokenizer=object(), noise_tokens_fn="noise", gumbel_seed=123),
            "gumbel_noise_fn requires vocab_size",
            id="state-seeded-gumbel-without-vocab",
        ),
        pytest.param(
            lambda: _from_state(vocab_size=99, noise_tokens_fn="noise", gumbel_seed=0),
            "positive nonzero",
            id="state-zero-gumbel-seed",
        ),
        pytest.param(lambda: _from_state(eos_token_id=-1), "non-negative", id="state-negative-eos"),
    ],
)
def test_entry_points_reject_impossible_requests(call, message, expect_error):
    """Each entry point validates its arguments before it touches the tokenizer, the model or the
    device: every helper above wires `_boom` into the work that must not have started yet."""
    with expect_error(ValueError, match=message):
        call()


# --- generation sequences and decoding ------------------------------------------------------


def test_generation_sequences_appends_prompt_and_generated_tokens():
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    assert torch.equal(generation_sequences(prompt_tokens, generation), torch.tensor([[1, 2, 3, 4, 5]]))


def test_generation_sequences_allows_empty_generated_continuation():
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.empty((1, 0), dtype=torch.long),
        prompt_len=3,
        next_pos=3,
        trajectories=[],
    )

    assert torch.equal(generation_sequences(prompt_tokens, generation), prompt_tokens)


def test_decode_generation_defaults_to_generated_continuation():
    tokenizer = _FakeTokenizer()
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    assert decode_generation(tokenizer, prompt_tokens, generation, skip_special_tokens=True) == ["4 5"]
    assert tokenizer.calls == [([[4, 5]], {"skip_special_tokens": True})]


def test_decode_generation_can_include_prompt_tokens():
    tokenizer = _FakeTokenizer()
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    assert decode_generation(tokenizer, prompt_tokens, generation, skip_prompt=False) == ["1 2 3 4 5"]


def test_generation_token_ids_applies_max_new_tokens_and_eos_to_continuation():
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5, 9, 6]], dtype=torch.long),
        prompt_len=3,
        next_pos=7,
        trajectories=[],
    )

    assert generation_token_ids(prompt_tokens, generation, max_new_tokens=4, eos_token_id=9) == [[4, 5, 9]]
    assert generation_token_ids(prompt_tokens, generation, max_new_tokens=2, eos_token_id=9) == [[4, 5]]


def test_generation_token_ids_can_return_full_trimmed_sequences():
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5, 9, 6]], dtype=torch.long),
        prompt_len=3,
        next_pos=7,
        trajectories=[],
    )

    assert generation_token_ids(prompt_tokens, generation, skip_prompt=False, eos_token_id=[9]) == [[1, 2, 3, 4, 5, 9]]


_PROMPT_TOKENS = torch.tensor([[1, 2, 3]], dtype=torch.long)


def _generation(**overrides):
    kwargs = dict(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )
    kwargs.update(overrides)
    return G.DeviceGeneration(**kwargs)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda: generation_sequences(torch.tensor([[1, -2, 3]], dtype=torch.long), _generation()),
            "non-negative",
            id="prompt-token-out-of-range",
        ),
        pytest.param(
            lambda: generation_sequences(
                _PROMPT_TOKENS, _generation(generated=torch.tensor([[4, INT32_MAX + 1]], dtype=torch.long))
            ),
            "fit int32",
            id="generated-token-out-of-range",
        ),
        pytest.param(
            lambda: generation_sequences(_PROMPT_TOKENS, _generation(prompt_len=2, next_pos=4)),
            "prompt_tokens length",
            id="prompt-len-mismatch",
        ),
        pytest.param(
            lambda: generation_sequences(_PROMPT_TOKENS, _generation(next_pos=7)),
            "generation.next_pos",
            id="next-pos-mismatch",
        ),
        pytest.param(
            lambda: generation_token_ids(_PROMPT_TOKENS, _generation(), eos_token_id=[9, -1]),
            "non-negative",
            id="eos-out-of-range",
        ),
        pytest.param(
            lambda: decode_generation(_FakeTokenizer(), _PROMPT_TOKENS, _generation(), max_new_tokens=-1),
            "non-negative",
            id="negative-max-new-tokens",
        ),
    ],
)
def test_generation_helpers_reject_bad_inputs(call, message, expect_error):
    with expect_error(ValueError, match=message):
        call()


# --- host to device transfers ---------------------------------------------------------------


class _FakeMesh:
    shape = (1, 4)

    def get_num_devices(self):
        return 4


def test_host_canvas_to_device_uses_controller_token_layout(monkeypatch):
    calls = {}

    class _FakeTtnn:
        TILE_LAYOUT = "tile"
        uint32 = "uint32"

        @staticmethod
        def ReplicateTensorToMesh(mesh_device):
            return ("replicate", mesh_device)

        @staticmethod
        def from_torch(value, **kwargs):
            calls["from_torch"] = (value.clone(), kwargs)
            return "device-canvas"

    monkeypatch.setattr(G, "ttnn", _FakeTtnn)
    canvas = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = host_canvas_to_device(_FakeMesh(), canvas)

    value, kwargs = calls["from_torch"]
    assert out == "device-canvas"
    assert value.shape == (1, 1, 3, 1)
    assert value.dtype == torch.int32
    assert kwargs["layout"] == "tile"
    assert kwargs["dtype"] == "uint32"
    assert kwargs["mesh_mapper"] == ("replicate", kwargs["device"])


def test_host_gumbel_noise_to_device_uses_logits_layout(monkeypatch):
    calls = {}

    class _FakeTtnn:
        TILE_LAYOUT = "tile"
        float32 = "float32"

        @staticmethod
        def ReplicateTensorToMesh(mesh_device):
            return ("replicate", mesh_device)

        @staticmethod
        def from_torch(value, **kwargs):
            calls["from_torch"] = (value.clone(), kwargs)
            return "device-gumbel"

    monkeypatch.setattr(G, "ttnn", _FakeTtnn)
    noise = torch.arange(24, dtype=torch.float64).reshape(1, 4, 6)

    out = host_gumbel_noise_to_device(_FakeMesh(), noise)

    value, kwargs = calls["from_torch"]
    assert out == "device-gumbel"
    assert value.shape == (1, 1, 4, 6)
    assert value.dtype == torch.float32
    assert kwargs["layout"] == "tile"
    assert kwargs["dtype"] == "float32"
    assert kwargs["mesh_mapper"] == ("replicate", kwargs["device"])


def test_host_tokens_to_device_uses_embedding_token_layout(monkeypatch):
    calls = {}

    class _FakeTtnn:
        TILE_SIZE = 32
        ROW_MAJOR_LAYOUT = "row-major"
        uint32 = "uint32"

        @staticmethod
        def ReplicateTensorToMesh(mesh_device):
            return ("replicate", mesh_device)

        @staticmethod
        def from_torch(value, **kwargs):
            calls["from_torch"] = (value.clone(), kwargs)
            return "device-tokens"

    monkeypatch.setattr(G, "ttnn", _FakeTtnn)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = host_tokens_to_device(_FakeMesh(), tokens)

    value, kwargs = calls["from_torch"]
    assert out == "device-tokens"
    assert torch.equal(value, torch.tensor([[1, 2, 3]], dtype=torch.int32))
    assert kwargs["layout"] == "row-major"
    assert kwargs["dtype"] == "uint32"
    assert kwargs["mesh_mapper"] == ("replicate", kwargs["device"])


# --- prompt tokenization --------------------------------------------------------------------


def test_tokenize_prompt_applies_chat_template_to_string_prompt():
    tokenizer = _FakeChatTokenizer()

    out = tokenize_prompt(tokenizer, "hello", system_prompt="be helpful")

    assert torch.equal(out, torch.tensor([[2, 1, 99]], dtype=torch.long))
    assert tokenizer.calls == [
        (
            [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hello"}],
            True,
            True,
        )
    ]


def test_tokenize_prompt_passes_chat_messages_through():
    tokenizer = _FakeChatTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    out = tokenize_prompt(tokenizer, messages, add_generation_prompt=False)

    assert torch.equal(out, torch.tensor([[1, 0, 99]], dtype=torch.long))
    assert tokenizer.calls == [(messages, False, True)]


def test_tokenize_prompt_uses_callable_tokenizer_without_chat_template():
    tokenizer = _FakeCallableTokenizer()

    out = tokenize_prompt(tokenizer, "plain prompt")

    assert torch.equal(out, torch.tensor([[7, 8, 9]], dtype=torch.long))
    assert tokenizer.calls == [("plain prompt", "pt")]


def test_tokenize_prompt_accepts_existing_token_tensor():
    assert torch.equal(tokenize_prompt(object(), torch.tensor([1, 2, 3], dtype=torch.int32)), torch.tensor([[1, 2, 3]]))


class _BoomModel:
    def prepare_inputs_decode(self, *args, **kwargs):
        raise AssertionError("the model must not run for an invalid commit")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda: host_canvas_to_device("mesh", torch.tensor([[1, -2]], dtype=torch.long)),
            "non-negative",
            id="canvas-token-out-of-range",
        ),
        pytest.param(
            lambda: host_tokens_to_device("mesh", torch.tensor([[1.5, 2.0]], dtype=torch.float32)),
            "integer token ids",
            id="tokens-not-integer",
        ),
        pytest.param(
            lambda: host_gumbel_noise_to_device("mesh", torch.zeros(1, 2, 3, 4, 5)),
            "gumbel_noise",
            id="gumbel-bad-rank",
        ),
        pytest.param(
            lambda: host_gumbel_noise_to_device("mesh", torch.zeros(1, 4, 0)),
            "dimensions must be positive",
            id="gumbel-empty-dimension",
        ),
        pytest.param(
            lambda: commit_canvas_tokens(
                _BoomModel(), torch.tensor([[1, INT32_MAX + 1]], dtype=torch.long), start_pos=0
            ),
            "fit int32",
            id="commit-token-out-of-range",
        ),
        pytest.param(
            lambda: commit_canvas_tokens(_BoomModel(), torch.tensor([[7, 8]], dtype=torch.long), start_pos=-1),
            "start_pos",
            id="commit-negative-start-pos",
        ),
        pytest.param(
            lambda: tokenize_prompt(object(), torch.empty((1, 0), dtype=torch.long)),
            "length",
            id="tokenize-empty-prompt",
        ),
        pytest.param(
            lambda: tokenize_prompt(object(), torch.tensor([[1, -2]], dtype=torch.int32)),
            "non-negative",
            id="tokenize-token-out-of-range",
        ),
    ],
)
def test_host_transfer_helpers_reject_bad_tensors(call, message, expect_error):
    with expect_error(ValueError, match=message):
        call()


# --- prompt prefill -------------------------------------------------------------------------


def test_prefill_prompt_tokens_embeds_and_writes_kv(monkeypatch):
    calls = {}

    class _FakeDeviceTensor:
        def __init__(self, name):
            self.name = name
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        TILE_SIZE = 32
        ROW_MAJOR_LAYOUT = "row-major"
        TILE_LAYOUT = "tile"
        uint32 = "uint32"

        @staticmethod
        def from_torch(value, **kwargs):
            calls["from_torch"] = (value.clone(), kwargs)
            return _FakeDeviceTensor("tokens")

        @staticmethod
        def reshape(value, shape):
            calls["reshape"] = (value, shape)
            return _FakeDeviceTensor("reshaped-embeds")

        @staticmethod
        def to_layout(value, layout):
            calls["to_layout"] = (value, layout)
            return _FakeDeviceTensor("tile-embeds")

        @staticmethod
        def embedding(tokens, weight, dtype=None):
            calls["embedding"] = (tokens, weight, dtype)
            return _FakeDeviceTensor("embeds")

        @staticmethod
        def mul(value, scale):
            calls["mul"] = (value, scale)
            return _FakeDeviceTensor("scaled-embeds")

        bfloat16 = "bfloat16"

    class _FakeModel:
        mesh_device = object()
        hidden_size = 16
        # DG embeds through its own op sequence rather than Gemma4Model.embed_tokens,
        # so it can route the TP all-gather through the semaphore-passing
        # ccl_allgather. Deliberately NO embed_tokens attribute: reverting to the
        # shared method must fail here, not silently reintroduce the plain
        # ttnn.all_gather. mesh_config None is the single-device case (no gather).
        embedding_weight = "embedding-weight"
        embed_scale = 4.0
        mesh_config = None

        def __call__(self, hidden_states, **kwargs):
            calls["model"] = (hidden_states, kwargs)
            return _FakeDeviceTensor("logits")

    monkeypatch.setattr(G, "ttnn", _FakeTtnn)
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)

    out = prefill_prompt_tokens(_FakeModel(), prompt_tokens, page_tables_per_layer=["pages"])

    assert out == PromptPrefill(prompt_len=3, cache_len=32)
    assert calls["embedding"][1] == "embedding-weight"
    assert calls["embedding"][0].deallocated is True
    assert calls["mul"][1] == 4.0
    assert calls["reshape"][1] == (1, 1, 32, 16)
    hidden_states, kwargs = calls["model"]
    assert hidden_states.name == "tile-embeds"
    assert kwargs["is_decode"] is False
    assert kwargs["input_ids_torch"].shape == (1, 32)
    assert torch.equal(kwargs["input_ids_torch"][:, :3], prompt_tokens)
    assert torch.equal(kwargs["input_ids_torch"][:, 3:], torch.zeros((1, 29), dtype=prompt_tokens.dtype))
    assert kwargs["get_last_token"] == 0
    # prefill writes the prompt KV via stock Gemma4 defaults (no diffusion kv_phase kwarg).
    assert "kv_phase" not in kwargs
    assert kwargs["page_tables_per_layer"] == ["pages"]

    bucketed = prefill_prompt_tokens(
        _FakeModel(),
        prompt_tokens,
        page_tables_per_layer=["pages"],
        execution_len=128,
    )
    assert bucketed == PromptPrefill(prompt_len=3, cache_len=32), "compute padding must not move canvas start"
    assert calls["reshape"][1] == (1, 1, 128, 16)
    assert calls["model"][1]["input_ids_torch"].shape == (1, 128)
    assert torch.equal(calls["model"][1]["input_ids_torch"][:, :3], prompt_tokens)
    assert torch.count_nonzero(calls["model"][1]["input_ids_torch"][:, 3:]) == 0


def test_prefill_execution_len_rejects_shape_smaller_than_logical_cache(expect_error):
    with expect_error(ValueError, match="cannot cover cache_len 64"):
        prefill_prompt_tokens(object(), torch.ones((1, 33), dtype=torch.long), execution_len=32)


@pytest.mark.parametrize(
    ("prompt_len", "execution_len", "expected_cache_len", "expected_compute_len"),
    [
        pytest.param(32768, 32768, 32768, 32768, id="exact-32k-boundary"),
        pytest.param(32769, 65536, 32800, 36864, id="above-32k-boundary"),
    ],
)
def test_prefill_bucket_at_or_above_32k_uses_fixed_bounded_chunks(
    monkeypatch,
    prompt_len,
    execution_len,
    expected_cache_len,
    expected_compute_len,
):
    from models.experimental.diffusion_gemma.tt import chunked_prefill as CP

    calls = []
    model = SimpleNamespace(
        _dg_model_owned_hybrid_kv=True,
        _dg_hybrid_host_page_tables_per_layer=["host-pages"],
        _dg_hybrid_page_tables_per_layer=["device-pages"],
        _dg_hybrid_block_size=64,
        tt_kv_cache=["kv"],
    )
    monkeypatch.setenv("DG_PREFILL_CHUNK_SIZE", "4096")
    monkeypatch.setattr(
        CP,
        "chunked_prefill",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    prompt = torch.ones((1, prompt_len), dtype=torch.long)
    result = prefill_prompt_tokens(
        model,
        prompt,
        page_tables_per_layer=["device-pages"],
        execution_len=execution_len,
    )

    assert result == PromptPrefill(prompt_len=prompt_len, cache_len=expected_cache_len)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (model,)
    assert kwargs["input_ids_torch"].shape == (1, expected_compute_len)
    assert kwargs["valid_prompt_len"] == prompt_len
    assert kwargs["chunk_size"] == 4096
    assert kwargs["return_last_logits"] is False
    assert kwargs["page_tables_torch_per_layer"] == ["host-pages"]
    assert kwargs["page_tables_per_layer"] == ["device-pages"]


def test_fixed_prefill_chunks_use_one_program_for_short_prompts(monkeypatch):
    from models.experimental.diffusion_gemma.tt import chunked_prefill as CP

    calls = []
    model = SimpleNamespace(
        _dg_model_owned_hybrid_kv=True,
        _dg_hybrid_host_page_tables_per_layer=["host-pages"],
        _dg_hybrid_page_tables_per_layer=["device-pages"],
        _dg_hybrid_block_size=64,
        tt_kv_cache=["kv"],
    )
    monkeypatch.setenv("DG_PREFILL_FIXED_CHUNKS", "1")
    monkeypatch.setenv("DG_PREFILL_CHUNK_SIZE", "4096")
    monkeypatch.setattr(
        CP,
        "chunked_prefill",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    prompt = torch.ones((1, 129), dtype=torch.long)
    result = prefill_prompt_tokens(
        model,
        prompt,
        page_tables_per_layer=["device-pages"],
        execution_len=256,
    )

    assert result == PromptPrefill(prompt_len=129, cache_len=160)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (model,)
    assert kwargs["input_ids_torch"].shape == (1, 4096)
    assert kwargs["valid_prompt_len"] == 129
    assert kwargs["chunk_size"] == 4096
    assert kwargs["return_last_logits"] is False


# --- device generation hook factories -------------------------------------------------------


def test_make_host_canvas_init_fn_replays_fixed_canvases(monkeypatch):
    calls = []

    def fake_host_canvas_to_device(mesh_device, canvas):
        first_token = int(canvas[0, 0])
        calls.append((mesh_device, canvas.clone()))
        canvas[0, 0] = 99
        return f"device-{first_token}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)
    canvases = [torch.tensor([[4, 5]]), torch.tensor([[6, 7]])]
    init_fn = make_host_canvas_init_fn("mesh", canvases)
    canvases[0][0, 0] = 88

    assert init_fn(0, 32) == "device-4"
    assert init_fn(1, 34) == "device-6"
    assert init_fn(0, 36) == "device-4"
    assert torch.equal(calls[0][1], torch.tensor([[4, 5]]))
    assert torch.equal(calls[1][1], torch.tensor([[6, 7]]))
    assert torch.equal(calls[2][1], torch.tensor([[4, 5]]))


def test_make_seeded_host_canvas_init_fn_generates_reproducible_tokens(monkeypatch):
    calls = []

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls.append((mesh_device, canvas.clone()))
        return f"device-canvas-{len(calls)}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)

    init_a = make_seeded_host_canvas_init_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=11)
    init_b = make_seeded_host_canvas_init_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=11)

    assert init_a(0, 32) == "device-canvas-1"
    assert init_a(1, 36) == "device-canvas-2"
    assert init_b(0, 32) == "device-canvas-3"
    assert torch.equal(calls[0][1], calls[2][1])
    assert not torch.equal(calls[0][1], calls[1][1])
    assert int(calls[0][1].min()) >= 0 and int(calls[0][1].max()) < 16


def test_make_seeded_host_noise_tokens_fn_generates_step_noise(monkeypatch):
    calls = []

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls.append((mesh_device, canvas.clone()))
        return f"device-noise-{len(calls)}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)

    noise_a = make_seeded_host_noise_tokens_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=23)
    noise_b = make_seeded_host_noise_tokens_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=23)

    block0 = noise_a(0)
    assert block0(0) == "device-noise-1"
    assert block0(1) == "device-noise-2"
    assert noise_b(0)(0) == "device-noise-3"
    assert torch.equal(calls[0][1], calls[2][1])
    assert not torch.equal(calls[0][1], calls[1][1])
    assert int(calls[0][1].min()) >= 0 and int(calls[0][1].max()) < 16


def test_seeded_host_token_helpers_allow_zero_seed(monkeypatch):
    calls = []

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls.append((mesh_device, canvas.clone()))
        return f"device-tokens-{len(calls)}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)

    init_a = make_seeded_host_canvas_init_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=0)
    init_b = make_seeded_host_canvas_init_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=0)
    noise_a = make_seeded_host_noise_tokens_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=0)
    noise_b = make_seeded_host_noise_tokens_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=0)

    assert init_a(0, 32) == "device-tokens-1"
    assert init_b(0, 32) == "device-tokens-2"
    assert noise_a(0)(0) == "device-tokens-3"
    assert noise_b(0)(0) == "device-tokens-4"
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[2][1], calls[3][1])


def test_make_host_noise_tokens_fn_replays_fixed_tokens(monkeypatch):
    calls = []

    def fake_host_canvas_to_device(mesh_device, tokens):
        calls.append((mesh_device, tokens.clone()))
        tokens[0, 0] = 99
        return f"device-noise-{len(calls)}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)
    host_tokens = [
        [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])],
        [torch.tensor([[7, 8, 9]])],
    ]
    noise_fn = make_host_noise_tokens_fn("mesh", host_tokens)
    host_tokens[0][0][0, 0] = 88

    assert noise_fn(0)(0) == "device-noise-1"
    assert noise_fn(0)(1) == "device-noise-2"
    assert noise_fn(0)(0) == "device-noise-3"
    assert noise_fn(1)(0) == "device-noise-4"
    assert torch.equal(calls[0][1], torch.tensor([[1, 2, 3]]))
    assert torch.equal(calls[1][1], torch.tensor([[4, 5, 6]]))
    assert torch.equal(calls[2][1], torch.tensor([[1, 2, 3]]))
    assert torch.equal(calls[3][1], torch.tensor([[7, 8, 9]]))


def test_make_host_gumbel_noise_fn_replays_fixed_noise(monkeypatch):
    calls = []

    def fake_host_gumbel_noise_to_device(mesh_device, noise):
        calls.append((mesh_device, noise.clone()))
        noise.reshape(-1)[0] = 99.0
        return f"device-gumbel-{len(calls)}"

    monkeypatch.setattr(G, "host_gumbel_noise_to_device", fake_host_gumbel_noise_to_device)
    host_noise = [
        [torch.full((1, 4, 8), 1.0), torch.full((1, 4, 8), 2.0)],
        [torch.full((1, 4, 8), 3.0)],
    ]
    noise_fn = make_host_gumbel_noise_fn("mesh", host_noise)
    host_noise[0][0].fill_(88.0)

    assert noise_fn(0)(0) == "device-gumbel-1"
    assert noise_fn(0)(1) == "device-gumbel-2"
    assert noise_fn(0)(0) == "device-gumbel-3"
    assert noise_fn(1)(0) == "device-gumbel-4"
    assert torch.equal(calls[0][1], torch.full((1, 4, 8), 1.0))
    assert torch.equal(calls[1][1], torch.full((1, 4, 8), 2.0))
    assert torch.equal(calls[2][1], torch.full((1, 4, 8), 1.0))
    assert torch.equal(calls[3][1], torch.full((1, 4, 8), 3.0))


def test_make_host_gumbel_noise_fn_allows_equivalent_3d_and_4d_shapes(monkeypatch):
    calls = []

    def fake_host_gumbel_noise_to_device(mesh_device, noise):
        calls.append((mesh_device, tuple(noise.shape)))
        return f"device-gumbel-{len(calls)}"

    monkeypatch.setattr(G, "host_gumbel_noise_to_device", fake_host_gumbel_noise_to_device)
    noise_fn = make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 4, 8)], [torch.zeros(1, 1, 4, 8)]])

    assert noise_fn(0)(0) == "device-gumbel-1"
    assert noise_fn(1)(0) == "device-gumbel-2"
    assert calls == [("mesh", (1, 4, 8)), ("mesh", (1, 1, 4, 8))]


def test_make_seeded_gumbel_noise_fn_generates_block_step_seeds(monkeypatch):
    """The production draw is vocab-innermost: the permuted variant would put the canvas
    positions on ttnn.rand's degenerate axis (154/256 vs 253/256 distinct winners)."""
    calls = []

    def fake_sample_gumbel_noise(shape, *, device, seed):
        calls.append((shape, device, seed))
        return f"gumbel-{len(calls)}"

    monkeypatch.setattr(G.TS, "sample_gumbel_noise", fake_sample_gumbel_noise)

    noise = make_seeded_gumbel_noise_fn("mesh", batch=2, canvas_len=4, vocab_size=16, seed=31)

    assert noise(0)(0) == "gumbel-1"
    assert noise(0)(1) == "gumbel-2"
    assert noise(1)(0) == "gumbel-3"
    assert calls == [
        ((2, 1, 4, 16), "mesh", 31),
        ((2, 1, 4, 16), "mesh", 32),
        ((2, 1, 4, 16), "mesh", 1_000_034),
    ]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        pytest.param(
            lambda: make_host_canvas_init_fn("mesh", [torch.tensor([4, 5])]),
            ValueError,
            "host_canvases",
            id="canvas-replay-not-2d",
        ),
        pytest.param(
            lambda: make_host_canvas_init_fn("mesh", [torch.tensor([[1, -2]], dtype=torch.long)]),
            ValueError,
            "non-negative",
            id="canvas-replay-token-out-of-range",
        ),
        pytest.param(
            lambda: make_host_canvas_init_fn("mesh", [torch.tensor([[4, 5]])])(1, 32),
            IndexError,
            "block index 1 out of range",
            id="canvas-replay-block-out-of-range",
        ),
        pytest.param(
            lambda: make_host_noise_tokens_fn("mesh", [[torch.tensor([[1, 2, 3]])], [torch.tensor([[4, 5]])]]),
            ValueError,
            "host_noise_tokens",
            id="noise-replay-ragged",
        ),
        pytest.param(
            lambda: make_host_noise_tokens_fn("mesh", [[torch.tensor([[1, 2, 3]])]])(0)(1),
            IndexError,
            "step index 1 out of range",
            id="noise-replay-step-out-of-range",
        ),
        pytest.param(
            lambda: make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 4, 8)], [torch.zeros(1, 4, 9)]]),
            ValueError,
            "host_gumbel_noise",
            id="gumbel-replay-ragged",
        ),
        pytest.param(
            lambda: make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 4, 0)]]),
            ValueError,
            "dimensions must be positive",
            id="gumbel-replay-empty-dimension",
        ),
        pytest.param(
            lambda: make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 2, 3)]])(0)(0.5),
            IndexError,
            "step index must be an integer",
            id="gumbel-replay-non-integer-step",
        ),
        pytest.param(
            lambda: make_seeded_host_canvas_init_fn("mesh", batch=1.5, canvas_len=4, vocab_size=16, seed=1),
            ValueError,
            "batch",
            id="seeded-non-integer-batch",
        ),
        pytest.param(
            lambda: make_seeded_host_canvas_init_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=-1),
            ValueError,
            "non-negative",
            id="seeded-negative-seed",
        ),
        pytest.param(
            lambda: make_seeded_gumbel_noise_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=0),
            ValueError,
            "positive nonzero",
            id="device-gumbel-zero-seed",
        ),
    ],
)
def test_generation_hook_factories_reject_bad_inputs(call, error, message, expect_error):
    with expect_error(error, match=message):
        call()
