# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Key end-to-end and runtime regressions for block-autoregressive generation."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.reference.generate import generate_blocks as reference_generate_blocks
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import (
    GeneratedBlock,
    PromptPrefill,
    denoise_and_commit_block,
    generate_blocks,
    generate_from_prompt_tokens,
    generate_text_from_checkpoint_state,
    generation_sequences,
    prefill_prompt_tokens,
)


def _gen(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class _PrefixDependentModel:
    """Peaked logits whose target changes as committed blocks extend the prefix."""

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
    for block_idx in range(blocks):
        block_tokens = hf_generated[:, block_idx * canvas_len : (block_idx + 1) * canvas_len]
        assert torch.all(block_tokens == 2 + block_idx)


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
    )

    assert logits_fn.q_rope_offset == 544
    assert calls["denoise"] == (logits_fn, "init-canvas", config, gumbel_noise_fn, noise_tokens_fn)
    assert calls["commit"] == ("model", committed, 544, None, page_tables_per_layer)
    assert calls["advance_prefix"] == 547
    assert out.committed is committed
    assert out.next_pos == 547
    assert out.trajectory is trajectory


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
            committed=committed,
            next_pos=kwargs["start_pos"] + config.canvas_length,
            trajectory=trajectory,
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

    out = generate_from_prompt_tokens(
        "model",
        "logits",
        prompt_tokens,
        num_blocks=2,
        config=config,
        init_canvas_fn="init",
        page_table="page-table",
        page_tables_per_layer=["layer-pages"],
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out is generation
    assert calls[0] == ("prefill", "model", prompt_tokens, "page-table", ["layer-pages"])
    assert calls[1][0:3] == ("blocks", "model", "logits")
    assert calls[1][3]["prompt_len"] == prompt_tokens.shape[1]
    assert calls[1][3]["num_blocks"] == 2
    assert calls[1][3]["config"] is config
    assert calls[1][3]["page_table"] == "page-table"
    assert calls[1][3]["page_tables_per_layer"] == ["layer-pages"]


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
        return PromptPrefill(prompt_len=tokens.shape[1], cache_len=32)

    def fake_builder(tt_model, **kwargs):
        calls.append(("builder", kwargs))
        return "built-logits"

    def fake_blocks(tt_model, logits_fn, **kwargs):
        calls.append(("blocks", kwargs))
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
    assert calls[0][1]["prompt_len"] == 32
    assert calls[1][1]["prompt_len"] == 32
    assert torch.equal(generation_sequences(prompt_tokens, out), torch.tensor([[1, 2, 3, 4, 5, 7, 8]]))


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
    assert calls["builder"] == (
        {"raw": "state"},
        {"adapter": "kwarg", "max_denoise_steps": 48, "temperature_start": 0.8, "temperature_end": 0.4},
    )
    assert calls["generate"][0:4] == ("model", None, "tokenizer", "hello")
    assert calls["generate"][4]["logits_fn_builder"] == "builder"
    assert calls["generate"][4]["max_new_tokens"] == 8


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
        bfloat16 = "bfloat16"

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

    class _FakeModel:
        mesh_device = object()
        hidden_size = 16
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
    assert "kv_phase" not in kwargs
    assert kwargs["page_tables_per_layer"] == ["pages"]


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
    monkeypatch.setattr(CP, "chunked_prefill", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = prefill_prompt_tokens(
        model,
        torch.ones((1, prompt_len), dtype=torch.long),
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
