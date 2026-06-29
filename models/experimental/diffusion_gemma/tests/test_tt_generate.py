# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import (
    GeneratedBlock,
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


class _FakeLogitsFn:
    q_rope_offset = None


class _FakeMesh:
    shape = (1, 4)

    def get_num_devices(self):
        return 4


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
    assert out.committed is committed
    assert out.next_pos == 547
    assert out.trajectory is trajectory


def test_denoise_and_commit_block_rejects_missing_commit():
    trajectory = DenoiseTrajectory(committed=None, num_steps=0, halted=False, per_step=[])

    with pytest.raises(RuntimeError, match="did not produce committed"):
        denoise_and_commit_block(
            object(),
            object(),
            object(),
            DiffusionConfig(canvas_length=3),
            start_pos=0,
            denoise_block_fn=lambda *args, **kwargs: trajectory,
            commit_fn=lambda *args, **kwargs: None,
        )


def test_denoise_and_commit_block_rejects_bad_committed_shape_before_commit():
    trajectory = DenoiseTrajectory(
        committed=torch.tensor([[7, 8]], dtype=torch.long), num_steps=1, halted=True, per_step=[]
    )

    def fail_commit(*args, **kwargs):
        raise AssertionError("commit should not run for malformed committed canvas")

    with pytest.raises(ValueError, match="block.committed"):
        denoise_and_commit_block(
            object(),
            object(),
            object(),
            DiffusionConfig(canvas_length=3),
            start_pos=0,
            denoise_block_fn=lambda *args, **kwargs: trajectory,
            commit_fn=fail_commit,
        )


@pytest.mark.parametrize(
    ("start_pos", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
        (-1, "start_pos"),
        (torch.iinfo(torch.int32).max, "fit int32"),
    ],
)
def test_commit_canvas_tokens_rejects_bad_position_before_model_call(start_pos, message):
    class _FailModel:
        def prepare_inputs_decode(self, *args, **kwargs):
            raise AssertionError("prepare_inputs_decode should not run for invalid positions")

    with pytest.raises(ValueError, match=message):
        commit_canvas_tokens(
            _FailModel(),
            torch.tensor([[7, 8]], dtype=torch.long),
            start_pos=start_pos,
        )


@pytest.mark.parametrize(
    ("canvas_tokens", "message"),
    [
        (torch.tensor([[1.5, 2.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_commit_canvas_tokens_rejects_invalid_token_ids_before_model_call(canvas_tokens, message):
    class _FailModel:
        def prepare_inputs_decode(self, *args, **kwargs):
            raise AssertionError("prepare_inputs_decode should not run for invalid canvas tokens")

    with pytest.raises(ValueError, match=message):
        commit_canvas_tokens(_FailModel(), canvas_tokens, start_pos=0)


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


def test_generate_blocks_rejects_negative_num_blocks():
    with pytest.raises(ValueError, match="num_blocks must be non-negative"):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=-1,
            config=DiffusionConfig(canvas_length=3),
            init_canvas_fn=lambda *args: "canvas",
        )


@pytest.mark.parametrize(
    ("num_blocks", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_blocks_rejects_non_integer_num_blocks(num_blocks, message):
    with pytest.raises(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=num_blocks,
            config=DiffusionConfig(canvas_length=3),
            init_canvas_fn=lambda *args: "canvas",
        )


def test_generate_blocks_rejects_non_positive_canvas_length():
    with pytest.raises(ValueError, match="canvas_length must be positive"):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=0),
            init_canvas_fn=lambda *args: "canvas",
        )


@pytest.mark.parametrize(
    ("prompt_len", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
        (-1, "prompt_len"),
        (torch.iinfo(torch.int32).max, "fit int32"),
    ],
)
def test_generate_blocks_rejects_bad_prompt_position_span(prompt_len, message):
    with pytest.raises(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=prompt_len,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn=lambda *args: "canvas",
            block_fn=lambda *args, **kwargs: None,
        )


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


@pytest.mark.parametrize(
    ("batch_size", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_blocks_rejects_non_integer_batch_size(batch_size, message):
    with pytest.raises(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=0,
            config=DiffusionConfig(canvas_length=4),
            batch_size=batch_size,
        )


def test_generate_blocks_rejects_block_next_pos_mismatch():
    def bad_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        committed = torch.full((1, config.canvas_length), 7, dtype=torch.long)
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + committed.shape[1] + 1,
            trajectory=trajectory,
        )

    with pytest.raises(ValueError, match="block.next_pos"):
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
    "committed",
    [
        torch.tensor([7, 7, 7, 7], dtype=torch.long),
        torch.tensor([[7, 7, 7]], dtype=torch.long),
        torch.tensor([[7, 7, 7, 7], [8, 8, 8, 8]], dtype=torch.long),
    ],
)
def test_generate_blocks_rejects_bad_committed_shape(committed):
    def bad_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + config.canvas_length,
            trajectory=trajectory,
        )

    with pytest.raises(ValueError, match="block.committed"):
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
    ("committed", "message"),
    [
        (torch.tensor([[1.5, 2.0, 3.0, 4.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2, 3, 4]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1, 3, 4]], dtype=torch.long), "fit int32"),
    ],
)
def test_generate_blocks_rejects_invalid_committed_token_ids(committed, message):
    def bad_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + config.canvas_length,
            trajectory=trajectory,
        )

    with pytest.raises(ValueError, match=message):
        generate_blocks(
            "model",
            "logits",
            prompt_len=32,
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn=lambda *args: "canvas",
            block_fn=bad_block,
        )


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
    ("stop_token_ids", "message"),
    [
        ("9", "stop_token_ids"),
        (True, "stop_token_ids"),
        (-1, "non-negative"),
        (torch.iinfo(torch.int32).max + 1, "fit int32"),
        ([9, True], "stop_token_ids"),
        ([9, -1], "non-negative"),
        ([9, torch.iinfo(torch.int32).max + 1], "fit int32"),
    ],
)
def test_generate_blocks_rejects_invalid_stop_token_ids(stop_token_ids, message):
    committed = torch.tensor([[1, 2]], dtype=torch.long)

    def fake_block(tt_model, logits_fn, init_canvas, config, **kwargs):
        trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])
        return GeneratedBlock(
            committed=committed,
            next_pos=kwargs["start_pos"] + committed.shape[1],
            trajectory=trajectory,
        )

    with pytest.raises(ValueError, match=message):
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


def test_generate_from_prompt_tokens_rejects_bad_prompt_shape_before_zero_block_fast_path():
    with pytest.raises(ValueError, match="prompt_tokens must have shape"):
        generate_from_prompt_tokens(
            "model",
            None,
            torch.tensor([1, 2, 3], dtype=torch.long),
            num_blocks=0,
            config=DiffusionConfig(canvas_length=4),
            prefill_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "message"),
    [
        (torch.empty((0, 3), dtype=torch.long), "batch size"),
        (torch.empty((1, 0), dtype=torch.long), "length"),
    ],
)
def test_generate_from_prompt_tokens_rejects_empty_prompt_tokens_before_zero_block_fast_path(prompt_tokens, message):
    with pytest.raises(ValueError, match=message):
        generate_from_prompt_tokens(
            "model",
            None,
            prompt_tokens,
            num_blocks=0,
            config=DiffusionConfig(canvas_length=4),
            prefill_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "message"),
    [
        (torch.tensor([[1.5, 2.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_generate_from_prompt_tokens_rejects_invalid_token_ids_before_zero_block_fast_path(prompt_tokens, message):
    with pytest.raises(ValueError, match=message):
        generate_from_prompt_tokens(
            "model",
            None,
            prompt_tokens,
            num_blocks=0,
            config=DiffusionConfig(canvas_length=4),
            prefill_fn=lambda *args, **kwargs: None,
        )


def test_generate_from_prompt_tokens_rejects_logits_and_builder_together():
    with pytest.raises(ValueError, match="either logits_fn or logits_fn_builder"):
        generate_from_prompt_tokens(
            "model",
            "logits",
            torch.tensor([[1, 2]], dtype=torch.long),
            num_blocks=1,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn="init",
            logits_fn_builder=lambda *args, **kwargs: "built",
            prefill_fn=lambda *args, **kwargs: 2,
            blocks_fn=lambda *args, **kwargs: None,
        )


def test_generate_from_prompt_tokens_rejects_logits_and_builder_together_for_zero_blocks():
    def fail_prefill(*args, **kwargs):
        raise AssertionError("prefill should not run for conflicting logits inputs")

    with pytest.raises(ValueError, match="either logits_fn or logits_fn_builder"):
        generate_from_prompt_tokens(
            "model",
            "logits",
            torch.tensor([[1, 2]], dtype=torch.long),
            num_blocks=0,
            config=DiffusionConfig(canvas_length=2),
            logits_fn_builder=lambda *args, **kwargs: "built",
            prefill_fn=fail_prefill,
            blocks_fn=lambda *args, **kwargs: None,
        )


def test_generate_from_prompt_tokens_rejects_negative_num_blocks_before_prefill():
    def fail_prefill(*args, **kwargs):
        raise AssertionError("prefill should not run for invalid num_blocks")

    with pytest.raises(ValueError, match="num_blocks must be non-negative"):
        generate_from_prompt_tokens(
            "model",
            "logits",
            torch.tensor([[1, 2]], dtype=torch.long),
            num_blocks=-1,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn="init",
            prefill_fn=fail_prefill,
            blocks_fn=lambda *args, **kwargs: None,
        )


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


def test_generate_text_rejects_negative_num_blocks_before_tokenize():
    class _FailTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not run for invalid num_blocks")

    with pytest.raises(ValueError, match="num_blocks must be non-negative"):
        generate_text(
            "model",
            "logits",
            _FailTokenizer(),
            "hello",
            num_blocks=-1,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn="init",
            blocks_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("num_blocks", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_text_rejects_non_integer_num_blocks_before_tokenize(num_blocks, message):
    class _FailTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not run for invalid num_blocks")

    with pytest.raises(ValueError, match=message):
        generate_text(
            "model",
            "logits",
            _FailTokenizer(),
            "hello",
            num_blocks=num_blocks,
            config=DiffusionConfig(canvas_length=2),
            init_canvas_fn="init",
            blocks_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_rejects_non_positive_canvas_length_before_tokenize():
    class _FailTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not run for invalid canvas_length")

    with pytest.raises(ValueError, match="canvas_length must be positive"):
        generate_text(
            "model",
            "logits",
            _FailTokenizer(),
            "hello",
            num_blocks=1,
            config=DiffusionConfig(canvas_length=0),
            init_canvas_fn="init",
            blocks_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_rejects_max_new_tokens_beyond_block_capacity_before_tokenize():
    class _FailTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not run for impossible length budget")

    with pytest.raises(ValueError, match="num_blocks is too small"):
        generate_text(
            "model",
            "logits",
            _FailTokenizer(),
            "hello",
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            max_new_tokens=5,
            blocks_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("max_new_tokens", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_text_rejects_non_integer_max_new_tokens_before_tokenize(max_new_tokens, message):
    class _FailTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not run for invalid max_new_tokens")

    with pytest.raises(ValueError, match=message):
        generate_text(
            "model",
            "logits",
            _FailTokenizer(),
            "hello",
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            max_new_tokens=max_new_tokens,
            blocks_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_allows_zero_blocks_without_init_canvas():
    tokenizer = _FakeChatTokenizer()

    def fake_prefill(tt_model, tokens, *, page_table=None, page_tables_per_layer=None):
        raise AssertionError("prefill should not run for zero generated blocks")

    def fake_blocks(tt_model, logits_fn, **kwargs):
        raise AssertionError("blocks should not run for zero generated blocks")

    out = generate_text(
        "model",
        None,
        tokenizer,
        "hello",
        num_blocks=0,
        config=DiffusionConfig(canvas_length=4),
        max_new_tokens=0,
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out.text == [""]
    assert torch.equal(out.generation.generated, torch.empty((1, 0), dtype=torch.long))
    assert out.generation.prompt_len == 3


def test_generate_text_can_build_logits_after_prompt_prefill():
    calls = []
    tokenizer = _FakeChatTokenizer()
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
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

    out = generate_text(
        "model",
        None,
        tokenizer,
        "hello",
        num_blocks=1,
        config=DiffusionConfig(canvas_length=2),
        init_canvas_fn="init",
        page_table="page-table",
        page_tables_per_layer=["layer-pages"],
        logits_fn_builder=fake_builder,
        prefill_fn=fake_prefill,
        blocks_fn=fake_blocks,
    )

    assert out.text == ["4 5"]
    assert calls[0][0] == "prefill"
    assert calls[1][0:2] == ("builder", "model")
    builder_kwargs = calls[1][2]
    assert torch.equal(builder_kwargs["prompt_tokens"], torch.tensor([[1, 1, 99]], dtype=torch.long))
    assert builder_kwargs["prompt_len"] == 3
    assert builder_kwargs["page_table"] == "page-table"
    assert builder_kwargs["page_tables_per_layer"] == ["layer-pages"]
    assert calls[2][0:3] == ("blocks", "model", "built-logits")


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
    assert calls["builder"] == ({"raw": "state"}, {"adapter": "kwarg"})
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
    assert calls["builder"] == ({"raw": "state"}, {"config": "model-config"})


def test_generate_text_from_checkpoint_state_preserves_explicit_adapter_config():
    calls = {}

    class _Model:
        hf_config = "model-config"

    def fake_builder_factory(dg_state_dict, **kwargs):
        calls["builder"] = (dg_state_dict, kwargs)
        return "builder"

    generate_text_from_checkpoint_state(
        _Model(),
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        adapter_kwargs={"config": "explicit-config", "dtype": "bf16"},
        logits_fn_builder_factory=fake_builder_factory,
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["builder"] == ({"raw": "state"}, {"config": "explicit-config", "dtype": "bf16"})


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


@pytest.mark.parametrize(
    ("max_new_tokens", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_text_from_checkpoint_state_rejects_non_integer_max_new_tokens(max_new_tokens, message):
    def fail_builder_factory(*args, **kwargs):
        raise AssertionError("logits builder should not run for invalid max_new_tokens")

    with pytest.raises(ValueError, match=message):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            max_new_tokens=max_new_tokens,
            logits_fn_builder_factory=fail_builder_factory,
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_allows_zero_new_tokens_without_canvas_or_logits():
    calls = {}

    class _Tokenizer:
        def __len__(self):
            raise AssertionError("vocab inference is not needed for zero generated blocks")

    class _Model:
        @property
        def vocab_size(self):
            raise AssertionError("model vocab inference is not needed for zero generated blocks")

    def fail_builder_factory(*args, **kwargs):
        raise AssertionError("logits builder is not needed for zero generated blocks")

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = (tt_model, logits_fn, tokenizer, prompt, kwargs)
        return "result"

    out = generate_text_from_checkpoint_state(
        _Model(),
        _Tokenizer(),
        "hello",
        dg_state_dict={"raw": "state"},
        max_new_tokens=0,
        noise_seed=123,
        gumbel_seed=456,
        logits_fn_builder_factory=fail_builder_factory,
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"][1] is None
    assert calls["generate"][3] == "hello"
    kwargs = calls["generate"][4]
    assert kwargs["num_blocks"] == 0
    assert kwargs["logits_fn_builder"] is None
    assert callable(kwargs["init_canvas_fn"])


def test_generate_text_from_checkpoint_state_defaults_diffusion_config():
    calls = {}

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    out = generate_text_from_checkpoint_state(
        "model",
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        init_canvas_fn="init",
        max_new_tokens=257,
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["config"] == DiffusionConfig()
    assert calls["generate"]["num_blocks"] == 2


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


def test_generate_text_from_checkpoint_state_uses_tokenizer_vocab_size_for_seeded_hooks(monkeypatch):
    calls = {}

    class _Model:
        mesh_device = "mesh"

    class _Tokenizer:
        vocab_size = 99

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = (mesh_device, kwargs)
        return "init"

    def fake_noise_tokens_fn(mesh_device, **kwargs):
        calls["noise_tokens"] = (mesh_device, kwargs)
        return "noise"

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)
    monkeypatch.setattr(G, "make_seeded_host_noise_tokens_fn", fake_noise_tokens_fn)

    generate_text_from_checkpoint_state(
        _Model(),
        _Tokenizer(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        seed=123,
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["canvas_init"][1]["vocab_size"] == 99
    assert calls["noise_tokens"][1]["vocab_size"] == 99


def test_generate_text_from_checkpoint_state_can_use_tokenizer_len_for_vocab_size(monkeypatch):
    calls = {}

    class _Model:
        mesh_device = "mesh"

    class _Tokenizer:
        def __len__(self):
            return 101

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = (mesh_device, kwargs)
        return "init"

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)

    generate_text_from_checkpoint_state(
        _Model(),
        _Tokenizer(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        seed=123,
        noise_tokens_fn="noise",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["canvas_init"][1]["vocab_size"] == 101


def test_generate_text_from_checkpoint_state_can_use_model_vocab_size(monkeypatch):
    calls = {}

    class _Model:
        mesh_device = "mesh"
        vocab_size = 103

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = (mesh_device, kwargs)
        return "init"

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)

    generate_text_from_checkpoint_state(
        _Model(),
        object(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        seed=123,
        noise_tokens_fn="noise",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["canvas_init"][1]["vocab_size"] == 103


def test_generate_text_from_checkpoint_state_can_use_model_config_vocab_size(monkeypatch):
    calls = {}

    class _Config:
        vocab_size = 105

    class _Model:
        mesh_device = "mesh"
        hf_config = _Config()

    def fake_canvas_init_fn(mesh_device, **kwargs):
        calls["canvas_init"] = (mesh_device, kwargs)
        return "init"

    monkeypatch.setattr(G, "make_seeded_host_canvas_init_fn", fake_canvas_init_fn)

    generate_text_from_checkpoint_state(
        _Model(),
        object(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        seed=123,
        noise_tokens_fn="noise",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=lambda *args, **kwargs: "result",
    )

    assert calls["canvas_init"][1]["vocab_size"] == 105


def test_generate_text_from_checkpoint_state_preserves_explicit_noise_tokens(monkeypatch):
    calls = {}

    class _Model:
        mesh_device = "mesh"

    def fail_noise_tokens_fn(*args, **kwargs):
        raise AssertionError("explicit noise_tokens_fn should not be replaced")

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    monkeypatch.setattr(G, "make_seeded_host_noise_tokens_fn", fail_noise_tokens_fn)

    out = generate_text_from_checkpoint_state(
        _Model(),
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        vocab_size=99,
        seed=123,
        noise_tokens_fn="explicit-noise",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["noise_tokens_fn"] == "explicit-noise"


def test_generate_text_from_checkpoint_state_preserves_explicit_gumbel_noise(monkeypatch):
    calls = {}

    class _Model:
        mesh_device = "mesh"

    def fail_gumbel_noise_fn(*args, **kwargs):
        raise AssertionError("explicit gumbel_noise_fn should not be replaced")

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    monkeypatch.setattr(G, "make_seeded_gumbel_noise_fn", fail_gumbel_noise_fn)

    out = generate_text_from_checkpoint_state(
        _Model(),
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        vocab_size=99,
        seed=123,
        gumbel_noise_fn="explicit-gumbel",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["gumbel_noise_fn"] == "explicit-gumbel"


def test_generate_text_from_checkpoint_state_uses_tokenizer_eos_by_default():
    calls = {}

    class _Tokenizer:
        eos_token_id = 9

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    out = generate_text_from_checkpoint_state(
        "model",
        _Tokenizer(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["eos_token_id"] == 9


def test_generate_text_from_checkpoint_state_preserves_explicit_eos():
    calls = {}

    class _Tokenizer:
        eos_token_id = 9

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    generate_text_from_checkpoint_state(
        "model",
        _Tokenizer(),
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        eos_token_id=None,
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert calls["generate"]["eos_token_id"] is None


def test_generate_text_from_checkpoint_state_defaults_to_skip_special_tokens():
    calls = {}

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    out = generate_text_from_checkpoint_state(
        "model",
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert out == "result"
    assert calls["generate"]["decode_kwargs"] == {"skip_special_tokens": True}


def test_generate_text_from_checkpoint_state_preserves_explicit_decode_kwargs():
    calls = {}

    def fake_generate_text(tt_model, logits_fn, tokenizer, prompt, **kwargs):
        calls["generate"] = kwargs
        return "result"

    generate_text_from_checkpoint_state(
        "model",
        "tokenizer",
        "hello",
        dg_state_dict={"raw": "state"},
        num_blocks=1,
        config=DiffusionConfig(canvas_length=4),
        init_canvas_fn="init",
        decode_kwargs={"clean_up_tokenization_spaces": False},
        logits_fn_builder_factory=lambda *args, **kwargs: "builder",
        generate_text_fn=fake_generate_text,
    )

    assert calls["generate"]["decode_kwargs"] == {"clean_up_tokenization_spaces": False}


def test_generate_text_from_checkpoint_state_requires_canvas_source():
    with pytest.raises(ValueError, match="init_canvas_fn"):
        generate_text_from_checkpoint_state(
            object(),
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_requires_length_budget():
    with pytest.raises(ValueError, match="num_blocks"):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_non_positive_canvas_length():
    with pytest.raises(ValueError, match="canvas_length must be positive"):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            config=DiffusionConfig(canvas_length=0),
            init_canvas_fn="init",
            max_new_tokens=1,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_negative_num_blocks():
    with pytest.raises(ValueError, match="num_blocks must be non-negative"):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=-1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("num_blocks", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_text_from_checkpoint_state_rejects_non_integer_num_blocks(num_blocks, message):
    def fail_builder_factory(*args, **kwargs):
        raise AssertionError("logits builder should not run for invalid num_blocks")

    with pytest.raises(ValueError, match=message):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=num_blocks,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            logits_fn_builder_factory=fail_builder_factory,
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_max_new_tokens_beyond_explicit_blocks():
    with pytest.raises(ValueError, match="num_blocks is too small"):
        generate_text_from_checkpoint_state(
            "model",
            "tokenizer",
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            max_new_tokens=5,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_nonpositive_batch_before_seeded_hooks():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            noise_tokens_fn="noise",
            vocab_size=99,
            gumbel_seed=123,
            batch=0,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_generate_text_from_checkpoint_state_rejects_non_integer_batch_before_seeded_hooks(batch, message):
    with pytest.raises(ValueError, match=message):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            noise_tokens_fn="noise",
            vocab_size=99,
            batch=batch,
            gumbel_seed=123,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_nonpositive_vocab_before_seeded_hooks():
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            noise_tokens_fn="noise",
            vocab_size=0,
            gumbel_seed=123,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_requires_vocab_for_seeded_noise():
    with pytest.raises(ValueError, match="noise_tokens_fn requires vocab_size"):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            seed=123,
            gumbel_noise_fn="gumbel",
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_requires_vocab_for_seeded_gumbel():
    with pytest.raises(ValueError, match="gumbel_noise_fn requires vocab_size"):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            gumbel_seed=123,
            noise_tokens_fn="noise",
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


def test_generate_text_from_checkpoint_state_rejects_nonpositive_gumbel_seed():
    with pytest.raises(ValueError, match="positive nonzero"):
        generate_text_from_checkpoint_state(
            object(),
            object(),
            "hello",
            dg_state_dict={"raw": "state"},
            num_blocks=1,
            config=DiffusionConfig(canvas_length=4),
            init_canvas_fn="init",
            noise_tokens_fn="noise",
            vocab_size=99,
            gumbel_seed=0,
            logits_fn_builder_factory=lambda *args, **kwargs: "builder",
            generate_text_fn=lambda *args, **kwargs: None,
        )


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


@pytest.mark.parametrize(
    ("prompt_tokens", "message"),
    [
        (torch.tensor([[1.5, 2.0, 3.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2, 3]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1, 3]], dtype=torch.long), "fit int32"),
    ],
)
def test_generation_sequences_rejects_invalid_prompt_token_ids(prompt_tokens, message):
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    with pytest.raises(ValueError, match=message):
        generation_sequences(prompt_tokens, generation)


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (torch.tensor([[4.5, 5.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[4, -5]], dtype=torch.long), "non-negative"),
        (torch.tensor([[4, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_generation_sequences_rejects_invalid_generated_token_ids(generated, message):
    generation = G.DeviceGeneration(
        generated=generated,
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    with pytest.raises(ValueError, match=message):
        generation_sequences(torch.tensor([[1, 2, 3]], dtype=torch.long), generation)


def test_generation_sequences_rejects_prompt_len_mismatch():
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=4,
        next_pos=6,
        trajectories=[],
    )

    with pytest.raises(ValueError, match="generation.prompt_len"):
        generation_sequences(torch.tensor([[1, 2, 3]], dtype=torch.long), generation)


def test_generation_sequences_rejects_next_pos_mismatch():
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=7,
        trajectories=[],
    )

    with pytest.raises(ValueError, match="generation.next_pos"):
        generation_sequences(torch.tensor([[1, 2, 3]], dtype=torch.long), generation)


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


@pytest.mark.parametrize(
    ("eos_token_id", "message"),
    [
        ("9", "eos_token_id"),
        (True, "eos_token_id"),
        (-1, "non-negative"),
        (torch.iinfo(torch.int32).max + 1, "fit int32"),
        ([9, "bad"], "eos_token_id"),
        ([9, True], "eos_token_id"),
        ([9, -1], "non-negative"),
        ([9, torch.iinfo(torch.int32).max + 1], "fit int32"),
    ],
)
def test_generation_token_ids_rejects_invalid_eos_token_ids(eos_token_id, message):
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5, 9, 6]], dtype=torch.long),
        prompt_len=3,
        next_pos=7,
        trajectories=[],
    )

    with pytest.raises(ValueError, match=message):
        generation_token_ids(prompt_tokens, generation, eos_token_id=eos_token_id)


@pytest.mark.parametrize(
    ("max_new_tokens", "message"),
    [
        (-1, "non-negative"),
        (1.5, "integer"),
        (True, "integer"),
    ],
)
def test_decode_generation_rejects_invalid_max_new_tokens(max_new_tokens, message):
    tokenizer = _FakeTokenizer()
    prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generation = G.DeviceGeneration(
        generated=torch.tensor([[4, 5]], dtype=torch.long),
        prompt_len=3,
        next_pos=5,
        trajectories=[],
    )

    with pytest.raises(ValueError, match=message):
        decode_generation(tokenizer, prompt_tokens, generation, max_new_tokens=max_new_tokens)


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


@pytest.mark.parametrize(
    ("canvas_tokens", "message"),
    [
        (torch.tensor([[1.5, 2.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_host_canvas_to_device_rejects_invalid_token_ids(canvas_tokens, message):
    with pytest.raises(ValueError, match=message):
        host_canvas_to_device("mesh", canvas_tokens)


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


def test_host_gumbel_noise_to_device_rejects_bad_shape():
    with pytest.raises(ValueError, match="gumbel_noise"):
        host_gumbel_noise_to_device("mesh", torch.zeros(1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="gumbel_noise"):
        host_gumbel_noise_to_device("mesh", torch.zeros(1, 2, 4, 6))


def test_host_tokens_to_device_uses_embedding_token_layout(monkeypatch):
    calls = {}

    class _FakeTtnn:
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


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (torch.tensor([[1.5, 2.0]], dtype=torch.float32), "integer token ids"),
        (torch.tensor([[1, -2]], dtype=torch.long), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_host_tokens_to_device_rejects_invalid_token_ids(tokens, message):
    with pytest.raises(ValueError, match=message):
        host_tokens_to_device("mesh", tokens)


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


def test_tokenize_prompt_preserves_empty_string_user_message():
    tokenizer = _FakeChatTokenizer()

    out = tokenize_prompt(tokenizer, "")

    assert torch.equal(out, torch.tensor([[1, 1, 99]], dtype=torch.long))
    assert tokenizer.calls == [([{"role": "user", "content": ""}], True, True)]


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


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (torch.empty((0, 3), dtype=torch.long), "batch size"),
        (torch.empty((1, 0), dtype=torch.long), "length"),
    ],
)
def test_tokenize_prompt_rejects_empty_token_ids(token_ids, message):
    with pytest.raises(ValueError, match=message):
        tokenize_prompt(object(), token_ids)


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (torch.tensor([[1.5, 2.0]], dtype=torch.float32), "integers"),
        (torch.tensor([[1, -2]], dtype=torch.int32), "non-negative"),
        (torch.tensor([[1, torch.iinfo(torch.int32).max + 1]], dtype=torch.long), "fit int32"),
    ],
)
def test_tokenize_prompt_rejects_invalid_token_ids(token_ids, message):
    with pytest.raises(ValueError, match=message):
        tokenize_prompt(object(), token_ids)


def test_prefill_prompt_tokens_embeds_and_writes_kv(monkeypatch):
    calls = {}

    class _FakeDeviceTensor:
        def __init__(self, name):
            self.name = name
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
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

    class _FakeModel:
        mesh_device = object()
        hidden_size = 16

        def embed_tokens(self, tt_tokens):
            calls["embed_tokens"] = tt_tokens
            return _FakeDeviceTensor("embeds")

        def __call__(self, hidden_states, **kwargs):
            calls["model"] = (hidden_states, kwargs)
            return _FakeDeviceTensor("logits")

    monkeypatch.setattr(G, "ttnn", _FakeTtnn)
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)

    out = prefill_prompt_tokens(_FakeModel(), prompt_tokens, page_tables_per_layer=["pages"])

    assert out == 3
    assert calls["embed_tokens"].deallocated is True
    assert calls["reshape"][1] == (1, 1, 3, 16)
    hidden_states, kwargs = calls["model"]
    assert hidden_states.name == "tile-embeds"
    assert kwargs["is_decode"] is False
    assert kwargs["input_ids_torch"] is prompt_tokens
    assert kwargs["kv_phase"] is G.KVCachePhase.PREFILL_WRITE
    assert kwargs["page_tables_per_layer"] == ["pages"]


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


def test_make_host_canvas_init_fn_rejects_bad_replay_shapes():
    with pytest.raises(ValueError, match="host_canvases"):
        make_host_canvas_init_fn("mesh", [torch.tensor([4, 5])])
    with pytest.raises(ValueError, match="host_canvases"):
        make_host_canvas_init_fn("mesh", [torch.tensor([[4, 5]]), torch.tensor([[6, 7, 8]])])


def test_make_host_canvas_init_fn_rejects_bad_block_index():
    init_fn = make_host_canvas_init_fn("mesh", [torch.tensor([[4, 5]])])

    with pytest.raises(IndexError, match="host canvas replay block index 1 out of range for 1 blocks"):
        init_fn(1, 32)


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch": 1.5}, "batch"),
        ({"batch": True}, "batch"),
        ({"canvas_len": 1.5}, "canvas_len"),
        ({"canvas_len": True}, "canvas_len"),
        ({"vocab_size": 1.5}, "vocab_size"),
        ({"vocab_size": True}, "vocab_size"),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        make_seeded_host_canvas_init_fn,
        make_seeded_host_noise_tokens_fn,
        make_seeded_gumbel_noise_fn,
    ],
)
def test_seeded_generation_helpers_reject_non_integer_shapes(factory, kwargs, message):
    args = {"batch": 1, "canvas_len": 4, "vocab_size": 16, "seed": 123}
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        factory("mesh", **args)


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


def test_make_host_noise_tokens_fn_rejects_bad_shapes():
    with pytest.raises(ValueError, match="host_canvases"):
        make_host_noise_tokens_fn("mesh", [[torch.tensor([1, 2, 3])]])
    with pytest.raises(ValueError, match="host_canvases"):
        make_host_noise_tokens_fn("mesh", [[torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])]])
    with pytest.raises(ValueError, match="host_noise_tokens"):
        make_host_noise_tokens_fn("mesh", [[torch.tensor([[1, 2, 3]])], [torch.tensor([[4, 5]])]])


def test_make_host_noise_tokens_fn_rejects_bad_block_or_step_index():
    noise_fn = make_host_noise_tokens_fn("mesh", [[torch.tensor([[1, 2, 3]])]])

    with pytest.raises(IndexError, match="host noise-token replay block index 1 out of range for 1 blocks"):
        noise_fn(1)
    with pytest.raises(IndexError, match="host noise-token replay step index 1 out of range for block 0 with 1 steps"):
        noise_fn(0)(1)


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


def test_make_host_gumbel_noise_fn_rejects_bad_shape():
    with pytest.raises(ValueError, match="gumbel_noise"):
        make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 2, 3, 4, 5)]])
    with pytest.raises(ValueError, match="host_gumbel_noise"):
        make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 4, 8)], [torch.zeros(1, 4, 9)]])


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


def test_make_host_gumbel_noise_fn_rejects_bad_block_or_step_index():
    noise_fn = make_host_gumbel_noise_fn("mesh", [[torch.zeros(1, 2, 3)]])

    with pytest.raises(IndexError, match="host gumbel replay block index 1 out of range for 1 blocks"):
        noise_fn(1)
    with pytest.raises(IndexError, match="host gumbel replay step index 1 out of range for block 0 with 1 steps"):
        noise_fn(0)(1)


def test_make_seeded_gumbel_noise_fn_generates_permuted_vocab_block_step_seeds(monkeypatch):
    calls = []

    def fake_sample_gumbel_noise_with_permuted_vocab(shape, *, device, seed):
        calls.append((shape, device, seed))
        return f"gumbel-{len(calls)}"

    monkeypatch.setattr(G.TS, "sample_gumbel_noise_with_permuted_vocab", fake_sample_gumbel_noise_with_permuted_vocab)

    noise = make_seeded_gumbel_noise_fn("mesh", batch=2, canvas_len=4, vocab_size=16, seed=31)

    assert noise(0)(0) == "gumbel-1"
    assert noise(0)(1) == "gumbel-2"
    assert noise(1)(0) == "gumbel-3"
    assert calls == [
        ((2, 1, 4, 16), "mesh", 31),
        ((2, 1, 4, 16), "mesh", 32),
        ((2, 1, 4, 16), "mesh", 1_000_034),
    ]


@pytest.mark.parametrize("seed", [0, -3])
def test_make_seeded_gumbel_noise_fn_rejects_nonpositive_seed(seed):
    with pytest.raises(ValueError, match="positive nonzero"):
        make_seeded_gumbel_noise_fn("mesh", batch=1, canvas_len=4, vocab_size=16, seed=seed)
