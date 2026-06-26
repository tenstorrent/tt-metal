# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import (
    GeneratedBlock,
    denoise_and_commit_block,
    generate_blocks,
    generate_from_prompt_tokens,
    host_canvas_to_device,
    host_tokens_to_device,
    make_host_canvas_init_fn,
    make_seeded_host_canvas_init_fn,
    make_seeded_host_noise_tokens_fn,
    prefill_prompt_tokens,
)


class _FakeLogitsFn:
    q_rope_offset = None


class _FakeMesh:
    shape = (1, 4)

    def get_num_devices(self):
        return 4


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
        calls.append((mesh_device, canvas.clone()))
        return f"device-{int(canvas[0, 0])}"

    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)
    init_fn = make_host_canvas_init_fn("mesh", [torch.tensor([[4, 5]]), torch.tensor([[6, 7]])])

    assert init_fn(0, 32) == "device-4"
    assert init_fn(1, 34) == "device-6"
    assert torch.equal(calls[0][1], torch.tensor([[4, 5]]))
    assert torch.equal(calls[1][1], torch.tensor([[6, 7]]))


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
