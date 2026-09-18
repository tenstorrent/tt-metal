# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical up-front trace capture, replay, and request-reuse regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import serving
from models.experimental.diffusion_gemma.tt import traced_denoise as TD

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
_DG_CKPT_INPUT = Path(
    os.path.expanduser(
        os.environ.get(
            "DG_CKPT",
            "~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it",
        )
    )
)
_DG_CKPT_REF = _DG_CKPT_INPUT / "refs" / "main"


def _checkpoint_has_weights(path: Path) -> bool:
    single = path / "model.safetensors"
    if single.is_file():
        return True
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return False
    filenames = set(json.loads(index.read_text())["weight_map"].values())
    return bool(filenames) and all((path / filename).is_file() for filename in filenames)


def _resolve_test_checkpoint(path: Path) -> Path:
    candidates = [path]
    if _DG_CKPT_REF.is_file():
        candidates.append(path / "snapshots" / _DG_CKPT_REF.read_text().strip())
    candidates.extend(sorted((path / "snapshots").glob("*")) if (path / "snapshots").is_dir() else [])
    return next((candidate for candidate in candidates if _checkpoint_has_weights(candidate)), path)


DG_CKPT = str(_resolve_test_checkpoint(_DG_CKPT_INPUT))


class _FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def is_allocated(self):
        return not self.deallocated

    def deallocate(self, force):
        assert force is True
        self.deallocated = True


class _FakeTtnn:
    TILE_SIZE = 32
    copies = []
    executions = []
    syncs = 0

    @classmethod
    def reset(cls):
        cls.copies = []
        cls.executions = []
        cls.syncs = 0

    @staticmethod
    def clone(tensor):
        return _FakeTensor(f"clone({tensor.name})")

    @classmethod
    def copy(cls, source, destination):
        cls.copies.append((source.name, destination.name))

    @classmethod
    def execute_trace(cls, mesh, trace_id, blocking=False):
        assert mesh == "mesh"
        assert blocking is False
        cls.executions.append(trace_id)

    @classmethod
    def synchronize_device(cls, mesh):
        assert mesh == "mesh"
        cls.syncs += 1


@pytest.fixture
def fake_ttnn(monkeypatch):
    _FakeTtnn.reset()
    monkeypatch.setattr(TD, "ttnn", _FakeTtnn)
    return _FakeTtnn


def _config(**overrides):
    values = {
        "canvas_length": 32,
        "max_denoise_steps": TD.UPFRONT_DENOISE_STEPS,
        "stable_steps_to_halt": 1,
    }
    values.update(overrides)
    return DiffusionConfig(**values)


def _controller():
    return TD.UpfrontTracedDenoiseController("mesh", _config())


def test_upfront_block_reuses_the_single_controller_attribute(monkeypatch):
    instances = []

    class _Controller:
        def __init__(self, mesh, config):
            self.calls = []
            instances.append(self)

        def denoise_block(self, logits_fn, init_canvas, config, **kwargs):
            self.calls.append((logits_fn, init_canvas, config, kwargs))
            return len(self.calls)

    monkeypatch.setattr(TD, "UpfrontTracedDenoiseController", _Controller)
    logits_fn = SimpleNamespace(
        tt_model=SimpleNamespace(mesh_device="mesh"),
        _upfront_capture_phase=True,
    )
    config = _config()

    assert (
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas-0",
            config,
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )
        == 1
    )
    assert (
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas-1",
            config,
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )
        == 2
    )
    assert len(instances) == 1
    assert logits_fn._upfront_traced_denoise_controller is instances[0]


def test_replay_reuses_reveal_buffers_and_stops_on_materialized_halt(fake_ttnn, monkeypatch):
    controller = _controller()
    controller.captured = True
    controller.reveal_pmax = 1024
    controller._last_prompt_len = 32
    controller.traces = [f"trace-{step}" for step in range(48)]
    controller.canvas_buf = _FakeTensor("canvas-buf")
    controller.committed_buf = _FakeTensor("committed-buf")
    controller.gumbel_buf = _FakeTensor("gumbel-buf")
    controller.noise_buf = _FakeTensor("noise-buf")
    controller.halt_bufs = SimpleNamespace()

    events = []
    monkeypatch.setattr(controller, "_refresh_noise", lambda fn, step: events.append(("noise", step)))
    monkeypatch.setattr(controller, "_refresh_gumbel", lambda fn, step: events.append(("gumbel", step)))
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[19]], dtype=torch.long))
    halt_values = iter([(1.0, 1.0), (0.1, 0.0)])
    monkeypatch.setattr(TD, "read_halt_scalars", lambda buffers: next(halt_values))
    monkeypatch.setattr(TD, "eval_halt", lambda mean, mismatch, step, **kwargs: step == 1)

    adapter = SimpleNamespace(
        prompt_len=64,
        q_rope_offset=64,
        update_canvas_rope_buffers=lambda start: events.append(("rope", start)),
        update_reveal_mask_buffer=lambda prompt: events.append(("reveal", prompt)),
        reset_signal_buffer=lambda: events.append("signal-reset"),
    )
    init_canvas = _FakeTensor("init-canvas")

    trajectory = controller.denoise_block(
        adapter,
        init_canvas,
        controller.config,
        gumbel_noise_fn=lambda step: _FakeTensor(f"unused-gumbel-{step}"),
        noise_tokens_fn=lambda step: _FakeTensor(f"unused-noise-{step}"),
    )

    assert trajectory.num_steps == 2
    assert trajectory.halted is True
    assert torch.equal(trajectory.committed, torch.tensor([[19]]))
    assert _FakeTtnn.executions == ["trace-0", "trace-1"]
    assert controller.capture_events == 0
    assert controller.traces_captured == 0
    assert controller.adapter_rebinds == 1
    assert ("reveal", 64) in events
    assert [event for event in events if isinstance(event, tuple) and event[0] == "gumbel"] == [
        ("gumbel", 0),
        ("gumbel", 1),
    ]
    assert [event for event in events if isinstance(event, tuple) and event[0] == "noise"] == [
        ("noise", 0),
        ("noise", 1),
    ]
    assert init_canvas.deallocated


def test_session_prefill_rebinds_injected_adapter_instead_of_building(monkeypatch):
    rebound = []
    prefill_calls = []
    adapter = SimpleNamespace(rebind_prompt=lambda n, *, true_prompt_len=None: rebound.append(n))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session.tt_model = SimpleNamespace()
    session.page_table = None
    session.page_tables_per_layer = None
    session.prefill_execution_len = 128
    session.prefill_reused = False
    session.prefill_time_s = 0.0
    session._persistent_adapter = adapter
    session._logits_fn = None
    session._logits_fn_builder = lambda *args, **kwargs: pytest.fail("persistent prefill must not rebuild adapter")
    session.prompt_len = None
    session.cache_len = None
    session.next_pos = None
    session.block_idx = 0
    session.finished = False
    monkeypatch.setattr(
        serving,
        "prefill_prompt_tokens",
        lambda *args, **kwargs: (prefill_calls.append(kwargs) or SimpleNamespace(prompt_len=3, cache_len=32)),
    )

    assert session.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long)) == 32
    assert rebound == [32]
    assert session._logits_fn is adapter
    assert prefill_calls == [{"page_table": None, "page_tables_per_layer": None, "execution_len": 128}]


def _set_valid_upfront_env(monkeypatch):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "1073741824")


def test_vllm_warmup_captures_48_traces_and_detaches_persistent_adapter(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    controller = SimpleNamespace(
        captured=True,
        stats=lambda: {"capture_events": 1, "traces_captured": 48},
        release=lambda: None,
    )
    adapter = SimpleNamespace(
        use_reveal_mask=True,
        _upfront_traced_denoise_controller=controller,
        reset=lambda: None,
    )
    resets = []

    class _Session:
        def __init__(self):
            self._logits_fn = adapter

        def prefill(self, tokens):
            assert tokens.shape == (1, 1)
            return 32

        def decode_block(self):
            return SimpleNamespace(tokens=torch.zeros((1, 256), dtype=torch.long), next_pos=288)

        def trace_stats(self):
            return [controller.stats()]

        def reset(self):
            resets.append(self._logits_fn)

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = [
        SimpleNamespace(
            mesh_device=None,
            tt_kv_cache=[(SimpleNamespace(shape=(1, 1, 1024, 1)), None)],
        )
    ]
    wrapper.canvas_length = 256
    wrapper._tokenizer = SimpleNamespace(bos_token_id=2)
    wrapper._config = DiffusionConfig()
    wrapper._gumbel_mode = "device"
    wrapper._upfront = True
    wrapper._persistent_adapter = None
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._upfront_pmax = 1024
    wrapper._max_model_len = 1024
    wrapper._make_session = _Session
    monkeypatch.setattr(generator_vllm, "_dram_snapshot", lambda *args, **kwargs: {})
    metrics = []
    monkeypatch.setattr(generator_vllm, "_metric", lambda event, **fields: metrics.append((event, fields)))

    wrapper.warmup_model_prefill(None, True, True)

    assert wrapper._persistent_adapter is adapter
    assert resets == [None]
    assert metrics[0][0] == "upfront_capture"
    assert metrics[0][1]["trace_stats"] == [{"capture_events": 1, "traces_captured": 48}]


@pytest.fixture(scope="module")
def upfront_device_bundle():
    if not DEVICE_GATED:
        pytest.skip("up-front capture device tests require DG_RUN_DEVICE=1")
    if not _checkpoint_has_weights(Path(DG_CKPT)):
        pytest.skip(f"complete checkpoint weights not available at {DG_CKPT}")
    raw_trace_region = os.environ.get("DG_TRACE_REGION_SIZE", "").strip()
    if not raw_trace_region or int(raw_trace_region) <= 0:
        pytest.skip("up-front capture device tests require an explicit DG_TRACE_REGION_SIZE > 0")
    pytest.importorskip("vllm")

    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device

    p_max = int(os.environ.get("DG_DENOISE_REVEAL_PMAX", "1024"))
    old_env = {
        name: os.environ.get(name)
        for name in (
            "DG_UPFRONT_CAPTURE",
            "DG_DENOISE_REVEAL_PMAX",
            "DG_TRACE_REGION_SIZE",
            "DG_VLLM_GUMBEL_MODE",
            "DG_UPFRONT_PREFILL_WARMUP_LENS",
        )
    }
    os.environ.update(
        {
            "DG_UPFRONT_CAPTURE": "1",
            "DG_DENOISE_REVEAL_PMAX": str(p_max),
            "DG_VLLM_GUMBEL_MODE": "device",
        }
    )

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    try:
        model_kwargs = {"max_seq_len": p_max, "create_kv_cache": True}
        num_layers = os.environ.get("DG_UPFRONT_NUM_LAYERS", "1")
        if num_layers.lower() != "full":
            model_kwargs["num_layers"] = int(num_layers)
        yield build_tt_model_from_checkpoint_dir(
            mesh,
            DG_CKPT,
            tokenizer_kwargs={"local_files_only": True},
            **model_kwargs,
        )
    finally:
        _close_mesh_device(mesh)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tokenize(bundle, text: str) -> torch.Tensor:
    from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

    return tokenize_prompt(bundle.tokenizer, text)


def _aligned_prompt_lengths(prompts) -> list[int]:
    return sorted({((int(prompt.shape[1]) + 31) // 32) * 32 for prompt in prompts} | {32})


def _make_upfront_wrapper(bundle, prompts):
    from models.experimental.diffusion_gemma.tt import generator_vllm

    os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = ",".join(str(value) for value in _aligned_prompt_lengths(prompts))
    wrapper = generator_vllm.DiffusionGemmaForCausalLM(
        [bundle.tt_model],
        [bundle.model_args],
        bundle.tt_model.mesh_device,
        dg_state_dict=bundle.state_dict,
        tokenizer=bundle.tokenizer,
        config=DiffusionConfig(),
        gumbel_mode="device",
    )
    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_prefill(None, True, True)
    return wrapper


def _persistent_controller(wrapper):
    controller = wrapper._persistent_adapter._upfront_traced_denoise_controller
    stats = controller.stats()
    assert stats["capture_events"] == 1
    assert stats["traces_captured"] == TD.UPFRONT_DENOISE_STEPS
    return controller


def _serve_once(wrapper, tokens: torch.Tensor, *, num_blocks: int = 2):
    controller = _persistent_controller(wrapper)
    outputs = []
    steps = []
    halted = []
    for block_idx in range(num_blocks):
        halted_before = controller.halted_blocks
        output = (
            wrapper.prefill_forward(tokens, prompt_lens=[int(tokens.shape[1])])
            if block_idx == 0
            else wrapper.decode_forward()
        )
        outputs.append(output)
        steps.append(len(controller.last_halt_trace))
        halted.append(controller.halted_blocks > halted_before)
    wrapper.release_request(0)
    return torch.cat(outputs, dim=1), steps, halted, controller.stats()


def _serve_eager(bundle, tokens: torch.Tensor, *, num_blocks: int = 2):
    session = serving.BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=DiffusionConfig(),
        tokenizer=bundle.tokenizer,
        gumbel_mode="device",
        seed=0,
        stop_token_ids=[],
    )
    try:
        session.prefill(tokens)
        emissions = [session.decode_block() for _ in range(num_blocks)]
        return (
            torch.cat([emission.tokens.reshape(1, -1) for emission in emissions], dim=1),
            [emission.num_denoise_steps for emission in emissions],
            [emission.halted for emission in emissions],
        )
    finally:
        session.reset()


def test_device_startup_capture_reuses_one_48_trace_set(upfront_device_bundle):
    prompts = [
        _tokenize(upfront_device_bundle, "Write one sentence about rain."),
        _tokenize(upfront_device_bundle, "Explain in detail why rainbows form. " * 8),
    ]
    wrapper = _make_upfront_wrapper(upfront_device_bundle, prompts)
    try:
        initial = _persistent_controller(wrapper).stats()
        first, _, _, after_first = _serve_once(wrapper, prompts[0])
        second, _, _, after_second = _serve_once(wrapper, prompts[1])
        first_again, _, _, after_third = _serve_once(wrapper, prompts[0])

        assert torch.equal(first, first_again)
        assert not torch.equal(first, second)
        for stats in (initial, after_first, after_second, after_third):
            assert stats["capture_events"] == 1
            assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()


def test_device_upfront_matches_eager_tokens_realized_k_and_halt(upfront_device_bundle):
    prompt = _tokenize(upfront_device_bundle, "Name the capital of France.")
    eager_tokens, eager_steps, eager_halted = _serve_eager(upfront_device_bundle, prompt)

    wrapper = _make_upfront_wrapper(upfront_device_bundle, [prompt])
    try:
        upfront_tokens, upfront_steps, upfront_halted, stats = _serve_once(wrapper, prompt)
        assert torch.equal(upfront_tokens, eager_tokens)
        assert upfront_steps == eager_steps
        assert upfront_halted == eager_halted
        assert stats["capture_events"] == 1
        assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()


def test_device_two_sequential_requests_match_eager_without_recapture(upfront_device_bundle):
    prompts = [
        _tokenize(upfront_device_bundle, "Give a friendly greeting."),
        _tokenize(upfront_device_bundle, "Describe a black hole in one sentence. " * 4),
    ]
    eager = [_serve_eager(upfront_device_bundle, prompt) for prompt in prompts]

    wrapper = _make_upfront_wrapper(upfront_device_bundle, prompts)
    try:
        for prompt, (expected_tokens, expected_steps, expected_halted) in zip(prompts, eager):
            tokens, steps, halted, stats = _serve_once(wrapper, prompt)
            assert torch.equal(tokens, expected_tokens)
            assert steps == expected_steps
            assert halted == expected_halted
            assert stats["capture_events"] == 1
            assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()
