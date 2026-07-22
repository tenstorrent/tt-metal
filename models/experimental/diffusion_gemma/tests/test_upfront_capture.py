# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable up-front denoise-trace capture tests.

CPU coverage pins the request-boundary ownership and rebind contract. Device tests are gated by
``DG_RUN_DEVICE=1`` and require an explicit ``DG_TRACE_REGION_SIZE``; they use the direct
block-serving harness rather than a live vLLM server so trace evidence is profiler-free.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import serving
from models.experimental.diffusion_gemma.tt.denoise_forward import DenoiseLogitsAdapter, MutablePrefixKVReader
from models.experimental.diffusion_gemma.tt.traced_denoise import upfront_capture_enabled

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
BASELINE_CONTROL_GATED = os.environ.get("DG_UPFRONT_BASELINE_CONTROL", "0") == "1"
EARLY_HALT_DEVICE_GATED = os.environ.get("DG_UPFRONT_EARLY_HALT_TEST", "0") == "1"
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


def _sha256_tokens(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def _write_evidence(filename: str, payload: dict) -> None:
    root = os.environ.get("DG_UPFRONT_EVIDENCE_DIR", "").strip()
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ── CPU: flag, rebind, and ownership contract ───────────────────────────
@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True), ("true", True), ("off", False)])
def test_upfront_capture_flag_defaults_off_and_parses_truthy(monkeypatch, value, expected):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", value)
    assert upfront_capture_enabled() is expected


def test_mutable_prefix_reader_request_reset_can_shrink_only_with_fixed_span(expect_error):
    reader = object.__new__(MutablePrefixKVReader)
    reader.prompt_len = 288
    reader.read_span = 1024

    reader.reset_prompt_len(32)
    assert reader.prompt_len == 32
    reader.reset_prompt_len(992)
    assert reader.prompt_len == 992

    with expect_error(ValueError, match="exceeds reveal read span"):
        reader.reset_prompt_len(1056)
    reader.read_span = None
    with expect_error(RuntimeError, match="fixed reveal-mask read span"):
        reader.reset_prompt_len(32)


def test_adapter_rebind_refreshes_reader_mask_and_rope_in_place():
    events = []
    reader = SimpleNamespace(reset_prompt_len=lambda n: events.append(("reader", n)))
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prompt_hidden_by_layer = reader
    adapter.prompt_len = 288
    adapter.q_rope_offset = 288
    adapter.use_reveal_mask = True
    adapter._reveal_p_max = 1024
    adapter._canvas_rope_len = 256
    adapter.update_reveal_mask_buffer = lambda n: events.append(("mask", n))
    adapter.update_canvas_rope_buffers = lambda n: events.append(("rope", n))

    adapter.rebind_prompt(64)

    assert adapter.prompt_len == 64
    assert adapter.q_rope_offset == 64
    assert events == [("reader", 64), ("mask", 64), ("rope", 64)]


def test_adapter_rebind_rejects_prompt_without_one_canvas_of_pmax_capacity(expect_error):
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prompt_hidden_by_layer = SimpleNamespace(
        reset_prompt_len=lambda n: pytest.fail("capacity must be checked before reader mutation")
    )
    adapter.use_reveal_mask = True
    adapter._reveal_p_max = 1024
    adapter._canvas_rope_len = 256

    with expect_error(ValueError, match=r"800 \+ 256 = 1056 > 1024"):
        adapter.rebind_prompt(800)


def test_adapter_rebind_rejects_prefix_baked_trace(expect_error):
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.use_reveal_mask = False
    with expect_error(RuntimeError, match="DG_DENOISE_REVEAL_MASK"):
        adapter.rebind_prompt(32)


def test_session_reset_detaches_borrowed_persistent_adapter_without_releasing_it():
    events = []
    adapter = SimpleNamespace(
        _traced_denoise_controller=SimpleNamespace(release=lambda: events.append("trace_release")),
        reset=lambda: events.append("adapter_reset"),
    )
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = None
    session._persistent_adapter = None
    session.next_pos = None
    session.finished = False
    session.block_idx = 0

    session.attach_persistent_adapter(adapter)
    session._logits_fn = adapter
    session.next_pos = 288
    session.reset()

    assert events == []
    assert hasattr(adapter, "_traced_denoise_controller")
    assert session._logits_fn is None
    assert session._persistent_adapter is None


def test_session_prefill_rebinds_injected_adapter_instead_of_building(monkeypatch):
    rebound = []
    adapter = SimpleNamespace(rebind_prompt=lambda n: rebound.append(n))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session.tt_model = SimpleNamespace()
    session.page_table = None
    session.page_tables_per_layer = None
    session.prefix_cache = None
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
        lambda *args, **kwargs: SimpleNamespace(prompt_len=3, cache_len=32),
    )

    assert session.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long)) == 32
    assert rebound == [32]
    assert session._logits_fn is adapter


def _set_valid_upfront_env(monkeypatch):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_MASK", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    monkeypatch.setenv("DG_VLLM_TRACE", "1")
    monkeypatch.setenv("DG_DENOISE_LAZY_CAPTURE", "0")
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "1073741824")


def test_vllm_upfront_configuration_fails_loud_on_unsafe_combinations(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    assert generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256) == 1024

    with expect_error(RuntimeError, match="requires DG_VLLM_TRACE=1"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=False, canvas_length=256)

    monkeypatch.setenv("DG_DENOISE_REVEAL_MASK", "0")
    with expect_error(RuntimeError, match="requires DG_DENOISE_REVEAL_MASK=1"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256)

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256)

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "0")
    with expect_error(RuntimeError, match="DG_TRACE_REGION_SIZE > 0"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256)

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1000")
    with expect_error(RuntimeError, match="positive 32-token multiple"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256)

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.setenv("DG_DENOISE_LAZY_CAPTURE", "1")
    with expect_error(RuntimeError, match="all trace windows must be captured at startup"):
        generator_vllm._validate_upfront_capture_configuration(trace_enabled=True, canvas_length=256)


def test_vllm_warmup_captures_and_detaches_persistent_adapter(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    controller = SimpleNamespace(
        captured=True,
        stats=lambda: {"capture_events": 1},
        release=lambda: None,
    )
    adapter = SimpleNamespace(
        use_reveal_mask=True,
        _traced_denoise_controller=controller,
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
    wrapper._upfront = True
    wrapper._trace_enabled = True
    wrapper._persistent_adapter = None
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._upfront_pmax = 1024
    wrapper._make_session = _Session
    monkeypatch.setattr(generator_vllm, "_dram_snapshot", lambda *args, **kwargs: {})
    metrics = []
    monkeypatch.setattr(generator_vllm, "_metric", lambda event, **fields: metrics.append((event, fields)))

    wrapper.warmup_model_prefill(None, True, True)

    assert wrapper._persistent_adapter is adapter
    assert resets == [None]
    assert metrics[0][0] == "upfront_capture"
    assert metrics[0][1]["trace_stats"] == [{"capture_events": 1}]


def test_vllm_upfront_warmup_defers_capture_until_trace_phase(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._upfront = True
    wrapper._persistent_adapter = None
    wrapper._make_session = lambda: pytest.fail("compile-only warmup must not build a capture session")

    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_decode()

    assert wrapper._persistent_adapter is None


def test_vllm_upfront_compile_phase_warms_configured_prefill_lengths(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = [SimpleNamespace(mesh_device=None)]
    wrapper._upfront = True
    wrapper._upfront_pmax = 1024
    wrapper._persistent_adapter = None
    wrapper.canvas_length = 256
    monkeypatch.setenv("DG_UPFRONT_PREFILL_WARMUP_LENS", "192,160,192")
    warmed = []
    monkeypatch.setattr(
        generator_vllm,
        "prefill_prompt_tokens",
        lambda model, tokens: warmed.append(tuple(tokens.shape)),
    )
    monkeypatch.setattr(generator_vllm.ttnn, "synchronize_device", lambda mesh: None)

    wrapper.warmup_model_prefill(None, False, True)

    assert wrapper._upfront_compile_phase_seen is True
    assert wrapper._upfront_prefill_warmup_lens == frozenset({160, 192})
    assert warmed == [(1, 160), (1, 192)]


def test_vllm_upfront_trace_phase_rejects_missing_prefill_warmups(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._upfront = True
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset()

    with expect_error(RuntimeError, match="requires a compile-only warmup"):
        wrapper.warmup_model_prefill(None, True, True)


def test_vllm_upfront_prefill_rejects_unseen_aligned_length(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    class _Session:
        def attach_persistent_adapter(self, adapter):
            assert adapter == "persistent"

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._sessions = {}
    wrapper._upfront = True
    wrapper._persistent_adapter = "persistent"
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._make_session = _Session

    with expect_error(RuntimeError, match="unseen aligned prefill length 64"):
        wrapper.prefill_forward(torch.zeros((1, 33), dtype=torch.long))


def test_vllm_destructor_releases_persistent_controller_then_adapter_exactly_once():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []
    adapter = SimpleNamespace(
        _traced_denoise_controller=SimpleNamespace(release=lambda: events.append("trace_release")),
        reset=lambda: events.append("adapter_reset"),
    )
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._sessions = {}
    wrapper._persistent_adapter = adapter

    wrapper.__del__()
    wrapper.__del__()

    assert events == ["trace_release", "adapter_reset"]
    assert wrapper._persistent_adapter is None
    assert not hasattr(adapter, "_traced_denoise_controller")


# ── Device: direct wrapper/session evidence ─────────────────────────────
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
            "DG_DENOISE_REVEAL_MASK",
            "DG_DENOISE_REVEAL_PMAX",
            "DG_VLLM_TRACE",
            "DG_DENOISE_EARLY_HALT",
            "DG_DENOISE_LAZY_CAPTURE",
            "DG_UPFRONT_PREFILL_WARMUP_LENS",
        )
    }
    early_halt = os.environ.get("DG_UPFRONT_EARLY_HALT", "0")
    prefill_warmup_lens = os.environ.get("DG_UPFRONT_PREFILL_WARMUP_LENS", "32,64,320")
    os.environ.update(
        {
            "DG_UPFRONT_CAPTURE": "1",
            "DG_DENOISE_REVEAL_MASK": "1",
            "DG_DENOISE_REVEAL_PMAX": str(p_max),
            "DG_VLLM_TRACE": "1",
            "DG_DENOISE_EARLY_HALT": early_halt,
            "DG_DENOISE_LAZY_CAPTURE": "0",
            "DG_UPFRONT_PREFILL_WARMUP_LENS": prefill_warmup_lens,
        }
    )

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    try:
        model_kwargs = {"max_seq_len": p_max, "create_kv_cache": True}
        num_layers = os.environ.get("DG_UPFRONT_NUM_LAYERS", "1")
        if num_layers.lower() != "full":
            model_kwargs["num_layers"] = int(num_layers)
        bundle = build_tt_model_from_checkpoint_dir(
            mesh,
            DG_CKPT,
            tokenizer_kwargs={"local_files_only": True},
            **model_kwargs,
        )
        yield bundle
    finally:
        _close_mesh_device(mesh)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_upfront_wrapper(bundle):
    from models.experimental.diffusion_gemma.tt import generator_vllm

    config = DiffusionConfig(
        canvas_length=256,
        max_denoise_steps=int(os.environ.get("DG_UPFRONT_STEPS", "2")),
    )
    wrapper = generator_vllm.DiffusionGemmaForCausalLM(
        [bundle.tt_model],
        [bundle.model_args],
        bundle.tt_model.mesh_device,
        dg_state_dict=bundle.state_dict,
        tokenizer=bundle.tokenizer,
        config=config,
        gumbel_mode=os.environ.get("DG_UPFRONT_GUMBEL_MODE", "argmax"),
    )
    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_prefill(None, True, True)
    return wrapper


def _tokenize(bundle, text: str) -> torch.Tensor:
    from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

    return tokenize_prompt(bundle.tokenizer, text)


def _serve_once(wrapper, tokens: torch.Tensor) -> torch.Tensor:
    output = wrapper.prefill_forward(tokens, prompt_lens=[int(tokens.shape[1])])
    wrapper.release_request(0)
    return output


def _persistent_stats(wrapper) -> dict:
    adapter = wrapper._persistent_adapter
    controllers = [
        getattr(adapter, attr, None)
        for attr in (
            "_traced_denoise_controller",
            "_traced_denoise_multistep_controller",
            "_traced_early_halt_controller",
        )
    ]
    controllers = [controller for controller in controllers if controller is not None]
    assert len(controllers) == 1
    controller = controllers[0]
    return controller.stats()


def _device_evidence_config(bundle) -> dict:
    return {
        "mesh": os.environ.get("DG_MESH", "P150x4"),
        "checkpoint": DG_CKPT,
        "num_layers": len(bundle.tt_model.layers),
        "max_denoise_steps": int(os.environ.get("DG_UPFRONT_STEPS", "2")),
        "reveal_pmax": int(os.environ["DG_DENOISE_REVEAL_PMAX"]),
        "trace_region_size": int(os.environ["DG_TRACE_REGION_SIZE"]),
        "gumbel_mode": os.environ.get("DG_UPFRONT_GUMBEL_MODE", "argmax"),
    }


def _decode_committed(tokenizer, tokens: torch.Tensor) -> str:
    ids = tokens.reshape(-1).tolist()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos in ids:
        ids = ids[: ids.index(eos)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _serve_per_request_trace(bundle, tokens: torch.Tensor, config: DiffusionConfig) -> torch.Tensor:
    from models.experimental.diffusion_gemma.tt.traced_denoise import traced_denoise_block

    session = serving.BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode=os.environ.get("DG_UPFRONT_GUMBEL_MODE", "argmax"),
        seed=0,
        stop_token_ids=[],
        denoise_block_fn=traced_denoise_block,
    )
    try:
        session.prefill(tokens)
        return session.decode_block().tokens
    finally:
        session.reset()


def test_device_upfront_trace_reuses_one_capture_across_different_prompt_lengths(upfront_device_bundle):
    wrapper = _make_upfront_wrapper(upfront_device_bundle)
    try:
        prompt_a = _tokenize(upfront_device_bundle, "Write one sentence about rain.")
        # Longer than mock_cache_len + canvas (32 + 256): real prefill must overwrite the
        # mock commit span, while the reveal mask hides only the remaining tail.
        prompt_b = _tokenize(upfront_device_bundle, "Explain in detail why rainbows form. " * 40)
        aligned_a = ((prompt_a.shape[1] + 31) // 32) * 32
        aligned_b = ((prompt_b.shape[1] + 31) // 32) * 32
        assert aligned_a != aligned_b
        assert aligned_b > 32 + 256

        output_a = _serve_once(wrapper, prompt_a)
        stats_after_a = _persistent_stats(wrapper)
        output_b = _serve_once(wrapper, prompt_b)
        stats_after_b = _persistent_stats(wrapper)
        output_a_again = _serve_once(wrapper, prompt_a)
        stats_after_a_again = _persistent_stats(wrapper)

        assert output_a.equal(output_a_again)
        assert not output_a.equal(output_b)
        assert stats_after_a["capture_events"] == 1
        assert stats_after_b["capture_events"] == 1
        assert stats_after_a_again["capture_events"] == 1
        _write_evidence(
            "upfront_reuse_across_prompts.json",
            {
                "status": "passed",
                "config": _device_evidence_config(upfront_device_bundle),
                "aligned_prompt_lengths": [aligned_a, aligned_b, aligned_a],
                "committed_sha256": [
                    _sha256_tokens(output_a),
                    _sha256_tokens(output_b),
                    _sha256_tokens(output_a_again),
                ],
                "trace_stats": stats_after_a_again,
            },
        )
    finally:
        wrapper.release_persistent_capture()


def test_device_upfront_commit_is_bit_exact_to_per_request_trace_and_eager(upfront_device_bundle):
    from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block as eager_denoise_block
    from models.experimental.diffusion_gemma.tt.traced_denoise import traced_denoise_block

    prompt = _tokenize(upfront_device_bundle, "Name the capital of France.")
    config = DiffusionConfig(canvas_length=256, max_denoise_steps=int(os.environ.get("DG_UPFRONT_STEPS", "2")))
    gumbel_mode = os.environ.get("DG_UPFRONT_GUMBEL_MODE", "argmax")

    traced_session = serving.BlockDiffusionServingSession(
        upfront_device_bundle.tt_model,
        upfront_device_bundle.state_dict,
        config=config,
        tokenizer=upfront_device_bundle.tokenizer,
        gumbel_mode=gumbel_mode,
        seed=0,
        stop_token_ids=[],
        denoise_block_fn=traced_denoise_block,
    )
    try:
        traced_session.prefill(prompt)
        per_request = traced_session.decode_block().tokens
    finally:
        traced_session.reset()

    eager_session = serving.BlockDiffusionServingSession(
        upfront_device_bundle.tt_model,
        upfront_device_bundle.state_dict,
        config=config,
        tokenizer=upfront_device_bundle.tokenizer,
        gumbel_mode=gumbel_mode,
        seed=0,
        stop_token_ids=[],
        denoise_block_fn=eager_denoise_block,
    )
    try:
        eager_session.prefill(prompt)
        eager = eager_session.decode_block().tokens
    finally:
        eager_session.reset()

    # Run the model-lifetime capture last: release_persistent_capture is terminal shutdown,
    # so no device work may follow it in this process.
    wrapper = _make_upfront_wrapper(upfront_device_bundle)
    try:
        upfront = _serve_once(wrapper, prompt)
    finally:
        wrapper.release_persistent_capture()

    hashes = {
        "upfront": _sha256_tokens(upfront),
        "per_request_reveal_trace": _sha256_tokens(per_request),
        "eager": _sha256_tokens(eager),
    }
    assert len(set(hashes.values())) == 1
    _write_evidence(
        "upfront_bit_exactness.json",
        {
            "status": "passed",
            "config": _device_evidence_config(upfront_device_bundle),
            "committed_sha256": hashes,
        },
    )


def test_device_upfront_multi_request_smoke_has_no_stale_cross_request_state(upfront_device_bundle):
    wrapper = _make_upfront_wrapper(upfront_device_bundle)
    try:
        prompt_a_text = "Give a friendly greeting."
        prompt_b_text = "Describe a black hole in one sentence. " * 4
        prompt_a = _tokenize(upfront_device_bundle, prompt_a_text)
        prompt_b = _tokenize(upfront_device_bundle, prompt_b_text)
        a_first = _serve_once(wrapper, prompt_a)
        b_middle = _serve_once(wrapper, prompt_b)
        a_last = _serve_once(wrapper, prompt_a)
        stats = _persistent_stats(wrapper)

        text_a = _decode_committed(upfront_device_bundle.tokenizer, a_first)
        text_b = _decode_committed(upfront_device_bundle.tokenizer, b_middle)
        assert a_first.equal(a_last)
        assert not a_first.equal(b_middle)
        assert stats["capture_events"] == 1
        assert "\ufffd" not in text_a
        assert "\ufffd" not in text_b
        if len(upfront_device_bundle.tt_model.layers) == 30:
            assert text_a.strip()
            assert text_b.strip()

        _write_evidence(
            "upfront_multi_request_smoke.json",
            {
                "status": "passed",
                "config": _device_evidence_config(upfront_device_bundle),
                "prompt_format": {
                    "tokenizer_class": type(upfront_device_bundle.tokenizer).__name__,
                    "chat_template_present": bool(getattr(upfront_device_bundle.tokenizer, "chat_template", None)),
                    "prompt_mode": "chat",
                    "rendering_method": "tokenize_prompt/apply_chat_template(add_generation_prompt=True)",
                    "prompt_a": prompt_a_text,
                    "prompt_b": prompt_b_text,
                    "prompt_a_token_ids": prompt_a.reshape(-1).tolist(),
                    "prompt_b_token_ids": prompt_b.reshape(-1).tolist(),
                },
                "a_roundtrip_exact": True,
                "a_vs_b_different": True,
                "a_sha256": _sha256_tokens(a_first),
                "b_sha256": _sha256_tokens(b_middle),
                "text_a": text_a,
                "text_b": text_b,
                "controls": None,
                "trace_stats": stats,
            },
        )
    finally:
        if wrapper is not None:
            wrapper.release_persistent_capture()


@pytest.mark.skipif(
    not EARLY_HALT_DEVICE_GATED,
    reason="up-front early-halt repro requires DG_UPFRONT_EARLY_HALT_TEST=1",
)
def test_device_upfront_early_halt_serves_two_sequential_requests(upfront_device_bundle):
    wrapper = _make_upfront_wrapper(upfront_device_bundle)
    try:
        prompts = [
            _tokenize(upfront_device_bundle, "Give a friendly greeting."),
            _tokenize(upfront_device_bundle, "Describe a black hole in one sentence. " * 4),
        ]
        capture_events = None
        for request_idx, prompt in enumerate(prompts):
            print(f"DG_UPFRONT_EH_MARK request={request_idx} prefill_forward_begin", flush=True)
            output = wrapper.prefill_forward(prompt, prompt_lens=[int(prompt.shape[1])])
            print(f"DG_UPFRONT_EH_MARK request={request_idx} prefill_forward_end", flush=True)
            stats = _persistent_stats(wrapper)
            controller = wrapper._persistent_adapter._traced_early_halt_controller
            steps = controller.last_halt_trace[-1][0]
            halted = steps < wrapper._config.max_denoise_steps
            print(
                f"DG_UPFRONT_EH_MARK request={request_idx} steps={steps} halted={halted} "
                f"capture_events={stats['capture_events']}",
                flush=True,
            )
            assert output.shape == (1, wrapper.canvas_length)
            if request_idx == 0:
                assert halted, "request 0 must halt early to exercise partial trace replay"
                capture_events = stats["capture_events"]
            else:
                assert stats["capture_events"] == capture_events
            wrapper.release_request(0)
            print(f"DG_UPFRONT_EH_MARK request={request_idx} release_end", flush=True)
    finally:
        wrapper.release_persistent_capture()


@pytest.mark.skipif(
    not BASELINE_CONTROL_GATED,
    reason="full per-request qualitative control requires DG_UPFRONT_BASELINE_CONTROL=1",
)
def test_device_per_request_prompt_b_matches_upfront_qualitative_artifact(upfront_device_bundle):
    evidence_root = os.environ.get("DG_UPFRONT_EVIDENCE_DIR", "").strip()
    if not evidence_root:
        pytest.skip("baseline comparison requires DG_UPFRONT_EVIDENCE_DIR")
    artifact_path = Path(evidence_root) / "upfront_multi_request_smoke.json"
    if not artifact_path.is_file():
        pytest.skip(f"up-front qualitative artifact not available at {artifact_path}")
    artifact = json.loads(artifact_path.read_text())

    prompt_b = _tokenize(upfront_device_bundle, "Describe a black hole in one sentence. " * 4)
    config = DiffusionConfig(
        canvas_length=256,
        max_denoise_steps=int(os.environ.get("DG_UPFRONT_STEPS", "2")),
    )
    control_b = _serve_per_request_trace(upfront_device_bundle, prompt_b, config)
    control_sha256 = _sha256_tokens(control_b)
    assert control_sha256 == artifact["b_sha256"]

    artifact["controls"] = {
        "mode": "per-request reveal-mask trace in a fresh process",
        "b_exact": True,
        "b_sha256": control_sha256,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
