# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving tests for the DiffusionGemma vLLM adapter (#47466).

CPU tests cover the vLLM-free block-emission scaffolding (sampler modes, argmax Gumbel hook,
session validation) that ``tt/generator_vllm.py`` delegates to, plus the wrapper's own row
contract, hybrid KV cache spec and per-request failure cleanup.

The device test (``DG_RUN_DEVICE=1``) runs the reduced-surface serving driver
``tests/serving_smoke.py`` — prefill + N committed 256-token blocks with a non-256-aligned
prompt — and asserts the block-granular contract: number of blocks, tokens emitted, position
advancement by ``canvas_length``, and that a per-block metrics dict (TTFT, per-block latency,
tokens-per-block) was produced. RUN-first: text quality is NOT gated (degenerate output expected
until #48291).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

serving = pytest.importorskip("models.experimental.diffusion_gemma.tt.serving")

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
DG_CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


# --- gumbel modes and the argmax hook ---------------------------------------------------------
def test_argmax_gumbel_hook_returns_none_per_step():
    per_step = serving._argmax_gumbel_noise_fn(0)
    assert per_step(0) is None
    assert per_step(5) is None


def test_argmax_gumbel_hook_rejects_bad_block_index(expect_error):
    with expect_error(ValueError):
        serving._argmax_gumbel_noise_fn(True)  # bool is not a valid block index


# --- session construction ---------------------------------------------------------------------
class _VocablessModel:
    mesh_device = None
    hf_config = None
    vocab_size = None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"vocab_size": 262144, "gumbel_mode": "nope"}, id="unknown_gumbel_mode"),
        # No tokenizer, no vocab_size, no model vocab metadata → must raise.
        pytest.param({"gumbel_mode": "argmax"}, id="no_vocab_size_source"),
    ],
)
def test_session_rejects_bad_arguments(kwargs, expect_error):
    with expect_error(ValueError):
        serving.BlockDiffusionServingSession(_VocablessModel(), {}, **kwargs)


# --- degeneracy guard stop set ----------------------------------------------------------------
# The guard's stop set is not the session's stop policy (2026-07-28). The vLLM path sets
# stop_token_ids=[] on purpose ("vLLM owns the stop decision"). While the degeneracy guard read that
# same field, it could not tell an answer's terminal <eos> padding from a collapsed canvas, so it
# rejected the terminal block of 110 of 198 requests on tt-shield run 30285823000 and each of those
# requests lost the block its answer was in.
def _tokenizer(eos=1, specials=(0, 1, 2, 50, 106)):
    return SimpleNamespace(eos_token_id=eos, all_special_ids=list(specials), vocab_size=262144)


def test_vllm_empty_stop_policy_still_leaves_the_guard_a_stop_set():
    ids = serving._resolve_degeneracy_stop_ids(None, stop_token_ids=[], tokenizer=_tokenizer())
    assert ids, "an empty stop policy must not blind the guard"
    assert 1 in ids and 106 in ids


@pytest.mark.parametrize(
    ("explicit", "stop_token_ids", "tokenizer", "expected"),
    [
        pytest.param([7, 8], [1], _tokenizer(), {7, 8}, id="explicit_ids_win"),
        pytest.param(None, [1, 106], _tokenizer(), {1, 106}, id="session_stop_policy_when_not_empty"),
        pytest.param(None, None, SimpleNamespace(eos_token_id=1), {1}, id="scalar_eos_from_a_bare_tokenizer"),
        # A partial tokenizer must not take generation down; the guard falls back to its
        # whole-canvas rule and says so.
        pytest.param(None, [], SimpleNamespace(), None, id="nothing_knowable_degrades"),
        pytest.param(
            None,
            [],
            SimpleNamespace(all_special_ids=["<eos>"], eos_token_id=None),
            None,
            id="non_int_specials_degrade",
        ),
    ],
)
def test_degeneracy_stop_ids_resolution(explicit, stop_token_ids, tokenizer, expected):
    resolved = serving._resolve_degeneracy_stop_ids(explicit, stop_token_ids=stop_token_ids, tokenizer=tokenizer)
    if expected is None:
        assert resolved is None
    else:
        assert resolved == expected


# --- session lifecycle ------------------------------------------------------------------------
def test_session_reset_detaches_persistent_upfront_adapter_without_releasing():
    events = []
    controller = SimpleNamespace(
        release=lambda: events.append("trace_release"),
        stats=lambda: {"traces_captured": 48},
    )
    logits_fn = SimpleNamespace(
        _upfront_traced_denoise_controller=controller,
        reset=lambda: events.append("logits_reset"),
    )
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = logits_fn
    session._persistent_adapter = logits_fn
    session.next_pos = 288
    session.finished = False
    session.block_idx = 1

    assert session.trace_stats() == [{"traces_captured": 48}]
    session.reset()

    assert events == []
    assert hasattr(logits_fn, "_upfront_traced_denoise_controller")
    assert session._logits_fn is None
    assert session._persistent_adapter is None
    assert session.next_pos is None
    assert session.block_idx == 0


def test_session_reset_releases_eager_logits_state():
    events = []
    logits_fn = SimpleNamespace(reset=lambda: events.append("logits-reset"))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = logits_fn
    session._persistent_adapter = None
    session.next_pos = 288
    session.finished = True
    session.block_idx = 2

    session.reset()

    assert events == ["logits-reset"]
    assert session._logits_fn is None
    assert session._persistent_adapter is None
    assert session.next_pos is None
    assert session.finished is False
    assert session.block_idx == 0


def test_next_block_capacity_accepts_exact_boundary_after_nonaligned_prompt():
    # A 265-token prompt aligns to cache position 288; one 256-token block
    # exactly fills a 544-token model-owned cache.
    model = SimpleNamespace(max_seq_len=544)
    serving._validate_next_block_capacity(model, start_pos=288, canvas_length=256)


def test_decode_rejects_block_overrun_before_device_execution(monkeypatch, expect_error):
    # A 289-token prompt aligns to 320, so a whole 256-token canvas would end at
    # 576 and must be rejected before denoise or commit touches the device/cache.
    device_called = False

    def _unexpected_device_call(*args, **kwargs):
        nonlocal device_called
        device_called = True
        raise AssertionError("device execution must not begin")

    monkeypatch.setattr(serving, "denoise_and_commit_block", _unexpected_device_call)
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = object()
    session.next_pos = 320
    session.finished = False
    session.tt_model = SimpleNamespace(max_seq_len=544)
    session.canvas_length = 256

    with expect_error(ValueError, match=r"320 \+ 256 = 576 > 544"):
        session.decode_block()
    assert device_called is False


# --- vLLM row contract for a terminal (zero-token) emission -----------------------------------
# serving.decode_block ends a request on a degenerate canvas by returning a ZERO-token emission with
# stop=True, so the caller keeps the healthy blocks it already produced. generator_vllm reshaped that
# unconditionally into [1, canvas_length] and killed EngineCore on the first degenerate block of a
# served run (tt-shield 30269947661): "shape '[1, 256]' is invalid for input of size 0". The graceful
# path was graceful only in the smoke driver and fatal in serving, the one place it exists for.
#
# The contract these pin: every row contributes exactly one [1, canvas_length] block, a terminal
# emission fills it with the stop id, and an emission that is neither empty nor a full canvas is an
# error rather than a confusing reshape.
class _TerminalEmission:
    def __init__(self, count, block_idx=3):
        self.tokens = torch.zeros(count, dtype=torch.long)
        self.block_idx = block_idx


class _SessionStub:
    def __init__(self, stop_token_ids=None):
        self.stop_token_ids = stop_token_ids


def _wrapper(canvas_length=256):
    """A generator_vllm wrapper shell exposing only the emission-block helpers."""
    GV = pytest.importorskip("models.experimental.diffusion_gemma.tt.generator_vllm")
    wrapper = object.__new__(GV.DiffusionGemmaForCausalLM)
    wrapper.canvas_length = canvas_length
    return wrapper


@pytest.mark.parametrize(
    ("stop_token_ids", "expected_fill"),
    [
        pytest.param([106, 1], 106, id="pads_with_the_FIRST_stop_id_of_a_list"),
        pytest.param(None, 0, id="no_stop_ids_falls_back_to_zero"),
        # stop_token_ids may be a bare int rather than a sequence; indexing one would raise.
        pytest.param(106, 106, id="bare_int_stop_token_id"),
    ],
)
def test_terminal_emission_fills_the_row(stop_token_ids, expected_fill):
    """A refused canvas must still fill its [1, C] slot, not raise on reshaping 0 elements."""
    wrapper = _wrapper()
    block = wrapper._emission_block(_TerminalEmission(0), _SessionStub(stop_token_ids=stop_token_ids), row=0)
    assert block.shape == (1, 256)
    assert (block == expected_fill).all()


def test_terminal_emission_uses_tokenizer_eos_when_vllm_stop_policy_is_empty():
    """Synthetic terminal blocks must carry an id the vLLM scheduler recognizes."""
    wrapper = _wrapper()
    wrapper._tokenizer = SimpleNamespace(eos_token_id=[106, 1])

    block = wrapper._emission_block(_TerminalEmission(0), _SessionStub(stop_token_ids=[]), row=0)

    assert (block == 106).all()


def test_full_canvas_emission_passes_through_unchanged():
    wrapper = _wrapper(canvas_length=4)
    emission = _TerminalEmission(0)
    emission.tokens = torch.tensor([7, 8, 9, 10], dtype=torch.long)
    block = wrapper._emission_block(emission, _SessionStub(), row=0)
    assert block.shape == (1, 4) and block.tolist() == [[7, 8, 9, 10]]


def test_partial_emission_is_an_error_not_a_reshape(expect_error):
    """Neither empty nor a full canvas means something upstream is wrong; say so with the size."""
    wrapper = _wrapper(canvas_length=4)
    emission = _TerminalEmission(0)
    emission.tokens = torch.tensor([1, 2], dtype=torch.long)
    with expect_error(RuntimeError, match="returned 2 tokens"):
        wrapper._emission_block(emission, _SessionStub(), row=0)


# --- vLLM wrapper: KV spec, session selection, per-request cleanup -----------------------------
def test_vllm_hybrid_kv_spec_keeps_one_full_attention_head_per_device():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt.generator_vllm import DiffusionGemmaForCausalLM

    text_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        num_key_value_heads=8,
        head_dim=256,
        sliding_window=1024,
        num_global_key_value_heads=2,
        global_head_dim=512,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=text_config), dtype=torch.bfloat16),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )

    specs = DiffusionGemmaForCausalLM.get_kv_cache_spec(vllm_config)

    assert specs["model.layers.0.self_attn"].num_kv_heads == 2
    assert specs["model.layers.1.self_attn"].num_kv_heads == 1


@pytest.mark.parametrize(("flag", "expected_upfront"), [("1", True), ("0", False)])
def test_vllm_session_selects_upfront_or_eager_from_sole_model_trace_flag(monkeypatch, flag, expected_upfront):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_UPFRONT_CAPTURE", flag)
    captured = {}

    def fake_session(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(generator_vllm, "BlockDiffusionServingSession", fake_session)
    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._upfront = generator_vllm.upfront_capture_enabled()
    model.model = [SimpleNamespace()]
    model._dg_state_dict = {}
    model._config = SimpleNamespace(canvas_length=256, max_denoise_steps=48)
    model._tokenizer = None
    model._gumbel_mode = "device"
    model.canvas_length = 256

    model._make_session()

    expected = generator_vllm.upfront_traced_denoise_block if expected_upfront else None
    assert captured["denoise_block_fn"] is expected


def test_vllm_prefill_failure_resets_unregistered_session(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []

    class _Session:
        def prefill(self, prompt):
            assert prompt == "prompt"
            return 32

        def decode_block(self):
            raise RuntimeError("injected block-0 failure")

        def reset(self):
            events.append("reset")

    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._sessions = {}
    model._make_session = lambda: _Session()
    model._prompt_tokens_for_row = lambda tokens, prompt_lens, row: "prompt"

    with expect_error(RuntimeError, match="injected block-0 failure"):
        model.prefill_forward(SimpleNamespace(shape=(1, 32)))

    assert events == ["reset"]
    assert model._sessions == {}


def test_vllm_decode_failure_releases_registered_session(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    class _Session:
        finished = False

        def decode_block(self):
            raise RuntimeError("injected replay failure")

    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._sessions = {3: _Session()}
    released = []

    def release(row):
        released.append(row)
        model._sessions.pop(row)

    model.release_request = release

    with expect_error(RuntimeError, match="injected replay failure"):
        model.decode_forward()

    assert released == [3]
    assert model._sessions == {}


# --- device: block-granular emission through the serving driver --------------------------------
@pytest.mark.skipif(not DEVICE_GATED, reason="device serving smoke requires DG_RUN_DEVICE=1")
@pytest.mark.skipif(not os.path.isdir(DG_CKPT), reason=f"checkpoint not available at {DG_CKPT}")
def test_serving_smoke_emits_blocks_and_advances_position():
    from models.experimental.diffusion_gemma.tests.serving_smoke import build_arg_parser, run

    num_layers = os.environ.get("DG_VLLM_SMOKE_NUM_LAYERS", "1")
    canvas = 256
    argv = [
        "--checkpoint",
        DG_CKPT,
        "--mesh",
        os.environ.get("DG_MESH", "P150x4"),
        "--num-layers",
        num_layers,
        "--max-seq-len",
        "1024",
        "--num-blocks",
        "2",
        "--canvas-length",
        str(canvas),
        "--max-denoising-steps",
        os.environ.get("DG_VLLM_SMOKE_STEPS", "2"),
        "--gumbel-mode",
        os.environ.get("DG_VLLM_SMOKE_GUMBEL", "argmax"),
        "--local-files-only",
    ]
    args = build_arg_parser().parse_args(argv)
    metrics = run(args)

    # Block-granular contract assertions (NOT text quality — RUN-first).
    assert metrics["canvas_length"] == canvas
    assert metrics["blocks_emitted"] >= 1
    assert metrics["tokens_emitted"] == metrics["blocks_emitted"] * canvas
    # Non-aligned prompt carve-out: prompt length is not a multiple of 256.
    assert metrics["prompt_aligned_256"] is False
    # Position advanced by canvas_length per emitted block from the aligned cache_len.
    assert metrics["final_next_pos"] == metrics["cache_len"] + metrics["blocks_emitted"] * canvas
    # Per-block metrics present.
    assert metrics["ttft_s"] > 0.0
    assert metrics["mean_block_latency_s"] > 0.0
    assert metrics["tokens_per_block_per_s"] > 0.0
    assert len(metrics["per_block_latency_s"]) == metrics["blocks_emitted"]
