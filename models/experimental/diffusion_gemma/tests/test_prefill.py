# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill: chunked long-context prefill, the paged-prefix reveal masks, and the prefill guards.

Everything here is host-only except the two ``#47466`` chunked-prefill numerical gates,
which are behind ``DG_RUN_DEVICE=1``.
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    build_canvas_reveal_denoise_mask,
    build_canvas_reveal_denoise_window_mask,
)
from models.experimental.diffusion_gemma.tt import chunked_prefill as cp
from models.experimental.diffusion_gemma.tt import ccl as dg_ccl
from models.experimental.diffusion_gemma.tt import commit_decode as CD
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import diffusion_attention as DA
from models.experimental.diffusion_gemma.tt import generate
from models.experimental.diffusion_gemma.tt import traced_denoise as TD
from models.experimental.diffusion_gemma.tt.diffusion_attention import (
    _is_distinct_buffer,
    _sdpa_q_chunked,
    _slice_rope_cache,
    validate_q_rope_offset,
)
from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model
from models.experimental.diffusion_gemma.tt.prefill_logits import discard_prefill_logits
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.utils_for_testing import assert_with_pcc

# --- chunk math ------------------------------------------------------------------------------


def test_blocks_in():
    assert cp._blocks_in(0, 64) == 0
    assert cp._blocks_in(1, 64) == 1
    assert cp._blocks_in(64, 64) == 1
    assert cp._blocks_in(65, 64) == 2
    assert cp._blocks_in(512, 64) == 8


def test_reference_page_table():
    pt = cp.make_reference_page_table(8, mesh_device=None)
    assert pt.shape == (1, 8)
    assert pt.dtype == torch.int32
    assert pt.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_chunk_page_table_slicing_matches_reference_contract():
    """Chunk c's fill table == page_table[:, chunk_start_block:chunk_end_block].

    Mirrors tt_transformers generator: chunk_start_block = start // block_size,
    chunk_end_block = ceil((start+len)/block_size). For 512 tokens as 2x256 with
    block_size 64, chunk 0 owns blocks [0:4], chunk 1 owns [4:8].
    """
    block_size, chunk_size, prompt_len = 64, 256, 512
    pt = cp.make_reference_page_table(prompt_len // block_size, mesh_device=None)
    expected = {0: [[0, 1, 2, 3]], 1: [[4, 5, 6, 7]]}
    for c in range(prompt_len // chunk_size):
        start = c * chunk_size
        end = start + chunk_size
        sb = start // block_size
        eb = cp._blocks_in(end, block_size)
        assert pt[:, sb:eb].tolist() == expected[c]


def test_chunked_prefill_adapter_forwards_shared_prefill_api(monkeypatch):
    captured = {}
    expected = object()

    def original(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(cp, "_ORIG_PREFILL_FORWARD", original)
    monkeypatch.setattr(cp, "_CHUNK_CTX", None)

    result = cp.chunked_prefill_attention_forward(
        *(object() for _ in range(8)),
        chunk_start_idx=256,
        chunk_page_table="chunk-table",
        sliding_tail_in="sliding-tail",
    )

    assert result is expected
    assert captured["chunk_start_idx"] == 256
    assert captured["chunk_page_table"] == "chunk-table"
    assert captured["sliding_tail_in"] == "sliding-tail"


def test_chunked_prefill_adapter_returns_shared_three_value_contract(monkeypatch):
    class FakeTensor:
        shape = (1, 1, 1, 1)

        def deallocate(self, *_args):
            pass

    q, projected_k, projected_v = FakeTensor(), FakeTensor(), FakeTensor()
    shared_kv = (FakeTensor(), FakeTensor())
    output = object()
    monkeypatch.setattr(
        cp,
        "_CHUNK_CTX",
        SimpleNamespace(sliding_state=SimpleNamespace()),
    )
    monkeypatch.setattr(cp, "apply_qkv_projection", lambda *_args: object())
    monkeypatch.setattr(
        cp,
        "split_qkv_heads_prefill",
        lambda *_args, **_kwargs: (q, projected_k, projected_v),
    )
    monkeypatch.setattr(cp, "apply_per_head_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(cp, "apply_rope", lambda value, *_args: value)
    monkeypatch.setattr(cp, "_bounded_sliding_sdpa", lambda *_args: output)
    monkeypatch.setattr(cp, "concat_heads", lambda value, **_kwargs: value)
    monkeypatch.setattr(cp, "apply_output_projection", lambda value, *_args: value)
    monkeypatch.setattr(cp, "apply_allreduce", lambda value, *_args: value)

    weights = SimpleNamespace(
        is_global=False,
        kv_replicated=False,
        q_norm_weight=None,
    )
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        is_sliding=True,
        sliding_window=1024,
        head_dim=32,
        hidden_size=128,
    )
    result = cp.chunked_prefill_attention_forward(
        object(),
        object(),
        object(),
        weights,
        object(),
        config,
        None,
        None,
        page_table=object(),
        shared_kv=shared_kv,
    )

    assert result == (output, None, None)


# --- device chunked prefill (#47466) ---------------------------------------------------------
#
# Prefilling a prompt in chunks must reproduce a single full-length prefill (last-token logits
# PCC >= 0.999), proving the fixes the shared gemma4 backbone lacks: the per-chunk RoPE offset
# (``chunk_start_idx``), cross-chunk attention for full-attention layers through the paged KV
# cache, and correct sliding-window attention for prompts LONGER than the sliding window via a
# bounded rolling in-memory K/V window buffer (the paged chunked SDPA op is causal-only, so it
# over-attends past the window).
#
# The vehicle is a tiny random-weight 2-layer model (one sliding + one full attention layer) so
# the check isolates the chunked-prefill logic from the 26B MoE fidelity ceiling. MoE is
# downstream of and identical between the two paths, so it does not affect chunked-vs-single
# equivalence; a tiny model with MoE off fully exercises the RoPE + KV-fill + SDPA changes.
#
# Run on QB2 (4x Blackhole):
#
#     source /home/zni/venvs/tt-diffusion-gemma/bin/activate
#     export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
#     DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest \
#       models/experimental/diffusion_gemma/tests/test_prefill.py -v -s

CHUNK_SIZE = 256
BLOCK_SIZE = 64
HIDDEN = 128
HEAD_DIM = 32
VOCAB = 256

requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run chunked-prefill correctness on a Tenstorrent device",
)


def _tiny_config(sliding_window):
    """Two-layer Gemma4 text config: one sliding + one full attention layer, MoE off."""
    layer_types = ["sliding_attention", "full_attention"]
    config = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=HEAD_DIM,
        global_head_dim=HEAD_DIM,
        layer_types=layer_types,
        sliding_window=sliding_window,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_k_eq_v=False,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    )
    config._attn_implementation = "eager"
    return config


def _to_tt_state(config):
    """Random but self-consistent tiny backbone weights, remapped to gemma4 keys."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextScaledWordEmbedding,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
                embed_scale=config.hidden_size**0.5,
            )
            self.layers = torch.nn.ModuleList(
                [Gemma4TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    m = _Tiny().eval()
    return {f"model.{k}": v for k, v in m.state_dict().items()}


def _alloc_caches(mesh_device, model, prompt_len, *, paged):
    """One [k, v] pair per layer: paged (block pool) or contiguous."""
    pac = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=prompt_len // BLOCK_SIZE) if paged else None
    caches = []
    for layer in model.layers:
        caches.append(
            init_kv_cache(
                mesh_device=mesh_device,
                config=layer.self_attn.config,
                max_batch_size=1,
                max_seq_len=prompt_len,
                paged_attention_config=pac,
            )
        )
    return caches


def _last_token_logits(tt_logits, row):
    t = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)  # [seq, vocab]
    elif t.dim() == 3:
        t = t.squeeze(0)
    return t[row, :VOCAB]


def _chunked_vs_single_pcc(device, prompt_len, sliding_window):
    """Build a tiny model, run 1×prompt_len vs (prompt_len/CHUNK_SIZE)×CHUNK_SIZE, return PCC."""
    torch.manual_seed(47466)
    tp = device.shape[1] if hasattr(device, "shape") else 1

    config = _tiny_config(sliding_window)
    model_args = Gemma4ModelArgs.from_hf_config(config)
    model_args._hf_text_config = config
    mesh_config = MeshConfig(device.shape, decode=ModeConfig(tp=tp)) if hasattr(device, "shape") else None

    state = _to_tt_state(config)
    model = DiffusionGemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=state,
        ccl_manager=CCLManager(device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=prompt_len,
        max_local_batch_size=1,
        num_layers=config.num_hidden_layers,
        create_kv_cache=False,
    )

    # Prompt embeddings [1, 1, prompt_len, HIDDEN], tile-laid.
    input_ids = torch.randint(0, VOCAB, (1, prompt_len), dtype=torch.int64)
    replicate = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "shape") else None
    tokens_tt = ttnn.from_torch(
        input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
    )
    embeds = model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, prompt_len, HIDDEN))
    embeds_single = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    # ── baseline: single 1×prompt_len prefill (contiguous cache, stock gemma4 SDPA) ──
    baseline_cache = _alloc_caches(device, model, prompt_len, paged=False)
    logits_single = model(
        embeds_single,
        is_decode=False,
        page_table=None,
        kv_caches=baseline_cache,
        input_ids_torch=input_ids,
        get_last_token=-1,
        batch_size=1,
    )
    single_last = _last_token_logits(logits_single, prompt_len - 1)
    logits_single.deallocate(True)

    # ── chunked: N×CHUNK_SIZE over a paged cache via the DG-local fixed prefill ──────
    paged_cache = _alloc_caches(device, model, prompt_len, paged=True)
    page_table_torch = cp.make_reference_page_table(prompt_len // BLOCK_SIZE, mesh_device=device)
    # embeds_single was consumed (lm_head deallocs its input chain); re-embed for the chunked run.
    embeds2 = model.embed_tokens(tokens_tt)
    embeds2 = ttnn.reshape(embeds2, (1, 1, prompt_len, HIDDEN))
    embeds_chunked = ttnn.to_layout(embeds2, ttnn.TILE_LAYOUT)
    logits_chunked = cp.chunked_prefill(
        model,
        embeds_chunked,
        input_ids_torch=input_ids,
        embeds_torch=None,
        kv_cache=paged_cache,
        page_table_torch=page_table_torch,
        block_size=BLOCK_SIZE,
        chunk_size=CHUNK_SIZE,
    )
    chunked_last = _last_token_logits(logits_chunked, CHUNK_SIZE - 1)
    logits_chunked.deallocate(True)
    embeds_chunked.deallocate(True)

    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(single_last, chunked_last, pcc=0.999)
    return single_last, chunked_last, pcc


@requires_device
@pytest.mark.use_module_device
def test_chunked_prefill_matches_single(device):
    """512 tokens as 2×256 vs 1×512, window 1024 > prompt (sliding == causal)."""
    prompt_len, sliding_window = 512, 1024
    single_last, chunked_last, pcc = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    print(f"[chunked-prefill] last-token logits PCC (2x{CHUNK_SIZE} vs 1x{prompt_len}, window {sliding_window}): {pcc}")
    assert_with_pcc(single_last, chunked_last, 0.999)


@requires_device
@pytest.mark.use_module_device
def test_chunked_prefill_sliding_past_window(device):
    """2048 tokens as 8×256 vs 1×2048, window 1024 < prompt — sliding layers EXCEED the window.

    This is the sliding-window bounded-buffer gate. The last-token (pos 2047)
    sliding query attends only (1023, 2047]; a causal-only path (the old paged
    chunked SDPA, which raised NotImplementedError here) would over-attend the
    full prefix. Matching the single 1×2048 prefill (which applies
    sliding_window_size=1024) at PCC >= 0.999 proves the rolling K/V window
    buffer applies the exact same window mask across chunk boundaries.
    """
    prompt_len, sliding_window = 2048, 1024
    assert prompt_len > sliding_window, "gate must exceed the sliding window"
    single_last, chunked_last, pcc = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    n_chunks = prompt_len // CHUNK_SIZE
    print(
        f"[chunked-prefill] SLIDING last-token logits PCC "
        f"({n_chunks}x{CHUNK_SIZE} vs 1x{prompt_len}, window {sliding_window}): {pcc}"
    )
    assert_with_pcc(single_last, chunked_last, 0.999)


# --- RoPE offset / cache-slice guards --------------------------------------------------------
#
# These guards used to live in the shared Gemma4 prefill op; they now belong to DiffusionGemma so
# the backbone stays untouched. The denoise pass is single-user, so ``validate_q_rope_offset``
# only enforces tile alignment (no batched-prefill case).

ROPE_CACHE_LEN = 262144


def _rope_cache_model(cache_len):
    return SimpleNamespace(
        hf_config=SimpleNamespace(layer_types=["sliding_attention"]),
        rope_caches={
            "sliding_attention": (
                torch.zeros(1, 1, cache_len, 8),
                torch.zeros(1, 1, cache_len, 8),
            )
        },
    )


def test_q_rope_offset_accepts_tile_aligned_offsets():
    validate_q_rope_offset(32)
    validate_q_rope_offset(0)


def test_get_rope_mats_reaches_256k():
    model = _rope_cache_model(ROPE_CACHE_LEN)
    cos, sin = DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=ROPE_CACHE_LEN)
    assert cos.shape[-2] == ROPE_CACHE_LEN
    assert sin.shape[-2] == ROPE_CACHE_LEN


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: validate_q_rope_offset(1),
            "q_rope_offset must be a multiple of 32",
            id="q_rope_offset-unaligned",
        ),
        pytest.param(
            lambda: _slice_rope_cache(None, 1, 32),
            "RoPE cache start must be a multiple of 32",
            id="rope_cache_start-unaligned",
        ),
        pytest.param(
            lambda: _slice_rope_cache(SimpleNamespace(shape=[1, 1, 262144, 8]), 262144, 32),
            r"RoPE cache slice \[262144, 262176\) exceeds cache length 262144",
            id="rope_cache_slice-past-end",
        ),
        pytest.param(
            lambda: DiffusionGemma4Model._get_rope_mats(
                _rope_cache_model(ROPE_CACHE_LEN), 0, seq_len=ROPE_CACHE_LEN + 32
            ),
            "requested RoPE seq_len 262176 exceeds cache length 262144",
            id="get_rope_mats-past-cache",
        ),
    ],
)
def test_rope_guards_reject(call, match, expect_error):
    with expect_error(ValueError, match):
        call()


def test_model_call_establishes_diffusion_activation_context(monkeypatch):
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.experimental.diffusion_gemma.tt import prefill_moe

    events = []

    @contextmanager
    def fake_context(model):
        events.append(("enter", model))
        try:
            yield
        finally:
            events.append(("exit", model))

    monkeypatch.setattr(prefill_moe, "use_tuned_prefill_moe", fake_context)
    monkeypatch.setattr(Gemma4Model, "__call__", lambda self, *args, **kwargs: (args, kwargs))
    model = object.__new__(DiffusionGemma4Model)

    assert model("hidden", is_decode=False) == (("hidden",), {"is_decode": False})
    assert events == [("enter", model), ("exit", model)]


# --- SDPA GQA fallback ----------------------------------------------------------------------


def test_sdpa_q_chunked_falls_back_to_manual_gqa_on_l1_clash(monkeypatch):
    calls = []

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

        def device(self):
            # ``_denoise_sdpa_program_config`` queries the device for the SDPA grid
            # (``DG_SDPA_GRID=device``). A host-only fake has none; ``None`` is the documented
            # input for that case and ``_resolve_sdpa_grid`` falls back to the historical pin.
            return None

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def CoreCoord(x, y):
            return ("grid", x, y)

        @staticmethod
        def SDPAProgramConfig(**kwargs):
            calls.append(("program", kwargs))
            return "program"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config=None):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(
                f"{tensor.name}[h{starts[1]}:{ends[1]},s{starts[2]}:{ends[2]}]",
                [ends[idx] - starts[idx] for idx in range(4)],
            )

        @staticmethod
        def clone(tensor, *, memory_config):
            calls.append(("clone", tensor.name, memory_config))
            return _Tensor(f"clone({tensor.name})", tensor.shape)

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            shape = list(tensors[0].shape)
            shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
            return _Tensor(f"concat{dim}", shape)

        @staticmethod
        def permute(tensor, order, *, memory_config):
            calls.append(("permute", tensor.name, order, memory_config))
            shape = [tensor.shape[idx] for idx in order]
            return _Tensor(f"permute({tensor.name})", shape)

        @staticmethod
        def matmul(lhs, rhs, *, transpose_b=False, memory_config):
            calls.append(("matmul", lhs.name, rhs.name, transpose_b, memory_config))
            rhs_name = f"transpose({rhs.name})" if transpose_b else rhs.name
            out_width = rhs.shape[-2] if transpose_b else rhs.shape[-1]
            return _Tensor(f"matmul({lhs.name},{rhs_name})", [lhs.shape[0], lhs.shape[1], lhs.shape[2], out_width])

        @staticmethod
        def softmax(tensor, *, dim, numeric_stable):
            calls.append(("softmax", tensor.name, dim, numeric_stable))
            return _Tensor(f"softmax({tensor.name})", tensor.shape)

    class _FakeTransformer:
        @staticmethod
        def scaled_dot_product_attention(q, k, v, **kwargs):
            calls.append(("sdpa", q.name, k.name, v.name, kwargs))
            raise RuntimeError("Statically allocated circular buffers in program clash with L1 buffers")

    _FakeTtnn.transformer = _FakeTransformer
    monkeypatch.setattr(DA, "ttnn", _FakeTtnn)

    out = _sdpa_q_chunked(
        _Tensor("q", [1, 4, 32, 256]),
        _Tensor("k", [1, 2, 288, 256]),
        _Tensor("v", [1, 2, 288, 256]),
        head_dim=256,
    )

    assert out.shape == [1, 4, 32, 256]
    assert [call[0] for call in calls].count("sdpa") == 1
    assert [call for call in calls if call[0] == "softmax"] == [
        ("softmax", "matmul(q[h0:2,s0:32],transpose(concat1))", -1, True),
        ("softmax", "matmul(q[h2:4,s0:32],transpose(concat1))", -1, True),
    ]


def test_sdpa_q_chunked_warns_about_gqa_fallback_once(monkeypatch):
    warnings = []

    monkeypatch.setattr(DA, "_FALLBACK_WARNED", False)
    DA.reset_sdpa_fallback_counts()
    monkeypatch.setattr(DA.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(DA, "_manual_gqa_attention", lambda q, k, v: "staged")

    def raising_sdpa(q, k, v, **kwargs):
        raise RuntimeError("Statically allocated circular buffers in program clash with L1 buffers")

    monkeypatch.setattr(DA.ttnn.transformer, "scaled_dot_product_attention", raising_sdpa)
    monkeypatch.setattr(DA, "_denoise_sdpa_program_config", lambda *args, **kwargs: "program")

    # ``device()`` returning None is what a host-only fake gives the SDPA grid resolver;
    # ``_resolve_sdpa_grid`` falls back to the historical pin for it.
    tt_q = SimpleNamespace(shape=[1, 4, 32, 256], device=lambda: None)
    tt_k = SimpleNamespace(shape=[1, 2, 288, 256])
    tt_v = SimpleNamespace(shape=[1, 2, 288, 256])

    first = _sdpa_q_chunked(tt_q, tt_k, tt_v, head_dim=256, layer_idx=5)
    second = _sdpa_q_chunked(tt_q, tt_k, tt_v, head_dim=256, layer_idx=5)

    assert first == "staged"
    assert second == "staged"
    assert len(warnings) == 1
    assert "staged GQA fallback" in warnings[0]
    assert DA.get_sdpa_fallback_counts() == {5: 2}


# --- skipping the prefill lm_head -----------------------------------------------------------
#
# ``discard_prefill_logits`` sets one attribute the shared backbone already honours, so the
# contract worth pinning is the flag lifecycle (set, restore, restore-on-raise, no-op when
# disabled) plus the fact that the two DG prefill call sites actually enter it. The device
# consequence -- no ``create_global_semaphore`` on the prefill path, so no command-queue drain --
# is covered by the prefill-ramp harness, not here.


class _Model:
    """Stand-in for Gemma4Model: only the flag matters."""


class _ClassFlagModel:
    _prefill_trace_mode = False


def test_sets_flag_inside_and_removes_it_after():
    model = _Model()
    assert not hasattr(model, "_prefill_trace_mode")
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert not hasattr(model, "_prefill_trace_mode")


def test_restores_a_preexisting_value():
    model = _Model()
    model._prefill_trace_mode = False
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is False


def test_restores_a_preexisting_true_value():
    # gemma4's traced-prefill generator sets the same flag; nesting must not
    # clear it on the way out.
    model = _Model()
    model._prefill_trace_mode = True
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is True


def test_restores_on_exception(expect_error):
    model = _Model()
    with expect_error(RuntimeError, "prefill blew up"):
        with discard_prefill_logits(model):
            assert model._prefill_trace_mode is True
            raise RuntimeError("prefill blew up")
    assert not hasattr(model, "_prefill_trace_mode")


def test_class_level_flag_is_not_left_shadowed_as_true():
    # delattr on an instance that never had its own copy raises; the fallback
    # must not leave the instance pinned to True.
    model = _ClassFlagModel()
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is False


def test_disabled_is_a_passthrough():
    model = _Model()
    with discard_prefill_logits(model, enabled=False):
        assert not hasattr(model, "_prefill_trace_mode")
    assert not hasattr(model, "_prefill_trace_mode")


def test_prefill_prompt_tokens_discards_logits():
    src = inspect.getsource(generate.prefill_prompt_tokens)
    assert "discard_prefill_logits(tt_model)" in src, (
        "DG prefill must not end in the shared lm_head: its all-gather calls "
        "create_global_semaphore, which drains the command queue once per prefill"
    )


def test_chunked_prefill_keeps_logits_only_for_the_final_chunk():
    src = inspect.getsource(cp)
    assert "discard_prefill_logits(model, enabled=not want_logits)" in src, (
        "non-final chunks discard their output, so they must skip the lm_head; "
        "the final chunk's logits are returned and must not be skipped"
    )


class _GatherSpyModel:
    mesh_device = object()
    hidden_size = 16
    embedding_weight = "w"
    embed_scale = 2.0

    class mesh_config:  # noqa: N801 - stands in for the model's mesh_config object
        tp = 4
        tp_axis = 1

    ccl_manager = "ccl-manager"


def test_embed_routes_the_tp_gather_through_dg_not_the_shared_allgather(monkeypatch):
    """The embed all-gather is the other per-prefill plain ``ttnn.all_gather``.

    ``embed_host_tokens`` is called by prefill AND by every block commit, so if this
    reverts to ``Gemma4Model.embed_tokens`` both paths go back to the semaphore-creating
    factory. Verified bit-identical on QB2 (max_abs 0 at [1,1,32,262144] and
    [1,1,8192,10240]); this test only pins the routing.
    """
    seen = {}

    class _T:
        def __init__(self, name):
            self.name = name

        def deallocate(self, force):
            seen[f"dealloc_{self.name}"] = force

    class _FakeTtnn:
        bfloat16 = "bfloat16"
        TILE_LAYOUT = "tile"

        @staticmethod
        def embedding(tokens, weight, dtype=None):
            return _T("embeds")

        @staticmethod
        def mul(value, scale):
            return _T("scaled")

        @staticmethod
        def unsqueeze_to_4D(value):
            seen["unsqueezed"] = value.name
            return _T("4d")

        @staticmethod
        def to_layout(value, layout):
            seen["tilized"] = (value.name, layout)
            return _T("tiled")

        @staticmethod
        def all_gather(*args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError("DG must not call the shared plain ttnn.all_gather")

    import models.experimental.diffusion_gemma.tt.ccl as dg_ccl

    monkeypatch.setattr(generate, "ttnn", _FakeTtnn)
    monkeypatch.setattr(dg_ccl, "ccl_allgather", lambda t, cfg, mgr, **kw: seen.setdefault("dg_gather", (t.name, mgr)))

    out = generate._embed_tokens_dg(_GatherSpyModel(), _T("tokens"))

    assert seen["unsqueezed"] == "scaled"
    assert seen["tilized"] == ("4d", "tile")
    assert seen["dg_gather"] == ("tiled", "ccl-manager")
    assert out == seen["dg_gather"]


def test_dg_allgather_rejects_composite_broadcast_fallbacks(expect_error):
    row_major = SimpleNamespace(shape=[1, 1, 32, 2816], layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(ValueError, match="composite all_broadcast"):
        dg_ccl._validate_minimal_allgather_input(row_major, 3)

    unaligned_tile = SimpleNamespace(shape=[1, 1, 32, 2815], layout=ttnn.TILE_LAYOUT)
    with expect_error(ValueError, match="tile-aligned"):
        dg_ccl._validate_minimal_allgather_input(unaligned_tile, 3)


def test_commit_and_denoise_avoid_shared_collectives():
    commit_src = inspect.getsource(CD.commit_decode_forward)
    denoise_src = inspect.getsource(DF.denoise_logits_forward)
    lm_head_src = inspect.getsource(DF._apply_denoise_lm_head)

    assert "_embed_tokens_dg(tt_model, x)" in commit_src
    assert "tt_model.embed_tokens(x)" not in commit_src
    assert "_apply_denoise_lm_head(hidden_states, tt_model)" in denoise_src
    assert "models.experimental.diffusion_gemma.tt.ccl" in lm_head_src


# --- paged-prefix reader plumbing -----------------------------------------------------------
#
# ``MutablePrefixKVReader`` read-span decoupling: once a fixed ``read_span`` (p_max) is set,
# ``__call__`` always reads p_max rows regardless of the growing committed ``prompt_len``. That
# constant read shape is what makes the trace capture-once/replay-many.


def _reader_with_recording_read_fn(prompt_len=256):
    seen = []

    def read_fn(tt_model, *, prompt_len, seq_len_start, layer_idx):
        seen.append(prompt_len)
        return ("K", "V")

    reader = DF.MutablePrefixKVReader(tt_model=object(), prompt_len=prompt_len, read_fn=read_fn)
    return reader, seen


def test_read_span_decouples_read_from_committed_len():
    reader, seen = _reader_with_recording_read_fn(prompt_len=256)
    reader.set_read_span(8192)
    reader(0)  # committed 256, but read span 8192
    reader.set_prompt_len(512)  # a block committed -> committed grows
    reader(1)
    reader.set_prompt_len(768)
    reader(2)
    assert seen == [8192, 8192, 8192], "read must be the fixed p_max span, not the growing committed len"


def test_read_span_defaults_to_prompt_len_when_unset():
    reader, seen = _reader_with_recording_read_fn(prompt_len=256)
    reader(0)
    assert seen == [256], "without a read_span the reader reads the committed prompt_len (legacy behavior)"


def test_set_prompt_len_accepts_aligned_growth_within_the_read_span():
    reader, _ = _reader_with_recording_read_fn(prompt_len=256)
    reader.set_read_span(1024)
    reader.set_prompt_len(768)
    assert reader.prompt_len == 768


@pytest.mark.parametrize(
    "read_span, setter, value",
    [
        pytest.param(1024, "set_prompt_len", 128, id="prompt_len-shrinks"),
        pytest.param(1024, "set_prompt_len", 300, id="prompt_len-unaligned"),
        pytest.param(1024, "set_prompt_len", 2048, id="prompt_len-past-read-span"),
        pytest.param(None, "set_read_span", 300, id="read_span-unaligned"),
        pytest.param(None, "set_read_span", 128, id="read_span-below-committed"),
    ],
)
def test_reader_span_guards_reject(read_span, setter, value, expect_error):
    reader, _ = _reader_with_recording_read_fn(prompt_len=256)
    if read_span is not None:
        reader.set_read_span(read_span)
    with expect_error(ValueError):
        getattr(reader, setter)(value)


class _FakeKCache:
    def __init__(self, seq):
        self.shape = [1, 8, seq, 128]


class _FakeModel:
    # `layers` + `hf_config.layer_types` are required, not decoration: the bounded sliding read
    # inspects each layer's TYPE, and since DG_DENOISE_SLIDING_SPAN was deleted (2026-07-29) that
    # path runs whenever retention is enforced -- which is the default. A fake without layers only
    # passed while the span was gated off behind its own flag.
    SLIDING_WINDOW = 1024

    def __init__(self, seq):
        self.tt_kv_cache = [(_FakeKCache(seq), _FakeKCache(seq))]
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        self.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=self.SLIDING_WINDOW)))
            for _ in layer_types
        ]
        self.hf_config = SimpleNamespace(layer_types=layer_types, sliding_window=self.SLIDING_WINDOW)


class _FakeAdapter:
    """Records the reveal plumbing calls the controller makes before capture."""

    def __init__(self, cache_seq, prompt_len):
        self.tt_model = _FakeModel(cache_seq)
        self.prompt_len = prompt_len
        self.calls = []
        self.use_reveal_mask = False
        self.prompt_hidden_by_layer = self  # acts as the reader too
        # The real adapter keeps one reveal buffer per layer type; the controller logs their keys
        # on the retention-window path, so the double has to carry them too.
        self._reveal_mask_bufs = {}

    # reader surface
    def set_read_span(self, p_max):
        self.calls.append(("set_read_span", p_max))

    def prepare_window_buffers(self, window_layers):
        self.calls.append(("prepare_window_buffers", dict(window_layers)))

    def refresh_windows(self, prompt_len):
        self.calls.append(("refresh_windows", prompt_len))

    # adapter reveal surface
    def prepare_reveal_mask_buffers(self, *, canvas_len, p_max, prompt_len, enforce_window=False, sliding_span=None):
        self.calls.append(("prepare", canvas_len, p_max, prompt_len))
        self.use_reveal_mask = True
        layer_types = ("full_attention", "sliding_attention") if enforce_window else ("full_attention",)
        self._reveal_mask_bufs = {layer_type: object() for layer_type in layer_types}

    def update_reveal_mask_buffer(self, prompt_len):
        self.calls.append(("update", prompt_len))


def test_resolve_pmax_requires_explicit_aligned_value(monkeypatch, expect_error):
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX", raising=False)
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        TD._resolve_reveal_pmax(a)

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    assert TD._resolve_reveal_pmax(a) == 4096

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4097")
    with expect_error(RuntimeError, match="positive 32-token multiple"):
        TD._resolve_reveal_pmax(a)


def test_prepare_fixed_reveal_wires_read_span_and_mask(monkeypatch):
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    assert TD._prepare_fixed_reveal(a, canvas_len=256) == 4096
    assert ("set_read_span", 4096) in a.calls
    assert ("prepare", 256, 4096, 256) in a.calls
    assert ("update", 256) in a.calls


@pytest.mark.parametrize(
    "flag, expect_window, expect_masks, expect_span",
    [
        ("0", False, ["full_attention"], None),
        ("1", True, ["full_attention", "sliding_attention"], _FakeModel.SLIDING_WINDOW),
    ],
)
def test_prepare_fixed_reveal_forwards_the_retention_flag(monkeypatch, flag, expect_window, expect_masks, expect_span):
    """The env gate has to reach `prepare_reveal_mask_buffers`, and add the sliding mask when on.

    HF's sliding layers retain only `sliding_window - 1` committed keys (#51080). With the flag off
    one shared full-attention mask serves all 30 layers; with it on the sliding layers need their
    own mask, so the buffer set is what tells the two regimes apart.

    It also carries the BOUNDED READ now. This used to assert `sliding_span is None` because the perf
    half was its own gate (DG_DENOISE_SLIDING_SPAN); that flag was deleted on 2026-07-29 because the
    bounded read is bit-identical whenever retention is enforced, so the span follows this flag and
    nothing else. Retention off must still mean no bounded read -- that is the one part of the
    bounded read which is NOT bit-identical.
    """
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", flag)
    seen = {}

    adapter = _FakeAdapter(cache_seq=8192, prompt_len=256)
    real_prepare = adapter.prepare_reveal_mask_buffers

    def recording_prepare(*, canvas_len, p_max, prompt_len, enforce_window=False, sliding_span=None):
        seen["enforce_window"] = enforce_window
        seen["sliding_span"] = sliding_span
        return real_prepare(
            canvas_len=canvas_len,
            p_max=p_max,
            prompt_len=prompt_len,
            enforce_window=enforce_window,
            sliding_span=sliding_span,
        )

    adapter.prepare_reveal_mask_buffers = recording_prepare
    TD._prepare_fixed_reveal(adapter, canvas_len=256)

    assert seen["enforce_window"] is expect_window
    assert seen["sliding_span"] == expect_span, "the bounded read must follow the retention flag"
    assert sorted(adapter._reveal_mask_bufs) == expect_masks
    allocated = [c for c in adapter.calls if c[0] == "prepare_window_buffers"]
    if expect_span is None:
        assert not allocated, "no block-resident window buffers without the retention mask"
    else:
        assert allocated and set(allocated[0][1]) == {0, 1, 2, 3, 4}, "only the sliding layers bounded"


# --- paged-prefix Phase-1 reveal mask -------------------------------------------------------
#
# The load-bearing invariants of ``build_canvas_reveal_denoise_mask`` (see
# ``doc/optimize_perf/paged_prefix_denoise_design.md``):
#
#   (1) NO LEAK — uncommitted prefix slots ``[prompt_len:p_max]`` are ALWAYS masked, at every
#       ``prompt_len`` and every ``p_max``.
#   (2) BIT-EXACT-TO-GOLDEN — the Phase-1 reveal mask, restricted to the committed key columns
#       ``[0:prompt_len] ++ canvas``, equals the current all-attend golden
#       (``build_canvas_denoise_mask``), so Phase-1 does not change any committed decision.
#   (3) FIXED SHAPE — the mask shape is ``[C, p_max+C]`` independent of ``prompt_len`` (the
#       property that makes the trace capture-once/replay-many).

NEG = float("-inf")
CANVAS = 256  # DG output block granularity


def _committed_columns(prompt_len: int, p_max: int, canvas_len: int) -> torch.Tensor:
    """Indices into the [p_max+C] key axis that correspond to committed keys."""
    prefix = torch.arange(prompt_len)  # committed prefix slots 0..prompt_len-1
    canvas = p_max + torch.arange(canvas_len)  # canvas columns live at [p_max:p_max+C]
    return torch.cat([prefix, canvas])


@pytest.mark.parametrize("prompt_len", [0, 32, 256, 288, 1024, 4096])
@pytest.mark.parametrize("p_max", [4096, 8192])
def test_fixed_shape_independent_of_prompt_len(prompt_len, p_max):
    mask = build_canvas_reveal_denoise_mask(prompt_len, CANVAS, p_max, layer_type="full_attention")
    assert tuple(mask.shape) == (CANVAS, p_max + CANVAS)


@pytest.mark.parametrize("prompt_len", [0, 32, 256, 288, 544, 1024, 4096])
@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("enforce_window", [False, True])
def test_no_leak_uncommitted_prefix_always_masked(prompt_len, layer_type, enforce_window):
    p_max = 8192
    mask = build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        p_max,
        layer_type=layer_type,
        sliding_window=1024,
        enforce_sliding_window=enforce_window,
    )
    # Every uncommitted prefix column [prompt_len:p_max] must be -inf for every canvas row.
    uncommitted = mask[:, prompt_len:p_max]
    assert (
        torch.isinf(uncommitted).all() and (uncommitted < 0).all()
    ), f"uncommitted prefix leaked at prompt_len={prompt_len} {layer_type} window={enforce_window}"


@pytest.mark.parametrize("prompt_len", [32, 256, 288, 544, 1024, 2048])
def test_phase1_full_attn_bit_exact_to_allattend_golden(prompt_len):
    """Phase-1 full-attn: committed columns must be exactly the all-attend golden (zeros)."""
    p_max = 8192
    reveal = build_canvas_reveal_denoise_mask(prompt_len, CANVAS, p_max, layer_type="full_attention")
    golden = build_canvas_denoise_mask(prompt_len, CANVAS, layer_type="full_attention")  # [C, prompt_len+C], all 0
    cols = _committed_columns(prompt_len, p_max, CANVAS)
    got = reveal[:, cols]
    assert got.shape == golden.shape
    assert torch.equal(got, golden), "Phase-1 full-attn reveal diverges from all-attend golden on committed span"


@pytest.mark.parametrize("prompt_len", [1024, 1281, 2048, 4096])
def test_phase2_sliding_matches_golden_on_committed_span(prompt_len):
    """Phase-2 sliding: committed columns must match the HF bidirectional-window golden."""
    p_max = 8192
    W = 1024
    reveal = build_canvas_reveal_denoise_mask(
        prompt_len, CANVAS, p_max, layer_type="sliding_attention", sliding_window=W, enforce_sliding_window=True
    )
    golden = build_canvas_denoise_mask(prompt_len, CANVAS, layer_type="sliding_attention", sliding_window=W)
    cols = _committed_columns(prompt_len, p_max, CANVAS)
    got = reveal[:, cols]
    # Compare mask topology (attend vs masked) rather than raw -inf bit patterns.
    assert torch.equal(
        torch.isfinite(got), torch.isfinite(golden)
    ), f"Phase-2 sliding visibility diverges from HF golden at prompt_len={prompt_len}"


def test_softmax_invariance_masked_tail_is_noop():
    """The -inf tail must contribute exactly 0 to softmax (bit-exact no-op vs the committed-only mask)."""
    torch.manual_seed(0)
    prompt_len, p_max = 288, 4096
    H, C, hd = 2, CANVAS, 64
    total = p_max + C
    scores = torch.randn(H, C, total, dtype=torch.float64)
    mask = build_canvas_reveal_denoise_mask(prompt_len, C, p_max, layer_type="full_attention", dtype=torch.float64)
    # Full masked softmax over the fixed span.
    full = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
    # Reference: softmax over ONLY the committed columns.
    cols = _committed_columns(prompt_len, p_max, C)
    ref = torch.zeros_like(full)
    ref[:, :, cols] = torch.softmax(scores[:, :, cols], dim=-1)
    assert torch.allclose(full, ref, atol=1e-12), "masked tail is not a softmax no-op"


# --- hidden prefix span: the prefill pad slots ----------------------------------------------
#
# Prefill right-pads the prompt to a tile multiple and writes K/V for the pad tokens, while the
# reveal predicate uses the PADDED length -- so those garbage keys are revealed, and they sit
# immediately before the canvas. Injecting that geometry into the reference (seeded canvas,
# otherwise identical) took q096 to the 48-step cap and q106/q095 to 35 steps; hiding the pads
# restored 20/12/11, i.e. baseline. See doc/decision_fidelity/device_gumbel_restored.md section 16.

PAD_P_MAX = 4096  # a reveal span wide enough that the pad slots sit well inside it
PAD_W = 1024  # the real sliding window, for the composition test


def test_hidden_span_hides_exactly_those_slots():
    """270 real tokens padded to 288 inside a 320-slot span: only 270..287 change."""
    today = build_canvas_reveal_denoise_mask(288, CANVAS, 320)
    fixed = build_canvas_reveal_denoise_mask(288, CANVAS, 320, hidden_prefix_span=(270, 288))
    assert (fixed[:, :270] == 0).all(), "real prompt keys must stay revealed"
    assert (fixed[:, 270:288] == NEG).all(), "pad slots must be hidden"
    assert torch.equal(fixed[:, 288:], today[:, 288:]), "uncommitted tail and canvas must be untouched"
    assert (fixed[:, 320:] == 0).all(), "canvas columns are always revealed"


def test_hidden_span_is_inert_when_absent():
    """The mechanism must not change any existing caller until one passes a span."""
    for prompt_len in (0, 32, 288, PAD_P_MAX):
        assert torch.equal(
            build_canvas_reveal_denoise_mask(prompt_len, CANVAS, PAD_P_MAX),
            build_canvas_reveal_denoise_mask(prompt_len, CANVAS, PAD_P_MAX, hidden_prefix_span=None),
        ), f"prompt_len={prompt_len}"


def test_empty_hidden_span_hides_nothing():
    """An aligned prompt has no pad slots, so lo == hi, which must be a no-op rather than an error."""
    assert torch.equal(
        build_canvas_reveal_denoise_mask(32, CANVAS, PAD_P_MAX),
        build_canvas_reveal_denoise_mask(32, CANVAS, PAD_P_MAX, hidden_prefix_span=(32, 32)),
    )


def test_hidden_span_composes_with_the_retention_window():
    """Both predicates are per-key, so hiding pads and enforcing retention must intersect.

    A key is attended only if it is committed AND not a pad AND still retained, so the pad slots
    stay hidden even when they fall inside the retained window.
    """
    prompt_len = PAD_P_MAX
    lo, hi = prompt_len - 4, prompt_len  # pads at the very end, inside any retained window
    windowed = build_canvas_reveal_denoise_mask(
        prompt_len, CANVAS, PAD_P_MAX, layer_type="sliding_attention", sliding_window=PAD_W, enforce_sliding_window=True
    )
    both = build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        PAD_P_MAX,
        layer_type="sliding_attention",
        sliding_window=PAD_W,
        enforce_sliding_window=True,
        hidden_prefix_span=(lo, hi),
    )
    assert (windowed[:, lo:hi] == 0).all(), "precondition: those keys are retained without the span"
    assert (both[:, lo:hi] == NEG).all(), "pads must be hidden even inside the retained window"
    assert torch.equal(both[:, :lo], windowed[:, :lo]), "keys outside the span must be unchanged"


# --- bounded sliding span + hidden pads -----------------------------------------------------
#
# This combination used to raise NotImplementedError; the bounded builder now takes the same
# ABSOLUTE-position span, because its key axis already carries absolute positions (prefix column
# r -> lo + r) and needs no column arithmetic.

SPAN = 1024  # tile-aligned read span for a sliding layer, sliding_read_span(1024, p_max)


def test_window_hidden_span_hides_exactly_the_pad_columns():
    """Pads at absolute 270..287 with the window still anchored at lo=0."""
    prompt_len, lo = 288, 0
    plain = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W)
    fixed = build_canvas_reveal_denoise_window_mask(
        prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=(270, 288)
    )
    assert (plain[:, 270:288] == 0).all(), "precondition: those columns are attended without the span"
    assert (fixed[:, 270:288] == NEG).all(), "pad columns must be hidden"
    assert torch.equal(fixed[:, :270], plain[:, :270]), "real prompt keys must be unchanged"
    assert torch.equal(fixed[:, 288:], plain[:, 288:]), "everything past the pads must be unchanged"


def test_window_hidden_span_is_a_noop_once_the_window_scrolls_past_the_pads():
    """The self-retiring property: pads sit at the START, so a scrolled window cannot see them.

    This is what keeps the steady-state mask prompt_len-independent — the reason the bounded read
    is worth having in the first place. It is also why the span is NOT bounded to [lo, lo+span):
    ``lo <= hi`` is the only requirement, and a span outside the window must be a no-op rather
    than an error, or this very case would raise.
    """
    pads = (270, 288)
    prompt_len = 8192
    lo = prompt_len - SPAN  # 7168, far past the pads
    assert lo > pads[1], "precondition: the window starts after the pad span"
    assert torch.equal(
        build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W),
        build_canvas_reveal_denoise_window_mask(
            prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=pads
        ),
    )


def test_window_hidden_span_is_inert_when_absent_or_empty():
    for pads in (None, (288, 288)):
        assert torch.equal(
            build_canvas_reveal_denoise_window_mask(288, CANVAS, SPAN, 0, sliding_window=PAD_W),
            build_canvas_reveal_denoise_window_mask(
                288, CANVAS, SPAN, 0, sliding_window=PAD_W, hidden_prefix_span=pads
            ),
        ), f"pads={pads}"


def test_window_hidden_span_partially_overlapping_the_window_hides_only_the_overlap():
    """A window that has scrolled INTO the pad span must hide the part it can see, and only that."""
    pads = (270, 288)
    lo = 280  # window covers 280.. ; pads 280..287 are visible, 270..279 are not in this window
    prompt_len = lo + SPAN
    plain = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W)
    fixed = build_canvas_reveal_denoise_window_mask(
        prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=pads
    )
    hidden_cols = slice(0, 288 - lo)  # absolute 280..287 -> columns 0..7
    assert (fixed[:, hidden_cols] == NEG).all(), "the visible part of the pad span must be hidden"
    assert torch.equal(fixed[:, 288 - lo :], plain[:, 288 - lo :]), "nothing past the pads may change"


# --- reveal-mask argument validation --------------------------------------------------------


@pytest.mark.parametrize(
    "build, match",
    [
        pytest.param(
            lambda: build_canvas_reveal_denoise_mask(4096, CANVAS, 2048, layer_type="full_attention"),
            None,
            id="p_max-below-prompt_len",
        ),
        pytest.param(
            lambda: build_canvas_reveal_denoise_mask(PAD_P_MAX, CANVAS, PAD_P_MAX, hidden_prefix_span=(-1, 8)),
            "hidden_prefix_span",
            id="fixed-span-negative-lo",
        ),
        pytest.param(
            lambda: build_canvas_reveal_denoise_mask(PAD_P_MAX, CANVAS, PAD_P_MAX, hidden_prefix_span=(8, 4)),
            "hidden_prefix_span",
            id="fixed-span-lo-above-hi",
        ),
        pytest.param(
            lambda: build_canvas_reveal_denoise_mask(
                PAD_P_MAX, CANVAS, PAD_P_MAX, hidden_prefix_span=(0, PAD_P_MAX + 1)
            ),
            "hidden_prefix_span",
            id="fixed-span-hi-past-p_max",
        ),
        pytest.param(
            lambda: build_canvas_reveal_denoise_window_mask(
                288, CANVAS, SPAN, 0, sliding_window=PAD_W, hidden_prefix_span=(-1, 8)
            ),
            "hidden_prefix_span",
            id="window-span-negative-lo",
        ),
        pytest.param(
            lambda: build_canvas_reveal_denoise_window_mask(
                288, CANVAS, SPAN, 0, sliding_window=PAD_W, hidden_prefix_span=(8, 4)
            ),
            "hidden_prefix_span",
            id="window-span-lo-above-hi",
        ),
    ],
)
def test_reveal_mask_rejects_bad_arguments(build, match, expect_error):
    with expect_error(ValueError, match=match):
        build()


# --- borrowed-prefix buffer ownership (#51080) ----------------------------------------------
#
# The fixed full-span prefix read can hand back the model-owned KV cache itself instead of
# cloning it (~2 whole-cache copies per layer per step of block-invariant data). Two independent
# guards make that safe, and BOTH were needed -- the second one was found only on device, after
# the first one alone still produced ``TT_FATAL: Input Tensor is not allocated``:
#
# 1. ``denoise_hidden_forward`` consults ``MutablePrefixKVReader.owns_result`` before freeing the
#    per-layer prompt source.
# 2. ``denoise_attention`` compares BUFFERS, not object identity, before freeing its
#    ``ttnn.to_memory_config`` result. ``to_memory_config`` returns a *fresh Tensor object that
#    aliases the input buffer* when no conversion is needed (device-observed:
#    ``distinct_buffer=False, same_object=False``), so the original ``is not`` check deallocated
#    the model KV cache.
#
# COVERAGE BOUNDARY -- read before trusting these four tests alone. They test the *predicate*
# (``_is_distinct_buffer``) with fakes; they do NOT execute the two real call sites in
# ``denoise_attention``, and no CPU test invokes ``denoise_attention`` at all (building a faithful
# fake for the whole attention path is not worth it). Those call sites are covered by:
#
# * the consumer-side guard test
#   ``test_denoise_forward.py::test_denoise_hidden_forward_honours_prompt_source_ownership``, which
#   drives the real ``denoise_hidden_forward`` and is mutation-verified -- deleting the
#   ``owns_result`` guard makes it fail; and
# * the device A/B ``doc/optimize_perf/verify_prefix_borrow.sh``, where ``DG_PREFIX_BORROW=1`` vs
#   ``0`` must produce an identical ``committed_sha256`` on the full 30-layer traced path. A
#   regression that freed the borrowed cache would abort that run outright.
#
# So a revert of ``_is_distinct_buffer`` to an ``is not`` identity check is caught by the device
# gate, not by CPU CI. If you touch the prefix ownership contract, run that A/B.


class _FakeTensor:
    """Minimal stand-in exposing the buffer_address() surface the guard relies on."""

    def __init__(self, address, *, raises=False):
        self._address = address
        self._raises = raises
        self.freed = False

    def buffer_address(self):
        if self._raises:
            raise RuntimeError("buffer_address unavailable for this storage type")
        return self._address

    def deallocate(self, force=True):
        self.freed = True


def test_alias_of_the_same_buffer_is_not_freed():
    """The device-observed case: different object, SAME buffer -> must not be freed."""
    source = _FakeTensor(0xDEAD0000)
    alias = _FakeTensor(0xDEAD0000)
    assert alias is not source, "the whole point is that object identity says 'distinct'"
    assert _is_distinct_buffer(alias, source) is False


def test_genuine_copy_is_freed():
    """A real conversion produces a new buffer we own and must free, or it leaks per layer."""
    source = _FakeTensor(0xDEAD0000)
    copy = _FakeTensor(0xBEEF0000)
    assert _is_distinct_buffer(copy, source) is True


def test_same_object_is_not_freed():
    source = _FakeTensor(0xDEAD0000)
    assert _is_distinct_buffer(source, source) is False


def test_unknowable_buffer_defaults_to_not_freeing():
    """If ownership cannot be proven, leak rather than free a caller-owned tensor.

    A leaked conversion is recoverable; freeing the model-owned KV cache is not.
    """
    source = _FakeTensor(0xDEAD0000)
    opaque = _FakeTensor(0xBEEF0000, raises=True)
    assert _is_distinct_buffer(opaque, source) is False
