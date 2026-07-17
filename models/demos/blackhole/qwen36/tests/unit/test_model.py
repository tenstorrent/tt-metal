# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device Generator-interface contract tests (Qwen3.5-9B).

Validates the top-level decode pipeline: the RoPE host/device pack round-trip,
Generator-driven decode (untraced and traced) matching the model-owned
``decode_paged`` reference (guards GDN recurrent-state stability across trace
capture), and the demo's chunk-outer trace-selection gate.

Each comparison test is self-contained: it computes the ``decode_paged`` reference,
then resets + re-prefills to a clean post-prefill state and runs the Generator path
from there — comparing the two in one process (no on-disk baseline / ordering
dependency between tests).

Uses the repo-root ``device`` fixture parametrized with ``device_params`` (the
tracing path needs ``l1_small_size``/``num_command_queues``), so this file does NOT
use the bespoke single-device fixture from conftest.

Run:
  HF_MODEL=Qwen/Qwen3.5-9B \
  pytest models/demos/blackhole/qwen36/tests/unit/test_model.py -v -s
"""
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.generator_interface import pack_rope_host, prime_decode_trace, unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.generator import Generator

# Single-device test: default to the 9B checkpoint (the 27B needs a multi-device mesh for TP).
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]

BLOCK, NBLK = 64, 32
PROMPT = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)  # T=16, short
DECODE_POSITIONS = range(16, 20)  # 4 incremental decode steps


def _build(device, n_layers=4):
    return Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, n_layers=n_layers)


def _allocate(model):
    """Allocate the paged KV caches + GDN external state once; return the page table.

    (allocate_kv_caches must be called only once per model — it binds the GDN external state.)"""
    args = model.args
    model.allocate_kv_caches([NBLK, args.n_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    return torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)


def _fresh_prefill(model, page_table):
    """Prefill the fixed prompt and return the next-token logits.

    ``prefill_paged`` resets all layer state (KV + GDN recurrent/conv) internally, so calling this
    again on the same model restores an identical clean post-prefill state — letting the reference
    path and the path under test each start from the same state within one test."""
    return ttnn.to_torch(model.prefill_paged(PROMPT, page_table)).squeeze().float()


def _reference_decode(model, page_table, pf):
    """Reference next-token logits via the model-owned ``decode_paged`` (advances state)."""
    dec = []
    tok = torch.tensor([[int(pf.argmax())]], dtype=torch.long)
    for pos in DECODE_POSITIONS:
        dl = ttnn.to_torch(model.decode_paged(tok, current_pos=pos, page_table=page_table)).squeeze().float()
        dec.append(dl)
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)
    return dec


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_pack_roundtrip(device):
    # Mirrors the decode flow: pack on host, copy to device, unpack on device.
    cos = torch.randn(1, 1, 64)
    sin = torch.randn(1, 1, 64)
    cos_host = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_host = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    packed = ttnn.to_device(pack_rope_host(cos_host, sin_host), device)
    c2, s2 = unpack_rope(packed)
    assert tuple(c2.shape) == (1, 1, 64)
    assert tuple(s2.shape) == (1, 1, 64)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generator_decode_matches_reference(device):
    """Generator-driven (untraced) decode must match the model-owned decode_paged reference."""
    device.enable_program_cache()
    model = _build(device)
    args = model.args
    page_table = _allocate(model)

    # Reference: model-owned prefill + decode_paged.
    ref_pf = _fresh_prefill(model, page_table)
    ref_dec = _reference_decode(model, page_table, ref_pf)
    assert ref_pf.shape[-1] == args.vocab_size

    # Path under test: Generator-driven decode from a fresh, identical post-prefill state.
    pf = _fresh_prefill(model, page_table)
    assert torch.allclose(pf, ref_pf, atol=1e-2, rtol=1e-2), "re-prefill not reproducible"
    gen = Generator([model], [args], device)
    tok = torch.tensor([[int(pf.argmax())]], dtype=torch.long)
    for i, pos in enumerate(DECODE_POSITIONS):
        out = gen.decode_forward(
            tok, torch.tensor([pos]), page_table=page_table, kv_cache=None, enable_trace=False, read_from_device=True
        )
        dl = out[0].squeeze().float() if isinstance(out, tuple) else out.squeeze().float()
        assert torch.allclose(dl, ref_dec[i], atol=1e-2, rtol=1e-2), f"decode step {i} drifted from decode_paged"
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generator_decode_traced_matches_reference(device):
    """Generator-driven TRACED decode must match the model-owned decode_paged reference
    token-for-token. Guards against GDN recurrent-state corruption during Generator's trace
    capture (the stock capture runs the forward twice; prime_decode_trace save/restores the GDN
    state so the replay continues from the correct post-prefill state). Coverage for the traced
    path that test_generator_decode_matches_reference (enable_trace=False) does not exercise.
    """
    device.enable_program_cache()
    model = _build(device)
    args = model.args
    page_table = _allocate(model)

    # Reference first (no trace parked yet) — model-owned decode_paged.
    ref_pf = _fresh_prefill(model, page_table)
    ref_dec = _reference_decode(model, page_table, ref_pf)

    # Path under test: traced Generator decode from a fresh, identical post-prefill state.
    pf = _fresh_prefill(model, page_table)
    gen = Generator([model], [args], device)
    tok = torch.tensor([[int(pf.argmax())]], dtype=torch.long)
    # Capture the decode trace with GDN-state save/restore so the loop replays from the
    # correct post-prefill state (Generator's capture would otherwise double-advance it).
    prime_decode_trace(gen, model, tok, torch.tensor([16]), page_table)
    for i, pos in enumerate(DECODE_POSITIONS):
        out = gen.decode_forward(
            tok, torch.tensor([pos]), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True
        )  # TRACED (trace already captured)
        dl = out[0].squeeze().float() if isinstance(out, tuple) else out.squeeze().float()
        assert torch.allclose(dl, ref_dec[i], atol=1e-2, rtol=1e-2), f"traced decode step {i} drifted from decode_paged"
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_chunk_seq_flag_selects_chunk_outer(device):
    """The demo's chunk-outer trace selection must be driven by the GDN weights'
    use_chunk_seq_prefill (always True now that chunk-seq is the only prefill path). Guards the
    demo gate _should_use_chunked_trace against regressing to the slow whole-sequence trace.
    """
    from models.demos.blackhole.qwen36.demo.text_demo import _should_use_chunked_trace

    model = _build(device)  # n_layers=4
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    assert gdn, "expected at least one GDN (linear-attention) layer"
    # Chunk-seq prefill is always on now; the demo's gate must select chunk-outer.
    assert all(a.weights.use_chunk_seq_prefill for a in gdn)
    assert _should_use_chunked_trace(model) is True
