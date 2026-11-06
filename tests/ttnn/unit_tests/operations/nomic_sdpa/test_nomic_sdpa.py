# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_sdpa (immutable spec; see ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/nomic_sdpa/op_design.md).

out[b, 0, i, 64 h + d] = sum_j softmax_j(scale * q[b, h, i, :] . k[b, h, j, :]) * v[b, h, j, d], the 12 heads
concatenated into 768 columns. Phase 0 cell: bfloat16, tile-aligned S, no key mask, q / k / v and the output
interleaved in L1 or DRAM.

Two input families:
- torch.randn (seed 42) against a float64 PyTorch reference, PCC keyed by dtype (bfloat16: 0.995);
- the golden suite's "real" profile (the model's own score statistics) against the golden suite's own bar
  (TOLERANCES[bfloat16]: PCC and relative RMS no worse than stock SDPA).

Shapes cover a single tile row per head (1x32, 12 busy cores), multi-tile (1x128), non-square multi-batch (4x288,
S_t = 9), cores holding rows of several (b, h) pairs (16x256, 32x128), and the perf shapes 2x512, 4x512, 8x512.
Every output value, the tile padding included, must be finite.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_sdpa.helpers import (
    TOLERANCES,
    check_finite,
    check_values,
    make_qkv,
    reference as golden_reference,
)
from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.nomic_sdpa import nomic_sdpa

HEADS, HEAD_DIM, HIDDEN = 12, 64, 768

PCC_BY_DTYPE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}

SHAPES = [
    (1, 32),  # single tile row per head: 12 query rows on 12 cores
    (1, 128),  # multi-tile, fewer rows than cores
    (4, 288),  # non-square, multi-batch, S_t = 9 (odd key extent)
    (16, 256),  # multi-batch: up to 3 (b, h) pairs on one core
    (32, 128),  # multi-batch: up to 4 (b, h) pairs on one core, 4096 tokens
    (2, 512),  # the 1k perf shape
    (4, 512),  # the 2k perf shape
    (8, 512),  # the 4k perf shape (perf focus)
]

BUFFERS = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


def _ids(shape):
    return f"{shape[0]}x{shape[1]}"


def torch_reference(q, k, v, scale):
    """(B, 1, S, 768) float64: softmax(scale q k^T) v per head, heads concatenated."""
    scores = scale * (q.double() @ k.double().transpose(-1, -2))
    out = torch.softmax(scores, dim=-1) @ v.double()
    batch, heads, seqlen, dim = q.shape
    return out.permute(0, 2, 1, 3).reshape(batch, 1, seqlen, heads * dim)


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def to_device(t, device, config):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)


def check_contract(output, shape, config):
    batch, seqlen = shape
    assert tuple(output.shape) == (batch, 1, seqlen, HIDDEN)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == config


def randn_qkv(batch, seqlen, prescale):
    torch.manual_seed(42)
    shape = (batch, HEADS, seqlen, HEAD_DIM)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    if prescale:
        q = q / 8.0  # the model's fused QKV op leaves q pre-scaled by 1 / sqrt(64)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


@pytest.mark.parametrize("buffer", ["l1", "dram"])
@pytest.mark.parametrize("shape", SHAPES, ids=_ids)
def test_nomic_sdpa_randn(device, shape, buffer):
    """torch.randn inputs, q pre-scaled by 1/8 and scale = 1.0 (the model's call)."""
    batch, seqlen = shape
    config = BUFFERS[buffer]
    q, k, v = randn_qkv(batch, seqlen, prescale=True)
    tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))

    output = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=config)

    check_contract(output, shape, config)
    check_finite(output, (batch, 1, seqlen, HIDDEN))
    got = ttnn.to_torch(output).reshape(batch, 1, seqlen, HIDDEN)
    expected = torch_reference(q, k, v, 1.0)
    value = pcc(got, expected)
    assert value >= PCC_BY_DTYPE[ttnn.bfloat16], f"PCC {value:.6f} < {PCC_BY_DTYPE[ttnn.bfloat16]}"


@pytest.mark.parametrize("shape", [(1, 128), (4, 288), (8, 512)], ids=_ids)
def test_nomic_sdpa_applies_scale(device, shape):
    """Unscaled q with scale = 1/8: the op must apply `scale` itself."""
    batch, seqlen = shape
    config = ttnn.L1_MEMORY_CONFIG
    q, k, v = randn_qkv(batch, seqlen, prescale=False)
    tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))

    output = nomic_sdpa(tq, tk, tv, scale=0.125, memory_config=config)

    check_contract(output, shape, config)
    check_finite(output, (batch, 1, seqlen, HIDDEN))
    got = ttnn.to_torch(output).reshape(batch, 1, seqlen, HIDDEN)
    value = pcc(got, torch_reference(q, k, v, 0.125))
    assert value >= PCC_BY_DTYPE[ttnn.bfloat16], f"PCC {value:.6f} < {PCC_BY_DTYPE[ttnn.bfloat16]}"


@pytest.mark.parametrize("buffer", ["l1", "dram"])
@pytest.mark.parametrize("shape", SHAPES, ids=_ids)
def test_nomic_sdpa_real_profile(device, shape, buffer):
    """The model's score statistics against the golden suite's bar (PCC and relative RMS vs float64)."""
    batch, seqlen = shape
    config = BUFFERS[buffer]
    q, k, v = make_qkv((batch, HEADS, seqlen, HEAD_DIM), "real", seed=0)
    tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))

    output = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=config)

    check_contract(output, shape, config)
    check_finite(output, (batch, 1, seqlen, HIDDEN))
    check_values(output, golden_reference(q, k, v, 1.0), TOLERANCES[ttnn.bfloat16])


def test_nomic_sdpa_default_memory_config_is_l1(device):
    q, k, v = randn_qkv(1, 128, prescale=True)
    tq, tk, tv = (to_device(t, device, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v))
    output = nomic_sdpa(tq, tk, tv)
    assert output.memory_config() == ttnn.L1_MEMORY_CONFIG
    got = ttnn.to_torch(output).reshape(1, 1, 128, HIDDEN)
    assert pcc(got, torch_reference(q, k, v, 1.0)) >= PCC_BY_DTYPE[ttnn.bfloat16]


def test_nomic_sdpa_repeated_calls_rebind_addresses(device):
    """The cached program must follow new tensors: two calls with different inputs give their own results."""
    config = ttnn.L1_MEMORY_CONFIG
    for seed_shift in (0, 1):
        torch.manual_seed(42 + seed_shift)
        shape = (2, HEADS, 512, HEAD_DIM)
        q, k, v = ((torch.randn(shape) / (8.0 if i == 0 else 1.0)).bfloat16() for i in range(3))
        tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))
        output = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=config)
        got = ttnn.to_torch(output).reshape(2, 1, 512, HIDDEN)
        assert pcc(got, torch_reference(q, k, v, 1.0)) >= PCC_BY_DTYPE[ttnn.bfloat16]
        for t in (tq, tk, tv, output):
            ttnn.deallocate(t)


def test_nomic_sdpa_refuses_long_sequences(device, expect_error):
    """S > 512: K and V of a (b, h) pair no longer fit one core; the op refuses so the model falls back."""
    q = torch.zeros(1, HEADS, 544, HEAD_DIM).bfloat16()
    tq = to_device(q, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(SupportRefusal, ""):  # the message is the implementer's
        nomic_sdpa(tq, tq, tq)


def test_nomic_sdpa_rejects_malformed_calls(device, expect_error):
    q = to_device(torch.zeros(1, HEADS, 128, HEAD_DIM).bfloat16(), device, ttnn.DRAM_MEMORY_CONFIG)
    k = to_device(torch.zeros(1, HEADS, 96, HEAD_DIM).bfloat16(), device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(ValueError, ""):
        nomic_sdpa(q, k, q)  # q / k / v shapes differ
    with expect_error(ValueError, ""):
        nomic_sdpa(q, q, q, key_lengths=[128, 128])  # len(key_lengths) != B
    with expect_error(ValueError, ""):
        nomic_sdpa(q, q, q, key_lengths=[0])  # a length outside 1 .. S


def test_nomic_sdpa_kv_depth_falls_back_under_l1_pressure(device):
    """Refinement 2: DRAM placement plans a 2-deep K/V ring (the pair multicast's pre-send slot); when live L1 leaves
    room only for the 1-deep ring, the call falls back to it (and its first-segment multicast groups) instead of
    refusing, and stays correct."""
    import importlib

    from eval.golden_tests.nomic_sdpa.helpers import reserve_l1

    op = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa")
    batch, seqlen = 8, 512
    deep, shallow = (op.circular_buffer_bytes(batch, seqlen, kv_depth=d) for d in (2, 1))
    assert shallow < deep
    free = op._l1_free_below_live(device)
    filler = reserve_l1(device, max(1, (free - (deep + shallow) // 2) // 1024))
    assert shallow <= op._l1_free_below_live(device) < deep, "the filler must leave room for the 1-deep ring only"

    config = ttnn.DRAM_MEMORY_CONFIG
    q, k, v = make_qkv((batch, HEADS, seqlen, HEAD_DIM), "real", seed=0)
    tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))
    output = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=config)
    check_values(output, golden_reference(q, k, v, 1.0), TOLERANCES[ttnn.bfloat16])
    assert any(state.plan.kv_depth == 1 and state.plan.batch == batch for state in op._CALL_CACHE.values())
    ttnn.deallocate(filler)


@pytest.mark.parametrize(
    "knobs",
    [{"ROW_GROUP_ROWS": 4}, {"ROW_GROUP_ROWS": 2}, {"EXP_OVERLAP": False}],
    ids=lambda k: ",".join(f"{n}={v}" for n, v in k.items()),
)
def test_nomic_sdpa_row_group_knobs(device, monkeypatch, knobs):
    """Refinement 3: the row-group schedule's non-default knobs stay correct. 8x384 has q_block_rows = 11 and
    12-row segments, so G = 4 runs ragged groups inside ragged blocks (G = 2 too: the normalize in DEST runs one P V
    window a row inside a multi-row group; PV_WINDOW_COLS was removed in Perf 1); EXP_OVERLAP = False is the block-sequential A/B path under the default (approximate-exp) config.
    """
    import importlib

    op = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa")
    for name, value in knobs.items():
        monkeypatch.setattr(op, name, value)
    monkeypatch.setattr(op, "_CALL_CACHE", {})
    batch, seqlen = 8, 384
    config = ttnn.L1_MEMORY_CONFIG
    q, k, v = make_qkv((batch, HEADS, seqlen, HEAD_DIM), "real", seed=3)
    tq, tk, tv = (to_device(t, device, config) for t in (q, k, v))
    output = nomic_sdpa(tq, tk, tv, scale=1.0)
    check_values(output, golden_reference(q, k, v, 1.0), TOLERANCES[ttnn.bfloat16])
    assert all(s.plan.exp_overlap == knobs.get("EXP_OVERLAP", True) for s in op._CALL_CACHE.values())
