# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests op_design.md's Work Distribution requires of the thread plan, beside the acceptance test.

1. Every hart ran: the plan's count comes from one host constant (GATHER_THREADS); the kernel deals the runs
   by that count, so a hart that does not run leaves its runs (with R = 2, every right-face half) unwritten.
   Each call draws a fresh input, so a stale shard left by an earlier call cannot pass for a correct one.
2. The block knob (heads_per_block) stays live: smaller blocks give the same bytes.
3. One device session: the op after another program, and a program of tt-metal's own after the op.
"""

import importlib

import pytest
import torch

import ttnn
from ttnn.operations.nlp_create_qkv_heads_decode import nlp_create_qkv_heads_decode

# The package's `nlp_create_qkv_heads_decode` attribute is the function; the modules by their full names.
op_module = importlib.import_module("ttnn.operations.nlp_create_qkv_heads_decode.nlp_create_qkv_heads_decode")
artifacts = importlib.import_module(
    "ttnn.operations.nlp_create_qkv_heads_decode.nlp_create_qkv_heads_decode_program_artifacts"
)

HEIGHT_SHARDED_L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

LLAMA = ((1, 1, 1, 3072), 32, 8)
QWEN = ((1, 1, 1, 6144), 32, 8)
PHASE0_ENTRIES = [
    LLAMA,
    QWEN,
    ((1, 1, 1, 96), 1, 1),
    ((1, 1, 1, 384), 4, 1),
    ((1, 1, 1, 4608), 12, None),
    ((1, 1, 1, 4096), 8, 4),
    ((1, 1, 1, 6144), 16, 16),
    ((1, 1, 1, 8192), 16, 8),
]


def _entry_id(entry):
    return "x".join(map(str, entry[0])) + f"_h{entry[1]}_kv{entry[2]}"


def _expected(x, nh, nkv):
    users, width = x.shape[2], x.shape[3]
    hd = width // (nh + 2 * nkv)
    rows = x[0, 0]
    return [rows[:, a * hd:(a + n) * hd].reshape(1, users, n, hd) for a, n in ((0, nh), (nh, nkv), (nh + nkv, nkv))]


def _run_and_check(device, entry, seed):
    shape, nh, nkv = entry
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = nlp_create_qkv_heads_decode(t, num_heads=nh, num_kv_heads=nkv, memory_config=HEIGHT_SHARDED_L1)
    assert isinstance(out, list) and len(out) == 3
    for name, got, want in zip("qkv", out, _expected(x, nh, nkv if nkv is not None else nh)):
        got = ttnn.to_torch(got)
        assert list(got.shape) == list(want.shape), name
        assert torch.equal(got, want), f"{name}: {(got.float() - want.float()).abs().max()}"
    return out


@pytest.fixture
def captured_specs(monkeypatch):
    """The ProgramSpecs the op builds, captured from its call into the program-artifacts module."""
    specs = []
    original = op_module.create_program_artifacts

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        specs.append(result[0])
        return result

    monkeypatch.setattr(op_module, "create_program_artifacts", capture)
    return specs


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("entry", [LLAMA, QWEN], ids=_entry_id)
def test_every_hart_runs(device, monkeypatch, captured_specs, entry, threads):
    monkeypatch.setattr(artifacts, "GATHER_THREADS", threads)
    _run_and_check(device, entry, seed=100 + threads)
    (gather,) = [k for k in captured_specs[-1].kernels if k.unique_id == "gather"]
    assert gather.num_threads == threads


@pytest.mark.parametrize("entry", PHASE0_ENTRIES, ids=_entry_id)
def test_phase0_entries_at_the_plan(device, captured_specs, entry):
    _run_and_check(device, entry, seed=7)
    (gather,) = [k for k in captured_specs[-1].kernels if k.unique_id == "gather"]
    assert gather.num_threads == artifacts.GATHER_THREADS


@pytest.mark.parametrize("heads_per_block", [1, 5, 13])
@pytest.mark.parametrize("entry", [LLAMA, ((1, 1, 1, 4096), 8, 4)], ids=_entry_id)
def test_block_knob(device, monkeypatch, entry, heads_per_block):
    monkeypatch.setattr(artifacts, "HEADS_PER_BLOCK", heads_per_block)
    _run_and_check(device, entry, seed=200 + heads_per_block)


def test_one_device_session(device):
    """In one device session: a tt-metal Quasar program with implicit sync (quasar.add) before the op, two
    calls of the op back to back, then tt-metal programs on its outputs: `to_layout` to row-major on each sharded
    result, a reshard to interleaved L1, and another quasar.add."""
    add = ttnn.experimental.quasar.add
    torch.manual_seed(3)
    a = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert torch.allclose(ttnn.to_torch(add(ta, ta)).float(), (a + a).float(), atol=0.05)

    q, k, v = _run_and_check(device, LLAMA, seed=11)
    _run_and_check(device, QWEN, seed=12)  # two of ours back to back

    torch.manual_seed(11)
    want = _expected(torch.randn(LLAMA[0], dtype=torch.bfloat16), 32, 8)
    for got, expected in zip((q, k, v), want):
        assert torch.equal(ttnn.to_torch(ttnn.to_layout(got, ttnn.ROW_MAJOR_LAYOUT)), expected)
        assert torch.equal(ttnn.to_torch(ttnn.to_memory_config(got, ttnn.L1_MEMORY_CONFIG)), expected)

    after = ttnn.to_torch(ttnn.to_layout(add(ta, ta), ttnn.ROW_MAJOR_LAYOUT))
    assert torch.allclose(after.float(), (a + a).float(), atol=0.05)
