# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implementer tests of nlp_create_qkv_heads beyond the acceptance test (op_design.md, Work Distribution).

- every first-phase INPUTS entry, exactly;
- the work split: a cell with more pages than clusters and one with fewer, and the clusters engaged;
- the knobs (thread plan, blocks per cluster, ring depth) turned away from their defaults, exactly — the kernels
  stride by the plan's counts, so a hart that did not run leaves pages unwritten;
- device sessions: the op after another program, and a tt-metal program (to_layout) after the op.
"""

import pytest
import torch

import ttnn
from ttnn.operations.nlp_create_qkv_heads import nlp_create_qkv_heads
from ttnn.operations.nlp_create_qkv_heads import nlp_create_qkv_heads_program_artifacts as artifacts
from ttnn.operations.nlp_create_qkv_heads.nlp_create_qkv_heads import output_shapes

DRAM = ttnn.DRAM_MEMORY_CONFIG

# The first-phase INPUTS entries of the specification: (shape, num_heads, num_kv_heads).
PHASE0_ENTRIES = [
    ((1, 1, 128, 3072), 16, 16),
    ((1, 1, 128, 6144), 32, 8),
    ((1, 1, 32, 96), 1, 1),
    ((1, 1, 32, 384), 4, 1),
    ((1, 1, 128, 3072), 32, 8),
    ((1, 1, 64, 4608), 12, None),
    ((1, 1, 32, 4096), 8, 4),
    ((1, 1, 32, 8192), 16, 8),
    ((1, 1, 4096, 96), 1, 1),
]


def _entry_id(entry):
    shape, nh, nkv = entry
    return "x".join(map(str, shape)) + f"_h{nh}_kv{nkv}"


def _reference(x, nh, nkv):
    kv = nh if nkv is None else nkv
    b, _, s, w = x.shape
    hd = w // (nh + 2 * kv)
    q, k, v = torch.split(x[:, 0], [nh * hd, kv * hd, kv * hd], dim=-1)
    return [t.reshape(b, s, n, hd).permute(0, 2, 1, 3) for t, n in ((q, nh), (k, kv), (v, kv))]


def _input(device, shape, seed):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    return x, ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _check_exact(outputs, x, nh, nkv):
    assert isinstance(outputs, list) and len(outputs) == 3
    for name, got, want in zip("qkv", outputs, _reference(x, nh, nkv)):
        got = ttnn.to_torch(got)
        assert list(got.shape) == list(want.shape), (name, got.shape, want.shape)
        assert torch.equal(got, want), f"{name}: {(got.float() - want.float()).abs().max()}"


def _run_with_knobs(device, tx, nh, nkv, **knobs):
    """The op's single dispatch, with the program's knobs overridden."""
    kv = nh if nkv is None else nkv
    outs = [
        ttnn.allocate_tensor_on_device(ttnn.TensorSpec(ttnn.Shape(s), tx.dtype, ttnn.TILE_LAYOUT), device)
        for s in output_shapes(tx, nh, nkv, False)
    ]
    spec, run_args, indices = artifacts.create_program_artifacts(tx, *outs, num_heads=nh, num_kv_heads=kv, **knobs)
    ttnn.generic_op([tx, *outs], spec, run_args, indices)
    return outs


@pytest.mark.parametrize("entry", PHASE0_ENTRIES, ids=_entry_id)
def test_phase0_entries_exact(device, entry):
    shape, nh, nkv = entry
    x, tx = _input(device, shape, seed=1)
    _check_exact(nlp_create_qkv_heads(tx, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False,
                                      memory_config=DRAM), x, nh, nkv)


@pytest.mark.parametrize(
    "entry,clusters",
    [(((1, 1, 128, 3072), 16, 16), 32), (((1, 1, 32, 96), 1, 1), 3), (((1, 1, 32, 4096), 8, 4), 32)],
    ids=["384pages", "3pages", "128pages"],
)
def test_work_split_engages_clusters(device, entry, clusters):
    """More pages than clusters fills the grid; fewer engages one cluster per page. Every cluster's range is
    written by its own reader and writer, so an exact result shows that each of them ran."""
    shape, nh, nkv = entry
    x, tx = _input(device, shape, seed=2)
    _, counts, starts = artifacts.split_pages(tx.buffer_num_pages(), device.compute_with_storage_grid_size())
    assert len(counts) == clusters
    assert sum(counts.values()) == tx.buffer_num_pages()
    assert max(counts.values()) - min(counts.values()) <= 1
    _check_exact(nlp_create_qkv_heads(tx, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False,
                                      memory_config=DRAM), x, nh, nkv)


KNOBS = [
    {"threads": (1, 1)},
    {"threads": (2, 2)},
    {"threads": (1, 2)},
    {"threads": (2, 1)},
    {"threads": (2, 4)},
    {"threads": (3, 3)},
    {"blocks_per_core": 2},
    {"blocks_per_core": 5},
    {"blocks_per_core": 3, "threads": (2, 2)},
    {"ring_depth_tiles": 4},
    {"ring_depth_tiles": 32},
]


@pytest.mark.parametrize("knobs", KNOBS, ids=lambda k: "-".join(f"{a}{b}" for a, b in k.items()))
@pytest.mark.parametrize("entry", [((1, 1, 128, 6144), 32, 8), ((1, 1, 64, 4608), 12, None)], ids=_entry_id)
def test_knobs_exact(device, entry, knobs):
    """Each knob away from its default: still an exact permutation. Pages per cluster (24 and 18) are not a
    multiple of the thread counts or the block counts, so ragged shares and short last blocks are covered."""
    shape, nh, nkv = entry
    x, tx = _input(device, shape, seed=3)
    _check_exact(_run_with_knobs(device, tx, nh, nkv, **knobs), x, nh, nkv)


def test_after_another_program(device):
    """The op in a device session after other programs: tt-metal's own softmax, then this op on another shape."""
    torch.manual_seed(4)
    a = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.to_torch(ttnn.experimental.quasar.softmax(ta, dim=-1))
    x1, t1 = _input(device, (1, 1, 32, 384), seed=5)
    _check_exact(nlp_create_qkv_heads(t1, num_heads=4, num_kv_heads=1, transpose_k_heads=False,
                                      memory_config=DRAM), x1, 4, 1)
    x2, t2 = _input(device, (1, 1, 128, 3072), seed=6)
    _check_exact(nlp_create_qkv_heads(t2, num_heads=16, num_kv_heads=16, transpose_k_heads=False,
                                      memory_config=DRAM), x2, 16, 16)


def test_a_tt_metal_program_follows(device):
    """tt-metal's to_layout (untagged transfers) after the op, in one session: the op's harts hand the tag back."""
    x, tx = _input(device, (1, 1, 128, 6144), seed=7)
    outputs = nlp_create_qkv_heads(tx, num_heads=32, num_kv_heads=8, transpose_k_heads=False, memory_config=DRAM)
    for name, out, want in zip("qkv", outputs, _reference(x, 32, 8)):
        got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
        assert torch.equal(got, want), name
