# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The thread plan and the blocking of layer_norm (op_design.md, Work Distribution, "Tests the plan needs").

(a) threads ran: the kernels size their loops from the plan's counts, so a NEO, a reader or a writer that did
    not run leaves its rows unwritten (or the program waiting); a correct result over every row shows each ran.
(b) the regime-pinned values of C and the block knob, read off select_blocking() and checked for values.
(c) one device session: the op after another program, and the op before a tt-metal program (to_layout).
"""

import pytest
import torch

import ttnn
from ttnn.operations.layer_norm import layer_norm
from ttnn.operations.layer_norm.layer_norm_program_artifacts import (
    THREAD_PLAN,
    create_program_artifacts,
    resolve_compute_config,
    select_blocking,
)

HIFI4 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False)


def _config(fp32):
    return ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32, **HIFI4)


def _inputs(device, shape, seed=0):
    torch.manual_seed(seed)
    W = shape[-1]
    x = torch.randn(shape).bfloat16()
    w = torch.randn(1, 32, W).bfloat16()  # 32 different rows: only row 0 is gamma (rule R8)
    b = torch.randn(1, 32, W).bfloat16()
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return (x, w, b), (put(x), put(w), put(b))


def _reference(x, w, b, eps):
    x = x.double()
    c = x - x.mean(-1, keepdim=True)
    var = c.pow(2).mean(-1, keepdim=True)
    return c * torch.rsqrt(var + eps) * w.double()[0, 0] + b.double()[0, 0]


def _check(got, expected, pcc_min=0.999):
    got, expected = got.double().flatten(), expected.double().flatten()
    pcc = torch.corrcoef(torch.stack([got, expected]))[0, 1].item()
    assert pcc >= pcc_min, f"PCC {pcc}"
    assert (got - expected).abs().max().item() < 0.25, (got - expected).abs().max().item()


def _run_with(tx, tw, tb, eps, fp32, **overrides):
    """The op's program with test overrides of the plan and the blocking, one generic_op dispatch."""
    out = ttnn.allocate_tensor_on_device(tx.spec, tx.device())
    spec, run_args, idx = create_program_artifacts(
        tx, tw, tb, out, epsilon=eps, config=resolve_compute_config(_config(fp32)), **overrides
    )
    tensors = [None] * len(idx)
    for name, t in (("x", tx), ("weight", tw), ("bias", tb), ("out", out)):
        tensors[idx[name]] = t
    return ttnn.generic_op(tensors, spec, run_args, idx)


# --- (a) every thread of the plan ran -------------------------------------------------------------------------


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 256, 64), (1, 1, 192, 64)], ids=["8rows_2rounds", "6rows_padded_round"]
)
def test_every_thread_of_the_plan_ran(device, shape, fp32):
    (x, w, b), (tx, tw, tb) = _inputs(device, shape)
    blocking = select_blocking(tx, tw, fp32, thread_plan=(4, 4, 2), max_clusters=1)
    assert (blocking.reader_threads, blocking.compute_threads, blocking.writer_threads) == (4, 4, 2)
    assert blocking.rounds_max == 2 and blocking.block_tile_rows == 2  # a NEO's whole share is one block
    got = ttnn.to_torch(_run_with(tx, tw, tb, 1e-5, fp32, thread_plan=(4, 4, 2), max_clusters=1))
    _check(got, _reference(x, w, b, 1e-5))


# --- (b) regime-pinned values of the plan and the block --------------------------------------------------------


@pytest.mark.parametrize(
    "shape,expected_c",
    [((1, 1, 128, 1024), 1), ((1, 1, 2048, 256), 2), ((1, 1, 4096, 64), 4)],
    ids=["C1_model", "C2", "C4"],
)
def test_thread_plan_selection(device, shape, expected_c):
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 32:
        pytest.skip("the expected values are for a 32-cluster grid")
    (x, w, b), (tx, tw, tb) = _inputs(device, shape)
    blocking = select_blocking(tx, tw, True)
    assert blocking.compute_threads == expected_c
    assert blocking.reader_threads == min(THREAD_PLAN[0], expected_c)
    assert blocking.writer_threads == min(THREAD_PLAN[2], expected_c)
    got = ttnn.to_torch(layer_norm(tx, weight=tw, bias=tb, epsilon=1e-6, compute_kernel_config=_config(True)))
    _check(got, _reference(x, w, b, 1e-6))


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32acc", "bf16acc"])
def test_several_blocks_per_neo(device, fp32):
    """block_tile_rows = 1 with 3 rounds per NEO: three blocks through the same operations, with the ALU formats
    re-aimed at the bfloat16 input after the previous block's Float32 affine step."""
    (x, w, b), (tx, tw, tb) = _inputs(device, (1, 1, 32 * 12, 128))
    blocking = select_blocking(tx, tw, fp32, max_clusters=1, max_block_tile_rows=1)
    assert blocking.compute_threads == 4 and blocking.rounds_max == 3
    assert blocking.block_tile_rows == 1 and blocking.blocks_per_neo_max == 3
    got = ttnn.to_torch(_run_with(tx, tw, tb, 1e-5, fp32, max_clusters=1, max_block_tile_rows=1))
    _check(got, _reference(x, w, b, 1e-5))


def test_ragged_last_block(device):
    """block_tile_rows = 2 with 3 rounds: a full block, then a one-row block through the same call."""
    (x, w, b), (tx, tw, tb) = _inputs(device, (1, 1, 32 * 12, 64))
    blocking = select_blocking(tx, tw, True, max_clusters=1, max_block_tile_rows=2)
    assert blocking.block_tile_rows == 2 and blocking.blocks_per_neo_max == 2
    got = ttnn.to_torch(_run_with(tx, tw, tb, 1e-5, True, max_clusters=1, max_block_tile_rows=2))
    _check(got, _reference(x, w, b, 1e-5))


# --- (c) one device session with other programs ----------------------------------------------------------------

SESSION_SHAPES = {1: (1, 1, 64, 256), 4: (1, 1, 32 * 8, 64)}  # C = 1 on the grid, C = 4 on one cluster


@pytest.mark.parametrize("c", [1, 4], ids=["C1", "C4"])
def test_after_another_program(device, c):
    shape = SESSION_SHAPES[c]
    (x, w, b), (tx, tw, tb) = _inputs(device, shape)
    # tt-metal's own Quasar add (implicit sync) first, in the same device session: our reader's explicit-sync
    # constants and NoC zero writes run after another program.
    other = ttnn.to_torch(ttnn.experimental.quasar.add(tx, tx))
    torch.testing.assert_close(other.float(), (x.float() * 2), atol=0.05, rtol=0.02)
    clusters = None if c == 1 else 1
    assert select_blocking(tx, tw, True, max_clusters=clusters).compute_threads == c
    got = ttnn.to_torch(_run_with(tx, tw, tb, 1e-5, True, max_clusters=clusters))
    _check(got, _reference(x, w, b, 1e-5))


@pytest.mark.parametrize("c", [1, 4], ids=["C1", "C4"])
def test_before_a_tt_metal_program(device, c):
    """to_layout to row-major is an untilize with plain NoC transfers: our reader and writer must have handed
    their transaction tags back."""
    shape = SESSION_SHAPES[c]
    (x, w, b), (tx, tw, tb) = _inputs(device, shape)
    clusters = None if c == 1 else 1
    assert select_blocking(tx, tw, True, max_clusters=clusters).compute_threads == c
    out = _run_with(tx, tw, tb, 1e-5, True, max_clusters=clusters)
    got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
    _check(got, _reference(x, w, b, 1e-5))
