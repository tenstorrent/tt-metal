# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_moe_route_dispatch (immutable spec; see op_design.md).

The op's routing is checked against its OWN selection (read back from combine_index and expert_counts):
expert_counts, the region layout, combine_index and the dispatched rows are exact. The selection and the
weights are checked against the fp64 reference with the contract's bars. T >= 1024 is the op's domain, so
the smallest case is 2x512 (no single-tile case exists).
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.nomic_moe_route_dispatch import nomic_moe_route_dispatch

HIDDEN = 768
NUM_EXPERTS = 8
TOP_K = 2
TILE = 32
WEIGHT_STD = 0.034  # the real router weight's scale

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
MAX_WEIGHT_ERROR = 3.0e-3
MAX_REROUTED_FRACTION = 0.006
MAX_REROUTE_PROB_GAP = 6.0e-3


def rows_for(tokens, spare_tile_rows):
    return TILE * (math.ceil(TOP_K * tokens / TILE) + NUM_EXPERTS - 1) + TILE * spare_tile_rows


def reference_probabilities(x, w):
    """fp64 softmax over the experts and the reference top-2 (ties to the lower expert)."""
    tokens = x.shape[0] * x.shape[2]
    p = torch.softmax(x.reshape(tokens, HIDDEN).double() @ w.double()[:, :NUM_EXPERTS], dim=-1)
    _, order = torch.sort(p, dim=-1, descending=True, stable=True)
    return p, order[:, :TOP_K]


def region_layout(sel):
    """Rows of every slot (T, 2), per-expert counts and region starts of a selection."""
    counts = torch.bincount(sel.flatten(), minlength=NUM_EXPERTS)
    padded = TILE * ((counts + TILE - 1) // TILE)
    starts = torch.cumsum(padded, 0) - padded
    flat = sel.flatten()
    order = torch.argsort(flat, stable=True)  # by expert, then by slot index (token order)
    ranks = torch.empty_like(flat)
    first = torch.cumsum(counts, 0) - counts
    ranks[order] = torch.arange(flat.numel()) - first[flat[order]]
    rows = (starts[flat] + ranks).reshape(sel.shape)
    return rows, counts, starts


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def nd_norm_memory_config(shape, device):
    from ttnn.operations.nomic_residual_layer_norm import input_memory_config

    return input_memory_config(tuple(shape), device)


def run_case(device, batch, seq, spare_tile_rows=0, compute_kernel_config=None, x_memory_config=None):
    torch.manual_seed(42)
    shape = (batch, 1, seq, HIDDEN)
    x = torch.randn(shape).bfloat16()
    w = (torch.randn(HIDDEN, NUM_EXPERTS) * WEIGHT_STD).float()

    x_dev = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=x_memory_config or nd_norm_memory_config(shape, device),
    )
    w_dev = ttnn.from_torch(
        w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    outputs = nomic_moe_route_dispatch(
        x_dev, w_dev, spare_tile_rows=spare_tile_rows, compute_kernel_config=compute_kernel_config
    )
    assert len(outputs) == 4
    dispatched, combine_index, combine_weights, expert_counts = outputs

    tokens = batch * seq
    t_pad = TILE * math.ceil(tokens / TILE)
    total_rows = rows_for(tokens, spare_tile_rows)

    from ttnn.operations.nomic_moe_route_dispatch import output_memory_config

    expected = (
        ("dispatched", dispatched, (1, 1, total_rows, HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ("combine_index", combine_index, (TOP_K, t_pad), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ("combine_weights", combine_weights, (1, TOP_K, t_pad, 1), ttnn.float32, ttnn.TILE_LAYOUT),
        ("expert_counts", expert_counts, (1, NUM_EXPERTS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )
    for name, tensor, out_shape, dtype, layout in expected:
        assert tuple(tensor.shape) == out_shape, f"{name} shape {tuple(tensor.shape)} != {out_shape}"
        assert tensor.dtype == dtype, f"{name} dtype {tensor.dtype} != {dtype}"
        assert tensor.layout == layout, f"{name} layout {tensor.layout} != {layout}"
    assert dispatched.memory_config() == output_memory_config(total_rows, device, ttnn.BufferType.L1)
    for tensor in (combine_index, combine_weights, expert_counts):
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG

    rows_out = ttnn.to_torch(dispatched).reshape(total_rows, HIDDEN)
    index = ttnn.to_torch(combine_index).to(torch.int64).reshape(TOP_K, t_pad)
    weights = ttnn.to_torch(combine_weights).to(torch.float64).reshape(TOP_K, t_pad)
    counts = ttnn.to_torch(expert_counts).to(torch.int64).reshape(NUM_EXPERTS)

    # The op's own selection, from combine_index and the region layout of its counts.
    padded = TILE * ((counts + TILE - 1) // TILE)
    starts = torch.cumsum(padded, 0) - padded
    slot_rows = index[:, :tokens].T.contiguous()
    sel = torch.searchsorted(starts, slot_rows, right=True) - 1
    assert (sel >= 0).all() and (slot_rows - starts[sel] < counts[sel]).all(), "a slot row lies outside its region"
    assert (sel[:, 0] != sel[:, 1]).all(), "a token is routed twice to one expert"

    # Exact: counts, region layout and combine_index follow from the op's selection.
    own_rows, own_counts, _ = region_layout(sel)
    assert torch.equal(own_counts, counts), f"expert_counts {counts.tolist()} != {own_counts.tolist()}"
    assert torch.equal(own_rows, slot_rows), "combine_index disagrees with the region layout of the selection"
    assert (index[:, tokens:] == 0).all() and (weights[:, tokens:] == 0).all(), "padded tokens must be 0"

    # Exact: every slot row is its token's x row, every other row a copy of some x row.
    xt = x.reshape(tokens, HIDDEN)
    x_bits = xt.view(torch.int16)
    out_bits = rows_out.contiguous().view(torch.int16)
    assert torch.equal(
        out_bits[slot_rows.flatten()], x_bits.repeat_interleave(TOP_K, dim=0)
    ), "a slot row of dispatched is not a bit-exact copy of its token's x row"
    free = torch.ones(total_rows, dtype=torch.bool)
    free[slot_rows.flatten()] = False
    known = {bytes(row.numpy().tobytes()) for row in x_bits}
    for row in out_bits[free]:
        assert bytes(row.numpy().tobytes()) in known, "a padding row of dispatched is not a copy of an x row"

    # Accuracy against fp64.
    p64, ref = reference_probabilities(x, w)
    selected = weights[:, :tokens].T
    expected_weights = torch.gather(p64, 1, sel)
    assert torch.isfinite(selected).all()
    assert (selected[:, 0] >= selected[:, 1]).all(), "k = 0 weight below k = 1 weight"
    weight_error = (selected - expected_weights).abs().max().item()
    assert weight_error <= MAX_WEIGHT_ERROR, f"max |w - p64[sel]| = {weight_error:.3e}"
    assert pcc(selected, expected_weights) >= PCC[ttnn.float32]

    rerouted = (sel != ref).any(-1)
    fraction = rerouted.double().mean().item()
    assert fraction <= MAX_REROUTED_FRACTION, f"{100 * fraction:.3f}% of tokens rerouted"
    if rerouted.any():
        gap = (torch.gather(p64, 1, sel) - torch.gather(p64, 1, ref)).abs()[rerouted].max().item()
        assert gap <= MAX_REROUTE_PROB_GAP, f"rerouted choice {gap:.3e} from the reference expert"

    # Inputs are not modified.
    assert torch.equal(ttnn.to_torch(x_dev).view(torch.int16), x.view(torch.int16))


@pytest.mark.parametrize(
    "batch, seq",
    [
        (2, 512),  # smallest token count (T = 1024)
        (4, 256),  # same T, more batches
        (4, 512),
        (8, 256),
        (6, 512),  # non-power-of-two batch
        (8, 384),
        (8, 512),  # primary production shape (T = 4096)
    ],
)
def test_nomic_moe_route_dispatch(device, batch, seq):
    run_case(device, batch, seq)


@pytest.mark.parametrize("batch, seq, spare_tile_rows", [(4, 512, 2), (8, 512, 3), (2, 512, 1)])
def test_spare_tile_rows(device, batch, seq, spare_tile_rows):
    run_case(device, batch, seq, spare_tile_rows=spare_tile_rows)


def test_explicit_compute_config(device):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    run_case(device, 2, 512, compute_kernel_config=config)


# Regime R2 (interleaved x, G = 1, tile rows split over the grid) and R3 (ragged T, B = 1).
@pytest.mark.parametrize(
    "batch, seq, placement",
    [
        (8, 512, "dram"),  # 128 tile rows: 18 cores with 2, 92 with 1
        (2, 512, "l1"),  # 32 tile rows: 32 token cores
        (1, 1000, "nd_norm"),
        (1, 3000, "dram"),
        (1, 1000, "l1"),
    ],
)
def test_interleaved_and_ragged(device, batch, seq, placement):
    memory = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG, "nd_norm": None}[placement]
    run_case(device, batch, seq, x_memory_config=memory)
