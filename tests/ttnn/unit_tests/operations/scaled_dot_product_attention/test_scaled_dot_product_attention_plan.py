# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design (op_design.md) requires beyond the acceptance test:

- regime-pinned tests: the streamed regime (several key chunks, online rescale) forced through the
  k_chunk_tiles knob, including a ragged last chunk; a multi-row NEO unit (q_chunk_tiles = 2); the resident
  default;
- the thread plan: results identical under several plans, and every NEO computing the unit the host plan
  gives it (a correct result alone would not show that four NEOs ran);
- one device session: another program, then this op, then a tt-metal program (ttnn.to_layout) that issues
  untagged transfers.
"""

import pytest
import torch

import ttnn
from eval.specs.scaled_dot_product_attention.spec import reference
from ttnn.operations.scaled_dot_product_attention import run_attention, scaled_dot_product_attention
from ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention_program_artifacts import (
    THREAD_PLAN,
    make_plan,
    split_rounds,
)


def _inputs(device, q_shape, kv_shape, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(q_shape).bfloat16()
    k = torch.randn(kv_shape).bfloat16()
    v = torch.randn(kv_shape).bfloat16()
    put = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)  # noqa: E731
    return (q, k, v), (put(q), put(k), put(v))


def _check(got, expected):
    """The specification's limits for bfloat16 with the float32 accumulator: PCC 0.999 (the numeric list's
    entry) and relative RMS 0.05."""
    got, expected = got.double().flatten(), expected.double().flatten()
    pcc = torch.corrcoef(torch.stack([got, expected]))[0, 1].item()
    rms = ((got - expected).pow(2).mean().sqrt() / expected.std()).item()
    assert torch.isfinite(got).all()
    assert pcc >= 0.999 and rms <= 0.05, f"pcc={pcc:.6f} rms={rms:.5f}"


LONG = ((1, 2, 512, 64), (1, 1, 512, 64))  # 16 tiles of tokens, two query heads on one key/value head


@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "q_chunk_tiles,k_chunk_tiles",
    [(None, None), (None, 4), (None, 3), (2, None), (2, 3)],
    ids=["resident", "streamed_kc4", "streamed_kc3_ragged", "qc2_resident", "qc2_streamed_kc3"],
)
def test_regimes(device, q_chunk_tiles, k_chunk_tiles, is_causal):
    (q, k, v), tensors = _inputs(device, *LONG)
    plan = make_plan(tensors[0], tensors[1], is_causal=is_causal, fp32_dest_acc_en=True,
                     q_chunk_tiles=q_chunk_tiles, k_chunk_tiles=k_chunk_tiles)  # fmt: skip
    if k_chunk_tiles is not None:
        assert plan.kc == k_chunk_tiles and plan.streamed
    else:
        assert not plan.streamed  # the resident regime is the default on this shape
    if q_chunk_tiles is not None:
        assert plan.qc == q_chunk_tiles
    out = run_attention(*tensors, is_causal=is_causal, scale=0.125,
                        q_chunk_tiles=q_chunk_tiles, k_chunk_tiles=k_chunk_tiles)  # fmt: skip
    _check(ttnn.to_torch(out), reference(q, k, v, is_causal=is_causal, scale=0.125))


@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("threads", [(1, 1, 1), (2, 4, 2), (1, 4, 1), (4, 4, 1), (2, 2, 2)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize(
    "shapes",
    [((1, 8, 96, 64), (1, 2, 96, 64)), ((1, 4, 64, 64), (1, 4, 64, 64)), LONG],
    ids=["grouped_8on2", "padded_rounds", "long"],
)
def test_plans_give_the_same_result(device, shapes, threads, is_causal):
    """Every plan computes the same thing, bit for bit, as the op's default plan."""
    _, tensors = _inputs(device, *shapes)
    base = ttnn.to_torch(run_attention(*tensors, is_causal=is_causal, threads=THREAD_PLAN))
    got = ttnn.to_torch(run_attention(*tensors, is_causal=is_causal, threads=threads))
    assert torch.equal(got, base), (got.float() - base.float()).abs().max()


def _unit_pages(plan, round_index, t):
    """(head, first row tile, rows) of the unit NEO t computes in round `round_index` (sdpa_geometry.hpp)."""
    head_round = round_index % plan.rounds_per_head
    c = (round_index // plan.rounds_per_head) % plan.Hkv
    u = head_round * plan.C + t
    padded = u >= plan.units_per_head
    if padded:
        u = head_round * plan.C
    h = c * plan.g + u % plan.g
    start = (u // plan.g) * plan.qc
    return h, start, min(start + plan.qc, plan.Sqt) - start, padded


@pytest.mark.parametrize(
    "shapes", [LONG, ((1, 4, 64, 64), (1, 4, 64, 64))], ids=["no_padding", "padded_rounds"]
)
def test_each_neo_computes_its_own_unit(device, shapes):
    """With tag_threads every NEO writes t + 1 into its unit's output tiles: each unit must carry the id of
    the NEO the host plan deals it to. A padded NEO rewrites NEO 0's unit, so those pages may carry its id."""
    _, tensors = _inputs(device, *shapes)
    plan = make_plan(tensors[0], tensors[1], is_causal=False, fp32_dest_acc_en=True)
    assert plan.C == THREAD_PLAN[1]
    got = ttnn.to_torch(run_attention(*tensors, is_causal=False, tag_threads=True)).float()

    _, counts, starts = split_rounds(plan.total_rounds, device.compute_with_storage_grid_size())
    expected = {}  # (head, row tile) -> set of NEO ids that write it
    for core, n in counts.items():
        for r in range(starts[core], starts[core] + n):
            for t in range(plan.C):
                h, start, rows, _ = _unit_pages(plan, r, t)
                for row in range(start, start + rows):
                    expected.setdefault((h, row), set()).add(t + 1)
    neos_seen = set()
    for (h, row), ids in expected.items():
        for d in range(plan.Dt):  # per tile: a shared page's tiles may land from different NEOs
            tile = got[0, h, row * 32 : row * 32 + 32, d * 32 : d * 32 + 32]
            value = tile[0, 0].item()
            assert torch.all(tile == value), (h, row, d)
            assert value in ids, (h, row, d, value, ids)
            neos_seen.add(int(value))
    # Every NEO that owns a page alone must show up; without padding that is every NEO of the plan.
    sole_writers = {next(iter(ids)) for ids in expected.values() if len(ids) == 1}
    assert sole_writers <= neos_seen, (sole_writers, neos_seen)
    if all(len(ids) == 1 for ids in expected.values()):
        assert neos_seen == set(range(1, plan.C + 1)), neos_seen


def test_one_device_session(device):
    """Another program, then this op (causal and not), then tt-metal's untilize, which issues untagged
    transfers: the op's harts must neither inherit nor leave a ring's transaction tag."""
    (q, k, v), tensors = _inputs(device, (1, 2, 64, 128), (1, 1, 64, 128))
    # another program first: a tilize of a row-major tensor on the device
    torch.manual_seed(1)
    x = torch.randn(1, 1, 64, 64).bfloat16()
    x_rm = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(x_tile), x)
    for is_causal in (True, False):
        out = scaled_dot_product_attention(*tensors, is_causal=is_causal, scale=0.08838834764831845)
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        _check(ttnn.to_torch(ttnn.from_device(out_rm)),
               reference(q, k, v, is_causal=is_causal, scale=0.08838834764831845))  # fmt: skip


@pytest.mark.parametrize("fidelity", ["LoFi", "HiFi2", "HiFi3", "HiFi4"])
def test_traced_call_arguments(device, fidelity):
    """The traced calls' argument forms: a ttnn.SDPAProgramConfig (ignored, rule R8) and tt-metal's device
    compute config (rule R7), at every fidelity (rule R6: none is refused)."""
    (q, k, v), tensors = _inputs(device, (1, 2, 64, 128), (1, 1, 64, 128))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(2, 1), q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )  # fmt: skip
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity), math_approx_mode=True, fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )  # fmt: skip
    out = scaled_dot_product_attention(*tensors, is_causal=True, scale=0.08838834764831845,
                                       program_config=program_config, compute_kernel_config=config)  # fmt: skip
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    _check(ttnn.to_torch(out), reference(q, k, v, is_causal=True, scale=0.08838834764831845))
