# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode routing (``tt/v41/routing.py``) against the prefill ops it replaces, on the same device inputs.

No weights: random bf16 gate logits per SP rank (the TP chips of a rank hold the same, as after the gate's
all-reduce) and a random recentred bias. The old path is ``moe_grouped_topk`` on the 64-row tile the batched step
feeds it (the tile repeated, the first ``REAL`` rows of every chip real under a padding config), then
``masked_bincount`` + ``offset_cumsum``. The new path is ``DecodeGate``'s ops on the 32-row tile (fp32 scores on the
SFPU, ``router_select``) and ``DecodeRoutingSetup`` (one index tile all-gathered, ``route_offsets``). Bars: the same
set of experts per row (the sentinel on the pad rows), except where the two differ on a near-tie at the boundary
(``moe_grouped_topk`` sorts TF32 keys and merges its per-tile results by chain position; ``router_select`` is exact
fp32, the reference's rule; at most one row in twenty, each within one TF32 ulp); weights within one bf16 ulp (the
old normalises on the SFPU, the new in fp32 on a RISC core); offsets / totals / region offsets exact on every chip,
both routing setups fed the same indices. Both paths are timed eagerly (a bound on the device time; the replay's
Tracy profile is the measure).

Run: pytest models/demos/deepseek_v3_d_p/tests/v41/test_v41_routing.py -k torus-xy-8x4
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.routing import DecodeRoutingSetup, router_select

E, K, SCALE, EPS = 384, 6, 1.5, 1e-20
ROWS, OLD_ROWS, REAL = 32, 64, 4
ITERS = 20

MESHES = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
] + galaxy_meshes()


def _per_device(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _sharded_rows(mesh_device, host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    """``host`` ``[sp * rows, cols]``: rows split over the mesh rows, replicated over the columns."""
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
    )


def _replicated(mesh_device, host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _old_gate(logits64, bias64, padding64):
    return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        logits64,
        bias64,
        n_groups=1,
        summed_experts_per_group=1,
        topk_groups=1,
        n_activated_experts=K,
        route_scale=SCALE,
        stable_sort=True,
        epsilon=EPS,
        score_func="sqrtsoftplus",
        padding_config=padding64,
        weights_layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _new_gate(logits32, zero32, bias32):
    activations = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
    ]
    l32 = ttnn.typecast(logits32, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    score = ttnn.add(l32, zero32, input_tensor_a_activations=activations, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rank = ttnn.add(score, bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    weights, indices = router_select(rank, score, K, EPS, SCALE, REAL)
    ttnn.deallocate(l32)
    ttnn.deallocate(rank)
    ttnn.deallocate(score)
    return weights, indices


def _time(mesh_device, fn, iters: int = ITERS) -> float:
    fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
def test_v41_decode_routing(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    epc = E // sp // tp
    torch.manual_seed(7)
    logits = torch.randn(sp * ROWS, E) * 1.5
    bias = torch.randn(E) * 0.02
    bias = bias - bias.mean()
    logits64 = torch.cat([logits.view(sp, ROWS, E)] * (OLD_ROWS // ROWS), dim=1).reshape(sp * OLD_ROWS, E)

    tt_logits = _sharded_rows(mesh_device, logits, ttnn.bfloat16)
    tt_logits64 = _sharded_rows(mesh_device, logits64, ttnn.bfloat16)
    bias64 = _replicated(mesh_device, bias.repeat(OLD_ROWS).view(OLD_ROWS, E), ttnn.bfloat16)
    bias32 = _replicated(mesh_device, bias.repeat(ROWS).view(ROWS, E), ttnn.float32)
    zero32 = _replicated(mesh_device, torch.zeros(ROWS, E), ttnn.float32)
    padding64 = _sharded_rows(
        mesh_device, torch.tensor([[REAL, 0]] * sp, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    table = ExpertMapping.create_dispatch_table(E, sp, tp)
    setup = TtMoERoutingSetup(mesh_device, table, num_links=fabric_num_links(), experts_per_chip=epc)
    new_setup = DecodeRoutingSetup(setup, mesh_device, ROWS, K, REAL)

    # --- the gate: the same set per row, or a TF32-level near-tie at the boundary (the old kernel's resolution) ---
    old_w, old_i = _old_gate(tt_logits64, bias64, padding64)
    new_w, new_i = _new_gate(tt_logits, zero32, bias32)
    host_rank = torch.sqrt(torch.nn.functional.softplus(logits.bfloat16().float())) + bias.bfloat16().float()
    near_ties, exact_weights, rows = 0, 0, 0
    for chip, (ow, oi, nw, ni) in enumerate(
        zip(_per_device(old_w), _per_device(old_i), _per_device(new_w), _per_device(new_i))
    ):
        oi, ni = oi.int(), ni.int()
        assert tuple(ni.shape) == (ROWS, K) and tuple(nw.shape) == (ROWS, K), (tuple(ni.shape), tuple(nw.shape))
        assert (ni[REAL:] == E).all() and (oi[REAL:] == E).all(), f"chip {chip}: pad rows not sentinel"
        assert (nw[REAL:].float() == 0).all(), f"chip {chip}: pad rows' weights not zero"
        rank_rows = host_rank[(chip // tp) * ROWS : (chip // tp) * ROWS + REAL]
        for r in range(REAL):
            rows += 1
            new_set, old_set = set(ni[r].tolist()), set(oi[r].tolist())
            assert len(new_set) == K, f"chip {chip} row {r}: repeated expert {ni[r].tolist()}"
            if new_set != old_set:
                # the experts the two select differently must rank within one TF32 ulp of each other
                diff = sorted(new_set ^ old_set)
                ranks = rank_rows[r, diff]
                spread = (ranks.max() - ranks.min()) / ranks.max()
                logger.warning(
                    f"chip {chip} row {r}: {sorted(new_set)} vs {sorted(old_set)}: {diff} spread {spread:.2e}"
                )
                assert (
                    spread < 2**-9
                ), f"chip {chip} row {r}: not a near-tie: {[(c, rank_rows[r, c].item()) for c in diff]}"
                near_ties += 1
                continue
            o = {c: w for c, w in zip(oi[r].tolist(), ow[r].float().tolist())}
            n = {c: w for c, w in zip(ni[r].tolist(), nw[r].float().tolist())}
            exact_weights += all(o[c] == n[c] for c in new_set)
            for c in new_set:
                assert (
                    abs(n[c] - o[c]) <= abs(o[c]) * 2**-7 + 1e-6
                ), f"chip {chip} row {r} expert {c}: {n[c]} vs {o[c]}"
    assert near_ties <= rows // 20, f"{near_ties} of {rows} rows select differently"
    logger.info(
        f"gate: {rows - near_ties} of {rows} rows the same set ({exact_weights} with bit-exact weights), "
        f"{near_ties} near-ties at the boundary"
    )

    # --- the routing setup, both paths on the new gate's indices ---
    per_rank = [_per_device(new_i)[rank * tp].int() for rank in range(sp)]  # the TP chips of a rank are the same
    idx64 = torch.full((sp, OLD_ROWS, K), E, dtype=torch.int32)
    for rank, rows_ in enumerate(per_rank):
        idx64[rank, :REAL] = rows_[:REAL]
    tt_idx64 = _sharded_rows(mesh_device, idx64.reshape(sp * OLD_ROWS, K), ttnn.uint16)
    old = setup(tt_idx64, E, K)
    new = new_setup(new_i, E, K)
    for name, o, n in zip(("offsets", "totals", "region"), old[:3], new[:3]):
        for chip, (a, b) in enumerate(zip(_per_device(o), _per_device(n))):
            a, b = a.flatten().int(), b.flatten().int()
            assert tuple(b.shape) == (E,), tuple(b.shape)
            assert torch.equal(a, b), f"chip {chip} {name}: {(a != b).nonzero().flatten().tolist()[:16]}"
    totals = _per_device(new[1])[0].flatten().int()
    logger.info(f"routing setup: offsets, totals and region offsets exact; {int((totals > 0).sum())} experts active")

    # --- eager timing (a bound; the replay's device profile is the measure) ---
    t_old_gate = _time(mesh_device, lambda: _old_gate(tt_logits64, bias64, padding64))
    t_new_gate = _time(mesh_device, lambda: _new_gate(tt_logits, zero32, bias32))
    t_old_setup = _time(mesh_device, lambda: setup(tt_idx64, E, K))
    t_new_setup = _time(mesh_device, lambda: new_setup(new_i, E, K))
    logger.info(
        f"eager us per call: gate {t_old_gate:.0f} -> {t_new_gate:.0f}; routing setup {t_old_setup:.0f} -> "
        f"{t_new_setup:.0f}"
    )


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_decode_routing_recall(mesh_device, device_params):
    """Both gates on the reference's real gate logits (layer 2, 5,120 tokens of the oracle's chunk, as bf16, with
    the checkpoint's bias recentred as ``TtV41Moe`` stores it), scored by top-6 recall against the reference's own
    selection: which selection rule, the exact fp32 one or the TF32-keyed sort, is the reference's more often."""
    from models.demos.deepseek_v3_d_p.tests.v41.moe_reference import LAYER, reference
    from models.demos.deepseek_v3_d_p.tt.v41 import weights as W

    sp, tp = tuple(mesh_device.shape)
    ref, _ = reference("real")
    ref_logits, ref_idx = ref["logits"].float(), ref["indices"].long()
    tokens = ref_logits.shape[0]
    bias = W.load_layer_dense(W.resolve_checkpoint(), LAYER)["gate_weights"]["e_score_correction_bias"].float()
    bias = (bias - bias.mean()).bfloat16().float()
    per_call = sp * ROWS
    bias64 = _replicated(mesh_device, bias.repeat(OLD_ROWS).view(OLD_ROWS, E), ttnn.bfloat16)
    bias32 = _replicated(mesh_device, bias.repeat(ROWS).view(ROWS, E), ttnn.float32)
    zero32 = _replicated(mesh_device, torch.zeros(ROWS, E), ttnn.float32)
    padding64 = _sharded_rows(
        mesh_device, torch.tensor([[ROWS, 0]] * sp, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    old_sel, new_sel = [], []
    for start in range(0, tokens - per_call + 1, per_call):
        chunk = ref_logits[start : start + per_call].bfloat16().float()  # the device gate's bf16 logits
        logits64 = torch.cat([chunk.view(sp, ROWS, E), torch.zeros(sp, OLD_ROWS - ROWS, E)], dim=1)
        tt_logits = _sharded_rows(mesh_device, chunk, ttnn.bfloat16)
        tt_logits64 = _sharded_rows(mesh_device, logits64.reshape(sp * OLD_ROWS, E), ttnn.bfloat16)
        _, oi = _old_gate(tt_logits64, bias64, padding64)
        rank = ttnn.add(
            ttnn.add(
                ttnn.typecast(tt_logits, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                zero32,
                input_tensor_a_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
                ],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            bias32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        score = ttnn.subtract(rank, bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # the weights are not scored here
        _, ni = router_select(rank, score, K, EPS, SCALE, ROWS)
        old_sel.append(torch.cat([_per_device(oi)[r * tp].int()[:ROWS] for r in range(sp)]))
        new_sel.append(torch.cat([_per_device(ni)[r * tp].int()[:ROWS] for r in range(sp)]))
    old_sel, new_sel = torch.cat(old_sel).long(), torch.cat(new_sel).long()
    n = old_sel.shape[0]
    ref_n = ref_idx[:n]

    def recall(sel):
        hits = (sel.unsqueeze(2) == ref_n.unsqueeze(1)).any(dim=2)
        return hits.float().sum(dim=1) / K

    r_old, r_new = recall(old_sel), recall(new_sel)
    same = torch.tensor([set(a.tolist()) == set(b.tolist()) for a, b in zip(old_sel, new_sel)])
    differ = (~same).nonzero().flatten()
    old_better = int((r_old[differ] > r_new[differ]).sum())
    new_better = int((r_new[differ] > r_old[differ]).sum())
    logger.info(
        f"recall against the reference over {n} real tokens: old {r_old.mean():.5f}, new {r_new.mean():.5f}; "
        f"the two select the same set on {int(same.sum())} rows; of the {len(differ)} rows that differ, "
        f"the old is nearer the reference on {old_better}, the new on {new_better}, neither on "
        f"{len(differ) - old_better - new_better}"
    )
    assert r_new.mean() >= r_old.mean() - 1e-4, (r_old.mean().item(), r_new.mean().item())
