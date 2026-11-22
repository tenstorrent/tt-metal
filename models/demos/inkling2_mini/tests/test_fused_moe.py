# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE expert step (tt/fused/moe_expert.py) vs torch and vs the baseline ``Inkling2MoE.experts``.

    flock /tmp/tt-card.lock pytest -s models/demos/inkling2_mini/tests/test_fused_moe.py -k unit

``unit``: 16 experts, routing fixed on the host, torch golden on the device's own (round-tripped) weights.
``moe``:  the T4 MoE test with the fused expert step swapped in (HF reference, both dtypes, trace x2, timing).
"""

import json
import math
import os
import time

import pytest
import torch

import ttnn

HIDDEN, INTER, TOP_K = 2304, 1152, 8


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _golden(x, gu, dn, ids, w):
    """x [H], gu [E,H,2I], dn [E,I,H] (device layouts), ids [8], w [8] -> [H] (fp32 math)."""
    out = torch.zeros(HIDDEN)
    for e, wt in zip(ids.tolist(), w.tolist()):
        y = x @ gu[e].float()
        g, u = y[:INTER], y[INTER:]
        h = torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25) * 4
        out += wt * (h @ dn[e].float())
    return out


DTYPES = {"bfp4": ("bfloat4_b", "bfloat4_b"), "bfp8": ("bfloat8_b", "bfloat8_b"), "mixed": ("bfloat4_b", "bfloat8_b")}
# E12_PLACE=offring: workers on logical rows 3..9 (11 x 7), collector a spare core of that rectangle, clear of the
# tensor prefetcher's receiver ring (0..6, 0..2).
PLACES = {"default": {}, "offring": {"origin": (0, 3), "shape": (11, 7), "collector": (10, 9)}}


def _place():
    return PLACES[os.environ.get("E12_PLACE", "default")]


def _time(device, fn, repeats=20, iters=20):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    start = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ttnn.release_trace(device, tid)
    return (time.perf_counter() - start) / iters / repeats * 1e6


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("dtype_name", ["bfp4", "bfp8", "mixed"])
@pytest.mark.parametrize("layout", ["tiles", "sticks"])
def test_fused_moe_unit(device, dtype_name, layout):
    from models.demos.inkling2_mini.tt.fused.moe_expert import FusedMoeExperts, stick_weights

    dt, dtd = (getattr(ttnn, n) for n in DTYPES[dtype_name])
    n_exp = int(os.environ.get("E12_UNIT_EXPERTS", "16"))
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(HIDDEN, generator=g)).bfloat16()
    gu = torch.cat(
        [
            torch.randn(n_exp, HIDDEN, INTER, generator=g) * (3.0 / math.sqrt(HIDDEN)),
            torch.randn(n_exp, HIDDEN, INTER, generator=g) * (8.0 / math.sqrt(HIDDEN)),
        ],
        dim=2,
    )
    dn = torch.randn(n_exp, INTER, HIDDEN, generator=g) / math.sqrt(INTER)
    ids = torch.randperm(n_exp, generator=g)[:TOP_K]
    w = torch.rand(TOP_K, generator=g) + 0.5

    dram = ttnn.DRAM_MEMORY_CONFIG
    gu_t = ttnn.from_torch(gu.unsqueeze(0), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    dn_t = ttnn.from_torch(dn.unsqueeze(0), dtype=dtd, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    gu_b, dn_b = ttnn.to_torch(gu_t)[0].float(), ttnn.to_torch(dn_t)[0].float()
    l1 = ttnn.L1_MEMORY_CONFIG
    x_t = ttnn.from_torch(x.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=l1)  # fmt: skip
    w_t = ttnn.from_torch(w.reshape(1, 1, 1, TOP_K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=l1)  # fmt: skip
    i_t = ttnn.from_torch(ids.reshape(1, 1, 1, TOP_K).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=l1)  # fmt: skip

    if layout == "sticks":  # same quantized values, stick order
        gu_t, dn_t = stick_weights(device, gu_b, dt), stick_weights(device, dn_b, dtd)
    fused = FusedMoeExperts(device, gu_t, dn_t, sticks=layout == "sticks", **_place())
    out = ttnn.to_torch(fused(x_t, w_t, i_t)).float()
    got = out.reshape(-1)[:HIDDEN] if out.numel() == HIDDEN else out[0, 0, 0, :HIDDEN]
    ref = _golden(x.float(), gu_b, dn_b, ids, w.bfloat16().float())
    pcc = _pcc(got, ref)
    us = _time(device, lambda: fused(x_t, w_t, i_t))
    res = {"dtype": dtype_name, "layout": layout, "place": os.environ.get("E12_PLACE", "default"), "pcc_vs_torch": round(pcc, 6), "fused_us": round(us, 1),
           "max_abs": float((got - ref).abs().max()), "ref_absmax": float(ref.abs().max())}  # fmt: skip
    print("E12-UNIT", json.dumps(res))
    assert torch.isfinite(got).all()
    assert pcc >= 0.999, res


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("dtype_name", ["bfp4", "bfp8", "mixed"])
@pytest.mark.parametrize("layout", ["tiles", "sticks"])
def test_fused_moe_block(device, dtype_name, layout):
    """T4's gate (tests/test_moe.py) with the expert step replaced by the fused program: PCC >= 0.999 vs HF on the
    round-tripped weights, top-8 overlap >= 0.97, trace replayed twice bit-equal; baseline vs fused expert time."""
    from models.demos.inkling2_mini.tests import test_moe as t4
    from models.demos.inkling2_mini.tt.fused.moe_expert import FusedMoeExperts, stick_weights
    from models.demos.inkling2_mini.tt.moe import Inkling2MoE, prepare_moe_weights

    expert_dtype, down_dtype = (getattr(ttnn, n) for n in DTYPES[dtype_name])
    x, norm_w, router_w, bias, gate_up, down = t4._random_inputs()
    weights = prepare_moe_weights(norm_w, router_w, bias, gate_up, down, expert_dtype)
    if down_dtype != expert_dtype:  # mixed layer: BFP4 gate_up + BFP8 down
        weights["experts.w_down"] = ttnn.from_torch(
            down.transpose(1, 2).unsqueeze(0).contiguous(), dtype=down_dtype, layout=ttnn.TILE_LAYOUT
        )
    gate_up_dev = ttnn.to_torch(weights["experts.w_gate_up"])[0].bfloat16()  # [E, H, 2I], round-tripped
    down_dev = ttnn.to_torch(weights["experts.w_down"])[0].bfloat16()  # [E, I, H]
    gate_up_b, down_b = gate_up_dev.transpose(1, 2), down_dev.transpose(1, 2)
    # The HF reference is CPU-only but slow; cache it so later card runs hold the lock for less time.
    cache = f"/work/inkling-bringup/E12-scratch/ref_{dtype_name}.pt"
    if os.path.exists(cache):
        r = torch.load(cache)
        refs, ref_idx = r["refs"], r["idx"]
    else:
        refs, ref_idx = t4._reference(x, norm_w, router_w, bias, {"A": (gate_up, down), "B": (gate_up_b, down_b)})
        torch.save({"refs": refs, "idx": ref_idx}, cache)
    del gate_up_b, down_b, gate_up, down

    moe = Inkling2MoE(device, weights)
    del weights
    if layout == "sticks":
        fused = FusedMoeExperts(
            device,
            stick_weights(device, gate_up_dev, expert_dtype),
            stick_weights(device, down_dev, down_dtype),
            sticks=True,
            **_place(),
        )
    else:
        fused = FusedMoeExperts(device, moe.w_gate_up, moe.w_down, **_place())
    del gate_up_dev, down_dev
    baseline_experts = moe.experts

    # baseline expert time first (same routing input), then swap in the fused step
    x_dev = t4._to_dev(x[0], device)
    x_norm = moe.norm(x_dev)
    w_k, ids_k = moe.route(x_norm)
    base_us = t4._time_traced(device, lambda: baseline_experts(x_norm, w_k, ids_k))
    fused_us = t4._time_traced(device, lambda: fused(x_norm, w_k, ids_k))
    base_out = ttnn.to_torch(baseline_experts(x_norm, w_k, ids_k)).reshape(-1)[:HIDDEN].float()
    fused_out = ttnn.to_torch(fused(x_norm, w_k, ids_k)).reshape(-1)[:HIDDEN].float()
    pcc_vs_base = _pcc(fused_out, base_out)
    base_block_us = t4._time_traced(device, lambda: moe(x_dev))
    moe.experts = fused  # same signature: (x, weights, indices) -> [1,1,1,H]

    outs, overlaps = [], []
    for t in range(t4.N_TOKENS):
        out, idx = moe(t4._to_dev(x[t], device))
        outs.append(ttnn.to_torch(out).reshape(-1)[:HIDDEN].float())
        got = set(ttnn.to_torch(idx).reshape(-1)[:TOP_K].long().tolist())
        overlaps.append(len(got & set(ref_idx[t].tolist())) / TOP_K)
    out_tt = torch.stack(outs)
    pcc_a, pcc_b = _pcc(out_tt, refs["A"]), _pcc(out_tt, refs["B"])
    overlap = sum(overlaps) / len(overlaps)

    moe(x_dev)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace, _ = moe(x_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    run1 = ttnn.to_torch(out_trace).reshape(-1)[:HIDDEN].float()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    run2 = ttnn.to_torch(out_trace).reshape(-1)[:HIDDEN].float()
    ttnn.release_trace(device, tid)
    trace_same = torch.equal(run1, run2)
    trace_vs_eager = _pcc(run1, outs[0])
    block_us = t4._time_traced(device, lambda: moe(x_dev))

    res = {
        "dtype": dtype_name,
        "layout": layout,
        "place": os.environ.get("E12_PLACE", "default"),
        "pcc_vs_B_roundtrip": round(pcc_b, 6),
        "pcc_vs_A_bf16": round(pcc_a, 6),
        "pcc_fused_vs_baseline_step": round(pcc_vs_base, 6),
        "top8_overlap": round(overlap, 4),
        "trace_same_twice": trace_same,
        "trace_vs_eager_pcc": round(trace_vs_eager, 6),
        "expert_step_us_baseline": round(base_us, 1),
        "expert_step_us_fused": round(fused_us, 1),
        "block_us_baseline": round(base_block_us, 1),
        "block_us_fused": round(block_us, 1),
    }
    print("E12-RESULT", json.dumps(res))
    if os.environ.get("E12_RESULTS"):
        with open(os.environ["E12_RESULTS"], "a") as f:
            f.write(json.dumps(res) + "\n")
    assert overlap >= 0.97, res
    assert pcc_b >= 0.999, res
    assert trace_same and trace_vs_eager > 0.9999, res


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("dtype_name", ["bfp4", "mixed"])
def test_fused_moe_stress(device, dtype_name):
    """Multi-step traced run: one trace holds STEPS MoE blocks (fused expert step) on different tokens,
    replayed REPLAYS times; every replay must be bit-equal to the first and match eager."""
    from models.demos.inkling2_mini.tests import test_moe as t4
    from models.demos.inkling2_mini.tt.fused.moe_expert import FusedMoeExperts
    from models.demos.inkling2_mini.tt.moe import Inkling2MoE, prepare_moe_weights

    steps, replays = int(os.environ.get("E12_STEPS", "16")), int(os.environ.get("E12_REPLAYS", "30"))
    expert_dtype, down_dtype = (getattr(ttnn, n) for n in DTYPES[dtype_name])
    # Host RSS <= 4 GB: only n_w experts exist on device; a +10 router bias keeps every top-8 inside them.
    n_w = int(os.environ.get("E12_STRESS_EXPERTS", "32"))
    g = torch.Generator().manual_seed(3)
    x = (3.0 * torch.randn(t4.N_TOKENS, HIDDEN, generator=g)).bfloat16()
    norm_w = 1.0 + 0.1 * torch.randn(HIDDEN, generator=g)
    router_w = torch.randn(t4.NUM_EXPERTS, HIDDEN, generator=g) / math.sqrt(HIDDEN)
    bias = 0.05 * torch.randn(t4.NUM_EXPERTS, generator=g)
    bias[:n_w] += 10.0
    weights = prepare_moe_weights(
        norm_w, router_w, bias, torch.zeros(1, 2 * INTER, HIDDEN), torch.zeros(1, HIDDEN, INTER), expert_dtype
    )
    gu = torch.cat(
        [torch.randn(n_w, HIDDEN, INTER, generator=g) * (3.0 / math.sqrt(HIDDEN)),
         torch.randn(n_w, HIDDEN, INTER, generator=g) * (8.0 / math.sqrt(HIDDEN))], dim=2
    ).bfloat16()  # fmt: skip
    dn = (torch.randn(n_w, INTER, HIDDEN, generator=g) / math.sqrt(INTER)).bfloat16()
    weights["experts.w_gate_up"] = ttnn.from_torch(gu.unsqueeze(0), dtype=expert_dtype, layout=ttnn.TILE_LAYOUT)
    weights["experts.w_down"] = ttnn.from_torch(dn.unsqueeze(0), dtype=down_dtype, layout=ttnn.TILE_LAYOUT)
    del gu, dn
    moe = Inkling2MoE(device, weights)
    del weights
    moe.experts = FusedMoeExperts(device, moe.w_gate_up, moe.w_down, **_place())
    xs = [t4._to_dev(x[t % t4.N_TOKENS], device) for t in range(steps)]
    eager = [ttnn.to_torch(moe(xi)[0]).reshape(-1)[:HIDDEN].float() for xi in xs]
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [moe(xi)[0] for xi in xs]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    first = None
    for r in range(replays):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = [ttnn.to_torch(o).reshape(-1)[:HIDDEN].float() for o in outs]
        if first is None:
            first = got
            assert all(_pcc(g, e) > 0.9999 for g, e in zip(got, eager))
        assert all(torch.equal(g, f) for g, f in zip(got, first)), f"replay {r} differs"
    ttnn.release_trace(device, tid)
    print("E12-STRESS", json.dumps({"dtype": dtype_name, "place": os.environ.get("E12_PLACE", "default"),
                                    "steps": steps, "replays": replays, "ok": True}))  # fmt: skip
