# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gate test of the RING decode MoE (tt/moe_ring.py) vs the unified decode MoE (tt/moe_decode.py, already gated vs the fp32 oracle) on REAL layers 1, 5, 16 (MIMO_LAYERS) on 4x8:
for T tokens per mesh row in MIMO_TVALS (1,2,4,8,16,32; T = 8U covers the U<=4 verify path), live slots MIMO_LIVE_FRAC of T (dead slots exercise the zero-weight dead routing):
  alpha / rel L2 / PCC of ring vs unified (live tokens) and of both vs the dump ffn_out (L1, L5 bf16 dump; L16 device capture dump), 5 eager runs bit-identical, trace == eager,
  ALTERNATING traced time ring vs unified (3 rounds, capture/replay/release per round), free DRAM per bank after both copies.
Needs the patched library (LD_LIBRARY_PATH) and the weight cache dir (MIMO_WCACHE)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_decode import MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_ring import MiMoRingMoEDecode, RingExpertWeights
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
D2 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,5,16").split(",")]
TVALS = [int(v) for v in os.environ.get("MIMO_TVALS", "1,2,4,8,16,32").split(",")]
FRAC = float(os.environ.get("MIMO_LIVE_FRAC", "0.75"))
REPLAYS = int(os.environ.get("MIMO_REPLAYS", "30"))


def stats(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (b @ b)), float((a - b).norm() / b.norm()), float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def free_mb(md):
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    return mv.total_bytes_free_per_bank / 2**20 if hasattr(mv, "total_bytes_free_per_bank") else float("nan")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_ring_layers(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    sel = torch.arange(0, 512, 16)[:32]
    uni, ring, xs, refs = {}, {}, {}, {}
    print(f"RESULT free DRAM per bank before: {free_mb(md):.0f} MB", flush=True)
    for L in LAYERS:
        a = torch.load(f"{D1 if L <= 5 else D2}/layer_{L}.pt", mmap=True)["prefill"]
        xs[L] = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].clone()
        refs[L] = a["ffn_out"].reshape(ROWS, 512, HIDDEN)[:, sel].float()
        W = load_layer_experts(L)
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=True)
        gw, gb = load_gate(L)
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))
        moe = MiMoPrefillMoE(
            md, wts, ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=True), expert_op="unified", cf=4
        )
        uni[L] = MiMoMoEDecode(md, gate, moe, router="fast", dead="spread32")
        t0 = time.time()
        ex = RingExpertWeights(md, L, w_of_expert=lambda e: W[e])
        ring[L] = MiMoRingMoEDecode(md, gate, ex, router="fast")
        print(
            f"RESULT layer {L}: ring weights {'loaded from cache' if ex.warm else 'built+cached'} in {time.time() - t0:.0f}s; free DRAM/bank {free_mb(md):.0f} MB",
            flush=True,
        )
        del W
    d0 = uni[LAYERS[0]]
    failures = []
    for T in TVALS:
        nlive = max(1, int(round(T * FRAC)))
        xd = {}
        for L in LAYERS:
            x = xs[L][:, :T].clone()
            x[:, nlive:] = 0
            xd[L] = ttnn.from_torch(
                x.reshape(ROWS, 1, T, HIDDEN),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=m2,
            )
        mask = d0.alloc_mask(d0.make_mask_host([nlive] * ROWS))

        def run_all(which, upto=99):
            dd = uni if which == "unified" else ring
            return [dd[L].forward(xd[L], mask) for L in LAYERS]

        def host(t):
            d = ttnn.get_device_tensors(t)
            return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(-1, HIDDEN).float()[:nlive] for r in range(ROWS)])

        res, outs = {}, {}
        for which in ("unified", "ring"):
            o = run_all(which)
            ttnn.synchronize_device(md)
            h = [host(t) for t in o]
            ident = True
            for _ in range(4):
                o2 = run_all(which)
                ttnn.synchronize_device(md)
                ident &= all(bool(torch.equal(host(t), e)) for t, e in zip(o2, h))
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            ot = run_all(which)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            same = all(bool(torch.equal(host(t), e)) for t, e in zip(ot, h))
            ttnn.release_trace(md, tid)
            outs[which] = h
            res[which] = (ident, same)
        for i, L in enumerate(LAYERS):
            r_u = stats(outs["ring"][i], outs["unified"][i])
            r_d = stats(outs["ring"][i], refs[L][:, :nlive])
            u_d = stats(outs["unified"][i], refs[L][:, :nlive])
            print(
                f"RESULT T={T} live={nlive} L{L}: ring vs unified alpha {r_u[0]:.4f} rel {r_u[1]:.4f} PCC {r_u[2]:.5f}; vs dump: ring (alpha {r_d[0]:.4f} rel {r_d[1]:.4f} PCC {r_d[2]:.5f}) unified (alpha {u_d[0]:.4f} rel {u_d[1]:.4f} PCC {u_d[2]:.5f})",
                flush=True,
            )
        print(
            f"RESULT T={T}: 5 eager runs identical / trace==eager: unified {res['unified']} ring {res['ring']}",
            flush=True,
        )
        if not (all(res["ring"])):
            failures.append((T, res))
        tm = {w: [] for w in ("unified", "ring")}
        for rnd in range(3):
            for which in ("unified", "ring"):
                run_all(which)
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                run_all(which)
                ttnn.end_trace_capture(md, tid, cq_id=0)
                for _ in range(5):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                t0 = time.time()
                for _ in range(REPLAYS):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                tm[which].append((time.time() - t0) / REPLAYS * 1e3 / len(LAYERS))
                ttnn.release_trace(md, tid)
        print(
            "RESULT TIME T=%d (ms/layer, alternating, host-timed): %s"
            % (T, {w: f"{sorted(v)[1]:.3f}" for w, v in tm.items()}),
            flush=True,
        )
    print(f"RESULT free DRAM per bank after both copies of {len(LAYERS)} layers: {free_mb(md):.0f} MB", flush=True)
    assert not failures, failures
