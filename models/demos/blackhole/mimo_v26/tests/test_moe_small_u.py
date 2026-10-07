# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small-U decode MoE table: MiMoMoEDecode at U in {1,2,4,8,16,32} tokens per mesh row (MIMO_UVALS) on REAL layers (MIMO_LAYERS=1,5,16; L1/L5 dump, L16 device capture dump; real weights).
Per U: variants dead=spread (old: dead slots round-robin over all experts) vs dead=sentinel (dead slots routed to the sentinel expert: never dispatched). For each: PCC and (alpha, rel) of the live
outputs vs the dump ffn_out, 5 eager runs bit-identical, trace == eager, ALTERNATING traced time per layer (3 rounds, capture/replay/release), and the cumulative stage breakdown of the sentinel variant.
Needs the patched _ttnncpp.so (LD_LIBRARY_PATH). Router: MIMO_ROUTER (fast)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_decode import MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
D2 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,5,16").split(",")]
UVALS = [int(v) for v in os.environ.get("MIMO_UVALS", "1,2,4,8,16,32").split(",")]
ROUTER = os.environ.get("MIMO_ROUTER", "fast")
REPLAYS = int(os.environ.get("MIMO_REPLAYS", "30"))
VARIANTS = os.environ.get("MIMO_DEAD", "spread,sentinel").split(",")


def stats(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (b @ b)), float((a - b).norm() / b.norm())


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
def test_moe_small_u(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    dec, xs, refs = {}, {}, {}
    sel = torch.arange(0, 512, 16)[:32]  # 32 tokens spread over the 4 users of a mesh row
    for L in LAYERS:
        a = torch.load(f"{D1 if L <= 5 else D2}/layer_{L}.pt", mmap=True)["prefill"]
        xs[L] = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].clone()  # [4, 32, D]
        refs[L] = a["ffn_out"].reshape(ROWS, 512, HIDDEN)[:, sel].float()
        W = load_layer_experts(L)
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=True)
        del W
        gw, gb = load_gate(L)
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))
        moe = MiMoPrefillMoE(
            md, wts, ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=True), expert_op="unified", cf=4
        )
        dec[L] = MiMoMoEDecode(md, gate, moe, router=ROUTER)
    d0 = dec[LAYERS[0]]
    mask_dev = d0.alloc_mask(d0.make_mask_host([32] * ROWS))
    failures = []
    for U in UVALS:
        xd = {}
        for L in LAYERS:
            xd[L] = ttnn.from_torch(
                xs[L][:, :U].reshape(ROWS, 1, U, HIDDEN),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=m2,
            )
        d0.update_mask(mask_dev, d0.make_mask_host([U] * ROWS))

        def run_all(upto=99):
            return [dec[L].forward(xd[L], mask_dev, upto=upto) for L in LAYERS]

        def to_host(t):
            d = ttnn.get_device_tensors(t)
            return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(-1, HIDDEN).float()[:U] for r in range(ROWS)])

        def cfg(dead):
            for L in LAYERS:
                dec[L].dead = dead
                rep = ttnn.ReplicateTensorToMesh(md)
                sidx = torch.full((1, 64, 8), 256) if dead == "sentinel" else None
                from models.demos.blackhole.mimo_v26.tt.moe_prefill import spread_pad_routing

                si, sw = spread_pad_routing(1, 64)
                if dead == "sentinel":
                    si, sw = torch.full_like(si, 256), torch.zeros_like(sw)
                for name, t in (("sp_idx", si), ("sp_w", sw)):
                    old = getattr(dec[L], name)
                    setattr(
                        dec[L],
                        name,
                        ttnn.from_torch(
                            t.reshape(1, 1, 64, 8).bfloat16(),
                            device=md,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=rep,
                        ),
                    )
                    ttnn.deallocate(old)

        res = {}
        for dead in VARIANTS:
            cfg(dead)
            outs = run_all()
            ttnn.synchronize_device(md)
            h = [to_host(o) for o in outs]
            ident = True
            for o in outs:
                ttnn.deallocate(o)
            for _ in range(4):
                o2 = run_all()
                ttnn.synchronize_device(md)
                ident &= all(bool(torch.equal(to_host(o), e)) for o, e in zip(o2, h))
                for o in o2:
                    ttnn.deallocate(o)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            to = run_all()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            same = all(bool(torch.equal(to_host(o), e)) for o, e in zip(to, h))
            ttnn.release_trace(md, tid)
            pc = [round(min(pcc(h[i][r], refs[L][r, :U]) for r in range(ROWS)), 5) for i, L in enumerate(LAYERS)]
            st = [stats(h[i], refs[L][:, :U]) for i, L in enumerate(LAYERS)]
            print(
                f"RESULT U={U} dead={dead}: min PCC per layer {dict(zip(LAYERS, pc))}; (alpha, rel) vs dump ffn_out {dict(zip(LAYERS, [(round(a, 4), round(r, 4)) for a, r in st]))}; 5 eager runs identical {ident}; trace==eager {same}",
                flush=True,
            )
            if not (ident and same):
                failures.append((U, dead, ident, same))
            res[dead] = h
        if len(VARIANTS) > 1:
            print(
                f"RESULT U={U}: live outputs sentinel vs spread identical: {all(bool(torch.equal(a, b)) for a, b in zip(res['sentinel'], res['spread']))} (max abs diff {max(float((a - b).abs().max()) for a, b in zip(res['sentinel'], res['spread'])):.4f})",
                flush=True,
            )
        # alternating timing
        tm = {d: [] for d in VARIANTS}
        for rnd in range(3):
            for dead in VARIANTS:
                cfg(dead)
                run_all()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                run_all()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                for _ in range(5):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                t0 = time.time()
                for _ in range(REPLAYS):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                tm[dead].append((time.time() - t0) / REPLAYS * 1e3 / len(LAYERS))
                ttnn.release_trace(md, tid)
        print("RESULT TIME U=%d: %s" % (U, {d: f"{sorted(v)[1]:.3f} ms/layer" for d, v in tm.items()}), flush=True)
        # stage breakdown (last variant = sentinel by default)
        cfg(VARIANTS[-1])
        prev = 0.0
        names = {
            0: "pad+router+mask",
            1: "+counts",
            2: "+dispatch",
            3: "+experts",
            4: "+combine",
            5: "+post_combine_reduce",
            6: "+all_reduce+slice",
        }
        row = []
        for st in range(7):
            fn = lambda st=st: run_all(upto=st if st < 6 else 99)
            outs = fn()
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            fn()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            t0 = time.time()
            for _ in range(REPLAYS):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            v = (time.time() - t0) / REPLAYS * 1e3 / len(LAYERS)
            ttnn.release_trace(md, tid)
            row.append(f"{names[st]} {v - prev:+.3f}")
            prev = v
        print(f"RESULT STAGES U={U} ({VARIANTS[-1]}, ms/layer deltas, total {prev:.3f}): " + "; ".join(row), flush=True)
    assert not failures, failures
