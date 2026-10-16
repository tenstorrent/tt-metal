# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check and time the MoE decode routing kernel (tt/moe_route.py) on Galaxy row 0.

Real layer-1 router logits of the golden prompts (fp32, from the checkpoint's router), 4 users per step:
the device sigmoid + bias as in TtMiMoMoE, then the kernel. Compared against an fp32 torch reference
(top-8 set and weights) and against the current ttnn.topk / gather path. Prints the kernel time traced.

    python models/autoports/xiaomimimo_mimo_v2_6_flash_rl/tests/perf/moe_route_check.py
"""

import time

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import MeshPlan
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.moe_route import MoERoute

USERS = 4
K = 8
LAUNCHES = 50


def main():
    cfg = common.config()
    ckpt = common.checkpoint(cfg)
    rows = common.golden_moe_inputs(cfg, ckpt)
    router = ckpt.router(common.LAYER)
    logits_all = rows.float() @ router["weight"].float().T  # [N, E] fp32
    bias = router["e_score_correction_bias"].float()
    steps = logits_all.shape[0] // USERS

    mesh, row = common.open_row(trace_region_size=16 << 20)
    try:
        plan = MeshPlan(row)
        rep = plan.replicate()
        route = MoERoute(plan, cfg.n_routed_experts, K, USERS)
        bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), device=row, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )
        mask = torch.zeros(1, 1, 32, 1)
        mask[0, 0, :USERS] = 1
        mask_tt = ttnn.from_torch(mask, device=row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)

        set_match = ref_match = 0
        worst = 0.0
        for s in range(steps):
            lg = torch.zeros(1, 1, 32, cfg.n_routed_experts)
            lg[0, 0, :USERS] = logits_all[s * USERS : (s + 1) * USERS]
            lg_tt = ttnn.from_torch(lg, device=row, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
            scores = ttnn.sigmoid(lg_tt)
            biased = ttnn.add(scores, bias_tt)
            ids, w = route(scores, biased, mask_tt)
            got_ids = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).to(torch.int64).reshape(USERS, K)
            got_w = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float().reshape(USERS, K)
            # fp32 torch reference on the device's own sigmoid (isolates the kernel)
            sc = ttnn.to_torch(ttnn.get_device_tensors(scores)[0]).float()[0, 0, :USERS]
            bi = ttnn.to_torch(ttnn.get_device_tensors(biased)[0]).float()[0, 0, :USERS]
            ref_idx = bi.topk(K, dim=-1).indices
            ref_w = sc.gather(-1, ref_idx)
            ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20)
            # full fp32 reference from the logits (what TtMiMoMoE aims at)
            s_true = torch.sigmoid(logits_all[s * USERS : (s + 1) * USERS])
            true_idx = (s_true + bias).topk(K, dim=-1).indices
            for u in range(USERS):
                set_match += set(got_ids[u].tolist()) == set(ref_idx[u].tolist())
                ref_match += set(got_ids[u].tolist()) == set(true_idx[u].tolist())
                for i in range(K):
                    j = (ref_idx[u] == got_ids[u, i]).nonzero()
                    if len(j):
                        worst = max(worst, abs(got_w[u, i].item() - ref_w[u, j[0, 0]].item()))
            for t in (lg_tt, scores, biased):
                t.deallocate(True)
        n = steps * USERS
        print(f"top-8 set == device-sigmoid reference: {set_match}/{n}; == fp32 logits reference: {ref_match}/{n}")
        print(f"max |weight - reference| {worst:.3e}")

        # Timing: kernel alone, traced
        lg = torch.zeros(1, 1, 32, cfg.n_routed_experts)
        lg[0, 0, :USERS] = logits_all[:USERS]
        lg_tt = ttnn.from_torch(lg, device=row, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        scores = ttnn.sigmoid(lg_tt)
        biased = ttnn.add(scores, bias_tt)
        route(scores, biased, mask_tt)
        ttnn.synchronize_device(row)
        trace = ttnn.begin_trace_capture(row, cq_id=0)
        for _ in range(LAUNCHES):
            route(scores, biased, mask_tt)
        ttnn.end_trace_capture(row, trace, cq_id=0)
        times = []
        for _ in range(5):
            ttnn.synchronize_device(row)
            t0 = time.perf_counter()
            ttnn.execute_trace(row, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(row)
            times.append((time.perf_counter() - t0) / LAUNCHES)
        ttnn.release_trace(row, trace)
        print(f"MOE_ROUTE_US {sorted(times)[2] * 1e6:.2f} per launch (traced, includes dispatch gaps)", flush=True)
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
