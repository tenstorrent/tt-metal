# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2: DSpark drafter on the device vs the CPU reference (reference/ref_spec_mtp.py -> mtp_ref.pt, built from the GSM8K greedy run of ref_spec_accept).
Per round: ``write_main`` for a block of n verify positions (hidden states of the 3 target layers) then ``draft`` -> 5 draft tokens, markov-biased logits and
confidence vs the reference; then a traced timing of (write_main + draft).
Env: DSV41_TILE (users = 8 * TILE, default 1 -> 2 users/row, 2 -> 4 users/row), DSV41_NV (verify rows per user n, default 1).
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import BLOCK, NOISE_ID, DSparkDrafter, load_mtp_stage
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

REF = os.environ.get("DSV41_MTP_REF", "/mnt/tt-data/ssinghal/dsv4-spec-accept/mtp_ref.pt")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_spec_mtp(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    tile = int(os.environ.get("DSV41_TILE", "1"))
    n = int(os.environ.get("DSV41_NV", "1"))
    ref = torch.load(REF)
    S = ref["S"]
    B0 = ref["ring"][0].shape[0]
    B = int(os.environ.get("DSV41_NUSERS", B0 * tile))  # DSV41_NUSERS=4 -> 1 user per mesh row (B=4)
    U = B // rows
    T_d = BLOCK * U
    rep = lambda t: t.repeat(tile, *([1] * (t.dim() - 1)))[:B]
    log = lambda m: print(m, flush=True)
    sh = _Shards()
    t0 = time.time()
    stage_w = [load_mtp_stage(i, sh) for i in range(3)]
    log(f"stage weights loaded {time.time() - t0:.0f}s")
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    emb = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=T_d)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=1e-20)
    t0 = time.time()
    dr = DSparkDrafter(md, mc, ccl, stage_w, emb.weight, head, users_per_row=U, n=n)
    ttnn.synchronize_device(md)
    log(f"drafter built {time.time() - t0:.0f}s")
    for s, a in enumerate(dr.attn):
        a.seed_ring(rep(ref["ring"][s]), S)
    sm = lambda dim: ttnn.ShardTensor2dMesh(md, dims=(dim, None), mesh_shape=(rows, cols))
    # persistent device inputs (refreshed in place -> usable inside a trace)
    hid_d = ttnn.from_torch(
        torch.zeros(1, 1, B * n, 15360),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sm(2),
    )
    posv_d = ttnn.from_torch(
        torch.zeros(B * n, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sm(0),
    )
    tok_d = ttnn.from_torch(
        torch.zeros(rows * T_d, 1, dtype=torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sm(0),
    )
    f_d = ttnn.from_torch(
        torch.zeros(rows * T_d, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sm(0),
    )

    def upload(t0_step):
        steps = ref["steps"][t0_step : t0_step + n]
        hid = torch.stack([rep(s["main_hidden"]) for s in steps], 1).reshape(B * n, 15360)  # user-major rows
        pos = (torch.full((B, 1), S + t0_step) + torch.arange(n).reshape(1, n)).reshape(-1)
        tokt = rep(steps[-1]["tok"])  # [B]
        tok = torch.full((rows, BLOCK, U), NOISE_ID, dtype=torch.int64)
        tok[:, 0, :] = tokt.reshape(rows, U)
        f = torch.full((B,), S + t0_step + n - 1)
        fr = f.reshape(rows, 1, U).repeat(1, BLOCK, 1).reshape(rows * T_d)
        h = lambda t, dt, lay, dim: ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=sm(dim))
        ttnn.copy_host_to_device_tensor(
            h(hid.reshape(1, 1, B * n, 15360).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, 2), hid_d
        )
        ttnn.copy_host_to_device_tensor(h(pos.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0), posv_d)
        ttnn.copy_host_to_device_tensor(
            h(tok.reshape(rows * T_d, 1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 0), tok_d
        )
        ttnn.copy_host_to_device_tensor(h(fr.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0), f_d)

    def run():
        dr.write_main(hid_d, dr.state.build_verify(posv_d))
        return dr.draft(tok_d, f_d)

    host_tok = lambda t: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1) for r in range(rows)]
    ).long()
    results = []
    for t0_step in range(0, len(ref["steps"]) - n + 1, n):
        st = ref["steps"][t0_step + n - 1]
        upload(t0_step)
        out = run()
        ttnn.synchronize_device(md)
        toks = torch.stack([host_tok(t) for t in out["tokens"]], 1)  # [B, 5] = d_1..d_5
        want = rep(st["drafts"])[:, 1:]
        pcc_l = []
        for i in range(BLOCK):
            lg = (
                ttnn.to_torch(
                    out["logits"][i], mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
                )
                .reshape(B, -1)
                .float()
            )
            pcc_l.append(R.pcc(lg, rep(st["logits"][:, i])))
        conf = torch.cat(
            [
                ttnn.to_torch(ttnn.get_device_tensors(out["conf"])[r * cols])[0, 0, :, 0].reshape(BLOCK, U).T
                for r in range(rows)
            ]
        ).float()  # [B, 5]
        pc = R.pcc(conf, rep(st["conf"]))
        agree = [(toks[:, i] == want[:, i]).float().mean().item() for i in range(BLOCK)]
        log(
            f"step {t0_step + n - 1} (n={n}): draft token agreement per position {[round(a, 2) for a in agree]}  logits PCC {[round(p, 4) for p in pcc_l]}  conf PCC {pc:.4f}"
        )
        results.append((agree, pcc_l, pc))
    # traced timing
    upload(0)
    run()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    run()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    N = 20
    t = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    log(f"MTP_TIMING users={B} U={U} n={n}: write_main + draft traced {(time.perf_counter() - t) / N * 1e3:.2f} ms")
    assert (
        min(sum(r[1][i] for r in results) / len(results) for i in range(BLOCK)) > 0.95
    ), results  # mean over steps per draft position (one near-tie flip lowers a single step)

    # breakdown (cumulative, traced): write_main alone, then draft truncated after each part
    def timed(fn):
        fn()
        ttnn.synchronize_device(md)
        tid2 = ttnn.begin_trace_capture(md, cq_id=0)
        fn()
        ttnn.end_trace_capture(md, tid2, cq_id=0)
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(md, tid2, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t0) / N * 1e3
        ttnn.release_trace(md, tid2)
        return r

    res_b = {"write_main": timed(lambda: dr.write_main(hid_d, dr.state.build_verify(posv_d)))}
    for stop in ("embed", "stage0", "stage1", "stage2", "head", None):
        dr.stop = stop
        res_b[f"draft<={stop}"] = timed(lambda: dr.draft(tok_d, f_d))
    dr.stop = None
    log("MTP_BREAKDOWN (cumulative ms) " + ", ".join(f"{k_} {v:.2f}" for k_, v in res_b.items()))
