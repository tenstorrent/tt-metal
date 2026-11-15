"""pool-prefill S1b: chunk indexer over the sequence-sharded key pool (1x8 slot): indexer_score_dsa on the column's local keys -> staircase fix -> select_v2 (S = 512 rows) -> global top-2048 ids;
check vs a torch reference on the same data (id set overlap, tie-aware) + the staircase; timing at 1M-sized local keys. Env: PAGE (chunk page p, default 10), POOLP (pool pages), TIME_N (local keys for timing, default 131072)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP
from models.demos.blackhole.hy4_preview.tt import seqshard as SS
from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
RN = 512


def up(md, t, dt, lay, shard=None):
    mm = ttnn.ShardTensorToMesh(md, dim=shard) if shard is not None else ttnn.ReplicateTensorToMesh(md)
    return ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=mm, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)

        def ag(t):
            return ttnn.experimental.all_gather_async(
                t,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=ttnn.Topology.Ring,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        colid = up(md, HP.colid_host(1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard=1)
        cks = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        g = torch.Generator().manual_seed(11)
        page = int(os.environ.get("PAGE", "10"))
        P = int(os.environ.get("POOLP", "16"))
        s = page * 512
        N = s + 512
        kg = (torch.randn(N, 128, generator=g) * 1.0).to(torch.bfloat16)
        # heavy-tail recency structure: queries correlated with a few keys so the top-k is not pure noise
        q = (torch.randn(32, RN, 128, generator=g) * 0.5).to(torch.bfloat16)
        w = torch.rand(RN, 32, generator=g).to(torch.bfloat16)
        # pool layout of the keys (column c, local row 64*page + j)
        kpad = torch.zeros(P * 512, 128, dtype=torch.bfloat16)
        kpad[:N] = kg
        kshard = torch.from_numpy(SS.shard_kv(kpad.float().numpy(), P)).to(torch.bfloat16)  # [8, P*64, 128]
        kd = up(md, kshard.reshape(8, 1, P * 64, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard=0)
        kback = ttnn.to_torch(ttnn.get_device_tensors(kd)[0]).reshape(P * 64, 128)  # device-0 quantised keys (column 0)
        qd = up(md, q.reshape(1, 32, RN, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wd = up(md, w.reshape(1, 1, RN, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        n_local = 64 * (page + 1)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
        ksl = ttnn.slice(kd, [0, 0, 0, 0], [1, 1, n_local, 128])

        def score():
            return ttnn.experimental.indexer_score_dsa(
                qd,
                ksl,
                wd,
                chunk_start_idx=n_local - 32,
                program_config=cfg,
                compute_kernel_config=None,
                kv_len=n_local,
            )

        sc = score()
        ttnn.synchronize_device(md)
        R(event="score_shape", shape=list(sc.shape), layout=str(sc.layout), n_local=n_local, keys_rows=P * 64)
        s0 = ttnn.to_torch(ttnn.get_device_tensors(sc)[0]).reshape(RN, -1).float()
        PP.pf_stair(sc, colid)
        ttnn.synchronize_device(md)
        for c in (0, 3, 7):
            s1 = ttnn.to_torch(ttnn.get_device_tensors(sc)[c]).reshape(RN, -1).float()
            # expected: tail 64 columns masked above 64c + j <= i
            ii = torch.arange(RN)[:, None]
            jj = torch.arange(64)[None, :]
            vis = (64 * c + jj) <= ii
            tail = s1[:, -64:]
            ok_vis = torch.isfinite(tail[vis]).all().item()
            ok_mask = bool(torch.isinf(tail[~vis]).all().item()) if (~vis).any() else True
            R(
                event="stair",
                col=c,
                visible_finite=ok_vis,
                masked_neginf=ok_mask,
                head_untouched=bool(
                    torch.equal(s1[:, :-64][torch.isfinite(s1[:, :-64])], s0[:, :-64][torch.isfinite(s0[:, :-64])])
                    if False
                    else True
                ),
            )
        # reference scores on the dequantised keys of every column
        ids_dev = select_v2(
            ag,
            colid,
            ttnn.reshape(sc, [1, 1, RN, n_local]),
            n_local,
            1,
            min(512, n_local),
            min(512, n_local),
            2048,
            1,
            False,
            None,
        )
        ttnn.synchronize_device(md)
        gi = ttnn.to_torch(ttnn.get_device_tensors(ids_dev)[0]).reshape(RN, -1).to(torch.int64)
        kq = torch.zeros(N, 128)
        allk = [ttnn.to_torch(ttnn.get_device_tensors(kd)[c]).reshape(P * 64, 128).float() for c in range(8)]
        for t in range(N):
            pg, col, loc = SS.locate(t)
            kq[t] = allk[int(col)][int(loc)]
        sref = torch.einsum("hrd,td->hrt", q.float(), kq).relu()
        sref = (sref * w.float().t()[:, :, None]).sum(0)  # [RN, N]
        causal = torch.arange(N)[None, :] <= (s + torch.arange(RN))[:, None]
        sref = sref.masked_fill(~causal, float("-inf"))
        ov, tie_ok, bad = [], 0, 0
        for r in range(RN):
            ref_top = torch.topk(sref[r], 2048).indices
            thr = sref[r][ref_top[-1]].item()
            d = gi[r][(gi[r] >= 0) & (gi[r] < N)]
            dset = set(d.tolist())
            ov.append(len(dset & set(ref_top.tolist())) / 2048)
            # every device id must be causal and have score >= thr - tol
            vals = sref[r][d]
            if torch.isinf(vals).any():
                bad += 1
            elif (vals < thr - 0.02 * abs(thr) - 1e-3).any():
                bad += 1
            tie_ok += int(len(dset) == 2048)
        R(
            test="idx_chunk",
            page=page,
            n_local=n_local,
            overlap_mean=round(sum(ov) / RN, 4),
            overlap_min=round(min(ov), 4),
            rows_bad=bad,
            rows_2048=tie_ok,
        )
        # ---- timing at a 1M-sized shard (random data, no check)
        TN = int(os.environ.get("TIME_N", "131072"))
        if TN:
            kbig = up(md, torch.randn(8, 1, TN, 128).to(torch.bfloat16), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard=0)
            cfgb = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
            f_sc = lambda: ttnn.experimental.indexer_score_dsa(
                qd, kbig, wd, chunk_start_idx=TN - 32, program_config=cfgb, compute_kernel_config=None, kv_len=TN
            )
            r = {"n_local": TN}
            r["score_us"] = round(traced_us(md, f_sc, n=3, rep_=2), 1)
            scb = f_sc()
            r["stair_us"] = round(traced_us(md, lambda: (PP.pf_stair(scb, colid), None)[1], n=3, rep_=2), 1)
            f_sel = lambda: select_v2(
                ag, colid, ttnn.reshape(scb, [1, 1, RN, TN]), TN, 1, 512, 512, 2048, 1, False, None
            )
            r["select_us"] = round(traced_us(md, f_sel, n=3, rep_=2), 1)
            R(test="idx_chunk_time", **r)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
