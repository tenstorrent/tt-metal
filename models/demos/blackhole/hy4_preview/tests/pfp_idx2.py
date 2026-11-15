"""pool-prefill probe: indexer_score_dsa with kv_len < allocated key rows (shape/time), key slice cost, select_v2 time at 512 rows."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.blackhole.hy4_preview.tests.pfp_idx import RN, R, up
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP
from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2
from models.demos.gpt_oss.tt.ccl import CCLManager

TT = int(os.environ.get("TT", "131072"))


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
        qd = up(md, torch.randn(1, 32, RN, 128).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wd = up(md, torch.rand(1, 1, RN, 32).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kbig = up(md, torch.randn(8, 1, TT, 128).to(torch.bfloat16), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard=0)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=256, head_group_size=1)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
        fids = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}
        kv = TT
        ks = kbig
        for fn in ("hifi4", "hifi2", "lofi", "none"):
            ck = (
                None
                if fn == "none"
                else ttnn.init_device_compute_kernel_config(
                    md.arch(),
                    math_fidelity=fids[fn],
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
            )
            f3 = lambda: ttnn.experimental.indexer_score_dsa(
                qd, ks, wd, chunk_start_idx=kv - 32, program_config=cfg, compute_kernel_config=ck, kv_len=kv
            )
            R(op="score_fid", fid=fn, kv_len=kv, us=round(traced_us(md, f3, n=3, rep_=2), 1))
        ck = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        f3 = lambda: ttnn.experimental.indexer_score_dsa(
            qd, ks, wd, chunk_start_idx=kv - 32, program_config=cfg, compute_kernel_config=ck, kv_len=kv
        )
        o = f3()
        R(op="stair", kv=kv, us=round(traced_us(md, lambda: (PP.pf_stair(o, colid), None)[1], n=3, rep_=2), 1))
        for B in (1,):
            kl = 512
            f_sel = lambda: select_v2(
                ag, colid, ttnn.reshape(f3(), [1, 1, RN, kv]), kv, B, kl, kl, 2048, 1, False, None
            )
            R(op=f"score+select_v2_B{B}", kv=kv, kloc=kl, us=round(traced_us(md, f_sel, n=3, rep_=2), 1))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
