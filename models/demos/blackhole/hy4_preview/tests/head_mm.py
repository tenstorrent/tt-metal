"""lm_head matmul program configs: [1,1,T,6144] x [1,1,6144,15104] per chip (bf16 / bfp8 weights), traced us and max abs diff vs the default config."""

import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])

for hd in os.environ.get("HDS", "bf16,bfp8").split(","):
    head = Hy4DeviceHead(md, mc, ccl, W, head_dtype=hd, out_dtype="fp32", fidelity="HiFi4")
    for T in TS:
        h = head.hid(streams_tt(X[:T]))
        base = head.matmul_only(h)
        base_t = head.gather_logits(base)
        us0 = timed_trace(lambda: head.matmul_only(h))
        res(event="mm", hd=hd, T=T, cfg="default", us=round(us0, 1))
        K, N = 6144 // 32, 15104 // 32
        gx, gy = 8, 8
        for pcn, ibw in itertools.product([4, 6, 8, 12, 16], [4, 8, 16, 32]):
            # N split over ncores (per_core_N tiles); the grid is the smallest rectangle in 10 rows x 13 cols (BH worker grid)
            ncores = -(-N // pcn)
            gxx = min(12, ncores)
            gyy = -(-ncores // gxx)
            name = f"1d_nc{ncores}_pcn{pcn}_ibw{ibw}"
            try:
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gxx, gyy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=min(pcn, 4) if pcn % min(pcn, 4) == 0 else 1,
                    per_core_M=1,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                f = lambda: ttnn.matmul(
                    h, head.head_w, program_config=pc, compute_kernel_config=head.ckc, dtype=head.out_tt
                )
                out = head.gather_logits(f())
                d = float((out - base_t).abs().max())
                res(event="mm", hd=hd, T=T, cfg=name, us=round(timed_trace(f), 1), maxdiff=d)
            except Exception as e:
                res(event="mm", hd=hd, T=T, cfg=name, err=str(e)[:140].replace("\n", " "))
    del head
ttnn.close_mesh_device(md)
