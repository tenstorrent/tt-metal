# prefill-roof phase 3 (1): per-chip matmul peak (dtype x fidelity x shape, explicit 2D-mcast configs, best of in0_block_w) + streaming DRAM read. Prints P json lines.
import json
import torch
from idx_common import *
import ttnn

def pcfg(M, K, N, bw, h, sw, gx, gy):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=h, out_subblock_w=sw,
        per_core_M=(M // 32) // gy, per_core_N=(N // 32) // gx, transpose_mcast=False, fused_activation=None, fuse_batch=True)

def main():
    md = open_md()
    shapes = [(512, 6144, 18432), (512, 2048, 6144), (512, 6144, 4608), (512, 6144, 2048), (512, 2048, 2048), (2048, 6144, 4608), (4096, 4096, 4096)]
    dts = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
    fids = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
    for (M, K, N) in shapes:
        x = up(md, torch.randn(1, 1, M, K).to(torch.bfloat16))
        for dn, dt in dts.items():
            w = up(md, torch.randn(1, 1, K, N).to(torch.bfloat16), dt)
            for fn, fid in fids.items():
                ckc = ttnn.init_device_compute_kernel_config(md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=(fn == "HiFi4"), packer_l1_acc=False)
                best = None
                mt, kt, nt = M // 32, K // 32, N // 32
                cands = [None]
                gy = 8 if mt % 8 == 0 else 4
                for gx in sorted({d for d in range(1, 13) if nt % d == 0}, reverse=True)[:3]:
                    pn = nt // gx
                    for bw in (8, 16, 24, 32):
                        if kt % bw: continue
                        for h, sw in ((1, 4), (2, 2), (2, 4)):
                            if pn % sw or ((mt // gy) % h) or (fn == "HiFi4" and h * sw > 4): continue
                            cands.append((bw, h, sw, gx, gy))
                for c in cands[:40]:
                    try:
                        kw = {} if c is None else {"program_config": pcfg(M, K, N, *c)}
                        us = traced_us(md, lambda: ttnn.matmul(x, w, compute_kernel_config=ckc, **kw), n=3, reps=2)
                    except Exception as e:
                        continue
                    if best is None or us < best[0]: best = (us, c)
                if best:
                    print("P", json.dumps({"M": M, "K": K, "N": N, "w": dn, "fid": fn, "us": round(best[0], 1), "tflops": round(2 * M * K * N / best[0] / 1e6, 1), "cfg": best[1]}), flush=True)
            ttnn.deallocate(w)
        ttnn.deallocate(x)
    # streaming read: reduction over a big tensor (read-dominated) and unary (read+write)
    for mb in (256, 1024):
        rows = mb * 1024 * 1024 // 2 // 4096
        t = up(md, torch.randn(1, 1, rows, 4096).to(torch.bfloat16))
        by = rows * 4096 * 2
        for nm, f, mult in (("sum_last", lambda: ttnn.sum(t, dim=3), 1), ("relu", lambda: ttnn.relu(t), 2), ("typecast_bf8", lambda: ttnn.typecast(t, ttnn.bfloat8_b), 1.53)):
            try:
                us = traced_us(md, f, n=3, reps=2)
                print("P", json.dumps({"stream": nm, "MB": mb, "us": round(us, 1), "GBps": round(by * mult / us / 1e3, 1)}), flush=True)
            except Exception as e:
                print("P", json.dumps({"stream": nm, "err": str(e)[:100]}), flush=True)
        ttnn.deallocate(t)
    close_md(md)
main()
