"""Dequant fidelity (exactness + time) and rotation stage split, layer 0, chip from env. Wall ms, median of 5."""
import os, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

dev = ttnn.open_device(device_id=0, l1_small_size=24576)
def timed(f, n=5):
    r = f(); [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]; ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
        [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
    return float(np.median(ts))
try:
    sh = Shared(dev); mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
    ref = torch.from_numpy(read_ternary(sh.gguf, "blk.0.ffn_down.weight").dequant().T.copy()).bfloat16()
    F = ttnn.MathFidelity
    for name, fid in (("LoFi", F.LoFi), ("HiFi2", F.HiFi2), ("HiFi3", F.HiFi3), ("HiFi4", F.HiFi4)):
        wd = pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120, dtype=ttnn.bfloat16, fidelity=fid)
        ex = (ttnn.to_torch(wd).reshape(ref.shape) == ref).float().mean().item(); ttnn.deallocate(wd)
        tg = timed(lambda: pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True, fidelity=fid))
        t4 = timed(lambda: pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True, fidelity=fid, dtype=ttnn.bfloat4_b))
        print(f"dequant {name}: down bf16 exact {ex*100:.3f}%  gu bfp8 {tg:.3f} ms  gu bfp4 {t4:.3f} ms", flush=True)
    rot = pf.rotator(sh)
    for s in (128, 2048):
        for k in (5120, 17408):
            x = ttnn.from_torch(torch.randn(1, 1, s, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            D = ttnn.DRAM_MEMORY_CONFIG
            xt = ttnn.transpose(x, -2, -1, memory_config=D)
            xt4 = ttnn.reshape(xt, [1, k // 1024, 1024, s])
            r = {"total": timed(lambda: rot(x)),
                 "transpose": timed(lambda: ttnn.transpose(x, -2, -1, memory_config=D)),
                 "bmm": timed(lambda: ttnn.matmul(rot.mt[k], xt4, compute_kernel_config=pf._HIFI4, memory_config=D, dtype=ttnn.bfloat16)),
                 "bmm_hifi2": timed(lambda: ttnn.matmul(rot.mt[k], xt4, compute_kernel_config=pf._LOFI, memory_config=D, dtype=ttnn.bfloat16)),
            }
            # slice + per-block matmul + concat, no transposes
            blocks = [ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(pf.a3.hadamard_blocks(sh.had, k).numpy()[b*1024:(b+1)*1024])).reshape(1,1,1024,1024),
                                      dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D) for b in range(k // 1024)]
            def sl():
                parts = [ttnn.matmul(ttnn.slice(x, (0, 0, 0, b * 1024), (1, 1, s, (b + 1) * 1024)), blocks[b], compute_kernel_config=pf._HIFI4,
                                     memory_config=D, dtype=ttnn.bfloat16) for b in range(k // 1024)]
                return ttnn.concat(parts, -1, memory_config=D)
            r["slice_mm_concat"] = timed(sl)
            print(f"rot S={s} K={k}: " + "  ".join(f"{a} {b:.3f}" for a, b in r.items()), flush=True)
            for t in [x, xt] + blocks: ttnn.deallocate(t)
finally:
    ttnn.close_device(dev)
