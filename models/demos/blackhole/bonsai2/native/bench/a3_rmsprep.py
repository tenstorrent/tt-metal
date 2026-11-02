# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + prep (a3.rmsprep) vs float64 rmsnorm -> rotate -> mask, and vs stock rms_norm -> prep
in device time. Usage: python -m models.demos.blackhole.bonsai2.native.bench.a3_rmsprep"""
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us

GGUF = os.path.expanduser("~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")


def main():
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)
    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import op as a3
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_gguf
    from models.demos.blackhole.bonsai2.native.hadamard import BlockHadamard

    had = BlockHadamard.from_metadata(read_gguf(GGUF).kv)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        k, eps = 5120, 1e-6
        workers = a3.worker_cores(device, 4)
        m = a3.hadamard_tensor(device, had, k)
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((1, 1, 32, k)).astype(np.float32) * 3).bfloat16().float()
        w = torch.from_numpy(rng.standard_normal(k).astype(np.float32) * 0.1).bfloat16().float()
        x0 = x[0, 0, 0].double().numpy()
        y = x0 / np.sqrt((x0 ** 2).mean() + eps) * (1.0 + w.double().numpy())
        want = a3.mask_x(torch.from_numpy(had.rotate(y[None]).astype(np.float32)).flatten()).numpy()
        g = a3.norm_gamma(device, w)
        for mem in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
            x8 = a3.masked_input_buffer(device, workers, k)
            a3.rmsprep(xt, g, m, x8, workers, eps)
            got = ttnn.to_torch(x8).float().numpy().reshape(len(workers), 8, k)
            np.savez(os.path.expanduser("~/avan/native2bit/agentA/tmp/rmsprep.npz"), got=got[0], want=want,
                     x=x0, w=w.double().numpy(), signs=had.signs[k])
            err = float(np.abs(got - want[None]).max())
            ulp = 2.0 ** (np.floor(np.log2(np.abs(want).max())) - 7)
            pcc = float(np.corrcoef(got[0].sum(0), want.sum(0))[0, 1])
            device_kernel_us(ttnn, device)
            for _ in range(9):
                a3.rmsprep(xt, g, m, x8, workers, eps)
            ttnn.synchronize_device(device)
            dev = float(np.median(all_kernel_us(ttnn, device)))
            ttnn.ReadDeviceProfiler(device)
            a3.rmsprep(xt, g, m, x8, workers, eps)
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for progs in ttnn.get_latest_programs_perf_data().values():
                for pr in progs:
                    res = pr.program_analyses_results
                    print("   per-RISC us:", {k.replace("DEVICE ", "").replace(" KERNEL DURATION [ns]", ""):
                                             round(int(v.duration) / 1e3, 1) for k, v in res.items()
                                             if "KERNEL DURATION [ns]" in k and "PER CORE" not in k}, flush=True)
            device_kernel_us(ttnn, device)
            for _ in range(9):
                a3.rmsprep(xt, g, m, x8, workers, eps, debug=1)
            ttnn.synchronize_device(device)
            dev_nosync = float(np.median(all_kernel_us(ttnn, device)))
            device_kernel_us(ttnn, device)
            for _ in range(9):
                a3.rmsprep(xt, g, m, x8, workers, eps, debug=2)
            ttnn.synchronize_device(device)
            print(f"   no re-layout (wrong values): {np.median(all_kernel_us(ttnn, device)):.1f} us", flush=True)
            device_kernel_us(ttnn, device)
            for _ in range(9):
                a3.rmsprep(xt, g, m, x8, workers, eps, debug=4)
            ttnn.synchronize_device(device)
            print(f"   no scatter (wrong values): {np.median(all_kernel_us(ttnn, device)):.1f} us", flush=True)
            for td in (2, 4, 6):
                device_kernel_us(ttnn, device)
                for _ in range(9):
                    a3.rmsprep(xt, g, m, x8, workers, eps, debug=1, tdebug=td)
                ttnn.synchronize_device(device)
                print(f"   no exchange, tdebug={td}: {np.median(all_kernel_us(ttnn, device)):.1f} us", flush=True)
            print(f"rmsprep x in {'DRAM' if mem == ttnn.DRAM_MEMORY_CONFIG else 'L1':4s}: {dev:6.1f} us (no rstd exchange "
                  f"{dev_nosync:5.1f})  pcc {pcc:.7f}  "
                  f"max|err| {err:.3e} ({err / ulp:.1f} bf16 ulp at max)  all shards equal "
                  f"{bool((got == got[0:1]).all())}", flush=True)
            if mem == ttnn.L1_MEMORY_CONFIG:  # emit mode: also the normalized rows for a dense consumer
                yo = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                    device, ttnn.L1_MEMORY_CONFIG)
                a3.rmsprep(xt, g, m, x8, workers, eps, y_out=yo)
                got8 = ttnn.to_torch(x8).float().numpy().reshape(len(workers), 8, k)
                yt = ttnn.to_torch(yo).float().numpy()[0, 0]
                yerr = float(np.abs(yt[0] - y).max() / np.abs(y).max())
                device_kernel_us(ttnn, device)
                for _ in range(9):
                    a3.rmsprep(xt, g, m, x8, workers, eps, y_out=yo)
                ttnn.synchronize_device(device)
                print(f"emit: {np.median(all_kernel_us(ttnn, device)):6.1f} us  x8 unchanged {bool((got8 == got).all())}  "
                      f"y rel max err {yerr:.2e}  rows 1-31 zero {bool((yt[1:] == 0).all())}", flush=True)
                ttnn.deallocate(yo)
                # add mode: h = bf16(x + d) normed instead of x, and written out
                d = torch.from_numpy(rng.standard_normal((1, 1, 32, k)).astype(np.float32)).bfloat16().float()
                hw = (x + d).bfloat16().float()
                h0 = hw[0, 0, 0].double().numpy()
                yh = h0 / np.sqrt((h0 ** 2).mean() + eps) * (1.0 + w.double().numpy())
                want_h = a3.mask_x(torch.from_numpy(had.rotate(yh[None]).astype(np.float32)).flatten()).numpy()
                for dmem in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
                    dt = ttnn.from_torch(d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dmem)
                    ho = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                        device, ttnn.L1_MEMORY_CONFIG)
                    a3.rmsprep(xt, g, m, x8, workers, eps, add=dt, h_out=ho)
                    g8 = ttnn.to_torch(x8).float().numpy().reshape(len(workers), 8, k)
                    ht = ttnn.to_torch(ho).float().numpy()[0, 0]
                    herr = float(np.abs(g8 - want_h[None]).max()) / 2.0 ** (np.floor(np.log2(np.abs(want_h).max())) - 7)
                    device_kernel_us(ttnn, device)
                    for _ in range(9):
                        a3.rmsprep(xt, g, m, x8, workers, eps, add=dt, h_out=ho)
                    ttnn.synchronize_device(device)
                    print(f"add (d in {'DRAM' if dmem == ttnn.DRAM_MEMORY_CONFIG else 'L1'}): "
                          f"{np.median(all_kernel_us(ttnn, device)):6.1f} us  h exact {bool((ht[0] == hw[0, 0, 0].numpy()).all())}  "
                          f"rows 1-31 zero {bool((ht[1:] == 0).all())}  x8 max err {herr:.1f} bf16 ulp  "
                          f"pcc {float(np.corrcoef(g8[0].sum(0), want_h.sum(0))[0, 1]):.7f}", flush=True)
                    ttnn.deallocate(ho)
                    ttnn.deallocate(dt)
            ttnn.deallocate(x8)
            ttnn.deallocate(xt)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
