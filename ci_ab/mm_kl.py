"""Scratch, not for merge. minimal_matmul (compute_metal2) and conv3d op-level cases, device kernel durations per case from tracy, output hash per case."""
# 2048x4096x4096 bf16 matmul neighbours: LoFi, HiFi2, HiFi4 with bfp8_b and bf16 outputs (device time A/B).
import hashlib, os, sys
import numpy as np
import torch
import ttnn
from tracy import signpost

N = int(os.environ.get("MM_N", "8"))
ONLY = os.environ.get("MM_ONLY", "")
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
D = ttnn.DRAM_MEMORY_CONFIG
F = ttnn.MathFidelity


def ckc(fid, fp32=False, l1acc=True, full=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc, dst_full_sync_en=full)


def tt(shape, dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=layout, device=dev, memory_config=D)


def case(name, f):
    if ONLY and ONLY not in name:
        return
    try:
        for _ in range(2):
            o = f()
        ttnn.synchronize_device(dev)
        h = hashlib.sha1(ttnn.to_torch(o).float().numpy().tobytes()).hexdigest()[:12]
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, h, flush=True)
    except Exception as e:  # keep going so one unsupported case does not lose the others
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)



def mmm(name, m, k, n, wdt, fid, kb, sh, sw, fp32=False, mb=8, nb=8):
    a, w = tt((m, k), ttnn.bfloat16), tt((k, n), wdt)
    cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw, compute_with_storage_grid_size=grid)
    case(name, lambda: ttnn.experimental.minimal_matmul(a, w, compute_kernel_config=ckc(fid, fp32), config=cfg))


FN = (("lofi", F.LoFi), ("hifi2", F.HiFi2), ("hifi4", F.HiFi4))
for fn, fid in FN:
    for kb in (1, 2, 4, 8, 16):
        mmm(f"mm_bf16_{fn}_kb{kb}_s2x4", 2048, 4096, 4096, ttnn.bfloat16, fid, kb, 2, 4)
    for sh, sw in ((4, 2), (1, 8), (8, 1), (1, 1), (2, 2)):
        mmm(f"mm_bf16_{fn}_kb8_s{sh}x{sw}", 2048, 4096, 4096, ttnn.bfloat16, fid, 8, sh, sw)
    mmm(f"mm_bfp8_{fn}_kb8_s2x4", 2048, 4096, 4096, ttnn.bfloat8_b, fid, 8, 2, 4)
mmm("mm_bf16_hifi2_kb8_s2x2_fp32", 2048, 4096, 4096, ttnn.bfloat16, F.HiFi2, 8, 2, 2, fp32=True)
mmm("mm_bf16_hifi4_kb4_s1x4_fp32", 2048, 4096, 4096, ttnn.bfloat16, F.HiFi4, 4, 1, 4, fp32=True)


def conv3d(name, cin, cout, ks, shape, cinb, coutb, t, h, w, fid=F.HiFi2, fp32=True):
    n, d, hh, ww = shape
    x = ttnn.from_torch(torch.randn(n, d, hh, ww, cin), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    cfg = ttnn.Conv3dConfig(weights_dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT, T_out_block=t, W_out_block=w, H_out_block=h,
                            C_out_block=coutb, C_in_block=cinb, compute_with_storage_grid_size=(grid.x, grid.y))
    wt = ttnn.from_torch(torch.randn(cout, cin, *ks), dtype=ttnn.bfloat16, pad_value=0)
    wt = ttnn.experimental.prepare_conv3d_weights(weight_tensor=wt, groups=1, C_in_block=cinb, alignment=32, device=dev)
    b = ttnn.from_torch(torch.randn(1, cout), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
    kc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
    pad = tuple(k // 2 for k in ks)
    case(name, lambda: ttnn.experimental.conv3d(input_tensor=x, weight_tensor=wt, device=dev, bias_tensor=b, dtype=ttnn.bfloat16, output_channels=cout,
                                                 kernel_size=ks, stride=(1, 1, 1), groups=1, padding=pad, dilation=(1, 1, 1), padding_mode="zeros",
                                                 config=cfg, compute_kernel_config=kc))


conv3d("c3d_hifi2_fp32_128_128", 128, 128, (3, 3, 3), (1, 8, 32, 32), 128, 128, 1, 8, 4)
conv3d("c3d_hifi2_fp32_96_96", 96, 96, (3, 3, 3), (1, 8, 32, 32), 96, 96, 1, 4, 8)
conv3d("c3d_hifi4_nofp32_128_128", 128, 128, (3, 3, 3), (1, 8, 32, 32), 128, 128, 1, 8, 4, fid=F.HiFi4, fp32=False)
conv3d("c3d_lofi_nofp32_64_128", 64, 128, (1, 3, 3), (1, 8, 32, 32), 64, 128, 1, 4, 8, fid=F.LoFi, fp32=False)
ttnn.close_device(dev)
print("ALL DONE", flush=True)
