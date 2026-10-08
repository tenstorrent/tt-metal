# Round 3 matmul: compile, on the mock cluster, the ttnn matmul kernels the branch can reach: the metal2 bmm kernel through the
# 2D and 1D mcast and reuse program configs with every sub block shape (rows of one tile up to 8 and 16 tiles), bf16, bfp8 and
# fp32 inputs, LoFi to HiFi4, fp32 DEST on and off, full sync, bias and activation; plus auto-config and batched matmuls.
# Outputs are not read; only the JIT builds matter.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import torch
import ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
print("arch", dev.arch(), flush=True)
T = ttnn.TILE_LAYOUT
ok = fail = 0


def run(name, fn):
    global ok, fail
    try:
        fn()
        ttnn.synchronize_device(dev)
        ok += 1
    except Exception as e:  # keep going: the point is the set of kernels compiled
        fail += 1
        print("FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


def t(shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.rand(shape) - 0.5, dtype=dtype, layout=T, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def ckc(fid, fp32, full=False, l1acc=False):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc, dst_full_sync_en=full
    )


SUB = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (2, 2), (1, 8), (8, 1), (2, 4), (4, 2), (1, 6), (6, 1)]
FIDS = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
grid = (2, 2)
for (sh, sw) in SUB:
    per_m, per_n = 2 * sh, 2 * sw
    M, N, K = per_m * 32 * grid[1], per_n * 32 * grid[0], 128
    for dtn, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        a, b = t((1, 1, M, K), dt), t((1, 1, K, N), dt)
        for fid in FIDS:
            for fp32 in (False, True):
                if fp32 and sh * sw > 4:
                    continue
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=sh, out_subblock_w=sw,
                    per_core_M=per_m, per_core_N=per_n, transpose_mcast=False, fused_activation=None,
                )
                run(f"2d {sh}x{sw} {dtn} {fid} fp32={fp32}", lambda: ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc(fid, fp32)))
        pc1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=per_m * grid[1], per_core_N=per_n // 2 if per_n >= 2 else 1, fuse_batch=True, fused_activation=None, mcast_in0=True,
        )
        if (per_n // 2 if per_n >= 2 else 1) % sw == 0:
            run(f"1d {sh}x{sw} {dtn}", lambda: ttnn.matmul(a, b, program_config=pc1, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2, False)))
        if sh * sw <= 8:
            pcf = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=per_m, per_core_N=per_n, transpose_mcast=False, fused_activation=None,
            )
            run(f"2d full sync {sh}x{sw} {dtn}", lambda: ttnn.matmul(a, b, program_config=pcf, compute_kernel_config=ckc(ttnn.MathFidelity.LoFi, sh * sw <= 4, True)))
        bias = t((1, 1, 1, N), dt)
        pcb = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=per_m, per_core_N=per_n, transpose_mcast=False, fused_activation=ttnn.UnaryOpType.GELU,
        )
        run(f"2d bias gelu {sh}x{sw} {dtn}", lambda: ttnn.linear(a, b, bias=bias, program_config=pcb, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2, False)))
        # spill over two K blocks with packer L1 accumulation and the partials reload
        a2, b2 = t((1, 1, M, 256), dt), t((1, 1, 256, N), dt)
        for l1acc in (False, True):
            pcs = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=per_m, per_core_N=per_n, transpose_mcast=False, fused_activation=None,
            )
            run(f"2d spill {sh}x{sw} {dtn} l1acc={l1acc}", lambda: ttnn.matmul(a2, b2, program_config=pcs, compute_kernel_config=ckc(ttnn.MathFidelity.LoFi, False, False, l1acc)))
# fp32 inputs
a, b = t((1, 1, 256, 256), ttnn.float32), t((1, 1, 256, 256), ttnn.float32)
for fid in FIDS:
    run(f"auto fp32 {fid}", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc(fid, True)))
# auto configs, batched and small matmuls
for shp in ((1, 1, 32, 32, 32), (1, 1, 64, 1024, 64), (1, 1, 1024, 1024, 1024), (2, 4, 128, 256, 512), (1, 1, 32, 4096, 128), (1, 1, 2048, 512, 2048)):
    bb, hh, m, k, n = shp
    for dtn, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        a, b = t((bb, hh, m, k), dt), t((1, 1, k, n), dt)
        for fid in FIDS:
            run(f"auto {shp} {dtn} {fid}", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc(fid, False)))
print(f"done ok={ok} fail={fail}", flush=True)
ttnn.close_device(dev)
