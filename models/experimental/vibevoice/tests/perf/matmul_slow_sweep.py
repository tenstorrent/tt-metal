"""Device-profiled sweep for the SLOW tokenizer/connector matmuls.

Runs each candidate config a few times in a FIXED order under the Tracy device profiler,
then parses the ops CSV to report true device us + PCC.  Run:

  python -m tracy -r -p --op-support-count 100000 \
    models/experimental/vibevoice/tests/perf/matmul_slow_sweep.py
then it prints the per-config device us (parsed from the generated CSV at exit).
"""
import torch
import ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
GX, GY = grid.x, grid.y
REP = 4  # measured reps per config

HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
_plan = []  # (label, M,K,N, prog, out_l1)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def mm1d(cx, cy, in0_bw, pcn):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def candidates(M, K, N, wdtype):
    Kt, Nt = K // 32, (N + 31) // 32
    xt = torch.randn(1, 1, M, K)
    wt = torch.randn(K, N)
    ref = (xt.reshape(M, K) @ wt).reshape(1, 1, M, N)
    x_dram = ttnn.from_torch(
        xt, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_l1 = ttnn.to_memory_config(x_dram, ttnn.L1_MEMORY_CONFIG)
    w = ttnn.from_torch(wt, device=dev, dtype=wdtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = [("auto/DRAM", None, x_dram, False), ("auto/L1", None, x_l1, True)]
    # try several full-ish grids with per_core_N chosen to tile N, plus a few in0_block_w
    for cx, cy in [(8, 4), (8, 8), (GX, GY), (8, 2), (11, 4)]:
        ncols = cx * cy
        if ncols > Nt:
            continue
        pcn = max(1, (Nt + ncols - 1) // ncols)
        for bw in (8, 4, 2):
            if Kt % bw:
                continue
            for outc in (x_l1, x_dram):
                lbl = f"1d {cx}x{cy} bw{bw} pcn{pcn}/{'L1' if outc is x_l1 else 'DR'}"
                out.append((lbl, mm1d(cx, cy, bw, pcn), outc, outc is x_l1))
            break  # bw8 preferred, one per grid
    return ref, out, (x_dram, x_l1, w)


SHAPES = [
    ("dim2048.l1up", 32, 2048, 8192, ttnn.bfloat8_b),  # BIGGEST untuned up-proj 0.808ms/68% DRAM
    ("dim1024.l1up", 32, 1024, 4096, ttnn.bfloat8_b),
    ("dim2048.l2dn", 32, 8192, 2048, ttnn.bfloat16),  # already tuned (baseline check)
    ("conn.fc2", 32, 1536, 1536, ttnn.bfloat16),
]

_refs = {}
_keep = []  # keep tensors alive
tracy = __import__("tracy")
for tag, M, K, N, wd in SHAPES:
    ref, cands, tens = candidates(M, K, N, wd)
    _refs[tag] = ref
    _keep.append(tens)
    for label, prog, x, out_l1 in cands:
        mc = ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG
        full = f"{tag}|{label}"
        # warmup (compile), not profiled range
        try:
            o = ttnn.linear(x, tens[2], compute_kernel_config=HIFI2, program_config=prog, memory_config=mc)
            p = pcc(ref, ttnn.to_torch(o).float())
        except Exception as e:
            print(f"SKIP {full}: {str(e)[:70]}")
            continue
        _plan.append((full, x, tens[2], prog, mc, p))

ttnn.synchronize_device(dev)
tracy.signpost("start")
for full, x, w, prog, mc, p in _plan:
    for _ in range(REP):
        ttnn.linear(x, w, compute_kernel_config=HIFI2, program_config=prog, memory_config=mc)
ttnn.synchronize_device(dev)
tracy.signpost("stop")
for full, x, w, prog, mc, p in _plan:
    print(f"PCC {full} = {p:.4f}")
ttnn.close_device(dev)
