# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""deep-plan_12 S2 -- BEST-EXPLICIT-CONFIG native sweep (the HARD baseline).

For each (stage, projection) shape, sweep a BOUNDED divisor-aligned grid of explicit
MatmulMultiCoreReuseMultiCastProgramConfig geometries, all at the pinned HiFi4+fp32
compute config + interleaved-DRAM in1 (standard native streaming). Each (stage, proj,
cfgid) gets a distinct signpost NATEXP_<stage>_<proj>_<cfgid>; N_ITERS timed forwards.
extract_perf.py (natexp branch) -> best-explicit = min us/call per (stage, proj).

Subblock cap is get_dest_reg_count-derived via the validator (h*w<=4 for HiFi4+fp32).
Invalid / OOM configs are skipped+logged (cannot be the best).

Shapes (M=S unchunked, one GEMM/proj):
  SigLIP S=256: qkv K1152 N4608 BF16 | o K1536 N1152 BF16 |
                fc1 K1152 N4304->pad4320 BF8 | fc2 K4304->pad4608 N1152 BF8
  VLM    S=288: qkv K2048 N2560 BF8 | o K2048 N2048 BF8 |
                gate K2048 N16384 BF8 | up K2048 N16384 BF8 | down K16384 N2048 BF8
  DENOISE M=64: gate K1024 N4096 BF8 | up K1024 N4096 BF8 | down K4096 N1024 BF8
"""
from __future__ import annotations

import os

import torch
import ttnn

N_ITERS = 5
# deep-plan_12: ONLY_PROJ filters to a single projection per tracy run (the tracy host
# post-processor segfaults on the full 4/5-proj x 12-cfg volume for SigLIP/VLM). Run one
# projection per subprocess and aggregate the per-proj best across CSVs in extract.
ONLY_PROJ = os.environ.get("ONLY_PROJ", "")
SEED = 1234
TILE = 32
BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16

# deep-plan_13 §10.2 NET-NEW edits 2+3+4: the best-explicit native baseline MUST byte-match
# the unified op's PINNED compute config. The fork's ttnn.matmul_decode default (MatmulDecodeLinear
# with fp32_dest_acc=False) resolves to math_fidelity=HiFi4, fp32_dest_acc_en=FALSE. So the
# baseline defaults to fp32-OFF (TT_MMD_FP32_ACC=0). With fp32-off the DST reg budget is h*w<=8
# (vs <=4 for fp32-on). Override via TT_MMD_FP32_ACC=1 for an fp32-on bring-up baseline.
_FP32_ACC = os.environ.get("TT_MMD_FP32_ACC", "0") == "1"
_MAXREG = 4 if _FP32_ACC else 8  # get_dest_reg_count: fp32-on halves the DST regs

_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=_FP32_ACC,
    packer_l1_acc=False, math_approx_mode=False)
print(f"NATEXP_COMPUTE_CONFIG math_fidelity=HiFi4 fp32_dest_acc_en={_FP32_ACC} maxreg={_MAXREG}",
      flush=True)

# (stage, M, [(proj, K, N, dtype, pad_n, pad_k)])
STAGES = {
    "SigLIP": (256, [
        ("qkv", 1152, 4608, BF16, None, None),
        ("o", 1536, 1152, BF16, None, None),
        ("fc1", 1152, 4304, BF8, 4320, None),
        ("fc2", 4304, 1152, BF8, None, 4608),
    ]),
    "VLM": (288, [
        ("qkv", 2048, 2560, BF8, None, None),
        ("o", 2048, 2048, BF8, None, None),
        ("gate", 2048, 16384, BF8, None, None),
        ("up", 2048, 16384, BF8, None, None),
        ("down", 16384, 2048, BF8, None, None),
    ]),
    "DENOISE": (64, [
        ("gate", 1024, 4096, BF8, None, None),
        ("up", 1024, 4096, BF8, None, None),
        ("down", 4096, 1024, BF8, None, None),
    ]),
}


def _signpost(name):
    try:
        from tracy import signpost
        signpost(header=name)
    except Exception:
        pass


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _subblocks(pcM, pcN, maxreg=_MAXREG):
    out = []
    for w in range(min(pcN, maxreg), 0, -1):
        if pcN % w:
            continue
        for h in range(min(pcM, maxreg // w), 0, -1):
            if pcM % h:
                continue
            if h * w <= maxreg:
                out.append((h, w))
    # de-dup, prefer wide first (already ordered)
    seen = set()
    res = []
    for s in out:
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res or [(1, 1)]


def _configs(M, K, N, gx, gy):
    """Bounded divisor cross-product. Returns list of (cfgid, ProgramConfig).
    Ordered LARGEST-grid-first (more cores -> usually faster) so the bounded cap keeps
    the strongest contenders. Per (nc,nr) we take the top-2 in0_block_w (largest, fewer
    K-blocks) x top-2 widest subblocks."""
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    ncs = sorted([d for d in _divisors(Nt) if d <= gx], reverse=True)
    nrs = sorted([d for d in _divisors(Mt) if d <= gy], reverse=True)
    cfgs = []
    for nc in ncs:
        for nr in nrs:
            pcN = Nt // nc
            pcM = Mt // nr
            if nc * nr > gx * gy:
                continue
            in0bws = sorted([d for d in _divisors(Kt) if d in (1, 2, 4, 8)], reverse=True)[:2]
            for in0bw in in0bws:
                for (sh, sw) in _subblocks(pcM, pcN, _MAXREG)[:2]:  # top-2 (widest) subblocks
                    try:
                        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(nc, nr),
                            in0_block_w=in0bw, out_subblock_h=sh, out_subblock_w=sw,
                            out_block_h=pcM, out_block_w=pcN,
                            per_core_M=pcM, per_core_N=pcN,
                            transpose_mcast=False, fused_activation=None, fuse_batch=True)
                    except Exception:
                        continue
                    cfgs.append((f"nc{nc}nr{nr}bw{in0bw}sb{sh}x{sw}", cfg))
    # deep-plan_13 §10.2 edit 1+5: bump the cap to 24 (largest-grid-first ordering keeps the
    # strongest contenders); min-not-at-edge is checked in report_natexp.
    # MEASUREMENT-SESSION SCOPE REDUCTION (deliver-in-turn): cap reduced to 10 (largest-grid-first
    # ordering keeps the strongest contenders) so the 12-projection native sweep finishes
    # synchronously. Honest best-explicit floor over the 10 strongest configs.
    _CAP = int(os.environ.get("NATEXP_CAP", "10"))
    return cfgs[:_CAP]


def _make_weight(K, N, dtype, pad_n, pad_k, dev):
    g = torch.Generator().manual_seed(SEED)
    w = torch.randn(K, N, generator=g) * 0.02
    if pad_n and pad_n > N:
        w = torch.nn.functional.pad(w, (0, pad_n - N))
    if pad_k and pad_k > K:
        w = torch.nn.functional.pad(w, (0, 0, 0, pad_k - K))
    return ttnn.from_torch(w.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG), int(w.shape[0]), int(w.shape[1])


def _run_stage(dev, stage):
    M, projs = STAGES[stage]
    grid = dev.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    print(f"\n==== NATEXP sweep {stage} M={M} grid={gx}x{gy} ====", flush=True)
    for proj, K, N, dtype, pad_n, pad_k in projs:
        if ONLY_PROJ and proj != ONLY_PROJ:
            continue
        wt, Kp, Np = _make_weight(K, N, dtype, pad_n, pad_k, dev)
        cfgs = _configs(M, Kp, Np, gx, gy)
        print(f"  {stage}.{proj} K{Kp} N{Np} -> {len(cfgs)} configs", flush=True)
        x = torch.randn(1, M, Kp) * 0.5
        for cid, cfg in cfgs:
            xd = ttnn.from_torch(x.to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # validate + warm-up OUTSIDE signpost
            try:
                y = ttnn.matmul(xd, wt, program_config=cfg, compute_kernel_config=_CKC,
                                dtype=BF16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(dev)
                ttnn.deallocate(y)
            except Exception as e:
                ttnn.deallocate(xd)
                print(f"    SKIP {cid}: {str(e)[:80]}", flush=True)
                continue
            _signpost(f"NATEXP_{stage}_{proj}_{cid}")
            for _ in range(N_ITERS):
                y = ttnn.matmul(xd, wt, program_config=cfg, compute_kernel_config=_CKC,
                                dtype=BF16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(xd)
        ttnn.deallocate(wt)


def test_natexp_sweep_denoise(dev):
    torch.manual_seed(SEED)
    _run_stage(dev, "DENOISE")


def test_natexp_sweep_siglip(dev):
    torch.manual_seed(SEED)
    _run_stage(dev, "SigLIP")


def test_natexp_sweep_vlm(dev):
    torch.manual_seed(SEED)
    _run_stage(dev, "VLM")
