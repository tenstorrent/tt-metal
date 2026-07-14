# SPDX-License-Identifier: Apache-2.0
"""Deep minimal_matmul block/subblock sweep for seq-parallel TP2 (M=49152).

Matmuls run at only 43-59% of the GEMM_FLOPS utilization formula ceiling
(ideal_cycles = M*K*N/32^3 * 16/64cores). The greatest lever is the
(M_block, K_block, N_block, subblock_h, subblock_w) tuple: subblock_h*subblock_w
tiles are computed per DEST acquire (max 8 for bf16 DEST / fp32_dest_acc_en=False),
and (M_block/subblock_h)*(N_block/subblock_w) acquire/release cycles per block —
larger subblocks amortize that overhead. K_block sets how many K-tiles accumulate
before a DEST flush (more = fewer flushes but bigger in0/in1 CBs).

This sweep uses DEVICE KERNEL DURATION (profiler + start/stop signposts), mirrors
sweep_ring_sdpa_tp2.py, and wraps every config in try/except.

Run:
  TT_VISIBLE_DEVICES=0 pytest \\
    models/demos/wormhole/bge_m3/tests/sweeps/sweep_matmul_subblock_tp2.py::test_matmul_subblock_sweep -s -q
"""
import os

import pytest
import torch
from tracy import signpost

import ttnn

M = 49152  # 12 * 4096

# label -> (K, N, has_gelu)
SHAPES = {
    "qkv": (1024, 3072, True),
    "mlpwi": (1024, 4096, True),
    "mlpwo": (4096, 1024, False),
    "attnout": (1024, 1024, False),
}

# Valid subblocks for bf16 DEST (h*w <= 8).
SUBBLOCKS = [(4, 2), (2, 4), (8, 1), (1, 8), (2, 2), (4, 1), (1, 4)]


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_matmul_subblock_worker(mesh_device):
    label = os.environ["BGE_MM_SHAPE"]
    mb = int(os.environ["BGE_MM_MB"])
    kb = int(os.environ["BGE_MM_KB"])
    nb = int(os.environ["BGE_MM_NB"])
    sbh = int(os.environ["BGE_MM_SBH"])
    sbw = int(os.environ["BGE_MM_SBW"])
    gx = int(os.environ["BGE_MM_GX"])
    gy = int(os.environ["BGE_MM_GY"])
    K, N, gelu = SHAPES[label]

    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    a = ttnn.from_torch(
        torch.randn(1, 1, M, K, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fa = (ttnn.UnaryOpType.GELU, True) if gelu else None
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb, K_block_size=kb, N_block_size=nb,
        subblock_h=sbh, subblock_w=sbw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )

    def run():
        o = ttnn.experimental.minimal_matmul(
            input_tensor=a, weight_tensor=w, bias_tensor=None, fused_activation=fa,
            config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(o)

    run()
    ttnn.synchronize_device(mesh_device)
    signpost("start")
    run()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")


def _configs():
    """Focused sweep. Stage A: at fixed M_block=16, sweep (K_block, N_block,
    subblock) — the highest-leverage dims. Stage B: for a few promising subblocks,
    vary M_block. Kept ~120 configs (~35 min) instead of the full 832."""
    cfgs = []
    for label, (K, N, gelu) in SHAPES.items():
        kt = K // 32
        kbs = [x for x in [4, 8, 16, 32] if x <= kt]
        # Stage A: M_block=16 fixed, full K/N/subblock grid
        for kb in kbs:
            for nb in [4, 8]:
                for sbh, sbw in SUBBLOCKS:
                    if 16 % sbh or nb % sbw:
                        continue
                    cfgs.append((label, 16, kb, nb, sbh, sbw, 8, 8))
        # Stage B: vary M_block for the two canonical subblocks at mid K_block
        kb_mid = kbs[len(kbs) // 2]
        for mb in [8, 24, 32]:
            for sbh, sbw in [(4, 2), (2, 4), (8, 1)]:
                if mb % sbh:
                    continue
                cfgs.append((label, mb, kb_mid, 4, sbh, sbw, 8, 8))
    return cfgs


@pytest.mark.timeout(7200)
def test_matmul_subblock_sweep():
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    results = []
    cfgs = _configs()
    print(f"MM_SUBBLOCK sweeping {len(cfgs)} configs", flush=True)
    for label, mb, kb, nb, sbh, sbw, gx, gy in cfgs:
        tag = f"{label}_m{mb}k{kb}n{nb}_sb{sbh}x{sbw}_g{gx}x{gy}"
        subdir = f"bge_mmsb_{tag}"
        os.environ.update({
            "BGE_MM_SHAPE": label, "BGE_MM_MB": str(mb), "BGE_MM_KB": str(kb),
            "BGE_MM_NB": str(nb), "BGE_MM_SBH": str(sbh), "BGE_MM_SBW": str(sbw),
            "BGE_MM_GX": str(gx), "BGE_MM_GY": str(gy),
        })
        cmd = (
            "pytest models/demos/wormhole/bge_m3/tests/sweeps/"
            "sweep_matmul_subblock_tp2.py::test_matmul_subblock_worker -s -q"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            perf = post_process_ops_log(
                subdir, float_columns=["DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=True
            )
            durs = perf["DEVICE KERNEL DURATION [ns]"]
            ns = int(max(durs)) if len(durs) else 0
            print(f"MMSB_RESULT {tag} device_us={ns/1e3:.1f}", flush=True)
            results.append((label, ns, tag))
        except Exception as e:
            print(f"MMSB_FAILED {tag}: {type(e).__name__}: {str(e)[:60]}", flush=True)

    print("\nMMSB_RANKED (per shape):", flush=True)
    for label in SHAPES:
        rows = sorted([(ns, tag) for lbl, ns, tag in results if lbl == label and ns > 0])
        print(f"  --- {label} ---", flush=True)
        for ns, tag in rows[:6]:
            print(f"    {ns/1e3:8.1f} us  {tag}", flush=True)
    assert results
