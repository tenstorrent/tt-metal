# SPDX-License-Identifier: Apache-2.0
"""Device-profiler sweep: matmul BACKENDS for seq-parallel TP2 shapes (M=49152).

Compares ttnn.matmul 2D-multicast (with out_block_h/w streaming so the large-M
output does not have to stay fully L1-resident) against the current
minimal_matmul, using DEVICE KERNEL DURATION (not wall clock), mirroring
sweep_ring_sdpa_tp2.py.

Each config runs in a fresh subprocess under the device profiler, delimited by
start/stop signposts; the driver reads the max device-kernel-duration.

RESULT (2026-07-14): minimal_matmul beats best-2D 1.7-3.2x on every shape
(qkv 2894 vs 8344us, mlpwi 3430 vs 10960, mlpwo 2709 vs 4615, attnout 856 vs
1682). M=49152 is very tall/skinny so minimal_matmul's M-streaming wins over 2D
mcast weight-refetch/mcast overhead. DRAM-sharded MM needs an L1-sharded
activation (impossible for the 53MB activation) so it is not applicable here.
Kept as the device-time reference harness for future re-tuning.

Run:
  TT_VISIBLE_DEVICES=0 pytest \\
    models/demos/wormhole/bge_m3/tests/sweeps/sweep_matmul_variants_tp2.py::test_matmul_variants_sweep -s -q
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


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_matmul_variant_worker(mesh_device):
    """Profile one (shape, backend, block) config, delimited by start/stop."""
    label = os.environ["BGE_MM_SHAPE"]
    backend = os.environ["BGE_MM_BACKEND"]  # "minimal" | "2d"
    ibw = int(os.environ["BGE_MM_IBW"])
    obh = int(os.environ["BGE_MM_OBH"])
    obw = int(os.environ["BGE_MM_OBW"])
    K, N, gelu = SHAPES[label]
    mt, kt, nt = M // 32, K // 32, N // 32

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

    if backend == "minimal":
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=16, K_block_size=min(ibw, kt), N_block_size=obw,
            subblock_h=4, subblock_w=2, compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )
        fa = (ttnn.UnaryOpType.GELU, True) if gelu else None

        def run():
            o = ttnn.experimental.minimal_matmul(
                input_tensor=a, weight_tensor=w, bias_tensor=None, fused_activation=fa,
                config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
                compute_kernel_config=ck,
            )
            ttnn.deallocate(o)
    else:  # 2d
        pcm, pcn = (mt + 7) // 8, (nt + 7) // 8
        sbh = next((h for h in [4, 2, 1] if obh % h == 0), 1)
        sbw = next((x for x in [4, 2, 1] if obw % x == 0 and sbh * x <= 8), 1)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8), in0_block_w=ibw,
            out_subblock_h=sbh, out_subblock_w=sbw, out_block_h=obh, out_block_w=obw,
            per_core_M=pcm, per_core_N=pcn, transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU if gelu else None),
        )

        def run():
            o = ttnn.matmul(
                a, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
                compute_kernel_config=ck, program_config=pc,
            )
            ttnn.deallocate(o)

    run()  # compile + warmup
    ttnn.synchronize_device(mesh_device)
    signpost("start")
    run()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")


def _configs():
    cfgs = []
    for label, (K, N, gelu) in SHAPES.items():
        nt = N // 32
        pcn = (nt + 7) // 8
        # minimal baseline (ibw=K_block, obw=N_block)
        cfgs.append((label, "minimal", 8, 16, 4))
        # 2d: sweep in0_block_w, out_block_h, out_block_w
        for ibw in [2, 4]:
            for obh in [8, 16]:
                for obw in sorted({pcn, max(1, pcn // 2), 4}):
                    cfgs.append((label, "2d", ibw, obh, obw))
    return cfgs


@pytest.mark.timeout(3600)
def test_matmul_variants_sweep():
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    results = []
    for label, backend, ibw, obh, obw in _configs():
        tag = f"{label}_{backend}_ibw{ibw}_obh{obh}_obw{obw}"
        subdir = f"bge_mm_{tag}"
        os.environ["BGE_MM_SHAPE"] = label
        os.environ["BGE_MM_BACKEND"] = backend
        os.environ["BGE_MM_IBW"] = str(ibw)
        os.environ["BGE_MM_OBH"] = str(obh)
        os.environ["BGE_MM_OBW"] = str(obw)
        cmd = (
            "pytest models/demos/wormhole/bge_m3/tests/sweeps/"
            "sweep_matmul_variants_tp2.py::test_matmul_variant_worker -s -q"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            perf = post_process_ops_log(
                subdir, float_columns=["DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=True
            )
            durs = perf["DEVICE KERNEL DURATION [ns]"]
            ns = int(max(durs)) if len(durs) else 0
            print(f"MM_RESULT {tag} device_us={ns/1e3:.1f}", flush=True)
            results.append((label, ns, tag))
        except Exception as e:
            print(f"MM_FAILED {tag}: {type(e).__name__}: {str(e)[:70]}", flush=True)

    print("\nMM_RANKED (per shape):", flush=True)
    for label in SHAPES:
        rows = sorted([(ns, tag) for lbl, ns, tag in results if lbl == label])
        print(f"  --- {label} ---", flush=True)
        for ns, tag in rows[:5]:
            print(f"    {ns/1e3:8.1f} us  {tag}", flush=True)
    assert results, "no matmul config completed"
