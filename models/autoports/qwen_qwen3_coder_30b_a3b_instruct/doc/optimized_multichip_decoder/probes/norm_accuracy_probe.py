# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04: pick the core count and compute config for the sharded decode RMSNorm.

``norm_router_probe.py`` showed the shipped one-core ``rms_norm`` at 19.80 us and
a width-sharded one at ~4.2 us, but reported the difference only against the
shipped op. This probe adds a **torch fp64 reference**, so "different" can be
split into "less accurate" and "more accurate", and sweeps the compute config
(the shipped interleaved call passes none at all).

    python norm_accuracy_probe.py

Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")

REPS = 32
HIDDEN = 2048
ROWS = 32
EPS = 1e-6


def bank_row(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def width_sharded(dim, cores):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(bank_row(cores), [ROWS, dim // cores], ttnn.ShardOrientation.ROW_MAJOR),
    )


def norm_pc(dim, cores):
    block_w = dim // cores // 32
    subblock_w = next(w for w in (4, 3, 2, 1) if block_w % w == 0)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[cores, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )


mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=60_000_000, l1_small_size=32768)


def slope(fn):
    def build(n):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        s = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            s.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(s)

    fn()
    ttnn.synchronize_device(mesh)
    return (build(REPS + 1) - build(1)) / REPS


def rep(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


try:
    torch.manual_seed(0)
    x_t = torch.randn(1, 1, 1, HIDDEN) * 0.5
    w_t = torch.randn(1, 1, 1, HIDDEN) * 0.3 + 1.0
    # torch reference, computed in fp64 from the *bf16-rounded* inputs the device sees
    xb = x_t.to(torch.bfloat16).double()
    wb = w_t.to(torch.bfloat16).double()
    ref = (xb / (xb.pow(2).mean(-1, keepdim=True) + EPS).sqrt() * wb).float()

    x = rep(x_t)
    w_tile = rep(w_t)
    w_rm = rep(w_t.reshape(1, 1, HIDDEN // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)

    def report(tag, out, t):
        got = ttnn.to_torch(
            out if not out.memory_config().is_sharded() else ttnn.sharded_to_interleaved(out),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )[0:1].float()
        err = (got - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"P|{tag:44s} {t:7.2f} us   max|err vs fp64| {err:.3e}  rel {rel:.2e}", flush=True)

    o = ttnn.rms_norm(x, weight=w_tile, epsilon=EPS)
    report("interleaved, no compute config (shipped)", o, slope(lambda: ttnn.rms_norm(x, weight=w_tile, epsilon=EPS)))

    CKS = [
        ("default", None),
        (
            "HiFi4 fp32acc",
            ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        ),
    ]
    for cores in (4, 8, 16):
        mc, pc = width_sharded(HIDDEN, cores), norm_pc(HIDDEN, cores)
        xs = ttnn.to_memory_config(x, mc)
        for ck_tag, ck in CKS:
            try:

                def leg(xs=xs, mc=mc, pc=pc, ck=ck):
                    return ttnn.rms_norm(
                        xs, weight=w_rm, epsilon=EPS, program_config=pc, memory_config=mc, compute_kernel_config=ck
                    )

                report(f"sharded {cores:2d} cores, {ck_tag}", leg(), slope(leg))
            except Exception as exc:
                print(f"P|sharded {cores:2d} cores, {ck_tag}: FAILED {str(exc)[:120]}", flush=True)
        print(
            f"P|  ({cores} cores) i2s {slope(lambda mc=mc: ttnn.to_memory_config(x, mc)):.2f} us  "
            f"s2i {slope(lambda xs=xs: ttnn.sharded_to_interleaved(xs)):.2f} us",
            flush=True,
        )
finally:
    ttnn.close_mesh_device(mesh)
print("P|done")
