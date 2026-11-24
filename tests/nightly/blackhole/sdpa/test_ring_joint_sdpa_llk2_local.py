# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# ring_joint_scaled_dot_product_attention on ONE device (1x1 mesh, ring_size 1) at the Wan2.2 480p QB2 per-device
# shape (q_chunk 128, k_chunk 512, 20 heads, 16384 local tokens, head dim 128, bf16, HiFi2, logical_n with a partial
# last key tile). One ring iteration of the real 2x2-mesh run, without the DRAM staging between iterations.
# Device time of the 1.0 and the LLK 2.0 kernel (TT_RING_SDPA_LLK2=0/1) in the same session, then the accuracy of both.
# Logs "LLK2_RING_PERF <kernel> median_ns=<n> runs=<n>" and "LLK2_RING_ACC <kernel> pcc=... rmse=..." lines, appended
# to $LLK2_RING_PERF_OUT when that is set.

import os
import statistics

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.operations.ccl import Topology

from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

B, NH, N_LOCAL, D = 1, 20, 16384, 128
LOGICAL_N = int(os.environ.get("RING_LOCAL_LOGICAL_N", N_LOCAL - 8))  # 16376: 24 valid rows in the last key tile
Q_CHUNK, K_CHUNK = int(os.environ.get("RING_LOCAL_Q_CHUNK", 128)), int(os.environ.get("RING_LOCAL_K_CHUNK", 512))
RUNS = int(os.environ.get("RING_LOCAL_RUNS", 5))
ACC_HEADS = (0, NH - 1)


def _out(line):
    logger.info(line)
    path = os.environ.get("LLK2_RING_PERF_OUT")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _fa_rand(*shape):
    return torch.randn(shape) + torch.randn(shape) * 10 * torch.bernoulli(torch.full(shape, 0.001))


@pytest.fixture(scope="module")
def ring1():
    # The fused all-gather's host code needs a 1D fabric config even with no peer (ccl_common.cpp, line unicast), but
    # metal refuses to launch fabric on a 1-chip mesh. FabricManagerMode.ENABLED (no INIT/TERMINATE flag: "an external
    # fabric manager runs the routers") sets the config without launching anything; with ring_size 1 nothing is sent.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.ENABLED,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    # No sub-device manager (loading one segfaults under the mock build); the op then runs on the default one.
    sub_device_id = None
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)]
    yield mesh, grid, sub_device_id, sems
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_ring_joint_sdpa_llk2_local(ring1, monkeypatch):
    mesh, grid, sub_device_id, sems = ring1
    torch.manual_seed(1234)
    Q, K, V = (_fa_rand(B, NH, N_LOCAL, D) for _ in range(3))
    # Keys past logical_n are padding: make them dominate the scores, so a missing partial-key mask fails accuracy.
    K[:, :, LOGICAL_N:, :] = 8.0

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    tq, tk, tv = dev(Q), dev(K), dev(V)
    pk, pv = dev(torch.zeros(B, NH, N_LOCAL, D)), dev(torch.zeros(B, NH, N_LOCAL, D))
    ttnn.synchronize_device(mesh)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x - 1, grid.y),  # the last column is the CCL column, as in Wan
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("RING_LOCAL_FIDELITY", "HiFi2")),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def run(kernel):
        monkeypatch.setenv("TT_RING_SDPA_LLK2", "1" if kernel == "llk2" else "0")
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            None,
            None,
            None,
            persistent_output_buffer_k=pk,
            persistent_output_buffer_v=pv,
            joint_strategy="rear",
            logical_n=LOGICAL_N,
            program_config=program_config,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=sems,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh,
            topology=Topology.Linear,
            subdevice_id=sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
        )
        return out

    kernels = os.environ.get("RING_LOCAL_KERNELS", "llk1,llk2").split(",")
    for k in kernels:  # compile outside the profiler window (a mock prebuild stops at require_realtime_profiler)
        run(k)
    ttnn.synchronize_device(mesh)
    require_realtime_profiler("the LLK 2.0 ring SDPA perf comparison")
    medians = {}
    for k in kernels:
        _, per_program = profile_realtime_program_merged(mesh, lambda: [run(k) for _ in range(RUNS)])
        # The op's programs only (a stray program can land in the window): those with the ring reader.
        durations = [
            e["duration_ns"] for _, e in sorted(per_program.items()) if "ring_joint_reader" in str(e["kernel_sources"])
        ]
        assert len(durations) == RUNS, f"expected {RUNS} programs, got {len(durations)}"
        medians[k] = statistics.median(durations)
        _out(
            f"LLK2_RING_PERF {k} median_ns={medians[k]:.0f} runs={RUNS} min_ns={min(durations):.0f} "
            f"q{Q_CHUNK}/k{K_CHUNK} logical_n={LOGICAL_N}"
        )
    if "llk1" in medians and "llk2" in medians:
        _out(f"LLK2_RING_PERF ratio llk2/llk1={medians['llk2'] / medians['llk1']:.4f}")

    # Accuracy: heads ACC_HEADS, every Q row below logical_n, against fp32 attention over the first logical_n keys.
    ref = {}
    for h in ACC_HEADS:
        q, k_, v = Q[:, h : h + 1, :LOGICAL_N], K[:, h : h + 1, :LOGICAL_N], V[:, h : h + 1, :LOGICAL_N]
        ref[h] = torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(q[:, :, s : s + 2048], k_, v)
                for s in range(0, LOGICAL_N, 2048)
            ],
            dim=2,
        )
    outs = {}
    for k in kernels:
        out = ttnn.to_torch(ttnn.get_device_tensors(run(k))[0]).float()
        outs[k] = out
        g = torch.cat([ref[h] for h in ACC_HEADS], dim=1).flatten()
        o = torch.cat([out[:, h : h + 1, :LOGICAL_N] for h in ACC_HEADS], dim=1).flatten()
        pcc = torch.corrcoef(torch.stack([g, o]))[0, 1].item()
        rmse = torch.sqrt(((g - o) ** 2).mean()).item()
        _out(f"LLK2_RING_ACC {k} pcc={pcc:.6f} rmse={rmse:.6f} heads={ACC_HEADS} rows<{LOGICAL_N}")
        assert pcc > 0.994, f"{k}: PCC {pcc} below 0.994"
    if "llk1" in outs and "llk2" in outs:
        a, b = outs["llk1"][:, :, :LOGICAL_N].flatten(), outs["llk2"][:, :, :LOGICAL_N].flatten()
        _out(
            f"LLK2_RING_ACC llk2_vs_llk1 pcc={torch.corrcoef(torch.stack([a, b]))[0, 1].item():.6f} "
            f"max_abs={(a - b).abs().max().item():.4f} all_heads"
        )
