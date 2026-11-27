# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The H3 DiT's ring joint SDPA alone, called exactly as `MiniMaxH3Attention` calls it on the mesh.

Two modes, chosen by `H3_SDPA_ROWS` (padded rows per device; default 13664, the WH 15 s length):
  * timing: `H3_SDPA_REPEATS` (default 5) calls after one warm call, wall clock per call after a device sync.
  * accuracy: when the full sequence is small enough for a CPU reference (`H3_SDPA_CHECK=1`, default
    on below 2048 rows per device), PCC and max abs error against torch's SDPA in FP32.

Heads and head dim come from the block config; the SP / TP axes, link count and topology from the mesh's
pipeline preset; the grid, chunk sizes and `dynamic_local_n` from the attention module's rules for the denoise
path the preset resolves to (`MINIMAX_H3_DYNAMIC_LENGTH` overrides it, as in the pipeline). So a kernel change
measured here is the change the pipeline sees. `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` sweeps the
H3 chunk pairs by device kernel time under the profiler, on its own mesh setup; this times the whole call, ring
all-gather included, on the preset's mesh and CCL setup, with no profiler.

    pytest models/tt_dit/tests/models/minimax_h3/test_sdpa_bench_minimax_h3.py -k 4x8_WH -s
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ....models.transformers.minimax_h3.attention_minimax_h3 import DYNAMIC_RING_SDPA_CHUNKS, MiniMaxH3Attention
from ....models.transformers.minimax_h3.length_classes import length_class
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import _resolve_denoise_path, resolve_mesh_preset
from ....utils.device_constants import compute_grid_key
from .common import GALAXY_MESHES, REAL_BLOCK_CONFIG, fabric_topology

HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_h3_ring_sdpa_bench(mesh_device):
    rows = int(os.environ.get("H3_SDPA_ROWS", 13664))
    repeats = int(os.environ.get("H3_SDPA_REPEATS", 5))
    check = os.environ.get("H3_SDPA_CHECK", "1" if rows <= 2048 else "0") == "1"
    preset = resolve_mesh_preset(tuple(mesh_device.shape))
    tp_axis, sp_axis = preset["tp_axis"], preset["sp_axis"]
    # The pipeline's rule: the open fabric wins over the preset on a line.
    topology = ttnn.Topology.Linear if fabric_topology() == ttnn.Topology.Linear else preset["topology"]
    dynamic = _resolve_denoise_path(
        preset,
        trace_denoise=None,
        bucket_denoise=None,
        bucket_ladder=None,
        dynamic_length=None,
        coresident=preset.get("coresident", True),
        topology=topology,
    ).dynamic_length
    tp, sp = mesh_device.shape[tp_axis], mesh_device.shape[sp_axis]
    local_heads = HEADS // tp
    seq = rows * sp
    logical = int(os.environ.get("H3_SDPA_LOGICAL", seq - (rows // 7 if rows > 1024 else 13)))

    ccl = CCLManager(mesh_device, num_links=preset["num_links"], topology=topology)
    full_grid = mesh_device.compute_with_storage_grid_size()
    worker_grid = (full_grid.x - 1, full_grid.y)

    # The model's chunk rule, without building the whole attention module.
    if dynamic:
        q_chunk, k_chunk = DYNAMIC_RING_SDPA_CHUNKS[compute_grid_key(full_grid)][length_class(rows).name]
    else:
        q_chunk, k_chunk = MiniMaxH3Attention.measured_sdpa_chunk_sizes.get(rows, (256, 512))
    q_chunk = int(os.environ.get("H3_SDPA_Q", q_chunk))
    k_chunk = int(os.environ.get("H3_SDPA_K", k_chunk))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*worker_grid),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)
    # Heads over TP (dim 1) on the TP axis, rows over SP (dim 2) on the SP axis.
    dims = [None, None]
    dims[tp_axis] = 1
    dims[sp_axis] = 2

    def to_mesh(t, mapper):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    if check:
        shape = (1, HEADS, seq, HEAD_DIM)
        q, k, v = torch.randn(shape) * 0.5, torch.randn(shape) * 0.5, torch.randn(shape)
        for t in (q, k, v):
            t[:, :, logical:] = 0
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    else:
        # Timing only: every device gets the same local shard (the full sequence is ~10 GB of host
        # tensors at 15 s). The kernel's work does not depend on the values.
        shape = (1, local_heads, rows, HEAD_DIM)
        q, k, v = torch.randn(shape) * 0.5, torch.randn(shape) * 0.5, torch.randn(shape)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_q, tt_k, tt_v = to_mesh(q, mapper), to_mesh(k, mapper), to_mesh(v, mapper)
    dummy = ttnn.from_torch(
        torch.zeros((1, local_heads, 0, HEAD_DIM)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logical_n = ttnn.from_torch(
        torch.tensor([[[[logical]]]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    local_shape = tt_k.shape

    def run():
        out, _prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            dummy,
            dummy,
            dummy,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(local_shape, 2, sp_axis, dtype=tt_k.dtype),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(local_shape, 2, sp_axis, dtype=tt_v.dtype),
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ccl.topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(worker_grid[0], 0),
            use_column_major_ccl=True,
            dynamic_local_n=dynamic,
        )
        return out

    out = run()
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = run()
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    # Per-core steps: each core runs ceil(q chunks x heads / cores) q chunks, each against every k chunk
    # of the gathered sequence; the median call over that is the inner-loop step period.
    cores = worker_grid[0] * worker_grid[1]
    rounds = math.ceil(math.ceil(rows / q_chunk) * local_heads / cores)
    steps = rounds * math.ceil(seq / k_chunk)
    median = times[len(times) // 2]
    logger.info(
        f"H3_SDPA rows/dev {rows} seq {seq} logical {logical} heads/dev {local_heads} q{q_chunk} k{k_chunk} "
        f"dynamic_local_n {dynamic} links {ccl.num_links} {topology} "
        f"grid {worker_grid}: median {median:.2f} ms, min {times[0]:.2f}, max {times[-1]:.2f}; "
        f"{rounds} rounds x {steps // rounds} k chunks = {median * 1e3 / steps:.2f} us/step"
    )

    if check:
        got = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        ).float()[:, :, :logical]
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :logical].float(), k[:, :, :logical].float(), v[:, :, :logical].float()
        )
        passing, pcc = comp_pcc(got, ref, 0.999)
        err = (got - ref).abs().max().item()
        logger.info(f"H3_SDPA check: PCC {pcc}, max abs err {err:.4e}")
        assert passing, pcc
