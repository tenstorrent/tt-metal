# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-pass parameter sweep for gpt-oss-120b attention o_proj fused MM+RS.

Shape: M=128, K=512, N=3072 (per-device). TG 4x8, num_links=4, cluster_axis=1.

Runs every (core_grid, M/K/N block, chunk_width) combo within one pytest call
so device init/fabric setup is paid once. Each config runs N_ITERS iterations,
wrapped in signposts "cfg=<id>". The companion driver parses the Tracy CSV.
"""
from __future__ import annotations

import itertools
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    create_global_semaphores,
)
from tracy import signpost


M = int(os.environ.get("GPT_OSS_SWEEP_M", "128"))
K = 512
N = 3072
TILE = 32
DIM = 3
N_ITERS = int(os.environ.get("GPT_OSS_SWEEP_ITERS", "10"))
N_WARMUP = int(os.environ.get("GPT_OSS_SWEEP_WARMUP", "5"))
SWEEP_FILTER = os.environ.get("GPT_OSS_SWEEP_FILTER", "")
# Run only configs with index in [BATCH_START, BATCH_END). Enables batched
# sweeps: clear the JIT cache between runs to bound per-batch disk usage.
BATCH_START = int(os.environ.get("GPT_OSS_SWEEP_BATCH_START", "0"))
BATCH_END = int(os.environ.get("GPT_OSS_SWEEP_BATCH_END", "1000000"))

# Subblock sweep mode: pins block params to the S=1024 winner and iterates
# subblock_h × subblock_w combinations instead.
SUBBLOCK_SWEEP = os.environ.get("GPT_OSS_SUBBLOCK_SWEEP", "0") == "1"
# Ring-RS worker-count sweep mode: pins block params + subblocks, iterates
# num_workers_per_link. Requires GPT_OSS_SUBBLOCK_SWEEP=0.
WORKER_SWEEP = os.environ.get("GPT_OSS_WORKER_SWEEP", "0") == "1"
# Subblock override for worker sweep (so we can lock in the subblock winner).
SWEEP_SBH = int(os.environ.get("GPT_OSS_SWEEP_SBH", "1"))
SWEEP_SBW = int(os.environ.get("GPT_OSS_SWEEP_SBW", "1"))


def _build_sweep():
    """Build the sweep config list.

    Pruning rules (learned from the first run, which hung after the 16th cfg):
    - Skip K_block == K_tiles (single-pass K) — hung at y3_mb1_kb16_...
    - Skip N_block == Nt_per_core (single-pass N) — same concern.
    - Only keep M_block values that divide Mt_per_core evenly.
    - M_tiles=4 is small; at grid_y=6 some rows have zero M-work — drop.
    """
    configs = []
    Nt = N // TILE
    Kt = K // TILE
    M_tiles = M // TILE  # 4 at S=128, 32 at S=1024
    Nt_per_core_x8 = Nt // 8  # 12
    # At larger M the MM has more work, so more grid_y is viable. Span 2..7.
    grid_y_candidates = (2, 3, 4, 5, 6, 7) if M_tiles >= 8 else (2, 3, 4)
    for grid_y in grid_y_candidates:
        Mt_per_core = max(1, (M_tiles + grid_y - 1) // grid_y)
        # Keep only M_block values that divide Mt_per_core evenly.
        all_mbs = (1, 2, 4, 8, 16)
        m_blocks = [mb for mb in all_mbs if mb <= Mt_per_core and Mt_per_core % mb == 0]
        for mb, kb, nb, cw in itertools.product(
            m_blocks,
            (2, 4, 8),  # K_block (skip 16 = full K single-pass)
            (2, 3, 4, 6),  # N_block (skip 12 = full Nt_per_core)
            (1, 2, 4),
        ):
            if kb > Kt or Kt % kb != 0:
                continue
            if nb > Nt_per_core_x8:
                continue
            cfg_id = f"y{grid_y}_mb{mb}_kb{kb}_nb{nb}_cw{cw}"
            if SWEEP_FILTER and SWEEP_FILTER not in cfg_id:
                continue
            configs.append(
                dict(
                    id=cfg_id,
                    mm_grid=ttnn.CoreCoord(8, grid_y),
                    M_block=mb,
                    K_block=kb,
                    N_block=nb,
                    chunk_width=cw,
                )
            )
    # Batch slicing for incremental sweeps (avoids JIT cache ballooning when
    # the full sweep is large). Default range is open so behaviour is unchanged
    # when BATCH_* env vars aren't set.
    return configs[BATCH_START:BATCH_END]


def _build_subblock_sweep():
    """Pin the winning shape-level config and sweep only subblock_h × subblock_w.

    Active when GPT_OSS_SUBBLOCK_SWEEP=1. Designed for the S=1024 winner:
    y4, M_block=8, K_block=8, N_block=6, chunk_width=2. Subblock constraint:
    subblock_h divides M_block and subblock_w divides N_block, and with
    fp32_dest_acc_en=True the product subblock_h*subblock_w should stay ≤8.
    """
    if M // TILE == 32:
        mb, kb, nb, cw, grid_y = 8, 8, 6, 2, 4
    else:  # S=128 fallback winner
        mb, kb, nb, cw, grid_y = 2, 8, 6, 2, 2

    # Divisors of M_block and N_block respectively, with h*w ≤ 8.
    sbh_vals = [h for h in (1, 2, 4, 8) if h <= mb and mb % h == 0]
    sbw_vals = [w for w in (1, 2, 3, 6) if w <= nb and nb % w == 0]

    configs = []
    for sbh in sbh_vals:
        for sbw in sbw_vals:
            if sbh * sbw > 8:
                continue
            cid = f"sb_y{grid_y}_mb{mb}_kb{kb}_nb{nb}_cw{cw}_sbh{sbh}_sbw{sbw}"
            configs.append(
                dict(
                    id=cid,
                    mm_grid=ttnn.CoreCoord(8, grid_y),
                    M_block=mb,
                    K_block=kb,
                    N_block=nb,
                    chunk_width=cw,
                    subblock_h=sbh,
                    subblock_w=sbw,
                )
            )
    return configs


def _build_worker_sweep():
    """Pin block config + subblock and iterate num_workers_per_link."""
    if M // TILE == 32:
        mb, kb, nb, cw, grid_y = 8, 8, 6, 2, 4
    else:
        mb, kb, nb, cw, grid_y = 2, 8, 6, 2, 2

    # With 4 links on TG and RS zone of (8 - grid_y) * 8, the legal workers
    # range is 1..(rs_zone/(2*num_links) - 1) inclusive.
    num_links = 4
    rs_zone = (8 - grid_y) * 8
    max_workers = rs_zone // (2 * num_links) - 1
    configs = []
    for w in range(1, max_workers + 1):
        cid = f"wk_y{grid_y}_mb{mb}_kb{kb}_nb{nb}_cw{cw}_w{w}"
        configs.append(
            dict(
                id=cid,
                mm_grid=ttnn.CoreCoord(8, grid_y),
                M_block=mb,
                K_block=kb,
                N_block=nb,
                chunk_width=cw,
                subblock_h=SWEEP_SBH,
                subblock_w=SWEEP_SBW,
                num_workers_per_link=w,
            )
        )
    return configs


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 30000000}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_gpt_oss_attn_o_proj_s128_sweep(mesh_device, device_params):
    num_links = 4
    cluster_axis = 1
    num_devices = mesh_device.shape[cluster_axis]

    # Setup sub-device / fabric cores
    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    if SUBBLOCK_SWEEP:
        configs = _build_subblock_sweep()
        logger.info(f"SUBBLOCK sweep: {len(configs)} configs, {N_ITERS} iters each")
    elif WORKER_SWEEP:
        configs = _build_worker_sweep()
        logger.info(f"WORKER sweep: {len(configs)} configs, {N_ITERS} iters each")
    else:
        configs = _build_sweep()
        logger.info(f"Sweep: {len(configs)} configs, {N_ITERS} iters each")

    # Shared torch goldens (input and per-device weight chunks)
    torch.manual_seed(0)
    input_shape = [1, 1, M, K]
    weight_shape_global = [num_devices, 1, K, N]
    torch_input = torch.randn(input_shape, dtype=torch.float32)
    torch_weight_global = torch.randn(weight_shape_global, dtype=torch.float32)

    torch_weight_chunks = torch.chunk(torch_weight_global, num_devices, dim=0)
    mm_outputs = [torch.matmul(torch_input, torch_weight_chunks[d]) for d in range(num_devices)]
    torch_rs_reduced = torch.sum(torch.stack(mm_outputs), dim=0)
    torch_rs_scattered = torch.chunk(torch_rs_reduced, num_devices, dim=DIM)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 0

    # Build device tensors once.
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_weight = ttnn.from_torch(
        torch_weight_global,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    failures = []
    for ci, cfg in enumerate(configs):
        mm_grid = cfg["mm_grid"]
        rs_offset = ttnn.CoreCoord(0, mm_grid.y)
        mm_config = ttnn.MinimalMatmulConfig(
            M_block_size=cfg["M_block"],
            K_block_size=cfg["K_block"],
            N_block_size=cfg["N_block"],
            subblock_h=cfg.get("subblock_h", 1),
            subblock_w=cfg.get("subblock_w", 1),
            compute_with_storage_grid_size=mm_grid,
        )

        # Fresh semaphores per config.
        sems = create_global_semaphores(mesh_device, all_cores, 0)
        barrier_sem = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

        def run_op(_cfg=cfg, _sems=sems, _barrier=barrier_sem, _mm=mm_config):
            return ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                tt_input,
                tt_weight,
                DIM,
                _sems,
                rs_offset,
                num_links=num_links,
                memory_config_mm=mem_config,
                rs_output_mem_config=mem_config,
                topology=ttnn.Topology.Ring,
                cluster_axis=cluster_axis,
                config=_mm,
                compute_kernel_config=compute_config,
                barrier_semaphore=_barrier,
                chunk_width_in_mm_blocks=_cfg["chunk_width"],
                num_workers_per_link=_cfg.get("num_workers_per_link"),
            )

        try:
            # Warmup / compile — N_WARMUP iters OUTSIDE the signpost window so
            # the first-call JIT/program-cache-entry overhead is excluded
            # from the measured average.
            for _ in range(N_WARMUP):
                mm_out, rs_out = run_op()
                mm_out.deallocate(True)
                rs_out.deallocate(True)
            ttnn.synchronize_device(mesh_device)

            # Correctness sanity on first config only (cheap compared to perf loop).
            if ci == 0:
                mm_out, rs_out = run_op()
                ttnn.synchronize_device(mesh_device)
                concat_mesh_shape = list(mesh_device.shape)
                concat_mesh_shape[1 - cluster_axis] = 1
                concat_dims = [DIM, DIM]
                concat_dims[1 - cluster_axis] = 0 if DIM != 0 else 1
                tt_rs_torch = ttnn.to_torch(
                    rs_out,
                    mesh_composer=ttnn.create_mesh_composer(
                        mesh_device,
                        ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(concat_mesh_shape)),
                    ),
                )
                for d in range(num_devices):
                    try:
                        eq, pcc = comp_pcc(tt_rs_torch[d : d + 1], torch_rs_scattered[d], 0.98)
                        logger.info(f"cfg={cfg['id']} dev{d} RS PCC: {pcc}")
                    except Exception as e:
                        logger.warning(f"PCC skipped for dev{d}: {e}")
                        break
                mm_out.deallocate(True)
                rs_out.deallocate(True)

            # Perf loop with signpost — measured iters only.
            signpost(f"start cfg={cfg['id']}")
            for _ in range(N_ITERS):
                mm_out_i, rs_out_i = run_op()
                mm_out_i.deallocate(True)
                rs_out_i.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            signpost(f"stop cfg={cfg['id']}")
        except Exception as e:
            logger.error(f"cfg={cfg['id']} FAILED: {e}")
            failures.append((cfg["id"], str(e)))

    logger.info(f"Sweep done. {len(configs) - len(failures)} configs ok, {len(failures)} failed")
    if failures:
        for cid, msg in failures:
            logger.warning(f"FAIL cfg={cid}: {msg[:200]}")
