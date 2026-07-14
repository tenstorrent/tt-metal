# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Why does all_gather_minimal_matmul_async not hide its gather behind its matmul?

The op gathers along K (the matmul's contraction dim) and accumulates one K-block at a time in L1,
so overlap is not merely possible — it is the design. The suspect is the *gather's own* critical
path: the fabric relay is issued by the LAST cores of the in0 core chain, so every ring hop must
first walk the whole on-chip store-and-forward chain.

That chain's length is `grid.y` (force_transpose=True routes in0 down a column). This harness
varies grid.y and reads the fused op's cost off the slope.

The gate matmul is the clean probe: ColParallelLinear(4096, 32) with TP=4 leaves N_local=8, i.e.
N_tiles=1. N is parallelized over grid.y, so shrinking grid.y removes NOTHING from the gate's real
matmul work (one N tile, wherever it lands) and removes ONLY chain hops. A drop in fused cost is
therefore attributable to the chain, not to the matmul.

Priced by slope, as in test_ccl_census: trace K copies, replay, take (T(K_HI) - T(K_LO)) / dK.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import ring_params

VIDEO_DIM = 4096
NUM_HEADS = 32
V_ROWS_S1 = 9728 // 8  # 1216 rows/device at SP=8

K_LO = 8
K_HI = 32
REPLAYS = 10


def _time_trace(mesh_device, body, k: int) -> float:
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        for _ in range(k):
            out = body()
            for t in out if isinstance(out, (list, tuple)) else [out]:
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / REPLAYS

    ttnn.release_trace(mesh_device, trace_id)
    return elapsed_ms


def _price(mesh_device, name: str, body, results: dict) -> None:
    t_start = time.perf_counter()
    out = body()
    for t in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)

    t_lo = _time_trace(mesh_device, body, K_LO)
    t_hi = _time_trace(mesh_device, body, K_HI)
    per_op_us = (t_hi - t_lo) * 1000 / (K_HI - K_LO)
    results[name] = per_op_us
    logger.info(
        f"CHAINPROBE {name:34s} per_op={per_op_us:8.2f} us "
        f"(t{K_LO}={t_lo:.3f}ms t{K_HI}={t_hi:.3f}ms warm={time.perf_counter() - t_start:.1f}s)"
    )


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {**ring_params, "trace_region_size": 90000000}, id="ring_bh_4x8")],
    indirect=True,
)
def test_agmm_chain_probe(mesh_device: ttnn.MeshDevice) -> None:
    tp_axis = 0
    num_links = 2
    tp = tuple(mesh_device.shape)[tp_axis]
    assert tp == 4

    full = mesh_device.compute_with_storage_grid_size()
    logger.info(f"CHAINPROBE full compute grid = {full.x} x {full.y}")

    ccl = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Ring)
    compute_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    v_loc = VIDEO_DIM // tp  # 1024

    x = ttnn.from_torch(
        torch.randn(1, 1, V_ROWS_S1, v_loc), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
    )
    x_full = ttnn.from_torch(
        torch.randn(1, 1, V_ROWS_S1, VIDEO_DIM), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
    )

    # gate: N_local = 32/4 = 8 -> padded to one tile. qkv: N_local = 3*4096/4 = 3072 -> 96 tiles.
    def _w(n_local):
        return ttnn.from_torch(
            torch.randn(1, 1, VIDEO_DIM, n_local) * 0.02,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh_device,
        )

    def _b(n_local):
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, n_local), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
        )

    w_gate, b_gate = _w(NUM_HEADS // tp), _b(NUM_HEADS // tp)  # (4096, 8) -> N_tiles=1
    w_qkv, b_qkv = _w(3 * VIDEO_DIM // tp), _b(3 * VIDEO_DIM // tp)  # (4096, 3072) -> N_tiles=96

    def cfg(gy, m_blk, k_blk, n_blk, sub=(2, 2)):
        return ttnn.MinimalMatmulConfig(
            M_block_size=m_blk,
            K_block_size=k_blk,
            N_block_size=n_blk,
            subblock_h=sub[0],
            subblock_w=sub[1],
            compute_with_storage_grid_size=ttnn.CoreCoord(full.x, gy),
        )

    def agmm(w, b, gy, m_blk, k_blk, n_blk, sub=(2, 2)):
        def body():
            return ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=w,
                bias_tensor=b,
                config=cfg(gy, m_blk, k_blk, n_blk, sub),
                compute_kernel_config=compute_cfg,
                persistent_output_buffer=ccl.get_ag_ping_pong_buffer(x.shape, 3, tp_axis, dtype=x.get_dtype()),
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(tp_axis),
                num_links=num_links,
                topology=ccl.topology,
                cluster_axis=tp_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full.x // num_links,
                num_buffers_per_channel=24,
                chunks=1,
            )[0]

        return body

    def mm(w, b, gy, m_blk, k_blk, n_blk, sub=(2, 2)):
        def body():
            return ttnn.experimental.minimal_matmul(
                x_full,
                w,
                bias_tensor=b,
                config=cfg(gy, m_blk, k_blk, n_blk, sub),
                compute_kernel_config=compute_cfg,
            )

        return body

    results: dict[str, float] = {}
    which = os.environ.get("PROBE", "chain")

    if which == "chain":
        # in0 chain length == grid.y. For the gate (N_tiles=1) grid.y buys no matmul parallelism,
        # so any change in the fused cost is chain, not compute.
        for gy in (9, 7, 5, 4):
            _price(mesh_device, f"agmm_gate_gy{gy}", agmm(w_gate, b_gate, gy, 8, 8, 2), results)
            _price(mesh_device, f"mm_gate_gy{gy}", mm(w_gate, b_gate, gy, 8, 8, 2), results)
    elif which == "kblock":
        # K_blocks_per_device = 32 / K_block_size -> 4, 2, 1. Fewer, fatter ring steps = fewer
        # chain traversals for the same bytes. Separates per-hop fixed cost from bandwidth.
        for k_blk in (8, 16, 32):
            _price(mesh_device, f"agmm_gate_kblk{k_blk}", agmm(w_gate, b_gate, 9, 8, k_blk, 2), results)
        for k_blk in (8, 16, 32):
            _price(mesh_device, f"mm_gate_kblk{k_blk}", mm(w_gate, b_gate, 9, 8, k_blk, 2), results)
    elif which == "qkv":
        for gy in (9, 5):
            _price(mesh_device, f"agmm_qkv_gy{gy}", agmm(w_qkv, b_qkv, gy, 8, 8, 8, (2, 4)), results)
            _price(mesh_device, f"mm_qkv_gy{gy}", mm(w_qkv, b_qkv, gy, 8, 8, 8, (2, 4)), results)
        for k_blk in (16, 32):
            _price(mesh_device, f"agmm_qkv_kblk{k_blk}", agmm(w_qkv, b_qkv, 9, 8, k_blk, 8, (2, 4)), results)

    logger.info("CHAINPROBE ==== summary ====")
    for k, v in results.items():
        logger.info(f"CHAINPROBE  {k:34s} {v:8.2f} us")
