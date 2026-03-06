#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark: prefill_moe_compute latency sweep over P=32/64/128.

Measures wall-clock latency for each P value with fabric dispatch on a 1x2 mesh.
Reports min/median/mean for N=5 timed iterations (after 1 warmup).

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/bench_prefill_moe_p_sweep.py
"""

import time
import statistics
import torch
import ttnn
from loguru import logger

# Fixed parameters
TILE = 32
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_EXPERTS_PER_DEVICE = 1
NUM_EXPERTS_TOTAL = 2
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
N_ITERS = 5

# P values to sweep
P_VALUES = [32, 64, 128]


def create_tensors_for_p(mesh_device, P):
    """Create all input tensors for a given P value."""
    torch.manual_seed(42)

    N_TOKENS_PER_DEVICE = P
    N_TOKENS_TOTAL = P * 2

    D_tiles = D // TILE
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    n_tiles_dn = D // TILE
    D_padded = n_tiles_dn * TILE
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE

    # Generate data
    all_hs = torch.randn(N_TOKENS_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]

    # Shuffle gate_up weights
    shuffled_ws = []
    for w in gate_up_ws_raw:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # Dispatch metadata: ~60% local, ~40% remote
    n_local = int(P * 0.625)  # 20/32, 40/64, 80/128
    n_send = P - n_local
    n_recv = n_send

    dev0_dispatch = [n_local, n_send, n_recv] + list(range(n_local)) + list(range(n_local, P))
    dev1_dispatch = [n_local, n_send, n_recv] + list(range(n_send, P)) + list(range(n_send))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]

    # K=1 weight = 1.0
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    all_rows = list(range(P))

    # Create tensors
    hs_dev0 = all_hs[:P].unsqueeze(0).unsqueeze(0)
    hs_dev1 = all_hs[P:].unsqueeze(0).unsqueeze(0)
    stacked_hs = torch.cat([hs_dev0, hs_dev1], dim=0)
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gu_mesh_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
        dev0_w = shuffled_ws[e].unsqueeze(0).unsqueeze(0)
        dev1_w = shuffled_ws[e + NUM_EXPERTS_PER_DEVICE].unsqueeze(0).unsqueeze(0)
        stacked_w = torch.cat([dev0_w, dev1_w], dim=0)
        gu_t = ttnn.from_torch(
            stacked_w,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gu_mesh_list.append(gu_t)

    dn_mesh_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
        dev0_w = down_ws_raw[e].unsqueeze(0).unsqueeze(0)
        dev1_w = down_ws_raw[e + NUM_EXPERTS_PER_DEVICE].unsqueeze(0).unsqueeze(0)
        stacked_w = torch.cat([dev0_w, dev1_w], dim=0)
        dn_t = ttnn.from_torch(
            stacked_w,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dn_mesh_list.append(dn_t)

    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_buf_list = []
    for _ in range(NUM_EXPERTS_PER_DEVICE):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P, D_padded, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_buf_list.append(ob)
    staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.from_torch(
        torch.zeros(1, 1, P, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Combine metadata
    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())

    def make_combine_meta():
        meta = []
        meta.append(out_buf_addrs[0])
        meta.append(P)
        meta.extend(all_rows)
        meta.extend([w_1_bf16] * P)
        return meta

    per_device_combine_metadata = [make_combine_meta(), make_combine_meta()]

    return {
        "hs_tile": hs_tile,
        "gu_mesh_list": gu_mesh_list,
        "dn_mesh_list": dn_mesh_list,
        "pkt_buf": pkt_buf,
        "inter_buf": inter_buf,
        "out_buf_list": out_buf_list,
        "output": output,
        "combine_metadata": per_device_combine_metadata,
        "hs_rm": hs_rm,
        "staging_buf": staging_buf,
        "dispatch_metadata": dispatch_metadata,
    }


def run_op(mesh_device, tensors):
    """Run prefill_moe_compute and synchronize."""
    result = ttnn.experimental.prefill_moe_compute(
        tensors["hs_tile"],
        gate_up_weights=tensors["gu_mesh_list"],
        down_weights=tensors["dn_mesh_list"],
        pkt_buf=tensors["pkt_buf"],
        inter_buf=tensors["inter_buf"],
        out_bufs=tensors["out_buf_list"],
        output=tensors["output"],
        combine_metadata=tensors["combine_metadata"],
        num_experts=NUM_EXPERTS_PER_DEVICE,
        num_cores=NUM_CORES,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        hidden_states_rm=tensors["hs_rm"],
        staging_buf=tensors["staging_buf"],
        enable_fabric_dispatch=True,
        dispatch_metadata=tensors["dispatch_metadata"],
    )
    ttnn.synchronize_device(mesh_device)
    return result


def bench():
    logger.info("Setting up fabric and opening mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    submesh = full_mesh.create_submesh(
        ttnn.MeshShape(1, 2),
        ttnn.MeshCoordinate(0, 0),
    )
    logger.info(f"Mesh ready: {submesh.get_num_devices()} devices")

    results = {}

    try:
        for P in P_VALUES:
            M_tiles = P // TILE
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking P={P} (M_tiles={M_tiles})")
            logger.info(f"{'='*60}")

            # Create tensors
            tensors = create_tensors_for_p(submesh, P)
            logger.info(f"  Tensors created for P={P}")

            # Warmup (compile kernels)
            logger.info("  Warmup run...")
            run_op(submesh, tensors)
            logger.info("  Warmup complete")

            # Timed runs
            latencies = []
            for i in range(N_ITERS):
                t0 = time.perf_counter()
                run_op(submesh, tensors)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000
                latencies.append(lat_ms)
                logger.info(f"  Run {i+1}/{N_ITERS}: {lat_ms:.3f} ms")

            min_lat = min(latencies)
            med_lat = statistics.median(latencies)
            mean_lat = statistics.mean(latencies)

            results[P] = {
                "M_tiles": M_tiles,
                "min": min_lat,
                "median": med_lat,
                "mean": mean_lat,
                "all": latencies,
            }

            logger.info(f"  P={P}: min={min_lat:.3f}ms  median={med_lat:.3f}ms  mean={mean_lat:.3f}ms")

    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.close_mesh_device(full_mesh)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'P':>6} {'M_tiles':>8} {'Min(ms)':>10} {'Median(ms)':>12} {'Mean(ms)':>10} {'vs P=32':>10}")

    baseline = results.get(32, {}).get("median", 1.0)
    for P in P_VALUES:
        if P in results:
            r = results[P]
            delta = f"+{((r['median'] / baseline) - 1) * 100:.1f}%" if P > 32 else "baseline"
            logger.info(
                f"{P:>6} {r['M_tiles']:>8} {r['min']:>10.3f} {r['median']:>12.3f} {r['mean']:>10.3f} {delta:>10}"
            )

    logger.info(f"\nIdeal scaling: latency ~ M_tiles (linear with tile rows)")
    logger.info(f"P=64 ideal: {baseline * 2:.3f}ms (2x), P=128 ideal: {baseline * 4:.3f}ms (4x)")


if __name__ == "__main__":
    bench()
