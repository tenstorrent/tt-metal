#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Clean profiling script — reads device profiler before shutdown."""

import time
import os
import torch
import ttnn
from loguru import logger

TILE = 32
P = 32
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_EXPERTS = 4
K = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
N_DEVICES = 4
LOCAL_SPLIT = 8
GROUP_SIZE = P // N_DEVICES

WARMUP_ITERS = 3
MEASURE_ITERS = 10


def get_dest_device(src_dev, token_idx):
    group = token_idx // GROUP_SIZE
    return (src_dev + group) % N_DEVICES


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    submesh = full_mesh.create_submesh(ttnn.MeshShape(1, N_DEVICES), ttnn.MeshCoordinate(0, 0))

    torch.manual_seed(42)

    # ---- Setup tensors ----
    hs_per_dev = [torch.randn(P, D, dtype=torch.bfloat16) for _ in range(N_DEVICES)]

    raw_weights = torch.randn(P, NUM_EXPERTS, dtype=torch.float32)
    weights_float = torch.softmax(raw_weights, dim=1)
    weights_bf16 = weights_float.bfloat16()

    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE

    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

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

    stacked_hs = torch.stack([h.unsqueeze(0) for h in hs_per_dev])
    hs_tile = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gate_up_tensors = [
        ttnn.from_torch(
            shuffled_ws[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for e in range(NUM_EXPERTS)
    ]
    down_tensors = [
        ttnn.from_torch(
            down_ws_raw[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for e in range(NUM_EXPERTS)
    ]

    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # out_bufs: ROW_MAJOR fragment pages for writer-side untilize (Phase 3)
    n_per_core_dn = (D // TILE) // NUM_CORES  # 6
    out_bufs = [
        ttnn.from_torch(
            torch.zeros(1, 1, P * NUM_CORES, n_per_core_dn * TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for e in range(NUM_EXPERTS)
    ]
    output = ttnn.from_torch(
        torch.zeros(NUM_EXPERTS, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    total_remote_per_dev = (P - LOCAL_SPLIT) * NUM_EXPERTS
    recv_staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, total_remote_per_dev, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_buf_dev_tensors = [ttnn.get_device_tensors(ob) for ob in out_bufs]
    out_buf_addrs = [out_buf_dev_tensors[e][0].buffer_address() for e in range(NUM_EXPERTS)]
    dest_slot_counters = {d: 0 for d in range(N_DEVICES)}

    return_metadata = []
    for d in range(N_DEVICES):
        dev_meta = []
        for e in range(NUM_EXPERTS):
            dev_meta.append(out_buf_addrs[e])
            dev_meta.append(P)
            for t in range(P):
                dest_device = get_dest_device(d, t)
                dest_page = e * P + t
                if dest_device == d:
                    dev_meta.extend([t, dest_device, dest_page, 0])
                else:
                    slot = dest_slot_counters[dest_device]
                    dest_slot_counters[dest_device] += 1
                    dev_meta.extend([t, dest_device, dest_page, slot])
        dev_meta.append(total_remote_per_dev)
        return_metadata.append(dev_meta)

    meta_words_per_dev = [rm[:-1] for rm in return_metadata]
    max_meta_words = max(len(mw) for mw in meta_words_per_dev)
    stacked_int32 = torch.zeros(N_DEVICES, 1, 1, max_meta_words, dtype=torch.int32)
    for d in range(N_DEVICES):
        words = meta_words_per_dev[d]
        stacked_int32[d, 0, 0, : len(words)] = torch.tensor(words, dtype=torch.int32)
    stacked_bf16 = stacked_int32.view(torch.bfloat16)
    meta_tensor = ttnn.from_torch(
        stacked_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"Setup complete. P={P}, D={D}, E={NUM_EXPERTS}, K={K}, devices={N_DEVICES}")

    def run_op():
        result = ttnn.experimental.prefill_moe_compute(
            hs_tile,
            gate_up_weights=gate_up_tensors,
            down_weights=down_tensors,
            pkt_buf=pkt_buf,
            inter_buf=inter_buf,
            out_bufs=out_bufs,
            output=output,
            per_device_combine_metadata=[[] for _ in range(N_DEVICES)],
            num_experts=NUM_EXPERTS,
            num_cores=NUM_CORES,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            dispatch_metadata=[],
            enable_fabric_return=True,
            return_metadata=return_metadata,
            recv_staging_buf=recv_staging_buf,
            return_metadata_tensor=meta_tensor,
        )
        for dev_tensor in ttnn.get_device_tensors(result):
            ttnn.synchronize_device(dev_tensor.device())
        return result

    # Warmup
    for i in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        run_op()
        t1 = time.perf_counter()
        logger.info(f"Warmup {i}: {(t1-t0)*1000:.2f} ms")

    # Measured
    latencies = []
    for i in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        run_op()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies.sort()
    avg = sum(latencies) / len(latencies)
    median = latencies[len(latencies) // 2]

    logger.info("=" * 60)
    logger.info(f"WALL-CLOCK RESULTS: 1x4 K={K} P={P} D={D}")
    logger.info(f"  Avg:    {avg:.2f} ms")
    logger.info(f"  Median: {median:.2f} ms")
    logger.info(f"  Min:    {min(latencies):.2f} ms")
    logger.info(f"  Max:    {max(latencies):.2f} ms")
    logger.info(f"  All:    {['%.2f' % l for l in latencies]}")
    logger.info("=" * 60)

    # Read device profiler BEFORE closing mesh
    logger.info("Reading device profiler data...")
    try:
        for dev_tensor in ttnn.get_device_tensors(hs_tile):
            device = dev_tensor.device()
            ttnn.ReadDeviceProfiler(device)
            logger.info(f"  ReadDeviceProfiler done for device {device.id()}")
    except Exception as e:
        logger.warning(f"ReadDeviceProfiler failed: {e}")

    # Check for profiler artifacts
    profiler_dir = "/data/sraizada_2/tt-metal/generated/profiler/.logs"
    if os.path.exists(profiler_dir):
        for f in os.listdir(profiler_dir):
            fpath = os.path.join(profiler_dir, f)
            size = os.path.getsize(fpath)
            logger.info(f"  Profiler artifact: {f} ({size} bytes)")

    # Cleanup - close submeshes before parent mesh (matches official fixture pattern)
    logger.info("Cleaning up...")
    del hs_tile, gate_up_tensors, down_tensors, pkt_buf, inter_buf, out_bufs, output
    del recv_staging_buf, meta_tensor, out_buf_dev_tensors

    for sm in full_mesh.get_submeshes():
        ttnn.close_mesh_device(sm)
    ttnn.close_mesh_device(full_mesh)
    logger.info("Done.")


if __name__ == "__main__":
    main()
