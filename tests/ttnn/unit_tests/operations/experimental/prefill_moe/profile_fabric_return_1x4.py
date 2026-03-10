#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile prefill_moe_compute with fabric return on 1x4 mesh.

Measures:
  1. Wall-clock op latency (warmup + N iterations)
  2. Device profiler per-kernel breakdown (via ReadDeviceProfiler)

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/profile_fabric_return_1x4.py
"""

import time
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
    try:
        _profile(submesh)
    finally:
        for sm in full_mesh.get_submeshes():
            ttnn.close_mesh_device(sm)
        ttnn.close_mesh_device(full_mesh)


def _profile(mesh_device):
    torch.manual_seed(42)

    # ---- Setup (same as test) ----
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

    # Create tensors
    stacked_hs = torch.stack([h.unsqueeze(0) for h in hs_per_dev])
    hs_tile = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gate_up_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            shuffled_ws[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_tensors.append(t)

    down_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            down_ws_raw[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_tensors.append(t)

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
    out_bufs = []
    for e in range(NUM_EXPERTS):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_bufs.append(ob)

    output = ttnn.from_torch(
        torch.zeros(NUM_EXPERTS, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    total_remote_per_dev = (P - LOCAL_SPLIT) * NUM_EXPERTS
    recv_staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, total_remote_per_dev, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Build metadata
    out_buf_dev_tensors = [ttnn.get_device_tensors(ob) for ob in out_bufs]
    out_buf_addrs = [out_buf_dev_tensors[e][0].buffer_address() for e in range(NUM_EXPERTS)]
    dest_slot_counters = {d: 0 for d in range(N_DEVICES)}

    return_metadata = []
    for d in range(N_DEVICES):
        dev_meta = []
        for e in range(NUM_EXPERTS):
            dev_meta.append(out_buf_addrs[e])
            expert_tokens = list(range(P))
            dev_meta.append(len(expert_tokens))
            for t in expert_tokens:
                src_row = t
                dest_device = get_dest_device(d, t)
                dest_page = e * P + t
                if dest_device == d:
                    recv_slot_id = 0
                else:
                    recv_slot_id = dest_slot_counters[dest_device]
                    dest_slot_counters[dest_device] += 1
                dev_meta.extend([src_row, dest_device, dest_page, recv_slot_id])
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
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"Setup complete. P={P}, D={D}, E={NUM_EXPERTS}, K={K}, devices={N_DEVICES}")
    logger.info(f"Running {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iterations...")

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

    # ---- Warmup ----
    for i in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        run_op()
        t1 = time.perf_counter()
        logger.info(f"Warmup {i}: {(t1-t0)*1000:.2f} ms")

    # ---- Measured iterations ----
    latencies = []
    for i in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        run_op()
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        latencies.append(lat_ms)

    # ---- Report ----
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    median = latencies[len(latencies) // 2]
    p10 = latencies[max(0, len(latencies) // 10)]
    p90 = latencies[min(len(latencies) - 1, 9 * len(latencies) // 10)]

    logger.info("=" * 60)
    logger.info(f"PROFILE RESULTS: prefill_moe_compute 1x4 K={K} P={P} D={D}")
    logger.info(f"  Iterations: {MEASURE_ITERS}")
    logger.info(f"  Avg:    {avg:.2f} ms")
    logger.info(f"  Median: {median:.2f} ms")
    logger.info(f"  P10:    {p10:.2f} ms")
    logger.info(f"  P90:    {p90:.2f} ms")
    logger.info(f"  Min:    {min(latencies):.2f} ms")
    logger.info(f"  Max:    {max(latencies):.2f} ms")
    logger.info(f"  All:    {['%.2f' % l for l in latencies]}")
    logger.info("=" * 60)

    # ---- Device profiler data ----
    try:
        # Read device profiler for per-kernel breakdown
        for dev_tensor in ttnn.get_device_tensors(
            ttnn.experimental.prefill_moe_compute(
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
        ):
            ttnn.synchronize_device(dev_tensor.device())

        perf_data = ttnn.profiler.get_latest_programs_perf_data()
        if perf_data:
            logger.info("Device profiler data:")
            for entry in perf_data:
                logger.info(f"  {entry}")
        else:
            logger.info("No device profiler data available (may need TT_METAL_DEVICE_PROFILER=1)")
    except Exception as e:
        logger.warning(f"Could not read device profiler data: {e}")

    # ---- Theoretical analysis ----
    logger.info("=" * 60)
    logger.info("THEORETICAL ANALYSIS:")
    # Matmul: [32, 2880] x [2880, 5760] = 32 * 2880 * 5760 * 2 FLOPs = 1.06 GFLOP
    flops_gate_up = P * D * D_FF * 2
    # Matmul: [32, 2880] x [2880, 2880] = 32 * 2880 * 2880 * 2 FLOPs = 0.53 GFLOP
    flops_down = P * D_FF_HALF * D * 2
    flops_per_expert = flops_gate_up + flops_down
    total_flops = flops_per_expert * NUM_EXPERTS
    logger.info(
        f"  FLOPs per expert: {flops_per_expert/1e9:.2f} GFLOP (gate_up={flops_gate_up/1e9:.2f} + down={flops_down/1e9:.2f})"
    )
    logger.info(f"  Total FLOPs (4 experts): {total_flops/1e9:.2f} GFLOP")
    # BH peak: ~150 TFLOPS BFP4 matmul at 15 cores
    peak_tflops = 150  # rough estimate
    ideal_compute_ms = total_flops / (peak_tflops * 1e9)
    logger.info(f"  Ideal compute time (at {peak_tflops} TFLOPS): {ideal_compute_ms:.3f} ms")
    # DRAM bandwidth for weight reads
    weight_bytes_gu = D * D_FF * 0.5  # BFP4 = 0.5 bytes per element (approximate)
    weight_bytes_dn = D_FF_HALF * D * 0.5
    total_weight_bytes = (weight_bytes_gu + weight_bytes_dn) * NUM_EXPERTS
    # BH DRAM BW: ~400 GB/s per device
    dram_bw_gbps = 400
    ideal_dram_ms = total_weight_bytes / (dram_bw_gbps * 1e6)
    logger.info(f"  Weight bytes (BFP4): {total_weight_bytes/1e6:.1f} MB")
    logger.info(f"  Ideal DRAM time (at {dram_bw_gbps} GB/s): {ideal_dram_ms:.3f} ms")
    logger.info(f"  Compute intensity: {total_flops / total_weight_bytes:.1f} FLOPs/byte")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
