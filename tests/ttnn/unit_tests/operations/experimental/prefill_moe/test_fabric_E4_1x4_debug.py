#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Debug test for 1x4 fabric dispatch — isolates metadata parsing vs fabric traffic.

Phase 1: No fabric traffic (send_dest_count=0, recv_device_count=0).
          Tests that the metadata parsing and pkt_buf assembly work.
Phase 2: Full fabric traffic (if Phase 1 passes).
"""

import sys
import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
NUM_DEVICES = 4
E_LOCAL = 4
E_TOTAL = NUM_DEVICES * E_LOCAL  # 16
K = 1
M_PER_DEVICE = 128
M_TOTAL = NUM_DEVICES * M_PER_DEVICE  # 512
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3

M_PER_EXPERT = M_TOTAL // E_TOTAL  # 32
M_PADDED = M_PER_EXPERT
M_PADDED_TILES = M_PADDED // TILE


def build_routing():
    expert_assignments = {e: [] for e in range(E_TOTAL)}
    for t in range(M_TOTAL):
        expert_assignments[t % E_TOTAL].append(t)

    per_device_send = {}
    for src_dev in range(NUM_DEVICES):
        sends = {dst: [] for dst in range(NUM_DEVICES) if dst != src_dev}
        for local_idx in range(M_PER_DEVICE):
            global_idx = src_dev * M_PER_DEVICE + local_idx
            expert_id = global_idx % E_TOTAL
            dst_dev = expert_id // E_LOCAL
            if dst_dev != src_dev:
                sends[dst_dev].append(local_idx)
        per_device_send[src_dev] = sends

    per_device_recv_offsets = {}
    for recv_dev in range(NUM_DEVICES):
        senders = sorted([d for d in range(NUM_DEVICES) if d != recv_dev])
        offsets = {}
        offset = 0
        for sender in senders:
            offsets[sender] = offset
            offset += len(per_device_send[sender][recv_dev])
        per_device_recv_offsets[recv_dev] = offsets

    return expert_assignments, per_device_send, per_device_recv_offsets


def build_multi_dest_dispatch_metadata(per_device_send, per_device_recv_offsets):
    metadata = []
    for dev in range(NUM_DEVICES):
        md = []
        senders = [d for d in range(NUM_DEVICES) if d != dev and len(per_device_send[d][dev]) > 0]
        md.append(len(senders))
        dests = sorted([d for d in per_device_send[dev] if len(per_device_send[dev][d]) > 0])
        md.append(len(dests))
        for dst in dests:
            if dst > dev:
                direction = 0
                hops = dst - dev
            else:
                direction = 1
                hops = dev - dst
            remote_offset = per_device_recv_offsets[dst][dev]
            indices = per_device_send[dev][dst]
            md.append(direction)
            md.append(hops)
            md.append(remote_offset)
            md.append(len(indices))
            md.extend(indices)
        metadata.append(md)
    return metadata


def build_no_traffic_metadata():
    """Build metadata with no fabric traffic — tests parsing only."""
    metadata = []
    for dev in range(NUM_DEVICES):
        md = [0, 0]  # recv_device_count=0, send_dest_count=0
        metadata.append(md)
    return metadata


def build_per_expert_dispatch_sources_local_only(expert_assignments):
    """Build expert sources using only local tokens (no staging_buf references)."""
    sources = []
    for dev in range(NUM_DEVICES):
        src = [E_LOCAL]
        for local_e in range(E_LOCAL):
            global_expert = dev * E_LOCAL + local_e
            global_tokens = expert_assignments[global_expert]
            # Only use local tokens
            expert_sources = []
            for global_t in global_tokens:
                src_dev = global_t // M_PER_DEVICE
                if src_dev == dev:
                    local_t = global_t % M_PER_DEVICE
                    expert_sources.append(local_t)
            # Pad to M_PER_EXPERT with zeros (repeated first token)
            while len(expert_sources) < M_PER_EXPERT:
                expert_sources.append(expert_sources[0] if expert_sources else 0)
            src.append(len(expert_sources))
            src.extend(expert_sources)
        sources.append(src)
    return sources


def build_per_expert_dispatch_sources(expert_assignments, per_device_send, per_device_recv_offsets):
    sources = []
    for dev in range(NUM_DEVICES):
        src = [E_LOCAL]
        for local_e in range(E_LOCAL):
            global_expert = dev * E_LOCAL + local_e
            global_tokens = expert_assignments[global_expert]
            expert_sources = []
            for global_t in global_tokens:
                src_dev = global_t // M_PER_DEVICE
                local_t = global_t % M_PER_DEVICE
                if src_dev == dev:
                    expert_sources.append(local_t)
                else:
                    recv_offset = per_device_recv_offsets[dev][src_dev]
                    sent_indices = per_device_send[src_dev][dev]
                    pos_in_block = sent_indices.index(local_t)
                    staging_row = recv_offset + pos_in_block
                    expert_sources.append((1 << 31) | staging_row)
            src.append(len(expert_sources))
            src.extend(expert_sources)
        sources.append(src)
    return sources


def run_debug_test():
    D_tiles = D // TILE
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    n_tiles_dn = D // TILE
    n_per_core_dn = n_tiles_dn // NUM_CORES
    D_padded = n_tiles_dn * TILE
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE
    k_tiles_dn = D_FF_HALF_padded // TILE

    expert_assignments, per_device_send, per_device_recv_offsets = build_routing()

    # Log routing info
    for dev in range(NUM_DEVICES):
        total_send = sum(len(per_device_send[dev][d]) for d in per_device_send[dev])
        senders = [d for d in range(NUM_DEVICES) if d != dev and len(per_device_send[d][dev]) > 0]
        total_recv = sum(len(per_device_send[s][dev]) for s in senders)
        logger.info(f"Device {dev}: send={total_send}, recv={total_recv}")

    # Build metadata for both phases
    no_traffic_meta = build_no_traffic_metadata()
    local_sources = build_per_expert_dispatch_sources_local_only(expert_assignments)
    full_meta = build_multi_dest_dispatch_metadata(per_device_send, per_device_recv_offsets)
    full_sources = build_per_expert_dispatch_sources(expert_assignments, per_device_send, per_device_recv_offsets)

    # Validate metadata sizes
    for dev in range(NUM_DEVICES):
        logger.info(
            f"Device {dev}: no_traffic_meta len={len(no_traffic_meta[dev])}, "
            f"local_sources len={len(local_sources[dev])}, "
            f"full_meta len={len(full_meta[dev])}, "
            f"full_sources len={len(full_sources[dev])}"
        )

    # Check for large values in full_sources (bit31 set)
    for dev in range(NUM_DEVICES):
        for val in full_sources[dev]:
            if val >= (1 << 31):
                staging_row = val & 0x7FFFFFFF
                if staging_row >= 96:
                    logger.error(f"INVALID staging_row {staging_row} on device {dev}")
                    sys.exit(1)

    logger.info("Metadata validation passed")

    # Dump metadata for device 0
    logger.info(f"Device 0 full_meta: {full_meta[0][:20]}... (len={len(full_meta[0])})")
    logger.info(f"Device 0 full_sources: {full_sources[0][:20]}... (len={len(full_sources[0])})")

    # ---- Setup mesh ----
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    logger.info(f"Full mesh opened with {full_mesh.get_num_devices()} devices")

    submesh = full_mesh.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    logger.info(f"Submesh: {submesh.get_num_devices()} devices, ids={submesh.get_device_ids()}")

    try:
        _run_phase(
            submesh,
            expert_assignments,
            per_device_send,
            per_device_recv_offsets,
            no_traffic_meta,
            local_sources,
            full_meta,
            full_sources,
            D_tiles,
            n_weight_per_core_gu,
            n_out_per_core,
            cols_per_core,
            half_cols,
            n_tiles_dn,
            n_per_core_dn,
            D_padded,
            D_FF_HALF_padded,
            k_tiles_dn,
        )
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.close_mesh_device(full_mesh)


def _run_phase(
    mesh_device,
    expert_assignments,
    per_device_send,
    per_device_recv_offsets,
    no_traffic_meta,
    local_sources,
    full_meta,
    full_sources,
    D_tiles,
    n_weight_per_core_gu,
    n_out_per_core,
    cols_per_core,
    half_cols,
    n_tiles_dn,
    n_per_core_dn,
    D_padded,
    D_FF_HALF_padded,
    k_tiles_dn,
):
    torch.manual_seed(42)

    all_hs = torch.randn(M_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(E_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(E_TOTAL)]

    # Pre-shuffle gate_up weights
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

    dispatch_metadata = [[0, 0, 0] for _ in range(NUM_DEVICES)]

    # Create tensors
    stacked_hs = torch.cat(
        [all_hs[d * M_PER_DEVICE : (d + 1) * M_PER_DEVICE].unsqueeze(0).unsqueeze(0) for d in range(NUM_DEVICES)]
    )
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hs_rm")

    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gu_list = []
    for local_e in range(E_LOCAL):
        stacked_gu = torch.cat(
            [shuffled_ws[d * E_LOCAL + local_e].unsqueeze(0).unsqueeze(0) for d in range(NUM_DEVICES)]
        )
        gu_mesh = ttnn.from_torch(
            stacked_gu,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gu_list.append(gu_mesh)
    logger.info("Created gate_up weights")

    dn_list = []
    for local_e in range(E_LOCAL):
        stacked_dn = torch.cat(
            [down_ws_raw[d * E_LOCAL + local_e].unsqueeze(0).unsqueeze(0) for d in range(NUM_DEVICES)]
        )
        dn_mesh = ttnn.from_torch(
            stacked_dn,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dn_list.append(dn_mesh)

    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, E_LOCAL * M_PADDED, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_buf_list = []
    for _ in range(E_LOCAL):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, M_PADDED, D_padded, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_buf_list.append(ob)

    max_recv = max(
        sum(len(per_device_send[s][dev]) for s in range(NUM_DEVICES) if s != dev) for dev in range(NUM_DEVICES)
    )
    staging_rows = ((max_recv + TILE - 1) // TILE) * TILE
    staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, staging_rows, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("Created all buffers")

    # Combine metadata
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    out_buf_addrs = []
    for e in range(E_LOCAL):
        devs = ttnn.get_device_tensors(out_buf_list[e])
        out_buf_addrs.append(devs[0].buffer_address())

    per_device_combine_metadata = []
    for dev in range(NUM_DEVICES):
        cm = []
        for local_e in range(E_LOCAL):
            cm.append(out_buf_addrs[local_e])
            cm.append(M_PER_EXPERT)
            # Pack (row, weight) into single uint32: low16=row, high16=weight
            for r in range(M_PER_EXPERT):
                cm.append(r | (w_1_bf16 << 16))
        per_device_combine_metadata.append(cm)

    # ====== PHASE 1: No fabric traffic ======
    logger.info("=" * 60)
    logger.info("PHASE 1: No fabric traffic (send=0, recv=0)")
    logger.info("=" * 60)

    try:
        result = ttnn.experimental.prefill_moe_compute(
            hs_tile,
            gate_up_weights=gu_list,
            down_weights=dn_list,
            pkt_buf=pkt_buf,
            inter_buf=inter_buf,
            out_bufs=out_buf_list,
            output=output,
            combine_metadata=per_device_combine_metadata,
            num_experts=E_LOCAL,
            num_cores=NUM_CORES,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            hidden_states_rm=hs_rm,
            staging_buf=staging_buf,
            enable_fabric_dispatch=True,
            dispatch_metadata=dispatch_metadata,
            per_expert_dispatch_sources=local_sources,
            multi_dest_dispatch_metadata=no_traffic_meta,
            enable_fpu_combine=True,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info("PHASE 1 PASSED — op completed without fabric traffic")
    except Exception as e:
        logger.error(f"PHASE 1 FAILED with exception: {e}")
        return

    # ====== PHASE 2: Full fabric traffic ======
    logger.info("=" * 60)
    logger.info("PHASE 2: Full fabric traffic (3 dests, 3 senders per device)")
    logger.info("=" * 60)

    # Re-create zero output
    output2 = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    try:
        result2 = ttnn.experimental.prefill_moe_compute(
            hs_tile,
            gate_up_weights=gu_list,
            down_weights=dn_list,
            pkt_buf=pkt_buf,
            inter_buf=inter_buf,
            out_bufs=out_buf_list,
            output=output2,
            combine_metadata=per_device_combine_metadata,
            num_experts=E_LOCAL,
            num_cores=NUM_CORES,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            hidden_states_rm=hs_rm,
            staging_buf=staging_buf,
            enable_fabric_dispatch=True,
            dispatch_metadata=dispatch_metadata,
            per_expert_dispatch_sources=full_sources,
            multi_dest_dispatch_metadata=full_meta,
            enable_fpu_combine=True,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info("PHASE 2 PASSED — op completed with full fabric traffic")
    except Exception as e:
        logger.error(f"PHASE 2 FAILED with exception: {e}")


if __name__ == "__main__":
    run_debug_test()
