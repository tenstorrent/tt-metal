#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fused fabric dispatch + expert compute on 1x4 submesh of TG 6U.

Each device has 4 experts (E_local=4), 16 experts total across 4 devices.
128 tokens per device, 512 total. K=1, uniform routing: token t -> expert (t % 16).

Each device sends 32 tokens to each of 3 remote devices (96 sent).
Each device receives 32 tokens from each of 3 remote senders (96 received).
staging_buf: 96 rows per device, partitioned by sender.

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_E4_1x4.py
"""

import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
NUM_DEVICES = 4
E_LOCAL = 4  # experts per device
E_TOTAL = NUM_DEVICES * E_LOCAL  # 16
K = 1
M_PER_DEVICE = 128  # tokens per device
M_TOTAL = NUM_DEVICES * M_PER_DEVICE  # 512
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3

# Derived
M_PER_EXPERT = M_TOTAL // E_TOTAL  # 32 tokens per expert with uniform routing
M_PADDED = M_PER_EXPERT  # already tile-aligned (32)
M_PADDED_TILES = M_PADDED // TILE  # 1


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def build_routing():
    """Build the token routing tables for uniform routing.

    Returns:
        expert_assignments: dict[int, list[int]]
            Maps global_expert_id -> list of global token indices
        per_device_send: dict[int, dict[int, list[int]]]
            Maps src_device -> {dst_device -> list of local token indices to send}
        per_device_recv_offsets: dict[int, dict[int, int]]
            Maps recv_device -> {src_device -> starting row in staging_buf}
    """
    # Assign tokens to experts: global_token_idx % E_TOTAL
    expert_assignments = {e: [] for e in range(E_TOTAL)}
    for t in range(M_TOTAL):
        expert_assignments[t % E_TOTAL].append(t)

    # Per-device send tables: which local tokens go to which remote device
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

    # Per-device recv offsets: where each sender's tokens land in staging_buf
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
    """Build multi_dest_dispatch_metadata for each device.

    Returns list of N flat vectors, one per device:
      [recv_device_count, send_dest_count,
       {direction, hops, staging_offset, count, indices...} x send_dest_count]
    """
    metadata = []
    for dev in range(NUM_DEVICES):
        md = []
        # recv_device_count = number of devices sending to us
        senders = [d for d in range(NUM_DEVICES) if d != dev and len(per_device_send[d][dev]) > 0]
        md.append(len(senders))

        # send_dest_count = number of devices we send to
        dests = sorted([d for d in per_device_send[dev] if len(per_device_send[dev][d]) > 0])
        md.append(len(dests))

        for dst in dests:
            # Direction and hops on 1xN mesh (row 0)
            if dst > dev:
                direction = 0  # EAST
                hops = dst - dev
            else:
                direction = 1  # WEST
                hops = dev - dst

            # Remote staging offset: where our tokens land in dst's staging_buf
            remote_offset = per_device_recv_offsets[dst][dev]

            indices = per_device_send[dev][dst]
            md.append(direction)
            md.append(hops)
            md.append(remote_offset)
            md.append(len(indices))
            md.extend(indices)

        metadata.append(md)
    return metadata


def build_per_expert_dispatch_sources(expert_assignments, per_device_send, per_device_recv_offsets):
    """Build per_expert_dispatch_sources for each device.

    Returns list of N flat vectors, one per device:
      [num_experts, M_0, source_0_0..., M_1, source_1_0..., ...]
    Each source: bit31=is_recv (0=local hs_rm, 1=staging_buf), bits0-30=row index.
    """
    sources = []
    for dev in range(NUM_DEVICES):
        src = [E_LOCAL]  # num_experts

        for local_e in range(E_LOCAL):
            global_expert = dev * E_LOCAL + local_e
            global_tokens = expert_assignments[global_expert]

            expert_sources = []
            for global_t in global_tokens:
                src_dev = global_t // M_PER_DEVICE
                local_t = global_t % M_PER_DEVICE

                if src_dev == dev:
                    # Local token: bit31=0, row = local_t in hs_rm
                    expert_sources.append(local_t)
                else:
                    # Remote token: bit31=1, row = staging_buf position
                    # Find which row in staging_buf this token occupies
                    recv_offset = per_device_recv_offsets[dev][src_dev]
                    # Find position within sender's block
                    sent_indices = per_device_send[src_dev][dev]
                    pos_in_block = sent_indices.index(local_t)
                    staging_row = recv_offset + pos_in_block
                    expert_sources.append((1 << 31) | staging_row)

            src.append(len(expert_sources))  # M_e
            src.extend(expert_sources)

        sources.append(src)
    return sources


def test_fabric_E4_1x4():
    # ---- Tile dimension calculations ----
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

    # ---- Build routing ----
    expert_assignments, per_device_send, per_device_recv_offsets = build_routing()
    multi_dest_metadata = build_multi_dest_dispatch_metadata(per_device_send, per_device_recv_offsets)
    expert_sources = build_per_expert_dispatch_sources(expert_assignments, per_device_send, per_device_recv_offsets)

    # Log routing summary
    for dev in range(NUM_DEVICES):
        total_send = sum(len(per_device_send[dev][d]) for d in per_device_send[dev])
        senders = [d for d in range(NUM_DEVICES) if d != dev and len(per_device_send[d][dev]) > 0]
        total_recv = sum(len(per_device_send[s][dev]) for s in senders)
        logger.info(
            f"Device {dev}: send={total_send} tokens to {len(per_device_send[dev])} dests, "
            f"recv={total_recv} tokens from {len(senders)} senders"
        )

    # ---- Setup mesh with fabric ----
    logger.info("Setting up fabric and opening full 8x4 mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    logger.info(f"Full mesh opened with {full_mesh.get_num_devices()} devices")

    submesh = full_mesh.create_submesh(
        ttnn.MeshShape(1, 4),
        ttnn.MeshCoordinate(0, 0),
    )
    logger.info(f"Submesh created with {submesh.get_num_devices()} devices")
    logger.info(f"  submesh device ids: {submesh.get_device_ids()}")

    try:
        _run_test(
            submesh,
            expert_assignments,
            per_device_send,
            per_device_recv_offsets,
            multi_dest_metadata,
            expert_sources,
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


def _run_test(
    mesh_device,
    expert_assignments,
    per_device_send,
    per_device_recv_offsets,
    multi_dest_metadata,
    expert_sources_meta,
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

    # ---- Generate data ----
    all_hs = torch.randn(M_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(E_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(E_TOTAL)]

    # Pre-shuffle gate_up weights for per-core SwiGLU
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

    # ---- Build dispatch_metadata (simple format for backward compat path) ----
    # We still need dispatch_metadata for the factory's validation check.
    # Use a minimal format: [0, 0, 0] (no local/recv/send) since multi_dest overrides it.
    dispatch_metadata = [[0, 0, 0] for _ in range(NUM_DEVICES)]

    # ---- Create mesh tensors ----
    # hidden_states_rm (ROW_MAJOR, sharded: each device gets its M_PER_DEVICE tokens)
    stacked_hs = torch.cat(
        [all_hs[d * M_PER_DEVICE : (d + 1) * M_PER_DEVICE].unsqueeze(0).unsqueeze(0) for d in range(NUM_DEVICES)]
    )  # [4, 1, 128, D]
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hidden_states_rm (ROW_MAJOR, sharded)")

    # hidden_states (TILE, replicated — used for shape derivation only)
    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gate_up_weights: list of 4 mesh tensors, one per local expert index
    # gate_up_weights[local_e] is sharded: dev d gets global expert (d*E_LOCAL + local_e)
    gu_list = []
    for local_e in range(E_LOCAL):
        stacked_gu = torch.cat(
            [shuffled_ws[d * E_LOCAL + local_e].unsqueeze(0).unsqueeze(0) for d in range(NUM_DEVICES)]
        )  # [4, 1, D, D_FF]
        gu_mesh = ttnn.from_torch(
            stacked_gu,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gu_list.append(gu_mesh)
    logger.info(f"Created {E_LOCAL} gate_up_weight mesh tensors (sharded per expert)")

    # down_weights: same structure
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

    # pkt_buf: [1, 1, E_LOCAL*M_PADDED, D] = [1,1,128,2880] — one region per local expert
    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, E_LOCAL * M_PADDED, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # inter_buf: [1, 1, M_PADDED, D_FF_HALF_padded]
    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # out_bufs: one per local expert
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

    # staging_buf: needs to hold max received tokens
    max_recv = max(
        sum(len(per_device_send[s][dev]) for s in range(NUM_DEVICES) if s != dev) for dev in range(NUM_DEVICES)
    )
    staging_rows = ((max_recv + TILE - 1) // TILE) * TILE  # pad to tile boundary
    staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, staging_rows, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # output: [1, 1, M_PADDED, D_padded]
    output = ttnn.from_torch(
        torch.zeros(1, 1, M_PADDED, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("Created all scratch buffers")

    # ---- Build per-device combine metadata ----
    # Each device has 4 experts, each with M_PER_EXPERT tokens, weight=1.0
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    out_buf_dev_tensors_0 = ttnn.get_device_tensors(out_buf_list[0])
    out_buf_addrs = []
    for e in range(E_LOCAL):
        devs = ttnn.get_device_tensors(out_buf_list[e])
        out_buf_addrs.append(devs[0].buffer_address())

    per_device_combine_metadata = []
    for dev in range(NUM_DEVICES):
        cm = []
        for local_e in range(E_LOCAL):
            cm.append(out_buf_addrs[local_e])  # out_buf_addr
            cm.append(M_PER_EXPERT)  # M_e
            # Pack (row, weight) into single uint32: low16=row, high16=weight
            for r in range(M_PER_EXPERT):
                cm.append(r | (w_1_bf16 << 16))
        per_device_combine_metadata.append(cm)

    # ---- Run the C++ op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with multi-dest fabric dispatch...")
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
        per_expert_dispatch_sources=expert_sources_meta,
        multi_dest_dispatch_metadata=multi_dest_metadata,
        enable_fpu_combine=True,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Op completed, reading back results...")

    # ---- Read back results ----
    result_dev_tensors = ttnn.get_device_tensors(result)

    # Read dequantized weights for reference
    gu_devs = [ttnn.get_device_tensors(gu) for gu in gu_list]
    dn_devs = [ttnn.get_device_tensors(dn) for dn in dn_list]

    all_pass = True
    for dev in range(NUM_DEVICES):
        dev_result = ttnn.to_torch(result_dev_tensors[dev]).squeeze().float()[:, :D]

        # Read per-expert out_bufs
        for local_e in range(E_LOCAL):
            global_expert = dev * E_LOCAL + local_e
            global_tokens = expert_assignments[global_expert]

            # Get expert's input hidden states
            expert_hs = torch.stack([all_hs[t].float() for t in global_tokens])  # [M_PER_EXPERT, D]

            # Get dequantized weights for this expert on this device
            gu_dq = ttnn.to_torch(gu_devs[local_e][dev]).squeeze().float()
            dn_dq = ttnn.to_torch(dn_devs[local_e][dev]).squeeze().float()

            # Compute reference
            ref_gu = expert_hs @ gu_dq
            ref_inter = torch.empty(M_PER_EXPERT, D_FF_HALF_padded, dtype=torch.float32)
            for c in range(NUM_CORES):
                g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
                u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
                ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
            ref_out = ref_inter.bfloat16().float() @ dn_dq

            # Read actual out_buf for this expert on this device
            ob_devs = ttnn.get_device_tensors(out_buf_list[local_e])
            dev_expert_out = ttnn.to_torch(ob_devs[dev]).squeeze().float()[:M_PER_EXPERT, :D]

            pcc_expert = torch.corrcoef(torch.stack([dev_expert_out.flatten(), ref_out[:, :D].flatten()]))[0, 1].item()

            logger.info(f"Device {dev}, Expert {global_expert} (local {local_e}): out_buf PCC={pcc_expert:.6f}")

            if pcc_expert < 0.96:
                logger.error(f"FAIL: Device {dev} Expert {global_expert} PCC {pcc_expert:.6f} < 0.96")
                all_pass = False

    # Note: combined output check is more complex with K=1 and multi-expert.
    # With K=1, each token's output comes from exactly one expert.
    # The combine kernel accumulates per-expert results into the output tensor.
    # Since we use enable_fpu_combine and weight=1.0, the output should equal
    # the sum of all expert out_bufs (but since K=1, each output row gets
    # contributions from exactly one expert at each device).

    assert all_pass, "Some expert PCC checks failed"
    logger.info("test_fabric_E4_1x4 PASSED")


if __name__ == "__main__":
    test_fabric_E4_1x4()
