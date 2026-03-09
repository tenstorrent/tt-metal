#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fused fabric dispatch + expert compute on 1x4 submesh of TG 6U.

Paired exchange: D0↔D1, D2↔D3 (each pair 1 hop apart).

Each device has 32 tokens and 1 expert.
Routing per pair (symmetric):
    D0: tokens 0-15 local (expert 0), tokens 16-31 sent to D1 (expert 1)
    D1: tokens 48-63 local (expert 1), tokens 32-47 sent to D0 (expert 0)
    D2: tokens 64-79 local (expert 2), tokens 80-95 sent to D3 (expert 3)
    D3: tokens 112-127 local (expert 3), tokens 96-111 sent to D2 (expert 2)

Each expert gets exactly 32 tokens (16 local + 16 from peer).

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_1x4.py
"""

import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
P = 32
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
N_TOKENS_PER_DEVICE = 32
N_DEVICES = 4
N_TOKENS_TOTAL = N_TOKENS_PER_DEVICE * N_DEVICES
NUM_EXPERTS_PER_DEVICE = 1
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def pack_combine_metadata(old_meta, num_experts):
    """Convert old-format combine metadata to packed (row, weight) pair format."""
    packed = []
    idx = 0
    for e in range(num_experts):
        addr = old_meta[idx]
        m_e = old_meta[idx + 1]
        packed.append(addr)
        packed.append(m_e)
        if m_e == 0:
            idx += 2
            continue
        rows = old_meta[idx + 2 : idx + 2 + m_e]
        weights = old_meta[idx + 2 + m_e : idx + 2 + 2 * m_e]
        for i in range(m_e):
            row = rows[i] & 0xFFFF
            w = weights[i] & 0xFFFF
            packed.append(row | (w << 16))
        idx += 2 + 2 * m_e
    return packed


def test_fabric_dispatch_1x4():
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

    # ---- Setup mesh with fabric ----
    logger.info("Setting up fabric and opening full 8x4 mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    logger.info(f"Full mesh opened with {full_mesh.get_num_devices()} devices")

    # Create 1x4 submesh (first four devices in first row)
    submesh = full_mesh.create_submesh(
        ttnn.MeshShape(1, 4),
        ttnn.MeshCoordinate(0, 0),
    )
    logger.info(f"Submesh created with {submesh.get_num_devices()} devices")
    logger.info(f"  submesh device ids: {submesh.get_device_ids()}")

    try:
        _run_test(
            submesh,
            D_tiles,
            n_weight_per_core_gu,
            n_out_per_core,
            cols_per_core,
            half_cols,
            n_tiles_dn,
            n_per_core_dn,
            D_padded,
            D_FF_HALF_padded,
        )
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.close_mesh_device(full_mesh)


def _run_test(
    mesh_device,
    D_tiles,
    n_weight_per_core_gu,
    n_out_per_core,
    cols_per_core,
    half_cols,
    n_tiles_dn,
    n_per_core_dn,
    D_padded,
    D_FF_HALF_padded,
):
    torch.manual_seed(42)

    # ---- Generate data ----
    all_hs = torch.randn(N_TOKENS_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(N_DEVICES)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(N_DEVICES)]

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

    # ---- Routing (paired: D0↔D1, D2↔D3) ----
    # dispatch_target_cols: device i exchanges tokens with device dispatch_target_cols[i]
    dispatch_target_cols = [1, 0, 3, 2]

    # Dispatch metadata: [local_count, recv_count, send_count, local_indices..., send_indices...]
    # Each device: 16 local + 16 recv + 16 send
    # D0: local=rows 0-15, send=rows 16-31 to D1
    # D1: local=rows 16-31, send=rows 0-15 to D0
    # D2: local=rows 0-15, send=rows 16-31 to D3
    # D3: local=rows 16-31, send=rows 0-15 to D2
    dispatch_metadata = [
        [16, 16, 16] + list(range(16)) + list(range(16, 32)),  # D0
        [16, 16, 16] + list(range(16, 32)) + list(range(16)),  # D1
        [16, 16, 16] + list(range(16)) + list(range(16, 32)),  # D2
        [16, 16, 16] + list(range(16, 32)) + list(range(16)),  # D3
    ]

    for i in range(N_DEVICES):
        logger.info(f"Dispatch metadata dev{i}: local=16, recv=16, send=16, target=D{dispatch_target_cols[i]}")

    # ---- Create mesh tensors ----
    # hidden_states_rm (ROW_MAJOR, sharded: each device gets 32 tokens)
    hs_per_dev = []
    for d in range(N_DEVICES):
        start = d * N_TOKENS_PER_DEVICE
        end = start + N_TOKENS_PER_DEVICE
        hs_per_dev.append(all_hs[start:end].unsqueeze(0).unsqueeze(0))  # [1, 1, 32, D]
    stacked_hs = torch.cat(hs_per_dev, dim=0)  # [4, 1, 32, D]
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hidden_states_rm (ROW_MAJOR, sharded to 4 devices)")

    # hidden_states (TILE, replicated - used for shape derivation only)
    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gate_up_weights: sharded (1 expert per device, 4 devices)
    stacked_gu = torch.cat(
        [shuffled_ws[i].unsqueeze(0).unsqueeze(0) for i in range(N_DEVICES)],
        dim=0,
    )  # [4, 1, D, D_FF]
    gu_mesh = ttnn.from_torch(
        stacked_gu,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created gate_up_weights (sharded, 4 experts)")

    # down_weights: sharded
    stacked_dn = torch.cat(
        [down_ws[i].unsqueeze(0).unsqueeze(0) for i in range(N_DEVICES)],
        dim=0,
    )
    dn_mesh = ttnn.from_torch(
        stacked_dn,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Scratch buffers (replicated to all 4 devices)
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
    out_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
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

    logger.info("Created all scratch buffers")

    # ---- Build per-device combine metadata ----
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    out_buf_dev_tensors = ttnn.get_device_tensors(out_buf)
    out_buf_addr = out_buf_dev_tensors[0].buffer_address()
    logger.info(f"out_buf address: {out_buf_addr}")

    per_device_combine_metadata = []
    for d in range(N_DEVICES):
        combine = [out_buf_addr, P]
        combine.extend(list(range(P)))  # rows 0..P-1
        combine.extend([w_1_bf16] * P)  # all weights = 1.0
        per_device_combine_metadata.append(pack_combine_metadata(combine, NUM_EXPERTS_PER_DEVICE))

    # ---- Run the C++ op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with fabric dispatch on 1x4...")
    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=[gu_mesh],
        down_weights=[dn_mesh],
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=[out_buf],
        output=output,
        combine_metadata=per_device_combine_metadata,
        num_experts=1,
        num_cores=NUM_CORES,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        hidden_states_rm=hs_rm,
        staging_buf=staging_buf,
        enable_fabric_dispatch=True,
        dispatch_metadata=dispatch_metadata,
        dispatch_target_cols=dispatch_target_cols,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Op completed, reading back results...")

    # ---- Read back results ----
    result_dev_tensors = ttnn.get_device_tensors(result)
    out_buf_devs = ttnn.get_device_tensors(out_buf)
    gu_devs = ttnn.get_device_tensors(gu_mesh)
    dn_devs = ttnn.get_device_tensors(dn_mesh)
    gu_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_devs]
    dn_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_devs]

    # ---- Verify each device's expert output ----
    # Token mapping per expert (determined by dispatch routing):
    # Expert 0 (D0): local tokens 0-15, recv tokens 32-47 (from D1)
    # Expert 1 (D1): local tokens 48-63, recv tokens 16-31 (from D0)
    # Expert 2 (D2): local tokens 64-79, recv tokens 96-111 (from D3)
    # Expert 3 (D3): local tokens 112-127, recv tokens 80-95 (from D2)
    expert_token_lists = [
        list(range(0, 16)) + list(range(32, 48)),  # Expert 0
        list(range(48, 64)) + list(range(16, 32)),  # Expert 1
        list(range(64, 80)) + list(range(96, 112)),  # Expert 2
        list(range(112, 128)) + list(range(80, 96)),  # Expert 3
    ]

    all_pass = True
    for d in range(N_DEVICES):
        dev_out = ttnn.to_torch(out_buf_devs[d]).squeeze().float()[:, :D]
        dev_result = ttnn.to_torch(result_dev_tensors[d]).squeeze().float()[:, :D]

        # Compute reference for expert d
        expert_tokens = expert_token_lists[d]
        expert_hs = torch.stack([all_hs[t].float() for t in expert_tokens])  # [32, D]

        ref_gu = expert_hs @ gu_dequant[d]
        ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_dequant[d]

        pcc_expert = torch.corrcoef(torch.stack([dev_out.flatten(), ref_out[:, :D].flatten()]))[0, 1].item()
        pcc_combined = torch.corrcoef(torch.stack([dev_result.flatten(), ref_out[:, :D].flatten()]))[0, 1].item()
        logger.info(f"Device {d} (Expert {d}): out_buf PCC={pcc_expert:.6f}, combined PCC={pcc_combined:.6f}")

        if pcc_expert < 0.96:
            logger.error(f"Device {d} out_buf PCC {pcc_expert:.6f} < 0.96")
            all_pass = False
        if pcc_combined < 0.96:
            logger.error(f"Device {d} combined PCC {pcc_combined:.6f} < 0.96")
            all_pass = False

    assert all_pass, "One or more devices failed PCC check"
    logger.info("test_fabric_dispatch_1x4 PASSED — all 4 devices verified")


if __name__ == "__main__":
    test_fabric_dispatch_1x4()
