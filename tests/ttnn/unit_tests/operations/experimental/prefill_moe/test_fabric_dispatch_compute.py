#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fused fabric dispatch + expert compute on 1x2 submesh of TG 6U.

Each device has 32 tokens and 1 expert.
Routing: tokens 0-15 -> expert 0 (device 0 local)
         tokens 16-31 -> expert 1 (device 0 sends to device 1)
         tokens 32-47 -> expert 0 (device 1 sends to device 0)
         tokens 48-63 -> expert 1 (device 1 local)

Each expert gets exactly 32 tokens (16 local + 16 from peer).

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_compute.py
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
N_TOKENS_TOTAL = 64
NUM_EXPERTS_PER_DEVICE = 1
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
BF16_TILE_BYTES = 2048


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def _compute_n_block(n_per_core, max_block=7):
    for b in range(min(n_per_core, max_block), 0, -1):
        if n_per_core % b == 0:
            return b
    return 1


def test_fabric_dispatch_compute():
    # ---- Tile dimension calculations ----
    D_tiles = D // TILE
    k_tiles_gu = D_tiles
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

    # ---- Setup mesh with fabric ----
    # On TG 6U, we must open all 32 devices for fabric to initialize.
    # Then create a 1x2 submesh for our 2-device test.
    logger.info("Setting up fabric and opening full 8x4 mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    logger.info(f"Full mesh opened with {full_mesh.get_num_devices()} devices")

    # Create 1x2 submesh (first two devices in first row)
    submesh = full_mesh.create_submesh(
        ttnn.MeshShape(1, 2),
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
            k_tiles_dn,
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
    k_tiles_dn,
):
    torch.manual_seed(42)

    # ---- Generate data ----
    all_hs = torch.randn(N_TOKENS_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(2)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(2)]

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

    # ---- Routing ----
    # Device 0 tokens (0-31): 0-15 -> expert 0 (local), 16-31 -> expert 1 (send)
    # Device 1 tokens (32-63): 32-47 -> expert 0 (send), 48-63 -> expert 1 (local)
    #
    # Expert 0 pkt_buf (on device 0): rows 0-15 = tokens 0-15, rows 16-31 = tokens 32-47
    # Expert 1 pkt_buf (on device 1): rows 0-15 = tokens 48-63, rows 16-31 = tokens 16-31

    # Dispatch metadata: [local_count, recv_count, send_count, local_indices..., send_indices...]
    dev0_dispatch = [16, 16, 16] + list(range(16)) + list(range(16, 32))
    dev1_dispatch = [16, 16, 16] + list(range(16, 32)) + list(range(16))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]

    logger.info(f"Dispatch metadata dev0: local={16}, recv={16}, send={16}")
    logger.info(f"Dispatch metadata dev1: local={16}, recv={16}, send={16}")

    # ---- Create mesh tensors ----
    # hidden_states_rm (ROW_MAJOR, sharded: dev0 gets tokens 0-31, dev1 gets tokens 32-63)
    hs_dev0 = all_hs[:32].unsqueeze(0).unsqueeze(0)  # [1, 1, 32, D]
    hs_dev1 = all_hs[32:].unsqueeze(0).unsqueeze(0)
    stacked_hs = torch.cat([hs_dev0, hs_dev1], dim=0)  # [2, 1, 32, D]
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hidden_states_rm (ROW_MAJOR, sharded)")

    # hidden_states (TILE, replicated - used for shape derivation only)
    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gate_up_weights: sharded (expert 0 on dev0, expert 1 on dev1)
    stacked_gu = torch.cat(
        [
            shuffled_ws[0].unsqueeze(0).unsqueeze(0),
            shuffled_ws[1].unsqueeze(0).unsqueeze(0),
        ],
        dim=0,
    )  # [2, 1, D, D_FF]
    gu_mesh = ttnn.from_torch(
        stacked_gu,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created gate_up_weights (sharded per expert)")

    # down_weights: sharded
    stacked_dn = torch.cat(
        [
            down_ws[0].unsqueeze(0).unsqueeze(0),
            down_ws[1].unsqueeze(0).unsqueeze(0),
        ],
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

    # Scratch buffers (replicated)
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
    # Each device: 1 expert, 32 tokens, weight = 1.0
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    # Get out_buf address (same on all devices due to deterministic allocation)
    out_buf_dev_tensors = ttnn.get_device_tensors(out_buf)
    out_buf_addr = out_buf_dev_tensors[0].buffer_address()
    logger.info(f"out_buf address: {out_buf_addr}")

    dev0_combine = [out_buf_addr, 32]
    dev0_combine.extend(list(range(32)))  # token rows 0-31
    dev0_combine.extend([w_1_bf16] * 32)  # all weights = 1.0

    dev1_combine = [out_buf_addr, 32]
    dev1_combine.extend(list(range(32)))
    dev1_combine.extend([w_1_bf16] * 32)

    per_device_combine_metadata = [dev0_combine, dev1_combine]

    # ---- Run the C++ op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with fabric dispatch...")
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
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Op completed, reading back results...")

    # ---- Read back results ----
    result_dev_tensors = ttnn.get_device_tensors(result)
    dev0_result = ttnn.to_torch(result_dev_tensors[0]).squeeze().float()[:, :D]
    dev1_result = ttnn.to_torch(result_dev_tensors[1]).squeeze().float()[:, :D]

    # Also read back out_bufs for per-expert PCC
    out_buf_devs = ttnn.get_device_tensors(out_buf)
    dev0_out = ttnn.to_torch(out_buf_devs[0]).squeeze().float()[:, :D]
    dev1_out = ttnn.to_torch(out_buf_devs[1]).squeeze().float()[:, :D]

    # Read dequantized weights for reference
    gu_devs = ttnn.get_device_tensors(gu_mesh)
    dn_devs = ttnn.get_device_tensors(dn_mesh)
    gu_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_devs]
    dn_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_devs]

    # ---- Compute reference ----
    # Device 0 (expert 0): pkt_buf rows 0-15 = tokens 0-15, rows 16-31 = tokens 32-47
    expert0_tokens = list(range(16)) + list(range(32, 48))
    expert0_hs = torch.stack([all_hs[t].float() for t in expert0_tokens])  # [32, D]

    ref_gu0 = expert0_hs @ gu_dequant[0]
    ref_inter0 = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
    for c in range(NUM_CORES):
        g = ref_gu0[:, c * cols_per_core : c * cols_per_core + half_cols]
        u = ref_gu0[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_inter0[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
    ref_out0 = ref_inter0.bfloat16().float() @ dn_dequant[0]

    pcc0_expert = torch.corrcoef(torch.stack([dev0_out.flatten(), ref_out0[:, :D].flatten()]))[0, 1].item()
    pcc0_combined = torch.corrcoef(torch.stack([dev0_result.flatten(), ref_out0[:, :D].flatten()]))[0, 1].item()
    logger.info(f"Device 0 (Expert 0): out_buf PCC={pcc0_expert:.6f}, combined PCC={pcc0_combined:.6f}")

    # Device 1 (expert 1): pkt_buf rows 0-15 = tokens 48-63, rows 16-31 = tokens 16-31
    expert1_tokens = list(range(48, 64)) + list(range(16, 32))
    expert1_hs = torch.stack([all_hs[t].float() for t in expert1_tokens])  # [32, D]

    ref_gu1 = expert1_hs @ gu_dequant[1]
    ref_inter1 = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
    for c in range(NUM_CORES):
        g = ref_gu1[:, c * cols_per_core : c * cols_per_core + half_cols]
        u = ref_gu1[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_inter1[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
    ref_out1 = ref_inter1.bfloat16().float() @ dn_dequant[1]

    pcc1_expert = torch.corrcoef(torch.stack([dev1_out.flatten(), ref_out1[:, :D].flatten()]))[0, 1].item()
    pcc1_combined = torch.corrcoef(torch.stack([dev1_result.flatten(), ref_out1[:, :D].flatten()]))[0, 1].item()
    logger.info(f"Device 1 (Expert 1): out_buf PCC={pcc1_expert:.6f}, combined PCC={pcc1_combined:.6f}")

    # ---- Assertions ----
    assert pcc0_combined >= 0.96, f"Device 0 combined PCC {pcc0_combined:.6f} < 0.96"
    assert pcc1_combined >= 0.96, f"Device 1 combined PCC {pcc1_combined:.6f} < 0.96"

    logger.info("test_fabric_dispatch_compute PASSED")


if __name__ == "__main__":
    test_fabric_dispatch_compute()
