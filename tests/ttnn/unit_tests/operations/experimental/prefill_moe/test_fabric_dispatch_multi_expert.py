#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fused fabric dispatch + multi-expert compute on 1x2 submesh of TG 6U.

4 experts total: experts 0,1 on device 0; experts 2,3 on device 1.
K=2: each token goes to exactly 2 experts (co-located on same device).

Routing:
  tokens  0-15 -> experts 0,1 (device 0 local)
  tokens 16-31 -> experts 2,3 (device 0 sends to device 1)
  tokens 32-47 -> experts 0,1 (device 1 sends to device 0)
  tokens 48-63 -> experts 2,3 (device 1 local)

Each device's pkt_buf has 32 tokens (16 local + 16 received).
Both experts on a device process the same pkt_buf.
Combine does weighted accumulation: output = w0*out_buf[0] + w1*out_buf[1].

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_multi_expert.py
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
NUM_EXPERTS_PER_DEVICE = 2
NUM_EXPERTS_TOTAL = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
BF16_TILE_BYTES = 2048


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def test_fabric_dispatch_multi_expert():
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
        )
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.close_mesh_device(full_mesh)


def pack_combine_metadata(old_meta, num_experts):
    """Convert old-format combine metadata to packed (row, weight) pair format.

    Old format per expert: [addr, M_e, row0..rowM, w0..wM]
    New format per expert: [addr, M_e, rw0..rw_{M_e-1}]
      where rw_i = (row_i & 0xFFFF) | (weight_bf16_i << 16)
    """
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

    # 4 experts: gate_up [D, D_FF], down [D_FF/2, D]
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]

    # Pre-shuffle gate_up weights for per-core SwiGLU (gate|up interleave per core)
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
    # Device 0 tokens (0-31):  0-15 -> experts 0,1 (local), 16-31 -> experts 2,3 (send)
    # Device 1 tokens (32-63): 32-47 -> experts 0,1 (send), 48-63 -> experts 2,3 (local)
    #
    # pkt_buf on device 0: rows 0-15 = tokens 0-15, rows 16-31 = tokens 32-47
    # pkt_buf on device 1: rows 0-15 = tokens 48-63, rows 16-31 = tokens 16-31
    #
    # Both experts on each device process the same pkt_buf.

    # Dispatch metadata: [local_count, recv_count, send_count, local_indices..., send_indices...]
    dev0_dispatch = [16, 16, 16] + list(range(16)) + list(range(16, 32))
    dev1_dispatch = [16, 16, 16] + list(range(16, 32)) + list(range(16))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]

    logger.info(f"Dispatch metadata: same as K=1 test (16 local, 16 recv, 16 send per device)")

    # ---- Routing weights (K=2) ----
    # For each token, two expert weights that sum to ~1.0
    # Use deterministic weights: w0=0.6, w1=0.4 for all tokens
    W_EXPERT_0 = 0.6
    W_EXPERT_1 = 0.4
    w0_bf16 = int(torch.tensor(W_EXPERT_0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    w1_bf16 = int(torch.tensor(W_EXPERT_1, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    logger.info(f"Routing weights: w0={W_EXPERT_0} (0x{w0_bf16:04x}), w1={W_EXPERT_1} (0x{w1_bf16:04x})")

    # ---- Create mesh tensors ----
    # hidden_states_rm (ROW_MAJOR, sharded)
    hs_dev0 = all_hs[:32].unsqueeze(0).unsqueeze(0)
    hs_dev1 = all_hs[32:].unsqueeze(0).unsqueeze(0)
    stacked_hs = torch.cat([hs_dev0, hs_dev1], dim=0)
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hidden_states_rm (ROW_MAJOR, sharded)")

    # hidden_states (TILE, replicated - for shape derivation)
    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gate_up_weights: 2 per device, sharded
    # Device 0 gets experts 0,1; Device 1 gets experts 2,3
    # ShardTensorToMesh with dim=0: stack [2, dev0_expert0], [2, dev0_expert1] won't work directly.
    # We need to create per-device tensors. Use list of mesh tensors.
    gu_mesh_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
        # Expert e on dev0, expert (e+2) on dev1
        dev0_w = shuffled_ws[e].unsqueeze(0).unsqueeze(0)  # [1, 1, D, D_FF]
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
    logger.info("Created gate_up_weights (2 per device, sharded)")

    # down_weights: same pattern
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
    logger.info("Created down_weights (2 per device, sharded)")

    # Scratch buffers (replicated)
    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, NUM_EXPERTS_PER_DEVICE * P, D, dtype=torch.bfloat16),
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
    # 2 out_bufs per device
    out_buf_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
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
    logger.info("Created all scratch buffers (2 out_bufs)")

    # ---- Build per-device combine metadata ----
    # Each device: 2 experts, 32 tokens each, weighted combine.
    # Format: [out_buf_0_addr, M_0, token_rows_0..., weights_0...,
    #          out_buf_1_addr, M_1, token_rows_1..., weights_1...]

    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())
    logger.info(f"out_buf addresses: {out_buf_addrs}")

    # Device 0 combine: expert 0 (w=0.6) + expert 1 (w=0.4) over 32 rows
    dev0_combine = []
    # Expert 0
    dev0_combine.append(out_buf_addrs[0])
    dev0_combine.append(32)
    dev0_combine.extend(list(range(32)))  # token rows 0-31
    dev0_combine.extend([w0_bf16] * 32)  # weight 0.6
    # Expert 1
    dev0_combine.append(out_buf_addrs[1])
    dev0_combine.append(32)
    dev0_combine.extend(list(range(32)))  # same rows
    dev0_combine.extend([w1_bf16] * 32)  # weight 0.4

    # Device 1 combine: same structure (experts 2,3 with same weights)
    dev1_combine = []
    dev1_combine.append(out_buf_addrs[0])
    dev1_combine.append(32)
    dev1_combine.extend(list(range(32)))
    dev1_combine.extend([w0_bf16] * 32)
    dev1_combine.append(out_buf_addrs[1])
    dev1_combine.append(32)
    dev1_combine.extend(list(range(32)))
    dev1_combine.extend([w1_bf16] * 32)

    per_device_combine_metadata = [
        pack_combine_metadata(dev0_combine, NUM_EXPERTS_PER_DEVICE),
        pack_combine_metadata(dev1_combine, NUM_EXPERTS_PER_DEVICE),
    ]
    logger.info(f"Combine metadata: 2 experts per device, " f"{len(dev0_combine)} args per device")

    # ---- Run the C++ op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with fabric dispatch, 2 experts...")
    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=gu_mesh_list,
        down_weights=dn_mesh_list,
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=out_buf_list,
        output=output,
        combine_metadata=per_device_combine_metadata,
        num_experts=NUM_EXPERTS_PER_DEVICE,
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

    # Read back per-expert out_bufs
    dev0_outs = []
    dev1_outs = []
    for ob in out_buf_list:
        devs = ttnn.get_device_tensors(ob)
        dev0_outs.append(ttnn.to_torch(devs[0]).squeeze().float()[:, :D])
        dev1_outs.append(ttnn.to_torch(devs[1]).squeeze().float()[:, :D])

    # Read dequantized weights for reference
    gu_dequant_per_dev = [[], []]  # [dev][expert]
    dn_dequant_per_dev = [[], []]
    for e in range(NUM_EXPERTS_PER_DEVICE):
        gu_devs = ttnn.get_device_tensors(gu_mesh_list[e])
        dn_devs = ttnn.get_device_tensors(dn_mesh_list[e])
        gu_dequant_per_dev[0].append(ttnn.to_torch(gu_devs[0]).squeeze().float())
        gu_dequant_per_dev[1].append(ttnn.to_torch(gu_devs[1]).squeeze().float())
        dn_dequant_per_dev[0].append(ttnn.to_torch(dn_devs[0]).squeeze().float())
        dn_dequant_per_dev[1].append(ttnn.to_torch(dn_devs[1]).squeeze().float())

    # ---- Compute reference ----
    # Device 0 pkt_buf tokens: rows 0-15 = tokens 0-15, rows 16-31 = tokens 32-47
    dev0_pkt_tokens = list(range(16)) + list(range(32, 48))
    dev0_pkt_hs = torch.stack([all_hs[t].float() for t in dev0_pkt_tokens])  # [32, D]

    # Device 1 pkt_buf tokens: rows 0-15 = tokens 48-63, rows 16-31 = tokens 16-31
    dev1_pkt_tokens = list(range(48, 64)) + list(range(16, 32))
    dev1_pkt_hs = torch.stack([all_hs[t].float() for t in dev1_pkt_tokens])  # [32, D]

    # Reference per-expert computation
    def compute_expert_ref(pkt_hs, gu_w, dn_w):
        """Reference: gate_up matmul -> SwiGLU -> down matmul."""
        ref_gu = pkt_hs @ gu_w
        ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w
        return ref_out[:, :D]

    # Device 0: 2 experts
    ref_dev0_expert0 = compute_expert_ref(dev0_pkt_hs, gu_dequant_per_dev[0][0], dn_dequant_per_dev[0][0])
    ref_dev0_expert1 = compute_expert_ref(dev0_pkt_hs, gu_dequant_per_dev[0][1], dn_dequant_per_dev[0][1])

    # Device 0 combined: w0 * expert0 + w1 * expert1
    w0_actual = torch.tensor(W_EXPERT_0, dtype=torch.bfloat16).float()
    w1_actual = torch.tensor(W_EXPERT_1, dtype=torch.bfloat16).float()
    ref_dev0_combined = w0_actual * ref_dev0_expert0 + w1_actual * ref_dev0_expert1

    # Device 1: 2 experts
    ref_dev1_expert0 = compute_expert_ref(dev1_pkt_hs, gu_dequant_per_dev[1][0], dn_dequant_per_dev[1][0])
    ref_dev1_expert1 = compute_expert_ref(dev1_pkt_hs, gu_dequant_per_dev[1][1], dn_dequant_per_dev[1][1])
    ref_dev1_combined = w0_actual * ref_dev1_expert0 + w1_actual * ref_dev1_expert1

    # ---- PCC checks ----
    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    # Per-expert PCC
    pcc_d0_e0 = pcc(dev0_outs[0], ref_dev0_expert0)
    pcc_d0_e1 = pcc(dev0_outs[1], ref_dev0_expert1)
    pcc_d1_e0 = pcc(dev1_outs[0], ref_dev1_expert0)
    pcc_d1_e1 = pcc(dev1_outs[1], ref_dev1_expert1)

    logger.info(f"Device 0 Expert 0: PCC={pcc_d0_e0:.6f}")
    logger.info(f"Device 0 Expert 1: PCC={pcc_d0_e1:.6f}")
    logger.info(f"Device 1 Expert 2: PCC={pcc_d1_e0:.6f}")
    logger.info(f"Device 1 Expert 3: PCC={pcc_d1_e1:.6f}")

    # Combined PCC
    pcc_d0_combined = pcc(dev0_result, ref_dev0_combined)
    pcc_d1_combined = pcc(dev1_result, ref_dev1_combined)

    logger.info(f"Device 0 Combined (0.6*e0 + 0.4*e1): PCC={pcc_d0_combined:.6f}")
    logger.info(f"Device 1 Combined (0.6*e2 + 0.4*e3): PCC={pcc_d1_combined:.6f}")

    # ---- Assertions ----
    for label, val in [
        ("D0 Expert 0", pcc_d0_e0),
        ("D0 Expert 1", pcc_d0_e1),
        ("D1 Expert 2", pcc_d1_e0),
        ("D1 Expert 3", pcc_d1_e1),
    ]:
        assert val >= 0.96, f"{label} PCC {val:.6f} < 0.96"

    assert pcc_d0_combined >= 0.96, f"D0 combined PCC {pcc_d0_combined:.6f} < 0.96"
    assert pcc_d1_combined >= 0.96, f"D1 combined PCC {pcc_d1_combined:.6f} < 0.96"

    logger.info("test_fabric_dispatch_multi_expert PASSED")


if __name__ == "__main__":
    test_fabric_dispatch_multi_expert()
