#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test E=4 per-expert dispatch on 1x2 mesh (no fabric exchange).

4 experts per device, each processing a different 32-token subset.
128 tokens total per device, split: expert 0 gets 0-31, expert 1 gets 32-63,
expert 2 gets 64-95, expert 3 gets 96-127.

K=1: each token goes to exactly 1 expert with weight=1.0.
No fabric exchange: send_count=0, recv_count=0 on each device.
Both devices run independently with different weights but same token layout.

Uses per_expert_dispatch_sources to specify per-expert token routing.
pkt_buf shape: [1, 1, 128, D] (4 experts × 32 tokens).
output shape: [1, 1, 128, D_padded].
out_buf shape: [1, 1, 128, D_padded] (same as output for tile-row alignment).

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_E4_single_device.py
"""

import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
P = 32  # tokens per expert
P_TOTAL = 128  # total tokens per device = NUM_EXPERTS * P
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_EXPERTS_PER_DEVICE = 4
NUM_EXPERTS_TOTAL = 8  # 4 per device × 2 devices
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
PCC_THRESHOLD = 0.96


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


def test_E4_single_device():
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
    # Each device gets 128 tokens
    all_hs_dev0 = torch.randn(P_TOTAL, D, dtype=torch.bfloat16)
    all_hs_dev1 = torch.randn(P_TOTAL, D, dtype=torch.bfloat16)

    # 8 experts: 4 per device
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

    # ---- Routing: no fabric exchange ----
    # Each device keeps all 128 tokens locally, no send/recv.
    # dispatch_metadata (old format): [local_count=128, recv_count=0, send_count=0, indices 0..127]
    dispatch_meta_per_dev = [128, 0, 0] + list(range(128))
    dispatch_metadata = [dispatch_meta_per_dev, dispatch_meta_per_dev]

    # ---- Per-expert dispatch sources ----
    # Each expert gets 32 tokens from different hs_rm rows, all local (bit 31 = 0).
    # Expert 0: rows 0-31, Expert 1: rows 32-63, Expert 2: rows 64-95, Expert 3: rows 96-127
    expert_sources = [NUM_EXPERTS_PER_DEVICE]  # num_experts
    for e in range(NUM_EXPERTS_PER_DEVICE):
        start_row = e * P
        expert_sources.append(P)  # M_e = 32
        expert_sources.extend(range(start_row, start_row + P))  # local hs_rm rows (bit 31 = 0)
    per_expert_dispatch_sources = [expert_sources, expert_sources]  # same layout for both devices
    logger.info(f"Per-expert sources: {NUM_EXPERTS_PER_DEVICE} experts × {P} tokens, all local")

    # ---- K=1 weight in BF16 ----
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    # ---- Create mesh tensors ----
    # hidden_states_rm (ROW_MAJOR, sharded) — 128 tokens per device
    stacked_hs = torch.cat([all_hs_dev0.unsqueeze(0).unsqueeze(0), all_hs_dev1.unsqueeze(0).unsqueeze(0)], dim=0)
    hs_rm = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # hidden_states (TILE, replicated — for shape derivation only)
    hs_tile = ttnn.from_torch(
        torch.zeros(1, 1, P_TOTAL, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gate_up_weights: 4 per device, sharded
    # Device 0 gets experts 0-3; Device 1 gets experts 4-7
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
    logger.info(f"Created weights ({NUM_EXPERTS_PER_DEVICE} experts per device)")

    # pkt_buf: [1, 1, E*P, D] = [1, 1, 128, D]
    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, P_TOTAL, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # inter_buf: [1, 1, P, D_FF_HALF_padded] (per-expert, shared across experts)
    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # out_bufs: [1, 1, P_TOTAL, D_padded] each (same size as output for tile-row alignment)
    out_buf_list = []
    for _ in range(NUM_EXPERTS_PER_DEVICE):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P_TOTAL, D_padded, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_buf_list.append(ob)

    # staging_buf: needed for fabric dispatch (even if unused)
    staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, P_TOTAL, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # output: [1, 1, P_TOTAL, D_padded]
    output = ttnn.from_torch(
        torch.zeros(1, 1, P_TOTAL, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created all buffers")

    # ---- Build per-device combine metadata ----
    # K=1: each expert contributes to disjoint output rows with weight=1.0
    # Expert 0 -> rows 0-31, Expert 1 -> rows 32-63, etc.
    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())

    def make_combine_meta():
        meta = []
        for e in range(NUM_EXPERTS_PER_DEVICE):
            start_row = e * P
            meta.append(out_buf_addrs[e])
            meta.append(P)  # M_e = 32
            meta.extend(range(start_row, start_row + P))  # output rows
            meta.extend([w_1_bf16] * P)  # weight=1.0
        return meta

    per_device_combine_metadata = [
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
    ]
    logger.info(f"Combine metadata: {NUM_EXPERTS_PER_DEVICE} experts, {P} tokens each, K=1 w=1.0")

    # ---- Run the C++ op ----
    logger.info("Calling prefill_moe_compute with E=4 per-expert dispatch...")
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
        per_expert_dispatch_sources=per_expert_dispatch_sources,
        enable_fpu_combine=True,
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
    gu_dequant = [[], []]
    dn_dequant = [[], []]
    for e in range(NUM_EXPERTS_PER_DEVICE):
        gu_devs = ttnn.get_device_tensors(gu_mesh_list[e])
        dn_devs = ttnn.get_device_tensors(dn_mesh_list[e])
        gu_dequant[0].append(ttnn.to_torch(gu_devs[0]).squeeze().float())
        gu_dequant[1].append(ttnn.to_torch(gu_devs[1]).squeeze().float())
        dn_dequant[0].append(ttnn.to_torch(dn_devs[0]).squeeze().float())
        dn_dequant[1].append(ttnn.to_torch(dn_devs[1]).squeeze().float())

    # ---- Compute reference ----
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

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    # ---- Verify per-expert out_bufs ----
    all_passed = True
    for dev_idx, (all_hs, gu_d, dn_d, outs, label) in enumerate(
        [
            (all_hs_dev0, gu_dequant[0], dn_dequant[0], dev0_outs, "Dev0"),
            (all_hs_dev1, gu_dequant[1], dn_dequant[1], dev1_outs, "Dev1"),
        ]
    ):
        for e in range(NUM_EXPERTS_PER_DEVICE):
            start_row = e * P
            # Expert e sees tokens [start_row : start_row + P]
            expert_hs = all_hs[start_row : start_row + P].float()
            ref = compute_expert_ref(expert_hs, gu_d[e], dn_d[e])

            # out_buf has results at tile rows [start_row//32 : start_row//32 + P//32]
            # For P=32: 1 tile row at offset e
            tile_row_offset = start_row // TILE
            actual = outs[e][start_row : start_row + P]

            p = pcc(actual, ref)
            status = "PASS" if p >= PCC_THRESHOLD else "FAIL"
            logger.info(f"{label} Expert {e} (rows {start_row}-{start_row+P-1}): PCC={p:.6f} [{status}]")
            if p < PCC_THRESHOLD:
                all_passed = False

    # ---- Verify combined output ----
    # K=1: output = sum of per-expert contributions (each with weight=1.0)
    # Expert 0 contributes to rows 0-31, Expert 1 to rows 32-63, etc.
    # So output[0:32] = expert_0_output, output[32:64] = expert_1_output, etc.
    for dev_idx, (all_hs, gu_d, dn_d, result_out, label) in enumerate(
        [
            (all_hs_dev0, gu_dequant[0], dn_dequant[0], dev0_result, "Dev0"),
            (all_hs_dev1, gu_dequant[1], dn_dequant[1], dev1_result, "Dev1"),
        ]
    ):
        ref_combined = torch.zeros(P_TOTAL, D, dtype=torch.float32)
        for e in range(NUM_EXPERTS_PER_DEVICE):
            start_row = e * P
            expert_hs = all_hs[start_row : start_row + P].float()
            ref = compute_expert_ref(expert_hs, gu_d[e], dn_d[e])
            ref_combined[start_row : start_row + P] = ref  # weight=1.0

        p = pcc(result_out, ref_combined)
        status = "PASS" if p >= PCC_THRESHOLD else "FAIL"
        logger.info(f"{label} Combined (E=4, K=1): PCC={p:.6f} [{status}]")
        if p < PCC_THRESHOLD:
            all_passed = False

    assert all_passed, "One or more PCC checks failed"
    logger.info("test_E4_single_device PASSED")


if __name__ == "__main__":
    test_E4_single_device()
