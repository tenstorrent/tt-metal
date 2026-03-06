#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fabric dispatch + compute at P=256 (8 tile rows).

Single expert per device, K=1, weight=1.0.
Asymmetric dispatch: 160 local / 96 remote per device.

Packed combine metadata: header(4) + 1*(2+256) = 262 < 341 (fits).

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_p256.py
"""

import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
P = 256
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
N_TOKENS_PER_DEVICE = 256
N_TOKENS_TOTAL = 512
NUM_EXPERTS_PER_DEVICE = 1
NUM_EXPERTS_TOTAL = 2
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
PCC_THRESHOLD = 0.96


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
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


def test_p256():
    D_tiles = D // TILE
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    n_tiles_dn = D // TILE
    D_padded = n_tiles_dn * TILE
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE

    logger.info("Setting up fabric and opening full 8x4 mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    logger.info(f"Full mesh opened with {full_mesh.get_num_devices()} devices")

    submesh = full_mesh.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))
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
    D_padded,
    D_FF_HALF_padded,
):
    torch.manual_seed(42)

    all_hs = torch.randn(N_TOKENS_TOTAL, D, dtype=torch.bfloat16)
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]

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

    # Asymmetric dispatch: 160 local, 96 remote
    dev0_dispatch = [160, 96, 96] + list(range(160)) + list(range(160, 256))
    dev1_dispatch = [160, 96, 96] + list(range(96, 256)) + list(range(96))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]
    logger.info("Dispatch: 160 local + 96 recv per device (asymmetric)")

    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    all_rows = list(range(P))

    # Create mesh tensors
    hs_dev0 = all_hs[:N_TOKENS_PER_DEVICE].unsqueeze(0).unsqueeze(0)
    hs_dev1 = all_hs[N_TOKENS_PER_DEVICE:].unsqueeze(0).unsqueeze(0)
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

    # Single expert per device
    dev0_w = shuffled_ws[0].unsqueeze(0).unsqueeze(0)
    dev1_w = shuffled_ws[1].unsqueeze(0).unsqueeze(0)
    stacked_w = torch.cat([dev0_w, dev1_w], dim=0)
    gu_mesh = ttnn.from_torch(
        stacked_w,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    dev0_dw = down_ws_raw[0].unsqueeze(0).unsqueeze(0)
    dev1_dw = down_ws_raw[1].unsqueeze(0).unsqueeze(0)
    stacked_dw = torch.cat([dev0_dw, dev1_dw], dim=0)
    dn_mesh = ttnn.from_torch(
        stacked_dw,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created weights (1 expert per device)")

    # Scratch buffers
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
    logger.info(f"Created all scratch buffers (P={P})")

    # Build combine metadata
    out_buf_dev_tensors = ttnn.get_device_tensors(out_buf)
    out_buf_addr = out_buf_dev_tensors[0].buffer_address()

    def make_combine_meta():
        meta = [out_buf_addr, P]
        meta.extend(all_rows)
        meta.extend([w_1_bf16] * P)
        return meta

    per_device_combine_metadata = [
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
    ]
    logger.info(
        f"Combine metadata: 1 expert, M_e={P}, w=1.0, {len(per_device_combine_metadata[0])} packed args per device"
    )

    # Run op
    logger.info(f"Calling prefill_moe_compute with P={P}...")
    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=[gu_mesh],
        down_weights=[dn_mesh],
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=[out_buf],
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

    # Read results
    result_dev_tensors = ttnn.get_device_tensors(result)
    dev0_result = ttnn.to_torch(result_dev_tensors[0]).squeeze().float()[:, :D]
    dev1_result = ttnn.to_torch(result_dev_tensors[1]).squeeze().float()[:, :D]

    ob_devs = ttnn.get_device_tensors(out_buf)
    dev0_out = ttnn.to_torch(ob_devs[0]).squeeze().float()[:, :D]
    dev1_out = ttnn.to_torch(ob_devs[1]).squeeze().float()[:, :D]

    # Dequantized weights
    gu_devs = ttnn.get_device_tensors(gu_mesh)
    dn_devs = ttnn.get_device_tensors(dn_mesh)
    gu0 = ttnn.to_torch(gu_devs[0]).squeeze().float()
    gu1 = ttnn.to_torch(gu_devs[1]).squeeze().float()
    dn0 = ttnn.to_torch(dn_devs[0]).squeeze().float()
    dn1 = ttnn.to_torch(dn_devs[1]).squeeze().float()

    # Expected pkt_buf (asymmetric 160/96)
    dev0_expected_tokens = list(range(160)) + list(range(256, 352))
    dev1_expected_tokens = list(range(352, 512)) + list(range(160, 256))

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    # Verify pkt_buf (spot check every 16th row)
    pkt_devs = ttnn.get_device_tensors(pkt_buf)
    dev0_pkt = ttnn.to_torch(pkt_devs[0]).squeeze().float()[:, :D]
    dev1_pkt = ttnn.to_torch(pkt_devs[1]).squeeze().float()[:, :D]
    dev0_expected_pkt = torch.stack([all_hs[t].float() for t in dev0_expected_tokens])
    dev1_expected_pkt = torch.stack([all_hs[t].float() for t in dev1_expected_tokens])

    pkt_ok = True
    for r in range(0, P, 16):
        for label, actual, expected in [
            ("Dev0", dev0_pkt, dev0_expected_pkt),
            ("Dev1", dev1_pkt, dev1_expected_pkt),
        ]:
            row_pcc = pcc(actual[r], expected[r])
            if row_pcc < 0.99:
                logger.warning(f"  {label} pkt_buf row {r}: PCC={row_pcc:.6f}")
                pkt_ok = False
    if pkt_ok:
        logger.info(f"PKT_BUF: sampled rows PCC >= 0.99 (P={P})")

    # Compute reference
    dev0_pkt_hs = torch.stack([all_hs[t].float() for t in dev0_expected_tokens])
    dev1_pkt_hs = torch.stack([all_hs[t].float() for t in dev1_expected_tokens])

    def compute_expert_ref(pkt_hs, gu_w, dn_w):
        ref_gu = pkt_hs @ gu_w
        ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w
        return ref_out[:, :D]

    ref_d0 = compute_expert_ref(dev0_pkt_hs, gu0, dn0)
    ref_d1 = compute_expert_ref(dev1_pkt_hs, gu1, dn1)

    # PCC checks
    pcc_d0_out = pcc(dev0_out, ref_d0)
    pcc_d1_out = pcc(dev1_out, ref_d1)
    pcc_d0_comb = pcc(dev0_result, ref_d0)
    pcc_d1_comb = pcc(dev1_result, ref_d1)

    logger.info(f"Dev0 Expert 0 ({P} tokens, P={P}): PCC={pcc_d0_out:.6f}")
    logger.info(f"Dev1 Expert 1 ({P} tokens, P={P}): PCC={pcc_d1_out:.6f}")
    logger.info(f"Dev0 Combined (P={P}): PCC={pcc_d0_comb:.6f}")
    logger.info(f"Dev1 Combined (P={P}): PCC={pcc_d1_comb:.6f}")

    all_passed = True
    for label, val in [
        ("D0 Expert 0", pcc_d0_out),
        ("D1 Expert 1", pcc_d1_out),
        ("D0 Combined", pcc_d0_comb),
        ("D1 Combined", pcc_d1_comb),
    ]:
        if val < PCC_THRESHOLD:
            logger.error(f"FAIL: {label} PCC {val:.6f} < {PCC_THRESHOLD}")
            all_passed = False

    assert all_passed, "One or more PCC checks failed"
    logger.info(f"test_p256 PASSED")


if __name__ == "__main__":
    test_p256()
