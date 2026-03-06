#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test K=2 weighted combine + asymmetric dispatch + non-uniform M_e at P=64.

Same as test_fabric_dispatch_k2_nonuniform but scaled to P=64 (2 tile rows).

Expert assignments:
  Expert 0/2 (primary):   M_e=64, all 64 rows, weight=0.7
  Expert 1/3 (secondary): M_e=32, rows 0-31 only, weight=0.3

Output pattern:
  Rows  0-31: 0.7 * expert_0_out + 0.3 * expert_1_out (K=2 weighted)
  Rows 32-63: 0.7 * expert_0_out                       (K=1 partial)

Dispatch (asymmetric 40/24):
  Device 0: 40 local (indices 0-39), 24 sent (40-63), 24 received
  Device 1: 40 local (indices 24-63), 24 sent (0-23), 24 received

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_k2_p64.py
"""

import torch
import ttnn
from loguru import logger

# Constants
TILE = 32
P = 64
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
N_TOKENS_PER_DEVICE = 64
N_TOKENS_TOTAL = 128
NUM_EXPERTS_PER_DEVICE = 2
NUM_EXPERTS_TOTAL = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
BF16_TILE_BYTES = 2048
PCC_THRESHOLD = 0.96

# K=2 weights
W_PRIMARY = 0.7
W_SECONDARY = 0.3
# M_e per expert
M_E_PRIMARY = 64  # all rows
M_E_SECONDARY = 32  # rows 0-31 only


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


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


def test_k2_p64():
    # ---- Tile dimension calculations ----
    D_tiles = D // TILE
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    n_tiles_dn = D // TILE
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

    # ---- Generate data ----
    all_hs = torch.randn(N_TOKENS_TOTAL, D, dtype=torch.bfloat16)

    # 4 experts: gate_up [D, D_FF], down [D_FF/2, D]
    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS_TOTAL)]

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

    # ---- Asymmetric dispatch: 40 local, 24 remote per device ----
    dev0_dispatch = [40, 24, 24] + list(range(40)) + list(range(40, 64))
    dev1_dispatch = [40, 24, 24] + list(range(24, 64)) + list(range(24))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]
    logger.info("Dispatch: 40 local + 24 recv per device (asymmetric)")

    # ---- K=2 weights in BF16 ----
    def bf16_bits(val):
        return int(torch.tensor(val, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    w_primary_bf16 = bf16_bits(W_PRIMARY)
    w_secondary_bf16 = bf16_bits(W_SECONDARY)

    # Expert row assignments
    e0_rows = list(range(M_E_PRIMARY))  # all 64 rows
    e1_rows = list(range(M_E_SECONDARY))  # rows 0-31
    logger.info(f"Expert 0/2 (primary):   M_e={M_E_PRIMARY}, rows [0..{M_E_PRIMARY - 1}], w={W_PRIMARY}")
    logger.info(f"Expert 1/3 (secondary): M_e={M_E_SECONDARY}, rows [0..{M_E_SECONDARY - 1}], w={W_SECONDARY}")

    # ---- Create mesh tensors ----
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

    # gate_up_weights: 2 per device, sharded
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

    # down_weights
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
    logger.info("Created weights (2 experts per device, sharded)")

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
    logger.info("Created all scratch buffers")

    # ---- Build per-device combine metadata ----
    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())

    def make_combine_meta():
        meta = []
        # Expert 0/2 (primary): M_e=64, all rows, weight=0.7
        meta.append(out_buf_addrs[0])
        meta.append(M_E_PRIMARY)
        meta.extend(e0_rows)
        meta.extend([w_primary_bf16] * M_E_PRIMARY)
        # Expert 1/3 (secondary): M_e=32, rows 0-31, weight=0.3
        meta.append(out_buf_addrs[1])
        meta.append(M_E_SECONDARY)
        meta.extend(e1_rows)
        meta.extend([w_secondary_bf16] * M_E_SECONDARY)
        return meta

    per_device_combine_metadata = [
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
    ]
    logger.info(
        f"Combine metadata: E0/2 M_e={M_E_PRIMARY} w={W_PRIMARY}, "
        f"E1/3 M_e={M_E_SECONDARY} w={W_SECONDARY}, "
        f"{len(per_device_combine_metadata[0])} packed args per device"
    )

    # ---- Run the C++ op ----
    logger.info("Calling prefill_moe_compute with K=2 non-uniform, P=64...")
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

    # ---- Read back pkt_buf to check dispatch correctness ----
    pkt_devs = ttnn.get_device_tensors(pkt_buf)
    dev0_pkt = ttnn.to_torch(pkt_devs[0]).squeeze().float()[:, :D]
    dev1_pkt = ttnn.to_torch(pkt_devs[1]).squeeze().float()[:, :D]

    # Expected pkt_buf contents (asymmetric 40/24 dispatch)
    # Dev0: rows 0-39 = global tokens 0-39 (local), rows 40-63 = global tokens 64-87 (received)
    # Dev1: rows 0-39 = global tokens 88-127 (local=[24-63]), rows 40-63 = global tokens 40-63 (received)
    dev0_expected_tokens = list(range(40)) + list(range(64, 88))
    dev1_expected_tokens = list(range(88, 128)) + list(range(40, 64))
    dev0_expected_pkt = torch.stack([all_hs[t].float() for t in dev0_expected_tokens])
    dev1_expected_pkt = torch.stack([all_hs[t].float() for t in dev1_expected_tokens])

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    # Verify pkt_buf
    pkt_ok = True
    for r in range(P):
        for label, actual, expected, tokens in [
            ("Dev0", dev0_pkt, dev0_expected_pkt, dev0_expected_tokens),
            ("Dev1", dev1_pkt, dev1_expected_pkt, dev1_expected_tokens),
        ]:
            row_pcc = pcc(actual[r], expected[r])
            if row_pcc < 0.99:
                logger.warning(f"  {label} pkt_buf row {r} (token {tokens[r]}): PCC={row_pcc:.6f}")
                pkt_ok = False
    if pkt_ok:
        logger.info(f"PKT_BUF: all {P} rows PCC >= 0.99")

    # ---- Read back results ----
    result_dev_tensors = ttnn.get_device_tensors(result)
    dev0_result = ttnn.to_torch(result_dev_tensors[0]).squeeze().float()[:, :D]
    dev1_result = ttnn.to_torch(result_dev_tensors[1]).squeeze().float()[:, :D]

    # Per-expert out_bufs
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

    ref_d0_e0 = compute_expert_ref(dev0_pkt_hs, gu_dequant[0][0], dn_dequant[0][0])
    ref_d0_e1 = compute_expert_ref(dev0_pkt_hs, gu_dequant[0][1], dn_dequant[0][1])
    ref_d1_e2 = compute_expert_ref(dev1_pkt_hs, gu_dequant[1][0], dn_dequant[1][0])
    ref_d1_e3 = compute_expert_ref(dev1_pkt_hs, gu_dequant[1][1], dn_dequant[1][1])

    # BF16-quantized weights for reference accumulation
    w_pri = torch.tensor(W_PRIMARY, dtype=torch.bfloat16).float().item()
    w_sec = torch.tensor(W_SECONDARY, dtype=torch.bfloat16).float().item()

    # Reference combined output:
    #   Rows 0-31:  w_pri * e0 + w_sec * e1
    #   Rows 32-63: w_pri * e0
    ref_d0_combined = torch.zeros(P, D, dtype=torch.float32)
    for r in e0_rows:
        ref_d0_combined[r] += w_pri * ref_d0_e0[r]
    for r in e1_rows:
        ref_d0_combined[r] += w_sec * ref_d0_e1[r]

    ref_d1_combined = torch.zeros(P, D, dtype=torch.float32)
    for r in e0_rows:
        ref_d1_combined[r] += w_pri * ref_d1_e2[r]
    for r in e1_rows:
        ref_d1_combined[r] += w_sec * ref_d1_e3[r]

    # ---- PCC checks ----
    pcc_d0_e0 = pcc(dev0_outs[0], ref_d0_e0)
    pcc_d0_e1 = pcc(dev0_outs[1], ref_d0_e1)
    pcc_d1_e2 = pcc(dev1_outs[0], ref_d1_e2)
    pcc_d1_e3 = pcc(dev1_outs[1], ref_d1_e3)

    logger.info(f"Dev0 Expert 0 (full {P}-row): PCC={pcc_d0_e0:.6f}")
    logger.info(f"Dev0 Expert 1 (full {P}-row): PCC={pcc_d0_e1:.6f}")
    logger.info(f"Dev1 Expert 2 (full {P}-row): PCC={pcc_d1_e2:.6f}")
    logger.info(f"Dev1 Expert 3 (full {P}-row): PCC={pcc_d1_e3:.6f}")

    pcc_d0_comb = pcc(dev0_result, ref_d0_combined)
    pcc_d1_comb = pcc(dev1_result, ref_d1_combined)

    logger.info(f"Dev0 Combined (0.7*e0 + 0.3*e1[0:{M_E_SECONDARY-1}]): PCC={pcc_d0_comb:.6f}")
    logger.info(f"Dev1 Combined (0.7*e2 + 0.3*e3[0:{M_E_SECONDARY-1}]): PCC={pcc_d1_comb:.6f}")

    # Verify K=2 rows and K=1 rows separately
    pcc_d0_k2_rows = pcc(dev0_result[:M_E_SECONDARY], ref_d0_combined[:M_E_SECONDARY])
    pcc_d0_k1_rows = pcc(dev0_result[M_E_SECONDARY:], ref_d0_combined[M_E_SECONDARY:])
    logger.info(f"Dev0 K=2 rows (0-{M_E_SECONDARY-1}): PCC={pcc_d0_k2_rows:.6f}")
    logger.info(f"Dev0 K=1 rows ({M_E_SECONDARY}-{P-1}): PCC={pcc_d0_k1_rows:.6f}")

    # ---- Assertions ----
    all_passed = True
    for label, val in [
        ("D0 Expert 0", pcc_d0_e0),
        ("D0 Expert 1", pcc_d0_e1),
        ("D1 Expert 2", pcc_d1_e2),
        ("D1 Expert 3", pcc_d1_e3),
        ("D0 Combined", pcc_d0_comb),
        ("D1 Combined", pcc_d1_comb),
    ]:
        if val < PCC_THRESHOLD:
            logger.error(f"FAIL: {label} PCC {val:.6f} < {PCC_THRESHOLD}")
            all_passed = False

    assert all_passed, "One or more PCC checks failed"
    logger.info("test_k2_p64 PASSED")


if __name__ == "__main__":
    test_k2_p64()
