#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test fused fabric dispatch + multi-expert compute with non-uniform token counts.

4 experts: 0,1 on device 0; 2,3 on device 1.
K=1: each token goes to exactly 1 expert.
Non-uniform distribution:
  Expert 0: 24 tokens, Expert 1: 8 tokens
  Expert 2: 24 tokens, Expert 3: 8 tokens

Asymmetric dispatch (20 local / 12 remote per device):
  Device 0 tokens (global 0-31, device-local 0-31):
    local_indices 0-19  -> stay on dev0 (20 tokens)
    local_indices 20-31 -> sent to dev1 (12 tokens)

  Device 1 tokens (global 32-63, device-local 0-31):
    local_indices 0-11  -> sent to dev0 (12 tokens)
    local_indices 12-31 -> stay on dev1 (20 tokens)

  Device 0 pkt_buf (32 rows):
    rows 0-19:  local tokens (global 0-19)
    rows 20-31: received tokens (global 32-43)

  Device 1 pkt_buf (32 rows):
    rows 0-19:  local tokens (global 44-63)
    rows 20-31: received tokens (global 20-31)

  Expert assignment within pkt_buf (same pattern on both devices):
    Expert 0/2 (M_e=24): rows [0..15, 20..27]
    Expert 1/3 (M_e=8):  rows [16..19, 28..31]

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_nonuniform.py
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
PCC_THRESHOLD = 0.96


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def test_nonuniform_token_counts():
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

    # ---- Asymmetric dispatch: 20 local, 12 remote per device ----
    # Dev0: 20 local (indices 0-19), 12 sent (indices 20-31), 12 received
    # Dev1: 20 local (indices 12-31), 12 sent (indices 0-11), 12 received
    dev0_dispatch = [20, 12, 12] + list(range(20)) + list(range(20, 32))
    dev1_dispatch = [20, 12, 12] + list(range(12, 32)) + list(range(12))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]
    logger.info("Dispatch: 20 local + 12 recv per device (asymmetric)")

    # K=1 weight = 1.0 in BF16
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    # Expert row assignments within pkt_buf (same on both devices)
    e0_rows = list(range(16)) + list(range(20, 28))  # 24 rows
    e1_rows = list(range(16, 20)) + list(range(28, 32))  # 8 rows
    logger.info(f"Expert 0/2 rows ({len(e0_rows)}): {e0_rows}")
    logger.info(f"Expert 1/3 rows ({len(e1_rows)}): {e1_rows}")

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
        # Expert 0/2 (24 tokens)
        meta.append(out_buf_addrs[0])
        meta.append(24)
        meta.extend(e0_rows)
        meta.extend([w_1_bf16] * 24)
        # Expert 1/3 (8 tokens)
        meta.append(out_buf_addrs[1])
        meta.append(8)
        meta.extend(e1_rows)
        meta.extend([w_1_bf16] * 8)
        return meta

    per_device_combine_metadata = [make_combine_meta(), make_combine_meta()]
    logger.info(
        f"Combine metadata: Expert 0/2 M_e=24, Expert 1/3 M_e=8, "
        f"{len(per_device_combine_metadata[0])} args per device"
    )

    # ---- Run the C++ op ----
    logger.info("Calling prefill_moe_compute with non-uniform token counts...")
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

    # Expected pkt_buf contents (asymmetric 20/12 dispatch)
    # Dev0: rows 0-19 = tokens 0-19 (local), rows 20-31 = tokens 32-43 (received)
    # Dev1: rows 0-19 = tokens 44-63 (local), rows 20-31 = tokens 20-31 (received)
    dev0_expected_tokens = list(range(20)) + list(range(32, 44))
    dev1_expected_tokens = list(range(44, 64)) + list(range(20, 32))
    dev0_expected_pkt = torch.stack([all_hs[t].float() for t in dev0_expected_tokens])
    dev1_expected_pkt = torch.stack([all_hs[t].float() for t in dev1_expected_tokens])

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    # Verify pkt_buf row-by-row
    pkt_ok = True
    for r in range(32):
        for label, actual, expected, tokens in [
            ("Dev0", dev0_pkt, dev0_expected_pkt, dev0_expected_tokens),
            ("Dev1", dev1_pkt, dev1_expected_pkt, dev1_expected_tokens),
        ]:
            row_pcc = pcc(actual[r], expected[r])
            if row_pcc < 0.99:
                logger.warning(f"  {label} pkt_buf row {r} (token {tokens[r]}): PCC={row_pcc:.6f}")
                pkt_ok = False
    if pkt_ok:
        logger.info("PKT_BUF: all rows PCC >= 0.99")

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
    gu_dequant = [[], []]  # [dev][expert]
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
        """Reference: gate_up matmul -> SwiGLU -> down matmul."""
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

    # Reference combined output
    ref_d0_combined = torch.zeros(P, D, dtype=torch.float32)
    for r in e0_rows:
        ref_d0_combined[r] = ref_d0_e0[r]
    for r in e1_rows:
        ref_d0_combined[r] = ref_d0_e1[r]

    ref_d1_combined = torch.zeros(P, D, dtype=torch.float32)
    for r in e0_rows:
        ref_d1_combined[r] = ref_d1_e2[r]
    for r in e1_rows:
        ref_d1_combined[r] = ref_d1_e3[r]

    # ---- PCC checks ----
    pcc_d0_e0 = pcc(dev0_outs[0][e0_rows], ref_d0_e0[e0_rows])
    pcc_d0_e1 = pcc(dev0_outs[1][e1_rows], ref_d0_e1[e1_rows])
    pcc_d1_e2 = pcc(dev1_outs[0][e0_rows], ref_d1_e2[e0_rows])
    pcc_d1_e3 = pcc(dev1_outs[1][e1_rows], ref_d1_e3[e1_rows])

    logger.info(f"Dev0 Expert 0 (24 tokens): PCC={pcc_d0_e0:.6f}")
    logger.info(f"Dev0 Expert 1 (8 tokens):  PCC={pcc_d0_e1:.6f}")
    logger.info(f"Dev1 Expert 2 (24 tokens): PCC={pcc_d1_e2:.6f}")
    logger.info(f"Dev1 Expert 3 (8 tokens):  PCC={pcc_d1_e3:.6f}")

    pcc_d0_comb = pcc(dev0_result, ref_d0_combined)
    pcc_d1_comb = pcc(dev1_result, ref_d1_combined)

    logger.info(f"Dev0 Combined: PCC={pcc_d0_comb:.6f}")
    logger.info(f"Dev1 Combined: PCC={pcc_d1_comb:.6f}")

    # ---- Assertions ----
    all_passed = True
    for label, val in [
        ("D0 Expert 0 (24 tokens)", pcc_d0_e0),
        ("D0 Expert 1 (8 tokens)", pcc_d0_e1),
        ("D1 Expert 2 (24 tokens)", pcc_d1_e2),
        ("D1 Expert 3 (8 tokens)", pcc_d1_e3),
        ("D0 Combined", pcc_d0_comb),
        ("D1 Combined", pcc_d1_comb),
    ]:
        if val < PCC_THRESHOLD:
            logger.error(f"FAIL: {label} PCC {val:.6f} < {PCC_THRESHOLD}")
            all_passed = False

    assert all_passed, "One or more PCC checks failed"
    logger.info("test_nonuniform_token_counts PASSED")


if __name__ == "__main__":
    test_nonuniform_token_counts()
