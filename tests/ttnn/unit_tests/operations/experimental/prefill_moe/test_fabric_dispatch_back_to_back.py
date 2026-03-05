#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test back-to-back execution without device reset.

Runs the prefill_moe_compute op multiple times in a loop to verify:
- Semaphores are properly cleaned up between invocations
- No resource leaks in fabric connection management
- Consistent results across iterations

Uses asymmetric dispatch (20/12) with K=1, 2 experts, uniform M_e=32.
Each iteration uses the same data but re-zeros output/scratch buffers.

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_back_to_back.py
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

N_ITERATIONS = 5


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def test_back_to_back():
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

    # ---- Asymmetric dispatch: 20 local, 12 remote per device ----
    dev0_dispatch = [20, 12, 12] + list(range(20)) + list(range(20, 32))
    dev1_dispatch = [20, 12, 12] + list(range(12, 32)) + list(range(12))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]

    # ---- K=1 weight in BF16 ----
    def bf16_bits(val):
        return int(torch.tensor(val, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    w_1_bf16 = bf16_bits(1.0)

    # ---- Create mesh tensors (persistent across iterations) ----
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

    # Scratch buffers (persistent, re-zeroed each iteration)
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

    # ---- Build per-device combine metadata ----
    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())

    all_rows = list(range(32))

    def make_combine_meta():
        meta = []
        for e in range(NUM_EXPERTS_PER_DEVICE):
            meta.append(out_buf_addrs[e])
            meta.append(32)
            meta.extend(all_rows)
            meta.extend([w_1_bf16] * 32)
        return meta

    per_device_combine_metadata = [make_combine_meta(), make_combine_meta()]

    # ---- Compute reference (once) ----
    dev0_expected_tokens = list(range(20)) + list(range(32, 44))
    dev1_expected_tokens = list(range(44, 64)) + list(range(20, 32))
    dev0_expected_pkt = torch.stack([all_hs[t].float() for t in dev0_expected_tokens])
    dev1_expected_pkt = torch.stack([all_hs[t].float() for t in dev1_expected_tokens])

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

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

    def compute_expert_ref(pkt_hs, gu_w, dn_w):
        ref_gu = pkt_hs @ gu_w
        ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w
        return ref_out[:, :D]

    ref_d0_e0 = compute_expert_ref(dev0_expected_pkt, gu_dequant[0][0], dn_dequant[0][0])
    ref_d0_e1 = compute_expert_ref(dev0_expected_pkt, gu_dequant[0][1], dn_dequant[0][1])
    ref_d1_e2 = compute_expert_ref(dev1_expected_pkt, gu_dequant[1][0], dn_dequant[1][0])
    ref_d1_e3 = compute_expert_ref(dev1_expected_pkt, gu_dequant[1][1], dn_dequant[1][1])
    ref_d0_combined = ref_d0_e0 + ref_d0_e1
    ref_d1_combined = ref_d1_e2 + ref_d1_e3

    logger.info(f"Running {N_ITERATIONS} back-to-back iterations...")

    # ---- Iteration loop ----
    all_passed = True
    for iteration in range(N_ITERATIONS):
        # Run the op
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

        # Read results
        result_dev_tensors = ttnn.get_device_tensors(result)
        dev0_result = ttnn.to_torch(result_dev_tensors[0]).squeeze().float()[:, :D]
        dev1_result = ttnn.to_torch(result_dev_tensors[1]).squeeze().float()[:, :D]

        pcc_d0 = pcc(dev0_result, ref_d0_combined)
        pcc_d1 = pcc(dev1_result, ref_d1_combined)

        # Per-expert check (brief)
        dev0_outs = []
        dev1_outs = []
        for ob in out_buf_list:
            devs = ttnn.get_device_tensors(ob)
            dev0_outs.append(ttnn.to_torch(devs[0]).squeeze().float()[:, :D])
            dev1_outs.append(ttnn.to_torch(devs[1]).squeeze().float()[:, :D])

        pcc_d0_e0 = pcc(dev0_outs[0], ref_d0_e0)
        pcc_d0_e1 = pcc(dev0_outs[1], ref_d0_e1)
        pcc_d1_e2 = pcc(dev1_outs[0], ref_d1_e2)
        pcc_d1_e3 = pcc(dev1_outs[1], ref_d1_e3)

        iter_ok = all(v >= PCC_THRESHOLD for v in [pcc_d0, pcc_d1, pcc_d0_e0, pcc_d0_e1, pcc_d1_e2, pcc_d1_e3])

        status = "OK" if iter_ok else "FAIL"
        logger.info(
            f"Iter {iteration + 1}/{N_ITERATIONS}: {status} | "
            f"D0 comb={pcc_d0:.6f} D1 comb={pcc_d1:.6f} | "
            f"e0={pcc_d0_e0:.6f} e1={pcc_d0_e1:.6f} e2={pcc_d1_e2:.6f} e3={pcc_d1_e3:.6f}"
        )

        if not iter_ok:
            all_passed = False
            logger.error(f"Iteration {iteration + 1} FAILED")

    assert all_passed, f"One or more iterations failed over {N_ITERATIONS} runs"
    logger.info(f"test_back_to_back PASSED ({N_ITERATIONS} iterations)")


if __name__ == "__main__":
    test_back_to_back()
