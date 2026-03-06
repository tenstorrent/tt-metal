#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test K=4 weighted combine (gpt-oss config) at P=64.

4 experts per device (8 total). All experts process all 64 rows.
Combine: 0.4*e0 + 0.3*e1 + 0.2*e2 + 0.1*e3.

Dispatch: asymmetric 40/24 per device.

Run on TG 6U:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_dispatch_k4_p64.py
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
NUM_EXPERTS_PER_DEVICE = 4
NUM_EXPERTS_TOTAL = 8
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
PCC_THRESHOLD = 0.96

EXPERT_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


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


def test_k4_p64():
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

    # Asymmetric dispatch: 40 local, 24 remote
    dev0_dispatch = [40, 24, 24] + list(range(40)) + list(range(40, 64))
    dev1_dispatch = [40, 24, 24] + list(range(24, 64)) + list(range(24))
    dispatch_metadata = [dev0_dispatch, dev1_dispatch]
    logger.info("Dispatch: 40 local + 24 recv per device (asymmetric)")

    def bf16_bits(val):
        return int(torch.tensor(val, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    w_bf16s = [bf16_bits(w) for w in EXPERT_WEIGHTS]
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

    gu_mesh_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
        dev0_w = shuffled_ws[e].unsqueeze(0).unsqueeze(0)
        dev1_w = shuffled_ws[e + NUM_EXPERTS_PER_DEVICE].unsqueeze(0).unsqueeze(0)
        stacked_w = torch.cat([dev0_w, dev1_w], dim=0)
        gu_mesh_list.append(
            ttnn.from_torch(
                stacked_w,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    dn_mesh_list = []
    for e in range(NUM_EXPERTS_PER_DEVICE):
        dev0_w = down_ws_raw[e].unsqueeze(0).unsqueeze(0)
        dev1_w = down_ws_raw[e + NUM_EXPERTS_PER_DEVICE].unsqueeze(0).unsqueeze(0)
        stacked_w = torch.cat([dev0_w, dev1_w], dim=0)
        dn_mesh_list.append(
            ttnn.from_torch(
                stacked_w,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    logger.info(f"Created weights ({NUM_EXPERTS_PER_DEVICE} experts per device)")

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
        out_buf_list.append(
            ttnn.from_torch(
                torch.zeros(1, 1, P, D_padded, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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

    # Build combine metadata
    out_buf_addrs = []
    for ob in out_buf_list:
        dev_tensors = ttnn.get_device_tensors(ob)
        out_buf_addrs.append(dev_tensors[0].buffer_address())

    def make_combine_meta():
        meta = []
        for e in range(NUM_EXPERTS_PER_DEVICE):
            meta.append(out_buf_addrs[e])
            meta.append(P)
            meta.extend(all_rows)
            meta.extend([w_bf16s[e]] * P)
        return meta

    per_device_combine_metadata = [
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
        pack_combine_metadata(make_combine_meta(), NUM_EXPERTS_PER_DEVICE),
    ]
    logger.info(f"Combine metadata: K=4, {len(per_device_combine_metadata[0])} packed args per device")

    # Run op
    logger.info("Calling prefill_moe_compute with K=4, P=64...")
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

    # Read results
    result_dev_tensors = ttnn.get_device_tensors(result)
    dev0_result = ttnn.to_torch(result_dev_tensors[0]).squeeze().float()[:, :D]
    dev1_result = ttnn.to_torch(result_dev_tensors[1]).squeeze().float()[:, :D]

    dev_outs = [[], []]
    for ob in out_buf_list:
        devs = ttnn.get_device_tensors(ob)
        dev_outs[0].append(ttnn.to_torch(devs[0]).squeeze().float()[:, :D])
        dev_outs[1].append(ttnn.to_torch(devs[1]).squeeze().float()[:, :D])

    gu_dequant = [[], []]
    dn_dequant = [[], []]
    for e in range(NUM_EXPERTS_PER_DEVICE):
        gu_devs = ttnn.get_device_tensors(gu_mesh_list[e])
        dn_devs = ttnn.get_device_tensors(dn_mesh_list[e])
        gu_dequant[0].append(ttnn.to_torch(gu_devs[0]).squeeze().float())
        gu_dequant[1].append(ttnn.to_torch(gu_devs[1]).squeeze().float())
        dn_dequant[0].append(ttnn.to_torch(dn_devs[0]).squeeze().float())
        dn_dequant[1].append(ttnn.to_torch(dn_devs[1]).squeeze().float())

    # Expected pkt_buf (asymmetric 40/24)
    dev0_expected_tokens = list(range(40)) + list(range(64, 88))
    dev1_expected_tokens = list(range(88, 128)) + list(range(40, 64))

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

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

    refs = [[], []]
    for d in range(2):
        pkt_hs = dev0_pkt_hs if d == 0 else dev1_pkt_hs
        for e in range(NUM_EXPERTS_PER_DEVICE):
            refs[d].append(compute_expert_ref(pkt_hs, gu_dequant[d][e], dn_dequant[d][e]))

    w_quants = [torch.tensor(w, dtype=torch.bfloat16).float().item() for w in EXPERT_WEIGHTS]
    ref_combined = [torch.zeros(P, D, dtype=torch.float32) for _ in range(2)]
    for d in range(2):
        for e in range(NUM_EXPERTS_PER_DEVICE):
            ref_combined[d] += w_quants[e] * refs[d][e]

    # PCC checks
    all_passed = True
    for d in range(2):
        for e in range(NUM_EXPERTS_PER_DEVICE):
            p = pcc(dev_outs[d][e], refs[d][e])
            global_e = e + d * NUM_EXPERTS_PER_DEVICE
            logger.info(f"Dev{d} Expert {global_e} (full {P}-row): PCC={p:.6f}")
            if p < PCC_THRESHOLD:
                logger.error(f"FAIL: Dev{d} Expert {global_e} PCC {p:.6f}")
                all_passed = False

    for d in range(2):
        p = pcc([dev0_result, dev1_result][d], ref_combined[d])
        logger.info(f"Dev{d} Combined (4-way weighted sum, P={P}): PCC={p:.6f}")
        if p < PCC_THRESHOLD:
            logger.error(f"FAIL: Dev{d} Combined PCC {p:.6f}")
            all_passed = False

    assert all_passed, "One or more PCC checks failed"
    logger.info("test_k4_p64 PASSED")


if __name__ == "__main__":
    test_k4_p64()
