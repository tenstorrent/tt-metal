#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test Step 5b: Fabric return on 1x2 mesh (2 devices, 4 experts, K=1).

Each device computes 4 experts on its 32 local tokens. The return kernel
sends results for tokens 16-31 to the other device via fabric unicast,
and writes results for tokens 0-15 locally.

Routing: token t -> expert (t % 4)
  - Tokens 0-15 on each device: local return (dest_device = self)
  - Tokens 16-31 on each device: remote return (dest_device = other)

Each device's output:
  - Rows 0-15: from local expert compute
  - Rows 16-31: from remote device's expert compute

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_return_1x2.py
"""

import torch
import ttnn
from loguru import logger

TILE = 32
P = 32
D = 2880
D_FF = 5760
D_FF_HALF = D_FF // 2
NUM_EXPERTS = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
N_DEVICES = 2
LOCAL_SPLIT = 16  # tokens 0..15 local, 16..31 remote


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def compute_expert_reference(hs_f, gu_dequant, dn_dequant, num_cores):
    """Compute full expert MLP reference matching hardware."""
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // num_cores
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = (n_weight_per_core_gu // 2) * TILE
    n_out_per_core = n_weight_per_core_gu // 2
    D_FF_HALF_padded = n_out_per_core * num_cores * TILE

    ref_gu = hs_f @ gu_dequant
    ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
    for c in range(num_cores):
        g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
        u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
    return (ref_inter.bfloat16().float() @ dn_dequant)[:, :D]


def test_fabric_return_1x2():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    submesh = full_mesh.create_submesh(ttnn.MeshShape(1, N_DEVICES), ttnn.MeshCoordinate(0, 0))
    try:
        _run_test(submesh)
    finally:
        ttnn.close_mesh_device(full_mesh)


def _run_test(mesh_device):
    torch.manual_seed(42)

    # ---- Generate per-device hidden_states ----
    hs_per_dev = [torch.randn(P, D, dtype=torch.bfloat16) for _ in range(N_DEVICES)]

    # ---- Generate weights (replicated across devices) ----
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE

    gate_up_ws_raw = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws_raw = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

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

    # ---- Create mesh tensors ----

    # hidden_states: sharded (each device gets different data)
    stacked_hs = torch.stack([h.unsqueeze(0).unsqueeze(0) for h in hs_per_dev])  # [2,1,1,P,D]
    stacked_hs = stacked_hs.squeeze(1)  # [2,1,P,D]
    hs_tile = ttnn.from_torch(
        stacked_hs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created sharded hidden_states")

    # Weights: replicated
    gate_up_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            shuffled_ws[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_tensors.append(t)

    down_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            down_ws_raw[e].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_tensors.append(t)
    logger.info(f"Created {NUM_EXPERTS} replicated weight tensors")

    # Scratch buffers: replicated
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE
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
    out_bufs = []
    for e in range(NUM_EXPERTS):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_bufs.append(ob)

    # Output: ROW_MAJOR, replicated
    output = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created all mesh tensors")

    # ---- Allocate recv_staging_buf ----
    # Each device receives up to LOCAL_SPLIT * NUM_EXPERTS / NUM_EXPERTS = LOCAL_SPLIT
    # remote tokens from the other device (16 tokens for 1x2)
    total_remote_per_dev = P - LOCAL_SPLIT  # 16
    recv_staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, total_remote_per_dev, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"Created recv_staging_buf: {total_remote_per_dev} slots")

    # ---- Build per-device return_metadata ----
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    # Get out_buf addresses (same on all devices due to deterministic allocation)
    out_buf_dev_tensors = [ttnn.get_device_tensors(ob) for ob in out_bufs]
    out_buf_addrs = [out_buf_dev_tensors[e][0].buffer_address() for e in range(NUM_EXPERTS)]

    # Global per-destination-device slot counters for recv_slot_id assignment
    dest_slot_counters = {d: 0 for d in range(N_DEVICES)}

    return_metadata = []
    for d in range(N_DEVICES):
        other_d = 1 - d
        dev_meta = []

        for e in range(NUM_EXPERTS):
            dev_meta.append(out_buf_addrs[e])
            # Expert e processes tokens where t % NUM_EXPERTS == e
            expert_tokens = [t for t in range(P) if t % NUM_EXPERTS == e]
            dev_meta.append(len(expert_tokens))  # M_e

            for t in expert_tokens:
                src_row = t
                if t < LOCAL_SPLIT:
                    dest_device = d  # local
                    dest_token = t
                    recv_slot_id = 0  # unused for local
                else:
                    dest_device = other_d  # remote
                    dest_token = t
                    recv_slot_id = dest_slot_counters[dest_device]
                    dest_slot_counters[dest_device] += 1
                dev_meta.extend([src_row, dest_device, dest_token, w_1_bf16, recv_slot_id])

        # total_expected_remote: how many results this device receives from others
        total_remote = sum(1 for e in range(NUM_EXPERTS) for t in range(P) if t % NUM_EXPERTS == e and t >= LOCAL_SPLIT)
        dev_meta.append(total_remote)

        return_metadata.append(dev_meta)
        logger.info(f"Device {d}: {len(dev_meta)} metadata values, {total_remote} expected remote")

    # ---- Create metadata DRAM tensor ----
    # Pack uint32 metadata as BF16 (bitcast: each uint32 -> 2 BF16 values)
    # Per-device metadata (excluding total_expected_remote at the end)
    meta_words_per_dev = [rm[:-1] for rm in return_metadata]
    max_meta_words = max(len(mw) for mw in meta_words_per_dev)

    # Stack per-device metadata, pad to same length
    stacked_int32 = torch.zeros(N_DEVICES, 1, 1, max_meta_words, dtype=torch.int32)
    for d in range(N_DEVICES):
        words = meta_words_per_dev[d]
        stacked_int32[d, 0, 0, : len(words)] = torch.tensor(words, dtype=torch.int32)

    stacked_bf16 = stacked_int32.view(torch.bfloat16)  # [N_DEVICES, 1, 1, max_meta_words*2]
    meta_tensor = ttnn.from_torch(
        stacked_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"Created metadata tensor: {max_meta_words} uint32 words per device")

    # ---- Run the op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with fabric return...")
    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=gate_up_tensors,
        down_weights=down_tensors,
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=out_bufs,
        output=output,
        per_device_combine_metadata=[[], []],
        num_experts=NUM_EXPERTS,
        num_cores=NUM_CORES,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        dispatch_metadata=[],
        enable_fabric_return=True,
        return_metadata=return_metadata,
        recv_staging_buf=recv_staging_buf,
        return_metadata_tensor=meta_tensor,
    )
    for dev_tensor in ttnn.get_device_tensors(result):
        ttnn.synchronize_device(dev_tensor.device())
    logger.info("Op completed")

    # ---- Read back results ----
    result_devs = [ttnn.to_torch(t).squeeze().float()[:, :D] for t in ttnn.get_device_tensors(result)]

    # Read back dequantized weights (from device 0 — same on both due to replication)
    gu_dequant = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).squeeze().float() for t in gate_up_tensors]
    dn_dequant = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).squeeze().float() for t in down_tensors]

    # ---- Compute reference per device per expert ----
    ref_per_dev = []  # ref_per_dev[d][e] = [P, D] expert output
    for d in range(N_DEVICES):
        hs_f = hs_per_dev[d].float()
        dev_refs = []
        for e in range(NUM_EXPERTS):
            ref = compute_expert_reference(hs_f, gu_dequant[e], dn_dequant[e], NUM_CORES)
            dev_refs.append(ref)
        ref_per_dev.append(dev_refs)

    # ---- Verify out_bufs (expert compute correctness) ----
    for d in range(N_DEVICES):
        for e in range(NUM_EXPERTS):
            ob = ttnn.to_torch(ttnn.get_device_tensors(out_bufs[e])[d]).squeeze().float()[:, :D]
            pcc = torch.corrcoef(torch.stack([ob.flatten(), ref_per_dev[d][e].flatten()]))[0, 1].item()
            logger.info(f"Device {d} Expert {e} out_buf PCC: {pcc:.6f}")
            assert pcc >= 0.96, f"Device {d} Expert {e} out_buf PCC {pcc:.6f} < 0.96"

    # ---- Verify output (return kernel correctness) ----
    for d in range(N_DEVICES):
        other_d = 1 - d

        # Local rows (0..LOCAL_SPLIT-1): from this device's expert compute
        for t in range(LOCAL_SPLIT):
            e = t % NUM_EXPERTS
            expected_row = ref_per_dev[d][e][t]
            actual_row = result_devs[d][t]
            pcc = torch.corrcoef(torch.stack([actual_row, expected_row]))[0, 1].item()
            if pcc < 0.96:
                logger.warning(f"Device {d} local row {t} PCC: {pcc:.6f}")

        # Remote rows (LOCAL_SPLIT..P-1): from other device's expert compute
        for t in range(LOCAL_SPLIT, P):
            e = t % NUM_EXPERTS
            expected_row = ref_per_dev[other_d][e][t]
            actual_row = result_devs[d][t]
            pcc = torch.corrcoef(torch.stack([actual_row, expected_row]))[0, 1].item()
            if pcc < 0.96:
                logger.warning(f"Device {d} remote row {t} PCC: {pcc:.6f}")

        # Overall PCC per device
        ref_output = torch.zeros(P, D, dtype=torch.float32)
        for t in range(P):
            e = t % NUM_EXPERTS
            if t < LOCAL_SPLIT:
                ref_output[t] = ref_per_dev[d][e][t]
            else:
                ref_output[t] = ref_per_dev[other_d][e][t]

        overall_pcc = torch.corrcoef(torch.stack([result_devs[d].flatten(), ref_output.flatten()]))[0, 1].item()
        logger.info(f"Device {d} overall output PCC: {overall_pcc:.6f}")

        # Local PCC
        local_mask = list(range(LOCAL_SPLIT))
        local_pcc = torch.corrcoef(
            torch.stack([result_devs[d][local_mask].flatten(), ref_output[local_mask].flatten()])
        )[0, 1].item()
        logger.info(f"Device {d} local PCC: {local_pcc:.6f}")

        # Remote PCC
        remote_mask = list(range(LOCAL_SPLIT, P))
        remote_pcc = torch.corrcoef(
            torch.stack([result_devs[d][remote_mask].flatten(), ref_output[remote_mask].flatten()])
        )[0, 1].item()
        logger.info(f"Device {d} remote PCC: {remote_pcc:.6f}")

        assert overall_pcc >= 0.96, f"Device {d} overall PCC {overall_pcc:.6f} < 0.96"

    logger.info("test_fabric_return_1x2 PASSED")


if __name__ == "__main__":
    test_fabric_return_1x2()
