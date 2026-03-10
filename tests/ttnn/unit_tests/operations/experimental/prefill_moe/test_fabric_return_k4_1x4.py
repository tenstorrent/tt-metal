#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test Step 5 on 1x4: Pure DM fabric return with K=4, pipelined, 4 devices.

Each device computes 4 experts on its 32 local tokens. Every token is routed
to all 4 experts. The return kernel is pure data movement — it gathers raw
(unscaled) expert results into an output of shape [NUM_EXPERTS, 1, P, D],
with one page per (expert, token) pair.

Scaling and accumulation happen externally (on CPU in this test, on compute
cores via ttnn.mul + ttnn.sum in the model pipeline), matching the
selective_reduce_combine pattern.

Routing: token t -> experts {0, 1, 2, 3}
  - Tokens 0-7 on each device:  local return (dest_device = self)
  - Tokens 8-15:  dest_device = (self + 1) % 4
  - Tokens 16-23: dest_device = (self + 2) % 4
  - Tokens 24-31: dest_device = (self + 3) % 4

Output layout: [NUM_EXPERTS, 1, P, D] ROW_MAJOR
  Page index: e * P + dest_token

Verification:
  1. Raw expert output slots match reference (per-expert PCC)
  2. CPU-side weighted sum matches reference (end-to-end PCC)

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_fabric_return_k4_1x4.py
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
K = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
N_DEVICES = 4
LOCAL_SPLIT = 8  # tokens [0,8) local, rest remote
GROUP_SIZE = P // N_DEVICES  # 8 tokens per destination group


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


def get_dest_device(src_dev, token_idx):
    """Determine destination device for a token based on its index."""
    group = token_idx // GROUP_SIZE
    return (src_dev + group) % N_DEVICES


def get_source_device(dest_dev, token_idx):
    """Determine which source device sent results for this output row."""
    group = token_idx // GROUP_SIZE
    return (dest_dev - group) % N_DEVICES


def test_fabric_return_k4_1x4():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    submesh = full_mesh.create_submesh(ttnn.MeshShape(1, N_DEVICES), ttnn.MeshCoordinate(0, 0))
    try:
        _run_test(submesh)
    finally:
        for sm in full_mesh.get_submeshes():
            ttnn.close_mesh_device(sm)
        ttnn.close_mesh_device(full_mesh)


def _run_test(mesh_device):
    torch.manual_seed(42)

    # ---- Generate per-device hidden_states ----
    hs_per_dev = [torch.randn(P, D, dtype=torch.bfloat16) for _ in range(N_DEVICES)]

    # ---- Generate K=4 routing weights ----
    raw_weights = torch.randn(P, NUM_EXPERTS, dtype=torch.float32)
    weights_float = torch.softmax(raw_weights, dim=1)
    weights_bf16 = weights_float.bfloat16()
    weights_bf16_float = weights_bf16.float()

    logger.info("Routing weights (BF16 rounded, first 4 tokens):")
    for t in range(4):
        ws = [f"{weights_bf16_float[t, e]:.4f}" for e in range(NUM_EXPERTS)]
        logger.info(f"  Token {t}: [{', '.join(ws)}] sum={weights_bf16_float[t].sum():.4f}")

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
    stacked_hs = torch.stack([h.unsqueeze(0) for h in hs_per_dev])  # [4,1,P,D]
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
    # out_bufs: ROW_MAJOR fragment pages for writer-side untilize (Phase 3).
    # Shape [1, 1, P*NUM_CORES, n_per_core_dn*TILE] where each row = one fragment page.
    # Page(r, c) = r * NUM_CORES + c, page_size = n_per_core_dn * TILE * 2 = 384 bytes.
    n_per_core_dn = (D // TILE) // NUM_CORES  # 90 / 15 = 6
    out_bufs = []
    for e in range(NUM_EXPERTS):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P * NUM_CORES, n_per_core_dn * TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_bufs.append(ob)

    # Output: ROW_MAJOR, replicated, zero-initialized
    # Shape [NUM_EXPERTS, 1, P, D] — one page per (expert, token) pair
    output = ttnn.from_torch(
        torch.zeros(NUM_EXPERTS, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created all mesh tensors")

    # ---- Allocate recv_staging_buf ----
    # Each device receives from 3 other devices:
    # (P - LOCAL_SPLIT) * NUM_EXPERTS = 24 * 4 = 96 remote results
    total_remote_per_dev = (P - LOCAL_SPLIT) * NUM_EXPERTS  # 96
    recv_staging_buf = ttnn.from_torch(
        torch.zeros(1, 1, total_remote_per_dev, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"Created recv_staging_buf: {total_remote_per_dev} slots per device")

    # ---- Build per-device return_metadata ----
    # New format: 4 fields per token (no weight — scaling done externally)
    #   [src_row, dest_device, dest_page, recv_slot_id]
    # where dest_page = e * P + dest_token
    out_buf_dev_tensors = [ttnn.get_device_tensors(ob) for ob in out_bufs]
    out_buf_addrs = [out_buf_dev_tensors[e][0].buffer_address() for e in range(NUM_EXPERTS)]

    # Global per-destination-device slot counters
    dest_slot_counters = {d: 0 for d in range(N_DEVICES)}

    return_metadata = []
    for d in range(N_DEVICES):
        dev_meta = []

        for e in range(NUM_EXPERTS):
            dev_meta.append(out_buf_addrs[e])
            # K=4: every token visits every expert
            expert_tokens = list(range(P))
            dev_meta.append(len(expert_tokens))  # M_e = P = 32

            for t in expert_tokens:
                src_row = t
                dest_device = get_dest_device(d, t)
                dest_page = e * P + t  # page index in [NUM_EXPERTS, 1, P, D] output

                if dest_device == d:
                    # Local
                    recv_slot_id = 0  # unused for local
                else:
                    # Remote
                    recv_slot_id = dest_slot_counters[dest_device]
                    dest_slot_counters[dest_device] += 1

                dev_meta.extend([src_row, dest_device, dest_page, recv_slot_id])

        # total_expected_remote for this device
        dev_meta.append(total_remote_per_dev)

        return_metadata.append(dev_meta)
        logger.info(f"Device {d}: {len(dev_meta)} metadata values, " f"{total_remote_per_dev} expected remote")

    # Verify slot counters
    for d in range(N_DEVICES):
        logger.info(f"dest_slot_counters[{d}] = {dest_slot_counters[d]}")
        assert dest_slot_counters[d] == total_remote_per_dev, (
            f"Expected {total_remote_per_dev} slots for device {d}, " f"got {dest_slot_counters[d]}"
        )

    # ---- Create metadata DRAM tensor ----
    meta_words_per_dev = [rm[:-1] for rm in return_metadata]
    max_meta_words = max(len(mw) for mw in meta_words_per_dev)

    stacked_int32 = torch.zeros(N_DEVICES, 1, 1, max_meta_words, dtype=torch.int32)
    for d in range(N_DEVICES):
        words = meta_words_per_dev[d]
        stacked_int32[d, 0, 0, : len(words)] = torch.tensor(words, dtype=torch.int32)

    stacked_bf16 = stacked_int32.view(torch.bfloat16)
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
    logger.info("Calling ttnn.experimental.prefill_moe_compute with K=4 pure DM return on 1x4...")
    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=gate_up_tensors,
        down_weights=down_tensors,
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=out_bufs,
        output=output,
        per_device_combine_metadata=[[] for _ in range(N_DEVICES)],
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
    # Result shape: [NUM_EXPERTS, 1, P, D] per device
    result_devs = [
        ttnn.to_torch(t).squeeze(1).float()[:, :, :D] for t in ttnn.get_device_tensors(result)  # [NUM_EXPERTS, P, D]
    ]

    # Read back dequantized weights (from device 0 -- same on all due to replication)
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

    # ---- Verify raw output slots (return kernel data movement correctness) ----
    # Each slot output[e, :, t, :] on device d should contain the raw (unscaled)
    # expert e result for token t, computed on get_source_device(d, t).
    for d in range(N_DEVICES):
        for e in range(NUM_EXPERTS):
            # Build expected raw output for expert e on device d
            expected_rows = []
            for t in range(P):
                source_dev = get_source_device(d, t)
                expected_rows.append(ref_per_dev[source_dev][e][t])
            expected = torch.stack(expected_rows)  # [P, D]

            actual = result_devs[d][e]  # [P, D]
            slot_pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
            logger.info(f"Device {d} Expert {e} raw output slot PCC: {slot_pcc:.6f}")
            assert slot_pcc >= 0.96, f"Device {d} Expert {e} raw output slot PCC {slot_pcc:.6f} < 0.96"

    # ---- Verify weighted sum (end-to-end pipeline correctness) ----
    # Simulate what ttnn.mul + ttnn.sum would do on compute cores
    for d in range(N_DEVICES):
        # Build reference weighted sum
        ref_output = torch.zeros(P, D, dtype=torch.float32)
        for t in range(P):
            source_dev = get_source_device(d, t)
            for e in range(NUM_EXPERTS):
                ref_output[t] += weights_bf16_float[t, e] * ref_per_dev[source_dev][e][t]

        # Compute weighted sum from actual raw output
        actual_output = torch.zeros(P, D, dtype=torch.float32)
        for e in range(NUM_EXPERTS):
            actual_output += weights_bf16_float[:, e : e + 1] * result_devs[d][e]

        # Overall PCC per device
        overall_pcc = torch.corrcoef(torch.stack([actual_output.flatten(), ref_output.flatten()]))[0, 1].item()
        logger.info(f"Device {d} weighted sum PCC: {overall_pcc:.6f}")

        # Local PCC (tokens 0..LOCAL_SPLIT-1)
        local_mask = list(range(LOCAL_SPLIT))
        local_pcc = torch.corrcoef(
            torch.stack(
                [
                    actual_output[local_mask].flatten(),
                    ref_output[local_mask].flatten(),
                ]
            )
        )[0, 1].item()
        logger.info(f"Device {d} local weighted PCC: {local_pcc:.6f}")

        # Remote PCC (tokens LOCAL_SPLIT..P-1)
        remote_mask = list(range(LOCAL_SPLIT, P))
        remote_pcc = torch.corrcoef(
            torch.stack(
                [
                    actual_output[remote_mask].flatten(),
                    ref_output[remote_mask].flatten(),
                ]
            )
        )[0, 1].item()
        logger.info(f"Device {d} remote weighted PCC: {remote_pcc:.6f}")

        # Per-token diagnostics for first few tokens
        for t in range(min(4, P)):
            token_pcc = torch.corrcoef(torch.stack([actual_output[t], ref_output[t]]))[0, 1].item()
            ws = [f"{weights_bf16_float[t, e]:.3f}" for e in range(NUM_EXPERTS)]
            src = get_source_device(d, t)
            logger.info(
                f"  Device {d} token {t} weighted PCC: {token_pcc:.6f} " f"weights=[{','.join(ws)}] src_dev={src}"
            )

        assert overall_pcc >= 0.96, f"Device {d} weighted sum PCC {overall_pcc:.6f} < 0.96"

    logger.info("test_fabric_return_k4_1x4 PASSED")


if __name__ == "__main__":
    test_fabric_return_k4_1x4()
