#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test Step 5a: Local return only (single device, 4 experts, K=1).

No fabric dispatch. All 32 tokens go into pkt_buf, all 4 experts compute
on the same 32 tokens, and the return kernel gathers the correct expert
result for each token into a ROW_MAJOR output.

Routing: token t -> expert (t % 4), weight = 1.0
Output: ROW_MAJOR [1, 1, 32, 2880]

Run:
    cd /data/sraizada_2/tt-metal
    python tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_local_return.py
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
NUM_EXPERTS = 4
NUM_CORES = 15
GRID_X = 5
GRID_Y = 3
BF16_TILE_BYTES = 2048


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu."""
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def test_local_return():
    device = ttnn.open_device(device_id=0)
    try:
        _run_test(device)
    finally:
        ttnn.close_device(device)


def _run_test(device):
    torch.manual_seed(42)

    # ---- Tile dimension calculations ----
    D_tiles = D // TILE  # 90
    n_weight_tiles_gu = D_FF // TILE  # 180
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES  # 12
    n_out_per_core = n_weight_per_core_gu // 2  # 6
    cols_per_core = n_weight_per_core_gu * TILE  # 384
    half_cols = n_out_per_core * TILE  # 192
    n_tiles_dn = D // TILE  # 90
    n_per_core_dn = n_tiles_dn // NUM_CORES  # 6
    D_padded = n_tiles_dn * TILE  # 2880
    D_FF_HALF_padded = n_out_per_core * NUM_CORES * TILE  # 2880

    # ---- Generate data ----
    all_hs = torch.randn(P, D, dtype=torch.bfloat16)
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

    # ---- Create device tensors ----

    # hidden_states (TILE_LAYOUT)
    hs_tile = ttnn.from_torch(
        all_hs.unsqueeze(0).unsqueeze(0),  # [1, 1, P, D]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created hidden_states (TILE)")

    # gate_up_weights: 4 separate tensors
    gate_up_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            shuffled_ws[e].unsqueeze(0).unsqueeze(0),  # [1, 1, D, D_FF]
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_tensors.append(t)
    logger.info(f"Created {NUM_EXPERTS} gate_up_weight tensors")

    # down_weights: 4 separate tensors
    down_tensors = []
    for e in range(NUM_EXPERTS):
        t = ttnn.from_torch(
            down_ws_raw[e].unsqueeze(0).unsqueeze(0),  # [1, 1, D_FF_HALF, D]
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_tensors.append(t)
    logger.info(f"Created {NUM_EXPERTS} down_weight tensors")

    # Scratch buffers
    pkt_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inter_buf = ttnn.from_torch(
        torch.zeros(1, 1, P, D_FF_HALF_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_bufs = []
    for e in range(NUM_EXPERTS):
        ob = ttnn.from_torch(
            torch.zeros(1, 1, P, D_padded, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_bufs.append(ob)
    logger.info(f"Created scratch buffers (pkt_buf, inter_buf, {NUM_EXPERTS} out_bufs)")

    # Output: ROW_MAJOR for fabric return mode
    output = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Created output tensor (ROW_MAJOR)")

    # ---- Build return_metadata ----
    # Routing: token t -> expert (t % NUM_EXPERTS), all local, weight=1.0
    # Format (flat, appended after factory prepends output_addr/D_tiles/D_bytes/num_experts):
    #   Per expert e:
    #     out_buf_addr, M_e,
    #     [src_row, dest_device, dest_token_index, weight_bf16, recv_slot_id] * M_e
    w_1_bf16 = int(torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    return_meta = []
    for e in range(NUM_EXPERTS):
        out_buf_addr = out_bufs[e].buffer_address()
        tokens = [t for t in range(P) if t % NUM_EXPERTS == e]
        return_meta.append(out_buf_addr)
        return_meta.append(len(tokens))  # M_e
        for t in tokens:
            return_meta.extend(
                [
                    t,  # src_row (row in out_buf — same as token index since all see same pkt_buf)
                    0,  # dest_device (0 = local)
                    t,  # dest_token_index (row in output)
                    w_1_bf16,  # weight (1.0)
                    0,  # recv_slot_id (unused for local tokens)
                ]
            )

    logger.info(f"Return metadata: {len(return_meta)} uint32 values")
    for e in range(NUM_EXPERTS):
        tokens = [t for t in range(P) if t % NUM_EXPERTS == e]
        logger.info(f"  Expert {e}: {len(tokens)} tokens, rows {tokens}")

    # ---- Create metadata DRAM tensor ----
    # Pack uint32 metadata as BF16 (bitcast: each uint32 -> 2 BF16 values)
    meta_int32 = torch.tensor(return_meta, dtype=torch.int32)
    meta_bf16 = meta_int32.view(torch.bfloat16)  # [N*2]
    meta_tensor = ttnn.from_torch(
        meta_bf16.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1, 1, 1, N*2]
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"Created metadata tensor: {len(return_meta)} uint32 words -> {meta_bf16.numel()} BF16 elems")

    # ---- Run the C++ op ----
    logger.info("Calling ttnn.experimental.prefill_moe_compute with fabric return...")
    # return_metadata: per-device list (single device = 1 element).
    # Append total_expected_remote=0 at the end.
    return_meta.append(0)  # total_expected_remote (0 for single device)

    result = ttnn.experimental.prefill_moe_compute(
        hs_tile,
        gate_up_weights=gate_up_tensors,
        down_weights=down_tensors,
        pkt_buf=pkt_buf,
        inter_buf=inter_buf,
        out_bufs=out_bufs,
        output=output,
        per_device_combine_metadata=[[]],  # no-op when fabric return is enabled
        num_experts=NUM_EXPERTS,
        num_cores=NUM_CORES,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        dispatch_metadata=[],  # not using fabric dispatch
        enable_fabric_return=True,
        return_metadata=[return_meta],  # per-device list
        return_metadata_tensor=meta_tensor,
    )
    ttnn.synchronize_device(device)
    logger.info("Op completed, reading back results...")

    # ---- Read back results ----
    result_torch = ttnn.to_torch(result).squeeze().float()  # [P, D]

    # Also read back out_bufs to verify expert compute independently
    out_buf_torch = []
    for e in range(NUM_EXPERTS):
        obt = ttnn.to_torch(out_bufs[e]).squeeze().float()[:, :D]
        out_buf_torch.append(obt)

    # Read back dequantized weights for reference
    gu_dequant = [ttnn.to_torch(t).squeeze().float() for t in gate_up_tensors]
    dn_dequant = [ttnn.to_torch(t).squeeze().float() for t in down_tensors]

    # ---- Compute reference ----
    all_hs_f = all_hs.float()
    ref_per_expert = []

    for e in range(NUM_EXPERTS):
        # All P tokens through expert e's MLP (same as hardware)
        ref_gu = all_hs_f @ gu_dequant[e]
        ref_inter = torch.empty(P, D_FF_HALF_padded, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out_e = ref_inter.bfloat16().float() @ dn_dequant[e]
        ref_per_expert.append(ref_out_e[:, :D])

    # Verify out_bufs (expert compute correctness)
    for e in range(NUM_EXPERTS):
        pcc_e = torch.corrcoef(torch.stack([out_buf_torch[e].flatten(), ref_per_expert[e].flatten()]))[0, 1].item()
        logger.info(f"Expert {e} out_buf PCC: {pcc_e:.6f}")
        assert pcc_e >= 0.96, f"Expert {e} out_buf PCC {pcc_e:.6f} < 0.96"

    # Build reference output: token t uses expert (t % NUM_EXPERTS)
    ref_output = torch.zeros(P, D, dtype=torch.float32)
    for t in range(P):
        e = t % NUM_EXPERTS
        ref_output[t] = ref_per_expert[e][t]

    # Verify return kernel output
    result_trimmed = result_torch[:, :D]
    overall_pcc = torch.corrcoef(torch.stack([result_trimmed.flatten(), ref_output.flatten()]))[0, 1].item()
    logger.info(f"Overall output PCC: {overall_pcc:.6f}")

    # Per-expert return PCC
    for e in range(NUM_EXPERTS):
        mask = [t for t in range(P) if t % NUM_EXPERTS == e]
        hw_rows = result_trimmed[mask].flatten()
        ref_rows = ref_output[mask].flatten()
        e_pcc = torch.corrcoef(torch.stack([hw_rows, ref_rows]))[0, 1].item()
        logger.info(f"Expert {e} return PCC: {e_pcc:.6f}")

    assert overall_pcc >= 0.96, f"Overall PCC {overall_pcc:.6f} < 0.96"
    logger.info("test_local_return PASSED")


if __name__ == "__main__":
    test_local_return()
