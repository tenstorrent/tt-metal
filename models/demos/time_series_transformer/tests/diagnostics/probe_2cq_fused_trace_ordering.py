# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
import torch

import ttnn

NUM_HEADS = 2
HEAD_DIM_PADDED = 32
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED
T_MAX = 8
NUM_LAYERS = 2

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")


def make_marker_input(step, BS):
    val = (step + 1) * 0.01
    return torch.full((BS, 1, PADDED_WIDTH), val, dtype=torch.bfloat16)


def blend_step_k(k_cache, pos_onehot, new_k_buf):
    keep_mask = ttnn.rsub(pos_onehot, 1.0)
    ttnn.multiply(k_cache, keep_mask, output_tensor=k_cache)
    new_k_bcast = ttnn.multiply(new_k_buf, pos_onehot)
    ttnn.add(k_cache, new_k_bcast, output_tensor=k_cache)


def make_layer_weights(device):
    qkv_w = torch.randn(PADDED_WIDTH, 3 * PADDED_WIDTH, dtype=torch.bfloat16) * 0.1
    qkv_b = torch.zeros(3 * PADDED_WIDTH, dtype=torch.bfloat16)
    out_w = torch.randn(PADDED_WIDTH, PADDED_WIDTH, dtype=torch.bfloat16) * 0.1
    out_b = torch.zeros(PADDED_WIDTH, dtype=torch.bfloat16)
    return {
        "qkv_weight": ttnn.from_torch(qkv_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        "qkv_bias": ttnn.from_torch(qkv_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        "out_proj_weight": ttnn.from_torch(out_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        "out_proj_bias": ttnn.from_torch(out_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    }


def one_layer_fused(hidden_in, w, k_cache, pos_onehot, mask):
    fused_qkv = ttnn.linear(hidden_in, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    key_pos = ttnn.multiply(key, pos_onehot)
    blend_step_k(k_cache, pos_onehot, key_pos)
    k_tile = k_cache
    scores = ttnn.matmul(query, k_tile)
    scale = 13.0**-0.5
    scaled = ttnn.multiply(scores, scale)
    masked = ttnn.add(scaled, mask)
    probs = ttnn.softmax(masked, dim=-1)
    v_bcast = ttnn.repeat(value, (1, 1, T_MAX, 1))
    context = ttnn.matmul(probs, v_bcast)
    context = ttnn.transformer.concatenate_heads(context)
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


def run_fused_two_layer_probe(device, BS, num_steps, use_2cq):
    w0 = make_layer_weights(device)
    w1 = make_layer_weights(device)

    def zero_cache():
        return ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_MAX),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    k_cache0, k_cache1 = zero_cache(), zero_cache()
    captured_input = ttnn.from_torch(
        torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_onehot_row = ttnn.from_torch(
        torch.zeros(1, 1, 1, T_MAX), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    mask = ttnn.from_torch(torch.zeros(1, 1, 1, T_MAX), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def fused_two_layer(hidden_in):
        out0 = one_layer_fused(hidden_in, w0, k_cache0, pos_onehot_row, mask)
        out1 = one_layer_fused(out0, w1, k_cache1, pos_onehot_row, mask)
        return out1

    traced_out = fused_two_layer(captured_input)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_out = fused_two_layer(captured_input)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    for c in (k_cache0, k_cache1):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_MAX), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            c,
        )
    ttnn.synchronize_device(device)

    CQ_COMPUTE = 0
    CQ_WRITE = 1 if use_2cq else 0
    per_step_outputs = []

    if use_2cq:
        op_event = ttnn.record_event(device, CQ_COMPUTE)

    for step in range(num_steps):
        marker = make_marker_input(step, BS)
        host_tt = ttnn.from_torch(marker, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
        oh = torch.zeros(1, 1, 1, T_MAX)
        oh[..., step] = 1.0
        oh_tt = ttnn.from_torch(oh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)

        if use_2cq:
            ttnn.wait_for_event(CQ_WRITE, op_event)
            ttnn.copy_host_to_device_tensor(host_tt, captured_input, cq_id=CQ_WRITE)
            ttnn.copy_host_to_device_tensor(oh_tt, pos_onehot_row, cq_id=CQ_WRITE)
            write_event = ttnn.record_event(device, CQ_WRITE)
            ttnn.wait_for_event(CQ_COMPUTE, write_event)
            ttnn.execute_trace(device, trace_id, cq_id=CQ_COMPUTE, blocking=False)
            op_event = ttnn.record_event(device, CQ_COMPUTE)
        else:
            ttnn.copy_host_to_device_tensor(host_tt, captured_input, cq_id=CQ_WRITE)
            ttnn.copy_host_to_device_tensor(oh_tt, pos_onehot_row, cq_id=CQ_WRITE)
            ttnn.execute_trace(device, trace_id, cq_id=CQ_COMPUTE, blocking=True)

        ttnn.synchronize_device(device)
        per_step_outputs.append(ttnn.to_torch(traced_out).float().clone())

    ttnn.release_trace(device, trace_id)
    return per_step_outputs


def main():
    device = ttnn.open_device(device_id=0, num_command_queues=2, trace_region_size=1_000_000)
    BS, num_steps = 2, 6
    try:
        print("=== Reference (use_2cq=False) ===")
        ref = run_fused_two_layer_probe(device, BS, num_steps, use_2cq=False)
        print("=== Probe (use_2cq=True, single fused two-layer trace) ===")
        probe = run_fused_two_layer_probe(device, BS, num_steps, use_2cq=True)
    finally:
        ttnn.close_device(device)

    for step in range(num_steps):
        diff = (ref[step] - probe[step]).abs().max().item()
        record(f"step {step}: matches single-queue reference", diff < 0.05, f"max_abs_diff={diff:.6f}")
    for step in range(num_steps - 1):
        d_own = (ref[step] - probe[step]).abs().max().item()
        d_next = (ref[step + 1] - probe[step]).abs().max().item()
        record(
            f"step {step}: closer to own ref than to step {step+1}'s ref",
            d_own < d_next,
            f"own={d_own:.6f} next={d_next:.6f}",
        )

    n_pass = sum(1 for _, p, _ in RESULTS if p)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed.")
    print("GO (template-level)" if n_pass == len(RESULTS) else "NO-GO")


if __name__ == "__main__":
    main()
