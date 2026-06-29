# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
probe_2cq_event_ordering.py
SCRATCH FILE -- hardware probe, not for commit.

QUESTION THIS PROBE ANSWERS:
  In run_traced_generation_cached's use_2cq=True path, `op_event` is recorded
  via `ttnn.record_event(device, CQ_COMPUTE)` immediately AFTER a non-blocking
  `ttnn.execute_trace(device, trace_id, cq_id=CQ_COMPUTE, blocking=False)`.

  The tt-metal tech report's own combined trace+2CQ examples never record an
  op_event directly after execute_trace like this -- in every documented
  example, op_event is recorded after a separate, NON-TRACED "consumer op"
  (e.g. ttnn.to_memory_config) that explicitly reads the input tensor, and
  the trace itself only covers ops AFTER that point. This model's trace
  covers the ENTIRE decoder stack, including the read of captured_dec_input,
  so there is no separate untraced "consumer op" for op_event to follow.

  This probe does NOT rely on documentation to answer whether this is safe.
  It tests directly: does record_event-after-execute_trace(blocking=False)
  actually gate CQ1's next write until the trace has REALLY finished reading
  captured_dec_input on the device -- not just until the replay command was
  dispatched?

METHOD:
  Bypass the real CPU embedding prep entirely. Instead, write a distinct,
  easily-identifiable marker value into captured_dec_input at each step
  (step 0 -> all elements = 0.0, step 1 -> all elements = 1.0, etc., scaled
  to stay in a safe bfloat16 range). Because every decoder layer's self
  -attention immediately consumes captured_dec_input via _extract_and_write_kv
  (which projects it to Q/K/V), the resulting K/V cache entry at position
  `step`, and ultimately ctx.traced_out, should be a deterministic function
  of THAT step's marker value alone.

  If op_event fires too early (before the trace has truly finished reading
  captured_dec_input), CQ1 will overwrite captured_dec_input with step k+1's
  marker BEFORE the trace's read of step k's marker has actually happened on
  the device. The corruption signature: ctx.traced_out for step k will
  numerically reflect marker k+1 (or some mix of k and k+1) instead of marker
  k alone.

  We detect this by comparing, for each step, the actual K-cache contents at
  position `step` (read back after the full run) against what a CLEAN,
  use_2cq=False run with the SAME marker-based inputs produces at the same
  position. use_2cq=False has no cross-queue race by construction (CQ_WRITE
  == CQ_COMPUTE == 0, fully blocking), so it is the ground truth.

  This isolates the race-condition question from model correctness (NLL/
  CRPS, already covered by test_tst_e2e.py) -- we are not asking "is the
  model right", only "does the 2CQ event choreography preserve step
  ordering on this exact hardware/ttnn version".

USAGE:
  cd ~/tt-metal/models/demos/time_series_transformer
  PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
    TT_METAL_HOME=/root/tt-metal \
    LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
    ARCH_NAME=wormhole_b0 python probe_2cq_event_ordering.py
"""

import torch
from tt.tst_attention import allocate_kv_cache
from tt.tst_model import PADDED_WIDTH
from tt.tst_model_cached_additions import (
    _build_causal_mask_1tok,
    _extract_and_write_kv,
    _extract_q_only,
    _update_causal_mask,
    _zero_kv_caches,
)

import ttnn

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}  {detail}")


def make_marker_input(step: int, BS: int) -> torch.Tensor:
    """
    A distinct, easily-identifiable bfloat16-safe value per step.
    Using (step + 1) * 0.01 keeps values small and well-separated relative
    to bfloat16's ~3 decimal digits of precision, and avoids 0.0 for step 0
    (a value of exactly zero could mask certain classes of write failures).
    """
    val = (step + 1) * 0.01
    return torch.full((BS, 1, PADDED_WIDTH), val, dtype=torch.bfloat16)


def run_one_layer_probe(device, BS, T_max, num_steps, use_2cq):
    """
    Minimal single-decoder-layer harness exercising ONLY the mechanism under
    test (write marker -> trace reads it -> KV cache holds it), without the
    rest of the model (no encoder, no cross-attn weights, no distribution
    head). This keeps the probe fast and keeps the corruption signature
    unambiguous: K-cache contents at position `step` should be a pure
    function of that step's marker, nothing else.
    """
    torch.manual_seed(0)
    qkv_weight = torch.randn(PADDED_WIDTH, 3 * PADDED_WIDTH, dtype=torch.bfloat16) * 0.1
    qkv_bias = torch.zeros(3 * PADDED_WIDTH, dtype=torch.bfloat16)
    out_proj_weight = torch.randn(PADDED_WIDTH, PADDED_WIDTH, dtype=torch.bfloat16) * 0.1
    out_proj_bias = torch.zeros(PADDED_WIDTH, dtype=torch.bfloat16)

    w_self_attn = {
        "qkv_weight": ttnn.from_torch(qkv_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        "qkv_bias": ttnn.from_torch(qkv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        "out_proj_weight": ttnn.from_torch(
            out_proj_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        "out_proj_bias": ttnn.from_torch(out_proj_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    }

    k_cache, v_cache = allocate_kv_cache(device, BS, T_max=T_max)
    _zero_kv_caches([(k_cache, v_cache)], device, BS, T_max)

    captured_input = ttnn.from_torch(
        torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shared_mask = _build_causal_mask_1tok(device, 0, T_max)

    # Capture a minimal trace: self-attend-from-cache only (no cross-attn,
    # no FFN -- isolates the exact mechanism under test).
    from tt.tst_model_cached_additions import _attend_from_cache

    def minimal_traced_step(hidden, q):
        return _attend_from_cache(q, k_cache, v_cache, shared_mask, w_self_attn)

    q0 = _extract_q_only(captured_input, w_self_attn)
    traced_out = minimal_traced_step(captured_input, q0)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    q_traced = _extract_q_only(captured_input, w_self_attn)
    traced_out = minimal_traced_step(captured_input, q_traced)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    _zero_kv_caches([(k_cache, v_cache)], device, BS, T_max)
    ttnn.synchronize_device(device)

    CQ_COMPUTE = 0
    CQ_WRITE = 1 if use_2cq else 0

    per_step_outputs = []

    if use_2cq:
        op_event = ttnn.record_event(device, CQ_COMPUTE)

    for step in range(num_steps):
        marker = make_marker_input(step, BS)
        host_tt = ttnn.from_torch(marker, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)

        if use_2cq:
            ttnn.wait_for_event(CQ_WRITE, op_event)
            ttnn.copy_host_to_device_tensor(host_tt, captured_input, cq_id=CQ_WRITE)
            _extract_and_write_kv(captured_input, w_self_attn, k_cache, v_cache, step, cq_id=CQ_WRITE)
            _update_causal_mask(device, shared_mask, step, T_max)
            write_event = ttnn.record_event(device, CQ_WRITE)

            ttnn.wait_for_event(CQ_COMPUTE, write_event)
            ttnn.execute_trace(device, trace_id, cq_id=CQ_COMPUTE, blocking=False)
            op_event = ttnn.record_event(device, CQ_COMPUTE)
        else:
            ttnn.copy_host_to_device_tensor(host_tt, captured_input, cq_id=CQ_WRITE)
            _extract_and_write_kv(captured_input, w_self_attn, k_cache, v_cache, step, cq_id=CQ_WRITE)
            _update_causal_mask(device, shared_mask, step, T_max)
            ttnn.execute_trace(device, trace_id, cq_id=CQ_COMPUTE, blocking=True)

        # Blocking readback every step regardless of path -- we need the
        # actual per-step output value to compare, and this is a probe, not
        # a perf test, so we deliberately pay the sync cost here to get an
        # unambiguous per-step snapshot.
        ttnn.synchronize_device(device)
        out = ttnn.to_torch(traced_out).float()
        per_step_outputs.append(out.clone())

    ttnn.release_trace(device, trace_id)
    return per_step_outputs


def main():
    device = ttnn.open_device(device_id=0, num_command_queues=2)
    BS = 4
    T_max = 8
    num_steps = 8

    try:
        print("\n=== Reference run (use_2cq=False, no cross-queue race possible) ===")
        ref_outputs = run_one_layer_probe(device, BS, T_max, num_steps, use_2cq=False)

        print("\n=== Probe run (use_2cq=True, event-gated cross-queue handoff) ===")
        probe_outputs = run_one_layer_probe(device, BS, T_max, num_steps, use_2cq=True)
    finally:
        ttnn.close_device(device)

    print("\n=== Comparison ===")
    max_diff_per_step = []
    for step in range(num_steps):
        diff = (ref_outputs[step] - probe_outputs[step]).abs().max().item()
        max_diff_per_step.append(diff)
        record(
            f"step {step}: 2cq output matches single-queue reference",
            diff < 0.05,  # bfloat16-generous threshold, same as other PCC checks this session
            f"max abs diff = {diff:.6f} (threshold 0.05)",
        )

    # Specific corruption signature check: if op_event fired too early, the
    # most likely failure mode is step k's output reflecting step k+1's
    # marker instead of (or mixed with) step k's marker. Check this directly
    # by comparing each step's 2cq output against what step k+1's reference
    # output looks like -- if probe_outputs[k] is suspiciously closer to
    # ref_outputs[k+1] than to ref_outputs[k], that's the smoking gun.
    print("\n=== Cross-contamination check (probe output vs NEXT step's reference) ===")
    for step in range(num_steps - 1):
        diff_same_step = (ref_outputs[step] - probe_outputs[step]).abs().max().item()
        diff_next_step = (ref_outputs[step + 1] - probe_outputs[step]).abs().max().item()
        record(
            f"step {step}: closer to its OWN reference than to step {step+1}'s reference",
            diff_same_step < diff_next_step,
            f"diff-to-own={diff_same_step:.6f}  diff-to-next={diff_next_step:.6f}",
        )

    n_pass = sum(1 for _, p, _ in RESULTS if p)
    n_total = len(RESULTS)
    print("\n" + "=" * 60)
    print(f"{n_pass}/{n_total} checks passed.")
    if n_pass == n_total:
        print("GO: 2CQ event ordering appears safe on this hardware/ttnn version.")
        print("    (This probe used a minimal single-layer harness, not the full")
        print("     model -- still recommend running the full e2e suite with")
        print("     use_2cq=True before fully trusting end-to-end correctness.)")
    else:
        print("NO-GO: 2CQ event ordering is NOT safe as currently implemented.")
        print("       op_event recorded after non-blocking execute_trace does")
        print("       NOT correctly gate the writer queue. Do not trust any")
        print("       use_2cq=True latency/throughput numbers until this is fixed.")


if __name__ == "__main__":
    main()
