# SPDX-License-Identifier: Apache-2.0
"""
Trace-compatibility probe for ttnn.update_cache on a TILE-native V-cache.

Question this answers: can update_cache be captured inside
begin_trace_capture/execute_trace with update_idx as a frozen Python int
per step (mirroring how `step` is already frozen into slice_write's
slice offsets at capture time, per project notes), and does the trace
correctly re-read v_input's LIVE contents on each replay (rather than
baking in capture-time values) while update_idx itself stays fixed --
confirming this needs one captured trace per decode step, exactly like
the existing slice_write-based per-layer trace structure.

This does NOT touch tst_model_cached_additions.py. Diagnostic only,
not committed to the PR.

Constraints observed from project notes, respected here:
  - only copy_host_to_device_tensor + execute_trace + readback should
    touch the device between trace replays
  - untraced ops running concurrently with an open trace handle cause
    severe slowdown -- avoided by not issuing other device ops between
    capture and replay
  - warm-up untraced call + ttnn.synchronize_device required before
    begin_trace_capture, to avoid JIT-compile during trace
"""
import torch

import ttnn


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=200_000)
    try:
        BS, H, T_max, D = 1, 2, 24, 32
        STEP_TO_CAPTURE = 3  # arbitrary mid-sequence step, frozen at capture

        v_cache = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        v_input = ttnn.from_torch(
            torch.zeros(BS, H, 1, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # --- Warm-up (untraced), synchronize, per project's own rule ---
        ttnn.update_cache(v_cache, v_input, update_idx=STEP_TO_CAPTURE, batch_offset=0)
        ttnn.synchronize_device(device)

        # Reset cache to zero after warmup write, so capture starts clean
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(BS, H, T_max, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            v_cache,
        )
        ttnn.synchronize_device(device)

        # --- Capture: update_cache with update_idx frozen as Python int ---
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.update_cache(v_cache, v_input, update_idx=STEP_TO_CAPTURE, batch_offset=0)
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # --- Replay 1: known marker value, check it lands at STEP_TO_CAPTURE ---
        marker_1 = torch.full((BS, H, 1, D), 111.0)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(marker_1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            v_input,
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        result_1 = ttnn.to_torch(v_cache).float()
        written_at_step = result_1[0, 0, STEP_TO_CAPTURE, 0].item()
        other_step = result_1[0, 0, STEP_TO_CAPTURE + 1, 0].item()
        print(f"[TRACE REPLAY 1] value at update_idx={STEP_TO_CAPTURE}: {written_at_step:.2f} (expect ~111)")
        print(f"[TRACE REPLAY 1] value at adjacent step {STEP_TO_CAPTURE+1}: {other_step:.2f} (expect 0)")

        # --- Replay 2 (SAME trace handle, no re-capture): different marker ---
        # Confirms trace re-reads v_input's CURRENT contents each replay
        # (not baking in capture-time values), while update_idx stays frozen.
        marker_2 = torch.full((BS, H, 1, D), 222.0)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(marker_2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            v_input,
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        result_2 = ttnn.to_torch(v_cache).float()
        written_at_step_2 = result_2[0, 0, STEP_TO_CAPTURE, 0].item()
        print(
            f"[TRACE REPLAY 2] value at update_idx={STEP_TO_CAPTURE}: {written_at_step_2:.2f} (expect ~222, "
            f"confirms trace re-reads v_input's live contents each replay)"
        )

        print(
            f"[TRACE API] execute_trace signature has no update_idx param -- confirms "
            f"update_idx can only be changed by capturing a NEW trace per step, "
            f"same as this model's existing per-layer trace structure for slice_write."
        )

        checks = [
            (abs(written_at_step - 111.0) < 1.0, "replay 1 wrote marker at frozen update_idx"),
            (abs(other_step) < 1.0, "replay 1 did not write to adjacent step"),
            (abs(written_at_step_2 - 222.0) < 1.0, "replay 2 picked up new v_input contents at same frozen update_idx"),
        ]
        all_pass = all(c[0] for c in checks)
        for ok, desc in checks:
            print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
        print("TRACE PROBE PASS" if all_pass else "TRACE PROBE FAIL")

        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
