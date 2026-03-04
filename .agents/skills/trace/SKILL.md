---
name: trace
description: Bring up and debug TTNN tracing for modules or models (capture once, replay many). Use when enabling metal trace in TTNN tests/demos, converting per-step execution to traced execution, or diagnosing trace hangs/perf regressions.
---

# Trace

## Overview

Enable reliable TTNN trace capture/execute for model or module runs, including warm-up, persistent inputs, trace replay, and safe teardown.

## Workflow

1. Confirm trace suitability
- Keep input shapes, dtypes, and layouts constant across steps.
- Avoid dynamic control flow or variable sequence lengths inside a single trace.
- Ensure device creation includes `trace_region_size` in `device_params`.

2. Warm-up compile (no trace)
- Run the op once without tracing to compile kernels and keep compilation out of capture.
- Synchronize the device after warm-up.

3. Allocate persistent device inputs
- Create device tensors for inputs that will be updated each step.
- Match mesh mapping, dtype, layout, and memory config to runtime usage.
- If CCL is involved, reset sem counters before capture and before each replay.

4. Capture once
- Begin trace capture, run the op exactly once, end capture.
- Keep the traced output tensor alive for reuse during replay.
- If you need derived tensors (e.g., RoPE mats), generate them inside capture using TTNN ops.

5. Replay N times
- Update persistent inputs each step via `ttnn.copy_host_to_device_tensor` (or `ttnn.copy` for device-to-device).
- Assert shapes/dtypes/layouts are unchanged before copying.
- Execute trace each step with `ttnn.execute_trace`.

6. Release and deallocate
- Call `ttnn.release_trace` after the loop.
- Deallocate persistent input/output tensors.

## Minimal Skeleton

```python
# 1) Warm-up
op()  # no trace
ttnn.synchronize_device(device)

# 2) Persistent inputs (device tensors)
tt_in = make_device_inputs(...)

# 3) Capture once
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
out = op()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 4) Replay
for step in range(N):
    update_inputs(tt_in, step)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    read(out)

# 5) Cleanup
ttnn.release_trace(device, trace_id)
```

## Guardrails And Checks
- Assert input `shape`, `dtype`, and `layout` match the first captured step.
- If inputs change, assert.
- Avoid host tensor creation inside capture (no `from_torch` inside trace).
- Use a dedicated trace region size; capture can fail or hang without it.

## Common Failure Modes

- Capture per step (slow): capture once, then replay.
- Shape changes across steps: assert.
- From-torch inside capture: move host-to-device outside capture.
- Trace hangs: verify `trace_region_size`, avoid writes during capture, ensure sem counters reset for CCL.
