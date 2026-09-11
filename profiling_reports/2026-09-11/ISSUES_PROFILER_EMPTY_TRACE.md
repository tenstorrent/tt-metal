# Profiler loses replay attribution when a segmented capture ends with an empty trace

Draft; not posted. The empty trace is a suspected cause, not a confirmed reproducer.

## Observation

A September 10 Mistral Small 4 PP4 run recorded eight host replay sessions. On the final rank, the C++ profiler report attributed only the first session for nonempty traces; the final trace was empty on all eight devices. This blocked validation of sessions 2–8 and a complete PP4 operation breakdown. Model execution and all four capture wrappers completed successfully; the attribution failure does not establish a model correctness failure.

The raw captures were in a `/tmp` worktree that is no longer available. [PROFILING_REPORT.md](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/PROFILING_REPORT.md) preserves the observation, but the raw data cannot currently be rechecked. Source inspection below was performed September 11.

## Environment

- Revision: `b48bf4095de1601786c4cf0e91a647a532c730c5`, `akhan/mistral4-prefill-followups`.
- Host: `bh-glx-120-b10u14`, 32-device Blackhole Galaxy, PP4 with eight devices per rank.
- Workload: 36 layers, 5,120-token chunks, two users, maximum sequence length 10,240, traced execution, `GPT_DEVICE` routing and a KV-only final layer.
- Ranks 0–3 recorded 28/28/28/26 trace IDs. Ranks 0–2 passed eight-session/eight-device coverage checks. Rank 3 had eight host replays for each of its 26 trace IDs, but device attribution was incomplete before the custom offline join.

Expected: every nonempty replay retains its trace ID and replay ordinal on each device, even when another segment has no programs.

## Hypothesis

At the revision above:

- `models/demos/deepseek_v3_d_p/utils/sub_device_trace.py:81`: `end_capture()` always appends the final trace. `_split()` closes a trace and opens another, allowing an empty trailing segment. `replay()` executes every recorded trace.
- `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:450`: capture uses this controller to split around subdevice switches and layer acknowledgments.
- `tt_metal/distributed/mesh_device.cpp:1586`, `tt_metal/tools/profiler/tt_metal_tracy.hpp:37` and `tt_metal/impl/profiler/profiler.cpp:1586`: each replay is registered in the host-side trace sequence before dispatch.
- `tt_metal/impl/profiler/profiler.cpp:1968`: attribution indexes that sequence with `device_trace_counter - 1`, then checks whether the program belongs to the resolved trace. The comment near line 1947 also notes incomplete trace/subdevice association.

An empty trace might advance host bookkeeping without the matching device counter advance, causing later markers to resolve to the wrong trace. **Device counter behavior for empty traces has not been verified.** A subdevice-specific mismatch or another cause remains possible.

## Reproduction plan, not yet run

1. Warm a small deterministic operation and capture it as trace A, with Tracy and device trace tracking enabled.
2. Capture empty trace B. Replay A then B twice and synchronize. Keep host metadata, raw device markers and the C++ report.
3. Compare against A replayed twice without B. Vary B's position; if these cases pass, repeat with subdevice manager switches.
4. Compare host replay lists, device trace counters, program IDs and reported `(trace_id, replay_count)`. Check operation outputs separately.

## Acceptance

- A retained small reproduction identifies the cause before a fix is proposed.
- Nonempty replays retain correct trace IDs and ordinals across empty segments and subdevice switches. Empty traces do not need synthetic operation rows.
- Nonempty-only attribution and operation results remain correct.
- A fresh eight-request PP4 capture passes coverage checks for every rank, device and trace, with artifacts retained outside `/tmp`.

No source fix or fresh hardware reproduction is included.
