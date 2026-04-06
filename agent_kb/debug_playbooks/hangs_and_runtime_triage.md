---
title: Hangs and Runtime Triage
type: debug
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - debug
  - watcher
  - dprint
  - hang
source_files:
  - CONTRIBUTING.md
  - docs/source/tt-metalium/tools/watcher.rst
  - docs/source/tt-metalium/tools/kernel_print.rst
---

# Hangs and Runtime Triage

## Default Bring-Up Posture

During kernel development, prefer running with Watcher enabled first and only disable it after the design is stable.

Common environment settings called out by the repo docs:

- `TT_METAL_WATCHER=10`
- `TT_METAL_DPRINT_CORES=(x0,y0)-(x1,y1)`
- `TT_METAL_RISCV_DEBUG_INFO=1`

## First Pass When A Kernel Hangs

1. Confirm whether Watcher reported an explicit error on stdout.
2. Inspect `generated/watcher/watcher.log`.
3. Find the latest dump and read:
   - `k_ids`
   - waypoint state
   - the core coordinates involved
4. Map `k_ids` back to kernel source files.
5. Instrument the suspected kernel with more logging or waypoints.

## When Watcher Changes Timing

If the hang disappears under Watcher, treat it as a timing-sensitive bug, not a clean bill of health.

The repo docs suggest progressively disabling Watcher features to narrow this down:

- `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1`
- `TT_METAL_WATCHER_DISABLE_WAYPOINT=1`
- `TT_METAL_WATCHER_DISABLE_ASSERT=1`

## Compile-Time Inspection

If behavior appears to depend on compile-time arguments or generated defines:

1. Inspect generated kernel configuration artifacts under `built/<device>/kernels/`.
2. Materialize compile-time args into `constexpr` variables in the kernel if needed.
3. Use the relevant dump tool against the generated ELF.

## Agent Guidance

- Do not answer a hang report only from static code inspection if Watcher data exists.
- When a bug is likely in data movement, inspect NoC alignment and barrier usage first.
- When a bug is likely in compute, inspect Dst register lifecycle and CB ordering first.

## Sources

- `CONTRIBUTING.md`
- `docs/source/tt-metalium/tools/watcher.rst`
- `docs/source/tt-metalium/tools/kernel_print.rst`
