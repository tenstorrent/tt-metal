---
title: Kernel Codegen Gotchas
type: pitfall
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - pitfalls
  - kernel
  - accuracy
  - synchronization
source_files:
  - docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst
  - tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md
  - CONTRIBUTING.md
---

# Kernel Codegen Gotchas

## Synchronization And Register Ownership

- Failing to respect the `tile_regs_acquire -> tile_regs_commit -> tile_regs_wait -> tile_regs_release` lifecycle leads to undefined behavior.
- Even kernels that do not pack output still need correct Dst register state transitions.
- `acquire_dst` and `release_dst` are deprecated; prefer `tile_regs_*`.

## Accuracy Traps

- `fp32_dest_acc_en = true` changes Dst storage width but does not automatically guarantee full-FP32 computation semantics.
- If intermediate results are written to a CB and later copied back to Dst, the CB may need `UnpackToDestFp32` host-side configuration to avoid truncation.
- A CB configured for `UnpackToDestFp32` cannot be used as if it were a normal SrcA/SrcB input for APIs such as `add_tiles()`.
- Divide-then-sum patterns can be less accurate than sum-then-divide.
- Welford-style accumulation can be both more accurate and more bandwidth-efficient for mean/variance style reductions.

## Shape And Layout Traps

- Non-tile-aligned shapes require explicit handling; garbage in partial tiles can contaminate reductions.
- Use logical shape when correctness depends on the real tensor extent; padded shape can be wrong for boundary logic.
- Do not assume interleaved and sharded layouts can share the same movement pattern.

## Debugging Traps

- A kernel that appears to work without Watcher may still be invalid.
- Timing-sensitive hangs can disappear when Watcher is enabled. That does not mean the bug is gone.
- Compile-time arguments are easy to misuse because generated kernels can differ by defines and constexpr values.

## Agent Rules

- Before inventing a new kernel structure, inspect the nearest working example.
- Before changing accumulator behavior, inspect both the host `ComputeConfig` and the device-side compute loop.
- Before adding a special-case branch for shapes, verify whether the operation already assumes tile alignment elsewhere in the stack.

## Sources

- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
- `tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md`
- `CONTRIBUTING.md`
