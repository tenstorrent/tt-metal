---
title: New Kernel Codegen Checklist
type: recipe
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - recipe
  - codegen
  - checklist
source_files:
  - tt_metal/programming_examples
  - METALIUM_GUIDE.md
  - tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md
  - CONTRIBUTING.md
---

# New Kernel Codegen Checklist

## 1. Choose A Reference Shape

- Find the closest existing programming example or test kernel.
- Match on kernel type, data layout, and communication pattern before matching on operation name.
- Reuse a known-good host/device split.

## 2. Lock Down Scope

- Which hardware generation is in scope?
- Is the kernel single-core or multi-core?
- Is the movement interleaved, sharded, multicast, or something else?
- Is correctness sensitive to dtype, accumulation mode, or non-tile-aligned tails?

## 3. Define The Pipeline Explicitly

- Reader inputs and runtime args
- Input CB IDs and tile counts
- Compute-kernel init calls
- Dst register usage
- Output CB IDs and writer behavior

Do not let these details remain implicit in the prompt.

## 4. Verify Host And Device Configuration Together

- CB declarations on the host must match CB usage in kernels.
- Compute configuration must match device-side assumptions about Dst capacity and accumulation behavior.
- Compile-time arguments, runtime arguments, and defines must all be audited together.

## 5. Add Observability Early

- Run with Watcher during bring-up.
- Add `DPRINT` or waypoints where a hang is plausible.
- If behavior depends on compile-time args, make them inspectable.

## 6. Handle Accuracy And Boundary Cases Up Front

- Check whether partial tiles exist.
- Check whether accumulation should be FP32.
- Check whether intermediate CBs need special unpack-to-Dst configuration.

## 7. Add A Minimal Validation Path

- Prefer a small analytic or reference-backed test first.
- Validate both happy path and one boundary condition.
- For performance work, separate correctness bring-up from tuning.

## 8. After The Change

- Update this KB if the work uncovered a new durable rule, trap, or recipe.

Related pages:

- [Kernel Pipeline](../concepts/kernel_pipeline.md)
- [Circular Buffers](../concepts/circular_buffers.md)
- [Kernel Codegen Gotchas](../pitfalls/kernel_codegen_gotchas.md)
- [Hangs and Runtime Triage](../debug_playbooks/hangs_and_runtime_triage.md)

## Sources

- `tt_metal/programming_examples`
- `METALIUM_GUIDE.md`
- `tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md`
- `CONTRIBUTING.md`
