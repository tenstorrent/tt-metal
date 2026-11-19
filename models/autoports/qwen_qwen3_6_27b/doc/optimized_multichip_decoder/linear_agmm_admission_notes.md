# Linear conventional AGMM admission repair

2026-11-13. CPU-only AutoFix subtask following
`debug_conventional_agmm/AUTODEBUG.md`. No build, device command, reset, or commit.

The repaired contract is persistent gathered-scratch ownership between merged
AG/matmul launches. The previous host factory disabled the supplied admission
barrier whenever scratch was persistent. A new launch could therefore overwrite
a peer's previous matmul input. Enabling the old barrier alone was insufficient:
its clear-to-zero could erase a future launch's arrival credit. These were masked
hazards identified by source inspection, not observed silicon failures.

Changed only:

- `all_gather_async/device/all_gather_async_default_program_factory.cpp`: requires
  the explicit `AllGatherFusedOpSignaler::is_linear_full_slice()` capability;
  restricts it to TP4 Linear, exact FABRIC_1D, one link, two workers per direction,
  MULTI signaling, non-reverse ordering, and a supplied barrier. Persistent scratch
  now participates in startup admission for this capability. Existing Ring and
  unrelated all-gather behavior retain their original decisions. The new kernel
  define `LINEAR_FULL_SLICE_ADMISSION` is emitted only for this capability.
- `all_gather_async/device/kernels/minimal_default_writer.cpp`: inside that define,
  wait for the three other ranks, subtract exactly three credits with a non-posted
  local NoC atomic, and wait for atomic acknowledgement before proceeding to
  scratch writes. The default branch remains the original clear operation.

All source paths above are relative to
`ttnn/cpp/ttnn/operations/experimental/ccl/`. Capability metadata, full consumer
validation, A/B slice schedules, and physical multicast-reservation checks belong
to the companion native repair, not this subtask. They must land before building.

The factory also verifies that each serialized directional worker-vector index
and each runtime worker coordinate match allocation order. This prevents a
changed set ordering from assigning the signaler master to the wrong core.
The two candidate placements `(0,2)` and `(0,5)` each use six cores on one row
(two mux cores and four workers). A cross-row/multilink ordering is not silently
accepted. Generalized support would need explicit ordering changes and validation.

Validation commands:

```sh
python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/linear_agmm_admission_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/linear_agmm_admission_cpu.json
git diff --check -- ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp
```

Both pass. The CPU ledger checks all 16 rank/direction/worker destinations receive
three credits, seven possible future-credit counts survive subtraction, and both
candidate worker placements preserve identity. Negative controls reproduce lost
future credits from clearing and a changed master under cross-row set sorting.
Source binding checks confirm the capability and admission define are present.
An additional direct text comparison removed only the new conditional branch and
verified the remaining writer source is **byte-identical to HEAD**; no unguarded
kernel behavior changed. A first source-binding assertion failed after formatting
split an expression across lines; the expression was made a named host boolean
and the checks rerun. No device inference follows from that CPU assertion.

SHA256 at subtask handoff:

- Factory: `d14f28181b6c559b142ba4d65932a31cf8a13698aa69586c56390e8b9e676069`
- Writer kernel: `f6e4c1d5a83031f40e6468e02942902a25f35d1ae95efce69005cb6d44a54313`

Hardware remains unverified. Required parent validation includes fresh-cache
watcher execution; changed input/weights/scratch/output/global semaphore cache
rebinding; 50 queued generations with rank skew; per-rank gathered-scratch and
matmul correctness; and complete coherent-family timing including admission.
The ownership argument assumes one owner and one ordered CQ/subdevice context:
arrival at the next merged program implies previous local matmul reads completed.
Concurrent queues sharing scratch are outside the admitted contract.
