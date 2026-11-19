# Compact prefill adoption audit

CPU-only independent inspection, 2026-11-14. No hardware was run, runtime was
not edited, and this is not the final stage review.

The adopted selection preserves canonical horizons, caps, profile/boundary
metadata and consumed-row counts. Only canonical 16384/32768 execution with at
most 8192 consumed rows selects the already-prepared 8192 profile. Missing 8192
preparation retains the old extent. Warmup still covers both large profiles,
8192 and the terminal head before capture. The version bump identifies the new
policy. Nonaligned overlap and saved input/conv/recurrent boundaries remain in
the planner/runtime path; compact selection happens after those decisions.

The pre-adoption [full-stack result](compact_prefill_full_model.json) contains
1,544 finite, bit-equal comparisons across two schedules, including all-layer,
all-shard conv/recurrent state, valid KV rows, fragment-terminal/final logits
and following decode. Compact executions cover consumed counts 9, 18 and 1024;
completion also exercises canonical 14336/16384/30720 extents. The 822 warmed
program count and six trace IDs were retained. This is substantial numerical
evidence for the measured shapes, not proof of every prompt/shape.

**Probe coverage defect fixed:** after adoption, its candidate wrapper supplied
8192 to `_execute` before the production selector ran. The canonical control
was valid, so this does not invalidate the earlier numerical result, but it
could not detect a broken production selector. The probe now observes the real
`_execution_height` via a call-through wrapper, leaves candidate `_execute`
unchanged, records its selections and requires a canonical-to-8192 transition.
The canonical baseline still explicitly disables the selector. A reduced
tracked rerun of this corrected probe is required; another full-stack run of
the same two schedules is unnecessary solely for this fix.

Head skipping occurs only after all layer execution, boundary/state updates and
request metadata writes. It omits `_head`, whose norm/projection has no model
cache or sampler mutation. `discard_logits` is rejected with all-logit mode;
final/device sampling paths retain the head. Prefix rebuilds explicitly discard
their unused head. The expanded horizon probe compares legacy versus skipped
heads through profile rebuilds and a nonaligned 129-to-160 continuation, checking
exact terminal/next-decode outputs, complete state, sampler state, head counts
and trace/program retention. That device result is still parent-owned.

No additional source defect was identified. Remaining necessary evidence is the
corrected production-selector probe, the already-running head-skip/state probes,
and planned real-serving mixed-trace, chunk-budget, long-context/load, sampling
and qualitative gates. Those must exercise adopted code without monkeypatches.
The host selector test covers the 8192/8193 switch and omitted-profile fallback;
do not label those boundary cases hardware-proven. If later numerical results
differ, use the established accuracy/resume gates rather than extending this
limited exactness claim or relaxing batch/peer preservation.
