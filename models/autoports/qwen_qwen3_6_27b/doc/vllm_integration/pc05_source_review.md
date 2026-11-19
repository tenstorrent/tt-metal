# PC-05 source audit (not stage review)

Independent AutoFix subagent inspected the current classifier, scheduler,
lane coordinator, runner and input-batch diff. No blocking correctness finding.

- Scheduler filtering precedes allocation; hidden requests retain KV/state and
  count against capacity. Per-owner phase counters preserve bounded progress.
- Host lane metadata supplies private CPU RNG generators only for scheduled
  rows. Paused rows cannot consume those generators. Built-in filters modify
  logits and do not advance output histories.
- Front-packed cached request identity survives unscheduling and preemption.
  Async results remain attributed to submitted IDs. Lane cohort changes mark
  layout changes and drain/reload; inactive positions and controls are neutral.
- Homogeneous device traffic retains the resident/async path with no additional
  device operation or readback. Scheduler bookkeeping adds host work.

Performance concern: global lane predicates still conservatively disable overlap
when paused occupants have host-only filters/logprobs. This affects mixed-cohort
traffic, not homogeneous device headline traffic. Mixed-cohort cost must be
reported separately. No performance claim follows from this source audit.

This was a read-only source audit, not the required final stage-review round.
Hardware serving, accuracy, context and benchmark acceptance remain pending.
