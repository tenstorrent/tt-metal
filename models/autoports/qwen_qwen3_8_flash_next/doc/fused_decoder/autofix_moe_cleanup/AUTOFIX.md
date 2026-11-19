# AutoFix: trace-safe zero creation

The unchanged contract is device-only prefill/decode and capture-safe execution. The experimental `zeros_like(logits,dtype=BF16)` rewrite fails at logical2 during capture; logical1 bypasses this branch. The fresh [AutoTriage report](AUTOTRIAGE.md) and current `creation.cpp:282` prove that the dtype mismatch selects `full_impl`; its32KiB output is below the128KiB device-fill threshold and is uploaded from host. The original cast-then-zeros path instead selects native fill directly.

Minimal repair in the stage prototype: allocate `empty_like(logits,dtype=BF16)` then `fill(output,0,output_tensor=output)`. This removes the numerical cast while keeping fill on device. Independently, end an active trace capture in `finally` before releasing the trace and closing the mesh. No production runtime or global TTNN code is changed.

Live failure evidence was captured with bounded tt-triage before the original600-second timeout; no process was killed early and no reset was requested or performed. The first failure source is archived under `sources/next_prototype_plan_v1/`. The already-running Python process retains its loaded original code; edits prepare subsequent jobs only.

Validation pending: after timeout, bounded chip0 operation smoke; then corrected component output/replay and clean timing on both expert dtypes, followed by full-model gates if selected. Device health and the proposed fix are not yet claimed verified.

At 2026-11-13 13:16 UTC the original bounded job exited 124. No early
termination or reset was performed. The subsequent 90-second bounded chip-0
open/add/read/close smoke passed (`post_timeout_smoke.log`); the device was
released before resuming `repaired_prototype_plan_v1.json`. Repair validation
is pending in that plan, with fresh source archives and artifact names.

Repair validated: `repaired_prototype_plan_v1_progress.json` completed all5jobs.
Both expert dtypes pass35 rows each under Watcher (five policies times seven
logical lengths), with exact eager outputs and16 repeated trace replays per row.
Clean BF16 T1 current/all2.284214/2.281729ms and T32 35.577953/35.560580ms;
BFP8 T1 2.272421/2.270262ms and T32 44.686564/44.660975ms. These small component
gains do not replace integrated ranking. The runtime candidate is gated by
`fuse_moe_cleanup`, initially disabled, and full acceptance continues in
`exact_qkv_integrated_plan_v1.json`. The trace host-write failure is repaired;
no hardware reset or quality-gate relaxation was needed.
