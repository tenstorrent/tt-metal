# Trace capacity investigation

Status: investigation complete for the tested allocation schedules. Three/four expert-layer upgrades remain rejected; two upgrades with global HiFi4 pass the full trace inventory. The datatype stage continues.

The four-layer expert BFP8 / global expert HiFi2 candidate must retain its full
262144-token cache and safely replay all supported traces. It loaded and
prepared, then failed the first readiness prefill when a new dynamic trace
entered an earlier live trace's DRAM address envelope. The exact failure is in
`e8_four_hifi2_v3.json`. Baseline `inherited_e4_lofi_v3.json` passes.

The isolated AutoDebug report identifies a valid placement check, not a numerical
failure or an assertion to remove. All trace allocation checks and program-cache
tracking stay enabled. No context reduction, trace eviction, or eager ranking is
permitted.

The first hypothesis is that a fixed region, reserved before ordinary allocation,
can separate trace storage from transient workspace. The harness now accepts
`--trace-region-size` and an optional capture ledger; these are diagnostic setup
options, with the original zero-region default unchanged. The proposed 768 MiB
reservation is below the observed 1000.14 MiB prepared free space and above the
baseline's 612.69 MiB prepared-to-final growth. Those figures motivate an
experiment; they do not prove the candidate's trace inventory or peak fits.
The ledger records each captured signature and actual TRACE allocator occupancy.
An ordinary allocation failure would show that this reservation leaves too little
workspace; an undersized trace region would require revisiting its sizing.

The second hypothesis is that the complete trace/workspace footprint requires
one fewer expert-layer upgrade. `e8_three_hifi2` retains expert HiFi2 globally and
upgrades layers 0, 1, 47, recovering exactly 300 MiB/rank relative to the failing
four-layer candidate. It remains unverified until the same full-model references
and traced decode pass. A passing three-layer run plus a failed adapted four-layer
run establishes a practical bracket for the current implementation, not a proof
that no future allocation optimization could fit four.

Changed source: `tests/run_datatype_candidate.py` only for the diagnostic mesh
argument and optional capture ledger. No model arithmetic or allocation guard
has been changed for this investigation. Formatter/hook evidence is
`logs/trace_diagnostic_hooks.log`. Device results and the final conclusion will
be appended after the serial experiments finish.

## Fixed768 result

`e8_four_fixed768_v1.json` exited1 during `generator.prepare()`, before capture. `causal_indexer_scores` needed a134217728B DRAM output across8banks (16777216B/bank), but only2176320B/bank was free (largest1591616). This proves the768MiB reservation leaves insufficient ordinary workspace in this allocation schedule; it does not prove every smaller reservation fails. No accuracy row accepted. Next discriminating experiment is640MiB (`e8_four_fixed640_v1`), still above baseline prepared-to-final growth, with the same references and the complete six-bucket/sampling retention smoke. Commands/provenance and logs use those stems. No reset or guard change.

## Fixed640 result

`e8_four_fixed640_v1.json` also exited 1 during preparation. The attention indexer needed a 268435456-byte output (33554432 bytes per bank), with only 5400640 bytes free per bank and a 4730944-byte largest block. Thus 640 MiB still reserves too much ordinary workspace for this policy. The three-upgrade dynamic-region experiment `e8_three_hifi2_v1` is now running at unchanged full context with the complete trace-inventory smoke. These failures reject the tested reservations, not every possible allocation optimization.

## Three-upgrade dynamic-region result

`e8_three_hifi2_v1` passed all four reference/runner accuracy rows, including traced teacher forcing, then failed capture of the retained 16-token prefill bucket. Trace base 4176820224 overlapped the prior live-trace high-water mark 4179454912. This policy is not accepted. The ledger measured 773849088 bytes (738.00 MiB) of cumulative allocation growth through that failed eighth capture, before adding sampled generation. A fixed region therefore needs at least this much space. The four-upgrade 640 MiB preparation failure needed another225+ MiB of working memory; a three-upgrade region near749 MiB would recover only191 MiB relative to it (300 MiB fewer weights minus109 MiB more reservation). This observed schedule leaves no credible fitting reserved-region interval.

Next bracket: `e8_two_hifi4_v1`, expert BFP8 layers0/47 and global expert HiFi4, unchanged full cache and complete trace inventory. HiFi4 is legal and actually reaches the fused Blackhole kernel (`p1_plan_audit.md`). The unrun three-layer HiFi4 policy is retained as a configuration only, not evaluated evidence.

## Accepted fit bracket

`e8_two_hifi4_v1.json` completed and passed both full-model references, all six retained prefill buckets, nonaligned31/32/33 prompts, and sampled-to-greedy switching. It retains the full262144-token BF16 cache, with BFP8 experts in layers0/47 and global expert HiFi4. This is the highest-precision tested policy that fits the complete current trace inventory. Three upgrades passed numerical rows but failed the16-token trace; four upgrades failed dynamic capture and both tested reserved-region workspace plans. AutoFix did not repair three/four-upgrade fit under the unchanged capability contract. Those policies are rejected; stage work continues with the valid two-upgrade P1 control. No hardware reset or allocation-guard relaxation occurred.
