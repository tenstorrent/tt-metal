# AutoFix: review contract findings

Current repair source: `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`. Both full-tracker reuse regressions and all eight tight-capacity cases pass. Current headline, Watcher and four pytest cases pass; v7 full-only QKV fidelity also passes affected-contract checks. Final profiling and review remain in progress. This document is not stage signoff.

## Trace allocation lifetime

The independent reviewer found allocator warnings in both v5 request-reuse logs. `run_trace_allocation_controls.py` reproduced the issue with allocation tracking and traceback collection enabled, without program-cache exclusions. `trace_alloc_v5_commands.json` records both failures. Sliding attention retained two final-tail clones; full attention initialized 53 surviving program-cache allocations after capture. [AutoDebug](AUTODEBUG_trace_lifetime.md) follows their ownership through the model, program initialization and allocator. The tracker proves live allocations, not actual overlapping addresses or observed numerical corruption. Top-down binary allocation does not establish safety against captured temporary addresses.

The runtime now passes `retain_prefill_tail` through the existing attention kwargs. Only a subsequent chunk of the same prefill retains cloned K/V history; the final chunk drops the previous tail and creates no final clones. Decode and prefix continuation consume the paged cache. No host read or replacement arithmetic is introduced.

The reuse test now initializes its exact nine prefill signatures before capture, initializes decode, then forbids program-cache misses for the entire live-trace interval. It records initialization counts and stable post-capture counts. This fixes the missing setup rather than suppressing the tracker. A caller adding a previously unseen prefill signature must initialize it before capture or release/rebuild the trace. This is a trace allocation lifecycle requirement, not a logical sequence-length restriction; the decoder supports all valid logical lengths.

Completed acceptance: `run_trace_allocation_controls_v6.py` passes both kinds with the full tracker enabled (`trace_alloc_v6_commands.json`). Both reports contain all nine actual requests, no retained post-capture buffers at every replay, stable program-cache entries and unchanged PCC thresholds. Tracking includes program-cache allocations; no corruptible scope is used. No claim is made that the original 53 buffers actually overlapped a captured address.

## Tight paged-cache capacity

The public contract permits capacity rounded to 128 tokens. The selected K256 paged prefill could issue a final read beyond that logical page table: full attention at S1025, Q padded to64 and offset1024, rounded K read end1280 versus capacity1152 (36 pages). The reader indexes the page table without a logical-width guard. The original narrow Watcher run passed (`tight_cache_v5_layer5_1025`), showing that a passing value/Watcher check alone does not prove absence of reads from aligned padding.

`ConfiguredChunkedPrefillAttention` now prepares a boundary program during setup and chooses it only when the wider rounded read would exceed the supplied logical page-table capacity. The boundary uses Q<=128/K<=128; the production full-attention case is Q64/K128. Normal sufficient-capacity chunks retain Q64/K256 and the same precision.

The repaired S1025/cache1152 run passes HF prefill/decode PCC and Watcher. `probe_optimized_cache_capacity.py` guards the actual native calls, recording Q, offset, K block and maximum read end. Its report proves K128/read_end1152 for that 36-page table; it is not inferred from output PCC. `run_tight_cache_regressions.py` covers adjacent tails and both K128/K256 branches, including exact prefill-only capacity and continuation decode.

## Evidence scope

The v5 source is retained byte-for-byte as `runtime_v5_before_review.py.txt`; its SHA remains169c0d97. Existing maximum-context, batched, stress and policy experiments remain evidence for unchanged arithmetic/configuration branches. New targeted tests, current-default headline reproduction and profiles bind the repaired source. Historical reports are never relabeled with a new runtime hash.

Current acceptance is indexed by `validated_v6_validation_summary.json` at status `current_targeted_and_inherited_gates_passed`. The source-delta mapping explicitly retains v5 maximum-context, batch and stress evidence for unchanged branches. `validated_v6_contract_commands.json` contains five successful commands (two headline, two Watcher and four-case pytest); `tight_cache_v6_commands.json` adds seven cases to the initial1025 Watcher probe. Both K-block branches are observed. `trace_alloc_v6_commands.json` records stable370/378 program-cache entries across nine requests per kind.

The selected v7 source`daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`
retains both fixes unchanged. Its full-attention request-reuse run again uses the
full allocation tracker with no exclusions, and its tight1025/cache1152 Watcher
run again observes K128 with maximum native read1152. `validated_v7_validation_summary.json`
binds these current checks, both current headlines/Watcher runs, full maximum
and near-maximum contexts, four small-tail pytest cases and both512-step stress
comparisons. The broader v6 geometry and sliding lifecycle checks remain
explicit inherited evidence under `source_delta_v7.json`.
The selected v8 source `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898` again retains both repairs. `validated_v8_request_reuse_layer5.json` passes all nine requests with the full allocation tracker and379 program-cache entries unchanged after capture; the extra entry versus v7 is the setup-warmed L1 producer signature. `tight_cache_v8_layer5_1025.json` and its separate Watcher log verify the K128 bound at actual capacity1152. `validated_v8_validation_summary.json` and `source_delta_v8.json` bind the fresh affected-path checks and unchanged inherited branches.
