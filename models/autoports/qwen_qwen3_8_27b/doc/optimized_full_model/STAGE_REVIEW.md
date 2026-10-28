# Stage Review

Verdict: clean-pass

Reviewed 2026-10-28 as the independent xhigh reviewer for stage 07, Qwen/Qwen3.8-27B. Acceptance covers the explicitly requested local B1 configuration **QWEN38_COMPACT_B1_RECURRENCE=1**, on the 1x4 p300c mesh. The reviewed worktree is uncommitted at base 206dd3a0ec6f6e1494d4408e6590e47d4bc97ae0; its 37 frozen files match final-source-freeze-v4.json, SHA256 2dd2e6053371e913d28b1c094eb9f943e73a5d4aae8a6e1f3244c2801740424a.

## Required Work

- None for the scoped local optimization.

## Other Concerns

- The opt-in requirement is explicit. Unflagged throughput remains 36.090930 TSU. The final opt-in median is 40.673091 TSU (+12.696158%); all three repeats exceed 40. Both arms use all 64 layers, native 262144 cache allocation, 4096 input/128 output tokens, physical chunk 4096, and three warmed repeats. Their complete token trajectories match. TTFT is effectively flat, 436.932691 versus 436.612106 ms.
- Device-time accounting is an explicitly labeled representative extrapolation: 26.790020 versus 23.694282 ms, with first-layer-only conversions counted once. Candidate E2E is 24.586280 ms, 3.76% above that estimate. The 7.700880 ms traffic floor includes stored BFP exponent payload, chunked KV reads and FP32 recurrent read/write; it is not measured DRAM utilization.

## Hard-Check Gaps

- None for this scope. Full64 tracked acceptance uses allocation tracking with program-cache allocations included. Five complete 512-entry conv/recurrent/paged-K/V maps and all 12 continuation-logit hashes match exactly, including ragged and nonaligned continuations and the all-inactive no-replay bridge. Three captures remain stable.
- Fresh native-chat AIME prefill and teacher forcing reach 100% top-5/top-100, with top-1 99% and 98%. The shared 512-token suite uses identical HF prompt token IDs; explanation, JSON and Python outputs are correct and terminate at EOS. The final scoped free-running checker exits 0. Teacher-forcing timing is kept separate from native token-out timing.
- The integrated watcher pair is correctly labeled as a reduced real-layer 0/3 fixture with buckets 1/32/64/4096. It passes exact state/logit/token comparisons with watcher10, NOINLINE, all features enabled and tracking1/skip0. Full64 all-129-bucket preparation is established separately by the tracked pair. Profiling and watcher runs are separate.

## Anomaly Ledger

- Observed anomaly: fused prefill rows report HiFi4 while the matmul policy is LoFi. Evidence: final raw CSV attributes and compiled matmul hashes 17099178834929360030/2082464955539158442 versus ring-reduction hash 16435510841809839828. Affected path: profiler reporting. Control: both arms show the same behavior. Likely subsystem: scalar fidelity export for a multi-kernel op. Investigation: matmul descriptors are LoFi, reduction is HiFi4, and the exporter overwrites the scalar per kernel. Resolution: controlled; raw CSV remains intact.
- Observed anomaly: initial memory inventory exceeds ordinary DRAM by 25,624,576 bytes. Evidence: raw control events, live buffer-address census and B1_TRACE_DOMAIN_REPLAY.json. Affected path: setup accounting. Control: numeric DRAM0 totals exactly 8,865,768,448 bytes; three TRACE4 allocations explain the entire excess. Likely subsystem: graph placement labels versus allocator domains. Investigation: source, addresses, aligned sizes and independent candidate DRAM/TRACE views agree. Resolution: controlled. The fixed 128 MiB trace arena remains separate; regular DRAM increases only 98,304 bytes, and reduced trace occupancy returns no ordinary DRAM capacity.
- Observed anomaly: multiplying the first representative linear layer repeats embedding-DRAM conversions. Evidence: final-v4-profile-attribution.json and raw operation signatures. Affected path: device-work estimate. Control: source carries subsequent residuals in L1. Likely subsystem: extrapolation boundaries. Investigation: two first-layer conversions are counted once, with unchanged prefix/full/terminal signatures verified on every ASIC. Resolution: corrected and disclosed as an estimate.
- Observed anomaly: the experimental tiny16 SwiGLU differed numerically. Evidence: staged prototype receipts and measured BinaryNg SFPU source/configuration. Affected path: unselected external experiment. Control: real layer 0/3 gate/up was exact. Likely subsystem: default unpack/FPU multiply versus selected direct-FP32-unpack/SFPU multiply. Investigation: the corrected recipe passes all-device component equality and full-window equality before/after 300 replays per arm. Clean windows measure 193.797350 to 194.179593 us and 191.883817 to 192.193408 us. Resolution: fixed; this configuration is unselected because it establishes no gain. No global tiny-tile/fusion rejection is claimed.
- Observed anomaly: earlier sampler/profiler probes failed. Evidence: retained triage, failed setup logs, legal pad128 sampler controls and current profiles. Affected path: discarded experiments. Control: pad128 matches real logits and 13 boundary/tie cases but is slower; final matched profiles complete successfully. Likely subsystems: sampler shape handshake and profiler launcher configuration. Investigation: adapted shapes/settings were tested before rejection. Resolution: controlled; failed timings are excluded from the final result.

## Scope Inspected

- Goal/skills: installed 07-optimized-full-model.txt; stage-review, optimize, multichip, tracing, qualitative-check, target-workload and device-usage requirements.
- Code: multichip_decoder.py, precision.py, compact_b1_recurrence.py, fp32_head_rows.py, new NoC kernels, unchanged FP32 compute, generator/model/sampling boundaries, benchmark/quality/profile harnesses and external setup-inventory wrappers. The B1/T1/unmasked/nonprefill guard preserves inactive, prepared, larger-batch and prefill paths. Runtime rows retain BF4/LoFi projections, FP32/HiFi4 recurrence and BF8/HiFi2 head.
- Artifacts: final-v4 native/strict/quality receipts, both final-v4 profiles and CSVs, watcher pair, domain-aware memory receipts, adapted head/sampler/projection/prefill trials, coherent tiny16 v8, and final hardware-health receipt under /home/mvasiljev/build/qwen38-swe/local-tsu-ttft. Canonical README, work_log, perf_summary, memory_summary, workload and context contract agree with those results.
- Commands: read-only source/log inspection; stdlib JSON, CSV, hash, AST and allocation arithmetic; final free-running checker by file path. No TTNN import, device/server execution or implementation edit occurred in this review. This report is the reviewer's sole file write.

## Residual Risk

- Acceptance is for the documented standalone B1 opt-in workload. Serving integration, concurrent capacity and maximum-context accuracy retain their existing evidence and are not new claims from this optimization. Native context and selected precision remain unchanged.
- The local checkpoint commit is the stage owner's next workflow step after this clean review; no push is part of this acceptance.
