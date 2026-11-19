# d810 selection assessment and final adoption plan

**d810 remains a candidate.** Its source audit and layer34 stream pass, but its single measured decode advantage over85ad is0.10423% (1.511microseconds). That is not enough evidence of a repeatable whole-decoder improvement. Production is not changed by this assessment. The active near-capacity c823 run is neither interrupted nor counted as completed evidence.

## Source identities and observed results

| Source | Exact SHA256 | S256 warmed prefill ms | S256 traced decode ms |
|---|---|---:|---:|
|44f87|`44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3`|484.135794919|1.450283686|
|85ad|`85adacf5df348f815e684a6ad1b1a36ecf12e862b72ef156681bfabb86dccd92`|478.206970263|1.449556323|
|d810|`d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`|478.809280088|1.448045485|

Sources: [44 benchmark](v6_transport_conv_norm_decay_l0_s256_10pf.json), [85 benchmark](v6_transpose_l0_s256_10pf.json), [d810 benchmark](v6_transpose_interleaved_l0_s256_10pf.json), with adjacent provenance. All use layer0, S256, context512,10prefill/60decode samples, the same device/versions, uninstrumented serial execution and identical reported PCC. d810 versus44 is0.15433% faster decode; d810 versus85 prefill is0.12595% slower. These small differences need repeat confirmation.

85ad changes only prefill matmul transpose reads relative to44; its decode graph is unchanged. d810 adds the weighted-tail reader to85 and changes decode transport while preserving arithmetic. Exact source scope: [d810 audit](v6_interleaved_tail_d810_source_audit.md), [transpose assessment](v6_matmul_transpose_integration_assessment.md). d810's [layer34 seed154 stream](v6_transpose_interleaved_layer34_seed154.json) passes8192decode+33continuation with no failed rows; prefill PCC0.9997892198726543 and output metrics match the prior correct graph. This cached-reference diagnostic does not replace fresh final acceptance.

The supplemental S1023/context262144 smoke is a different workload and harness: [c823](v6_long_benchmark_c823_smoke.json)4203.748804983ms prefill/2.016284969ms decode versus [d810](v6_long_benchmark_d810_smoke.json)1905.842443928ms/1.444863854ms, both PASS. Both have1warm+3timed prefill and60decode samples with the same pinned oracle and seed154. This supports a longer-prefill benefit overc823; it cannot rank d810 against44/85 or substitute for final S256 comparison. The running S262143 experiment has no completed-result claim here.

## Comparator eligibility:44 is already mandatory

The unchanged `check_fused_evidence.py` SHA256 is `65a2763f23e275831cf74d81f2bc87a0f83389552ada039393d89f944ece072b`. Its actual `check_comparator_endurance` passed all four44 records during this assessment, reading archived originals into temporary files without modifying repo evidence. Execution took0.031seconds, imported neither Torch nor TTNN, and did not run the full audit suite.

|44 record suffix, prefix `v6_transport_conv_norm_decay_`|Original JSON SHA256|Adjacent provenance SHA256|
|---|---|---|
|`layer0_seed913.json`|`236924dfaec7123de8a657bc7d1bd37c94aaaa32622ba6ec92e3404c5d2447e2`|`c7ac9213fc53490ceab218d3d21d7a91a1279f0ed47e23affd088f6489fbb8ae`|
|`layer0_seed42.json`|`ee10fb2595f5cd7d2779e801f5e6f7f44692dce241e83f901a9719560eb2163c`|`e68a7e6dcc6ff8f6044adaf89c8035a2bcd2305c857dedf64f17fdea50da3728`|
|`layer0_seed154.json`|`8804d82718cb7b5ee456535d158d0ce4891b07d4598d56d3416fabb5d8d7eb07`|`c7eafac77e50b2da3e6b1113c67aa9cef031729340704ec212b0d2a115ed05dd`|
|`layer34_seed154.json`|`c12326b6700f8daf3ceab579d080bbb919842b8a54731014815978eff6480b23`|`d12ffaeaa721b17ecaf675ffcc21b8a5415df2af2f3f8aa836c78065390b3c80`|

The required contract at checker lines77–172 is source-bound ordinary stress: prompt128,8192decode,33continuation, at least1complete repeat, unique/full position sets, finite output and contribution PCC≥0.995, determinism, one capture, allocation tracking including program cache, `decoder=fused`, serial device provenance, current exact harness/reference hashes and matching oracle parameters/source hashes. Cached HF **outputs** are allowed for comparator selection; state/output injection, diagnostic intervention harnesses, inconsistent snapshot identity or generic decoder labels are forbidden. Layer0 seeds913/42/154 are mandatory and cannot be removed by a manifest. Separate `representative_comparator_correctness` must include layer34 seed154; it does not relabel that run as layer0.

44 satisfies these gates already, so using803 alone would omit a faster correct comparator. 85 has the source-bound layer34 record but its three layer0 records were not found; obtain them before calling85 checker-eligible. The exact decode-AST relation to44 does not satisfy the checker's source-bound endurance requirement and is not a reason to weaken it. Keep85 in the selection experiment even while its formal eligibility is pending.

## Serial work plan after the active long run closes

1. Preserve the current production bytes and source pins. Do not promote d810 just to measure it. Use the existing snapshot stress harness for85 at layer0 seeds913,42,154, prompt128/decode8192/continuation33, tracking enabled with program cache. Ordinary cached-HF runs are acceptable for these controls. Write `v6_transpose_layer0_seed{seed}.json` plus logs/provenance and run the unchanged eligibility checker. Do not transplant44 evidence or change mandatory seeds.
2. Run three serial matched timing blocks after an identical warmup policy: block1 order44,85,d810; block2 d810,85,44; block3 85,44,d810. Every invocation uses the existing `benchmark_fused_snapshot.py`, exact snapshot pin, layer0/length256/context512, `--prefill-repeats 10 --decode-repeats 60`, same device, versions, environment, weights and default seeded input. Watcher/profiling/tracking must be disabled as in the existing clean benchmarks. Name each output `v6_selection_{44f87|85ad|d810}_repeat{1|2|3}.json`; retain every run and sample. No parallel CPU/device workload during measured intervals.
3. Compare within-block decode medians, their across-block variability, PCC/determinism, and prefill medians. A favorable single sample or pooled median that hides a reversed paired result is insufficient. If d810's advantage changes sign or is comparable to drift, record it as unresolved and continue performance work; do not claim that the current1.511microsecond edge proves improvement. No new percentage acceptance threshold is being introduced. These repeats establish whether the existing strict faster-decode contract is supported.
4. If d810 remains the consistently faster correct choice, adopt its exact frozen bytes as `tt/fused_decoder.py` and switch the source-audit entrypoint to the already-frozen d810 auditor. Record the new source SHA and exact byte/AST equality. If a formatter changes bytes, first make and audit a separately pinned candidate; do not reuse d810 runtime evidence under an unproved new identity. Production and the final harness stay frozen throughout final validation.
5. Regenerate final source-bound artifacts, including final layer0 S256/context51210PF/60decode and a full-kind benchmark with counts matching its historical a478 control. Run the unchanged `check_best_correct` against both eligible44 and85, plus the full-kind a478 control. Append **every complete matched repeat control** to the template as another comparison against `final_0.json`; do not pick the slowest control or discard an inconvenient result. This makes the final measured source beat the fastest observed correct control as well as the original records. If that fails, it remains work, not permission to fall back to803 orc823 as the sole control.
6. Complete the separate final gates using the exact adopted source: fresh-HF stress at all required seeds/layers with≥3repeats, representative prefill/decode/continuation and paged-cache/nonaligned tests, max advertised context including relevant layer34, watcher-clean correctness, valid final prefill/decode profiler reports, no unnecessary measured movement, updated pattern/StopRule evidence and independent stage review. Existing historical capacity/profile records remain historical unless their source transfer is explicitly proven and accepted. Supplemental long-context timing does not replace these gates.

## Manifest and checker details

[Pending manifest template](v6_d810_best_correct_pending_template.json) retains44 and85 and the full a478 control. It deliberately references pending85 layer0 paths and final selected-source timings: it is **not runnable completion evidence yet**. Paths are relative to this endurance directory; moving it requires preserving/rebasing those paths.

At checker lines175–286, benchmark identities must use the actual measured snapshot SHA, not the live file recorded incidentally by snapshot provenance. Candidate source must equal final production; control snapshot hash must equal measured control; source identities must differ. Layer/kind/length/context, device, versions, and both sample counts must match. Benchmarks must be deterministic, runtime audit PASS, clean serial latency with positive finite samples, and all four recorded PCC fields≥0.995. The decoder median must be strictly lower; prefill before/after is reported without inventing an unchanged-prefill speed gate. The checker does not by itself establish statistical repeatability or discover all faster controls; the repeat/ranking assessment above closes that selection gap without changing the checker.

No production, checker, shared README/work log, or entrypoint edits are part of this preparation. No hardware or heavy host jobs were run. All writes are limited to this assessment and its pending template.
