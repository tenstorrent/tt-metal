# Final benchmark stage work log

## Current checkpoint

Timed attempt 1 completed successfully. All measurements and required checks are
finished; independent review returned clean-pass with no required work. Evidence
packaging and the local evidence commit are complete; closure details follow below. Accuracy used
one frozen GPQA document out of 198 and one IFEval document out of 541. Both
responses stopped normally, with nonblank final content and no token-cap stop.
Completion lengths were 1826 and 12504 tokens with the unchanged 32768 cap.
Flexible GPQA extraction and IFEval extraction were each 1/1; GPQA strict extraction
was 0/1 because the upstream zero-shot prompt does not request its required literal
phrase. Both upstream scores remain unchanged. Strict-match is a format diagnostic;
no accuracy verdict is made. See [protocol review](run/protocol-review.json).

- B2 4096/128: 8/8 completed, failed 0, wall 2030.509356 s.
- One user on B2, 128/128: 4/4 completed, failed 0, wall 82.113245 s.
- One user on B2, 32768/128: 2/2 completed, failed 0, wall 3923.564069 s.
- Runner client clock: 7627.747302 / 9000 s. Runner setup: 2503.071863 / 3600 s.
  Pre-invocation setup: 5256.310899 s, recorded separately. Client time through
  final reporting and independent review: 8298.676009 / 9000 s; see
  [reporting-timing.json](run/reporting-timing.json).

The B2 collector mapped all 8 requests, with 4096 prefill and 1016 decode events.
Every observed decode event had one active sequence. Nominal FIFO host-observation
attribution was 1920.376579 s prefill and 110.126940 s decode; polling uncertainty
and conservative byte-accounting limitations are explicit in the report and
[b2-measurement-audit.json](run/b2-measurement-audit.json).

The evidence check returned exit 1 with exactly the documented strict-format
advisory, and no critical failure. The context check returned exit 0 at full
262144 capacity; historical Stage 9/10 textual cap advisories remain in its log.
All eight recorded implementation/context/precision source hashes still match.
The final observer export retained 44644 events, with zero errors or pending
tickets. Owned shutdown succeeded after observing zero running/waiting requests;
no owned processes or device holders remained. The enclosing reservation was not
changed and no reset was issued.

CPU vLLM performance-client logs contain Triton nanobind reference-leak warnings
at interpreter teardown. Every client exited successfully and all measured
request counts/lengths validated. These are distinct from the TT phase observer,
which exported without errors. The CPU help-command control did not reproduce
the warning; it is not claimed as a reproduction or a repair.

See [REPORT.md](run/REPORT.md), [check results](run/check-summary.json), and
[evidence restoration instructions](EVIDENCE.md).

## Chronological setup and execution record

The paragraphs below record successive checkpoints. Statements about pending
work or the then-current server apply at that checkpoint, not to the final state.

Stage 10 parent: `28ca67d4`. Selected policy `combined_shared4_mixed`, full48, TP4/EP4 P300x2, C262144, prefix caching off for recurrent-state compatibility. Stage10 validated resident2; 32 HTTP users were queued, not32 resident slots. Stage11 will measure B2; defaults1/8/16/32 are unvalidated and skipped with explicit reasons. A later independent audit confirmed that historical B1 standalone generator rows explicitly disclaim vLLM-capacity validation. Stage10 c1 ran on the resident2 server. No new one-slot configuration is substituted for the requested working Stage10 config.

No server was running at start and no process held `/dev/tenstorrent/*`. Existing enclosing reservation and device assignment0,1,2,3 are preserved; no acquisition/release command. Source modifications from prior stages are preserved. Installed tt-autodebug0.1.8 and tt-model-bringup0.1.24 are enabled (codex plugin list).

Setup before timed invocation includes separate lm-eval/vLLM0.26 clients, dataset downloads/freezing, phase collector native query extension, instrumentation validation, initial server startup/warmup and pilot. These are recorded separately from the runner's client and setup clocks. No timed benchmark attempt invoked yet.

Accuracy protocol: native HF template once via structured chat; exact checkpoint/tokenizer revision; GPQA Diamond zero-shot CoT and IFEval; model-card xhigh thinking with T1/top_p.95/top_k20. Output cap initially32768; sample count determined from speed pilots. No accuracy threshold or verdict.

Phase timing design: model-local binding to existing C++ EventQuery, host-only nonblocking polling of CQ0 completion counters; no additional device synchronization or live profiler. Prefill completion events include all chunks; decode uses existing async readback event and records subsequent host token processing. Requests mapped using first KV page + monotonic slot generations; measured client start_times exclude probes/warmups. Work estimates document selected dtypes, active expert sharing bounds and physical traffic assumptions.

### Local commits and preparation checkpoints

Pending stage preparation and evidence commits; never push.

Preparation commits: `7184c2f226222eb86a6e09c037ca36ae2ecea1f2` and `48220c1057d97ce92da36a1d660e0e6d8e9092b2`; hooks passed. Ten recorder/adapter CPU regressions and two collector regression tests pass. Native extension import succeeds. Initial B2 server is owned by setup/owned-server.json; launched at source48220c1 with full48 runtime precision evidence and all four devices. No reset. Collector readiness check passes before any accuracy request. Setup perf pilot uses upstream vLLM0.26 at4096/128, c2,2measured requests plus its unmeasured initial probe.

Exact source provenance: all2576 upstream vLLM Python/template files match v0.31.0 `db9527a46873454610df6dbedf79a36d6bf1a7f6`; generated _version.py is the sole extra. All14737 lm-eval Python/task YAML files match v0.4.13 `ddd67220430a2470529f25fd5c05a576ca1057a0`. See setup/vllm-server-provenance.json.

Scope audit before invocation: doc/optimized_vllm/run_server.sh, perf_summary.json and launch provenance fix resident2; full_model_comparison standalone B1 rows carry validated_capacity_claim=false. Configured batch_sizes=[1,2,8,16,32], measured=[2], others skipped with distinct factual reasons. The required lone-user rows remain on B2. Client bound9000s; explicit setup bound3600s allows the Stage10-observed 32K warmup (~1954s) plus 4K warmup.

First pre-invocation performance pilot completed2/2 in534.728s, but server input counts4096/4529 violate exact4096. Preserved setup/perf-b2.json. CPU diagnosis found divergent tokenizer segmentation and upstream first-only alignment; model-local token-ID transport wrapper preserves upstream sampling/timing and retains all exact inputs. A fresh transport-fixed setup pilot is running before accuracy.

Collector/token transport commit: 772f7a07e58a35697d9353cafbe6adc16db1dbf7 (hooks pass after formatting). Sixteen CPU recorder, adapter, partition and exact-prompt mapping regressions pass. Actual server native Qwen2Tokenizer/transformers5.14.1 renders the exact same native chat template as the newer client; setup/template-smoke-server.json confirms xhigh system instruction and assistant think prefix without importing TTNN. Runtime model source remains launch commit48220c1; only host-side accounting/transport changed afterward.

ABI repair commit: 45bdcf46660436ef99f5c1c502b65ed2865cf544; hooks and16 CPU tests pass. Old stable binary and original server launch/empty phase snapshot remain under setup/. Corrected token pilot was exact-shape but could not export phases; it is diagnostic, not a completed roofline profile. No benchmark_stage run has been invoked. Owned idle server stopped, allfour devices unheld, no reset. New launch will use this commit with same model/precision/context and Stage10 resident2 config.

Live ABI smoke passes on the replacement full48 server: exact8in/4out,1 prefill+3 decode events, no pending tickets/errors, all existing host decode completions retained. Evidence setup/abi-fixed/native-smoke-proof.json and native-smoke-phases.json. Full4096/128 collector pilot follows before accuracy.

Native FIFO validator host-only commit: 377741b2ea24bc2b2ae496e50b9eb121e3811e98;10 collector CPU tests pass, including rejection of native event regression/reuse/CQ mismatch. Runtime model remains45bdcf46; native8/4 smoke validates1050–1053 on CQ0 across the full1x4 mesh.

Full pre-invocation collector pilot passed: 2/2 exact4096in/128out requests, native event FIFO validated, zero missing/pending events; upstream load_roofline required profile2 accepted. Independent read-only audit found no blocker. Prefill479.662905s+decode27.489294s versus client507.158617s. Observer gaps/brackets are retained and point phase utilization must be qualified as FIFO-observation estimates; modeled DRAM is not measured transactions. Pilot had one active decode slot per call despite resident2/concurrency2. Accuracy2+2 speed/protocol pilot started with unchanged intended generation policy.

Accuracy pilot completed4/4 in1913.705s,17735completiontokens,9.267tokens/s; GPQA lengths2660/6914, IFEval812/7349. Allstop; no blank-at-cap. GPQA flexible extraction2/2, strict0/2; upstream zero-shot prompt lacks required strict literal phrase. Both upstream filters retained; strict metric is a format diagnostic, flexible is the parsed-choice result. IFEval extraction2/2. Selected1/task from unchanged common seed/ranking, based on p90 time sensitivity and full performance/reporting reserve (setup/sample-sizing.json), before any cap/thinking change. Final manifest retains population/doc hashes.

Report finalizer commits:690672a64c270c0b8b8c08b80a4ee37f67245b06 and4f4524de2c2a9cdd81582f027ba3bf70beab671b. Lossless scope-bundle archival helper:6ad77f962804512eae44c28551dd36d404e1e847. CPU fixtures and commit hooks passed; runtime remains45bdcf46.

Timed attempt1 invoked 2026-11-18T02:42:23.310899+00:00; setup before invocation5256.311s, logged separately. Config benchmark/run_config.json; fresh output benchmark/run. Client9000s; setup3600s; outer process bound13000s.

Host report precision improvement: `c3c54169304cda0af3ff76d15e35f2120d71d9af`.
The appendix retains six decimal places for request/output/total-token throughput,
avoiding a misleading rounded `0.00` requests/s on long-context rows. Runtime
source and generation settings remain unchanged.

## Final evidence closure

Independent review: [STAGE_REVIEW.md](run/STAGE_REVIEW.md), clean-pass with no
required work. Full-scope lossless evidence bundles are finalized for setup and
run. Production restoration verified all 153 setup originals and all 50 run
originals byte-identical, without overwriting them. The run bundle includes the
final report, reporting clock, check summary and independent review. Raw evidence
remains unchanged. Local evidence commit follows; no push is authorized or made.

Evidence commit: `5e5bc63bb40e3e5fd38ba297e38d6c80b0c3e970`. Repository commit hooks all passed.
At 2026-11-18T05:43:13.943234+00:00, client elapsed
including report, review, archive verification and the evidence commit was
**8347.560416 / 9000 seconds**. Setup remained **2503.071863 / 3600
seconds**, with pre-invocation preparation separate. This final log-only commit
records the evidence SHA. No push was made; unrelated work remains untouched.
