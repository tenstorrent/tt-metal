# Stage 10 work log

Status: in progress. Baseline implementation f2abf54e (runtime checkpoint9706b2bc); selected head_bfp4 policy and context262144 retained. No optimized result or clean-pass claimed.

Startup read run_contract.json, agentic inputs, installed startup, serving/plugin-change/prefill, optimize, tracing and device requirements. AutoDebug enabled in personal Codex home. Existing user changes and untracked artifacts preserved. No profiler collection, plugin runtime change or push authorized by this stage.

Pre-change measurements are preserved in baseline.json with source paths: full60-layer4096/128/B1/C1 TTFT754.870287ms, meanTPOT28.311535ms, decode35.321292t/s/user. CI-shaped100/100/32 burst measured on local QB2; label is workload parity, not remote execution. Full-model selected4096/128/B1 reference34.83t/s/user. These are earlier-stage measurements, not stage10 after measurements.

Optimization hypothesis: per-row serial prefill repeats all60 layers and weight traffic. Prepared batched dense passes should reduce burst TTFT while preserving independent per-row attention/cache and exact last-token logits. Async decode already uses persistent token/position state and nonblocking split replay, so decoder math is retained initially. Implementation effort delegated per agentic-serving; all hardware jobs remain coordinator-owned and serialized.

2026-11-05: /proc device-owner audit empty (startup_device_owners.json). Started serial128/128/B1 baseline through tests/run_serving.sh, serve,benchmark, prefix caching enabled, no CI profile, output readiness_vllm/stage10_serial_128. Inherited bounded7200s runner. Same stock runtime plugin90f33d2 test-only descendant of ca7a182. No watcher/profiler attached. Runtime was imported before candidate changes.

Validation order: focused batched/solo exactness on real layers0/5 and boundary lengths; prepared-shape watcher and allocation-tracked replay with program-cache checks; full60 burst versus serial; primary4096/128 and128/128 regression; full sampling and qualitative controls; same-server mixed trace reuse, prefix/growing traffic; batch8/16/32 and prefill/host-contention diagnostics; final independent stage review then local isolated commits. Any failed gate remains work.

Telemetry packet initialized from exact supplied template; optimized performance remains null pending final measurement.

Serial128 run passed exit0: all128 output tokens, TTFT111.007509ms, TPOT25.695860ms, decode38.916775t/s/user. Server shut down cleanly; /proc owner audit empty before next job. B8/S128 direct selected-policy layers0/5 exactness probe passed across all4 shards for all8 rows: logits and valid K/V bit-equal (probe_b8_s128.json). No full-model/serving performance is inferred from this reduced probe. Candidate remains off by default. Removed a temporary list reference before sampler replay so last-group logits cannot remain alive across the trace boundary; tracked acceptance follows.

Tracked B8/S128 generator acceptance passes both row orders, exact full-vocab logits and valid K/V on all devices, matching32-token continuations. Tracking1 with skip-program-cache0; program entries2894 stable through both prefill and final decode, three original trace IDs reused. Evidence tracked_b8_s128.json/log. B32 mixed32/128/129 plus positive control running next under same tracking settings.

B32 mixed32/128/129 generator check: all6 row-order cases pass exact logits/cache, greedy continuation and stable program/trace counts. The run exits1 at the positive-control cleanup: deliberate TILE upload introduced a persistent TilizeDeviceOperation program-cache allocation. Deleting the tensor cannot remove that late program. This is a test-control construction bug, not accepted production behavior. Changed control to uint32 ROW_MAJOR copy (no tilize), retaining the failed tracked_b32_mixed.json/log. Fresh tracked_b32_control run tests same tracker with the corrected control and clean replay. No allocation exemptions or program-cache skips.

Corrected B32 positive control passed: tracker rejected exactly one ttnn.to_device buffer; deletion restored clean replay with program-cache tracking included. Both row orders of B32x128 pass. Evidence tracked_b32_control.json. Testing a wider candidate (one32x128 pass,4096 physical rows) against four8x128 groups using same selected policy and allocations. No production default change yet.

Wide4096-row candidate fails exactness before timing; all4 shards differ already in layer0 K/V, max terminal-logit delta9.15625. It remains a probe-only constructor option, never selected. AutoDebug launched with personal Codex home, gpt-6-astra/medium, API-key variables removed, inspection-only. Known bubblewrap preflight limitation from stage09 justifies explicit child-sandbox skip under unrestricted operator authorization. AutoFix follows diagnosis. Meanwhile watcher B32 full acceptance matrix validates the passing1024-row candidate, one local hardware job at a time.

Watcher initial mesh startup failed before model construction: fabric_erisc_router29264B exceeds ACTIVE_ETH config26624B by2640B. Device closed cleanly. Retried exactly as optimize tooling guidance prescribes, TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1; compute assertions remain enabled. No profiler or reset. Original watcher_b32_all.log preserved; scoped compute run watcher_b32_compute.json/log pending.

AutoFix started from `autodebug_wide.md`: isolate the normalization dispatch boundary before changing production code. `AUTOFIX.md` records hypotheses; independent agents own source inspection and the focused probe, while root serializes hardware.

Independent AutoFix source audit found no concrete first-QKV matmul defect:
wide M4096 equals N4096, so the strict M>N orientation branch is not crossed;
explicit M4/K8/N16/sub1x8 keeps K reduction geometry. If bounded hidden norms
fail to restore equality, compare identical normalized inputs through wide
DRAM and partitioned DRAM projection, then B8 L1/DRAM, head splitting,
unweighted head norm and BF8 cast separately. This is a hypothesis audit,
not device proof. A focused `tests/probe_batched_norm.py` is ready; production
normalization remains unchanged. Source hooks pass.

The bounded candidate alone plausibly reduces a 32-request burst to four B8
prefill groups. Real serving performance, not that operation-count estimate,
will determine acceptance. Fresh serial and candidate burst arms retain the
runner scheduler budget and record chunking/admission behavior.

Compute-watcher acceptance completed exit0: `watcher_b32_compute.json`,22/22
cases pass, logical lengths5/31/32/33/127/128/129/257/640/1000/1024 in two
orders atB32, reduced real layers0/5; all prepared batched buckets warmed.
ETH instrumentation exclusion remains explicit due the measured firmware
capacity failure, with compute assertions enabled. Subsequent source refinements
add startup-only physical-buffer accounting and repeat the last valid RoPE
position in padded positions (per prefill contract); final-path validation
will follow. These do not constitute a new measured serving performance result.

Wide-candidate AutoFix evidence: `norm_discriminator_b32.json` and `probe_b32_bounded_norm.json` prove bounded named norms restore reduced logits/KV exactly. See `autofix_wide.md`; no production norm change. Fresh serial B32 server now runs benchmark (128/128 single request plus CI100/100/32) and cold128/32/32 prefill burst, with original2048 scheduler token budget.

Fresh serial B32 runner exit0. Cold128input/32output/32requests/C32 burst
meanTTFT1981.696865ms,p992627.053158ms,cache4096queries/0hits,all32completed.
CI-shaped100/100/32 aggregate551.296602tokens/s,meanTTFT2065.012946ms,
meanTPOT37.594690ms,all32completed. These fresh numbers supersede stage09
for the paired burst comparison; they are not headline single-user decode.
The extra128/128/C1 row used serverB32 (TTFT215.67ms,TPOT28.84ms) and is
only a capacity-configuration diagnostic, not the max-num-seqs1 regression.
Candidate arm launched with identical runner args plusenvGEMMA4_BATCHED_PREFILL=1
and the serial-result path. Runtime audit records whole-server counters,
including benchmark warmups, explicitly separate from timed-window counts.

First candidate serving attempt failed before any CI burst request completed:
`stage10_grouped_burst/server.log` shows `TypeError: Object of type int64 is not
JSON serializable` in the new generator_batched_prefill event, after the first
group computed. vLLM supplies NumPy lengths; reduced tests supplied Python lists.
Changed only event lengths/physical_size to explicitPythonint, and changed the
generator acceptance harness to submit numpy.int64 lengths as vLLM does.
Source hooks pass; paired v2 serving rerun in progress. Failed raw artifacts
are retained and excluded from performance claims. API/EngineCore are gone;
/proc device-owner scan was empty before the v2 launch. No reset was needed.
First candidate preparation measured55,181,312bytes/device in new input buffers,
3180programs before/aftercapture,3traces; these are allocation/preparation data,
not a successful serving result.

Paired candidatev2 runner exit0; all32burst requests completed. Cold128/32/32/C32
meanTTFT854.132602ms,p991158.485013ms: meanratio0.431011<=0.5,
p99ratio0.440983<=0.6,config_matches=true,4096cachequeries/0hits in botharms.
CI100/100/32 outputthroughput740.355375tokens/s,meanTTFT877.821669ms,
meanTPOT34.603665ms. All benchmark lifetime sampling steps are audited separately.
This passes the burst candidate gate, not final stage closure. Full60 bucket
calibration now runs with advertisedcontext262144 and pool1048576, ABBA2;
it measures host-wall prefill cost only, not device time or vLLM headline.

Allocator comparison (`memory_comparison.json`): candidate prepared32,321,636,864B
perdevice versus serial32,248,072,704B,delta73,564,160B. New explicit input
buffers55,181,312B; difference includes program-cache/runtime reservations.
UsableDRAM34,044,513,280B; after retaining805,306,368B transient budget,
917,570,048B remains. Trace reserve134,217,728B is already excluded from usable
and is not subtracted again. The short group cap keeps total hidden rows<=1024;
normalization/dense hidden volumes do not exceed the prior1024-row solo bound.
Terminal logicalrows<=32 occupy the same tile-height as prior one-row terminal,
and retained per-row logits are bounded by the existing31-extra-row budget.
This is an analytical carry-forward of the prior transient bound, not a new
full-stack transient peak measurement. No serving profiler was used.

Full60 bucket calibration exit0: all15prepared groups are faster than corresponding solo calls, stableprogramcache in everyABBA sample. Ratios range0.093880..0.616487; context262144,pool1048576. Source `bucket_calibration.json`; hostwall only, no sampler or serving claim. Final-input-type equality matrices B2/B4/B8 now run sequentially over all11required lengths in bothorders.

Final NumPy-length B2/B4/B8 matrices pass22/22cases each, all11required logical lengths in bothorders. `equality_summary.json` records compact scope/counts and exact32-token continuation status; raw row-level logits/KV checks remain local artifacts. Final-source B32 compute-watcher matrix now running with same documented ETH exclusion.

Final-source B32 watcher passes22/22cases, all32generated tokens exact perrow,
programcache stable. Together B2/B4/B8/B32 random matrices compare1012rows.
Tracked mixedchat attempt stopped beforecases because apply_chat_template's
returned dictionary was treated as tokenIDs. Added explicitreturn_dict=False;
pinned-tokenizer hostprobe confirms list[int] length2824, covering fixturelengths.
No model outputs failed. Failed fixture artifact retained; v2 tracked run now
covers12mixedchatcases and unsafe-allocation positivecontrol.

Tracked B32 mixedchat v2 exit0:12/12cases pass with tracking1/skipprogramcache0. Positivecontrol detects exactlyone live ttnn.to_device buffer; afterrelease cleanreplay passes. Source `tracked_b32_chat_v2.json`. Smallerbatch chat checks and current adapter stale/deferred-read check now serialized.

B2/B4/B8 real-chat prefix equality each passes6cases (5/33/127 tokens,twoorders). Current tracked adapter stale-token/current-position, page-only-refresh/deferred-read and uint32 seed controls pass (`adapter_state_current.json`), with allocationtracking1 and programcache included. Full60/B32 exact logits/KV/continuation comparison now runs5/33/96/127/128 real-chat prefix lengths in twoorders. Its2048-token cache is an explicit correctness fixture, not a serving, quality, performance or advertised-context measurement.

Full60 B32 chat equality completed: all 10 cases pass, including nonaligned lengths5/33/127 and aligned96/128, both row orders. Starting primary B1 128/128 regression.

Candidate B1 128/128 regression passes: TTFT109.671220ms vs111.007509ms (ratio0.987962), TPOT25.684934ms vs25.695860ms (ratio0.999575). Batched prefill is now default-on; env0 remains a serial control. Starting fresh full60 B32 tracked synchronous mixed traffic on final runtime source.

Final-source tracked synchronous full60/B32 mixed test passes68requests, all checks, all four batched physical sizes32/64/96/128, three stable traces. Sampling:2914 device decode,34 device prefill,12 host-compatible intermediate prefill steps. Clean shutdown and no device owners before final async server launch. Exact commands/source hashes: readiness_vllm/stage10_sync_tracked/launch_manifest.json.

Final-default B32 server passes full sampling76pass75skip, shared chat outputs and prefix cache. Cold128/32/C32 burst meanTTFT857.826786ms,p991160.511272ms; serial ratios0.432874877 and0.441754012 meet gates and reproduce candidate. Local CI100/100/32 aggregate741.364865tokens/s. Runtime audit is bounded to the performance client interval; all7prefill/259decode steps device sampled.

Async mixed traffic passes all68requests and sync/async controls, with34deviceprefill/2914devicedecode/12host-compatibleprefill steps. Growing client failed before HTTP due to direct-file invocation of a relative-import module. Corrected command to python -m; no model/device failure. Initial server cleaned up, no device owners. Remaining growing/load checks run in stage10_final_load on identical frozen source; passed suites are retained.

Corrected growing-context module invocation passes160requests:32conversations,5turns at4096/8192/16384/24576/32768 tokens,8outputtokens/turn. Prefix reuse/ceiling0.994669269; late0.996534388; zero preemptions and zero captures after warmup. Sampling476host-compatible intermediate prefill,44deviceprefill,287devicedecode. Artifact: readiness_vllm/stage10_final_load/growing_traffic.json.

Growing160requests pass. Load diagnostic was invalidated by repeated len(tokenizer) in per-token HTTP validation, causing CPU backlog after server finished. No model/adapter change; cache vocabulary_size once. Interrupted owned client after server idle and confirmed no stress children. Runner cleaned up; corrected load-only rerun is stage10_final_load_v2. Prior load timings are not acceptance evidence.

Corrected load diagnostic passes98requests and cleans16stressworkers.31decoding requests remain live through32768-token prefill (9.734428s;279other tokens). B32/C32 128input/256output decode-window delivery983.003tok/s idle,650.398tok/s busy. Process CPU/step187.480ms idle,83.228ms busy sums all engine threads, not Python-exclusive or wall time; lower CPU under contention is not faster decode. Artifact: readiness_vllm/stage10_final_load_v2/load_costs.json. Server cleaned; no device owners before final profiles.

Final-default exact headline4096input128outputB1C1 passes: TTFT754.431132ms, TPOT28.312670ms, decode35.319877tokens/s/user, outputthroughput29.421170tokens/s. Server command/config and benchmark config match baseline; effective timed_trace dataset and exact token totals verified. Headline sampling1host-compatible intermediate prefill,1devicefinalprefill,127devicedecode. Root readiness artifacts reconciled with explicit source index; prior root checkpoint archived in stage09_root_snapshot.

Final default B8/B16 sampling smoke3pass1unchangedskip each, concurrent128/128 profiles and32-requestCI profiles all pass. All final runtime launch hashes match. No TT device owners, vLLM/API/EngineCore runners or stress workers remain; cleanup.json records observation. Stage checks and independent review follow; no further hardware work is planned unless a finding requires it.

Final stage checker returns1(advisory): no degenerate output, context262144 and required batched-prefill gates pass; agentic measurements complete. Two carried structural advisories remain documented: BF16 lookup/BFP4 head require distinct representations; pinned pre-weight-load pool query uses fixed synthetic sizing, validated by allocator fit and growing traffic. No critical/missing checks. Hooks pass0. Independent review requested against raw evidence and live worktree.

## Final independent review

Fresh stage-review returned **clean-pass**, with no required work; see `stage_review.md`. The reviewed runtime matches all seven final launch manifests. Statistical headline uncertainty, analytical transient reserve and inherited quality limits remain documented. Stage checks have two reviewed structural advisories, no critical findings; hooks pass. Local source/docs checkpoint and telemetry closure follow; no push.

## Local checkpoint closure

Implementation, tests and reviewed summaries committed locally as `a785c2626c4552af083c4cc190c325d615d392c3`. Commit hooks passed using the existing workspace PRE_COMMIT_HOME after the default home cache was inaccessible. No runtime source changed after validation. Plugin remains at authorized test-only commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a`, stock runtime base `ca7a18246e4111629ed6cb4a7d1192002749fccf`; no stage-10 plugin or vLLM changes. No push. The following documentation-only closure commit is recorded in local `commit_manifest.json`. Exact-ID telemetry is finalized in the requested bringup artifacts packet; prohibited device/roofline fields remain null.

## Runner advisory reproduction and refutation (2026-11-06)

Ran the supplied installed `10-optimized-vllm.check.sh` with
`MODEL_DIR=models/autoports/google_gemma_4_31b_it` from this checkout, capturing
`doc/optimized_vllm/check_reproduction.log`. Exit was 1, reproducing exactly the
reported two ADVISORY lines. Degeneracy, context 262144 and prefill features pass.
The installed checker `check_agentic_serving.py:check_memory` deliberately returns
1 for copies >1 with copy_reason and fixed pool sizing with pool_sizing_reason;
missing reasons would return 2. The agentic-serving skill explicitly permits
these documented advisory shortfalls. Thus exit 1 is real; interpreting these
particular findings as a failed serving correctness/completion gate is refuted.
No checker was edited, warning suppressed, or ledger policy relabeled.

The two representations are real: `tt/model.py` constructs row-major BF16
hidden-sharded embedding and transposed tiled BFP4 vocabulary-sharded head.
The ledger accounts for 704643072 +198180864 =902823936 bytes per device.
These representations cannot alias; selected datatype and token-out evidence
remain in `doc/datatype_sweep/selected_default/`. Dropping either representation
would change the selected implementation and require new correctness/performance
work; it is not an evidence-only repair.

The fixed pool is also real. `doc/vllm_integration/pool_plan.json` records the
pinned worker's pre-load query, normalized 36896 IDs and failed smaller-pool
control. `memory_comparison.json` measures prepared allocation32321636864 of
34044513280 usable bytes per device, leaving1722876416 bytes. Retaining the
analytical805306368-byte transient budget leaves917570048 bytes. This is not a
measured full-stack peak. Final `stage10_final_load/growing_traffic.json` has
160 completed requests,32 conversations growing to32768 tokens, reuse0.994669269,
late reuse0.996534388, zero preemptions and stable allocation endpoints.
These data justify the documented fixed pool; they do not make sizing dynamic.

Found and fixed a documentation inconsistency: `doc/agentic_serving.json:ledger`
still described pre-batched preparation totals. Reconciled prepared state,
allocated/free bytes and transient budget to the existing stage10 measurements.
Runtime source and serving settings are unchanged. Existing independent stage
review and unaffected device evidence remain applicable. No new hardware runs,
benchmarks, profiler collection, remote dispatch or publication were needed.

Created the previously absent exact-attempt telemetry packet from its supplied
template at `bringup/artifacts/multigoal-runs/v3-20261105T172350Z/telemetry/packets/bf89528a-30b5-4467-b42f-617ed982c062.json`.
It uses existing final4096/128/B1/C1 vLLM evidence: TTFT754.4311319943517ms,
decode35.31987658862003tokens/s/user. Device time and rooflines remain null;
accuracy is empty with a missing reason. No measurement was fabricated.

Reverification: the supplied stage script again exits1 with exactly the same two
advisories (`check_reverification.log`); all other checks pass. Fresh independent
stage-review agent `/root/advisory_review` returned **clean-pass**, no required
work, after checking the original goal, checker, code, raw memory/growing traffic,
benchmark and packet/template. This focused review supplements the existing full
review. The two shortfalls remain for goal12 pricing. Repository hooks pass with
`PRE_COMMIT_HOME=$PWD/bringup/artifacts/pre-commit` (default cache inaccessible).
Local repair checkpoint follows; no push.
