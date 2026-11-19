# Full-model stage work log

Stage 6 started from clean tt-metal `7be8c434` on four unheld Blackhole P300c devices, mesh 1x4. Enabled tt-autodebug 0.1.8 and tt-model-bringup 0.1.24 verified in the active Codex plugin inventory. Environment: `source python_env/bin/activate; source /home/ttuser/qwen36-27b/lane-env.sh`. Bounded `tt-smi -ls --local` passed; no reset.

Current status: **all full-model gates passed; independent stage review round 1 clean-pass**. See the closure entries and local commit receipt at the end. The chronological entries below preserve the evidence and status as recorded during development; prior decoder max-context evidence was not used as full-model acceptance.

Read-only serving stack pin: `/home/ttuser/qwen36-27b/vllm-tt-plugin`, commit 042a94863ad27d88207bcf71abdf67122c6f4c27, current default main at resolution; `docs/install-vllm-tt.sh` recommends upstream vLLM 0.31.0. No stack install/server/adapter work authorized or started.

Preserved decoder: `tt/multichip_decoder.py` final29 defaults, hidden-sharded BF16 residuals, BF8 projection/KV policy and narrow GDN correction, GDN HiFi4, per-kind MLP fidelities, persistent Linear CCL, BF16 recurrent persistence/FP32 live recurrence. Rejections and carried items remain in `doc/optimized_multichip_decoder/final29_candidate_decisions.md` and `final29_handoff.md`.

Reference command (CPU, OMP/MKL threads16, HF_HUB_OFFLINE=1, timeout7200): `python -u -m readiness_check.generate --hf-model Qwen/Qwen3.6-27B --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 --model-dir models/autoports/qwen_qwen3_6_27b --output models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt`. Log `reference.log`. Cache exact-match semantics retained; checkpoint pinned by offline cached revision 8f33b98937c6031bb49e713dee3e8e185aafd26f.

## Initial wrapper and reference

`probe_full_model --length33` reached prefill but decode RoPE embedding rejected int32 positions. Device-side UINT32 conversion fixes it (`probe_positions.json`: finite prefill1x33x248320 and decode1x1x32x248320, 9.82s reduced probe). This is execution evidence only, not HF numerical agreement. Device close succeeded without reset.

Reference retry uses the exact local snapshot as `--hf-model` because the readiness CLI has no revision flag and the offline cache lacks `main`. All other arguments are unchanged. `reference_pinned.log` shows100 generated tokens and exact-match cache entry. Artifact metadata records the full snapshot path, not an unpinned HF main revision.

`probe_full_generator` first required list-valued penalty defaults for common SamplingGenerator (`generator_initial.log`), then device-only zeros/copy for RM convolution reset (`generator_params.log`). `generator_reset.log` reached capture and failed `Input Tensor is not allocated`, then hung in shutdown. Live evidence captured before timeout in `triage/sampling_capture.txt` and `triage/sampling_summary.txt`. No external process was killed.

AutoDebug sandbox preflight failed because unprivileged user namespaces are disabled (`autodebug_sampling_launch.log`). Retried inspection-only child with `AUTODEBUG_SKIP_CHILD_SANDBOX=1` under the existing unrestricted task authorization, log `autodebug_sampling_retry.log`. No permission escalation or remote modification.

Sampler alternate source pin: tt-transformers25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73, read-only checkout. No installation.

## AutoFix sampling ownership

Failing contract: common sampler may deallocate its returned collective tensors; decoder-owned persistent CCL outputs must survive every call and trace. Hypotheses: borrowed-output ownership violation (first); hidden reset host writes and incomplete capture cleanup (separate); post-capture prefill warmup gaps (later contract). Source and independent AutoDebug support the first cause. Minimal repair returns a clone at the sampling/CCL boundary, preserving decoder persistent buffers and policy; reset now copies shared immutable zero sources allocated before capture; failed graph capture ends/releases its partial trace.

After the600s timeout (`generator_reset`, exit124), no device holders remained. `timeout -k10 60 ...full_model_mesh_smoke` passed (`mesh_after_capture_timeout.log`); no reset, no lock cleanup, no process killed by the agent.

Discriminating layer-free test `probe_full_sampler` uses exact local-vocab62080/padded65536, physical32 candidates/rank and semantic k1/p0/temp1. `sampler_ownership_fixed2.json` passes three traced greedy replays for32 distinct expected winners. The explicit borrowed-output control fails on its second eager call with the original invalid-allocation signature (`sampler_ownership_negative.json`), without opening a trace or hanging.

Reduced generator now captures decode/sample/reset and generates5 tokens through4 decode replays (`generator_ownership_fixed.json`,17.41s process scope). Position advances33→37 with one request-boundary position write, zero free-running token feedback writes,5 sampled-token readbacks. Its2-layer output is not a language-quality verdict; full64 numerical gates remain. Allocation tracking and all supported prefill variants remain work.

### First all64 prefill result

`READINESS_TRACE_REGION_SIZE=100000000 timeout -k 20 1800 python -u -m readiness_check.run_prefill_check --model-dir models/autoports/qwen_qwen3_6_27b --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt --mesh-device P300x2 --fabric-config FABRIC_1D --build-kwargs '{"max_seq_len":1024}'` passed (exit0), top1/top5/top100 all100/100. Evidence: `prefill_first_all64.log`; this predates the bounded-prefill refactor and is baseline correctness, not final acceptance. All64 loading plus check took about6m45s. No hardware reset.

Traced all64 teacher forcing launched with the same arguments through `readiness_check.run_teacher_forcing`, log `teacher_forcing_first_all64.log`. This run is baseline decode evidence; arbitrary-length preparation remains open. Source investigation report `prefill_preparation_hypothesis.md` proves bounded-chunk changes are needed; AutoFix implementation is in progress. Fixed-slot GDN preservation and remap implementations are unverified pending reduced batch testing.

### First all64 traced teacher forcing

`readiness_check.run_teacher_forcing` with the same model/reference/mesh/fabric/max_seq_len1024 arguments passed, top1=99/100, top5=100/100, top100=100/100 (`teacher_forcing_first_all64.log`, exit0). Harness decode30.59t/s/u; TTFT17266.14ms includes prepare/capture under harness timing and is not warmed128/128 performance. This predates bounded preparation and fixed-slot repairs, so final gates still need rerun after reduced tests. Loaded model DRAM measured13,857,710,080 bytes/device, free20,221,012,992 bytes/device before cache/generator buffers. Recorded in context contract; supported context262144 unchanged.

### Bounded-prefill AutoFix control

The first reduced bounded probe failed on Python `ttnn.Shape` slicing, before the attention op; changed `hidden.shape[:2]` to `tuple(hidden.shape)[:2]`. Rerun `timeout -k20 600 python -u .../tests/probe_bounded_prefill.py --output .../doc/full_model/bounded_prefill_shape_fixed.json` passed native-vs-public lengths33,128,513 and resumes31/512 through545. Ordinary lengths and aligned resume had exact logits; resume31 had PCC0.999999382 and top1/top5/top100 agreement100%. GDN states and stable cache addresses passed. Real layers0/3, same accepted policy; this is wrapper parity, not all64/HF acceptance. Detailed metrics/log: `bounded_prefill_shape_fixed.{json,log}`. The initial failure remains `bounded_prefill_initial.{json,log}`.

Trace allocation positive control passed with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, detecting a deliberately live late allocation (`trace_alloc_tracking/positive_control.log`). No blanket tracker exemptions introduced.

### Fixed-slot repair probe

Reduced B3 slot probe passed (`slots_initial.{json,log}`). B32 passed under allocation tracking, including identity preparation, inactive recurrence preservation, whole-slot reverse remap, release of one slot, and replay afterward (`slots32_tracking.{json,log}`). Command: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 timeout -k20 1800 python -u .../tests/probe_full_slots.py --batch 32 --output .../slots32_tracking.json`. The probe uses diagnostic prefill-length1 preparation only because it isolates decode/slot ownership; it is not complete arbitrary-prefill preparation acceptance. Additional resident inactive token/RoPE/seed preservation checks are being run after extending this focused test.

Extended B32 resident inactive token/RoPE/RNG preservation passed under allocation tracking (`slots32_resident_tracking.{json,log}`), with5 decode/sample replays and only3 authoritative refreshes. Two resident steps made no token/position/RoPE refresh. Test inputs include token IDs above65536 to catch lossy integer masking.

### Bounded batch and trace reuse

`bounded_prefill_batch3.{json,log}` passed all five native/public comparisons (33,128,513 tokens; resume31 and512 through545). Ordinary and aligned-resume logits are exact; unaligned resume PCC0.999999267 with top1/top5/top100 agreement100%. Stable cache addresses and recurrent-state comparisons pass.

`trace_contract_quick.{json,log}` passed under `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`: retained decode/sample/reset trace IDs, constant program-cache count, exact persistent token feedback and on-device position/RoPE advance, zero steady host input rebuilds, and changed physical page writes. This prepares a diagnostic subset only, so it is not full arbitrary-prefill acceptance. Complete1..512 preparation is running in `trace_contract_full`. Exact commands are in `commands.json`.

Source-only serving audit `serving_gap_audit.md` identified legitimate capability exclusions and three generator fixes: disabled logprob sentinels, seeded reload without history, and low-level prefill sampling. Sentinel and absolute-position seed fixes are implemented; the prefill sampling implementation and tests are in progress. No vLLM adapter work.

At02:13UTC, the CPU HF qualitative job (PID364898, verified command) was lowered to nice10 for all79 threads so cold kernel compilation could progress on the16-logical-CPU host. No process was stopped and no numerical setting changed. These concurrent timings are not performance evidence.

### Source audit follow-ups before acceptance

`runtime_audit.md` found stale initial external page-table binding. Constructor now validates every external table and cache tensor's mesh/layout/memory contract, validates physical page bounds, and force-copies the supplied mapping into existing table storage. External cache ownership is an exclusive whole-pool lease; `prepare/reset` clears that pool, while request release clears only its GDN slot.

Low-level sampled prefill now stages last logits into fixed slots, invokes the existing sample trace once after all submitted rows, preserves omitted token/parameter/RNG state, and returns request-order token IDs. Mask backup is allocated before capture; temporary logits are released before replay. The focused probe additionally tests signed64 seeds, no-history resets, ignored stale resident host inputs, deterministic stochastic repeats and request/slot permutations. These additions remain hardware-unverified until their queued probe passes.

Explicit high-level host sampling now leaves host policy to its callback without applying the device top-k limit. A diagnostic mode-switch probe checks host/device greedy parity and return to retained device traces. Source pre-commit checks passed (`source_hooks_initial.log`); no acceptance inference is made from syntax/format checks.

Full1..512 reduced trace contract passed in2310.44s (`trace_contract_full.{json,log}`), including cold compilation of23691 cached programs. Counts stayed23691; three captures retained;43 decode and57 sample replays; zero token refreshes or full-logit readbacks. Fourteen position/RoPE writes were request setup only. This process loaded the generator before later prefill-sampling/external-cache repairs, so final source replay is still required. It closed normally before its2400s timeout; no reset.

### Prefill sampling and greedy ties

`prefill_sampling_tracking.{json,log}` passed under allocation tracking, including signed64 seed reset without history, stale resident input suppression, stochastic repeats and request/slot permutations. Intermediate chunks did not sample; one final sample trace handled all requested rows while preserving omitted state. External initial table rebinding and invalid bounds checks passed.

`sampler_ties.{json,log}` passed semantic greedy for changed unique winners, cross-rank ties and 65 tied maxima in one shard. Every chosen token maximizes the supplied logits. Tie identity is recorded separately from HF lowest-index identity. Exact commands are in `commands.json`.

`host_compatibility.{json,log}` passed with allocation tracking: five explicit full-logit reads, zero sampler replays in host mode, and exact greedy parity after switching back to the retained device traces. `watcher_generator.{json,log}` repeats this reduced real-layer test with watcher enabled and profiler disabled; passed and closed normally. Watcher raw capture: `watcher_capture.log`. These diagnostic timings are not performance measurements.

### Active-page ownership repair

`active_fill_page_proof.md` proves native fill writes only ceil(logical_count/32) pages per row. `bounded_prefill.py` now certifies precisely that active range, bounds it against physical cache size, and prepares both native branches using legal disjoint writes. Synthetic warmup overflow uses the native -1 skip sentinel, while attention reads use valid IDs. Original runtime tables are restored in finally. `active_fill_pages_cpu.json` passes6,152 checks; no accepted decoder source change.

`mixed_prefill_tracking.{json,log}` passed under allocation tracking: uniform33 uses accepted bulk fill despite repeated unused virtual entries; mixed33/128/513 matches solo prefill and traced decode across request/slot permutations. Physical64-page pool backs96 virtual entries. Trace IDs/program counts remain constant. Current-source complete1..512 B1 trace verification is running in `trace_contract_final`.

`trace_contract_final.{json,log}` passes on the repaired source (embedded SHA256 identities): all1..512 preparation, arbitrary and resumed prompt cases, three retained traces, stable23,705 programs and no free-running host token feedback. This is the complete reduced-layer trace/lifecycle acceptance, separate from all64 numerical/capacity/performance evidence.

### Final-source all64 numerical checks

`prefill_final_all64.log` passes AIME24 top1/top5/top100=100/100/100, all64 real layers and context2048 allocation. Loaded DRAM is unchanged13,857,710,080 bytes/device. Final traced teacher forcing is running. HF shared suite finished all12 records; reviewed late French/Fibonacci outputs are coherent thinking-mode planning.

The CPU-only `qualitative/prepare_autoregressive_reference.py` invokes the installed readiness runner's exact HF generation/cache helper for the later run_autoregressive CLI. It uses the identical stripped rendered-chat prompt IDs, pinned snapshot and128-token settings. It does not manually seed the cache or copy unrelated shared-suite outputs. No mesh opens in this helper.

`teacher_forcing_final_all64.log` passes final64-layer traced AIME24 decode top1/top5/top100=99/100/100. Harness decode31.04t/s/u; TTFT92978.04ms includes full1..512 preparation/capture and is not warmed latency. CPU HF reference completed during loading/preparation, before steady decode. Final standalone128/128 performance remains separate.

The official autoregressive HF cache helper completed128 tokens at exact key `hf_greedy-097759decf630d1938ab0fdd8ce612b9.json`. Readable control receipt is `qualitative/autoregressive_hf_cache_receipt.json`: coherent supervised/unsupervised explanation. The readiness harness strips final prompt whitespace, yielding24 prompt IDs versus25 in the unstripped template metadata; both CLI sides use the identical24 IDs. The canonical shared suite separately retains the exact full chat template.

### Reduced profiler collection repair

First `terminal_profile` executed all reduced model/terminal/sample windows and retained traces, but Tracy postprocessing failed because profiler markers overflowed during setup before the first drain. `terminal_profile.json` describes model execution only; it is not a valid complete profiler report. Exact failure: Op1042432 missing from cpp_device_perf_report.csv. Raw initial traces retained under `tracy_terminal/.logs/`; no reset. AutoFix adds diagnostic-only setup/warmup drains outside timed windows, leaving production model/generator unchanged.

`mixed_prefill_all64.{json,log}` passed allocation tracking with all64 real layers: mixed33/128/513 prompts match solo across slot orders; preemption release/rebuild produces exact prefill/decode logits and preserves other slots. Trace IDs and834 programs stay constant. Uniform33 takes accepted bulk fill with top1/top5/top100 agreement100%; numerical logits differ slightly from solo due accepted batch arithmetic (minimumPCC0.999833). Diagnostic preparation covers1/33/128/512 for this functional probe, not full B3 warmup acceptance.

### 2026-11-14 03:18 UTC — valid terminal profiler report
The reduced profile rerun exited0 and produced the ops CSV under `tracy_terminal_fixed/reports/2026_11_14_03_14_06/`. Setup drains preserved all markers; analysis with tt-perf-report is underway. The all64 TT shared qualitative suite is running. The pending maximum-context test now calls public sampled `prefill_forward` directly before its final legal resident decode. Production source remains unchanged.

### 2026-11-14 03:23 UTC — shared qualitative suite complete
Read all12 HF and12 TT completions for the six canonical prompts, greedy and sampled, seed0,128 tokens. All remain coherent and on topic; no sustained collapse or wrong-language drift observed. Greedy first differences are11,74,96,29,97,55. Both sides mostly remain in default thinking at the cap; this is not an EOS or final-answer-completeness gate. `qualitative/shared_suite_verdict.json` records exact scope and limits. The serialized `acceptance_queue.py` is now running autoregressive CLI, resumed-prefill, all64 B32 preparation/pool, then uninstrumented128/128 performance; it stops on any failure and supports an inter-job stop file.

### 2026-11-14 03:28 UTC — autoregressive readiness128-token exact match
`run_autoregressive` exited0 with an exact-match HF cache hit. Read both output files: all128 token IDs match, coherent English explanation and fruit analogy, no early divergence or sustained repetition. `qualitative/autoregressive_verdict.json` records prompt-format caveat and scope. The queue paused safely between jobs via `acceptance.stop`; the reduced same-precision LM-head comparison is now running before the remaining gates. No production source change.

### 2026-11-14 03:30 UTC — LM-head geometry control
Whole-head two-reader block1 BF16/HiFi4 DRAM-sharded candidate failed compile admission:1,628,096-byte static CB exceeds1,572,864-byte L1. No hang/reset; mesh closed. Adapted two-split/two-reader/block2 candidate passed allfour real-weight activation checks and common greedy sampling under strict allocation tracking. Same-regime terminal+sampling/readback median2.34860ms versus current2.20238ms (6.64% slower); head+staging1.81070 versus1.66497ms. Artifacts under `perf_terminal/lm_head_candidates_b1` and `lm_head_split2_b1`. Production head unchanged.

### 2026-11-14 03:31 UTC — retain measured faster terminal
Whole-head three-reader block1 adapted geometry also passed strict tracking, real-weight numerical and sampler checks, but terminal+sample/readback2.68125ms is slower than current2.20328ms; head+staging2.14538 versus1.66532ms. Four readers are explicitly unsupported by native API. Thus both adapted same-precision DRAM-sharded variants are measured slower for primaryB1; current terminal remains selected with no production changes. Resumed `acceptance_queue.py --start-at prefill_resume_final` after removing its stop file.

### 2026-11-14 03:36 UTC — resumed-prefill readiness passes
`run_prefill_resume_check` exits0: natural2560-token prompt, starts32/68/2528/2559, allfour cases pass. MeanPCC minima0.999432, individual-positionPCC minimum0.985755 (defaultbar0.98), minimumtop1 agreement0.992218 (bar0.98), all final-positiontop1 equal. Both no_history and fresh_suffix controls fail at both eligible257-position windows. Artifact `readiness/prefill_resume_check.json`; exact command/log in manifest. Prefix caching remainsFalse under recurrent_state, independently of this successful existing-state resume test. The queue has started all64 B32 full preparation with8192-column virtual tables and8224 physical pages.

### 2026-11-14 — primary performance construction
Before the queued performance run starts, adjusted its test-only allocation to advertised262144 context instead of diagnostic2048. This includes full-capacity reset and SDPA window costs in published128/128 primary figures. Reduced profiler mode remains2048 and is labeled accordingly. Production source and prior accuracy/trace results are unchanged. Pinned Black hook passes.

### Pending final terminal timing confirmation
The adapted LM-head comparisons used strict allocation tracking for correctness, which is explicitly labeled. Before finalizing their performance rejection, rerun the two successful geometries together with tracking disabled; no repeat of the known L1-admission failure. Set the queue stop file so this short reduced timing confirmation runs safely after B32 closes and before primary all64 performance.

### B32 full preparation allocation failure and AutoFix

`slots32_all64_pool` exited 1 during uniform preparation at inferred length 161 (first after printed 160). The failed BF16 logits allocation was 762,839,040 bytes/device. Free DRAM was 1,429,402,112 bytes/device, but the largest hole only 51,515,008 bytes/bank: immediate fragmentation, not a proven physical context limit. Mesh closed normally; no reset. Fresh AutoDebug and two independent source hypotheses investigate ownership. Reduced real layers [0,1,3] with the same B32 virtual table/physical pool now measure native buffers, CCL owners, program count, and diagnostic pre-capture cache-clear behavior. Early B32 32→64 measurements show large tensor-sized retention, not predominantly kernel-binary bytes. All capability claims remain pending this repair; no context reduction.

### Native mapping metadata root cause and repair

The reduced [0,1,3] B32 measurement completed lengths1..192: prepared DRAM17,247,420,928B/device fell to2,503,948,288B after a synchronized diagnostic-only program-cache clear. Source and two independent CPU investigations identified native tiled reshape mapping tensors retained in `ProgramArtifacts.op_owned_tensors`. Six size/multiplicity bins match exactly and explain896.5MB of the1.34GB length32→64 growth. See `prepare_memory_baseline_summary.json`, `reshape_mapping_reconciliation.md`, and `reshape_mapping/independent_review.md`; full raw censuses stay local.

Applied only the native same-width mapping compression in the tiled reshape factory and reader: cache one tile-column map per output tile row, add tile-column index to source page IDs. Other widths remain on the general mapping. Accepted decoder and full-model Python are unchanged. Canonical ttnn build and both installation components passed. Focused36-case all-rank test passes exact BF16/FP32/UINT32 data, fresh input addresses, cache hits, traced replay and width fallbacks. Initial test-only memory assertion retained its last rank alias; correcting that ownership produced155,648B map+program footprint for a formerly22,282,240B mapping. No reset.

Full1..512 reduced preparation is running. `native_acceptance_queue.py` records the serialized affected acceptance suite, uninstrumented LM-head comparison, final full-model timing, and the single max-context acceptance; it respects `acceptance.stop` between jobs. Native source identities are now included by `tests/full_model_evidence.py`. Format-only native factory edits followed the canonical build and will be rebuilt before full acceptance.

The single maximum-context acceptance is configured with8193 physical pages at B1: the pinned plugin adds one output-reservation block to Q262144. Page0 is left unused, active full-history mapping is1..8192, and reset clears the pool before the public prefill. The report separates actual pool bytes from logical-context KV bytes. This closes the source-only addendum's B1 extra-page uncertainty without another maximum-context run. Test has not yet launched.

Reduced [0,1,3] complete B1 and B32 preparation1..512 passes after native compression:59,324 cached programs,5,169,064,448B/device prepared,8,639,015,424B with S512 logits alive,2,503,948,288B after diagnostic pre-capture cache clear. Full native buffer census remains local; `prepare_memory_compressed_summary.json` is the compact committed receipt. Program-count matching at shared length checkpoints isolates metadata-size reduction, without evicting programs in production. The all64 queue will use the final formatted/rebuilt native source.

Uninstrumented serialized confirmation (`lm_head_uninstrumented.json`, profiler/tracker off, five repeats of100 replays) retains the current head: split2_r2_k2 2.3492ms, whole_r3_k1 2.6892ms, current 2.2032ms for complete head+sampling+token readback. Both alternatives remain slower; all numerical/greedy checks pass and program/trace identities stay fixed.

`trace_contract_native.json` passes after final native rebuild: complete1..512 preparation,13 public request/resume/sampling/page-map cases,23,705 program entries unchanged, all three trace IDs retained, token feedback and signed/unsigned position coherence, strict allocation tracking including program-cache allocations. Current-source receipt replaces `trace_contract_final.json` in active serving/audit docs; earlier evidence is retained.

The first all64 B32 retry completed both full1..512 preparation passes, all three captures, public uniform33 prefill equality, inactive-state/token/RNG preservation, remap, release and final replay assertions. It then failed only while constructing its result dictionary: `ccl_inventory` omitted `ttnn.uint16`, used by the common sampler's CCL index buffer. `slots32_all64_pool_fixed_reporting_error.log` preserves the run; no final memory JSON was written, so this is not claimed as a complete acceptance receipt. Added the native2048-byte UINT16 tile to the test-only inventory table. A reduced B32 focused report test precedes the full retry. Production/native sources are unchanged; no reset.

`slots32_inventory_fixed.json` passes on reduced layers [0,3], B32/C262144/P8224, focused lengths1/33, strict tracking. UINT16 inventory now succeeds. The full-slot reporter also writes completed assertions plus measured memory before optional inventory enrichment, and sets `passed=true` only after the complete report succeeds, preventing a metadata exception from losing useful evidence. Full acceptance restarts at `slots32_all64_pool_fixed`; earlier head/trace passes are retained.

### Acceptance report preflight and numerical gate enforcement

The source-only AutoFix helper found two slot-probe CLI defects outside the active B32 invocation: unsupported batch1 and a new output directory failing only after device assertions. The parent restricted this multi-slot probe to2..32 and creates the report directory before device setup. `pre-commit run black --files models/autoports/qwen_qwen3_6_27b/tests/probe_full_slots.py` passes (`hooks_slots_cli.log`). These changes do not alter the already-running B32 workload or production generator.

The pinned readiness prefill/teacher-forcing commands return0 even when their printed numerical accuracy is below the user gate. `collect_accuracy.py` therefore reads the exact aggregate numerator/denominator counts, requires100 targets in all three top-k columns, verifies all seven current model/native source hashes against the native trace receipt, and separately enforces top5>=0.98/top100==1.0. `pre-commit run --files models/autoports/qwen_qwen3_6_27b/doc/full_model/collect_accuracy.py` passes (`hooks_accuracy_collector.log`). The collector will run only after both current logs complete; no accuracy pass is inferred from subprocess exit0 alone.

### Repaired all64 batch32 capacity acceptance — PASS

`slots32_all64_pool_fixed.json` now completes with `passed=true` after the reporting helper repair. The exact command and strict tracking environment are in `commands.json`; no diagnostic preparation override was used. Both B1 and B32 prepare every1..512 shape, then retain decode/sample/reset traces. Uniform33-token logits are finite and bit-equal across32 slots; inactive GDN/token/RoPE/RNG preservation, remap, release and final replay assertions pass. Three captures and five decode/sample replays are recorded. The one full-logit read is the explicit prefill correctness comparison, not token-out decode.

At virtual context262144 and physical pool8224 pages/layer/rank, the post-assertion resident snapshot is23,217,463,808 bytes/device; free10,861,259,264; largest contiguous free block1,184,727,168 bytes/bank. Owned persistent CCL inventory is2,206,482,432 bytes/device. K/V payload is2,290,614,272 and shared reset-zero71,581,696 bytes/device. These are separately scoped, overlapping components of the measured total, not additional memory to add to it. Complete preparation demonstrates transient feasibility; the end snapshot is not a measured high-water mark. The accepted per-request context remains262144 and still awaits its one B1 maximum-context runtime acceptance. No reset, cache eviction, policy override, or context reduction was used.

### Current-native all64 mixed lifecycle — PASS

`mixed_prefill_native_all64.json` reruns the all64 external-cache mixed workload after the final native rebuild. Ragged33/128/513 and row reorder/repeat are exact solo parity; release/rebuild prefill and decode are exact with other GDN slots preserved. Uniform33 bulk retains top1/top5/top100100% agreement, minimum PCC0.9998330967452547, maximum absolute error0.25. All834 prepared programs and three traces remain unchanged under strict tracking. Physical64 pages and virtual96 entries per row are explicitly a focused external-cache test, not the full-context capacity claim.

### Current-native AIME24 numerical gates — PASS

The serialized readiness reruns completed after the final native rebuild. `prefill_native_all64.log`: top1/top5/top100100/100/100%; `teacher_forcing_native_all64.log`:99/100/100%, decode31.05t/s/u. `python models/autoports/qwen_qwen3_6_27b/doc/full_model/collect_accuracy.py` independently enforces exact100-target counts, top5>=98% and top100=100%, and matches all seven current source hashes; receipt `accuracy_native.json`, log `accuracy_native_gate.log`. Both numerical gates pass. The readiness TF TTFT92,787.35ms includes complete preparation and is not the warmed primary TTFT. These current results supersede the old-native receipts without deleting them.

### Current-native resumed prefill — PASS

`run_prefill_resume_check` passes all four starts32/68/2528/2559 for the natural2560-token prompt. Both no-history and fresh-suffix controls fail at each eligible257-position window (starts32/68); short tails have no eligible control window. Minimum case-mean PCC0.9994319661055144, minimum individual PCC0.9857546325547079 against runner threshold0.98, minimum top1 agreement0.9922178983688354, and every final token agrees. Raw artifact `readiness/prefill_resume_check.json`, current log `prefill_resume_native_all64.log`, compact source-linked receipt `prefill_resume_native_receipt.json`. Values reproduce the earlier pre-native-repair receipt. This validates existing recurrent/KV state continuation; automatic KV-only prefix caching remains excluded for `recurrent_state`.

### Current-native autoregressive review — PASS

The current `run_autoregressive` completes128 tokens per side. Both HF and TT completions were read in full: identical coherent English explanation of supervised/unsupervised learning, followed by the same fruit analogy; no sustained repetition, unintended language drift or early divergence. The128-token cap truncates the answer mid-sentence, equally on both sides. All128 token IDs match each other and the TT output archived before the native repair. The leading closing-think marker appears identically in HF and TT. Updated `qualitative/autoregressive_verdict.json` links current source hashes and `autoregressive_native_all64.log`.

`check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_6_27b --missing-artifacts critical --scope autoregressive` passes (`degenerate_native.log`): adjacent duplication0.0127, trigram loop fraction0.075. This observed output does not trigger a qualitative-failure investigation or a new statistical claim. Existing12-per-side shared-suite review and its prompt/template scope remain in `qualitative/shared_suite_verdict.json`.

### Warmed full-model primary performance — PASS

`performance_final.json` completes all64, batch1, prompt128/generate128, context allocation262144, three uninstrumented repetitions. Median TTFT86.924313ms, token-out30.9675576t/s/u (32.2918589ms/token); individual rates30.96756/30.96961/30.96485. All three token sequences agree. Each measured request performs127 decode replays,128 sample replays/readbacks, one initial position/RoPE setup and zero token-feedback/page-table/host-sampling/teacher-forcing uploads. Three trace IDs,24456 programs and persistent input/output addresses remain unchanged. Preparation93.05582s and model load are outside warmed TTFT; request reset, prefill and first sample/read are inside it.

The separately labeled fixed-input, advancing-position logits-only trace gives31.50349t/s/u (31.74251ms/step); it is neither autonomous generation nor teacher forcing. The observed token-out/logits-only gap is0.54935ms/step, consistent with the focused sampler/readback profile. AIME24 teacher-forcing uses161-token input/100 targets, context2048, and includes traced sampling/readback plus reference input uploads:31.05t/s/u, not directly the same workload.

Compared with the preserved best observed optimized layer stack31.0554549ms, full token-out32.2918589ms is1.03981x. The1.23640ms difference is not a component decomposition: layer data uses prefix512/repeated per-kind weights, while this all64 run uses prompt128 and advancing autonomous tokens. The independent terminal profile and adapted LM-head candidates remain the evidence for terminal choices. No sampler dominates full-model latency, no full-vocabulary all-gather/host argmax occurs, and no precision/layout strategy was relaxed. `performance_summary.json` and both READMEs report the exact scope. Maximum-context acceptance is now the queue's sole remaining hardware gate.

### Maximum-context acceptance and final native Watcher — PASS

The one maximum-context acceptance completes on all64 distinct real layers with strict allocation tracking including programs. `max_context_final.json`: physical8193 pages (page0 reserved), public nonaligned prefill262143, then resident decode at262143, next position262144; valid output IDs13205/7189. All24456 programs and three traces remain unchanged. The final decode has one model/sample replay and token read, with zero token/position/RoPE/page-table refreshes and no full-logit readback. Prepared/after-prefill/after-decode resident memory is identically19,072,073,728 bytes/device, free15,006,649,344; largest free block1,870,288,192 bytes/bank. Loaded baseline13,857,710,080 is reproduced. Tracked long-prefill313.5994s and final-decode40.1303ms are acceptance timings, not primary performance. No context reduction, hard-limit search, reset or repeated maximum-context attempt occurred.

After the queue exited and closed its mesh, the focused native probe ran with `TT_METAL_WATCHER=1 TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=0`, timeout900s. All36 exact reshape/cache/trace cases pass. `reshape_mapping_watcher_receipt.json` records source identities and the raw capture digest;1,630,537-byte `reshape_mapping_watcher_capture.log` remains local/ignored. No error/assertion pattern was reported. All hardware work is finished before review.

`context_contract.json:full_model` is validated with recomputed physical KV/reset coefficients, loaded/setup/prepared/runtime memory, CCL inventories and exact B1/B32 operating scopes. `capacity_accounting.md` distinguishes sampled residence from peaks and makes no maximum-pool extrapolation. `serving_contract.json` now maps79 pinned rows with zero missing items, including Q262144 before headroom for physical B1/P8193 and B32/P8224. Fixed B32 supports1..32 live requests; other constructor batches are not assigned unmeasured budget certification. Null reservation, page rounding, exclusive pool ownership and scheduler admission remain explicit. No vLLM implementation or server was started.

Final source hooks pass (`hooks_final.log`, exact argv `hooks_final_command.json`) with no implementation changes. Stage gate and fresh independent review follow completed hardware evidence.

### Pre-review gate and self-check — PASS

`06-full-model.check.sh` exits0 (`stage_gate.log`): no degenerate output, target/supported context262144, and79 serving rows at the pinned plugin SHA. All current-native numerical, mixed, B32, maximum-context and performance artifacts carry the same seven current model/native source hashes. All completed command records are terminal; no hardware job remains. Untracked archive inventory contains no file above500KB (large raw captures/censuses are preserved locally and ignored). Working/staged `git diff --check` passes. Fresh independent stage-review round1 is the next gate; local commits remain pending its verdict.

### Independent stage review round 1 — CLEAN-PASS

Fresh independent inspection returned `clean-pass`, with no required fixes (`stage_review_round1.md`). The reviewer inspected the complete implementation, all actual qualitative outputs, seven current source hashes, numerical counts, trace ownership/feedback, current B1/B32/context evidence, and timing arithmetic. The only optional housekeeping note was the missing reference-generation manifest row; `commands.json` now includes the pinned command reconstructed from the original work-log command and snapshot substitution, explicitly labeled as such. No reference regeneration or additional hardware run was needed.

Capacity, qualitative-window, tie identity and future serving-integration limits remain explicit in the report and contracts. All device work is stopped. Only stage-owned source, tests, compact evidence and documentation are included in the local checkpoint; large raw captures remain local under the documented ignore rules. No vLLM work or push was performed. The implementation SHA will be recorded in a subsequent documentation receipt commit.

Commit preflight staged236 stage-owned files. Source, Markdown, JSON and ignore-rule whitespace checks pass. Once newly tracked evidence was staged, the unrestricted whitespace check also reported original CSV CRLF line endings and whitespace in generated console/log artifacts; these raw evidence bytes are intentionally preserved. The earlier pre-review diff check covered the then-tracked changes and did not inspect untracked artifacts. Production sources remain byte-identical to the reviewed acceptance hashes.

The first commit attempt ran all repository hooks: source/format/API/structural checks passed, while trailing-whitespace and end-of-file-fixer rewrote only captured evidence files. Those14 evidence files were restored byte-for-byte from the staged originals. The implementation checkpoint skips only those two formatting hooks (`SKIP=trailing-whitespace,end-of-file-fixer`) to preserve raw logs, profiler text and HF/TT completion artifacts. Source hooks, including both whitespace checks, already pass in `hooks_final.log`; all other commit hooks run normally.

### Local checkpoint and goal closure

Implementation and reviewed evidence committed locally as `9ed6549ad7332f424750e3e2890a18ac379836c5` (`Add Qwen3.6-27B TP4 full model and traced generator`), based on `7be8c4348666622686a6bdf5efa72e73ecf4c9b0`. All seven production source hashes still match the final acceptance receipt. Independent round1 verdict is `clean-pass`; all required gates are complete. `commit_receipt.json` records the checkpoint, review and scope. This SHA entry is committed in a subsequent documentation-only receipt commit. No push or vLLM implementation was performed.
