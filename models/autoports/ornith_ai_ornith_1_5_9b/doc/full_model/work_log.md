# Full model work log

Stage 6, complete; independent round-1 clean-pass. Target `ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Started from clean main
`c2254f91` following stage 5 implementation/evidence `1b91dfed` and its
round-2 clean review. No vLLM adapter, installation, server, push or reset is
part of this stage. Four available P300c Blackhole devices, physical IDs 0–3,
mesh 1x4, serialized hardware work. No foreign device owners found.

Environment: source `/home/ttuser/ornith-9b/lane-env.sh`; active interpreter
`/home/ttuser/ornith-9b/tt-metal/python_env/bin/python`. Installed/enabled
skills: tt-model-bringup 0.1.23, tt-autodebug 0.1.8. Package environment setup
is applied by lane-env.sh. Installed skill packages remain read-only.

## Initial contract retained

The initial wrapper used the unchanged production decoder. Its class defaults,
shared `MultichipCollectives`, pooled recurrent state and packed decode entry
are used directly. Source SHA256 `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`.
All prior precision/geometry/collective rejection evidence remains in
`../optimized_multichip_decoder/`; no previously rejected candidate is selected.
Residuals stay locally width-sharded through every decode layer. Context
262144 remains advertised; no reduction or all-layer capacity claim yet.

## 2026-11-13: setup and reduced wrapper

- `timeout 60 tt-smi -ls --local`: four expected chips available.
- Bounded 1x4 mesh open/add/read/close: `logs/mesh_smoke.log`.
- Separate read-only serving checkout pinned at
  `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`,
  `042a94863ad27d88207bcf71abdf67122c6f4c27`; recommended vLLM 0.31.0 from
  `docs/install-vllm-tt.sh`. No serving packages installed.
- `timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_probe --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/reduced_smoke.json`:
  one real linear layer (0), one real full-attention layer (3), real sharded
  embedding/head and offset terminal RMSNorm. 33-token prefill and one decode
  pass; finite outputs. Cache is allocated for 262144 tokens. This is a wrapper
  smoke, not all-layer accuracy or context execution evidence.
- `OMP_NUM_THREADS=8 timeout 1800 python -m readiness_check.generate --hf-model ornith-ai/Ornith-1.5-9B --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 --model-dir models/autoports/ornith_ai_ornith_1_5_9b --output models/autoports/ornith_ai_ornith_1_5_9b/readiness_aime24_chat.refpt`:
  succeeded; 161-token rendered prompt, 100-token continuation, exact-match
  reference cache retained under `readiness/reference_cache/`.
- Split sampling implementation is under test. `trace_smoke.log` and
  `trace_smoke2.log` exposed scalar sampling parameter shape mismatch before
  trace capture; corrected caller to use common `format_sampling_params`.
  No decoder or precision change. Clean device closure in both failures.
- Initial wrapper/probe Ruff check exposed import ordering and dictionary style;
  fixed. Repository hooks recorded in `logs/hooks.log`; no failures.

## Initial acceptance checklist (historical)

All-layer capacity and top-k accuracy; independent HF/TT qualitative controls;
serving interface matrix; mixed/resumed prefill; external cache and fixed slots;
complete generator-wide warmup; cross-request trace reuse and allocation positive
control; sampling modes/seeds; reduced profiler and terminal optimization;
128/128 batch-1 performance; runtime fallback audit; independent stage review;
local stage commits. No stage completion claim.

## Initial all-layer evidence and outstanding defects

- `run_prefill_check` on the AIME24 reference: top1 99/100, top5 100/100,
  top100 100/100. `logs/prefill_check.log`.
- `run_teacher_forcing` with trace allocation tracking and program buffers
  included: top1 98/100, top5/top100 100/100. `logs/teacher_forcing.log`.
  Its diagnostic timing is not the primary performance result.
- Exact initial source and hashes: `initial_source/` and
  `initial_accuracy_provenance.json`. Neither result closes serving/state gates.
- `full_model_trace_probe --cross-request`: same33-token shape reuses both
  traces;34tokens causes20 unsafe program-cache allocations. No replay was
  forced past the tracker. `logs/cross_request.log`. AutoDebug launcher namespace
  preflight failed; explicit skip of the additional child sandbox was used
  under the session's unrestricted execution authorization. Fresh inspection
  process is running; `logs/autodebug_retry.log`.
- `run_prefill_resume_check --lengths513 --window128`: FAILED at starts32
  and68 (minimum logit PCC0.8744/0.7978); start512passes. Both negative controls
  fail as expected. `readiness/prefill_resume_check.json`. This is required work,
  not a numerical-drift waiver or a prefix-cache support claim.
- `run_autoregressive --max-new-tokens128`, normal HF tokenizer chat template,
  sky-color prompt: HF and TT outputs read directly. Both produce coherent
  English reasoning about Rayleigh scattering and reach128tokens while drafting
  the answer. No doubled-token or single-token collapse is visible. Their
  wording diverges early; this is one smoke, not the complete shared suite or
  a matched sampled quality verdict. Raw files: `readiness_autoregressive/`.

## AutoFix experiments: resume numerics and live-trace prefill

All commands source `/home/ttuser/ornith-9b/lane-env.sh`, use
`OMP_NUM_THREADS=8`, and run serially on chips 0–3 under bounded timeouts.
Production decoder source remains unchanged. Diagnostic candidates are not
selected by the current generator.

- Fresh AutoDebug completed: `autofix/AUTODEBUG.md`. Its principal finding is
  late program-cache allocation from unprepared logical shapes and absolute
  offsets. Inspection subsequently corrected one proposed detail: ordinary RM
  slice hashes absolute offsets; runtime-index embedding lookup over residual
  storage is used instead. No blanket trace exemptions or trace retirement.
- BF16 norm/CCL diagnostic: reduced resume passes, all-layer resume still fails
  (minimum PCC 0.919710/0.854162 at starts 32/68). Artifacts
  `reduced_resume_bf16.json`, `full_resume_bf16.json`. Rejected as a selected
  repair because it changes the required policy and does not pass the stack.
- Fixed32 diagnostic: reduced resume passes (`reduced_resume_fixed32.json`).
  Rejected as a selected repair because it replaces optimized chunk strategy.
- `timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_resume_policy_probe --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/resume_policy_probe.json`:
  identical-input GDN outputs and final FP32 state match bit-for-bit at aligned
  starts 32/512. Start68 changes recurrence grouping (output PCC 0.99999983,
  state PCC 0.99999864). Most packed-input projections match exactly; short MLP,
  normalization and collective policies differ. First attempt's synthetic
  single-row reduction lacked native sharding; the probe now restores that
  layout. Logs `resume_policy_probe.log` and `resume_policy_probe2.log`.
- Canonical BF8 norm/CCL diagnostic improves reduced logit PCC but still fails
  top1 (0.9767/0.9690); `reduced_resume_canonical_bf8_storage.json`.
  `reduced_resume_canonical_bf8.json` is an invalid activation control: readiness
  loaded a fresh generator class and bypassed the monkeypatch. The subsequent
  active run exposed native decode's BF16 cache-update input contract; the final
  diagnostic preserves BF16 storage after value quantization. No runtime kernel
  input validation was weakened.
- Normalized layer probes pass: linear lengths 1,2,33,34,65,129,481,511,512,513
  (`normalized_linear_smoke.json`, `normalized_linear_boundaries.json`); full
  attention 12 private/pooled cases with lengths 33,34,65 and starts 0/32
  (`normalized_attention_smoke.json`). Real-token outputs and subsequent decode
  match exactly in these tests; small FP32 recurrent-state differences remain
  for some padded linear cases. Untouched attention pages remain byte-identical.
- Request-wide physical profiles fix aligned resume through all32 layers:
  `full_resume_profiles.json`. Adding device-only boundary state/input tails and
  at most31 replayed overlap tokens fixes all tested starts:
  `full_resume_overlap.json`, top1=1.0 for starts32/68/512, both negative controls
  fail as expected. Exact command:
  `timeout 1200 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_resume_profiles --aligned-overlap --model-dir models/autoports/ornith_ai_ornith_1_5_9b --lengths 513 --window 128 --mesh-device P300x2 --fabric-config FABRIC_1D --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/full_resume_overlap.json`.
  This is a diagnostic wrapper limited to1024, with eager allocations; it is not
  the final public-generator gate. Runtime integration remains work.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_prefill_staging --trace --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/prefill_staging.json`:
  all15 reduced cases pass at lengths33/34/65/513, starts0/32/68, with live split
  traces. Program count stays398; no misses allowed. Context2048; this is shape
  preparation evidence, not full-context correctness or performance.
- `full_model_state_probe` with both trace-tracker flags passes external-cache
  warmup restoration, device token feedback, position increments, and changed
  versus unchanged page tables. `state_probe.json` uses layers0/3, context256.
  Generator warmup now snapshots/restores the actual passed cache. No-history
  controls now clear full KV as well as recurrence; old negative controls did
  not clear full KV. Zero KV buffers preserve independent K/V dtypes.
- `full_model_sampler_probe`: common force-argmax needed two shared-CCL accessor
  methods; first failure preserved in `logs/sampler_probe.log`. After adding
  those methods, tracked traces pass for both greedy strategies. Initial
  sampling-only timing: top-k0.482ms, force-argmax2.537ms (`sampler_probe2.json`).
  Further nontrivial-logit and untracked timing evidence is being collected.

No final performance, serving completeness, review, or stage-pass claim.

## Integrated runtime and serving repairs (2026-11-13)

- The public runtime now retains canonical whole-request precision profiles,
  device input tails, and FP32 recurrent snapshots at the previous tile boundary.
  Unaligned resumes recompute at most the required overlap under that profile.
  Implicit chunk horizons rebuild the known prefix only when the selected
  profile/history cannot be reused. This preserves the existing chunk caps and
  policy rather than selecting the rejected fixed-small-chunk or BF16 controls.
- Full-context warmup initially failed with an L1 clash in MLP down projection.
  Normalized prefill adapters retained a no-longer-used norm tensor, unlike the
  original decoder. Releasing it before `finish` fixed the reduced full-context
  warmup. Artifacts: `logs/full_context_runtime_trace.log` (failure),
  `logs/reduced_context_runtime_trace.log` (pass).
- `run_prefill_check --model-dir $MODEL_DIR --reference
  $MODEL_DIR/readiness_aime24_chat.refpt --mesh-device P300x2
  --fabric-config FABRIC_1D`: top1 99/100, top5/top100 100/100.
  `logs/final_prefill_check.log`. Uses the complete layer stack and context262144.
- Same flags with `run_teacher_forcing`: top1 98/100, top5/top100 100/100.
  `logs/final_teacher_forcing.log`. Diagnostic 100.23 t/s/u includes token-out
  sampling plus teacher-forced input upload; this is not the primary128/128 result.
- `run_prefill_resume_check --model-dir $MODEL_DIR --lengths 513 --window 128
  --mesh-device P300x2 --fabric-config FABRIC_1D`: PASS all three starts and both
  negative controls. Exact accepted output: `../../readiness/prefill_resume_check.json`.
- Production decoder additions are limited to a masked-only safe RoPE lookup
  and shared global-page-pool addressing/allocation. Original local all-active
  math, precision, collectives and residual layout are retained. Inactive
  recurrent rows are restored by preallocated wrapper scratch; native attention
  kernels retain their signed -1 inactive sentinel.
- `full_model_batch_state_probe --batch 3` and `--batch 4`: pass native-active
  equivalence, exact inactive preservation, reactivation oracle, pool remap,
  peer-preserving reset and traced position handling. Artifacts:
  `batch_state_probe_b3.json`, `batch_state_probe_b4.json` and matching logs.
- `full_model_sampling_state_probe`: pass scoped history reset/snapshot, full32
  permutation and inactive RNG/count preservation. First run exposed only an
  unsupported torch uint32 scalar addition in the test assertion; explicit
  integer comparison fixes the test. `sampling_state_probe.json` and
  `logs/sampling_state_probe2.log`.
- `full_model_serving_prefill_probe`: pass B3 mixed lengths/starts/order, solo
  equality, persistent device-logit assembly, inactive peers, transactional
  invalid-row admission, explicit reset and implicit33→68→513 horizon changes.
  Program count stays614 with live traces. `serving_prefill_probe.json`.
- `full_model_serving_decode_probe`: initial pass for B3 fixed slots, resident
  feedback, owning async outputs, stale-host rejection, cache/RNG invariants and
  host-logit output. The subsequent source audit identified authoritative forced
  reload and preflight validation gaps; these require the updated probe before
  this evidence closes the serving matrix.
- `full_model_capacity_probe` reduced layers0/3: public length262143 prefill
  passes after trace creation with cache misses forbidden and finite last logits.
  `capacity_reduced.json`; measured8.044s was concurrent with a CPU HF control,
  so it is a capacity result, not ranked performance.

All device jobs above were bounded by timeout and serialized across chips0–3,
with TT_METAL_TRACE_ALLOC_TRACKING=1 and program buffers included. No reset,
vLLM server, vLLM adapter or push occurred. Full-capacity, full-batch, external
pool, final qualitative/performance/audit/review evidence is still in progress.

## Serving boundary follow-up (2026-11-13)

- Full32 B32/context512 passes mixed prefill, selected solo equivalence, reordered rows, native packed decode, inactive preservation and device feedback;597 cached programs remain constant. `batch32_full.json`, `logs/batch32_full.log`.
- New serving decode probe first failed only arbitrary inactive sampled outputs; active outputs matched. Return immutable-submission inactive sentinels (-1 token,0 logit) and retain all fixed rows. `serving_decode_probe_final2.json` passes async two-pending ownership, observational reads, writable logits, invalid transition atomicity, authoritative page reload commands, unchanged-page zero uploads and resident full-pool remap. Diagnostic failures retained.
- Positive trace allocation control rejected one deliberately live late buffer and succeeded after releasing it. Exact command: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout120 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_trace_positive_control`; `trace_alloc_tracking/positive_control.log`.
- Sampled mixed prefill exposed a physical32 common-formatter requirement (`logs/sampling_transition.log`); repair and targeted rerun are required before acceptance.
- HF shared-suite seed0 run is screening; it is not the minimum10 matched-seed failure-control experiment. Any quality failure must get that control before diagnosis.

- Watcher10, separate from profiler, passed reduced cross-request33/34/31/65 logical prompts:35model+35sampling replays,6captures,0token/page refreshes,5initialposition writes; `watcher_trace.json`, `logs/watcher_trace.log`, copied`logs/watcher_device.log` with no watcher violations. Instrumented compilation dominates the310s diagnostic, not a performance number.
- Full32 ownership probe passed caller-injected cache with zero duplicate model allocations; actual external default-release routing, inactive peer/standalone state preservation, invalid binding atomicity, repeat release, pending-output close/drain/read, idempotent teardown and backing storage retention. `cache_ownership_full.json`, `logs/cache_ownership_full.log`. Configuration-specific768total tokens atB3/context256 over29physicalpages is validated; this is not a maximum-context allocation claim.

- `sampling_transition2.json` passes public sampled-prefill seed advancement, exact seeded-history resume, mode transitions, scoped resets, all three BF16 penalty winners against host arithmetic on real frozen logits, eight unseeded varied draws, and unchanged520programs. Corrected physical32 common formatting and retained raw temperatures to avoid double inversion.
- `prefill_generation.json` passes nine invalid admission cases before mutation, physical32/logical1,3,32 parameters, sampledB3, first/warmed high-level nativeB3 execution with inactive-peer preservation and zero token feedback uploads. Prefill explicitly supports prepared eager execution; enable_trace=True rejects before mutation, while decode_only remains the declared trace mode.
- `runtime_fallback_probe.json` passes seven traced device-only decode steps with host logits/argmax and >32-word reads forbidden, H2D forbidden inside replay, and eight token IDs equal to explicit host compatibility. No default-path host fallback.
- Final all-layer gates run serially via exact command/source manifests in `final_gate_commands.json`; source snapshot `final_source_snapshot.json`. Their accepted-prefixed logs supersede earlier provisional accuracy records.

## Final correctness and capacity acceptance (2026-11-13)

Final commands and exact production hashes are in `final_gate_commands.json`. All five jobs exited 0. Prefill top-1/top-5/top-100: 99%/100%/100%; teacher-forced decode: 98%/100%/100%. Resumed prefill passes starts 32/68/512 for length 513 with both negative controls failing. Full 32-layer maximum-context prefill 262144 and terminal decode position 262143 pass with 820 cached programs unchanged.

The accepted default raw-story autoregressive run produces 128 tokens on each side. Both saved completions were read directly: coherent English narrative continuations, no mechanical repetition, both cut off mid-explanation. Early wording divergence is recorded in `autoregressive_review.json`; exact free-running equality is not claimed. The prior chat-template sky-color completion and six shared chat prompts provide the instruction-format coverage. `check_degenerate_output.py --hf-model ornith-ai/Ornith-1.5-9B --model-dir models/autoports/ornith_ai_ornith_1_5_9b --missing-artifacts critical --scope autoregressive --json .../doc/full_model/degenerate_output.json` passes both retained runs.

## Primary full-model performance (2026-11-13)

`performance_primary.json` passes: all 32 layers, TP4, batch 1, context 2048,
128 prompt tokens / 128 generated tokens, three warmed repetitions. Median
TTFT 92.579 ms, token-out decode 8.9363 ms/token = 111.9037 tokens/s/user.
The same-workload teacher-forcing diagnostic includes sampled predictions and
forced host-token uploads: 8.9529 ms/token = 111.6953 tokens/s/user. A separate
128-replay model-only fixed-device-token diagnostic measures 8.3143 ms/token;
it excludes sampling/readback and must not be called teacher forcing.

Each warmed token-out run has 127 model and 127 sampling replays, zero token or
page-table uploads, one request-initial position write, 128 caller readbacks,
and zero captures/releases. Program count remains 712. Primary runs disable
allocation tracking/profiler/watcher; the tracked correctness gates are separate.
Owned CPU-only HF workers were paused for the complete 33-second primary job,
then resumed in `finally`. Exact command, environment, source hashes and timing
are in `final_performance_commands.json`. Neither concurrent HF runs nor the
reduced device profile supply the ranked host-wall result.

The ten-seed TT prompt04 control is complete. Original-HF controls remain in
progress; CPU scheduling changes are logged in `hf_cpu_scheduling*.json` and
`hf_cpu_affinity.json`, without changing model, dtype, prompts, seeds, sampling
or thread count. No TT-owning process was paused or terminated.

## Reduced profiler and terminal accounting (2026-11-13)

Final reduced profile succeeds with one real linear and one real full-attention
layer, actual shapes, TP4, terminal norm, vocabulary-sharded head and common
traced greedy sampler. `reduced_profile_final.json` and its command manifest
record one model plus one sampling replay with zero host input/position/page
refresh, zero readbacks and no recapture in the signposted window.

Two profiler infrastructure faults were diagnosed without changing production
code. The 16-bit source ID 1831 collided between chunk-GDN `pp_kdec` and `rd_kick`.
The existing byte-identical source-path overlay resolves it only when the profiler
runs from the model directory: the repository working directory otherwise wins
kernel resolution. Then the full warmup catalogue overflowed profiler buffers.
The profile-only test runner now drains between eager preparation profiles and
restores the original method before the measurement. Final collection and CSV
export exit 0 with no dropped-marker warnings. All failed logs/commands remain.
The initial tt-perf-report alias `blackhole/p150` was rejected by the installed
parser; the supported `blackhole` argument produces the final report.

`profile/decode_perf_report.txt` includes advice. `profile/provenance.json`
records raw CSV SHA256, exact report commands and lossless window reproduction:
all 104 operation rows match the full raw report in operation, timings, gaps,
core counts, dtypes and fidelity. Full raw captures remain on persistent storage;
only compact report/window artifacts are committed.

`profile/layer_stack_accounting.json` weights the layer kinds independently:
24 × 0.243417 + 8 × 0.264020 = 7.954 ms instrumented stack estimate. Terminal
norm/head/masking adds 0.784 ms; sampling and feedback add 0.605 ms (6.8% of
full token-out), embedding/entry 0.036 ms and device position 0.001 ms. The
9.381 ms extrapolation differs from measured 8.936 ms because instrumentation,
merged device/gap estimates and position windows differ. It is not mislabeled as
an all-layer timing. The preserved layer stack dominates full-model latency.

The head reports 345 GB/s, 67.4% of modeled bandwidth. A DRAM-sharded geometry
is an unmeasured optimized-full-model candidate, explicitly not a selected or
rejected measurement. The ideal bandwidth-only bound saves at most 0.244 ms
(2.7% whole-model latency); it is not a speedup claim. The common greedy sampler
has no full-vocabulary gather and remains much faster than the semantically
identical forced-argmax candidate measured in `sampler_probe3.json`.

## Final repository hooks and source identity

Repository hooks added missing final newlines to three JSON artifacts without
changing their parsed values. Isort and standalone Ruff disagreed on two aliased
imports in `tt/generator.py`; use qualified calls to the same cache-contract
functions so both formatters agree. `import_format_provenance.json` records the
before/after hashes and an AST comparison: after canonicalizing those exact two
function names and top-level imports, all code is identical. Model/sampler math,
trace state and timing boundaries are unchanged, so measured device results are
not retroactively relabeled with a new execution hash.

The AST-only external-ownership test needed its injected namespace to expose the
module instead of the two old aliases. The first host rerun caught that fixture
mismatch; after repair all 40 tests pass in `logs/final_host_contract_tests2.log`.
This is a host fixture repair, not a device or cache-behavior fix.

Immutable generated text, prompt templates, logs and initial source snapshots
are losslessly archived to prevent formatting hooks from changing their exact
bytes. Raw local copies remain; `archive_manifest.json` records hashes and the
restoration command. Serving evidence paths resolve to tracked archives where
appropriate. The serving-contract checker still passes all 114 rows.

## Readiness extra-keyword compatibility

The installed standard Generator contract asks high-level `generate` to ignore
unknown per-model extras. Its strict rejection was inconsistent with that
convention. Filter only recognized named sampling controls, ignore other extras,
and retain conflicts/validation among recognized controls; low-level serving
admission remains strict. `readiness_extras_change.json` records source hashes.
The prior measured no-extra/recognized-control paths are unchanged.

`readiness_extras_probe.json` passes the actual B3 native-pool generator with an
unknown object-valued extra alongside explicit sampling parameters: output equals
the no-extra run, peer state is preserved, feedback remains device-only and traces
are reused. `delivered_teacher_forcing.log` reruns the full32 AIME gate after the
change: top-1 98%, top-5/top-100 100%. Exact tracked commands and final hashes are
in `readiness_extras_commands.json`. Its 161/100 diagnostic timing under allocation
tracking and concurrent HF work is not the primary quiet 128/128 performance.
The primary benchmark uses configured context2048; maximum-context capacity is a
separate full32 proof and is not a maximum-context performance claim.

## HF control completion and execution regime

All ten original-HF and ten TT prompt04 samples are complete, with matched seeds
0–9, the original tokenizer chat template, default thinking, a 256-token cap,
temperature1/top-p0.95/top-k20/presence1.5. The initial CPU jobs reached their
2400-second timeout after saving four exact completed samples. The retry reused
only completed exact-signature samples and generated the six remaining outputs;
both retry workers exited 0. Passive OpenMP waits and disjoint physical-core
sets changed only CPU scheduling, not model, dtype, thread count, prompts, seeds
or sampling. `qualitative_entropy_hf_retry_commands.json` records the complete
commands and environment. No device job was terminated, paused or reset.

A fresh blind reader receives only the anonymous 20-output file and the rubric
fixed before generation. Labels remain hidden until the scores are written.
The blind output and subsequent numerical unblinding verdict will be linked
below. The six-prompt shared screen remains separate screening evidence.

## Final blind HF control — 2026-11-13

Fresh independent reader `qualitative_entropy_blind` saved all 20 scores before
labels were opened. `qualitative_entropy_control/unblinded_results.json` binds
the score hash, assignments, flags and seeds: HF failures 4/10 (0,1,4,7), TT
1/10 (3); all are scientific overstatements. Difference -30 percentage points
passes the pre-generation >20-point excess rule. Both sides are cap-limited
incomplete 10/10 at 256 tokens. No gibberish, mechanical-loop, wrong-language,
leakage or corruption flags. Shared limitations are recorded with
`original_model: same` in `doc/eval_readiness/limitations.json`.
The initial four HF samples survived exact-cache reuse after the 40-minute
CPU timeout; six were completed with passive OpenMP waits and disjoint CPU
affinities. Prompt, checkpoint, seeds, sampling settings and runner identity
were unchanged. Commands/manifests and retry logs remain in the control folder.
`qualitative_verdict.json` links the six-prompt screen, matched control and
separate HF/TT autoregressive reviews. Independent stage review is next.

## Stage closure — 2026-11-13

Independent fresh xhigh reviewer `full_model_stage_review_round1` returned
`clean-pass` with no Required Work; see `stage_review_round1.md`. It checked
all 50 delivered source hashes, 174 lossless evidence archives, actual generated
outputs, full-stack gates, primary performance, pinned serving interfaces and
trace lifetime/probe assertions. Source stayed unchanged during review.

Final checks: 40 host contract tests passed (`logs/final_host_contract_tests2.log.gz`),
Ruff passed (`logs/final_lint.log.gz`), applicable repository hooks passed
(`logs/final_hooks_review.log.gz`), and serving/context contract gates passed.
Prefill top-1/top-5/top-100 is 99/100/100%; delivered-source decode is
98/100/100%. Maximum context remains 262144. Warmed full-stack B1 128/128,
configured context 2048: TTFT 92.579 ms, traced device-sampled token-out decode
111.904 tokens/s/user (8.936 ms/token); same-workload teacher forcing
111.695 tokens/s/user (8.953 ms/token).

Next-stage notes (not unresolved gates): shared scientific overstatements and
256-token thinking truncation have matched HF controls; maximum-context
capacity does not establish maximum-context HF accuracy; B32/context512 does
not imply B32 maximum-context capacity; the DRAM-sharded LM head remains an
unmeasured optimization candidate. No vLLM adapter/server was started.

Only the tt-metal checkout has stage-owned changes. The inspected plugin and
tt-transformers checkouts are clean. Local checkpoint commits follow this
passing review; never push. Commit identifiers are recorded below.

### Local checkpoint

- Repository: `/home/ttuser/ornith-9b/tt-metal`
- Branch: `main`
- Stage implementation, tests, evidence and clean review commit:
  `6f1d98b4f5288448a99c9c523bcfbdc64dfc2c77`
- Commit hooks passed; worktree was clean immediately after this checkpoint.
- External dependency repositories: no stage-owned changes or commits.
- Push state: never pushed by this stage. This follow-up documentation commit
  records the completed checkpoint; its identifier is returned with the final
  handoff.
