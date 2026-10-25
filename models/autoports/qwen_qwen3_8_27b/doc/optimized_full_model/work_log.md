# Stage07 optimized full model work log

2026-10-25: Started from clean checkpoint 6bcbce2d6e2 (stage06 accepted).
Enabled skills are supplied by the session inventory; environment uses the existing
qwen38-functional-build container and installed model-bringup/AutoDebug packages.
Read model-bringup startup, tracing correctness, multichip/optimization/agentic/device
contracts, full-model README, sampling contract, and stage05 performance evidence.
Hardware list: bounded `docker exec qwen38-functional-build timeout 60 tt-smi -ls --local`
passed; four p300c ASICs, TP4 mesh1x4. Hardware commands serialized.

Baseline: stage06 benchmark_b{1,8,16,32}_perf.json, 4096/128 autoregressive,
B1 TTFT433.812ms,35.186t/s/user; B32 TTFT14760.040ms,8.822t/s/user.
Existing loop has persistent device feedback but reads every output token.
First hypothesis: buffered traced token collection removes those synchronization
boundaries without changing token feedback or numerical policy. Compare exact token
sequences against the existing streaming path, tracker on, then full-model timing.
Second target: terminal BF16 vocab-sharded head currently uses generic linear;
compare explicit DRAM-sharded geometry with the same BF16 policy.
No vLLM or broad datatype search is authorized for this stage.

Buffered-output reduced B1/B32 checks pass exact streamed equivalence at k1 and
k8/p.9, including130 outputs crossing the128-token collection boundary, mixed
33/65/97 prompts and inactive slots. Tracking includes program-cache allocations.
No acknowledgments/scopes were added; all collector buffers are setup-owned.

Head candidate c8192/r1/K5 first attempt failed `bad optional access` before candidate
execution. Inspection found LMHead1D._load_input_device_tensor returns an existing
TT tensor unchanged: input_memcfg only applies to LazyWeight input. Adapted caller
now explicitly reshards real normalized activation to the configured width-sharded
L1 input. This is not a rejection of DRAM-sharded head. Device closed normally;
next full benchmark opened all four devices successfully before this log inspection.

B1 paired full-stack command (diagnostics off):
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh env TT_METAL_TRACE_ALLOC_TRACKING=0 python -m models.autoports.qwen_qwen3_8_27b.tests.benchmark_full_model --buffered-output --compare-buffered --teacher-forcing-control --output models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/buffered_full_b1.json`.
This is an initial candidate measurement; final default will be remeasured after
head selection. During loading the source was refined to move initial output-index
reset before the first-token timing boundary. The already-imported candidate retains
the old reset placement, as its recorded source hashes identify; do not call this
artifact the final-default measurement.

Initial paired all64 B1 benchmark passed exact streamed/buffered outputs, three repetitions.
streamed_baseline_records: median TTFT433.083416ms; decode35.178698t/s/user.
records: median TTFT433.470728ms; decode35.252290t/s/user.
Small collector gain; full-model already near inherited stack bound. Teacher-forcing control is separate in same JSON.

Persistent sampler candidate uses Sampling1D's existing line_all_gather callback.
Source audit caught unconditional deallocation of gathered values in common sampler;
the adapter returns an owned transient clone after async persistent gather so the
CCL output allocation survives. This copy is included in timing. It does not modify
shared sampler behavior or exempt allocations. Candidate has not yet been measured.

After the initial candidate benchmark, serialized bounded reset completed(exit0),
then local listing found all four p300c devices and ring1x4 mesh open/close passed.
Logs: post_head_validation_reset.log, post_reset_list.log, post_reset_smoke.log.
No hung process, missing link, lock removal or operator recovery was involved.

Reference reuse is valid: current autoport readiness_aime24_chat.refpt.json records
exact model/revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,Qwen2Tokenizer,
AIME24/chat=true,100outputs/top100 and generation command. Stage07 reruns the
unmodified readiness comparators against it; no reference regeneration is necessary.

Adapted head c8192/r1/K5 reached the matmul but static dataflow region ended at
1,311,872 bytes versus first live L1 allocation1,187,456, an exact L1 clash.
Preserved head_c8192_r1_k5_s8.log. Revised the geometry matrix to lower chunk×K
products while still testing larger legal K16/20/32 with wider input shards and
smaller output chunks. No dtype/fidelity fallback. Bounded reset runs before retry.

The first four legal DRAM-sharded head candidates are slower than the generic
baseline (~1683us): c8192/r1/K4~2187us; c6144/r1/K4~2226us;
c6144/r2/K10~2149us;c6144/r3/K10/core16~2288us. Same BF16/HiFi4 FP32 policy,
real hidden, each includes input reshard/chunks/slices/concat/output copy.
Continue remaining legal wider-shard candidates before rejecting the family.

Inspected actual baseline profiler attributes: generic head is
MatmulMultiCoreReuseMultiCast1DProgramConfig(grid11x10,K2,sub1x3,N18,M1,
fuse_batch=False,mcast_in0=True), using108 worker cores. Next same-precision
candidate family tunes this existing interleaved-weight topology at K4/5/8/10
(and compatible grid/subblock), rather than accepting its smallK2 by default.

All eight adapted DRAM-sharded head geometries passed the same-precision real
logit check, but none beat the baseline. K16/20/32 and all legal reader counts were
executed, not rejected from the initial L1 error. Exact rows are head_c*_s*.json.
Generic topology sweep launched next with explicit K4/5/8/10 and compatible grids.

Sampler full controls: B1 baseline451.94us,async458.89,persistent455.48;
B32 baseline488.01us,async460.65,persistent471.37. Same semantic k1/p0/temp1,
real logits and vocabulary boundary maxima pass; force-argmax remains much slower.
Intermediate batches will be measured before final per-batch selection. Persistent
adapter includes the required transient ownership copy, no sampler deallocation
contract weakened.

Generic head K4/sub2 ~1680.43us versus1683.31 baseline. HiFi2 micro-control gives
1678.61us generic but changes numerics(PCC.999993); DRAM-sharded HiFi2 remains slower
2121.40us. These are terminal-only experiments, no decoder precision change and no
full-model datatype frontier selection. Preserve existing HiFi4 for stage07 unless
same full-model evidence justifies a precision-policy change; stage08 owns frontier.
GenericK16 requires static region1,467,520bytes versus first L1 allocation1,293,952;
log head_generic_k16.log. Retry larger blocks with two/four interleaved head chunks
before selecting smallK4. Bounded reset follows this failed geometry attempt.

Initial all64 candidate provenance is preserved exactly: sources/buffered_candidate_generator.py
and sources/buffered_candidate_benchmark.py SHA256 match buffered_full_b1.json.
These snapshots reconstruct the measured earlier source, not the later reset-boundary
refinement; the final-default harness will record its own current hashes.

Final defaults selected after terminal controls: packed last-token prefill terminal,
window-buffered device autoregression (teacher-forcing/host compatibility retain
explicit streaming path), async nonpersistent sampler gathers only for B32.
B8 baseline472.27us versus async478.13; B16 baseline476.52 versus async482.80.
B32 packed prefill logits are bitwise equal for lengths33,65 with inactive rows,
and4097; reduced whole-prefill host medians86.46->36.48ms,87.30->37.41ms,
559.47->457.90ms respectively. These are not full-model performance measurements.
Generic split2/K16 and split4/K20 are legal and correct but slower1729.64/1713.17us
versus1683.22/1683.38 baseline. Retain existing auto head configuration: explicit
K4/sub2 improvement~2.88us is negligible relative to whole-model decode and lacks
an all-batch benefit. Preserve BF16/HiFi4. All larger topology candidates have
executed controls rather than being dismissed at the first L1 failure.
Buffered default preserves first-token TTFT read, queues up to128 subsequent tokens
without per-token host synchronization, then reads an output window. EOS is trimmed
at first EOS per row; device work can run to the end of that bounded window.
Long generations reset the window cursor only between windows. Fixed slots stay
allocated; finished rows do not change device feedback or force recapture.

Interim independent stage-review requested tracker-on full-active/mixed/full-active
B32 collector transitions. Added both greedy and top-k/p comparisons to the reduced
probe; final evidence path buffered_final_b32_transition.json.
First all64 validation passed prefill accuracy, then failed the readiness signature
check because the test wrapper hid explicit enable_trace. Removed the redundant
wrapper (the generator now selects buffering itself); failure log retained as
full_validation_signature_failure.log. This is harness failure, not model accuracy.
Reviewer also identified avoidable EOS overscheduling:128-step windows can delay
short replies. EOS-enabled default now collects16-step windows; fixed-length
throughput keeps128. Both use the same preallocated128-row collector and captured
sampling trace. B1 clean benchmark will compare streamed/buffered128/buffered16
on exact factual4096 input with1024 cap, three warmed repeats per arm. No latency
claim for this choice until those paired measurements complete.

Final all64 validation completed after wrapper repair: prefill and teacher-forcing
99/100 top1,100/100 top5/top100. All shared and long output token arrays exactly
match stage06; long EOS-aware windows schedule49 for47 factual tokens and897 for892
nearby prose tokens. Independent reviewer rederived these results and found no new
correctness concern. Degenerate check exits0. Initial EOS metadata concern was
withdrawn: full checkpoint legitimately lists248046 and248044; the reduced probe
checkpoint differs. Final metadata clarifies this source distinction.
Final benchmark suite launched in B1->B32->B8->B16 order via
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/benchmark_optimized_full_model.sh`.
All headline rows use3 warmed4096/128 repeats; B32 additionally warms and measures
one32768/128 target run. B1 adds separate AIME teacher-forcing and3-repeat EOS
latency controls. Allocation-event audits occur after all timed runs.

Final B1 benchmark completed:431.360885ms TTFT,35.252433t/s/u median3 runs;
zero host refreshes and one finalread in127step interval. Separate teacher-forced
AIME token-out control35.524841t/s/u,114.055946ms TTFT at203/100. Allocation auditpass.
EOS paired3repeat factual47token median total times: streamed1739.872636ms,
buffered1284064.639178ms(schedules129),buffered161797.595501ms(schedules49).
Keep16-step EOSdefault:57.722865ms/3.32% latency cost versus streaming, bounded
extra work; reject128 EOSdefault's133.62% regression. Fixed-length headline remains
128-window uninterrupted replay. Exact tokens match across all controls. B32 started.

B32 final3repeat4096/128 medians14715.772549ms TTFT,8.84441985t/s/u.
Agent target32768/128 warmed single measured run144643.64ms TTFT,7.4415t/s/u
(exact values in benchmark_b32_perf.json). Full-stack mixed-slot trace reusepass;
allocation-event reconciliationpass; peak28,440,140,288bytes/device and isolated
trace allocation118,554,624bytes/device. No context/capability reduction. B8 next.
B8 final3repeat medians3564.809091ms TTFT,21.015013t/s/u. Full-stack mixed-slot
trace reuse and allocation audits pass. B16 is the last clean performance batch.

B16 final3repeat medians7302.092257ms TTFT,13.724910t/s/u; mixed-slot trace and
allocator audits pass. summarize_optimized_full_model.py reconciles all batches,
appends stage07 to context_contract.json, and refreshes agentic_serving.json without
reducing native context or the32x32768 target. CPU checks: degeneration0,context0,
agentic1 solely for the six inherited/accounted norm and gate/up representations.

Initial watcher launch omitted the decoder campaign's NOINLINE setting and failed
before model execution: ACTIVE_ETH config29264bytes exceeds26624 by2640bytes.
Failure stdout and watcher log retained. Bounded tt-smi reset/list and ring mesh
smoke succeed on all four devices (post_watcher_size_*). Retry uses
TT_METAL_WATCHER=10,TT_METAL_WATCHER_NOINLINE=1 with per-batch log roots; no Ethernet,
assertion, NOC or CB checks disabled. B1 watcher probe already passes; B32 underway.
Watcher retries pass: B1 four cases, B32 ten cases including full/mixed/full modes
and sampled16-step collector boundaries. No watcher checks disabled; raw logs have
no error/exception/overflow/invalid/corrupt markers; minimum recorded stackfree1216.
Evidence watcher_summary.json and watcher_b*/generated/watcher/watcher.log.

All post-benchmark diagnostics now pass. External preparation verifies injected
failure/capture-end cleanup, preservation of original sampled cache/map/request/RNG,
and replay updates without recapture. Mixed B8 tests include nonzero inactive-state
preservation, absolute-position continuation, remapped pages, odd page-window tails,
slot permutations and all-inactive decoding. Both reduced profiles completed via
validate_optimized_full_model.sh; tables, filtered CSVs and scope/runtime provenance
are under tracy/b1 and tracy/b32. CPU `audit_optimized_full_model_profile.py` exits0,
verifying exact inherited opcode-prefix layer boundaries, BF4/LoFi ordinary/fused
matmuls, native prefillK4/K17, B32decodeK6/K17,K24/K17 and BF16/HiFi4 head.
Profiler conclusions and gap accounting: profiler_conclusions.md. No >10–15% gap;
all host-tokenout excesses over inherited layer-stack guide are <=4.49% before
terminal allowance. Fresh reduced whole-layer device windows independently recorded;
never substituted for all64 device timing/rooflines.

CPU commands: `python models/autoports/qwen_qwen3_8_27b/tests/summarize_optimized_full_model.py`;
`python models/autoports/qwen_qwen3_8_27b/tests/audit_optimized_full_model_profile.py`;
three exact checker commands/exit codes are in stage_checks.json. AIME before/after
both phases remains99%top1/100%top5/100%top100 over100 reference positions, one problem.
Packet writer uses the supplied exact template/identifiers and actual B1 headline;
completed_samples is one AIME problem, with100 compared positions stated in evidence.
Final independent review and local checkpoint receipts follow; no push authorized.

Final independent xhigh stage review: CLEAN-PASS, no required work. Fresh reviewer
/root/stage07_final_review verified current source identities,227 runtime receipts,
prior/final benchmark prompt/output hashes, all shared/long output tokens, raw
profile attribution and four allocation-event peak reconciliations. Report:
stage_review.md, SHA256 cc6ef22e7e2703147f86fffd41d77177b1e8dcd5e771960ba7bf6025226d09fb.
Limits remain explicit: small timing gain,3.32% factual EOS latency cost versus
streaming, inherited long-answer delay, one-problem accuracy and inherited native
context execution evidence. No vLLM integration or datatype frontier was started.

Checkpoint preparation follows the existing stage convention: raw profiler captures
remain local under stage .gitignore and manifest receipts; compact human tables,
filtered perf-report CSVs, head-candidate CSV, summaries and provenance are committed.
Generated table CRLF/trailing whitespace is normalized without changing metrics or
raw op/device captures. git diff --cached --check passes. Checkpoint SHA follows.
