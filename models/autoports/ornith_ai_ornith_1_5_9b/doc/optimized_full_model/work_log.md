# Optimized full model work log

Stage 7 starts at ebd521f7, clean worktree, TP4 P300c Blackhole IDs 0–3.
Enabled installed packages: tt-model-bringup 0.1.23 and tt-autodebug 0.1.8.
Environment: source python_env/bin/activate; source /home/ttuser/ornith-9b/lane-env.sh.
No vLLM work and no pushes. Devices were unowned according to fuser; bounded
tt-smi listing passed. No reset authorization assumed or reset performed.

## Preserved contracts

The previous stage selected replicated-across-ranks BF16 residuals, locally
width-sharded L1. This is the measured selected layout, not a stage-7 fallback.
Retain BF8 projection/CCL/KV, attention HiFi2, MLP LoFi, FP32 recurrence,
shape-dependent prefill precision and existing rejection ledger. Context262144
and B32 support remain unchanged. No datatype-frontier search.

## Operation topology audit before tuning

| Boundary | Current path | Candidate/decision |
|---|---|---|
| Embedding | BF16 hidden-sharded lookup, gather once to residual | Keep existing TP lookup; profile entry cost |
| Decoder stack | packed QKV, DRAM-sharded projections, shared persistent CCL, selected residual | Carry stage5 coherent-family experiments and rejection ledger; no interlayer conversion |
| Final norm/head | local norm in residual layout, DRAM conversion, interleaved vocab-sharded head | Sweep DRAM-sharded BF8 HiFi2 head, reader count and K block; retain sampler-ready shards |
| Sampling | local tile-shaped top32, candidate gather, semantic k1, feedback | Compare correct split greedy vs force argmax; keep sampled modes |
| Host loop | nonblocking model+sampling replay, synchronous caller token read each step | Add/measure resident multi-step replay with no per-token reads/uploads |
| Cache/positions | persistent KV/table/position, device increment | Keep unchanged and rerun tracked/mixed-state gates |

## Baseline

`OMP_NUM_THREADS=8 TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 900 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_perf_probe --model-only --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/baseline.json`

Exact output: baseline.json; console logs/baseline.log. Serial devices0–3.

## Integrated head and resident path

Head geometry retained BF8/HiFi2/FP32 accumulation and BF16 output. Unsplit
three-reader K2/target64 measured0.570ms versus0.767ms interleaved on real
reduced-stack activations; PCC0.99999994, max absolute logit delta0.03125 and identical argmax. K4/K8/K16,64/110 output targets
and1/2/3/4 readers were compared; illegal static L1 configurations are retained
in head_sweep/head_split4/head_unsplit artifacts. Four-way splitting adapts the
one-reader path successfully but does not beat the multi-reader whole shard.
LMHead1D is used directly from tt-transformers25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73,
installed editable with --no-deps --no-build-isolation. No source edits there.

The public decode_resident method queues fixed steps with borrowed final-device
tokens, no retained intermediate sequence. Streaming/readiness interfaces remain
unchanged. It rejects uninitialized/reset/host-logit state, pending submissions,
closed generators and context overruns. Its benchmark compares final sampled
tokens and exact device position against streaming generation and traps host
conversions/copies/synchronizations inside submission.

Initial integrated B1 128/128:92.028ms TTFT;114.592t/s/u with readback;
114.853t/s/u resident split sampling. Baseline resident112.477t/s/u. These are
intermediate measurements; final default reruns will supply the headline.
AIME24 prefill99/100/100 and traced decode98/100/100 percent pass.

## Masked native-batch gap

Full-model B32 all-active measured10.741ms/token (93.102t/s/u);31active slots
with one inactive measured18.050ms/token (55.402t/s/u). Existing before/after
protection copies16MiB of FP32 recurrent pool, merges every row, then copies
the full result back, per each of24linear layers. This explains a large
full-model-only traffic term; the selected decoder arithmetic is unchanged.
A source-local raw-copy candidate saves/restores only inactive recurrent rows.
The isolated tracking-enabled probe passes30mask transitions at B1/B3/B32
on all four ranks with bit-exact FP32 results and unchanged program counts.
It covers all-active/all-inactive/first/last/alternating rows and reversed order.
No allocation acknowledgment or tracker exemption is introduced.

The initial reduced B32 profile failed before measurement on the already-known
16-bit profiler source-location collision. Retained log:profile_masked_before.log.
AutoDebug child sandbox preflight could not create an unprivileged namespace;
retried the inspection-only child with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the
user-authorized unrestricted session. Prior full-model evidence names the
byte-identical source overlay and model-directory cwd repair. The retry uses
that exact overlay; this changes profiler source IDs, not production math.
No device reset, process termination, or discarded failure evidence.

## Final selected state and validation

Selective recurrent save/restore passes bitwise B1/B3/B32 mask transitions,
full32-layer B32 trace allocation tracking (including program caches), and
Watcher on the B4 scheduler/mask/remap lifecycle. Full-pool control and selected
path both preserve identical final token vectors and inactive state hashes.
Matched128-step B32/31active:18.017388ms control versus11.471057ms selected,
36.333% latency reduction. B32 all-active remains10.741069ms; the remaining
masked overhead is0.730ms across24linear layers, including retained convolution
protection. No decoder arithmetic, precision or layout changes.

Final B1 128/128:92.049307ms warmed TTFT,114.407258t/s/u caller-readback,
114.827306t/s/u resident nonblocking split sampling. Baseline90.017093ms,
112.028965t/s/u readback and112.477446t/s/u resident. TTFT regresses2.26%; it is
reported, not hidden by the decode gain. Same-workload traced teacher forcing
113.846658t/s/u (baseline111.581816), includes token uploads/sampling/readback.
The reference-driven AIME24 decode measurement is a separate workload/setup.

Final AIME24 chat-template gates retain prefill99/100/100 and decode98/100/100
percent top1/top5/top100. Commands/logs carry exact reference and source hashes.
All device jobs use the serial wrapper with explicit timeout; final queue is
final_validation_jobs.json. The original queue was stopped between jobs after
capacity to incorporate the discovered masked-pool optimization.

## Accepted profiles and final accounting

The effective overlay fixes the source-ID collision. Broad reduced B32 device
execution passes, but its huge warmup catalogue causes host report enrichment
to fail; it is not accepted as a perf report. Reduced B1 full-path and B32
one-layer guard-only captures both finish, followed by a32769-token reduced
full-model chunk-boundary capture. All accepted CSV windows are complete.
Exact commands:commands.jsonl.xz, final_validation_jobs.json and chunk_profile_job.json.
AutoDebug/AutoFix reports are under profiler_debug; no shared profiler source changed.

104-op decode rows give linear0.243511ms, full0.262858ms, stack7.947128ms,
entry/terminal1.231759ms. Extrapolation9.178887ms versus full resident8.708730ms
has no positive unexplained gap. Same-run reduced FW maximum1.729251ms and
synchronized host1.808981ms reconcile separately; no negative-host-time claim.
Head kernel556.647us; norm/head/layout601.917us versus previous784.385us.
Guard profile predicts6.656112ms saved versus measured6.546332ms. Raw matmul
attributes prove selected BF8/HiFi2 or LoFi policy. Kernel shape metadata for
row-polymorphic MinimalMatmul is cached by hash: executed row counts are annotated
from the validated prefill plan without modifying raw CSVs. See profile/README.md.

Full-context final tracking passes262143/262144 prefill and terminal decode,
with822programs and six live decode traces. Added head270745600bytes/device;
free DRAM20214922240bytes/device after preparation. context_contract.json adds
this evidence without changing any prior stage's historical scope or capability.
The final raw and AIME24 chat-template AR runs are coherent but cap-limited;
mechanical stage gate passes. Six shared prompts exactly preserve prior-stage
seed0 tokens; known HF-controlled scientific overstatement/thinking-cap limits
remain recorded. All45 host contract tests and touched-source pre-commit hooks pass.

Completion uses optimize stop rule1: checklist items have measured/kept/rejected
decisions, with unchanged decoder experiments carried by source equivalence.
No70% hardware-limit claim or eight-hour time-box claim is made. Prefill tracing
and shape-specialized prefill head reuse remain possible further optimizations;
they are disclosed decisions rather than unproven runtime limitations.

## Round1 review and resident page repair

Independent round1 returned `more-work-needed`: public resident decode could
advance from position31 into page1=-1, and ordinary one-step resident decode
had the same admission omission. See stage_review_round1.md for the source
counterexample and paged-cache writer address calculation. No unsafe request
was executed. The cause was directly established, so the minimal direct fix
adds `_validate_resident_window` before either public path mutates state or
submits traces. It checks every active row's full causal prefix through the
last requested position, using CPU metadata only. Remap-only admission can
check the original pairs because both positions and table rows are permuted.
Decoder arithmetic, sampler, copy kernel, allocation and trace bodies did not
change; the earlier accuracy, capacity, Watcher and profiler evidence remains
scoped to those unchanged bodies, not claimed to have current generator hashes.

New host cases cover last mapped/first unmapped, a multi-step hole, inactive
unmapped rows, context overrun, ordinary resident admission and unchanged
submission/position counts on rejection. All49 host contract tests pass.
`resident_page_guard.json` uses real layers0/3, B3/context128, tracking1 and
fixed program-cache entries. Mapped-boundary replay and changed-table
continuation match explicit-feedback control; sparse requests reject before
any replay; six traces and412 programs are unchanged. The first probe's
`changed_only_page_refresh` assertion wrongly expected one copied row. The
existing explicit reload command intentionally uploads all three fixed-slot
rows (serving_decode_probe already asserts this); the test was corrected,
without changing that contract. Failed evidence is retained in
resident_page_guard_probe_assertion.json and logs/resident_page_guard.log.gz.
Steady resident replay performs no page-table uploads. This was a diagnostic
assertion error, not an invalid-address device execution.

The tracking-enabled existing serving command/ownership probe also passes
(serving_guard_final.json): resident remap, parameter changes, inactive rows,
owned pending outputs and explicit table reload semantics remain intact.
Post-repair full32 B1 (optimized_guard_final.json): warmed TTFT89.839262ms,
resident114.841770t/s/u (8.707633ms), caller-readback114.599219t/s/u,
traced teacher-forcing113.956140t/s/u. Earlier92.049307ms TTFT remains in
optimized_final.json; run-to-run spread prevents claiming a material TTFT gain.
Post-repair full32 B32 with31 active rows (batch32_guard_final.json):
11.473304ms/token,87.158854t/s/u,36.321% lower latency than18.017388ms control;
128 replay pairs, zero token/position/page/readback/capture counters inside the
window and599 unchanged prepared programs. The largest masked gap stays closed.

Exact post-repair commands, source hashes, environment and logs are named
resident_guard_host_final, resident_page_guard_final, optimized_guard_final,
batch32_guard_final, serving_guard_final, stage_gate_guard_final and
degenerate_guard_final in commands.jsonl.xz. Every command exits0. Stage gate
and degenerate-output check pass. final_audit.json explicitly separates earlier
compute evidence from current generator source. No broad accuracy, dtype or
geometry sweep was repeated for this host-only admission fix.

## Passing independent review and checkpoint

Round2 returned **clean-pass**, with no Required Work; see stage_review_round2.md.
The reviewer independently reconstructed both pre-guard production and measured
probe hashes, checked all107 archive hashes/lengths, and re-derived the new
metrics. The page-admission P1 is closed. The nonblocking inventory wording
was clarified to scheduler-managed page tables; explicit reload semantics were
already correct. Known HF-controlled output limitations and TTFT variability
remain documented, not waived accuracy failures.

probe_format_provenance.json retains the exact import-only isort reconstruction
for the measured probe. Current production matches all guard reruns. The command
journal uses lossless XZ (commands.jsonl.xz; restore with `xz -dc`) because its
gzip archive exceeded the repository500KB limit. Other archived files remain
gzip. Pre-commit's generated-JSON final-newline fix is also retained. Final
precommit_guard_pass2 passed; git diff checks passed. No device reset, push,
vLLM integration or external dependency source edit occurred. The pinned
tt-transformers dependency checkout remains clean.

The following local checkpoint records all stage-owned implementation, tests
and evidence under this model directory on branch main. Its SHA is logged in
the subsequent documentation-only bookkeeping commit.

| Repository | Branch | Local stage checkpoint |
|---|---|---|
| `/home/ttuser/ornith-9b/tt-metal` | `main` | `0c8c478bd049c7f74a58db50c61dcca0ff5e7091` |

This checkpoint contains227 stage-owned files, including the round2 clean-pass
report. Commit hooks passed and the worktree was clean immediately afterward.
This SHA record is a documentation-only follow-up; its own commit is discoverable
with `git log -1 -- doc/optimized_full_model/work_log.md` from the model directory.
The external tt-transformers checkout remains unchanged at
`25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73`; no other repository needed a commit.
Neither commit is pushed. Optimized-full-model is complete; vLLM was not started.
