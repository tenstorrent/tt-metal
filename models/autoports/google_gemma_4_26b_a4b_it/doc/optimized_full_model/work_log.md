# Optimized full model work log — Stage 07

Target google/gemma-4-26B-A4B-it, revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52.
Starting checkpoint 941b23a758; clean worktree. Target mesh 1x4, four P300c ASICs.
`timeout 60 tt-smi -ls --local` succeeded, devices 0–3 visible. No reset needed.
Hardware commands are serialized by the stage owner. The existing Tracy web viewer
is not a device workload and was left alone.

## Baseline and scope

Preserve Stage05 decoder precision, residual layout, pooled CCL and rejection ledger.
The accepted residual is replicated BF16 DRAM, established by measured alternatives;
this stage does not introduce replication. Embeddings are hidden-sharded, head is
vocabulary-sharded, local vocabulary width65536 (already power-of-two).
The existing sampler uses32 physical candidates and greedy k1/p0, no full-vocab gather.

Baseline command: `python models/autoports/google_gemma_4_26b_a4b_it/tests/run_full_readiness.py --skip-readiness --performance --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/baseline/readiness.json`.
Log: `baseline/run.log`. Workload4096 input,128 output,B1,C1; all30 layers.

## Operation topology audit

| Boundary | Current path and policy | Candidate/action |
| --- | --- | --- |
| Embedding | hidden-sharded row-major BF16 lookup, one hidden gather, BF16 scale | Retain stack-compatible entry gather; inspect reduced profile cost |
| Decoder stack | Stage05 TP4/EP4, packed projections, policy by layer kind, pooled persistent CCL | Preserve accepted cumulative contract and measured rejection ledger |
| Final norm | FP32 norm arithmetic then BF16 cast | Preserve numerics; inspect terminal movement |
| LM head | BF16/HiFi4 vocab-sharded interleaved linear, K2816 N65536 per chip | Compare DRAM-sharded layout/config/readers at same precision |
| Logits | BF16 softcap then32-row sampler input | Preserve local vocab and semantic softcap |
| Sampling | local physical top32, candidate gather, semantic greedy k1/p0 | Retain common traced path; refresh correct greedy comparison |
| Feedback/cache | persistent tt_out_tok, device position/RoPE advance, changed-only tables | Preserve and test replay/request reuse |
| Host loop | nonblocking model+sampler replay then per-token readback | Device output buffer with one final bulk read; compare exact tokens |

AutoFix agent owns generator buffered-output experiment. Separate terminal agent
owns precision-preserving LM-head experiment. Neither may run devices concurrently.
All-layer profiling is prohibited; reduced profile retains layers0/5 and real terminal.

## Baseline and split-sampler results

Baseline command exits0: TTFT2114.794ms, token-out49.2561t/s/u;127 per-token
readbacks. `baseline/performance.json`. Initial source checkpoint941b23a758.
Fresh `probe_full_sampler.py --pad-batch --batch-sizes 1 --force-mode both
--iterations 100 --output .../sampler_comparison.json` exits0. Split physical32/
semantic k1,p0:508.518us; force-argmax3300.512us. Both B1 changed-input
known-winner checks pass. These are isolated warmed host replay times, not full-model
performance. Keep split greedy. Stage06 B3 force-argmax failure remains rejected.

`TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_buffered_generation --recorder-only --output .../buffered_recorder.json`
and the same without `--recorder-only` to `buffered_reduced.json` both exit0.
Exact uint32 recording, buffered/control token equivalence, request reuse,
resizing and feedback state pass. Tracker overhead invalidates these timings
as performance evidence. Extended sampled/Watcher checks remain pending.

## Watcher setup recovery

`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_WATCHER=10 timeout600 python -m
...tests.check_buffered_generation --gen-len8 --prompt-lengths31 32 33 127 129
1023 1024 1025 4097 --output .../buffered_extended_watcher.json` failed at mesh
open before model execution: ACTIVE_ETH program28464 bytes >26624 config limit.
Preserved `buffered_extended_watcher.log`. The evidence-backed environment workaround is a scoped retry with
`TT_METAL_WATCHER_DISABLE_ETH=1`; worker assertions remain enabled, while
Ethernet Watcher coverage is unavailable for this run.
Bounded list/reset/list logs: `recovery_list_before.log`, `recovery_reset.log`,
`recovery_list_after.log`. No foreign processes killed or locks removed.
The test harness allocation was corrected to include the largest requested
boundary prompt before retry; the first run never reached this test logic.

Recovery list/reset/list completed successfully; all4 devices visible. `timeout60 python` fabric1D/mesh1x4 open-close succeeds (`recovery_mesh.log`, MESH_SMOKE_OK). Scoped Watcher retry uses TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1, trace allocation tracking enabled; `buffered_extended_watcher_retry.log`.

## Selected path and expanded checks

LM-head real-input controls (`terminal_lm_head_plan.md`,44-row CSV) select the
11x10 interleaved BF16/HiFi4 K4 candidate:936.85us vs K8 control955.30us.
DRAM-sharded adapted families are slower; larger K blocks22/44/88 run after
reducing chunk width but lose total projection latency. No decoder policy changes.

Scoped Watcher retry exits0 (`buffered_extended_watcher.json`): exact greedy
and seeded top-k16/top-p0.9 control equality, mode alternation, request isolation,
zero/single output, capacity growth, reset, and runtime host-boundary audits.
Logical prompt lengths31,32,33,127,129,1023,1024,1025,4097 all preserve exact
control tokens. Worker Watcher and trace-allocation tracking enabled; Ethernet
Watcher disabled only due the measured firmware config-size limit.

All-layer command: `OMP_NUM_THREADS=8 timeout1200 python -m
models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_full_model`,
log `final_validation.log`. It refreshes AIME24 prefill/traced teacher forcing,
all six shared prompt controls, buffered128-token autoregression, and three
warmed4096/128/B1/C1 requests per buffering mode. Exact verified HF reference
reuse is recorded in `reference_metadata.json`; no mismatched reference copied.

## Final all-layer results and reduced profiling

The all-layer command above exited 0. `readiness.json`: prefill top1/5/100
96/100/100%, teacher-forcing 94/100/100%, 100 continuation positions each.
`performance_comparison.json`: selected buffered median TTFT 2113.192867 ms,
decode 49.354078 t/s/user; same-head streaming control 2115.115080 ms and
49.264327 t/s/user. Original baseline is 2114.794180 ms and 49.256080 t/s/user.
All are 4096-input/128-generated/B1/C1 all-30-layer autoregressive requests.
Throughput is essentially unchanged; the verified improvement is removing
per-token host readbacks. Exact output equality passes. Traced teacher forcing
49.966053 t/s/user is a separate 161-input/100-position experiment.

All six shared qualitative outputs match the Stage06 control exactly; buffered
128-token outputs preserve the EOS-stopping prefix. Actual HF and TT texts were
read, including refreshed sky explanation (`qualitative_verdict.md`).
`python models/common/readiness_check/check_degenerate_output.py --hf-model
google/gemma-4-26B-A4B-it --missing-artifacts critical --scope autoregressive
--json models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/degeneracy.json`
exited 0 (`degeneracy.log`).

Profiler command, run with python_env/bin on PATH:
```sh
OMP_NUM_THREADS=8 timeout 900 python -m tracy -r -p -v --op-support-count 100000 --no-op-info-cache --disable-device-data-dump-to-files --disable-device-data-push-to-tracy -o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/profile -n reduced_full_path -m models.autoports.google_gemma_4_26b_a4b_it.tests.profile_optimized_full_model
```
Exit 0; `profile.log`. Raw CSV is
`profile/reports/reduced_full_path/2026_09_27_08_30_38/ops_perf_results_reduced_full_path_2026_09_27_08_30_38.csv`.
Only real layers 0/5 and terminals were instrumented, as required by the safe
profiling contract. No reduced result is labeled full-model device time.

For each phase PREFILL/DECODE, ran `tt-perf-report RAW --start-signpost PERF_PHASE
--end-signpost PERF_PHASE_END --active-experts 8 --no-summary`, adding
`--tracing-mode` for decode, once with `--csv profile/phase_perf_report.csv` and
once without for the human `.txt` table. Advice remains enabled. Console outputs
are retained. `python -m models.autoports.google_gemma_4_26b_a4b_it.tests.summarize_full_profile
RAW --output .../profile/summary.json` records SHA256 and whole windows including
gaps. Device prefill 667191.590 us, decode 3082.769 us; sampler max 514.067 us,
recorder max 9.821 us. Full advice/checklist dispositions are in
`final_perf_findings.md`, with precision-locked terminal candidate evidence and
links to inherited decoder geometry/CCL/layout rejection experiments.

`OMP_NUM_THREADS=8 timeout 600 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace
--batch 3 --output .../trace_mixed_slots.json` exited 0: mixed lengths 31/63/127,
active fixed slots 0/2, inactive cache unchanged, independent isolated-slot logits
and feedback pass. Batch32 all-layer and B1 page-table/reset checks follow.

Final state checks both exited 0:
```sh
OMP_NUM_THREADS=8 timeout 1200 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --batch 32 --all-layers --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/trace_full_batch32.json
OMP_NUM_THREADS=8 TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 600 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --batch 1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/trace_page_tables.json
```
Batch32 runs all 30 layers with alternating lengths 31/127 and checks independent
slot controls (minimum decode PCC 0.9999999404). B1 verifies persistent token and
position feedback, unchanged tables avoiding refresh, changed tables updating in
place, reset and repeated/changed prompts. Matching `.log` files retain output.

Installed Stage07 gate with `MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it`
and `07-optimized-full-model.check.sh` exits 0 (`stage_check.log`).
The large prefill CSV and human advice tables are preserved byte-for-byte as gzip
for the repository's 500 KB file limit; uncompressed originals remain local.
Only Python/model documentation changed, so no native build is required.

Final pre-commit first pass normalized trailing whitespace in three text logs
and the rendered completion, plus final newlines in completion/degeneracy JSON.
Token IDs and numeric evidence are unchanged. A second pass follows.

Before/after accuracy is unchanged: Stage06 `../full_model/readiness.json`
and Stage07 `readiness.json` both report prefill 96/100/100% and decode
94/100/100% top1/5/100 across the same 100 reference positions. Final pre-commit
second pass exits 0 (`final_precommit_pass.log`).

## Independent review and checkpoint

Fresh xhigh stage-review subagent returned **clean-pass**, no Required Work;
`stage_review.md` records inspected files, hashes, commands and residual risks.
Its documentation attribution correction was applied before the verdict.
No implementation changed after final device validation. Local checkpoint follows;
no remote push.

Commit preparation also normalized CSV CRLF to LF (values unchanged). The first
commit attempt found no configured Git identity; reuse the established local
checkpoint identity without changing global Git configuration.

Stage-owned implementation/evidence checkpoint: tt-metal, branch
`gemma-4-26b-a4b-it`, commit `9473a2e1b18be1dc9e5fe0add73a0f62ffcc115f`.
Commit hooks all pass. No push performed. This following documentation-only
checkpoint records the implementation SHA; telemetry records both checkpoints.
