# Accepted-candidate release validation — 2026-10-03

## User contract and current status

The user accepts the measured **2/5 SWE** from combined run
[36983437902](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36983437902)
and requests an actual release workflow with serving benchmarks, **GPQA10,
TerminalBench5 and the original SWE5**. This supersedes the earlier objective
of making all five SWE tasks correct. It does not turn 2/5 into 5/5 or waive
other missing/failed execution, accuracy or API rows. No new quality-policy
experiments are part of this run. Configuration audit began 2026-10-03;
dispatch/results are tracked below. Starting documentation TT checkpoint95d6e17caa,
TTIed5cd392. No local hardware or inference process is started.

## Exact serving provenance

- Image (reused; **no build**):
  `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:ad58effd178b9d8c7689a392159532d30aea0c1fe24bec82c06dbc0d3ebf4bbd`.
- Native TT-Metal: `c9ec3469f1b875e7e5e505660c4421e5126e8dad`.
- Bundled autoport vLLM/plugin: `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`;
  imported from the autoport's `vllm_plugin_snapshot/src`, not an unrelated checkout.
- HF/tokenizer revision: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- Read-only precision overlay: TTI `reference_config/precision/gemma4_eval_bfp8.json`,
  SHA256 `46389c08f1c99f068669f66e902cc34daf00e54e7c7d8014dc1539b3dd1af954`.
  This is the accepted configurable/decode-BFP8 policy; it does **not** promote
  the subsequently rejected new-prefill-BFP8 experiment. Its individual existing
  prefill dtype entries are unchanged; the short name is not an all-BFP8 claim.
- QB2 P300x2 / four Blackhole ASICs; TP4, 262144 context, 32 server slots,
  async scheduling, on-device sampling, decode-only trace, no model-wide warmup.
  One 4096-input/4-output readiness warmup. Prefix caching and chunked prefill off.
  Persistent HF/converted/kernel caches and one loaded server across release children.
- Full SHAs, image digest, resolved startup config and cache attestation must be
  checked again in actual CI artifacts. Documentation HEAD is not native-image HEAD.

## Evaluation contract

All model requests use T1.0, top-p0.95, top-k20, max output32768, explicit
request seed9472 and repetition guard `{min_pattern_size:16,max_pattern_size:1024,min_count:8}`.
These are declared accepted-candidate policy deviations, not upstream defaults.
No focused prompts, task hints, review/advisory intervention, history truncation,
greedy sampling or output-budget reduction. GPQA uses the existing thinking chat
template and original r1 zero-shot task. Benchmarks use their existing synthetic
serving harness parameters, not agentic generation sampling.

| Family | Exact selection | Scheduling / budgets |
|---|---|---|
| GPQA | `r1_gpqa_diamond`, integer limit10, processed dataset doc_ids0–9 | C1, batch1; harness RNG seed42 retained, request seed9472; request timeout7200s |
| TerminalBench | `terminal-bench/caffe-cifar-10`, `terminal-bench/password-recovery`, `terminal-bench/portfolio-optimization`, `terminal-bench/hf-model-inference`, `terminal-bench/financial-document-processor` | terminus-2 JSON parser; serialC1, one attempt,10800s agent budget/task,3600s LLM timeout;16CPU/48GiB task limit |
| SWE | `astropy__astropy-14096`, `django__django-11299`, `matplotlib__matplotlib-25332`, `scikit-learn__scikit-learn-14629`, `sympy__sympy-13551` | mini-swe-agent2.2.8; serialC1, one attempt,7200s agent budget/task,1800s LLM timeout |

Listed task order is selection order; Harbor's actual execution order is recorded
from timestamps after completion (it differed in the prior combined run).
SWE alone retains corrected testbed PATH/CONDA_DEFAULT_ENV and the generic
submission-marker adapter. TerminalBench receives neither SWE environment overrides
nor that submission adapter. Validated owned-process cleanup is enabled only for
SWE; abandoned-request cancellation and compact request telemetry are enabled for
both agentic families. These harness repairs prevent verification racing timed-out
commands; they do not add instructions or promise better reward.

The previous SWE-only catalog and `evals -> agentic` transport override were
removed: otherwise release would omit GPQA and invoke agentic twice. Actual
release order is **GPQA → benchmarks → spec/API tests → TerminalBench → SWE**.
Structured repetition settings cannot traverse lm-eval's comma-split generation
CLI, so a tested explicit request-override adapter attaches the guard/seed to
raw HTTP payloads without altering prompts or harness RNGs. No image rebuild.

## Established serving matrix

Restore the historical23-row Qwen-style release matrix from CI36490378756,
not the intervening29-row C1/C8/C16 optimization-qualification matrix.
Every row reports TTFT, TPOT/TSU, E2EL, successful/failed requests where emitted;
missing fields are not fabricated. No token-cap or32-slot capability reduction.

| Input | Output | concurrency / number of requests |
|---:|---:|---|
|128|128|1/8;32/256|
|128|1024|1/4;32/128|
|1024|128|1/4;32/128|
|2048|128|1/4;32/128|
|4096|128|1/4;32/128|
|8192|128|1/2;31/62|
|8192|1024|1/2;28/56|
|10000|1024|1/2;23/46|
|16384|128|1/2;15/30|
|32768|128|1/1;7/7|
|65536|128|1/1;3/3|
|131072|128|1/1|

## Pre-dispatch runtime forecast and limits

- GPQA historical C1 evidence:139.31s/question (40-sample run36501826660),
  giving23m13s for10 under the old policy. Planning estimate20–40m; guard/seed/
  precision can change reasoning lengths. Ten full32K generations alone need
  about1h51m at49tok/s, plus prefill; this is not a measured new-policy duration.
- SWE observed accepted-candidate trial span2h01m06s, dispatch2h14m54s,
  reward2/5. Use~2h agentic-family planning center,1–4h scenario range;
  stochastic/environment/cache differences mean neither time nor reward is guaranteed.
- Historical23-row serving workflow took3h13m47s including startup. Planning
  center3h, scenario2–4h; no claim that decode speedup scales every prefill-heavy row.
- TerminalBench has **no matched serial accepted-policy baseline**. Old C5 run
  had2/5 successes (portfolio753s and HF5419s) and three10800s timeouts.
  A3–10h family allowance is only a scheduling scenario, not an evidence-derived
  speedup prediction. The exact serial agent-budget ceiling is15h plus setup/verifier.
- Thus a rough workflow planning envelope is~7–18h including setup/API checks,
  not a confidence interval or solve-time claim. The reused orchestration job's
  **18h outer timeout can truncate the worst case**; unchanged per-task budgets
  alone sum25h across TB/SWE. Monitor family progress and report any censoring.
  Do not shorten tasks or treat timeout as solve time to fit a wall-clock target.

This explicitly requested small release subset is not unrestricted readiness:
GPQA full198, TerminalBench full dataset and SWE500 would be substantially more
expensive. Even GPQA's historical mean projects7h40m for198 before agentic suites.
Only exact10/5/5 subset scores will be reported; published full-set references
are not matched-subset controls. Workflow success and readiness/quality verdicts
are separate. No all-five-correct claim is required by the updated user contract.

## Validation and handoff

Before dispatch:207 host tests pass across eval config, request adapter, agentic
driver, telemetry, cancellation, release routing and standard eval command;
131 additional benchmark/overlay/startup/config checks pass (overlapping subsets).
Final combined host suite: **311 passed** in8.44s, one pre-existing pytest
collection warning. TTI **c2d737f215201dbda02880c85f2a529ec70a0e11**, initial
plan/docs TT **db4eb3cfa4**, both pushed.

### Live CI checkpoint — 13:28 UTC

[Actual release37126247213](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37126247213),
[hardware job111212039304](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37126247213/job/111212039304),
runner`120-qb2-p04t07`, dispatched**2026-10-03T13:27:35Z**. Builds are skipped;
runner setup is in progress. `workflow=release`, `run-full-evals=false`
(ci-nightly exact10/5/5), AI summary and issue-comment publication disabled.
All git inputs use the full native TT/plugin/TTI SHAs above; immutable image override.

Initial dispatch37126220935 accidentally used abbreviated TTIc2d737f2 and was
cancelled immediately before a hardware job was created. It reached terminal
`cancelled` at13:27:54; replacement above uses the full40-character SHA. This
is an orchestration correction, not an eval rerun or failed model result.

Read-only30-second monitor evidence:
`/home/mvasiljevic/gemma4-eval-speed-evidence/release_20261003_37126247213/live_metrics.jsonl`.
Results, artifact IDs, exact runtime provenance and final elapsed agent-work window
remain pending. Monitor dispatched CI through completion; retry only infrastructure/
configuration failures, not low reward under the accepted policy.
Keep large logs/CSV/Tracy outside git under `/home/mvasiljevic/gemma4-eval-speed-evidence`.
