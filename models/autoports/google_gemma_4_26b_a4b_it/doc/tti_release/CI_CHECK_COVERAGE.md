# CI check coverage for this port: what has run here, and what has not

Cross-checked 2026-08-18 against `agentic-research@bh-model-status`
`reports/bh-model-evals.md`, which tracks eight CI checks per port and scores this
one **7 of 8**. That report is generated and not authoritative on config detail —
two of its statements are stale against
`tt-inference-server@vvukoman/add-8-models-to-release-flow` (`60f80c4b`), noted
below — but its checklist is the right frame.

## The eight checks

| CI check | status here | evidence |
|---|---|---|
| Graded benchmark point (isl 128 / osl 128 / c1) | **run, PASS** | `local_release_report.md`: TTFT 261.2 ms vs 300 target, 26.7 TPS vs 26 |
| Benchmark sweep, long ISL | **NOT RUN** — see below | — |
| Benchmark at concurrency 32 | not run on the release recipe; see the c32 note | — |
| Layer PCC + AIME24 on target weights | pre-existing on the branch | `doc/functional_decoder/`, `doc/datatype_sweep/` |
| Eval — GPQA | **run** | old task 4/10 → 9/10 serial; `r1_gpqa_diamond` 3/3 smoke, CI-nightly 0.2 running |
| Eval — IFEval | **run, at the bar** | 82.62, reproduces the recorded row exactly |
| Spec tests / API conformance | **no-op** — no suite matches this model/device | `tti_spec_tests.log`: "No spec test suites match model='gemma-4-26B-A4B-it' device='p300x2'" |
| Full release workflow, end to end | **run** | acceptance FAIL on 2 Docker-dependent agentic rows; evals `NA`; benchmarks PASS |

## Why the ISL sweep was never run — it is suppressed by an env var

`ONLY_BENCHMARK_TARGETS` **skips the sweep entirely** and runs only the
perf-reference point (`reference_config/benchmarking/benchmark_config.py:616`).
`RUN_NOTES.md` records that variable in the definitive release command, and every
run on this machine copied it. So the missing check is not a hardware or model
limit; the recorded invocation switches it off.

Dropping it runs `BENCHMARK_ISL_OSL_PAIRS` (`benchmark_config.py:91`) filtered by
`isl + osl <= max_context`. With the release recipe's `max_context = 49152`:

| point | in sweep at 49152? |
|---|---|
| (128,128) (128,1024) (1024,128) (2048,128) | yes |
| (4096,128) (8192,128) (16384,128) (32768,128) | yes |
| (65536,128) (131072,128) | **no** — 65,664 and 131,200 exceed 49,152 |

So **8 of 10 points**, capping after isl 32,768. That confirms the report's
prediction that "the sweep will cap below CI's 131,072 ceiling"; the cap sits one
point lower than its 65,535 guess.

## Two places the generated report is stale

1. **GPQA CI-nightly fraction.** The report says `CI_NIGHTLY` runs "**5 %** — about
   10 of GPQA Diamond's 198". `60f80c4b` sets `EvalLimitMode.CI_NIGHTLY: 0.2` for
   this model's `r1_gpqa_diamond`, i.e. ~40 documents. The run in progress here uses
   0.2, taken from the config rather than the report.
2. **"Benchmark at c32 with a long output".** The release entry sets
   `max_concurrency: 1`, and the sweep expands concurrency from
   `model_max_concurrency`, so c32 is not part of this model's graded path any more.
   A c32 measurement is still worth having — the serving-concurrency defect in
   `POST_FIX_EVAL_RESULTS.md` only appears above one row — but it is a diagnostic
   here, not a CI check.

## Not reachable on this host

- **A paired control for `r1_gpqa_diamond`** (report item 3, the fix for the `NA`
  grade). The old-task HF control took 2,978 s on CPU for 10 documents of a few
  hundred tokens. Thinking mode generates 6.8k–12.7k tokens per document, so the
  same control over 40 documents is orders of magnitude more compute — weeks of CPU.
  This needs a GPU reference, not this machine.
- **`terminal_bench_2` / `swe_bench_verified`** — no Docker in this container.
- **TTI-managed server startup** — TTI has no launcher for `models/autoports/*`.
