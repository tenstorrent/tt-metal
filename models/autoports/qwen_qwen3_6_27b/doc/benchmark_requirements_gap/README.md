# What the benchmarks measure, and what Rev 0.11 asks for

Source of truth: `doc/QB2 Model Support Requirements - qwen3.8-27b (agentic
coding) - p150x4 _ p300x2 - qwen3.8-27b (coding).csv`, Rev 0.11. "What we run"
is read out of the CI benchmarks job for `38153c48c8a`
(run `34360774551`, job `102496979872`), which is the most complete benchmarks
run this model has.

**The headline: of the 21 measurements the requirements ask for, CI produces
zero of them.** Not "fails" — does not measure. Every CI point differs from every
required point in at least one of OSL, concurrency or ISL, and the two most
load-bearing requirements (warm-prefill/APC, and batch 8/16) are absent
entirely. Meanwhile 10 of CI's 18 hours are spent on a concurrency the
requirements never ask for, and it is those points that get killed.

## 1. What Rev 0.11 requires

21 measurements. OSL is **252** for every one of them — the TraceLab per-step
median output for Claude (arXiv:2606.30560 Table 8).

### Batch 1, cold prefill — "agentic coding without subagents"

| ISL | OSL | batch | TTFT cold target | decode t/s/u | agg t/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 252 | 1 | 60 ms | **50** | 50 |
| 1024 | 252 | 1 | 150 ms | **50** | 50 |
| 4096 | 252 | 1 | 500 ms | **49** | 49 |
| 16384 | 252 | 1 | 1800 ms | **47** | 47 |
| 32768 | 252 | 1 | 3500 ms | **46** | 46 |
| 65536 | 252 | 1 | 8000 ms | **43** | 43 |
| 131072 | 252 | 1 | 22000 ms | **40** | 40 |
| 262144 | 252 | 1 | 60000 ms | **34** | 34 |

### Warm prefill with APC — the steady-state agentic path

Prefix cached, ≤1K appended. This is the row that decides whether an agent loop
is usable, because >90% of coding steps append <1K over a large cached prefix.

| ISL (cached prefix) | warm TTFT target | cold ref | required speedup |
| ---: | ---: | ---: | ---: |
| 128 | 60 ms | 60 ms | 1.0x |
| 1024 | 150 ms | 150 ms | 1.0x |
| 4096 | 150 ms | 500 ms | 3.3x |
| 16384 | 160 ms | 1800 ms | 11x |
| 32768 | 170 ms | 3500 ms | 21x |
| 65536 | 190 ms | 8000 ms | 42x |
| 131072 | 220 ms | 22000 ms | **100x** |
| 262144 | 300 ms | 60000 ms | **200x** |

Measured with `vllm bench serve --dataset-name prefix_repetition`
(`--prefix-repetition-prefix-len 131072 --prefix-repetition-suffix-len 1024
--prefix-repetition-output-len 252 --max-concurrency 8`). Pass bar: ≥50% E2EL
reduction vs cold.

### Batch 8 and 16 — "with subagents"

| profile | ISL | OSL | batch | TTFT cold | decode t/s/u | agg t/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch 8 | 4096 | 252 | 8 | 500 ms | **34** | 272 |
| batch 8 | 32768 | 252 | 8 | 3500 ms | **28** | 224 |
| batch 8 | 131072 | 252 | 8 | 22000 ms | **22** | 176 |
| batch 16 | 4096 | 252 | 16 | 500 ms | **32** | 512 |
| batch 16 | 32768 | 252 | 16 | 3500 ms | **24** | 384 |

Per-user decode must clear a **20 t/s usability floor**. ≥128K is served at
≤ batch 8; batch 16 is short/mid context only.

## 2. What CI actually runs

19 points attempted, `(ISL, OSL, concurrency)`:

| ISL | OSL | conc | n | outcome |
| ---: | ---: | ---: | ---: | --- |
| 128 | 128 | 1 | 8 | ok |
| 128 | 128 | 32 | 256 | ok |
| 128 | 1024 | 1 | 4 | ok |
| 128 | 1024 | 32 | 128 | ok |
| 1024 | 128 | 1 | 4 | ok |
| 1024 | 128 | 32 | 128 | ok |
| 2048 | 128 | 1 | 4 | ok |
| 2048 | 128 | 32 | 128 | ok |
| 4096 | 128 | 1 | 4 | ok |
| 4096 | 128 | 32 | 128 | **KILLED** at 7200 s, 33/128 |
| 8192 | 128 | 1 | 2 | ok |
| 8192 | 128 | 32 | 64 | **KILLED**, 1/64 |
| 8192 | 1024 | 1 | 2 | ok |
| 8192 | 1024 | 32 | 64 | **KILLED**, 1/64 |
| 10000 | 1024 | 1 | 2 | ok |
| 10000 | 1024 | 32 | 64 | **KILLED**, 1/64 |
| 16384 | 128 | 1 | 2 | ok |
| 16384 | 128 | 31 | 62 | **KILLED**, 0/62 |
| 32768 | 128 | 1 | 1 | run cancelled during it (18 h job cap) |

## 3. The gap, ranked by consequence

### 3.1 OSL is 128 or 1024; the requirement is 252 — every point, no exceptions

Not a rounding difference. OSL sets the prefill/decode balance of the
measurement. At OSL 128 a point is prefill-dominated and reports mostly TTFT; at
OSL 1024 it is decode-dominated. The requirement picks 252 because that is the
measured median agentic step. **No CI point can be compared to any requirement
row without re-running at OSL 252.**

### 3.2 Concurrency 32 is measured and not required; 8 and 16 are required and not measured

The requirements ask for batch **1, 8, 16** and explicitly cap concurrency at
≤8 for ≥128K. CI runs batch **1 and 32**.

This is not merely off-target, it is where the budget goes: **all five killed
points are conc 32**, 2 h each = **10 of the 18 h**, producing no result. The
requirements never ask for batch 32, and batch 16 — which they do ask for — is
capped at ≤32K precisely because 16×128K KV plus weights leaves no headroom.
CI is spending its entire long-context budget on an unrequested concurrency, in
a regime the requirements rule out.

### 3.3 The long-context batch-1 points are never reached

Required batch-1 ISLs: 128, 1024, 4096, 16384, 32768, 65536, 131072, 262144.
CI reaches 128, 1024, 4096, 16384 and dies during 32768. **65536, 131072 and
262144 are never measured.** Those are the sizes the use-case is about — the
workload basis puts Claude's P50 cached prefix at ~126K.

CI also spends time on three ISLs that are not required at all: 2048, 8192,
10000.

### 3.4 Warm-prefill / APC: 8 required measurements, 0 run

The requirements list automatic prefix caching as **REQUIRED**, state that
steady-state agentic latency is decode-bound *once the prefix is cached*, and
give an 8-row warm-TTFT table plus a named benchmark
(`--dataset-name prefix_repetition`) and a pass bar (≥50% E2EL reduction).

CI measures **cold TTFT only**. It therefore measures the one path the
requirements say APC removes from the steady state, and does not measure the
path that decides usability. The required speedups at the long end are 100x and
200x — this is the largest single gap in the whole comparison.

### 3.5 Required serving features are off or absent

| requirement | status in this port |
| --- | --- |
| Automatic prefix caching | **not enabled, not measured** |
| Chunked prefill | **disabled** — `generator_vllm` prefills one request per scheduler step for `model_type=qwen3_5` |
| Speculative decoding (MTP) | **not implemented** — `config.json` ships `mtp_num_hidden_layers: 1`, unused |
| max_model_len 262144 | supported |

Chunked prefill being off is not cosmetic: it is *why* the conc-32 long-ISL
points blow the 7200 s cap. With one request prefilled per step, 32 concurrent
requests serialize their prefills — CI's own numbers show mean TTFT of
1,292,573 ms (21.5 min) at ISL 2048 conc 32, and 643,445 ms at ISL 1024 conc 32.
The requirements call chunked prefill REQUIRED specifically because it "heavily
impacts user-observed TTFT p99, especially at batch 8".

### 3.6 Eval set

Four eval tasks are required. **The CI evals run executes one.** Checked
against the genuine eval run `34414429853`: the strings `terminal_bench`,
`swe_bench` and `livecodebench` appear **zero** times in its log, and
`Selected Tasks: ['r1_gpqa_diamond']` is the whole task list.

| required task | reference | pass bar (−5% rel) | in CI? |
| --- | ---: | ---: | --- |
| `r1_gpqa_diamond` | 89.2 | ≥84.7 | yes — measures **40**, fails |
| `terminal_bench_2_1` | 73.0 | ≥69.4 | **no — does not run** |
| `livecodebench` (v6) | 90.3 | ≥85.8 | **no — does not run** |
| `swe_bench_verified` | 61.7 | ≥58.6 | **no — does not run**; and the card's 61.7 is SWE-bench **Pro**, so the variant needs aligning before it can gate anything |

`doc/ci_dispatch_qb2` records that the eval *config* resolves
`r1_gpqa_diamond`, `terminal_bench_2_1` and `swe_bench_verified`. Resolving is
not running: this dispatch ran only the first. So three of the four accuracy
gates are unmeasured, and the one that does run fails by a wide margin
(40 against ≥84.7).

### 3.7 The decode target is a curve, not the number previously carried

Earlier work in this repo (and `doc/decode_perf`) measured against
`tput_user 41.0` from `model_performance_reference.json`, flagged there as
"ASSUMED, NOT VALIDATED" and extrapolated from Qwen3-32B on a t3k. Rev 0.11
supersedes it with an ISL-dependent curve — **50 t/s/u at 128/1K falling to 34
at 256K for batch 1** — derived from a stated roofline (27 GB FP8 weights, 2048
GB/s, 128 GB) and held at 45-66% of that ceiling. It also sets targets *without*
spec-decode credit, treating MTP as headroom.

## 4. Where we actually stand against the real targets

Measured on this branch (4x Blackhole p300c = the p300x2 QB2 config), all
64 layers, real weights:

| profile | required | measured | gap |
| --- | ---: | ---: | ---: |
| batch 1, decode t/s/u (ISL 128) | **50** | **20.59** (48.56 ms/token) | **2.4x short** |
| batch 1, TTFT cold (ISL 128) | **60 ms** | ~3600 ms warm (recorded) | **~60x short** |
| batch 8, decode t/s/u (ISL 4096) | **34** | not measured at that ISL | — |
| batch 16 | **32 / 24** | never measured | — |
| warm-prefill TTFT, any ISL | 60-300 ms | never measured | — |
| `r1_gpqa_diamond` | ≥84.7 | **40** | fails |
| 3 of 4 eval gates | measured | **not run at all** | — |

**The 2.68x in `doc/decode_perf` is a batch-32 result, and batch 32 is not a
required profile.** At batch 1 — the requirements' primary profile — the same
work is **1.03x** (19.96 → 20.59 t/s/u), because the fused KDA conv needs
`kernel * batch` tile-aligned and batch 1 keeps the composite path. So against
Rev 0.11 the decode work delivered almost nothing on the profile that matters
most, and the honest statement is that batch 1 is still 2.4x short of target.

Two things follow. First, the remaining decode levers in `doc/decode_perf`
(state traffic, one-matmul, ~13 ms at batch 32) are also batch-32-shaped and
will not move batch 1 either. Making batch 1 fast needs the composite conv path
fixed or the fused path generalized to `B=1` — which is a different piece of
work from anything done so far. Second, at batch 1 the step is 48.56 ms against
the requirements' own 13.2 ms roofline, so there is 3.7x of headroom there in
principle.

## 5. What to change in the benchmark configuration

In rough order of value per hour of runner time:

1. **Set OSL to 252.** Without this no CI number maps to a requirement.
2. **Replace concurrency 32 with 8 and 16.** Frees the 10 h currently burned on
   killed conc-32 points and starts measuring two required profiles.
3. **Add the `prefix_repetition` warm-prefill benchmark** at the eight required
   ISLs, concurrency 1 (and 8 for the cache-efficiency block). This is the
   largest missing requirement and it is a benchmark that already exists in
   vLLM.
4. **Extend batch-1 ISL coverage to 65536, 131072, 262144**, and drop 2048,
   8192, 10000 — which no requirement asks for.
5. **Enable APC and chunked prefill**, then re-check whether the long-ISL
   multi-concurrency points still hit the 7200 s cap. They likely will not:
   serialized per-request prefill is the direct cause.
6. **Make the other three eval tasks actually run** — `terminal_bench_2_1`,
   `livecodebench`, `swe_bench_verified` — and settle SWE-bench Pro vs Verified
   before gating on 61.7.
7. Only then is MTP worth measuring, as the requirements' stated headroom above
   the non-spec-decode targets.

Items 1, 2 and 4 are edits to the benchmark point list. Item 3 is a new
benchmark invocation. Items 5 and 7 are model/serving work, not configuration.
