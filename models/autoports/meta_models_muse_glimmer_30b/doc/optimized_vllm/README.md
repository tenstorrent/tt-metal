# Muse-Glimmer-30B — optimized vLLM serving

Starting point: the completed vLLM-integration stage (`doc/vllm_integration/`), serving
`meta-models/Muse-Glimmer-30B` through the shared TT vLLM path on a 4-die Blackhole
P300_X2 (mesh `1x4`, `FABRIC_1D_RING`) at the full advertised 131072-token context, with
on-device split sampling and a traced decode step. Precision policy unchanged:
`c14-attn4-cclbfp8-kv8`, read from `doc/datatype_sweep/selected_precision_config.json` on
every build.

**Read this first.** Serving decode entered this stage at 100 % of the standalone decoder,
so the only optimization target was TTFT and the only candidate was tracing the serving
prefill. That candidate is worth a measured **1.29x–1.34x on TTFT**, is token-identical to
the eager path in every offline comparison, and is **shipped off**, because capturing a
prefill trace was measured to change the output of *other* requests — in two distinct ways,
only one of which has a known mechanism. Four fixes were tried and measured insufficient, including shrinking the
bucket set to one and then widening its largest entry. What ships is therefore **no latency change**, a fully
characterised blocker with a reproducer, an interlock, three correctness fixes to the port,
and two shared evidence tools that were silently reporting the wrong thing — one of which
passed a completely corrupted server.

## Headline — primary single-user profile

Workload: **128-token prompt, 128 output tokens, 1 request, `--max-concurrency 1`,
greedy (`--temperature 0.0`), `ignore_eos`**, server at `--max-model-len 131072`,
`--max-num-seqs 32`, `sample_on_device_mode=all`, mesh `P300x2`. Every arm ran the
benchmark stage **six times back to back as the first traffic after a server start**; the
figures are the median of runs 4–6. Raw per-run JSON under `<arm>/run<N>/`, folded by
`bench/collect_metrics.py` into `metrics.json`.

| metric (128/128/1, greedy) | before | **after (shipped)** | delta |
|---|---|---|---|
| **TTFT** median = p99 | 81.48 ms | **77.42 ms** | −5.0 %, inside the run spread |
| **Decode t/s/u** (`1000 / mean_tpot_ms`) | 43.480 | **43.428** | −0.12 % (unchanged) |
| TPOT mean = p99 | 22.999 ms | 23.027 ms | +0.12 % |
| ITL p50 / p99 | 23.015 / 23.222 ms | 23.011 / 23.245 ms | — |
| Aggregate output throughput | 42.63 tok/s | 42.65 tok/s | +0.05 % |
| E2E latency median | 3002.6 ms | 3001.2 ms | −0.05 % |
| Completed | 1/1, 0 missing tokens | 1/1, 0 missing tokens | — |

**Nothing measurable moved, and this says so rather than dressing up noise.** TTFT is 5.0 %
lower with run ranges that overlap heavily (before 77.79–91.83 ms, after 76.64–87.32 ms),
and none of the shipped changes was expected to move it. The one that could have — the
slot-independent page table removing a per-slot `ttnn.slice` program compile — costs one
compile per slot amortised over 100 output tokens in the burst profile, below what this
harness resolves.

**Decode is unchanged, and that is the correct result.** Against
`doc/datatype_sweep/evidence_perf.json`'s standalone traced token-out decode for the same
128/128/1 shape and the same precision policy — **23.078 ms/token, 43.331 t/s/u** —
serving measures 23.027 ms/token, 43.428 t/s/u, i.e. **100.2 %** of the standalone rate.
Serving orchestration, sampling, token feedback, request handling and readback stay inside
the decoder's own run-to-run spread. The vLLM-integration stage had already measured the
same parity, which is why decode was not a target here: there was no serving-side decode
overhead to remove, and `$optimize`'s rule for the opposite case ("treat the gap as
orchestration overhead") had no gap to apply to.

## Secondary — CI serving-burst profile (capacity, not headline)

Workload: **100-token prompts, 100 output tokens, 32 requests, no explicit
`--max-concurrency`** (the vLLM-nightly shape), greedy, `ignore_eos`. Same six-run
protocol, same median-of-4–6.

| metric (100/100/32, greedy) | before | **after (shipped)** | delta |
|---|---|---|---|
| Aggregate output throughput | 721.88 tok/s | **717.56 tok/s** | −0.6 % (noise) |
| TTFT p50 / p99 | 2147.53 / 2148.76 ms | 2175.86 / 2177.19 ms | +1.3 % (noise) |
| TPOT mean | 23.039 ms | 23.046 ms | +0.03 % |
| Decode t/s/u from mean TPOT | 43.405 | 43.392 | −0.03 % |
| ITL p50 / p99 | 23.042 / 26.214 ms | 23.033 / 25.449 ms | — |
| E2E latency median | 4431.2 ms | 4457.8 ms | +0.6 % |
| Completed | 32/32, 0 missing tokens | 32/32, 0 missing tokens | — |

This is **not** the headline decode number: all 32 prompts are admitted as one burst, so
TTFT carries 32 queued prefills and TPOT sees burst admission. It is here for vLLM-nightly
parity and to show 32 concurrent sequences still serve.

## The optimization that works, is fast, and does not ship

Enabled with `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1`, everything else identical, same six-run
protocol:

| arm | buckets | primary TTFT | vs before | burst TTFT | burst throughput | primary decode t/s/u |
|---|---|---|---|---|---|---|
| `before/` | — | 81.48 ms | — | 2147.53 ms | 721.88 tok/s | 43.480 |
| `after/` (shipped) | none | 77.42 ms | −5.0 % | 2175.86 ms | 717.56 tok/s | 43.428 |
| `after_prefill_traced_1bucket/` | `[128]` | **62.97 ms** | **−22.7 % (1.29x)** | **1654.70 ms** | **812.10 tok/s (+12.5 %)** | 43.430 |
| `after_prefill_traced/` | 20 buckets | **60.66 ms** | **−25.5 % (1.34x)** | 1691.07 ms | 805.38 tok/s (+11.6 %) | 43.469 |

Both traced arms ran the full gate set — six benchmark runs, the whole sampling suite, both
qualitative arms, determinism — and both passed it, **with one caveat that matters**: the
interlock releases their prefill traces during the sampling step, and the sweep runs sampling
before the qualitative and determinism steps, so only their *benchmarks* were measured with a
trace resident. Traces-resident output evidence comes from `soak_traced_bucket/`.
Decode is untouched in every arm. The
TTFT run ranges do not overlap the before arm's (77.79–91.83 vs 59.37–70.45 for the
1-bucket arm), and per-user burst TPOT is identical, so the +12.5 % aggregate is entirely
the shorter prefill queue.

Those numbers are real. The next section is why they are not the default.

## What changed

### 1. The serving prefill can be traced, in declared buckets captured at warmup (off by default)

The capability existed before this stage: `GeneratorConfig.prefill_trace`, added by
optimized-full-model, which measured a warmed replay at **44.96 ms against 59.80 ms eager
(1.33x)**, **bit-identical** to it. It shipped **off**, for a stated reason — the graph is
keyed by the *padded* row count, so one trace serves one 32-row bucket, capture costs
~98 ms, and a generator cannot know whether its caller's prompt lengths bucket.

A server can. `MuseGlimmerForConditionalGeneration` declares its bucket set
(`PREFILL_TRACE_BUCKETS`) and captures it during warmup, so no request pays a capture. The
machinery, the bucket declaration, the capacity ladder and three acceptance tests all ship;
the *default* does not, for the reason below.

### 2. The prefill graph no longer bakes in the cache slot

The optimized-full-model form captured with the whole `[32, blocks]` page table and
`user_id=slot`, so the slot landed in a `ttnn.slice` *offset*, which is part of the program
hash — and the generator only offered the trace when `user_id == 0`. vLLM picks the slot,
so that form would have served one request in a 32-request burst.

`MuseGlimmerModel.page_table_row` returns the target slot's `[1, blocks_per_seq]` row and
the layer stack is driven with `user_id=0`. Prefill writes exactly one slot, and both
places the stack reads the table in prefill — the `paged_fill_cache` chunk row and the
chunked-SDPA prefix row — want that single row, so this computes the same thing. It also
helps the eager path: each of the 32 serving slots used to compile its own slice program,
so a request landing in slot 7 paid a program-cache miss a slot-0 warmup could not cover.
The per-request page-table staging drops from 32x2048 int32 to 1x2048.

The slot bound moved with it: the layer's `user_id >= max_batch_size` guard can no longer
see the caller's slot, so `_prefill_user` raises instead. Dropping it would have been
silent, because `normalize_page_table` aliases rows past the last private one, so an
out-of-range slot would have prefilled into another user's blocks.

### 3. Every sampling mode is warmed before serving

`warmup_model_decode` drives a penalised decode as well as a greedy one, in both warmup
phases, so the sampler's `(penalties, log_probs, force_argmax)` trace slots are all captured
before the first request. That removes a capture from whichever request first uses a
penalty, and removes one trace capture from the middle of a live process.

### 4. The interlock

`MuseGlimmerGenerator._guard_late_sampling_capture` releases the prefill traces **before**
the sampler allocates device buffers, whenever it is about to: a sampling mode with no
captured trace, or an explicit per-request seed (which makes the shared sampler bypass its
trace on purpose). One-way, logged as
`DEGRADED PATH prefill_traces_released_for_sampling_capture`, reported in
`capability_report()['prefill_traces_released_for_sampling']`, and covered by a host-level
acceptance test that also pins the four sampler attributes it reads.

It never fires in the shipped arm, which has no prefill traces to release. It fires **once
in each traced arm** (`after_prefill_traced_1bucket/`, `after_prefill_traced/`), during the
sampling suite's seeded tests — and `bench/run_arm.sh` runs `sampling` *before* `qualitative`,
`qualchat` and `determinism`, so **those traced arms' qualitative, determinism, cross-batch
and non-aligned results are eager-path results**, not traced ones. Their benchmarks ran
before the release, with the trace resident, which is exactly what the audit's
benchmark-window split exists to make checkable. Qualitative evidence *with a trace resident*
comes from `soak_traced_bucket/` instead, whose prompts are asserted to land in the bucket.

### 5. Two evidence tools that were reporting the wrong thing

* **`audit_serving.py` could never say clean.** It scanned the whole server log as one
  window, counting the plugin's by-design phase-1 untraced decode warmup as a serving
  degradation. Now split into warmup / benchmark / checks / shutdown windows, with `clean`
  judged on the window whose numbers are being reported and the by-design markers of the
  other windows listed with the reason each is expected. Offsets are recorded by
  `bench/run_arm.sh` when they happen; the log is sliced as **bytes**, because those offsets
  are `wc -c` and these logs are not pure ASCII.
* **`check_degenerate_output.py` passed a completely corrupted server**, and did not look at
  the chat arm at all. Three fixes:
  - it now measures `replacement_char_fraction` on the raw text (the duplication and looping
    metrics are computed over `\w+` words, and text made of U+FFFD has almost none, so a
    corrupted server scored 0.0 on both). Critical above 0.10, advisory above 0.02,
    calibrated **per completion** — the granularity it is applied at: every healthy
    completion in this stage's corpus measures exactly 0.0000 and the corrupted ones span
    0.187–0.617. An earlier revision used 0.25 taken from a per-artifact-*set* aggregate, at
    which the six *sampled* completions of a fully corrupted server (0.187–0.248) passed;
  - `discover()` now globs `qualitative_tt_chat.json` as well, so the prompt-correct chat arm
    `$qualitative-check` reads its verdict from is inside the gate for the first time;
  - deliberately-broken evidence is excluded by a `DEGENERATE_CHECK_EXCLUDE` marker file that
    the checker **reports** (`excluded: <artifact> [<marker>] <reason>`), instead of being
    renamed out of the glob. The artifacts keep their real names and stay readable, including
    the two sets that straddle the threshold. Naming a file explicitly on the command line
    still scans it, which is how the negative control is produced.

## The bug this stage found

**Enough resident prefill traces corrupt a live vLLM server, silently.** Anyone enabling
prefill tracing on any TT model needs this, which is why it is in the README.

### What it looks like

Served output decays into U+FFFD replacement characters partway through ordinary traffic.
Greedy output stays perfectly *deterministic* — the same prompt returns the same tokens — it
is simply wrong. The shared stage gate **passed it**; that hole is fixed above.

### The minimal reproducer

`traced_qualitative/vllm_qualitative_outputs.json` and
`soak_blocking/runner_qual1/vllm_qualitative_outputs.json`, both at **20 buckets**. Each is
preceded on its server by one clean 12-generation chat round, then its own p0–p3 measure
exactly 0.0000 and the decay starts at p4 — so onset is the **22nd generation**, identically
in both arms, with no seeds and the guard never firing. The p5 greedy corrupt string is
**byte-identical between the two servers**, so this is deterministic, not a drifting race.

### Bisection

| # | experiment | artifact | result |
|---|---|---|---|
| 1 | Is the traced prefill itself wrong? Eager vs traced on the real pinned prompts — one process, one build, one KV cache, one decode trace | `prefill_trace_bisect.json` | **No.** Token-identical on all three prompts, both arms coherent English |
| 2 | Is the server broken from its first request? Chat qualitative before any traffic, after a benchmark, after the sampling suite | `bisect_server/qualitative{1,2,3}/` | healthy, healthy, **corrupted** |
| 3 | Are the prefill traces causal? Same binary, tracing off, same sequence | `ctrl_notrace/qualitative{1,2}/` | **healthy both sides**, and the sampling suite reproduces the vLLM-integration stage's failure set (10 failed / 62 passed) |
| 4 | Which part of the suite does it? One test file at a time, pinned prompt after each | `corruption_localization_unguarded.json` | `test_config`, `test_build_logprobs_from_topk`, `test_logprobs` healthy; **`test_seeding_and_variety.py` corrupts** |
| 5 | Is it the *number* of traces? One bucket instead of twenty | `soak_1bucket/` | Clean — but **void**: its prompts pad to 32/64/96, so none of them took the traced path. Kept as the record of the mistake |
| 6 | Repeat (5) with prompts that actually land in the bucket | `soak_traced_bucket/` | **84 generations, all 0.0000, byte-stable across 14 rounds**, spanning an out-of-bucket qualitative round |
| 7 | Does one bucket change anything *outside* it? A probe series across several configurations | `prefill_trace_discriminators.json` | **Yes — see the next two sections.** Long eager prefills diverge with one bucket resident |

### Mechanism

ttnn states the constraint:

> Allocating device buffers is unsafe due to the existence of an active trace. These buffers
> may be corrupted once a trace is executed.
> … buffers allocated when a trace is active have to have a lifetime that ends before the
> trace is executed. — `tt_metal/impl/allocator/allocator.cpp:113-126`

A captured trace's intermediates are freed at `end_trace_capture`, but their addresses stay
baked into the replay, so anything the allocator later hands out from that range is
overwritten when the trace runs. The decode and sampling traces put a small, decode-shaped
range under that rule and this port has lived with it since optimized-full-model. Twenty
prefill traces put twenty 52-layer *prefill* working sets under it, and a vLLM server
allocates continuously from code this adapter does not own.

`test_seeding_and_variety` is the first *identified* trigger, because the shared sampler
deliberately bypasses its trace when a per-request seed is active ("run them directly so
trace replay cannot observe stale seed state", `models/common/sampling/generator.py`) and
allocates instead. It is not the only one — the decay reproduces with no seeds at all.

**This explains the 20-bucket decay and nothing else.** It does *not* explain the one-bucket
failure below, where no trace is ever replayed, and this report does not stretch it to. Two
failures, one confirmed mechanism, and one open question is the honest count.

### Two fixes that did not work, recorded as refuted

* **Warm every sampling mode at warmup**, so nothing is captured at runtime. `fixcheck/`:
  both modes captured at warmup, no runtime capture, **still corrupted**. Kept anyway —
  correct on its own terms, and it stops an ordinary penalised request from tripping the
  interlock — but it is not the fix.
* **Make the traced prefill replay blocking**, so the host cannot allocate the clone, the
  sampler's buffers or the readback while the replay is in flight. `soak_blocking/`: ~80 real
  requests, **corrupted inside the first sustained round**. Kept, because non-blocking bought
  nothing that queue order does not already give and the race it opened is real; but it is
  not the fix either.
* **Shrink the bucket set to one**, and then **widen the largest bucket while keeping the fast
  one** — the two configuration fixes the matrix in the next section rules out.

### The one-bucket configuration, and why it does not rescue this

The obvious response to "20 traces poison too much address range" is to keep one, and that
was tried properly.

The first attempt at qualifying it was void, and the round-2 stage review caught it:
`soak_1bucket/` drove four runner qualitative rounds and two chat rounds whose prompts are
7–79 tokens, i.e. padded 32/64/96, so with the bucket set to `[128]` **every one of those 84
completions ran the eager path**. It is kept as the record of the mistake.

`soak_traced_bucket/` is the corrected experiment. `bench/soak_traced_bucket.py` builds six
ordinary chat questions whose *rendered* length is 108–115 tokens, **asserts every one pads
to 128 before sending anything** (it exits 3 rather than produce evidence about a path the
prompts do not take), then serves them for 10 rounds, reads the text, and checks
`replacement_char_fraction` per completion and byte-stability against round 0 — the sharper
signal, because this corruption is deterministic. Then the runner's own out-of-bucket
qualitative arm runs on the same server, then 4 more rounds:

```text
60 generations (10 rounds) -> worst replacement 0.0000, all rounds byte-stable
   ... runner qualitative arm (out-of-bucket traffic) on the same server ...
24 generations (4 rounds)  -> worst replacement 0.0000, all rounds byte-stable
```

Plus 198 traced replays across `after_prefill_traced_1bucket/`'s six benchmark rounds, and
that arm's full gate set passing. **On everything that goes through the bucket, one trace is
clean.**

### And then it changes a request that does not

The adapter probe runs prompt lengths in one session and compares each against a reference:
the vLLM-integration stage's committed probe where it has that length, otherwise this stage's
own tracing-off run of the same length. Fifteen probes across six configurations are
tabulated in `prefill_trace_discriminators.json` and reproduced by
`bench/run_discriminators.sh`. Every request, by configuration:

| configuration | largest bucket | 37 | 100 | 128 | 1024 | 4097 | 8192 |
|---|---|---|---|---|---|---|---|
| tracing off | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1 bucket `[96]` | 96 | ✅ | ✅ | ✅ | — | ❌ | — |
| 1 bucket `[128]` | 128 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 1 bucket `[1024]` | 1024 | — | — | — | ✅ | ✅ | — |
| 2 buckets `[128,1024]` | 1024 | — | — | ✅ | ✅ | ✅ | ❌ |
| 20 buckets | 1024 | ✅ | ✅ | ✅ | — | ✅ | — |

**No traced configuration measured is correct at every length it was measured at**, and that
— not a bucket count — is why tracing ships off. Three of the five traced configurations have
a measured wrong length; the other two (`[1024]` alone, and the 20-bucket set) were not run at
8192, which is the matrix's main coverage limit and is recorded as such rather than read as a
pass. The failures are all on prompts the trace never serves: 4097 and 8192 take the eager
path in every row above.

**What is not measured**, stated so it is not mistaken for a result: bucket `[1024]` alone at
8192, and any bucket size between 128 and 1024. Both observed 8192 failures contain bucket
128, so "a large largest bucket is what helps" and "any small resident bucket poisons long
eager prefills" are not separated by this matrix. The decision does not rest on that cell —
`[128,1024]` failing 8192 settles it either way — but the gap is real.

What the series establishes:

* **The capture is sufficient; no replay is needed.** The `4097 ALONE` and `bucket [96]` runs
  contain no traced request at all — the bucket is captured at warmup and never replayed —
  and 4097 is still wrong. That rules out the "a buffer allocated under a live trace is
  overwritten when the trace runs" reading, which is what the wide-set decay looks like.
* **A larger largest bucket helps, and not enough.** Bucket 1024 makes 4097 correct, alone or
  as the top of `[128,1024]` or of the 20-bucket set. It does **not** make 8192 correct:
  `[128,1024]` fails 8192 byte-identically to `[128]`. So the effect tracks the size of the
  largest captured trace, but not in a way that any measured set makes safe.
* **`[128,1024]` was the configuration worth wanting.** Two traces, largest 1024, keeping the
  bucket the 1.29x was measured on. Round 5 of the stage review named it; it was measured; it
  fails 8192.
* **It is not an unwarmed shape.** 8192 *is* in `PREFILL_WARMUP_LENGTHS`, so its programs are
  compiled before any capture, and it diverges anyway.
* **It is not a bucket-value quirk.** 96 and 128 both do it.
* **The wide set is not the safe end either.** Twenty buckets get 4097 right — re-measured on
  the shipped revision after round 3 flagged the earlier datum as cross-revision — but they
  are the configuration whose *short*-prompt output decays after ~22 generations.
* **One negative result worth recording**, because it looks alarming and is not: the 1024-token
  probe's output is a 2-token cycle (`distinct_tokens 4`). That is a property of the synthetic
  `arange(1000, …)` prompt, not corruption — the tracing-off control produces it
  byte-identically. Round 5 caught that this had no control; it now does.

**The mechanism is not established.** The ttnn allocator-lifetime rule quoted above explains
the wide-set decay but not a capture-only failure with no replay in it, and nothing measured
explains why a larger captured trace fixes 4097 but not 8192. That is recorded as unexplained
rather than narrated away, and it is the first of the two upstream asks below.

### What ships

Tracing **off**. Not because nothing was measured — this is the most measured part of the
stage — but because **no traced configuration measured is correct at every length it was
measured at**, the failure is silent, and none of the stage's own gates caught the long-prompt
case: the short-prompt soaks never sent one, the server-side non-aligned check asserts only
that the request succeeded, and the degenerate-output gate never saw that text. Three of the
five traced sets have a measured wrong length; the other two were not run at 8192, which is a
coverage gap and not a pass. Tracing off is the only configuration measured at every length and clean
at all of them; `[1024]` alone also has no measured failure, but it was only run at two
lengths.

Everything else ships: the machinery, `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1` for a deployment
that has soaked its own prompt-length distribution *including lengths outside the buckets*,
the bucket declaration, the capacity ladder, the interlock, the acceptance tests, and this
write-up.

### The upstream ask

Two asks, in order of usefulness:

1. **Why does capturing a prefill trace change the result of a later, unrelated, eager
   prefill — and why does a *larger* captured trace make it stop?**
   `prefill_trace_discriminators.json` is a small, reproducible matrix — one build, one script
   (`bench/run_discriminators.sh`) — and this stage could not explain it.
   It is not the allocator-lifetime rule (no replay occurs), not an unwarmed shape (8192 is
   warmed), not bucket-value-specific (96 and 128 both do it), and it tracks the size of the
   largest captured trace rather than the number of them. Whoever owns mesh trace capture will
   recognise the shape of that faster than a model port can.
2. **A way to reserve a captured trace's freed intermediate range**, or to ask whether an
   allocation falls inside one. Without it, "capture several large traces in a long-lived
   serving process" is not expressible safely, which is what the 20-bucket decay is.

The cost of leaving both open, for this model, is 1.29x–1.34x of TTFT and +12 % of
serving-burst throughput.

## Status

| gate | result |
|---|---|
| Primary single-user benchmark, 6 runs | **pass**, 1/1 completed each run, 0 missing output tokens |
| CI serving-burst benchmark, 6 runs | **pass**, 32/32 completed each run, 0 missing output tokens |
| Plugin sampling suite, `--sampling-profile full` | 62 passed, 10 failed, 1 skipped — see *Sampling suite* |
| Qualitative, prompt-correct chat arm | **pass** — coherent, and character-identical to the standalone model over every character compared. `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json` reports `identical: false, first_divergence: 2` on all six prompts, and that is the OpenAI API stripping `<\|message\|>` — which the standalone text carries — so every id after position 1 shifts. With that one token removed the two are byte-identical over all 149 compared characters on every prompt, which is the comparison `determinism_vllm.json`'s `special_tokens_stripped` performs |
| Qualitative, runner raw-completion arm | **pass with a classified caveat** — **3 of its 12** completions contain a long verbatim loop (p0 sampled 0.53, p1 greedy 0.71, p2 greedy 0.94 coverage). Pre-existing and prompt-shaped, not stage-introduced: the vLLM-integration arm is also 3 of 12 and shares p1 and p2 at identical coverage, the **chat** arm this verdict is read from is 0 of 6, and so is the HF control. See *Qualitative* |
| Qualitative with the prefill trace resident (**not** the shipped path) | in-bucket prompts **pass** — `soak_traced_bucket/`, 84 generations, 0.0000, byte-stable across 14 rounds; an out-of-bucket 4097-token prompt **fails**, which is why tracing is off |
| Degenerate-output check, `--scope all` | **pass**, exit 0, `logs/degenerate_check_all.log` |
| Determinism run-to-run / cross-batch / standalone baseline | **pass** / **pass** (8 concurrent, 1 distinct output, equal to the single request) / **pass** (identical over the 79-char common prefix the check compares at `max_tokens 24`; the longer 127-token comparison is in `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`) |
| Non-aligned prompt lengths | **pass** — 9/9 (1, 37, 127, 129, 1023, 2049, 4097, 8193, 12345) |
| Served context vs `doc/context_contract.json` | **131072 = 131072**, no reduction |
| Fallback + process audit (benchmark window) | **clean**, `after/serving_audit.json` |
| Acceptance tests (29 selected) | **29 passed**, `logs/pytest_final.log` |
| Same 29 under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` | **29 passed, watcher clean** — 20 dumps, zero error-shaped lines (`logs/pytest_watcher.log`, `watcher/watcher_excerpt.log`) |

## Serving configuration

Byte-identical across every arm. The **only** difference between them is the code under test
and `MUSE_GLIMMER_VLLM_PREFILL_TRACE` / `..._BUCKETS`.

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/meta_models_muse_glimmer_30b \
  --hf-model meta-models/Muse-Glimmer-30B \
  --mesh-device P300x2 \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --sampling-profile full \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING",
                "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144,
                "trace_mode": "decode_only"}'
```

Wrapped as `bench/serve.sh`; arms are driven by `bench/run_arm.sh <arm> <steps>`:

```bash
# before: the vLLM-integration stage's committed code, restored with git stash
bash doc/optimized_vllm/bench/run_arm.sh before bench1,bench2,bench3,bench4,bench5,bench6
# after: this stage's shipped configuration (tracing off — needs no env variable)
bash doc/optimized_vllm/bench/run_arm.sh after \
    bench1,bench2,bench3,bench4,bench5,bench6,sampling,qualitative,qualchat,determinism
# the traced arms, measured and not shipped (1 bucket shown; the 20-bucket arm is the same
# with MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=32,64,...,512,640,768,896,1024)
MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
  bash doc/optimized_vllm/bench/run_arm.sh after_prefill_traced_1bucket \
    bench1,bench2,bench3,bench4,bench5,bench6,sampling,qualitative,qualchat,determinism
# the FIRST, VOID soak -- kept only as the record of the mistake: these prompts pad to
# 32/64/96, so with the bucket at 128 none of them took the traced path
MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
  bash doc/optimized_vllm/bench/run_arm.sh soak_1bucket \
    qualitativerep1,qualchatrep1,qualitativerep2,bench1,qualitativerep3,qualchatrep2,qualitativerep4
# the corrected in-bucket soak: prompts asserted to pad to 128 before anything is sent,
# 14 rounds spanning the runner's out-of-bucket qualitative arm
MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
  python doc/optimized_vllm/bench/soak_traced_bucket.py --server-url http://localhost:8000 \
    --rounds 10 --out doc/optimized_vllm/soak_traced_bucket/soak_traced_bucket.json
# the traced-vs-eager pair the 4097 finding started from
MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
  python doc/vllm_integration/bench/adapter_probe.py --prompt-lens 128,100,37,4097 \
    --decode-steps 8 --out doc/optimized_vllm/probe_repro_traced.json
MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 \
  python doc/vllm_integration/bench/adapter_probe.py --prompt-lens 128,100,37,4097 \
    --decode-steps 8 --out doc/optimized_vllm/probe_repro_eager.json
# and the discriminator probes that characterised it, each with what it separates
bash doc/optimized_vllm/bench/run_discriminators.sh
```

`trace_mode` stays `decode_only`. It is the *plugin's* knob for the plugin's own
prefill-trace hook, which this port does not use — the plugin calls
`warmup_model_prefill(enable_trace=True)` **before** the decode warmup
(`model_runner.py::warmup_model`), so capturing buckets there would let TTFT work compete
with the per-token path for the trace region. Keeping the TT config identical is also what
makes the before/after a comparison of code rather than of harnesses.

## Contract evidence — the measured path

From `probe_repro_eager.json` — the adapter driven through the TT plugin's exact call
sequence on the real 52-layer build **in the shipped configuration**, i.e. tracing off — and
`after/serving_audit.json`. `probe_full_prefill_traced.json` (20 buckets),
`probe_full_shipped.json` and `probe_repro_traced.json` (1 bucket) are the same probe with
tracing on, and are what the traced-path claims cite; note that `probe_full_shipped.json` is
misnamed, having been taken during the few hours when the default was briefly one bucket.

**Async decode.** `decode_forward(read_from_device=False)` returns a device-resident carrier,
`read_decode_output(async_read=True)` enqueues the minimal deferred read and returns its
event, `process_decode_output_host` does host formatting only. `supports_async_decode=True`
is claimed because that split is exercised: the probe runs every decode step through it, and
the vLLM-integration stage's `--async-scheduling` arm confirmed the plugin accepts the
capability and that overlapped output is byte-identical to the non-overlap arm. The decode
path is unchanged by this stage.

**Non-blocking traced decode.** `ttnn.execute_trace(..., cq_id=0, blocking=False)` for both
the model decode trace and the sampler's. Everything after it — the deferred read, the
event, the host format — is behind the async boundary. The *prefill* replay is blocking, on
purpose; see *Two fixes that did not work*.

**On-device sampling, no host fallback on the measured path.** `sample_on_device_mode=all`;
`force_argmax` is `False`; serving replays the full-model generator's canonical split
sampling with `tt_out_tok` pointing at the persistent decode token input. The benchmark
window of `after/serving_audit.json` contains **no** degraded markers — in particular no
`serving_full_logits_readback`, the marker the host-sampling route emits.

**Steady-state counters** from `probe_repro_eager.json`, the shipped configuration — 8
multi-slot decode steps over three concurrent rows, with deliberately wrong host
token/position values fed on every steady step:

```text
trace_replays 8   token_refreshes 1   position_refreshes 1
page_table_refreshes 1   synchronizations 0   readbacks 8
sampling_param_refreshes 1   sampling_param_reuses 7
```

One refresh each, for eight tokens, and zero synchronizations. The 16-step probes
(`probe_full_prefill_traced.json`, `probe_full_shipped.json`) show the same shape at
`trace_replays 16 / refreshes 1,1,1 / synchronizations 0 / readbacks 16 / 1 refresh + 15
reuses`; the counter that matters is that the refreshes stay at 1 however many steps run.

**Page-table refresh, changed and unchanged, read back off the device.** The page table is
the explicit exception to the stale-input rule — it changes when a sequence crosses a block
boundary, which has nothing to do with the sampled token — so it is compared every step.
Both halves are measured through the adapter rather than only at the generator:

```text
unchanged: 3 steps -> page_table_refreshes 0, device table byte-identical to before
changed  : 3 steps -> page_table_refreshes 1, device table matches the new host table
                       and differs from the old one
```

**Traced prefill, per request** (`probe_full_prefill_traced.json`): `trace_replays 1,
token_refreshes 1, page_table_refreshes 1, synchronizations 0` at the traced padded lengths,
`trace_replays 0` past the bucket bound.

**Bit-identity against the before code.** In the **shipped configuration**
(`probe_repro_eager.json`) every request — 128, 37, 4097 — and the three-slot multi-request
section reproduce `doc/vllm_integration/probe_full_fixed.json` exactly, which is the statement
that this stage's shipped changes are numerically inert.

With tracing **on**, the same is true of every *short* length and of the multi-request
section, including when the short prompts are served by a trace replay
(`probe_full_prefill_traced.json`, 20 buckets) — that is the strongest available statement
that the traced graph itself computes the right thing. It is **not** true of the long lengths
at one bucket: `probe_full_shipped.json` (misnamed — it was taken while the default was
briefly one bucket) diverges at 4097, and that divergence is this stage's central finding
rather than a caveat to this paragraph. The full matrix is
`prefill_trace_discriminators.json`.

## Qualitative

The verdict arm is the **prompt-correct chat arm** (`after/qualitative/`), because this
checkpoint has a chat template. It is coherent, on topic, and character-identical to the
standalone model over the full common prefix; 0 of its 6 completions contain a long verbatim
loop, and neither does the HF control.

The runner's **raw-completion arm** posts the bare prompt strings to a chat/instruct model,
which the vLLM-integration stage already labelled continuation stress coverage rather than a
verdict. It loops, and the numbers below are recomputed over each arm's *own* artifact by
`bench/`-free arithmetic recorded in `loop_classification.json`: for every completion, the
longest word block (4–80 words) that repeats at least twice, and the fraction of the
completion its non-overlapping repeats cover. Anything over 0.40 is listed.

| artifact | completions over 0.40 | which |
|---|---|---|
| `readiness_vllm/` runner arm (vLLM-integration stage) | **3 / 12** | p1 greedy 0.708, p2 greedy 0.938, p2 sampled 0.600 |
| `after/` runner arm (shipped) | **3 / 12** | p0 sampled 0.529, p1 greedy 0.708, p2 greedy 0.938 |
| `after_prefill_traced_1bucket/` runner arm (1 bucket, not shipped) | 3 / 12 | p1 greedy 0.708, p2 greedy 0.938, and **p0 sampled 1.000** — a 6-word block repeated 32 times, the worst single completion in the corpus |
| `after_prefill_traced/` runner arm (20 buckets, not shipped) | 5 / 12 | p0 sampled 0.942, p1 greedy 0.708, p2 greedy 0.938, p2 sampled 0.741, p4 sampled 0.982 |
| `after/qualitative/qualitative_tt_chat.json` chat arm (**the verdict**) | **0 / 6** | — |
| `doc/full_model/qualitative/qualitative_hf_chat.json` HF control | **0 / 6** | — |

So the runner arm loops on 3 of 12 in this stage and 3 of 12 in the previous one, sharing p1
greedy (an 80-word answer restated verbatim) and p2 greedy (a 33-word story sentence, six
times) at *identical* coverage — pre-existing and prompt-shaped, not stage-introduced. The arm
the verdict is read from has none, and neither does the HF control on the same prompts.

The two non-shipped traced arms are in the table for completeness and **not** as evidence
about tracing: their qualitative steps ran *after* the interlock released their prefill traces
(the sweep runs `sampling` first), so those completions came off the eager path. The
differences between the arms are all in *sampled* completions — including the 1.000 one — i.e.
run-to-run draws on an arm that feeds bare prompts to a chat model. The one thing they do show
is limitation 9 at its worst: a completion that is a single 6-word block repeated 32 times
passes the shared gate, because `replacement_char_fraction` is 0.0000 and
`trigram_loop_fraction` measures trigrams.

The shared gate does not catch these, and that is a real gap in it rather than a judgement
about this model: `trigram_loop_fraction` measures the coverage of the single most common
*trigram*, so a completion repeating a 33-word sentence scores about 3/33. It is recorded in
`work_log.md` §7 as the metric's known blind spot; fixing it is a shared-checker change with
calibration implications across every model, and this stage did not make it.

## Sampling suite — 62 passed, 10 failed, 1 skipped

`after/sampling_tests.log`. Member by member, the shipped arm's failures:

```text
test_host_only_params.py::TestHostOnlyParameters::test_allowed_token_ids
test_tt_penalties.py::TestPresencePenalty::test_different_presence_penalties
test_tt_penalties.py::TestPresencePenalty::test_presence_penalty_mixed_batch
test_seeding_and_variety.py::TestSeedingAndVariety::test_seeding
test_seeding_and_variety.py::TestSeedingAndVariety::test_same_seeds_reproduce_across_batches
test_seeding_and_variety.py::TestSeedingAndVariety::test_specific_seed_reproducible[0]
test_seeding_and_variety.py::TestSeedingAndVariety::test_uniform_seed_deterministic[10-0]
test_seeding_and_variety.py::TestSeedingAndVariety::test_uniform_seed_deterministic[10-1]
test_seeding_and_variety.py::TestSeedingAndVariety::test_uniform_seed_deterministic[32-0]
test_seeding_and_variety.py::TestSeedingAndVariety::test_uniform_seed_deterministic[32-1]
```

Seven are the seeded-reproducibility-at-batch class the vLLM-integration stage documented;
the other three (`test_allowed_token_ids`, the two presence-penalty tests) are the three that
stage resolved by measurement without a code change, and that analysis is unchanged.

**The class has a floating member, so the set is not literally identical to the previous
stage's, and saying "the baseline set exactly" would be wrong.** The previous stage's seventh
reproducibility failure was
`test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`; the shipped arm
fails `test_specific_seed_reproducible[0]` instead. Both assert seeded reproducibility
across a `max_batch_size` concurrent batch, so both are the same class, and which one trips
varies run to run — measured, not assumed:

| run | seventh (and any eighth) member |
|---|---|
| `ctrl_notrace/` (this binary, tracing off) | `test_mixed_params_batch` |
| `sampling_variance/sampling1` (same server as ↓) | `test_mixed_params_batch` |
| `sampling_variance/sampling2` (same server as ↑) | `test_mixed_params_batch` **and** `test_specific_seed_reproducible[999]` — 11 failures |
| `after_prefill_traced/` (20 buckets) | `test_mixed_params_batch` **and** `test_specific_seed_reproducible[999]` |
| `after/` (shipped, tracing off) | `test_specific_seed_reproducible[0]` |
| `after_prefill_traced_1bucket/` | `test_specific_seed_reproducible[42]` |

`sampling_variance/` is the controlled version of that: the suite run **twice against one
server**, failing 10 then 11 with no code, config or server change between them.

Batch-1 seeding is reproducible, `test_top1_is_greedy`, `test_topk` and
`test_different_seeds_produce_different_outputs` pass, and correctness, logprobs, crash-free
serving and qualitative output all pass — the condition `$vllm-integration` attaches to
classifying these separately. Cross-request contamination is ruled out independently by the
cross-batch-position check in `after/determinism_vllm.json`: 8 concurrent copies of a prompt
return one distinct completion, equal to the single-request one.

## Rejected and deferred

| candidate | verdict |
|---|---|
| Any traced bucket set | **Rejected, five traced bucket sets measured against a tracing-off control.** Wide sets decay short-prompt output by the 22nd generation; narrow sets change long eager prompts. `[128,1024]` — two traces, largest 1024, keeping the fast bucket — was the one worth wanting and it fails at 8192. No traced set measured is correct at every length it was measured at, and the coverage limits are stated with the matrix. |
| Trace the 8192-row prefill bucket | **Rejected, measured.** What a trace removes is host dispatch, so the win falls with prompt length: 1.33x at 128 padded rows, **1.00x at 8192** (`doc/optimized_full_model/prefill_trace_probe.json`, `prefill_trace_probe_8192.json`). |
| Keep tracing by fixing the hazard | **Four fixes measured insufficient**: warm every sampling mode at warmup, make the traced replay blocking, shrink the bucket set to one, widen the largest bucket to 1024 while keeping the fast one. A fifth — allocating a ballast buffer over the freed range immediately after capture — was **not attempted**: it depends on first-fit allocator behaviour this stage cannot verify, and it would compete for DRAM with the eager prefill working set that long prompts still need. Recorded as the untried option, not as a blocker. `$autofix` was not invoked either, and `work_log.md` §8d says why: the investigation never stalled — each round produced a cheap decisive experiment — and what is missing at the end is device-level visibility rather than a debugging pass. |
| Plugin `trace_mode: all` | **Rejected.** Its warmup captures prefill before decode, letting a TTFT optimization compete with the per-token path for the trace region — and it would have changed the TT config between arms. |
| Retune decode | **Not attempted, deliberately.** Serving decode was at parity with the standalone decoder before this stage and is after. Decoder-internal work belongs to `$optimize` on the decoder and the datatype frontier to `$datatype-sweep`, both complete. |
| Raise `KV_CACHE_TOKEN_BUDGET` from 16416 blocks toward the measured-feasible 28672 | **Deferred, named.** `doc/vllm_integration/kv_budget_probe.json` proves 28672 feasible and the previous stage handed the change here. It is a *capacity* change: it moves no metric this stage measures, and it would spend the DRAM margin that absorbs allocator fragmentation across a long-lived serving process — which this stage has now shown matters more than it looked. |
| Profiler evidence (Tracy / `tt-perf-report` / device profiler) | **Not collected, by instruction.** `$optimize` and `$vllm-integration` both forbid it in vLLM stages; the device-time and roofline fields in `perf_summary.json` are `null` with that reason. Device-op context is carried from `doc/optimized_full_model/`. |

## `$optimize` checklist, mapped

The items that apply to a vLLM serving path — the skill scopes the decoder/module items out
of this stage and forbids the profiler-based ones here:

| item | where |
|---|---|
| Decode path fully traced, no host fallbacks | `probe_repro_eager.json` steady-state counters (shipped configuration); `after/serving_audit.json` benchmark window |
| Batch capability preserved to 32 | both benchmark profiles at `--max-num-seqs 32`, burst 32/32 completed; probe multi-request rows distinct |
| Same-harness before/after, primary and CI burst | `metrics.json`, six runs per arm per profile |
| Async split exercised, non-blocking decode replay | *Contract evidence* |
| On-device sampling, no host argmax or full-logits readback | *Contract evidence*; `force_argmax` False |
| Persistent trace inputs, refresh only on scheduler change | steady-state counters; page-table changed/unchanged |
| Runtime fallback audit clean | `after/serving_audit.json` |
| Context contract preserved | 131072 served; `.agents/scripts/check_context_contract.py` passes |
| Non-aligned lengths preserved | 9/9 in `after/determinism_vllm.json` |
| `$qualitative-check` after the optimization | `after/qualitative/` (shipped), plus the traces-resident arm in `soak_traced_bucket/` |
| Watcher clean | `logs/pytest_watcher.log`, `watcher/watcher_excerpt.log` |
| Rejected options recorded with evidence | *Rejected and deferred*, *The bug this stage found* |
| Performance accounting, `perf_summary.json` | written; device-time/roofline `null` with the no-profiler reason |
| Tracy / `tt-perf-report` / device profiler | **intentionally not collected** |

## Artifacts

| what | path |
|---|---|
| metrics for **every** arm, per run and folded, plus the before/after deltas | `metrics.json`, `bench/collect_metrics.py` |
| performance accounting | `perf_summary.json` |
| before arm (committed vLLM-integration code), 6 runs | `before/`, plus a 3-run first sweep in `before_sweep0/` |
| after arm, shipped configuration (tracing off), 6 runs + all gates | `after/` |
| traced arms, 6 runs + all gates each: 1 bucket, 20 buckets | `after_prefill_traced_1bucket/`, `after_prefill_traced/` |
| in-bucket soak, prompts asserted inside the traced bucket | `soak_traced_bucket/`, `bench/soak_traced_bucket.py` |
| driver logs | `logs/` — note that the shipped `after/` arm's own driver logs were overwritten by the 1-bucket arm, which ran under the same arm name before the rename; those logs are relabelled `after_prefill_traced_1bucket_*` with a header saying so, and the shipped arm's own artifacts are intact |
| the first, void soak (prompts outside the bucket) | `soak_1bucket/` |
| adapter probe, 52 layers, **shipped default** (tracing off) | `probe_repro_eager.json` |
| the prefill-trace discriminator matrix — every probe request, with the largest resident bucket | **`prefill_trace_discriminators.json`**; reproduced by `bench/run_discriminators.sh`, which carries 13 invocations covering 13 of the 15 matrix probes, and names the two 16-step probes it deliberately does not re-run |
| its inputs | `probe_repro_{traced,eager}.json`, `probe_full_shipped.json` (misnamed: taken while the default was briefly 1 bucket), `probe_full_prefill_traced.json`, and `probe_disc_{20bucket,bucket96,bucket1024,bucket128_1024,4097only_traced,4097only_eager,8192_traced,8192_eager,8192_bucket128_1024,1024_eager,1024_bucket128}.json` |
| long-verbatim-loop classification, every arm | `loop_classification.json` |
| trace-region capacity ladder | `probe_trace_capacity.json` |
| eager-vs-traced prefill on real prompts | `prefill_trace_bisect.json`, `bench/prefill_trace_bisect.py` |
| corruption bisection (in-server) | `bisect_server/`, `ctrl_notrace/`, `fixcheck/`, `soak_blocking/`, `traced_qualitative/` |
| corruption localization, unguarded / guarded | `corruption_localization_unguarded.json`, `corruption_localization.json`, `bench/localize_corruption.py` |
| sampling-suite variance, one server, two runs | `sampling_variance/` |
| degenerate-output check + negative control | `logs/degenerate_check_all.log`, `logs/degenerate_check_negative_control.log` |
| acceptance tests, plain and watcher | `logs/pytest_final.log`, `logs/pytest_watcher.log`, `watcher/watcher_excerpt.log` |
| stage narrative | `work_log.md` |
| independent stage reviews, eight rounds | `stage_review.md`, `stage_review_round2.md` … `stage_review_round8.md`; the sequence is summarised in `work_log.md` §8c |

Raw `server.log` files (50–130 MB each) are re-ignored by `.gitignore`; each arm commits
`<arm>/server_excerpt.log`. The before arm's raw log was overwritten by an accidental
relaunch after its sweep completed; its excerpt and all six benchmark results predate that
and are intact, the stub is renamed
`before/server/server_STUB_FROM_ACCIDENTAL_RELAUNCH.log`, and `before/serving_audit.json` is
the audit of the **real** log, restored from `logs/before_audit.log` with its provenance
recorded in the file.

## Limitations

1. **The one available TTFT optimization does not ship.** Traced serving prefill is
   implemented, tested, token-identical offline and worth 1.29x–1.34x, and it is off because
   capturing a prefill trace was measured to change the output of requests outside the traced
   buckets. Four fixes were measured insufficient. See *The bug this stage found*; the
   underlying constraint is ttnn's, and it is worth raising upstream.
2. **`_guard_late_sampling_capture` is a partial interlock**, and it is not what makes the
   shipped configuration safe — the bucket count is. It has exactly one call site,
   `_decode_submit_traced`, so it covers the *decode* sampler's capture and eager-seed paths
   and nothing else. In particular the traced prefill path allocates twice per request that
   the guard never sees: `_sample_eager`'s sampler intermediates and the `ttnn.clone` of the
   trace's persistent logits. Those are legal under the allocator rule rather than
   unguarded-and-lucky — both are allocated *after* the blocking replay has completed and
   both are freed before the next one, so their lifetimes end before the trace runs again,
   which is precisely what the rule asks and precisely why that replay is blocking. Long-lived
   allocations are the dangerous ones, and the adapter makes none per request.
   When the guard does fire, that server's TTFT reverts to the eager figure for the rest of
   its life — which is what happens in the sampling stage of both *traced* arms
   (`after_prefill_traced_1bucket/`, `after_prefill_traced/`). It never fires in the shipped
   `after/` arm, which has no prefill traces to release.
3. **Prefix caching is off** and declared off, unchanged from vLLM integration.
4. **Uniform KV-cache spec**; sliding-window layers are allocated full-attention sized
   blocks. Unchanged from vLLM integration; the blocker is vLLM's page-table zero-padding,
   described there.
5. **Seeded reproducibility at batch > 1** — unchanged from vLLM integration, and now measured
   twice against one server to show which member of the class trips is a run-to-run draw.
6. **Watcher evidence excludes ETH cores.** `TT_METAL_WATCHER=10` alone aborts every test at
   `TT_FATAL: Program size (28656) too large for kernel config buffer (25600) on ACTIVE_ETH`
   before any model code runs — the signature `$optimize` names, with
   `TT_METAL_WATCHER_DISABLE_ETH=1` as the prescribed retry.
7. **No device-time or roofline term was measured here.** The roofline in `perf_summary.json`
   is carried from optimized-full-model and labelled an upper bound, because it was computed
   under that stage's precision policy and the shipped policy moves strictly fewer bytes.
8. **The shipped default still carries the decode and sampling traces**, which are under the
   same allocator rule and have been since optimized-full-model. Nothing in this stage
   suggests they are unsafe — they are a small, decode-shaped address range and the port has
   served on them throughout — but this stage has demonstrated that the failure mode is
   silent, which is why the new `replacement_char_fraction` gate exists.
9. **The shared long-loop metric is blind to long-period verbatim repetition**, which this
   stage's own runner raw-completion arm exhibits on **3 of 12** completions — identically to
   the previous stage's, and absent from the chat verdict arm and the HF control. See
   *Qualitative* and `loop_classification.json`.
