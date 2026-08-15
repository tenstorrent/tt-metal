# Optimized vLLM serving — work log

Model: `meta-models/Muse-Glimmer-30B`
Autoport: `models/autoports/meta_models_muse_glimmer_30b`
Stage input: the completed vLLM-integration stage (`doc/vllm_integration/`) and the
selected datatype-sweep policy `c14-attn4-cclbfp8-kv8`.
Device: 4-die Blackhole P300_X2, mesh `(1, 4)`, `FABRIC_1D_RING`, healthy at stage start
(`timeout 60 tt-smi -ls --local` → 4 Blackhole p300c boards, no leftover vLLM/EngineCore
processes).

The result and the reasoning are in `README.md`. This is the narrative: what was
measured, in what order, what was wrong, and what was thrown away.

---

## 1. Where the stage started, and what was left to optimize

The vLLM-integration stage closed with serving decode at **100.1 %** of the standalone
decoder's token-out rate for the same 128/128/1 shape. That is not a number with headroom
in it: there was no serving-side decode overhead to remove, and `$optimize`'s own rule for
this case — *"When direct traced generator decode is already fast but vLLM/serving decode
is slower, treat the gap as orchestration overhead"* — did not apply, because there was no
gap.

What that stage did leave, in writing, was TTFT:

> **Prefill is eager.** Traced prefill exists (`GeneratorConfig.prefill_trace`) but is
> keyed by padded prompt length and off by default, so TTFT carries host dispatch. It is
> worth 1.33x at 128 rows and 1.00x at 8192; a serving deployment with bucketed prompt
> lengths should turn it on. — `doc/vllm_integration/README.md`, limitation 2

and the optimized-full-model stage had already measured why prefill is where the time is:
batch-1 prefill on this mesh is **host-issue bound**, 4122 ttnn dispatches at 9–60 µs of
issue each, 54.9 ms of issue against 55.1 ms to drain, with no collective implementation
or persistent-buffer variant moving the per-call cost across 12 arms
(`doc/optimized_full_model/ccl_host_probe_bfp8.json`). Tracing is the only mechanism that
removes host issue.

So the stage plan was: take the existing prefill-trace capability, make it usable from a
server, and prove decode did not move. The first half of that worked and is measured; the
second half held; and the candidate still does not ship, for the reason in §6.

## 2. Before arm

Ran first, on the vLLM-integration stage's **committed** code, restored with
`git stash push` over the six files this stage touches. Same harness, same TT config,
same `--max-num-seqs 32 --max-model-len 131072`, same greedy benchmark.

The benchmark stage was run **six times back to back as the first traffic after the
server start**, and that was not padding. The first three runs gave TTFT 91.83 → 81.99 →
81.22 ms: there is a warm-up curve, and a single sample per arm would have compared two
different points on it. Six runs per arm, in the same position, is the protocol both arms
use; the reported figure is the median of runs 4–6.

```
before  TTFT ms : 91.83  81.99  81.22  81.48  82.12  77.79     (warm median 81.48)
before  t/s/u   : 43.42  43.46  43.48  43.49  43.48  43.42     (warm median 43.480)
```

A first exploratory sweep of three runs (`before_sweep0/`) is kept; it is the same
configuration measured in a separate process and agrees (81.5 warm).

*Process note.* The before arm's raw `server.log` was overwritten by an accidental second
launch a few minutes after its sweep had completed. The six benchmark JSON files and
`before/server_excerpt.log` predate that and are intact. The **audit did not**: an earlier
revision of this stage committed a `before/serving_audit.json` computed against the 21 KB
stub the relaunch left behind, which reported `clean: true` about a different, aborted
process. The stage review caught it. `before/serving_audit.json` is now the audit of the
real 397 KB log, recovered from this tool's own stdout in `logs/before_audit.log`, with its
provenance recorded inside the file; the stub is renamed
`before/server/server_STUB_FROM_ACCIDENTAL_RELAUNCH.log` and `before/server_log_size.txt`
says which is which.

## 3. The three changes

### 3.1 The page table is a row, not a table (`tt/model.py`, `tt/generator.py`)

Prefill writes exactly one cache slot. Both places the layer stack reads the page table
in prefill — `_chunk_page_table` for `paged_fill_cache`, and `_page_table_row` for the
chunked-SDPA prefix on continuation chunks — want that one row. The old code handed the
stack the whole `[32, blocks]` table with `user_id=slot`, which turns the slot into a
`ttnn.slice` *offset*; slice offsets are baked into the program hash.

Two consequences, and the second is why the trace mattered:

* each of the 32 serving slots compiled its own slice program, so a request landing in
  slot 7 paid a program-cache miss that a slot-0 warmup could not cover;
* a prefill trace captured this way is a **slot-0 trace**, which is why the
  optimized-full-model form gated itself on `user_id == 0`.

`MuseGlimmerModel.page_table_row(page_table, user_id)` returns the slot's
`[1, blocks_per_seq]` row; `page_table_row_to_device` replicates it; `_prefill_user`
drives the stack with `user_id=0`. With a 1-row table and `user_id=0`,
`_page_table_row` returns the tensor itself for the SDPA prefix case and a
bucket-constant slice for the fill case.

The slot bound had to move with it: the layer's `user_id >= max_batch_size` guard can no
longer see the caller's slot, so `_prefill_user` raises instead. Dropping it would have
been silent rather than loud — `normalize_page_table` aliases rows past the last private
one, so an out-of-range slot would have prefilled into another user's blocks.

### 3.2 Serving captures its prefill buckets at warmup (`tt/generator_vllm.py`)

`GeneratorConfig.prefill_trace` stays off by default, because the generator's reasoning
for that default is still right. What changed is that the *serving adapter* knows its
bucket set, so it can turn it on with `MuseGlimmerGenerator.enable_prefill_trace(...)` and
capture every bucket during warmup — no request pays a capture, and the bucket set is
declared rather than discovered.

Two ordering decisions:

* **after the decode trace, not before.** The plugin's two-phase warmup calls
  `warmup_model_prefill` before `warmup_model_decode` (`model_runner.py::warmup_model`),
  so capturing there would let a TTFT optimization compete with the per-token path for
  the trace region. The buckets are captured at the end of the `enable_trace=True` decode
  warmup instead. This is also why `trace_mode` stays `decode_only`: that knob selects the
  *plugin's* prefill hook, which this port does not use, and changing it would have moved
  the TT config between the arms.
* **ascending, stopping at the first failure**, so if the region does run out it is the
  widest, least valuable bucket that is lost.

### 3.3 Traced prefill replay: non-blocking, then blocking again (`tt/generator.py`)

Shipped `blocking=False` first, on the reasoning that the clone, the sampler and the
readback are enqueued on cq0 behind the replay so queue order already orders them, and the
blocking form only added a host wait. Queue order does order the *reads*; what it does not
order is the **allocations** those consumers make, and ttnn requires a buffer allocated
under a live trace to die before that trace runs. It is back to `blocking=True`, with the
reasoning in the code. That change did not fix the corruption either (§6) — the race it
closes is real but is not the only one.

## 4. Choosing the bucket set — measured, not picked

A trace is keyed by the exact padded row count (the graph slices the last 32-row tile), so
covering every 32-row bucket is the only way the optimization reaches arbitrary prompt
lengths rather than only the benchmark's 128.

`bench/localize_corruption.py`'s sibling probe, `probe_trace_capacity.json`, walks a
deliberately excessive list — every 32-row bucket to 2048, 64 candidates — against the
shipped 400 MB `trace_region_size` on the real 52-layer build with the decode and sampling
traces already resident:

```
resident: 32 … 896            (28 traces, 0 failures)
bucket 928: TT_FATAL @ mesh_trace.cpp:81
            mesh_cq.device()->get_trace_buffers_size() <= trace_region_size
            -> capture disabled, remaining buckets serve eagerly, warmup continues
```

That is both the capacity number and an end-to-end exercise of the graceful-stop path, and
it is why the stage first shipped 20 buckets: 71 % of a proven-feasible 28. 8192 was
deliberately absent, because the win falls to 1.00x there
(`doc/optimized_full_model/prefill_trace_probe_8192.json`).

**That reasoning was measuring the wrong resource.** Trace-region capacity turned out not to
be the binding constraint at all -- §6 is what is -- and the shipped list is one bucket. The
capacity ladder is kept because it is still the right answer to "how many will fit", and
because a deployment widening the set with `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS` needs
both bounds.

## 5. Correctness before performance

Before any serving arm, three checks on the reduced and full stacks:

* **Reduced-stack acceptance tests.** The 20 existing prefill/page-table/trace tests pass
  unchanged. Three new ones were added:
  `test_page_table_row_is_the_slot_row_and_bounds_are_checked` (host-only),
  `test_prefill_trace_serves_every_cache_slot` (four prompts into four slots, eager arm
  against traced arm, compared on a **decode** step rather than prefill logits — the same
  prompt prefilled into any slot returns the same logits whatever the page table says, so
  only a decode can tell a correctly-routed K/V write from one that landed in another
  slot's blocks), and `test_prefill_trace_enable_seam_and_width_bound`.
* **52-layer adapter probe.** `probe_repro_eager.json` (the shipped configuration) reproduces
  the vLLM-integration probe's **identical token sequences** for every shared prompt length and
  for the three-slot multi-request section. `probe_full_prefill_traced.json` (20 buckets) does
  too. `probe_full_shipped.json` is misnamed -- it was taken while the default was briefly one
  bucket -- and its 4097 request does *not* match, which is §6's finding rather than a caveat
  here.
* **Eager vs traced on the real pinned prompts.** `prefill_trace_bisect.json`: one
  process, one build, one KV cache, one decode trace; prompts prefilled eagerly, then the
  buckets captured and the same prompts prefilled again into different slots. Token-
  identical, both arms coherent English.

## 6. The traced arms, the failure they exposed, and why nothing traced ships

The traced arm (`after_prefill_traced/`) produced exactly the TTFT win the plan predicted
— 81.48 → 60.66 ms, 1.34x, with decode untouched — and then the plugin sampling suite ran,
and every completion after it came back as replacement characters. `README.md` → *The bug
this stage found* is the write-up; this is the order the experiments actually ran in,
because the order is the argument:

1. `prefill_trace_bisect.json` — **the traced prefill is not wrong.** Rules out the
   obvious suspect before spending a server on it.
2. `bisect_server/` — chat qualitative before any traffic (healthy), after a benchmark
   round (healthy), after the sampling suite (**corrupted**). So the server is not born
   broken and the benchmark does not break it.
3. `ctrl_notrace/` — same binary, `MUSE_GLIMMER_VLLM_PREFILL_TRACE=0`, same sequence:
   **healthy on both sides**, and the sampling suite reproduces the vLLM-integration
   stage's failure set *exactly* (10 failed / 62 passed). Two things at once: the prefill
   traces are causal, and the other two changes are not.
4. `corruption_localization_unguarded.json` — one test file at a time, pinned prompt after
   each. `test_config`, `test_build_logprobs_from_topk`, `test_logprobs` healthy;
   **`test_seeding_and_variety.py` corrupts.**

**A fix that did not work.** Between (2) and (4) the theory was the lazily-captured
penalised sampling trace: the sampler keys a trace per
`(penalties, log_probs, force_argmax)` and captures it on the first request that needs it,
which is *after* warmup. `warmup_model_decode` was changed to warm both reachable modes.
`fixcheck/` says it did not fix it — both modes captured at warmup, no runtime capture,
still corrupted. The warmup change is kept because it is correct on its own terms and it
keeps an ordinary penalised request from tripping the interlock below, but it is recorded
here as a refuted hypothesis rather than folded into the fix.

**The mechanism**, once (4) named the file, is in ttnn's own words
(`tt_metal/impl/allocator/allocator.cpp:113-126`): buffers allocated while a trace is
active must have a lifetime that ends before that trace executes, because a captured
trace's intermediates are freed at `end_trace_capture` while their addresses stay baked
into the replay. `test_seeding_and_variety` is the first file to send explicit per-request
seeds, and the shared sampler bypasses its trace when a request seed is active, allocating
instead. With only the decode and sampling traces resident the poisoned range is
decode-shaped and this port had lived with it since optimized-full-model; with 20 resident
prefill traces it is a 52-layer prefill working set.

**The fix** is `_guard_late_sampling_capture`: ask, before every traced decode submit,
whether the sampler is about to allocate — a missing trace for the current mode, or an
active request seed — and if it is, release the prefill traces *first*. One-way, logged as
`DEGRADED PATH prefill_traces_released_for_sampling_capture`, surfaced in
`capability_report()`. Verified by re-running (4) with it in place:
`corruption_localization.json`, all eight files, model healthy after every one,
guard fired exactly once at `test_seeding_and_variety.py`.

**And the interlock was not enough.** With it in place, a soak — `soak_blocking/`, three
rounds of chat qualitative plus the runner's 12-request raw arm plus a benchmark, ~80 real
requests, no seeds, guard never fired — corrupted anyway, `replacement_char_fraction`
rising 0.418 → 0.617 from the first sustained round. A second fix attempt, making the
traced prefill replay blocking so the host cannot allocate the clone, the sampler's
buffers or the readback while the replay is in flight, is what that soak was testing; it
did not hold either. Both attempts are kept in the code because each is correct on its own
terms, and both are recorded as refuted rather than folded into a success.

So the finding is broader than the seeded path: a serving process allocates continuously,
from code this adapter does not own, and the ttnn rule cannot be honoured for twenty
52-layer prefill working sets' worth of freed-but-baked addresses.

**At which point the stage was about to ship the optimization off, and the stage review
stopped it.** Its P1 was that the *reduced* bucket configuration had never been measured —
the env knob to do it (`MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS`) already existed and had
been used only to sweep *upward* to 64 — and that `traced_qualitative/` contained a
deterministic reproducer that needs only 22 generations that had gone unused. Both were correct.

`soak_1bucket/` was that experiment -- one bucket (128), four runner qualitative rounds, two
prompt-correct chat rounds and a benchmark round through one server, 84 completions all at
`replacement_char_fraction` 0.0000 -- and it was **void**, which round 2 of the stage review
caught: every one of those prompts is 7-79 tokens, i.e. padded 32/64/96, so with the bucket at
128 none of them took the traced path at all. It is kept only as the record of the mistake.
`soak_traced_bucket/` is the corrected experiment (§6, and `bench/soak_traced_bucket.py`
refuses to run if its prompts do not pad into the bucket).

That looked like the answer, and for a while the stage shipped it: one bucket, on by
default, with the honest two-point ladder of 1 clean and 20 corrupt.

**Then the shipped-configuration adapter probe found the hole.** Running four prompt lengths
in one session -- 128 and 100 inside the bucket, 37 and 4097 outside it -- the **4097-token
prompt diverges from its first token** with the single bucket resident, and is exactly
correct with tracing off.

Round 3 of the stage review was right that the first version of this finding was
under-investigated: the "correct at 20 buckets" datum was cross-revision, and the mechanism
being quoted could not produce a failure with no replay in it. Fifteen probes across six
configurations now pin it down, and every request from every one of them is tabulated in
`prefill_trace_discriminators.json`:

* **the capture is sufficient** -- a run whose *only* request is the 4097 one, with the bucket
  captured at warmup and never replayed, is still wrong;
* **it is not an unwarmed shape** -- 8192 is in `PREFILL_WARMUP_LENGTHS` and diverges too;
* **it is not bucket 128** -- bucket 96 reproduces it;
* **it is not monotone** -- 20 buckets get 4097 right, re-measured on the shipped revision;
* **the effect tracks trace SIZE, not count** -- a single *large* bucket ``[1024]`` also gets
  4097 right, so what separates the configurations is the size of the largest captured trace
  rather than how many there are;
* **and that is still not a way to ship it.** Round 5 of the review named the configuration
  worth wanting -- ``[128,1024]``: two traces, largest 1024, keeping the bucket the 1.29x was
  measured on. It was measured. 128, 1024 and 4097 are all correct in it, and **8192 is wrong**,
  byte-identically to the single-bucket failure. Six configurations measured; no traced one is correct at every
  length it was measured at, and the two cells not measured are named in
  `bench/run_discriminators.sh`.

So the mechanism the wide-set decay has (a buffer allocated under a live trace, overwritten
when it replays) does **not** explain this one, and the stage did not find the one that
explains both. That is recorded as open. The operational conclusion does not depend on it: no
traced configuration measured is correct at every length it was measured at, and the two cells
that were not measured -- ``[1024]`` alone at 8192, and any bucket size between 128 and 1024 --
are stated as the matrix's coverage limit rather than read as passes.

Every gate this stage had missed it. The short-prompt soaks were clean because they never
sent a long prompt; the server-side non-aligned check sends 4097 and 8193 but asserts only
that the request succeeded; the degenerate-output gate never saw that text. The one thing
that caught it was comparing a probe against the previous stage's committed token sequences
-- which is exactly the kind of cheap invariant that is worth keeping.

So **tracing ships off**, and the stage's headline is that nothing measurable moved. The
1.29x and 1.34x arms are kept with their full gate sets, because the next person to look at
TTFT here should start from "this is available the moment the allocator contract allows a
caller to reserve a captured trace's freed range" rather than from scratch.

## 6b. What the stage review changed

The independent review (`stage_review.md`) returned `more-work-needed` and was right on both
P1s. Recording what it moved, because the stage's conclusion is different because of it:

* **The reduced bucket set had never been measured.** The stage had bisected the failure
  carefully, tried two fixes, and then generalised from 20 buckets to "not expressible
  safely" — while the knob to test 1 bucket already existed and had only ever been swept
  upward. `soak_1bucket/` took ~12 minutes and turned a shipped-nothing stage into a shipped
  1.29x.
* **The deterministic reproducer was sitting unused.** `traced_qualitative/` corrupts at a
  fixed prompt index after the 22nd generation with byte-identical output across two servers; the
  narrative was describing the failure as sampling-suite-triggered and "after a few dozen
  requests" instead. It is now the reproducer the README leads the mechanism section with,
  and it is what `soak_1bucket/` was run against.
* **`before/serving_audit.json` was an audit of the wrong log** (§2), the degenerate-output
  gate log predated the arm it was credited with checking, the new
  `replacement_char_fraction` threshold was calibrated at a different granularity than it
  was applied (0.25 from per-set aggregates, applied per completion — at which the six
  *sampled* completions of a fully corrupted server passed), `check_degenerate_output.py`
  never globbed the chat arm at all, and the sampling failure set was described as identical
  to the baseline when it is a swap within the same class. All fixed; see §7 and the README.
* **~400 MB of raw server logs were staged for commit** under a `!doc/**/*.log` un-ignore.
  Re-ignored.

## 7. Two evidence tools were reporting the wrong thing

Both were found by this stage's own arms, and both would have made a reviewer's job
harder rather than easier.

* **`audit_serving.py` could never say clean.** It scanned the whole server log as one
  window, so the plugin's by-design phase-1 untraced decode warmup counted as a serving
  degradation. Now split into warmup / benchmark / checks / shutdown windows, with `clean`
  judged on the window whose numbers are being reported and the by-design markers of the
  other windows listed with the reason each is expected. The offsets are recorded by
  `bench/run_arm.sh` at the moment they happen rather than inferred afterwards.
* **`check_degenerate_output.py` passed a completely corrupted server**, and never looked
  at the chat arm at all. Its duplication and looping metrics are computed over `\w+`
  words; text made of U+FFFD replacement characters has almost none, so both scored 0.0.
  Three fixes, two of them from the round-1 review:
  - `replacement_char_fraction` on the raw text. Critical above **0.10**, advisory above
    **0.02**, calibrated **per completion**, which is the granularity it is applied at:
    every healthy completion in this corpus measures exactly 0.0000, the corrupted ones
    span 0.187-0.617. The first revision used 0.25 taken from a per-artifact-*set*
    aggregate, and at that threshold the six *sampled* completions of a fully corrupted
    server (0.187-0.248) passed. Negative control:
    `logs/degenerate_check_negative_control.log`, exit 2.
  - `discover()` now globs `qualitative_tt_chat.json` as well. The prompt-correct chat arm
    -- the one `$qualitative-check` reads its verdict from -- had never been inside this
    gate, on this model or any other.
  - deliberately-broken evidence is excluded by a `DEGENERATE_CHECK_EXCLUDE` marker file
    that the checker **reports** by artifact, marker and reason. The first revision renamed
    those files out of the glob instead, which also hid the two sets that straddle the
    threshold -- exactly the ones a reviewer needs -- and justified itself with a claim
    about the glob that was not true. The artifacts are back under their real names.

  What it still cannot see is a long-period verbatim loop: `trigram_loop_fraction` measures
  the coverage of the single most common trigram, so a completion that repeats a 40-word
  sentence seven times scores about 3/40. That is a real gap, it is pre-existing, and it is
  visible in this stage's own runner-arm output (see §8).

## 8. Final evidence

| arm | what it ran | audit |
|---|---|---|
| `before/` | 6 benchmark runs, committed vLLM-integration code | `clean: false` on the pre-windowing tool, whose single degraded marker is the plugin's phase-1 untraced decode warmup -- the by-design one the windowed revision classifies as expected-in-warmup. See the `_provenance` block in the file. |
| `after/` | 6 benchmark runs, then sampling (full), qualitative, chat qualitative, determinism — **the shipped configuration, tracing off** | **clean** (benchmark window) |
| `after_prefill_traced_1bucket/` | the same full arm at **1** bucket | benchmark window clean, gates pass; the arm the 1.29x comes from, not shipped |
| `after_prefill_traced/` | the same full arm at **20** buckets | benchmark window clean, gates pass; the arm the 1.34x comes from, not shipped. Its *post-sampling* qualitative is healthy only because the interlock had already released the traces |
| `soak_1bucket/` | 4 runner qualitative rounds + 2 chat rounds + 1 benchmark | **void** — its prompts pad to 32/64/96, so none took the traced path. Kept as the record of the mistake |
| `soak_traced_bucket/` | 14 rounds of prompts asserted inside the bucket, spanning an out-of-bucket qualitative round | clean; 84 in-bucket generations at 0.0000, byte-stable |
| `traced_qualitative/` | chat qualitative + runner qualitative + 1 benchmark at 20 buckets | the minimal reproducer: p0-p3 clean, decay from p4, the 22nd generation |
| `soak_blocking/` | ~80 real requests at 20 buckets with a blocking traced replay | the second refuted fix |
| `ctrl_notrace/` | control: tracing off, qualitative either side of the sampling suite | diagnostic; no benchmark window, so its expected sampling-suite markers are not separable |
| `bisect_server/`, `fixcheck/`, `localize/`, `sampling_variance/` | diagnostic arms | as above |

Acceptance tests: 29 selected tests pass plain (`logs/pytest_final.log`) and pass again
under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` with 20 watcher dumps and zero
error-shaped lines (`logs/pytest_watcher.log`, `watcher/watcher_excerpt.log`). Plain
`TT_METAL_WATCHER=10` aborts before any model code with
`TT_FATAL: Program size (28656) too large for kernel config buffer (25600) on ACTIVE_ETH`,
which is the signature `$optimize` names and prescribes that exact retry for; the scoped
limitation is that ETH cores were not watched.

`metrics.json` folds the seven arms that ran benchmark stages -- `before`, `after`,
`after_prefill_traced_1bucket`, `after_prefill_traced`, `before_sweep0`, `soak_1bucket`,
`soak_traced_bucket` -- so every performance number the README quotes is re-derivable from the
same script. The diagnostic arms that ran no benchmark (`ctrl_notrace`, `bisect_server`,
`fixcheck`, `localize`, `sampling_variance`, `soak_blocking`, `traced_qualitative`) are not in
it, by construction.

No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER` or `ttnn.ReadDeviceProfiler` was
run at any point in this stage, on a live server or otherwise. The device-time and
roofline fields of `perf_summary.json` are `null` with that reason recorded, and the
roofline carried from optimized-full-model is labelled an upper bound because it was
computed under that stage's precision policy.

## 8b. Artifact hygiene, and one thing that was lost

Two sweeps ran under the arm name `after` before the arms were renamed to what they are now,
and the second overwrote the first's **driver** logs. The surviving `logs/after_*.log` set
therefore belongs to the 1-bucket traced arm, not to the shipped one, and its header line
reads `prefill_trace=unset` -- which at the time meant the then-default one traced bucket,
i.e. the opposite of what shipped. They are renamed `after_prefill_traced_1bucket_*` with a
header recording all of that. The shipped arm's own driver logs are gone; its artifacts
(`run1..6`, `sampling_tests.log`, `qualitative/`, `determinism_vllm.json`,
`serving_audit.json`, `server_excerpt.log`, `bench_window_end_bytes.txt`) are intact and are
what the report cites. The same happened to the before arm's raw server log (§2).

Both are disclosed rather than tidied away, and the lesson is recorded rather than just the
loss: an arm should carry its identity *inside* its artifacts, not only in its directory
name. `serving_audit.json` files written before the rename now carry both `path` and
`path_as_run`, and each arm's `server_log_size.txt` names its own directory.

## 8c. Review rounds

Eight independent `$stage-review` rounds ran against this stage, and every one of the first seven
changed the result rather than rubber-stamping it. Recorded because several of the conclusions
are theirs as much as this stage's, and because the shape of the sequence -- each round finding
the next thing the evidence did not actually support -- is the useful part:

| round | verdict | what it changed |
|---|---|---|
| 1 (`stage_review.md`) | more-work-needed | the reduced-bucket configuration had never been measured, and a deterministic reproducer sat unused. Turned a shipped-nothing stage into a measured 1.29x -- temporarily |
| 2 (`stage_review_round2.md`) | more-work-needed | the soak that qualified that 1.29x never touched the traced path; also caught the wrong-log `before` audit, the mis-calibrated degenerate threshold, and the chat arm never being globbed |
| 3 (`stage_review_round3.md`) | more-work-needed | the 4097 mechanism was inconsistent with the observation and the 20-bucket datum was cross-revision. Six further probes; the mechanism is now scoped and the one-bucket case labelled unexplained |
| 4 (`stage_review_round4.md`) | more-work-needed | contract evidence certified with the wrong probe, refuted mechanism still in shipped docstrings, loop count 2/12 vs 3/12, discriminator commands unrecorded -- and it named the one probe still missing, a single *large* bucket, which reframed the matrix from trace count to trace size |
| 5 (`stage_review_round5.md`) | more-work-needed | the decisive `[1024]` probe's own traced output had no control (it turned out to be a property of the synthetic prompt), and `[128,1024]` -- the configuration that would have kept the 1.29x -- had never been measured. Both were run; `[128,1024]` fails at 8192, which is what finally settles the rejection |
| 6 (`stage_review_round6.md`) | more-work-needed | wording that overstated the matrix ("six configurations, each wrong at some length"), a driver script that reproduced 7 of 15 probes, and a work log not carried through round 5 |
| 7 (`stage_review_round7.md`) | more-work-needed | the overstated claim surviving in the one paragraph that states the ship decision, and a causal attribution for a non-shipped arm's loop count that the report's own interlock account ruled out |
| 8 (`stage_review_round8.md`) | **clean-pass** | rebuilt the matrix, the loop metric, every headline number and the shipped code's bytecode independently; no required work |

## 8d. On `$autofix`

`$optimize` and `$tt-enable-tracing` both point at `$autofix` when a failure crosses op,
kernel or runtime boundaries and progress stalls. It was not invoked here, and the reason is
worth stating rather than leaving as an omission: the failure never stalled. Each round
produced a cheap, decisive experiment -- an in-server bisection, a file-at-a-time
localisation, then a fifteen-probe matrix over one build -- and the loop that skill exists to
run (propose a cause, verify or refute it in isolation, keep only what survives) is exactly
the loop these probes ran, with the stage reviews supplying the adversarial half. What is
missing at the end is not a debugging pass but device-level visibility this port does not
have: the freed-intermediate address range of a captured trace, which is the first upstream
ask in the README.

## 9. Hardware

No resets, no hangs, no ARC/ERISC/remote-Ethernet events, no `tt-triage` capture needed.
Devices were serialized one job at a time throughout. Two servers were killed mid-launch
by this stage (an accidental duplicate `before` launch, and a sampling-repetition arm
abandoned once the corruption bisection made it redundant -- its tree is deleted rather
than committed as evidence of nothing); in both cases the launcher and
`VLLM::EngineCore` were killed, `/dev/tenstorrent/*` was confirmed free, and the next job
opened the mesh normally.
