# Stage Review (round 2)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against `.agents/prompts/model_bringup_multigoal/10-optimized-vllm.txt`,
`$optimize`, `$vllm-integration`, `$tt-enable-tracing`, `$tt-device-usage`.
Worktree live, changes uncommitted (8 modified files + untracked `doc/optimized_vllm/`).
Re-review of the same stage after remediation of `stage_review.md`.

Verdict: more-work-needed

**Round-1 items: 10 of 11 are genuinely fixed, not reworded.** I re-derived each one.
The reduced-bucket configuration was really measured (`soak_1bucket/`), the before-arm
audit is now the real 397 KB log with `clean: false` and an honest provenance block, the
degenerate gate log postdates every artifact it scans (15:00:39 vs 14:50), the threshold
is recalibrated per completion (0.10/0.02) with a working negative control that fires 3
criticals, `qualitative_tt_chat.json` is globbed, `DEGENERATE_CHECK_EXCLUDE` reports 14
exclusions and hides nothing a gate should catch (I checked every marker directory: all
are genuinely corrupted, and `traced_qualitative/qualitative1` — the clean round in an
excluded arm's tree — is still scanned), the log is sliced as bytes, the sampling failure
set is now correct member-by-member and its floating-member table reproduces exactly from
all six `sampling_tests.log` files, the interlock has a host test that pins the sampler
attributes, the raw logs are re-ignored (409 MB → 11 MB staged), `after_sampling_reps/` is
gone, the watcher dump count is right (20), and the guarded-localization result is
correctly restated in the README. Every headline number reproduces from raw JSON: TTFT
81.48 → 62.97 (1.294x), decode t/s/u 43.480 → 43.430, burst 721.88 → 812.10 (+12.50 %),
burst TTFT 2147.53 → 1654.70 (−22.95 %), and both probes are token-identical to
`doc/vllm_integration/probe_full_fixed.json`.

What does not hold up is the **new** headline's safety argument. The shipped bucket is
padded 128, i.e. prompts of 97–128 tokens. The `soak_1bucket/` prompts are 64–79 tokens
(chat) and 7–22 tokens (runner). **None of the 84 completions the stage cites as the
qualifying evidence was produced by a traced prefill** — they all ran eager, which is the
path `ctrl_notrace/` already showed was healthy. The 20-bucket arm corrupted on exactly
that traffic *because* buckets 32 and 96 were resident and it did replay traces. So the
one experiment the round-1 review asked for was run with the variable it was supposed to
isolate switched off for the traffic that does the checking. Separately, the adapter's own
prefill path allocates device buffers twice per request while prefill traces are resident,
in a code path the interlock does not cover, while the README concludes the allocations
come "from code this adapter does not own".

---

## Required Work

- **P1: The soak that qualified the shipped configuration never exercised the shipped
  traced path.**

  Evidence:
  - The trace is selected by exact padded bucket:
    `tt/generator.py:_prefill_traced` computes `padded_len = ceil(L/32)*32` and does
    `self._prefill_traces.get(padded_len)`. `soak_1bucket/server/server.log` records
    `prefill tracing enabled (max_entries=1, max_padded_len=128)` and
    `prefill traces resident for padded buckets [128]`. So only prompts of **97–128
    tokens** replay; everything else takes the eager path.
  - `soak_1bucket/qualitative1/qualitative_prompts.json` — the pinned chat prompts are
    **64, 68, 79, 64, 70, 66 tokens** → padded **64/96**. Not 128.
  - The runner arm posts the raw strings in
    `models/common/readiness_check/vllm_prompts.txt` ("Write a haiku about machine
    learning.", …) → roughly **7–22 tokens** → padded **32**. Not 128.
  - Therefore all **84** completions in `soak_1bucket/` ran the **eager** prefill. The
    only traffic in that arm that replayed the trace is `soak_1bucket/run1`: one 128-token
    single-user request plus 32 100-token burst requests = **33 replays**.
  - The 20-bucket arms corrupted on precisely this traffic *because* buckets 32 and 96
    were resident (`after_prefill_traced/server_excerpt.log` lists all 20). Dropping to
    `[128]` removed both the poisoned range **and** essentially all traced activity on the
    checking traffic. The two are confounded and the stage does not say so.
  - The claims that rest on this:
    - README *Status*: "Qualitative with the prefill trace resident | **pass** —
      `soak_1bucket/`, 84 completions, `replacement_char_fraction` 0.0000 throughout".
    - README §4: "Qualitative evidence *with the trace resident* comes from
      `soak_1bucket/` instead."
    - README *What is shipped*: "Measured clean over `soak_1bucket/` (84 completions …)".
    - `work_log.md` §6: "**84 completions, every one at `replacement_char_fraction`
      0.0000**, on the exact traffic pattern that corrupted the 20-bucket arm".
    - `perf_summary.json` `prefill_trace_bucket_ladder.measured_clean.evidence`.
    - `tt/generator_vllm.py` `PREFILL_TRACE_BUCKETS` docstring.
    "Resident" is the wrong safety property. ttnn's rule is about a trace being
    **executed** (`allocator.cpp:113-126`, quoted by the stage itself); a resident trace
    that is never replayed cannot overwrite anything.
  - Total traced-path exposure at bucket `[128]` in the whole stage: 33 replays in
    `soak_1bucket/run1` plus 198 in `after/run1..6` (6 × (1 + 32), all before the guard
    fired in the sampling stage) ≈ **231 replays**, every one of them
    `vllm bench serve --dataset-name random` traffic whose output text nobody reads
    (`ignore_eos`, random token prompts, only `missing_output_tokens` checked).
  - Secondary: the "84 completions" figure double-counts. `qualitative_tt_chat.json` is a
    re-serialisation of the sibling `vllm_qualitative_outputs.json`'s **greedy**
    completions — I compared all six in `after/qualitative/` and they are equal strings.
    So `soak_1bucket/` is 4×12 + 2×12 = **72 distinct generations**, not 84.

  Why this matters: this is the entire basis for flipping the default from off to on for a
  failure the stage documents as silent, deterministic, and invisible to the naked eye.
  The real evidence is "231 benchmark-shaped traced replays across two servers, each
  followed by readable output that stayed clean" — which is a defensible thing to ship, but
  it is roughly a third of what the stage claims and it covers none of the mixed
  traced/eager/sampling traffic that broke 20 buckets.

  Required next step: run one more one-bucket soak whose *checked* prompts land in the
  bucket — either pad the qualitative prompts into 97–128 tokens, or add bucket 32 and 96
  so the existing runner and chat prompts replay — and report it. If that is not run,
  restate every claim above as what it is: 231 traced replays in benchmark traffic,
  followed by clean readable output, with the qualitative rounds serving as the detector
  rather than as traced-path coverage.

- **P1: The adapter allocates device buffers twice per request inside the traced prefill
  path, unguarded, and the mechanism narrative blames external code.**

  Evidence:
  - `tt/generator_vllm.py:prefill_forward` calls
    `generator.prefill_forward(..., sample_on_device=sampling_params is not None)`; with
    `sample_on_device_mode=all` that is true for every serving request.
  - `tt/generator.py:1492-1510` → `_sample_eager` (`tt/generator.py:599-618`) runs
    `self.sampling.sample(logits, enable_trace=False, ...)` — an **untraced** sampling call
    that allocates its own intermediates — once per prefill, for every request.
  - `tt/generator.py:_prefill_traced` allocates again on every replay:
    `logits = ttnn.clone(entry["logits"], memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
  - `_guard_late_sampling_capture` is called from exactly one place:
    `tt/generator.py:1633`, inside `_decode_submit_traced`. The prefill path never
    consults it. `grep -n "_guard_late_sampling_capture" tt/generator.py` → definition +
    one call site.
  - README §4 and *Limitations* 2 scope the interlock as covering "the sampler's own
    allocation sites". `_sample_eager` **is** a sampler allocation site and is not covered.
  - README *Mechanism* concludes: "a vLLM server allocates continuously from code this
    adapter does not own." `work_log.md` §6 repeats it: "a serving process allocates
    continuously, from code this adapter does not own". Both are contradicted by the two
    adapter-owned allocations above, which happen once per request in the traced path.
  - This also weakens the seeded-request story. README: "`test_seeding_and_variety` is the
    first *identified* trigger, because the shared sampler deliberately bypasses its trace
    when a per-request seed is active … and allocates instead." The adapter already runs an
    untraced, allocating sample on **every** prefill regardless of seeds — which is
    consistent with `traced_qualitative/` corrupting with no seeds at all and with the guard
    never firing there (`grep -c prefill_traces_released_for_sampling_capture
    traced_qualitative/server/server.log` = 0).
  - `Rejected and deferred` lists only one untried mitigation (a ballast buffer over the
    freed range). A persistent prefill-logits output buffer instead of a per-request
    `ttnn.clone`, and a warmed/persistent prefill sampling buffer, are neither tried nor
    named — and both are inside code this stage owns. `$optimize` and `$tt-enable-tracing`
    both prescribe `$autofix` at this point; `grep -n autofix README.md work_log.md` still
    returns nothing.

  Why this matters: `$stage-review` requires more work when "logs or code show a plausible
  bug in a stage-critical subsystem, such as … trace replay" and when "the stage dismisses
  a material anomaly with prose instead of investigation". The stage's own hypothesis
  (allocations landing in a captured trace's freed-but-baked range) points straight at two
  per-request allocations in its own prefill path, and the write-up instead attributes the
  cause to code it does not own and stops.

  Required next step: name these two allocation sites in the mechanism section; either
  extend the interlock/persistent-buffer plan to cover them and measure the 20-bucket soak
  again, or record the exact blocker. Correct the two "code this adapter does not own"
  sentences and the "sampler's own allocation sites" scoping.

- **P2: `probe_full_shipped.json` is a probe of a configuration that is no longer
  shipped, and the docs present it as the shipped default.**

  Evidence:
  - `probe_full_shipped.json` → `prefill_trace: {"enabled": false, "buckets_requested":
    [32,64,…,1024], "buckets_resident": []}`, `capability_report.prefill_trace: false`, and
    `requests[0]` (prompt_len 128) → `prefill_counters.trace_replays: 0`.
  - The shipped configuration is `enabled: true`, buckets `[128]`, and a 128-token prompt
    replays (`after/server_excerpt.log`: `prefill traces resident for padded buckets [128]`).
  - mtime 13:57 — before `soak_1bucket/` (14:37) and the decision to ship tracing on.
  - `work_log.md` §5: "`probe_full_shipped.json` (shipped default)". README *Contract
    evidence*: "From `probe_full_prefill_traced.json` and `probe_full_shipped.json` — the
    adapter driven through the TT plugin's exact call sequence".
  - Consequence: **no adapter probe covers the shipped configuration.** The decode-path
    contract items (steady-state counters, page-table changed/unchanged, async split,
    bit-identity) are covered by the union of the two probes because prefill tracing does
    not touch decode, but that argument is nowhere in the docs and the file name asserts
    the opposite.

  Required next step: rerun the probe with the shipped default (it takes one build), or
  rename it (`probe_full_prefill_eager.json`) and state explicitly which probe evidences
  which contract item for the shipped path.

- **P2: The shipped arm's runner qualitative output contains blatant mechanical loops, the
  README calls it "coherent", and the shared loop metric cannot see it.**

  Evidence (all from `after/vllm_qualitative_outputs.json`, the shipped arm):
  - `prompt[0] sampled_completion` is "Write a haiku about machine learning." repeated
    **31 times** verbatim.
  - `prompt[2] greedy_completion` is a 40-word sentence ("Once upon a time … (fill in the
    blank)") repeated **7 times** verbatim.
  - `prompt[1] greedy_completion` is the same shape.
  - `check_degenerate_output.py` scored these **0.5000, 0.0995, 0.0664** and the run
    reported "No degenerate output detected", exit 0
    (`logs/degenerate_check_all.log`). The reason is structural:
    `trigram_loop_fraction` (`check_degenerate_output.py:153-169`) measures only the
    **single most common trigram's** non-overlapping coverage, so a verbatim N-word block
    repeated any number of times scores ≈ `3/N`. That is a hole of exactly the class this
    stage just spent a hardening pass closing, present in this stage's own gated artifact.
  - README *Status*: "Qualitative, runner raw-completion arm | **pass** — coherent
    continuation-style output". False for three of twelve completions.
  - Control: the identical values (0.415 greedy / 0.842 greedy) appear in
    `readiness_vllm/vllm_qualitative_outputs.json` from the previous stage and in
    `after_prefill_eager/`, `after_prefill_traced/` and `traced_qualitative/`, so this is
    pre-existing and deterministic, almost certainly the instruct model echoing a raw
    completion prompt. It is not a regression — but it is unclassified here, and
    `$qualitative-check` requires the classification, not the assertion.

  Required next step: classify it (an HF raw-completion control on the same prompts settles
  it in minutes and the HF chat control already exists), correct the Status row, and either
  extend the loop metric to catch long-period verbatim repeats or record the blind spot
  next to the `replacement_char_fraction` calibration.

- **P2: `after_prefill_eager/` was measured with six runs and its numbers appear nowhere,
  so the win is never decomposed — and one README statement is wrong because of it.**

  Evidence (recomputed from `after_prefill_eager/run*/vllm_benchmark.json`, median of runs
  4–6, the same protocol as every other arm):
  - single-user TTFT **77.42 ms** vs before 81.48 → the non-tracing changes (the
    page-table-row change) are worth **~22 % of the total TTFT gain** on their own;
  - CI burst throughput **717.56 tok/s** vs before 721.88 → the non-tracing changes are
    **−0.6 %** on burst.
  - `grep -n "77.42\|717.56\|after_prefill_eager" README.md work_log.md perf_summary.json`
    returns no number for this arm anywhere; `metrics.json` folds it correctly.
  - `work_log.md` §8 calls it "the control that isolates the other changes as numerically
    inert". Token-identity is inert; 5 % of TTFT is not.
  - README line 41: "Every other prompt length serves on the eager path at **the
    before-arm TTFT**." Measured, it is 77.42 ms, not 81.48 ms.

  Why this matters: `$stage-review` for optimization stages requires that performance
  claims compare like with like. The stage measured the exact arm that separates "new code"
  from "tracing" and then reported only the combined delta.

  Required next step: quote the eager-control TTFT and burst throughput in the README and
  `perf_summary.json`, attribute the split, and fix the "at the before-arm TTFT" sentence.

- **P2: `work_log.md` §7 still carries the refuted 0.25 calibration that §6b says was
  fixed.**

  Evidence:
  - `work_log.md` §7: "It now measures `replacement_char_fraction` on the raw text,
    **critical above 0.25**, calibrated on this stage's own artifacts (**10 healthy sets at
    0.0000, 3 corrupted at 0.512–0.539**)".
  - `check_degenerate_output.py:104-108`: `REPLACEMENT_CHAR_CRITICAL = 0.10`,
    `REPLACEMENT_CHAR_ADVISORY = 0.02`, and the comment above it explicitly labels
    "0.512-0.539" as a per-artifact-set aggregate that "overstated the margin in the
    direction that matters".
  - `work_log.md` §6b: "the new `replacement_char_fraction` threshold was calibrated at a
    different granularity than it was applied … All fixed; **see §7 and the README**."
    §7 is the section that still contains the wrong numbers.
  - Minor, while this is being edited: the code comment says the corrupted range is
    "0.187-0.617"; recomputed over every artifact the lowest corrupted completion is
    **0.1860** (`soak_blocking/runner_qual2` prompt[2] sampled).

  Required next step: rewrite §7 to match the code.

- **P2: `work_log.md` §8 still records the `before/` audit as "clean" while the committed
  artifact says `clean: false`.**

  Evidence:
  - `work_log.md` §8 table: "| `before/` | 6 benchmark runs, committed vLLM-integration
    code | clean |".
  - `before/serving_audit.json`: `"clean": false`, `degraded_markers:
    ["DEGRADED PATH untraced_eager_decode"]`, with a `_provenance` block explaining it was
    computed by the pre-windowing revision so the union covers the warmup window.
  - Round 1 asked for §2 **and** §8 to be corrected. §2 was rewritten well; §8 was not.

  Required next step: say "clean under the windowed reading; `clean: false` as committed
  because the pre-windowing tool reported the union" in the §8 row, or drop the word.

- **P2: The "about nine ordinary requests" framing of the 20-bucket onset undercounts by
  more than half, in the direction that flatters the shipped configuration.**

  Evidence:
  - `traced_qualitative/` steps are `qualchatrep1, qualitative, bench1`
    (`logs/traced_qualitative.log`). `qualchatrep1` produced
    `traced_qualitative/qualitative1/vllm_qualitative_outputs.json` — **12 clean
    generations** — before the corrupting round started.
  - Same in `soak_blocking/`: `qualchatrep1` → `qualitative1/` 12 clean, then
    `runner_qual1` decays at p4 sampled.
  - So the onset is the **22nd** generation of the server's life in both arms, not the 9th
    or 10th. README *The minimal reproducer* and `work_log.md` §8 both say "~nine ordinary
    greedy/sampled requests".
  - The comparison the stage draws ("84 clean vs corrupts after ~9") is therefore really
    "72 distinct clean generations, none of them traced, vs corruption at generation 22".
  - Worth keeping: the onset index is **identical** in the two servers and the p5 greedy
    corrupt string is byte-identical (256 chars, verified), which is a stronger
    determinism statement than the README makes.

  Required next step: state the onset as the 22nd generation with the preceding clean chat
  round included, in the README and the work log.

---

## Other Concerns

- **Two stacked, contradictory docstrings on `PREFILL_TRACE_BUCKETS`**
  (`tt/generator_vllm.py`). The old block ("Padded prefill buckets that additionally get a
  **captured trace** … these are exactly the short padded lengths `PREFILL_WARMUP_LENGTHS`
  already compiles") is still there immediately above the new one-bucket block that says
  the opposite. Shipped code, and the first thing a reader hits.
- **The env-gate docstring and warning quote the not-shipped arm's numbers.**
  `_PREFILL_TRACE_ENV` says the default is worth "a measured **1.34x** on single-user TTFT
  … and **+11.5 %** on CI serving-burst throughput"; the shipped configuration measures
  **1.294x** and **+12.5 %** (the quoted pair is `after_prefill_traced/`'s). The
  `MUSE_GLIMMER_VLLM_PREFILL_TRACE=0` warning says the same 1.34x.
- **`tt/generator.py:938-957` records the blocking-replay change as if it worked.** "Blocking
  here costs the host wait between submit and consumer and **removes the race entirely**",
  with no note that `soak_blocking/` corrupted anyway with `blocking=True` at 20 buckets.
  The README records it as refuted; the code comment does not, and the code is what a
  future reader will find.
- **Stale `server_log_size.txt` after the arm rename.** Both
  `after_prefill_eager/server_log_size.txt` and `after_prefill_traced/server_log_size.txt`
  name `.../doc/optimized_vllm/**after**/server/server.log`. The sizes (81M, 53M) are
  correct for their own logs; the paths are not.
- **`soak_1bucket/` ran a 12-line-earlier `generator_vllm.py`** than the shipped one
  (`_prefill_trace_enabled:233` vs 243 today; `get_max_tokens_all_users:476` vs 488) — the
  pre-default-flip revision, which is why it logs the "off by default" warning. `generator.py`
  matched exactly (`_capture_sampling_trace:1612`, `build_generator:2440`), so the trace
  behaviour is the shipped one and this is immaterial, but it is unrecorded and the README
  presents the `soak_1bucket` command as reproducing the current code.
- **`work_log.md` §8: "metrics.json folds every arm above".** It folds 6 arms
  (`before`, `after`, `after_prefill_traced`, `after_prefill_eager`, `before_sweep0`,
  `soak_1bucket`); the §8 table lists 13.
- **The interlock test pins names but not shapes.** `test_the_prefill_trace_interlock_…`
  asserts `hasattr(real, "_trace_slot")` but never that the real `_trace_slot` takes three
  positional booleans and returns `(key, slot)`. An arity change still lands in
  `except Exception: return None` and the guard fails open with the test green.
- **`audit_serving.scan` reports `"bytes": len(text)`**, which is the decoded character
  count after the shutdown truncation, not bytes. `after/serving_audit.json` says 84121001
  where `wc -c` is 84125747. Cosmetic, but the field is named for the unit the offsets use.
- **`traced_qualitative/serving_audit.json` reports `clean: true`** with
  `benchmark_window_end_bytes` equal to the whole non-shutdown log, because `bench1` ran
  last in that arm — so the "benchmark window" label covers the corrupted qualitative
  traffic. Harmless here (the corruption emits no marker, which is the stage's point), but
  the window semantics are meaningless for any arm whose bench step is not first.
- **"character-identical to the standalone model over the full common prefix"** (README
  Status) is 79 characters of a 392-character baseline — `determinism_vllm.py` requests
  `max_tokens: 24`. The stronger comparison exists
  (`after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`, 127 tokens, diverging
  at index 2 on the API-invisible `<|message|>` token) and is the one worth citing.
- **`tt/generator.py:173` cites `doc/optimized_full_model/ccl_host_probe.json`**, which does
  not exist (only `_bf16`, `_bfp8`, `_bfp8_loaded`). Not in this stage's diff — pre-existing
  — but round 1 flagged the same dangling name in two other places and both were fixed.

---

## Hard-Check Gaps

- The freed-intermediate address range of one prefill trace is **still unmeasured**.
  `doc/optimized_full_model/prefill_trace_probe.json` records
  `capture_retained_dram_bytes: 3280896` (3.1 MB retained per bucket) but no peak-during-capture
  reading, so "twenty 52-layer *prefill* working sets" versus "a small, decode-shaped range"
  — the quantitative core of the ship/don't-ship argument — is still an assertion. This was
  a round-1 ask and was not addressed.
- Nothing between 2 and 19 buckets is measured. Honestly disclosed, but it means the
  ship decision rests on a two-point ladder whose clean end is now known (P1) to have been
  measured with the mechanism largely disengaged.
- No long-duration soak of the shipped default. Longest continuous traced-path exposure is
  the `after/` arm's six benchmark rounds (~2.5 min, 198 replays) before the guard fired.
- `supports_async_decode=True` is still justified by the previous stage's
  `--async-scheduling` arm; the decode path is unchanged, so this is reasonable.
- No qualitative or eval evidence at long context; the 131072 contract is evidenced by
  served `max_model_len` (verified in `after/server_excerpt.log`) and `doc/context_contract.json`
  (`capability_reduction: none`), not by a long-context serving generation. Unchanged from
  round 1 and acceptable for this stage.

---

## Anomaly Ledger

- Observed anomaly: served output decays into U+FFFD with prefill traces resident.
  Evidence: `bisect_server/qualitative3` (0.207–0.539 per completion),
  `fixcheck/qualitative{2,3}`, `soak_blocking/qualitative{2,3}` + `runner_qual{1,2,3}`,
  `traced_qualitative/vllm_qualitative_outputs.json`; onset at generation 22 in two
  independent servers with a byte-identical 256-char p5 greedy string.
  Affected path: serving prefill trace replay + adapter/serving allocations.
  Control or comparison: `ctrl_notrace/` (tracing off, same binary, healthy either side of
  the sampling suite, reproduces the baseline 10-failure set); `prefill_trace_bisect.json`
  and `probe_full_prefill_traced.json` (traced prefill is token-identical, so it is a
  memory-lifetime bug, not math).
  Likely subsystem: ttnn trace/allocator lifetime, plus the adapter's own per-request
  `ttnn.clone` and untraced prefill sampling call.
  Investigation performed: 4-step in-server bisection, two refuted fixes, a one-way
  interlock, a capacity ladder, and a one-bucket arm.
  Resolution: **more-work-needed** — the one-bucket arm does not exercise the mechanism on
  its checked traffic (P1), and the two adapter-owned allocation sites inside the traced
  prefill path were never considered (P1).

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner qualitative
  output (31× and 7× sentence repeats) that the stage gate passes.
  Evidence: `after/vllm_qualitative_outputs.json` prompt[0] sampled, prompt[2] greedy,
  prompt[1] greedy; `logs/degenerate_check_all.log` scores them 0.5000/0.0995/0.0664 and
  reports "No degenerate output detected".
  Affected path: raw-completion prompts against an instruct checkpoint; the shared
  `trigram_loop_fraction` metric.
  Control or comparison: byte-identical greedy values in `readiness_vllm/` from the previous
  stage and in `after_prefill_eager/`/`after_prefill_traced/`; the prompt-correct chat arm
  is clean (max trigram 0.244) and the HF chat control is comparable.
  Likely subsystem: prompt format, not the port.
  Investigation performed: none in this stage.
  Resolution: **more-work-needed** (classification + README correction + metric blind spot).

- Observed anomaly: the interlock fires in `after/` during the sampling suite, so the
  shipped arm's qualitative, determinism and non-aligned evidence all ran eagerly.
  Evidence: `after/serving_audit.json` `degraded_checks_window` contains
  `DEGRADED PATH prefill_traces_released_for_sampling_capture`; `logs/after_arm.log` orders
  sampling before qualitative.
  Affected path: what the `after/` arm proves about the traced configuration.
  Control or comparison: README §4 discloses it correctly.
  Likely subsystem: the one-way guard.
  Investigation performed: disclosed in §4 but not carried into *What is shipped* or
  `perf_summary.json.measured_clean.evidence`, both of which cite the after arm's
  "both qualitative arms and determinism" without the caveat.
  Resolution: **more-work-needed** (documentation; folded into P1).

- Observed anomaly: leading `" to=self"` / `" to=user"` and first divergence from the
  standalone baseline at token 2 on all six chat prompts.
  Evidence: `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`
  (`first_divergence: 2`, heads differ only by `<|message|>`).
  Control or comparison: HF chat control present; `determinism_vllm.json`
  `standalone_baseline.identical_over_common_prefix: true`.
  Likely subsystem: Harmony-style channel tokens invisible over the OpenAI API.
  Investigation performed: carried from earlier stages, controls present here.
  Resolution: **controlled**.

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every
  pytest and sampling log.
  Evidence: `logs/pytest_final.log`, every `*/sampling_tests.log`.
  Control or comparison: identical in the before arm and previous stages.
  Likely subsystem: ttnn Python bindings teardown.
  Investigation performed: none needed.
  Resolution: **controlled**.

---

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); goal contract as
  supplied; `doc/optimized_vllm/stage_review.md` (round-1 report, each item re-derived).
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,metrics.json}`;
  every `run*/vllm_benchmark.json` and `run*/vllm_ci_serving_benchmark.json` in
  `before/`, `before_sweep0/`, `after/`, `after_prefill_traced/`, `after_prefill_eager/`,
  `soak_1bucket/`, `soak_blocking/`, `bisect_server/`, `fixcheck/`, `traced_qualitative/`;
  every `serving_audit.json`, `sampling_tests.log`, `server_excerpt.log`,
  `bench_window_end_bytes.txt`, `server_log_size.txt`;
  every `vllm_qualitative_outputs.json`, `qualitative_tt_chat.json`,
  `qualitative_prompts.json`, `qualitative_prompt_format.json`,
  `qualitative_hf_chat.json`, `qualitative_vllm_vs_datatype_sweep_chat.json`;
  `after/determinism_vllm.json`; all nine `DEGENERATE_CHECK_EXCLUDE` markers;
  `probe_full_shipped.json`, `probe_full_prefill_traced.json`, `probe_trace_capacity.json`,
  `prefill_trace_bisect.json`, `corruption_localization.json`,
  `corruption_localization_unguarded.json`; `logs/` (degenerate_check_all,
  degenerate_check_negative_control, before_audit, run_tests, run_watcher, pytest_final,
  pytest_watcher, every `*_arm.log`/driver log, traced_qualitative_audit);
  `watcher/watcher_excerpt.log`; `bench/run_arm.sh`;
  `doc/vllm_integration/{README.md,probe_full_fixed.json}`;
  `doc/datatype_sweep/{evidence_perf.json,qualitative/qualitative_tt_chat.json}`;
  `doc/context_contract.json`;
  `doc/optimized_full_model/prefill_trace_probe{,_8192}.json`;
  `readiness_vllm/vllm_qualitative_outputs.json`;
  `models/common/readiness_check/vllm_prompts.txt`; `.gitignore`.
- Code paths: `tt/generator.py`, `tt/generator_vllm.py`, `tt/model.py`,
  `tests/test_full_model.py`, `doc/vllm_integration/bench/{audit_serving,adapter_probe}.py`,
  `models/common/readiness_check/check_degenerate_output.py` (all via `git diff HEAD` plus
  direct reads of `_prefill_traced`, `_sample_eager`, `_guard_late_sampling_capture`,
  `_sampling_allocates_this_step`, `trigram_loop_fraction`, `discover`).
- Commands run (all read-only; no server, device, or hardware use):
  `git status/diff/check-ignore`, `find`, `stat`, `du`, `wc`, `grep`, and Python scripts
  that recomputed warm medians for every arm and profile, recomputed
  `replacement_char_fraction` and trigram-loop metrics over every qualitative artifact,
  diffed probe token sequences against `probe_full_fixed.json`, compared chat and runner
  artifacts for duplication, measured pinned prompt token lengths against the bucket, and
  checked every file path cited in the docs and shipped code for existence.

---

## Residual Risk

- The shipped default enables, on by default, a mechanism this stage proved corrupts
  silently and deterministically at 20 buckets, on the strength of ~231 traced replays in
  benchmark traffic whose output text was never read. If P1 is closed by re-soaking with
  in-bucket prompts, this drops to normal; as it stands the safety margin is unquantified.
- The interlock does not cover the adapter's own per-request prefill allocations, so a
  server serving 97–128-token prompts is doing an unguarded allocating sample plus a
  `ttnn.clone` on every request while the trace is resident.
- Once the interlock fires (any seeded request), the 1.29x is gone for that server's
  lifetime. Disclosed, but it means the advertised headline is not the steady-state
  headline for a workload that uses seeds.
- The shared `trigram_loop_fraction` metric is blind to long-period verbatim loops, so a
  future regression of that shape passes `--scope all` on any model.
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`;
  the new test pins attribute existence but not signatures.
- Nothing between 2 and 19 buckets is measured, and the freed address range per trace is
  still unmeasured, so a deployment widening the set with
  `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS` has no interpolation to reason from.
