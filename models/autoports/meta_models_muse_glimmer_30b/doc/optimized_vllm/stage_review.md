# Stage Review

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against `.agents/prompts/model_bringup_multigoal/10-optimized-vllm.txt`,
`$optimize`, `$vllm-integration`, `$tt-enable-tracing`, `$tt-device-usage`.
Worktree live, changes uncommitted (7 modified files + untracked `doc/optimized_vllm/`).

Verdict: more-work-needed

The core performance story reproduces exactly from raw JSON: I re-derived every
headline and secondary number (TTFT 81.48 → 77.42, decode t/s/u 43.480 → 43.428,
burst 721.88 → 717.56 tok/s, traced-arm TTFT 60.66 = 1.343x, 1.33x/1.00x prefill-trace
probes, 43.331 standalone t/s/u) and all of them are correct. Bit-identity against
`doc/vllm_integration/probe_full_fixed.json` is real for both probes. Non-aligned
prompt lengths, the 131072 context contract, async decode, non-blocking traced decode,
on-device sampling, persistent trace inputs and stale-input coverage all hold up.

What does not hold up is (a) the rejection of the one optimization this stage had,
which was closed out with prose and two fixes while a deterministic ~9-request
reproducer, an untried reduced-bucket configuration, and `$autofix` were all sitting
there unused; and (b) a set of committed evidence artifacts that do not say what the
report says they say — a `before` audit computed over the wrong log, a stage gate log
that predates the shipped arm by an hour, a degenerate-output gate that provably
cannot see the chat arm it is credited with checking, and a sampling failure set
described as identical to the baseline when it is a swap.

---

## Required Work

- **P1: The 1.34x TTFT optimization was rejected without measuring the reduced-risk
  configuration, and without `$autofix`.**

  Evidence:
  - README *Rejected and deferred*: "A *smaller* traced bucket set as a safer
    compromise | **Not claimed.** Fewer buckets shrink the poisoned address range but
    do not remove the hazard, and no soak was run to support a reduced set." That is
    prose, and the stage says so itself.
  - The knob to test it already exists and is already wired:
    `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS` (`tt/generator_vllm.py:_prefill_trace_buckets`).
    `grep -c PREFILL_TRACE_BUCKETS` over every arm server log
    (`after_prefill_traced`, `soak_blocking`, `traced_qualitative`, `fixcheck`,
    `bisect_server`, `localize`, `sampling_variance`) returns **0**. The only use in
    the stage is `logs/probe_trace_capacity.log`, and that sweep goes *upward* to 64
    buckets. No 1-bucket, 2-bucket or short-bucket arm was ever run.
  - The stage's own framing makes the reduced set the obvious candidate: "The decode
    and sampling traces put a small, decode-shaped range under that rule and this port
    has lived with it since optimized-full-model." The benchmark's whole primary win
    comes from a single bucket (padded 128); `doc/optimized_full_model/prefill_trace_probe.json`
    measures 1.33x at 128 padded rows with `with_decode_traces: true`,
    `capture_retained_dram_bytes: 3280896` — one bucket, ~3.3 MB retained.
  - The size of the "poisoned range" is asserted ("a 52-layer prefill working set"),
    never measured, even though `prefill_trace_probe.json` already records
    `dram_free_before_capture_bytes` / `dram_free_after_capture_bytes` and a
    peak-during-capture reading would size it directly.
  - `$autofix` appears **nowhere** in `README.md` or `work_log.md`
    (`grep -n "autofix" → no matches`), although `$optimize` ("If the failure crosses
    several ops, kernels, layouts, or planner/runtime boundaries and you are not
    making progress, use `$autofix`") and `$tt-enable-tracing` ("If you are still stuck
    after isolating the failing block, use `$autofix`") both prescribe it for exactly
    this situation.
  - The claim "There is no way for a caller to tell the allocator to keep a captured
    trace's intermediate address range reserved" is not demonstrated. The cited
    `tt_metal/impl/allocator/allocator.cpp:113-126` is a warning, and
    `tt_metal/distributed/mesh_device.cpp:1315-1361` shows `allocations_unsafe_` is
    set at `end_mesh_trace` and cleared only when every trace is released — that is
    the state, not a proof that a caller-side moat (a persistent buffer allocated
    immediately after capture to cover the freed range) cannot work. No such attempt
    is recorded.

  Why this matters: this is the only optimization in an optimization stage. The
  `$stage-review` standard for optimization stages requires that a rejection be earned
  — "Accept rejection only when the adapted path is measured slower, fails correctness
  for an understood reason, or a minimal repro proves the op cannot express the
  required contract." Two refuted fixes on the *full* 20-bucket configuration do not
  reject the 1–2 bucket configuration, which is the one that would deliver essentially
  the whole measured benchmark win at a poisoned-range size the port already tolerates.

  Required next step: run at least one traced arm with a minimal bucket set (e.g.
  `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128`, and one with 128+96+160) through the
  same soak that broke the 20-bucket arm, and report the result. If it also corrupts,
  the wholesale rejection is earned and should cite that arm. If it does not, ship it
  and report the measured TTFT for the shipped bucket set. Additionally measure the
  freed-intermediate range per bucket, and either try or record a concrete blocker for
  a post-capture reservation buffer. If this crosses op/allocator boundaries, use
  `$autofix` as the skills require.

- **P1: A deterministic, minimal corruption reproducer is sitting in the committed
  artifacts and was never used; the README's mechanism narrative does not match it.**

  Evidence:
  - `traced_qualitative/CORRUPTED_ARM_vllm_qualitative_outputs.json`, per completion:
    p0–p3 greedy and sampled all `replacement_char_fraction = 0.0000`; **p4 sampled
    0.2841, p5 greedy 0.4180, p5 sampled 0.2054**. `logs/traced_qualitative.log` shows
    that arm ran only `qualchatrep1, qualitative, bench1` — **no sampling suite, no
    per-request seeds**, and `grep -c prefill_traces_released_for_sampling_capture
    traced_qualitative/server/server.log` = 0 (the guard never fired). So the server
    corrupted **partway through a single 12-request greedy/sampled runner arm**, after
    roughly nine ordinary requests.
  - `soak_blocking/runner_qual1/CORRUPTED_ARM_vllm_qualitative_outputs.json` reproduces
    the same onset at the same prompt index: p0–p3 clean, p4 sampled 0.2371, p5 greedy
    0.4180, p5 sampled 0.2135 — and its p5 greedy corrupt string is **byte-identical**
    to the traced_qualitative one (`'�&!)�0!*...'`, 256 chars, 0.4180 in both).
  - README *The bug this stage found* instead says "after a few dozen ordinary
    requests" and bisects to `test_seeding_and_variety.py` as the trigger, then falls
    back to "a serving process allocates continuously, from code this adapter does not
    own" as the conclusion. Neither `traced_qualitative` nor `runner_qual1` is used in
    that argument; the README's artifact table even misdescribes `traced_qualitative`
    as "qualitative1/ (healthy) and its later rounds (corrupted)" when there is exactly
    one runner round and it is *partly* healthy.

  Why this matters: a reproducer that (i) needs ~9 requests instead of a whole sampling
  suite, (ii) fires at a fixed prompt index, and (iii) produces byte-identical corrupt
  output across two different servers is not "a serving process allocates continuously".
  It is a deterministic handle that can identify which allocation collides — by bucket
  bisection, by allocator/memory-state dump around prompt p4, or by varying only the
  request that precedes the onset. The stage concluded "not expressible safely" without
  using it.

  Required next step: use `traced_qualitative`-shaped traffic as the minimal repro.
  Bisect the resident bucket set against it, dump device memory state at the onset
  step, and record which allocation lands in a captured trace's freed range — or record
  the exact blocker that prevents doing so. Then correct the README's mechanism section
  to cite this arm rather than describing the failure as sampling-suite-triggered.

- **P2: `before/serving_audit.json` is an audit of the wrong server log, and the work
  log states the opposite.**

  Evidence:
  - `before/serving_audit.json` reports `"bytes": 21262`, `"warmup_window_split": false`,
    `"benchmark_window_end_bytes": null`, `"markers": {Prefix caching…, Chunked prefill…}`
    only, `"clean": true`.
  - `before/server/server.log` is 21,347 B, mtime 11:16:43, first line
    `INFO 08-15 11:16:03` — i.e. the *accidental relaunch*. The real before arm launched
    at 11:05:38 (`before/server_excerpt.log` first line) and finished its six benchmark
    runs by 11:11:47 (`before/run6/vllm_benchmark.json` mtime).
  - `before/serving_audit.json` mtime is **13:10:54**, two hours after the overwrite —
    it was regenerated against the stub, not preserved.
  - The real audit survives only as `logs/before_audit.log` (`"bytes": 397104`,
    `degraded_markers: ["DEGRADED PATH untraced_eager_decode"]`, `clean: false` under
    the old whole-log semantics).
  - `before/server_log_size.txt` still says `388K`, contradicting the 21 KB file at
    that path.
  - `work_log.md` §2: "The excerpt, the audit and all six benchmark JSON files predate
    that and are intact." False for the audit. `work_log.md` §8 lists `before/` audit
    as "clean".
  - `logs/before_arm.log` (121 B) and `logs/before_serve_hold.log` were also overwritten
    by the relaunch, so the before arm has no surviving driver log either.

  Why this matters: `before` is the baseline half of every number in this stage. A
  committed `clean: true` audit over a 21 KB log from a different, aborted process is
  not evidence about the baseline; it is an artifact that reads as evidence.
  (Substantively the baseline looks fine — `logs/before_audit.log` shows no
  `serving_full_logits_readback` anywhere in the real 397 KB log, and
  `before/server_excerpt.log` confirms `max_model_len=131072`,
  `sample_on_device_mode=all`, `warmup_model_decode:552` i.e. the stashed
  previous-stage code — but that is not what is committed as the audit.)

  Required next step: replace `before/serving_audit.json` with the audit of the real
  log (`logs/before_audit.log` content, re-run through the new windowed tool if the
  offsets can be reconstructed, otherwise kept as-is and labelled), delete or clearly
  rename the stub `before/server/server.log`, fix `before/server_log_size.txt`, and
  correct work_log §2/§8.

- **P2: The degenerate-output stage gate (`logs/degenerate_check_all.log`) predates the
  shipped arm and never saw its output.**

  Evidence:
  - `logs/degenerate_check_all.log` mtime **12:43:03**. `after/qualitative/vllm_qualitative_outputs.json`
    and `after/qualitative/qualitative_tt_chat.json` mtime **13:50:48**;
    `after/vllm_qualitative_outputs.json` mtime 13:50:07.
  - The `after/...` rows in that log are byte-for-byte the numbers now in
    `after_prefill_traced/` (token counts 63/191/226/187/211/216/221/92/6/165/134/163 —
    I recomputed both). At 12:43 the directory named `after/` held the traced arm.
  - README Status table cites it as the shipped arm's gate: "Degenerate-output check,
    `--scope all` | **pass**, exit 0, `logs/degenerate_check_all.log`".
  - `soak_blocking/` (13:32) and `traced_qualitative/` (13:08) also postdate it.

  Why this matters: a stage gate log that was produced before the artifacts it is
  credited with checking is stale evidence, and here it also silently attributes the
  traced arm's numbers to the shipped arm. (I re-derived the check by hand over the
  current `after/` artifacts: `replacement_char_fraction` 0.0000 on all 12 completions
  and all 6 chat completions, max trigram-loop 0.0531 — so the shipped arm does pass.
  The committed evidence just is not that.)

  Required next step: re-run `check_degenerate_output.py --scope all` after the final
  arm and re-commit the log.

- **P2: `check_degenerate_output.py` does not glob `qualitative_tt_chat.json`, so the
  chat/`$qualitative-check` arm is outside the gate — and the `CORRUPTED_ARM_*`
  renaming is justified by that false claim.**

  Evidence:
  - `models/common/readiness_check/check_degenerate_output.py:329-343`, `discover()`:
    only `root.rglob("vllm_qualitative_outputs.json")` and
    `root.rglob("autoregressive_meta.json")`. No `qualitative_tt_chat.json`.
  - `logs/degenerate_check_all.log` confirms it: every scanned artifact is a
    `vllm_qualitative_outputs.json` or an `autoregressive_meta.json`; no chat artifact
    appears.
  - `work_log.md` §7, README *Evidence tools*, and all three
    `*/README.md` files beside the renamed artifacts
    (`bisect_server/qualitative3/README.md`, `fixcheck/qualitative{2,3}/README.md`,
    `soak_blocking/qualitative{2,3}/README.md`, `soak_blocking/runner_qual{1,2,3}/README.md`)
    all state: "whose discovery is an `rglob` for the exact names
    `vllm_qualitative_outputs.json` and `qualitative_tt_chat.json`". That is wrong.
  - Consequence: renaming the corrupted `qualitative_tt_chat.json` files was
    unnecessary, and the prompt-correct chat arm — the one `$qualitative-check` cares
    about — is never covered by the shared degenerate gate at all.

  Required next step: either extend `discover()` to include `qualitative_tt_chat.json`
  (which is the fix that matches the intent of the stage's own hardening) or correct
  every statement that claims it already does.

- **P2: The new `replacement_char_fraction` threshold is calibrated at a different
  granularity than it is applied, and the stage's own artifacts contain sub-threshold
  corruption.**

  Evidence:
  - `check_degenerate_output.py` applies `REPLACEMENT_CHAR_CRITICAL = 0.25` **per
    completion** (`check_completion`, `replacement = text.count(...) / len(text)`).
  - `logs/degenerate_check_negative_control.log`: on a completely corrupted server, all
    six **greedy** completions fire (0.3359–0.5120) but all six **sampled** completions
    measure 0.1871–0.2483 and **pass**.
  - `soak_blocking/runner_qual1` p4 sampled 0.2371, p5 sampled 0.2135;
    `traced_qualitative` p4 sampled 0.2841, p5 sampled 0.2054 — real corruption, three
    of those four below threshold.
  - The code comment claims "13 serving artifact sets, 10 healthy at **0.0000** and 3
    corrupted at **0.512-0.539**, so 0.25 has an order of magnitude of margin on both
    sides." I could not reproduce 0.512–0.539 at any granularity: per-set aggregates
    over the corrupted chat sets are 0.4340/0.4340/0.4829/0.5040/0.5040 and over the
    corrupted runner sets 0.0211/0.0247/0.3205/0.3237/0.3278/0.3416/0.3643/0.3945/0.3973.
    Per-completion the corrupted range is 0.187–0.617.

  Why this matters: the fix is genuinely valuable, but a server that corrupts only its
  sampled output would still pass, and the recorded calibration overstates the margin
  by an order of magnitude in the direction that matters.

  Required next step: recalibrate against the per-completion distribution actually
  present in these artifacts (or add a per-artifact-set aggregate rule), and correct
  the calibration numbers in the code comment, README and work log.

- **P2: Two `CORRUPTED_ARM_*` sets are mostly healthy, and renaming them removes the
  straddling calibration data from the gate.**

  Evidence: `soak_blocking/runner_qual1/CORRUPTED_ARM_vllm_qualitative_outputs.json`
  aggregates to 0.0211 and `traced_qualitative/CORRUPTED_ARM_vllm_qualitative_outputs.json`
  to 0.0247, with prompts p0–p3 at exactly 0.0000 in both. These are the two artifacts
  that show the onset boundary, and they are the two the gate can no longer see.

  Why this matters: the renaming mechanism itself is defensible — corrupted evidence
  should not fail a stage gate — but a filename convention is a blunt instrument that
  also removes the near-threshold cases a reviewer needs. Combined with the false claim
  about what the gate globs (above), the net effect is that `--scope all` provably
  cannot see any of this stage's corruption evidence, which is not obviously
  distinguishable from gate-dodging without reading the artifacts by hand.

  Required next step: keep the artifacts discoverable and add an explicit exclusion
  mechanism to the checker (a `--exclude` glob, or a sibling marker file the checker
  honours and reports), so the gate records "N artifact sets excluded as deliberate
  corruption evidence" rather than silently not finding them.

- **P2: The shipped arm's sampling failure set is not the baseline set, and
  `test_request_isolation` is never named.**

  Evidence (all from `FAILED` lines in the committed logs):
  - `after/sampling_tests.log` (shipped): `test_allowed_token_ids`, `test_seeding`,
    `test_same_seeds_reproduce_across_batches`, **`test_specific_seed_reproducible[0]`**,
    `test_uniform_seed_deterministic[32-1][32-0][10-1][10-0]`, 2 presence-penalty. 10.
  - `ctrl_notrace/sampling_tests.log`: same **except** it has
    **`test_request_isolation::TestBatchIsolation::test_mixed_params_batch`** and no
    `test_specific_seed_reproducible`. 10.
  - `doc/vllm_integration/README.md:451-470` defines the baseline 10 as including
    `test_mixed_params_batch` ("one of the seven reproducibility failures").
  - So the shipped arm's set is a **swap**, not the baseline set. It also appears in
    `sampling_variance/sampling1` (request_isolation, Request 5),
    `sampling_variance/sampling2` (request_isolation Request 2 + `[999]`),
    `after_prefill_traced` (request_isolation Request 1 + `[999]`) — request_isolation
    fails in 4 of the 5 full-suite runs in this stage.
  - README Status row: "62 passed, 10 failed, 1 skipped — the vLLM-integration baseline
    set exactly". False for the arm that row describes. README *Sampling suite* says
    "the vLLM-integration stage's set of 10 ... sometimes plus one further draw from the
    same class" — it is a swap, not an addition, and neither README nor work_log ever
    writes the words `test_request_isolation` or `test_mixed_params_batch`.
  - `corruption_localization.json` (the guarded run) also records
    `test_request_isolation.py rc=1` with `1 failed` and 0 passed, unmentioned anywhere.

  Why this matters: the class argument itself survives — I checked
  `test_specific_seed_reproducible` in `/home/ttuser/dev/vllm-tt-plugin/tests/tt/test_seeding_and_variety.py:151-178`
  and it runs a `max_batch_size` concurrent batch, so it is batch>1 and belongs to the
  documented class, and `test_batch1_seed_reproducible[0/1]` still passes. But the
  stage's evidence statement is inaccurate, and a reader of this README cannot learn
  that request isolation is in the failing class at all.

  Required next step: state the shipped arm's actual failure set member by member,
  name `test_mixed_params_batch`, and either reuse the previous stage's
  cross-batch-position argument for it explicitly or re-derive it here.

- **P2: The guarded corruption-localization result and the `after_prefill_traced` gates
  are evidence about the eager path, not the traced path.**

  Evidence:
  - `localize/server/server.log:1175` — the guard fired once at
    `eager_sampling_for_request_seed` and, per its own message and
    `tt/generator.py:_guard_late_sampling_capture`, "further prefill capture is switched
    off for this generator". `corruption_localization.json` shows
    `test_seeding_and_variety.py` at step 4 of 8; the four files after it therefore ran
    with **zero** prefill traces resident.
  - README: "Verified to work for what it covers: re-running experiment 4 with it in
    place gives `corruption_localization_guarded.json`, **all eight test files, model
    healthy after every one**". Steps 5–8 being healthy is the shipped default being
    healthy, not the interlock protecting a traced server.
  - Same for the arm: `after_prefill_traced/serving_audit.json` records
    `DEGRADED PATH prefill_traces_released_for_sampling_capture` in the checks window,
    and I measured `after_prefill_traced/qualitative/qualitative_tt_chat.json` and
    `after_prefill_traced/vllm_qualitative_outputs.json` at **0.0000** — healthy.
    `work_log.md` §8 says of that arm "its post-sampling qualitative is the
    corruption". That is false for the committed artifacts.

  Required next step: restate the guarded-localization result as "the guard fired at the
  first seeded file and the remaining files ran eagerly", correct work_log §8, and — if
  the interlock is to be claimed as verified for the sites it covers — add a case where
  it fires and tracing is then re-armed, or drop the "verified" framing.

- **P2: The traced arm's provenance does not match the documented reproduction command,
  and no arm ever exercised the shipped env gate.**

  Evidence:
  - `grep -c "serving prefill will be TRACED"` over all 11 arm server logs = **0**,
    including `after_prefill_traced` — yet `after_prefill_traced`, `bisect_server`,
    `fixcheck`, `localize`, `sampling_variance`, `soak_blocking` and `traced_qualitative`
    each log `prefill traces resident for padded buckets [32…1024]`.
  - Their driver logs record `prefill_trace=unset` (`logs/traced_qualitative.log`,
    `logs/soak_blocking.log`, `logs/bisect_server.log`, `logs/fixcheck.log`), and
    `logs/ctrl_notrace.log` records `prefill_trace=0`. So those arms ran a revision
    where tracing was on by default and `=0` disabled it; the current
    "default off, `=1` enables" gate shipped afterwards and has never run.
  - There is no `logs/after_prefill_traced*.log` driver log at all.
  - README *Serving configuration* presents
    `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 bash doc/optimized_vllm/bench/run_arm.sh after_prefill_traced …`
    as the command that produced the arm.
  - Logger line numbers date the arm: `after_prefill_traced` logs
    `generator:_capture_sampling_trace:1599` while the shipped file has it at 1612 and
    `soak_blocking`/`after` log 1612 — so the traced arm ran a `generator.py` 13 lines
    shorter, i.e. **before** the `blocking=True` change at `tt/generator.py:958`. The
    published 1.34x is from the non-blocking replay, not from what
    `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1` runs today.

  Why this matters: the headline not-shipped number is attributed to a code path and a
  command that did not produce it. (`soak_blocking/run1/vllm_benchmark.json` TTFT
  59.00 ms suggests the blocking replay costs nothing, but that run is from a corrupted
  server and is not cited anywhere.)

  Required next step: re-run the traced arm with the shipped code and the shipped env
  gate (at least one benchmark round, enough to confirm the warning fires and TTFT
  holds), or label the 1.34x explicitly as measured on the pre-`blocking=True`
  revision and cite `soak_blocking/run1` as the blocking-replay TTFT datapoint.

- **P2: ~400 MB of raw server logs are staged for commit, contradicting the README.**

  Evidence:
  - `du -sh doc/optimized_vllm` = **409M**; `after/server` 81M, `sampling_variance/server`
    131M, `ctrl_notrace/server` 80M, `after_prefill_traced/server` 53M,
    `localize/server` 52M.
  - `git check-ignore -v doc/optimized_vllm/after/server/server.log` →
    `models/autoports/meta_models_muse_glimmer_30b/.gitignore:7:!doc/**/*.log` — a
    negated pattern, i.e. **not ignored**. `git status --untracked-files=all` over the
    directory lists 386 paths.
  - README: "Server logs are 0.3–0.5 MB per arm and committed as `<arm>/server_excerpt.log`;
    raw `server.log` files are left on disk uncommitted." Not true as configured.
  - The previous stage put raw logs under `readiness_vllm/` precisely for this reason —
    see `.gitignore:10-15`: "server.log is deliberately NOT un-ignored because a single
    serving run produces ~81 MB of it."
  - Also staged: `after_sampling_reps/server/server.log` (an arm the work log says was
    abandoned) and `localize/server/server.log` (52 MB) with no README.

  Required next step: exclude `doc/optimized_vllm/*/server/server.log` before the
  checkpoint commit (add a `.gitignore` rule or move the raw logs out of `doc/`), and
  drop the abandoned `after_sampling_reps/` tree or document it.

- **P2: Broken artifact references in shipped docs and code.**

  Evidence (each checked for existence):
  - `README.md` cites `corruption_localization_guarded.json` twice (the bug section and
    the Artifacts table). The file is `corruption_localization.json`.
  - `tt/generator_vllm.py` (`_PREFILL_TRACE_ENV` docstring) cites
    `doc/optimized_vllm/probe_full_optimized.json`. The file is
    `probe_full_prefill_traced.json`.
  - `work_log.md` §1 and the `PREFILL_TRACE_BUCKETS` docstring cite
    `doc/optimized_full_model/ccl_host_probe.json`. Only `ccl_host_probe_bf16.json`,
    `ccl_host_probe_bfp8.json`, `ccl_host_probe_bfp8_loaded.json` exist.
  - README Status: "110 dumps". `watcher/watcher_excerpt.log` ends
    `Dump #55 completed at 545.315s` (55 dumps; 110 looks like start+complete lines
    double-counted).

  Required next step: fix the four references.

---

## Other Concerns

- **No test covers the one shipped safety interlock.** The 28 acceptance tests
  (`logs/pytest_final.log`) include three new prefill-trace tests but nothing for
  `_guard_late_sampling_capture` or `_sampling_allocates_this_step`. That helper reaches
  into sampler privates (`sampler._trace_slot`, `sampler._trace_states`,
  `sampler._penalties_active`, `seed_manager.has_active_request_seed`) behind a bare
  `except Exception: return None` — if any of those move, the guard silently stops
  guarding and nothing fails. A host-level test with a stub sampler would cover it.

- **`audit_serving.py` mixes byte offsets with character indices.** `run_arm.sh` records
  `wc -c` (bytes) into `bench_window_end_bytes.txt`, and `scan()` uses that value to
  slice `path.read_text(errors="replace")` (characters). The server logs contain
  multibyte characters (the vLLM banner box-drawing, and U+FFFD in the corrupted arms).
  The effect here is conservative — the benchmark window ends up slightly wide — and
  the `after` arm's window checks out against the driver log timeline (window ends at
  `13:45:59`, `bench6` ended `17:45:57 UTC`). Worth fixing anyway.

- **`metrics.json` folds only `before` and `after`.** The traced arm and
  `before_sweep0` are not in the folded metrics even though README and
  `perf_summary.json` quote traced-arm numbers. The per-run JSON is there and correct;
  the folded artifact just does not cover the arm the report leans on.

- **Unclassified repetition in the traced arm's sampled output.** The 12:43 gate log
  records `after/vllm_qualitative_outputs.json` (= today's `after_prefill_traced`) p0
  `sampled_completion` at `trigram_loop_fraction 0.4869` — "I cannot comply with that
  request." repeated ~30 times, just under the 0.50 advisory. The shipped arm's
  equivalent is 0.0455. Not the shipped path, but it is an unclassified anomaly in the
  arm the stage says is numerically inert.

- **`$optimize` checklist not written down.** The goal contract asks for "relevant
  `$optimize` checklist items completed with evidence". The README's *Rejected and
  deferred* table covers several implicitly, but there is no checklist mapping, so it
  is not possible to tell which items were considered and which were skipped as
  decoder-stage work.

- **`after/qualitative/qualitative_prompt_format.json` still records
  `"stage": "vllm_integration"`.** Cosmetic, but it is the provenance field of the
  `$qualitative-check` evidence for this stage.

- **`adapter_probe.py` comment overclaims.** The page-table block says the tail-cell
  change "is what lets the emitted tokens be asserted to continue unchanged across it",
  but `pt_report["ok"]` checks only refresh counts and device-table equality; tokens are
  recorded and not asserted (and they do differ between the two windows because they are
  later decode steps).

---

## Hard-Check Gaps

- No sustained soak was run on the **shipped default** configuration comparable to
  `soak_blocking` (~80 requests). The `after` arm's traffic (6+6 benchmark runs,
  sampling suite, two qualitative arms, determinism) is decent coverage and is clean,
  but the same allocator hazard exists at decode+sampling-trace scale and is inherited
  rather than retested here.
- `supports_async_decode=True` is justified by the *previous* stage's
  `--async-scheduling` arm. The decode path is unchanged, so this is reasonable, but no
  async-overlap arm was run in this stage.
- The freed-intermediate address range of a prefill trace is never measured, so "a
  52-layer prefill working set" versus "a small, decode-shaped range" — the quantitative
  core of the ship/don't-ship argument — rests on an assertion.
- Determinism/non-aligned coverage tops out at a 12345-token prompt; the 131072 context
  contract is evidenced by served `max_model_len` and `capability_report`, not by a
  long-context serving generation in this stage.

---

## Anomaly Ledger

- Observed anomaly: served output decays into U+FFFD replacement characters with prefill
  traces resident.
  Evidence: `bisect_server/qualitative3`, `fixcheck/qualitative{2,3}`,
  `soak_blocking/qualitative{2,3}` + `runner_qual{1,2,3}`, `traced_qualitative`
  (0.0211–0.5040 aggregate, 0.187–0.617 per completion);
  `corruption_localization_unguarded.json` `first_corrupting_file =
  test_seeding_and_variety.py`.
  Affected path: opt-in traced serving prefill only. Shipped default measured clean
  (`after/*` all 0.0000; `ctrl_notrace/*` all 0.0000).
  Control or comparison: `ctrl_notrace/` — same binary, tracing off, healthy either side
  of the sampling suite, and reproduces the previous stage's exact 10-failure set.
  Likely subsystem: ttnn trace/allocator lifetime
  (`tt_metal/impl/allocator/allocator.cpp:113-126`;
  `mesh_device.cpp:1315-1361` sets `allocations_unsafe_` at `end_mesh_trace` and clears
  it only when all traces are released).
  Investigation performed: 4-step in-server bisection; eager-vs-traced token identity
  (`prefill_trace_bisect.json`); two refuted fixes (`fixcheck/`, `soak_blocking/`); a
  one-way interlock.
  Resolution: **more-work-needed** — causality is established and the default is safe,
  but the deterministic ~9-request reproducer in `traced_qualitative` was not used, the
  reduced-bucket configuration was never measured, the poisoned range was never sized,
  and `$autofix` was not run. See P1 items.

- Observed anomaly: `test_request_isolation::test_mixed_params_batch` fails in 4 of 5
  full-suite runs in this stage and in the guarded localization run, and is not named
  anywhere in this stage's docs.
  Evidence: `ctrl_notrace/`, `after_prefill_traced/`, `sampling_variance/sampling{1,2}/`
  `sampling_tests.log`; `corruption_localization.json` step 7 `pytest_rc: 1`.
  Affected path: vLLM plugin sampling suite, seeded reproducibility at batch.
  Control or comparison: passes in `after/`; previous stage classified it as one of its
  seven reproducibility failures and ruled out cross-request contamination via
  cross-batch-position checks (which this stage re-ran clean: 8 concurrent, 1 distinct
  output, equal to the single request).
  Likely subsystem: shared sampler seed stream at batch > 1.
  Investigation performed: none in this stage; inherited classification not restated.
  Resolution: **more-work-needed** (documentation-level) — restate the shipped arm's
  actual failure set and carry the previous stage's classification forward explicitly.

- Observed anomaly: `after_prefill_traced` post-sampling qualitative is healthy while
  the work log says it is the corruption.
  Evidence: `after_prefill_traced/qualitative/qualitative_tt_chat.json` and
  `.../vllm_qualitative_outputs.json` at 0.0000; `after_prefill_traced/serving_audit.json`
  `markers_checks_window` contains `prefill_traces_released_for_sampling_capture`.
  Affected path: evidence description only.
  Control or comparison: `bisect_server/qualitative3` is the corrupted post-sampling arm
  the narrative describes (from the pre-guard revision).
  Likely subsystem: the one-way guard.
  Investigation performed: none.
  Resolution: **more-work-needed** — correct work_log §8.

- Observed anomaly: leading `" to=self"` / `" to=user"` on every chat completion, and
  first-divergence-from-HF at token 1–2 on all six prompts.
  Evidence: `after/qualitative/qualitative_tt_chat.json`,
  `after/qualitative/qualitative_comparison_chat.json`.
  Affected path: chat qualitative rendering.
  Control or comparison: `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`
  shows the same text with `<|message|>` present in the standalone arm and absent over
  the API; `after/determinism_vllm.json` `standalone_baseline` identical over the full
  79-char common prefix. HF control shows comparable trigram-loop and non-ascii rates.
  Likely subsystem: Harmony-style channel tokens, API-invisible special tokens.
  Investigation performed: carried from earlier stages; controls present here.
  Resolution: **controlled** — expected, evidenced.

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every
  pytest and sampling log.
  Evidence: `logs/pytest_final.log`, every `*/sampling_tests.log`.
  Affected path: process teardown.
  Control or comparison: present identically in the `before` arm and in the previous
  stage's logs.
  Likely subsystem: ttnn Python bindings teardown, not this port.
  Investigation performed: none needed.
  Resolution: **controlled** — pre-existing, unrelated to the stage.

---

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`, `.agents/skills/optimize/SKILL.md`,
  `.agents/skills/tt-enable-tracing/SKILL.md` (read in full); goal contract as supplied.
- Artifact paths (all under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/README.md`, `work_log.md`, `perf_summary.json`, `metrics.json`;
  `before/`, `before_sweep0/`, `after/`, `after_prefill_traced/` (all run1–6 benchmark
  JSON, `serving_audit.json`, `sampling_tests.log`, `qualitative/*`,
  `determinism_vllm.json`, `server_excerpt.log`, `server/server.log`);
  `probe_full_shipped.json`, `probe_full_prefill_traced.json`, `probe_trace_capacity.json`,
  `prefill_trace_bisect.json`, `corruption_localization.json`,
  `corruption_localization_unguarded.json`;
  `bisect_server/`, `ctrl_notrace/`, `fixcheck/`, `soak_blocking/`, `traced_qualitative/`,
  `localize/`, `sampling_variance/`, `after_sampling_reps/`;
  `logs/` (pytest_final, pytest_watcher, degenerate_check_all,
  degenerate_check_negative_control, before_audit, all `*_arm.log`/driver logs,
  probe_trace_capacity); `watcher/watcher_excerpt.log`; `bench/*.sh`, `bench/*.py`;
  `doc/vllm_integration/README.md` + `probe_full_fixed.json`;
  `doc/datatype_sweep/evidence_perf.json`; `doc/context_contract.json`;
  `doc/optimized_full_model/prefill_trace_probe{,_8192}.json`;
  `models/autoports/meta_models_muse_glimmer_30b/.gitignore`.
- Code paths: `tt/generator.py`, `tt/generator_vllm.py`, `tt/model.py`,
  `tests/test_full_model.py`, `doc/vllm_integration/bench/{audit_serving,adapter_probe}.py`,
  `models/common/readiness_check/check_degenerate_output.py` (all via `git diff HEAD`);
  `tt/functional_decoder.py` `_chunk_page_table`/`_page_table_row`;
  `tt_metal/impl/allocator/allocator.cpp:95-140`;
  `tt_metal/distributed/mesh_device.cpp:1295-1375`;
  `/home/ttuser/dev/vllm-tt-plugin/tests/tt/test_seeding_and_variety.py`.
- Commands run (all read-only; no server, device, or hardware use):
  `git status/diff/check-ignore/ls-files`, `find`, `stat`, `du`, `wc`, `grep`, `sed`,
  and small Python scripts that recomputed per-run medians from
  `*/run*/vllm_benchmark.json`, recomputed `replacement_char_fraction` and trigram-loop
  metrics over every qualitative artifact, diffed probe token sequences against
  `probe_full_fixed.json`, and cross-checked README-cited paths for existence.

---

## Residual Risk

- The shipped default carries the same allocator/trace-lifetime hazard at decode+sampling
  scale, inherited from optimized-full-model. It is unmeasured beyond the `after` arm's
  ~20 minutes of traffic, and this stage has now demonstrated the failure is silent.
- `_guard_late_sampling_capture` depends on four private attributes of the shared
  sampler and fails open (`except Exception: return None`). A change in
  `models/common/sampling/generator.py` disables it silently.
- The `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1` path as currently written has never been run
  end to end (no arm emitted its warning) and its replay is now blocking, unlike the arm
  the 1.34x came from. Anyone enabling it is running an unmeasured configuration.
- If the artifact tree is committed as-is, ~400 MB of raw server logs enter git history
  and cannot be cheaply removed later.
- `--scope all` degenerate checking currently cannot see any chat-arm output or any of
  this stage's corruption evidence, so a future regression of the same shape would again
  pass the gate.
