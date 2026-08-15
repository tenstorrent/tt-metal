# Stage Review (round 4)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the three previous reviews
(`stage_review.md`, `stage_review_round2.md`, `stage_review_round3.md`).
Worktree live, uncommitted (8 modified files + untracked `doc/optimized_vllm/`).

Verdict: more-work-needed

**Round 3's P1 is genuinely resolved.** I re-derived all 30 rows of
`prefill_trace_discriminators.json` from the ten `probe_*.json` files with my own
common-prefix comparator against `doc/vllm_integration/probe_full_fixed.json`: every
`matches_vllm_integration_reference` value reproduces exactly, including the `null`s where the
reference has no such length. Claims (a)–(d) all follow from the artifacts:

* **(a) same-revision non-monotonicity.** `probe_disc_20bucket.json` ran at 16:31; the last
  edit to `tt/generator_vllm.py` was 15:57 and to `tt/generator.py` 15:31. Its loguru
  fingerprints (`generator_vllm:allocate_kv_cache:566`, `_capture_prefill_traces:767/784`,
  `generator:_capture_prefill_trace:1048`, `build_generator:2452`) match the *current* source
  line-for-line, which I checked by reading those exact lines. 4097 is correct there, after
  three traced replays (128/100/37 all `trace_replays 1`). The cross-revision datum is retired.
* **(b) capture alone is sufficient.** `probe_disc_4097only_traced.json` has
  `buckets_resident [128]`, `capture_failures 0`, a single request, and
  `prefill_counters.trace_replays 0` — and is wrong (`[576, 5824, 761, 426, 426, …]`,
  `distinct_tokens 4`). `_capture_prefill_trace` (`tt/generator.py:998-1050`) does not replay
  after `end_trace_capture`, and the probe's multi-request section runs *after* the
  single-length section, so nothing replayed the bucket before the wrong logits. The
  "corrupted once a trace is executed" reading is correctly ruled out.
* **(c) not an unwarmed shape.** `PREFILL_WARMUP_LENGTHS = (32, 96, 128, 160, 256, 512, 1024,
  8192)` (`tt/generator_vllm.py:145`) — 8192 is warmed, and `probe_disc_8192_traced.json`
  differs from `probe_disc_8192_eager.json` at token 0.
* **(d) not bucket-128-specific.** `probe_disc_bucket96.json` has `buckets_resident [96]`, and
  **every** single-length request in it is `trace_replays 0` (128 and 100 pad to 128, 37 pads
  to 64, none of which is a bucket) — so that run contains no traced request at all, and 4097
  is still wrong.

Token comparisons are common-prefix throughout (9-token disc probes vs 17-token reference),
and the JSON's `what` field says so. The four headline arms all reproduce from raw
`run<N>/vllm_*benchmark.json` (median of runs 4–6), to the last digit, including the run
ranges and every derived percentage. The loop table reproduces exactly under my own
independent implementation of the stated metric (3/12, 3/12, 0/6, 0/6). Acceptance tests, the
watcher retry, the audit windows, the renamed logs, the corrected `server_log_size.txt` and
`serving_audit.json` path fields, the soak citations and the added commands all check out.

**"Ship tracing off" is now properly earned, and the unexplained mechanism is honestly
labelled** in `README.md` ("This explains the 20-bucket decay and nothing else… Two failures,
one confirmed mechanism, and one open question is the honest count") and in
`perf_summary.json.blocker.mechanism_status` ("UNEXPLAINED for the one-bucket case").

What does not hold up is a cluster of claims that the artifacts contradict — one of them in the
section that certifies the measured path against the goal contract, two of them in shipped
source, and three of them in numbers the same README recomputed correctly one section earlier.
None of them changes the ship decision; all of them are statements a future reader will act on.

---

## Required Work

- **P1: The *Contract evidence — the measured path* section certifies the shipped path with
  the wrong probe twice, and asserts something the stage's own central finding disproves.**

  Evidence:
  - `README.md:480-486`: "**Bit-identity against the before code.** Both probes produce
    **identical token sequences** to `doc/vllm_integration/probe_full_fixed.json` for every
    shared prompt length (128, 37, 4097) … For `probe_full_shipped.json` that says this
    stage's changes are numerically inert". `probe_full_shipped.json`'s 4097 request is
    `[576, 5824, 761, 426, 426, 426, 426, 426]`; the reference is
    `[198, 6453, 107177, 38, 372, 2556, 27326, 11974]`. It does **not** match, and that
    mismatch is the stage's entire "and then it changes a request that does not" finding, 200
    lines above in the same file. `work_log.md:173-176` repeats the same false claim and
    additionally calls `probe_full_shipped.json` "(shipped default)", which the README itself
    corrects two paragraphs earlier as misnamed.
  - `README.md:456-463`: the section opens "From `probe_repro_eager.json` … **in the shipped
    configuration**, i.e. tracing off", then quotes "**Steady-state counters**, 16 multi-slot
    decode steps … `trace_replays 16 … readbacks 16 … sampling_param_reuses 15`". Those are
    the 16-step *traced* probes. `probe_repro_eager.json`'s own multi-request block is
    `trace_replays 8, token_refreshes 1, position_refreshes 1, page_table_refreshes 1,
    synchronizations 0, readbacks 8, sampling_param_refreshes 1, sampling_param_reuses 7` —
    equally good evidence for the contract item, and not the numbers printed.
  - `README.md:587`, the `$optimize` checklist row "Decode path fully traced, no host
    fallbacks", still cites `probe_full_shipped.json` — round 3 asked for this row
    specifically and only the prose header moved.

  Why this matters: this is the section the goal contract's async-split / persistent-trace-input
  / on-device-sampling / bit-identity items are evidenced from. As written, two of its three
  quantitative claims are sourced from a non-shipped configuration and one of them is factually
  false about the artifact it names. `$stage-review` requires more work when "a required check,
  artifact, metric … from the goal contract is … stale, or contradicted by another artifact".

  Required next step: quote `probe_repro_eager.json`'s own counters (8/1/1/1/0/8, 1 refresh +
  7 reuses) under the shipped-configuration header; restate bit-identity as what the artifacts
  say — every shared length matches for `probe_full_prefill_traced.json` and for the eager
  probes, and `probe_full_shipped.json` matches at 128/37 and at the three-slot multi-request
  section but **not** at 4097; fix the checklist row and `work_log.md` §5.

- **P2: The shipped `tt/generator_vllm.py` docstrings still publish the mechanism the new
  probes refute, and still carry the stale 20-bucket bucket-set docstring (third round).**

  Evidence:
  - `_PREFILL_TRACE_ENV` (`tt/generator_vllm.py:199-241`) quotes the allocator rule and then
    says "**Two independent failures were measured**, at both ends of the bucket ladder",
    presenting both under it; and for the 1-bucket case: "correct with tracing off and correct
    with 20 buckets resident, **i.e. it depends on which *other* requests were traced**". The
    4097-ALONE and bucket-96 probes contain *no traced request at all* and still fail, so that
    attribution is refuted by this round's own evidence. It cites only
    `probe_repro_{traced,eager}.json`, never `prefill_trace_discriminators.json`, never
    mentions 8192, and never says UNEXPLAINED — the word the README and `perf_summary.json`
    now use. Round 3's required next step was "either state a mechanism consistent with … or
    record it explicitly as unexplained"; that was done in the report and not in the code.
  - `PREFILL_TRACE_BUCKETS` (`tt/generator_vllm.py:147-183`) still stacks two docstrings. The
    first (147-162) is the 20-bucket-era text — "these are exactly the short padded lengths
    `PREFILL_WARMUP_LENGTHS` already compiles" (false for `(128,)`) and "8192 is deliberately
    absent" — and runs straight into the replacement block at 163 with no separator. Flagged in
    round 2 and round 3; unfixed. It now also reads as a *contradiction* of the new finding,
    since 8192's absence from the bucket set is discussed as a throughput trade while 8192 is
    the second length the capture corrupts.

  Required next step: rewrite both docstrings against the discriminator matrix — capture (not
  replay) is the trigger, it is not request-ordering-dependent, it hits long eager prefills
  including the warmed 8192 shape, the mechanism is open — and delete the superseded
  147-162 block.

- **P2: Three surfaces still say the runner arm loops on 2 of 12, contradicting the README's
  own recomputed table, and the accompanying description matches no artifact in the tree.**

  Evidence (my own recomputation of "longest 4–80-word block repeating ≥2×, non-overlapping
  coverage > 0.40" agrees with `loop_classification.json` exactly):
  - `README.md:505` (*Qualitative*): `after/` 3/12 — p0 sampled 0.529, p1 greedy 0.708,
    p2 greedy 0.938. Correct.
  - `README.md:360` (Status): "2 of its 12 completions loop a 40-word sentence (**p0 sampled
    x26**, p2 greedy x6)". `after/` p0 sampled is a 12-word block x3; the x26/x32 completion
    belongs to `after_prefill_traced_1bucket/`; no arm has a 40-word block. p1 greedy (the
    80-word restatement round 2 named explicitly) is still missing from this row.
  - `README.md:675` (limitation 9): "(2/12, pre-existing)".
  - `perf_summary.json:85` (`named_limitations[4]`): "exhibits on **2 of 12** completions".

  Required next step: propagate 3/12 and the correct per-completion description to the Status
  row, limitation 9 and `perf_summary.json`.

- **P2: README limitation 2 attributes an interlock firing to the shipped arm, which README §4
  and the arm's own audit both say never happens.**

  Evidence: `README.md:654-655` — "When the guard does fire, that server's TTFT reverts to the
  eager figure for the rest of its life — **which is what happens in the `after/` arm's
  sampling stage**." `after/serving_audit.json` contains zero
  `prefill_traces_released_for_sampling_capture` markers (I counted over the whole file);
  the marker appears in `after_prefill_traced_1bucket/` and `after_prefill_traced/` only, in
  `degraded_markers_checks_window_expected`. `README.md:153` says the opposite in the same
  document: "It never fires in the shipped arm, which has no prefill traces to release."
  This is the same rename-induced misattribution round 3 raised for §4, surviving in
  limitation 2.

  Required next step: point that sentence at the traced arms.

- **P2: The six discriminator probes — now the stage's load-bearing evidence — have no recorded
  commands anywhere in the tree.**

  Evidence: `README.md`'s *Serving configuration* records the `probe_repro_{traced,eager}`
  pair only. `logs/probe_discriminate.log` and `logs/probe_warm.log` contain nothing but the
  labels `A:`…`F:` and `RC=0`; there is no driver script under `bench/` (contents:
  `collect_metrics.py`, `localize_corruption.py`, `prefill_trace_bisect.py`, `run_arm.sh`,
  `serve.sh`, `soak_traced_bucket.py`). So the exact invocations for 20 buckets, bucket 96,
  4097-alone and 8192-alone (env vars, `--prompt-lens`, `--decode-steps`, `--out`) exist only
  in the shell history. The goal contract requires the README/work log to record commands, and
  round 3 asked for "the probe commands".

  Required next step: add the four remaining probe command lines (or commit the driver script)
  next to the pair already recorded.

- **P2: `work_log.md` §6 still narrates the void soak as the qualifying experiment, and "~9
  requests" survives twice.**

  Evidence:
  - `work_log.md:249-257` presents `soak_1bucket/` as "**84 completions**, every one at
    `replacement_char_fraction` 0.0000, on the exact traffic pattern that corrupted the
    20-bucket arm", with no mention that it is void; the correction lives only in the §8 table
    and in the README's bisection row 5. Round 3 required the §6 narrative to be corrected.
    (Confusingly, `soak_traced_bucket/` also reports 84 generations, so the two are easy to
    conflate.)
  - `work_log.md:303` and `work_log.md:363` still say "~9 requests"; the README was corrected
    to "the 22nd generation" everywhere. Round 3 named `work_log.md` §8 specifically.

  Required next step: mark the §6 `soak_1bucket/` paragraph void at the point it is narrated,
  and reconcile the two onset figures (either state both frames — 9th request within the arm,
  22nd generation on the server — or use one).

---

## Other Concerns

- **The matrix's ✅ cells overstate two columns.** `README.md:293-298` heads its columns
  "✅ match", but the 100- and 8192-token columns have no entry in
  `doc/vllm_integration/probe_full_fixed.json`; the underlying rows correctly record
  `matches_vllm_integration_reference: null`. `prefill_trace_discriminators.json`'s
  `conclusions[1]` likewise folds 8192 into "do not [match]" when the actual 8192 comparison is
  traced-vs-eager *within this stage*. Worth one clause: "for 100 and 8192 the comparison is
  against this stage's tracing-off control, not the committed reference."
- **The 8192 artifact contains an unread clue.** The probe prompt is
  `torch.arange(1000, 1000+len)` (`doc/vllm_integration/bench/adapter_probe.py:247`). With the
  bucket resident, 8192 returns `[1767, 1330, 1331, 1332, 1333, 1334, …]` — a counting
  continuation starting ~330 tokens into a 8192-token prompt — while the eager control returns
  the same head as every other long prompt. That looks like a prefill that saw a truncated
  prompt, which is a sharper hypothesis than "unexplained" and costs nothing to state.
- **The obvious remaining discriminator was not run.** Every failing configuration has a
  *largest* captured bucket of 96 or 128; the passing one has 1024. "Not monotone in the trace
  count" is literally true of the data, but a single bucket at `[1024]` (or `[128,1024]`)
  would separate "how many traces" from "how large the largest captured trace is", and that
  distinction is exactly what the upstream ask in `README.md:339-344` needs. One probe
  invocation, no server.
- **`perf_summary.json:59`** still reports burst-before `decode_tps_u` 43.407; the raw median
  of runs 4–6 of `mean_tpot_ms` is 23.0387 → 43.4052, which is what `metrics.json` and the
  README say. Round-3 finding, unfixed.
- **`README.md:627`** cites `stage_review.md` as "independent stage review"; rounds 2 and 3
  (and this one) exist in the same directory and are not listed.
- **`work_log.md:376`** "metrics.json folds every arm above" — it folds 7 arms
  (`before`, `after`, `after_prefill_traced_1bucket`, `after_prefill_traced`, `before_sweep0`,
  `soak_1bucket`, `soak_traced_bucket`) against a 10-row table. Unchanged from round 3.
- **`README.md:359`** "character-identical to the standalone model over the full common
  prefix" is still the 79-character / `max_tokens 24` comparison in
  `after/determinism_vllm.json.standalone_baseline`; the stronger 127-token comparison in
  `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json` is still not the one cited.
  Carried unchanged from rounds 2 and 3.
- **`$autofix` is still unmentioned** in both documents (`grep -c` → 0, 0), and the
  ballast-buffer mitigation remains the named-but-untried option while the mechanism it
  targets is now recorded as not being the one that explains the 1-bucket failure.
- **`README.md:614`** has a stray blank line inside the Artifacts table, splitting it into two
  tables of which the second has no header row.

## Hard-Check Gaps

- The freed-intermediate address range of one prefill trace is **still unmeasured** (round 1,
  2 and 3 ask). `doc/optimized_full_model/prefill_trace_probe.json` gives
  `capture_retained_dram_bytes 3280896` and no peak-during-capture reading, so "twenty 52-layer
  prefill working sets" versus "a small, decode-shaped range" is still an assertion — and it is
  the quantitative core of the one mechanism the stage does claim.
- Nothing measured between 2 and 19 buckets; nothing at a single large bucket (above).
- No live-server evidence of the 4097/8192 divergence in any configuration. The server-side
  non-aligned check sends 4097 and 8193 and stores `text_head`, but it runs after the sampling
  step that releases the traces, so it cannot see it. One step reorder in `bench/run_arm.sh`
  would give a live-server datum; still not taken.
- The shipped headline arm (`after/`, 13:39) predates the final code by ~2 h 20 m. Both deltas
  are in the traced-prefill path, which is off in that arm; the final code is covered by 29
  passing acceptance tests (16:01) and by `probe_repro_eager.json` token-identity. Unchanged
  and acceptable, but the headline arm itself was not re-run on the shipped bits.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm;
  the decode path is unchanged by this stage.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` (verified
  in `after/server_excerpt.log`) and `doc/context_contract.json`. Unchanged from rounds 1–3.

## Anomaly Ledger

- Observed anomaly: with a prefill trace **captured** (not necessarily replayed) for one small
  bucket, long eager prefills (4097 → padded 4128, and the warmed 8192) diverge from their
  first token.
  Evidence: `prefill_trace_discriminators.json` and the ten underlying probes; all 30 rows
  re-derived independently here. Failing: `probe_repro_traced`, `probe_full_shipped`,
  `probe_disc_bucket96`, `probe_disc_4097only_traced`, `probe_disc_8192_traced`. Passing:
  every tracing-off control, and `probe_disc_20bucket` on the shipped revision.
  Affected path: eager prefill of a prompt outside the traced buckets, in a process that has
  captured at least one small prefill trace.
  Control or comparison: same-revision tracing-off controls at both lengths ✓; same-revision
  20-bucket comparison ✓ (round 3's cross-revision objection retired); no live-server control.
  Likely subsystem: ttnn mesh trace capture / allocator lifetime, or port-owned per-bucket
  persistent capture state (`entry["tokens"|"page_table"|"logits"]`). Unknown.
  Investigation performed: six discriminating probes isolating capture-vs-replay, bucket
  identity, warmed-shape and trace count; every request tabulated.
  Resolution: **controlled** — the configuration does not ship at any bucket count, the
  matrix is in the tree, the mechanism is labelled UNEXPLAINED in the README and
  `perf_summary.json`, and an upstream ask is written. Residual documentation work is P1/P2
  above (the shipped docstring still carries the refuted attribution).

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `prefill_trace_bisect.json` token-identical (not math); `soak_traced_bucket/` clean at 1
  bucket over 84 in-bucket generations.
  Likely subsystem: trace/allocator lifetime.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, a valid in-bucket soak.
  Resolution: **controlled** — does not ship; reproducer, refutations and interlock in tree.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm.
  Evidence: `after/vllm_qualitative_outputs.json` p0 sampled 0.529, p1 greedy 0.708, p2 greedy
  0.938 — reproduced here independently, matching `loop_classification.json`.
  Control or comparison: `readiness_vllm/` 3/12 with two identical coverages; chat verdict arm
  0/6; HF control 0/6. Pre-existing and prompt-shaped.
  Investigation performed: classified, metric written down, recomputed per arm.
  Resolution: **controlled** for the verdict; the count is still misreported in three places
  (P2 above).

- Observed anomaly: the qualifying soak's completions are the `" to=self"` analysis channel
  restating the question.
  Evidence: `soak_traced_bucket/soak_traced_bucket.json` head fields.
  Control or comparison: `after/qualitative/qualitative_tt_chat.json` shows the same channel
  prefix; classified in earlier stages as Harmony-style channel tokens invisible over the API.
  Resolution: **controlled** (weaker readable-text evidence than described; unchanged from
  round 3).

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest
  and sampling log.
  Evidence: `logs/pytest_final.log`, `*/sampling_tests.log`.
  Control or comparison: identical in the before arm and previous stages.
  Resolution: **controlled**.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `stage_review.md`, `stage_review_round2.md`, `stage_review_round3.md` (round 3's
  required items re-derived one by one);
  `.agents/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh`.
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,metrics.json,loop_classification.json,prefill_trace_discriminators.json}`;
  all ten `doc/optimized_vllm/probe_*.json` and their logs;
  `doc/vllm_integration/probe_full_fixed.json`; every `run<N>/vllm_benchmark.json` and
  `run<N>/vllm_ci_serving_benchmark.json` in `before/`, `after/`,
  `after_prefill_traced_1bucket/`, `after_prefill_traced/`; every `serving_audit.json`,
  `server_log_size.txt`, `sampling_tests.log`, `server_excerpt.log`;
  `soak_traced_bucket/soak_traced_bucket.json`; every arm's `vllm_qualitative_outputs.json`,
  `after/qualitative/qualitative_tt_chat.json`,
  `doc/full_model/qualitative/qualitative_hf_chat.json`, `readiness_vllm/`;
  `logs/` (run_tests, run_watcher, pytest_final, pytest_watcher, degenerate_check_all,
  probe_disc_*, probe_discriminate, probe_warm, probe_pair, the relabelled
  `after_prefill_traced_1bucket_*` set); `doc/context_contract.json`; `.gitignore`.
- Code paths: `tt/generator_vllm.py` (`PREFILL_WARMUP_LENGTHS`, `PREFILL_TRACE_BUCKETS`,
  `_PREFILL_TRACE_ENV`, `_prefill_trace_buckets`, `_prefill_trace_enabled`,
  `_capture_prefill_traces`, `warmup_model_prefill`, `capability_report`),
  `tt/generator.py` (`_capture_prefill_trace`, `_kv_cache_signature`, the three
  `ttnn.execute_trace` call sites), `tests/test_full_model.py` (test inventory),
  `doc/vllm_integration/bench/adapter_probe.py` (prompt construction, section ordering),
  `doc/optimized_vllm/bench/`.
- Commands run (all read-only; no server, device, hardware or vLLM use): `git status/log/
  check-ignore`, `ls`, `stat`, `grep`, `sed`, `head`/`tail`, and Python scripts that
  re-derived every discriminator row against the committed reference over common prefixes,
  recomputed warm medians and deltas for all four arms and both profiles, recomputed the
  long-verbatim-loop coverage over six qualitative artifacts, cross-checked audit markers and
  path fields, and matched loguru line fingerprints against the current source.

## Residual Risk

- The one-bucket mechanism is genuinely open, and the shipped `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1`
  docstring still tells a reader it is about *other traced requests*. A deployment following
  that text would soak its traced buckets and never test the lengths that actually break.
  (The runtime warning at `_prefill_trace_enabled` does say "including prompts outside the
  traced buckets", which is the right instruction; the docstring above it is not.)
- The suspected variable may be the size of the largest captured trace rather than the count;
  untested, and it changes what the upstream ask should say.
- The shipped headline arm was measured on code ~2 h older than what ships, with the deltas
  confined to the disabled path.
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops on
  every model, disclosed as limitation 9.
