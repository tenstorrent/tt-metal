# Stage Review (round 5)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the four previous reviews
(`stage_review.md`, `stage_review_round2.md`, `stage_review_round3.md`,
`stage_review_round4.md`). Worktree live, uncommitted (8 modified files + untracked
`doc/optimized_vllm/`).

Verdict: more-work-needed

**Every one of round 4's six required items is genuinely resolved, and I checked each
against the artifacts rather than the prose:**

1. **P1 (contract evidence certified with the wrong probe).** `README.md:438-508` now opens on
   `probe_repro_eager.json` and quotes *its* counters. I read them out of the file:
   `multi_request.decode_counters = {trace_replays 8, token_refreshes 1, position_refreshes 1,
   page_table_refreshes 1, synchronizations 0, readbacks 8}` and
   `decode_serving_counters = {sampling_param_refreshes 1, sampling_param_reuses 7}` — exactly
   what README:470-474 prints. The 16-step claim (`16 / 1,1,1 / 0 / 16 / 1+15`) reproduces from
   `probe_full_prefill_traced.json` and `probe_full_shipped.json`. The bit-identity paragraph
   (496-508) now says what the artifacts say: `probe_repro_eager.json` matches
   `doc/vllm_integration/probe_full_fixed.json` at 128/37/4097 and on the three-slot section
   (verified), tracing-on matches at the short lengths and the multi-request section (verified),
   and `probe_full_shipped.json` diverges at 4097 and that is the central finding. The
   `$optimize` checklist row (`README.md:609`) and `work_log.md:173-178` are fixed.
2. **P2 (refuted mechanism in shipped docstrings).** The duplicate `PREFILL_TRACE_BUCKETS`
   docstring is gone — `tt/generator_vllm.py:147-166` is now one block, with no "exactly the
   short padded lengths `PREFILL_WARMUP_LENGTHS` already compiles" and no "8192 is deliberately
   absent". `_PREFILL_TRACE_ENV` (182-220) no longer says "depends on which *other* requests
   were traced", cites `prefill_trace_discriminators.json`, names 8192 as warmed, and labels the
   one-bucket case **unexplained**.
3. **P2 (2/12 vs 3/12).** Status row (368), limitation 9 (700) and
   `perf_summary.json:85` all say 3 of 12 with per-completion coverages. I recomputed the
   metric independently (longest 4–80-word block repeating ≥2×, non-overlapping coverage
   > 0.40) over all six artifacts and reproduced `loop_classification.json` exactly:
   `after/` 3/12 (p0 sampled 0.529 12-word×3, p1 greedy 0.708 80-word×2, p2 greedy 0.938
   33-word×6), `readiness_vllm/` 3/12, chat 0/6, HF 0/6, `after_prefill_traced/` 5/12.
4. **P2 (limitation 2 arm misattribution).** `README.md:676-679` now points at the two traced
   arms and states the shipped arm never fires. `after/serving_audit.json` still contains zero
   release markers; they appear only in the traced arms.
5. **P2 (discriminator commands unrecorded).** `bench/run_discriminators.sh` exists, documents
   what each probe separates, and is cited at `README.md:428`. I checked every invocation in it
   against the probe it produced: bucket sets, `--prompt-lens` and `--decode-steps` all match
   the corresponding JSON's `prefill_trace.buckets_resident` and request list.
6. **P2 (work_log soak_1bucket / "~9 requests").** `work_log.md:251-257` marks `soak_1bucket/`
   void at the point it is narrated. One "~9-request" survives (see *Other Concerns*).

**The whole matrix still re-derives.** I recomputed all 30 rows of
`prefill_trace_discriminators.json` from the eleven `probe_*.json` files with my own
common-prefix comparator against `doc/vllm_integration/probe_full_fixed.json`: every
`path`, `matches_vllm_integration_reference`, `first_tokens` and `distinct_tokens` value
reproduces exactly, including the `null`s. **All four headline arms reproduce to the last
digit** from raw `run<N>/vllm_*benchmark.json` medians of runs 4–6 (TTFT 81.477 → 77.419,
t/s/u 43.4802 → 43.4278, TPOT, ITL, throughput, E2E, run ranges, 62.965/60.662, 812.096/805.384,
1.294x/1.343x, +12.5 %/+11.6 %, 100.2 % of `evidence_perf.json`'s 23.078 ms / 43.331 t/s/u).
Round 4's `perf_summary.json:59` nit (43.407) is fixed to 43.405. Gate logs check out: 29
passed plain and under the watcher, 62/10/1 sampling with the exact ten names README lists,
degenerate check exit 0 with only the deliberately-corrupted diagnostic dirs excluded, audit
benchmark window clean with no surviving processes, non-aligned 9/9, served
`max_model_len=131072` against a 131072 contract.

**New this round, and it closes round 4's staleness question:** the 17:28 edit to
`tt/generator_vllm.py` is provably comment/docstring-only. I compiled the current source and
compared it to `tt/__pycache__/generator_vllm.cpython-312.pyc` (16:28, i.e. before every
discriminator probe): 29 code objects, zero differences in bytecode, names or varnames. So the
entire discriminator series (16:31–17:27) ran on code semantically identical to what ships, and
the "same revision" premise of the size comparison holds.

What does not hold up is the load-bearing new probe itself. `probe_disc_bucket1024.json` is
the artifact the round-5 conclusion, the rewritten shipped docstring and the rewritten upstream
ask all rest on, and the one request that configuration actually traced looks like the failure
it is being used to rule out — with no control anywhere in the tree. And the variable the stage
has now identified names a bucket set that would keep the 1.29x, which was never measured.

---

## Required Work

- **P1: The decisive `probe_disc_bucket1024.json` has an uncontrolled in-bucket output carrying
  this stage's own corruption signature, and the report never mentions it.**

  Evidence:
  - `probe_disc_bucket1024.json` contains exactly two requests. The 4097 one is eager, replays
    0, and matches the reference (this is the datum the conclusion uses). The other is the
    **1024-token prompt, `path: traced`, `trace_replays 1`**, and its tokens are
    `[84, 198, 2223, 6453, 2223, 6453, 2223, 6453, 2223]` — `distinct_tokens 4`, a 2-token
    cycle from position 2.
  - Across all 30 matrix rows, `distinct_tokens == 4` occurs in exactly **five**: the four
    known-wrong 4097 rows (`[576, 5824, 761, 426, 426, 426]`, `matches…: false`) and this one.
    **Every row the matrix certifies correct has `distinct_tokens` 9, 12, 16 or 17.** That is
    the stage's own signature for a bad prefill.
  - There is **no control**. I scanned all eleven probes: `probe_disc_bucket1024.json` is the
    only artifact in the tree containing a 1024-length request. No eager 1024, no 1024 at 20
    buckets, no 1024 at bucket 128. (1024 *is* in `PREFILL_WARMUP_LENGTHS`, so it is not an
    unwarmed shape.)
  - `prefill_trace_discriminators.json:392-410` records the row with
    `matches_vllm_integration_reference: null` and the note "one LARGE bucket - separates trace
    size from trace count", with no remark on its content. `README.md:293-299`'s matrix has **no
    1024 column at all**, so the only request that configuration traced is invisible in the
    report. `README.md:301-304`, `work_log.md:278-281`, `perf_summary` (absent), the shipped
    docstring `tt/generator_vllm.py:203-210` and the upstream ask `README.md:345-352` all lead
    with "1024 does not [break it] … the one that makes the rest cohere".
  - I did rule out the two obvious confounds in the size comparison, so the problem is
    specifically this row and not the comparison design: the [1024] run is the only passing
    single-bucket run in which the trace was *replayed* before the 4097 request, but
    `probe_repro_traced.json` replays the [128] trace twice before its 4097 request and still
    fails, so "a replay repairs it" is refuted; and `probe_disc_20bucket.json` passes with only
    short preceding requests, so "a preceding long prefill repairs it" is refuted. Held at
    count = 1, largest-bucket 96 and 128 fail and 1024 passes on the 4097 row.

  Why this matters: the round-5 re-framing — "the discriminator is the *size* of the largest
  captured trace", the new upstream ask "why does a *larger* captured trace make it stop?", and
  the shipped `MUSE_GLIMMER_VLLM_PREFILL_TRACE` docstring a deployment will read — all require
  that the [1024] configuration is clean. An equally simple reading of the same artifact is that
  at bucket 1024 the damage lands on the **traced** request instead of on the later eager one,
  in which case "size" is the wrong variable and the upstream ask points at the wrong thing.
  `$stage-review`'s anomaly rule is explicit: an anomaly visible in model output with no control
  showing it is expected is required work.

  Required next step: one probe invocation, no server —
  `MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 python doc/vllm_integration/bench/adapter_probe.py
  --prompt-lens 1024,4097 --decode-steps 8` — as the eager control for the 1024 row, and
  ideally `--prompt-lens 1024` at bucket `[128]` for the third corner. Add the rows to
  `prefill_trace_discriminators.json` and the invocations to `bench/run_discriminators.sh`, and
  add a 1024 column to the README matrix. If the eager 1024 output is the same, say so
  explicitly and the size conclusion stands as written; if it is not, retract the size
  conclusion from `README.md`, `work_log.md`, `tt/generator_vllm.py` and the upstream ask, and
  state what the matrix actually supports.

- **P1: The new discriminator names a bucket set that would keep the measured 1.29x, and that
  set was never measured — so "rejected at any bucket count" is no longer earned by the
  evidence as re-framed.**

  Evidence:
  - Failing configurations all have largest resident bucket 96 or 128
    (`probe_repro_traced`, `probe_full_shipped`, `probe_disc_bucket96`,
    `probe_disc_4097only_traced`, `probe_disc_8192_traced`); passing ones have 1024
    (`probe_disc_bucket1024`, `probe_disc_20bucket`, `probe_full_prefill_traced`). That is the
    stage's own conclusion (`prefill_trace_discriminators.json:conclusions[1]`).
  - The 1.29x TTFT and +12.5 % burst throughput come from bucket **128**
    (`after_prefill_traced_1bucket/`, re-derived here: 62.965 ms, 812.096 tok/s).
  - A set containing 128 **and** 1024 — two traces, largest 1024 — is precisely the
    configuration the new variable predicts is correct on long eager prefills *and* keeps the
    win, and it is nowhere in the tree. `bench/run_discriminators.sh` runs `[1024]` alone.
    Round 4's *Other Concerns* named `[128,1024]` explicitly; only `[1024]` was run.
  - `README.md:594` concedes "Counts between 2 and 19 are unmeasured", and `README.md:328-334`
    nonetheless rejects tracing "at any bucket count" on the ground that "every configuration
    measured is wrong somewhere" — which, for `[1024]`, is true only if the P1 above resolves
    against the stage.

  Why this matters: `$optimize` and this skill require rejections to be earned, and specifically
  call out rejecting a family without measuring the compatible combination. 1.29x–1.34x of TTFT
  and +12 % of serving-burst throughput are the stage's entire optimization payload.

  Required next step: run the probe at
  `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128,1024 --prompt-lens 128,100,37,4097` (one
  invocation, no server). If 4097 is correct there, either qualify the configuration with an
  in-bucket soak or record explicitly that the unmeasured 2..19 count hazard is what still
  blocks it — the rejection then rests on a named, measured gap rather than on a claim the
  matrix contradicts. If 4097 is wrong there, the size conclusion needs revision and the
  blanket rejection is immediately earned.

- **P2: `perf_summary.json` was not updated to the round-5 finding and now contradicts the
  matrix it cites.**

  Evidence:
  - `perf_summary.json:115` `blocker.not_monotone_in_trace_count`: "20 buckets get 4097 right on
    the shipped revision; **1 bucket does not**." `prefill_trace_discriminators.json:411-429`
    has 1 bucket `[1024]` getting 4097 right. Directly contradicted by the artifact named two
    lines above it.
  - `perf_summary.json:112` `blocker.at_1_bucket` still lists only 96 and 128; the word "size"
    does not appear anywhere in the file, so the stage's central finding is absent from the
    required machine-readable performance-accounting artifact.
  - `perf_summary.json:110` `evidence_matrix`: "six probes, four configurations" — the file it
    points at holds seven discriminator probes over five configurations.

  Required next step: propagate the size discriminator and the corrected counts into
  `blocker.*`, after the P1s settle whether the conclusion survives.

- **P2: "six probes / four configurations" is stale in four places, the probe input list omits
  the decisive probe, and the Artifacts table is still broken in two.**

  Evidence:
  - `README.md:218` (bisection row 7) "Six probes across four configurations";
    `README.md:290` "Six further probes were run"; `README.md:638` "six probes, four
    configurations" — against `README.md:348` "five configurations" and `README.md:427` "the
    seven discriminator probes" in the same document.
  - `README.md:639` "its inputs" lists
    `probe_disc_{20bucket,bucket96,4097only_traced,4097only_eager,8192_traced,8192_eager}.json`
    and **omits `probe_disc_bucket1024.json`** — the one the round-5 conclusion comes from.
  - `work_log.md:270` "Six probes across four configurations now pin it down", immediately
    followed by five bullets whose fifth is the seventh probe.
  - `README.md:636` is still a blank line inside the Artifacts table, splitting it into two
    tables of which the second (the one listing every probe) has no header row. Flagged in
    round 4; unfixed.

  Required next step: one pass over those five lines.

- **P2: `prefill_trace_discriminators.json`'s top-line conclusion overstates what was compared,
  and the README matrix repeats it.**

  Evidence: `conclusions[0]` — "Tracing off: every request matches the reference, at every
  length." Two of the six tracing-off rows carry
  `matches_vllm_integration_reference: null` (prompt_len 100 and 8192), because
  `doc/vllm_integration/probe_full_fixed.json` has no entry at those lengths; the 8192
  comparison is traced-vs-eager *within this stage*, and the tracing-off 8192 row is itself the
  control. `conclusions[1]` folds 8192 into the same reference framing. `README.md:293-299`'s
  matrix marks ✅ in the "128 / 100" and "8192 (out)" columns with no note that two of those
  cells are not reference comparisons. Round 4 raised this as a concern; it has since been
  promoted into the artifact's headline conclusions.

  Required next step: one clause on the matrix and one on `conclusions[0]` — "for 100 and 8192
  the comparison is against this stage's tracing-off control, not the committed reference."

## Other Concerns

- **`README.md:16`**, the fourth sentence of the document, says tracing is off "because a
  resident prefill trace was measured to change the output of *other* requests **through a ttnn
  trace/allocator contract**". `README.md:241-243` says the opposite about the case that
  actually forces the default: "This explains the 20-bucket decay and nothing else… it does
  *not* explain the one-bucket failure below". The lead paragraph attributes both failures to a
  mechanism the body labels unexplained.
- **`work_log.md:249`** still says "a deterministic **~9-request** reproducer"; every other
  surface uses "the 22nd generation". One of round 4's two instances was fixed; this one was
  not, and the two frames (9th completion within the runner arm, 22nd generation on the server)
  are still not reconciled anywhere.
- **`README.md:367`** (Qualitative status row) still cites "character-identical to the
  standalone model over the full common prefix" for what `after/determinism_vllm.json` records
  as `compared_chars: 79` at `max_tokens 24`. The stronger 127-token comparison
  (`after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`) is disclosed only in the
  determinism row. Carried unchanged from rounds 2–4.
- **`$autofix` is still unmentioned** in both documents (`grep -c` → 0, 0), and the
  ballast-buffer mitigation remains the named-but-untried option — now targeting a mechanism the
  stage says does not explain the failure that forces the default off.
- `after_prefill_traced/`'s runner arm is **5 of 12** on the long-loop metric
  (`loop_classification.json`, reproduced here), against 3 of 12 for the shipped and previous
  arms. It is not a shipped arm and the README does not claim otherwise, but the number is in
  the tree unremarked and it is the arm whose server was later found corrupt.

## Hard-Check Gaps

- The freed-intermediate address range of one prefill trace is **still unmeasured** (rounds
  1–5). `doc/optimized_full_model/prefill_trace_probe.json` gives
  `capture_retained_dram_bytes 3280896` and no peak-during-capture reading, so "twenty 52-layer
  prefill working sets" versus "a small, decode-shaped range" remains an assertion — and it is
  now the quantitative core of the *only* mechanism the stage claims, and would also be the
  natural way to test the new size hypothesis.
- Nothing measured between bucket sizes 128 and 1024, nothing between counts 2 and 19.
- Still **no live-server evidence** of the 4097/8192 divergence. I confirmed why: the
  non-aligned check's 4097 and 8193 `text_head` values are byte-identical across `after/`,
  `after_prefill_traced_1bucket/` and `after_prefill_traced/`, because in the traced arms that
  step runs after the interlock released the traces. One step reorder in `bench/run_arm.sh`
  would produce the datum; still not taken (rounds 4 and 5).
- The shipped headline arm (`after/`, 13:39) still predates the final code. This round I
  bounded the risk further than before: `tt/generator.py` is unchanged since 15:31 (its `.pyc`
  is 15:44), and the 17:28 `generator_vllm.py` edit is provably comment-only by bytecode
  comparison against the 16:28 `.pyc`. The only unverifiable delta is a one-line insertion in
  `generator_vllm.py` between source lines 247 and 425, made between 15:52 and 16:28 — after
  `probe_repro_eager.json` (loguru fingerprints shift `allocate_kv_cache` 565 → 566 while
  `_prefill_trace_enabled:247` is unchanged). Acceptable, but the headline arm was not re-run on
  the shipped bits.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm; the
  decode path is unchanged by this stage.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` (verified in
  `after/server_excerpt.log` and in `probe_repro_eager.json`'s capability report) and
  `doc/context_contract.json`. Unchanged from rounds 1–4.

## Anomaly Ledger

- Observed anomaly: **the only traced request in `probe_disc_bucket1024.json` — the probe the
  round-5 conclusion rests on — returns `[84, 198, 2223, 6453, 2223, 6453, …]`,
  `distinct_tokens 4`.**
  Evidence: `probe_disc_bucket1024.json` requests[0]; `prefill_trace_discriminators.json`
  rows 392-410. `distinct_tokens == 4` appears in exactly five of thirty rows: the four
  known-wrong 4097 rows and this one; every certified-correct row is 9/12/16/17.
  Affected path: replay of a captured 1024-row prefill trace.
  Control or comparison: **none exists.** No eager 1024, no 1024 at another bucket set, no 1024
  in any other probe.
  Likely subsystem: ttnn mesh trace capture/replay at large padded row counts, or the port's
  per-bucket persistent capture state.
  Investigation performed: none by the stage; the row is tabulated without comment and the
  README matrix omits the length entirely.
  Resolution: **more-work-needed** (P1 above). One probe invocation, no server.

- Observed anomaly: with a prefill trace **captured** (not necessarily replayed) for one small
  bucket, long eager prefills (4097 → padded 4128, and the warmed 8192) diverge from their first
  token; a largest captured bucket of 1024 does not do this.
  Evidence: `prefill_trace_discriminators.json` and the eleven underlying probes; all 30 rows
  re-derived independently here over common prefixes. Failing: `probe_repro_traced`,
  `probe_full_shipped`, `probe_disc_bucket96`, `probe_disc_4097only_traced`,
  `probe_disc_8192_traced`. Passing: every tracing-off control, `probe_disc_20bucket` and
  `probe_disc_bucket1024`.
  Affected path: eager prefill of a prompt outside the traced buckets, in a process that has
  captured at least one small prefill trace.
  Control or comparison: same-revision tracing-off controls at both lengths ✓; same-revision
  20-bucket and 1-bucket-1024 comparisons ✓ (I verified all seven discriminator probes ran on
  bytecode-identical source); replay-order and preceding-request-length confounds refuted by
  `probe_repro_traced` and `probe_disc_20bucket` respectively; **no live-server control**.
  Likely subsystem: ttnn mesh trace capture / allocator lifetime. Unknown.
  Investigation performed: seven discriminating probes isolating capture-vs-replay, bucket
  identity, warmed-shape, trace count and trace size; every request tabulated.
  Resolution: **controlled** for the ship decision — the configuration does not ship, the matrix
  is in the tree, the mechanism is labelled UNEXPLAINED in the README, the docstring and
  `perf_summary.json`. The *characterisation* is not closed: the size conclusion depends on the
  uncontrolled row above, and the configuration the conclusion implies (`[128,1024]`) is
  unmeasured. P1s above.

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `prefill_trace_bisect.json` token-identical; `soak_traced_bucket/` clean over 84 in-bucket
  generations (60 + 24, both files verified, worst `replacement_char_fraction` 0.0000, all
  rounds byte-stable).
  Likely subsystem: trace/allocator lifetime.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, a valid in-bucket soak.
  Resolution: **controlled** — does not ship; reproducer, refutations and interlock in tree.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm.
  Evidence: `after/vllm_qualitative_outputs.json` p0 sampled 0.529, p1 greedy 0.708, p2 greedy
  0.938 — reproduced independently here, matching `loop_classification.json`.
  Control or comparison: `readiness_vllm/` 3/12 sharing p1 and p2 at identical coverage; chat
  verdict arm 0/6; HF control 0/6. Pre-existing and prompt-shaped.
  Resolution: **controlled**; the count is now correct on all three surfaces.

- Observed anomaly: the qualifying soak's completions are the `" to=self"` analysis channel
  restating the question.
  Evidence: `soak_traced_bucket/soak_traced_bucket.json` head fields;
  `after/qualitative/qualitative_tt_chat.json` p0/p2 show the same prefix while p1 is a clean
  `" to=user"` answer.
  Control or comparison: classified in earlier stages as Harmony-style channel tokens invisible
  over the API.
  Resolution: **controlled** (weaker readable-text evidence than described; unchanged from
  rounds 3–4).

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest
  and sampling log.
  Control or comparison: identical in the before arm and previous stages.
  Resolution: **controlled**.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `stage_review.md`, `stage_review_round2.md`, `stage_review_round3.md`,
  `stage_review_round4.md` (round 4's six required items re-derived one by one);
  `.agents/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh`.
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,metrics.json,loop_classification.json,prefill_trace_discriminators.json}`;
  all eleven `doc/optimized_vllm/probe_*.json` and their logs;
  `doc/vllm_integration/probe_full_fixed.json`; every `run<N>/vllm_benchmark.json` and
  `run<N>/vllm_ci_serving_benchmark.json` in `before/`, `after/`,
  `after_prefill_traced_1bucket/`, `after_prefill_traced/`; `after/serving_audit.json`,
  `after/sampling_tests.log`, `after/server_excerpt.log`, `after/determinism_vllm.json` and the
  traced arms' determinism files; `soak_traced_bucket/soak_traced_bucket{,_after_mixed}.json`;
  every arm's `vllm_qualitative_outputs.json`, `after/qualitative/qualitative_tt_chat.json`,
  `doc/full_model/qualitative/qualitative_hf_chat.json`, `readiness_vllm/`;
  `logs/` (pytest_final, pytest_watcher, degenerate_check_all, all seven `probe_disc_*.log`,
  probe_repro_*, probe_discriminate, probe_warm); every `DEGENERATE_CHECK_EXCLUDE` marker;
  `doc/context_contract.json`; `doc/datatype_sweep/evidence_perf.json`;
  `doc/optimized_full_model/prefill_trace_probe{,_8192}.json`; `bench/run_discriminators.sh`,
  `bench/run_arm.sh`.
- Code paths: `tt/generator_vllm.py` (`PREFILL_WARMUP_LENGTHS`, `PREFILL_TRACE_BUCKETS`,
  `_PREFILL_TRACE_ENV`, `_prefill_trace_buckets`, `_prefill_trace_enabled`), and its 16:28
  `.pyc`; `tt/generator.py` mtime/`.pyc`; `tests/test_full_model.py` (test inventory).
- Commands run (all read-only; no server, device, hardware or vLLM use): `git status/log`, `ls`,
  `stat`, `grep`, `sed/awk`, and Python scripts that re-derived every discriminator row against
  the committed reference over common prefixes, recomputed warm medians/deltas/speedups for all
  four arms and both profiles, recomputed the long-verbatim-loop coverage over six qualitative
  artifacts, tabulated `distinct_tokens` against the match column, resolved every file path
  cited in the README/work_log/perf_summary, and compared the current `generator_vllm.py`
  bytecode against its pre-edit `.pyc`.

## Residual Risk

- If the eager 1024 control comes back different from
  `[84, 198, 2223, 6453, 2223, 6453, …]`, then the [1024] configuration is *also* wrong — just
  in the traced request instead of the eager one — and the round-5 conclusion, the shipped
  docstring paragraph and the upstream ask all point in the wrong direction. That is the single
  largest open item, and it costs one probe.
- Even if the size conclusion holds, "96 and 128 fail, 1024 passes" is three points; the
  threshold, and whether it is absolute size or size relative to the eager prompt, are
  unconstrained. The report does not over-claim here, but a deployment reading the docstring
  might.
- `[128,1024]` — the configuration that would keep the 1.29x — is untested in both directions:
  correctness on long eager prefills, and whether two resident traces reproduce the 20-bucket
  decay.
- The one-bucket mechanism is genuinely open, and no measurement bounds the poisoned address
  range for any trace size.
- The shipped headline arm was measured on code ~4 h older than what ships, with the executable
  deltas confined to the disabled path (bounded by bytecode comparison, except one line inserted
  between 15:52 and 16:28).
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops on every
  model, disclosed as limitation 9.
