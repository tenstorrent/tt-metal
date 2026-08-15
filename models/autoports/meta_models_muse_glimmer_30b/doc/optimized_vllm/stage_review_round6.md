# Stage Review (round 6)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the five previous reviews. Worktree live,
uncommitted (8 modified files + untracked `doc/optimized_vllm/`).

Verdict: more-work-needed

**Both of round 5's P1s are genuinely and completely resolved, and I verified each from the raw
probe JSON rather than from the prose.**

1. **P1a — the uncontrolled 1024 output.** `probe_disc_1024_eager.json` exists, ran tracing-off
   (`prefill_trace.enabled false`, `buckets_resident []`), and its 1024 request returns
   `[84, 198, 2223, 6453, 2223, 6453, 2223, 6453, 2223]` — **byte-identical** to the traced
   `[1024]` row that round 5 flagged. I extended the check: the same nine tokens come back in
   **four independent sessions** — tracing off, `[1024]` traced, `[128,1024]` traced, and `[128]`
   eager. `distinct_tokens 4` at length 1024 is therefore a property of the synthetic
   `arange` prompt, exactly as claimed. The negative result is stated explicitly in
   `prefill_trace_discriminators.json:conclusions[5]` and `README.md:327-330`. Anomaly closed.
2. **P1b — the unmeasured `[128,1024]`.** `probe_disc_bucket128_1024.json` has
   `buckets_resident [128, 1024]` and 128 / 1024 / 4097 all matching. `probe_disc_8192_bucket128_1024.json`
   has the same resident set and returns `[1767, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337]`
   at 8192 — **byte-identical to the `[128]` failure** (`probe_disc_8192_traced.json`), against a
   tracing-off control of `[198, 6453, 107177, 38, 589, 6105, 2032, 14774, 328]`. So the
   configuration that would have kept the 1.29x was measured, and it fails. The ship-off decision
   is now earned on a measured failure of the only configuration worth wanting, not on an
   unmeasured generalisation. That is a real improvement over round 5.

**The matrix re-derives exactly.** I rebuilt all 38 rows from the fifteen `probe_*.json` files
with my own common-prefix comparator (vLLM-integration reference at 37/128/4097, this stage's
tracing-off run otherwise), and compared field by field against
`prefill_trace_discriminators.json`: **0 mismatches** across `path`, `reference_source`,
`matches_reference`, `first_tokens`, `distinct_tokens` and `largest_resident_bucket`. The README
matrix at `README.md:295-302` also reproduces cell for cell, including every `—`.

**Every headline number reproduces to the last digit** from the raw `run<N>/vllm_*benchmark.json`
medians of runs 4–6: TTFT 81.477 → 77.419 (−4.98 %), decode t/s/u 43.4802 → 43.4278 (−0.12 %),
TPOT 22.999 → 23.027, ITL 23.015/23.222 → 23.011/23.245, throughput 42.625 → 42.646,
E2E 3002.6 → 3001.2, run ranges 77.79–91.83 / 76.64–87.32; burst 721.877 → 717.560,
2147.527 → 2175.865, 43.4053 → 43.3920; traced arms 62.965 (1.294x, 812.096 = +12.50 %,
1654.70) and 60.662 (1.343x, 805.384 = +11.57 %); 43.428 / 43.331 = **100.2 %** of
`doc/datatype_sweep/evidence_perf.json`. Completed 1/1 and 32/32 with 0 missing tokens in all
six median runs of both profiles.

**Evidence-vs-shipped-code continuity still holds.** `tt/generator_vllm.py` was edited again at
17:59, after the last probe. I compiled the current source and diffed it against
`tt/__pycache__/generator_vllm.cpython-312.pyc` (17:42, i.e. before all four new probes):
29 code objects, **zero** bytecode, name, varname or non-docstring constant differences. The
17:59 edit is provably comment-only, and the probe log's `_prefill_trace_buckets:179` line
number matches the current source exactly.

**Gates re-checked:** 29 passed plain and under the watcher; 62/10/1 sampling with the ten names
`README.md:556-565` lists, verified against `after/sampling_tests.log`'s short summary;
`logs/degenerate_check_all.log` re-run at 17:59, "No degenerate output detected", 14 exclusions
all corruption-characterisation dirs; `after/serving_audit.json` `clean true`,
`markers_benchmark_window {}`, `degraded_benchmark_window []`, no surviving processes;
non-aligned 9/9; served 131072 against a 131072 contract with `capability_reduction: "none"`.

What remains is entirely documentation and one script — no hardware, no reruns. But it is not
cosmetic: the stage's central conclusion is stated **two different ways in the same tree**, and
the stronger of the two is contradicted by the stage's own matrix, including in the required
machine-readable performance-accounting artifact and in a shipped source comment. Two of round
5's five P2 items were also only partly carried out.

---

## Required Work

- **P2: "six configurations, each wrong at some length" is contradicted by this stage's own
  matrix, and it is the stated reason tracing ships off.**

  Evidence:
  - Re-derived from the matrix, the six configurations split as:
    | configuration | measured ❌ at any prompt length? |
    |---|---|
    | tracing off | **no** — ✅ at 37/100/128/1024/4097/8192 |
    | `[96]` | yes (4097) |
    | `[128]` | yes (4097, 8192) |
    | `[1024]` | **no** — ✅ at 1024 and 4097, nothing else measured |
    | `[128,1024]` | yes (8192) |
    | 20 buckets | **no** — ✅ at 37/100/128/4097; its failure is the 22nd-generation decay, which is not a prompt length |
    Only **three of six** have a measured wrong prompt length.
  - Yet: `perf_summary.json:115` — "six configurations, each wrong at some length";
    `work_log.md:284` — "Six configurations measured, each wrong at some prompt length";
    `work_log.md:290` — "every configuration measured is wrong somewhere";
    `README.md:340` — tracing ships off "because **every one of the six configurations measured
    is wrong at some prompt length**"; `tt/generator_vllm.py` `PREFILL_TRACE_BUCKETS` docstring —
    "Both a wide list and a single entry were measured and **both are wrong somewhere**"
    (`[1024]` is a single entry with no measured failure).
  - The correct form is present elsewhere in the same files: `README.md:304` — "**No traced
    configuration measured is correct at every length**" — and
    `prefill_trace_discriminators.json:conclusions[2]`. `README.md` therefore contradicts itself
    36 lines apart, and `perf_summary.json` contradicts the artifact it cites two lines above
    (`evidence_matrix`).
  - `perf_summary.json:115`'s **key is also still `not_monotone_in_trace_count`**, the framing
    round 5 flagged and the stage says it replaced; the value under that key is now about size.
  - Round 5 quoted `README.md:594` conceding "Counts between 2 and 19 are unmeasured". That
    sentence is gone: `grep -n "unmeasured"` over `README.md` and `work_log.md` now returns
    **nothing**. There is no longer any statement of what the matrix did not cover — in
    particular the one cell that would separate the stage's own hypothesis from its alternative:
    **`[1024]` alone at 8192 was never run**. Both 8192 failures ( `[128]` and `[128,1024]` )
    contain bucket 128, so "a large enough largest bucket is not enough" and "any *small*
    resident bucket poisons long eager prefills, and 1024 is merely big enough for 4097" fit the
    data equally well. The upstream ask (`README.md:354-361`) is written as if only the first
    reading survives.

  Why this matters: the false form converts "no traced configuration is *known* safe at every
  length" into "every traced configuration was *proven* broken", which is a materially stronger
  claim about the measurements than the matrix supports, and it is the sentence that justifies
  the shipped default. It appears in the machine-readable performance-accounting artifact the
  goal contract requires and in a comment a deployment reads. This is the same overstatement
  round 5 raised against `conclusions[0]`: it was fixed in the JSON conclusions and propagated
  into four other surfaces.

  Required next step: replace the four/five statements with the accurate one already used at
  `README.md:304` — no traced configuration measured is *correct at every length*, and name the
  three that have a measured failure versus the two whose coverage is incomplete. Rename or
  rewrite `perf_summary.json`'s `not_monotone_in_trace_count` key. Restore an explicit coverage
  statement listing what was not measured, `[1024]`@8192 first. Either that, or run the single
  probe `MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=1024 --prompt-lens 8192` and let the matrix
  settle it. The wording fix alone is sufficient; the ship-off decision does not depend on the
  cell, because `[1024]` alone does not trace the 128-row bucket the 1.29x was measured on.

- **P2: `bench/run_discriminators.sh` does not reproduce four of the fifteen matrix probes,
  including both decisive ones, while three artifacts state that it does. This was round 5's
  explicit required next step.**

  Evidence:
  - `bench/run_discriminators.sh` (mtime 17:24) contains seven invocations, A–G, ending at
    `bucket1024`. The four probes that answer round 5 ran **after** it and are **not in it**:
    `probe_disc_1024_eager.json` (17:46), `probe_disc_bucket128_1024.json` (17:49),
    `probe_disc_1024_bucket128.json` (17:53), `probe_disc_8192_bucket128_1024.json` (17:57).
  - `prefill_trace_discriminators.json:"what"` — "Reproduce with
    `doc/optimized_vllm/bench/run_discriminators.sh`"; `README.md:291-293` — "Fifteen probes
    across six configurations are tabulated … and **reproduced by** `bench/run_discriminators.sh`";
    `perf_summary.json:110` — "reproduce with …". All three are false for the rows that decide
    the stage.
  - `logs/probe_r5.log` shows H/I/J were driven by an ad-hoc script (`=== H: 1024+4097 with
    tracing OFF …`) that is not in the tree. The 17:57 `8192` at `[128,1024]` probe has no driver
    record at all — only its raw `logs/probe_disc_8192_bucket128_1024.log`.
  - `README.md:647` ("its inputs") still lists ten of the fifteen artifacts, omitting
    `probe_disc_bucket1024.json` (round 5 flagged exactly this omission),
    `probe_disc_1024_eager.json`, `probe_disc_1024_bucket128.json`,
    `probe_disc_bucket128_1024.json` and `probe_disc_8192_bucket128_1024.json`.
  - The script's own header still documents G as "**separates trace COUNT from trace SIZE** …
    every failing config's largest bucket is <= 128 and the passing 20-bucket config's is 1024",
    the framing `README.md` and the shipped docstring have since replaced.

  Why this matters: the discriminator matrix is the stage's load-bearing evidence and the stage
  says so. A reader who runs the one command the report gives reproduces seven rows and neither
  of the two that resolve round 5's P1s.

  Required next step: append the four invocations (with the `--prompt-lens` / bucket settings
  visible in each probe's `prefill_trace.buckets_requested` and request list) and refresh the
  header's what-each-one-separates block; complete `README.md:647`'s input list.

- **P2: `work_log.md` was not carried through the round-5 pass.**

  Evidence:
  - `work_log.md:269-270` — "**Six probes across four configurations** now pin it down" — the
    exact line round 5 flagged, unfixed, and now directly contradicted by `README.md:291`
    ("Fifteen probes across six configurations") in the same tree. The bullet list underneath it
    has six bullets.
  - `work_log.md:417-428` §8c — "**Four** independent `$stage-review` rounds ran against this
    stage" with a four-row table. Round 5 is absent, although `stage_review_round5.md` is in the
    directory, is cited by `README.md:657`, and is the round that produced both new probes. The
    section's own premise is that each round "changed the result rather than rubber-stamping it".
  - `work_log.md:428` (round 4's row) still says the single large bucket "turned out to be the
    discriminator that **explains the whole matrix**" — superseded by the 8192 result the stage
    now leads with.

  Required next step: one editing pass over §6 and §8c.

- **P2: the count of insufficient fixes is 3 in two places and 4 in three others.**

  Evidence: `README.md:15` (fourth sentence of the document) — "**Three** fixes were tried and
  measured insufficient, including shrinking the bucket set to one"; `README.md:672`
  (limitation 1) — "**Three** fixes were measured insufficient". Against `README.md:605` —
  "**Four fixes measured insufficient**: warm every sampling mode at warmup, make the traced
  replay blocking, shrink the bucket set to one, widen the largest bucket to 1024 while keeping
  the fast one"; `perf_summary.json:117-122` (four entries); and the shipped
  `_PREFILL_TRACE_ENV` docstring ("Four fixes were implemented and measured insufficient").
  `README.md:245` also still heads the section "**Two** fixes that did not work" over three
  bullets, the third of which covers two fixes.

  Required next step: one pass over those four lines.

## Other Concerns

- **`README.md:576` mislabels the arm.** "the shipped arm fails `test_specific_seed_reproducible[42]`
  instead". `after/sampling_tests.log`'s short summary and `README.md:586` both say the shipped
  arm fails `test_specific_seed_reproducible[0]`; `[42]` is `after_prefill_traced_1bucket/`. I
  extracted the failing set from all five sampling logs and the table at `README.md:582-587`
  reproduces exactly — only the prose sentence is wrong. The argument it supports (the class has
  a floating member) is unaffected.
- **`tt/generator_vllm.py` `_PREFILL_TRACE_ENV`: "No configuration measured is correct at every
  length."** In context the bullet is about bucket sets, but tracing off *is* a measured
  configuration and *is* correct at every length. One word — "traced" — fixes it.
- **`README.md:376`** (Qualitative status row) still claims "character-identical to the standalone
  model over the **full common prefix**" for what `after/determinism_vllm.json` records as
  `compared_chars: 79` at `max_tokens 24`. Carried unchanged from rounds 2–5.
- **`$autofix` is still unmentioned** in both documents (`grep -c` → 0, 0), and the ballast-buffer
  mitigation remains the named-but-untried option against a mechanism the stage says does not
  explain the failure that forces the default off.
- `after_prefill_traced/`'s runner arm is 5 of 12 on the long-loop metric against 3 of 12 for the
  shipped and previous arms — in the tree, unremarked. Not a shipped arm; carried from round 5.

## Hard-Check Gaps

- **`[1024]` alone at 8192 is the single unmeasured cell that discriminates the stage's
  hypothesis from its alternative** (see P2 above). One probe, no server. It does not change the
  ship decision.
- Nothing measured between bucket sizes 128 and 1024; counts 3–19 still unmeasured (count 2 now
  is). The report no longer says so anywhere.
- The freed-intermediate address range of one prefill trace is **still unmeasured** (rounds 1–6).
  `doc/optimized_full_model/prefill_trace_probe.json` gives `capture_retained_dram_bytes 3280896`
  and no peak-during-capture reading, so "twenty 52-layer prefill working sets" versus "a small,
  decode-shaped range" remains an assertion — and it is the quantitative core of the only
  mechanism the stage claims.
- Still **no live-server evidence** of the 4097/8192 divergence: the non-aligned check's 4097 and
  8193 `text_head` values are byte-identical across all three arms because in the traced arms that
  step runs after the interlock released the traces. One step reorder in `bench/run_arm.sh` would
  produce the datum; not taken (rounds 4–6).
- At 8192 and 100 the tracing-off "reference" is a **single** session, so those ✅ cells are
  self-comparisons with no cross-session reproduction. (1024 does have four independent sessions;
  4097 has three tracing-off sessions agreeing.) Adequate for the conclusion drawn, but worth one
  clause.
- The shipped headline arm (`after/`, 13:39) still predates the final code. Bounded as before:
  `tt/generator.py` unchanged since 15:31, and the 17:59 `generator_vllm.py` edit proven
  comment-only by bytecode comparison against the 17:42 `.pyc` that every late probe ran on.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm; the
  decode path is unchanged by this stage.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` and
  `doc/context_contract.json`.

## Anomaly Ledger

- Observed anomaly: the 1024-token probe returns a 2-token cycle, `distinct_tokens 4`.
  Evidence: `probe_disc_bucket1024.json`, `probe_disc_1024_eager.json`,
  `probe_disc_bucket128_1024.json`, `probe_disc_1024_bucket128.json`.
  Affected path: prefill of the synthetic `arange` probe prompt at length 1024.
  Control or comparison: **tracing-off control now exists** and is byte-identical; so are two
  traced configurations and one eager-under-a-small-bucket run — four independent sessions,
  same nine tokens.
  Likely subsystem: none — prompt property.
  Investigation performed: dedicated control probe (round 5 P1a).
  Resolution: **controlled**, and recorded as a negative result in the README and the JSON
  conclusions.

- Observed anomaly: with a prefill trace **captured** (not necessarily replayed), long eager
  prefills diverge from their first token; a largest bucket of 1024 fixes 4097 but **not** 8192.
  Evidence: all 38 matrix rows, independently re-derived here; `[128]` and `[128,1024]` fail 8192
  byte-identically; `[96]`, `[128]` fail 4097 byte-identically; `[1024]`, `[128,1024]` and the
  20-bucket set get 4097 right.
  Affected path: eager prefill of a prompt outside the traced buckets, in a process that captured
  at least one prefill trace.
  Control or comparison: same-revision tracing-off controls at every length ✓; capture-only
  (no replay) case isolated ✓; warmed-shape ruled out ✓; bucket-value quirk ruled out ✓;
  replay-order and preceding-request-length confounds refuted ✓; **no live-server control**.
  Likely subsystem: ttnn mesh trace capture / allocator lifetime. Unknown.
  Investigation performed: fifteen probes over six configurations, every request tabulated.
  Resolution: **controlled** for the ship decision — the configuration does not ship, the matrix
  is in the tree, the mechanism is labelled UNEXPLAINED in the README, the docstring and
  `perf_summary.json`, and the configuration that would have kept the win is now measured and
  measured broken. The *characterisation* is open in one direction: `[1024]`@8192 is unmeasured,
  so "largest-bucket size" versus "any small resident bucket" is not separated. P2 above asks for
  the wording, not the hardware.

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `soak_traced_bucket/` clean over 84 in-bucket generations.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, a valid in-bucket soak.
  Resolution: **controlled** — does not ship; reproducer, refutations and interlock in tree.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm.
  Evidence: `after/vllm_qualitative_outputs.json` 3/12; `readiness_vllm/` 3/12 sharing p1/p2 at
  identical coverage; chat verdict arm 0/6; HF control 0/6.
  Resolution: **controlled**; count consistent on all three surfaces.

- Observed anomaly: the qualifying soak's completions are the `" to=self"` analysis channel.
  Resolution: **controlled** (classified in earlier stages as Harmony-style channel tokens
  invisible over the API); weaker readable-text evidence than described, unchanged from rounds 3–5.

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest and
  sampling log. Identical in the before arm and previous stages. Resolution: **controlled**.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `stage_review_round5.md` (each of its five required items re-derived).
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,prefill_trace_discriminators.json}`;
  all sixteen `doc/optimized_vllm/probe_*.json`; `doc/vllm_integration/probe_full_fixed.json`;
  every `run<N>/vllm_benchmark.json` and `run<N>/vllm_ci_serving_benchmark.json` in `before/`,
  `after/`, `after_prefill_traced_1bucket/`, `after_prefill_traced/`; `after/serving_audit.json`,
  `after/determinism_vllm.json`, `after/sampling_tests.log` and the other four sampling logs;
  `logs/{pytest_final,pytest_watcher,degenerate_check_all,probe_r5,probe_warm,probe_discriminate,
  probe_pair,probe_disc_*}.log`; `bench/run_discriminators.sh`; `doc/context_contract.json`;
  `doc/datatype_sweep/evidence_perf.json`.
- Code paths: `tt/generator_vllm.py` (`PREFILL_WARMUP_LENGTHS`, `PREFILL_TRACE_BUCKETS`,
  `_PREFILL_TRACE_ENV`, `_prefill_trace_buckets`, `_prefill_trace_enabled`) and its 17:42 `.pyc`;
  `tests/test_full_model.py` (test inventory).
- Commands run (all read-only; no server, device, hardware or vLLM use): `git status`, `ls`,
  `stat`, `find -newermt`, `grep`, `sed`, and Python scripts that rebuilt all 38 discriminator
  rows from the raw probes and diffed them field-by-field against the committed matrix,
  recomputed warm medians/deltas/speedups for all four arms and both profiles, compared full
  token sequences across configurations at 1024/4097/8192, extracted the failing sampling sets
  from five logs, and compared the current `generator_vllm.py` bytecode against its pre-edit
  `.pyc`.

## Residual Risk

- `[1024]` alone at 8192 is unmeasured, so the stage's "largest captured trace size" variable is
  supported by three points and not separated from "any small resident bucket poisons long eager
  prefills". The upstream ask is written for the first reading only.
- The mechanism is genuinely open, and no measurement bounds the poisoned address range for any
  trace size.
- The shipped headline arm was measured on code ~4 h older than what ships, with the executable
  deltas confined to the disabled path (bounded by bytecode comparison).
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops on every
  model, disclosed as limitation 9.
