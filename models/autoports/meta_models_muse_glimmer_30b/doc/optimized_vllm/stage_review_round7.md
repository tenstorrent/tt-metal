# Stage Review (round 7)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the six previous reviews. Worktree live,
uncommitted (8 modified files + untracked `doc/optimized_vllm/`).

Verdict: more-work-needed

**Five of round 6's six items are genuinely done, and I verified each from the artifacts rather
than the prose.** The sixth — the one round 6 led with — survives in exactly one place, and it
is the paragraph that states why the shipped default is what it is.

Verified done:

* **`bench/run_discriminators.sh` now reproduces the matrix.** It carries **13** invocations
  writing 13 of the 15 matrix probe artifacts — including all four post-round-5 ones
  (`1024_eager`, `1024_bucket128`, `bucket128_1024`, `8192_bucket128_1024`) — grouped as
  controls / one-small-bucket / one-large-bucket-and-the-pair / the-wide-set, with a header
  saying what each group separates and a trailing note naming the two 16-step probes it does
  not re-run (`probe_full_shipped.json`, `probe_full_prefill_traced.json`) and the two
  unmeasured cells. The `what` field of `prefill_trace_discriminators.json` and
  `perf_summary.json`'s `evidence_matrix` now point at a script that is true.
* **The coverage statement is restored in all four places claimed** — `README.md:311-315`,
  `perf_summary.json:blocker.coverage_limits`, the shipped `_PREFILL_TRACE_ENV` docstring
  (`generator_vllm.py:208-211`) and `bench/run_discriminators.sh:73-77` — each naming
  `[1024]` alone at 8192 and any bucket size between 128 and 1024, and each saying the ship
  decision does not rest on that cell.
* **`perf_summary.json`'s `not_monotone_in_trace_count` key is gone**, replaced by
  `what_the_matrix_shows`, whose text ("three of the five traced sets have a measured wrong
  length, and the other two were not run at 8192") I re-derived and confirm is exactly right.
* **`work_log.md` is carried through.** §6 now says "Fifteen probes across six configurations";
  §8c has six rows; round 4's row no longer claims the large bucket "explains the whole
  matrix"; §8d records why `$autofix` was not invoked.
* **Fix count is four** in `README.md:15`, `README.md:682`, `README.md:615`,
  `perf_summary.json:fixes_measured_insufficient` (4 entries) and the docstring.
* **`test_specific_seed_reproducible[0]` is the shipped arm's failure everywhere.** I extracted
  the failing set from all six sampling logs: `after/` fails `[0]`,
  `after_prefill_traced_1bucket/` fails `[42]`, `after_prefill_traced/` and
  `sampling_variance/sampling2` fail `test_mixed_params_batch` **and** `[999]` (11 each),
  `ctrl_notrace/` and `sampling_variance/sampling1` fail `test_mixed_params_batch` (10 each).
  `README.md:565-576`, `:586-587` and the table at `:592-597` all reproduce exactly.

Re-derived independently this round:

* **The 38-row discriminator matrix rebuilds with zero mismatches.** I rebuilt every row from
  the 15 raw probe JSONs with my own common-prefix comparator (vLLM-integration committed probe
  at 37/128/4097, this stage's tracing-off runs otherwise) and diffed field by field against
  `prefill_trace_discriminators.json`: 0 mismatches on `matches_reference`, `first_tokens`,
  `distinct_tokens`, `largest_resident_bucket` and reference-source class, no extra rows, none
  missing. The README matrix at `:295-302` reproduces cell for cell including every `—`.
* **Every headline number reproduces to the last digit** from the raw `run<N>/vllm_*.json`
  medians of runs 4–6: TTFT 81.477 → 77.419 (−4.98 %), decode t/s/u 43.4802 → 43.4278
  (−0.12 %), TPOT 22.999 → 23.027, ITL 23.015/23.222 → 23.011/23.245, throughput 42.625 →
  42.646, E2E 3002.6 → 3001.2, run ranges 77.79–91.83 / 76.64–87.32; burst 721.877 → 717.560,
  2147.527/2148.757 → 2175.865/2177.187, 43.4053 → 43.3920; traced arms 62.965 (1.2940x,
  812.096 = +12.50 %, burst TTFT 1654.70, decode 43.4298) and 60.662 (1.3432x, 805.384 =
  +11.57 %, 1691.07, 43.4694). 1/1 and 32/32 completed with 0 missing tokens in all six runs of
  both profiles, every arm.
* **`loop_classification.json` recomputes exactly.** My own longest-repeating-word-block
  measure over each arm's own artifact returns 3/12 for `after/` (0.529/0.708/0.938), 3/12 for
  `readiness_vllm/` (0.708/0.938/0.600), 5/12 for `after_prefill_traced/` and 3/12 for
  `after_prefill_traced_1bucket/` — same completions, same coverages, to three decimals.
* **Evidence-vs-shipped-code continuity holds.** `tt/generator_vllm.py` was edited again at
  18:12, after every probe and after the `.pyc`. I compiled the current source and diffed all
  29 code objects against `tt/__pycache__/generator_vllm.cpython-312.pyc` (17:42): **zero**
  differences in bytecode, names, varnames or constants (docstrings included). Both late edits
  are provably `#:`-comment-only.
* **Gates re-checked:** 29 passed plain and under the watcher; 62/10/1 sampling; the
  degenerate check re-run at **18:12**, after the final edits, "No degenerate output detected"
  with 14 reported exclusions all in corruption-characterisation directories;
  `after/serving_audit.json` `clean true`, `degraded_markers_benchmark_window []`,
  `surviving_vllm_processes []`; non-aligned 9/9; served `max_model_len=131072`,
  `max_num_seqs=32`, `sample_on_device_mode=all` read out of `after/server_excerpt.log`
  against a 131072 contract with `capability_reduction: "none"`.

What remains is two README sentences. No hardware, no rerun. The first is not cosmetic: it is a
false statement about this stage's own measurements, it contradicts the same document 45 lines
earlier, and it sits in the paragraph a reader goes to for the ship decision.

---

## Required Work

- **P2: the overstated claim round 6 required be replaced *everywhere* survives at
  `README.md:349`, in "What ships" — the sentence that justifies the shipped default.**

  Evidence:
  - `README.md:348-350`: "Tracing **off**. Not because nothing was measured … but because
    **every one of the six configurations measured is wrong at some prompt length**, the failure
    is silent…".
  - Re-derived from the matrix I rebuilt from the raw probes, the six configurations are:

    | configuration | measured wrong at any prompt length? |
    |---|---|
    | tracing off | **no** — ✅ at 37/100/128/1024/4097/8192 |
    | `[96]` | yes (4097) |
    | `[128]` | yes (4097, 8192) |
    | `[1024]` | **no** — ✅ at 1024 and 4097, nothing else measured |
    | `[128,1024]` | yes (8192) |
    | 20 buckets | **no** — ✅ at 37/100/128/4097; its failure is the 22nd-generation decay, which is not a prompt length |

    Three of six, not six of six. And the one the sentence is defending — **tracing off, the
    shipped configuration** — is one of the six it declares wrong.
  - The correct form is in the same file 45 lines above (`README.md:304-309`: "**No traced
    configuration measured is correct at every length it was measured at** … Three of the five
    traced configurations have a measured wrong length; the other two … were not run at 8192"),
    and in `README.md:613`, `work_log.md:284`, `work_log.md:290-293`,
    `perf_summary.json:what_the_matrix_shows`, `prefill_trace_discriminators.json:conclusions[2]`
    and the `_PREFILL_TRACE_ENV` docstring. This is the last surviving instance; `grep -n` over
    the tree returns exactly this one.

  Why this matters: the false form converts "no traced configuration is *known* safe at every
  length" into "every configuration was *proven* broken", which is a materially stronger claim
  about the measurements than the matrix supports, and as written it asserts that the shipped
  default is wrong at some prompt length. It is the stage's decision sentence, and it directly
  contradicts a correct sentence in the same section.

  Required next step: replace `README.md:349-350` with the form already used at `README.md:304`.
  One sentence, no hardware.

- **P2: `README.md:546` explains the 20-bucket arm's 5/12 loop count with a cause the same
  README rules out, and the table omits the arm with the corpus's single worst completion.**

  Evidence:
  - `README.md:546`: the `after_prefill_traced/` row's *which* column ends "— this is the arm
    whose short-prompt output decays, and the loop metric is where that shows up before
    `replacement_char_fraction` does".
  - But `README.md:154-160` and `work_log.md:374` both state that in the traced arms
    `bench/run_arm.sh` runs `sampling` before `qualitative`, the interlock fired there, and
    therefore "those traced arms' qualitative, determinism, cross-batch and non-aligned results
    are **eager-path** results, not traced ones" / that arm's "*post-sampling* qualitative is
    healthy only because the interlock had already released the traces". I confirmed the guard
    fired exactly once in each traced arm's `server_excerpt.log` and zero times in `after/`.
    The 5/12 was therefore measured with **no trace resident**, so it cannot be the decay
    showing up.
  - The two extra completions are both **sampled** (p2 sampled 0.741, p4 sampled 0.982), i.e.
    run-to-run draws — and my recomputation shows the *1-bucket* traced arm, which the report
    presents as the clean traced configuration, has `p0 sampled` at coverage **1.000**
    ("6-word block x32", a fully degenerate completion) against 0.529 for the shipped arm.
    `after_prefill_traced_1bucket/` is the only arm with a benchmark directory that is **not**
    a row in the `README.md:544-548` table, and it holds the highest single coverage in the
    corpus. It passes the degenerate gate (`replacement_char_fraction` 0.0000) — an instance of
    limitation 9, which is disclosed generically but not for this arm.

  Why this matters: an unsupported causal attribution is the kind of prose resolution the
  anomaly rule exists to catch, and here the report attributes to the trace hazard something its
  own interlock account says could not have been traced, while leaving out the row that would
  have shown the simpler explanation (sampled draw). It concerns a non-shipped arm and does not
  touch the ship decision, which is why this is P2 and not P1.

  Required next step: drop or correct the causal clause at `README.md:546` (these arms'
  qualitative ran eager, after the release; the extra rows are sampled draws), and add the
  `after_prefill_traced_1bucket/` row (3/12, p0 sampled 1.000) to the same table.

## Other Concerns

- **`README.md:656` says the script "carries all 14 invocations".** It carries **13**
  (`grep -c '^run '` returns 14 because it counts the `run ()` definition), covering 13 of the
  15 matrix probes, with the other two named. The content is right; the number is off by one.
- **`README.md:385` and `:532`** still say the chat arm is "character-identical to the standalone
  model over the **full common prefix**". The supporting artifact
  `after/determinism_vllm.json` records `compared_chars: 79` at `max_tokens 24`, and the longer
  comparison the README points at (`after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`)
  reads `identical: false`, `first_divergence: 2` for **all six** prompts. I checked why: the
  standalone renders `<|message|>` and the API does not, and after stripping it the two heads
  are byte-identical over all 149 compared characters on every prompt — which is exactly what
  `determinism_vllm.json`'s `special_tokens_stripped: ["<|message|>"]` records. So the claim is
  true in substance, but the artifact a reader is sent to says `false` six times with no note
  saying why. One clause fixes it. Carried unchanged from rounds 2–6.
- **`generator_vllm.py:152-153`** (`PREFILL_TRACE_BUCKETS` docstring): "Both a wide list and a
  single entry were measured and both are wrong somewhere." True of the 20-bucket list and of
  `[128]`, but it reads as a statement about single entries in general, and `[1024]` alone has
  no measured failure. The `_PREFILL_TRACE_ENV` docstring 50 lines below now states the accurate
  form and names the coverage limits, so this is ambiguity rather than error.
- **`$autofix` is now addressed** (`work_log.md` §8d) but only in the work log; the README's
  *Rejected and deferred* row for the ballast buffer still records it as untried without saying
  the debugging skill was considered. Acceptable; noted because rounds 4–6 asked.

## Hard-Check Gaps

- `[1024]` alone at 8192 remains unmeasured, so "largest captured trace size" is not separated
  from "any small resident bucket poisons long eager prefills". **This is now disclosed in all
  four required surfaces**, which was round 6's ask; the cell itself is still one probe away and
  does not change the ship decision.
- Nothing measured between bucket sizes 128 and 1024; trace counts 3–19 unmeasured. Disclosed.
- The freed-intermediate address range of one prefill trace is still unmeasured (rounds 1–7).
  `doc/optimized_full_model/prefill_trace_probe.json` gives `capture_retained_dram_bytes
  3280896` and no peak-during-capture reading, so "twenty 52-layer prefill working sets" versus
  "a small, decode-shaped range" remains an assertion, and it is the quantitative core of the
  one mechanism the stage claims.
- Still no live-server evidence of the 4097/8192 divergence: the non-aligned check's 4097 and
  8193 `text_head` values are identical across arms because in the traced arms that step runs
  after the interlock released the traces. One step reorder in `bench/run_arm.sh` would produce
  the datum; not taken (rounds 4–7).
- At 8192 and 100 the tracing-off reference is a single session, so those ✅ cells are
  self-comparisons; 1024 has four independent sessions and 4097 has three. Adequate for the
  conclusion drawn.
- The shipped headline arm (`after/`, 13:39) predates the final source, bounded as before:
  `tt/generator.py` unchanged since 15:31 and both late `generator_vllm.py` edits proven
  comment-only by a 29-code-object bytecode comparison against the 17:42 `.pyc` that every late
  probe ran on.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm; the
  decode path is unchanged by this stage.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` and
  `doc/context_contract.json`.
- No device-time/roofline term, by instruction (`$optimize`/`$vllm-integration` forbid the
  profiler in vLLM stages); recorded as `null` with the reason.

## Anomaly Ledger

- Observed anomaly: with a prefill trace **captured** (not necessarily replayed), long eager
  prefills diverge from their first token; a largest bucket of 1024 fixes 4097 but not 8192.
  Evidence: all 38 matrix rows, independently rebuilt from the raw probes this round;
  `[128]` and `[128,1024]` fail 8192 byte-identically; `[96]`, `[128]` fail 4097
  byte-identically; `[1024]`, `[128,1024]`, 20 buckets get 4097 right.
  Affected path: eager prefill of an out-of-bucket prompt, in a process that captured at least
  one prefill trace.
  Control or comparison: same-revision tracing-off controls at every length ✓; capture-only
  (no replay) isolated ✓; warmed-shape ruled out ✓; bucket-value quirk ruled out ✓; no
  live-server control.
  Likely subsystem: ttnn mesh trace capture / allocator lifetime. Unknown.
  Investigation performed: fifteen probes over six configurations, every request tabulated and
  now reproducible by one script.
  Resolution: **controlled** for the ship decision — the configuration does not ship, the
  matrix is in the tree and re-derives, the mechanism is labelled UNEXPLAINED in four places,
  the configuration that would have kept the win (`[128,1024]`) is measured and broken, and the
  coverage limits are stated. Only `README.md:349`'s description of that result is wrong
  (P2 above).

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `soak_traced_bucket/` clean over 84 in-bucket generations.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, a valid in-bucket soak.
  Resolution: **controlled** — does not ship; reproducer, refutations and interlock in tree.

- Observed anomaly: the 1024-token probe returns a 2-token cycle, `distinct_tokens 4`.
  Control or comparison: tracing-off control byte-identical, four independent sessions.
  Resolution: **controlled**, recorded as a negative result.

- Observed anomaly: `after_prefill_traced_1bucket/`'s p0 sampled completion is a 6-word block
  repeated 32 times, coverage **1.000** — the worst single completion in the stage's corpus.
  Evidence: my recomputation over `after_prefill_traced_1bucket/vllm_qualitative_outputs.json`,
  matching `loop_classification.json:after_1bucket`.
  Affected path: runner raw-completion arm (bare prompts to a chat model), sampled,
  **eager path** — this arm's qualitative ran after the interlock released its traces.
  Control or comparison: shipped arm's p0 sampled 0.529 on the same prompt; 20-bucket arm 0.942;
  the chat verdict arm and HF control are 0/6 on all prompts. The arm is not shipped and the
  arm's total (3/12) equals the shipped and previous-stage arms'.
  Likely subsystem: sampling draw on a prompt-format-mismatched arm; not trace-related.
  Investigation performed: recomputed all four arms' loop sets; verified guard-fire ordering.
  Resolution: **controlled** in substance, but **not recorded** — it is absent from the README's
  loop table while the 20-bucket arm's 5/12 is present with a causal explanation the evidence
  does not support. That is P2 #2.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm,
  3/12. Resolution: **controlled**; recomputed here and consistent on all three surfaces, equal
  to the previous stage's arm and sharing p1/p2 at identical coverage.

- Observed anomaly: the qualifying soak's completions are the `" to=self"` analysis channel.
  Resolution: **controlled** (classified in earlier stages as Harmony-style channel tokens
  invisible over the API); weaker readable-text evidence than described, unchanged from
  rounds 3–6.

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest
  and sampling log. Identical in the before arm and previous stages. Resolution: **controlled**.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `stage_review_round6.md` (each of its four required items and five other concerns
  re-derived).
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,prefill_trace_discriminators.json,
  loop_classification.json}`; all 16 `doc/optimized_vllm/probe_*.json`;
  `doc/vllm_integration/probe_full_fixed.json`; every `run<N>/vllm_benchmark.json` and
  `run<N>/vllm_ci_serving_benchmark.json` in `before/`, `after/`,
  `after_prefill_traced_1bucket/`, `after_prefill_traced/`; `after/serving_audit.json`,
  `after/determinism_vllm.json`, `after/server_excerpt.log`, `after/qualitative/*`, all six
  sampling logs; `after_prefill_traced{,_1bucket}/vllm_qualitative_outputs.json` and
  `server_excerpt.log`; `readiness_vllm/vllm_qualitative_outputs.json`;
  `logs/{pytest_final,pytest_watcher,degenerate_check_all}.log`;
  `bench/{run_discriminators.sh,run_arm.sh}`; `doc/context_contract.json`.
- Code paths: `tt/generator_vllm.py` (`PREFILL_TRACE_BUCKETS`, `_PREFILL_TRACE_BUCKETS_ENV`,
  `_PREFILL_TRACE_ENV`, `_prefill_trace_buckets`, `_prefill_trace_enabled`) and its 17:42 `.pyc`.
- Commands run (all read-only; no server, device, hardware or vLLM use): `git status`, `ls`,
  `stat`, `find -newermt`, `grep`, `sed`, and Python scripts that rebuilt all 38 discriminator
  rows from the raw probes and diffed them field by field against the committed matrix,
  recomputed warm medians/deltas/speedups for all four arms and both profiles, recomputed the
  longest-repeating-block loop metric for all four qualitative corpora, extracted the failing
  sampling sets from six logs, compared the vLLM/standalone chat heads with `<|message|>`
  stripped, and compared the current `generator_vllm.py` bytecode against its pre-edit `.pyc`.

## Residual Risk

- `[1024]` alone at 8192 is unmeasured, so "largest captured trace size" is supported by three
  points and not separated from "any small resident bucket poisons long eager prefills". Now
  disclosed in four surfaces, including the shipped docstring.
- The mechanism is genuinely open, and no measurement bounds the poisoned address range for any
  trace size.
- The shipped headline arm was measured on code ~4.5 h older than what ships, with the
  executable deltas proven empty by bytecode comparison.
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops on every
  model, disclosed as limitation 9 — and one arm's fully degenerate sampled completion passes
  the gate because of it.
