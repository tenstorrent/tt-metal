# Stage Review (round 8)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the seven previous reviews. Worktree live,
uncommitted (8 modified files + untracked `doc/optimized_vllm/`).

Verdict: **clean-pass**

Both of round 7's Required Work items are done, and I verified each from the artifacts rather
than the prose. Three of its four Other Concerns are done; the fourth (the "full common prefix"
clause, carried since round 2) is **partially** done — the 79-character scope is now stated, the
`<|message|>` explanation is not — and it remains what it has been for six rounds: true in
substance, thin in presentation. What is left across the whole report is five wording nits, four
of which I had to reconstruct the exact numbers to even notice. None of them misleads a reader
about the model, the measurements, or the shipped configuration, and the skill is explicit that
that is not what `more-work-needed` is for.

Only three files changed since round 7 (`README.md` 18:23, `tt/generator_vllm.py` 18:24,
`logs/degenerate_check_all.log` 18:24). I re-derived the evidence that those edits could have
invalidated, and re-ran the independent derivations rather than inheriting round 7's.

### Round 7's two Required Work items — both closed

* **P2 #1, the "What ships" claim.** `README.md:346-355` now reads "**no traced configuration
  measured is correct at every length it was measured at**", followed by "Three of the five
  traced sets have a measured wrong length; the other two were not run at 8192, which is a
  coverage gap and not a pass" and "Tracing off is the one configuration with no measured failure
  at any length." I rebuilt the six-configuration table from `prefill_trace_discriminators.json`
  and it is exactly what the new text says: 3 of 5 traced sets wrong at a measured length,
  `[1024]` and the 20-bucket set unrun at 8192. `grep` over `README.md`, `work_log.md`, every
  `doc/optimized_vllm/*.json`, `tt/*.py` and `bench/` returns **zero** surviving instances of
  "every one of the six configurations measured is wrong at some prompt length" or any variant.
  The false form is gone from the tree.

* **P2 #2, the loop table.** The causal clause on the `after_prefill_traced/` row is gone; the
  row is now "5 / 12 | the shipped arm's three plus p2 sampled 0.741 and p4 sampled 0.982". The
  missing `after_prefill_traced_1bucket/` row is present at `README.md:549` with "3 / 12 … and
  **p0 sampled 1.000** — a 6-word block repeated 32 times, the worst single completion in the
  corpus". The replacement paragraph (`README.md:559-566`) makes a new, falsifiable claim — that
  the two traced arms' qualitative ran off the eager path and "the differences between the arms
  are all in *sampled* completions" — and **it is exactly right**: comparing the three arms'
  artifacts byte for byte, all **six greedy** completions are identical across `after/`,
  `after_prefill_traced_1bucket/` and `after_prefill_traced/`, and all **six sampled** ones
  differ. The guard fires 0 times in `after/server_excerpt.log` and once in each traced arm's.
  So the eager-path account is not just asserted, it is corroborated by the greedy output being
  bit-stable across three different trace configurations.

### Re-derived independently this round

* **The loop metric recomputes exactly.** My own longest-repeating-word-block comparator
  (blocks of 4–80 words, non-overlapping repeats, coverage of the `\w+` token count) over each
  arm's own artifact returns, to three decimals and with the same pattern strings:
  `after/` 3/12 (p0 sampled 0.529 "12-word block x3", p1 greedy 0.708 "80-word x2", p2 greedy
  0.938 "33-word x6"); `after_prefill_traced_1bucket/` 3/12 (p0 sampled **1.000** "6-word x32");
  `after_prefill_traced/` 5/12 (p0 sampled **0.942** "6-word x30", + p2 sampled 0.741, p4 sampled
  0.982); `readiness_vllm/` 3/12 (0.708/0.938/0.600); chat verdict arm **0/6**. Identical to
  `loop_classification.json` in every field.
* **The 38-row discriminator matrix still reproduces the README table.** Collapsing
  `prefill_trace_discriminators.json`'s 38 rows by (config, prompt_len) with a conflict check
  yields the `README.md:295-302` table cell for cell, including every `—`, with no cell where two
  rows of the same (config, length) disagree.
* **Every headline number reproduces to the last digit** from the raw `run<N>/vllm_*.json`
  medians of runs 4–6, recomputed by me: primary TTFT 81.477 → 77.419 (−4.98 %), decode t/s/u
  43.4802 → 43.4278 (−0.12 %), TPOT 22.999 → 23.027, ITL 23.015/23.222 → 23.011/23.245,
  throughput 42.625 → 42.646, E2E 3002.6 → 3001.2; burst 721.877 → 717.560, TTFT
  2147.527/2148.757 → 2175.865/2177.187, TPOT 23.039 → 23.046, t/s/u 43.4053 → 43.3920, E2E
  4431.2 → 4457.8. Traced arms: 62.965 (1.294x), burst 1654.704 and 812.096 tok/s (+12.50 %),
  decode 43.4298; 60.662 (1.343x), burst 1691.072 and 805.384 tok/s (+11.57 %), decode 43.4694.
  Six-run TTFT ranges reproduce too: before 77.79–91.83, after 76.64–87.32, 1-bucket 59.37–70.45.
  1/1 and 32/32 completed, 0 missing tokens, in all six runs of both profiles in all four arms.
* **The late `generator_vllm.py` edit is provably inert.** The file was edited again at 18:24,
  after round 7's bytecode check and after the 17:42 `.pyc` that every late probe ran on. I
  compiled the current source and compared all **29** code objects against
  `tt/__pycache__/generator_vllm.cpython-312.pyc` on `co_code`, `co_names`, `co_varnames`,
  `co_consts` (frozenset ordering normalised), `co_argcount`, `co_flags`, `co_freevars`,
  `co_cellvars`: **zero** real differences. The only apparent one was the repr order of
  `frozenset({'on','yes','1','true'})`, which is set-iteration nondeterminism, not a change. The
  `PREFILL_TRACE_BUCKETS` rewrite is entirely `#:`-comment text.
* **Gates re-checked against the logs**: 29 passed / 35 deselected plain **and** under
  `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`; sampling 10 failed / 62 passed / 1
  skipped; the degenerate check re-run at **18:24** (after the last source edit) ends "No
  degenerate output detected" with 14 reported exclusions, all in corruption-characterisation
  directories, and no critical lines; `after/serving_audit.json` `clean true` with
  `degraded_markers_benchmark_window []`; non-aligned **9/9** `ok:true` at exactly the nine
  advertised lengths; served `max_model_len=131072`, `max_num_seqs=32`,
  `sample_on_device_mode=all` in `after/server_excerpt.log` against a 131072 contract with
  `capability_reduction: "none"`; `soak_traced_bucket` 60 + 24 = **84** generations over 10 + 4 =
  **14** rounds, worst `replacement_char_fraction` 0.0000, `all_stable true`, and
  `all_prompts_in_traced_bucket true` with all six prompts padding to 128.
* **The chat-arm identity claim is true in substance.** On all six prompts of
  `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json` the only difference between
  the vLLM head and the standalone head is the standalone's `<|message|>` token at offset 2;
  strip it and the heads are byte-identical over the whole compared span. That is exactly what
  `after/determinism_vllm.json` records (`comparison: "characters, baseline stripped of
  API-invisible special tokens"`, `compared_chars: 79`, `identical_over_common_prefix: true`,
  `first_char_divergence: -1`).

### The other four round-7 items

3. **"carries all 14 invocations" → fixed.** `README.md:669` now says "carries 13 invocations
   covering 13 of the 15 matrix probes, and names the two 16-step probes it deliberately does not
   re-run". The script has exactly 13 `run` invocations (the 14th `grep` hit is the `run ()`
   definition on line 32), writing 13 distinct probe artifacts, and lines 68-71 name
   `probe_full_shipped.json` and `probe_full_prefill_traced.json` as the two it does not re-run.
4. **"full common prefix" → partially fixed.** See *Other Concerns*.
5. **`PREFILL_TRACE_BUCKETS` docstring → fixed**, with a wording nit. It no longer implies single
   entries are wrong in general; it states the six-set result and names `[1024]` alone at 8192 and
   any bucket between 128 and 1024 as the unmeasured cells, pointing at
   `bench/run_discriminators.sh:73-77`, where they are indeed stated.
6. **Ballast-buffer row → fixed.** `README.md:628` records that `$autofix` was not invoked and
   points at `work_log.md` §8d, which exists (line 436) and gives the reason.

---

## Required Work

None.

## Other Concerns

- **`README.md:550` understates the 20-bucket arm's worst loop by shorthand.** The row says
  "the shipped arm's three plus p2 sampled 0.741 and p4 sampled 0.982". The three *positions*
  match, but the coverages do not: that arm's p0 sampled is **0.942** ("6-word block x30"), not
  the shipped arm's 0.529. One row above, the 1-bucket arm's p0 sampled is quoted explicitly as
  1.000 and called the corpus's worst, so a reader comparing the two rows will infer 0.529 for an
  arm that is actually the corpus's second-worst. The exact values are one file away in
  `loop_classification.json`, cited in the same section, and the following paragraph does say the
  arms differ in their sampled completions — but "the shipped arm's three" resolves to a wrong
  number. Both arms are non-shipped; this touches no ship decision. A four-word edit ("the
  shipped arm's three, with p0 sampled at 0.942, plus …") closes it.
- **`README.md:354`, "Tracing off is the one configuration with no measured failure at any
  length", is a uniqueness claim the matrix does not quite support.** `[1024]` alone also has no
  measured failure at any length — it was measured only at 1024 and 4097, and passed both. The
  sentence is true only under the reading "the one configuration measured at every length and
  clean at all of them", and the *immediately preceding* sentence supplies exactly that
  qualification ("the other two were not run at 8192, which is a coverage gap and not a pass").
  The direction of any misreading is conservative — it credits the traced sets with less safety
  than measured — and it defends the correct shipped default. Wording, in context.
- **An off-by-one between "configurations" and "bucket sets", in two places.**
  `tt/generator_vllm.py:152` says "**Six bucket sets** were measured and no traced one is correct
  at every length it was measured at", and `README.md:626` says "Rejected, **six configurations**
  measured" under the candidate "Any traced bucket set". Five traced bucket sets were measured
  (`[96]`, `[128]`, `[1024]`, `[128,1024]`, the 20-bucket list); the sixth configuration is
  tracing off, which is not a bucket set. Both sentences immediately enumerate the five by name,
  and "no traced one" shows the author is counting tracing-off among the six, so the reader can
  reconstruct it — but the README is careful to say "five traced configurations" at `:305` and
  `:353`, and these two do not match that. Same family as round 7's 14-vs-13 concern.
- **The same docstring sentence calls the wide set both failing and not-failing.**
  `generator_vllm.py:152-158`: "the wide set decays short-prompt output, `[96]`, `[128]` and
  `[128,1024]` each change a long eager prompt, and the two that have no measured failure
  (`[1024]` alone, and the wide set at 8192) were simply not run there." The intended sense is
  "no measured failure *at a length*", and the bullets 40 lines below document the wide set's
  decay in full, so the paragraph corrects itself — but as one sentence it says the wide set both
  does and does not have a measured failure.
- **`README.md:388` and `:535` still say "character-identical to the standalone model over the
  full common prefix", and the artifact a reader is sent to says `identical: false` six times.**
  Round 7 asked for one clause; what landed is the *scope* half at `README.md:392` ("identical
  over the 79-char common prefix the check compares at `max_tokens 24`; the longer 127-token
  comparison is in `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`") and **not**
  the *explanation* half. `grep` for `message`, `first_divergence` or `149` over `README.md` and
  `work_log.md` returns nothing: no surface in the report says that the longer artifact reads
  `identical: false, first_divergence: 2` on all six prompts because the API does not render
  `<|message|>`, nor that the heads are byte-identical once it is stripped. I verified that they
  are, and `determinism_vllm.json` records the strip in its `comparison` field, so the claim is
  true and the evidence exists — but the README now *points at* the artifact that appears to
  contradict it without saying why. Carried from rounds 2–7; still one clause.
- **`README.md:680` and `work_log.md` §8c lag one review round.** The artifact table says
  "independent stage reviews, **six** rounds | `stage_review.md`, `stage_review_round2.md` …
  `stage_review_round6.md`", and §8c's table ends at round 6 with the header "Six independent
  `$stage-review` rounds ran against this stage". `stage_review_round7.md` (18:22) exists in the
  same directory and is not indexed anywhere, and `work_log.md` (18:12) predates it, so none of
  the six round-7 fixes is recorded in the stage narrative. Housekeeping the stage owner will
  presumably do when recording this round, but the goal contract does ask the work log to record
  the stage's decisions, and right now the last round of them is only in the README diff.

## Hard-Check Gaps

- `[1024]` alone at 8192 remains unmeasured, so "largest captured trace size" is not separated
  from "any small resident bucket poisons long eager prefills". Disclosed in all four required
  surfaces (`README.md:311-315`, `perf_summary.json:blocker.coverage_limits`, the shipped
  `_PREFILL_TRACE_ENV` docstring, `bench/run_discriminators.sh:73-77`); one probe away; does not
  change the ship decision, because `[128,1024]` failing 8192 settles it either way.
- Nothing measured between bucket sizes 128 and 1024; trace counts 3–19 unmeasured. Disclosed.
- The freed-intermediate address range of one prefill trace is still unmeasured (rounds 1–8).
  `doc/optimized_full_model/prefill_trace_probe.json` gives `capture_retained_dram_bytes
  3280896` and no peak-during-capture reading, so "twenty 52-layer prefill working sets" versus
  "a small, decode-shaped range" is still an assertion, and it is the quantitative core of the
  one mechanism the stage claims. It is also the stage's second upstream ask, which is the right
  place for it.
- Still no live-server evidence of the 4097/8192 divergence: the non-aligned check's 4097 and
  8193 `text_head` values are identical across arms because in the traced arms that step runs
  after the interlock released the traces. One step reorder in `bench/run_arm.sh` would produce
  the datum; not taken (rounds 4–8).
- At 8192 and 100 the tracing-off reference is a single session, so those ✅ cells are
  self-comparisons; 1024 has four independent sessions and 4097 has three. Adequate for the
  conclusion drawn.
- The shipped headline arm (`after/`, 13:39) predates the final source. Bounded: `tt/generator.py`
  unchanged since 15:31, and all three late `generator_vllm.py` edits (including the 18:24 one
  made after round 7) are proven comment-only by a 29-code-object bytecode comparison against the
  17:42 `.pyc` that every late probe ran on.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm; the
  decode path is unchanged by this stage.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` and
  `doc/context_contract.json`.
- No device-time/roofline term, by instruction (`$optimize`/`$vllm-integration` forbid the
  profiler in vLLM stages); recorded as `null` with the reason.

## Anomaly Ledger

- Observed anomaly: with a prefill trace **captured** (not necessarily replayed), long eager
  prefills diverge from their first token; a largest bucket of 1024 fixes 4097 but not 8192.
  Evidence: all 38 matrix rows; the collapsed table reproduces `README.md:295-302` exactly with
  no intra-cell conflicts; `[128]` and `[128,1024]` fail 8192 byte-identically; `[96]`, `[128]`
  fail 4097 byte-identically; `[1024]`, `[128,1024]`, 20 buckets get 4097 right.
  Affected path: eager prefill of an out-of-bucket prompt, in a process that captured at least
  one prefill trace.
  Control or comparison: same-revision tracing-off controls at every length ✓; capture-only (no
  replay) isolated ✓; warmed-shape ruled out ✓; bucket-value quirk ruled out ✓; no live-server
  control.
  Likely subsystem: ttnn mesh trace capture / allocator lifetime. Unknown.
  Investigation performed: fifteen probes over six configurations, every request tabulated, 13 of
  the 15 reproducible by one committed script and the other two named.
  Resolution: **controlled** — the configuration does not ship, the matrix is in the tree and
  re-derives, the mechanism is labelled UNEXPLAINED in four places, the configuration that would
  have kept the win (`[128,1024]`) is measured and broken, the coverage limits are stated in four
  surfaces, and as of this round every surviving description of the result is accurate.

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `soak_traced_bucket/` clean over 84 in-bucket generations, 14 rounds, all byte-stable.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, a valid in-bucket soak.
  Resolution: **controlled** — does not ship; reproducer, refutations and interlock in tree.

- Observed anomaly: `after_prefill_traced_1bucket/`'s p0 sampled completion is a 6-word block
  repeated 32 times, coverage **1.000** — the worst single completion in the stage's corpus.
  Evidence: my recomputation over that arm's `vllm_qualitative_outputs.json`, matching
  `loop_classification.json:after_1bucket`.
  Affected path: runner raw-completion arm (bare prompts to a chat model), sampled, **eager
  path** — the guard fired once in that arm's `server_excerpt.log` before its qualitative step.
  Control or comparison: all six **greedy** completions are byte-identical across `after/`,
  `after_prefill_traced_1bucket/` and `after_prefill_traced/`; all six **sampled** ones differ.
  So the arms differ only where the sampler introduces a draw, which is the report's claim.
  Chat verdict arm and HF control are 0/6.
  Investigation performed: byte comparison of all 36 completions across the three arms; guard-fire
  counts; loop recomputation for four corpora.
  Resolution: **controlled and now recorded** — the row is in the README table with its 1.000
  coverage and its pattern, the eager-path caveat is stated, and it is named as limitation 9 at
  its worst. Round 7's P2 #2 is closed.

- Observed anomaly: the 1024-token probe returns a 2-token cycle, `distinct_tokens 4`.
  Control or comparison: tracing-off control byte-identical, four independent sessions.
  Resolution: **controlled**, recorded as a negative result.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm,
  3/12. Resolution: **controlled**; recomputed here and consistent on all three surfaces, equal to
  the previous stage's arm and sharing p1/p2 at identical coverage; absent from the chat verdict
  arm and the HF control.

- Observed anomaly: the qualifying soak's completions are the `" to=self"` analysis channel.
  Resolution: **controlled** (classified in earlier stages as Harmony-style channel tokens
  invisible over the API); weaker readable-text evidence than described, unchanged from rounds
  3–7.

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest and
  sampling log. Identical in the before arm and previous stages. Resolution: **controlled**.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `stage_review_round7.md` (both Required Work items and all four Other Concerns
  re-derived from artifacts).
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,prefill_trace_discriminators.json,
  loop_classification.json}`; `doc/optimized_vllm/bench/run_discriminators.sh`;
  every `run<N>/vllm_benchmark.json` and `run<N>/vllm_ci_serving_benchmark.json` in `before/`,
  `after/`, `after_prefill_traced_1bucket/`, `after_prefill_traced/`;
  `after/{serving_audit.json,determinism_vllm.json,server_excerpt.log,sampling_tests.log}`;
  `after/qualitative/{qualitative_tt_chat.json,qualitative_vllm_vs_datatype_sweep_chat.json}`;
  `after/vllm_qualitative_outputs.json`, `after_prefill_traced{,_1bucket}/
  {vllm_qualitative_outputs.json,server_excerpt.log}`; `readiness_vllm/
  vllm_qualitative_outputs.json`; `soak_traced_bucket/{soak_traced_bucket.json,
  soak_traced_bucket_after_mixed.json}`; `logs/{pytest_final,pytest_watcher,
  degenerate_check_all}.log`; `doc/context_contract.json`.
- Code paths: `tt/generator_vllm.py` (`PREFILL_TRACE_BUCKETS`, `_PREFILL_TRACE_BUCKETS_ENV`,
  `_prefill_trace_buckets`, `_PREFILL_TRACE_ENV`, `PREFILL_WARMUP_LENGTHS`) and its 17:42 `.pyc`.
- Commands run (all read-only; no server, device, hardware or vLLM use): `git log`,
  `git status`, `ls`, `find -newermt`, `grep`, `sed`, `tail`, and Python scripts that collapsed
  the 38 discriminator rows into the README's matrix with a conflict check, recomputed warm
  medians, deltas, speedups and six-run ranges for all four arms and both profiles, recomputed
  the longest-repeating-block loop metric for five qualitative corpora, byte-compared all 36
  greedy/sampled completions across the three benchmark arms, compared the vLLM and standalone
  chat heads with `<|message|>` stripped, and compared all 29 code objects of the current
  `generator_vllm.py` against its pre-edit `.pyc` with frozenset ordering normalised.

## Residual Risk

- `[1024]` alone at 8192 is unmeasured, so "largest captured trace size" is supported by three
  points and not separated from "any small resident bucket poisons long eager prefills". Disclosed
  in four surfaces, including the shipped docstring.
- The mechanism is genuinely open, and no measurement bounds the poisoned address range for any
  trace size. Both facts are the stage's two upstream asks.
- The shipped headline arm was measured on code ~4.5 h older than what ships, with the executable
  deltas proven empty by bytecode comparison — re-verified this round after the 18:24 edit.
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class, measured
  twice against one server.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops on every
  model, disclosed as limitation 9 — and one non-shipped arm's fully degenerate sampled completion
  passes the gate because of it. Now recorded with its coverage and pattern.
- The stage narrative (`work_log.md` §8c) and the README artifact index are one review round
  behind the review files actually in the directory.
