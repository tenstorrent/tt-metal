# Stage Review (round 3)

Stage 10, optimized-vLLM serving — `meta-models/Muse-Glimmer-30B`
Reviewed against the supplied goal contract, `$optimize`, `$vllm-integration`,
`$tt-enable-tracing`, `$tt-device-usage`, and the two previous reviews
(`stage_review.md`, `stage_review_round2.md`).
Worktree live, uncommitted (8 modified files + untracked `doc/optimized_vllm/`).

Verdict: more-work-needed

**The decision to ship tracing off is the right call and I am not asking for it to be
reversed.** A candidate that silently changes another request's output, with a named
allocator-contract mechanism, an in-repo reproducer, an interlock and an upstream ask, is a
rejection `$optimize` accepts: "fails correctness for an understood reason". Three fixes were
implemented and measured, the machinery ships behind an env gate, and the headline honestly
says nothing moved.

**Every headline number reproduces from raw JSON.** I recomputed the median-of-runs-4–6 for
every arm and profile from `run*/vllm_benchmark.json` and `run*/vllm_ci_serving_benchmark.json`
independently of `bench/collect_metrics.py`: before 81.48 ms / 43.480 t/s/u / 22.999 / 23.015 /
23.222 / 42.63 / 3002.6; after 77.42 / 43.428 / 23.027 / 23.011 / 23.245 / 42.65 / 3001.2;
burst before 2147.53 / 2148.76 / 721.88 / 23.039 / 43.405 / 26.214 / 4431.2; burst after
2175.86 / 2177.19 / 717.56 / 23.046 / 43.392 / 25.449 / 4457.8; 1 bucket 62.97 / 1654.70 /
812.10 / 43.430; 20 buckets 60.66 / 1691.07 / 805.38 / 43.469. Run ranges, −5.0 %, −0.6 %,
1.294x, 1.343x, +12.50 %, +11.57 % all check out, and `metrics.json` matches to the last
digit. The sampling failure sets reproduce member-by-member from all six `sampling_tests.log`
files, including the floating seventh member table. Acceptance tests (29 plain at 16:01,
29 under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` at 16:05, 20 watcher dumps,
zero error-shaped lines) genuinely postdate the last code edit (15:57). Round-2's P1 about the
void soak is properly closed: `bench/soak_traced_bucket.py` asserts padding to 128 before it
sends anything, `soak_traced_bucket/`'s prompts are 108–115 tokens → padded 128, its server
logs `prefill traces resident for padded buckets [128]` and never released them, and 84
generations came back 0.0000 and byte-stable.

What does not hold up is (a) the characterisation of the 4097 divergence, which is the single
piece of evidence the whole "cannot ship at any bucket count" conclusion rests on and which is
propagated verbatim into shipped code docstrings, and (b) a cluster of stale-after-rename
errors: the arms were renamed at 15:33 and the README section that explains the interlock, the
qualitative classification, the `$optimize` checklist, the reproduction commands, two
`server_log_size.txt` files, two audit `path` fields and the entire `logs/after_*` set now
describe the wrong arm.

---

## Required Work

- **P1: The 4097 divergence — the stage's load-bearing evidence — is not explained by the
  mechanism the stage gives it, and the discriminating experiments were not run. One of its
  two data points comes from a different revision of the function under test.**

  Evidence:
  - The divergence itself is real and cleanly controlled at 1 bucket. I re-derived it:
    `probe_repro_traced.json` (bucket `[128]` resident, 15:48) returns
    `4097 -> [576, 5824, 761, 426, 426, ...]`, `distinct_tokens 4`; `probe_repro_eager.json`
    (tracing off, 15:52, same code) returns `[198, 6453, 107177, 38, 372, 2556, ...]`, which is
    byte-identical to `doc/vllm_integration/probe_full_fixed.json`. `probe_full_shipped.json`
    (15:34) reproduces the divergent sequence exactly. The other three lengths (128, 100, 37)
    and the three-slot multi-request section are identical in every arm. The decode-steps
    mismatch (16 vs 8) is not a confound: the *first* token — the prefill's own output —
    already differs, and `prefill_counters.trace_replays` is `0` for the 4097 request in every
    arm, so it is the eager path in all of them.
  - **The stated mechanism cannot produce this ordering.** README *Mechanism* and
    `perf_summary.json.blocker.rule` quote `allocator.cpp:113-126`: buffers allocated while a
    trace is active "may be corrupted **once a trace is executed**". In the probe the 4097
    request's buffers are allocated *after* the last prefill-trace replay (request 1) and
    nothing replays that trace again before the wrong logits are produced — `_prefill_traced`
    returns `None` for padded 4128 (`tt/generator.py:888`), so no trace executes between the
    allocation and the result. The write-up does not notice this, and the mechanism section is
    unchanged from when it only had to explain the 20-bucket decay.
  - **The non-monotonicity claim is cross-revision.** `probe_full_prefill_traced.json` (12:55,
    the only "correct at 20 buckets" datum) ran a generator with
    `_capture_prefill_trace:1023`, `_capture_decode_trace:1587`, `build_generator:2427`;
    the 1-bucket probes ran `1036/1600/2440` (15:34) and `1048/1612/2452` (15:48).
    `enable_prefill_trace:842` is identical in all three, so the ~25 inserted lines lie between
    842 and 1023 — the `_prefill_traced` / `_capture_prefill_trace` body. `work_log.md` §3.3
    says the replay "shipped `blocking=False` first … it is back to `blocking=True`", and
    `soak_blocking/` (13:25–13:38) is the arm that tested the blocking form — i.e. after 12:55.
    So the 20-bucket probe was almost certainly taken with a **non-blocking** traced replay,
    which is exactly the allocation-versus-replay ordering the stage's own mechanism says
    governs the hazard. No same-revision 20-bucket probe exists.
  - The conclusion drawn from that pair is quoted as fact in four shipped places:
    README *The one-bucket configuration* ("correct at 20 buckets and wrong at 1 … no bucket
    count that can be argued safe"), README *What ships*, `work_log.md` §6,
    `perf_summary.json.blocker.at_1_bucket`, and the `_PREFILL_TRACE_ENV` docstring in
    `tt/generator_vllm.py` ("this cannot ship at any bucket count: the failure is not monotone
    in the number of traces").
  - **Cheap discriminators that were not run** (all one probe invocation each, no server):
    4097 alone with the bucket resident (no in-bucket request first — separates "capture
    poisoned something" from "replay poisoned something"); 4097 ordered first; a bucket other
    than 128 (e.g. `[96]`, so the in-bucket requests are 37 rather than 128/100); 20 buckets on
    the shipped revision. The stage instead re-ran the same 4-length ordering twice.
  - There is also an untested port-owned hypothesis that fits the observation better than the
    allocator rule does: the capture path leaves per-bucket persistent state
    (`entry["tokens"]`, `entry["page_table"]`, and whatever `model.prefill_tokens_to_device` /
    the layer stack caches), and the *last* bucket captured is 128 in the failing arm and 1024
    in the passing one. If the divergence is port-owned state rather than a ttnn allocator
    constraint, it is fixable and the 1.29x is recoverable — and the same capture path ships
    behind `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1` with a README that tells deployments they may
    enable it after soaking their own prompt distribution.
  - No arm records a code fingerprint (no commit, no file digest). The only way I could date
    the revisions was loguru line numbers, and by them **every** benchmark arm ran a different
    revision of `tt/generator_vllm.py`: `after/` 13:39 (`allocate_kv_cache:552`),
    `after_prefill_traced_1bucket/` 14:39 (`:564`), `after_prefill_traced/` 12:31 (`:519`),
    current `:566`.

  Why this matters: `$stage-review` requires more work when "logs or code show a plausible bug
  in a stage-critical subsystem, such as … trace replay" and when "the stage dismisses a
  material anomaly with prose instead of investigation". The ship-off decision survives either
  way — but the stage states as measured fact something it measured across two builds, and
  publishes a mechanism that does not account for the observation, in shipped code comments a
  future reader will act on.

  Required next step: re-run the 20-bucket probe on the shipped revision; run the
  4097-only / 4097-first / different-bucket probes; then either state a mechanism consistent
  with an eager prefill being wrong with no intervening replay, or record it explicitly as
  unexplained and downgrade "not monotone in the number of traces" to what the same-revision
  evidence supports. Record a code fingerprint per arm.

- **P2: README §4 attributes the interlock firing to `after/`, an arm that has no prefill
  traces at all, so the caveat that both traced arms' qualitative/determinism ran eagerly is
  now attached to the wrong arm — and §4 still cites the void soak.**

  Evidence:
  - README lines 149–154: "It fires once in the `after/` arm, during the sampling suite's
    seeded tests, which is why that arm's qualitative and determinism ran on the eager path…
    Qualitative evidence *with the trace resident* comes from `soak_1bucket/` instead."
  - `after/` never enabled tracing. On the raw 84 MB log:
    `grep -c "prefill tracing enabled"` = 0, `"prefill traces resident"` = 0, `"PREFILL_TRACE"`
    = 0; `after/serving_audit.json` has no
    `DEGRADED PATH prefill_traces_released_for_sampling_capture`. The marker appears exactly
    once in `after_prefill_traced_1bucket/` and once in `after_prefill_traced/`.
  - So the real statement is: **in both traced arms the traces were released during the
    sampling step, and `bench/run_arm.sh` orders `sampling` before `qualitative`, `qualchat`
    and `determinism`** — therefore the traced arms' qualitative, determinism, cross-batch and
    non-aligned evidence is eager-path evidence. README line 89 says instead "Both traced arms
    ran the full gate set — six benchmark runs, the whole sampling suite, both qualitative
    arms, determinism — and both passed it", with no such caveat anywhere.
  - `soak_1bucket/` is declared **void** at README line 210 and `work_log.md` §8, yet it is
    still cited as the traces-resident qualitative evidence at README line 154 and in the
    `$optimize` checklist row at line 545 ("`$qualitative-check` after the optimization |
    `after/qualitative/`, plus the traces-resident arm in `soak_1bucket/`"). The correct
    citation is `soak_traced_bucket/`.

  Required next step: rewrite §4 for the renamed arms, move the release caveat onto the traced
  arms and repeat it next to the "both passed the full gate set" sentence, and replace the two
  `soak_1bucket/` citations with `soak_traced_bucket/`.

- **P2: The runner-qualitative classification describes a different arm's artifact and
  undercounts by its own stated criterion.**

  Evidence (recomputed over `after/vllm_qualitative_outputs.json`, longest word-block repeated
  ≥2× non-overlapping, coverage of the completion):
  - `p0 sampled` — 12-word haiku block ×3, coverage **0.507**;
    `p1 greedy` — the entire 61-word answer restated verbatim after its own summary, coverage
    **0.545**; `p2 greedy` — 32-word story sentence ×6, coverage **0.937**. That is **3 of 12**
    over the README's own ">40 % long-verbatim loop" threshold, not 2 of 12. The table is only
    2/12 under an unstated "≥3 occurrences" rule. `readiness_vllm/` is likewise 3/12
    (p1 greedy 0.545, p2 greedy 0.937, p2 sampled 0.597), not 2/12.
  - The prose is about a different arm: README Status and *Qualitative* say "p0 sampled repeats
    the prompt sentence 26 times". In `after/` p0 sampled contains the prompt sentence **twice**;
    the completion that repeats it verbatim (32 occurrences) is
    `after_prefill_traced_1bucket/vllm_qualitative_outputs.json` — the directory that *was*
    `after/` when round 2 counted 31. The 20-bucket arm's p0 sampled is a third text again
    ("I cannot comply with that request." ×N).
  - Round 2 named `prompt[1] greedy_completion` explicitly; it is still absent from the prose
    and from the count.
  - The chat verdict arm (0/6) and the HF control (0/6) do reproduce, so the *verdict* is
    unaffected; the classification of the caveat is what is wrong.

  Required next step: recompute the table against the shipped arm's own artifact with the
  metric written down (or committed as a script), fix the two counts and the p0 description,
  and name p1 greedy.

- **P2: The shipped arm's driver and per-step logs were overwritten by the 1-bucket arm, and
  the surviving files say the opposite of the truth about it.**

  Evidence:
  - `logs/after_arm.log` and `logs/after_{bench1..6,sampling,qualitative,qualchat,qualcompare,
    determinism,audit,serve_hold}.log` are all timestamped 14:43–14:50. The shipped `after/`
    arm ran 13:39–13:51 (`after/bench_window_end_bytes.txt` 13:45:57,
    `after/serving_audit.json` 13:51:12). The 14:39–14:50 run is the one that produced
    `after_prefill_traced_1bucket/` (`server/` created 14:39:18, `serving_audit.json`
    14:50:54, matching `logs/after_arm.log`'s "audit 18:50:53" UTC).
  - `logs/after_arm.log` header reads `arm=after steps=… prefill_trace=unset`. At 14:39 *unset*
    meant the then-default **1 traced bucket**, so a reader checking the shipped arm's
    configuration from its logs is told the exact opposite of what shipped.
  - `after_prefill_traced_1bucket/` has no logs under its own name; the shipped arm has none at
    all. The before arm's overwritten `server.log` was disclosed in `work_log.md` §2 and
    `before/server_log_size.txt`; this one is not disclosed anywhere.

  Required next step: relabel the logs to the arm that produced them (or record inside each
  file which arm it belongs to, as the before arm's stub does), and note in the work log that
  the shipped arm's driver log did not survive the rename.

- **P2: Stale paths, stale reproduction commands, and no command recorded for the qualifying
  experiment.**

  Evidence:
  - `after/server_log_size.txt` names
    `…/doc/optimized_vllm/**after_prefill_eager**/server/server.log` — a directory that no
    longer exists. `after_prefill_traced_1bucket/server_log_size.txt` names
    `…/**after**/server/server.log`. (Round 2 flagged this class; `after_prefill_traced/` was
    fixed and the rename broke two others. Both files were touched at 15:33:32, the rename
    moment, without their contents being updated.)
  - `after_prefill_traced_1bucket/serving_audit.json` and `after_prefill_traced/serving_audit.json`
    both record `logs[0].path = …/after/server/server.log`; the 20-bucket arm's real log is
    55,435,345 bytes at `after_prefill_traced/server/server.log`, while `after/server/server.log`
    is 84,124,291 bytes.
  - README *Serving configuration* still says `# after: this stage's shipped configuration
    (1 traced bucket, on by default)` and `# the one-bucket soak that qualified the shipped
    configuration` above the `soak_1bucket` command. Neither is true, and the command for
    `soak_traced_bucket/` — the experiment that actually qualifies the traced bucket in this
    round — appears in neither README nor `work_log.md`, nor does the `probe_pair` command.
    The goal contract requires the README/work log to record commands.
  - README *Rejected and deferred*, "More traced buckets (2–20)": "20 corrupts within **~9
    requests**, 1 is clean over **two independent soaks**; … Widening is one env variable plus a
    soak." The ~9 figure was corrected to the 22nd generation elsewhere in the same README,
    one of the two soaks is the void one, and "widening is one env variable plus a soak"
    contradicts "this cannot ship at any bucket count" two sections earlier.
    `work_log.md` §8 also still carries "~9 requests".
  - `work_log.md` §6 narrates `soak_1bucket/` as "**84 completions**, every one at
    `replacement_char_fraction` 0.0000, on the exact traffic pattern that corrupted the
    20-bucket arm" and never records that it was void or that `soak_traced_bucket/` replaced
    it; the correction lives only in the §8 table.

  Required next step: fix the four stale paths, the two command comments, the "~9 requests" and
  "two independent soaks" rows, add the `soak_traced_bucket` and probe commands, and correct
  the §6 narrative.

- **P2: `probe_full_shipped.json` still is not the shipped configuration, and the
  contract-evidence section cites only the two traced probes.**

  Evidence:
  - `probe_full_shipped.json`: `prefill_trace.enabled true`, `buckets_resident [128]`,
    `capability_report…prefill_trace_enabled True`. The shipped default is off.
  - README *Contract evidence — the measured path*: "From `probe_full_prefill_traced.json` and
    `probe_full_shipped.json`"; the quoted steady-state block (`trace_replays 16 …`) is the
    16-step traced probe. `$optimize` checklist: "Decode path fully traced, no host fallbacks |
    `probe_full_shipped.json` steady-state counters".
  - The probe that *does* cover the shipped configuration is `probe_repro_eager.json`, and I
    verified it covers every contract item: `prefill_trace_enabled False`, async read mode,
    stale inputs on, multi-request `trace_replays 8 / token_refreshes 1 / position_refreshes 1 /
    page_table_refreshes 1 / synchronizations 0 / readbacks 8`,
    `sampling_param_refreshes 1 / reuses 7`, page-table unchanged 0 refreshes + byte-identical,
    changed 1 refresh + matches new table, host-sampling logits `[32,1,202048]` finite,
    `rows_are_distinct true`, and token-identity with the previous stage on all three shared
    lengths. It is cited only in the Artifacts table.
  - This is round-2's P2 recurring after the default flip; the file name now asserts something
    false about the stage's own default.

  Required next step: rename the file (`probe_full_prefill_traced_1bucket.json`) or regenerate
  it, and cite `probe_repro_eager.json` in *Contract evidence* and the checklist row.

---

## Other Concerns

- **The two stacked contradictory docstrings on `PREFILL_TRACE_BUCKETS` are still there**
  (`tt/generator_vllm.py:147-183`). The first block describes the 20-bucket set — "these are
  exactly the short padded lengths `PREFILL_WARMUP_LENGTHS` already compiles", "8192 is
  deliberately absent" — and runs straight into the new block that says the opposite. Round 2
  flagged it; unfixed, in shipped code, above the constant a reader is looking up.
- **README *Decode is unchanged* quotes the wrong arm's decode numbers.** "serving measures
  23.026 ms/token, 43.430 t/s/u" is `after_prefill_traced_1bucket/`; the shipped arm is
  23.027 / 43.428, as the headline table says. Immaterial numerically (100.2 % either way),
  stale after the rename.
- **`perf_summary.json` burst before `decode_tps_u` is 43.407**; `metrics.json` and the README
  both say 43.405.
- **The qualifying soak's readable text is thinner than described.** All six completions in
  `soak_traced_bucket.json` begin `" to=self"` followed by a verbatim restatement of the
  question, and only `head[:100]` of a 96-token generation is retained, so "reads the text …
  and see that they stay English" is really "the analysis channel restates the prompt,
  stably". It is consistent with the chat arm's known Harmony-channel behaviour and is a
  perfectly good U+FFFD / byte-stability detector, but nobody appears to have read it.
- **The non-aligned check does record comparable text, contrary to the README.** README *What
  ships* says "the server-side non-aligned check asserts only that the request succeeded";
  `determinism_vllm.json.non_aligned_prompt_lengths` also stores `text_head`, and all nine
  heads are byte-identical across `after/`, `after_prefill_traced_1bucket/` and
  `after_prefill_traced/` — including 4097. That is *not* a contradiction of the probe, because
  `determinism` runs after `sampling`, which released the traces in both traced arms; but it
  means a live-server test of the 4097 case was one step-reordering away and was not taken.
- **`work_log.md` §8: "metrics.json folds every arm above"** — it folds 7 arms
  (`before`, `after`, `after_prefill_traced_1bucket`, `after_prefill_traced`, `before_sweep0`,
  `soak_1bucket`, `soak_traced_bucket`, the last with an empty `profiles`), against a 10-row
  table.
- **Status: "character-identical to the standalone model over the full common prefix"** is 79
  characters at `max_tokens 24` (`after/determinism_vllm.json.standalone_baseline`). The
  stronger comparison — `after/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`, 127
  tokens, diverging only on the API-invisible `<|message|>` — exists and is still not the one
  cited. Carried unchanged from round 2.
- **`$autofix` is still not mentioned anywhere** (`grep -c autofix README.md work_log.md` → 0,0),
  and the ballast-buffer mitigation remains the one named-but-untried option, while the stage's
  own evidence now says the hazard is not what it thought it was (P1).
- `logs/after_sampling_reps.log` and `logs/after_sampling_reps_serve_hold.log` survive from the
  abandoned arm whose tree `work_log.md` §9 says was deleted.

---

## Hard-Check Gaps

- The freed-intermediate address range of one prefill trace is **still unmeasured** — a round-1
  and round-2 ask. `doc/optimized_full_model/prefill_trace_probe.json` gives
  `capture_retained_dram_bytes 3280896` but no peak-during-capture reading, so "twenty 52-layer
  prefill working sets" versus "a small, decode-shaped range" remains an assertion, and it is
  the quantitative core of the mechanism section.
- Nothing between 2 and 19 buckets, and now nothing at 20 on the shipped revision (P1).
- No live-server evidence of the 4097 divergence in any configuration; the only evidence is the
  in-process adapter probe.
- The shipped headline arm (`after/`, 13:39) predates the final code by ~2 h 20 m
  (`tt/generator.py` +12 lines, `tt/generator_vllm.py` +14 lines). Both deltas are in the
  traced-prefill path, which is off in that arm; the final code is covered by 29 passing
  acceptance tests and by `probe_repro_eager.json`'s token-identity, so the risk is low, but
  the headline arm itself was not reproduced on the shipped bits.
- `supports_async_decode=True` still rests on the previous stage's `--async-scheduling` arm;
  the decode path is unchanged, so this remains reasonable.
- No long-context serving generation; 131072 is evidenced by served `max_model_len` (verified in
  `after/server_excerpt.log`) and `doc/context_contract.json`. Unchanged from rounds 1–2 and
  acceptable for this stage.

---

## Anomaly Ledger

- Observed anomaly: with one prefill trace resident, a 4097-token prompt served on the **eager**
  path diverges from its first token.
  Evidence: `probe_repro_traced.json` and `probe_full_shipped.json` (identical divergent
  sequence, `distinct_tokens 4`, `trace_replays 0` on that request) vs `probe_repro_eager.json`
  (same code, same session pair) and `doc/vllm_integration/probe_full_fixed.json`.
  Affected path: eager prefill with a captured prefill trace resident.
  Control or comparison: same-revision tracing-off control ✓; 20-bucket "correct" comparison is
  cross-revision ✗; no live-server control (the non-aligned check ran after the interlock had
  released the traces).
  Likely subsystem: ttnn trace/allocator lifetime **or** port-owned per-bucket capture state —
  the stage names only the first, and the first does not explain a wrong result with no trace
  execution between allocation and use.
  Investigation performed: reproduced twice; no isolation, ordering, alternate-bucket or
  same-revision 20-bucket probe.
  Resolution: **more-work-needed** (P1).

- Observed anomaly: served output decays into U+FFFD with 20 prefill traces resident, from the
  22nd generation, byte-identically across two servers.
  Evidence: `traced_qualitative/`, `soak_blocking/runner_qual1/`, `bisect_server/qualitative3`,
  `fixcheck/qualitative{2,3}`.
  Affected path: serving with 20 resident prefill traces.
  Control or comparison: `ctrl_notrace/` healthy either side of the sampling suite;
  `prefill_trace_bisect.json` token-identical, so not math; `soak_traced_bucket/` clean at 1
  bucket over 84 in-bucket generations with the guard never firing.
  Likely subsystem: trace/allocator lifetime.
  Investigation performed: 4-step in-server bisection, two refuted fixes, an interlock, a
  capacity ladder, a bucket-count ladder, and a valid in-bucket soak.
  Resolution: **controlled** — the configuration does not ship, and the reproducer, the
  refutations and the interlock are all in the tree.

- Observed anomaly: mechanical verbatim looping in the shipped arm's runner raw-completion arm.
  Evidence: `after/vllm_qualitative_outputs.json` p0 sampled 0.507, p1 greedy 0.545, p2 greedy
  0.937 by long-block coverage; `logs/degenerate_check_all.log` scores them by
  `trigram_loop_fraction` at ~0.08–0.09 and passes.
  Affected path: raw-completion prompts against an instruct checkpoint; the shared loop metric.
  Control or comparison: identical values in `readiness_vllm/` from the previous stage; chat
  verdict arm 0/6; HF control 0/6. Pre-existing and prompt-shaped, not stage-introduced.
  Likely subsystem: prompt format, plus a blind spot in the shared checker.
  Investigation performed: classified in this round with a control table — but the table's
  counts and the prose describe a different arm's artifact (P2).
  Resolution: **more-work-needed** (documentation/count only; the verdict arm is clean).

- Observed anomaly: the qualifying soak's completions are the model's `" to=self"` analysis
  channel restating the question.
  Evidence: all six `head` fields in `soak_traced_bucket/soak_traced_bucket.json`.
  Control or comparison: `after/qualitative/qualitative_tt_chat.json` shows the same channel
  prefix and the same restate-then-reason shape on the prompt-correct chat arm; classified in
  earlier stages as Harmony-style channel tokens invisible over the API.
  Likely subsystem: prompt format / channel tokens.
  Investigation performed: cross-checked against the chat arm here.
  Resolution: **controlled** (noted under *Other Concerns* as weaker readable-text evidence
  than described).

- Observed anomaly: `nanobind: leaked N instances/types/functions` at the end of every pytest
  and soak log.
  Evidence: `logs/pytest_final.log`, `logs/soak_traced_bucket.log`, every `*/sampling_tests.log`.
  Control or comparison: identical in the before arm and previous stages.
  Resolution: **controlled**.

---

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md` (read in full); the goal contract as
  supplied; `doc/optimized_vllm/stage_review.md` and `stage_review_round2.md` (each required-work
  item re-derived).
- Artifact paths (under
  `/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
  `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,metrics.json}`;
  every `run*/vllm_benchmark.json` and `run*/vllm_ci_serving_benchmark.json` in `before/`,
  `before_sweep0/`, `after/`, `after_prefill_traced_1bucket/`, `after_prefill_traced/`,
  `soak_1bucket/`; every `serving_audit.json` (all windows and marker classifications),
  `sampling_tests.log`, `server_excerpt.log`, `server_log_size.txt`,
  `bench_window_end_bytes.txt`; `soak_traced_bucket/{soak_traced_bucket.json,
  soak_traced_bucket_after_mixed.json,vllm_qualitative_outputs.json,server/server.log}`;
  `probe_repro_traced.json`, `probe_repro_eager.json`, `probe_full_shipped.json`,
  `probe_full_prefill_traced.json`, `probe_trace_capacity.json`,
  `doc/vllm_integration/probe_full_fixed.json`; `after/determinism_vllm.json` and the other
  arms' non-aligned sections; `after/qualitative/*` and every arm's
  `vllm_qualitative_outputs.json`; `readiness_vllm/vllm_qualitative_outputs.json`;
  `doc/full_model/qualitative/qualitative_hf_chat.json`;
  `doc/datatype_sweep/evidence_perf.json`; `doc/optimized_full_model/prefill_trace_probe*.json`,
  `ccl_host_probe_bfp8.json`; `doc/vllm_integration/kv_budget_probe.json`;
  `logs/` (degenerate_check_all, negative control, run_tests, run_watcher, pytest_final,
  pytest_watcher, adapter_probe_*, probe_repro_*, probe_pair, every `*_arm.log` and step log,
  soak_traced_*); `watcher/watcher_excerpt.log`; `.gitignore`.
- Code paths: `tt/generator.py` (`_prefill_user`, `_prefill_traced`, `_capture_prefill_trace`,
  `enable_prefill_trace`, the blocking-replay comment block), `tt/generator_vllm.py`
  (`PREFILL_TRACE_BUCKETS`, `_PREFILL_TRACE_ENV`, `_prefill_trace_enabled`,
  `_capture_prefill_traces`), `tt/model.py` (`prefill_forward`, page-table row helpers),
  `doc/vllm_integration/bench/adapter_probe.py`, `doc/optimized_vllm/bench/soak_traced_bucket.py`,
  `doc/optimized_vllm/bench/run_arm.sh`.
- Commands run (all read-only; no server, device, hardware or vLLM use): `git status/log/diff
  --stat/ls-files/check-ignore`, `ls`, `stat`, `wc`, `grep`, `sed`, and Python scripts that
  recomputed warm medians and deltas for every arm and profile, recomputed long-verbatim-loop
  coverage over every qualitative artifact, diffed probe token sequences against the previous
  stage's committed probe, cross-compared the non-aligned `text_head` values across arms, and
  extracted loguru `module:function:line` fingerprints from every arm and probe log to date the
  revision each ran.

---

## Residual Risk

- The stage's central "no bucket count is licensable" claim rests on a cross-revision pair and a
  mechanism that does not explain the observation. If the 4097 divergence turns out to be
  port-owned capture state rather than the ttnn allocator contract, both the upstream ask and
  the 1.29x rejection change, and `MUSE_GLIMMER_VLLM_PREFILL_TRACE=1` is documented as
  soak-and-enable for deployments.
- After the 15:33 rename, five independent documentation and artifact surfaces point at the
  wrong arm (README §4, the qualitative classification, the `$optimize` checklist, two
  `server_log_size.txt`, two audit `path` fields, and the whole `logs/after_*` set). Anyone
  reconstructing the shipped arm from the logs will conclude tracing was on.
- The shipped configuration's own headline arm was measured on a revision ~2 h older than what
  ships; the deltas are confined to the disabled traced path, and the final code is covered by
  the acceptance suite and `probe_repro_eager.json`, but not by a benchmark re-run.
- The shared `trigram_loop_fraction` metric remains blind to long-period verbatim loops, so a
  future regression of that shape passes `--scope all` on any model. Disclosed as limitation 9.
- Seeded reproducibility at batch > 1 remains a run-to-run draw within a known class, measured
  twice against one server.
- `_guard_late_sampling_capture` still fails open through `except Exception: return None`, and
  the new interlock test pins attribute names rather than signatures.
