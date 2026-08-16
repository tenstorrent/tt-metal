# Stage 06 work log — optimized full model

What actually happened, in order, including the parts that were wrong. Nothing
was committed and nothing was pushed. Every run is on this machine, 1x4 Blackhole
P300_X2, `FABRIC_1D_RING`, `python_env`, tree `8ea42a6b8ed` (stage 05) plus the
uncommitted stage-06 changes to `tt/model.py`, `tt/multichip_decoder.py` and
`tt/functional_decoder.py`. `doc/full_model/` was never written to and
`doc/full_model/probes/perf_full_model.py` was never run.

The stage ran in four passes. The first three are written up in full in
**`profile_48layer_work_log.md`**, which is the lever analysis and stays as
written; this file is the stage's log, and its last section is the documentation
and evidence pass, which is where the fresh profiles live.

| pass | what | token-out at ctx128 |
|---|---|---|
| baseline | stage 05, committed | 22.079 ms — 45.29 t/s/u |
| 1 | 48-layer profile + eight-lever analysis; **distributed argmax** adopted | 21.478 — 46.56 |
| 2 | **paged SDPA program config** adopted | 20.146 — 49.64 |
| 3 | **argmax over live rows** adopted; MoE skew rejected (re-argued in pass 5) | **19.693 — 50.78** |
| 4 | this pass: fresh profiles of the shipped state, prefill closed, documents | — |

---

## Passes 1–3, in one page

The detail is in `profile_48layer_work_log.md`. What matters for reading the
README:

**Pass 1** captured all 48 layers rather than stage 05's 2, ranked the ops, and
produced eight levers. Its two biggest calls were both right in direction and
both wrong in size:

* it called SDPA-decode's missing program config "0.9% at the profiled ctx128".
  Measured, it was **6.6%** — pass 1 under-called its own top lever;
* it called the sampler's 366 us `ArgMax` a bandwidth problem and recommended a
  two-stage reduction "to get within 5x of bandwidth". The op has no bandwidth
  problem: `argmax_multi_core_program_factory.cpp` places one *data-movement*
  kernel and no compute kernel, and the reduction is a scalar `bfloat16_greater`
  loop. The fix that worked was orthogonal — stop reducing 31 rows of padding.

It also closed the LM head lever on arithmetic (75 us of headroom, and 1187 is
prime) and that closure has survived every re-measurement since.

**Pass 2** adopted the SDPA config and hit two things a probe cannot see. The
config pass 1 recommended (`k512/c32`) was measured at the **wrong cache dtype** —
every pass-1 SDPA probe allocated `bfloat8_b` and `create_mesh_kv_cache`
allocates `bfloat16`. Re-swept at the real dtype the winner is `k256/c16`, and
`k512` turned out to be **memory-unsafe in-model**: `k_chunk_size` must not
exceed the cache's per-user allocated depth and exceeding it does not raise. It
only fails when another test has already run in the same process, which is the
tell — on a fresh device the memory past the cache is zeros and the softmax mask
hides it. **Two standalone probes were written specifically to reproduce it and
neither could**, because in a probe the cache is the only thing allocated. The
clamp in `_sdpa_k_chunk` is that operating range, not defensive coding.

Pass 2 also regressed `run_teacher_forcing` from 40.47 to 29.38 t/s/u on first
adoption, because `ttnn.SDPAProgramConfig` was being built inside the layer 48
times a token and each build calls `device.compute_with_storage_grid_size()` — a
device query. Free on the traced path, 96 device round-trips per token on the
untraced ones. Memoising took it to 41.74.

**Pass 3** adopted the live-row slice (2.52x on the sampler standalone, a flat
0.45 ms at every context in-model) and **rejected the MoE reduce-scatter skew
lever**, then on the strength of the stage-06 review it argued the rejection
again from a much larger sample. Pass 3's version — "the achievable saving is
0 ms, because the measured skew idle is already below what perfectly uniform
routing would imply" — did not hold; see section 6 below and the README. The
rejection stands, at a measured 0.024-0.112 ms/iteration rather than at zero.
Pass 3 also found the `argmax` `sub_core_grids` device hang.

---

## Pass 4 — the documentation and evidence pass

### 1. The published profile was of a tree that no longer exists

Pass 1's 48-layer profile was captured before the SDPA config and before the
live-row slice, so by the end of pass 3 the three artifacts under the canonical
names described **none of the shipped code**. Three tells, checkable from the
files themselves: `ArgMaxDeviceOperation` at 366.098 us (the pre-slice figure),
`SdpaDecodeDeviceOperation` at 20.704 us/layer (the pre-config figure), and a
per-layer total of 396.904 us.

That is exactly the hazard this project keeps hitting from the other direction —
a document quoting a number whose artifact has moved — so the fix is naming, not
deletion:

* the fresh capture takes the canonical names
  (`ops_perf_full_model_48layer_decode.csv.gz`,
  `tt_perf_report_full_model_48layer_decode.txt.gz`,
  `rank_full_model_48layer_decode.txt`);
* the pass-1 capture is renamed `*_part1_preadoption.*` and **kept**, because
  `profile_48layer_work_log.md` quotes it throughout and deleting it would leave
  that document's figures unbacked. Its three references were updated to the new
  names and a banner added at its head;
* `probes/moe_skew_analysis.py` still defaults to the canonical name, which now
  resolves to the *shipped* profile — so re-running it re-derives the skew
  conclusion on new data rather than reproducing an old JSON. It does (see below).

### 2. The fresh decode profile, and how the window was verified

`python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_final
probes/profile_full_model_48.py`, then `probes/window_full_model_48.py`, then
`tt-perf-report`, `probes/rank_full_model_48.py` and
`probes/profile_summary.py`. No `--sync-host-device`.

The window is one decode iteration and it is **checked, not eyeballed**: the
boundary is anchored on the three `EmbeddingsDeviceOperation` gathers that open
`decode_hidden`, and then **ten per-device tallies on all four devices — forty in
total — must all be exact** (`logs/window_full_model_48_final.log`). They are: 96
reduce-scatters, 96 all-gathers, 48 `SdpaDecode`, 96 `SparseMatmul`, 96
`PagedUpdateCache`, 3 `Embeddings`, 1 `ArgMax`, and 2 / 2 / 1 of `AllBroadcast` /
`Concat` / `Gather`.

The last three are the ones stage 05's constants get wrong, and they are worth
recording because the failure mode would have been silent: stage 05's windower
expects `2 * layers + 1` all-gathers because the old sampler gathered the whole
vocabulary with one `AllGatherAsync`. That op is gone, and its replacement —
the distributed argmax's two **4-wide** gathers — does not use `AllGatherAsync`
at all. At a gather dim of 4, padded to a 32 tile, `ttnn.all_gather` takes its
*composite* path: `AllBroadcast` + `UntilizeWithUnpadding` + `Permute` + `Concat`
+ `TilizeWithValPadding`. Running stage 05's constants against this capture would
have failed on two counts, which is the windower working.

Two independent cross-checks on top:

* per-device kernel totals **18889.5 / 18888.5 / 18888.3 / 18886.8 us** — 2.7 us
  of spread, 0.014%. One iteration too many or too few is a ~19 ms error;
* the window lands 0.803 ms (4.08%) under the independently measured 19.693 ms
  token-out, which is 0.23 us per dispatch across 3512 dispatches.

A third check fell out and is the nicest one in the stage. The sampler runs in
its **own** captured trace, so `token_out − model_trace` should equal the
sampler's kernel time and nothing else. Measured: `19.692514 − 19.566713 =
125.80 us` against a profiled `sampler_us` of **126.207 us** — **0.32%** apart. Two
completely separate measurements, one a wall-clock median of 128 traced replays
and the other a sum of 39 device-kernel durations, agreeing to half a
microsecond.

### 3. What the fresh profile shows, and what changed rank

| | pass-1 profile | shipped profile |
|---|---|---|
| iteration, device 0 | 19926.5 us | **18889.5** |
| in-model per layer | 396.904 us | **384.791** |
| `SdpaDecode` | 20.704 us/layer | **8.952** |
| terminal-post | 821.7 us | **366.5** |
| the sampler's `ArgMax` | 366.098 us | **10.350** |
| the two composite 4-wide gathers | ~141 us | **41.06** |
| the LM head | 225.384 us | 226.130 |

The per-layer figure fell by **12.113 us** and `SdpaDecode` alone accounts for
**11.752** of it — 97%, with 0.361 us of run-to-run drift across the other 71 ops
in the layer. That is what a program config on one op is supposed to look like.

**Two ranks changed, and both are ours.** `SdpaDecode` went from 6th in the
per-layer ranking to 13th. And the terminal path, which pass 1 described as
dominated by a 366 us `ArgMax` that was "the largest single op in the whole
iteration after the two `SparseMatmul`s", is now dominated by the **LM head** at
226.13 us — 61.7% of terminal-post. The op pass 1 wanted to fix is 2.8% of the
block it used to own.

Everything else held. The expert `SparseMatmul` pair is 41.215 + 39.513 us/layer
against 41.23 + 39.57 before; `TopK` is 26.357 against 26.36; the collectives are
unmoved. Nine of the ten leading per-layer ops changed by under 1%.

**One honest caveat about the profile, in the conservative direction.** It is a
single iteration at `cur_pos ≈ 131`, and the default SDPA path's cost is linear
in `cur_pos` while the configured path is nearly flat. So the profiled saving —
48 × 11.752 us = 0.564 ms — is smaller than the 1.332 ms token-out actually
gained from that lever, because token-out is a median over positions 128 to 256.
The profile understates its own headline change. The sampler lever is
position-independent and the two agree: the profile says terminal-post fell 455.2
us, token-out says 453.5.

### 4. Prefill, profiled — the gap stage 05 disclosed is closed

Stage 05 named "prefill is unprofiled" as a gap. It is now profiled, at the same
128-token prompt the published TTFT is measured at
(`probes/profile_full_model_48_prefill.py`).

The boundary check had to be different, because prefill is eager and has no
three-embedding marker — its rotary reads a slice of the precomputed tables. So
the script runs a warm-up prefill to compile every program and then **two**
measured prefills with no `reset()` between them, and the windower slices at the
last of the one-per-pass `EmbeddingsDeviceOperation` rows and then requires the
preceding pass to be the **identical sequence of op codes, row for row**. So what
the row-for-row check compares is **the last two of the three passes**, not all
three: 4606 ops per device, matched position by position on all four. That is
a stronger check than decode's, not a weaker one. Fourteen
per-device tallies (56 in all) are asserted on top, including one that is
length-dependent and had to be derived rather than guessed: prefill's MoE walks
the sequence in 32-row blocks, so it runs `2 × layers × ceil(S/32) = 384`
`SparseMatmul`s per device, not 96.

Result: **122920.8 us** of device kernel time on device 0 (0.048% spread across
dies) against an independently measured warmed TTFT of **125.431 ms** — 98.0% of
TTFT is device kernel time.

And it settles the open prefill question from the other side. The expert
`SparseMatmul` pair is **61.44%** of a prefill (39.56% gate/up + 21.88% down);
the four collectives together are 3.13%; and **SDPA is 0.58%**, 14.83 us/layer.
The prefill SDPA program config this stage built, measured and declined is
therefore bounded at 0.58% at this length even before the accuracy argument — and
at S=128 the standalone sweep says the config is *slower* than the default (25.72
vs 23.92 us). Its 6.3–6.8x lives at S ≥ 4096, which nothing in the gate set
reaches.

### 5. The lower bound, recomputed on the right basis

Stage 05 published `48 × 0.4286 = 20.573 ms` and concluded the model was
marginally *under* its own lower bound. Being under a lower bound should have
been the tell: 0.4286 ms is a **wall** figure for a traced model containing one
layer, so it carries one iteration's dispatch and host cost, and multiplying it
by 48 charges 48 layers for an overhead paid once.

On the shipped profile:

```
48 x 362.83 us   (stage-04 layer, isolated, device kernel)   = 17.416 ms
48 x 384.791 us  (optimized layer, in-model, device kernel)  = 18.470 ms
  + 0.053 ms terminal-pre + 0.367 ms terminal-post           = 18.890 ms
measured token-out                                            = 19.693 ms
                                                       gap    =  0.803 ms  (4.08%)
```

**4.08%**, against a goal that flags >10–15% as needing action. At 3512
dispatches that is 0.23 us each and there is nothing to act on. The in-model
layer is +6.1% on the isolated stage-04 layer, down from +9.4% before this
stage — the difference being that `SdpaDecode` no longer pays for decoding at a
real position.

### 6. The MoE skew conclusion re-derived on new data

`probes/moe_skew_analysis.py` is pure analysis of an archived profile and opens
no device, so pointing it at the shipped capture is a free replication. It
reproduces: the active-expert recovery still lands on exactly 8 in every one of
the 48 layers (base 29.35 us + 6.85 us/expert), the MoE reduce-scatter still
correlates with the lag at **0.988** with slope 1.046 us/us while the
attention-side one correlates at 0.063, and measured idle is 0.473 ms/iteration
against a uniform-routing floor of 0.506. Chi-square against uniform is **4.06**,
df **5**, p **0.54** on this capture, and the measured mean per-die maximum is
3.438 against a uniform expectation of 3.538.
`probes/moe_skew_analysis_final.json`.

**This section originally concluded from those numbers that the achievable
saving is 0 ms, and that conclusion is withdrawn.** Three things were wrong with
it and they are set out in full in the README under "The MoE-skew rejection is
withdrawn": the "already better than an arbitrarily chosen partition" reading was
z = −0.82 noise; the chi-square dropped the one bin most in tension with
uniformity (k=6, expected 0.74, **observed 3**) instead of pooling it; and the
whole sample is 48 layers of a *single decode token*, so no p-value in it is
evidence for uniformity. The measurement that settles it —
`probes/moe_routing_across_tokens_probe.py`, 128 decode tokens of real routing on
each of **three** prompts — finds per-expert hotness that persists strongly
across tokens (the top 8 experts of a layer take 47.5-57.4% of its selections
against 6.2% under uniform routing), so "0 by exchangeability" is not available.
Refitted per prompt a permutation is worth **0.173-0.194 ms/iteration**; held out
on routing the fit never saw, which is what a fixed layout has to be, it is worth
**0.024-0.112 ms/iteration** over every fit-and-score direction three prompts
allow. The candidate is still declined, but now on a measured basis rather than
on a statistic that did not say what it was read as saying.

The third prompt is the round-2 review's, and it is why that range is not what
this stage first published, "0.024-0.028" (now superseded). Two prompts yield
exactly two of the six single-prompt directions,
and the two this stage published turned out to be the two smallest of the six.

### 7. Two things that went wrong in this pass

**The prefill capture's ops CSV came out as a 0-byte file**, with the profiler
reporting success ("OPs csv generated at: …"). The device log (8.3 GB) and the
host log were both intact, so re-running the post-processing alone
(`python tools/tracy/process_ops_logs.py -o /tmp/prof_fm48_pf`) produced the CSV
correctly and no device time was wasted. Recorded because "the tool said it wrote
the file" is not the same as "the file has contents", and a windower run against
the empty file reports `window: 0 of 0 rows` and then dies in `csv.DictWriter`
rather than saying anything useful.

**The runtime-audit probe crashed on its first run** with `'NoneType' object has
no attribute 'shape'` from `_paged_cache_depth`: a `KVCache`'s `page_table` is
bound by the first prefill, not at allocation, so querying the SDPA program
config before running a pass reads `None`. The probe now queries it after a real
prefill and decode. This is a small, real property of the code worth having
written down: **`_sdpa_k_chunk` cannot be evaluated on a freshly allocated
cache**, which is also why the clamp is a runtime decision rather than a
construction-time one.

### 8. What was checked and did not move

Every readiness gate was re-run on the shipped tree and every one reads exactly
what stage 05 read: `run_prefill_check` 0.980 / 1.000 / 1.000, `run_teacher_forcing`
0.990 / 1.000 / 1.000, 145 passed, 145 passed with zero tripped asserts under
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`, and "No degenerate output
detected". The one thing that did move is the free-running HF/TT token agreement,
4 of 128 to **2 of 128** — the SDPA chunk order is not bit-identical (in-model
PCC 0.9994), so the greedy branch point moved. That number is informational by
construction; the number that gates accuracy is the teacher-forced one and it is
unchanged at 100/100 within top-5.

Policy is preserved: dtype, fidelity, KV-cache layout and dtype, activation
memory configs, CCL policy (`Topology.Ring`, no `num_workers_per_link` pinned)
and the inter-layer residual layout are all untouched. `Topology::Linear` +
`num_workers_per_link=1` was not introduced. Non-aligned prompt lengths still
work — the 145-test suite covers 1, 31, 33, 100, 127, 128, 129, 257 and 1000.

### 9. Discrepancies found and deliberately not fixed

* **`runtime_fallback_audit()` does not name the paged SDPA program config or the
  sampler's live-row count**, both of which are properties of the measured path
  and both of which stage 06 introduced. Adding fields would change a dict
  `test_runtime_fallback_audit_is_clean` pins field by field. They are recorded in
  `probes/runtime_fallback_audit.json` instead, read off the modules that own
  them.

Two figures in the task brief also did not match their artifacts and the
artifacts were used: the stage-05 baseline TTFT is **126.695 ms**
(`../full_model/probes/perf_full_model.json` `ttft_ms`), not 126.979; and the
stage-05 token-out is **22.079145 ms**, which rounds to 22.0791 rather than
22.0794.

### 10. The checker

`probes/check_published_figures.py` is the stage-06 continuation of the pattern
stage 04 left behind, under the rule the stage-05 review sharpened: **a checker
that reads the document it is checking checks nothing.** Every figure is parsed
out of an artifact — a CSV row, a JSON field, a probe log line — and the
README/work-log string is compared against the *computed* rounding of it; every
ratio is recomputed from its two artifact operands and the quoted string must be
the rounding of the result.

Every assertion in it was **mutation-tested, mechanically**.
`probes/mutation_test_checker.py` hard-links the model directory into a scratch
tree, applies one mutation at a time — corrupt a document's digits, corrupt its
words, scale an artifact's numbers, rename an op label, flip a contract boolean,
change the measured side of a boundary tally, rename a promised file, reverse a
ranking, add a `failed` line to a pytest log — and re-runs the checker against
each, then reports any assertion that never failed.
`logs/mutation_test_checker.log`: **473 assertions, 473 made to fail by at least
one mutation, none left over.** (That tally is this pass's and is **superseded** —
the stage-06 review found the crediting that produced it was broken, and the
archived log now carries the run described in the README. The figures in this
section are kept as the record of what this pass believed.)

That is the check the stage-05 review had to do by hand, after finding an
assertion whose second clause was already proven by an assertion above it — a
check that cannot fail is worse than no check, because it reads like cover.
Doing it mechanically found three classes of weakness that a reading would
probably have missed:

* **short numbers matched as substrings.** `"2.7"` was satisfied by `"12.75"`
  and `"13"` by `"13-wide grid"`. Every numeric comparison now goes through a
  boundary-anchored match, and the two counts the README spells out in words are
  checked as words against the computed number.
* **checks that crashed rather than failed.** Six assertions raised
  `KeyError`/`FileNotFoundError` when their artifact was corrupted, which detects
  drift but proves nothing about the assertion — and, worse, aborted the run so
  every assertion after them went unevaluated. Each is now guarded.
* **one check that could not be evaluated at all**: an artifact-existence
  assertion pointing at shared code outside this model directory, which does not
  exist in the scratch copy. Replaced with a check on what the README claims
  about that file — its path and line range — plus the reproducer that is in
  this tree.

It does not prove the assertions are the *right* ones. It proves none is a
tautology over the checker's own literals, and none is a search a corrupted
document could still satisfy.

---

## Pass 5 — three loose ends closed before review

Pass 4 shipped with three things open, all named in it. All three are now closed.
No model behaviour changed: one is prose, one is file naming, one is a
measurement.

### 11. The two stale docstrings in `tt/model.py`

Pass 4 recorded them as "discrepancies found and not fixed" on the grounds that
the stage was not to change model code. That was the wrong call for prose — a
docstring is documentation, and leaving a wrong number in the file a reader opens
first is not conservatism. Both are corrected.

`sample_greedy_argmax` now quotes the shipped sampler figures (0.928 ms against
6.155, **6.6x**) and the shipped token-out (19.693 ms, 50.78 t/s/u), and — the
part that is more than a find-and-replace — it now attributes **each** of the two
sampler levers to its own like-for-like delta instead of quoting one of them as
if it were the whole: the distributed reduction is 22.079 → 21.461 ms at
`context` 4096, and the live-row slice is 20.146 → 19.693 ms at `context` 8192.
The checker asserts the two context fields are equal within each pair, so the
"like-for-like" claim is checked and not merely asserted.

`_WatcherCleanSampling1D._sample_argmax` was the worse of the two. It priced the
baseline path at `AllGatherAsync 889 us + ArgMax 859 us` and called that
"essentially all of the 1.87 ms of non-layer work inside a 22.079 ms token-out
decode step". The two sides are not the same accounting: the per-op figures come
from stage 05's **2-layer** window, which charges the terminal path against 2
layers rather than 48 and so over-weights it by construction, and each is
device-kernel time summed over that op's own cores (2 for the gather, 110 for the
argmax) rather than wall clock. The sum resembling 1.87 ms was a coincidence. The
claim is **withdrawn in the docstring text** — not silently deleted — the 889/859
rows are kept but labelled as 27.5% and 26.5% of *that* window, and the terminal
block is priced from the 48-layer profile instead: 366.5 us of an 18889.5 us
iteration, 1.94%, sampler 126.2 us.

The baseline path was replaced before any 48-layer profile was taken, so it has
no 48-layer op row of its own. The docstring says so rather than inventing one.

### 12. `perf_full_model.{csv,json}` held the wrong run

The unsuffixed name is what `probes/perf_full_model.py` writes with no `--tag`,
which makes it the file a re-run overwrites and a reader reaches first. It held
the **part-1** measurement (token-out 21.4609 ms), left there only because a
docstring cited the path. Pass 4 disclosed that in a limitation, which is weaker
than fixing it — and this stage had already been bitten by exactly this failure
mode from the profile artifacts.

Resolved the same way the profile artifacts were: the canonical name holds the
shipped run (byte-identical to `perf_full_model_p128_argmaxrows.{csv,json}`) and
the part-1 run is `perf_full_model_part1_preadoption.{csv,json}`. Citations
updated in `tt/model.py`, this file, `README.md`,
`profile_48layer_work_log.md` and both checkers; a grep of the whole model
directory finds no reference to the old arrangement.

`check_published_figures.py` now asserts the rule in both directions: the
unsuffixed file must match the shipped run on **every** published row, not just
token-out, and the `_part1_preadoption` file must still differ from it by more
than a millisecond. Either half breaking is a failure.

### 13. The six-prompt qualitative suite, re-run

Pass 4 inherited stage 05's `../full_model/qualitative_check.log` and named the
inheritance as limitation 6. That was untenable: the sampler's reduction changed
twice this stage, so the inherited evidence described a path that no longer
exists.

`doc/full_model/probes/qualitative_probe.py` was **copied** to
`probes/qualitative_probe.py` and the copy was run, so nothing was written into
the committed stage-05 tree. The copy differs from the original in exactly two
ways: its docstring names the stage-06 log, and it archives its JSON beside
itself as `probes/vllm_qualitative_outputs_argmaxrows.json` in addition to
writing the `readiness_qualitative/` schema location the scorer discovers.

Twelve completions, six prompts × {greedy, sampled}, on the real 48-layer model.
`check_degenerate_output.py --scope vllm`: **"No degenerate output detected"**.
Read as well as scored — the verdict, the quotes and the two caveats are in the
README's qualitative section. Limitation 6 is replaced, not deleted: what
survives is that on **four of six** prompts the sampled leg is byte-identical to
the greedy one, because `top_k=20, top_p=0.9, temperature=0.7` is not hot enough
to move a confident model off its argmax. Stage 05's log collapses on five of
six, so that is the shared suite's property and not this stage's.

Four of the six greedy answers differ from stage 05's. Every difference read is
an ordinary lexical branch with both readings correct ("a teacher who **gives**
you the right answers" → "who **tells** you", "remains constant in ideal
**reversible processes**" → "in ideal **cases**") — the same `bfloat4_b`
expert-routing branch-point story as the free-running run, at sentence scale.

### 14. What was re-run to close the pass

* `probes/check_published_figures.py` — **473 figures**, all matching, up from
  466 before this pass. The added assertions are the ones that make the three
  closures checkable rather than merely stated: the docstrings' figures against
  the perf and profile artifacts, the naming rule in both directions, the
  qualitative artifacts and their score, and — new, because it had drifted
  silently once — the README's own counts against what the tester actually
  defines and what the archived mutation run measured.
* `probes/mutation_test_checker.py` — **193 mutations, 473 assertions, 473 made
  to fail by at least one mutation, none left over.** Four of the new assertions
  survived the first mutation run and each got a mutation written for it: a
  README that reinstates the two closed disclosures, a docstring that drops the
  "2-layer" qualifier, a qualitative artifact with a prompt removed, and a
  mutation tester that will not import. One more, the assertion on the archived
  mutation log, needed a mutation that corrupts that log.
* `pytest tests/ -m "not models_performance_bare_metal" -q` — **145 passed, 16
  deselected**, 410 s (`logs/pytest_stage06_loose_ends.log`). `tt/model.py` was
  touched, so the suite was re-run even though the change is prose; a docstring
  edit can break a test that greps prose, and this one does not.

The perf tests under `tests/` were **not** run: they rewrite committed CSVs.

---

## Pass 6 — answering the stage-06 review

The review returned `more-work-needed` with four blocking findings and several
smaller ones. **No model-correctness bug was found**: it independently confirmed
the SDPA clamp, the distributed argmax on hardware, both profile windows and
every headline number. What failed was the evidence layer, and that is the whole
of this pass. One `tt/` change (an assert), one new test, one new device probe;
everything else is checkers and documents.

### 16. The mutation tester was crediting coverage it had not earned

`ever_failed |= failed & set(clean)` keyed mutation credit by the check's
**formatted name**, and many of those names embed the artifact value under test
— `README quotes the degeneracy metric 109`. Mutating the artifact to 77 renamed
the check, the mutated run's `FAIL` line matched nothing from the clean run, and
the intersection discarded it. For every check of that shape the tester proved
only that the *document* side could fail, never the artifact side — and still
printed `473 assertions; 473 were made to fail`.

`check()` now emits a **stable id** (the source line of the call plus an ordinal
for calls inside a loop) that cannot vary with any artifact value, and the tester
keys on it. Two further gaps became visible once it did, and both are now
reported and both make the tester exit non-zero:

* **36 assertions were reachable only by a document-wide shotgun** — one of the
  four mutations that corrupt every digit or every word of a whole file. That is
  coverage in name only. Each got a targeted mutation. (This log said those four
  "trip 200+ assertions at once"; the tester measures them instead of asserting
  a constant, and the measured set moves as assertions are added — it read
  29/34/206/230 when the round-2 review first measured it, then 39/44/236/260,
  and it has moved twice since. This parenthetical used to restate the shipped
  quadruple as a third site and nothing checked it, so it went stale exactly as
  such a site must; the numbers are gone from here. The two sites that publish
  them are in `README.md`, each anchored to its own sentence and each with a
  mutation that corrupts only that site, and both are read from the archived
  run's `measured breadth of the 4 declared shotguns:` line.)
* **14 mutations broke nothing at all.** Nine were dropped as redundant and five
  were repaired, mostly because they asked the README for a number it happened
  to contain anyway (a bare `7` occurs everywhere in 55 KB).

Five assertions were then demonstrably unfailable and are rewritten: an
`all(...)` over a one-element default sliced to nothing; an index list checked
digit by digit against the whole document; a collapse count that passed at four
or at six; the `Topology.Ring` check, which was an unanchored case-folded search
for `"ring"` with 19 hits; and a regex parsed, discarded, and compared against a
hardcoded `4`. The direct assertion the review asked for — the archived run
covered *exactly* the assertions this run makes — is in, replacing a bare-integer
search for the count.

### 17. The MoE-skew rejection did not survive its own statistics

Retiring a ~2.5% lever permanently needs a sound argument and this one had
three unsound ones. They are set out in the README under **"The MoE-skew
rejection is withdrawn"**; the short version is that "0.69 us/layer better than
an arbitrarily chosen partition" was z = −0.82 noise, the chi-square dropped the
one bin most in tension with uniformity, and the whole sample was 48 layers of a
single decode token.

So the question was measured instead. `probes/moe_routing_across_tokens_probe.py`
wraps `ttnn.topk` and records the router's top-8 at every layer of 128
free-running decode tokens on the real 48-layer model, on three prompts.
Hotness **does** persist across tokens — the top 8 experts of a layer take
47.5-57.4% of its selections against 6.2% under uniform routing — so the
exchangeability argument that would give exactly zero is not available, and a
permutation fitted on half the tokens is worth 0.173-0.194 ms/iteration on the
other half.

Fitted on routing it will not see again, which is what a fixed layout has to be,
it is worth **0.024-0.112 ms/iteration**, 0.12-0.57% of token-out — the six
single-prompt directions span 0.024-0.111 and the three pooled-on-two,
held-out-on-the-third fits span 0.058-0.112. The top of that is a floor, because
pooling the fit transferred strictly better than any single-prompt fit for every
held-out prompt. The rejection stands; "0 ms" does not.

This stage first published **0.024-0.028** here (now superseded), from the only two
directions an n=2 sample can produce, and the round-2 review's third prompt
showed those two were the *smallest* of six — the mean over six is 2.2x the
published figure and the largest is 4.6x it. The lesson is not only the number:
"fit on one prompt, score on another" was justified in this log as the only gain
that counts, and that framing is withdrawn. It is the worst case for a fixed
layout, not the definition of one.

### 18. Three arithmetic corrections

* **"14 rows" is 16**, twice, for the composite gathers. The artifact said 16 and
  the README's own parenthetical enumerates 2+8+2+2+2. There is now an assertion
  on `composite_gather_rows` — only `composite_gather_us` was checked, which is
  why it passed 473 assertions.
* **The LM-head headroom mixed two devices.** `tt-perf-report` merges the mesh
  and prints the slowest device's row; 226.130 us is device 0 and 66.4% is device
  3. The self-consistent figure is **79.44 us on device 3**, and the checker now
  parses the device id out of the report row and pairs it with that device's own
  kernel time from a new `lm_head_us_all_devices` field.
* **The attention `ReduceScatter` max in the lever analysis's B2 table** was
  transcribed as 15.01; the CSV gives **15.276**. The checker now re-derives all
  three of that row's figures from the pre-adoption window.

### 19. The smaller ones

* `test_distributed_argmax_is_exact_at_batch_above_one` — the review found no
  device-sampling test above batch 1, which is a coverage gap stage 06 opened by
  making the sampler's reduction batch-dependent. The new test drives it at
  `max_batch_size=4` on random and on all-negative logits.
* `_sdpa_k_chunk`'s `max(32, depth)` floor could violate the invariant stated in
  its own docstring if `block_size < 32`. The model never configures that; there
  is now an assert saying so.
* The `ccl_watcher_ab.py` path in `profile_48layer_work_log.md` did not resolve.
* **"Three identical prefills" is a warm-up and two measured ones**, and the
  row-for-row check compares the last two. Corrected in both documents.
* **The LM head's "structurally impossible" is overstated** and is now split: the
  unpadded uniform shard is structural (1187 tiles, prime); the padded route is a
  design decision, its probe raised an undiagnosed `TT_THROW` rather than a
  tile-alignment error, and padding the vocabulary would also disable the
  distributed-argmax fast path — which is a good reason to decline and is now
  the stated one.
* `window_full_model_48.py --manifest` writes a cut-point manifest for the raw
  capture. The shipped windows predate it and cannot be given one retroactively;
  that is disclosed rather than papered over.

## Pass 7 — a rot in one mutation, and the class of vacuous assertions behind it

### 20. A reworded anchor turned a mutation into a no-op

The README's shotgun-coverage section had its anchor sentence reworded — it used
to attribute the measured breadths to a particular review round, which bakes in
a claim that goes stale every time the set moves. `check_published_figures.py`
was updated with it and stayed green. `mutation_test_checker.py` was **not
re-run**, and its `readme_shotgun_section_breadths` mutation still matched on
the old wording: it matched nothing, reported `*** BROKE NOTHING ***`, and the
second of the two anchored sites was left with no mutation of its own. The
rewording also moved the two digit shotguns' breadths by one, so the archived
log was stale as well.

Both are repaired. The mutation is a regex over the current wording, matched
with `\s+` between words so re-wrapping cannot break it, and the two sites are
verified one-to-one: corrupting limitation 10 breaks exactly its own assertion
and corrupting the shotgun-coverage section exactly its own, neither the other's.
The general lesson is in the comment above the mutation: an anchor that names
something a writer can edit out is a mutation waiting to rot, and a green
checker run is not evidence about the tester.

### 21. Seven assertions were satisfied by unrelated text

Instrumenting `appears()` showed 258 calls, 236 of them whole-document searches,
81 of those searching a string the document carries more than once. Seven were
demonstrably vacuous — each proven by corrupting the intended site and watching
the checker stay green, then corrupting the collider and watching it fire:
`16` was supplied by `bfloat16` (twice over), `256` by `SHA-256`, `5` by a
markdown list marker, `384` by an unrelated published crossover, and — the two
that mattered most — the decode tally of `40` by this log's sibling sentence
about being "40 assertions short", and the degeneracy metric `109` by the
README's own prose quoting the check's name. The last two collide with
self-referential text, which is the same feedback path that made the tester
oscillate between two fixpoints one apart.

All of them, and the four more short needles the same rule catches, are now
anchored: the phrase that is supposed to carry the figure is built from the
parsed artifact value and matched against the whitespace-flattened document.

### 22. The lint, which is the actual fix

Three rounds of reactive fixes each left more behind, so the fourth is
mechanical. `appears()` lints its own needle and the checker exits non-zero on a
violation: a numeric needle under four characters may not be searched over a
whole document at all, and a needle the document carries more than once must be
anchored or declared `restated=RESTATED`. It flagged **88 searches at 70 call
sites** on the run that introduced it — 20 anchored, 68 declared. The 68 are
distinctive decimals a table and a paragraph both quote; declaring them is an
explicit decision at the call site, not a claim that they are strong.

### 23. The archived log's provenance was not checked

`--bootstrap` exists so a first log can be produced after the assertion count
moves, and it prints a banner saying its breadths are inflated and that the log
is not the archive. Nothing read the banner: pasting it, and a clean-tree line
reporting failures, onto `logs/mutation_test_checker.log` passed everything. The
checker now requires the archive to open with `clean tree: <this run's count>
checks, 0 failing` and to carry no bootstrap marker, and the laundering itself
is a mutation so the assertion has to be able to fail.

That check is self-referential like the two count checks, and it is stricter
than they are: a bootstrap log can never satisfy it, so archiving one leaves the
clean tree failing and only another bootstrap run can proceed. The settling
procedure therefore gained one step, documented on `BOOTSTRAP_SELF_REFERENTIAL`:
strip the banner from the bootstrap log and set its clean-tree line to `0
failing` to make a **scaffold**, derive the documents' figures from that, run
normally, and throw the scaffold away. What ships is a log a normal run
reproduces byte for byte, which is what those three assertions then confirm.

### 24. The stale hand-maintained figures that were left

`SHOTGUN`'s docstring still said the four "trip 236, 39, 260 and 44 assertions",
two revisions out of date, and nothing asserted it — the exact defect this tool
exists to remove. The numbers are gone from the docstring and from the module
docstring above it; the run prints them and the README is tied to what it
printed. The same treatment was given to this log's own third restatement of the
quadruple in §16, which had gone stale for the same reason: a figure restated at
a site nothing checks will rot, and the fix is to stop restating it.

### 25. Six artifacts were over the repo's 500 KB limit, and the fix was width

The pre-commit `check-large-files` hook rejected the commit. Six files were over
it: the three `tt-perf-report` transcripts (804, 1332 and 800 KB) and the three
windowed ops CSVs (2728, 2732 and 2308 KB gzipped).

The transcripts are plain text and gzip to 32, 56 and 32 KB, so they are
archived as `.txt.gz` and every citation moved with them — `README.md`, this
log, `profile_48layer_work_log.md`, `check_published_figures.py` (whose
`log_text` already read gzip) and `mutation_test_checker.py` (whose
`mutate_text` did not, and now does). `gzip` in place refuses on these because
the mutation tester's scratch trees hard-link them, so it is `gzip -c > x.gz &&
rm x` — the same unlink-first rule `_rewrite` has always had, for the same
reason.

The CSVs could not be fixed that way. They are already the *windowed* captures,
cut out of a ~139 MB raw file by `window_full_model_48{,_prefill}.py`, and every
row in them is load-bearing: the boundary check that makes them evidence is ten
and fourteen exact per-device op tallies. Dropping a row breaks the check that
says the window is one iteration. What they have instead is width — Tracy writes
128 columns and this tree reads eleven.

So `probes/reduce_profile_csv.py` cuts them to 35 columns, 250/251/364 KB. The
kept set was **not guessed**. `--audit` greps every consumer for column
subscripts, expands the one f-string template `rank_full_model_48.shape` builds,
intersects the result with the capture's own header, and fails if anything
subscripted is not kept; it reports eleven, and `KEPT` is those plus a
deliberate margin (op identity, the two other durations, the INPUT_0/1 layout
triples, OUTPUT_0, three utilisations) on the grounds that a wrongly-dropped
column costs a re-capture on hardware and a wrongly-kept one costs a few KB.

**The check that decided it was re-derivation, not the audit.** Every consumer
was run against the full-width CSV and against the reduced one, and the outputs
compared byte for byte: `profile_summary.py` in both modes, `rank_full_model_48.py`
on the shipped and the pre-adoption windows, `moe_skew_analysis.py` on both, and
`window_full_model_48.py` re-checking the decode window's tallies. All
identical, including the ranking labels that depend on the `INPUT_*_PAD` shape
columns and the LM head that `profile_summary.py` finds by its 37984
vocabulary shard. The full-width run was first confirmed to reproduce the
archived artifacts themselves, so "identical to the full-width run" and
"identical to what is archived" are the same statement.

`ATTRIBUTES` is the one column dropped that a reader might miss: 280 KB
compressed on the decode window alone, more than the entire budget, and read by
nothing here. The cost is that `tt-perf-report` cannot be re-run from the
reduced CSV — it reads that column among about forty. Its output is archived
whole instead, and the README says so where the artifacts are listed rather than
leaving it to be discovered.
