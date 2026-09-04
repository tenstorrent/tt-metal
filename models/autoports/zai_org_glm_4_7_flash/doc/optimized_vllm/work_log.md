# GLM-4.7-Flash optimized vLLM serving: work log

Stage 10, `$vllm-integration` + `$optimize` + `$tt-device-usage` +
`$tt-enable-tracing`. One Blackhole p150, device 0, 1x1 mesh. Starts from the
vLLM-integration stage (`ab0fbebb4b1`) and the selected datatype-sweep config,
and optimizes serving in place.

Entries are in the order they happened, including the two turns that were wrong.

---

## OV-001 -- The inherited "2x vLLM overhead" claim is false; measure it first

`doc/vllm_integration/README.md` opened this stage with: the full model's traced
token-out decode is 22.994 ms/token, through vLLM it is 45.0 ms/token, therefore
"~22 ms/token, roughly 2x, of vLLM-path overhead ... the single largest
optimisation target for stage 08". Optimizing against that framing would have
sent this stage hunting for adapter and engine overhead that does not exist.

**Experiment.** `probe_scripts/adapter_decode_floor.py` drives the real
`GLM47FlashForCausalLM` exactly the way `vllm_tt_plugin/async_decode.py` does
(`decode_forward(read_from_device=False)` -> `read_decode_output(async_read=True)`
-> `ttnn.event_synchronize` -> `process_decode_output_host`), on the real 47-layer
model, with vLLM's own config: `max_batch_size=32`, `max_seq_len=202752`,
`num_blocks=7362` (the pool `server.log` recorded vLLM choosing),
`prefill_chunk_size=1024`. No vLLM engine anywhere.

**Result (first run, all rows the same token/position):** 1 live row 45.34
ms/token, 8 rows 45.39, 32 rows 40.96. Serving measured 45.0. So the vLLM engine
contributes nothing measurable, and the entire 22 ms is the model doing
different work.

**Why different work:** `build_generator`'s `max_batch_size` defaults to 1, so
the full-model perf harness builds a *one-physical-row* decoder, which takes
`OptimizedDecoder._moe_decode_indexed`. The adapter builds 32 physical rows
(vLLM's `max_num_seqs`, and the decoder pins per-slot shard grids at
construction), which takes `_moe_decode_union`. The stage-09 comparison was
batch-1-built model vs 32-row-built model, not serving vs model.

The inherited claim is corrected in this stage's README rather than repeated.

## OV-002 -- Union vs compact: where the 32-row cost actually is

`_moe_decode_union` asks `ttnn.sparse_matmul` to scan all `E = 64` expert groups
and zero-fill the skipped ones, so its output is `[1, 64, B, N]` and the two
slices, the fused SiLU multiply, the routing-weight multiply and the final
reduction all run over 64 groups regardless of how few experts were selected.
`_moe_decode_indexed` (batch-1 only) uses `sparse_matmul`'s INDEXED/GATHER mode,
whose output group axis is compact.

**Experiment.** `probe_scripts/moe_union_vs_compact.py`: one real MoE layer,
32 physical rows, deployment dtypes, traced replay, sweeping the number of
active experts for the union form and the compact width for the indexed form.
Post-matmul chain only (`moe_union_vs_compact.json`):

| active / kc | union | compact |
|---|---|---|
| 4 | 0.4029 ms | 0.1258 ms |
| 8 | 0.5232 | 0.2128 |
| 16 | 0.7636 | 0.3849 |
| 32 | 1.2444 | 0.7374 |
| 64 | 2.2264 | 1.4376 |

The compact form is 1.5-3.2x cheaper **at equal coverage**, including at 64
where it does identical math -- the difference is the full-group scan and the
zero-fill, not the expert matmuls. Over 46 MoE layers that is 12.7 ms/token at
4 experts and 36 ms/token at 64.

## OV-003 -- Prototype, and a trap worth recording

`probe_scripts/moe_compact_layer.py` built the real thing on one MoE layer:
routing -> inactive-row mask -> union -> top-`kc` union ids -> indexed
`sparse_matmul` -> compact chain, checked against the shipped union path.

* PCC vs union at 32 live rows: kc=64 **1.0** (exact), kc=32 0.83, kc=16 0.60.
  That is the correctness bound made visible: a `kc` below the real union
  silently drops the lowest-scoring selected experts. Not a tuning knob.
* PCC vs union at 1 live row with the mask: kc=4 **1.0**, kc=8 **1.0**, masked
  rows exactly zero.
* Whole routed block, 1 live row: union 0.507 ms vs compact kc=4 0.2945 ms.
* Whole routed block, 32 live rows: union 1.944 ms vs compact kc=64 1.631 ms.

**The trap.** The first end-to-end equivalence probe built *two* generators (one
compact, one union) on one device in one process and compared them. Every
comparison came back PCC ~0 with max absolute differences around 1e19-1e20. That
is not a compact-path bug: the second generator's persistent buffers were
allocated while the first generator's traces were live, which is exactly the
hazard `recapture_decode_traces`' docstring describes, and each one's replay then
clobbered the other's. Re-run one arm per process, the two agree to a row
checksum delta of exactly 0.0 at 1, 2, 4, 8, 16 and 32 live rows, with
identical argmax (`compact_decode_equivalence.json`). Recorded because the next
person will otherwise waste the same hour.

## OV-004 -- kc buckets: two wrong bucket sets before the right one

`kc` is part of the captured program's shapes, so a trace is only valid for the
`kc` it was captured with. The generator therefore captures one decode trace per
bucket and selects per step from its own live-row count
(`GLM47FlashGenerator.decode_kc_for_rows`, precomputed into a lookup table so
the per-token path is a list index).

Making one captured sampling trace serve several decode traces needs every
decode trace to write the *same* logits tensor, since the sampler binds to the
tensor it was captured against. `GLM47FlashModel.allocate_decode_logits`
pre-allocates that buffer before any capture and `lm_head_decode(out=...)` passes
it as `ttnn.linear`'s `optional_output_tensor`, so the sharing is free rather
than a copy. `ttnn` returns a fresh Python wrapper around the same device buffer,
so the post-capture assertion compares `buffer_address()`, not identity -- the
first version compared identity and refused a correct capture.

**Wrong bucket set 1: `(4, 16, 64)`.** Whole-model token-out decode at 32
physical rows, all rows carrying the same token and position: 1 row 45.34 ->
30.92, 8 rows 45.39 -> 92.19, 32 rows 40.96 -> 92.58. The regression is real but
the *baseline* was also wrong: identical rows all route to the same 4 experts, so
the union path looked far cheaper at batch than any real serving batch would.
Re-ran the probe with a different vocabulary id and decode position per row.

**Wrong bucket set 1, honest baseline:** before 45.28 / 53.37 / 60.41 / 78.21 ms
at 1 / 4 / 8 / 32 live rows; after 30.92 / 43.02 / 92.19 / 94.07. The two
regressions are both rows whose bound forces `kc = 64`. Fitting both curves,
compact beats union while `kc < 11.1 + 1.388 * union_size`; with `kc = 4 * rows`
that holds comfortably to 4 rows, marginally to 8, and fails at 16 and above.

**Shipped bucket set: `(4, 16, 32, union)`.** `kc = n_experts` is never captured;
row counts whose bound needs it replay the union trace, which is data-adaptive
and cannot be beaten by a fixed-width compact form when the batch's real union is
small.

The rejection is kept as a committed, re-runnable arm rather than as prose:
`probe_scripts/adapter_decode_floor.py kc64` replaces `_kc_buckets` with one
that does capture the full-width bucket, and
`adapter_decode_floor_kc64.json` records **91.77 ms/token at 32 live rows
against the shipped union fallback's 78.43** -- 13.3 ms/token slower, and worse
still at 12 and 16 live rows (+24.4 and +18.7 ms/token). (An
earlier draft of this log quoted 92.19 / 94.07 from an intermediate run whose
JSON had already been overwritten; that was a stage-review finding and is fixed
by the dedicated arm.)

Measured after, at that point: 29.52 / 41.61 / 57.55 / 78.21 ms at 1 / 4 / 8 /
32 live rows.

> **Superseded by OV-012.** Those are the four row counts where each bucket's
> bound is *saturated*, i.e. each bucket's best case. Sweeping every row count
> later showed this rule was a +2.1 ms/token regression at 5 live rows, and the
> 78.21 figure came from a run whose JSON was overwritten. The shipped rule and
> the committed numbers are in OV-012.

The inactive-row routing mask (`GLM47FlashModel.decode_active_mask`, derived on
device from the single current-position tensor so it cannot drift out of step
with it) is what makes the bound sound: without it the 31 idle rows' stale hidden
state contributes experts to the union and a `kc` derived from the live-row count
would be too small.

## OV-005 -- Page-table refresh off the per-token path

`_write_page_table_rows` unconditionally wrote every row into the adapter's
mirror, cloned the full `[32, 3168]` int32 mirror (405 KB), and handed it to
`refresh_page_table(only_if_changed=True)` which then diffed it -- every decode
step, even though vLLM re-sends the same block list every token and only extends
it when a request crosses a 64-token block boundary. Now the adapter diffs per
row against its own mirror first and returns without touching the mirror, the
clone, or the generator when nothing moved, with `page_table_calls_skipped` /
`page_table_calls_written` counters so this is a counted fact.

Live server: 98-100 of every 100 decode calls skipped across all 33 counter
windows, and 0 `page_table_refreshes` in the four windows that sit entirely
inside one request's steady state.

## OV-006 -- The per-admission trace recapture (the burst's real problem)

First after-arm serving run: single-user decode 45.22 -> 30.84 ms/token as
predicted (the shipped figure is 29.497 after OV-009's routing-prologue work), but burst TTFT went **14378.8 -> 31907.8 ms** and burst output
throughput 137.30 -> 78.36 tok/s. TPOT was unchanged, so it was admission, not
decode.

`server.log` gave the cadence directly: one "Resetting sampling trace" per
admitted request, ~0.52 s apart in the before arm and ~1.32 s apart in the after
arm -- a full decode-trace recapture per prefill, made ~2.2x more expensive by
this stage capturing four decode traces instead of one.

**Attribution.** `probe_scripts/prefill_recapture_probe.py` counts
`num_program_cache_entries()` around each call in the served prefill path:
`apply_prefill_sampling_state` compiles nothing; `prefill_and_sample` compiles
exactly **one** program the first time each `user_id` is used and nothing on
repeat; and that program is slot-keyed, not length-keyed (slot 7 compiles one
program at prompt length 100, then none at 200 or 400). Forbidding cache misses
made the first use of a fresh slot throw, confirming the miss was real.
`prefill_recapture_probe_before.json`.

That single compile happens while the decode traces are live, so
`_maybe_recapture_after_compile` correctly releases and re-captures all of them.
Correct behaviour, catastrophic cost: 22 recaptures inside a 32-request burst.

**Fix.** `GLM47FlashGenerator.warmup_prefill_slots` compiles one prefill per slot
at the shortest warmed bucket during warm-up, before any trace exists. After:
`prefill_recapture_probe_after.json` shows 0 programs compiled and 0 recaptures
for fresh slots, and `set_program_cache_misses_allowed(False)` no longer trips.

Serving effect, isolated by its own env knob (`GLM47_VLLM_PREFILL_SLOT_WARM`)
so it is not credited to the MoE change: recaptures over a whole run **31 -> 2**,
burst TTFT 14320.1 -> 9126.2 ms, burst output throughput 137.65 -> 177.25 tok/s,
burst e2e 23245.6 -> 18051.4 ms, and primary TPOT unchanged at 45.21 ms. This is
a pre-existing inefficiency, not one this stage introduced -- the before arm paid
it too, just less per event (~0.24 s with one decode trace against ~1.0 s with
four, which is the arithmetic behind Limitation 3 in the README).

## OV-007 -- The full sampling profile moved 5 failed -> 8 failed; it is not this change

> The counts in this entry are the round-1 arms. Across five measurements of the
> same suite the failing count has been 11 / 5 / 8 / 7 / 8; the argument and the
> controls are unchanged, and OV-013 records the later data points.

Both arms ran `--sampling-profile full` for the record (the gated profile is
smoke, inherited from stage 09 with owner acceptance).

* before arm (both knobs off): 5 failed, 68 passed, 1 skipped.
* after arm: 8 failed, 65 passed, 1 skipped.
* stage 09, same suite, same server config: 11 failed, 62 passed, 1 skipped.

The failing sets are exactly nested -- the after arm's eight are the before
arm's five (`test_mixed_params_batch`, `test_topk[19]`, and the three
`*_penalty_mixed_batch` tests) plus `test_topk[15]`, `test_topk[32]` and
`test_top1_is_greedy` -- and all of them are `assert_deterministic` failures at
full or near-full occupancy, drawn from the same pool the vLLM-integration
stage's eleven came from. (An earlier revision of this entry said the penalty
mixed-batch tests flip between arms; they do not, they fail in both.) Two pieces of evidence rule this stage's change out rather than
assuming it:

1. **The decode logits are bitwise identical whichever bucket runs.**
   `probe_scripts/bucket_numerics.py` forces each captured bucket in turn over
   identical persistent inputs on the real 47-layer model:
   `bitwise_identical: true`, `max_abs_diff: 0.0`, identical argmax, for
   compact-vs-union and compact-vs-wider-compact at 1 and 4 live rows, and for
   repeated replays of one bucket. A change that cannot move a single bit cannot
   be responsible for a sampling result changing.
   (The worry was real and worth testing: the compact path reduces the expert
   outputs in union-score order while the union path reduces them in expert-id
   order, and float addition is not associative.)
2. **All of them pass alone against a freshly started server.** The after arm's
   eight failures, run against a fresh after-arm server: **8 passed**
   (`logs/sampling_isolated_after.log`). That is the discriminator stage 09
   established for this defect.
3. **The convenient shortcut is checked and refused.** Most of the failing tests
   run at 15-32 concurrent, where the bound forces the union trace, i.e.
   unchanged code -- but `test_top1_is_greedy` runs a batch of **4**
   (`vllm-tt-plugin/tests/tt/test_seeding_and_variety.py:373`, one greedy config
   plus three `top_k=1` configs), which takes the compact `kc=16` bucket. So
   "all the failures are on the unchanged path" would be false and this log does
   not claim it. What rules the change out is the bitwise identity at 4 live rows
   plus the isolated pass.

That matches the characterization stage 09 filed as
[tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408):
a server-state-accumulation defect at full occupancy whose failing set varies
run to run. It remains open and remains this model's most significant serving
limitation; this stage did not narrow it further and does not claim to.

Output-level corroboration: all six qualitative greedy completions are
byte-identical between the arms (`qualitative_before_after.json`), prompts sent
in the model's chat template (`prompt_mode: chat`), degenerate-output check
clean on both arms.

## OV-008 -- Closing checks

* Reduced adapter suite: 28 passed, including 13 new tests for the kc bound,
  bucket selection, the union fallback, one-trace-per-bucket, the shared logits
  buffer, token feedback across a bucket switch, page-table skipping, and the
  per-slot prefill warm. One of them first failed under the full file because it
  asserted absolute counter values while other tests in the same module-scoped
  fixture deliberately exercise the host-sampling path; fixed to assert deltas.
* Full-model batch-32 suite: 10 passed on the real 47 layers, including
  `test_batch_slot_isolation_matches_single_user`.
* Full-model batch-1 suite: 47 passed, including
  `test_no_host_fallback_during_traced_decode`,
  `test_split_sampling_trace_feedback`, `test_unchanged_and_changed_page_table`,
  `test_traced_decode_matches_eager_decode`.
* Batch-1 full-model traced decode re-measured without touching any earlier
  stage's artifacts (`full_model_batch1_regression.json`): 21.826 ms/token
  token-out against 23.013 recorded in `doc/optimized_full_model/perf.json`, and
  20.680 model-only against 21.760. No regression from the `moe_kc`/`logits_out`
  plumbing or the persistent logits buffer, both of which are on the batch-1 path
  even though the compact bucketing is not.
* Watcher (`TT_METAL_WATCHER=10`), kept separate from any profiler run because
  no profiler run happened: 0 faults over the reduced adapter suite (22 passed)
  and over targeted batch-32 full-model tests. One of the three batch-32 targets
  hit pytest-timeout's 300 s cap under watcher instrumentation and passes without
  it; disclosed in `watcher/summary.json` and the README rather than dropped.
* Non-aligned prompt lengths through the live server: 37, 129, 777, 1039, 2051
  tokens, all served with the exact `prompt_tokens` echoed back, including two
  that cross the 1024-token serving prefill-chunk cap
  (`non_aligned_serving.json`).
* Runner gates: `check_degenerate_output --scope all` exit 0,
  `check_context_contract --stage optimized-vllm --require-contract` exit 0 with
  target = supported = 202752. The context contract is untouched by this stage.
* Cleanup: no `run_vllm_server`, `EngineCore`, or `vllm.entrypoints` process
  left behind after the last run; `ttnn.open_mesh_device`/`close_mesh_device`
  smoke passes.
* `doc/full_model/accuracy.json` is modified by this stage's commit, and that is
  deliberate: running the full-model suite regenerates it, and the regenerated
  file has **identical** accuracy values (prefill top-1/5/100 0.880/1.000/1.000,
  teacher-forced 0.850/1.000/1.000) with a refreshed `source_manifest` pointing
  at this stage's `tt/model.py`, `tt/generator.py` and `tt/optimized_decoder.py`
  hashes. That is the opposite of clobbering earlier evidence: it re-attests the
  full-model accuracy claim against the current source instead of leaving a
  manifest that no longer describes the code. Nothing else under `doc/` from an
  earlier stage changed.
* The vLLM-integration stage's own `readiness_vllm/` artifacts were copied to
  `doc/vllm_integration/readiness_vllm_stage09/` before this stage's runs
  overwrote that directory, so no earlier stage's committed evidence was
  destroyed in place.

## OV-009 -- Stage-review response: two P1s and seven P2s

An independent `$stage-review` returned `more-work-needed` on the first version
of this stage. Every finding and what was done:

**P1: the before arm's full-sampling record was stage 09's file.** True, and a
real evidence defect. The sequence: the genuine before-arm full run wrote to
`readiness_vllm/sampling_tests.log`; a later `cp -a readiness_vllm/. ->
readiness_vllm_before/` then copied `readiness_vllm/sampling_tests_full_RECORD.log`
-- still stage 09's file, untouched in that directory -- straight over the record
it had just saved. The md5s matched stage 09's, which is how the reviewer caught
it. The number the report quoted (5 failed, 68 passed) was real, but nothing in
the tree proved it. Fixed by re-running the before arm's full profile from the
committed tree: **5 failed, 68 passed, 1 skipped**, now committed with a distinct
md5.

**P1: the two arms did not run the same source revision.** Also true. Three
loguru line numbers in `server.log` were 17 lines apart between arms, and the
before arm had no decode-counter lines at all, because the counter logging was
added between the two runs. The reviewer further inferred the page-table diff was
also absent; that part was wrong (it predates both runs), but the finding stands
and the claim "the two arms differ only in the A/B flag" was false as written.

Fixed properly rather than by rewording. `warmup_prefill_slots` got its own
independent env knob (`GLM47_VLLM_PREFILL_SLOT_WARM`), and all three arms were
re-run from one commit:

| | primary TPOT | burst TTFT p50 | burst output tput |
|---|---|---|---|
| both knobs off (stage entry) | 45.218 ms | 14320.1 ms | 137.649 tok/s |
| per-slot warm only | 45.213 ms | 9126.2 ms | 177.254 tok/s |
| both on (shipped) | 29.497 ms | 9135.8 ms | 177.161 tok/s |

> Those are the round-1 arms. The shipped arms, re-measured after the kc=24
> bucket landed, are in the README and `before_after_summary.json`: 29.496 ms,
> 9132.9 ms, 177.185 tok/s. The attribution conclusion is unchanged.

which also answers a question the first version could not: the single-user
decode win is entirely the MoE change and the burst win is entirely the warm.

**P2: the `kc = n_experts` rejection had no artifact, and the layer table
contradicted it.** Both true. The 92.19/94.07 numbers came from an intermediate
run whose JSON was overwritten, and `moe_compact_layer.json`'s row showing
compact kc=64 *beating* union at 32 live rows was measured on `torch.randn`
activations, which drive the union to ~52-56 of 64 experts -- far wider than real
activations produce, which is exactly why the union path wins on the real model.
Fixed: the rejected variant is now a first-class arm of
`probe_scripts/adapter_decode_floor.py` (`kc64`) with its own committed JSON
(91.77 vs 78.43 ms/token at 32 live rows), and the README's layer table is gone
in favour of the whole-model table that decides the question.

**P2: the shipped kc=32 bucket had no end-to-end correctness evidence.** True,
and it is the bucket whose bound is exactly saturated (8 rows x top_k 4).
`bucket_numerics.py` now covers 8 live rows on the real 47-layer model: bitwise
identical to the union trace, identical argmax, deterministic across repeats.
`compact_decode_equivalence.json` was regenerated under the shipped
`(4, 16, 32, union)` set instead of the superseded `(4, 16, 64)` one.

**P2: `perf_summary.json` did not use the shape `$optimize` mandates.** True.
Rewritten with `workload` / `ttft_ms` / `decode_ms_per_token_e2e` /
`decode_ms_per_token_device` (null, with the reason) /
`roofline_ms_per_token_estimate` / `named_limitations`, keeping the extra
reconciliation and rejected-candidate blocks as additional fields. The roofline
is no longer null: the previous stage's byte model carries over unchanged,
because with one live row and the inactive-row mask the compact `kc=4` bucket
reads exactly the four routed experts the batch-1 build reads.

**P2: a named 7.8 ms/token optimization was deferred on "risk budget" with no
attempt.** Half true, and the more useful half was that the 7.8 ms number itself
was wrong. `moe_prologue_ablation.json` splits the prologue: 0.1202 ms/layer is
the router/top-k/normalize both paths pay, the union path's own tail is 0.1234,
and the compact tail was 0.1904 -- so the compact-specific cost was 3.08
ms/token, not 7.8. Two adapted alternatives were then measured and the fastest
shipped: extending `_moe_decode_indexed`'s `ttnn.embedding` trick to a per-row
`[E, B]` table (0.1599 ms/layer) beat both the original `repeat` + `ttnn.gather`
(0.1904; the repeat is mandatory because `ttnn.gather` does not broadcast its
index against its input) and a one-hot + matmul variant (0.1678). Net **-1.40
ms/token**, which is why the shipped single-user figure moved from 30.84 to
29.497 ms/token between review rounds. The README's residual attribution is
corrected accordingly: 1.68 ms/token compact-path tail, ~6.04 ms/token 32-row
shape, not "all of it is the shard grids".

**P2: trace-buffer footprint had no accounting and `context_contract.json` was
not updated.** True. Measured with `ttnn.get_memory_view(BufferType.TRACE)` on
the real serving build: 15,179,776 of 43,753,472 bytes per bank, 34.7% of the
350 MB reservation, 28.6 MB per bank free with four decode traces plus the
sampling trace. Recorded in `context_contract.json` under `optimized_vllm`, along
with the persistent logits buffer's size and why it exists. The reservation
itself is unchanged, so no other DRAM budget line moves.

**P2: the commit added 51 `__pycache__/*.pyc` files.** True; `models/autoports/`
is force-added because `.git/info/exclude` excludes it, and the `-f` swept those
in. Removed from the index.

**P2: the recapture cost quadrupled, undisclosed, with a stale docstring.** True.
`recapture_decode_traces`' docstring now carries the measured multi-bucket cost
(~1.0 s against ~0.24 s) instead of the pre-stage "~0.17 s", names the remaining
triggers, and the README has it as Limitation 3.

Smaller findings, all fixed: the counter-window quote was cherry-picked and is
now reported as a distribution with the request-boundary windows named; the burst
TTFT attribution now separates the ~5.2 s of recaptures from the ~9.1 s of
irreducible prefill; "8 new tests" was 7;
`test_all_decode_traces_share_one_logits_buffer` now actually replays every
bucket and asserts the logits buffer address never moves and that each replay
wrote it; `_select_decode_trace` raises instead of silently replaying whatever
bucket ran last if a bucket has no trace; `teardown()` now frees the persistent
logits buffer (it previously leaked ~10 MB for the process lifetime, which the
single-trace build did not, because the buffer used to die with the trace) and
the dead `_decode_logits_owned` flag is gone; the `ttnn.split` DRAM-downgrade
warnings and the single trace-allocation warning are classified in the README
against the existing `doc/full_model/logits_memory_ab.json` A/B and
`probe/trace_alloc_probe.py`; the qualitative outlier the reviewer flagged
(trigram-loop fraction 0.2045) does not appear in the final run, whose worst is
0.075; and the README now leads with sampling and qualitative status alongside
the metrics.

Two reviewer observations were checked and **not** adopted as written:

* "All the failing sampling tests run at >= 9 live rows, so they are on the
  unchanged union path." `test_top1_is_greedy` runs a batch of 4, which takes the
  compact `kc=16` bucket. The exculpatory argument is the bitwise identity plus
  the isolated pass, not the row count. Recorded in OV-007.
* "The `ttnn.split` warnings are a prefill/QKV layout fallback." They are
  `TTSampling`'s split of the sampler-ready logits tensor, already classified in
  `tt/model.py`'s constructor comment with a committed A/B showing the L1 default
  is the faster arm and the tokens identical.

## OV-010 -- Closing re-verification after the review response

Everything below was re-run on the final source, not carried over:

* reduced adapter suite 22 passed; full-model batch-32 suite 10 passed;
  full-model batch-1 suite 47 passed; full-model accuracy re-derived identical
  (0.880/1.000/1.000 prefill, 0.850/1.000/1.000 teacher-forced).
* batch-1 full-model traced decode 21.803 ms/token token-out (recorded 23.013),
  20.665 model-only (recorded 21.760): no regression from this stage's plumbing.
* watcher over the adapter suite: 22 passed, 0 faults; watcher over three
  targeted batch-32 tests: 2 passed, 0 faults, 1 pytest-timeout at 300 s under
  instrumentation (passes un-instrumented).
* all three serving arms re-run; both gates re-run (exit 0); non-aligned prompt
  lengths 37/129/777/1039/2051 re-sent through the final server; the after arm's
  eight full-profile failures re-run alone on a fresh server (8 passed).
* no `run_vllm_server`, `EngineCore` or `vllm.entrypoints` process left behind;
  mesh open/close smoke passes.

## OV-011 -- Second stage-review round

The re-review confirmed round 1's ten findings closed and raised five more. What
was done:

**P1: `doc/full_model/accuracy.json`'s refreshed manifest did not describe the
shipped source.** True, and the sharper form of a real problem. The manifest is a
`sha256[:16]` per source file (`tt/provenance.py`), and the `tt/generator.py`
entry had gone stale because the full-model suites ran *before* the last round of
edits (the `prefill_slot_warmup` knob, the `_select_decode_trace` raise, the
`teardown()` deallocate). So OV-008's "it re-attests the accuracy claim against
the current source" was false, and worse than leaving the file alone. Fixed by
re-running the suites on the shipped source rather than by editing the file:
`test_full_model.py` **47 passed**, `test_full_model_batch.py` **10 passed**,
adapter suite **22 passed**, and all seven manifest entries now match the
committed blobs, with the accuracy values unchanged (0.880/1.000/1.000 prefill,
0.850/1.000/1.000 teacher-forced). `full_model_batch1_regression.json` and
`adapter_decode_floor_after.json` were re-measured on the shipped source too.

The three serving arms were **not** re-run, and that is deliberate rather than
an oversight: the edits that landed after them are a `raise` on an unreachable
`_kc_by_rows` branch, a `raise` on an unreachable `_select_decode_trace` branch,
a `B % TILE` guard evaluated once per layer at capture, a `teardown()`
deallocate that runs at process end, and docstrings. None of them can move a
serving measurement, and the decode-floor re-run on the shipped source lands
within 0.01 ms/token of the pre-edit one (29.510 vs 29.518), which is the check
that would have caught it if they could.

**P2: stage 09's README cited `readiness_vllm/`, which now holds this stage's
numbers.** True. The bytes were preserved at
`doc/vllm_integration/readiness_vllm_stage09/`, but stage 09's own report still
pointed a reader at the live directory. Its README and work log now point at the
snapshot, and the snapshot has a `README.md` saying what it is and who moved it.

**P2: the shipped `COMPACT_KC_BUCKETS` docstring justified the bucket set with
numbers from the superseded intermediate run**, including "8 live rows 60.41 ->
92.19" which is not even true of the shipped set (8 live rows takes kc=32). This
was round-1's P2-3 surviving in the artifact that outlives the report. Rewritten
as a table against the three committed decode-floor JSONs.

**P2: two committed artifacts gave opposite signs for compact kc=64 vs union at
32 live rows, and the reconciliation was prose.** The most useful finding of
either round, because the missing measurement was the one number the whole
bucket design turns on. `probe_scripts/moe_union_width.py` now measures it:
wrapping `_routing_weights_decode` on the **eager** decode path (legal to read
back from) over the real 47-layer model, the distinct-expert union averages 4.00
/ 6.41 / 10.41 / 16.65 / 23.43 / 32.17 at 1 / 2 / 4 / 8 / 16 / 32 live rows,
against a bound of 4 / 8 / 16 / 32 / 64 / 64. The same measurement under the
single-layer probes' synthetic `torch.randn` activations gives **52** at 32 rows.
That is the sign flip, measured: the layer probes let the union path's active
count float up to 52 while the real model's is ~32, so they flatter the
fixed-width compact form. `moe_compact_layer.json` and
`moe_union_vs_compact.json` are now stamped with the caveat and with which of
their comparisons remain valid (fixed-active-count ones), the README carries the
width table, and `_moe_decode_compact`'s docstring no longer asserts the "pure
win at equal coverage" the whole-model arm refutes.

**P2: the commit message contradicted every headline it committed** (it was
written for the round-1 numbers and not updated when the commit was amended).
Rewritten against the committed artifacts.

Smaller findings, all fixed: `perf_summary.json`'s
`compact_specific_routing_prologue_ms_per_token` said 1.40 (the amount
*recovered*) where the arithmetic and the prose said 1.68 (the residual); OV-007
claimed the penalty mixed-batch tests flip between arms when they fail in both;
the work log still said "8 new tests"; the README's "~24 counter windows" is 33,
of which 32 show the contract, and the one that does not is now explained
(seeded eager sampling plus the smoke profile's `test_min_p` logprobs request,
which vLLM itself forces to host-sample on a single-chip mesh); `_kc_by_rows`
kept a `buckets[-1]` fallback that the new `_select_decode_trace` raise could not
catch, because a too-narrow bucket *is* captured -- it raises now too;
`_moe_decode_compact` asserts its `[E, B]` embedding table needs a whole-tile
decode batch instead of assuming it; the README says which `server.log` each
claim is against; and the attribution arm's `--stages serve,benchmark` scope is
disclosed next to the attribution table rather than only in the command block.

## OV-012 -- Third stage-review round

Round 3 confirmed all thirteen round-2 findings closed and raised six more, one
of them a real shipped regression that two review rounds and this stage's own
sweep had all missed.

**P2: "no regression at any batch size" was false at 5 and 6 live rows.** The
reviewer noticed that `adapter_decode_floor.py` swept only 1 / 4 / 8 / 32 --
exactly the row counts where each bucket's bound is *saturated*, i.e. each
bucket's best case -- and derived, from the committed union-width curve and
three independently measured union-cost segments, that kc=32 must be a net loss
at 5 rows. Measured, sweeping every row count where the bucket choice changes:

| live rows | union | compact bucket | delta |
|---|---|---|---|
| 1 | 45.208 | 29.515 (kc 4) | -15.69 |
| 2 | 48.436 | 41.585 (kc 16) | -6.85 |
| 3 | 51.283 | 41.600 (kc 16) | -9.68 |
| 4 | 53.397 | 41.603 (kc 16) | -11.79 |
| **5** | **55.377** | **57.519 (kc 32)** | **+2.14** |
| **6** | **56.749** | **57.516 (kc 32)** | **+0.77** |
| 7 | 58.949 | 57.524 (kc 32) | -1.43 |
| 8 | 60.213 | 57.537 (kc 32) | -2.68 |

The mechanism is structural, not incidental: a compact bucket's cost is flat
across the rows it serves (every op in `_moe_decode_compact` is shaped by `kc`
and `B`, not by live rows) while the union path's grows with the real union
width, so inside a bucket's range the compact form starts behind and crosses
over. Fixed with a measured crossover table, `COMPACT_KC_MIN_ROWS = {4: 1,
16: 2, 32: 7}`: a bucket is only selected from the row count where it was
measured to win, and rows below that replay the union trace, which is always
correct and here also cheaper. Re-measured across the same sweep, the worst
delta anywhere is now **+0.009 ms/token**, i.e. noise. Pinned by
`test_kc_bucket_is_only_chosen_where_it_is_the_cheaper_path`.

This did not move any headline: the primary benchmark runs at one live row
(kc=4) and the burst at 32 (union), so neither measured workload was in the
5-6 window. It would have shipped a 4% regression over an ordinary serving
range, which is exactly the kind of thing a saturated-points-only sweep hides.

**P3: the new `B % TILE` guard hard-failed every `max_batch_size` in 2..31.**
Round 2 asked for a guard on the `[E, B]` embedding table's width and got a
`raise`, which turned a previously working configuration (`GLM47_FM_BATCH=8`,
a documented knob) into a fixture-setup failure. Compaction now simply does not
engage for a non-tile decode batch -- `_kc_buckets` returns `()` and the model
takes the batch-agnostic union path it always used -- with the `ValueError`
demoted to a backstop for a wiring bug. Pinned by
`test_compaction_is_off_rather_than_fatal_for_a_non_tile_decode_batch`.

**P3: superseded numbers left in shipped artifacts.** `perf_summary.json` was
regenerated from the final artifacts rather than hand-edited, the
`COMPACT_KC_BUCKETS` and `_moe_decode_compact` docstrings now carry the final
sweep, and `moe_prologue_ablation.json`'s key that called the *rejected*
repeat+gather variant "shipped" is renamed
(`compact_prologue_first_implementation_repeat_gather` /
`compact_prologue_SHIPPED_embedding_both_indices`). OV-005's "99-100 of every
100" is corrected to the measured 98-100.

**P3: `watcher/summary.json` claimed "on the final source" while the runs
predated the last edits.** Re-run rather than reworded: adapter suite under
`TT_METAL_WATCHER=10`, **24 passed, 0 faults**, on the shipped source.

**P3: the batch-1 no-regression guard compared a median against a mean.** True
and worth checking -- the baseline it guards
(`tests/test_full_model_perf.py`'s `bench()`) is a mean. The probe now emits
both and the comparison is mean-to-mean. The reviewer's hypothesis that the
reproducible ~5% improvement was a median-vs-mean artifact turns out to be
wrong: median 21.821 and mean 21.822 agree to 0.001 ms, so the improvement over
the recorded 23.013 is real (the persistent logits buffer replaced a per-step
allocation). Recorded either way, because the guard was genuinely not
like-for-like.

**P3: a stale artifact path and a stale docstring summary.**
`prefill_recapture_probe.py` wrote an unsuffixed JSON that matched neither
committed arm; it now takes a `before`/`after` argument, drives the
`prefill_slot_warmup` knob itself, and both committed arms were regenerated by
running it. `recapture_decode_traces`' summary line no longer says "both decode
traces".

The final full sampling profile on the shipped source is **7 failed, 66 passed,
1 skipped** -- a fourth measurement of the same suite (11 / 5 / 8 / 7 across the
vLLM-integration stage, this stage's before arm, an intermediate after arm, and
the final one). `test_top1_is_greedy`, the batch-4 case round 2 correctly
refused to explain away with the row-count shortcut, **passes** in the final
run with no code change touching it, which is one more direct data point that
the failing set is drawn run to run from one pool rather than caused by
anything this stage did.

## OV-013 -- Fourth stage-review round

Round 4 confirmed all six round-3 findings closed in substance and raised nine
more, one P2 and eight P3s. The P2 was the round-3 fix landing everywhere except
the two places round 2 had specifically called out as "the artifact that
outlives the report".

**P2: the shipped `COMPACT_KC_BUCKETS` and `_moe_decode_compact` docstrings still
carried the superseded intermediate table**, and OV-012 claimed they had been
rewritten. Only the *new* `COMPACT_KC_MIN_ROWS` docstring had been. Both are now
written against the committed 11-point sweep, and the claim is corrected.

**P3s, all fixed:** the retired `78.20` and its derived `13.6 ms/token` (the
delta is 13.3 against the committed 78.43); the README's sampling heading and
"three more failures"; the degenerate-check worst value (0.0865 overall, 0.075
worst *greedy*); Limitation 8's freshness sentence, now an explicit
re-measured/not-re-measured list; `moe_prologue_ablation.py` still *generating*
the key that called the rejected variant "shipped"; the reduced-equivalence
"every live-row count from 1 to 32" overclaim (it is six counts) plus its stale
`kc_by_rows` row; OV-004's superseded table still labelled "Final"; and the
stage-09 correction block's `29.497`.

Two of the reviewer's "other concerns" turned into real changes:

**The permissive crossover default.** `_kc_by_rows` used
`COMPACT_KC_MIN_ROWS.get(pick, 1)`, so adding a bucket to `COMPACT_KC_BUCKETS`
-- documented as "the knob" -- would give it an unmeasured crossover of 1 and
silently reintroduce exactly the regression the table exists to prevent. The
lookup is strict now, `_kc_buckets` raises on a bucket with no measured
crossover, and `test_every_bucket_has_a_measured_crossover` pins it.

**Finer buckets, rejected on pre-crossover reasoning.** The reviewer pointed out
that the "finer buckets" rejection was written before the crossover was known,
and that rows 5-6 had just been moved onto the union trace, so a `kc=20`/`24`
bucket was plausible and unmeasured. Both halves turned out to be right:

* `ttnn.topk` accepts a non-power-of-two `k` (checked on device for k = 4, 8,
  12, 16, 20, 24, 32, 48, 64), so there is no op-contract blocker.
* A **kc=24** bucket for rows 5-6 measures 49.745 / 49.756 ms/token against the
  union trace's 55.377 / 56.749 -- 10-12% better on that range, nothing worse
  anywhere else. **Shipped.** `COMPACT_KC_BUCKETS = (4, 16, 24, 32)`,
  `COMPACT_KC_MIN_ROWS = {4: 1, 16: 2, 24: 5, 32: 7}`.
* The limit case, one bucket per reachable row count (`kc = live_rows * top_k`
  for rows 1..8, eight compact traces plus the union trace), was measured too:
  faster everywhere it differs (-8.0 / -3.9 / -4.1 / -3.8 ms/token at 2 / 3 / 5
  / 7 live rows), nothing worse. It is **deferred on a measured resource cost,
  not on latency**: 78.4% of the 350 MB trace region against the shipped set's
  43.5%, leaving 9.4 MB per bank against 23.6 MB for the additional traces
  `models/common/sampling` captures per sampling mode. Adopting it should come
  with a multi-sampling-mode serving run proving the region still fits.
  `adapter_decode_floor_kcexact.json`.

The two remaining coverage gaps the reviewer named were closed by running them:
`GLM47_FM_BATCH=8 pytest tests/test_full_model_batch.py` -- the concrete case
the non-tile fallback exists for -- **10 passed**, and the `kc64` rejection arm
was re-run under the current script so it is reproducible from its own probe
(and is worse than previously recorded: +24.4 and +18.7 ms/token at 12 and 16
live rows, +13.3 at 32).

Everything was then re-measured on the shipped source: the after serving arm
(both benchmark profiles, smoke sampling, qualitative, full sampling record),
all five decode-floor arms, both recapture-probe arms, the batch-1 regression
probe, both gates, the adapter suite (25 passed) plain and under watcher (0
faults), and the full-model batch-1 (47) and batch-32 (10, plus 10 at batch 8)
suites, which regenerated `doc/full_model/accuracy.json` with a matching source
manifest. README Limitation 8 lists what was and was not re-measured, and why
the remainder is unaffected, instead of asserting that everything was.

The final full sampling profile is 8 failed / 65 passed / 1 skipped -- a fifth
measurement of the same suite (11 / 5 / 8 / 7 / 8), with `test_top1_is_greedy`
failing in this one and passing in the previous one with no code change touching
it, which is the run-to-run variance the exculpatory argument rests on.

## OV-014 -- Fifth stage-review round, and a mechanical guard against the pattern

Round 5 found that the kc=24 bucket added in the round-4 response was carried
through the latency evidence and the selection code but **not** through the
correctness probes, the trace-count narrative, or the two docstrings round 2 had
called "the artifact that outlives the report". Three findings, all the same
omission from different angles, plus six smaller ones.

**The one that could have hidden a defect: kc=24 shipped with no numerical
evidence.** `bucket_numerics.py` hard-coded `{1: 4, 4: 16, 8: 32}`, so
re-running it would never have touched kc=24, and
`compact_decode_equivalence.py` swept rows that skip 5 and 6. Both probes now
**derive** their row list from the shipped bucket table instead of hard-coding
it: `bucket_numerics.py` picks the live-row count where each bucket's bound is
exactly saturated (its zero-slack case) and *raises* if any shipped bucket has
no such row, and `compact_decode_equivalence.py` sweeps every row count where
the selection table changes bucket. Re-run:

* real 47-layer model, saturated row per bucket (1/kc4, 4/kc16, **6/kc24**,
  8/kc32): every bucket **bitwise identical** to the union path, identical
  argmax, identical repeats;
* reduced 2-layer model at 1, 2, 4, **5**, **6**, 8, 16, 32 live rows: row
  checksum delta exactly 0.0 and identical argmax at every count.

6 live rows is kc=24's saturated bound (6 x top_k 4 = 24), so the bucket is now
covered at exactly the point where an imperfection would drop a genuinely
selected expert. The reviewer also established independently that `ttnn.topk`
rounds `k` up to a tile multiple and slices back
(`ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp`), so k=24 takes the same
`adjusted_k = 32` path as k=4 and k=16.

**The docstring tables, again.** Round 4's fix rewrote them, but with numbers
from the pre-kc24 measurement run, and cited artifacts that did not contain
them. Rather than retype them a third time, both tables are now **generated
from the committed JSONs**, and three new device-free tests make that
structural:

* `test_shipped_bucket_docstring_table_matches_the_committed_measurements`
  parses `COMPACT_KC_BUCKETS`' docstring table and asserts every union value,
  shipped value, bucket name and delta against
  `adapter_decode_floor_{before,after}.json`;
* `test_report_states_the_shipped_bucket_set` asserts the README's prose bucket
  set is the code's, and that it states no other;
* `test_every_shipped_bucket_has_bitwise_equivalence_evidence` asserts
  `bucket_numerics.json` covers every shipped bucket at its saturated row and
  that each is bitwise identical.

That is the mechanical guard the reviewer asked for in Hard-Check Gaps. Four of
five review rounds found this class of defect; it cannot recur silently now.

**The rest, all fixed:** the README still said the bucket set was
`(4, 16, 32, union)`, "four decode traces", "34.7%" and "28.6 MB" while
`context_contract.json` and `perf_summary.json` said five / 43.5% / 23.6 MiB;
`perf_summary.json`'s kc=32-at-5-6-rows block cited arms that do not contain
that measurement (it comes from `adapter_decode_floor_kc64.json`, whose bucket
set selects kc=32 there); the sampling bullets still described the final arm as
7 failures with `test_top1_is_greedy` passing, when the committed final record
is 8 with it failing; `test_kc_bucket_is_only_chosen_where_it_is_the_cheaper_path`'s
docstring contradicted its own assertions and its loop still used the permissive
`.get(kc, 1)`; Limitation 8's freshness list claimed re-measurement for three
arms that predate the last edit and gave a false reason for
`compact_decode_equivalence` (which does consult the selection table); "9 new
tests" was 10 and "OV-001..OV-011" was OV-013; the `readiness_vllm/server.log`
provenance sentence was wrong (it is the same after-arm run, differing only by
the pre-commit hook's whitespace fixes); the stage-09 correction block quoted
45.34 ms/token, which is OV-001's superseded first run rather than the committed
45.208; `watcher/summary.json` described the batch-32 run's gap as "host-side
Python" without naming the kc=24 bucket, which is a new device program; and
MB/MiB were mixed in one sentence.

**The `kc24` probe arm is retired.** Once kc=24 shipped, that arm's monkeypatch
became identical to the defaults, so its JSON was a duplicate of the `after` arm
presented as a distinct measurement. Arm and artifact removed; the kc=24 numbers
are the shipped arm's.

## Commits

Branch `ttmodelmanager/glm47-flash-probe`. Not pushed.

* **`3fdafd7b4da5598b2da4e603864393f1c054b957`** -- the whole stage as one commit: implementation
  (`tt/optimized_decoder.py`, `tt/model.py`, `tt/generator.py`,
  `tt/generator_vllm.py`), 13 new adapter tests, all three serving arms, every
  probe script and artifact, this report, and all five `$stage-review`
  responses (OV-009, OV-011, OV-012, OV-013, OV-014). It also preserves the
  vLLM-integration stage's runner artifacts at
  `doc/vllm_integration/readiness_vllm_stage09/`, repoints that stage's report
  at them, and corrects that report's "~22 ms/token of vLLM-path overhead"
  claim in place.
* the follow-up commit recording this line.
