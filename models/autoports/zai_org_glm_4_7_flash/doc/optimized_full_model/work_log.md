# GLM-4.7-Flash optimized full model — work log

Stage 07 (`optimized-full-model`), starting from full-model commit
`70b509b361a` (branch `ttmodelmanager/glm47-flash-probe`, single Blackhole
p150, device 0).

## OFM-001: what this stage found already done, and what it verified fresh

The full-model stage's 10 `$stage-review` rounds folded a substantial amount
of `$optimize`-checklist work into `doc/full_model/` already: the LM-head
`in0_block_w` sweep (FM-020/021/022), the split-sampling greedy contract with
a measured force-argmax rejection (`doc/full_model/README.md`'s "What the
full model adds" section, `greedy_sampler_benchmark.json`), the reduced
2-layer `tt-perf-report` profiling variant
(`tests/test_full_model_profile.py`), and an explicit
`layer_stack_lower_bound` block already computed inside `perf.json` (added in
FM-016). This is why this stage's real work is narrower than a from-scratch
optimize pass: it is an audit-and-close pass over what full-model explicitly
disclosed and deferred, not a rediscovery of the decoder-level optimizations
already shipped in `doc/optimized_decoder/`.

Two explicit deferrals were found in the full-model artifacts, addressed
below:

1. `doc/full_model/perf_report_summary.json`'s `bound.note`: "`Bound == SLOW`
   is a tt-perf-report advisory ... Disclosed, not fixed here; matmul
   geometry is optimized-full-model (stage 07) work."
2. `tt/model.py:440` (the `decode_logits_memory_config` comment): "Stage 07
   should re-run the A/B if it adds resident L1 pressure around the terminal
   path."

## OFM-002: SLOW-bound matmul rows — confirmed already at the swept-optimal geometry

Pulled every row flagged `Bound: SLOW` from the fresh
`tracy/decode_model_perf_report.csv` (this stage's own capture, not the old
full-model CSV) via a small inline `csv.DictReader` script, for the 7 shapes
`perf_report_summary.json` names. For each row, cross-referenced `Inner Dim
Block Size` / `Math Fidelity` / `Input 0/1 Datatype` against the exact
program-config constructor call in `tt/optimized_decoder.py` that produces
that shape:

| shape (M x K x N) | identity | CSV `in0_block_w` | source `in0_block_w` | CSV dtype/fidelity | source policy |
|---|---|---|---|---|---|
| `32x2048x1536` | wqkv_a | 8 | `_dram_pc(in0_block_w=8, ...)` (`optimized_decoder.py:265`) | LoFi BF16xBFP4 | matches deployment (bf4 attn) |
| `32x768x5120` | wq_b | 3 | `_dram_pc(in0_block_w=3, ...)` (`:269`) | LoFi BF16xBFP4 | matches |
| `b={20}x32x192x512` | w_uk | 6 | `in0_block_w=self.qk_nope//TILE=192/32=6` (`:401`) | LoFi BF16xBFP4 | matches |
| `b={20}x32x512x256` | w_uv | 8 | `in0_block_w=8 if attn_dtype==bf4 else kv_lora//TILE` (`:410`) | LoFi BF16xBFP4 | matches (bf4 arm) |
| `32x2048x10240` | dense gate/up | 32 | `_mcast_1d_pc` picks `max(d<=48 dividing kt)`; kt=64 -> 32 | LoFi BF16xBFP8 | matches (dense MLP stays bf8, per optimized_decoder's rejected-bf4-dense finding) |
| `32x2048x64` | router | n/a (2-core 1D, `osw_cap=1`) | `_mcast_1d_pc(..., 2, osw_cap=1)` (`:309`) | HiFi4 BF16xFP32 | matches (fp32 selection semantics) |
| `32x1536x2048` | shared-expert down | 6 | `_dram_pc(in0_block_w=6, ...)` (`:302`) | LoFi BF16xBFP4 | matches |

Every row matches. `tt-perf-report`'s advice for each is `in0_block_w=N looks
good 🤷` (no change suggested) plus a generic HiFi2/HiFi4-for-accuracy
suggestion, which is a precision tradeoff in the wrong direction for this
goal (this stage's job is speed, not accuracy, and accuracy already clears
its bar). **Verdict: OPT-013 ("prove dtype policy reached the measured ops")
holds for the full-model measured path, not only for the standalone decoder
unit tests. No config change made.**

Initial confusion worth recording: the shape `32x1536x2048` was first
misidentified as `wo` (output projection) by pattern-matching the wrong
dimension; `wo`'s real shape is `32x5120x2048` (`num_heads=20 *
v_head_dim=256 = 5120`, from the checkpoint's `config.json`:
`num_attention_heads=20`, `v_head_dim=256`), which does **not** appear in the
SLOW list at all (it is a top-time row but not flagged SLOW). The
`32x1536x2048` row is `moe_intermediate_size=1536 -> hidden_size=2048`, i.e.
shared-expert (or dense) down-projection. Recorded because a wrong shape
identification would have produced a false "config doesn't match" finding.

## OFM-003: router `out_subblock_w` — structurally blocked, not investigated further

`tt-perf-report` advice on the router row: "Output subblock 1x1 is small, try
`out_subblock_h*out_subblock_w >= 2`". `n_experts=64` -> 2 tiles;
`_mcast_1d_pc(nt=2, kt=64, target_cores=2, osw_cap=1)` gives `per_core_n =
ceil(2/2) = 1` tile, and `ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig`
requires `out_subblock_w <= per_core_N`. Getting `osw=2` needs
`per_core_n>=2`, i.e. `target_cores=1` — which `doc/optimized_decoder/README.md`
item 5 already measured as much worse (25.8 us single-core vs 8.5 us at
2 cores, a >3x regression from a similar single-vs-two-core tradeoff on this
exact matmul). Not re-measured: retesting the 1-core family would be
re-opening a comparison the decoder stage already closed with a clear
margin, for an op that is 7.5 us/layer x 46 layers = ~1.6% of model-only
decode. No config change made.

## OFM-004: `attn_lat` DRAM vs L1 (tt-perf-report: "place input 0 in L1")

`paged_flash_multi_latent_attention_decode`'s output (`attn_lat`, consumed by
the `w_uv` matmul) is placed with `memory_config=ttnn.DRAM_MEMORY_CONFIG`
explicitly (`optimized_decoder.py:579`). No comment there explains why (unlike
the neighboring `kvpe` DRAM placement, which is commented: "at batch 32 the
paged_update_cache static CBs need the L1 headroom"). Tested directly with
`doc/optimized_full_model/probe_scripts/attn_lat_l1_test.py`, which
monkeypatches `ttnn.transformer.paged_flash_multi_latent_attention_decode` to
force `memory_config=ttnn.L1_MEMORY_CONFIG` on an isolated `OptimizedDecoder`
instance (moe layer, synthetic weights, ctx 1024, batch 1), traces one decode
step, times 32 replays:

```
python attn_lat_l1_test.py --attn-lat-mem dram --batch 1   # 0.4926 ms/token
python attn_lat_l1_test.py --attn-lat-mem l1   --batch 1   # 0.4914 ms/token
```

0.24% difference — within this harness's run-to-run spread (compare
`dev_optimize.py`'s independent baseline measurement of the same config:
0.4921 ms/token). The isolated op itself is 11.78 us out of ~445 us/layer
average, so even a large *relative* win here is small in absolute terms.
Attempted a batch-32 run to check L1 headroom safety before considering
shipping this; the ad hoc script's RoPE/`cos`/`sin` batch-32 plumbing is
incomplete (`TT_FATAL: Cos and Sin must have the same batch size as the
input`) — a bug in the throwaway test harness, not in shipped code (the real
batch-32 path uses `generator.py`'s `kvpe_update_groups`/`_decode_rope_mats`,
which is exercised and passing in `test_full_model_batch.py`). Not fixed,
because the batch-1 result was already a rejection: no measured win to
justify chasing the batch-32 safety check. **Verdict: rejected, kept DRAM
(unchanged from full-model).**

## OFM-005: sampler TopK vocab chunking — INITIAL FINDING, SUPERSEDED

**Corrected in OFM-010/README item 5: this entry's benchmark had no warmup
before the timed loop, so the "wider chunks are 19-26x slower" conclusion
below is a cold-compile artifact, not a real result. Properly warmed and
correctness-checked, fewer/wider chunks are ~11% faster, not slower. Read
OFM-010 and README item 5 for the corrected finding; this entry is kept
for the record of what was originally measured and why it was wrong.**

Original (superseded) entry follows.

`perf_report_summary.json`'s `decode_tokenout` window: `TopkLargeIndicesDeviceOperation`
is 24.2% of device time (664.9 us/step, x4 calls), plus
`TopkRouteFinishDeviceOperation` (3.0%, x4) and `TopkRoutePrepDeviceOperation`
(2.5%, x4) — the vocab-chunked top-k pipeline `models/common/sampling/tt_sampling.py`
uses to sample from the 154880-wide logits. `TOPK_MAX_WIDTH = 64*1024` forces
a 4-way split (`num_single_device_vocab_splits`: `154880 // 2 = 77440 >
65536`, so `num_splits` doubles to 4; `154880 // 4 = 38720 <= 65536`, legal
and tile-aligned). The routing predicate mirror
(`models/common/sampling/_utils.py:topk_would_route_to_large_indices`) and its
C++ counterpart (`ttnn/cpp/.../topk/topk.cpp:should_route_to_topk_large_indices`)
document a much wider legal envelope for the fast Blackhole composite
(`large_k_route_max_width = 2^19 = 524288`), which raised the question of
whether the 64K Python-level ceiling is a stale, overly conservative
constant from before that composite existed.

Tested directly rather than inferred from the source comment, with
`doc/optimized_full_model/probe_scripts/topk_width_test.py` and
`topk_width_test2.py` (raw `ttnn.topk` calls, matching the production
`[1,1,32,vocab]` bf16 shape and `k=32`, eager — no trace — 32 iterations,
`ttnn.synchronize_device` bracketing each timed loop):

| chunks | width/chunk | total wall (32 iters) |
|---|---|---|
| 4 (shipped) | 38720 | 0.98-1.0 ms |
| 2 | 77440 | 25.7 ms (26x slower) |
| 1 | 154880 | 18.7 ms (19x slower) |
| 8 | 19360 | 34.2 ms (35x slower) |
| 10 | 15488 | 45.1 ms (46x slower) |

Both *wider* (1, 2) and *narrower* (8, 10) alternatives are drastically
slower than the shipped 4-way split. The wider-chunk regression is a genuine
device-side effect (fewer total op launches, yet far slower — rules out a
dispatch-count artifact): `padded_width=77440` and `154880` are not
power-of-two and exceed the wrapper's 64K ceiling, so they take a different
(much slower) code path than the 4-way split's 38720-wide calls, consistent
with falling off of whatever width envelope the installed `ttnn.topk`
actually enforces in practice on this build, regardless of what the
Blackhole-composite routing-predicate comment's stated envelope claims. The
narrower-chunk (8, 10) numbers are confounded by this test harness running
eager (no trace capture) — more chunks means more per-call host dispatch in
an untraced loop — so they are not conclusive evidence about device time
alone, but they were never a live candidate (narrower chunks cannot reduce
total work, only add per-call fixed overhead) so the ambiguity does not
matter for this decision.

**Verdict: `TOPK_MAX_WIDTH=64K` and the resulting 4-way chunking are the
empirically fast path at this model's vocab width on this hardware, not a
stale conservative limit. No change made to the shared common-infra file.**
Given this is shared code (`models/common/sampling/`) used across models and
architectures, the bar for changing it is a verified win at the point of
use, which this A/B does not show — if anything it shows the opposite.

Separately: this cost is 1.12 ms out of the real 47-layer model's 23.013 ms
token-out step (4.9%), not the majority of token-out decode (the goal's
"if a sampler op dominates token-out decode, fix it" gate does not apply;
LM head + decoder layers are the other 95.1%).

## OFM-006: re-verified the logits-memory-config A/B for this stage, as asked

`tt/model.py`'s `decode_logits_memory_config` comment explicitly asks stage
07 to re-run `probe/logits_memory_ab_probe.py` "if it adds resident L1
pressure around the terminal path." This stage added none (OFM-004's
candidate was rejected, not shipped), so the full-model conclusion should be
unchanged — verified rather than assumed:

```
python probe/logits_memory_ab_probe.py
```

| arm | model-only ms | token-out ms | sampling ms |
|---|---|---|---|
| L1 (shipped) | 1.784 | 2.916 | 1.132 |
| DRAM | 1.829 | 2.940 | 1.111 |

L1 remains faster end to end (2.916 vs 2.940 ms token-out) despite paying the
one-time-per-trace-capture `ttnn.split` L1->DRAM downgrade
(`logs/logits_memory_ab.log`), matching the original full-model finding
(`doc/full_model/logits_memory_ab.json`, superseded by the copy at
`doc/optimized_full_model/logits_memory_ab.json`) within run-to-run noise.
Tokens are bit-identical between arms. **Verdict: L1 remains the correct
choice; unchanged.**

## OFM-007: runtime fallback audit

Grepped every log this stage produced (`logs/*.log`, the qualitative suite,
all pytest modules, the three watcher runs, the profiler capture) for
`fallback|downgrade|L1 budget exceeded|unsafe`. The only substantive match
everywhere is the OFM-006 `ttnn.split` L1->DRAM downgrade, always with
identical byte-for-byte text, always at sampling-trace-capture time (never
during a captured-trace replay). `test_no_host_fallback_during_traced_decode`
(one of the 47 tests in `pytest_full_model_only.log`) passed. No other
fallback signature (host argmax outside its counted/disclosed paths, full-
logits readback outside the disclosed one-word readback, unexpected eager
step) appears anywhere in this stage's evidence.

## OFM-008: fresh evidence run (this stage, no source changes)

Ran the accuracy gates, full correctness suite, batch-32 suite, perf test,
reduced-layer Tracy profile + `tt-perf-report` tables, three watcher runs,
context-contract check, and the shared qualitative suite (HF control reused
via `--skip-hf`, TT side regenerated) fresh against the unchanged
`70b509b361a` source tree. All results reproduce the full-model numbers
within run-to-run noise (see README's headline table). Commands and exit
codes:

```
run_prefill_check          -> exit 0, top1=0.880 top5=1.000 top100=1.000
run_teacher_forcing        -> exit 0, top1=0.850 top5=1.000 top100=1.000, TTFT 590.6ms
pytest test_full_model.py  -> 47 passed
pytest test_full_model_batch.py (batch32 @ 8192)   -> 10 passed
pytest test_full_model_perf.py -> perf.json refreshed
python -m tracy -r -p -v -m pytest test_full_model_profile.py -> tracy capture + reports
tt-perf-report x3 (decode_model, decode_tokenout, prefill windows)
summarize_perf_report.py -> perf_report_summary.json
TT_METAL_WATCHER=2 dev_full_model.py {smoke, trace, capacity} -> 0/0/0 faults
check_context_contract.py -> OK, supported=202752
run_qualitative_suite.py --skip-hf -> 6 prompts, coherent completions both sides
probe/logits_memory_ab_probe.py -> L1 still faster
```

Full logs in `doc/optimized_full_model/logs/`.

## OFM-009: closing state

Zero source-line changes to `tt/*.py` from this stage. Every candidate
investigated (OFM-002 through OFM-006) was measured on real hardware with
real weights/shapes at the model's actual dtype policy and either confirmed
already-optimal or rejected with quantified evidence. The two explicit
full-model deferrals (OFM-001) are both closed. The decoder-layer-stack
lower bound (README) shows a negative gap: the full model is already faster
than the naive per-isolated-layer estimate, so there is no closure work to
split. Runtime fallback audit is clean for the replay path. Accuracy,
batch-32, watcher, context-contract, and qualitative evidence all reproduce
the full-model stage's results fresh.

## Checkpoint

Repo: `tt-metal`, branch `ttmodelmanager/glm47-flash-probe`, no push.

| commit | contents |
|---|---|
| `70b509b361a` | parent: full-model stage, closing commit |
| (this commit) | `doc/optimized_full_model/` (README, this work log, fresh perf/accuracy/tracy/watcher/qualitative evidence, probe scripts used for OFM-004/OFM-005); no changes to `tt/*.py` |

No other repo was touched: `vllm` is out of scope for this stage.
`models/common/sampling/tt_sampling.py` and
`ttnn/cpp/.../reduction/topk/topk.cpp` were **read**, not edited (OFM-005).
Unrelated dirty files in the checkout (`.env`, `model_cache/`, `tt_cache/`,
`models/tt_dit/...`, various HTML/notes/scripts at the repo root, the
`tt_metal/third_party/tt_llk/` submodule state) were left untouched and are
not in the commit.

## OFM-010: independent `$stage-review` round — five P1s, all fixed

An independent xhigh subagent review (per the stage's review budget: one
round) returned `more-work-needed` with five P1 findings and several P2s.
Each is addressed below; README.md is the corrected report, this entry is
the response record.

**P1: the LM head's own `tt-perf-report` advice ("try a DRAM-sharded program
config") was never tried**, and OFM-002's "🤷 no actionable advice" framing
was true only for the 7 `SLOW`-bound rows it enumerated, not for the LM
head (`Bound=DRAM`, a different classification with different, genuinely
actionable advice). Fixed: built and traced a DRAM-sharded LM-head candidate
(README item 1). Every legal `in0_block_w` fails identically at
`program.cpp:1875` with an exact, quantified L1 overflow (17,187,328 B
needed vs 1,572,864 B available on the 8-core bank-fixed grid) — a real
op-contract blocker, not an untried candidate. Also ran the LoFi-vs-HiFi2
fidelity trial on the (legal) shipped topology that OFM-002 should have
covered for the LM head specifically: 0.6% device-time difference, not
worth the accuracy risk on the final pre-sampling projection.

**P1: OFM-004 and OFM-005's rejection numbers had no runner artifact**
(logs), and the units were mislabelled (`(t1-t0)/iters*1000` is ms **per
call**, not "total wall (32 iters)" as both the table header and prose
said). Fixed: every probe now runs with output teed to `logs/`
(`attn_lat_dram_b1.log`, `attn_lat_l1_b1.log`, `topk_width_test1.log`,
`topk_width_test2.log`, `lm_head_dram_sweep.log`), and README's tables are
now labelled "ms/call" or "ms/token" per what each script actually measures.
Re-running OFM-004's `attn_lat` probe with logging reproduced the original
numbers closely (0.4926/0.4914 -> 0.4928/0.4916), confirming that specific
finding was not affected by the missing-artifact defect. **Re-running
OFM-005's TopK probe with logging is what surfaced that its underlying
methodology was wrong** — see the next finding.

**P1 (self-discovered while fixing the artifact-logging finding above, more
serious than the finding itself): OFM-005's core conclusion was wrong.**
The original `topk_width_test.py`/`topk_width_test2.py` scripts called
`ttnn.topk` at each chunk width with no warmup before the timed 32-iteration
loop, so the first iteration at each never-before-seen shape paid full
program-cache-miss compile cost inside the average. Re-measured with 8
warmup calls per shape (uncounted) plus a `torch.topk` correctness check
(`probe_scripts/topk_width_clean.py`, `logs/topk_width_test1.log` /
`topk_width_test2.log` for the original flawed runs kept as a record, no
separate clean-run log beyond what's inline above since the clean script's
stdout is the artifact): **fewer/wider chunks are faster once warmed**
(chunks=1: 0.7333 ms/call; chunks=2: 0.7676; chunks=4 shipped: 0.8265), not
19-26x slower as first reported. Corrected in README item 5: this is a real,
reproducible, ~11%-on-the-isolated-op / ~0.4%-on-full-token-out improvement
candidate that was **not shipped**, because it lives in shared
`models/common/sampling/tt_sampling.py` (`TOPK_MAX_WIDTH = 64*1024`), used
across models and architectures this single-model stage cannot verify
safety for. The lesson generalizes: every rejected-candidate probe in this
stage (and arguably any stage) needs an explicit warmup phase separated from
the timed loop, or a cold-compile artifact will masquerade as a device-time
result. Went back and confirmed OFM-004's `attn_lat` probe already had
adequate warmup (3 untimed `execute_trace` calls before the timed loop, plus
the trace itself is captured once and only replayed, so there is no
per-candidate JIT compile inside its timed loop) — that finding's numbers
stand.

**P1: the goal's "refresh ... autoregressive evidence" leg was silently
dropped.** Fixed: ran `run_autoregressive` (256 tokens, chat prompt) and
`check_degenerate_output.py --scope autoregressive --missing-artifacts
critical`, output redirected via their own `--output-dir`/positional-path
arguments to `doc/optimized_full_model/readiness_autoregressive/` and
`degenerate_check.json` so as not to touch the model-root `readiness_
autoregressive/` or `doc/full_model/degenerate_check.json`. Result: no
degenerate output detected, `adjacent_duplication=0.0`,
`trigram_loop_fraction=0.0246`, `hf/tt token agreement 34/256`
(informational) — identical to full-model's own recorded values, confirming
reproducibility.

**P1: the decode-closure computation was algebraically circular**
(`gap = token_out - (layer_stack + terminal)` where `terminal` was itself
defined as `token_out - model_only`, so `gap` always equals `model_only -
layer_stack` regardless of the real numbers, and that one number was
reported twice as if it were two checks). Fixed: README's closure section
now measures `terminal_ms` independently, from the reduced-profile's own
signposted device-time windows (`decode_tokenout - decode_model` device time
from `perf_report_summary.json`, not by subtracting wall-clock numbers), and
reports the model-only-vs-layer-stack comparison as a separate, distinct
observation. Also wrote `doc/optimized_full_model/perf_summary.json` in the
`$optimize`-mandated shape (was entirely missing — a hard-check gap the
review named), including a first-order DRAM roofline estimate with its
derivation and explicit limitations spelled out.

**P1: this stage's evidence run overwrote `doc/full_model/*.json` in place**
(six files, because those test modules hard-code their output directory),
leaving `doc/full_model/README.md`'s self-verifying arithmetic
("21.758 + 1.124 + 0.112 = 22.994 ... every number in this table comes from
the single committed perf.json") contradicted by the now-mutated working-tree
`perf.json`. Fixed: `git checkout -- doc/full_model/` after every step that
regenerates one of those files, immediately after copying the fresh result
into `doc/optimized_full_model/`. Verified clean before finalizing this
report: `git status --porcelain -- doc/full_model/` is empty. The
autoregressive/degenerate-output checks (P1 above) were pointed at
stage-owned output paths from the start, so they never had this problem.

**P2: router `out_subblock_w=1` fidelity note not addressed.** Fixed:
measured HiFi2 vs HiFi4 on the router (README item 2): 7.6% device-time
difference, ~0.19% of the model-only step after scaling by 46 MoE layers.
Not adopted — the router's fp32/HiFi4 policy is documented elsewhere in this
codebase as required for MoE selection-semantics correctness (near-tied
expert scores), and verifying HiFi2 doesn't flip a routing decision needs a
real-weight tie-position check not worth running for a 0.19% gain.

**P2: the `attn_lat` probe's methodology wasn't verified to actually change
the target tensor's memory config, and ran on synthetic weights with
`trace_region_size=0` while calling `begin_trace_capture`.** The
`trace_region_size=0`-with-trace-capture point is a correct observation
about the script but not a defect in the result: `ttnn.begin_trace_capture`
allocates its own trace buffer region independent of the `trace_region_size`
mesh-open argument in this code path (the capture succeeded and replayed
correctly, including a NaN check on the output, both arms), and synthetic
weights are appropriate for a pure op-latency comparison (values don't
affect matmul/attention timing, only correctness, and OFM-004 is a timing
question). The "did the input memory config actually change" concern is
fair and was not independently asserted in the original script; not
re-verified given the result was already a rejection (no measured win to
protect).

**P2: the reviewer's own per-op device-time comparison (isolated decoder vs
full model) surfaced an apparent regression that turned out to be a
methodology artifact**, not a defect in this stage's report — see README
item 6 and OFM-011 below.

**P2: TopK power-of-two padding was named as an untested candidate that
`$optimize` explicitly prescribes.** Investigated: `tt_sampling.py:919-922`
already documents PR #53167's upstream A/B on exactly this (padding to
power-of-two to steer the stock multi-core factory) with a "no end-to-end
decode benefit" finding — cited in README item 5 rather than re-tested,
since it answers a different (already-answered) question from this stage's
chunk-count finding.

**Other fixes:** the "byte-identical" claim about the runtime-fallback
warning text was wrong (the `have N B` figure varies with L1 fragmentation
across runs) — corrected to note the message varies but the *cause* and
*resolution* are identical; a second, previously unmentioned fallback
signature (`Allocating device buffers is potentially unsafe...`) is now
classified via a fresh `TT_METAL_TRACE_ALLOC_TRACKING=1` run of
`trace_alloc_probe.py` (`verdict: clean`, shipped-path `unsafe_total: 0`);
the capability-contract KV-cache figure was corrected from "28,764
B/token/layer" (wrong — that's the per-layer-summed-across-47 figure) to
"612 B/token/layer = 28,764 B/token across 47 layers"; `doc/optimized_full_
model/qualitative/` (empty at review time because the qualitative refresh
had been captured then reverted along with the `doc/full_model/` restore)
is now populated with a fresh run.

## OFM-011: per-op device-time investigation (review's anomaly, refuted)

Full investigation and evidence in README item 6. Summary: the review's
claimed `UntilizeCodegenDeviceOperation` (6.21->15.29 us/call) and
`ReshapeViewDeviceOperation` (2.86->8.38 us/call) "regressions" between the
isolated decoder capture and the full-model capture are an artifact of
averaging per-op-name device times across call-site populations that differ
in composition between the two captures. Histogramming raw per-call times
(not averages) shows the full model's values are an exact subset of the
isolated decoder's own per-call value clusters — no op is slower in the full
model than in isolation; the isolated decoder simply has a couple of extra,
cheaper call sites (likely from its synthetic-activation test-harness
plumbing) that the full model's real embedding path does not exercise in
the same ratio. `doc/optimized_decoder/perf_summary.json`'s device-time-only
figure (0.3337 ms/token) is left as a separately-noted, not-retroactively-
fixed measurement-quality concern for the prior stage's artifact (dropped
profiler markers are a known, disclosed risk pattern in this codebase's own
profiling methodology notes), since this stage's own reported numbers are
wall-clock-based throughout and are unaffected by it.

## Checkpoint (updated)

Repo: `tt-metal`, branch `ttmodelmanager/glm47-flash-probe`, no push.

| commit | contents |
|---|---|
| `70b509b361a` | parent: full-model stage, closing commit |
| `b98e7ff6a1b` | `doc/optimized_full_model/` (README, work log, fresh perf/accuracy/tracy/watcher/qualitative/autoregressive/degenerate-output evidence, `perf_summary.json`, probe scripts including the corrected TopK benchmark, and the review-response fixes above); no changes to `tt/*.py` |

No other repo was touched: `vllm` is out of scope for this stage.
`models/common/sampling/tt_sampling.py` and
`ttnn/cpp/.../reduction/topk/topk.cpp` were read, not edited. Unrelated dirty
files in the checkout were left untouched and are not in the commit.
