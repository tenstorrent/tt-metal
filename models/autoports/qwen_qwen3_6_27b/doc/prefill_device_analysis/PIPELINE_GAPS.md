# Follow-up experiments and pipeline gaps

Status: full-model follow-up measured; serving integration and CI validation remain
in progress. This log records the work after the first performance
report, including rejected attempts. The objective is to make the bringup
pipeline find and validate these opportunities without a user asking again.

## Current result and how to read this ledger

Final default full64 generator prefill is **119.301ms at S128** and
**1299.242ms at S4096**, with B1 and three warmed samples. The matched original
S128 is2480.102ms: **20.79× faster**, still **1.988× / 2.598× above the
60/500ms requirements**. Both pinned-reference checks score99/100 top1 and
100/100 top5. Full active B32 aligned/ragged lifecycle checks pass on four
layers. The latest completed seed-policy serving run measured median HTTP TTFT
**910.998ms at ISL128/OSL252/C1** and **12812.076ms at ISL4096/OSL252/C8**.
The sampled rank-divergence repair passed574 full-model decode observations
and live shared-suite review, with documented256-token completion limits.
A subsequent device-fill change saves175–184ms in the isolated logits-scatter
operation; its final HTTP rerun is in progress. Required-point CI is pending.
The completed short HTTP point is **15.18× above60ms**; generator speedup is
not an HTTP speedup claim.

The sections below preserve the sequence of hypotheses and evidence. A row
called “selected” in an intermediate section describes that checkpoint;
`native_final_full_validation.json` is the latest complete B1 measurement.

| Additional work | Kept outcome | Pipeline obligation exposed |
| --- | --- | --- |
| Native GDN, flat QKV, direct log-decay, larger outer chunks | Main structural speedup; FP64 state oracle supports the algorithm | Search available dedicated operators, inspect their fast-path contracts, retune surrounding graph |
| Fused convolution and gated RMSNorm, matmul K-block8 | Further measured gains after the first native result | Reprofile after each dominant bottleneck changes; do not stop at the first relative speedup |
| Duplicate SiLU and CCL role repair | Correct intended activation; route prefill policies correctly | Verify actual fused program configs and phase-specific policy consumers |
| Failed traces/scans/packing/CCL/L1/binary fusion | Rejected or superseded with recorded measurements and controls | Preserve failed evidence, test finite outputs/states, investigate disagreements instead of inventing explanations |
| Pinned HF reference and aligned fast-path quality checks | Correct checkpoint identity; full64 quality and traced generation | Tie reference identity and physical-shape branch coverage to the measured default graph |
| Sampler/sharded-logit and native batch-capacity repair | B32 lifecycle and sampling paths restored and tested | Cover active batch, allocated batch, prompt padding, slot lifetime and serving integration independently |
| Live seed/reload contract repair | Host proofs exposed order-dependent seed salting and ignored plugin commands; canonical live smoke3passed/1skipped | Test the exact CI dependency revision, same-seed request order, penalty history, stale-host overlap and recapture |
| Exact prior CI reconstruction and live checks | Same model/plugin/matrix prepared; execution tracked below | Carry immutable runtime identity and requested ISL/OSL/concurrency through CI |

## Minimum automatic acceptance contract

This is the implementation checklist for the pipeline owners; the detailed
ledger below supplies evidence and counterexamples. These gates are proposed,
not implemented by this performance follow-up.

| Gate | Required durable evidence | Fail or reopen when |
| --- | --- | --- |
| Operator discovery | Dedicated-op and same-family candidate inventory, API/shape blockers, repository revision | A primitive dominant recurrence has no completed candidate search, including after a no-fusion starting stage |
| Measured graph identity | Model/checkpoint/tokenizer revision, code SHA, selected config, actual branch counters and physical shapes | Correctness and timing use different default branches, checkpoints or dependency revisions |
| Graph and movement audit | Device-op totals, kernels versus dispatch gaps, fused activation contracts, casts/layouts, CCL input/output shapes and payload dtypes | Source operation count or nominal precision policy substitutes for observed execution |
| Hypothesis loop | Before/after matched timings, predicted whole-model benefit, output/state quality, rejected candidates and adjudication | A relative speedup closes an unmet requirement; explaining the gap is required but does not satisfy the target |
| Correctness beyond logits | Real-input high-precision state oracle, finite intermediate/state checks, follow-on decode | Incumbent disagreement is treated as candidate failure without adjudication, or finite final logits hide invalid intermediate tensors |
| Serving lifetime | Active and allocated batch, aligned/ragged prompt branches, nonzero slots, reset/remap, cache ownership and trace recapture | A B1 layer probe is the sole evidence for a B32 serving implementation |
| Sampling feedback | Full selected graph, actual per-rank feedback token IDs, seeded/unseeded cases, exact plugin reload contract, reviewed shared-suite text | HTTP200, identical seed buffers, a small-stack pass or coherent greedy output is treated as sampled-quality acceptance |
| Real TTFT | Actual serving benchmark with full requested OSL, request completion/token counts, setup versus first-response timing | Generator prefill or output1 is reported as the requested multi-token HTTP TTFT |
| Required CI matrix | Immutable metal/plugin/inference-server refs, requested ISL/OSL/concurrency JSON, dispatch URL and executed matrix | Generic sweeps replace requested points, dependency refs drift, or queued CI is called passed |

For this model, the most consequential misses were available native GDN
discovery, surrounding graph retuning after adopting it, failure to benchmark
the final serving route, and weak sampled-quality validation. Some integration
defects arose after upstream dependency changes; the ledger distinguishes
those from omissions in the original bringup.

## What the first pass delivered, and what it did not

The first pass measured a full 64-layer S128 prefill A/B (2537 → 999 ms),
reduced S4096 A/B, a ragged nonzero-slot control, device profiles and multiple
graph/config candidates. It did **not** integrate its trace prototype into
the serving implementation or dispatch the required-point CI benchmark. The
runtime would therefore still benchmark its old prefill path. It also stopped
with a performance result still 16.6× over the S128 requirement.

## Additional work ledger

| Opportunity / action | Why the first pass missed it | Hypothesis / required verification | Current outcome |
| --- | --- | --- | --- |
| Native `ttnn.transformer.chunk_gated_delta_rule` and existing Blackhole Qwen GDN integration | Discovery focused on KDA affine scan and the current autoport implementation; did not complete a model-family-wide fused-op search | Replace the sequential recurrence with its dedicated implementation; check exact shapes, gate semantics, masks, entry/exit state, real-weight output/state error and latency | Selected native adapter integrated in model/generator; full64 measurements and fresh HF checks passed (details below) |
| Explicit ownership for recurrence tracing | Experiment used global post-op retention/deallocation hooks; a passing probe was not serving integration | Own all Python-visible scratch explicitly, create before decode trace capture, share a single S32 trace across layers, preserve it across slot resets, release at teardown | Measured 20.854 ms/layer at S128 with exact checked values; superseded by faster native GDN; archived in `owned_trace_experiment.py`, not a runtime mode |
| Required-point CI reproduction | Full-model prefill was treated as the final validation boundary | Match prior workflow, inference-server requirement-only branch, runner/device and model; upload validated implementation before dispatch | Prior run 34479389468 and workflow inputs recovered; dispatch pending readiness |

## Pipeline changes to derive from this work

These are proposed process changes, not claims that the pipeline has already
been modified:

1. Search both the operator library and other implementations of the same model
   family before authoring a primitive recurrence or choosing a generic scan.
   Record dedicated-op candidates and concrete contract blockers.
2. Require a cumulative serving-path result. A fast monkeypatched decoder probe
   cannot complete a stage while generator/vLLM still selects the old graph.
3. Reconcile the best full-model result with the requirement. If the remaining
   gap is an order of magnitude, explicitly revisit algorithm and operator
   selection rather than stopping after the first large relative improvement.
4. Separate output PCC from state correctness. Validate recurrence and cache
   states with real inputs, a higher-precision oracle and follow-on decode.
5. Use matched baseline/candidate measurements and actual device kernels/gaps;
   never infer kernel time by dividing TTFT by source-op count.
6. Treat trace ownership, memory capacity, reset/reuse and decode interleaving
   as integration gates, not follow-ups after a benchmark claim.
7. Persist exact CI matrix selection and dispatch inputs. Check the run log
   confirms the requested ISL/OSL/BS set instead of the generic sweep.

Every additional candidate will be appended with the measured outcome, failed
hypotheses and evidence paths before the final CI handoff.

## Evidence added during the follow-up

### Discovery and provenance

The native operator was added by `045f77046d6` on 2026-07-17. It is an
ancestor of the carried-forward base `d58cb341c70`, so this was not an operator
that appeared only after the bringup. The autoport's initial commit
`f8e7461e725` (author date 2026-08-13) explicitly says it copies the
**advchal-v3 nofuse-noadvise** decoder cell, then continues from multichip through
release. The copied optimized-decoder README explicitly says linear prefill
reuses the functional mixer. The same README gives a long topology/config
ledger, but does not inventory the dedicated GDN operator.

This provenance matters: a no-fusion experiment can intentionally omit a fused
stage. The failure is carrying that starting graph into the full-model product
without reopening the fusion/algorithm decision, and later explaining its cost
as a hardware floor. The current graph-fusing skill already mandates dedicated
operator and family-implementation discovery. The proposed improvement is an
evidence/acceptance gate, not another generic paragraph saying to optimize.

The September first pass also missed that operator even though it read the
current guidance. Its general affine-scan experiments were useful but were
not a complete dedicated-op search. This log includes that investigation miss,
not only historical pipeline failures.

### Quantitative chain so far

| Step, reduced real-weight linear layer | S128 ms | S4096 ms | What changed |
| --- | ---: | ---: | --- |
| Original | 51.119 | 1670.163 | Many primitive recurrence ops, outer chunk32 |
| Protected trace prototype | 20.689 | 573.517 | Remove eager dispatch gaps |
| Explicitly owned runtime trace | 20.854 | not measured | Same arithmetic, no global hooks; exact checked S128 values |
| Native GDN, outer128/512 | 5.475 | 69.316 | Dedicated delta-rule algorithm plus fewer outer boundaries |
| Native flat QKV and log decay | 4.076 | 39.696 | In-kernel head mapping/norm, remove repeat/reshape round trips and exp→log |
| Also fused causal convolution + SiLU | 3.402 | pending | Reuse dedicated convolution with existing TP tap weights |

The final rows are not full-model or serving measurements. First flat-path
measurements overlapped an eight-thread CPU HF job: full attention inflated
from 2.19 to 6.60 ms, invalidating timing comparisons. The uncontended S128
rerun and S4096 result with its unchanged full-attention control replace them;
contended JSONs remain explicitly labeled.

### A better oracle changed the acceptance decision

Original-vs-native S4096 recurrent-cache PCC near 0.969 initially looked like a
failed rewrite. Both implementations were then compared against FP64 over the
**same captured real preprocessed inputs** across all4096 tokens. Original
final-state relative L2 was 26.10%; native was 2.03%. An implementation can be
more accurate while differing from the incumbent. The oracle does not itself
prove full-model quality; the fresh full-model HF checks below supply separate
evidence for the selected implementation.

### Checkpoint-specific validation

The checked-in `doc/full_model/readiness_aime24_chat.refpt` metadata names
Qwen3.6 and its revision. Reusing that file under the Qwen3.8 environment would
silently measure the wrong teacher. `make_reference.py` generated a fresh
pinned Qwen3.8 BF16 CPU AIME24 chat reference: 203 prompt tokens, 100 greedy
continuation tokens, top100. Its metadata/hash/text are retained in
`artifacts/qwen38_reference_metadata.json`; the tensor artifact stays under
`/tmp/qwen38_prefill_followup_aime100.refpt`.

The full-model A/B tested the double-SiLU gate and the locally verified
single-SiLU repair separately. Results are recorded below, separating the
recurrence/graph performance change from the MLP correctness repair.

## Concrete automatic gates suggested by the evidence

| Gate | Required artifact / assertion | Miss caught |
| --- | --- | --- |
| Operator discovery at decoder, multichip and full-model entry | Native operator + same-family implementation inventory; each candidate measured or rejected with an exact contract blocker; record source commit availability | Missing native GDN and fused convolution |
| Phase-specific runtime selection | Loaded graph/config fingerprint from the actual generator and serving entry points, including overrides; verify selected prefill operator name in a reduced profile | Single-chip optimization shadowed by TP override; experimental probe never reaching CI |
| Requirement reconciliation | Same-harness warmed medians with multiple samples; absolute target gap and remaining cost breakdown; keep unmet requirements open and require another bottleneck investigation or an evidence-backed limitation | Stopping at 2.54× while16.6× over target; ranking cold single samples |
| State-aware equivalence | Real prefill→decode state comparisons; on disagreement, identical-input FP64 oracle before rejecting an algorithm; later full-model HF check | Rejecting a more accurate native state because original cache differed |
| Checkpoint identity | Assert reference model/revision/tokenizer/prompt format matches runtime, including copied stage artifacts | Using a Qwen3.6 teacher for Qwen3.8 |
| Serving geometry/lifetime | Tests at allocated batch32, inactive slots, nonzero active slot, ragged lengths, resets/remaps, changed page tables and subsequent decode | B2 tests missing fused B32 state layout and serving lifetimes |
| Required CI matrix | Machine-readable exact (ISL,OSL,concurrency) list, actual selected refs, log confirmation and observed matrix count | Shared sweep omitting OSL252 and running unrequested concurrency32 |

The current stage07 shell gate only invokes the degenerate-output checker. That
is useful but cannot enforce native-op discovery, perf evidence quality, or a
remaining target gap. The goal text is richer than that gate. Strengthening the
structured evidence and independently checking it is more actionable than
adding another unchecked optimization checklist. These proposed gates should
allow precise unsupported/capability outcomes; they must not encourage invented
measurements merely to satisfy a required field.


## Intermediate native full-model checkpoint: measured results and selected code

These are full **64-layer generator** measurements, not a layer-count
extrapolation. `validate_runtime.py` loads the same pinned Qwen3.8 weights,
resets state between inputs, synchronizes around the complete prefill, discards
the first iteration and reports the median of three warmed samples. There is
no per-layer synchronization in this harness. This is prefill latency, not an
HTTP serving TTFT or a CI result.

| Full model, B1 | S128 median ms | S4096 median ms |
| --- | ---: | ---: |
| Original eager recurrence, outer32, double SiLU | 2480.102 | not rerun in this matching harness |
| Native flat GDN + fused conv, outer512, double SiLU | 134.478 | 1852.063 |
| Selected native graph with single SiLU | 136.498 | 1834.551 |
| Requirement | 60 | 500 |

Evidence: `artifacts/eager_full_matching_baseline.json` and
`artifacts/native_full_validation.json`. The selected S128 improvement is
**18.17×** against the matched original. Its remaining target gaps are
**2.275× at S128** and **3.669× at S4096**. There is no matched full-model S4096
speedup claim here. Removing the extra SiLU did not give a demonstrated S128
speedup (the measured median increased 1.5%); its acceptance is based on the
intended math and improved model accuracy. It reduced the observed S4096 median
by about 0.95%, but these short sample sets do not establish a small standalone
performance effect.

On the fresh 100-token AIME24 chat reference, native GDN with double SiLU had
91/100 prefill top1 and 94/100 teacher-forced top1. Single SiLU raised **both to
99/100**, with 100/100 top5 and top100. Here top5 means the TT argmax belongs to
the HF top5 set. Both variants generated 100 coherent tokens of the math setup;
the budget truncates the reasoning, so this is not a solved-answer accuracy
claim. The selected generation used 99 decode trace replays with zero host
updates to token/position/page-table buffers. This one prompt is a regression
check, not an IFEval/GPQA or full release qualification.

### Initial native implementation decisions

| Change / experiment | Selection and reason | What an autonomous pipeline should have done |
| --- | --- | --- |
| Dedicated native GDN instead of a general affine scan | Kept; direct delta-rule operator avoids materializing generic transition matrices and removes per-token primitive dispatch | Inventory the native transformer operators and same-family Blackhole code before selecting the recurrence algorithm |
| Flat rank-3 Q/K/V into native GDN | Kept; native head mapping and L2 normalization replace explicit replication, normalization and permutations | Read shape-dependent operator fast paths, then remove now-redundant producers |
| Direct FP32 log-decay and beta | Kept; avoid computing exp only to log it again in the native adapter; mask inactive beta/log-decay to zero | Match the consumer's mathematical input contract and dtype, rather than mechanically preserving the old graph |
| Outer prefill chunks128/512 versus32 | Native outer512 selected; internal native chunk remains32; fewer projections, cache boundaries and collectives | Distinguish algorithm-internal tile size from generator chunk size and retune the latter after fusion |
| Fused causal conv + SiLU | Kept for supported flat-native B1 path; larger batches retain composite conv | Search adjacent operations again after replacing the dominant recurrence; validate convolution history and slot narrowing |
| Model-owned constant tiles | Kept and shared across layers; release at generator teardown; no experimental allocation hooks | Include allocation lifetime and ownership in production integration, not only the benchmark closure |
| Explicit scratch-owning trace | Correct S128 experiment, but rejected as selected runtime because native GDN was faster; class archived | Compare the integrated alternatives and remove superseded runtime modes |
| Suppress second prefill MLP SiLU | Kept; program config already fuses the activation; full-model top1 improved | Audit actual fused program configs against surrounding Python operations, and check the intended formula |
| Recognize `mlp_down_prefill` in CCL role selection | Kept; otherwise prefill selects token-mixer policy. Shipped policies are both BF16, so no default payload or speed change is claimed | Verify policy names reach each phase-specific call site, not just the JSON |
| Native/eager runtime switch and default wiring | Native selected in model; explicit eager remains a diagnostic fallback | Require production generator selection, not a monkeypatched probe, before claiming a deployable optimization |
| Fresh pinned HF reference | Rebuilt because carried reference named Qwen3.6; recorded revision/hash/prompt/output | Validate artifact identity before trusting copied stage passes |
| CPU-contention control | Rejected contaminated flat-path timing and repeated without HF CPU generation | Use unchanged control-layer latency and process isolation to detect invalid performance comparisons |
| Matching complete-model baseline | Added same-harness original S128 after reduced experiments | Require an end-to-end measured result; do not mix synchronized layer probes with normal generator timing |
| B32 slot lifecycle and serving checks | Running after default integration; results must be appended before CI readiness | Exercise actual allocated batch, ragged prompt, narrowed slot, reset and subsequent decode before dispatch |
| Prior CI reconstruction | Recovered exact workflow/ref/matrix; dispatch still pending | Carry the requirement matrix as a machine-readable artifact from requirements through benchmark execution |

### Hypotheses that failed or needed reinterpretation

The first trace result did not satisfy the target because it removed dispatch
gaps but retained the expensive recurrence graph. General KDA/Hillis scan
improvements failed the real-state correctness criteria; raising precision in
the generic scan did not fix the modeled recurrence discrepancy. Initial native
state comparisons also appeared to fail, but the FP64 control showed that the
incumbent was less accurate. These are distinct failure mechanisms and must
not be collapsed into “fusion failed.”

Packed projections and BF8 CCLs did not improve the original recurrence-heavy
short-prefill measurement. That does **not** reject them for the new graph:
replacing its dominant cost changes the priority of the remaining projections,
collectives and layout conversions. The next device profile must establish
those costs before assigning a numerical gain. No unsupported projection of
another 2× is included in this report.

### Exact required-point CI contract recovered

Prior run: https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/34479389468.
Workflow: `manual-tt-shield-dispatch.yml`; model `Qwen/Qwen3.8-27B`, runner
`bh-qb-ge`, device `p300x2`, workflow `benchmarks`. The inference-server branch
`mvasiljevic/qwen38-bench-required` resolves to
`0fcbc763bfcb95ba6e87eb70aa4360d23c0c2c96`; the prior vLLM SHA is
`c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`.

All requested points use **OSL252**:

- Concurrency1: ISL128, 1024, 4096, 16384, 32768, 65536, 131072, 261892.
- Concurrency8: ISL4096, 32768, 131072.
- Concurrency16: ISL4096, 32768.

There are exactly **13 points**, with final ISL262144−252 preserving the
advertised total context. The prior run passed nine points and killed four;
recovering its inputs is not evidence that the new implementation passes them.
A dispatch link and actual CI outcome will be recorded separately.


## Integration/review follow-up after the first full-model result

1. **Production B32 lifecycle:** the default path passed
   `tests/slot_lifecycle_b32.py --num-layers 4 --prompt-tokens 65`. Fused decode
   was active in all three linear layers; reset slots3/17 had zero uncleared
   or disturbed peers; remap had zero misplaced states; narrowed slot9 changed
   six state tensors and disturbed no other slots. Artifact:
   `artifacts/native_slot_lifecycle_b32.json`.
2. **Actual B32 adapter exposed a separate blocker:** prefill completed, then
   shared sampling attempted a 9024512-byte L1 buffer against a1572864-byte
   limit. Source investigation found a lost TP4 local-topk branch after rebase,
   plus an undefined split width in its retained body. This was not caused by
   native recurrence or a missing greedy flag. The shared sampler repair and
   its boundary-ID and adapter reruns are tracked in `SAMPLER_FOLLOWUP.md`.
   It demonstrates why model-only benchmarks cannot certify the serving path.
3. **Fresh independent review caught a branch-coverage gap:** the AIME prompt
   has203 tokens; earlier full-model HF checks exercised the rank4 native
   adapter, whereas tile-aligned performance inputs exercised flat QKV and
   fused convolution. The reduced S33 probe itself supplied physical128; the
   generator did not pad it. A new full-model run explicitly pads physical
   inputs while preserving logical lengths and records flat/rank4 call counts.
   This is a required coverage correction, not a performance regression.
4. **Context evidence separated by checkpoint:** historical Qwen3.6 capacity
   results remain labeled with their original identity. A Qwen3.8 native
   follow-up section in `doc/context_contract.json` records outer512/internal32,
   six shared FP32 constant tiles (24576 bytes/device), exact tested lengths,
   and pending long-context capacity validation. The served context target
   remains262144; old capacity numbers are not relabeled as new measurements.
5. **Reprofile after removing the bottleneck:** native S128 linear layer has
   **91 device operations versus2218 originally**, with merged kernel time
   **2.020 ms** and inter-op gaps **2.180 ms** in a4.231ms synchronized window.
   Full attention has61 ops,1.233ms kernels and1.594ms gaps. These profiled
   windows are slower than unprofiled medians and cannot replace the full-model
   result. Both profiles use the selected BFP4/LoFi projection policy.

The native linear kernel breakdown now puts matmuls first (0.642ms), followed
by device-backed reshapes (0.435ms), norms (0.155ms), and collectives
(0.154ms). Native GDN prep+scan is only0.110ms and fused conv0.033ms. This is a
very different optimization problem from the original recurrence. Typecasts
are five operations totaling0.016ms kernel time but0.083ms adjacent gaps;
untilize/tilize and permutation/transpose chains carry further launch and
movement costs. Raw operation rows, dtypes, program configs and advice are in
`artifacts/native_linear_perf.csv` and `native_full_perf.csv`, with grouped
numbers in the matching summary JSONs. Missing dedicated-op roofline labels
are tool classification limitations, not evidence that those operations ran
on the host.

New automatic gate implied by the review: **tie correctness coverage to the
fast-path dispatch predicate**, including physical shape, batch, dtype and
environment. Matching model IDs and logical prompt length alone is not enough.
After a rebase, independently rerun downstream sampling and slot tests even
when the model implementation is unchanged.


### Evidence locations for the pipeline diagnosis

- Initial copied-stage provenance: `git show f8e7461e725` and the
  `nofuse-noadvise/doc/optimized_decoder/README.md` under
  `/home/mvasiljevic/agentic-research/shard-advisor-experiments/03-advisor-stage-v2/autoports/qwen_qwen3_6_27b/`.
- Native operator availability: `git show 045f77046d6` and
  `git merge-base --is-ancestor 045f77046d6 d58cb341c70` (exit0).
- Existing fusion requirement: `.agents/skills/graph-fusing/SKILL.md`.
- Rich optimization contract versus narrow executable gate:
  `.agents/prompts/model_bringup_multigoal/07-optimized-full-model.txt` and
  `07-optimized-full-model.check.sh`. The latter only invokes
  `check_degenerate_output.py --scope autoregressive`.

The native prefill operator uses its own default **HiFi4 with FP32
accumulation**; the model policy's `linear_recurrent_fidelity=HiFi2` describes
the existing decode recurrence and does not configure this native prefill op.
Projection dtypes/fidelity are independently verified as BFP4/LoFi in the
profile. A future runtime-config artifact should report phase-specific
operator config instead of implying one recurrent policy controls both.


### Multi-request output assembly and sampler repair

The B32 adapter failure had multiple independent causes. With
`QWEN36_PREFILL_PER_REQUEST=1`, the generator gathered each full-vocabulary
terminal row to the CPU, trimmed its padding, assembled the rows, and uploaded
that full vocabulary **replicated on every rank**. The sampler expects a
padded local TP vocabulary shard, so its tensor contract was wrong even if the
sampler's chunk branch were restored. The new assembly takes each selected
row on device and concatenates device rows plus inactive zeros; it preserves
TP vocabulary shards and removes the host gather/re-upload. A caller that
explicitly requests host logits reads only the assembled final result.

The sampler independently needed its explicit local-chunk branch and split
width restored, tile-layout padding to bound L1 buffering, and an obsolete
`dtype` keyword removed from its index gather call. The gather derives dtype
from its input, retaining uint16 indices. Six focused host tests and the
hardware split-topk trace probe pass; the latter selects token IDs0,32767,
32768 and248063 on every rank. B32 adapter and live serving are the next
checks. This is not a claimed TTFT gain until measured; it repairs correctness,
capacity and graph continuity at the sampling boundary.

The first padded full64 run completed HF checks (prefill100/100 top1,
teacher99/100; both100/100 top5), then failed generation at that obsolete
sampler keyword. Its partial result is explicitly marked failed in
`artifacts/native_flat_full_partial.json`; it is not represented as a clean
end-to-end pass. The validation script now saves partial accuracy evidence
before entering generation so a downstream failure does not erase it.


### Residual graph/config sweep registered after native profiling

The native flat output is immediately reshaped from head-major `[B*12,S,128]`
to `[B*12,S,1,128]`, then the tail reshapes it to `[B,12,S,128]`. In tiled
storage the singleton penultimate dimension can create padding and movement.
Hypothesis: pass the native output directly to the tail and remove those
redundant conversions; approximately0.1–0.3ms/layer is plausible from the
profiled reshape cost, but this requires exact output/cache checks.

The selected projection program uses K-block limit4. Hypothesis: limits8/16
reduce loop overhead and improve K reuse under the **same BFP4/LoFi policy**,
potentially saving10–25% of the0.642ms projection kernel budget (about0.06–
0.16ms/layer), not10–25% of full-model TTFT. The helper chooses the largest
legal divisor at or below the limit. These reduced-stack experiments run after
all other TT work closes and keep unchanged full attention as a timing control.
The original single-chip packed-MLP L1 blocker cannot by itself reject these
smaller TP-local projection geometries.

Commands, return codes and individual measurements are retained by
`run_native_residual_sweep.py` under `artifacts/native_residual_*.json`.
Selected baseline captures stay in `/tmp/qwen_native_selected.pt`. A faster
result is not promoted without tensor/state comparison and final full-model
and serving revalidation.


The residual sweep completed with these warmed S128 layer medians:

| Candidate | Linear ms | Full-attention ms | Decision |
| --- | ---: | ---: | --- |
| Selected native baseline, limit4 | 3.361 | 2.293 | Matched control |
| Remove rank expansion only | 3.547 | 2.271 | Rejected: checked outputs/caches exact, but no measured speedup |
| Remove rank expansion + limit8 | 3.193 | 2.125 | Faster; isolate config before promotion |
| Remove rank expansion + limit16 | 3.384 | 2.123 | No improvement over limit8 |
| Limit8 alone | 3.296 | 2.138 | Full-model check running |

All config variants retained terminal-logit top1; limit8 alone preserves high
PCC, while changing accumulation order is not bit-exact. The predicted reshape
saving did not materialize: source-level reshape cancellation did not establish
which device-backed reshape rows dominated the profile, and wall time got
worse. Limit8's measured short-prefill benefit is modest, so the next full64
S128/S4096 measurement must determine whether it is worth selecting. A config
win is not multiplied directly by all64 layers and advertised as TTFT.

The repaired full sampler/adapter hardware results are now in
`artifacts/sampler_followup_validation.json`: split boundary IDs pass on every
rank; B32 prefill, decode, stale-input replay and slot remap pass. Replay
count advanced1→2 with token, position, page-table and readback refresh counters
unchanged at zero. Those reduced-layer outputs are used only for lifecycle
validation, not text quality.


### Full-model confirmation of limit8 and the flat branch

`artifacts/native_block8_full_validation.json` completed successfully after
sampler repairs. The script counted **1872 flat native calls and zero rank4
calls**. Quality prefill had logical303/physical320 tokens; teacher forcing and
autoregressive generation had logical203/physical224. Thus the HF checks cover
the same flat/fused dispatch path as the aligned benchmark, including masks.

Full64 warmed medians were **131.164ms S128** and **1787.089ms S4096**. Against
the preceding native limit4 result (136.498/1834.551), the measured reductions
are3.91% and2.59%. Compared with the matching original2480.102ms S128 baseline,
this is **18.91×**, still **2.186× above60ms**. S4096 remains **3.574× above500ms**.
Limit8 prefill top1 was99/100 and teacher top1 was98/100; both top5/top100 were
100/100. This small top1 change is disclosed, not described as bit-exact or as
an accuracy improvement. It satisfies the existing top5 acceptance threshold.

Autoregressive100-token output was coherent and matched the HF control's math
setup; both are reasoning-budget excerpts. The run recorded99 decode trace
replays, no host token/position/page updates, and the expected100 readbacks for
the high-level Python token-list API. This closes the earlier flat-branch full
quality/generation gap; live serving is still a separate gate.

A further legal-divisor check with K-block limit20 is recorded separately
(the helper may choose non-power-of-two divisors, e.g.17 for K/32=136). A new
source investigation is checking dedicated head concatenation at the native
tail: removing the standalone rank expansion failed, but does not reject
replacing the actual permute-plus-head-flatten producer/consumer sequence.


### Tail fusion attempts and a new numeric failure

Limit20 measured3.311ms linear and2.097ms full attention, versus limit8's
3.296/2.138ms. Weighting the two layer types48/16 for prioritization makes
these effectively tied; this is not a measured full-model result. Limit8 has
the full-model accuracy/performance evidence and remains the preferred config.

Dedicated `nlp_concat_heads` replaced the actual permute-plus-head-flatten
sequence and was **bit-exact** for all checked outputs/caches, measuring
3.277ms linear versus3.361ms matched native baseline. Combining it with
`multiply(..., input_tensor_b_activations=[SILU])` measured3.218ms but produced
**non-finite layer0/3 values**. Terminal logits were finite but PCC fell to
0.9666, and full-attention KV PCC fell near0.55. Conv/recurrent state remained
exact, locating the regression downstream of GDN. The fused activation is not
accepted; a minimal operator repro/source investigation is tracked in
`TAIL_FUSION_FOLLOWUP.md`.

The experiment process had exited0 because the probe originally collected
comparisons without enforcing them. The probe now records all checked tensor
finiteness and exits with failure on a non-finite layer/logit/cache tensor,
after saving its evidence. This is another concrete automation miss: **a
successful subprocess and finite final logits do not imply a passing graph**.

A dedicated sigmoid-gated RMSNorm candidate uses the native head-major output
and flat gate, then multiplies by the flat gate again to recover SiLU's
`z*sigmoid(z)` factor. It removes the normalization/head-shuffle chain, but
changes rounding order. Initial S128 output/cache checks are finite with
layer0 PCC0.999991 and exact conv/recurrent state; timing was3.067ms linear
with an elevated2.598ms full-attention control. This timing needs an
uncontended repeat and long/full-model validation before selection. Source
contracts and candidate code are retained in `tail_followup.py` and
`TAIL_FUSION_FOLLOWUP.md`.


### Dependency identity before CI

The local installed plugin checkout is `d7a6008b03c7afba001444f2d7a4cfde9ef6d498`;
the previous CI plugin is `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`, seven
commits ahead, including centralized async-decode reload planning and changed
chunked-prefill capability handling. These are material runtime differences.
The exact CI commit was fetched and checked out in the isolated
`/tmp/qwen-ci-vllm-plugin` worktree. Live checks use that checkout's `src` on
PYTHONPATH, without installing dependencies or modifying the user's plugin
checkout. The standalone generator and direct-adapter tests do not establish
compatibility with a different scheduler/plugin; CI readiness must use the
actual selected plugin ref.

This adds an automatic gate: include model, runtime, plugin and benchmark
harness refs in the evidence manifest, and reject “same CI” claims based only
on matching the model branch name. The benchmark requirements branch was
rechecked and still resolves to the recorded exact SHA.


### Dedicated tail fusion: long-context gain and full-model prediction

With limit8, the clean S128 gated-norm rerun measured **3.048ms linear** and
2.118ms full attention, versus3.296/2.138ms without it. At S4096, a matched
baseline/candidate pair measured **32.548→22.251ms linear**, with essentially
identical15.644/15.640ms full attention. The recurrent and convolution states
were exact; layer0 output PCC was0.999993 and terminal-logit PCC0.999730. This
is a **31.64% long-prefill linear-layer reduction**, not yet a full-model claim.

Before full-model measurement, the hypothesis is approximately **131→119ms
S128** and **1787→1293ms S4096**, using the changed linear-layer cost across48
layers. Per-layer synchronization and complete-stack overlap can make this
estimate fail, so the full-model medians must replace it. Even matching the
hypothesis leaves the60/500ms requirements unmet.

The selected flat-native implementation now calls the dedicated norm directly
on native head-major output, consumes flat z, multiplies by z for the SiLU
factor, and projects the resulting flat token-major tensor. The unaligned
rank4 native fallback retains its previous composite tail; it does not call a
fused op requiring tile-aligned sequence. This avoids turning an internal
kernel shape restriction into a public prompt-length restriction. Only the
measured native full-model path receives K-block limit8; standalone decoders
without that model integration retain their policy limit.

A final matched-policy probe tests the existing prefill L1-input option in the
new graph, because the profile's DRAM-input recommendation must be evaluated
after recurrence fusion. It includes copy overhead and the same BFP4/LoFi
matmul config. Results are recorded separately; L1 movement is not selected
merely because a kernel's modeled bandwidth looks low.

### Binary-SiLU rejection refined by controls

The minimal BF16 range/shape test, a padded-head-permutation test, and a
synchronized test of actual model inputs were all finite and bit-exact against
standalone SiLU. Actual norm input ranged−0.875…10.1875 and gate input
−16.25…15.625. Unsupported activation and simple range/rounding explanations
are therefore refuted. The unsynchronized whole-graph failure remains
unresolved; ordering/lifetime interaction is a hypothesis, not an established
root cause. The selected dedicated-norm graph does not use that failing binary
activation fusion. Controls and commands are in `TAIL_FUSION_FOLLOWUP.md`.


The L1-input candidate measured3.155ms linear and2.161ms full attention,
versus3.048/2.118ms for the matching gated-norm/limit8 DRAM-input graph. It
remained finite and retained terminal top1, but the extra copies did not pay
for themselves at S128. The selected runtime keeps DRAM inputs. This rejects
that measured interleaved-L1 candidate, not every possible persistent-sharded
L1 or fractured-residual design.

The final runtime validation now runs **without experimental program/norm
patches**. Its artifact records the invocation, actual model config summary,
physical/logical prompt shapes, native branch counts, warmed samples,
checkpoint-specific HF scores and generated text. This turns the pipeline's
“selected configuration reaches the generator” recommendation into concrete
follow-up evidence for this model, rather than a claim based on constructors.


## Final default full-model measurement and prediction check

`artifacts/native_final_full_validation.json` completed with no experimental
program/tail override. Actual warmed medians: **119.301ms S128** and
**1299.242ms S4096**. The predicted119/1293ms were matched to about0.3/6.2ms,
respectively. The long-prompt deviation is under0.5%; a representative-layer
delta is not an exact complete-stack timing model, and the three warmed
samples themselves vary by a few milliseconds. There is no large unexplained
failure to match the tail-fusion hypothesis.

Compared with the matched original, S128 is **20.79× faster**.
The absolute gaps remain **1.988× / 2.598×** versus60/500ms. Prefill and
teacher forcing both measured99/100 top1 and100/100 top5/top100. The final
100-token generation was coherent against the same-budget pinned HF control,
with99 trace replays and no host token/position/page-table updates.

### Full active batch differs from allocated batch

The additional `slot_lifecycle_b32.py --prompt-tokens 128` run deliberately
omitted `QWEN36_PREFILL_PER_REQUEST=1`, so all32 rows reached native GDN in one
call. It failed with `num_heads 384 exceeds compute cores 110`. Earlier B32
checks validated allocation, state layout, ragged prompts and per-request
narrowing; they did not prove this fully active batch shape. The native phased
scan maps at least one core to each batch/head pair, creating the limit.

The repair must tile the batch on device within the native adapter, preserve
head-major output/state ordering, and concatenate results without host
readbacks. Disabling the test or forcing per-request mode would hide the
public full-batch regression. The B1 graph must remain unchanged; both aligned
and ragged full-B32 calls must be rerun after the repair. This adds a precise
pipeline gate: **allocated batch, active batch and physical prompt extent are
three separate axes**, not interchangeable labels in a readiness result.

### Shared sampler constraints from independent review

Review also identified configurations outside Qwen's validated local62080
vocabulary: uint16 local index offsets would wrap above65536 entries, and
explicit local chunks on a single sampling device conflicted with its older
split-offset path. The opt-in now rejects those unsupported geometries and
chunks narrower than max_top_k before device uploads. The default single-device
sampler is unchanged. Focused host tests cover these guards; no broader
sampling capability is claimed from Qwen-only measurements.


### Batch-capacity repair verified

The adapter now derives its maximum native batch from the actual compute grid
and local value-head count. On this TP4 mesh, B32 executes device batches
9+9+9+5 and concatenates head-major outputs and final states in slot order.
B1 passes the original tensor objects directly to the same native operation;
there is no host fallback or change to B1 arithmetic.

Both all-active B32 tests passed with per-request mode explicitly disabled:
S128 exercised the flat path and S65 exercised the rank4 adapter. These are
four-layer lifecycle checks, including exact peer preservation, reset, remap
and narrowed-slot updates, not full-stack B32 quality claims. Five host tests
cover batch boundaries and ordering. See
[BATCH_CAPACITY_FOLLOWUP.md](BATCH_CAPACITY_FOLLOWUP.md). The shared sampler's
nine host tests also pass, including its new unsupported-geometry guards.

This was an additional integration defect found after the speedup, not a
performance candidate. A pipeline should exercise the actual maximum active
batch before declaring an optimized replacement compatible with its public
interface, even when serving normally narrows prefills to individual requests.


## Final device profile and remaining opportunities

The final default graph was reprofiled after integration and batch repair;
`artifacts/native_final_profile_summary.json` and the two
`native_final_*_perf.csv` files contain the advice-enabled report. Linear layer0
now has **84 operations**, versus2,218 originally. Its merged-device profile
contains **1.499ms kernel execution + 2.289ms between-operation gaps**; full
attention has61 operations and1.113ms kernels +1.251ms gaps. Profiling and
explicit layer synchronization raise wall time, so these numbers must not be
substituted for the unprofiled full-stack119.301ms result.

The final linear profile's eight matmuls use0.575ms of kernels. Native GDN
prep+scan use0.110ms, fused convolution0.032ms and fused gated norm0.014ms.
Five typecasts use0.015ms of kernels plus0.079ms charged gaps. Two reduce
scatters and two all-gathers use0.147ms of kernels plus0.042ms charged gaps.
Untilize/tilize still account for17 calls,0.127ms kernels and0.699ms gaps;
binary ops account for12 calls,0.121ms kernels and0.549ms gaps. These are the
remaining graph boundaries to investigate, rather than assuming recurrence
fusion alone reaches the absolute target.

Potential next work is a persistent tiled/sharded layout through normalization,
gating and row projections, or a safe whole-layer/static-shape trace that owns
its scratch. Either needs a new memory and serving-lifecycle validation, and
its savings overlap with other graph changes. Merely eliminating all five
casts cannot account for the remaining59ms S128 gap. Changing CCL dtype also
requires counting conversion/layout costs and preserving model quality;
the tested BFP8 CCL and interleaved-L1 candidates did not win. The native
operator names are not yet classified by tt-perf-report's stacked chart;
this report uses their raw CSV names explicitly rather than losing them in
an unclassified bucket. No unmeasured additional speedup is claimed.


### Serving configuration is verified from the loaded runtime

The live runner was started against the isolated CI-plugin revision, with
max_num_seqs32, max_model_len262144 and a200MB trace reservation. Its startup
fingerprint confirms native recurrence, outer512/internal32 and K-block8.
It advertises1,726,400 cache tokens, and the plugin allocates1,728,448 including
lookahead pages. A proposed `QWEN36_MAX_TOKENS_ALL_USERS=525312` environment
setting had no consumer: `generator_vllm.get_max_tokens_all_users` returns the
constant. The ineffective setting was removed from the reproduction script;
no cache capacity reduction was made. This is another reason to record loaded
runtime values instead of copying intended environment variables into results.


## Live sampling failure: retained as a separate gate

The first live server completed startup at the unchanged public context/cache
configuration. Canonical CI-plugin sampling smoke returned **1 failed, 2 passed,
1 skipped**: mixed-parameter requests changed their seeded output after request
order changed; top1-is-greedy and min-p passed; the logprobs case skipped.
The observed mismatch was temperature0.5/seed42. Raw continuation outputs in
that test are functional stress evidence, not a chat-quality verdict.

Logs are preserved in `readiness_vllm/initial_sampling_attempt/`. The shared
runner intentionally stops after a sampling failure, so it had not run its
qualitative or benchmark stages. A separate held server is being used to run
those remaining checks, while a fresh source-only AutoDebug investigates the
seed/slot/trace contract. Skipping past the failed stage does not make it pass.
The optimization is not declared serving-ready on the strength of B1 generator
accuracy or successful server startup alone.


### Seed isolation hypothesis verified on the host

The actual Qwen SamplingArgs did not set `salt_duplicate_seeds`, so the common
sampler enabled its default. An AST-extracted test of the real SeedManager
showed seed42 requestA receiving effective seed275414 or62798 depending on
whether requestA or requestB arrived first. Disabling salting preserves each
request's effective seed across both orders, including the next token counter.
`artifacts/serving_seed_host_adjudication.json` retains the exact result.
Qwen now sets `salt_duplicate_seeds=False`; the common demo default is unchanged.
The original live mixed-request test must still be rerun to establish sufficiency.

The second server startup was deliberately interrupted before any request or
benchmark ran, to apply the verified seed repair and investigate the plugin
reload contract. Its log is retained in `readiness_vllm/interrupted_startup/`.
The runner received SIGINT; its orphan EngineCore was then stopped with SIGTERM.
This was planned process cleanup, not a device hang or a performance result.


### Final CCL payload and dtype audit

For the normal narrowed B1 serving prefill, the linear output and MLP down
projection each reduce a BF16 `[1,1,C,5120]` residual contribution. C is at most
512 in the selected graph (128 at the short benchmark), rather than32 at every
old recurrence boundary. One contribution is1,310,720 bytes per rank at C128
or5,242,880 bytes at C512, before the collective's ring protocol. The
reduce-scatter result contains one quarter of those elements; its all-gather
restores the replicated residual required by the next norm/projection.
These sizes describe tensor payloads, not measured link traffic or bandwidth.

Native GDN's FP32 beta/log-decay and internal state stay local to each TP rank;
they are not sent through these collectives. Persistent recurrent/KV caches
remain BFP8. The CCL role repair makes MLP prefill consume the MLP policy, but
both current CCL policies are BF16. Further removing an all-gather requires
changing the following residual, normalization and projection layout together;
deleting it in isolation would violate the model's replicated-residual contract.


### Decode reload contract repaired and host-tested

The adapter advertised contract v1 while swallowing the newer plugin's
`reload_inputs`, `reload_page_table`, `reload_sampling_params`, and
`reset_sampling_state` arguments. A host invocation of the real method proved
that a requested sampling-history reset became `reset_batch=False`. It also
realigned RNG counters from host positions on every step, although overlapped
decode deliberately permits those positions to lag the device.

The repair consumes the explicit commands, retains active seed slots/counters
on resident replay, handles page-table-only refresh without token reload, and
restores authoritative penalty history **after** trace setup's synthetic
sampling executions. Direct callers that omit v1 commands retain the legacy
selection path. Seven host regressions pass, using the actual AST-loaded
adapter method, SeedManager/hash and Qwen SamplingArgs. The full diagnosis,
source references and exact scope are in
[SERVING_SEED_AUTODEBUG.md](SERVING_SEED_AUTODEBUG.md).

This defect was exposed by testing the exact plugin revision used in CI. The
pipeline needs a tested capability/contract handshake, not just a capability
integer in a class. It must also distinguish a successful greedy path from
seeded sampling and penalty-history correctness. The latest full64 prefill
measurements remain measurements of the same prefill math; these subsequent
repairs concern sampler and serving lifecycle. The final live checks are being
rerun with the repaired adapter before CI readiness is decided.


### Live sampling repair confirmed; provenance matters

The unchanged canonical mixed-request test now passes. The final sampling smoke
reports **3 passed, 1 skipped in30.29s** in `readiness_vllm/sampling_tests.log`.
This confirms the combined seed/reload repairs on the actual plugin and full
model; the two independent host proofs identify their distinct mechanisms.
It does not assign a percentage of the original text mismatch to each defect.
The top-k100 stress request is clipped by the current max-device-top-k32 path,
min-p uses the explicit host compatibility route, and the all-vocabulary
logprobs case skipped. Those details are preserved rather than calling the
smoke a comprehensive device-sampling feature pass.

The old checked-in sampling pass was committed on2026-08-15 (`a96c1c900c8`).
The common salting opt-out was added on2026-08-27 (`1ec5b723981`), while Qwen's
contract-v1 declaration dates to2026-08-14 (`dced79f19f4`). The plugin added
centralized reload planning on2026-09-02 (`cacf1e7`). Thus a historical
pass cannot certify compatibility after dependency changes. These newly
exposed issues should not be retroactively described as operator-discovery
mistakes in the original no-fusion experiment. The automation miss is failure
to revalidate the integrated model at its actual dependency versions.


### Additional unmeasured concurrency opportunity

The required CI specification still uses per-request prefill. Now that native
batch capacity is repaired, comparing genuinely batched B8 prefill with eight
serialized B1 calls is a separate useful experiment: B8 supplies96 local
batch/head pairs to110 cores, whereas B1 supplies12. This could amortize
projections and collectives too, but it changes workspace, convolution path,
MLP chunking and scheduler behavior. The B32 lifecycle test establishes state
handling, not a B8 full-model throughput gain. No such gain or new serving
default is claimed here; retain the prior CI configuration for comparison and
require full-model quality/capacity/serving evidence before selecting batching.


## First full serving measurements: the generator result is insufficient

The actual shared vLLM benchmark completed4/4 requests at ISL128/OSL252/C1,
with every requested output token present. Median HTTP TTFT is **890.998ms**
(mean912.520ms), median TPOT88.758ms and decode11.267tokens/s/user.
The ISL4096/OSL252/C8 burst completed8/8 with all output tokens present;
median TTFT is **13039.632ms**, mean TPOT90.320ms and aggregate output
throughput56.580tokens/s. JSONs and commands are in `readiness_vllm/`.

These are not the119.301/1299.242ms B1 generator measurements. The short-point
HTTP median remains **14.85× above60ms**. The C8 burst includes admission and
serialized per-request prefills and must not be compared to the B1 500ms target
as if it were an isolated prefill. The gap between generator and HTTP TTFT is
an additional integration bottleneck to localize, not measurement noise.
The completed one-output-token versus multi-output-token HTTP control below
localizes about521ms to decode setup/capture before first-token delivery.

### Shared-suite quality did not pass just because requests completed

All12 correctly formatted chat requests completed, but human review found
sampled text degradation despite coherent greedy output. Examples: sampled
story has “glmed with she close the sunlight”; sampled Python changes `seq`
to undefined `fib`; sampled learning explanation drops words (“you give the
with the answers”). These are not dismissed as temperature variation or
as the expected reasoning-budget truncation. The correctly formatted greedy
and prior same-checkpoint suite are controls; a device-vs-host sampling
comparison is running to localize the additional regression.

The unchanged mixed-request smoke passing proves the repaired tested seed/order
contract; it does not prove general unseeded sampled text quality. The pipeline
must actually read both shared-suite outputs, rather than equating valid HTTP
responses, finite logits or a passing smoke subprocess with coherent generation.


### HTTP controls localize two further problems

The six same-prompt quality controls use temperature0.7, top-p0.9, top-k20 and
160tokens. Both unseeded device completions are malformed; both seed42 device
completions are coherent and byte-identical; both seed42 host completions are
coherent and byte-identical. The host control requests logprobs/top_logprobs1,
which makes this TP4 plugin use host sampling without changing the requested
distribution. Different host/device text is expected from different RNGs.
`artifacts/serving_quality_controls.json` preserves every request and response.
Explicit seeds also switch the common sampler from its internal trace to eager
execution, so this control does not identify RNG alone as the cause. A2x2
seeded/unseeded and traced/eager device test will check per-rank token equality.

The matched streaming TTFT control uses the same128 synthetic token IDs, no
prefix cache, one discarded warmup and three samples for each output limit.
Median first-token times: **330.309ms at output1**, **850.969ms at output2**,
**868.426ms at output8**. Asking for a second token adds about521ms before
the first token is delivered. This supports a decode-setup/capture delay in
serving first-token delivery. The output1 latency still exceeds the119ms B1
model prefill, so eliminating recapture alone cannot meet60ms. Exact requests,
stream chunks and timings are in `artifacts/serving_ttft_controls.json`.
A source audit is checking safe trace reuse rather than assuming the prototype
retention hooks from the original recurrence experiment are production-safe.


### Warmup-history branch verified on hardware; rank hypothesis still open

The new history-preserving recapture branch passed with B32, four real layers
and all four TP ranks. All three resident output-history buffers were exactly
unchanged after setup; one real model/sampler trace replay then added exactly
one sampled token per row. The counting warmup is retained, and temporary
history snapshots are restored/freed before model trace capture. Canonical v1
transitions opt out of these snapshots because they restore authoritative host
history after setup. This avoids extra snapshot work on the measured CI path.

The same load ran16 steps for each seeded/unseeded × traced/eager sampler
combination. All TP ranks agreed on seeds and active sampled tokens; seed42
traced and eager sequences matched. This does **not** reproduce the full64
quality failure and refutes any unqualified claim that these paths necessarily
produce rank disagreement. A full64, actual-chat, longer control is required.
The source's old SFPU/RNG-clobber comment is not itself proof: the current
Blackhole rand implementation restores mutable lane-register state on each
call. Evidence is `artifacts/penalty_recapture_b32.json` and
`penalty_recapture_b32.log`; no speculative RNG fix has been selected.


### Full64 sampling control reproduces a defect that the reduced test missed

The same actual chat prompt (67 logical tokens,128 physical), allocated B32,
full64 layers and up to160 generated tokens changed the verdict. Both unseeded
traced and unseeded eager sampling produced malformed text and different active
token IDs on different TP ranks. The first disagreement was decode step1:
traced `[4087,310,310,4087]`, eager `[310,4087,4087,310]`, while every rank
had the identical SKIP seed4294967295. There were36 divergent steps in the
160-step traced case and33 in the137-step eager case. Prefill tokens agreed.

Both explicit-seed42 cases, including forced internal tracing, produced the
same coherent149-token completion and zero divergent decode steps. Therefore
internal tracing is not necessary for the failure, and disabling it is not a
supported fix. The full-model graph exposes unseeded rank-local random-state
divergence despite identical seed commands; the reduced four-layer test did
not. Exact token/seed vectors and generated text are in
`artifacts/sampling_rank_full64.json`. The nonzero process exit is the intended
functional rank-agreement assertion; device teardown completed normally.

A probe-only intervention now refreshes a replicated entropy seed vector at
every unseeded sampling step, preserving both the unseeded request contract
and the traced/eager comparison. This is a hypothesis under test, not a
selected production fix. Explicit seeded streams must remain reproducible,
and unseeded requests must remain stochastic.

**Pipeline acceptance gap:** a passing small-model sampling test, identical
seed buffers, coherent greedy text, and a passing mixed-request seed smoke
are insufficient. On the selected full graph, compare actual sampled feedback
IDs on every TP rank for seeded and unseeded requests; read generated text;
exercise enough steps to cover the failing workload. Retain the reduced
negative result so automation does not mistake non-reproduction for refutation.


### Isolated unseeded seed-refresh intervention passes full64

The probe-only intervention completed134 traced and141 eager decode steps with
**zero token/seed rank disagreement**, coherent explanations and EOS in both
cases. Actual explicit-seed activity remained false throughout; the traced arm
still replayed the sampler trace. `artifacts/sampling_rank_full64_refresh.json`
preserves the evidence. This supports refreshing a shared entropy seed vector
per unseeded draw rather than letting TP ranks independently advance their
random state. It does not claim to identify the exact intervening kernel that
advanced each rank's state.

The next production candidate is an optional common SeedManager policy with
the existing default unchanged and Qwen opting in. Explicit seeded and mixed
streams must retain their semantics, and unseeded sampling must still use
tracing. Host regressions, a full64 run without the probe override and actual
HTTP sampled-suite review remain necessary before selecting this repair.

### Further serving prefill data-movement audit

The B32 single-active-slot path borrows all48 linear convolution caches into
composite layout, slices/clones each slot's convolution and recurrent state,
then splices the produced rows through slice/concat/full-cache copy before
restoring decode layout. Existing source says request-boundary conversions
“cost nothing that matters”; that assumption needs a synchronized measurement
now that layer compute is much faster. Full64 B32 control prefill is314–330ms
versus119ms in the B1 generator harness. This difference does not by itself
prove the state adapters account for all211ms. A warmed empty slot-view
roundtrip is planned to isolate their cost.

Source-only alternatives were checked before proposing a patch. The generic
`ttnn.experimental.slice_write` converts interleaved tiled inputs/output to
row-major and converts output back; it is not a direct tiled in-place scatter
for the BFP8 recurrent cache. `ttnn.indexed_fill` supports tiled batch axes,
but creates a fresh full-sized output, so preserving the captured cache
allocation would still require a copy. Neither API name justifies claiming
zero-copy state updates. No state-adapter speedup has yet been measured.

**Pipeline acceptance gap:** once a kernel bottleneck shrinks, remeasure
request setup, state conversion, allocated-slot work and first-response
scheduling on the serving path. Revisit old comments that dismissed overhead
under the former slow graph.


### Guarded serving trace reuse: correct first attempt, slower end to end

An instance-only experiment retained decode/sampler traces across the exact
B32-allocated, C1/slot0, S128, unseeded greedy request envelope. Unknown shapes,
sampler modes, ownership changes, program-cache growth or surviving unsafe
allocations force the ordinary release/recapture route before replay. It uses
no global deallocation hooks and changes no production default.

Four-layer A/B/A/B requests each ran prefill and three resident decode steps.
All active tokens, all32 token rows on every TP rank, all-layer cache digests
for all slots/ranks, and device positions matched forced recapture exactly.
The control captured four times; the candidate captured once and reused three
times. The allocation tracker, including program-owned allocations, reported
zero unsafe survivors and no program-cache growth.

Despite passing correctness, the first measured prototype was slower:
median prefill+first-decode **857.591ms control versus1455.190ms reuse**.
Both arms enabled allocation tracking, so these are reduced diagnostic
request-boundary timings, not production HTTP TTFT. The reuse guard performs
expensive Python garbage collection and allocation inspection; eliminating
capture calls alone does not imply an end-to-end gain. The initial artifacts
are `artifacts/serving_trace_reuse_correctness.json` and
`artifacts/serving_trace_reuse_timing.json`.

A safe refinement under consideration removes explicit garbage collection
while retaining the exact zero-unsafe-allocation requirement: uncollected
objects would conservatively cause recapture, rather than authorize an unsafe
replay. This is not permission to remove ownership guards or claim the521ms
HTTP setup delay has already been eliminated.


### Benchmark metadata must reflect overrides

Independent review found the shared runner's secondary `comparison_scope`
string still said100/100/32 even though the actual command, structured config
and completed token counts correctly recorded4096/252/C8. The old raw result
artifacts retain that stale label as historical evidence. The runner now builds
this descriptive string from the actual requested ISL/OSL/concurrency. This
changes metadata only; it does not change benchmark execution or measurements.
The next live run will validate the corrected label.

This is another pipeline failure mode: free-text default labels can disagree
with executed overrides. Check command and structured config against the
requirements matrix and generate human-readable labels from those same values.


The refinement removed explicit `gc.collect()` from the experiment's guard
without weakening the zero-unsafe condition. All four repeated requests again
matched all token/cache/position checks, with no guard fallback. Across two
timed requests, median prefill+first-decode was **847.928→821.355ms**,
a26.573ms (3.13%) reduction. This is modest, reduced-stack, tracker-enabled
evidence (`artifacts/serving_trace_reuse_without_gc_timing.json`), not an HTTP
claim. TTNN's own allocation-tracked replay still performs garbage collection
on both arms, contributing roughly620ms decode overhead.

**Decision:** keep trace reuse experimental, not a production default. The
full64 graph needs its own ownership validation and an implementation whose
normal serving path does not pay debug allocation-tracker costs, followed by
actual HTTP A/B and slot/shape transition tests. The observed521ms production
setup opportunity remains open. The failed first attempt and refined result
both remain evidence; neither is erased by quoting fewer capture calls.


### Final production seed policy passes; slot-view timing refutes a broad explanation

Without the probe's reseed override, the selected Qwen policy passed all four
full64 cases:131 unseeded traced,147 unseeded eager, and148 steps in each
seed42 case. All574 observed decode steps agreed across TP ranks on tokens
and seeds. Text was coherent and ended at EOS. Seed42 traced/eager sequences
matched each other and the original seeded baseline exactly.
`artifacts/sampling_rank_full64_production.json`, its source-hash sidecar and
log preserve the production-path evidence. Host seed-update enqueue medians
were0.112/0.125ms for the unseeded cases (not synchronized kernel latency);
actual serving benchmarks remain the performance authority.

The warmed empty B32 slot-view roundtrip measured **20.20/20.20/20.14ms**.
All448 touched-cache hashes across four ranks were unchanged. This refutes
treating the view alone as an explanation for the approximately200ms
serving-prefill excess. A further source audit found a separate suspect:
`_scatter_slot_logits` creates a large tiled BFP8 zero tensor for inactive
rows using `ttnn.zeros`; the BFP8 creation implementation constructs, packs
and uploads a host tensor. An isolated comparison against device
`zeros_like(single_row)` and concatenation of repeated zero rows is being
prepared. No gain is assumed until measured.


For the logits-scatter candidate, the pre-measurement hypothesis is that
removing host BFP8 zero construction can save roughly100–200ms per
B32-allocated/single-active prefill. Reusing one device-filled row and
concatenating repeated references should take only a few milliseconds, but
still writes the padded output and is not zero-copy. If this accounts for
the residual overhead, expect warmed full64 prefill to move from roughly
330ms toward130–230ms and HTTP TTFT from roughly890ms toward690–790ms;
this still does not meet60ms. Static operation timing and actual serving
measurements will adjudicate that estimate separately.

The microbenchmark must derive its width from the pinned checkpoint rather
than an old model constant: Qwen3.8 vocab248320 gives local62080 after the
model's TP4 tile padding. The sampler's internal65536 power-of-two width is
a later tensor contract. Earlier boundary tests using248063 do not establish
the current vocabulary size.


The source-prepared scatter experiment records both the logical host float
vector (7,697,920bytes for31 rows) and padded float extent
(246,333,440bytes) before BFP8 packing. This distinguishes a cheap-looking
`zeros` call in Python from its actual host work. It is outside the decoder
layer profile, so a pipeline gate that only inspects per-layer device kernels
can miss it even after the layer graph is optimized.


### Live production seed-policy rerun: sampled corruption absent, budget limits remain

The unchanged canonical smoke again reports3passed/1skipped (30.37s). All
twelve shared greedy/sample completions were manually read. The dropped-word
and malformed sampled-text regression is absent: the learning explanations
and stories are coherent, both translations are correct, and the sampled
Python uses consistent variable names. Both generated Fibonacci function
bodies were parsed and executed at n0/1/10 with expected results.

This is not a blanket complete-answer pass. The fixed256-token budget cuts
off the sampled haiku during coherent reasoning, both stories, thermodynamics
answers, and the sampled Python example output. The function bodies themselves
are complete. Keep these completion limits visible rather than relabeling
request completion as task success. Exact outputs are under
`final_serving/readiness_vllm/`; judgments and executable code checks are in
`artifacts/final_serving_qualitative_review.json`. The production repair's
full64 per-rank proof and this actual HTTP review complement each other.


The seed-policy serving rerun completed4/4 short-point requests and8/8 burst
requests with all1008/2016 requested output tokens. Median HTTP TTFT was
**910.998ms at128/252/C1**, and **12812.076ms at4096/252/C8**. Mean TPOT
was89.073/90.353ms respectively; C8 aggregate output throughput56.934tokens/s.
This repairs correctness, not the remaining TTFT problem. The current short
point is15.18× above60ms. The slight change from891ms is an observed rerun
difference, not an isolated estimate of seed-policy cost. The unchanged
13-point CI dispatch remains pending the current
logits-scatter experiment and final selected-code validation.

The shared runner's corrected secondary metadata now says
ISL4096/OSL252/C8 and agrees with its command and structured config.
Artifacts are in `final_serving/readiness_vllm/`; the original earlier run
remains under `readiness_vllm/` for comparison.


### Device logits fill matches the operation-level hypothesis and is integrated

Static TP4 tests at local width62080 passed all rank-local active and inactive
rows for slots0/7/31. The shared canonical input stayed allocated and unchanged;
each method consumed only its private input clone. B1 returned the exact same
object without accessing TTNN, preserving the existing B1 generator path.
Three warmed alternating samples per slot gave:

| Active slot | Original host-zero scatter | Device-fill scatter | Saved |
| --- | ---: | ---: | ---: |
| 0 | 175.957ms | 0.506ms | 175.451ms |
| 7 | 184.647ms | 0.997ms | 183.650ms |
| 31 | 175.891ms | 0.521ms | 175.370ms |

The100–200ms operation-level saving hypothesis held. The wider output is still
materialized on device; no zero-copy claim is made. This small replacement is
now integrated in `_scatter_slot_logits`, with its original consumed-input
contract intact. Formatting and all12 seed/reload host tests pass. A fresh
full serving run under `device_fill_serving/readiness_vllm/` will adjudicate
the predicted HTTP benefit. The B1 119.301ms prefill path is unchanged because
its early return bypasses this helper body.

`SLOT_LOGITS_SCATTER.md` and `artifacts/slot_logits_scatter.json` contain
source references, exact shapes/dtypes, commands, correctness and timing.
The script can replay the baseline method saved in the artifact even after
production integration; rerunning an A/B against the already-replaced current
method would otherwise silently compare the candidate to itself. That
reproduction trap is another pipeline check to automate.
