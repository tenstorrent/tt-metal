# AutoDebug: 1025-token prefill and 512 traced decode steps

## Scope and status

Fresh-context, source-only investigation under repo-local AutoFix/AutoDebug.
The AutoDebug CLI previously failed because `bwrap` is unavailable; this
subagent follows its inspection-only workflow. The investigator opened no TT
device and changed no runtime or test implementation. The only executed
numerical check loads the parent's saved output tensors on CPU, checks
finiteness, and computes float64 Pearson correlations. All proposed device
controls below are unrun by this investigator.

**Confirmed and experimentally repaired:** the initial optimized default
differs from fused beyond the unchanged 0.995 equivalence gate on this
workload. Independent controls followed by a justified combination restore
every direct comparison: sliding uses three QKV terms and BFP8 expert down;
full attention restores the original post-attention norm while retaining
sharded common norm. Final text-input precision selection and default
verification remain the parent's work. Some exact-HF failures persist in
both paths and are not relabeled as passes or universally attributed to cache.

## Evidence read directly

Original failing commands are in `final_stress_commands.json`; same-input
controls and exit codes are in `stress_pair_commands.json`. Both use real
checkpoint weights, seed 42, logical prefill 1025, batch 1, 512 decode steps,
tracing, and program-cache-miss guards. All repeated first-position outputs
are equal and the runtime audits are clean. The default disallows the
functional forward fallback.

The paired optimized runs record runtime SHA256
`5366f2b21541c48e55fc9fa2200990e8afbb7dc742396f45c4d1265f46065b45`.
The original failing runs record
`1ca822320732924d7c6510a418c0e4ec28d92422810aef4fcd0fda8f7d10f4b2`;
their reported numerical results match the paired optimized results. Do not
replace either provenance record with the hash of a later runtime.

| Path | Prefill PCC vs HF | Minimum decode PCC vs HF | Failed positions / 512 |
| --- | ---: | ---: | ---: |
| Fused sliding, layer 0 | 0.998553878 | 0.982455098 | 16 |
| Optimized sliding | 0.996941559 | 0.971129869 | 18 |
| Fused full, layer 5 | 0.999477389 | 0.987054399 | 5 |
| Optimized full | 0.999504124 | 0.985001352 | 5 |

Sliding has 15 shared failing positions, three optimized-only HF failures
(1108, 1175, 1519), and one fused-only HF failure (1428). Full attention has
four shared failures, one optimized-only failure (1398), and one fused-only
failure (1396). Equal failure counts do not imply preservation.

`stress_pair_cpu_comparison.json` records the CPU comparison of all saved
prefill/decode tensors, tensor-file SHA256 hashes, finite-value checks, and
both HF PCCs at every position. Prefill direct PCC is 0.998173080 sliding and
0.999856784 full. All 512 decode outputs were compared, without exclusions:

| Layer | Position | Fused vs optimized PCC | Fused vs HF | Optimized vs HF |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1066 | 0.994938538 | 0.996982668 | 0.998195161 |
| 0 | 1108 | 0.994870912 | 0.999699939 | 0.994996942 |
| 0 | 1175 | 0.994301590 | 0.999604680 | 0.994828371 |
| 0 | 1428 | 0.980942478 | 0.982703439 | 0.997230422 |
| 0 | 1519 | 0.971447063 | 0.999722754 | 0.971129869 |
| 5 | 1396 | 0.986259157 | 0.988383114 | 0.998097099 |
| 5 | 1398 | 0.993124603 | 0.999842374 | 0.993106589 |

Position 1066 demonstrates why an HF-failure-only probe is insufficient:
both paths pass HF individually, but they fail direct equivalence. Positions
1428 and 1396 improve against HF while failing direct equivalence. Neither
observation establishes whether their selected experts actually differ.

## Dataflow and precision ledger

`run_decoder.py` creates independent BF16-rounded random decoder inputs; no
output is fed back as a later input. It advances the FP32 HF cache once per
logical position. Trace warmup, capture, and repeat overwrite the same first
device cache row. Every subsequent input and both position tensors are copied
before replay; output readback synchronizes before its comparison. The saved
outputs are readbacks from this same loop. The initial position is represented
twice in JSON checks but only once in the saved 512-output list.

| Boundary | Fused baseline | Optimized stress default |
| --- | --- | --- |
| QKV weights / decode activation | BF16 / FP32 | Same |
| Decode QKV computation | FP32 broadcast products and reduction | Two BF16 activation terms, 16 disjoint K lanes, HiFi4, FP32 destination/output |
| QKV geometry | Broadcast grouping | Layer 0: 8x8, K11, subblock 1; layer 5: 9x8, K11, subblock 4 |
| Q/K/V head normalization; attention | Existing FP32 paths | Same; no native decode SDPA |
| Cache | BF16 TILE, 32-token pages | Same dtype/layout/update/mapping contract |
| RoPE table storage | BF16 | Same |
| Input/post/common hidden norms | Existing fused policy | Sharded input+common for sliding; post+common for full |
| Residual | FP32 | FP32, width-sharded working storage |
| Router scores / top-k | FP32, choose scores before BF16 route storage | Same operators; normalized input can differ |
| Decode expert weights | BF16 gate/up/down | BFP8 gate/up, BFP4 down, LoFi, 11x4/K11 |
| Shared decode MLP | BF16 | BFP8, LoFi, DRAM-sharded, one reader/K11 |
| Prefill expert weights | BF16 | BFP8 gate/up; BFP4 down sliding, BFP8 down full; LoFi |
| Prefill expert work | Dense expert batches of 64 | Active expert union in batches of 32 |
| Output | BF16 | BF16 |
| CCL | None | None |

`FusedDecoder._forward` writes attention K/V before calling the router and
experts. `OptimizedSharedMLP` delegates prefill to its source. Sharded hidden
norms and lane-partition QKV only apply to single-row decode. At prefill the
lane wrapper delegates its source projection, and the sharded norm override
delegates the fused norm. Therefore changed prefill expert precision/fidelity
cannot ordinarily change later decode caches or inputs in this layer-only
harness. A prefill-memory corruption hypothesis would require separate
evidence; reduced prefill PCC alone is not such evidence.

Likewise, decode expert and shared-MLP errors occur after routing and cache
updates, and do not accumulate into future inputs or K/V in this harness.
The longer run supplies more independent sensitive inputs and more cache
updates; it is not autoregressive accumulation through decoder outputs.

### Exact paging check

Allocation extent is `ceil((1025 + max(128,512))/1024)*1024 = 2048`, with
64 pages and reverse mapping `physical_page = 63 - logical_page`. Actual
cache shapes are `[64,8,32,256]` sliding and `[64,2,32,512]` full attention.
Prefill contains a full 1024-token chunk and a one-logical-token tail; the
tail is physically padded to 1024 for sliding and 32 for full attention.
The cache fill writes the rounded valid tail page, and each decode update
overwrites its own valid row before attention.

The active implementation is `BatchedPagedAttention`, not native dynamic
SDPA. Sliding reads 33 pages beginning at
`floor(max(position+1-1024,0)/32)` and masks future/out-of-window rows. Full
attention reads all 64 allocated pages and masks future rows. At position
1536 the sliding read is logical rows 512..1567, inside the allocation.

| Sentinel | Step, zero-based | Logical page / row | Physical page |
| --- | --- | --- | --- |
| First direct sliding miss, 1066 | 41 | 33 / 10 | 30 |
| First HF sliding miss, 1067 | 42 | 33 / 11 | 30 |
| Sliding 1108 | 83 | 34 / 20 | 29 |
| Sliding 1175 | 150 | 36 / 23 | 27 |
| Sliding 1428 | 403 | 44 / 20 | 19 |
| Sliding 1519 | 494 | 47 / 15 | 16 |
| First HF full miss, 1189 | 164 | 37 / 5 | 26 |
| Full 1396 / 1398 | 371 / 373 | 43 / 20,22 | 20 |

There is no observed cliff at a page, tile, power-of-two, or active-attention
chunk boundary. This source-level coverage check demotes under-allocation;
it does not substitute for an exact cache readback if localization implicates
the cache. An over-allocation experiment is not the first discriminating
control for these dispersed positions.

## Hypotheses and focused localization ladder

1. **Quantized MLP work consumes most of the output-error budget.** BFP4
   down-projection is a plausible source of the approximately 0.995 direct
   misses at 1066, 1108 and 1175. This remains a hypothesis: shared BFP8,
   BFP8 expert gate/up, and differing routes can also contribute. First make
   the one-variable control `expert_down_dtype=bfloat8_b`, leaving all other
   defaults unchanged. Save all outputs and compare against both fused and
   HF. Capture route IDs/weights and pre-expert input to confirm they stay
   unchanged. Improvement with unchanged routes localizes a downstream
   precision boundary; it does not prove a sparse-matmul kernel defect.
   If needed, compare same-input down output using the actual quantized
   device weight against FP32 CPU matmul and then original BF16 weight.
   This separates storage quantization from arithmetic error.

2. **An upstream perturbation changes a near-tied route.** The large isolated
   direct differences and opposite HF outcomes at 1428/1519 and 1396/1398
   are consistent with rank-eight/rank-nine sensitivity, but outputs alone
   do not prove it. Capture the actual stages in both paths on the same
   complete stream: input norm, QKV, post-RoPE Q, actual cache K/V, SDPA
   result, attention projection, router residual, common-normalized tensor,
   route scores/IDs/weights, expert mixture and shared output. Include all
   seven direct-miss positions, shared HF misses 1067/1189, and adjacent
   passing positions. Report top-eight set intersection and rank8-rank9
   logit gaps, not just stage PCC.

3. **Locate a route change at the first differing boundary.** Run CPU router
   arithmetic first on the actual TT residual, then on the actual TT common
   normalized value. The latter matters because optimized common RMSNorm
   is supplied directly to `BroadcastRouter`; recomputing CPU norm would
   otherwise conflate common-norm error with projection error. If CPU on
   the actual normalized value agrees with TT, localize upstream rather
   than increasing router fidelity. If attention is the first meaningful
   difference, run FP32 CPU SDPA and output projection on the captured Q and
   logical K/V reconstructed through the actual page table. High agreement
   there implicates Q/K/V production, not the attention arithmetic.

4. **Apply independent same-cache precision controls.** Keep BF16 cache
   dtype, page mapping, full allocation, update contract, and attention
   implementation fixed. Use `qkv_terms=3` as one control, then original
   broadcast QKV as a second. Separately restore just one hidden-norm site:
   sliding `input_common -> input` removes sharded common norm;
   sliding `input_common -> common` removes sharded input norm;
   full `post_common -> post` removes sharded common norm;
   full `post_common -> common` removes sharded post norm. These controls
   should be selected from captured evidence, not combined speculatively.
   Compare projection candidates on the identical captured normalized
   activation and keep an identical BF16 cache snapshot for attention-only
   comparisons. A full-stream QKV change also changes later stored K/V;
   passing that run alone does not distinguish Q from historical K/V.

5. **Confirm inherited HF outliers separately.** Only after the new direct
   differences are understood, reproduce shared failures with CPU FP32 HF
   plus BF16-rounded cache, keeping all other arithmetic and RoPE FP32.
   Add BF16 RoPE as a separate control. Use exact captured inputs and assert
   their equality to the harness seed reconstruction. The earlier
   `fused_decoder/AUTODEBUG_tail65.md` proved this for position 68 at S65;
   it does not establish the cause at every new S1025 position. Cache-only
   reproduction plus same-Q/K/V SDPA comparison provides the required
   evidence; cache dtype alone is not an explanation.

Do not combine higher precision at several sites before locating the first
divergence. A passing higher-precision candidate proves sensitivity to that
change, and must still pass the full unchanged stress, headline, trace and
public-contract gates before being retained. No threshold relaxation,
position exclusion, functional runtime fallback, or speculative runtime fix
is proposed.

## Minimal control invocation and instrumentation cautions

Ordinary candidate flags are ignored with `run_optimized_decoder --defaults`.
In particular, `--defaults --expert-down-dtype bfloat8_b` is **not** a down-only
control. The parent added `--default-overrides` during this investigation;
the preferred exact control is now
`--defaults --default-overrides '{"expert_down_dtype":"bfloat8_b"}'`.
The runner records these overrides and its resolved `precision_policy`.
The explicit cumulative command below remains a valid alternative.

The following is the exact explicit sliding policy for the down-only control;
changing only `bfloat8_b` after `--expert-down-dtype` back to `bfloat4_b`
reconstructs the tested default. For full attention use layer 5, QKV grid
9x8/subblock 4, prefill-down BFP8, and `post_common` norms. Run this only through
the parent-owned serialized device workflow:

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_decoder \
  --layer 0 --length 1025 --real --decode --steps 512 --verify-program-cache \
  --expert-gate-dtype bfloat8_b --expert-down-dtype bfloat8_b \
  --expert-grid 11 4 --expert-block-w 11 --expert-fidelity LoFi \
  --qkv-lanes 16 --qkv-terms 2 --qkv-grid 8 8 --qkv-block-w 11 \
  --qkv-subblock-w 1 --qkv-fidelity HiFi4 \
  --active-prefill --prefill-tokens 32 --prefill-grid 11 4 --prefill-block-w 11 \
  --prefill-dtype bfloat8_b --prefill-down-dtype bfloat4_b --prefill-fidelity LoFi --prefill-l1 \
  --shared-dtype bfloat8_b --shared-dram --shared-readers 1 --shared-block 11 --shared-fidelity LoFi \
  --sharded-norms --sharded-norm-site input_common --residual-sharded \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/stress_down8_layer0.json \
  --save-output-tensors models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/stress_down8_layer0.pt
```

For a norm-site control change only `--sharded-norm-site` relative to the full
baseline command. For a QKV three-term control change only `--qkv-terms`.
Broadcast control removes the QKV flags and leaves `--compensated-qkv` absent;
the candidate runner then explicitly supplies `qkv_lanes=0`.

Existing `probe_sequence_decode.py` cannot simply be pointed at optimized
defaults. It patches `FunctionalDecoder.from_state_dict` before fused and
optimized wrappers replace the objects it instruments. Its router probe also
accepts only one argument, while `FusedDecoder` calls the router with
`normalized=`. Its projection diagnostic assumes `.weight` and `.compute`
directly, while `LanePartitionQKV` retains the weight in `.source` and can
retain separate `.weights`. Attach probes after the selected decoder factory
has fully returned; preserve `*args, **kwargs` and attribute delegation.
Preserve probe tensor lifetimes across traced replay and read them only
outside capture. Do not use the initial-position-only `--diagnostic` check
as evidence about position 1519. Select explicit direct-miss sentinels even
when their HF PCC passes.

`hf_long_decode_precision_controls.py` is currently hard-coded to layer 0
and sliding RoPE. It needs a test-only layer argument plus the matching
layer-type RoPE selection before it can establish the full-attention control.
The `.pt` files analyzed here contain outputs, not input fixtures; do not feed
them to that control's `--input-fixture` argument. Save the actual harness
input stream first. CPU oracle substitutions belong exclusively to diagnosis;
they must not be retained as a runtime route or output fallback.

## Completed independent controls and justified combination

The parent executed the controls; this investigator read their JSON and
compared every saved tensor on CPU. Exact commands/exit codes are in
`stress_down8_commands.json`, `stress_normoff_commands.json`,
`stress_terms3_commands.json`, and `stress_combined_commands.json`.
`stress_comparison_layer{0,5}.json` records all per-position comparisons.
These controls preserve the initial cache, prefill and other policy settings;
the override dictionary records the independent variable.

| Layer / override | Minimum direct PCC | Direct failures | New HF failures vs fused |
| --- | ---: | --- | --- |
| 0, down BFP8 | 0.974616344 | 1428, 1519 | 1519 |
| 0, sharded norms off | 0.970910830 | 1036, 1066, 1108, 1175, 1428, 1519 | 1036, 1108, 1175, 1519 |
| 0, QKV three terms | 0.994308722 | 1036, 1108, 1175 | 1036, 1108, 1175 |
| 0, QKV three terms + down BFP8 | **0.998895224** | **None** | **None** |
| 5, down BFP8 | 0.987929281 | 1396, 1398 | 1398 |
| 5, sharded norms off | 0.996141155 | None | None |
| 5, QKV three terms | 0.986260160 | 1396 | None |
| 5, common sharded norm only | **0.996061028** | **None** | **None** |

This separates two sliding contributions: down precision repairs the small
direct misses, while the third QKV activation term repairs the large direct
differences. Neither independently passes every direct comparison; their
combination does. It retains sharded input/common norms and the reduced
prefill policy. Full attention only needs restoration of its post-attention
normalization boundary for this stream; it retains two-term QKV, sharded
common norm and BFP4 down. The three-term full control improves HF agreement
at 1396 but still differs from fused there, so it is not the demonstrated
full-path preservation fix.

These results verify boundary-specific accuracy effects, not a particular
router-ID substitution or kernel defect. No router capture was needed to
establish the successful direct controls. `probe_optimized_stress_routes.py`
is prepared if remaining source localization is useful: it wraps the fully
built fused/optimized factory, preserves `normalized=`, records actual scores
and top-k indices, and checks CPU routing from both the actual residual and
actual normalized input. It performs no host work inside the forward/capture
and requires exact prefill-plus-all-decode output identity against an
uninstrumented saved run. It passed Black and Python compilation; this
investigator has not run it on hardware.

The new CPU-only `compare_decoder_outputs.py` supports one baseline stem and
repeatable candidate stems, checks workload/position/shape coverage, preserves
all outputs, and reports shared/new/recovered HF failures separately from
direct-equivalence failures. It was run against all controls in the table.

## Remaining inherited checks and precision selection

The separate CPU investigation's `stress_cpu_precision_layer0.json` verifies
the seed reconstruction against every recorded FP32-HF-vs-TT PCC (maximum
discrepancy 2.69e-8). BF16 cache plus BF16 RoPE reproduces six shared failures:
1067, 1181, 1191, 1310, 1354, 1420. Both original TT paths pass 0.995 against
that CPU control at these six positions. Cache-only reproduces 1310. The
other baseline failures are not yet explained by these controls. Do not
generalize the proven S65 cache result or this six-position result to every
remaining HF miss; the CPU-control report records the exact boundaries.

The completed full-attention control reproduces shared positions 1189,
1289 and 1382 with BF16 cache alone; both TT variants pass against those CPU
outputs. It leaves shared position 1360 unexplained. For every reproduced
shared failure in either layer, the CPU control changes one selected expert
while attention PCC remains at least 0.9999959. See
`AUTODEBUG_stress_cpu.md` and `stress_cpu_precision_summary.json` for the
complete input-identity validation and per-boundary evidence. These controls
also create some CPU-only failures, emphasizing that they are precision
experiments rather than complete emulations of the TT path.

The stress stream uses real checkpoint weights with Gaussian activations.
It establishes preservation and sensitive numerical boundaries on that
stream; it does not replace the user's real-input precision selection.
A bounded real-text fixture needs no full model allocation: tokenize fixed
text, gather its checkpoint embedding rows with the HF embedding scale, and
record layer-0 inputs. To obtain genuine layer-5 inputs, run HF layers 0..4
sequentially with real weights, retaining only one layer at a time and the
activation stream. Use the same segmented prefill/teacher-forced decode
semantics, retain FP32 captured boundaries and the exact BF16 inputs uploaded
to TT, and compare HF on those same uploaded tensors. Record text/token and
tensor hashes. A collection of real-text fixtures is the appropriate next
evidence for retaining or rejecting reduced precision under that constraint.

**Present verdict:** the newly introduced stress differences are repaired
experimentally by the minimal layer-specific controls above. All 512 direct
outputs and prefill pass unchanged 0.995, and the exact-HF failure sets now
match fused. The exact-HF gates themselves still fail at the shared positions.
Final default selection, real-text precision evidence and final headline/
public-contract verification remain outstanding; no threshold or position
was relaxed, and no runtime fallback was introduced.
