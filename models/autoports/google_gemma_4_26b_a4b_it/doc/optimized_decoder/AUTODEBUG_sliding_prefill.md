# AutoDebug: sampled sliding-prefill row misses

Source-only fresh-context investigation, 2026-09-26. The parent requested this
fallback after the CLI AutoDebug isolation runner could not start its sandbox.
The repo-local AutoFix/AutoDebug skills and inspection prompt were read. This
investigation did not execute a device job, alter implementation, or regenerate
the oracle. Device evidence below comes from the parent's saved artifacts.

## Finding and next intervention

The isolated **prefill expert gate/up BFP8 control repairs all 291 sampled
rows**, keeping prefill down BFP4, SDPA LoFi, cache BFP8, and all other reported
policies unchanged. This is a demonstrated passing intervention at the
prefill gate/up weight boundary. The minimum repaired PCC is 0.995109737003
at position 32; row 140287 improves from 0.982345507834 to 0.997184060335.
Endpoint decode results are exactly identical to baseline.

This is not the established long full-attention LoFi failure. Raising only
sliding-prefill SDPA fidelity to HiFi2 repairs none of the original four
sampled-row misses and introduces a fifth. Isolated prefill down BFP8, QKV
weight BF16, and output weight BF16 controls also fail some sampled rows.
Retain their original policies rather than combine unproven promotions.

The source-backed localization is gate/up projection precision and its
subsequent nonlinear expert branch. The gate/up weight intervention occurs
**after routing has already selected experts**; it does not change the router
input or scores. No expert-ID change has been measured, and a router-flip
claim is unnecessary to explain this controlled result. No address, causal
mask, tail ownership, or kernel contract defect was established.

See `AUTOFIX_sliding_prefill.md` for the completed control ledger and the
requested minimal patch, subsequently applied by the parent. The initial hypotheses below are retained
with adjudication; they were predictions before the parent's controls finished.

## Direct evidence

The baseline and controls use the same actual-text layer-0 fixture SHA256
`74cfb212f7f18fbc2bb7ebc3aa5f8f10619766a437b0ea93f86c8ad09f15c596`
and the exact saved 291-query HF oracle. All optimized control runs identify runtime
`e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`.
Each control changes only its intended policy boundary. For the passing gate/up
control the only policy difference is `prefill_expert_gate`, BFP4 to BFP8. For
the attention control it is `prefill_attention_fidelity`, LoFi to HiFi2.

| Artifact | Aggregate PCC | Rows below .995 | Minimum row PCC |
| --- | ---: | ---: | ---: |
| `verified_long_262144_layer0.json` | 0.998877366858 | 4 | 0.982345507834 |
| `actual_long_sliding_hifi2.json` | 0.998962441965 | 5 | 0.981222010118 |
| `actual_fused_long_sliding.json` | 0.999906677988 | 0 | 0.999027582203 |
| `actual_long_sliding_pref_gate8.json` | 0.999310065697 | 0 | 0.995109737003 |
| `actual_long_sliding_pref_down8.json` | 0.999352150014 | 3 | 0.985997648375 |
| `actual_long_sliding_qkv_bf16.json` | 0.998863544311 | 5 | 0.983176989259 |
| `actual_long_sliding_output_bf16.json` | 0.998875459104 | 4 | 0.982213733387 |

| Absolute query | Optimized LoFi | SDPA HiFi2 only | Fused |
| --- | ---: | ---: | ---: |
| 32 | 0.994265170838 | 0.994732582342 | 0.999678835584 |
| 71679 | 0.994287855709 | 0.994071809254 | 0.999836400427 |
| 140287 | 0.982345507834 | 0.981222010118 | 0.999790895358 |
| 262120 | 0.993570005055 | 0.994015399962 | 0.999887427130 |

HiFi2 additionally misses position 39935 at 0.994802451312. Both optimized
runs pass the final two prefill-query checks and the three traced endpoint
decode checks. `verified_long_262143_layer0.json` has the same original four
misses with numerically identical row PCCs. Thus changing the final logical
length and its padded tail does not change these four outputs.

The LoFi mean sampled-row PCC in successive buckets is 0.998938
(4096–32767), 0.998915 (32768–65535), 0.998826 (65536–131071), 0.998636
(131072–196607), and 0.998996 (196608–262143). The late recovery and the
first-chunk miss are inconsistent with a simple monotonic context-length
accumulation story. This is sampled evidence, not an all-token statement.
Most sampled rows are chunk ends by construction, so the two interior misses
at chunk ends are **not** evidence of a chunk-boundary bug by themselves.

`tests/long_context.py:95–112` constructs 32-token pages with a randomized
physical page table, executes the complete device prefill, and reads output
back before calculating PCC. Lines 127–141 add row diagnostics and enforce
the existing aggregate and endpoint gates. Decode uses a blocking trace replay
before reading output (lines 174–176). This is an accuracy observation after
completion, not an enqueue-only assertion or a hang.

## Oracle and geometry audit

`tests/create_optimized_long_reference.py:25–63` validates model, revision,
layer, shape, context, finite values, BF16-roundtripped transport, fixture hash,
length, and slice before loading the oracle. Lines 72–86 build HF K/V for every
position. Lines 100–114 allow exactly `q - 1024 < k <= q` for sliding attention
and use the original absolute RoPE rows for each sampled query. The saved
reference is reused for controls; changing the oracle to accommodate a
candidate would invalidate the comparison.

The model config has hidden width 2816, sliding head width 256, 16 Q heads,
8 KV heads, window 1024, and top-8 routing over 128 experts.
`OptimizedDecoder.prefill_forward` uses physical 1024-token chunks at this
aligned maximum length (`tt/optimized_decoder.py:600–638`). The first attention
call has sequence extent 1024; subsequent calls concatenate a 1024-token BF16
tail and have extent 2048. They do not grow to 262144 attention rows.

The sliding path (`optimized_decoder.py:1158–1202`) does the following:

1. Projects and normalizes Q/K/V, applies the supplied absolute RoPE to Q/K,
   and rounds each to BF16.
2. Writes K/V to the BFP8 paged cache. This is a separate write branch.
3. Runs ordinary causal sliding SDPA on **BF16 current K/V plus the BF16 tail**.
   It never reads this cache for sliding prefill.
4. Discards outputs for the synthetic history Q rows and clones the current
   BF16 K/V as the next tail.

The wrapper in `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp:35–99`
passes no page table or chunk offset to this ordinary SDPA primitive.
`models/demos/gemma4/tt/attention/operations.py:236–261` selects grid 8×8,
Q256/K128, nonapproximate exponentials for H256 unless the environment
overrides chunk sizes. Capture those environment variables with each control.
The explicit compute config sets FP32 destination, full synchronization, and
no approximate math (`optimized_decoder.py:1114–1121`).

This selects standard SDPA, not its streaming variant
(`sdpa/device/sdpa_program_factory.cpp:76–78,483–493`). Its QK and sum
intermediates are FP32, but output ping-pong buffers, max statistics, and max
correction scales remain BF16 (lines 780–784,881–888). Those formats explain
why FP32 destination alone is not an exact oracle. They do not establish a
defect or the cause of these four misses. Unlike the full-attention failure,
this SDPA invocation has at most 2048 physical K rows, regardless of absolute
context position.

For the original misses, the exact legal key intervals are:

| Query | Current chunk start | Row inside chunk | Legal K interval | Prior-chunk keys needed |
| --- | ---: | ---: | --- | ---: |
| 32 | 0 | 32 | 0–32 | 0 |
| 71679 | 70656 | 1023 | 70656–71679 | 0 |
| 140287 | 139264 | 1023 | 139264–140287 | 0 |
| 262120 | 261120 | 1000 | 261097–262120 | 23 |

The first miss does not consume any tail. The two chunk-end misses also
mathematically exclude prior-chunk K/V. A broken stale-tail story therefore
cannot explain all four without an additional independent error. BF8 cache
read quantization cannot be their direct cause because there is no cache read
in this branch. Keep cache dtype unchanged in controls; a BF16-cache run would
primarily perturb allocation and the separate decode path here.

## Ranked hypotheses and falsifiable controls

### 1. Prefill expert projection precision: gate/up intervention verified

**Adjudication:** the gate/up BFP8-only control passes every sampled row and
leaves decode exactly unchanged. Down BFP8 alone leaves misses at 130047,
140287, and 262120. Thus promoting down alone is not sufficient, even though
its aggregate PCC exceeds the gate/up control. These observations support
selecting precision using the enforced row checks rather than aggregate PCC
alone. No combined promotion or higher expert fidelity is presently needed.

**Evidence:** `OptimizedExperts` uses separate BFP4 prefill gate/up and down
weights, LoFi with BF16 destination, 32-token groups, and the active-expert
union (`optimized_decoder.py:58–109,211–239`). The inherited fused expert path
has a different precision/execution policy and passes all sampled rows.
These are large known changes at a row-local nonlinear boundary. There is no
cross-chunk expert state that would make long history necessary.

**Prediction:** Improving one prefill expert weight group changes routed
expert outputs and final PCC while leaving attention outputs, post-attention
residual, router inputs/scores/IDs, and the shared MLP output unchanged. It
should also leave decode numerics unchanged: these prefill expert tensors do
not produce attention K/V or replace decode expert tensors. An improvement
does not prove that LoFi compute or active-sparsity geometry is correct.

**Minimal tests:** two independent full-context runs with
`{"prefill_dtype":"bfloat8_b"}` and
`{"prefill_down_dtype":"bfloat8_b"}`. Compare all 291 rows, especially the
four original misses and position 39935. Do not combine both changes first.
If each gives a partial improvement, a combined control can test whether
their independent contributions are sufficient. If dtype controls do little,
`{"prefill_fidelity":"HiFi2"}` is the next isolated compute control at the
same BFP4 weights and geometry. A passing promotion is a measured sufficient
intervention, not a declaration of inherent instability in the lower dtype.

### 2. BF8 QKV or output weights: isolated promotions refuted as sufficient fixes

**Adjudication:** QKV BF16 leaves five misses; output BF16 leaves the original
four. These controls do not establish that projection quantization has zero
error, but neither is necessary for the demonstrated gate/up repair.

**Evidence:** the factory quantizes the **prefill source** QKV weight as well
as decode weights (`optimized_decoder.py:496–504`), and quantizes the shared
output-projection weight (lines 511–514). Prefill delegates through
`DirectQKV` and `LanePartitionQKV` to `BroadcastQKV` (lines 1334–1336,
1561–1575; `fused_decoder.py:30–38`). Its arithmetic remains the original
FP32-output projection compute; the `qkv_fidelity` field is not its compute
control. Likewise prefill output projection delegates to
`DecodeAttention.project` (`optimized_decoder.py:1128–1130`,
`tt/decode_attention.py:57–66`), using the original HiFi4/FP32-output compute
with the newly quantized weight.

**Prediction:** restoring QKV BF16 first changes Q/K/V, then SDPA and the
post-attention residual. Restoring only output BF16 leaves Q/K/V and SDPA
identical but changes the projected attention and subsequent residual.
Unlike the expert-only controls, either may change routing inputs and selected
experts, but that must be observed, not assumed. QKV weight changes also alter
the K/V subsequently used by decode.

**Minimal tests:** independently
`{"qkv_weight_dtype":"bfloat16"}` and
`{"output_weight_dtype":"bfloat16"}`, leaving expert and SDPA policies at
baseline. If a control repairs the large row-140287 miss, observe that boundary
before choosing a broad precision promotion. A combined QKV/output BF16 run
alone would not identify which projection mattered.

### 3. Expert selection or norms: unmeasured amplification, not the required fix

**Adjudication:** gate/up-only precision leaves routing upstream unchanged.
A change in selected expert IDs is not needed to explain why this control
passes. Instrumentation below is reserved for future unexplained failures,
not a prerequisite to retain the measured gate/up repair.

**Evidence:** the common post-attention residual feeds both routing and expert
inputs (`fused_decoder.py:169–185`). Prefill routing selects top-8 from FP32
scores, then casts route probabilities to BF16 (`fused_decoder.py:407–430`).
Post-expert and shared outputs are normalized before final combination (lines
213–228). Either discrete selection or branch normalization can amplify small
upstream changes. No route or intermediate evidence is saved for these rows.

**Prediction/test:** capture projected attention, post-attention residual,
common normalized activation, router scores/IDs/weights, routed expert output,
and shared output at the failing rows and nearby passing rows. First compare
the same optimized execution with and without observation and require exact
saved-output identity. Then evaluate the HF router on the **same TT residual**
and on the **same TT normalized activation**, as the existing
`tests/probe_optimized_stress_routes.py:138–180` does for decode. That separates
an upstream residual difference from router implementation error. The existing
probe captures only `M=1`; it must be adapted explicitly for prefill chunks.

If TT/HF route IDs differ, record top-8 sets, rank-8/rank-9 margins, and weights;
substitute reference IDs/weights at identical TT expert inputs to test whether
selection explains the final miss. If IDs agree, test routed-expert output and
same-input normalization instead. Do not call a row a route flip based on its
PCC magnitude alone.

### 4. Active-expert batching/reduction or SDPA geometry: demoted

**Adjudication:** gate/up BFP8 passes with the same active union, 32-token
batching, reduction, and SDPA geometry. Changing any of those is not necessary
for this observed repair. There is no source-backed reason here to modify a
shared sparse-matmul or attention kernel.

**Evidence:** optimized experts infer a dynamic active union, change expert
batch size/geometry, weight the private down output in place, and reduce the
expert axis (`optimized_decoder.py:211–239`). Fused prefill computes all experts
with explicit `nnz` (`fused_decoder.py:286–342`). The optimized implementation
is mathematically consistent with sparse routed execution in the inspected
Python. No corrupt buffer, missing selected expert, or invalid reduction was
established. The first miss at row 32 is an expert-group boundary, but the
other misses and biased sample set do not make that a boundary diagnosis.

**Minimal test, only if earlier boundaries point here:** run the two expert
implementations on the exact same saved 32-token expert inputs and route
tensors, matching dtypes and compute fidelity before changing sparse union
versus dense execution. Compare each selected expert's output and final
weighted sum. `active_prefill=False` is a coarse localization control only;
it also changes weights, fidelity, group width, and geometry, so a pass cannot
identify a sparse-matmul defect. Similarly an SDPA geometry control changes
reduction order and cannot by itself establish a causal-mask bug.

## Controls that do not test this prefill path

For M=1024, the following settings are delegated away or guarded by M=1:
`router_direct_grid`, `router_direct_fidelity`, generalized-gate centering,
`qkv_fidelity`, direct QKV grid/cleanup, `native_sdpa_fidelity`,
`output_fidelity`/output decode grid, shared-decode dtypes/fidelity,
sharded-hidden-norm selection, and decode residual placement.
Source guards are `optimized_decoder.py:679–715,814–816,1128–1130,
1334–1336,1561–1575,1758–1760`. These may matter to the parent's separate
decode performance work; changing them is not evidence about these prefill
misses. The QKV/output **weight dtypes** are shared with prefill and do matter.

## Full-context command template

Reuse the parent's device-usage environment and serialize all hardware jobs.
The parent executed this gate/up control using its configured Python environment
(see `sliding_precision_commands.json`). It was **not executed by this
investigation**. The template below uses the repository Python environment.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract long_context --layer 0 --length 262144 --threads 4 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_long/actual_text_layer0_262144_0.pt \
  --reference-file models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_long/actual_text_layer0_262144_reference.pt \
  --default-overrides '{"prefill_dtype":"bfloat8_b"}' \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_long_sliding_pref_gate8.json
```

Record fixture/oracle hashes, runtime hash, actual policy, environment chunk
overrides, row deltas, and return code. The report's diagnostic count is useful
even when the unchanged established aggregate/endpoint assertion passes.

## Optional exact local reproduction

The parent reports that full-context controls are affordable; prefer those
before adding a new harness. A shorter mathematical oracle is possible, but
simply slicing 1024 tokens and resetting positions is not an identical test.

For a target chunk beginning at absolute position C, retain its full 1024 input
rows, the preceding 1024 input rows when C>0, and the original absolute RoPE
rows. Seed the tail by applying the same input normalization, QKV projection,
head normalization, RoPE, and BF16 conversion to the preceding input chunk.
Those K/V values depend only on that chunk's input, not on its own attention
output, so no earlier attention state is needed for a single-layer replay.
Alternatively execute the preceding private `_forward` with its absolute RoPE
and discard its outputs; its generated K/V tail is the required state.

Then call the current chunk through private `_forward` with the same kwargs
used at `optimized_decoder.py:626–637`, the absolute chunk page-table slice,
and valid length 1024. Preserve the original 1024-row expert grouping and
the 2048-row SDPA shape for C>0. Do **not** use public
`prefill_forward(start_pos=C)`: a nonzero start takes the token-by-token decode
path (lines 580–599). Do not drop history Q rows merely because their outputs
are discarded; that changes the SDPA shape and schedule.

For the unchanged HF oracle, use the saved original query row directly.
If recomputing local intermediates, construct K/V for the needed absolute
interval, apply `q-1024 < k <= q` using absolute indices, and use the same
FP32 HF weights and transported BF16-roundtripped input. For query 140287,
the relevant keys are exactly 139264–140287; for 262120, include the 23 keys
261097–261119 from the preceding chunk. Retaining a complete preceding chunk
preserves the original device tail geometry. Keep full-size cache/page-table
allocation if the local test claims exact allocation equivalence; otherwise
label allocation changes explicitly.

Finally compare saved local device outputs with the matching uninstrumented
full-run outputs before using a local replay to adjudicate precision. The
present JSONs store PCCs, not raw outputs, so exact local/full identity has not
yet been established. Matching the mathematical oracle alone does not prove
identical device scheduling or allocation.

## Status

The parent verified a sufficient repair with isolated prefill gate/up BFP8;
the competing SDPA, down, QKV, and output-only promotions do not pass all
sampled rows. The parent applied `sliding_prefill_gate8.patch` after its isolated controls. It selects
BFP8 gate/up for sliding prefill and BFP4 for full prefill, and adds the
requested actual-input sampled-row regression gate. Integration/default-path
reruns, near-maximum sliding context, nearby public contracts, and measured
performance remain the parent's next steps. No implementation edit or device
job was performed by this investigator. The parent confirms a new hardware
geometry batch is running on the immutable patched runtime; final regression
reruns are pending.
