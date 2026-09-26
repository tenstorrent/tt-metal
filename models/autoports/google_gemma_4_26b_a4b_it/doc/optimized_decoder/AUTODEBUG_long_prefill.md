# AutoDebug: actual-text maximum-context full prefill

Source-only investigation, 2026-09-26. The parent requested a fresh-context
fallback because the AutoDebug CLI sandbox was unavailable. This investigation
read the AutoDebug skill and inspection prompt; it did not run a device job,
change model runtime, or regenerate the oracle. Hardware results below were
read from artifacts produced by the parent.

## Finding

The actual-input maximum-context prefill failure is repaired at the existing
aggregate gate by changing **only prefill attention fidelity from LoFi to HiFi2**:
PCC rises from 0.99170327883148 to 0.9987005770651347. The fixture, sampled rows,
runtime hash, BFP4 expert weights, BFP8 cache, and chunk geometry are unchanged.
The saved precision policies differ only at `prefill_attention_fidelity`.
Decode remains identical and passing. Raising either or both routed-expert
prefill weight groups to BFP8 did not pass.

This localizes the demonstrated passing intervention to full-prefill SDPA math
fidelity. Recurrent BF16 output accumulation remains a source-supported
long-context sensitivity, but the HiFi2 pass disproves any claim that changing
those buffers or cache precision is necessary for the aggregate gate. No
cache-addressing, causality, or oracle bug was established. At the same legal paged Q64/K256 geometry, LoFi fails at 0.9945550525960524
while HiFi2 passes at 0.9990505377367619; all 291 sampled HiFi2 rows also exceed
.995. This is a second matched-geometry fidelity contrast. The parent may
separately test paged Q32/K512 to decide whether further blocking can retain
LoFi accuracy.

Do not lower context capability, change the .995 gate, or globally raise
precision. Retain the original expert policy while comparing legal local
attention candidates. Maximum-context HiFi2/K128 passes the established aggregate
contract but has two interior diagnostic misses; HiFi2/Q64/K256 passes every
sampled row. Neither sampled result is an all-token accuracy claim.

## Direct observations

All seven rows below use layer 5, full attention, actual input fixture
`52d5a811ff0735478d3fff537be518fc10c098c812c8a34ffede705b0385352a`,
262144 positions, and runtime
`d6f4d858d7358f6d52332d9d1747a8bdb25cfa359f5c6f5aedde2f2e4f5d9f93`.
The gate is .995 and compares 291 sampled query rows while executing all tokens
and retaining all causal K/V positions.

| Artifact | Prefill override | Prefill PCC | Result |
| --- | --- | --- | --- |
| `validated_long_262144_layer5.json` | None | 0.99170327883148 | Fail |
| `actual_long_control_down8_layer5.json` | Down BFP8 | 0.9920093534153488 | Fail |
| `actual_long_control_gate8_layer5.json` | Gate/up BFP8 | 0.9920183417617187 | Fail |
| `actual_long_control_both8_layer5.json` | Gate/up and down BFP8 | 0.9923329847405855 | Fail |
| `actual_long_attention_hifi2_layer5.json` | Prefill SDPA HiFi2 only | 0.9987005770651347 | Pass |
| `actual_long_paged_q64k256_lofi_layer5.json` | Paged Q64/K256 only | 0.9945550525960524 | Fail |
| `actual_long_paged_q64k256_hifi2_layer5.json` | Paged Q64/K256, prefill HiFi2 | 0.9990505377367619 | Pass |

Every run's traced decode at positions 262143, 262142, 262143 passes, with
identical PCCs 0.9977452721876459, 0.9990883887593417,
0.9977452721876459. The prefill and decode device-only audits pass. The log
identifies Blackhole. Failure occurs in the accuracy assertion after complete
prefill, blocking decode replay, and host output reads; it is not a hang or an
unsynchronized completion claim.

The parent also reports passing actual 4096 prefill and sliding maximum-context
tests. These are contrasts, not proof that the same precision policy is safe at
every full-attention length. The earlier Gaussian down4 miss is diagnostic
history, not the reason to reject this actual candidate. Conversely, the
actual-input failure cannot be waived by earlier actual short-context passes.

## Oracle and input audit

`tests/create_optimized_activation_fixture.py:58–68,86–127,130–154,197–203,255–272`
captures actual corpus tokens, FP32 checkpoint embeddings including the FP32
scale, and FP32 HF layers 0 through 4 to produce the layer-5 boundary. All five
preceding layers are sliding attention. Streamed capture explicitly constructs
HF `DynamicSlidingWindowLayer` instances and derives each mask from its current
cache geometry. Saved transport inputs are BF16-roundtripped FP32. The raw FP32
boundary is retained separately and is not substituted for the device input.

`tests/create_optimized_long_reference.py:25–47` validates model, revision,
layer, shape, context, source kind, finite values, and exact BF16 transport.
`51–63` matches reference fixture SHA256, used length, and prefix slice before
loading the oracle. The reference generator uses the same FP32 real-layer
loader semantics as the main harness.

`67–120` projects K/V for **every** position in 1024-token CPU batches. Q is
sampled only after complete K/V construction. `FixedCache.update` returns that
full K/V pair, and the explicit mask admits keys `position <= query_position`;
sliding layers additionally require `position > query_position - window`.
Query positions are supplied through the matching absolute RoPE rows. Full
attention has no sliding truncation in this oracle.

`actual_text_long/verification.json` records finite reference tensors of
`[1,291,2816]`. Its independent 262143-versus-262144 causal-prefix check has
290 shared query rows and minimum layer-5 per-token PCC 0.9999999987744099.
This supports the length/mask construction. It is not a full independent
implementation proof of the oracle. No mismatch in input transport, mask
extent, sampled-row indexing, or revision was found in the inspected path.

## Exact path differences that matter

`tt/optimized_decoder.py:532–610` uses outer prefill chunks of 1024. For the
failing length, all 256 chunks are full; there is no short-tail padding branch.
Page size is 32 and the table has 8192 entries. Each chunk's fill table is the
corresponding 32-page slice. The chunk reader receives the full table and an
absolute scalar offset.

`1052–1119` normalizes/projects/rotates Q/K/V, casts all three to BF16, and writes
each K/V chunk to the configured BFP8 cache. Its attention dispatch is:

| Case | Attention inputs and path |
| --- | --- |
| Full chunk 0 | BF16 Q/K/V tensors, ordinary causal SDPA |
| Full chunks 1–255 | BF16 Q plus BFP8 paged K/V, chunked SDPA |
| Sliding prefill | BF16 current K/V and BF16 saved tail, ordinary sliding SDPA |
| End-context decode | BF16 query plus BFP8 paged K/V, separate decode SDPA kernel |

Consequently, sliding maximum-context prefill does not exercise the BFP8
cache-fed full-prefill SDPA path. Passing end-context decode constrains gross
cache corruption but does not clear the prefill SDPA kernel: decode selects
`paged_scaled_dot_product_attention_decode` (`905–936`), while prefill selects
`chunked_scaled_dot_product_attention` (`839–902`). Also, the decode comparison
contains only two distinct rows, whereas aggregate prefill contains 291.

The original TT prefill output is not saved as a tensor. The original
aggregate PCC therefore cannot show whether divergence grows with position,
appears at a specific chunk boundary, or belongs to a few activation cohorts.

## Ranked hypotheses and adjudication

### 1. Prefill SDPA fidelity: demonstrated passing intervention

`tt/optimized_decoder.py:842–852,1006–1019` selects, for head dimension 512,
grid 8×4, Q chunk 128, K chunk 128, LoFi, FP32 destination accumulation,
`math_approx_mode=False`, `exp_approx_mode=False`, and full destination sync.
Environment overrides for Q/K chunk sizes already exist at setup time.

The scalar-offset API (`ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp:102–140`)
forwards these values and always enables causality. The validator
(`device/sdpa_device_operation.cpp:331–374`) requires the offset to be divisible
by both Q and K chunk sizes and requires sufficient K coverage. All actual
offsets are multiples of 1024, so the selected 128/128 configuration obeys
these constraints. At length 262144 the last outer chunk begins at 261120;
at length 4096 it begins at 3072. No offset-unit or cache-size contradiction
was found.

The program factory (`device/sdpa_program_factory.cpp:314–360`) sets the key
extent to `chunk_start_idx + Sq`, then divides its padded extent by K chunk
size. The last query therefore consumes 2048 K blocks at maximum context
versus 32 blocks at 4096. This is a concrete factor-of-64 difference in the
recurrent attention reduction, even though each outer prefill chunk has the
same shape.

**FP32 destination accumulation does not make that whole recurrence FP32.**
`sdpa_program_factory.cpp:76–78` selects the standard kernel when FP32
destination accumulation is enabled. `775–888` makes QK intermediates and
softmax sums FP32 but keeps `out_im_A`, `out_im_B`, max statistics, and
`exp_max_diff` BF16. `kernels/compute/sdpa.cpp:201–280` enters
`sdpa_standard`. Its shared implementation in
`kernels/compute/compute_common.hpp:1868–1939`:

1. Computes a K-block's softmax products and P×V.
2. Packs P×V into the current BF16 output intermediate.
3. Computes a max-change scale.
4. Accumulates `previous_output * scale` into the current BF16 output buffer.
5. Swaps these buffers for the next K block.

Final normalization divides this accumulated numerator by the accumulated
sum (`2058–2064`). Increasing fidelity changes matrix multiplication arithmetic
but does not change these intermediate formats. Increasing K chunk size
reduces the number of recurrent BF16 updates and also changes blocking, so its
effect is a useful control but is not a pure precision switch.

This recurrence can contribute to length sensitivity and differs from the
decode kernel. The subsequent HiFi2-only pass establishes a sufficient local
fidelity intervention for the aggregate gate. The internal BF16 recurrence is
unchanged by that passing control, so it is not by itself a blocker. The
reported row means still decrease modestly with context under HiFi2, consistent
with residual length sensitivity but not a proof of its source. No kernel
contract violation was established.

### 2. BFP8 cache contribution: possible residual error, not necessary to promote

Only full prefill consumes the BFP8 cache for these rows. Cache quantization
could perturb long-context attention weights enough to alter the post-attention
residual and routing. The two passing end-context decode rows show that the
cache can support a passing result at those positions, but do not prove that
all sampled prefill rows would pass under decode arithmetic.

A `kv_cache_dtype=bfloat16` control would isolate storage precision, but the
HiFi2 pass gives no reason to require that promotion. Reserve this experiment
for unresolved row-level concerns after local attention controls; keep the
same context and oracle if it is run.

### 3. Routed-expert prefill quantization: measurable contributor, insufficient fix

`tt/optimized_decoder.py:57–109,207–235` prepares separate prefill gate/up and
down tensors, performs sparse gate/up, GELU/multiply, sparse down, route
weighting, and expert reduction. These operations use 32-token groups and do
not recurrently accumulate across context chunks. Their errors can still
depend on long-context attention outputs and activation distributions.

The isolated BFP8 weight controls give small improvements, and both8 remains
below .995. This refutes the claim that promoting these two weights alone is
sufficient. It does not exonerate their LoFi compute or BF16 intermediates.
After attention is repaired, re-evaluate the original BFP4/BFP4 policy before
keeping any weight promotion. Identical decode results are expected: these
prefill expert projections do not produce the attention K/V cache.

### 4. Other stage errors: retain only if localized diagnostics point there

Input normalization, QKV projection, output projection, routing, shared MLP,
and residual normalization can propagate or amplify an attention difference.
No new long-context-specific branch was found there that explains this failure
better than the attention reduction. The policy's `native_sdpa_fidelity` and
`output_projection` metadata primarily describe decode paths; do not infer that
changing either field independently changes full prefill arithmetic. The
prefill attention knob is explicitly `prefill_attention_fidelity`.

## Legal attention controls and L1 estimates

The CB inventory follows `sdpa_program_factory.cpp:453–515,775–889` for full
layer-5 H512, BF16 Q/output, FP32 QK/sums, and grid 8×4. All candidate shapes
have more than one query chunk per core and therefore use double-buffered Q.
Paged chunks use BFP8 K/V and a 32-KiB table; **the first chunk uses BF16 K/V
and no table**. The imported first-chunk helper
`models/demos/gemma4/tt/attention/operations.py:222–257` reads the same Q/K
variables as the configured paged helper. Both must fit when using environment
controls.

For tiled Q count `q`, tiled K count `k`, and head width 16 tiles, CB bytes are:
`Q=2*q*16*2048`, `K+V=4*k*16*KV_tile_bytes`, `QK=q*k*4096`,
`output_pingpong_and_final=3*q*16*2048`, `max_and_exp=3*q*2048`,
`sums=2*q*4096`, `mask_and_scalars=4*2048`, plus `32768` for a paged table.
KV tile bytes are 2048 for initial BF16 and 1088 for paged BFP8.

The parent ran Q128/K256 through the shared environment variables. It failed
at the **first ordinary SDPA call** (`optimized_decoder.py:1110`), before an
accuracy result: `actual_long_attention_k256_layer5.log:23,43–49` reports CB
region end 2012160 B versus physical L1 1572864 B. The initial CB inventory
is 1900544 B, a 111616-B (109-KiB) difference consistent with CB base/reserved
space. `tt_metal/impl/program/program.cpp:1873–1879` checks the region end,
not merely the sum of CB sizes. The inferred 109-KiB offset below is calibrated
to that observed allocation, not a universal reservation across builds.

| Q / K chunk | Initial BF16 CB KiB | Initial end with 109-KiB offset | Paged BFP8 CB KiB | Paged end with offset | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| 128 / 128 | 1280 | 1389 | 1072 | 1181 | Existing |
| 128 / 256 | 1856 | 1965 | 1408 | 1517 | Observed first-chunk overflow |
| 64 / 256 | 1444 | 1553 | 996 | 1105 | First chunk predicted 17 KiB over capacity |
| 32 / 256 | 1238 | 1347 | 790 | 899 | Legal candidate across both paths |
| 64 / 512 | 2532 | 2641 | 1604 | 1713 | Both paths exceed capacity |
| 32 / 512 | 2294 | 2403 | 1366 | 1475 | Paged candidate only; first chunk cannot fit |

Physical capacity is 1536 KiB
(`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33`). Other live L1
allocations can still constrain apparently fitting candidates. All listed Q/K
dimensions satisfy tile alignment and divide outer offsets of 1024. A larger
grid does not reduce these per-core CB sizes. To try paged Q32/K512, preserve a
separate fitting first-chunk geometry; a global K512 override cannot work.
The first chunk has only 1024 keys, so keeping its existing geometry does not
remove the long-context experiment.

The parent's `tests/probe_optimized_prefill_chunks.py` now replaces only the
configured paged helper's program during construction. Source review confirms
that its nested factory patch composes with the contract wrapper and retains
explicit fidelity/dtype overrides; it does not change the initial BF16 SDPA
call. Clear global Q/K environment overrides when using this probe. The next
bounded candidates preserve the first chunk at Q128/K128. Paged Q64/K256
LoFi ran successfully and improved aggregate PCC to 0.9945550525960524, still
below .995. This is an accuracy miss, distinct from the first shared-environment
allocation failure. Paged Q64/K256 HiFi2 passes at 0.9990505377367619 and all 291 sampled rows
exceed .995; paged Q32/K512 LoFi remains a justified next larger-K candidate
if the parent needs to compare accuracy and performance.

HiFi2/K128 is already a passing control. Compare accuracy and measured prefill
performance before selecting it or a larger-K LoFi configuration. Recheck
maximum and nonaligned near-maximum context, short/representative inputs, and
cache-consuming decode on the selected defaults. A higher-fidelity or BF16
cache escalation is not required by the current aggregate evidence.

## Sampled-row diagnostics

After its expert-control batch completed, the parent added per-sampled-row PCC
and pass/fail diagnostics to `tests/long_context.py:127–133`. The existing
aggregate .995 gate and sample set are unchanged. No raw tensors are saved.
An earlier proposed richer diagnostic patch was not applied and was removed
from this task's deliverables to avoid conflicting with the parent's edit.

The HiFi2 artifact's 291 row diagnostics contain 289 rows at or above .995.
The misses are position 229375 at 0.9942408184463241 and position 249855 at
0.994988220728704. These do not alter the existing aggregate gate.

| Query positions | Samples | Minimum row PCC | Mean row PCC | Rows below .995 |
| --- | ---: | ---: | ---: | ---: |
| 0–1023 (direct BF16 K/V) | 4 | 0.998580564 | 0.999310656 | 0 |
| 1024–4095 | 3 | 0.998462234 | 0.999246533 | 0 |
| 4096–32767 | 28 | 0.998726126 | 0.999330485 | 0 |
| 32768–65535 | 32 | 0.997780391 | 0.999172943 | 0 |
| 65536–131071 | 64 | 0.995738882 | 0.998975924 | 0 |
| 131072–196607 | 64 | 0.996201374 | 0.998590803 | 0 |
| 196608–262143 | 96 | 0.994240818 | 0.998337447 | 2 |

All final 33 sampled rows pass individually; their minimum is 0.996876513 and
mean is 0.998486809. At position 262142, prefill PCC is 0.998378523 versus decode
0.999088389. At 262143, prefill is 0.997020468 versus decode 0.997745272. Thus the
remaining two diagnostic misses are not at the final rows already checked by
decode. Bucket means decrease modestly but individual row values fluctuate;
this is not proof of monotonic error growth. Original Q128/K128 LoFi per-row outputs were not saved, so its per-row
delta against HiFi2 cannot be reconstructed.

The new paged Q64/K256 LoFi run exposes substantial position-associated drift:
108 of 291 sampled rows miss .995. Its mean row PCC by the same increasing
context buckets (4096 onward) is 0.998208, 0.997463, 0.995374, 0.993756, and
0.992046, versus HiFi2/K128 means 0.999330, 0.999173, 0.998976, 0.998591, and
0.998337. The final 33 rows average 0.989825, with 20 misses and minimum
0.959289410 at position 262143. Position 262142 is 0.995549563. Decode at these
positions is unchanged and passing. This shows that the distinct full-prefill
attention path can produce large row-specific errors with LoFi at long context.
Both geometry and fidelity differ from the passing HiFi2/K128 comparison.
The subsequently completed HiFi2/Q64/K256 control uses the exact same paged
program and fixture as this LoFi run and passes at 0.9990505377367619. It has
zero misses among 291 rows, minimum 0.996061645296 at position 87039. The final
33 rows have minimum 0.997778215 and mean 0.998962970. End rows 262142 and
262143 are 0.999146879 and 0.998209209, with identical passing decode. Means
from 4096 onward in the same context buckets are 0.999358, 0.999269, 0.999202,
0.999073, and 0.998892. This matched geometry control proves fidelity
sensitivity at the prefill SDPA boundary; it does not identify a specific
incorrect kernel instruction. Larger K improves both fidelities, but the
observed LoFi/Q64/K256 geometry alone does not fix the bug.

## Targeted regression for the demonstrated endpoint failure

The paged Q64/K256 LoFi run gives a concrete bug reproduction at prefill row
262143: PCC 0.959289410 while decode at that same position passes. The model
returns every prefill output, including that endpoint. A future larger-K
candidate could pass a 291-row aggregate while retaining the known endpoint
error, so aggregate-only acceptance is insufficient to prove this bug fixed.

`long_context_tail_regression.patch` is a prepared, **unapplied** test-only
patch requiring the two final prefill rows `{max(0,length-2), length-1}` to meet
.995 against the existing oracle, matching the positions already checked
individually in decode. It preserves the aggregate result in
`prefill_aggregate_passed`, logs `prefill_tail_checks`, and makes the prefill
`passed` field require both. The existing assertion still also requires all
decode checks. This is an explicit new regression for the observed endpoint
failure, not a claim that the historical test already enforced per-row prefill
checks, and not a blanket all-sampled-row bar. Interior diagnostic misses remain
reported without becoming new gates.

The prospective file syntax compiles. Evaluating this proposed policy against
saved scalar diagnostics accepts both HiFi2/K128 and HiFi2/Q64/K256 and rejects
LoFi/Q64/K256's known endpoint. No new hardware execution is required to classify these existing
artifacts under the targeted regression. Apply after the current parent-owned
hardware batch; do not change a harness halfway through a controlled batch.

Only source inspection and arithmetic/JSON analysis were run by this agent.
No model runtime file or kernel was edited. The report supports a local
prefill-fidelity fix and legal geometry controls, not a broad kernel rewrite,
global precision escalation, oracle relaxation, or capability reduction.
