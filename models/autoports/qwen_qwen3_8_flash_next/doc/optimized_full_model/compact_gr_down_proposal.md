# Exact GR-down compaction: component repair and failed integration gate

The candidate is implemented in [compact_gr_down.py](../../tt/compact_gr_down.py),
[writer kernel](../../tt/kernels/compact_gr_down_writer.cpp) and
[component probe](../../tests/probe_compact_gr_down.py). Following exact component
checks, candidate `GatheredProjection` dispatch was tested for BF16 TILE/DRAM [1,1,Q,384],
valid rank width80, Q1..32 on Blackhole1x4. Other shapes retain the original
slice/concat path. The owner has reverted the compaction dispatch; the helper
remains an experimental reproduction. Component latency is measured below.
**Rejected at the integrated numerical gate:**
[final serving Watcher v1](final_serving_watcher_v1.json) failed mixed-prefill
logits row0 versus its solo control and closed cleanly. The
[restored exact-fusion serving control](final_serving_watcher_v2.json) passed all16 checks.
Compaction remains unselected in production v4; its full-model performance and
refreshed accuracy are not established.
The selected profile measures a **3.334706 ms/token zero-cost ceiling** for
four slices plus concat across 97 GR reads (two per layer and the final mixer).
The real saving must be smaller. This is padded projection-output traffic,
not KV-cache-capacity traffic.

## Smallest change

Keep the projection, BF16 cast and selected synchronous gather unchanged.
In `tt/multichip_decoder.py::GatheredProjection.__call__`, only the
`valid_width == 80` case compacts four rank-local 96-column padded
outputs into320 columns. The selected replacement removes four slices and one
concat with one data-movement operation plus integer BF16 canonicalization:

```
input:  BF16 TILE DRAM [1,1,Q,384], physical [1,1,32,384]
output: BF16 TILE DRAM [1,1,Q,320], physical [1,1,32,320]
output_bits[q,c] = legacy_canonicalize(input_bits[q, 96 * (c // 80) + (c % 80)])
```

Use the same operation for all `1 <= Q <= 32`, including nonaligned prompt
tails. The final mixer uses the same `GatheredProjection(valid_width=80)`
contract, so one narrow wrapper covers both call sites. Keep other gathered
projections unchanged. Do not alter weights, fold the following multiply,
change collective payloads, or change any rounding boundary in this first
candidate.

## Deterministic tiled copy

Assign one worker exclusive ownership of each of the ten output tiles. All
workers use a distinct 2,048-byte local tile buffer and write complete tiles;
there are no overlapping writers or cross-core assembly. Every tile contains
two 16-column output faces per row half. For output horizontal face `f` in
`0..19`, the corresponding input horizontal face is:

```
input_face = f + f // 5
input_tile = input_face // 2
input_face_in_tile = 2 * row_half + input_face % 2
output_face_in_tile = 2 * row_half + f % 2
```

This skips exactly one 16-column padding face after each five valid faces.
The initial raw-bit-copy assumption was disproved by the legacy control.
The selected slice/concat performs untilize/retilize and canonicalizes BF16
encodings. The native writer reproduces that mapping with integer masks:

```text
exponent = bits & 0x7f80
if exponent == 0:      return 0x0000
if exponent == 0x7f80: return bits & 0xff80
otherwise:            return bits
```

Thus negative zero and subnormals become positive zero; NaNs become signed
infinity; existing infinities and normal finite values keep their bits.
[Exhaustive legacy evidence](gr_compaction_canonicalization_v1.json) finds509
changed encodings out of65536, with no inconsistent mapping and518 Q1
controls per rank. This preserves observed selected behavior; it is not a new
precision policy. No unpack, SFPU arithmetic or floating-point reduction is
needed. Use tiled `TensorAccessor` addresses and face offsets; do not assume
physical DRAM tiles are contiguous.

For Q32, copy four complete 512-byte faces per worker. For a partial row half,
copy each live face row as one 32-byte NoC read. Initialize output padding
deterministically to positive zero; native physical padding is explicitly
read back and checked. Logical canonicalized values must match the legacy
output bitwise; raw source payload preservation is not the selected contract. Read barriers precede writes and the writer issues its
completion barrier before finishing. The output must not alias the input.

Warm every supported physical/program signature before first capture; store
the shape in compile arguments and persistent buffer addresses in runtime
arguments. The same device operation must work on the mesh without host
conversion or mesh-rank inspection during replay. Avoid an output-buffer API
change in the first candidate unless required by existing trace allocation
rules: normal pre-capture TTNN allocation already gives stable trace addresses.

## Discriminating validation and timing

1. CPU check the face-address mapping against direct concatenation for every
   Q1..32 and all 320 columns. Include adversarial raw BF16 payloads; this proves
   the proposed indexing only, not NoC/kernel correctness.
2. Test-local device harness compares logical output bits against the current
   four-slice/concat implementation on all four ranks for Q1,2,3,15,16,17,31,32.
   Use rank-distinct ramps, random BF16 bit patterns, signed zeros, infinities,
   NaNs, and deliberately poisoned per-rank 16-column padding. Check padded
   output separately and distinguish a canonical-zero requirement from
   undefined reference padding. Exercise different input/output allocations
   to catch stale program-cache runtime addresses.
3. Compare real GR-down gathered outputs and the complete following
   multiply/SiLU/up-projection chain at bit level. Preserve explicit BF16
   rounding and the selected exact FP32 projections.
4. Warm/capture before comparison, then change input contents and replay with
   persistent addresses. Run watcher and allocation tracking separately,
   including program-cache allocations, with no exemption. Verify no live
   host boundary or fallback in the operation.
5. Compare clean traced Q1 and Q32 windows for old five-op compaction versus
   the new one-op copy, followed by a reduced complete-layer regression. Only
   wire the candidate after it is correct and measurably faster. Refresh
   full-model accuracy and token-out timings after integration.

CPU indexing proof passed all32 logical Q sizes and320 columns with no
duplicate/missing output addresses. The actual C++ canonicalization helper also
matched all65536 observed encodings and paired-halfword controls. Subsequent
hardware tests validate logical canonicalized BF16 bits and native physical
positive-zero padding; results follow. Real captured GR-chain and full-model
refresh remain integrated evidence requirements, not claims of this synthetic
component probe.

## Evidence and limit

`terminal_profile_analysis_v1.json::candidate_budgets.GR_compact` gives exact
CSV ranges and the weighted ceiling. The first layer has slices at data rows
17:21 and concat at row21; its concat alone takes 30.319 us. The relevant source
is `GatheredProjection` in `tt/multichip_decoder.py` and `FinalMixer` in
`tt/model.py`. Existing gather and projection geometry is preserved, so this
proposal cannot reclaim their costs or the larger exact-projection budget.

## Component results and preserved failures

- [Watcher v1](compact_gr_down_watcher_v1.json), [log](compact_gr_down_watcher_v1.log): failed the legacy-versus-host control at Q1 before native execution because raw payload preservation was assumed.
- [Watcher v2](compact_gr_down_watcher_v2.json), [log](compact_gr_down_watcher_v2.log): added mismatch classification, but finite negative-zero/subnormal differences still failed. Neither failure was waived or relabeled a pass; the historical NaN-payload allowance was superseded by strict v3 equality.
- [Canonicalization diagnostic](gr_compaction_canonicalization_v1.json), [log](gr_compaction_canonicalization_v1.log): observed all65536 encodings; lookup SHA256 `5d279b4bf150f2fc1ca6fae8431f6beae34ec6a2e2097ecaebd314c4c2b71121` (int32 little endian).
- [Watcher v3](compact_gr_down_watcher_v3.json), [log](compact_gr_down_watcher_v3.log): PASS all Q1..32, four ranks, exhaustive Q32 families covering65536 patterns across ranks and518 exceptional/boundary Q1 controls per rank. Baseline, native and independently canonicalized host outputs agree exactly with no exceptions. Changed-input traces, native padding readback, runtime guard and allocation-guard positive controls pass; Watcher10 and tracking include program-cache allocations.
- [Clean v1](compact_gr_down_clean_v1.json), [log](compact_gr_down_clean_v1.log): PASS Q1/3/17/32 plus exhaustive controls, with tracking/Watcher disabled. Source hashes and diagnostic binding are embedded in both successful JSON files.

Clean median milliseconds per replay,64 nonblocking replays per window; input refresh/readback are outside the timed window and there is one synchronization at its end. These measure compaction alone, not gather, projection, downstream arithmetic or full-model decode.

| Q | Four-slice/concat baseline ms | Native canonical compaction ms | Component speedup |
|---:|---:|---:|---:|
| 1 | 0.040585247916 | 0.006272341125 | 6.4705× |
| 3 | 0.040609513235 | 0.006892609235 | 5.8917× |
| 17 | 0.040631908632 | 0.010886826203 | 3.7322× |
| 32 | 0.040602310037 | 0.013978860807 | 2.9046× |

Eligible `GatheredProjection` dispatch was integrated as a candidate after
these component gates, but its integrated mixed-prefill gate subsequently
failed. Production compaction dispatch was reverted; this is not a kept
optimization. The helper is retained for experimental reproduction. It retains the existing projection, explicit BF16 cast, synchronous
gather, downstream multiply and inter-layer residual policy. Module guards
require standard32×32 tiles, exact padded shapes, allocated buffers on the
same validated mesh and nonaliasing output. Full-model post-compaction
performance/accuracy is not established by this report. The historical
3.334706ms profile ceiling and a component speedup are not interchangeable
with a measured full48 gain.

The earlier exact-fusion candidate passed its serving controls; compaction is
the only new production behavior in this failed attempt. That comparison
localizes the next investigation but does not yet prove the root cause.
Historical component PASS results remain valid in their tested scope and do
not override the integrated failure. No implementation edit or device work
was performed during this documentation update.

## Final diagnostic status

[Native-only serving reproduction](compaction_serving_repro_v1.json) fails the
mixed/solo gate again: row0 and its cache match, while row1 has247197 differing
finite logits. [Eager real-operand comparison](compaction_eager_operands_v1.json)
passes all28 calls across Q1/2/3/32, including logical and physical compaction
outputs, final logits, and mutable state. This eager comparator also executes
the legacy compactor before each downstream consumer and synchronizes, so it
can mask LLK-state or allocation/lifetime dependencies. It does not establish
native-only integration correctness. See
[diagnosis v2](compaction_integration_diagnosis_v2.md) for exact scope and the
next discriminating hand-off check. No root cause or full-model gain is claimed;
no new performance experiment was run for this update.
