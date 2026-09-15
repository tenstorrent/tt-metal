# BF16 compute / native FP32 state, and QK4 / PV2

Blackhole P100A investigation, September 14, 2026. Isolated experimental
kernels only; production kernels and dispatch are unchanged.

## Results

| Variant | Resident time (ms) | TFLOP/s/core | Resident L2 (%) | PCC |
| --- | ---: | ---: | ---: | ---: |
| Retained main BF16 streaming (earlier matched control) | 275.713 | 1.994 | 20.4106 | 0.97958956 |
| Retained compensated BF16 / FAST | 345.445 | 1.591 | 2.4457 | 0.99970087 |
| New BF16 compute / native FP32 state, blocked pack | 474.167 | 1.159 | 2.4101 | 0.99971216 |
| FP32 streaming, QK2/PV2, cheaper exp | 575.753 | 0.955 | 0.6201 | 0.99998454 |
| New FP32 streaming, QK4/PV2, cheaper subtraction | 587.844 | 0.935 | 0.3637 | 0.99999338 |
| New FP32 streaming, QK4/PV2, full-FP32 subtraction | 631.665 | 0.870 | 0.3510 | 0.99999383 |
| Retained ACCURATE, QK4/PV4 | 723.523 | 0.760 | 0.1776 | 0.99999842 |

The main BF16 control is the prior same-card, same-shape measurement in
`../hifi2-fp32-resident-v1/REPORT.md`; the other controls were remeasured here.
Resident error is deliberately separated from genuine distinct-input error.

The cheaper QK4/PV2 change costs only **2.10% time** versus cheaper QK2/PV2,
while substantially reducing error. It delivers **23.08% more throughput**
than retained ACCURATE (18.75% less time). Full-FP32 subtraction gives a
smaller additional normal-input improvement and helps the common-K stress
case, but reduces the throughput advantage over ACCURATE to 14.54%.

The native-state hybrid is **not a replacement for FAST on these results**:
it takes 37.26% longer for approximately the same resident accuracy. It would
need another 27% time reduction just to tie FAST. This is not a proof that a
better schedule cannot win. The simple single-tile-pack prototype took
532.499 ms (1.032 TFLOP/s/core); restoring blocked packing cuts time 10.95%
with identical tested outputs.
Its reverse-order rerun is 474.180 ms with the same output hash. The retained
single-tile-pack hybrid also reproduced at 532.480 ms. Blocked packing matches
the unblocked hybrid's hashes on all overlapping normal and stress cases.

An ablation keeping full subtraction and the matching HiFi2 denominator but
using QK2/PV2 measures 0.923 TFLOP/s/core and 0.590% resident L2
(`qk2_pv2_fullsub-steady.json`). Thus the QK4 improvement is not merely
the denominator correction. The cheaper QK2/PV2 control already uses the
same matched denominator as cheaper QK4/PV2.

### Device-counter cross-check

Separate eight-Q-repetition profiles agree with the uninstrumented rates.
Counter intervals were checked against kernel timestamps and fit below 2^32
cycles. These are activity counters, not useful-FLOP utilization.

| Profile | Device TFLOP/s/core | FPU active | SFPU active | Both active | Neither active |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid, conservative single-tile pack | 1.032 | 40.61% | 19.41% | 9.20% | 49.18% |
| Hybrid, restored blocked pack | 1.158 | 45.59% | 21.79% | 10.35% | 42.97% |
| QK4/PV2, cheaper subtraction | 0.935 | 54.57% | 38.95% | 30.99% | 37.46% |

For comparison, the prior matched cheaper QK2/PV2 profile had 38.46% FPU
active and 16.54% both-active time. The extra QK4 work substantially raises
overlap while barely changing elapsed time, consistent with much of that
extra math being hidden behind other work. The native-state prototype still
has substantial neither-unit-active time. Its extra format transitions,
direct-to-DST loads and synchronized precision switches are engineering
targets; these aggregate counters do not isolate their individual costs.

The nominal HiFi2 peak is 2.7648 TFLOP/s/core at this clock; HiFi4 peak is
1.3824. Equal-work QK4/PV2 has an ideal mixed-fidelity peak of 1.8432,
before softmax/state work. Its useful-FLOP utilization is about 50.7% of
that mixed peak, or 33.8% when normalized to the HiFi2 peak.

### Distinct-input accuracy

All entries below are L2 percentages; normal rows span seeds 1236 and 1237.
Other rows use seed 1236 at 262,144 keys. Q/K/V are normal unless specified;
`scaled_qk` multiplies Q/K by two; `outliers` adds sparse 10x Gaussian terms;
common-mode cases add 32 before BF16 input rounding.

| Input | BF16 / FP32 state | QK4/PV2 cheap | QK4/PV2 full subtraction | QK4/PV4 ACCURATE |
| --- | ---: | ---: | ---: | ---: |
| Normal, 32K keys | 2.567–2.632 | 0.382–0.402 | 0.368–0.383 | 0.179–0.182 |
| Normal, 256K keys | 3.308–3.323 | 0.387–0.391 | 0.372–0.373 | 0.178–0.180 |
| Scaled Q/K | 5.499 | 0.351 | 0.283 | 0.199 |
| Sparse outliers | 4.847 | 0.286 | 0.249 | 0.231 |
| Large common Q | 0.287 | 0.0150 | 0.0150 | 0.0150 |
| Large common K | 17.159 | 1.079 | 0.803 | 0.730 |
| Constant V=1 | approximately 0 | approximately 0 | approximately 0 | approximately 0 |

Constant-V PCC is undefined, not a failure. Normal 256K PCC is
0.999704–0.999709 for the native-state hybrid, 0.99999231–0.99999236 for
cheaper QK4/PV2, 0.99999291–0.99999297 for full-subtraction QK4/PV2, and
0.99999836–0.99999838 for ACCURATE. Per-case PCCs are retained in the JSONs.

Additional held-out seed 1238, all at 256K keys (L2 percentages):

| Input | BF16 / FP32 state | QK2/PV2 cheap | QK4/PV2 cheap | QK4/PV2 full subtraction | ACCURATE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 3.315 | 0.666 | 0.395 | 0.377 | 0.179 |
| Uniform attention, Q=0 | 2.145 | 0.194 | 0.194 | 0.194 | 0.195 |
| Large common V | 0.673 | 0.0104 | 0.0104 | 0.0104 | 0.0104 |

For large common V, all FP32 variants produce constant BF16 outputs and PCC
is undefined. Their small global L2 is dominated by the common mode and is
not evidence of preserving the small centered signal. Hybrid PCC is only
0.00128 on this case. These remain stress diagnostics, not qualification wins.

The mixed variants meet the 0.5% target on the tested normal inputs without
Q preprocessing. They do **not** universally meet it: common K remains a
counterexample, including for the all-HiFi4 control in this harness. Neither
variant should be presented as universally interchangeable with ACCURATE.

Native FP32 state removes the repeated-resident long-context drift: its
resident error is already about 2.41% after one K chunk and remains about
2.41% after 512. It does not remove BF16 score/probability/bulk accumulation
error, nor all correction and final-reduction rounding. The distinct-input
increase from about 2.6% to 3.3% is important; this is not a claim that all
long-context errors have disappeared.

## Measurement contract

Reservation 219611 on yyzo-bh-08, logical core (0,0), 1350 MHz. Q chunk 256,
K chunk 512, D128, noncausal, original BF16 Q/K/V, no Q preprocessing.
The no-DM benchmark repeats one resident K/V block 512 times and Q 16 times.
Useful FLOPs are 549,755,813,888. Timings are uninstrumented blocking trace
replays, 20 warmups and 10 measurements. Every timed run checks finite outputs
and bit-identical eager/trace results. These are measured per-core rates, not
measured full-chip throughput.

The BF16 variants retain two Q slots and double-buffered K/V. FP32 variants
retain the previous single K/V slot at Q256. No input transfers or L1 copies
occur in the resident steady-state loop; internal score/state traffic remains.
Chunk sizes and the existing input buffering of each path are unchanged.

Separate correctness tests use 256 queries against genuinely distinct 32,768
or 262,144 keys/values, with blocked FP64 attention over the already-rounded
BF16 inputs as the reference. They stream inputs and are not no-DM timing
measurements. They are a bounded numerical investigation, not a full operator
qualification, model evaluation, or Galaxy result. Production improved-mode
guards exclude Q256; this wrapper explicitly instantiates the experimental
streaming kernels without fallback and does not expand production support.

## Implementations

`hybrid`: HiFi2 QK/PV with BF16 bulk destination and BF16 scores/probabilities.
Numerator, column-wise denominator partials, correction factors and reciprocal
scratch are native FP32 CBs; running maxima remain BF16. After bulk PV, one
FP32 recurrent-update section per K chunk loads
the old recurrent state directly into FP32 DST, multiplies by the correction
in the SFPU, and adds it to the current chunk using FP32 L1 accumulation.
The kernel then returns to BF16 DST. The final normalization uses FP32
reciprocal/multiply, but its final-only column reduction reads the FP32 partial
sums through a BF16 view. It is not an all-FP32 softmax implementation.
There is no high/low BF16 compensation in this variant. CB allocation is
1,351,680 bytes, versus 1,333,248 for retained compensated BF16.

`qk4_pv2`: existing cheaper FP32 streaming schedule, with QK initialization and
execution changed to HiFi4 and PV left at HiFi2. It keeps ordinary FPU
score-minus-max subtraction, the biased cubic approximate-exp refiner and
the LoFi denominator matched to the effective HiFi2 PV weights. This matching
matters: merely combining the accurate denominator with truncated HiFi2 PV
introduces a numerator/denominator mismatch.

`qk4_pv2_fullsub`: also uses full-FP32 score-minus-max subtraction. The fused
refiner is given the HiFi2-compatible biased coefficients, and the denominator
is changed to the matching effective-weight reduction. It is not simply the
ACCURATE variant with one matmul fidelity flag changed.

Both QK4/PV2 variants use FP32 destination, FP32 score/recurrent CBs, and the
existing 1x4 FP32 matmul subblocks. The retained ACCURATE control uses HiFi4
for both matmuls, full-FP32 subtraction and its unbiased exp refinement.

## Reproducibility and verification

`run.py` records arguments, compile defines, CB specifications and source hashes
next to each result. `finalize-tests.sh` contains the principal matched runs;
`qualify_mixed.sh` contains the initial distinct-input FP32 suite. Use fresh
labels when rerunning; result files are not overwritten. Historical debug
logs include rejected implementations and must not be treated as valid results.

The native-state microprobe checks FP32 round trips, 512 additions of 2^-20
to one, nonidentity rescaling and cancellation over three seeds. Round trips
and the small-increment case are bit exact; rescaling differs by at most
2.3842e-7 from a single-rounded FP64 reference. Explicit FP32 SFPU load/store
formats are used in the runtime-switched section.

The first row-interleaved hybrid and format-cache experiments were rejected.
The retained implementation batches state updates, explicitly reconfigures
formats, restores SFPU configuration, and stores correction factors in FP32.
With BF16 correction storage, distinct-input testing exposed a last-query-tile
error; widening that CB removes it in the tested cases. This isolates a format
integration issue, not a demonstrated fundamental limit of native FP32 state.

The initial distinct-input attempt on the retained Q256 compensated-BF16
wrapper also failed, so it is not used as a distinct-input correctness control
here. Its resident timing/error control remains reproducible. This does not
replace the earlier Q128 production-path qualification.

Host build completed successfully with
`CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install`
(`build.log`). Experimental kernels also JIT-compile and execute on the card.
Black and clang-format were run on the newly edited Python/C++ sources.
The four pre-existing modified production files retain their starting SHA256s.

Example reruns in the remote checkout (choose fresh labels):

```bash
python_env/bin/python experiments/sdpa-l2/hybrid-mixed-v1/run.py \
  --mode hybrid --hybrid-block-pack --q-repeats 16 --k-chunks 512 \
  --warmup 20 --iters 10 --label fresh-hybrid
python_env/bin/python experiments/sdpa-l2/hybrid-mixed-v1/run.py \
  --mode qk4_pv2 --distinct-kv --q-repeats 1 --k-chunks 512 \
  --seed 1236 --iters 0 --label fresh-mixed-accuracy
```

The home-filesystem quota filled during a final holdout JIT build. That failed
attempt is not a numerical result. Subsequent tests set
`TT_METAL_CACHE=/localdev/cglagovich/tt-metal-blackhole-20260908/generated/sdpa-hybrid-cache-20260914` and use local
disk; no existing cache or user data was deleted.
