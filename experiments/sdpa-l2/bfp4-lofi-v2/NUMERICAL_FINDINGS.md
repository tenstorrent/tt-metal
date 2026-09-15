# Measured numerical findings: native-exp LoFi and value centering

## Conclusions and scope

The native-exp FP32/LoFi points have useful ordinary-input accuracy bands,
but neither is uniformly accurate across this stress suite. K outliers,
large common K, and concentrated logits expose much larger errors. Device K
centering repairs the tested common-K low-precision failure. The unchanged
accurate HiFi4 control also has a smaller common-K sensitivity; a new device
diagnostic removes it by centering, while exposing a reason not to center
ordinary inputs unconditionally.

Value centering repairs a real long-context failure of the **denominator-only
compensated BF16** experiment, including constant V. However, merely adding
back the original V mean can badly hurt normal inputs. Correcting the mean
of the actually represented centered V improves that result. Common-V output
can reach its BF16 rounding floor while its residual-relative error remains
over100%; that is not evidence of remaining kernel error of that magnitude.

This report uses existing hardware JSON, plus the explicitly identified CPU
reference-rounding floor. No new device/CPU jobs were run for this report.
No result below establishes model-level quality or parity with SageAttention.
The four locked variants and all drivers are unchanged by this report.

## Metrics: keep the original reference

Let R be FP64 attention of the **original BF16 Q/K/V**, A the device output,
and b the FP64 token mean of original V. We report:

- Original L2: `100*||A-R||/||R||`.
- Residual L2: `100*||A-R||/||R-b||`. Subtracting the **same** b from both
  outputs changes only the denominator, not the error or reference operator.
- BF16 output floor: replace A with `BF16_RNE(R)` under the same metrics.
- PCC: Pearson correlation, undefined when either operand has no variance.

There is no gain fitting, independent output recentering, or substitution of
quantized inputs into the primary reference. Residual L2 is undefined for
constant V and analytically uniform attention with zero output residual.
Display rounding of tiny FP64-reference residuals to0.000% does not indicate
literal zero floating-point reference error.

## Native-exp qualification at32K

Evidence: [native-suite-32k-v1.jsonl](native-suite-32k-v1.jsonl), produced by
`native_exp_qualification.py`: N32,768,H2,D128,Q/K chunks256/512,22 cores,
seeds1240/1241. Each case computes the entire output; reference metrics cover
128 explicitly sampled Q rows per head and **all** K/V rows. All82 records
are finite and trace-replay-identical; the completion record says sources
were unchanged. This is an accuracy-only suite, with no timing measurements.

The B8/B4 columns are FP32-destination/state, LoFi QK/PV, native exp, matched
represented-P denominator, BF16 output. B8 means per-value RNE5 K/V followed
by native BFP8 RNA packing and effective LoFi consumption—not a full
seven-bit-magnitude BFP8 matmul. B4 uses native-group RNE BFP4. Q is RNE7.
The accurate control is unchanged HiFi4 QK/PV with FP32 destination/state,
accurate exp/full subtraction and original BF16 operands.

Each cell shows **original L2% range across two seeds; minimum PCC**. The
worst PCC and range endpoints need not belong to the same seed.

| Distribution | Native LoFi B8/B8 | Native LoFi B4/B4 | Accurate HiFi4 |
|---|---:|---:|---:|
| normal | 2.820–2.835; 0.999598 | 16.681–16.839; 0.985840 | 0.179–0.181; 0.999998 |
| outliers | 7.002–7.491; 0.997199 | 33.637–65.758; 0.770058 | 0.210–0.395; 0.999992 |
| scaled_qk | 4.531–4.647; 0.998948 | 31.976–32.481; 0.946702 | 0.193–0.194; 0.999998 |
| scaled_down | 1.558–1.700; 0.999856 | 11.349–12.019; 0.992785 | 0.169–0.174; 0.999998 |
| biased_v | 0.180–0.181; 0.980917 | 0.315–0.318; 0.966065 | 0.178–0.179; 0.981470 |
| common_q | 1.542–4.552; 0.998986 | 12.763–19.129; 0.984214 | 0.037–0.252; 0.999997 |
| common_k | 45.849–46.039; 0.899391 | 80.147–80.409; 0.594054 | 0.729–0.736; 0.999973 |
| common_v | 0.388–0.389; 0.021296 | 0.029–0.029; — | 0.029–0.029; — |
| constant_v | 0.000–0.000; — | 0.000–0.000; — | 0.000–0.000; — |
| uniform | 1.353–1.508; 0.999886 | 11.323–12.001; 0.992807 | 0.167–0.186; 0.999998 |
| uniform_constant_v | 0.000–0.000; — | 0.000–0.000; — | 0.000–0.000; — |
| channel_k | 12.772–13.615; 0.990710 | 54.309–57.756; 0.835892 | 0.358–0.369; 0.999993 |
| channel_v | 3.282–3.292; 0.999458 | 17.030–17.851; 0.983925 | 0.179–0.181; 0.999998 |

`scaled_qk` multiplies Q/K by2; `scaled_down` by0.25; `channel_k/v` multiplies
every16th channel by32. Common offsets are32. Other distributions follow the
pinned repro generator. These are synthetic operator tests, not activations
captured from a deployed model.

Important conditional observations:

- B4's normal16–17% band does not extend to sparse outliers or K-channel
  outliers. B8 is substantially better, but its7–14% stressed errors still
  exceed its ordinary2.8% band.
- Uniform/scaled-down attention makes the represented V mean important.
  For scaled-down seed1240, B8 original L2 is1.558% but residual L2 is24.962%;
  B4 is12.019% versus192.534%. Accurate residual L2 is2.788%, already close
  to the2.699% BF16-output floor under that stricter normalization.
- A low original L2 or low PCC can be misleading with offset-dominated V.
  For biased-V seed1240, accurate original L2 is0.178%, residual L2 is24.152%,
  and PCC is0.981874; the residual BF16 floor is24.144%. Almost all of that
  residual-relative loss is unavoidable in this output dtype.
- Common-V B4 and accurate outputs both reach the0.029% original-L2 floor;
  this does not establish that B4 retained the subtle attention-dependent
  signal. B8's0.388–0.389% is instead genuinely above that output floor.

## Native-exp qualification at256K

[native-suite-256k-v2.jsonl](native-suite-256k-v2.jsonl) is now complete:
the same H2/22-core, two-seed, thirteen-distribution design, with82 finite,
trace-identical hardware records and source-unchanged completion. All shared
source pins match the32K suite except the qualification driver itself;
kernel and reference pins match. Again, there is no timing in this suite.
Cells use the same **L2% range; minimum PCC** convention.

| Distribution | Native LoFi B8/B8 | Native LoFi B4/B4 | Accurate HiFi4 |
|---|---:|---:|---:|
| normal | 2.774–2.803; 0.999607 | 16.587–17.038; 0.985727 | 0.178–0.180; 0.999998 |
| outliers | 10.349–15.660; 0.987754 | 38.724–46.798; 0.891017 | 0.312–0.350; 0.999994 |
| scaled_qk | 4.939–5.429; 0.998587 | 35.196–35.930; 0.934753 | 0.198–0.200; 0.999998 |
| scaled_down | 1.614–1.625; 0.999869 | 11.539–12.197; 0.992923 | 0.170–0.173; 0.999999 |
| biased_v | 0.176–0.179; 0.872015 | 0.265–0.265; 0.891769 | 0.176–0.178; 0.874077 |
| common_q | 3.695–26.191; 0.965083 | 48.998–89.751; 0.570216 | 0.155–0.502; 0.999987 |
| common_k | 44.824–45.438; 0.902410 | 79.922–80.479; 0.592947 | 0.730–0.733; 0.999974 |
| common_v | 0.389–0.389; — | 0.010–0.010; — | 0.010–0.010; — |
| constant_v | 0.000–0.000; — | 0.000–0.000; — | 0.000–0.000; — |
| uniform | 1.482–1.491; 0.999889 | 11.504–12.189; 0.992943 | 0.171–0.175; 0.999999 |
| uniform_constant_v | 0.000–0.000; — | 0.000–0.000; — | 0.000–0.000; — |
| channel_k | 15.255–15.359; 0.988179 | 62.303–64.048; 0.796424 | 0.362–0.457; 0.999990 |
| channel_v | 2.878–3.134; 0.999509 | 15.357–16.380; 0.986486 | 0.178–0.185; 0.999998 |

Normal-input accuracy remains in essentially the32K bands. Common-Q and
outlier failures remain strongly input-dependent; B4 common-Q reaches89.751%
for seed1241, while B8 reaches26.191% for seed1240. This is not a controlled
proof of monotonically growing recurrent-state error: longer runs change
the K/V population and sampled Q rows, and near-winning logits can change.
Accurate common-Q seed1240 is0.501528%, just above0.5%; this deserves separate
attribution rather than rounding it down to claim a universal pass.

Large-V-offset metric effects become stronger. For biased-V seed1240,
accurate original L2 remains0.176% and residual L2 rises to67.822%, against
a67.802% BF16 residual floor. Its PCC0.877289 is largely a representability
effect. Common-V B8 has roughly0.389% original L2, well above the0.010% floor,
even though PCC is now undefined for its constant output. Undefined PCC
alone cannot distinguish a correct constant rounded output from a biased one.

## Common K: centering and the accurate-control diagnostic

The same suite includes a device-generated K mean and fused FP32
subtraction/quantization for the low-precision common-K cases. These remain
compared against original BF16 inputs. Values below are seeds1240 /1241.

| Variant | K centering | Original L2 (%) | PCC |
|---|---|---:|---:|
| LoFi B8/B8 | Off | 45.849 / 46.039 | 0.899972 / 0.899391 |
| LoFi B8/B8 | On | 2.376 / 2.411 | 0.999718 / 0.999710 |
| LoFi B4/B4 | Off | 80.409 / 80.147 | 0.594054 / 0.598066 |
| LoFi B4/B4 | On | 18.297 / 18.074 | 0.983352 / 0.983860 |

The256K common-K centering checks retain the repair: B8 reaches2.347/2.357%
and B4 reaches18.228/17.980%, against44.824–45.438% and79.922–80.479%
uncentered. The original-input reference is unchanged.

Subtracting a token-constant K vector adds a row-constant score shift, so it
is an exact softmax invariance before additional rounding. Here it prevents
the large common component from consuming shared-exponent precision. The
measured improvement is strong, but does not cure other outlier mechanisms.

Accurate common-K L2 is0.736/0.729%, versus0.181/0.179% normal and a common-K
BF16 output floor of0.166/0.166%. This excess is **not** explained by final
BF16 rounding. Product alignment/finite hardware arithmetic amplified by
the large offset is a plausible explanation, not yet an established causal
attribution from these records. The full-subtraction/accurate-exp control
already removes the most obvious low-quality softmax alternatives.

New hardware evidence from `accurate-kcenter-1024-v1.jsonl` and
`accurate-kcenter-32k-v1.jsonl` isolates the shift with the **unchanged accurate
kernel**, H2,seed1240. The1024 run references every output row;32K references
128 sampled rows/head. An immutable original-K device backup feeds a device
BF16 mean and accurate SFPU subtraction into the kernel's private K buffer.
All preprocessing oracles, immutable-input checks, and repeated traces pass;
both completion records verify unchanged sources.

| Length / input | Original L2 off→on (%) | PCC off→on | BF16-shift attention drift (%) | Centered kernel vs shifted-BF16 reference L2 (%) |
|---|---:|---:|---:|---:|
| 1024 / normal | 0.178→0.241 | 0.999998→0.999997 | 0.163 | 0.178 |
| 1024 / common K | 0.733→0.178 | 0.999974→0.999998 | 0.000 | 0.178 |
| 32K / normal | 0.181→0.240 | 0.999998→0.999997 | 0.159 | 0.181 |
| 32K / common K | 0.736→0.178 | 0.999973→0.999998 | 0.000 | 0.178 |

Exact FP64 shift invariance holds within2.2e-13% L2. Common-K subtraction
introduces **zero BF16 K-rounding error** for these inputs, so its0.736→0.178%
improvement is a genuine reduction of kernel sensitivity to an algebraically
irrelevant common offset. This strongly supports a hardware arithmetic
conditioning issue, but does not by itself distinguish product alignment
from every accumulation/subtraction effect or prove a particular FPU stage.

Normal inputs illustrate the engineering tradeoff: the extra centered-K
BF16 spill causes0.159% attention drift at32K. Kernel error against those
actual shifted inputs stays0.181%, but original-reference error rises to0.240%.
Conditional centering for substantial common K is therefore better supported
than enabling this BF16 preprocessing universally. At32K, mean+subtraction
costs0.619 ms; combined timing changes69.194→69.812 ms for normal and
69.123→69.807 ms for common K. The mean cost includes its currently unused
repeated-bias materialization; these are diagnostic costs, not optimized
production overhead. No256K accurate-centering result is claimed here.

## V-centered BF16 experiment at256K

Evidence: `valuecenter-262144-{none,original_mean,matched_mean}-{normal,common_v,constant_v}-v1.json`.
These nine hardware runs use H10,110 cores,D128,Q/K256/512,seed1240,
128 sampled Q rows/head. The variant is LoFi K8/V4,
`destination=fast_bf16, denom_only=true`: **denominator compensation, no
numerator compensation**. It is neither the FP32 suite above nor the locked
fully compensated FAST variant. All nine records have identical common
source pins and replay-identical outputs.

Original-mean mode centers V using its actual device BF16 mean, quantizes
the residual, and adds that mean back after attention. Matched-mean mode
also decodes the actual packed V4, measures its represented mean, and removes
that mean from the output bias. It corrects an **unweighted** represented
mean; it does not reconstruct attention-weighted quantization error. Both
centered paths have a BF16 core-output spill and BF16 epilogue result.

| Input | V treatment | Original L2 (%) | Residual L2 (%) | PCC | Combined ms |
|---|---|---:|---:|---:|---:|
| Normal | None | 12.566 | 15.806 | 0.992396 | 1610.8 |
| Normal | Original mean | 42.444 | 53.391 | 0.909043 | 1612.3 |
| Normal | Matched mean | 10.323 | 12.985 | 0.995011 | 1624.7 |
| Common V | None | 25.466 | 315880.347 | -0.000490 | 1576.9 |
| Common V | Original mean | 0.010 | 129.260 | — | 1615.9 |
| Common V | Matched mean | 0.010 | 129.260 | — | 1616.2 |
| Constant V | None | 25.467 | — | — | 1577.8 |
| Constant V | Original mean | 0.000 | — | — | 1510.6 |
| Constant V | Matched mean | 0.000 | — | — | 1519.9 |

This is a real constant-V failure in the uncentered denominator-only BF16
experiment: a normalized weighted average of all ones should be one. Its
25.467% loss is not a residual-denominator artifact. V centering removes
the common component from the low-precision numerator loop and restores it
in the epilogue. This result should not be generalized to every BF16 or
FP32 path: the32K FP32 suite's constant-V controls are already correct.

The normal original-mean regression is equally important. The1024 matched
preprocessing smoke records explicitly measure represented-centered V mean
bias and its correction: mean reconstruction max error falls0.010691→0.000284.
The normal32K attention check also improves12.046% uncentered→9.815% matched,
whereas original-mean-only gives16.093%. This supports testing matched
represented-mean correction, not blindly enabling original-mean centering.
The measured256K improvement does not isolate every source of recurrence
error or guarantee that matched correction helps other distributions.

Timings include real preprocessing and epilogue; useful attention FLOPs
exclude their arithmetic. On normal256K, preprocessing is9.0/12.7/19.9 ms
for none/original/matched and centered epilogues cost about5.3 ms. Combined
traces are independently measured aggregates, not sums of stage medians.
Core timing drifts across these runs, so small end-to-end differences need
interleaved A/B measurements before a precise overhead claim. The faster
constant-V centered case is degenerate, not a general throughput prediction.

## V8 centering follow-up

New hardware records `valuecenter-b8-{32768,262144}-{none,original_mean,matched_mean}-{normal,common_v}-v1.json`
use the same H10,seed1240,denominator-only BF16 compensation, now K8/V8.
The matched path decodes packed V8 **and truncates it to the actual five-bit
LoFi-consumed value before estimating its mean**. Using the decoded BFP8
mean without this truncation would correct a different operator.

| Length / input | V treatment | Original L2 (%) | Residual L2 (%) | PCC | Combined ms |
|---|---|---:|---:|---:|---:|
| 32K / normal | None | 3.318 | 4.155 | 0.999478 | 26.6 |
| 32K / normal | Original mean | 11.484 | 14.380 | 0.993459 | 27.8 |
| 32K / normal | Matched mean | 3.160 | 3.957 | 0.999529 | 29.0 |
| 256K / normal | None | 4.599 | 5.785 | 0.999178 | 1702.6 |
| 256K / normal | Original mean | 40.693 | 51.189 | 0.916147 | 1704.0 |
| 256K / normal | Matched mean | 4.404 | 5.539 | 0.999258 | 1711.1 |
| 32K / common V | None | 1.507 | 6587.084 | 0.019055 | 26.6 |
| 32K / common V | Original mean | 0.029 | 125.228 | — | 27.8 |
| 32K / common V | Matched mean | 0.029 | 125.228 | — | 29.0 |
| 256K / common V | None | 25.675 | 318472.737 | -0.000199 | 1669.0 |
| 256K / common V | Original mean | 0.010 | 129.260 | — | 1687.0 |
| 256K / common V | Matched mean | 0.010 | 129.260 | — | 1695.5 |

V8 does not eliminate the dangerous original-mean-only regression. Matched
correction recovers its normal band, but the improvement over uncentered V8
is modest relative to the extra work. Common-V centering again reaches the
same BF16-output floor; that repair is not specific to V4. At256K normal,
matched preprocessing costs23.6 ms versus9.1 ms uncentered and adds a5.2 ms
epilogue. Small combined-time differences still need interleaved verification.

All twelve records have replay-identical outputs and exact epilogue checks;
long-input preprocessing-oracle checks are disabled. Source pins agree
within each length's comparisons, and these newer V8 records include the
reference helper. These twelve rows are the earlier normal/common-V subset;
the subsequently completed long constant-V V8 controls are reported in
[the final value-smoothing note](VALUE_SMOOTHING.md). Their conclusions are
measured separately, not inferred from common-V results.

## Why common-V residual error can exceed100% at the best BF16 output

Evidence: [valuecenter-reference-floor-v3.jsonl](valuecenter-reference-floor-v3.jsonl).
This is a **CPU FP64-reference/BF16-rounding calculation**, not an additional
hardware result. It regenerates each selected device record's original
inputs and exact sampled rows. All three referenced-record SHA256 hashes
match the current JSON artifacts.

| Input / length | Best BF16 original L2 (%) | Best BF16 residual L2 (%) | Residual-reference RMS | Rounded reference |
|---|---:|---:|---:|---|
| Normal /32K | 0.166 | 0.208 | 0.007306 | 2923 distinct sampled values |
| Common V /32K | 0.029 | 125.228 | 0.007322 | All sampled values32 |
| Common V /256K | 0.010 | 129.260 | 0.002580 | All sampled values32 |

The centered device common-V errors hit these floors. BF16 spacing is0.125
just below32 and0.25 just above32, much larger than these residual signals.
Even ideal attention rounded to BF16 loses the tiny variation. PCC becomes
undefined for the constant rounded output; replacing it with1 or treating
the undefined value as a numerical failure would both be misleading.

This does **not** make the uncentered25.466% common-V loss acceptable. It
distinguishes that large, avoidable error from the0.010% best-representable
output. If preserving the residual itself is required, inspect an FP32
output or a separately represented residual—not a shifted primary reference
that silently changes the requested BF16-output operator.

## Evidence caveats and next decisions

The32K native suite pins the reference helper and records source stability,
but `check_preprocess=false`: it does not rerun exact preprocessing oracles
for all82 cases. The long V-centering runs likewise omit those expensive
oracles, while1024 smokes have exact Q/K/centered-V checks and all centered
long records have exact BF16 epilogue checks. Long V records originally
omitted the reference-helper source pin. The v3 floor artifact documents
that its current helper hash matches independently verified locked checkpoint
637d956; that is a retrospective provenance check, not a missing original
pin magically recovered. Current source files may have evolved since these
runs; use recorded hashes when reproducing them.

Recommended next steps, without declaring a new production SKU:

1. Pursue conditional accurate K centering: the diagnostic repairs common-K
   sensitivity, but its extra BF16 spill hurts normal-input accuracy. Extend
   its activation/length coverage and investigate the hardware stage before
   asserting product alignment as the sole cause. Preserve the original
   reference as primary and retain the shifted-input attribution.
2. Keep B8/LoFi and B4/LoFi as explicitly conditional candidates. Qualify K
   centering, asymmetric K/V formats, and transform policies on the same
   captured layer/head activations; do not extrapolate normal-input bands.
3. For V4, prioritize matched represented-mean correction over original-mean
   restoration alone. Compare its real cost with numerator compensation and
   V8 using identical inputs and interleaved timings. The V8 follow-up above
   narrows the numerical benefit of matched correction on normal inputs;
   its completed long constant-V control and final interpretation are in
   [the value-smoothing note](VALUE_SMOOTHING.md).
4. Keep original L2 and PCC plus residual/floor diagnostics. Use absolute
   error and the known constant-output identity when residual/PCC are
   undefined. Do not turn the arbitrary0.5% goal into a dtype-impossible
   residual criterion, or hide genuine excess error behind the floor.
5. Before model-quality claims, obtain real Q/K/V captures, layer/head and
   context metadata, softmax sharpness/margins, and task/evaluation results.
   These operator stress tests identify risks and mechanisms; they do not
   measure downstream model fidelity.
