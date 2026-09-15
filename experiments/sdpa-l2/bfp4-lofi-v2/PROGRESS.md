# Low-precision attention continuation — intermediate evidence

Eight-hour research window:2026-09-15 05:48:43–13:48:43 UTC.
This is an intermediate notebook, not a production recommendation. The four
selected SDPA variants and their frozen sources remain unchanged.

## One-pass LoFi, with input rounding

Blackhole LoFi consumes7 significant bits of logical-left Q/P and5 of
logical-right K/V. RNE Q to7 and K/V to5 before matmul removes much of the
one-sided truncation error. Storage can remain BF16; K/V can instead use
BFP8 with a little extra quantization error. FP32 recurrent state prevents
the long-context drift seen in uncompensated BF16.

Q128/K512/D128, distinct normal BF16 inputs, seed1240:

| Variant | 4K L2% | 32K L2% | 256K L2% |
|---|---:|---:|---:|
| LoFi FP32, pre-rounded BF16 storage, cheap exp |2.051|1.986|2.073|
| Same, BFP8 K/V storage |2.150|2.088|2.192|
| LoFi compensated BF16, HiFi2 state/output multiplies |pending|2.957|3.599|

These use original BF16 inputs for the FP64 reference; preprocessing loss
is included in L2. Output is BF16. Full results include PCC and relative
errors in `fused-suite-v1.jsonl` and `fused-safe-*.jsonl`.

Q256/K512/D128, repeated resident Q/K/V, one core, q_repeats16/k_chunks512:

| Candidate | TFLOP/s/core |
|---|---:|
| LoFi FP32, accurate exp/subtract, pre-rounded BF16 |1.001|
| LoFi FP32, cheap exp/subtract, pre-rounded BF16 |1.039|
| Same cheap FP32 with BFP8 K/V |1.035|
| LoFi compensated BF16 with HiFi2 state/output multiplies |1.891|
| LoFi uncompensated BF16 with HiFi2 state/output multiplies |2.488|

The last variant retains severe long-context BF16-state drift and is not
an accuracy recommendation. These timings exclude preprocessing and
recurring input data movement; they are not measured chip throughput.
Same-session selected controls reproduced1.9941/1.5914/0.9352/0.7598TF/core
for main/FAST/balanced/accurate respectively.

## Important integration findings

- Global LoFi also lowers recurrent broadcast-multiply precision. Keeping
  those small operations HiFi2 reduced uncompensated256K error from~202%
  to18.28%, and compensated error4.23→3.60%, at negligible measured cost.
- Frozen streaming code assumes Q and max use the same BF16 format. BFP8 Q
  needs an explicit SrcB format restore after max/subtract operations.
  This is fixed only in the private v2 header.
- Device RNE preprocessing:110cores,256K×128 elements, about0.38ms toBF16
  or0.27ms toBFP8; measured read+write bandwidth~350/379GB/s. Device-prepared
  inputs reproduce the fused one-pass numerical results.
- Tiny-value RNE ties initially exposed `copy_init`'s BF16 zero-flag mode:
  FP32-DST datacopy uses ELWADD, whose nominal zero is not zero under that
  mode. Clearing the source-zero flag after initialization fixed the four
  wide-exponent probes exactly. The separate v1 model clamp below2^-100
  was handled by exact groupwise power-of-two scaling in the test oracle.
  This is not a claim of subnormal/exceptional-value support.

## BFP4 residual direction

K/V≈BFP4(K/V)+BFP8(RNE5(residual)); Q/P use per-valueRNE7.
Both components fit LoFi. Two products per QK and per PV are required,
with1664B/tile total K/V storage versus2048B/tile BF16.

Unfused device LoFi QK/PV plus CPUFP32 softmax/state,4K, two seeds:

- Normal residual4+8 L2:0.669–0.671%, vsmodel0.670–0.671%.
- Outliers:1.033–1.073%; scaledQK:1.122–1.269%.
- Two native device-packed BFP4 components are much worse (~4.5% normal);
  a wider residual is important because the native BFP4 packer is biased.

Fused residual prototype runs at0.669%L2/PCC0.9999776 on4K normal; its
no-P-round ablation runs at0.681%L2. Explicit P rounding initially had an
SFPU address-modifier integration bug: exp leaves auto-increment active,
while SFPI advances DST explicitly. Resetting that modifier fixes it.
Resident timing is0.6090TF/core with explicit P rounding and0.8756 without;
resident L2 is0.6259%/0.6347%. This is not yet a Pareto improvement over
balanced0.9352TF/core. Longer-context validation is in progress.

## Preprocessing is not automatically harmless

Centering K removes its common-mode quantization issue without an output
correction. Centering Q requires restoring the higher-precision Q-mean×K
score correction. Centering V before coarse rounding can introduce a
shared quantization-error mean: normal256K model error rose to~40%.
Restoring the original V mean is insufficient; matching the represented
V mean removes that bias (~1.86% one-pass and0.605% residual4+8 in one
centered256K model case). This remains a model result to validate on device.

## Unbiased native BFP4 packing

A device SFPU pre-rounder now implements group-of16 RNE plus saturation
before native packing. Normal, tie-threshold, wide-exponent and zero probes
all match independent integer-alignment and FP32 arithmetic oracles exactly
(524,288values each); wide also passes FP32 DST. Supported group exponents
are[-124,106], excluding nonzero subnormal inputs and exceptional values.
110core,256K×128:0.26190ms/328.31GB/s read+write,33,554,432exact matches.
This is one preprocessing pass, not attention throughput. Sources/results
are in `bfp4_round.py` and `bfp4_round/`.

## Full-chip harness status

The experimental common-reader harness passes main and accurate on4K,
heads2/cores4 (2.5157%/0.1798% L2 on explicit sampled Q rows). LoFiFP32
BFP8K/V also passes2.1703%; all three device preprocessing tensors were
verified exactly. FAST-family output initially failed the accuracy gate.
The cause is a stale SFPU address modifier after compensation, exposed by
distinct-input Q256 recurrence. A private state reset repairs it: FAST2.4823%,
LoFi/B8 compensated3.0666% on4K/heads2/cores4. Frozen sources are unchanged.
See `FAST_Q256_CORRECTION_BUG.md`. Failed finite-output timings are INVALID
candidate evidence; prior resident and Q128 checks cover only their stated
cases, not this Q256 recurrence.

Initial110-core common-reader results (H10,N8192,Q256,K512,D128), including
device preprocessing: main69.64,FAST65.90,ACCURATE68.70,LoFi-compensated-B8
92.71,LoFi-FP32-B8 88.86TF/chip. These are memory-limited and are NOT
production-dispatch results. The raw JSON separates attention/preprocessing/
combined traces and explicitly records sampled-Q accuracy scope. The simple
reader needs investigation before claiming practical speedups.

Next: improve the common reader, scale full-chip controls, evaluate
host-RNE BFP4 residuals and device preprocessing integration.

## Update: chain reader and long full-chip runs

A production-style unicast K/V forwarding chain per head substantially reduces
redundant DRAM traffic. Q256/K512/D128, noncausal, H10, 110 cores; all numbers below
include device input preprocessing. Accuracy uses 128 explicitly sampled Q rows
per head against FP64 using all original BF16 K/V. These are experimental common
harness measurements, not production-dispatch performance.

| Candidate | 32K L2% | 32K TF/chip | 256K L2% | 256K TF/chip |
|---|---:|---:|---:|---:|
| Main BF16 HiFi2 | 2.785 | 172.98 | 18.781 | 153.22 |
| FAST, private correction-address reset | 2.614 | 168.11 | 3.296 | 143.16 |
| Balanced FP32 QK4/PV2 | 0.390 | 97.06 | 0.390 | 88.42 |
| ACCURATE FP32 QK4/PV4 | 0.179 | 79.26 | 0.179 | 71.01 |
| LoFi compensated BF16, RNE5 BFP8 K/V | 3.151 | 190.10 | 3.735 | 190.63 |
| LoFi FP32, RNE5 BFP8 K/V | 2.152 | 105.14 | 2.151 | 111.10 |
| LoFi compensated BF16, RNE BFP4 K/V | 16.938 | 191.28 | 17.168 | 200.04 |
| LoFi uncompensated BF16, RNE BFP4 K/V | 16.961 | 230.87 | 26.264 | 233.39 |
| LoFi FP32, RNE BFP4 K/V | 16.789 | 105.54 | 16.853 | 111.48 |

Sources: `chain-h10-{32768,262144}-*-v1.json`. Warmup3/iterations7;
Q/K chunks and input buffering unchanged. Compensated BF16 BFP4 is approximately
the same speed as BFP8 here, despite much greater representation loss. Plain
BF16 state is faster but retains long-context drift. These data do not establish
model-level tolerability of the roughly17% BFP4 attention error.

## Update: fully device-prepared residuals

Qualified fused preprocessing emits unbiased BFP4 first components and either
BFP4 or RNE5 BFP8 residual components. Input is read once. All component outputs
match independent normal/tie/wide/zero oracles exactly. For 256K×128 on110 cores,
2×BFP4 preprocessing is0.3657ms, 3×BFP4 is0.5364ms, and BFP4+BFP8 is0.3557ms.

Fused Q128/K512 attention, FP32 DST, LoFi, cheap exp, no extra P-rounding pass:

| Device-prepared K/V | 4K L2% | 32K L2% | 256K L2% |
|---|---:|---:|---:|
| Two unbiased BFP4 components | 1.545 | 1.498 | 1.580 |
| Unbiased BFP4 + RNE5 BFP8 residual | 0.599 | 0.587 | 0.601 |

The latter still costs two matmuls per QK and PV. Its cheap/no-P-round resident
implementation measured0.9137 useful TF/core (older biased first-component
input preparation); it is not yet a speed/accuracy Pareto improvement over the
selected balanced control. Skipping redundant same-format reinitialization for
the two-BFP4 form improved resident0.9147→0.9367TF/core with identical output.

The next numerical experiment prescales Q near1.003 before RNE7 and divides the
attention scale by the same factor. CPU modeling predicts about0.50% rather than
0.59% L2 by breaking BF16-to-7-bit halfway ties; this is not a robust <0.5% claim.
The device prescale kernel passes normal/wide and BF16-DST exactness probes.
Device attention validation is in progress.

## Update: Q-prescale and fullchip residual measurements

Device Q128, original BF16 reference, residual4+8, alpha1.0028:
4K/32K/256K L2=0.5150/0.5031/0.5237%. Alpha1.003 gives
0.5198/0.5078/0.5274%. Treat this as a near0.5% scheme, not a <0.5% contract.

Fullchip Q256/K512/H10/110cores chain, device-preprocessed residual4+8,
alpha1.0028, BF16 output:

| Length | L2% (128 sampled Q rows/head) | Attention TF/chip | Including preprocessing TF/chip |
|---:|---:|---:|---:|
| 8,192 | 0.5164 | 66.58 | 62.35 |
| 32,768 | 0.5176 | 68.60 | 67.51 |
| 262,144 | 0.5153 | 70.66 | 70.53 |

These are **useful** attention FLOPs; executed QK/PV work is2× higher.
This is not a Pareto improvement over balanced FP32 QK4/PV2 on normal inputs.
All-output1024/heads2/cores6 tests also pass; five input component tensors
match their independent preprocessing oracles exactly, including uneven chain
job counts and Q prescaling. See `fullchip_residual/` raw records.

## Update: device centering and BFP8 ties

`center_preprocess.py` fuses full-FP32 subtraction of a BF16 column bias with
RNE quantization, avoiding an intermediate centered-BF16 spill. All ten
mode/distribution probes (BFP4/BFP8 × normal/common/threshold/wide/zero) match
exactly, including head-boundary crossings. BF16 spilling is observably different:
on normal data it changes12,745 BFP4 outputs and80,185 BFP8 outputs of1,572,864.

`center_mean.py` creates that bias on device. AtH10/256K:

| Device mean method | Mean ms | Mean + fused center/quantize ms |
|---|---:|---:|
| BF16 input, HiFi4 FPU, FP32 DST | 3.6685 | 6.3132 |
| Device widening + FP32 SFPU mean | 11.0788 | 13.7225 |

BF16-input mean's reciprocal scaler is truncated BF16 (exact for these powers
of two), but any token-constant K bias preserves exact softmax invariance.
The main concern is effective centering, not exactness of the mean itself.
Fullchip normal/common-K tests using device means and preprocessing pass.

`bfp8_round.py` now implements shared-exponent RNE7 with native pack identity.
Normal, thresholds, wide, zeros and FP32-DST-wide probes pass524,288 exact values
each. Group exponent contract[-120,110], no nonzero subnormals/exceptional values.
RNE removes ties-away bias; saturation remains. HiFi2+BFP8 K/V with this packer
gives1.25245% L2 andgain0.999739 on4K/H2/Q256, versus roughly1.72%/+1.06% gain
for the earlier native-packed control. Separate pipeline attribution is in
progress because the CPU single-stage native-BFP8 model explains only part of
that gain. Do not conflate different native pack/DST conversion configurations.

## Update: compressed intermediate P investigation

Scalar copy→pack throughput is essentially unchanged across BF16/BFP8/BFP4/FP32
outputs: approximately50.8cycles/tile with BF16 DST and49.3 with FP32 DST at
the assumed1350MHz clock. This is a whole copy/unpack/pack/CB pipeline measurement,
not isolated packer throughput. Raw data: `pack_resident/*v3.json`.

Width4 default BFP packing fails exactness because its MOP closes only once
while exponent framing is configured for one tile. The experiment explicitly
rejects it; no BFP width4 performance claims. Headerless BF16/FP32 controls pass.

Padded2048/4096-byte BFP4/BFP8 L1 pages roundtrip exactly for BF16/FP32 DST in
all eight controls. This establishes page addressing, not safe alias scheduling.
Compact score/P aliases are unsafe under existing tile traversal.

`p8_streaming.py` uses a separate compact BFP8 P CB, matched packed-P denominator,
scalar packing and the unbiased cheap exp fit. Q/K/V buffering stays unchanged.
Initial all-output1024/onecore: FP32-P2.1810%L2 vsBFP8-P2.3764%. Both pass trace
replay, but this is not yet a speed improvement. Larger and resident tests follow.

## Update: BFP8 attribution resolved and negative optimization results

The generic compute descriptor defaults to `bfp8_pack_precise=False`: it rounds
each value to E8M6 ties-away **before** shared-exponent BFP8 rounding. The supported
`True` setting removes the first rounding. High-level BF16→BFP8 typecast already
sets it; host BFP8 conversion instead uses ties-even. Four normal/tie pipeline
controls match independent integer models exactly. This explains the formerly
unattributed gain, not a softmax-denominator bug. On32K/H10, HiFi2 native BFP8
single rounding gives1.3140% L2/gain1.00536, versus~1.724%/1.01063 for double
rounding. Shared-exponent RNE remains preferable numerically.

Inlining P8 scalar packing recovers about half its initial slowdown. At fixed
Q256/K512/D128, FP32 LoFi resident throughput is1.03467TF/core with FP32 P versus
1.00302 with inlined, precise BFP8 P; L2 is2.1942% versus2.3793% on the resident
reference. On32K/H10/110cores, FP32 P gives107.46TF/chip and2.1516% L2 versus
104.33TF/chip and2.3499% for BFP8 P. Both constant-V controls reproduce the constant
exactly. This P8 implementation is currently dominated on speed and accuracy.

BF16 LoFi bulk compute with FP32 recurrent state also fails to improve the
frontier:1.305TF/core, with2.916/2.921/3.601% L2 on distinct4K/32K/256K normal
inputs (BF16 K/V storage). Compensated BF16 reaches~1.891TF/core in the same
resident setup and similar accuracy. Neither negative result changes the four
locked selected variants.

A private cheap-exp FP32 HiFi4 diagnostic exposed an independently fixable
denominator initialization error: the phase0/2 override requires a two-phase
MOP, even without `SDPA_DIAG_EXP_MODE`. The v2 header now selects it whenever
`SDPA_DENOM_PHASES` is defined and statically checks the phase override. The
repaired resident diagnostic passes1.9335% L2 at0.81556TF/core. The failed
pre-fix run produced no accepted timing; frozen selected kernels are unchanged.

## Update: cheaper FP32 exp and asymmetric K/V

The FP32 cheap-exp path actually uses the polynomial refiner; BF16 FAST instead
calls native approximate exp directly. Degree1/2 FAST smoke flags initially
reached no changed arithmetic and produced identical output. These are explicitly
inapplicable controls, not optimizations; host guards now reject this combination.

Q256/K512/D128 resident, LoFi FP32, Q7/BF16 and RNE5/BFP8 K/V:

| Exp implementation | TF/core | Resident L2% |
|---|---:|---:|
| Cubic | 1.03469 | 2.1373 |
| Quadratic | 1.08652 | 2.1527 |
| Linear | 1.14383 | 2.8807 |
| Native, no refiner or repeated grid init | 1.56205 | 2.9375 |

Distinct Q128/4K native-exp smoke gives2.8554% L2/PCC0.9995947. The native
variant preserves accurate online correction exp and the matched represented-P
denominator; changing correction exp would introduce a different recurrent bias.
Fullchip/native and broader qualification are pending.

Independent K/V format smoke tests pass all12 destination/format combinations,
including exact device quantization checks and every output row atN1024/H2.
At32K/H10/110cores, chain reader, including preprocessing:

| K/V storage | MAIN LoFi L2% / TF | Full compensation L2% / TF | Denominator-only L2% / TF |
|---|---:|---:|---:|
| BFP8 / BFP8 | 3.287 / 206.89 | 3.151 / 190.10 | 3.318 / 206.93 |
| BFP4 / BFP8 | 12.397 / 218.46 | 12.367 / 190.93 | 12.399 / 218.02 |
| BFP8 / BFP4 | 12.041 / 220.59 | 12.007 / 190.94 | 12.046 / 218.59 |
| BFP4 / BFP4 | 16.961 / 230.90 | 16.938 / 191.30 | 16.960 / 226.92 |

All have Q256/K512/D128 and fixed double-buffered K/V. Denominator-only retains
the corrected FAST correction-exp/address handling and compensated denominator,
but ordinary BF16 output recurrence. Resident speed is2.25190TF/core versus
1.87717 full compensation and2.44112 MAIN. Repeated identical K/V gives31.87%
L2 for denominator-only: coherent numerator drift is not fixed and can be
partly cancelled by denominator drift in MAIN. Distinct256K tests are needed
before deciding whether its smaller cost is worthwhile.

At256K, the same fullchip experiment gives:

| K/V storage | MAIN LoFi L2% / TF | Full compensation L2% / TF | Denominator-only L2% / TF |
|---|---:|---:|---:|
| BFP8 / BFP8 | 18.641 / 212.64 | 3.735 / 191.46 | 4.599 / 208.39 |
| BFP4 / BFP8 | 22.799 / 220.48 | 12.529 / 195.91 | 12.797 / 219.17 |
| BFP8 / BFP4 | 22.734 / 222.00 | 12.293 / 196.22 | 12.566 / 220.08 |
| BFP4 / BFP4 | 26.264 / 233.48 | 17.168 / 200.21 | 17.352 / 228.84 |

Denominator-only is a useful coarse-precision point: for BFP4 K/V it recovers
most of full compensation's numerical benefit at~2% overhead versus MAIN,
rather than~14%. For BFP8, remaining BF16 numerator error is more visible:
4.60% versus3.74%, with~9% more throughput than full compensation. These are
normal-input results, not a broad robustness or model-quality acceptance.

## Update: native FP32, long-context stress, and preprocessing

Native-exp FP32 reaches153.83TF/chip at32K and161.19 at256K with RNE5/BFP8
K/V, including preprocessing; normal L2 is2.8244%/2.7984%. The BFP4 K/V
versions give155.04/166.38TF and16.8573%/16.9369% L2. Quadratic-refined
BFP8 at32K gives110.19TF and2.1584%. These preserve FP32 recurrent state,
accurate online rescale exp and matched represented-P denominator. A broader
two-seed native-exp suite is running; these are not universal accuracy bounds.

The denominator-only BF16 candidate fails coherent-V stress at256K:

| K8/V4 input | MAIN LoFi L2% | Full compensation L2% | Denominator-only L2% |
|---|---:|---:|---:|
| Constant V=1 |14.5917|0.5361|25.4669|
| V with common offset32 |14.5906|0.5359|25.4655|

Its attractive normal-input result does not fix numerator recurrence loss.
Centering V before its BFP4 quantization removes that coherent component,
but quantization can introduce a new mean. At32K/H10/K8V4/denominator-only:

| V preparation | Normal L2% | Common-V L2% | Combined TF/chip, normal |
|---|---:|---:|---:|
| None |12.0456|1.4278|218.72|
| Center + original BF16 mean |16.0933|0.02865|209.09|
| Center + matched represented mean |9.8149|0.02865|202.53|

Matched here uses BF16 means/subtraction and a BF16 output epilogue. These
precision choices are explicit limitations, not exact FP32 mean repair.
Common-V global L2 hides residual error: the matched32K case has125.23%
L2 after subtracting the same original-V mean from both outputs. A separate
BF16 reference-rounding floor calculation is needed before attributing all
of that residual error to the algorithm. Constant-V centered outputs recover1
exactly; their tiny reported~1e-13% difference is FP64 reference arithmetic.

Signed H16 Q/K rotation, denominator-only BF16, H10/32K:

| Outlier inputs | Unrotated L2%, seeds1240/1241 | H16 L2%, seeds1240/1241 |
|---|---:|---:|
| K4/V8 |30.514 /29.116|12.332 /7.825|
| K4/V4 |32.453 /31.617|17.188 /14.987|
| K8/V4 |16.573 /13.858|12.408 /12.989|

Normal-input error stays essentially unchanged. H16 combined throughput is
about207–215TF versus218–227TF without rotation. This is conditional:
CPU tests show common-Q and reciprocal channel imbalance can worsen badly.
Q centering repairs the former; diagonal channel balancing can help the latter.
See `Q_CENTER_HADAMARD.md`, `DIAGONAL_K_SMOOTHING.md` and raw device records.

The half-sync Q-centering integration now passes all128-query1024-KV smokes:
skip-add and zero-add controls both2.1930% L2; centered normal2.1697%;
common-Q32 centered1.5385%. The added FP32 correction pass uses two score
and two correction tiles at a time. Full-sync diagnostics remain invalid.
Longer runs are pending. The correction producer's~0.0304% FP32 residual is
explained by a bit-exact hardware product-alignment model, not BF16 spilling;
see `matmul-fp32-floor-audit.md` for the independent device discriminator.

Tiny correction matmul primitive at256K: eight separate1-row calls1.6688ms,
one8-row call0.2302ms, one padded32-row call0.3244ms. Outputs are identical.
This excludes Q-mean production, retiling and attention integration.

## Update: compressed P remains a negative optimization

A separate compact BF16 P buffer exceeds available L1 by13,312bytes at the
fixed geometry/buffering. An equal-pitch4096-byte alias avoids this allocation
failure and passes all-output smokes with independent score/P FIFO counters.
The custom four-tile padded packer also passes exact128-tile roundtrips in
both DST modes, preserving the active native-exp replay slots.

Resident Q256/K512, same prepared inputs and private controls:

| P storage / packing | Cubic TF/core | Native-exp TF/core |
|---|---:|---:|
| FP32, scalar |1.0023|1.4888|
| FP32, standard4 |1.0309|1.5513|
| BF16 padded alias, scalar |0.9676|1.3661|
| BF16 padded alias, custom4 |0.9722|1.4222|

Scalar/blocked outputs are identical within each format/exp pair. BF16 P
is still slower than FP32 P; less nominal L1 payload does not guarantee less
kernel time. This candidate is not advanced as a Pareto improvement.

At09:59UTC, the locked-variant validator again passes4variants/14pinned
sources/34cases and136identical repeated outputs. No frozen sources changed.

## Update: broader qualification and accurate K-offset attribution

The native FP32/LoFi suite has completed82 finite, replay-identical cases at
each of32K and256K:13 input distributions,2 seeds,3 main variants and the
additional common-K centering controls. References use original BF16 inputs,
FP64 attention and all KV, with128 sampled Q rows/head. This suite has no
timing claims. See `NUMERICAL_FINDINGS.md` and `native-suite-{32k-v1,256k-v2}.jsonl`.

Normal native B8/B8 remains approximately2.8% L2; B4/B4 approximately17%.
Neither is a uniform stress-input accuracy band. At256K, B8 outliers reach
10.35–15.66%, B4 outliers38.72–46.80%, and B4 channel-K outliers62.30–64.05%.
Common-K centering reduces B8's44.82–45.44% to2.347–2.357%, but cannot repair
arbitrary outlier mechanisms or make B4 a sub-percent representation.

An unchanged accurate HiFi4 kernel with original BF16 inputs has approximately
0.73% common-K error. The new isolated K-shift diagnostic at32K reduces
0.73617→0.17812% by subtracting a device-generated BF16 K mean. Exact FP64
shift invariance and BF16 shifted-input error are both approximately2e-13%
in that case; the original input reference is unchanged. This localizes the
extra error to finite-arithmetic sensitivity to the common component, without
uniquely proving which hardware step causes it. On ordinary normal inputs,
the extra centered-K BF16 spill instead worsens0.18061→0.23983%, with0.15862%
reference drift from that spill. Centering should be conditional, not automatic.

## Update: represented V means and denominator-only compensation

At256K/H10/Q256/K512, denominator-only compensated BF16 K8/V4 gives:

| V input | No centering L2% | Original mean restored L2% | Matched represented mean L2% |
|---|---:|---:|---:|
| Normal |12.566|42.444|10.323|
| Common V+32 |25.466|0.010421|0.010421|
| Constant V=1 |25.467|approximately0|approximately0|

Matched centering runs at216.55TF/chip on normal versus218.43 without it.
The common-V centered outputs equal the BF16 reference rounding floor;
their129.26% residual-relative L2 is also exactly the BF16 floor, not a
remaining kernel defect of that magnitude. See `valuecenter-reference-floor-v3.jsonl`.

With B8 V, matching must use the values actually consumed by LoFi, including
its five-significant-bit truncation after native packing. The effective-V
primitive now passes ordinary/wide finite BF16 and signed-zero controls in
both DST modes, with only documented negative-zero canonicalization.
B8 normal256K L2 is4.599% without centering,40.693% with original mean only,
and4.404% with matched mean, at206.65/206.49/205.62TF/chip respectively.
Centering repairs common V here too, but does not replace full numerator
compensation for general accuracy.

## Update: adaptive BFP4 exponent search on device

The private quantizer searches E, E−1 and optionally E+1 per native group16,
choosing minimum reconstructed MSE using an explicitly specified FP32 sum
tree. All8 initial device probes match decoded FP32 bits exactly, including
ties, wide exponents, zeros and group outliers. At256K×128/110 cores,
baseline/minus/three-candidate preprocessing costs0.4324/0.7314/1.1010ms;
the older optimized RNE quantizer is faster than this batch1 baseline.

Full-chip denominator-only BF16, H10/N32K, original-input reference and all
preprocessing costs included:

| K/V | Native RNE L2% / TF | E/E−1 L2% / TF | E/E−1/E+1 L2% / TF |
|---|---:|---:|---:|
| K8/V4 |12.0456 /218.75|11.3171 /213.84|11.3163 /209.75|
| K4/V4 |16.9596 /226.91|16.0211 /216.35|16.0235 /208.77|

The larger search improves all5 tested K8/V4 input cases, but K4/V4 sparse
outliers worsen32.4527→34.9261% even as quantizer MSE improves. Local MSE
does not optimize attention's sensitivity. The three-candidate search helps
channel-K error54.8147→51.0553%, still far outside the normal band.

## Update: LUT exp scheduling and cheap P-bit matching

The two-segment FP16 LUT corrects native-exp shape error without a cubic
refiner. A LOADMACRO schedule reduces its issued body from10 to8 instructions
per pair of vectors. Raw and macro implementations are bit-identical on all
outputs of N1024/H2, the resident test, and full-chip32K/256K normal tests.

| Scope | Raw LUT TF | Macro LUT TF | Unchanged L2% |
|---|---:|---:|---:|
| One resident core, Q256/K512 |1.14860|1.20356|2.24428|
| Full-chip H10/32K, including preprocessing |115.93|121.06|2.20760|
| Full-chip H10/256K, including preprocessing |123.01|128.66|2.20918|

Separately, native exp can emit exactly7 significant bits by quartering its
two affine constants and changing shift15→17. This changes no vector-loop
instructions. The standalone BF16 primitive matches all524,288 oracle values,
covering17,405 distinct supported BF16 deltas, ties and underflow. However,
full-FAST BF16 at32K worsens3.1511→3.1839% normal and0.5126→0.5794% constant-V
L2 at unchanged speed. Matching P operand bits alone does not make the
different BF16 reductions, recurrence and normalization exact.

An independent four-tile identity-correction scheduling change improves the
FP32 native resident loop1.56205→1.59690TF/core, bit-identically. Its full-chip
32K/256K gains are within timing variation; do not claim a chip-level win.
