# Beta180 measured attribution and remaining source opportunity

This note reads parent-produced device evidence. The source helper agent did
not run hardware or modify the selected/frozen implementation.

## V1 failure is the mixed-input addition boundary

`beta180_probe.json` records all12 V2 cases passing exact logical beta, decay,
add intermediate and softplus intermediate comparison, repeated bit equality,
and supplied-output address reuse. The add diagnostic also compares the same
native output against CPU full-FP32 addition of BF16 A and uploaded dt_bias.
Native add differs from that CPU sum by up to0.015625 in every case.

For all12 cases, the fraction of native-add values equal to CPU FP32 addition
is **identical** to the fraction of V1 final-decay values equal to native decay
in `beta177_probe.json`:

| Batch | Draw | Native add = CPU FP32 fraction | V1 decay = native fraction |
|---:|---:|---:|---:|
| 1 | 0 | 0.7708333333 | 0.7708333333 |
| 1 | 1 | 0.8541666667 | 0.8541666667 |
| 1 | 2 | 0.8750000000 | 0.8750000000 |
| 3 | 0 | 0.7777777778 | 0.7777777778 |
| 3 | 1 | 0.9375000000 | 0.9375000000 |
| 3 | 2 | 0.9097222222 | 0.9097222222 |
| 31 | 0 | 0.7782258065 | 0.7782258065 |
| 31 | 1 | 0.9126344086 | 0.9126344086 |
| 31 | 2 | 0.9112903226 | 0.9112903226 |
| 32 | 0 | 0.7773437500 | 0.7773437500 |
| 32 | 1 | 0.9114583333 | 0.9114583333 |
| 32 | 2 | 0.9069010417 | 0.9069010417 |

V2 differs from V1 by restoring native mixed BF16/FP32 FPU addition and its
input representation, while leaving sigmoid, softplus and final multiplication
mathematics intact. Exact V2 add, softplus and final outputs therefore isolate
V1's error to changing the native addition boundary to precise SFPU addition.
There is no evidence of a softplus-init or polynomial discrepancy.

Do not call this merely truncation of dt_bias: checkpoint dt_bias is BF16
before FP32 upload, so it already has fewer mantissa bits than TF32. The
probe establishes the **native FPU addition boundary** differs from full-FP32
addition, without separating internal alignment, accumulation and output
rounding at the hardware instruction level. Its mismatch samples list only
native-versus-V2 differences (empty because they match); this artifact does
not expose raw CPU/native unequal pairs to fit a precise bit-truncation rule.
No such rule is needed: V2 executes the actual native FPU add.

## Recorded whole-decoder validation

- `beta180_b32.json`: prefill PCC0.9999481019, decode PCC0.9999370435;
  warmed prefill24.451681ms, traced median decode1.645240ms. Prefill/decode
  repeated outputs are bit equal; changed input changes output and passes PCC.
- `beta180_sequence.json`: B32, prefix128,512 decode steps, minimum per-user
  PCC0.9955427456, traced median1.657564ms.
- `beta180_b1.json`: prefill PCC0.9999457541, decode PCC0.9999917837;
  warmed prefill6.328119ms, traced median decode1.219332ms. Repeated and
  changed-input checks pass.

Parent's matched integrated167 comparison is B32 short1.661943→1.645240ms
(16.703us,1.005%) and512-step1.671731→1.657564ms (14.167us,0.847%). These
measure the entire decoder, not an isolated kernel extrapolation. Input/output
precision, recurrence semantics and acceptance thresholds remain unchanged.

## Concrete untried reduction in work

The current fused helper fills only row0 of each input tile, but its unary
calls execute all four faces and eight iterations per face. The native LLK
`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h`
explicitly implements VectorMode::R as Face0 + Face1; the unary callable's
iteration parameter independently determines vectors processed in each face.
The following source-supported specializations should cover the logical row0
while retaining the identical per-lane formulas:

- Sigmoid: `SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_sigmoid,
  (false,false,1),idst,VectorMode::R)`.
- Softplus: a small wrapper around
  `SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_softplus,
  (false,true,1),idst,VectorMode::R,beta,beta_reciprocal,threshold)`.
  Keep INP_FLOAT32 and the same native softplus init.

This reduces nominal unary invocations from32 vector iterations to2. Preserve
full native FPU add, final multiply, CB contents and packing for the first
controlled experiment; do not combine with a different FPU implementation.
Logical rows are the contract. Physical padding would remain deterministic
but change from current unary-transformed padding, which no current consumer
reads. Own-input equality and trace reuse must still validate the face traversal
and callable iteration assumptions before any whole-decoder claim.

No new candidate has been created for this proposal. Reader/pack/dispatch
work remains, and the current kernel is small: whole-decoder timing is needed
to decide whether this saves a meaningful amount beyond the measured fusion.
