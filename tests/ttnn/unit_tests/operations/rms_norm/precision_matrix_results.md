# rms_norm — precision matrix results

Produced by `test_rms_norm_precision_matrix.py` (/numeric-formats-metal §10).

- **Last run**: 2026-07-28, after Refinement 1.
- **Device**: blackhole_p150b, AICLK 1350 MHz, 110-core compute grid.
- **Cells**: 8 shapes × 3 dtypes × 4 math_fidelity × 2 fp32_dest_acc_en × 2 input
  distributions = **384**, of which **320 run** and **64 are skipped** by the op's
  own `EXCLUSIONS`.
- **Gate**: PCC ≥ 0.99 asserted; every other metric printed for observability only.
- **Result**: **320 / 320 pass.** Worst PCC anywhere in the matrix is **0.999325**
  (bfloat8_b / LoFi / bf16 DEST) — ~75× inside the gate.

Shapes: `32x32`, `1x1x64x128`, `1x1x128x512`, `1x1x32x4096` (wide, chunked reduce
`NW > 1`), `32x48` (W non-aligned), `48x64` (H non-aligned), `1x1x17x50` and
`2x1x100x47` (both non-aligned). Gamma present, `epsilon = 1e-6`, TILE layout,
gamma at the activation's dtype.

## Worst case per config (over all 8 shapes × both distributions)

| dtype | fp32_dest_acc_en | math_fidelity | min PCC | max rel-RMS | max abs err |
|---|---|---|---:|---:|---:|
| bfloat16 | True | HiFi4 | 0.999992 | 2.511e-03 | 4.807e-02 |
| bfloat16 | True | HiFi3 | 0.999992 | 2.506e-03 | 4.807e-02 |
| bfloat16 | True | HiFi2 | 0.999973 | 1.012e-02 | 1.462e-01 |
| bfloat16 | True | LoFi | 0.999493 | 4.169e-02 | 5.721e-01 |
| bfloat16 | False | HiFi4 | 0.999980 | 6.163e-03 | 8.858e-02 |
| bfloat16 | False | HiFi3 | 0.999980 | 6.163e-03 | 8.858e-02 |
| bfloat16 | False | HiFi2 | 0.999964 | 9.554e-03 | 1.456e-01 |
| bfloat16 | False | LoFi | 0.999509 | 4.008e-02 | 5.721e-01 |
| float32 | True | HiFi4 | 0.999999 | 1.569e-03 | 2.243e-02 |
| float32 | True | HiFi3 | 0.999999 | 1.650e-03 | 2.403e-02 |
| float32 | True | HiFi2 | 0.999976 | 1.109e-02 | 2.128e-01 |
| float32 | True | LoFi | 0.999544 | 4.761e-02 | 5.946e-01 |
| float32 | False | — | **skipped — op `EXCLUSIONS`** | | |
| bfloat8_b | True | HiFi4 | 0.999820 | 1.794e-02 | 1.837e-01 |
| bfloat8_b | True | HiFi3 | 0.999820 | 1.800e-02 | 1.837e-01 |
| bfloat8_b | True | HiFi2 | 0.999792 | 1.743e-02 | 1.914e-01 |
| bfloat8_b | True | LoFi | 0.999346 | 3.405e-02 | 6.120e-01 |
| bfloat8_b | False | HiFi4 | 0.999827 | 2.235e-02 | 1.870e-01 |
| bfloat8_b | False | HiFi3 | 0.999827 | 2.235e-02 | 1.870e-01 |
| bfloat8_b | False | HiFi2 | 0.999821 | 2.051e-02 | 1.870e-01 |
| bfloat8_b | False | LoFi | 0.999325 | 2.967e-02 | 5.579e-01 |

## Reading the table

- **`math_fidelity` is the dominant axis, not dtype.** Dropping HiFi4 → LoFi costs
  roughly an order of magnitude in rel-RMS for every dtype; HiFi4 → HiFi3 costs
  nothing measurable (rms_norm's FPU work is multiplies and adds, and HiFi3 only
  adds a pass that matters to wider mantissas). This is expected hardware
  behaviour, not a bug.
- **`fp32_dest_acc_en=False` is cheap here.** bf16 loses ~2.5× rel-RMS at HiFi4
  and *nothing* at HiFi2/LoFi, where fidelity already dominates. The perf loose
  cases run at HiFi2 + bf16 DEST and clear their tighter `pcc_threshold = 0.9995`
  soft gate with room. The op's pairwise-add reduce (`AccumulateViaAdd`) is what
  keeps this cheap — see op_requirements.md Refinement 1's accuracy watch item.
- **bfloat8_b sits ~7× above bf16 in rel-RMS and is flat across fidelity.** Its
  error is set by block-float quantization of the input and output tensors, which
  happens outside the compute pipeline, so `math_fidelity` barely moves it until
  LoFi.
- **`float32 + fp32_dest_acc_en=False` is a permanent op-side refusal**, not a gap:
  accumulating an fp32 input through a 16-bit DEST is a silent precision trap. The
  matrix skips it from the op's own `EXCLUSIONS` list rather than a local copy.

## Skipped combinations

| Combination | Count | Why |
|---|---:|---|
| `float32 × fp32_dest_acc_en=False` | 64 | Op `EXCLUSIONS` — permanent refusal (`references/precision_convention.md`). |

## Not covered here (deliberately)

`ROW_MAJOR` input/gamma layout and the mixed-precision gamma cells
(`gamma_dtype ≠ dtype`) are covered by `test_rms_norm_precision_baseline.py` and
the golden suite's cartesian, not repeated in this matrix.

## The metric this matrix cannot provide

PCC and rel-RMS are both **scale-invariant or near-scale-invariant** on this op:
rms_norm's output is `x · rsqrt(mean(x²) + ε)`, so a reduce that under-counts
elements simply rescales every row by a near-constant. A bfloat8_b partial-W bug
that summed 32 of 49 elements scored PCC **0.9998** and passed the golden gate
(`probes/probe_005.py`). `test_partial_w_reduce_counts_every_element` covers that
blind spot by inverting an all-ones output back into the element count the reduce
actually accumulated — an *absolute* check, not a correlational one.
