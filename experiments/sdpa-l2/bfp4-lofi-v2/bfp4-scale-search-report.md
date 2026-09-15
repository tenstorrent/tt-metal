# Hardware-constrained BFP4 exponent search

## Decision

Feasible without scale tensors or changes to attention matmuls, but not an accuracy breakthrough. Empirical exponent search reduces normal-input K/V representation MSE by approximately 10%; normal attention L2 falls only about 0.5 percentage points. K8/V4 benefits consistently in this small suite, whereas K4/V4 regresses on two stress inputs despite lower representation MSE. Do not make a fixed clipping threshold the general default.

If V4 remains important for bandwidth, a private **V-only E versus E−1 MSE selector** is a reasonable low-priority follow-up. I would not prioritize full three-candidate K/V preprocessing ahead of better K representation or existing kernel optimizations. No device preprocessing or throughput benefit is demonstrated here.

## What transfers from the papers

- **ScaleSearch** searches nearby representable scale encodings and selects minimum reconstructed block MSE. Its implemented NVFP4 search uses nine FP8-scale offsets, −2 through +6. NVFP4 has E2M1 values and scales with mantissa bits; native TT BFP4 does not. The transferable contribution is empirical scale selection, not those offsets or the paper's end-model gains. Other contributions, including Q/K transforms and mixed-precision attention sinks, are independent of scale search. [ScaleSearch, May 2026](https://arxiv.org/html/2605.12464v1)
- **MXAttention** derives an E2M1 normalized-maximum threshold of 7.25 for its data-free integrated projection-error objective. That is not an arbitrary finite block's empirical MSE optimum, and the grid differs from TT. Its pre-normalization quantization uses represented P in both numerator and denominator: this corresponds to our existing matched-P approach. In exact arithmetic, effective normalized weights sum to one; finite state error and quantized weight-distribution error remain. [MXAttention, July 2026](https://arxiv.org/html/2607.24377v1)
- **SageAttention3** uses group16 FP4 microscaling, plus two-level P scaling to address the restricted FP8 scale range. TT's shared-exponent format has a different grid and exponent range; reproducing its FP8-scale treatment is not a direct native-pack optimization. NVIDIA's low-bit instruction speedups do not imply a speedup from changing B8 to B4 when both already execute TT LoFi. [SageAttention3, January 2026 revision](https://arxiv.org/html/2505.11594v3)

## Exact TT encoding constraint

For a finite, nonzero group of 16 values, let M=max|x| and E=floor(log2 M), using an unbiased exponent. Native BFP4 has three unsigned magnitude bits plus sign, so its decoded grid is

`sign × {0,1,…,7} × 2^(E−2)`.

This follows the shared-exponent/magnitude definition in the ISA, not the E2M1 grid `{0,.5,1,1.5,2,3,4,6}` of MXFP4/NVFP4. Source inspected: [FloatBitPatterns.md](https://github.com/tenstorrent/tt-isa-documentation/blob/5287a62727350bcef35f7b411d1b8a706172ec4c/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md#bfp), local ISA commit `5287a62727350bcef35f7b411d1b8a706172ec4c`. This definition is in the Wormhole-oriented shared documentation; prior Blackhole B4 preprocessor qualification independently established our baseline encoding contract.

For δ in {0,−1,+1}, construct

`Δδ = 2^(E+δ−2)`

`qδ(x) = sign(x) × Δδ × clamp(RNE(|x|/Δδ), 0, 7)`.

Choose the candidate minimizing the sum of squared reconstruction errors across the group, preferring the original exponent on ties. The E−1 candidate halves the spacing but clips the largest input; E+1 trades bulk resolution for better representation of an upper-bin maximum.

Pre-round **and cap to these final decoded values**, then let native packing run normally. The outputs have at most three significant bits and are exact BF16/FP32 values. The selected exponent must be induced by their actual maximum, not merely stored in an unattached metadata variable. For E−1, saturation produces a magnitude-seven maximum at the selected exponent. A strictly winning E+1 candidate must reach its own exponent bin; otherwise its coarser representable set is contained in the baseline set and cannot win. The model checks this invariant for every selected group.

All 80 tensor/policy outputs survived three CPU native-rounding models exactly: direct shared RNA3, per-value RNA7 then shared RNA3, and per-value RNA3 then shared RNA3. No selected-exponent mismatches occurred. This supports the mathematical pack contract; it is **not a new device pack test**, and excludes exponent extremes, nonfinite inputs, and special zero encodings.

Unlike a general learned scale, this scheme needs no residual scale multiplication inside QK/PV. It does change the approximated inputs: clipping is not an algebraically exact attention transformation.

## Model and results

Run on the reserved machine's CPU, four threads, 12.409 seconds for all 100 attention cases and 90 representation records. No device imports or jobs. New artifacts:

- [CPU model](bfp4_scale_search_model.py), SHA256 `0e926580cf9838d97f3364eec1d78fdf8c7868218e6f5ad325bce5b9d2537134`.
- [Raw JSONL](bfp4-scale-search-model-v1.jsonl), including gain, PCC, MSE, clipping rate, exponent selections, source pin, and timing.

H=1, D=128, N=32768, 128 Q rows, K chunks512, seeds1240/1241. Original BF16 Q/K/V reference. Q is RNE7; K4/V4 use unbiased final-grid RNE; K8 uses RNE5 followed by native shared B8 RNA and LoFi five-bit consumption. Native FP32 approximate-exp grid, P trunc7, and matched represented-P denominator are retained. QK, subtraction, online rescaling, PV, and recurrent state use FP64; final output is BF16. This deliberately omits FPU product alignment, the real running-max format, FP32 state roundoff, and reciprocal error. It isolates representation effects, not exact device output.

Distributions: standard normal; 0.1% additive N(0,10²) outliers in all Q/K/V; Q/K multiplied by two; or every sixteenth channel multiplied by32 in K alone or V alone. Channel multiplication follows BF16-equivalent power-of-two scaling. The normal/outliers/scaled inputs reproduce the existing repro's seeded generation.

Each cell is L2% for **seed1240 / seed1241**. Search here means the full E−1/E/E+1 empirical-MSE selector applied to every B4 tensor in that column.

| Distribution | K4/V4 baseline | K4/V4 search | K8/V4 baseline | K8/V4 search |
|---|---:|---:|---:|---:|
| Normal | 16.006 / 16.622 | 15.402 / 16.092 | 11.262 / 11.630 | 10.695 / 11.374 |
| Sparse outliers | 48.138 / 24.104 | 47.408 / 25.370 | 16.834 / 14.570 | 16.321 / 13.708 |
| Q/K ×2 | 31.575 / 32.333 | 29.512 / 32.662 | 12.655 / 12.548 | 11.950 / 11.931 |
| K channel outliers | 56.551 / 49.849 | 50.134 / 49.485 | 16.653 / 17.490 | 16.173 / 17.047 |
| V channel outliers | 18.023 / 18.744 | 17.517 / 18.552 | 13.036 / 13.124 | 12.679 / 12.998 |

Representation MSE reductions below average the two seeds; positive means better. They concern the named tensor, not attention output.

| Tensor distribution | E/E−1 search | E/E−1/E+1 search | Fixed uniform proxy |
|---|---:|---:|---:|
| Normal K | 10.00% | 10.00% | 8.17% |
| Sparse-outlier K | 9.09% | 9.10% | 7.08% |
| Channel-outlier K | 1.05% | 5.89% | −27.92% |
| Channel-outlier V | 1.06% | 5.82% | −27.89% |

Normal K selected E−1 in approximately18.6% of groups and E+1 in only0.032%. Channel-outlier K selected E+1 in approximately4.12%, making that candidate materially more relevant there. E/E−1 already captures essentially all normal-input representation benefit.

The lower MSE also introduces more shrinkage: normal K representation gain changes from approximately0.9949 to0.9858. Seed1240 normal K4/V4 output gain changes from0.9950 to0.9735, despite L2 improving16.006→15.402% and PCC improving0.98724→0.98799. Thus this is not a uniformly better approximation in every useful sense. K8/V4's gain changes0.9988→0.9866 while its L2 improves11.262→10.695%.

## Cheap threshold ablation: not recommended

As an independent TT-grid calculation, define the integrated squared projection error A(q) onto nonnegative integer codes0…7. Applying a UOS-style proxy gives the stationary equation `A(q)=8A(q/2)`. On the applicable intervals,

`A(q)=7/12+(q−7)^3/3`,

`A(q/2)=1/3+(q/2−4)^3/3`.

Their root is `q=7.5+sqrt(2)≈8.91421356`. Using `Δ=2^ceil(log2(M/q))` selects E−1 when `M/2^E≤1.1142767`, otherwise E. This is our grid-specific stationary proxy calculation, not the paper's constant and not a proof of empirical block optimality.

It captures most normal-input representation benefit but performs badly for channel outliers. K-channel-outlier attention becomes60.867/63.861% instead of56.551/49.849%; K8/V4 with V channel outliers becomes14.248/14.245% instead of13.036/13.124%. The within-range uniform projection proxy is a poor fit for one dominant value per group.

## Implementation cost and next decision

Native pack has no documented empirical-MSE scale-search operation. A preprocessor would need candidate generation, squared residuals, a group16 sum reduction, candidate comparison, and final selection in addition to the existing group maximum and RNE/cap. A full three-candidate implementation costs more than two candidates; exact instruction counts and speed require an implementation and measurement. Nothing here changes Q256/K512, K/V slots, or attention data movement.

The most defensible small experiment is V-only two-candidate search with K8 retained, if saving V bandwidth is already valuable. Require an exact quantizer oracle, distinct-input attention checks, combined preprocessing-plus-attention timing, and a regression gate on the stress distributions above. Do not infer a device performance win from these CPU timings, and do not describe this as solving BFP4's approximately10–15% normal-input attention error band.

Validation: Python syntax compilation passed; complete CPU sweep passed its representation-MSE, induced-exponent, and modeled pack-roundtrip assertions. Existing sources, frozen controls, and device jobs were untouched.
