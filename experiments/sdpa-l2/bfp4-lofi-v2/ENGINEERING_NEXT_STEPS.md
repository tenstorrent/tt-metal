# Engineering disposition and next steps

This is a research disposition, not authorization to change the four selected variants or public dispatch. The numerical candidates, controls and negative results are preserved independently. Performance figures below are the historical common-reader H10/D128/Q256/K512 normal-input measurements at 32K/256K, including real preprocessing; they are not hardware FPU counters or post-format timing results.

## Candidate disposition

| Candidate | Normal L2 band | Combined chip TFLOP/s | Disposition |
|---|---:|---:|---|
| Selected balanced / accurate | 0.390% / 0.179% | 88–97 / 71–79 | Keep accuracy references; stress exceptions still apply |
| LoFi, fully compensated BF16, B8 | 3.15–3.74% | about 190 | Fast wider-storage candidate for broader evaluation |
| LoFi FP32 B8, native exp | about 2.8% | 154–161 | Opt-in speed candidate; substantial order/common-mode sensitivity |
| LoFi FP32 B8, macro-LUT exp | about 2.2% | 121–129 | Better exp consistency, at measurable cost; not an accurate-mode substitute |
| HiFi2 FP32 BF16 K/V, macro-LUT | about 0.75% | 105–110 | Intermediate point worth qualification; does not meet the 0.5% target |
| B4 denominator-only/adaptive branches | about 11–17% | 209–229 | Codec/bandwidth diagnostics, not robust promotion candidates |

These are not universal accuracy bands. In the existing long native-exp stress suite, LoFi B8 reaches roughly 45% common-K and 26% common-Q L2. Source records and distribution-level qualifications govern any more specific claim. Conditional V-axis changes, rotations, centering and mean-error correction should not become additional default SKUs merely because they help one synthetic stress.

## 1. Establish correctness and dispatch contracts first

The smallest independently justified integration task is the **Q256 correction-address reset plus a targeted regression**, documented in [the private-harness defect investigation](FAST_Q256_CORRECTION_BUG.md). This is not a blanket claim that all production dispatch configurations exercise the defect. The frozen numerical choices remain unchanged in this research worktree.

The regression should use Q128 and Q256, one and several K512 chunks, original BF16 and rounded B8 operands, distinct K/V blocks that force successive maximum increases, multiple query jobs per core and head boundaries. Require all-output original-input FP64 reference at small N and two bitwise trace replays. Repeated resident blocks alone are inadequate: they can hide the address-state defect.

Before exposing a new variant, specify these contracts independently:

- Input dtype versus internal compute/storage precision, output dtype, exp quality, numerator/denominator state and row-maximum precision.
- Supported shapes, masks, causal semantics, head mapping, chunk sizes and buffering; explicit reject/fallback behavior outside them.
- Preprocessing range validity before device invocation, independently of optional expensive oracle checks.
- Buffer lifetime, source/destination format transitions, SFPU address-modifier state and trace replay behavior.

There is a concrete range-check gap in the experimental capture route: artifact validation accepts finite BF16 tensors, whereas the current B4 preprocessor supports normal values/zero with group-maximum exponents in [-124,106]. Its range check lives in an optional host oracle after device preprocessing and can be bypassed by disabling exact-preprocessing validation. A promoted route should reject unsupported subnormals/ranges **before** invoking the device, without implying new special-value support. This is a preflight/integration requirement, not an observed failure of the normal/stress cases reported here.

Changing the public input dtype alone is insufficient: the production streaming guards and output specification do not encode these experimental choices. Keep any integration explicitly opt-in until the dispatch contract is tested.

## 2. Qualify wider-storage LoFi before adding more low-bit machinery

Use the existing Q7/KV5 preprocessing and FP32 P/numerator/denominator as a minimal candidate; retain native and LUT exp as explicit, independently selectable research controls. Do not initially bundle new centering, rotation or adaptive scaling into that integration.

Add **block-permutation output distance** to the diagnostic suite. At 32K, the native FP32 LoFi output changes by 2.55% under reversal of the same represented K/V blocks, versus 0.87% with same-grid LUT and 0.52% with the cubic/finer-grid path. Original-reference aggregate L2 alone hid this difference. This metric diagnoses stability; it is not automatically an accuracy acceptance threshold or a requirement of bitwise permutation invariance.

The next broad qualification should compare the same original inputs for normal, scaled QK, common Q/K/V, outliers, channel-scaled K/V, uniform attention and constant V, using multiple seeds and both all-output small cases and sampled long cases. Preserve quiet-channel and centered-residual metrics. The small late-window exp stress study supplements this qualification; it does not cover long-context drift or real model activations.

Once representative captures are available, evaluate actual SDPA-boundary tensors and semantics through [the captured-input contract](CAPTURED_INPUTS.md). Model-quality evidence requires downstream evaluation as well; operator L2/PCC alone is insufficient. Do not relabel causal captures as an unmasked operator qualification.

One small performance-only opportunity is removing identity K/V preparation copies from the HiFi2/BF16 control. Require identical complete output hashes and separately time the combined route. This removes redundant work without intentionally changing arithmetic, but its actual gain is unmeasured and may be small at long N.

## 3. Give BFP4 a concrete memory or bandwidth objective

Current LoFi and BFP4 storage are independent choices. Narrower K/V does not multiply LoFi arithmetic throughput, and the compute-heavy full-compensation loop does not realize proportional speedup from the byte savings. A BFP4 proposal should first identify the target it improves: capacity, memory traffic, or a measured data-movement-bound workload.

Priorities, in order:

1. Retain B8 controls when testing V's sharing direction. N-axis grouping repairs quiet channels in both B4 and B8; comparing only two B4 layouts overstates what low precision buys.
2. Evaluate selective K8/V4 versus K4/V8 against that concrete memory objective, including transpose and preparation costs. Sensitivity in score routing can make coarse K more consequential than its reconstruction MSE suggests.
3. Apply H16, adaptive scaling or bias correction conditionally only after a supported selection rule is evaluated on held-out inputs. Already observed counterexamples rule out universal “smoothing helps” claims.
4. Park the residual B4+B8 path as a default candidate: roughly 0.515% normal L2 at 67.5–70.5 useful TFLOP/s does not beat the selected balanced point, and its extra matmuls are real costs.

The FP64 representation-only study leaves about 16.6% normal attention error from rounded B4 K/V alone. It is not a lower bound for every codec or model, but it is a clear stop signal for trying to solve these normal-input errors solely through accumulator or exp improvements. More expensive arithmetic cannot recover information that the selected representation removed.

## Decision gates

Promote a candidate only when it has: (a) a defined supported-input contract, (b) no unexplained deterministic/replay/finite failures, (c) a reproducible accuracy benefit on the intended inputs, (d) measured combined-time or memory benefit against the relevant wider-storage control, and (e) an explicit remaining stress/model-quality risk statement. Do not silently expand the four previously selected options based on this research alone.
