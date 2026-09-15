# Blackhole low-precision attention research

Research window: 2026-09-15 05:48:43–13:48:43 UTC. Numerical experiments, controls and qualification are reported below; limitations are explicit.

## Bottom line

The most promising candidate for broader evaluation is **LoFi with carefully rounded operands and wider recurrent state**, not indiscriminate BFP4. Native BFP4 is useful as a bandwidth/accuracy research point, but it does not currently approach the previously selected accurate modes. Several Sage-inspired transformations help specific failure mechanisms; none is a universal accuracy repair.

The four frozen SDPA choices remain unchanged. These are private experimental kernels and common-reader benchmarks, not a production API, dispatch change, or model-quality qualification. The raw numerical/evidence checkpoint before final formatting is `52b288a3213`.

## What transfers from SageAttention

[SageAttention2 v7](https://arxiv.org/html/2411.10958v7) combines low-bit QK, higher-precision PV, smoothing, and accumulation improvements. [SageAttention3 v3](https://arxiv.org/html/2505.11594v3) uses microscaled FP4 and a hardware-specific implementation. The transferable ideas are separating QK/PV precision, treating outliers explicitly, selecting a suitable block-sharing direction, and protecting recurrent accumulation—not assuming that two formats called “4-bit” have equivalent numerics or throughput. Published INT4 model results are configuration-dependent; [our Sage-style codec comparison](SAGE_INT4_COMPARISON.md) does not establish Sage model quality.

Blackhole's native BFP4_b has a sign and three magnitude bits with a shared power-of-two exponent across 16 values. It is not NVIDIA's **NVFP4** recipe: E2M1 values with per-16 E4M3 local scales and an FP32 tensor scale, as described in [Transformer Engine's format documentation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html). Other E2M1 formats such as MXFP4 use different scaling rules. This generic NVFP4 description does not reproduce Sage3's complete scaling/attention algorithm. In addition, Blackhole **LoFi and BFP4 storage are separate choices**: in the tested attention-matmul mapping, LoFi consumes seven significant bits from logical-left Q/P and five from logical-right K/V. Rounding to those observed widths before the multiply reduces one-sided truncation bias. This mapping must not be generalized to differently mapped elementwise operations. We need not throw away useful Q/P bits merely to imitate an FP4 recipe.

These statements are supported by the silicon probes and primary-source codec discussion in [the v1 report](../bfp4-lofi-v1/REPORT.md), [codec comparisons](CODEC_LIMITS.md), and [numerical findings](NUMERICAL_FINDINGS.md). The codec study is a CPU representation-only comparison, **not an execution of SageAttention or a prediction of NVIDIA kernel error**.

## Measurement contract

- Original BF16 Q/K/V define the FP64 reference; input preprocessing error is included. Final output is BF16. L2 means `100 * ||output-reference||₂ / ||reference||₂`.
- Full-chip tables below use noncausal square attention, H=10, D=128, Q256/K512, 110 active cores. BF16 streaming retains two K and two V slots; FP32 uses its existing one-slot configuration. The forwarding-chain reader is common to the compared candidates.
- “FP32 recurrent state” denotes the numerator and denominator state. Row-maximum CB10/11 remains BF16 in these harnesses; it does not mean every internal value is FP32.
- Full-chip accuracy samples 128 explicit Q rows per head and includes every K/V token. Small N=1024 qualification checks every output row. Sampled accuracy is not an all-output numerical bound; all outputs are separately checked finite and replay-stable where recorded.
- Reported full-chip throughput is useful attention FLOPs divided by blocking trace elapsed time. “Combined” includes real device preprocessing and any output correction. It is **not a hardware FPU activity counter** or a production-dispatch measurement. Warmup 3, seven timed replays unless a record says otherwise.
- Resident per-core tests intentionally remove recurring input data movement and preprocessing. They must not be plotted on the same unlabeled axis as full-chip results. Repeated K/V can also create a numerically different stress condition from distinct random tokens.
- Raw relative error is retained in records, but tiny reference entries can make its maximum enormous. PCC, gain, quiet-channel error, and centered residual error are important alongside aggregate L2.

Hardware: reservation 220027 on `yyzo-bh-08`, Blackhole P100A, 11×10 compute grid. The [saved active snapshot at 11:24:01 UTC](hardware-active-20260915-112401.json) shows 1306 MHz AICLK, versus a 1350 MHz configured maximum; an earlier active observation showed 1350 MHz. Clock variation is a potential confound for small sequential timing differences, not proof of their cause. Firmware 19.12.0.0, KMD 2.9.0, tt-smi 5.2.0. The remote checkout's Git HEAD is base `2ba6fc2339d53300ae87c5202f335ef56492cfb3`; explicitly synchronized source manifests identify the experimental code actually used. Benchmark interpreter: Python 3.10.19; tt-smi's separate interpreter reports 3.10.20. Linux 5.15.0-136-generic. No clock override or firmware change was made.

## Measured normal-input tradeoffs

All values below are **combined chip TFLOP/s**, not per-core throughput. These are selected seed-1240 examples, not the complete Pareto frontier. They are separate controlled runs, not a claim that every small timing difference is statistically significant. Adaptive rows use `pm` search over E−1/E/E+1; compensated LoFi BF16 uses both numerator and denominator compensation and the private address reset.

| Experimental common-harness choice | 32K L2 % | 32K TFLOP/s | 256K L2 % | 256K TFLOP/s |
|---|---:|---:|---:|---:|
| Main BF16 HiFi2 | 2.785 | 172.98 | 18.781 | 153.22 |
| FAST BF16, private correction-address reset | 2.614 | 168.11 | 3.296 | 143.16 |
| Balanced FP32 QK4/PV2 | 0.390 | 97.06 | 0.390 | 88.42 |
| ACCURATE FP32 QK4/PV4 | 0.179 | 79.26 | 0.179 | 71.01 |
| LoFi, compensated BF16, Q7/KV5 in BFP8 | 3.151 | 190.10 | 3.735 | 190.63 |
| LoFi, FP32 state, BFP8, native cheap exp | 2.824 | 153.83 | 2.798 | 161.19 |
| LoFi, FP32 state, BFP8, macro-LUT exp | 2.208 | 121.06 | 2.209 | 128.66 |
| HiFi2, FP32 state, RNE BFP8, native cheap exp | 2.215 | 129.70 | 2.167 | 126.28 |
| Same, macro-LUT exp | 1.339 | 110.54 | 1.328 | 111.10 |
| HiFi2, FP32 state, BF16 K/V, native cheap exp | 1.918 | 128.16 | 1.876 | 118.56 |
| Same, macro-LUT exp | 0.753 | 109.54 | 0.755 | 105.42 |
| LoFi, BF16 denominator-only compensation, K8/V4 RNE | 12.046 | 218.75 | 12.566 | 219.36 |
| Same, adaptive V exponent search | 11.316 | 209.75 | 11.902 | 217.69 |
| LoFi, BF16 denominator-only compensation, K4/V4 RNE | 16.960 | 226.91 | 17.352 | 228.54 |
| Same, adaptive K/V exponent search | 16.023 | 208.77 | 16.419 | 225.34 |

Evidence: `chain-h10-*`, native-exp full-chip records described in [Progress](PROGRESS.md), `lut-macro-*`, `hi2-lut-*`, `adaptive32k-*`, and `adaptive256k-*`. The [checkpoint index](CHECKPOINT_INDEX.md) explains the explicit evidence audit. “FAST, private reset” is intentionally not labeled an unchanged frozen-driver replay: see [the Q256 integration finding](FAST_Q256_CORRECTION_BUG.md).

The HiFi2 rows retain finer K/V values than the LoFi RNE5 rows; their cross-comparison changes both multiply fidelity and operand preparation. Their own LUT-off/on comparison changes only exp refinement. The BF16 K/V controls (`hi2-bf16-lut-*`, five timed replays) retain identity RNE8/BF16 K/V and Q7 preparation; combined timing includes the identity copies. They reduce LUT-path normal L2 to about 0.75%, separating shared-exponent quantization loss from exp/fidelity loss.

Removing BFP8 storage from the LoFi native path improves normal L2 slightly (32K 2.824→2.742%, 256K 2.798→2.713%) but reduces measured combined throughput (153.71→149.15 and 159.58→151.38 TFLOP/s). This changes quantization as well as storage bytes; it is not a lossless storage-width ablation. See `native-storage-*.json`.

The denominator-only rows are **not generally robust recommendations**. With repeated K/V blocks and exact preprocessing checks, full compensation preserves its short-block output, while denominator-only grows to about 32% L2. Constant/common V also expose large long-context error. Ordinary normal-input error alone would conceal this weakness.

## Numerical mechanisms established by the experiments

1. **Packing bias matters.** A device RNE preprocessor substantially improves native BFP4 over the default multi-stage packing route. Normal, tie, wide-exponent, and zero probes match independent oracles. Supported exponent ranges are explicit; this is not unrestricted IEEE special-value support.
2. **FP32 destination does not recover discarded multiplicand bits.** LoFi Q7/KV5 rounding helps; BFP4 still removes much more information. FP32 recurrent state prevents BF16 summation drift but cannot undo coarse K/V quantization.
3. **Exp quality is a separate cost/accuracy axis.** Two-segment LUT correction reduces LoFi FP32 normal L2 from about 2.8% to 2.2%. A LOADMACRO schedule preserves every output bit in four raw/macro paired tests while improving the LUT path's measured throughput. Direct seven-bit exp, intended to match BF16 P consumption, passes its primitive oracle but shows no consistent attention-error benefit and worsens normal-input L2 in the tested full/denominator-only cases.
4. **Lower reconstruction MSE does not guarantee lower attention error.** Adaptive BFP4 scale search improves normal-input error modestly. For one K4/V4 outlier test it worsens attention L2 from 32.45% to 34.93%, despite improving local quantization MSE. Clipping sensitive keys is not interchangeable with reducing average element error.
5. **Q, K, and V centering have different algebra.** Centering K is a softmax-invariant row shift in exact arithmetic. Centering Q requires a Q-mean×K score correction. Exact V centering subtracts a mean and restores that same mean after attention. Coarse quantization can subsequently introduce a new mean error; correcting the effective represented mean is an additional bias treatment, not a requirement of exact centering algebra. See [value smoothing](VALUE_SMOOTHING.md) and [Q correction](Q_CENTER_HADAMARD.md).
6. **Common modes can affect the accurate path too.** In the controlled 32K common-K diagnostic, unchanged accurate kernels improve from 0.736% to 0.178% L2 after device K centering. On normal inputs, the extra BF16 preparation spill instead changes 0.181% to 0.240%. This supports conditional treatment, not always-on centering. Exact-shift and actual-prepared-input references distinguish the mechanisms.
7. **FP32 FPU accumulation is not an ideal FP64/IEEE dot product.** Independent silicon discriminators establish product-alignment loss within reduction groups. The internal arithmetic model agrees with the tested device outputs; it is an internal explanatory model, not a public ISA guarantee. See [the matmul audit](matmul-fp32-floor-audit.md).

## Stress qualification and targeted transformations

The native-exp suite contains 82 cases at 32K and 82 at 256K (H2, 22 cores): two seeds, 13 distributions, LoFi B8/B4 and accurate controls, plus K-centering diagnostics. Complete coverage and finite/replay gates pass; **this does not mean every distribution passes a common numerical cutoff**. LoFi B8 normal error stays around 2.8%, but outliers, scaled QK and common modes can be substantially worse. The accurate control also has isolated stress cases above 0.5%. Full tables are in [Numerical findings](NUMERICAL_FINDINGS.md).

Device K centering is a successful targeted repair: the 256K LoFi B8 common-K cases improve from 44.82–45.44% to 2.35–2.36% L2 across two seeds. This is a separate preprocessing intervention, not an effect of LUT exp and not enabled in the uncentered stress tables. It supports a conditional Sage-inspired transformation; its cost and normal-input effects must remain part of any deployment decision.

Signed H16 Q/K rotation helps sparse outliers in coarse K, but does not consistently improve normal or scaled-QK inputs and can amplify common-mode problems. H128 is not uniformly better. V transposition changes shared-exponent groups from 16 channels to 16 tokens at one channel. The N1024/H2/4-core hardware test reduces quiet-channel L2 from 81.6% to 11.8%, with PCC 0.617→0.993; aggregate L2 alone barely exposes that repair.

The full-compensated K8/V4 long-context test confirms quiet-channel improvement: **79.66→12.01% at 32K; 80.58→12.41% at 256K**. Normal-input L2 is essentially unchanged, and aggregate channel-outlier L2 slightly worsens. The real V transpose costs about 0.52 ms and 3.73 ms respectively. Combined throughput is about 191→188 TFLOP/s at 32K. Interleaved 32K timing confirms about 1.6% extra combined time for the N-axis route; read barriers of 2/8/16 tiles show no material tuning win. Small sequential 256K differences remain susceptible to clock variation. A second 32K seed confirms quiet-channel improvement, 81.50→12.15%. See [V-axis results](V_TRANSPOSE_DESIGN.md).

This is not uniquely a BFP4 benefit: V8 quiet-channel error improves **11.45→3.14% at 32K and 11.68→3.73% at 256K**. Normal L2 remains approximately 3.15%/3.73%; 32K combined throughput changes 190.56→187.63 TFLOP/s. BFP4 must justify its extra representation loss against this wider-storage control, not only against badly grouped BFP4. BFP4 still saves 47.1% of V storage bytes and 23.5% of combined K8/V bytes; in this compute-heavy full-compensation loop it does not buy proportional extra throughput.

The combined K4/V4 recipe uses independent H16, V-axis, and adaptive-V switches. At 32K, H16 reduces sparse-outlier L2 from 32.44% to 17.18%, but raises common-K error from 80.35% to 95.72% and common-Q from 33.97% to 48.65%. H16 plus V-axis reduces the combined K-outlier/quiet-V-channel stress from 83.94% to 23.06% quiet-channel L2. Adding adaptive V gives only a small further aggregate improvement and costs additional preprocessing. See [the combined experiment](COMBINED_RECIPE.md); these are conditional repairs, not a universally better recipe.

## Mean quantization-error correction

A separate experiment keeps the quantizer input as **original V**, computes `delta = mean(V) - mean(actual LoFi-consumed Vq)` on device, and adds delta once after attention. The attention loop and its buffers are unchanged. This removes unweighted quantization bias in ideal arithmetic; it does not repair weighted quantization error or recurrent accumulation.

At N1024/H2, full-compensated K8/V4 improves normal L2 **12.18→9.78%** and uniform-attention L2 **12.45→0.87%**. K8/V8 improves normal **3.09→2.94%** and uniform **1.78→0.87%**. Constant-V output is unchanged, as expected when there is no V quantization error to correct. Two bit-exact combined replays, exact preparation/correction checks, and original-input immutability pass.

Long-context K8/V4 normal L2 improves **12.01→9.81% at 32K and 12.29→10.06% at 256K**; K8/V8 improves 3.15→3.04% and 3.74→3.62%. Uniform-attention V4 improves 11.67→1.72% and 12.38→4.21%. Here uniform means **Q=0 with normal K/V**, not uniformly distributed random tensors. The measured delta-estimation error is only about 0.14–0.25 L2 percentage points, so it cannot explain most remaining attention error. Extra stages cost about 2–2.5 ms at 32K and 16–19 ms at 256K. The three-replay combined comparisons give roughly 7–8% extra time at 32K and sub-percent differences at 256K, but varying attention times prevent treating the latter as pure overhead estimates. See [the implementation and limits](VALUE_MEAN_ERROR.md).

## Representation-only noise budget

The [CPU noise-isolation study](QUANTIZATION_NOISE_BUDGET.md) uses exact FP64 QK, softmax, PV and correction, retaining original BF16 Q/K/V as reference. Rounded native BFP4 K/V still gives **16.34–16.70% L2** across five normal input sets at 4K/32K/256K. Each operand has about 11.7% reconstruction error; approximately orthogonal K- and V-induced output errors explain the roughly 16.6% combined scale. Growing context shrinks both output and quantization noise, so relative error need not vanish.

Exact mean-error correction reduces KV error to 14.97–15.16%; uniform Q is the exact control, with zero corrected FP64 error. A representative NVFP4 codec gives about 13.35–14.04% uncorrected KV error in the same representation-only study. This is neither a SageAttention execution nor a universal lower bound: different codecs, transforms, residual representations and model activations can behave differently. It establishes that improving Blackhole arithmetic alone cannot eliminate the observed native-BFP4 normal-input loss.

## Final-normalization negative control

An isolated 2×2 test changes only the final BF16 reciprocal (existing seven-bit versus eight-bit routine) and final output-scale ELWMUL (HiFi2 versus HiFi4). Exp, attention matmuls, recurrence, buffers and data movement remain unchanged. At N1024/H2, K8/V8 normal L2 is 3.092% baseline, 3.085% with scale-only HiFi4, 3.094% with reciprocal-only, and 3.113% with both. Constant-V error worsens; uniform-attention cases are unchanged. All 16 rows pass exact preparation, finite, original-input integrity and two bitwise replay checks. There is no demonstrated accuracy benefit warranting long performance qualification.

The existing HiFi2 scale consumes only the high seven reciprocal bits; the HiFi4 scale control removes that specific limitation. The final denominator-reduction matmul is still LoFi, so this does not qualify fully accurate final normalization. See [the reciprocal/scale control](recip8_streaming/README.md), including the initial compile failure and its math-thread-only assertion fix.

## Block order exposes an additional cheap-exp weakness

Jointly permuting complete 512-token K/V blocks preserves exact attention and the represented input values. A 24-case test checks identity, reverse, and descending K-block-RMS order at 4K/32K, normal and block-scaled K, on LoFi FP32 B8 and full-compensated LoFi BF16 B8. Original/permuted FP64 references agree; exact represented-value preparation, finite outputs, input integrity and two bitwise replays pass.

Normal-input aggregate L2 stays in the same band, but the actual output changes substantially with order. At 32K, reversing K/V changes the FP32 output by **2.55% relative L2**, and compensated BF16 by 2.81%, measured over every output element. These are interorder differences, not additional error added to the original-reference L2.

An 18-case exp-only follow-up preserves the FP32 LoFi inputs, state, chunks and buffers. Its native control reproduces the previous complete-output hashes exactly. Reverse-order output differences are:

| Exp family | 4K interorder L2 % | 32K interorder L2 % | 32K identity-order original-reference L2 % |
|---|---:|---:|---:|
| Native cheap grid | 2.370 | 2.551 | 2.820 |
| Same grid, macro-LUT refinement | 0.791 | 0.866 | 2.198 |
| Existing cubic / finer grid | 0.484 | 0.522 | 2.142 |

This supports a substantial contribution from approximate-exp inconsistency during online maximum updates. Native score exp has input-dependent relative error, while online rescaling uses a more accurate exp; in general `F(x-m_old) * G(m_old-m_new) != F(x-m_new)`. FP32 accumulation cannot repair that identity. The same-grid LUT comparison isolates refinement; the cubic comparison changes the exp grid and approximation family more broadly. Remaining order sensitivity is not uniquely attributed to accumulation: represented P, BF16 row maxima and other arithmetic are still finite precision. See [the evidence and derivation](BLOCK_PERMUTATION.md). No block-reordering performance optimization or model-quality claim is made.

A separate [36-case idealized FP64 online model](EXP_ORDER_CPU_MECHANISM.md) reproduces the mechanism with continuous native exp but exact rescaling, state and PV: at 32K, reverse-order output differs by 2.576%, versus numerical roundoff with exact or constant-bias exp. Fixed global maxima remove order dependence but retain 1.827% original-reference error and require an unpriced prepass. These H1 inputs differ from the device experiment; the model is not a quantitative decomposition of device error.

An [84-case lattice-maximum extension](EXP_ORDER_LATTICE_CPU.md) avoids that prepass in the model by restricting maximum updates to log(2) steps. It removes order dependence but preserves about 1.81% normal error. More importantly, adding a mathematically harmless half-octave common score offset changes its 32K output by 3.49%. BF16 storage also moves snapped maxima off the exact lattice. This is an invariance tradeoff, not a free accuracy fix. The [Blackhole feasibility analysis](LATTICE_MAX_FEASIBILITY.md) identifies integer-code, range and SFPU scheduling requirements; no lattice device kernel or performance claim is made.

## Expanded exp stress qualification

The late-window [native/LUT stress cross-check](EXP_STRESS.md) covers ten distributions at N1024 with all-output references and six at 32K with sampled Q/all K/V, two seeds, four routes and fresh-process repetitions. At 32K, LoFi B8 normal L2 is 2.82–2.83% native and 2.20–2.21% with LUT, but outliers reach about 7% and common K about 46%; LUT does not repair these remaining failures. HiFi2 with BF16 K/V and LUT gives 0.748–0.752% normal L2, but common K remains 1.27–1.32% and common Q is seed-sensitive. These data support separate numerical axes, not a universal accuracy band.

The targeted 256K extension confirms stable normal bands: LoFi native 2.77–2.80%, LoFi LUT 2.20–2.22%, and HiFi2/BF16 LUT 0.750–0.752%. However, LoFi outlier L2 reaches 10.35–15.66% native and 10.38–15.95% with LUT, versus 0.785–1.371% for HiFi2/BF16 LUT. LoFi common K remains about 45%, versus 1.23–1.28% for the HiFi2/BF16 LUT point. The three late-window suites together pass their recorded gates for 304 records and 152 fresh-process complete-output comparisons. Their differing replay/input-hash guarantees are disclosed in the audit; none uses a universal L2 acceptance cutoff.

The [common-V output-floor analysis](COMMON_V_DERIVED_METRICS.md) separates unavoidable BF16 output loss from additional error. At 32K, nearest BF16 output rounding alone gives roughly 126% L2 relative to the small centered signal; the HiFi2 native/LUT records reach that error norm, while LoFi has 13.4–13.6 times the floor error norm. At 1K, LoFi has higher PCC despite about 2.5–2.6 times the BF16-floor error norm, whereas HiFi2 is almost floor-limited. These are algebraically derived residual metrics from recorded error norms and regenerated original references, not new output-tensor verification. Neither PCC alone nor a common-mode-dominated aggregate L2 is adequate qualification.

## What is not established

No real model activation capture, end-to-end model score, NVIDIA device comparison, production integration, backward pass, masked/causal attention, GQA/MQA, or general head-dimension qualification is claimed. The new [captured-input runner](CAPTURED_INPUTS.md) passes an explicitly synthetic six-variant interface smoke; that is readiness to evaluate supplied captures, not model validation.

Do not promote a BFP4 option on normal L2/PCC alone. The next selection should use representative model layers and include quiet-channel and common-mode diagnostics. The established accurate modes remain the reference/fallback choices while low-bit work is experimental.

## Recommended engineering direction

Separate correctness prerequisites from new numerical choices. First carry the private Q256 correction-address fix into a narrowly scoped integration proposal with distinct-KV/max-update regression tests and explicit dispatch/range contracts. Then qualify wider-storage LoFi and the HiFi2/BF16-LUT intermediate point on the intended inputs. Keep BFP4 focused on a concrete capacity or bandwidth objective, with B8 controls and preprocessing costs included. The [ranked engineering plan](ENGINEERING_NEXT_STEPS.md) spells out candidate disposition, integration gaps and stop/go evidence. No production integration was performed here.

## Reproduction and audit

Use the exact commands/configuration stored in each record and the linked driver README. Preserve failed or superseded evidence as such; do not relabel it a passing current-source result. The read-only validator separates missing evidence, recorded gates, historical source omissions, and current source drift:

Device work used container `yyzo-bh-08-special-cglagovich-for-reservation-220027`, from `/localdev/cglagovich/tt-metal-blackhole-20260908`. The local worktree is `tt-metal-blackhole`, branch `cglagovich/blackhole-work-20260908`. The four selected variants remain pinned at `637d956c8874d356c9080e467b9a1664133fa780`; the final Git commit containing this report identifies the new research checkpoint.

```sh
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_research_checkpoint.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_final_smokes.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_exp_stress.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_exp_stress.py --plan experiments/sdpa-l2/bfp4-lofi-v2/exp-stress-long-plan.json
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_exp_stress.py --plan experiments/sdpa-l2/bfp4-lofi-v2/exp-stress-256k-plan.json
python3 -B experiments/sdpa-l2/validate_selected_variants.py
```

An evidence PASS is not a universal accuracy acceptance threshold. Historical provenance warnings are disclosed in the audit rather than silently repaired in old records. The [final current-source qualification](FINAL_QUALIFICATION.md) records 85 complete attention outputs bit-identical to their pre-format controls, four exact BFP4 primitive checks, and 86 distinct passing tests. Its reference scopes and omitted gates are explicit; it does not requalify historical performance or real model quality.
