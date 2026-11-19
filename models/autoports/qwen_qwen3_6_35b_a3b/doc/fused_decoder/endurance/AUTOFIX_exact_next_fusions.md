# Remaining exact recurrence/normalization fusion candidates

The preparation record below is retained as written. Subsequent parent-run device evidence is appended at the end; statements that device checks are still required describe the original preparation state.

Base: frozen `selected_fused_decoder.py.snapshot`, SHA256 `61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178`. Production/core tests untouched; no device execution. Searched work log and endurance reports for prior attempts. These four candidates do not repeat norm-L1, weight staging, compact-TILE gather, BF16 SiLU-input, FMA refinement or native recurrent-FDR controls.

Four isolated `exact_next_<name>.py.snapshot` files and `.py.patch` siblings, plus `exact_next_combined.py.snapshot`, are recorded in `exact_next_fusions.provenance.json`. Combined includes the independent substitutions, not additional algebraic changes. Syntax checks pass.

## Candidate 1: recurrent_q_scale

Keep query head duplication and permutation, but omit its standalone multiply by128**-0.5. Apply `MUL_UNARY_SFPU(128**-0.5)` to the Q operand of the final `multiply(new,q)` using `input_tensor_b_activations`. Q has already crossed its L2 BF16 round/widen boundary and remains FP32. No activation is applied to a BF16 operand. The scaled Q is only used by this product, so the candidate removes one materialized FP32 operation without modifying memory/update arithmetic or the16-then8 output contraction.

Current native BinaryNG FP32 multiply selects SFPU. `unary_op_utils.cpp:467–471` emits `mul_unary_tile` with the scalar's FP32 bits; the original scalar multiply likewise uses FP32 scalar/data and the SFPU. `binary_ng_program_factory.cpp:1103–1104` binds the activation to the original RHS dtype; its preactivation CB retains FP32. This is not folding scale into k/decay or combining adjacent multiplications algebraically. The operand product remains rounded before the final product. Device bitwise comparison is still required.

## Candidate 2: l2_sum_boundary

Replace:

```
sums = unary_chain(sum(squares), round_widen)
shifted = add(sums, epsilon, activations=round_widen)
```

with:

```
sums = sum(squares)
shifted = add(sums, epsilon,
              input_tensor_a_activations=round_widen,
              activations=round_widen)
```

This removes one standalone chain. Preserve both BF16 boundaries: sum rounds before epsilon; epsilon result rounds afterward. Sum, inverse, final product and all output dtypes remain unchanged. The source operand is FP32, so preactivation packing retains widened BF16 exactly. This differs from the rejected BF16-Z input activation: that case packed a nonlinear FP32 SiLU result into a BF16 intermediate CB; here the desired widened result is already BF16-representable in FP32.

## Candidate 3: norm_square_input

For the two square terms in the square-root residual, replace `subtract(lhs,multiply(t,t))` with `subtract(lhs,t,input_tensor_b_activations=[SQUARE])`, separately for high and low. Keep the middle2*high*low term and the left-associated three subtractions unchanged. Two standalone square-product tensors disappear.

`ckernel_sfpu_square.h` on Blackhole computes each FP32 square as `v*v`; its BF16 rounding branch is disabled under FP32 destination. The existing `ckernel_sfpu_binary.h:129–130` multiply uses `in0*in1`. `unary_op_utils.cpp:832–850` selects `square_tile` for FP32. The preactivation is stored/reloaded through an FP32 CB before subtraction; this is not an FMA. A complete small residual must be compared bitwise because cancellation can amplify any native implementation difference.

## Candidate 4: norm_mean_scale

Replace standalone total*1/2048, reshape, epsilon add by a single epsilon add with FP32 input activation `MUL_UNARY_SFPU(1/2048)`, followed by the same logical reshape. This removes one scalar multiply. The power-of-two multiplication remains before the epsilon addition; the reshape only moves values. Keep shifted variance separately because the sqrt residual uses it. Do not fuse sqrt away or alter rsqrt/refinement arithmetic.

## Required discriminator before benchmarking

`tests/exact_next_fusion_probe.py` uses only explicit frozen/candidate snapshots. Its CPU-only fixture check passed (`exact_next_fusion_cpu.json`), loading real checkpoint norm/conv weights, five real singleton norm rows, a32-row prefill block, and three saved recurrence captures.

The device mode, to be run by the hardware owner, checks:

- Candidate1: full FP32 state×Q product and complete recurrent core/state, using the same captured operands for old and new paths.
- Candidate2: epsilon-shifted FP32 cut and complete normalized Q/K output, with CPU reconstruction of actual captured convolution history/current projection and checkpoint taps feeding the native old/new L2 adapter.
- Candidates3/4: hooks in the frozen norm capture actual device-produced high/low/total operands. Each fused subtract/add is compared against the original complete cut before downstream BF16 rounding. The complete candidate norm is then compared with the frozen norm on every singleton and the32-row prefill block.

Every consumed FP32 bit pattern is gated; signed-zero differences are reported but not waived. No timing or hardware-pass claim is made. No monkeypatch remains installed after a norm call (try/finally). Device closes in finally; source/probe/capture hashes and failure stage are recorded.

```
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.exact_next_fusion_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/exact_next_fusion_device.json
```

## Additional structural candidate, not part of these snapshots

Q/K duplication can be expressed by outer broadcasting: view state/products `[1,32,128,128]` as `[1,16,2,128,128]`, and permute original16-head K/Q to `[1,16,128,1]`, then view `[1,16,1,128,1]`. Their product broadcasts the group-of-two axis; metadata-reshape the result back to4D before the unchanged `_recurrent_sum`. For the update, view delta `[1,32,1,128]` as `[1,16,2,1,128]`, multiply by the same5D K, reshape back and add to decayed. Q scale must still precede the output product, either explicitly or through candidate1's tested activation. Beta/decay/state layouts remain32-head.

This removes two repeat_interleave calls but adds metadata views and changes native binary rank geometry. BinaryNG supports broadcasting through rank6 (`binary_ng_device_operation.cpp` rank-validation loop), and the last H/W dimensions of all these reshapes remain identical, satisfying the ordinary reshape view path. CPU verified all524288 product values with the old/new head mapping. Native product-bit and full-trace checks remain necessary; no speed claim or implementation change is made here.

## Exact blockers retained

- Parallelizing the512 ordered projection fold or changing the16-then8 state contraction association does not preserve the canonical FP32 sequence.
- Replacing `add(decayed,multiply(k,delta))` or cancellation residuals with addcmul/mac may introduce FMA behavior; prior device residual evidence already rejected that shortcut.
- Fusing nonlinear activations into BF16 input operands can introduce premature BF16 packing, as the rejected SiLU-Z control demonstrated.
- Moving all normalization intermediates to L1 and transiently staging ordered weights in L1 were already measured regressions and are not proposed again.

Elapsed wall time alone does not establish eight hours of device execution. This investigation supplies candidates; it does not establish the70% roofline target, device-time budget, or optimization stop condition.

## Parent-run device evidence, 2026-11-13

The hardware owner ran `exact_next_fusion_device.json`: **45/45 strict bitwise checks passed**, with clean device close. Candidate hashes match their original preparation snapshots. These are the primitive and complete-adapter checks described above; this subagent did not execute hardware.

Five diagnostic layer-0 S256/context512 snapshot benchmarks each recorded 60 decode samples. Frozen reference: `../qualified_moe_padding/final_0.json`, whose provenance binds production source to the frozen 61ac211a hash. The source-bound artifact paths and SHA256 hashes are recorded in `exact_next_fusions.provenance.json`.

| Candidate | Decode median ms | Difference from frozen, us | Latency change |
| --- | ---: | ---: | ---: |
| Frozen | 1.455763 | 0 | 0% |
| recurrent_q_scale | 1.452663 | -3.100 | -0.213% |
| l2_sum_boundary | 1.458063 | +2.300 | +0.158% |
| norm_square_input | 1.449030 | -6.733 | -0.463% |
| norm_mean_scale | 1.451231 | -4.533 | -0.311% |
| combined | 1.441802 | -13.962 | -0.959% |

All five retain exactly the same reported prefill PCC (0.999962395060449), prefill contribution PCC (0.9999594863488576), and eager/traced decode PCC (0.9999779064754709) as frozen; all report deterministic output. These S256 checks do not establish arbitrary-length or long-stream qualification. No candidate is selected for production.

The isolated L2 result is **2.300 us slower (+0.158%)**, a small observed regression whose cause is unresolved. Frozen sample interquartile range is 1.453930–1.457327 ms; L2 is 1.456094–1.460502 ms, with overlapping ranges. The 60 samples are repeated replays within one run, so treating them as independent process-level replications would overstate confidence. This result establishes neither an intrinsic L2 slowdown nor mere noise. The combined result is 13.962 us faster (-0.959%), but its component effects need not add. The parent will compare a combined-minus-L2 snapshot against combined before choosing whether to retain L2; if that difference remains similarly small, paired/interleaved complete-run measurements can resolve drift.

Rank-5 head broadcasting is now prepared separately in `exact_next_rank5_heads.py.snapshot`, with its own memo/probe. Its independent device probe subsequently passed (`rank5_recurrence_device.json`); its full-path benchmark remains pending and is not part of the five results above. Preparation snapshots, production source, core tests, and pattern assessment were not edited for this evidence update.
