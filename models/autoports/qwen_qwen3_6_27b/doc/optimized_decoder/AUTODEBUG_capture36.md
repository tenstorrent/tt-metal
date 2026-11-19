# CPU attribution of the 512-step failures

CPU-only analysis of `capture36/layer0_step240.pt` (user7) and `layer0_step411.pt` (user29), produced by boundary32 with its original policy. No device run or main source edit. Reproduction:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.analyze_long_capture \
  --capture-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture36 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture36_cpu.json
```

## Result: accumulated state mismatch dominates these two tokens

The compact SFPU kernel matches the CPU IEEE FP32 recurrence on identical TT raw inputs and stored state: maximum core error `1.49e-8` / `1.12e-8`, maximum state error `1.19e-7` at both steps. This sharply rejects an appreciable per-step SFPU recurrence implementation error at these coordinates. It does not establish bit identity over every input.

Counterfactuals below take the indicated recurrence inputs/state, then run the complete HF gated normalization, real BF16 output projection, residual, hidden norm, real BF16 MLP, and final residual. TT z is retained throughout this table. Actual captured HF outputs are the comparison target.

| Current recurrence inputs / initial state / qk normalization | Step240/user7 output PCC | Step411/user29 output PCC |
|---|---:|---:|
| Actual TT output | .980315624 | .985831938 |
| Captured TT core + exact HF tail | .981837915 | .988172060 |
| TT inputs / TT old state / HF qk norm | .982003412 | .989343848 |
| HF inputs / TT old state / HF qk norm | .985827887 | .989735126 |
| TT inputs / HF old state / HF qk norm | **.999213407** | **.999898696** |
| HF inputs / HF old state / HF qk norm | .999984722 | .999858872 |

Changing just the current q/k normalization or all current recurrence inputs leaves failures. Replacing the old state with HF's state clears both by a large margin, even with current TT inputs and TT gate. Thus fix the history-generating contract; further changes to the direct recurrence arithmetic are low priority for these failures. This experiment cannot by itself divide that accumulated state difference between historical cache rounding, historical input norm/projection errors, and prefix error. The independent measured cache-rounding control reducing failures11->2 and round+unsharded reducing2->1 is consistent with multiple accumulated contributors.

The full CPU HF tail reconstructed from captured HF core/z agrees with HF output at PCC `.999999989` / `.999999756`. The small reconstruction floor includes one-user versus original B32 CPU matmul/reduction behavior. These counterfactuals are diagnostic, not new hardware acceptance rows.

## Where the small state error becomes large

Old-state relative RMS differences are 2.084% / 2.033%, despite whole-state PCC .99981/.99982. The affected heads have small readouts:

| Step,user,head | HF core RMS | Core PCC | Gated squared-error share | TT/HF old-state fitted gain |
|---|---:|---:|---:|---:|
| 240,7,4 | .0005548 | .942402 | 64.14% | .990849 |
| 240,7,41 | .0008219 | .997369 | 19.43% | .989455 |
| 240,7,5 | .0010456 | .999053 | 8.22% | .991591 |
| 411,29,29 | .0004701 | .971979 | 43.64% | .993472 |
| 411,29,44 | .0006317 | .993484 | 17.50% | .989476 |
| 411,29,25 | .0013403 | .995547 | 16.52% | .991083 |

The state gain is descriptive, not an authorized correction factor. Orthogonal state error also exists. Do not compensate outputs by fitted constants.

## Hidden norm and projection attribution

Exact checkpoint BF16 projections on the captured **TT input norm** account for nearly all current projection discrepancy:

- QKV relative error caused by replacing HF input norm with TT input norm: .5631% / .5354%.
- Actual TT QKV versus exact BF16 projection on the same TT input: .1085% / .1115%.
- Actual TT QKV versus HF: .5594% / .5335%.
- Z has the same pattern: hidden-norm sensitivity .5534% / .5564%, versus same-input projection error .1116% / .1163%.
- Input-norm fitted TT/HF gain is .996323 / .995854, with orthogonal relative error .2869% / .2649%.

This favors repairing the measured sharded norm path before spending another matmul on projection precision. It does not prove hidden norm is the only historical state source. Folded norm weights in boundary32 are already prepared as FP32 `1 + weight`, so a BF16 folded-weight rounding hypothesis is contradicted by source. The current two-BFP8 residual correction keeps projections close to exact-weight same-input results; BFP8 defaults should remain.

## MLP is an amplifier, not the leading arithmetic fault

Exact HF MLP evaluated on captured TT post-attention norm and TT residual still yields output PCC `.981085607` / `.987039560` versus HF. It agrees with actual TT output at `.999985961` / `.999953472`. Replacing MLP arithmetic cannot repair the observed failures while retaining those upstream inputs.

HF gated normalization on identical TT core and gate agrees with actual TT gated values at `.999999050` / `.999999973`; the boundary32 gated-norm implementation is already sufficiently close at these coordinates. The attention PCC `.997176/.999012` is amplified through residual normalization and MLP into final `.980316/.985832`.

## Next decisive evidence

Analyze the remaining round+unsharded failure at209:25 using the same helper. If replacing old state still repairs it, compare a legal HF-normalization/state-rounding sequence and the prefix state. A diagnostic-only exact-HF-prefix-state initialization separates inherited prefix error from subsequent drift; it must never become the delivered runtime path. If instead current inputs dominate209:25, the same helper's exact-weight projections and same-input MLP comparisons identify the remaining local boundary. No .995 threshold reduction is supported.
