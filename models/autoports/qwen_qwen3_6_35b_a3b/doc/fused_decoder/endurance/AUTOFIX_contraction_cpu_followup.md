# CPU recurrence replay of the endurance probe

**The current step's contraction arithmetic is much smaller than the incoming-state discrepancy. Canonical CPU recurrence on the exact TT incoming state and inputs reproduces the device core to relative L2 5.6e-6–3.7e-5. Replacing only the incoming state with HF state restores canonical attention PCC to at least 0.999916 at all seven failing positions.** This localizes most attention error to the accumulated state history, rather than the current contraction or output normalization. It does not prove which small repeated error produced that history; a contraction-only full-history control is still needed to measure its cumulative contribution.

All work here used CPU PyTorch and safetensors; no TTNN imports, device calls, implementation edits, or test edits. Inputs are the ten `endurance_state_probe.json.pos*.pt` files at positions 128, 1024, 2048, 3979, 5011, 6595, 7551, 7668, 7703, and 8157. Machine-readable results and input SHA256 values are in [contraction_cpu_replay.json](contraction_cpu_replay.json). The parent reports the older 42ac baseline also fails the 8192-token test; therefore the newer matmul/post-norm changes cannot be assumed to originate all endurance failures.

## Validated local replay

The probe saves actual incoming recurrent state and normalized Q/K/V/g/beta for both implementations. I repeated TT Q/K heads twice, applied the exact HF `1/sqrt(128)` query scale, and computed CPU FP32:

```
decayed = state.float() * exp(g.float())
memory = sum(decayed * k.float(), key_dimension)
delta = (v.float() - memory) * beta.float()
new_state = decayed + k.float() * delta
core = sum(new_state * q_scaled.float(), key_dimension)
```

Using saved HF inputs, both BF16 core and BF16 final cache match the saved HF tensors **bit for bit at every sampled position**. Canonical HF output processing uses the saved HF input norm for Z projection, casts core to BF16, applies the exact HF gated RMSNorm rounding order, then performs CPU BF16 output projection. This reproduces saved HF attention bit for bit at all sampled positions. Checkpoint tensors are from the pinned revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.

The following controlled substitutions use canonical HF output processing and the same HF Z in every column, isolating recurrence sources. All values are attention PCC versus saved HF attention.

| Position | Actual saved TT attention | CPU recurrence with TT state + TT inputs | TT state + HF current inputs | HF state + TT current inputs |
| --- | ---: | ---: | ---: | ---: |
| 3979 | 0.995789 | 0.995922 | 0.995982 | 0.999983 |
| 5011 | 0.995303 | 0.995165 | 0.995071 | 0.999984 |
| 6595 | 0.995284 | 0.995484 | 0.996802 | 0.999916 |
| 7551 | 0.997781 | 0.997847 | 0.997765 | 0.999970 |
| 7668 | 0.999414 | 0.999416 | 0.999410 | 0.999974 |
| 7703 | 0.999047 | 0.999028 | 0.999019 | 0.999985 |
| 8157 | 0.999691 | 0.999697 | 0.999677 | 0.999984 |

Current inputs matter somewhat at 6595, but incoming state dominates every failing-position attention discrepancy. Incoming-state relative L2 is 0.01616–0.01823 at these positions, versus 0.01051 immediately after prefill (128). It is not monotonically increasing across the later samples.

## What can and cannot be measured about the memory matmul

The probe does **not** save `memory` directly. An exact measurement of its isolated device error is therefore unavailable. However, comparing device final FP32 state to the CPU update on *identical TT operands* gives relative L2 1.41e-5–4.01e-5 across all samples. Device core versus CPU on identical TT operands differs by only 5.6e-6–3.7e-5 relative L2. Replacing that current-step device core with canonical CPU core changes attention PCC by only a few millionths.

For a useful estimate, let `E = saved_device_new_state - canonical_CPU_new_state`, and project `E` onto the per-head key direction. Because a memory perturbation `dm` induces `E ≈ -beta*k*dm`, the least-squares effective memory perturbation is:

```
dm_effective = -sum(E * k, key_dimension) / (sum(k*k) * beta)
```

The residual after this rank-one fit is just 0.00078–0.00260 of the state-error norm. Thus over 99.999% of its squared norm lies in the expected update direction, strongly supporting contraction arithmetic as the source of the **small local update error**. Effective memory relative L2 is 0.000411–0.000731. This is an inferred effective perturbation, not a directly captured matmul tensor; it can include other arithmetic error aligned with the same direction.

CPU FP32 decay reassociation, `sum((S*exp(g))*k)` versus `sum(S*k)*exp(g)`, changes memory by only 8.2e-8–2.0e-7 relative L2. This is thousands of times smaller than the effective device perturbation, consistent with FP32-key input narrowing/reduction in the matmul path rather than simple decay reassociation. A hardware memory capture is still needed to separate the precise device components.

Although small locally, these update differences alter approximately 0.82%–1.23% of state elements after BF16 rounding for the next cache boundary. Therefore the CPU one-step evidence **does not rule out cumulative matmul influence**. The parent should compare a contraction-only SFPU revert from the same prefill across the full history, while retaining all cache-boundary behavior and other settings. The older baseline's failure establishes that the new contraction is not necessary for endurance failures, but does not determine whether it worsens or improves their magnitude.

## Nonlinear head and output sensitivity

At 5011 and 6595, head 23 core relative error is respectively 0.351 and 0.337, with HF core RMS 0.001794 and 0.001679. At 3979, head 25 relative error is 0.136 with RMS 0.000380. These head-specific deviations can matter far more after per-head normalization and output projection than aggregate state PCC suggests. Exact HF output processing preserves the poor attention scores in the table; its arithmetic is not the dominant source.

The worst final-output position, 7703, is instructive: actual attention PCC is already 0.999047, while final output/contribution PCC is much lower (~0.98402/0.98348 in the endurance artifact). Exact HF post-attention RMSNorm computed on the **same TT residual** differs from saved TT norm by relative L2 0.003063 (PCC 0.9999958). Across all samples this norm-only discrepancy is 0.00270–0.00369, so sharded norm is a small local perturbation. It cannot alter future recurrence in this fixed-input test, but downstream expert nonlinearities or routing-weight changes can amplify local differences.

Do not infer native expert selection from CPU `torch.topk` of saved BF16 logits: ties can choose a different eighth index. At 7703, experts 25 and 173 tie at BF16 logit 9.5. CPU topk of the saved TT logits picks 173, but the probe's **actual native IDs select 25**, matching HF's expert set. All seven failing positions except 3979 have matching HF/native expert sets (orders can differ). At 3979 the sets differ substantially. Thus an eighth-expert mismatch does not explain the worst 7703 result. This report does not reproduce the expert MLP, so it cannot assign that remaining final-output amplification to MoE arithmetic versus normalized-input/routing-weight sensitivity.

## Narrow next step

The most informative recurrence control is the full-history contraction-only revert; a one-step replacement will barely move the failing attention scores. Capture actual `memory` if an exact contraction error measurement is needed. Separately, use the saved TT residual at 7703 to run the same MoE on canonical CPU norm versus saved TT norm, preserving the actual expert IDs and their corresponding weights. That distinguishes small post-norm sensitivity from a downstream implementation discrepancy without mixing it with recurrence history.
