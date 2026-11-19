# Exact frontend recurrence: CPU isolation

Source/CPU investigation only. No TTNN import, device experiment, implementation edit, or test edit was performed by this investigator. The device tensors were supplied by the parent run. Source candidate: `../endurance_gates_norm_l2_conv.py.snapshot`. Evidence: `../endurance_exact_frontend_probe.json.pos{128,129,256,1024,3032,3979,4904,5011,6595,7551,7668,7703,8157}.pt`; per-file SHA256 and complete measurements are in `exact_recurrence_cpu_replay.json`.

## Finding

The remaining one-step recurrence mismatch is overwhelmingly aligned with the rank-one update, not the multiplicative decay. Replacing fused addcmul with separately rounded multiply/add is a weak first experiment: its CPU effect is about 1e-8 relative, while the observed same-input device state discrepancy is 8.40e-6–2.85e-5. The first discriminating control should instead replace only the memory contraction with accurate FP32 SFPU multiply/reduce on the already-decayed state.

**Correction to the earlier TF32-input explanation:** in every new probe, the incoming state and normalized key are exactly BF16-representable, despite being stored as FP32. Narrowing those operands to TF32 cannot itself lose their significant bits. A memory contraction problem remains consistent with the error signature, but its precise native cause is not established. The probes do not capture memory/delta separately; a rank-one fit is not a direct measurement of matmul error.

## Method and results

Pure PyTorch CPU replay uses each captured TT incoming state, Q/K/V, g and beta; it repeats each of the 16 Q/K heads twice and applies the exact HF scalar `1 / sqrt(128)`. Canonical recurrence is `dec = S*exp(g); memory = sum(dec*k); delta=(v-memory)*beta; new=dec+k*delta; core=sum(new*q)`. Replaying the separately captured HF operands reproduces **all 13 HF BF16 core and cache tensors bit-for-bit**, validating orientation, head mapping, scaling and cache convention.

| Position | Incoming TT/HF state relL2 | Same-input CPU/device new-state relL2 | Next BF16 cache elements differing | Effective memory perturbation relL2 |
|---|---:|---:|---:|---:|
| 128 | .010509 | 1.7521e-5 | .7084% | 3.9218e-4 |
| 1024 | .013212 | 2.1431e-5 | .6037% | 3.9612e-4 |
| 3979 | .013374 | 1.2670e-5 | .7095% | 2.4296e-4 |
| 5011 | .015760 | 1.9562e-5 | .5804% | 4.2869e-4 |
| 6595 | .013273 | 8.3989e-6 | .5898% | 2.7398e-4 |
| 7551 | .014220 | 1.3069e-5 | .6571% | 3.1063e-4 |
| 7668 | .014326 | 1.7405e-5 | .6786% | 4.3949e-4 |
| 7703 | .013895 | 2.8517e-5 | .6945% | 4.4214e-4 |
| 8157 | .013688 | 2.0039e-5 | .8009% | 5.6577e-4 |

The last column is an **inferred equivalent perturbation**, not a captured memory value. For each head fit `E = TTnew - canonical_new = S*de + k*da`, projecting perpendicular to k to identify de first. The two-component fit leaves only .126%–.407% of E's norm unexplained. The decay component is .0198%–.1235% of E's norm; the rank-one component dominates. Interpreting `da = -beta*dm` yields the displayed memory perturbation. Errors in another delta-producing operation can have the same rank-one signature.

Specific competing controls, replayed on identical operands:

- **FMA versus separate multiply/add:** full-precision-product-plus-add rounded once to FP32, independently cross-checked with CPU `torch.addcmul`, changes new-state relL2 by only 6.79e-9–1.31e-8. It changes at most 7 of 524,288 next-cache BF16 elements, versus 3,043–4,199 differing between canonical CPU and saved device. Both variants remain essentially equally far from the device result. This bounds the ordinary FP32 fusion semantic difference; it is not an emulation of every hardware instruction.
- **Reassociate decay after contraction:** `sum(S*k)*exp(g)` instead of `sum((S*exp(g))*k)` leaves the observed error essentially unchanged. CPU batched matrix multiplication in place of sum also leaves it unchanged. Algebraic reassociation / CPU reduction order alone does not explain the device error.
- **EXP:** computing exp in FP64 then rounding to FP32 also leaves the result essentially unchanged. The independent fitted decay error is at most 8.16e-8 relative for heads with exp(g)>1e-15, and its contribution is tiny. The evidence does not support EXP as the main one-step discrepancy.
- **Cache boundary:** all captured incoming TT states equal their own BF16->FP32 conversion exactly. The current input-boundary cast implements the HF public-call cache rounding convention. Existing incoming state disagreement is therefore accumulated value disagreement, not absence of the BF16 boundary. Changing the output buffer dtype is unnecessary for this convention, and rounding internal prefill steps would change a different semantic boundary.

Incoming-state disagreement is 1.05%–1.58% relative, much larger than the same-input one-step discrepancy. These data cannot establish that the remaining late output failures are caused solely by recurrence arithmetic: upstream differences can also accumulate. The small local mismatch does alter thousands of next-cache BF16 elements, so a stream-level control is justified after the local cause is identified.

## Minimal parent-run control

At snapshot lines 291–293 replace only `matmul(k,state)` followed by multiplication by exp(g) with:

```python
memory = ttnn.sum(
    ttnn.multiply(decayed, k),
    dim=2,
    keepdim=True,
    compute_kernel_config=self.compute,
    fast_and_approximate_mode=False,
)
```

Here `decayed` is `[1,32,128,128]`, k is `[1,32,128,1]`, both FP32; result is `[1,32,1,128]`. Keep `decayed`, delta subtraction/multiply, addcmul, output reduction, frontend, cache boundary and prefill unchanged. This matches HF's decayed-state contraction ordering and forces the documented accurate SFPU reduction path (self.compute has FP32 destination enabled). It is a model-only control, with expected extra work and no performance claim.

For an immediate one-step diagnosis, capture `decayed`, `memory`, `delta`, and `new` on the **same incoming tensors** for current and control paths at position 7703 (and an early position). Compare decayed to CPU `S*exp(g)`, memory to CPU `sum(decayed*k)`, delta to CPU arithmetic using the captured device memory, and new to separate/FMA CPU updates using captured device delta. This distinguishes contraction from downstream update error without attributing the rank-one fit exclusively to matmul. Then use an independent full-stream replay to measure accumulated benefit. If memory agrees but delta/new does not, switch only addcmul to `add(decayed, multiply(k,delta))`; the CPU evidence predicts a much smaller ordinary fusion effect.

## Source anchors

All paths below are relative to repository root.

- `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_gates_norm_l2_conv.py.snapshot:281–297`: recurrence; lines 363–365: public decode BF16->FP32 input cast.
- `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:325–366`: HF FP32 recurrence and final core cast; particularly 357–360 for decay/contraction/separate update.
- `python_env/lib/python3.12/site-packages/transformers/cache_utils.py:926,937,989`: cache dtype inherited from convolution and state copied into that dtype.
- `models/autoports/qwen_qwen3_6_35b_a3b/tests/routing_probe.py:40–47,63–67`: capture semantics and cloned HF incoming state.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h:46–59`: MUL/MAD sequence; FP32 destination stores without BF16 conversion.
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–66`: FP32 add/subtract/multiply select SFPU, including default fast mode.
- `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h:36–44,80–90`: EXP default approx=false. The FP32 accurate branch is in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h:708–736`.
- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:48,105–118`: FLOAT32 accurate SFPU reduction contract, FP32 destination requirement, and explicit supported kwargs.
