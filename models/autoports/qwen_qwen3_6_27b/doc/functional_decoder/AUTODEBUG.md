# AutoDebug: layer 0 Gated DeltaNet contribution PCC

## Findings

**The inspected failing implementation has confirmed normalization precision mismatches with the installed HF fallback. Its chunk inverse is algebraically correct, but has a demonstrated cancellation vulnerability. The saved evidence does not establish which discrepancy causes the real length-511 failure.**

In that implementation, the earliest definite mismatch is the BF16 upload of the folded input RMSNorm scale. Q/K normalization then uses FP32 arithmetic where this HF fallback uses BF16 arithmetic. After the recurrence, gated RMSNorm differs in both rounding order and gate precision. These are concrete fidelity repairs, not yet proven sufficient fixes of the failing contract. **A later concurrent edit already changed folded-scale storage to FP32; this report did not make that edit, and the supplied failure artifacts do not test it.**

Separately, the power-product inverse at `functional_decoder.py:377–380` can produce inaccurate results even with CPU FP32 matrix multiplication: a controlled 32-row example below has maximum inverse error about 0.125 while HF's row substitution has error about 1.15e-7. This deserves a direct discriminator on the actual failing intermediates; nilpotence alone does not prove numerical stability.

Investigation date: 2026-11-12. Inspection only: no devices, decoder execution, source edits by this investigator, resets, or launcher invocation. CPU checks used small synthetic tensors, selected cached weight tensors, and isolated reference functions. This report is the only artifact written by this investigator.

## Sources and evidence boundaries

Abbreviations used below:

- **TT:** `models/autoports/qwen_qwen3_6_27b/tt/functional_decoder.py`.
- **HF:** `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`.
- **Contract:** `models/autoports/qwen_qwen3_6_27b/tests/run_contract.py`.

Prepared checkout HEAD is `a0b0bd54e198d81676662b5a2d3a489e29bc9ab9`; this autoport is untracked. The runner pins checkpoint revision `8f33b98937c6031bb49e713dee3e8e185aafd26f` (`tests/run_decoder.py:22–23`). The report concerns the prepared files, not an inferred earlier implementation. Reference environment: Torch `2.11.0+cpu`, Transformers `5.12.1`.

**Concurrent-edit boundary:** the decoder and contract runner changed externally during inspection. A first re-read confirmed added GDN configuration assertions, list-prefill validation, and a later mixed-row contract case; the normalization, inverse, and original failing length loop were unchanged. **TT line numbers below are pinned to that captured snapshot, SHA256 `9f89505e…`, not to the continuously changing working tree.** A subsequent snapshot, `5b5933e1…`, changes folded-scale storage to FP32 (`folded_norms` at its lines 56–69); its Q/K L2, gated norm, and inverse remain as investigated. Initial, first-recheck, and later hashes are recorded at the end. The added mixed-row case and FP32 scale repair have no results in the supplied failure artifact. Do not splice those newer changes into the failing run's causal chain.

**Direct observations:**

`gdn_contract.json` and its log record:

| Logical length | Whole-output PCC | Minimum row contribution PCC | Finite output |
|---:|---:|---:|:---:|
| 1 | 0.99999708 | 0.99999659 | yes |
| 31 | 0.99997461 | 0.99955325 | yes |
| 32 | 0.99998909 | 0.99993949 | yes |
| 33 | 0.99993289 | 0.99992618 | yes |
| 127 | 0.99996805 | 0.99907603 | yes |
| 128 | 0.99992311 | 0.99969332 | yes |
| 129 | 0.99996305 | 0.99733276 | yes |
| 511 | 0.99996442 | **0.99407857** | yes |

The assertion at Contract:102 stopped execution after saving length 511. Lengths 512, 513, 1025, and the final 17, and **all GDN continuation controls**, were not reached in this artifact.

Contract:82 samples a fresh random input for each length. Thus the passing length 129 is not a prefix of the failing length 511. Only `min(rows)` is saved (Contract:91–98): there is no argmin, first failing row, row distribution, or intermediate capture. “seq511” identifies a sequence length, **not token index 511 or a known first divergent timestep**. A length-511 input has indices 0–510.

**Interpretations that remain unproven:** monotonic error accumulation, a last-row failure, a padding-only failure, a state-copy failure, and cancellation in the actual failing chunk. The longer case also tests more rows and different data, so a worse minimum alone establishes none of these.

The current installed reference path was checked without importing TTNN: `causal_conv1d_fn`, `causal_conv1d_update`, `FusedRMSNormGated`, `chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule` are all `None`; `is_fast_path_available=False`. HF:406–423 therefore selects `Qwen3_5RMSNormGated` and the torch delta-rule functions. The log's generic fast-path warning alone would not prove the norm selection, which is independent. Cached layer-0 parameters are all BF16; the harness also converts the HF module to BF16 before loading weights (Contract:34–35). Epsilon is `1e-6`.

## Confirmed normalization discrepancies, in execution order

### 1. Folded RMSNorm scales were rounded before HF rounds them

Status: confirmed in the initial and first-rechecked snapshots; the later concurrent `5b5933e1…` snapshot has applied the storage repair below. Runtime effect remains unverified here.

TT:58–73 computes `1.0 + weight.float()` for input/post-attention norms, then uploads the result as BF16. HF:745–750 instead computes the normalized activation in FP32, multiplies by `1.0 + weight.float()` in FP32, and converts the product to the input dtype afterward.

Converting the activation to FP32 in TT `_norm` (117–125) cannot recover the lost scale bits. On the actual cached layer-0 weights:

| Scale | Entries changed by BF16 folding | Maximum absolute scale error |
|---|---:|---:|
| `input_layernorm.weight` | 4,870 / 5,120 | 0.00390625 |
| `post_attention_layernorm.weight` | 3,800 / 5,120 | 0.00311279296875 |

This is the earliest identified source-level precision difference: it changes the first normalized hidden activation, then QKV, beta, decay, and the output gate. Post-attention scale rounding independently changes the MLP input.

**Minimal repair:** upload only the already-folded `(1+w)` norm scales as FP32, retaining FP32 activation normalization and the final BF16 cast. Keep the ordinary GDN norm weight unchanged: it uses `w`, not `(1+w)`. An explicit FP32 scale multiplication after an unweighted norm is another way to avoid BF16 scale storage. Verify actual kernel precision as discussed below; FP32 storage does not promise full FP32 arithmetic throughout.

**Discriminator:** compare normalized hidden activations on the identical saved input, then change only folded-scale storage. Record raw activation errors and downstream contribution errors. A pass after this change would be evidence; the scale statistics alone do not predict whether the 0.995 gate will pass.

### 2. Q/K L2 normalization upcasts at the wrong boundary for this reference

HF:240–243 executes:

```python
inv_norm = torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
return x * inv_norm
```

Here `x` arrives as BF16 from convolution/SiLU. The intermediate tensors remain BF16. HF only converts the normalized Q/K to FP32 afterward (258–264 for chunks; 330–335 for recurrence). TT `_l2` (134–137) converts `x` to FP32 **before** the square, sum, epsilon addition, reciprocal square root, and multiplication. Both `_delta_step` and `_delta_chunks` use this helper.

This is a mismatch with the installed fallback even if FP32 normalization is more accurate against real arithmetic. Merely rounding the final FP32 normalized result to BF16 does not reproduce HF's intermediate rounding. CPU seed 183, input `randn(2,17,16,128,dtype=BF16)*0.1`: 16,102 / 69,632 entries differ between HF and the FP32 calculation rounded back to BF16; maximum absolute HF-versus-FP32 difference is 0.00178394.

**Minimal repair:** retain BF16 boundaries for square, reduction output, epsilon addition, rsqrt, and final multiplication, then explicitly convert normalized Q/K to FP32 before recurrence. The reduction may accumulate internally at higher precision, but its result must have HF's BF16 boundary. Preserve FP32 state, beta promotion, decay, and recurrence arithmetic. Do not change epsilon to the RMSNorm convention: Q/K use a **sum**, not a mean.

**Discriminator:** supply identical captured raw post-convolution Q/K to CPU HF L2, current TT L2, and a parity version. Then compare the recurrence with matched normalized inputs. In a small random CPU reference comparison, this normalization choice alone changed core output relative L2 by about 0.00351, while changing chunk size or inverse evaluation on ordinary random data produced errors near 1e-7. These are synthetic core-output measurements, not the contract's 5120-wide layer contribution metric.

### 3. Gated RMSNorm differs before gamma and in SiLU precision

Both paths first cast the recurrent core output to BF16 (HF:323 / TT:175). The subsequent operations differ:

| Operation | Installed HF:193–202 | TT:176–183 |
|---|---|---|
| Variance/normalization | Promote BF16 core to FP32 | Weighted RMSNorm receives BF16 core, with FP32 destination config |
| Normalized activation | Cast to BF16 **before gamma** | Gamma applied before BF16 output packing |
| Gamma product | BF16 `weight * normalized_bf16` | Float32 normalized intermediate, gamma multiplication, then BF16 output |
| Gate | `silu(z.float())`, FP32 | `silu(z_bf16)`, BF16 output |
| Final gate product | Promote weighted BF16 norm, multiply FP32 gate, then BF16 | BF16 norm times BF16 gate, BF16 output |

The lowered kernel confirms the fused gamma-before-output-pack ordering; it is not inferred only from the Python call.

**Minimal repair, preserving HF boundaries:**

```python
# Pseudocode: device operations, no host conversion inside forward.
n32 = rms_norm(core_bf16.float(), weight=None, epsilon=eps, compute_kernel_config=ckc)
n16 = n32.bfloat16()
weighted16 = n16 * gamma_bf16
gated16 = (weighted16.float() * silu(z_bf16.float())).bfloat16()
# Existing reshape and out_proj follow.
```

CPU checks using the actual layer-0 gamma, seed 1023, shape `(128,128)` confirm that omitting the pre-gamma BF16 round changes 3,746 / 16,384 final BF16 values; rounding SiLU to BF16 changes 4,524 / 16,384. They isolate rounding boundaries under ideal CPU arithmetic and do not simulate TT hardware.

**Discriminator:** hold core BF16 output and projected `z` fixed. Separately toggle pre-gamma rounding and FP32 SiLU. Compare normalized values, weighted values, gated values, and out-projection. Casting the gate to FP32 alone leaves the pre-gamma mismatch unresolved.

## Chunk rule: correct algebra, conditional numerical vulnerability

TT:344–392 implements 32-token chunks; installed HF:246–324 defaults to 64. For each chunk, write `G_i=sum(g_0…g_i)`, `D_ij=exp(G_i-G_j)` on the lower triangle, and

`N = -strict_lower((beta*k) @ k.T * D)`.

HF's row substitution computes `L=(I-N)^-1`. TT computes

`L=(I+N)(I+N²)(I+N⁴)(I+N⁸)(I+N¹⁶)`.

Because N is strictly lower triangular of size 32, `N³²=0`; these are mathematically equivalent. The loop has the correct number of factors. The remaining formulas agree:

```text
delta = L @ (beta*v) - (L @ (beta*k*exp(G))) @ H
out   = (q/sqrt(128)*exp(G)) @ H + ((q/sqrt(128)) @ k.T * D) @ delta
Hnext = exp(G_last)*H + (k*exp(G_last-G)).T @ delta
```

The `g.T @ upper` calculation is an inclusive prefix sum, with the correct orientation. Repeating each Q/K head three times matches HF's 16-to-48 head expansion. The recurrent state orientation is `[batch,48,128,128]` in both implementations. No missing beta, incorrect sign, reversed decay, wrong scale, or transposed state was found.

Padding with zero Q/K/V/beta/g is neutral: padded rows emit zero, update no state, and add zero decay. For length 511, TT executes 16 chunks of 32 with one padded token; HF executes eight chunks of 64 with one padded token. For 129, the respective counts are five and three, with 31 and 63 padded tokens. Earlier lengths also cover partial tails. TT's outer 512-token splitting (TT:219–240) does **not** activate a second outer chunk for length 511.

CPU sanity comparisons with matched normalized inputs and small random tensors gave, at length 511, relative output L2 about 1.83e-7 for HF64 versus HF32 and 1.38e-7 for HF32 versus the power-product implementation. Thus chunk size 32 or the inverse identity is not intrinsically a logical error.

### Why the inverse implementation still needs a numerical discriminator

N can have large powers even when every entry has magnitude below one. In the power product, large intermediate matrices subsequently cancel. FP32 state/output buffers and HiFi4 do not make this cancellation harmless.

An independently checked CPU example uses 32 identical raw keys `[1,0,…,0]` of width 128, normalizes them with FP32 epsilon `1e-6`, sets `beta=0.75`, and uses constant `g=-0.01`. Compare both inverse algorithms on **the same N** against an FP64 triangular solve:

| Measurement | Power product | HF row substitution |
|---|---:|---:|
| Max absolute inverse error | **0.12499999** | 1.1485e-7 |
| Max absolute FP32 `(I-N) @ L - I` | **0.18273553** | 9.4072e-8 |

Maximum power magnitude reaches 1,140,258.25; intermediate product magnitude reaches 654,321.375, although the true inverse's largest magnitude is one. This is a verified numerical weakness of this algorithm for correlated keys and weak decay, not a sign/index mistake. It does **not** prove those conditions occurred in the real input. The saved JSON has no keys, beta, decay, inverse residuals, or state comparisons.

**Minimal conditional repair:** replace only TT:377–380 with HF's FP32 row substitution on the existing N, keeping chunk size, masks, state convention, and the rest of the equations unchanged. For row i, clone its original prefix, combine it with already-solved earlier rows exactly as HF:291–295 does, then add I. Avoid diagonal damping or clipping: they change the modeled recurrence. A direct forward solve against the required right-hand sides is another stable approach but is a broader implementation change.

**Decisive discriminator:** capture the failing chunks' N and successive powers; measure `(I-N)@L-I`, max intermediate magnitude, and output/state error against the HF row method on those same inputs. Alternatively, compare `_delta_chunks` with repeated `_delta_step` on slices of the **same precomputed raw Q/K/V/beta/g**, restoring the same initial state. Both currently use the same `_l2`; running the entire `_gdn` one token at a time would also change projection/convolution arithmetic and would confound the inverse test. Do not normalize the inputs twice.

## Lowered precision contract and other potential issues

`ttnn.rms_norm` lowers through `operations/normalization/rmsnorm/rmsnorm.cpp:70–83` to `prim::layer_norm(RMSNORM)`. Interleaved tiled tensors select `LayerNormMultiCoreProgramFactory` (`layernorm_device_operation.cpp:20–25`). Its intermediate buffers are Float32 with the supplied FP32-destination config (`layernorm_op_multi_core.cpp:488–503`). This does not mean every consumer receives all 23 FP32 mantissa bits:

- Ordinary consumed Float32 buffers use `UnpackToSrc` (`layernorm_op_multi_core.cpp:1233–1241`), with explicit exceptions such as the large-kernel running accumulator (1203–1204). JIT selects TF32 intake (`tt_metal/jit_build/genfiles.cpp:1073–1075`); Blackhole's `tt_llk_blackhole/common/inc/cunpack_common.h:311–322` documents FP32-to-TF32 Src intake versus FP32-to-FP32 Dest intake.
- The FP32 width-5120 input norm must stream: its full-row input, square, and BF16 gamma alone require 1,638,400 bytes, exceeding Blackhole's 1,572,864-byte L1. The planner is in `layernorm_op_multi_core.cpp:266–350`. The large kernel uses SFPU square, then separate normalization/gamma phases (`kernels/compute/layernorm_large_tensor.cpp:325–479`).
- Width-128 gated norm ordinarily uses `kernels/compute/layernorm.cpp`: FPU square at 271–282, normalization and gamma at 314–378, BF16 output. Both widths are exact multiples of 32; sequence padding is not part of a valid row's width reduction.
- Epsilon is stored in a BF16 buffer (`layernorm_op_multi_core.cpp:694–695`), generated by upper-half truncation (`ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp:14–21`). The mean's `1/W` scaling is a full-float destination operation; it should not be blamed as a BF16 scaler approximation.

These are supported precision limits, not an established callee bug causing this PCC failure. The default program has `use_welford=False`; RMSNorm does not support enabling that alternative (device validation at 333). The model's `WormholeComputeKernelConfig` name is harmless here: it aliases the universal config (`ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp:36–39`). The HiFi4 warning in the implementation is Wormhole-specific, whereas this log reports Blackhole.

The chunk matmuls really do request FP32 output: omitted matmul dtype defaults to input A (`matmul_device_operation.cpp:3454–3465`), and chunk operands are FP32. BF16 projections remain BF16 output despite FP32 destination accumulation. Tensor dtype, accumulator dtype, and operand intake precision are separate contracts; do not interpret CPU FP32 inverse checks as a complete TT numerical simulation.

**Other upstream precision difference:** TT:160–164 rounds each convolution tap product and each successive sum to BF16. HF:497 uses a BF16 convolution output followed by BF16 SiLU, without those extra per-tap tensor boundaries. This can alter raw Q/K before L2 normalization. If stage captures identify it as material, minimally accumulate the four BF16-valued tap products in FP32, cast the completed convolution to BF16, then apply SiLU. That aligns the rounding boundary more closely but still needs numerical verification against the reference convolution. It is not evidence of reversed taps or incorrect left history; the last-three-token state convention is sufficient for a width-four causal convolution.

**Metric/test gaps, not explanations of the recorded finite failure:**

- `row_pcc` uses FP64 centering and reduction after exact FP32 subtraction of BF16 output and input (Contract:18–21,91). No arithmetic defect was found that explains the finite 0.99407857. Whole-output PCC is flattened and includes the shared residual, which can mask contribution errors. The saved evidence lacks the norms needed to quantify that masking.
- `output-x` includes both BF16 residual-add roundings (TT:128,132 / HF:801,807); it is not the raw attention output or raw MLP output. Capture those separately to identify the responsible stage.
- Zero-variance contribution rows yield NaN PCC. Python `min()` can overlook a later NaN; `all_finite` checks output tensors, not row metrics. Explicitly require finite row PCCs and define a zero-variance policy. This edge case does not explain the saved finite minimum.
- The continuation section's “one full reference call” comment (Contract:103) actually uses a **TT full-prefill baseline** (108,117–121). It tests TT split consistency, not HF continuation fidelity; it was not reached in this GDN run.
- The assertion consumes `to_torch` output (Contract:55–56,90), rather than merely checking that work was enqueued. No evidence points to incomplete asynchronous work at the assertion; there is no hang in this artifact.

## Follow-up order and acceptance evidence

The candidates have different evidentiary status; none has a matched real-input passing/failing intervention yet:

| Candidate | Direct evidence and reached preconditions | What remains unexplained | Smallest intervention / falsifier |
|---|---|---|---|
| Folded-scale rounding | Actual weight bits differ; every original case uses these norms | Its share of the 511-row error; later storage repair has no supplied retest | Compare same input with FP32 scale storage, already present in the later snapshot |
| Q/K L2 boundary | Source and CPU dtype checks prove BF16-versus-FP32 normalization; both TT recurrence paths reach it | Whether this mismatch materially lowers the failing layer contribution | Change only L2 rounding; hold raw Q/K and subsequent recurrence inputs fixed |
| Gated norm boundaries | Source/kernel ordering and CPU controls prove two distinct rounding changes; all cases reach it | Contribution of each change after projection and MLP | Hold core/z fixed; isolate pre-gamma round and FP32 SiLU |
| Inverse cancellation | Same-matrix CPU comparison proves vulnerability; chunks run for all reported multi-token lengths | Correlated keys/weak decay and large inverse residual are unrecorded in the real failing case | Capture actual N/residuals; replace only inverse evaluation or compare matched step inputs |

1. Preserve the exact failing input and full per-row metrics before changing math. Record argmin, first row below threshold, contribution norms, error norms, and reference callable names. Use shared-prefix inputs to test sequence-length effects.
2. Capture input norm, projected QKV, convolution/SiLU, normalized Q/K, beta/g, each chunk's core output/final state, gated norm, mixer projection, and both residual boundaries. Compare one stage at a time with identical inputs; avoid feeding divergent upstream outputs into a supposed local test.
3. Test folded-scale FP32 storage, HF-boundary L2 normalization, and the two gated-norm boundary changes independently. Keep the 0.995 requirement. These are precise parity repairs with known source evidence, but none is certified sufficient here.
4. In parallel with those numeric controls, inspect actual inverse residuals and compare power versus row substitution. If inverse error is small throughout the failing case, demote cancellation as that run's cause; retain its correlated-key regression. If it spikes, replace the inverse evaluation alone and verify both output and recurrent state.
5. After a justified repair, run the complete originally intended length list, the unreached continuation/negative-history controls, and decoder state/trace checks. Report whole-output and per-row contribution results. Existing short whole-output smokes do not establish the stronger contract.

The draft received a second source pass and an independent claim review. The published CPU cancellation check was reproduced independently. No hardware claim or passing repair is asserted by this inspection.

## Reproducible CPU cancellation check

This is a small synthetic algorithm check; it imports neither TTNN nor the model:

```python
import torch
torch.set_num_threads(1)
E = torch.eye(32)
i = torch.arange(32, dtype=torch.float32)
k = torch.zeros(32, 128)
k[:, 0] = 1
k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
D = (-0.01 * (i[:, None] - i[None, :])).tril().exp().tril()
N = -((0.75 * k) @ k.T * D).tril(-1)
A, P = N.clone(), E + N
for _ in range(4):
    A = A @ A
    P = P @ (E + A)
R = N.clone()
for j in range(1, 32):
    row, sub = R[j, :j].clone(), R[:j, :j].clone()
    R[j, :j] = row + (row[:, None] * sub).sum(-2)
R = R + E
truth = torch.linalg.solve_triangular((E - N).double(), E.double(), upper=False)
for name, L in (("power", P), ("hf_row", R)):
    print(name, (L.double() - truth).abs().max().item(),
          ((E - N) @ L - E).abs().max().item())
```

File SHA256, initial inspection:

```text
functional_decoder.py  cbac7d00006d8b0429e59409ca0bb0eaed2efdc351ffdc68d7ac3806d72ec2fb
run_contract.py        b4bafe35832366d836af4db43aa9159d48692f365f9a817ef3ab6d43f9ae80b7
modeling_qwen3_5.py    395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6
gdn_contract.json     3d54a5e74c09021b7c539a4f1b6f5120192481b01fe8143ff3863585974ac668
```

Rechecked concurrent-edit snapshot (HF and failure JSON hashes unchanged):

```text
functional_decoder.py  9f89505e8521b3260c01603dff8d55df3e8209a6a8d463cc604e27ef5af8a41c
run_contract.py        13c9bad94bf494a4e0df0469363437f0f613a53425b26ac636b5f9f5256140f4
```

Later concurrent decoder snapshot, with folded-scale FP32 storage already applied (contract hash unchanged):

```text
functional_decoder.py  5b5933e1aaa6612605b8e7ec5e7867da659538d173b6cc095d6d1f2ed86068e5
```
