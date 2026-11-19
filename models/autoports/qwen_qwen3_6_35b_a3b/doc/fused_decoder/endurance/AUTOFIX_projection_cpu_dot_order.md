# CPU BF16 projection accumulation order

The remaining large BF16 code differences can be caused by canonical CPU projection reduction order. In particular, the HF result is **not** consistently the correctly rounded mathematical dot product: position 1068 QKV channel 2907 has FP64 dot `-4.015625728934538`, rounding to BF16 `-4.03125`, whereas canonical HF returns `-4.0`. The current TT result in the parent capture is the mathematically closer BF16 value. Improving ordinary dot-product accuracy alone cannot reproduce this reference.

CPU-only reproduction uses the real checkpoint, seed 913, canonical inputnorm and the original complete weight shapes. It imports no TTNN and performs no device work. Artifacts are `projection_cpu_dot_order.py`, `.json`, and `.log` in this directory. Run:

```sh
ONEDNN_VERBOSE=1 python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/projection_cpu_dot_order.py > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/projection_cpu_dot_order.log 2>&1
```

## Actual backend and schedule

The host is an AMD Ryzen 7 9700X with AVX512 BF16 support, no AMX. Installed torch is `2.11.0+cpu`, commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`; oneDNN is 3.10.2 commit `f1d471933dc852f956fd05389f9313c7148783d5`. Verbose execution explicitly reports `brg_matmul:avx512_core_bf16`, `src:bf16`, `wei:bf16`, `dst:bf16`, `nthr:8`, for actual `1x2048:2048x4096` and `1x2048:2048x32` shapes. This invocation does use oneDNN, rather than the alternative ATen reduced-precision GEMV implementation.

With `ONEDNN_JIT_DUMP=1`, disassembly of the generated `jit_brgemm_kernel_t` showed 64 loop iterations, each consuming four adjacent BF16 pairs, or **512 K elements per partial dot**. Each output register receives consecutive `vdpbf16ps`; partial blocks combine through FP32 `vaddps`, and final conversion uses `vcvtneps2bf16`. Both projection widths use that 512-element inner loop. Generated files remain under `/tmp/projection_jit` for inspection; they are not runtime modifications.

The oneDNN [BF16 dot-product emitter](https://raw.githubusercontent.com/uxlfoundation/oneDNN/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp) emits `vdpbf16ps` (lines 2055–2065). Its [matmul block dispatch](https://raw.githubusercontent.com/uxlfoundation/oneDNN/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul.cpp) uses configured K blocks and an FP32 accumulation kernel. The instruction accumulates the odd element of each pair before the even element. The executable CPU emulation therefore computes:

```python
# p is the exact FP32 product of two normal BF16 operands.
p = (weight.float() * input.float()).reshape(N, 4, 512)
acc = zeros(N, 4, dtype=float32)
for j in range(0, 512, 2):
    acc = (acc + p[..., j + 1]) + p[..., j]
y = (((acc[:, 0] + acc[:, 1]) + acc[:, 2]) + acc[:, 3]).bfloat16()
```

This schedule reproduces **all seven identified significant counterexamples**. Across all QKV/Z/A/B channels at four positions (129, 202, 315, 1068), there are zero differences for reference magnitudes above `1e-30`. The only remaining differences are QKV extremely tiny values, maximum `9.4039548065783e-38`; the emulator does not yet reproduce instruction denormal handling. Across 142 additional full A/B projection cases covering 71 positions through 8319, there are **zero BF16 code differences**. This is strong sampled evidence, not an all-stream equivalence claim.

| Position / projection / channel | Canonical HF BF16 | FP64 dot rounded BF16 | Ordered CPU emulation BF16 |
|---|---:|---:|---:|
| 129 / Z / 1846 | 0.130859375 | 0.130859375 | 0.130859375 |
| 315 / A / 0 | 0.01239013671875 | 0.012451171875 | 0.01239013671875 |
| 202 / B / 26 | -1.0967254638671875e-5 | -1.0907649993896484e-5 | -1.0967254638671875e-5 |
| 1068 / QKV / 2907 | -4 | -4.03125 | -4 |
| 1068 / QKV / 1267 | 0.0033416748046875 | 0.00335693359375 | 0.0033416748046875 |
| 1068 / QKV / 3125 | 2.193450927734375e-5 | 2.193450927734375e-5 | 2.193450927734375e-5 |
| 1068 / QKV / 7016 | -4.589557647705078e-6 | -4.649162292480469e-6 | -4.589557647705078e-6 |

## Minimal model-only device discriminator

There is an existing native **SFPU sequential cumsum** path that can implement this order without 512 separately dispatched adds or native edits:

1. Keep the current FP32 SFPU projection products `[2048,12352]`.
2. Reorder adjacent K rows as `[1,0,3,2,...]`. One model-only construction is reshape to `[1024,2,12352]`, slice the two rows in the middle dimension, concatenate odd then even on that dimension, and reshape to `[4,512,12352]`. Static setup weight/input indexing is another option.
3. Call `ttnn.cumsum(products, dim=-2, dtype=ttnn.float32, disable_compensated_sum=True)` on the four independent blocks. Select row 511 from each block.
4. Combine four partial rows with **three explicit left-associated FP32 SFPU adds**, then cast to BF16 at the existing projection boundaries.

Source support: `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp:440–503` selects a tile-axis scan, sets 32-bit Dest and `UnpackToDest` for FP32 input, and only enables Kahan compensation if the opt-out flag is false. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h:48–98` scans each column top to bottom using sequential SFPADD and carries FP32 LREG state between tiles. Thus this path avoids the already diagnosed FPU/TF32 state loss. The exact Python slicing/layout and the scan's end-to-end numerical behavior still require the parent's device microprobe.

First test the four positions and all projection groups using the existing frontend harness, checking the seven reported cells and significant BF16 code differences. If those match, rerun the full frontend stream before endurance. Do not use default compensated cumsum: a more accurate compensated result would defeat the intended canonical sequential rounding. The ordinary reduction and the 8-residue native matmul strategy do not imply this order.

This recommendation concerns decode `M=1` only. Prefill `M=128/256` may choose another oneDNN blocking schedule and must be audited separately before applying the same canonical emulation there. No runtime, F, tests, or acceptance thresholds were changed in this investigation.
