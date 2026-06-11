# tt-llk#951 — Data format handling for Math (FPU vs SFPU)

Investigation of the HW-team premise:

> `unpack0_dst_format` is used as the FPU data format, while `unpack1_dst_format`
> is used as the SFPU format.

**TL;DR:** The premise is correct and confirmed on an n150. The SFPU interprets
values it loads from DEST using the **SrcB** (`ALU_FORMAT_SPEC_REG1_SrcB`) format,
not SrcA. tt-metal *does* pass different srcA/srcB formats into the math HW config.
However, the behavior is **benign in tt-metal today** because a JIT-time guardrail
forces every float operand into the same exponent family, and the SFPU only derives
the exponent family (BF16 vs FP16) from SrcB. The contract gap is real at the LLK/HW
level for any caller that bypasses tt-metal's format inference.

---

## 1. How formats reach the math thread

`_llk_math_hw_configure_(srca, srcb)` programs the ALU format registers:

- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_common.h:88` —
  `ALU_FORMAT_SPEC_REG0_SrcA = srca`, `ALU_FORMAT_SPEC_REG1_SrcB = srcb`.

The metal wrapper picks the per-CB formats:

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_common_api.h:23` —
  `_llk_math_hw_configure_(unpack_dst_format[srca_operand], unpack_dst_format[srcb_operand])`.

Compute-API call sites (`tt_metal/hw/inc/api/compute/`):

| Path | hw_configure args | srcA vs srcB |
|------|-------------------|--------------|
| `eltwise_unary/eltwise_unary.h` | `(icb, icb)` | always equal |
| `bcast.h`, `transpose_wh.h` | `(icb, icb)` | always equal |
| `eltwise_binary.h`, `matmul.h`, `tilize.h`, `compute_kernel_hw_startup.h` | `(icb0, icb1)` | **can differ** |

So a pure unary SFPU op can never see a srcA/srcB mismatch. Binary/matmul paths can.

## 2. tt-metal really does pass different srcA/srcB formats

`unpack_dst_format[]` is per-CB and derived from each CB's data format
(`tt_metal/jit_build/data_format.cpp:171` `get_unpack_dst_formats` →
`get_single_unpack_dst_format`): a non-fp32 input keeps its own format. Two
different-format input CBs therefore produce two different entries.

Verified by inspecting a real JIT artifact for a bf16 × bfp8_b matmul (the common
LLM activation × weights case):

```
constexpr uint8_t unpack_dst_format[32] = {
    5, 6, 255, 255, ...      // [0]=Float16_b (srcA), [1]=Bfp8_b (srcB)
};
```

(`Float16_b = 5`, `Bfp8_b = 6` — see `tt_metal/api/tt-metalium/tt_backend_api_types.hpp`.)

→ `_llk_math_hw_configure_(5, 6)` — srcA and srcB formats differ.

## 3. The inference does NOT make them common — but constrains the exponent family

`compute_data_formats` (`tt_metal/jit_build/genfiles.cpp:505`) →
`get_data_exp_precision` → **`check_consistent_format_across_buffers`**
(`tt_metal/jit_build/data_format.cpp:65`):

```cpp
TT_FATAL(
    is_exp_b_format(format) == is_exp_b_format(last_valid_format),
    "All input data-formats must have the same exponent format.");
```

Float32 / integer / Fp8_e4m3 are exempt; every other float CB must share one
exponent family (all A=E5 or all B=E8). Formats may differ (Float16_b vs Bfp8_b)
but never across exponent families. This fires at JIT build time — e.g. trying to
compile a kernel with a Float16_b CB and a Float16 CB aborts here.

## 4. HW confirmation (ISA + n150 experiment)

**ISA docs.** `SFPLOAD` with `Mod0 = MOD0_FMT_SRCB` (the default for the SFPU
calculate path) resolves the DEST data format from `ALU_ACC_CTRL_SFPU_Fp32_enabled`
(→ FP32 if set) else from `ALU_FORMAT_SPEC_REG1_SrcB`, and from that picks
`MOD0_FMT_BF16` vs `MOD0_FMT_FP16`. The FPU uses REG0 for SrcA, REG1 for SrcB.

**Experiment.** `TensixSfpuSrcbFormatProbe951` in `test_sfpu_compute.cpp` +
`test_kernels/compute/sfpu_srcb_format_probe.cpp`.

Design — a controlled A/B test where the *only* difference between two runs is
`ALU_FORMAT_SPEC_REG1_SrcB`:

1. All CBs are `Float16_b` (so the §3 guardrail is satisfied and the kernel compiles).
2. `unary_op_init_common` sets REG0 = REG1 = Float16_b.
3. The kernel then *directly* overrides REG1_SrcB to a compile-arg format value via
   `_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE,false>(override)`, leaving
   SrcA = Float16_b.
4. `copy_tile` (FPU datacopy) writes DEST from SrcA (Float16_b, E8M7).
5. `square_tile` (SFPU) loads/stores DEST — using REG1_SrcB.
6. Pack out as Float16_b, read back, compare.

Result (input = 1.5, golden square = 2.25), `fp32_dest_acc_en = false`:

| REG1_SrcB override | output |
|--------------------|--------|
| `Float16_b` (E8M7 — matches DEST) | **2.25** ✓ |
| `Float16` (E5M10 — different exp family) | **inf** ✗ |

The output flips from correct to garbage by changing only REG1_SrcB → the SFPU's
interpretation of DEST is governed by the **srcB** format. Confirmed.

## 5. Conclusion

- Premise confirmed: SFPU keys off `unpack1_dst_format` (REG1_SrcB) for DEST; the
  FPU uses both REG0/REG1.
- Benign in tt-metal: the SFPU only reads the exponent family (BF16 vs FP16) from
  REG1, and `check_consistent_format_across_buffers` guarantees all float operands
  share that family, so REG1_SrcB always agrees with SrcA on what the SFPU cares about.
- Real gap at the LLK/HW layer: any caller that programs srcA/srcB directly (bypassing
  tt-metal's inference) can desync the SFPU's DEST format. Worth a guardrail:
  - a `@note` on `_llk_math_hw_configure_` documenting that REG1_SrcB doubles as the
    SFPU DEST format, and/or
  - an assert that srcA/srcB share an exponent family (mirroring the metal-side check).

---

## Repro

```bash
cd <tt-metal>
export TT_METAL_HOME=$PWD
export ARCH_NAME=wormhole_b0

# Build the LLK unit tests
cmake --build build --target unit_tests_llk -j

# (4) HW experiment — confirms SFPU uses REG1_SrcB
./build/test/tt_metal/unit_tests_llk \
    --gtest_filter='*SfpuSrcbFormatProbe951*'
# expect log line:
# [951] input=1.5 square_golden=2.25 control(srcB=Float16_b)=2.25 treatment(srcB=Float16)=inf
```

To regenerate the §2 artifact, run any compute op with two different-format input
CBs (e.g. a bf16 × bfp8_b matmul) and inspect the JIT'd
`chlkc_descriptors.h` under `~/.cache/tt-metal-cache/<hash>/kernels/<compute_kernel>/`:

```bash
grep -A1 'unpack_dst_format' \
  ~/.cache/tt-metal-cache/*/kernels/<compute_kernel>/*/chlkc_descriptors.h
```
