<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Quasar ↔ Wormhole LLK gaps — eltwise binary ops & fusable activations

**Audience: the LLK team.** This lists the binary-op and activation **SFPU/FPU primitives** that
Wormhole has but **Quasar does not yet** — the set that blocks `binary_ng` op parity on Quasar. It is
the LLK-facing companion to the op-side gap inventory
(`ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/QUASAR_PARITY_GAPS.md`, same gaps from the op
author's angle) and the qualification runbook (`QUASAR_BINARY_QUALIFICATION.md`).

## Scope & method

- **Scope:** the `BinaryOpType` set `binary_ng` routes (FPU `add/sub/mul` + the `is_binary_sfpu_op`
  subset) and the `UnaryOpType` activations it fuses as lhs/rhs/post.
- **How support was determined** — a primitive is usable on Quasar only if it is present at all three
  layers, none behind `#ifndef ARCH_QUASAR`:
  1. Compute API `tt_metal/hw/inc/api/compute/…` — the `<op>_tile()` / `<op>_binary_tile()` the kernel calls;
  2. Quasar LLK-API bridge `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/…` — `ckernel::sfpu::calculate_<op>`;
  3. Quasar ckernel `tt_metal/tt-llk/tt_llk_quasar/…` — `_calculate_<op>_`.
  Cross-checked against the Quasar LLK tests (`test_eltwise_binary_quasar.py` FPU,
  `test_eltwise_binary_sfpu_quasar.py` SFPU, `test_eltwise_unary_sfpu_quasar.py` unary) that run on the
  QSR simulator in craq-sim's `quasar-llk.yml`.
- **Classes:** **UNPORTED** = primitive `#ifndef ARCH_QUASAR` / absent (JIT-fails on Quasar);
  **WRAPPER-ONLY** = the Quasar ckernel already exists, only the LLK-API/compute-API binding is missing
  (cheap); **FORMAT** = the op is fine but a dtype (uint16/uint32/block-float) is `static_assert`-blocked;
  **SUPPORTED** = works.
- **Status is static compile-availability**, except where marked "sim-verified ✓" (actually run on the QSR
  sim). No op below is known **BROKEN** (compiles-but-wrong / hangs) — that needs a sim run; use the
  qualification harness.
- **Re-derive live:** `qualify_quasar_binary.py --coverage` and `--supports <op>` enumerate this straight
  from the LLK test collection (self-updating); the LLK tests themselves are the upstream source of truth.

## Baseline — what Quasar already supports

- **Binary:** FPU `add`/`sub`/`mul` (bf16) ✓ · SFPU `multiply`/`divide` (bf16 **and** fp32) ✓ ·
  `mul_int`/`add_int`/`gt_int` (int32) · `maximum`/`minimum` (float + int32) · `where` (ternary).
- **Activations (fusable):** `relu` ✓ · `silu` ✓ · `sigmoid` ✓ · `exp` · `sqrt` · `rsqrt` ·
  `reciprocal` · the six compare-to-zero (`eqz/nez/ltz/gtz/lez/gez`).

(✓ = sim-certified on craq-sim; the rest are compiled + LLK-test-covered.)

## 1. Binary-op gaps

| Op (ttnn) | Route | Class | LLK evidence (file:line) | To close |
|---|---|---|---|---|
| `add` / `sub` / `rsub` (fp32) | SFPU | **UNPORTED** | `*_binary_tile` under `#ifndef ARCH_QUASAR` — `eltwise_binary_sfpu.h:72-107`; Quasar ckernel `calculate_sfpu_binary` `static_assert(BINOP==MUL\|\|DIV)` — `ckernel_sfpu_binary.h:68` | relax the static_assert + add `_binop_add/_sub/_rsub` wrappers + un-gate |
| `eq`/`ne`/`lt`/`gt`/`le`/`ge` (fp32, bf16) | SFPU | **UNPORTED** | `*_binary_tile` under `#ifndef ARCH_QUASAR` — `eltwise_binary_sfpu.h:121-191`; no float-compare ckernel on Quasar | new ckernel + wrappers + un-gate |
| `lt`/`le`/`ge` (int32) | SFPU | **WRAPPER-ONLY** | Quasar ckernel already handles lt/gt/le/ge (`tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_binary_comp.h`); only the `gt_int` wrapper exists (`llk_math_eltwise_binary_sfpu_binary_comp.h`) | add `lt_int`/`le_int`/`ge_int` wrappers + un-gate (`binary_comp.h`) |
| `power` | SFPU | **UNPORTED** | `power_binary_tile` under `#ifndef ARCH_QUASAR` — `eltwise_binary_sfpu.h:109-119` | new ckernel + wrapper + un-gate |
| `bitwise_and`/`or`/`xor` | SFPU | **UNPORTED** | no `ARCH_QUASAR` branch, WH-only include — `binary_bitwise_sfpu.h:9` | port ckernel + wrapper + branch |
| `left_shift`/`right_shift`/`logical_right_shift` | SFPU | **UNPORTED** | `binary_shift.h:9` | port ckernel + wrapper + branch |
| `sub_int` / `rsub_int` | SFPU | **UNPORTED** | `sub_int_sfpu.h:9` (WH-only includes) | port ckernel + wrapper + branch |
| `div` (int32) / `floor_div` / `trunc_div` | SFPU | **UNPORTED** | `div_int32_sfpu.h:9`, `div_int32_floor.h:9` | port ckernel + wrapper + branch |
| `remainder` / `fmod` | SFPU | **UNPORTED** | `binary_remainder.h:9`, `binary_fmod.h:9` | port ckernel + wrapper + branch |
| `gcd` / `lcm` | SFPU | **UNPORTED** | `gcd.h:9`, `lcm.h:9` | port ckernel + wrapper + branch |
| `xlogy` | SFPU | **UNPORTED** | `xlogy.h` uses the WH `calculate_sfpu_binary` log-fusion (Quasar's is MUL/DIV-only) | extend Quasar `calculate_sfpu_binary` + branch |
| `atan2` | SFPU | **UNPORTED** | `atan2.h:9` | port ckernel + wrapper + branch |
| `isclose` | SFPU | **UNPORTED** | `isclose.h:9` | port ckernel + wrapper + branch |
| `quant` / `requant` / `dequant` | SFPU | **UNPORTED** | routed at `binary_ng_utils.cpp:496`; no quant ckernels under `quasar/.../llk_sfpu/` | port ckernels + wrappers + branch |
| `add_int`/`gt_int`/`maximum`/`minimum` @ **uint32/uint16** | SFPU | **FORMAT** | Int32-only `static_assert` — `add_int_sfpu.h:44`, quasar `binary_comp.h:48`, `ckernel_sfpu_binary_max_min.h:70` | relax the dtype static_assert (if uint is in scope for Quasar) |

**Derived ops inherit the SFPU add/sub gap.** `squared_difference`, `logical_and/or/xor`, `rsub`,
`logaddexp`, `logaddexp2`, `ldexp`, `bias_gelu`, `hypot` decompose to unary + FPU/SFPU. For **bf16** they
stay on the FPU path and work; for **fp32/int** dtypes where `is_binary_sfpu_op` routes them to SFPU
`add`/`sub`, they hit the UNPORTED gap above. Closing SFPU float add/sub unblocks all of them.

## 2. Activation gaps

Two tiers. **Tier 1 is cheap** (the Quasar kernel already exists — only the binding is missing); **Tier 2
is real kernel work.**

### Tier 1 — bridge-only (Quasar ckernel + `SfpuType` already exist)

`gelu` · `tanh` · `abs` · `leaky_relu` · `relu_max` · `relu_min` · `square`

For these, `_calculate_<op>_` exists in `tt_llk_quasar` and the op is even exercised by the Quasar LLK
unary test — **but through the test's standalone C++ source, not the fusable path.** The `binary_ng`
fusable path is gated: the compute-API `<op>_tile()` sits inside `#ifndef ARCH_QUASAR` (e.g. `gelu.h:16`,
`compute_kernel_api.h:177-1250` block) and there is no `ckernel::sfpu::calculate_<op>` bridge. **Fix =
add the bridge (mirror `ckernel_sfpu_silu.h` / `ckernel_sfpu_exp.h`) + change the compute-API gate to an
`#ifdef ARCH_QUASAR` branch. No `tt_llk_quasar` change needed.**

> "In the Quasar LLK unary test" ≠ "fusable." `gelu`/`tanh`/`abs`/`square` pass the raw-ckernel unary test
> but `binary_ng` cannot emit them as activations until the bridge + compute-API branch land.

### Tier 2 — kernel gaps (no Quasar ckernel / no `SfpuType` slot — genuine LLK work)

`log`/`log2`/`log10`/`log1p` · `exp2` · `neg` · `sign` · `erf`/`erfc`/`erfinv` · `elu`/`celu`/`selu` ·
`hardsigmoid`/`hardswish`/`hardtanh`/`hardshrink`/`softshrink`/`softsign` · `softplus` · `mish`/`hardmish` ·
trig family (`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`sinh`/`cosh`/`asinh`/`acosh`/`atanh`) ·
`power`/`heaviside`/`expm1`/`signbit`/`i0`/`i1`/`cbrt`/`logit`/`logsigmoid`/`lgamma`/`digamma` ·
unary **scalar**-compares (`eq/ne/gt/lt/ge/le` vs a scalar — only compare-to-**zero** exists on Quasar) ·
general `typecast` (only a few specific pairs exist).

Each needs a new `_calculate_<op>_` ckernel in `tt_llk_quasar/common/inc/sfpu/`, a `SfpuType` entry
(`tt_llk_quasar/llk_lib/llk_defs.h`), the LLK-API bridge, and the compute-API branch.

## 3. Formats (arch axis — confirm intent)

Independent of the op: several SFPU primitives `static_assert` **Int32-only** on Quasar
(`add_int`, `gt_int`, `maximum`/`minimum`), so **uint32/uint16** fail there while WH allows them. And
block-float **bf8_b/bf4_b** are rejected by the host format validator (Quasar uses MX/microscaling). These
may be **intentional** arch differences rather than "to port" — flag for the LLK/arch team to confirm which
of uint16/uint32 (and block-float) are in Quasar's supported set.

## 4. Priorities (by model impact)

1. **GELU** — BERT/transformer MLPs. Highest-value gap, and it is **Tier 1** (bridge-only): the ckernel
   and `SfpuType::gelu` already exist; add the bridge + un-gate `gelu.h`. Low effort.
2. **TANH, LEAKY_RELU** — common; also Tier 1 (bridge-only).
3. **SFPU float `add`/`sub`** (binary) — unblocks fp32 add/sub and every fp32/int derived op above.
   Real ckernel branch (relax the `MUL||DIV` static_assert; mirror the mul/div path).
4. **int `lt`/`le`/`ge`** (binary) — **WRAPPER-ONLY**, trivial (the ckernel already handles lt/gt/le/ge).
5. Everything else as op/model demand arises.

**Already done — no LLK work needed:** `silu` (Llama 3.2 SwiGLU) and `relu` (ResNet50), both sim-certified.

## 5. Closing a gap — the pattern

- **Bridge-only (Tier 1 / int lt-le-ge):** add `ckernel::sfpu::calculate_<op>` in
  `hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` (copy `ckernel_sfpu_silu.h` /
  `ckernel_sfpu_exp.h`), then change the compute-API `#ifndef ARCH_QUASAR` around `<op>_tile()` to an
  `#ifdef ARCH_QUASAR` branch.
- **Kernel gap (Tier 2):** add `_calculate_<op>_` in `tt_llk_quasar/common/inc/sfpu/`, a `SfpuType` entry
  (`tt_llk_quasar/llk_lib/llk_defs.h`), the bridge, and the compute-API branch.
- **No ops-side change is needed:** `binary_ng` already emits the correct `<op>_tile()` /
  `<op>_binary_tile()` calls (`binary_ng_utils.cpp`); every gap here is below the compute-API line.
- **Verify:** add the op to the Quasar LLK test and run it on the QSR sim (`run_test.sh`), then re-run
  `qualify_quasar_binary.py` to confirm the corresponding op cell flips to PASS.

## 6. Key files

- Op routing / SFPU-fn map: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`,
  `…/binary_ng_device_operation.cpp` (`is_binary_sfpu_op`).
- Activation emit: `binary_ng_utils.cpp` `add_activation_defines` → `unary/common/unary_op_utils.cpp`.
- Compute-API gates: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (binary SFPU),
  `compute_kernel_api.h` (`#ifndef ARCH_QUASAR` block ~`:177-1250`), `eltwise_unary/*.h`,
  and the per-op SFPU headers (`binary_bitwise_sfpu.h`, `binary_shift.h`, `gcd.h`, `xlogy.h`, …).
- Quasar LLK-API bridges: `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/`.
- Quasar ckernels + `SfpuType`: `tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu/`, `…/llk_lib/llk_defs.h`.
- Quasar LLK tests: `tt_metal/tt-llk/tests/python_tests/quasar/test_eltwise_{binary,binary_sfpu,unary_sfpu}_quasar.py`
  (WH baselines: `tests/python_tests/test_eltwise_binary.py`, `test_eltwise_unary_sfpu.py`).

## Caveats

- Static compile-availability, not an exhaustive sim sweep; only the ✓ items were run on the QSR sim.
- FORMAT gaps (block-float / uint16 / uint32) may be intentional Quasar arch choices (MX) — confirm before porting.
- Keep this current on tt-llk / craq-sim pin bumps: re-run `qualify_quasar_binary.py --coverage` and
  reconcile. The LLK team's `quasar-llk.yml` on the QSR sim is the upstream source of truth.
