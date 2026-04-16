# Metal Integration — LLK Change Propagation Guide

When you change code in `tt_metal/tt-llk/`, those changes may need to propagate to other layers in tt-metal. This reference documents the stack, the scenarios, and exactly which files to check.

---

## The Stack

```
Layer 4:  TTNN Operations          ttnn/cpp/ttnn/operations/*/device/kernels/compute/
Layer 3:  Compute API              tt_metal/hw/inc/api/compute/
Layer 2:  CKernels LLK API         tt_metal/hw/ckernels/{arch}/metal/llk_api/
Layer 1:  LLK Library (tt-llk)     tt_metal/tt-llk/tt_llk_{arch}/llk_lib/
```

Changes may propagate upward; Layer 2 is almost always the first place to check. Layer 3 is commonly edited; Layer 4 rarely needs changes.

### Layer 2: CKernels LLK API Wrappers

**Path:** `tt_metal/hw/ckernels/{arch}/metal/llk_api/`

Per-architecture wrappers (top-level files + SFPU wrappers in `llk_sfpu/` per arch). These `#include` LLK headers directly and expose the non-underscored API functions.

Example: `llk_math_binary_api.h` includes `llk_math_eltwise_binary.h` from tt-llk, wraps `_llk_math_eltwise_binary_init_<>()` into `llk_math_eltwise_binary_init<>()`.

### Layer 3: Compute API

**Path:** `tt_metal/hw/inc/api/compute/`

Architecture-agnostic `ckernel::` namespace (headers + unary ops in `eltwise_unary/`). Uses `MATH()`, `UNPACK()`, `PACK()` macros. Has `#ifndef ARCH_*` guards for architecture differences.

### Layer 4: TTNN Direct Consumers

Most TTNN operations use the Compute API cleanly. Some bypass it and include LLK headers directly:

```
# Find all direct LLK includes in TTNN
Grep for "#include.*llk_math_eltwise|#include.*llk_unpack|#include.*llk_pack|#include.*ckernel_sfpu" in ttnn/ (*.h *.cpp files)
```

**Always run the grep above to find current bypass locations**, as these change over time.

---

## Change Scenarios

### Scenario 1: Change an LLK Function Signature

*e.g., add/remove/rename a parameter on `_llk_math_eltwise_binary_init_<>()`*

**Check these files:**

```
# Layer 2 — find the wrapper that calls this function
Grep for "{function_name}" in tt_metal/hw/ckernels/ (*.h files)

# Layer 3 — find the compute API that calls the wrapper
Grep for "{wrapper_function_name}" in tt_metal/hw/inc/api/compute/ (*.h files)

# Layer 4 — find TTNN files that bypass the API
Grep for "#include.*{header_name}" in ttnn/ (*.h *.cpp files)

# Metal integration tests
Grep for "{operation_name}" in tests/tt_metal/tt_metal/llk/ (*.cpp files)

# Metal test kernel sources
Grep for "{function_name}" in tests/tt_metal/tt_metal/test_kernels/compute/ (*.cpp files)
```

**Update order:**
1. tt-llk implementation (Layer 1)
2. CKernels wrapper (Layer 2) — match the new signature
3. Compute API (Layer 3) — only if the API-visible interface changes
4. TTNN bypass files (Layer 4) — only if they include the changed header directly
5. Metal integration tests and test kernels — if test expectations changed

### Scenario 2: Add a New SFPU Operation

*e.g., add `ckernel_sfpu_new_op.h`*

**Create these files (per architecture):**

| Layer | File to Create | Template |
|-------|---------------|----------|
| 1 | `tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_new_op.h` | Existing SFPU op in same arch |
| 2 | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_new_op.h` | Existing wrapper in `llk_sfpu/` |
| 2 | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_new_op.h` | Existing `llk_math_eltwise_unary_sfpu_*.h` |
| 3 | `tt_metal/hw/inc/api/compute/eltwise_unary/new_op.h` | Existing op in `eltwise_unary/` |

**Update these files:**

```
# Include the new wrapper in the SFPU API aggregator
Read tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_math_unary_sfpu_api.h

# Include the new compute API header in the unary aggregator
Read tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h
```

**Pattern:** Look at how an existing SFPU op (e.g., `sigmoid`, `relu`) is wired through all 4 layers and replicate the pattern.

### Scenario 3: Change Unpack/Pack Behavior

*e.g., modify MOP configuration, address calculation, tile dimension handling*

**Check these files:**

```
# Layer 2 — unpack/pack wrappers
Grep for "{function_name}" in tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_unpack_*.h
Grep for "{function_name}" in tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_pack_*.h

# Layer 3 — compute API
Grep for "{wrapper_function_name}" in tt_metal/hw/inc/api/compute/pack*.h
Grep for "{wrapper_function_name}" in tt_metal/hw/inc/api/compute/tilize.h
Grep for "{wrapper_function_name}" in tt_metal/hw/inc/api/compute/untilize.h

# Metal integration tests
Grep for "{operation}" in tests/tt_metal/tt_metal/llk/ (*.cpp files)
```

---

## Quick Reference: Key File Paths

| What | Where |
|------|-------|
| CKernels wrappers (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` |
| CKernels wrappers (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/` |
| CKernels wrappers (QSR) | `tt_metal/hw/ckernels/quasar/metal/llk_api/` |
| SFPU wrappers (per arch) | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/` |
| Compute API | `tt_metal/hw/inc/api/compute/` |
| Unary ops (Compute API) | `tt_metal/hw/inc/api/compute/eltwise_unary/` |
| Metal LLK integration tests | `tests/tt_metal/tt_metal/llk/` |
| Metal test kernel sources | `tests/tt_metal/tt_metal/test_kernels/compute/` |
| Build system (CMake) | `tt_metal/CMakeLists.txt` (glob), `tt_metal/hw/CMakeLists.txt` (includes) |
| HAL include paths | `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp`, `bh_hal.cpp`, `tt-2xx/quasar/qa_hal.cpp` |
| JIT build includes | `tt_metal/jit_build/build.cpp` |
| Compute kernel entry point | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |

---

## Build System Notes

- **New `.h` files in tt-llk are auto-discovered** — `tt_metal/CMakeLists.txt` uses `GLOB_RECURSE` on all tt-llk headers
- **Include paths are set by the HAL layer** — only touch HAL files if you add new directories to tt-llk
- **`assembly.yaml` changes stay within tt-llk** — no metal files reference instruction definitions directly

---

## Propagation Checklist

Before completing any LLK change, verify:

- [ ] CKernels wrapper updated for each affected architecture (`tt_metal/hw/ckernels/{arch}/metal/llk_api/`)
- [ ] Compute API updated if the public interface changed (`tt_metal/hw/inc/api/compute/`)
- [ ] TTNN bypass files checked for direct includes of changed headers
- [ ] Metal integration tests still compile and pass (`tests/tt_metal/tt_metal/llk/`)
- [ ] tt-llk's own tests pass (`tt_metal/tt-llk/tests/`)
