# PR Descriptions for Generated Quasar SFPU Kernels

---

## 1. fill

**Branch:** `vvukomanovic/quasar-fill-kernel`

**Title:** `Add Quasar SFPU fill kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_fill_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `fill` SFPU kernel. This kernel fills a tile with a constant value, supporting float, integer, and bitcast modes.

### What's changed
AI-generated Quasar port of the Blackhole `fill` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_fill.h` (79 lines, 3 phases: float_fill, int_fill, bitcast_fill)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::fill` to `llk_defs.h` (required by test infrastructure)
- Added C++ test harness and Python functional test (78/78 tests passed on simulator)
- Run: `2026-03-30_fill_quasar_fc62fc04`
- Compile attempts: 4 | Debug cycles: 1 | 1 failure resolved (TTI_SFPSTORE requires compile-time constant for store_mode)

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 2. threshold

**Branch:** `vvukomanovic/quasar-threshold-kernel`

**Title:** `Add Quasar SFPU threshold kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_threshold_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `threshold` SFPU kernel. This kernel applies element-wise thresholding: if input > threshold, output = input; else output = value.

### What's changed
AI-generated Quasar port of the Blackhole `threshold` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_threshold.h` (49 lines, 1 phase)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::threshold` to `llk_defs.h` (required by test infrastructure)
- Added C++ test harness and Python functional test (78/78 tests passed on simulator)
- Run: `2026-03-30_threshold_quasar_91b927b3`
- Compile attempts: 1 | Debug cycles: 0 | Compiled and passed on first attempt

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 3. elu

**Branch:** `vvukomanovic/quasar-elu-kernel`

**Title:** `Add Quasar SFPU elu kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_elu_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `elu` SFPU kernel. ELU (Exponential Linear Unit): f(x) = x if x > 0, else alpha * (exp(x) - 1).

### What's changed
AI-generated Quasar port of the Blackhole `elu` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_elu.h` (50 lines, 1 phase)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::elu` to `llk_defs.h` (required by test infrastructure)
- Added C++ test harness and Python functional test (78/78 tests passed on simulator)
- Run: `2026-03-30_elu_quasar_7ade1cdf`
- Compile attempts: 1 | Debug cycles: 0 | Zero failures — compiled and passed on first attempt

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 4. exp2

**Branch:** `vvukomanovic/quasar-exp2-kernel`

**Title:** `Add Quasar SFPU exp2 kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_exp2_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `exp2` SFPU kernel. This kernel computes 2^x element-wise.

### What's changed
AI-generated Quasar port of the Blackhole `exp2` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_exp2.h` (50 lines, 1 phase)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::exp2` to `llk_defs.h` (required by test infrastructure)
- Added C++ test harness and Python functional test (78/78 tests passed on simulator)
- Run: `2026-03-30_exp2_quasar_e79aa3d1`
- Compile attempts: 4 | Debug cycles: 1 | 1 failure resolved (kernel computed e^x instead of 2^x — SFPMULI not working on Quasar, replaced with SFPLOADI+SFPMUL + SFPNOP for pipeline hazard)

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 5. log

**Branch:** `vvukomanovic/quasar-log-kernel`

**Title:** `Add Quasar SFPU log kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_log_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `log` SFPU kernel. This kernel computes natural logarithm element-wise using LUT-based approximation.

### What's changed
AI-generated Quasar port of the Blackhole `log` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_log.h` (94 lines, 1 phase, LUT-based)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::log` to `llk_defs.h` (required by test infrastructure)
- Added C++ test harness and Python functional test (114/114 tests passed on simulator)
- Run: `2026-03-30_log_quasar_d635d457`
- Compile attempts: 4 | Debug cycles: 1 | 2 failures resolved (CC corruption from SFPIADD mod=0 overwriting CC; MxFp8P quantization error amplified by log exceeding tolerance)

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 6. trigonometry

**Branch:** `vvukomanovic/quasar-trigonometry-kernel`

**Title:** `Add Quasar SFPU trigonometry kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_trigonometry_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `trigonometry` SFPU kernel. This kernel implements sine, cosine, acosh, asinh, and atanh functions.

### What's changed
AI-generated Quasar port of the Blackhole `trigonometry` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_trigonometry.h` (557 lines, 3 phases: sine_cosine, acosh_asinh, atanh)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::{sine, cosine, acosh, asinh, atanh}` to `llk_defs.h`
- Added C++ test harness and Python functional test (390/390 tests passed on simulator)
- Run: `2026-03-30_trigonometry_quasar_e5e44d34`
- Compile attempts: 3 | Debug cycles: 0 | Zero failures — all 3 phases compiled and passed on first attempt

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 7. activations

**Branch:** `vvukomanovic/quasar-activations-kernel`

**Title:** `Add Quasar SFPU activations kernel (celu, hardsigmoid)`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_activations_quasar.py test_sfpu_activations_hardsigmoid_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `activations` SFPU kernel. This kernel implements CELU and Hardsigmoid activation functions.

### What's changed
AI-generated Quasar port of the Blackhole `activations` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_activations.h` (112 lines, 2 phases: CELU, Hardsigmoid)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::{celu, hardsigmoid}` to `llk_defs.h`
- Added C++ test harnesses and Python functional tests for both CELU and Hardsigmoid (156/156 tests passed on simulator)
- Run: `2026-03-30_activations_quasar_6b720d80`
- Compile attempts: 4 | Debug cycles: 2 | 2 failures resolved:
  - CELU negative branch returned 0 — missing SFPNOP between SFPMUL and SFPNONLINEAR (pipeline hazard)
  - Hardsigmoid lower clamp failed — missing SFPENCC between upper and lower clamp sequences

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 8. sign

**Branch:** `vvukomanovic/quasar-sign-kernel`

**Title:** `Add Quasar SFPU sign kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_sign_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `sign` SFPU kernel. This kernel computes element-wise sign: -1 for negative, 0 for zero, +1 for positive.

### What's changed
AI-generated Quasar port of the Blackhole `sign` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_sign.h` (67 lines, 1 phase)
- Added `#include` in `ckernel_sfpu.h`
- Added `SfpuType::sign` to `llk_defs.h`
- Added `MathOperation.Sign` to `llk_params.py` and `_sign` golden generator to `golden_generators.py`
- Added C++ test harness and Python functional test (78/78 tests passed on simulator)
- Run: `2026-03-30_sign_quasar_912b17af`
- Compile attempts: 2 | Debug cycles: 0 | 1 failure resolved (compile checker requires template<bool APPROXIMATION_MODE> and _init_sign_())

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## 9. where

**Branch:** `vvukomanovic/quasar-where-kernel`

**Title:** `Add Quasar SFPU where (ternary conditional select) kernel`

**Test command:**
```bash
source tests/.venv/bin/activate && cd tests/python_tests/quasar && TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_sfpu_where_quasar.py
```

**PR body:**
```
### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `where` SFPU kernel. This is a ternary conditional select: result = (cond != 0) ? true_val : false_val, operating on 3 input tiles in Dest.

### What's changed
AI-generated Quasar port of the Blackhole `where` SFPU kernel using the LLK CodeGen system.

- Added `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_where.h` (174 lines, 2 phases: basic where + replay buffer optimization)
- Added `tt_llk_quasar/llk_lib/llk_math_eltwise_ternary_sfpu.h` — ternary SFPU math infrastructure (init, start, done)
- Added `#include` in `ckernel_sfpu.h` and `SfpuType::where` to `llk_defs.h`
- Added C++ test harness and Python functional test (24/24 tests passed on simulator)
- Run: `2026-03-31_where_quasar_79c8b251`
- Compile attempts: 8 | Debug cycles: 1 | Optimized: YES (replay buffer) | 1 failure resolved (SFPLOADMACRO path DATA_MISMATCH — fell back to DISABLE_SFPLOADMACRO)

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
