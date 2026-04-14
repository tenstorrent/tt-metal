#!/bin/bash
# Create draft PRs for all generated Quasar SFPU kernel branches.
# Usage:
#   ./codegen/scripts/create_prs.sh              # create all 9 PRs
#   ./codegen/scripts/create_prs.sh fill sign    # create specific PRs only
#   ./codegen/scripts/create_prs.sh --dry-run    # print commands without running
#
# Prerequisites:
#   gh auth login

set -euo pipefail

DRY_RUN=false
SELECTED=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [kernel1 kernel2 ...]"
      exit 0 ;;
    *) SELECTED+=("$1"); shift ;;
  esac
done

create_pr() {
  local kernel="$1"
  local branch="$2"
  local title="$3"
  local body="$4"

  if $DRY_RUN; then
    echo "[DRY RUN] Would create PR: $title"
    echo "  Branch: $branch"
    echo ""
    return
  fi

  echo "Creating PR: $title"
  gh pr create \
    --draft \
    --head "$branch" \
    --title "$title" \
    --body "$body"
  echo "  Done."
  echo ""
}

should_run() {
  local kernel="$1"
  if [[ ${#SELECTED[@]} -eq 0 ]]; then
    return 0  # no filter, run all
  fi
  for s in "${SELECTED[@]}"; do
    [[ "$s" == "$kernel" ]] && return 0
  done
  return 1
}

# --- fill ---
should_run fill && create_pr fill \
  "vvukomanovic/quasar-fill-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU fill kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `fill` SFPU kernel. This kernel fills a tile with a constant value, supporting float, integer, and bitcast modes.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- threshold ---
should_run threshold && create_pr threshold \
  "vvukomanovic/quasar-threshold-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU threshold kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `threshold` SFPU kernel. This kernel applies element-wise thresholding: if input > threshold, output = input; else output = value.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- elu ---
should_run elu && create_pr elu \
  "vvukomanovic/quasar-elu-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU elu kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `elu` SFPU kernel. ELU (Exponential Linear Unit): f(x) = x if x > 0, else alpha * (exp(x) - 1).

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- exp2 ---
should_run exp2 && create_pr exp2 \
  "vvukomanovic/quasar-exp2-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU exp2 kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `exp2` SFPU kernel. This kernel computes 2^x element-wise.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- log ---
should_run log && create_pr log \
  "vvukomanovic/quasar-log-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU log kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `log` SFPU kernel. This kernel computes natural logarithm element-wise using LUT-based approximation.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- trigonometry ---
should_run trigonometry && create_pr trigonometry \
  "vvukomanovic/quasar-trigonometry-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU trigonometry kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `trigonometry` SFPU kernel. This kernel implements sine, cosine, acosh, asinh, and atanh functions.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- activations ---
should_run activations && create_pr activations \
  "vvukomanovic/quasar-activations-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU activations kernel (celu, hardsigmoid)" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `activations` SFPU kernel. This kernel implements CELU and Hardsigmoid activation functions.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- sign ---
should_run sign && create_pr sign \
  "vvukomanovic/quasar-sign-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU sign kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `sign` SFPU kernel. This kernel computes element-wise sign: -1 for negative, 0 for zero, +1 for positive.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

# --- where ---
should_run where && create_pr where \
  "vvukomanovic/quasar-where-kernel" \
  "[Quasar AI codegen] Add Quasar SFPU where (ternary conditional select) kernel" \
  '### Ticket
N/A — AI-generated kernel port

### Problem description
Quasar architecture is missing the `where` SFPU kernel. This is a ternary conditional select: result = (cond != 0) ? true_val : false_val, operating on 3 input tiles in Dest.

### What'\''s changed
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)'

echo "=== All done ==="
