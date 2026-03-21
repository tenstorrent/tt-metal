# Make PolyNorm forward fused-only and refresh training comparison

## Kernel / op changes
- Make PolyNorm forward fused-only in `tt-train` and keep backward as composite.
- Add explicit fused-path guard for non-tile-aligned channels (`C % 32 != 0`) with a clear error.
- Refactor PolyNorm forward CB layout to contiguous grouped indices (input/intermediate/output).
- Remove stale debug/revision scaffolding and align tail handling to `std::min(...)`.
- Remove duplicate `onetile` definitions in compute kernels that already include `compute_utils.hpp` to avoid JIT redefinition failures.

## Test updates
- Add test coverage for fused path rejecting non-tile-aligned channel count.
- Keep consolidated fused-vs-reference PolyNorm op coverage and fused-path backward smoke checks (shape/finite gradients).

### Problem description
We need a stable PolyNorm fused forward implementation for tt-train that can be exercised in realistic training runs. During bring-up, we also hit JIT compilation failures caused by duplicate `onetile` definitions in kernels that already pull `compute_utils.hpp`.

### What's changed
1. Implemented and stabilized PolyNorm fused forward integration and test coverage.
2. Removed composite forward fallback from `ops::polynorm`; forward always uses fused kernel (with `C % 32 == 0` guard).
3. Re-ran end-to-end training comparisons on NanoLlama3 Shakespeare with fused PolyNorm:
   - PolyNorm fused log: `tt-train/logs/nanollama3_polynorm_fused_5000steps_20260320_231604.log`
   - SwiGLU baseline log: `tt-train/logs/nanollama3_swiglu_5000steps_20260320_193155.log`
   - Plots: `tt-train/logs/comparison_polynorm_fused_vs_swiglu/`
     - `losses.png`
     - `losses_diff.png`
     - `step_time.png`

Training comparison summary (`plot_training_comparison.py`):
- Mean step time:
  - `swiglu`: `178.13 ms`
  - `polynorm_fused`: `296.41 ms`
  - Relative throughput: `polynorm_fused = 0.601x` vs `swiglu`
- Final loss (last 100-step average):
  - `swiglu`: `0.103774`
  - `polynorm_fused`: `0.108271`

### Checklist
- [x] [![Sanity tests](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml/badge.svg?branch=mdragula/polynorm_fw)](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml?query=branch:mdragula/polynorm_fw)
- [x] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch=mdragula/polynorm_fw)](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:mdragula/polynorm_fw)
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch=mdragula/polynorm_fw)](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:mdragula/polynorm_fw)
- [x] New/Existing tests provide coverage for changes
