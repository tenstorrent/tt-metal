Consolidate duplicated random-data generation across `tt-train` tests and benchmarks into shared test utilities, and fix follow-up build issues so `--build-tt-train` completes successfully.

### Ticket
Link to Github Issue

### Problem description
Random test/benchmark data generation in `tt-train` was duplicated across many files (custom `parallel_generate` lambdas, local wrappers, `xt::random`, and manual vector copy/adapt code), making maintenance harder and behavior less consistent.

### What's changed
- Added shared helpers in `tt-train/sources/ttml/test_utils/random_data.hpp` and migrated tests/benchmarks to use them.
- Simplified many callsites from `xt::empty + fill_uniform` (or vector+adapt/copy) to direct `make_uniform_xarray` / `make_uniform_vector`.
- Removed unused normal-distribution helpers and redundant local wrappers.
- Renamed `tt-train/tests/ops/layer_norm_composite_test.cpp` to `tt-train/tests/ops/layernorm_composite_test.cpp` and updated `tt-train/tests/CMakeLists.txt`.
- Applied follow-up fixes required for successful build:
  - corrected xtensor include paths in `random_data.hpp`
  - fixed template `static_assert` placement in benchmark/serialization helpers
  - resolved duplicate-symbol issue in cross-entropy tests

### Validation
- `./build_metal.sh -b Release --build-tt-train` passed
- `ctest --test-dir build_Release/tt-train/tests --output-on-failure` passed (`475` passed, `0` failed, `2` disabled)
- Benchmarks executed successfully:
  - `adamw_benchmark`
  - `from_vector_benchmark`
  - `matmuls_benchmark`
  - `swiglu_fusion_benchmark`

### Checklist
- [ ] [![Sanity tests](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml/badge.svg?branch=mdragula/random-data-utils-tt-train)](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml?query=branch:mdragula/random-data-utils-tt-train)
- [ ] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch=mdragula/random-data-utils-tt-train)](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:mdragula/random-data-utils-tt-train)
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch=mdragula/random-data-utils-tt-train)](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:mdragula/random-data-utils-tt-train)
