# Codebase Testing: tt-metal

**Analyzed:** 2026-03-12

---

## Test Frameworks

| Language | Framework | Notes |
|----------|-----------|-------|
| Python | pytest 7.2+ | Primary test runner for model/integration tests |
| C++ | GoogleTest (gtest) | Used for unit and integration tests |
| CMake | CTest | Orchestrates C++ test execution |

**pytest configuration:** `pytest.ini` at repo root
- Default timeout: 300s
- Imports: `--import-mode=importlib`
- Output: JUnit XML at `generated/test_reports/most_recent_tests.xml`

---

## Directory Structure

```
tests/
├── tt_metal/               # Core metal layer tests
│   ├── tt_metal/           # C++ unit/integration tests
│   │   ├── perf_microbenchmark/
│   │   │   └── routing/    # Fabric router benchmarks (test_tt_fabric)
│   │   └── data_movement/
│   ├── microbenchmarks/    # Python benchmarks (includes ethernet/)
│   └── multihost/          # Multi-host fabric/CCL tests
├── ttnn/                   # TTNN Python API tests
│   └── conftest.py
├── sweep_framework/        # Parametric sweep tests
├── scale_out/              # Scale-out/multi-device tests
├── nightly/                # Nightly regression suites
│   ├── single_card/
│   ├── t3000/
│   └── blackhole/
├── device_perf_tests/      # Device-level performance tests
├── didt/                   # DI/DT (device-in/device-out) tests
│   └── op_test_base.py     # Base class for op tests
└── conftest.py             # Root conftest (not at tests/ level — at repo root)
```

---

## Test Types

### C++ Microbenchmark / Performance Tests (fabric focus)
- **Binary:** `build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric`
- **Config YAML files:** `tests/tt_metal/tt_metal/perf_microbenchmark/routing/*.yaml`
  - `test_fabric_ubench_at_least_2x2_mesh.yaml` — primary 2x2 bandwidth benchmark
  - `test_fabric_sanity_common.yaml` — sanity suite
  - `test_fabric_2d_torus_nightly.yaml` — nightly 2D torus
  - `test_fabric_test_2erisc_quick.yaml` — quick dual-ERISC test
- **Run pattern:** `./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --config <yaml>`
- **Hardware requirement:** Real TT device (cannot run in simulation)

### Python Integration Tests
- Located under `tests/ttnn/`, `tests/tt_metal/microbenchmarks/`
- Run via `pytest tests/ttnn/...`
- Hardware-dependent tests require `TT_METAL_HOME` to be set

### Multihost Tests
- `tests/tt_metal/multihost/fabric_tests/` — fabric-specific multi-host scenarios
- Some disabled due to Issue #36811 (descriptor merger tests)

### Sweep Framework
- `tests/sweep_framework/` — parametric sweeps for TTNN op coverage
- Generates coverage across dtype/shape combinations

---

## pytest Markers

Key markers used to control test execution:

| Marker | Purpose |
|--------|---------|
| `post_commit` | Run on every commit |
| `frequent` | Run every few hours |
| `slow` | Long-running tests |
| `models_performance_bare_metal` | Model perf on bare metal |
| `requires_grid_size(min_x, min_y)` | Skip if device grid too small |
| `use_module_device(device_params)` | Module-scoped device for speed |
| `frequently_hangs` | Known-flaky hardware tests |

---

## Fixtures (Python)

Defined in `conftest.py` files at various levels:

- **`device`** — Function-scoped TT device fixture (opens/closes per test)
- **`_device_module_impl`** — Module-scoped device (faster for batched tests; use with `use_module_device` marker)
- **`mesh_device`** — Multi-device mesh fixture for scale-out tests
- **`reset_seeds`** — Resets random seeds for determinism
- **`model_location_generator`** — Resolves model checkpoint paths

---

## C++ Test Patterns

### GTest Structure
```cpp
TEST(FabricTest, SomeFeature) {
    // Arrange
    // Act
    // Assert using ASSERT_EQ / EXPECT_EQ / EXPECT_TRUE
}

TEST_P(FabricParamTest, BandwidthSweep) {
    auto [packet_size, expected_bw] = GetParam();
    // ...
}
INSTANTIATE_TEST_SUITE_P(...)
```

### TT_FATAL / TT_ASSERT (not GTest)
- `TT_FATAL(cond, msg)` — Terminates process on failure (use for unrecoverable errors)
- `TT_ASSERT(cond)` — Debug-mode only assertion

---

## Mocking Strategy

**C++:** Minimal mocking; tests use real device hardware when possible.
- `gmock` is available but rarely used in fabric tests

**Python:** Minimal mocking; tests run against real devices.
- `unittest.mock` used for host-side logic only (not device kernels)

---

## Board Reset

If a test hangs or produces no output:
```bash
tt-smi -r 0,1,2,3
```

Required before re-running tests after a hang.

---

## Environment Setup

```bash
export TT_METAL_HOME=/home/snijjar/tt-metal
# Optional profiling:
export TT_FABRIC_PROFILE_SPEEDY_PATH=1
export TT_FABRIC_PROFILE_SPEEDY_TIMER_MASK=<bitmask>  # one-hot only
```

---

## Known Test Gaps

- No unit tests for kernel functions (device-side code can only be tested via hardware)
- No per-VC state isolation tests
- No chaos/fault injection tests
- Issue #36811: 9 descriptor merger tests disabled for multi-host

---

*Last updated: 2026-03-12 via gsd:map-codebase*
