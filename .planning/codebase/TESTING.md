# Testing Patterns

**Analysis Date:** 2026-03-16

## Test Framework

**Runner:**
- Python: pytest (version 7.2+, minversion = 7.2)
- C++: Google Test (gtest) with gmock for mocking
- Config: `pytest.ini` at project root

**Assertion Library:**
- Python: pytest assertions (`assert` statements)
- C++: gtest assertions (`EXPECT_EQ`, `EXPECT_THAT`, `ASSERT_EQ`)

**Run Commands:**
```bash
pytest                                # Run all Python tests
pytest -m post_commit                 # Run post-commit tests only
pytest --runslow                      # Include slow tests
cmake --build build --target fabric_unit_tests -j$(nproc)  # Build C++ unit tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests  # Run C++ tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*ChannelTrimming*"  # Run specific test
pytest -k "test_binary_mul" --num-iterations=10  # Run with custom parameters
```

**Test Configuration (pytest):**
```
Timeout: 300 seconds per test
Import mode: importlib (--import-mode=importlib)
Verbosity: -vvs (verbose with short traceback, show all assertions)
JUnit XML: generated/test_reports/most_recent_tests.xml
Durations: Show all test execution times (--durations=0)
```

## Test File Organization

**Location:**
- Python: Co-located in `tests/` directory tree matching source structure
  - `tests/tt_metal/` for tt_metal tests
  - `tests/ttnn/` for ttnn tests
  - `tests/didt/` for device-side integration and diagnostics tests
- C++: Co-located in source tree with `*_test.cpp` suffix
  - `tests/tt_metal/tt_metal/hal_codegen/codegen_test.cpp`
  - `tests/tt_metal/tt_metal/test_kernels/dataflow/*_test.cpp`

**Naming:**
- Python: `test_*.py` prefix (e.g., `test_binary_mul.py`, `test_metadata.py`)
- C++: `*_test.cpp` suffix (e.g., `codegen_test.cpp`, `reader_cb_test.cpp`)

**Structure:**
```
tests/
├── tt_metal/
│   ├── microbenchmarks/
│   │   ├── conftest.py          # pytest hooks for test customization
│   │   ├── ethernet/
│   │   │   ├── test_all_ethernet_links_bandwidth.py
│   │   │   └── conftest.py
│   │   └── tensix/
│   ├── tt_metal/
│   │   ├── data_movement/python/
│   │   │   ├── test_metadata.py
│   │   │   └── conftest.py
│   │   └── hal_codegen/
│   │       └── codegen_test.cpp
│   └── perf_microbenchmark/
├── ttnn/
│   ├── conftest.py               # Shared pytest fixtures and config hooks
│   └── unit_tests/
├── didt/
│   └── test_binary_mul.py         # Device-side integration tests
└── scripts/
    └── common.py                  # Shared test utilities
```

## Test Structure

**Suite Organization:**
```python
# Python test class pattern
class BinaryMulTest(OpTestBase):
    def __init__(self, *args, gelu=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gelu = gelu

    def run_device_operation(self):
        return ttnn.mul(
            self.activations,
            self.inputs[0],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)] if self.gelu else [],
        )

# Test function using class
@pytest.mark.parametrize("gelu, math_fidelity", [
    (False, ttnn.MathFidelity.LoFi),
    (True, ttnn.MathFidelity.HiFi2),
], ids=["without_gelu", "with_gelu"])
@pytest.mark.parametrize("mesh_device", [
    pytest.param(1, id="1chips"),
    pytest.param((8, 4), id="galaxy"),
], indirect=["mesh_device"])
def test_binary_mul(mesh_device, gelu, math_fidelity, ...):
    binary_mul_test = BinaryMulTest(mesh_device, OpParameter(...), ...)
    binary_mul_test.run_op_test()
```

```cpp
// C++ test pattern
namespace {

void verify_equal(const ::PacketInfo& expected, types::PacketInfo::ConstView view) {
    EXPECT_EQ(view.size(), sizeof expected);
    EXPECT_EQ(view.len(), expected.len);
    // ... more assertions
}

}  // namespace

TEST(CodegenTest, CodegenTest) {
    auto factory = create_factory();
    types::PacketInfo info = factory.create<types::PacketInfo>();
    ASSERT_EQ(info.size(), sizeof(::PacketInfo));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    // ... setup and assertions
}
```

**Patterns:**
- **Setup**: Fixture-based (pytest fixtures defined in conftest.py, C++ test constructors)
- **Teardown**: Implicit via fixture cleanup or RAII patterns
- **Assertion**: Direct comparisons (`EXPECT_EQ`, `assert`) and custom verify functions

## Mocking

**Framework:**
- Python: pytest fixtures with monkeypatch and indirect parameterization
- C++: gmock (Google Mock) for function and object mocking

**Patterns (Python):**
```python
@pytest.fixture(indirect=["mesh_device"])
def mesh_device(request):
    """Fixture that creates a mesh device with indirect parameterization."""
    num_devices = request.param
    # Create device configuration
    return mesh_device_context

# Usage in test
def test_with_fixture(mesh_device):
    # mesh_device is automatically provided by the fixture
    pass
```

**Patterns (C++):**
```cpp
// Google Mock usage with EXPECT_THAT
EXPECT_THAT(hop.addr(), testing::ElementsAreArray(hop_expected.addr));

// Manual mock-like verification via comparison
verify_equal(*raw_ptr, const_view);
```

**What to Mock:**
- External device communication (via fixtures providing mock mesh devices)
- File I/O operations (for unit tests of data processing)
- Network operations (ethernet link tests use actual hardware when available)

**What NOT to Mock:**
- Core device operations (use actual device or simulator)
- Memory allocation/deallocation (test real behavior)
- Kernel execution (run actual kernels via ttnn APIs)

## Fixtures and Factories

**Test Data (Python):**
```python
# conftest.py fixture pattern
@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    yield

@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield

# Custom parameterization with indirect
@pytest.mark.parametrize("mesh_device", [
    pytest.param(1, id="1chips"),
    pytest.param((8, 4), id="galaxy"),
], indirect=["mesh_device"])
def test_op(mesh_device):
    pass
```

**Fixture Scopes:**
- `session`: One-time setup across entire test run (is_ci_v2_env)
- `function`: Per-test setup (reset_seeds, function_level_defaults)
- `module`: Implicit for module-level conftest.py hooks
- `autouse=True`: Automatically applied to all tests (pre_and_post in conftest.py)

**Location:**
- Global fixtures: `/home/snijjar/tt-metal-2/conftest.py`
- Per-directory fixtures: `tests/*/conftest.py`
- Test-specific factories: In test file or imported from `op_test_base`, `test_utils`

## Coverage

**Requirements:**
- No enforced coverage target
- Coverage tracking via pytest plugins (JUnit XML output available)

**View Coverage:**
```bash
pytest --cov=tt_metal --cov-report=html  # Generate HTML coverage report
pytest --cov=tt_metal --cov-report=term  # Terminal coverage summary
```

**Coverage Output:**
- Generated to `generated/test_reports/most_recent_tests.xml`
- HTML reports typically in `htmlcov/` directory

## Test Types

**Unit Tests:**
- Scope: Individual functions, classes, or modules
- Approach: Test in isolation with mocked dependencies
- Example: `tests/tt_metal/tt_metal/data_movement/python/test_metadata.py` tests YAML loading and field mapping
- Framework: pytest for Python, gtest for C++

**Integration Tests:**
- Scope: Multiple components working together (e.g., device + operator)
- Approach: Use real fixtures and device API calls
- Example: `tests/didt/test_binary_mul.py` tests binary multiplication on mesh devices
- Framework: pytest with device fixtures (mesh_device with indirect parameterization)

**E2E Tests:**
- Framework: Not formally organized; some tests use full model execution
- Scope: Full model inference pipelines
- Location: `models/*/tests/` directories and `tests/ttnn/` integration suites
- Configuration: Parametrized across device configs (1chip, 8chips, galaxy)

**Microbenchmarks:**
- Location: `tests/tt_metal/microbenchmarks/`
- Purpose: Measure performance (bandwidth, latency) of operations
- Configuration: Custom pytest options via conftest.py (--num-iterations, --packet-size)
- Output: CSV reports with results per packet size

## Common Patterns

**Async Testing:**
```python
# Python: Using async fixtures with device operations
@pytest.fixture(indirect=["mesh_device"])
def mesh_device(request):
    # Setup async device context
    pass

# Device operations are implicitly async via ttnn API
result = ttnn.mul(tensor1, tensor2)  # Non-blocking, returns future-like object
```

**Error Testing:**
```python
# Test that preconditions are validated
class TestMetadataLoader:
    def get_test_metadata(self, test_id: int, arch: str):
        test_info = self.load_test_information()
        test_data = test_info.get("tests", {}).get(test_id, {})

        if "memory_type" not in test_data:
            raise KeyError(f"Test ID {test_id} does not have complete metadata")

        return result

# Asserting expected errors in test
with pytest.raises(KeyError):
    loader.get_test_metadata(invalid_id, arch)
```

**Parametrization:**
```python
@pytest.mark.parametrize("gelu, math_fidelity", [
    (False, ttnn.MathFidelity.LoFi),
    (True, ttnn.MathFidelity.HiFi2),
], ids=["without_gelu", "with_gelu"])
@pytest.mark.parametrize("mesh_device", [...], indirect=["mesh_device"])
def test_multiple_configs(mesh_device, gelu, math_fidelity):
    # Test runs with all combinations of parameters
    pass
```

## Pytest Markers

**Available Markers:**
- `@pytest.mark.post_commit`: Run in post-commit CI
- `@pytest.mark.slow`: Long-running tests (skip unless --runslow)
- `@pytest.mark.frequently_hangs`: Known to hang devices
- `@pytest.mark.requires_fast_runtime_mode_off`: Skip when fast runtime enabled
- `@pytest.mark.eager_host_side`: Host-side eager release builds
- `@pytest.mark.requires_grid_size(min_x, min_y)`: Skip if grid smaller than specified

**Custom Parameters:**
```python
def pytest_addoption(parser):
    parser.addoption("--num-iterations", action="store", type=int,
                     help="Number of iterations to run each test config")
```

## Test Organization Best Practices

**Device Tests:**
- Use indirect fixture parameterization for device configurations
- Initialize OpParameter objects with specific shapes, dtypes, memory configs
- Call `run_op_test()` from test base classes for standardized execution

**Microbenchmarks:**
- Capture profiler output to CSV files
- Use logger.info() for reporting results
- Include expected bandwidth/latency bounds and validate results

**Kernel Tests:**
- C++ kernel tests focus on codegen and accessor patterns
- Use random data generation for exhaustive verification
- Verify both read and write operations through accessor methods

---

*Testing analysis: 2026-03-16*
