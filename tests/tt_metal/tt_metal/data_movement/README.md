# Data Movement Test Suite

This test suite addresses the functionality and performance (i.e. bandwidth) of various data movement scenarios.

## Dispatch Mode Support
This test suite includes tests using both fast dispatch (Mesh Device API) and slow dispatch modes:

### Fast Dispatch (Mesh Device API)
Most test suites use the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. These tests use `GenericMeshDeviceFixture` and run on single-device unit meshes with fast dispatch mode for optimal performance.

### Slow Dispatch
Some test suites use slow dispatch mode for reliable program execution. These tests use `DeviceFixture` and execute programs directly using `tt::tt_metal::detail::LaunchProgram()`. Tests requiring slow dispatch include:
- **Deinterleave Hardcoded** (IDs 200-201)
- **Conv Hardcoded** (IDs 21-23)
- **Reshard Hardcoded** (IDs 17-20)

## Tests in the Test Suite

| Name                        | ID(s)                | Description                                                                             |
| ----------                  | -----                | ----------------------------------------------------                                    |
| DRAM Unary                  | 0-3                  | Transactions between DRAM and a single Tensix core.                                     |
| One to One                  | 4, 50, 150-151       | Write transactions between two Tensix cores.                                            |
| One From One                | 5, 51, 152-153       | Read transactions between two Tensix cores.                                             |
| One to all                  | 6-8, 52, 154-155     | Writes transaction from one core to all cores.                                          |
| One to all Multicast        | 9-14, 53-54, 100-102 | Writes transaction from one core to all cores using multicast.                          |
| One From All                | 15, 30, 156-157      | Read transactions between one gatherer Tensix core and multiple sender Tensix cores.    |
| Loopback                    | 16, 55               | Does a loopback operation where one cores writes to itself.                             |
| Reshard Hardcoded           | 17-20                | Uses existing reshard tests to analyse their bandwidth and latency. **(Slow Dispatch)** |
| Conv Hardcoded              | 21-23                | Uses existing conv tests to analyse their bandwidth and latency. **(Slow Dispatch)**    |
| Interleaved Page Read/Write | 61-69, 71-75         | Reads and writes pages between interleaved buffers and a Tensix core.                   |
| One Packet Read/Write       | 80-83                | Reads or writes packets between two Tensix cores.                                       |
| Core Bidrectional           | 140-148              | Tensix core reads from and writes to another Tensix core simultaneously.                |
| Deinterleave                | 200-201              | Tests deinterleaving. **(Slow Dispatch)**                                               |
| All to all                  | 300-308              | Write transactions from multiple cores to multiple cores.                               |
| All from all                | 310-318              | Read transactions from multiple cores to multiple cores.                                |
| I2S Hardcoded               | 400-405              | Tests interleaved to sharded data movement operations for different memory layouts.     |

## Running Tests
### C++ Gtests
Before running any tests, build the repo with tests: ```./build_metal.sh --build-tests```

Most tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement
```

To run a single test, add a gtest filter with the name of the test. Example:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*TensixDataMovementDRAMInterleavedPacketSizes*"
```

For tests that require slow dispatch mode (Deinterleave, Conv, and Reshard Hardcoded tests), run with:
```
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*DeinterleaveHardcoded*|*ConvHardcoded*|*ReshardHardcoded*"
```

### Pytest
Before running any tests, build the repo with profiler and tests: ```./build_metal.sh --enable-profiler --build-tests```
Then, for performance checks and more extensive testing, our Python test can be run as follows:
```
pytest tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py <options>
```

Options can be used to disable new profiling (i.e. use existing results), enable plotting of results etc.
An exhaustive list of options and their descriptions can be found in `./conftest.py`

## Adding Tests
Follow these steps to add new tests to this test suite.

1. **Choose dispatch mode:** Decide whether your test should use fast dispatch (Mesh Device API) or slow dispatch:
    - **Fast Dispatch (recommended)**: Use `GenericMeshDeviceFixture` for new performance tests
    - **Slow Dispatch**: Use `DeviceFixture` only if fast dispatch APIs don't work for your specific test case
2. Create a new directory with a descriptive name for the test.
    - **Example:** `./dram_unary`
3. In this directory, create the c++ test file with a filename that starts with "test_".
    - **Example:** `./dram_unary/test_unary_dram.cpp`
4. Write your test in this file and place the kernels you use within this test in "kernels" directory.
    - **Example:** `./dram_unary/kernels/reader_unary.cpp`
5. Assign your test a unique test id to make sure your test results are grouped together and are plotted separately from other tests.
    - Refer to the "Tests in the Test Suite" section for already taken test ids.
    - Preferably use the next integer available.
    - Add the test id and test name to this README and to `/python/test_mappings/test_information.yaml`. If test bounds are relevant, add these to `/python/test_mappings/test_bounds.yaml`
6. Create a README file within the test directory that describes:
    1. What your test does,
    2. What the test parameters are,
    3. What different test cases are implemented,
    4. And which dispatch mode it uses.
7. In the `CMakeLists.txt` file, add your test path in the `set(UNIT_TESTS_DATA_MOVEMENT_SRC ... )` call.
    - **Example:** `${CMAKE_CURRENT_SOURCE_DIR}/dram_unary/test_unary_dram.cpp`

**Note:** Make sure the tests pass by building and running as above.
