# Data Movement Test Suite

This test suite addresses the functionality and performance (i.e. bandwidth) of various data movement scenarios.

## Tests in the Test Suite

| Name                 | ID(s)      | Description                                                                          |
| ----------           | -----      | ----------------------------------------------------                                 |
| DRAM Unary           | 0-3        | Transactions between DRAM and a single Tensix core.                                  |
| One to One           | 4, 50      | Write transactions between two Tensix cores.                                         |
| One From One         | 5, 51      | Read transactions between two Tensix cores.                                          |
| One to all           | 6-8, 52    | Writes transaction from one core to all cores.                                       |
| One to all Multicast | 9-14       | Writes transaction from one core to all cores using multicast.                       |
| One From All         | 15, 30     | Read transactions between one gatherer Tensix core and multiple sender Tensix cores. |
| Loopback             | 16         | Does a loopback operation where one cores writes to itself.                          |
| Reshard Hardcoded    | 17-20      | Uses existing reshard tests to analyse their bandwidth and latency.                  |
| Conv Hardcoded       | 21-23      | Uses existing conv tests to analyse their bandwidth and latency.                     |

## Running Tests
### C++ Gtests
Before running any tests, build the repo with tests: ```./build_metal.sh --build-tests```
Then, to run the whole test suite execute the following command:
```
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement
```

To run a single test, add a gtest filter with the name of the test. Example:
```
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*TensixDataMovementDRAMInterleavedPacketSizes*"
```

### Pytest
Before running any tests, build the repo with profiler and tests: ```./build_metal.sh --enable-profiler --build-tests```
Then, for performance checks and more extensive testing, our Python test can be run as follows:
```
pytest tests/tt_metal/tt_metal/data_movement <options>
```

Options can be used to disable new profiling (i.e. use existing results), enable plotting of results etc.
An exhaustive list of options and their descriptions can be found in `./conftest.py`

## Adding Tests
Follow these steps to add new tests to this test suite.

1. Create a new directory with a descriptive name for the test.
    - **Example:** `./dram_unary`
2. In this directory, create the c++ test file with a filename that starts with "test_".
    - **Example:** `./dram_unary/test_unary_dram.cpp`
3. Write your test in this file and place the kernels you use within this test in "kernels" directory.
    - **Example:** `./dram_unary/kernels/reader_unary.cpp`
4. Assign your test a unique test id to make sure your test results are grouped together and are plotted separately from other tests.
    - Refer to the "Tests in the Test Suite" section for already taken test ids.
    - Preferably use the next integer available.
    - Update the `test_id_to_name` and `test_bounds` objects with the test id, test name and test bounds.
5. Create a README file within the test directory that describes:
    1. What your test does,
    2. What the test parameters are,
    3. And what different test cases are implemented.
6. In the `CMakeLists.txt` file, add your test path in the `set(UNIT_TESTS_DATA_MOVEMENT_SRC ... )` call.
    - **Example:** `${CMAKE_CURRENT_SOURCE_DIR}/dram_unary/test_unary_dram.cpp`

**Note:** Make sure the tests pass by building and running as above.
