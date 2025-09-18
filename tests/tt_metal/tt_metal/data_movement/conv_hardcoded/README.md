# Conv Hardcoded Data Movement Tests

This test suite implements tests that measure the performance (i.e. bandwidth) of conv transactions between Tensix cores.
They are based on kernel runtime arguments of existing metal tests.

## Slow Dispatch Support
This test suite uses the TT-Metal slow dispatch mode for reliable program execution. The tests use `MeshDeviceFixture` and are designed to work with TT-Mesh APIs.

**Note**: These tests require slow dispatch mode for proper operation. They use `tt::tt_metal::detail::LaunchProgram()` for direct program execution without command queues.

## Test Flow

This test creates a buffer for various conv patterns as gotten from `pytest test_conv2d.py::test_unet_conv_wh`. It plots the bandwidth of each core.

It does not check pcc as the afformentioned test does this.

The conv patterns are the exact ones as gotten from the base tests, as such this is a directed test is is not general.

## Running the Tests
The tests use slow dispatch mode. Run with the slow dispatch environment variable:
```
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ConvHardcoded*"
```

Or run normally (the MeshDeviceFixture will automatically enforce slow dispatch mode):
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ConvHardcoded*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| dest_core_set             | CoreRangeSet          | Set of destination cores to which the data will be moved. This is a set of logical coordinates. |
| dest_core_compile_args    | std::vector<uint32_t> | Compile-time arguments for the destination core. |
| dest_core_runtime_args    | std::vector<uint32_t> | Runtime arguments for the destination core. |
| noc                       | N/A                   | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. Conv Act with halo 3x3 - A test that reads convolution activation data with a halo of 3x3.
2. Conv Act with halo 3x3 Small - A test that reads convolution activation data with a halo of 3x3, with a transaction size of 16.
3. Conv Halo Gather - A test that gathers convolution halo data.
