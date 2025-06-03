# Reshard Hardcoded Data Movement Tests

This test suite implements tests that measure the performance (i.e. bandwidth) of reshard transactions between Tensix cores.
They are based on kernel runtime arguments of existing metal tests.

## Test Flow

This test creates a buffer for various sharding patterns as gotten from `pytest test_core.py -k test_reshard`. It plots the bandwidth of each core.

It does not check pcc as the afformentioned test does this.

The sharding patterns are the exact ones as gotten from the base tests, as such this is a directed test is is not general.

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

1. Reshard Hardcoded Small - A test that reshards a small amount of data using 2 cores.
2. Reshard Hardcoded Medium - A test that reshards a medium amount of data using 2 cores.
3. Reshard Hardcoded Many Cores - A test that reshards a medium amount of data using 8 cores.
4. Reshard Hardcoded 2 Cores to Many Cores - A test that reshards 2 cores to 8 cores.
