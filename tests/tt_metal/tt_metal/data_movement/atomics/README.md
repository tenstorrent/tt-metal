# Atomic Semaphore Tests

This directory contains tests for `noc_semaphore_inc` and `noc_async_atomic_barrier` operations with performance sweeping.

**Note**: Due to how the test framework works, `Transaction Size (bytes)` in the output actually means `Increment Amount`.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Sender kernel performs atomic semaphore increments with an atomic barrier after each write. Receiver kernel that waits for all expected atomic semaphore notifications. Performance is measured on the sender kernel from the first increment sent to the last barrier completed.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*AtomicSemaphore*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| sender_core_coord             | CoreCoord             | Logical coordinates for the sender core. |
| receiver_core_coord           | CoreCoord             | Logical coordinates for the receiver core. |
| num_of_transactions           | uint32_t              | Number of atomic increments that will be issued. |
| semaphore_addr_offset         | uint32_t              | Address offset in L1 for the semaphore. |
| atomic_inc_value              | uint32_t              | Amount to increment semaphore by on each transaction. |
| noc_id                           | N/A                   | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|

## Test Cases

- **`AtomicSemaphoreAdjacentIncrementValueSweep`**: Comprehensive sweep testing different:
  - NOC configurations (NOC_0, NOC_1)
  - Transaction counts (16 to 64)
  - Atomic increment values (1, 2, 3, 4)
  - Tests delay between adjacent cores ( 0,1 -> 0,0 )

- **`AtomicSemaphoreNonAdjacentIncrementValueSweep`**:
  - Same as previous test, but tests delay between cores at (almost) opposite ends of the device
  - Sender is positioned 1 row & 1 column from the far corner, to allow dispatch cores to use the far opposite position
