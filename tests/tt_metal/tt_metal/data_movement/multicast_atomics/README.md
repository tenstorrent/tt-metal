# Multicast Atomic Semaphore Tests

This directory contains tests for `noc_semaphore_inc_multicast` which performs atomic increment operations on semaphores using multicast to multiple destination cores simultaneously.

## Overview

The `noc_semaphore_inc_multicast` API allows a single core to atomically increment semaphores on a grid of destination cores in a single multicast operation. This is useful for synchronization patterns where multiple cores need to signal multiple destination cores.

## Mesh Device API Support

This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

## Device 2.0 API Support

This test suite includes tests using both the standard NOC API and the experimental Device 2.0 API

## Test Flow

1. **Sender Kernel**: Performs multicast atomic semaphore increments to a rectangular grid of destination cores
2. **Receiver Kernels**: Wait for the semaphore to reach the expected value

## Running the Tests

```bash
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*MulticastAtomic*"
```

## Test Cases

| Test Name                          | ID  | Description |
| ---------------------------------- | --- | ----------- |
| MulticastAtomicSingleSource        | 321 | Single sender core multicasts atomic increment to 12 destinations (NOC_0) |
| MulticastAtomicMultiSource         | 322 | 4 sender cores multicast atomic increment to 12 destinations (NOC_0)      |
| MulticastAtomicSingleSourceNOC1    | 323 | Single sender core multicasts atomic increment to 12 destinations (NOC_1) |
| MulticastAtomicMultiSourceNOC1     | 324 | 4 sender cores multicast atomic increment to 12 destinations (NOC_1)      |
| MulticastAtomicLargerIncrement     | 325 | 4 senders, 3 transactions each, increment value 5 (NOC_0)                 |
| MulticastAtomicLargerIncrementNOC1 | 326 | 4 senders, 3 transactions each, increment value 5 (NOC_1)                 |
| MulticastAtomicSingleSource_2_0    | 327 | Single sender using experimental API                                      |
| MulticastAtomicLargerIncrement_2_0 | 328 | 4 senders, 3 transactions each, increment value 5 using experimental API  |

### SingleSourceMulticastAtomic (IDs 321, 323, 327)
- **Description**: Single sender core multicasts atomic increment to a 3x4 grid of 12 destination cores
- **Expected Result**: All 12 destination cores have semaphore value = 1

### MultiSourceMulticastAtomic (IDs 322, 324)
- **Description**: 4 sender cores each multicast atomic increment to the same 3x4 grid of 12 destination cores
- **Expected Result**: All 12 destination cores have semaphore value = 4 (one increment from each sender)

### LargerIncrementMulticastAtomic (IDs 325, 326, 328)
- **Description**: 4 sender cores each perform 3 multicast atomic increments with value 5 to a 3x4 grid of 12 destination cores
- **Expected Result**: All 12 destination cores have semaphore value = 60 (4 senders × 3 transactions × 5 increment)

## Configuration Parameters

| Parameter            | Type                  | Description |
| -------------------- | --------------------- | ----------- |
| sender_cores         | vector<CoreCoord>     | List of sender core coordinates |
| dst_grid_start       | CoreCoord             | Start coordinate of destination grid |
| dst_grid_size        | CoreCoord             | Size of destination grid (width x height) |
| num_of_transactions  | uint32_t              | Number of multicast atomic increments per sender |
| atomic_inc_value     | uint32_t              | Value to increment semaphore by |
| noc_id               | NOC                   | Which NOC to use (NOC_0 or NOC_1) |
| use_2_0_api          | bool                  | Use experimental Device 2.0 API (default: false) |

## Important Notes

- The multicast sender **cannot** be part of the multicast destination grid
- Verification is implicit - if the program completes, receivers received the expected semaphore value
- Both NOC_0 and NOC_1 are tested for standard API scenarios
- Both API versions use `CreateSemaphore` for proper semaphore management
