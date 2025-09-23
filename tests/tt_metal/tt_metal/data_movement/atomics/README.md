# Atomic Semaphore Bandwidth Tests

This directory contains tests for `noc_semaphore_inc` and `noc_async_atomic_barrier` operations with bandwidth performance sweeping.

## Overview

The tests demonstrate and measure the performance of atomic semaphore operations between cores using the NOC (Network on Chip). These tests are designed to:

1. **Test atomic semaphore correctness**: Verify that `noc_semaphore_inc` operations correctly modify remote semaphore values
2. **Measure bandwidth performance**: Sweep through different transaction sizes and counts to characterize performance
3. **Validate synchronization**: Ensure `noc_async_atomic_barrier` properly synchronizes atomic operations

## Test Structure

### Kernels

- **`atomic_semaphore_sender.cpp`**: Sender kernel that performs data writes followed by atomic semaphore increments
- **`atomic_semaphore_receiver.cpp`**: Receiver kernel that waits for atomic semaphore notifications and validates data

### Test Cases

- **`AtomicSemaphoreBandwidthSweep`**: Comprehensive sweep testing different:
  - NOC configurations (NOC_0, NOC_1)
  - Transaction counts (16 to 1024)
  - Transaction sizes (1 to 256 pages)
  - Atomic increment values (1, 2, 4)

- **`AtomicSemaphoreDirectedPerformance`**: Focused performance test with optimal parameters

## Key Features

### Atomic Operations Pattern
```cpp
// Send data
noc_async_write(src_addr, dst_addr, size, noc_index, vc);
noc_async_write_barrier();

// Atomically signal completion
noc_semaphore_inc(remote_semaphore_addr, increment_value);
noc_async_atomic_barrier();  // Ensure atomic op completes
```

### Synchronization Model
- Sender cores use `noc_semaphore_inc` to atomically increment remote semaphores
- Receiver cores poll semaphore values to detect completion of data transfers
- `noc_async_atomic_barrier()` ensures atomic operations complete before proceeding

### Performance Measurement
- Sweeps transaction sizes from single pages to maximum supported
- Varies transaction counts to measure throughput scalability
- Tests both NOC_0 and NOC_1 for comparison
- Measures atomic operation overhead vs regular NOC writes

## Usage

The tests can be run as part of the standard tt-metal test suite and will output performance metrics and validation results for atomic semaphore operations.

## Implementation Notes

- Uses L1 memory for both data storage and semaphore locations
- Semaphores are placed at configurable offsets from base data addresses
- Tests verify both correctness (expected semaphore values) and performance characteristics
- Compatible with the existing data movement test framework and utilities
