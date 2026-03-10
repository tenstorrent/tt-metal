# One Packet Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between a sender and a receiver Tensix core using the one_packet APIs.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Runs either the reader kernel or the writer kernel on the master core.

If the reader kernel is run, data is written directly into the L1 memory of the subordinate core, and the kernel issues NOC instructions to read this data into the L1 address of the master core using the one_packet APIs. A read barrier is placed after these transactions in order to ensure data validity.

If the writer kernel is run, data is written directly into the L1 memory of the master core, and the kernel issues NOC instructions to write this data into the L1 address of the subordinate core using the one_packet APIs. A write barrier is placed after these transactions in order to ensure data validity.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*TensixDataMovementOnePacket*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord         | CoreCoord             | Logical coordinates for the master core. |
| subordinate_core_coord    | CoreCoord             | Logical coordinates for the subordinate core. |
| num_packets               | uint32_t              | Number of packet transactions. |
| packet_size_bytes         | uint32_t              | Size of a packet in bytes (Max 8kB for Wormhole, 16kB for Blackhole). |
| read                      | bool                  | True if the reader kernel is enabled, false if the writer kernel is enabled. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. One Packet Read Sizes: Tests packet reading over varying packet sizes and number of packets.
2. One Packet Write Sizes: Tests packet writing over varying packet sizes and number of packets.
3. One Packet Read Directed Ideal: Tests the most optimal transactions for reading packets by maximizing the number of packets and packet size to amortize initialization overhead and saturate the bandwidth.
4. One Packet Write Directed Ideal: Tests the most optimal transactions for writing packets by maximizing the number of packets and packet size to amortize initialization overhead and saturate the bandwidth.
