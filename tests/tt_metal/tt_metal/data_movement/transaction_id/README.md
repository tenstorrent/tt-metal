# Transaction ID Data Movement Tests

This test suite implements tests that measure the functionality and performance of NOC transaction ID (TRID) mechanisms during NOC transactions between multiple Tensix cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API enables only fast dispatch mode by default. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on three Tensix cores: one master core and two subordinate cores (sub0 and sub1). The test implements either a read-after-write or write-after-read scenario.

### Read-After-Write:
1. **Data Setup**: Random data is written to both the master core and sub1 core L1 memory
2. **Write Phase**: The master core issues NOC write transactions with unique transaction IDs to transfer data to the sub0 core
3. **Read-After-Write**: The master core waits for write transactions to complete using transaction ID barriers (`noc_async_write_flushed_with_trid`), then immediately reads data from the sub1 core. Note that the transaction ID barriers prevent write-after-read (WAR) hazards where the data we are writing from the master core is not corrupted by our reads from the sub1 core.
4. **Validation**: Data integrity is verified through a simple equality check. Initial data in sub1 and master cores are compared with resulting data in master and sub0 cores, respectively.

### Write-After-Read:
1. **Data Setup**: Random data is written to both the master core and sub1 core L1 memory
2. **Read Phase**: The master core issues NOC read transactions with unique transaction IDs to transfer data from the sub1 core
3. **Write-After-Read**: The master core waits for read transactions to complete using transaction ID barriers (`noc_async_read_barrier_with_trid`), then immediately writes that same data to the sub0 core.
4. **Validation**: Data integrity is verified through a simple equality check. Initial data in sub1 and master cores are compared with resulting data in master and sub0 cores, respectively.

This pattern tests the benefits of using transaction IDs on performance, ensuring that the NOC is kept busy at the highest rate possible.

Test attributes such as number of transaction IDs, transaction sizes, and latency measures are recorded by the profiler. The test validates both functional correctness and captures performance data for bandwidth analysis.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*TransactionId*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord         | CoreCoord             | Logical coordinates for the master core that performs writes and reads. |
| sub0_core_coord           | CoreCoord             | Logical coordinates for the receiver core (write destination). |
| sub1_core_coord           | CoreCoord             | Logical coordinates for the sender core (read source). |
| num_of_trids              | uint32_t              | Number of unique transaction IDs to use (1-16). |
| pages_per_transaction     | uint32_t              | Size of each transaction in pages. |
| bytes_per_page            | uint32_t              | Size of a page in bytes. Minimum flit size per architecture. |
| l1_data_format            | DataFormat            | Data format of data that will be moved (bfloat16). |
| noc_id                    | NOC                   | Specifies which NOC to use for transactions. |
| one_packet                | bool                  | Whether to use one-packet NOC API optimizations. |
| stateful                  | bool                  | Whether to use stateful NOC API optimizations. |
| read_after_write          | bool                  | Whether to run writer_reader or reader_writer kernels. |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **Transaction ID Read After Write (ID 600)**: Tests various combinations of transaction ID counts (1-15) and transaction sizes. Uses standard NOC write/read APIs with transaction ID barriers. Maximum transaction size is 64KB (2048 pages for WH, 1024 pages for BH).

2. **Transaction ID Read After Write One Packet (ID 601)**: Similar to test case 1 but uses optimized one-packet NOC APIs (`noc_async_write_one_packet_with_trid` and `noc_async_read_one_packet`) for improved performance. Limited to NOC_MAX_BURST_WORDS (256 pages) per transaction for optimal one-packet operation.

3. **Transaction ID Read After Write One Packet Stateful (ID 602)**: Similar to test case 2 but uses stateful one-packet NOC APIs (`noc_async_write_one_packet_with_trid_with_state` and `noc_async_read_one_packet_with_state`) for further performance optimization. Also limited to NOC_MAX_BURST_WORDS (256 pages) per transaction.

4. **Transaction ID Write After Read (ID 610)**: Similar to test case 1 but uses write-after-read methodology. Uses standard NOC write/read APIs with transaction ID barriers. Maximum transaction size is 64KB (2048 pages for WH, 1024 pages for BH).

5. **Transaction ID Read After Write One Packet Stateful (ID 611)**: Similar to test case 4 but uses stateful one-packet NOC APIs (`noc_async_read_one_packet_with_state_with_trid` and `noc_async_write_one_packet_with_state`) for further performance optimization. Limited to NOC_MAX_BURST_WORDS (256 pages) per transaction.

## Implementation Details
The test uses five different kernel implementations:

- **writer_reader.cpp**: Standard transaction ID implementation using `noc_async_write_set_trid()` and `noc_async_write_flushed_with_trid()` APIs
- **writer_reader_one_packet.cpp**: Optimized implementation using one-packet APIs (`noc_async_write_one_packet_with_trid` and `noc_async_read_one_packet`) for improved performance
- **writer_reader_one_packet_stateful.cpp**: Further optimized implementation using stateful one-packet APIs (`noc_async_write_one_packet_with_trid_with_state` and `noc_async_read_one_packet_with_state`) that reduce per-transaction overhead for smaller transactions
- **reader_writer.cpp**: Standard transaction ID implementation using `noc_async_read_set_trid()` and `noc_async_read_barrier_with_trid()` APIs
- **reader_writer_one_packet_stateful.cpp**: Further optimized implementation using stateful one-packet APIs (`noc_async_read_one_packet_with_state_with_trid` and `noc_async_write_one_packet_with_state`) that reduce per-transaction overhead for smaller transactions

All kernels implement the same read-after-write or write-after-read pattern but with different NOC API optimizations suited for different transaction size ranges and performance requirements.
