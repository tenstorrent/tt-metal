# All From All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions from all subordinate cores to all master cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 space is allocated on all Tensix cores involved in the data movement test. Based on a given number of reservable pages, subsets of these pages are designated as either requestor pages (masters) or responder pages (subordinates).

The subsets are determined as follows:
- num_responder_pages = num_reservable_pages / (1 + num_subordinate_cores)
- num_requestor_pages = num_responder_pages * num_subordinate_cores

The starting address of the requestor space in L1 immediately follows the final address allocated for the responder space.

Each subordinate core contains a portion of the entire data that is to be received buy each master core.
Each master core issues a NOC asynchronous read to each subordinate (responder) core, requesting and receiving data from all the subordinate cores and storing it its requestor pages.
By the end of the test, every master core should have received the same data from every subordinate core.

Test attributes such as pages per transaction and number of transactions per subordinate core, and latency measures such as kernel and pre-determined scope cycles are recorded by the profiler.

An equality check is performed by cross-checking the data of all the subordinate cores pieced-together against the data received by each master core. This ensures that the data integrity is maintained throughout the data movement process.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*AllFromAll*"
```

## Test Parameters
| Parameter                              | Data Type             | Description |
| -------------------------------------- | --------------------- | ----------- |
| test_id                                | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| mst_logical_start_coord                | CoreCoord             | Starting logical coordinates for the master core grid. |
| sub_logical_start_coord                | CoreCoord             | Starting logical coordinates for the subordinate core grid. |
| mst_grid_size                          | CoreCoord             | Size of the master core grid. |
| sub_grid_size                          | CoreCoord             | Size of the subordinate core grid. |
| num_of_transactions_per_subordinate    | uint32_t              | Number of noc transactions that each subordinate core will issue. |
| pages_reservable_per_transaction       | uint32_t              | Size of the issued noc transactions in pages. |
| bytes_per_page                         | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format                         | DataFormat            | Data format of data that will be moved. |
| noc_id                                 | NOC                   | Specifies which NOC to use for the test. |
| virtual_channel                        | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| posted                                 | N/A                   | Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to response packets) (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **All From All Packet Sizes**: Tests different number of transactions and transaction sizes by varying the num_of_transactions_per_subordinate and pages_reservable_per_transaction parameters across a full compute grid.

2. **All From All Directed Ideal**: Tests the most optimal data movement setup between all subordinate and master cores that maximizes the transaction size and performs enough transactions to amortize initialization overhead.

3. **Grid Configuration Tests**: Multiple test cases testing different master and subordinate grid configurations:
   - 2x2 From 1x1: 2x2 master grid receiving from 1x1 subordinate grid
   - 4x4 From 1x1: 4x4 master grid receiving from 1x1 subordinate grid
   - 1x1 From 2x2: 1x1 master grid receiving from 2x2 subordinate grid
   - 1x1 From 4x4: 1x1 master grid receiving from 4x4 subordinate grid
   - 2x2 From 2x2: 2x2 master grid receiving from 2x2 subordinate grid
