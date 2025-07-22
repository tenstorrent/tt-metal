# All to All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between Tensix cores.

## Test Flow

L1 space is allocated on all Tensix cores involved in the data movement test. Based on a given number of reservable pages, subsets of these pages are designated as either sender pages or receiver pages.

The subsets are determined as follows:
- num_sender_pages = num_reservable_pages / (1 + num_master_cores)
- num_receiver_pages = num_sender_pages * num_master_cores

The starting address of the receiver space in L1 immediately follows the final address allocated for the sender space.

Each master core contains a portion of the entire data that is to be received by each subordinate core.
Each master core issues a NOC asynchronous (unicast) write to each receiver core, writing its portion of data to its corresponding portion of receiver pages on each subordinate core.
By the end of the test, every subordinate core should have received the same data from every master core.

Test attributes such as pages per transaction and number of transactions per master core, and latency measures such as kernel and pre-determined scope cycles are recorded by the profiler.

A pcc check is performed by cross-checking the data of all the master cores pieced-together against the data received by each subordinate core. This ensures that the data integrity is maintained throughout the data movement process.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                         | Data Type             | Description                                                               |
| --------------------------------- | --------------------- | ------------------------------------------------------------------------- |
| test_id                           | uint32_t              | Test ID for identifying different test cases.                             |
| mst_logical_start_coord           | CoreCoord             | Logical starting coordinates for the master core range.                   |
| sub_logical_start_coord           | CoreCoord             | Logical starting coordinates for the subordinate core range.              |
| mst_grid_size                     | CoreCoord             | Grid size of the master core range.                                       |
| sub_grid_size                     | CoreCoord             | Grid size of the subordinate core range.                                  |
| num_of_transactions_per_master    | uint32_t              | Number of transactions issued per master core.                            |
| pages_reservable_per_transaction  | uint32_t              | Number of reservable pages per transaction in L1.                         |
| bytes_per_page                    | uint32_t              | Size of each page in bytes.                                               |
| l1_data_format                    | DataFormat            | Data format used for L1 data movement.                                    |
| noc_id                            | NOC                   | Specifies which NOC to use for the test.                                  |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size. Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **All to All Packet Sizes:** Tests different number of transactions and transaction sizes by varying the num_of_transactions and transaction_size_pages parameters.
a. **2x2 Packet Sizes**
2. **All to All Directed Ideal:** Tests the most optimal data movement setup between two ranges of cores.
a. **2x2 to 1x1**
b. **4x4 to 1x1**
c. **1x1 to 2x2**
d. **1x1 to 4x4**
