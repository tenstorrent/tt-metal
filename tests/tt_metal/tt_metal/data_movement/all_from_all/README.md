# All from All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between Tensix cores.

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

A pcc check is performed by cross-checking the data of all the subordinate cores pieced-together against the data received by each master core. This ensures that the data integrity is maintained throughout the data movement process.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                             | Data Type             | Description                                                               |
| ------------------------------------- | --------------------- | ------------------------------------------------------------------------- |
| test_id                               | uint32_t              | Test ID for identifying different test cases.                             |
| mst_logical_start_coord               | CoreCoord             | Logical starting coordinates for the master core range.                   |
| sub_logical_start_coord               | CoreCoord             | Logical starting coordinates for the subordinate core range.              |
| mst_grid_size                         | CoreCoord             | Grid size of the master core range.                                       |
| sub_grid_size                         | CoreCoord             | Grid size of the subordinate core range.                                  |
| num_of_transactions_per_subordinate   | uint32_t              | Number of transactions performed per subordinate core.                    |
| pages_reservable_per_transaction      | uint32_t              | Number of reservable pages per transaction in L1.                         |
| bytes_per_page                        | uint32_t              | Size of each page in bytes.                                               |
| l1_data_format                        | DataFormat            | Data format used for L1 data movement.                                    |
| noc_id                                | NOC                   | Specifies which NOC to use for the test.                                  |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size. Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **All from All Packet Sizes:** Tests different number of transactions and transaction sizes by varying the num_of_transactions and transaction_size_pages parameters.
a. **2x2 Packet Sizes**
2. **All from All Directed Ideal:** Tests the most optimal data movement setup between two ranges of cores.
a. **2x2 from 1x1**
b. **4x4 from 1x1**
c. **1x1 from 2x2**
d. **1x1 from 4x4**
