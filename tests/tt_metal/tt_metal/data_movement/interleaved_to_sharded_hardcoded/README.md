# Interleaved to Sharded Hardcoded Data Movement Tests

This test suite implements tests that measure the performance (i.e. bandwidth) of interleaved to sharded data movement operations between Tensix cores.
They are based on kernel runtime arguments of existing metal tests (test_interleaved_to_dram_height_sharded, test_interleaved_to_dram_width_sharded, test_interleaved_to_dram_sharded_via_to_memory_layout)

## Test Flow

This test creates buffers for various interleaved to sharded conversion patterns. It tests both reader and writer operations for different memory layouts (row major and tile) and storage types (DRAM and L1). It plots the bandwidth of each core.

It does not check pcc as the afformentioned test does this.

The interleaved to sharded patterns are the exact ones as gotten from the base tests, as such this is a directed test, it is not general.

## Test Parameters
| Parameter                 | Data Type                          | Description |
| ------------------------- | ---------------------              | ----------- |
| test_id                   | uint32_t                           | Test id for signifying different test cases (starts from 202). |
| compile_args              | std::vector<uint32_t>              | Compile-time arguments for the kernel. |
| runtime_args              | std::vector<uint32_t>              | Runtime arguments for the kernel. |
| master_core_coord         | CoreCoord                          | Coordinate of the master core running the kernel. |
| noc_id                    | NOC                                | Specify which NOC to use for the test |
| input_data_format         | tt::DataFormat                     | Data format for input/output (e.g., Float32, Float16_b) |

## Test Cases
Each test case uses various data formats and page sizes depending on the memory layout.

1. **I2S - DRAM Sharded Row Major Writer** - Tests writing data to DRAM in sharded row major layout.
2. **I2S - DRAM Sharded Tile Writer** - Tests writing data to DRAM in sharded tile layout.
3. **I2S - DRAM Interleaved Tile Reader** - Tests reading data from DRAM in interleaved tile format.
4. **I2S - L1 Interleaved Tile Reader** - Tests reading data from L1 memory in interleaved tile format.
5. **I2S - DRAM Interleaved Row Major Reader** - Tests reading data from DRAM in interleaved row major format.
6. **I2S - L1 Interleaved Row Major Reader** - Tests reading data from L1 memory in interleaved row major format.
