.. _multicore_reuse_detailed_guide:

Detailed Guide: From Basic Multi-Core to Optimized Data Reuse
**************************************************************

This section provides a comprehensive explanation for implementing ``matmul_multicore_reuse``
starting from a working ``matmul_multi_core`` implementation. The goal is to help you
understand the motivation, design decisions, and step-by-step changes required to achieve
significant performance improvements through data reuse.


Motivation: Why the Basic Multi-Core Implementation is Suboptimal
=================================================================

In the basic multi-core implementation (``matmul_multi_core.cpp``), each core computes
one output tile at a time. For each output tile ``C[i,j]``, the reader kernel fetches
**all** tiles along the inner dimension ``K`` for both the corresponding row of ``A``
and column of ``B``. This approach has a critical inefficiency: **tiles from input
matrices are re-fetched from DRAM multiple times**.

Consider a concrete example with matrices A (4x4 tiles), B (4x4 tiles), producing
C (4x4 tiles). To compute output tiles ``C[0,0]`` and ``C[0,1]``:

+----------------+----------------------------------------+----------------------------------------+
| Output Tile    | Tiles Read from A                      | Tiles Read from B                      |
+================+========================================+========================================+
| C[0,0]         | A[0,0], A[0,1], A[0,2], A[0,3]         | B[0,0], B[1,0], B[2,0], B[3,0]         |
+----------------+----------------------------------------+----------------------------------------+
| C[0,1]         | A[0,0], A[0,1], A[0,2], A[0,3]         | B[0,1], B[1,1], B[2,1], B[3,1]         |
+----------------+----------------------------------------+----------------------------------------+

Notice that **the entire row 0 of A is read twice** -- once for C[0,0] and once for C[0,1].
In general, for an MxK @ KxN matmul producing MxN output tiles:

- Each row of A is read N times (once per column of C in that row)
- Each column of B is read M times (once per row of C in that column)

This redundant DRAM traffic becomes the performance bottleneck, especially as matrix
dimensions grow.


Visualizing the Data Reuse Opportunity
======================================

The following ASCII diagram illustrates the basic approach versus the optimized approach
for a 4x4 output tile matrix with K=4 inner dimension tiles::

    BASIC APPROACH: Process output tiles one at a time (high DRAM traffic)
    ======================================================================

         K tiles (inner dimension)
         <---------------------->
       +----+----+----+----+
    M  | A0 | A1 | A2 | A3 |  Row 0 of A read for EACH output tile in row 0 of C
    t  +----+----+----+----+
    i  | A4 | A5 | A6 | A7 |  Row 1 of A read for EACH output tile in row 1 of C
    l  +----+----+----+----+
    e  | A8 | A9 |A10 |A11 |  ...
    s  +----+----+----+----+
       |A12 |A13 |A14 |A15 |
       +----+----+----+----+

    For C[0,0]: Read A row 0 (4 tiles) + B col 0 (4 tiles) = 8 DRAM reads
    For C[0,1]: Read A row 0 (4 tiles) + B col 1 (4 tiles) = 8 DRAM reads
    ...
    Total for row 0 of C: 4 outputs x 8 reads = 32 DRAM reads
    But A row 0 could be reused! Wasted: 3 x 4 = 12 redundant reads of A row 0


    OPTIMIZED APPROACH: Group outputs into blocks, reuse data within blocks
    =======================================================================

    Assign a BLOCK of output tiles to each core (e.g., 2x2 tiles per core):

       +----+----+----+----+
       | C0 | C0 | C1 | C1 |   <- Core 0 computes top-left 2x2 block
       +----+----+----+----+      Core 1 computes top-right 2x2 block
       | C0 | C0 | C1 | C1 |
       +----+----+----+----+
       | C2 | C2 | C3 | C3 |   <- Core 2 computes bottom-left 2x2 block
       +----+----+----+----+      Core 3 computes bottom-right 2x2 block
       | C2 | C2 | C3 | C3 |
       +----+----+----+----+

    Now Core 0 can:
    1. Read A rows 0-1 once (a "block" of A)
    2. Read B columns 0-1 once (a "block" of B)
    3. Compute partial products for ALL 4 output tiles simultaneously
    4. Accumulate partial results in L1 (intermediate buffer)
    5. Repeat for next block along K dimension
    6. Write final results only once

This blocking strategy reduces DRAM traffic by reusing input tiles across multiple
output tiles computed by the same core.


Understanding Blocks and Subblocks
==================================

The optimized implementation introduces a hierarchical structure:

**Tiles** (32x32 elements)
    The fundamental unit of computation on Tenstorrent hardware.

**Subblocks** (small groups of tiles, e.g., 2x4 tiles)
    The unit of partial result accumulation. Subblock dimensions are chosen from
    hardware-efficient configurations such as {4,2}, {2,4}, {8,1}, {1,8}, etc.
    These dimensions are constrained by the number of destination registers available.

**Blocks** (groups of subblocks)
    The unit of work assigned to each core. A core processes its entire block of
    output tiles, iterating over "inner blocks" along the K dimension.

**Inner Block Width** (``in0_block_w``)
    The number of tiles processed along K before moving to the next block.
    This determines how many partial results must be accumulated before
    writing final output.

The following table shows an example parameter configuration for a 640x640x640 matmul::

    +------------------------+--------+----------------------------------------------+
    | Parameter              | Value  | Description                                  |
    +========================+========+==============================================+
    | Mt (M in tiles)        | 20     | 640 / 32 = 20 tiles in M dimension           |
    +------------------------+--------+----------------------------------------------+
    | Nt (N in tiles)        | 20     | 640 / 32 = 20 tiles in N dimension           |
    +------------------------+--------+----------------------------------------------+
    | Kt (K in tiles)        | 20     | 640 / 32 = 20 tiles in K dimension           |
    +------------------------+--------+----------------------------------------------+
    | per_core_M             | 4      | Each core handles 4 rows of output tiles     |
    +------------------------+--------+----------------------------------------------+
    | per_core_N             | 4      | Each core handles 4 cols of output tiles     |
    +------------------------+--------+----------------------------------------------+
    | out_subblock_h         | 2      | Subblock height in tiles                     |
    +------------------------+--------+----------------------------------------------+
    | out_subblock_w         | 2      | Subblock width in tiles                      |
    +------------------------+--------+----------------------------------------------+
    | in0_block_w            | 2      | Inner dimension block width in tiles         |
    +------------------------+--------+----------------------------------------------+
    | num_blocks             | 10     | Kt / in0_block_w = 20/2 = 10 blocks along K  |
    +------------------------+--------+----------------------------------------------+

With these parameters:

- Total cores needed: (Mt / per_core_M) x (Nt / per_core_N) = 5 x 5 = 25 cores
- Subblocks per core: (per_core_M / out_subblock_h) x (per_core_N / out_subblock_w) = 2 x 2 = 4
- Output tiles per core: per_core_M x per_core_N = 16 tiles


The Role of the Intermediate Circular Buffer
=============================================

The key to data reuse is keeping partial results in L1 memory instead of writing them
to DRAM after each inner block. This is achieved through an **intermediate circular
buffer** (typically using CB index 24 or ``c_24``).

The compute kernel workflow with data reuse is::

    for block in 0..num_blocks:
        for subblock in all_subblocks:

            if block > 0:
                # Reload partial results from intermediate buffer
                cb_wait_front(c_24, subblock_tiles)
                copy_tile(c_24, i, dst_i) for each tile
                cb_pop_front(c_24, subblock_tiles)

            # Compute: accumulate A_block @ B_block into destination registers
            for h in subblock_height:
                for w in subblock_width:
                    for k in block_width:
                        matmul_tiles(c_0, c_1, a_idx, b_idx, dst_idx)

            if block < num_blocks - 1:
                # Store partial results to intermediate buffer
                cb_reserve_back(c_24, subblock_tiles)
                pack_tile(i, c_24) for each tile
                cb_push_back(c_24, subblock_tiles)
            else:
                # Last block: write final results to output buffer
                cb_reserve_back(c_16, subblock_tiles)
                pack_tile(i, c_16) for each tile
                cb_push_back(c_16, subblock_tiles)

The intermediate buffer and output buffer share the same L1 memory region but use
different indices, allowing efficient toggling between accumulation and output phases.


Implementing get_large_matmul_params Without bmm_op.hpp
=======================================================

The ``get_large_matmul_params`` function determines optimal block and subblock dimensions.
Here is the logic you need to implement:

**Step 1: Find minimum per-core dimensions**

For each dimension (M and N), find prime factors that exceed the number of cores
in that dimension. These factors must be included in ``per_core_M`` or ``per_core_N``
because they cannot be distributed across cores::

    # Example: Mt=20, num_cores_y=8
    # Prime factors of 20: [2, 2, 5]
    # Factor 5 > 8? No. All factors can be distributed.
    # Mpc_min = 1

    # Example: Mt=21, num_cores_y=8
    # Prime factors of 21: [3, 7]
    # Factor 7 > 8? No. Factor 3 > 8? No.
    # Mpc_min = 1

    # Example: Mt=35, num_cores_y=4
    # Prime factors of 35: [5, 7]
    # Factor 5 > 4? Yes! Factor 7 > 4? Yes!
    # Mpc_min = 5 * 7 = 35 (single core handles all M tiles)

**Step 2: Maximize per-core dimensions within L1 constraints**

Given the minimum per-core dimensions, try to increase them while respecting:

- L1 memory constraints: The formula ``(400 - 2 * in0_block_w * dim) / (2 * in0_block_w + dim)``
  gives the maximum allowable size for the other dimension.
- Core count constraints: ``Mt / per_core_M <= num_cores_y`` and ``Nt / per_core_N <= num_cores_x``

**Step 3: Select subblock dimensions**

Iterate through hardware-efficient subblock configurations and find one that divides
evenly into the per-core dimensions::

    SUBBLOCK_CHOICES = [(4,2), (2,4), (8,1), (1,8), (7,1), (1,7),
                        (3,2), (2,3), (6,1), (1,6), (5,1), (1,5),
                        (2,2), (4,1), (1,4), (3,1), (1,3), (2,1), (1,2), (1,1)]

    for (subblock_h, subblock_w) in SUBBLOCK_CHOICES:
        if per_core_M % subblock_h == 0 and per_core_N % subblock_w == 0:
            return (per_core_M, per_core_N, subblock_h, subblock_w)


Step-by-Step Transformation Guide
=================================

The following steps describe how to transform ``matmul_multi_core.cpp`` into
``matmul_multicore_reuse.cpp``. Each step identifies which components change and how.


Step 1: Replace Work Distribution Logic
---------------------------------------

**What changes:** Remove ``split_work_to_cores`` and implement 2D core assignment based
on output blocks.

**In matmul_multi_core.cpp:**

.. code-block:: cpp

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

**In matmul_multicore_reuse.cpp:**

.. code-block:: cpp

    // Get block/subblock parameters
    auto matmul_params = get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    // Calculate core grid dimensions
    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    CoreRangeSet all_cores(num_cores_to_corerangeset(num_blocks_total, compute_grid, true));


Step 2: Calculate Block and Subblock Parameters
------------------------------------------------

**What changes:** Add new parameters for block structure and compute kernel arguments.

**New parameters to calculate:**

.. code-block:: cpp

    uint32_t in0_block_w = 2;  // Inner block width (tunable parameter)
    uint32_t num_blocks = Kt / in0_block_w;  // Number of blocks along K

    // Input block parameters
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;

    // Subblock parameters
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;


Step 3: Resize Circular Buffers
-------------------------------

**What changes:** Increase CB sizes to hold full blocks (not just 2 tiles for double-buffering).

**In matmul_multi_core.cpp:**

.. code-block:: cpp

    uint32_t num_input_tiles = 2;  // Double buffer: 2 tiles per CB
    // CB size = 2 * single_tile_size

**In matmul_multicore_reuse.cpp:**

.. code-block:: cpp

    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // Double buffer full blocks
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;

    uint32_t in1_CB_tiles = in1_block_tiles * 2;
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;

    uint32_t out_CB_tiles = per_core_M * per_core_N;  // Full output block (no double buffer)
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;


Step 4: Add Intermediate Circular Buffer
----------------------------------------

**What changes:** Create an additional CB for storing partial results between blocks.

**New code to add:**

.. code-block:: cpp

    uint32_t output_cb_index = CBIndex::c_16;
    uint32_t interm0_cb_index = 24;  // Intermediate buffer for partial results

    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format},
        {interm0_cb_index, cb_data_format}
    };

    CircularBufferConfig cb_output_config =
        CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, single_tile_size)
            .set_page_size(interm0_cb_index, single_tile_size);

    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);


Step 5: Replace Kernels with Block-Aware Versions
-------------------------------------------------

**What changes:** Use different kernel files that understand block/subblock structure.

+------------------+-----------------------------------------------+-----------------------------------------------+
| Kernel Type      | matmul_multi_core                             | matmul_multicore_reuse                        |
+==================+===============================================+===============================================+
| Reader           | reader_mm_output_tiles_partitioned.cpp        | reader_bmm_tile_layout.cpp                    |
+------------------+-----------------------------------------------+-----------------------------------------------+
| Compute          | mm.cpp                                        | bmm_large_block_zm.cpp                        |
+------------------+-----------------------------------------------+-----------------------------------------------+
| Writer           | writer_unary_interleaved_start_id.cpp         | writer_bmm_tile_layout.cpp                    |
+------------------+-----------------------------------------------+-----------------------------------------------+


Step 6: Update Compute Kernel Compile-Time Arguments
-----------------------------------------------------

**What changes:** Pass block/subblock structure as compile-time arguments.

**In matmul_multi_core.cpp:**

.. code-block:: cpp

    // No compile-time args for compute kernel
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

**In matmul_multicore_reuse.cpp:**

.. code-block:: cpp

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,                 // Inner block width
        in0_num_subblocks,           // Number of subblocks in M dimension
        in0_block_num_tiles,         // Tiles per input block for A
        in0_subblock_num_tiles,      // Tiles per subblock for A

        in1_num_subblocks,           // Number of subblocks in N dimension
        in1_block_num_tiles,         // Tiles per input block for B
        in1_per_core_w,              // Width of B block per core

        num_blocks,                  // Number of blocks along K dimension

        out_subblock_h,              // Output subblock height
        out_subblock_w,              // Output subblock width
        out_subblock_num_tiles,      // Tiles per output subblock
        B                            // Batch dimension
    };

    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args}


Step 7: Restructure Runtime Arguments
-------------------------------------

**What changes:** Replace tile-offset-based arguments with block/stride-based arguments.

**Reader kernel runtime arguments (per core):**

.. code-block:: cpp

    std::vector<uint32_t> mm_reader_args = {
        src0_dram_buffer->address(),         // in0_tensor_addr
        Kt * per_core_M * output_idx_y,      // in0_tensor_start_tile_id
        1,                                   // in0_tensor_stride_w (adjacent tiles)
        Kt,                                  // in0_tensor_stride_h (next row)
        in0_block_w,                         // in0_tensor_next_block_stride

        in0_block_w,                         // in0_block_w
        per_core_M,                          // in0_block_h
        in0_block_w * per_core_M,            // in0_block_num_tiles

        src1_dram_buffer->address(),         // in1_tensor_addr
        per_core_N * output_idx_x,           // in1_tensor_start_tile_id
        1,                                   // in1_tensor_stride_w
        Nt,                                  // in1_tensor_stride_h
        in0_block_w * Nt,                    // in1_tensor_next_block_stride

        per_core_N,                          // in1_block_w
        in0_block_w,                         // in1_block_h
        per_core_N * in0_block_w,            // in1_block_num_tiles

        Kt / in0_block_w,                    // num_blocks

        Mt * Kt,                             // MtKt (for batch offset)
        Kt * Nt,                             // KtNt (for batch offset)
        B,                                   // batch
        bcast_batch                          // broadcast B across batches?
    };

**Writer kernel runtime arguments (per core):**

.. code-block:: cpp

    std::vector<uint32_t> writer_args = {
        dst_dram_buffer->address(),                                    // out_buffer_addr
        output_idx_x * per_core_N + output_idx_y * per_core_M * Nt,    // start_tile_id
        1,                                                             // stride_w
        Nt,                                                            // stride_h
        out_subblock_w,                                                // next_subblock_stride_w
        out_subblock_h * Nt,                                           // next_subblock_stride_h

        out_subblock_w,                                                // subblock_w
        out_subblock_h,                                                // subblock_h
        out_subblock_w * out_subblock_h,                               // tiles per subblock
        per_core_N / out_subblock_w,                                   // num_subblocks_w
        per_core_M / out_subblock_h,                                   // num_subblocks_h

        Mt * Nt,                                                       // MtNt (for batch)
        B                                                              // batch
    };


Step 8: Update Core Iteration Loop
-----------------------------------

**What changes:** Iterate over 2D core indices instead of linear work offsets.

**In matmul_multi_core.cpp:**

.. code-block:: cpp

    uint32_t work_offset = 0;
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Set runtime args using work_offset
                work_offset += work_per_core;
            }
        }
    }

**In matmul_multicore_reuse.cpp:**

.. code-block:: cpp

    uint32_t num_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            // Set runtime args using output_idx_x, output_idx_y
            // (these determine which block of output this core computes)

            num_blocks_read++;
        }
    }


Summary of Key Differences
==========================

+---------------------------+------------------------------------------+------------------------------------------+
| Aspect                    | matmul_multi_core                        | matmul_multicore_reuse                   |
+===========================+==========================================+==========================================+
| Work unit                 | Single output tile                       | Block of output tiles (subblocks)        |
+---------------------------+------------------------------------------+------------------------------------------+
| Work distribution         | split_work_to_cores (1D linear)          | 2D grid based on Mt/per_core_M, Nt/...   |
+---------------------------+------------------------------------------+------------------------------------------+
| CB sizes                  | 2 tiles (double buffer)                  | Full blocks (in0_block_tiles * 2)        |
+---------------------------+------------------------------------------+------------------------------------------+
| Intermediate buffer       | Not used                                 | c_24 for partial result accumulation     |
+---------------------------+------------------------------------------+------------------------------------------+
| Input tile reuse          | None (re-read for each output tile)      | Within-block reuse across subblocks      |
+---------------------------+------------------------------------------+------------------------------------------+
| Compute kernel loop       | Simple: for each tile, for each K        | Nested: blocks -> subblocks -> tiles     |
+---------------------------+------------------------------------------+------------------------------------------+
| Runtime args              | Linear offsets                           | 2D indices with strides                  |
+---------------------------+------------------------------------------+------------------------------------------+
| DRAM traffic              | O(Mt * Nt * Kt) reads per matrix         | Reduced by factor of block size          |
+---------------------------+------------------------------------------+------------------------------------------+
