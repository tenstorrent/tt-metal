
1. Motivation
=============

The multi-core example in ``tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.cpp`` parallelizes matrix multiplication by assigning **flat ranges of output tiles** to cores using ``split_work_to_cores``. Each core then uses a simple matmul kernel that, for every output tile, streams all ``Kt`` contributing tiles of ``A`` and ``B`` from DRAM through small circular buffers and immediately writes the result back to DRAM. This is easy to understand, but it is suboptimal in two key ways: (1) **A and B tiles are repeatedly loaded from DRAM** for different output tiles even when they could have been reused on core, and (2) there is **no explicit control over output block shape per core**, only over the number of tiles per core.

A more efficient design is to introduce a **block matrix multiply (BMM) layout** with explicit **per-core blocks** and **subblocks**, and to process the reduction dimension ``K`` in **blocks of tiles**. Each core owns a rectangular block of output tiles of shape ``per_core_M x per_core_N`` (in tiles), this block is subdivided into smaller subblocks of size ``out_subblock_h x out_subblock_w``, and the kernel iterates over ``K`` in chunks of width ``in0_block_w`` tiles. For each K-chunk, it reads in blocks of ``A`` and ``B`` tiles, accumulates partial products into an **intermediate circular buffer** in L1, and only writes fully accumulated results to the final output buffer. The high-level objective is to reduce DRAM bandwidth by **reusing both input tiles and partial results** while still fitting comfortably within the core’s L1 memory.


2. Examples and geometric intuition
===================================

Start from the tilized view of the matrices as in the basic example: ``A`` has shape ``Mt x Kt`` in tiles, ``B`` has shape ``Kt x Nt``, and ``C`` has shape ``Mt x Nt`` where ``Mt = M / TILE_HEIGHT``, ``Kt = K / TILE_WIDTH``, and ``Nt = N / TILE_WIDTH``. In the basic multi-core version, you conceptually flatten the output tile grid into a 1D range of size ``Mt * Nt`` and hand out contiguous segments of that range to cores. In a blocked design, you instead tile ``C`` hierarchically:

.. code-block:: text

   Global tiles:  Mt x Nt
       = (num_blocks_y x per_core_M)  x  (num_blocks_x x per_core_N)

where each core gets one block of size ``per_core_M x per_core_N`` tiles. Within each per-core block, you then choose a subblocking:

.. code-block:: text

   per-core block: per_core_M x per_core_N
       = (in0_num_subblocks_h x out_subblock_h)  x  (in1_num_subblocks_w x out_subblock_w)

so that the compute kernel can loop over subblocks in a regular way.

It is useful to keep a small table of the main geometric quantities you will manipulate:

.. table:: Geometric quantities in a reuse-oriented multi-core matmul

   +----------------------+-----------------------------------------------------------+
   | Name                 | Meaning                                                  |
   +======================+===========================================================+
   | Mt, Kt, Nt          | Global matrix dimensions in tiles                         |
   +----------------------+-----------------------------------------------------------+
   | per_core_M          | Output tiles per core along M                             |
   +----------------------+-----------------------------------------------------------+
   | per_core_N          | Output tiles per core along N                             |
   +----------------------+-----------------------------------------------------------+
   | out_subblock_h      | Subblock height (divides per_core_M)                      |
   +----------------------+-----------------------------------------------------------+
   | out_subblock_w      | Subblock width (divides per_core_N)                       |
   +----------------------+-----------------------------------------------------------+
   | in0_block_w         | K-tiles per K-block (how much reduction you do at once)   |
   +----------------------+-----------------------------------------------------------+
   | in0_block_tiles     | per_core_M * in0_block_w  (A-tiles per core per K-block)  |
   +----------------------+-----------------------------------------------------------+
   | in1_block_tiles     | per_core_N * in0_block_w  (B-tiles per core per K-block)  |
   +----------------------+-----------------------------------------------------------+
   | out_subblock_tiles  | out_subblock_h * out_subblock_w (tiles per subblock)      |
   +----------------------+-----------------------------------------------------------+

The logic in ``bmm_op.hpp`` (for example, ``bmm_op_utils::get_large_matmul_params`` and the ``SUBBLOCK_HW_CHOICES`` table) is essentially a **search over factorizations** of ``Mt`` and ``Nt`` and a small catalogue of candidate subblock sizes to find values of ``per_core_M``, ``per_core_N``, ``out_subblock_h``, and ``out_subblock_w`` that fit both the core grid and a simple L1 memory model. Recreating this logic yourself (prime factorization, generating possible products, and enforcing a "max tiles per block" constraint) is the key to arriving at a robust blocked layout without including the header.


3. Step-by-step construction of a reuse-style multi-core matmul
================================================================

First, **extend the host-side geometry and planner**. After computing ``Mt``, ``Kt``, and ``Nt`` from ``M``, ``K``, and ``N``, query the compute grid to get ``num_cores_x`` and ``num_cores_y`` from the device or mesh device. Write your own versions of the helpers from ``bmm_op.hpp``:

* ``get_prime_factors(uint32_t n)`` and ``get_possible_products(factors)`` compute prime factorizations and all distinct products of subsets of factors.

* ``get_maximum_block_dim(block_dim, in0_block_w)`` encodes a simple L1 budget: for a chosen per-core dimension in one direction and a fixed ``in0_block_w``, it returns the maximum allowed per-core dimension in the other direction so that the total tiles in input and output blocks stay below some approximate threshold.

* A small catalogue ``SUBBLOCK_HW_CHOICES`` of candidate ``(out_subblock_h, out_subblock_w)`` pairs, for example the 20-element list in the header.

* ``get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w)`` assembles these ideas: it trims away factors of ``Mt`` and ``Nt`` that cannot be distributed across the available cores, enumerates candidate ``per_core_M`` and ``per_core_N`` values bounded by ``get_maximum_block_dim``, and for each candidate checks whether some ``(out_subblock_h, out_subblock_w)`` in the catalogue evenly divides both per-core dimensions while satisfying ``Mt / per_core_M <= num_cores_y`` and ``Nt / per_core_N <= num_cores_x``.

Call this function in your host matmul and assert that the returned per-core dimensions are nonzero and divide ``Mt`` and ``Nt``.

Second, **construct a block-level multi-core mapping**. Instead of calling ``split_work_to_cores`` on the flat range of output tiles, compute

* ``num_blocks_y = Mt / per_core_M``,
* ``num_blocks_x = Nt / per_core_N``,
* ``num_blocks_total = num_blocks_y * num_blocks_x``,

assert ``num_blocks_total <= num_cores_x * num_cores_y``, then create a ``CoreRangeSet`` over the first ``num_blocks_total`` cores of the grid. When setting runtime arguments, iterate over ``output_idx_y`` and ``output_idx_x`` in nested loops and use a linear counter to map each block to a core:

.. code-block:: c++

   uint32_t num_blocks_read = 0;
   for (int output_idx_y = 0; output_idx_y < num_blocks_y; ++output_idx_y) {
       for (int output_idx_x = 0; output_idx_x < num_blocks_x; ++output_idx_x) {
           int core_idx_x = num_blocks_read % num_cores_x;
           int core_idx_y = num_blocks_read / num_cores_x;
           CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
           // ... SetRuntimeArgs(...) ...
           ++num_blocks_read;
       }
   }

Third, **size the circular buffers, including the intermediate buffer**. Use the per-core block shape and ``in0_block_w`` to define:

* ``in0_block_tiles = per_core_M * in0_block_w`` and ``in0_CB_tiles = 2 * in0_block_tiles`` (double-buffered), so ``in0_CB_size = in0_CB_tiles * single_tile_size`` for CB ``CBIndex::c_0``;

* ``in1_block_tiles = per_core_N * in0_block_w`` and ``in1_CB_tiles = 2 * in1_block_tiles`` for CB ``CBIndex::c_1``; and

* an output/partial-results buffer of size ``out_CB_size`` that simultaneously backs CB ``CBIndex::c_16`` (final output tiles) and an intermediate CB index (for example 24) used to store partial sums between K-blocks.

Fourth, **choose and configure the kernels**. On the host:

* Keep the general pattern of three kernel types (reader, compute, writer), but use the BMM tile-layout kernels:

  * Reader: ``matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp``
  * Writer: ``matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp``
  * Compute: ``matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp``

* Build compute kernel compile-time arguments that describe the K-blocking and subblocking:

  * ``num_blocks = Kt / in0_block_w``,
  * ``in0_num_subblocks = per_core_M / out_subblock_h``,
  * ``in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks``,
  * ``in0_subblock_num_tiles = out_subblock_h * in0_block_w``,
  * ``in1_num_subblocks = per_core_N / out_subblock_w``,
  * ``in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks``,
  * ``in1_per_core_w = out_subblock_w * in1_num_subblocks``,
  * ``out_subblock_num_tiles = out_subblock_h * out_subblock_w``, plus batch size ``B`` if needed.

* For each per-core output block, set reader runtime arguments that tell the reader which tiles of ``A`` and ``B`` to stream into the CBs (start tile, strides, block sizes, number of K-blocks) and writer runtime arguments that describe where each subblock of the per-core block should land in the output tensor (start tile id, tensor strides, subblock strides and counts, and global strides like ``MtNt`` and ``B``).

On the device, ``bmm_large_block_zm.cpp`` is structured to:

* for each K-block, optionally reload intermediate partials from CB index 24, run ``matmul_tiles`` over the appropriate input tile indices for each output subblock, and either store new partials back to CB 24 or, on the final K-block, pack the fully accumulated result tiles into the regular output CB;

* use the compile-time arguments above to compute indices into the input and intermediate CBs so that everything lines up with the host-side block and subblock layout.


4. From basic multi-core to a reuse-optimized multi-core matmul
================================================================

To adapt the program in ``matmul_multi_core/matmul_multi_core.cpp`` to this reuse-oriented design (without using ``bmm_op.hpp`` directly), it is helpful to think in terms of a sequence of targeted edits:

* **Keep the overall flow** identical: CPU golden matmul for validation, tilization of A and B, DRAM buffer allocation, circular buffer allocation, kernel creation, runtime argument setup, workload enqueueing, untilization of C, and a PCC check against the golden output.

* **Replace the work-splitting strategy**: remove the call to ``split_work_to_cores`` and the flat ``work_offset`` scheme used to assign contiguous output tile ranges to cores, and instead compute per-core blocks and block counts (``per_core_M``, ``per_core_N``, ``num_blocks_y``, ``num_blocks_x``) using your own reimplementation of the helpers from ``bmm_op.hpp``. Use nested loops over block indices to assign blocks to cores and to set reader/writer runtime arguments in terms of block geometry rather than simple counts of tiles per core.

* **Change the circular buffers and kernels**: instead of small 2-tile CBs and the simple multi-core reader/writer kernels, size your CBs according to per-core blocks and K-block width, introduce an intermediate CB index for partial results, and use the BMM tile-layout reader, writer, and large-block compute kernels. Their compile-time and runtime arguments must be constructed from the same geometric quantities that your planner produced.

* **Extend the compute-kernel contract**: instead of a compute kernel that just knows "number of output tiles" and ``Kt``, switch to a compute kernel that understands blocks and subblocks via a richer set of compile-time arguments. On the host side, construct those args (block widths, numbers of subblocks, tiles per block and subblock, numbers of K-blocks, and batch size) from the layout returned by your planner; on the device side, decode them and use them to compute CB indices and tile indices for ``matmul_tiles`` and for reading/writing partial results in the intermediate CB.

Working through these steps while deliberately reimplementing just the small subset of logic from ``bmm_op.hpp`` forces you to internalize how TT-Metalium expresses complex matmul dataflows in terms of blocks, subblocks, circular buffers, and kernel arguments. When you reach the point where your program uses a block layout and an intermediate CB to reuse partial results across K-blocks, you will have arrived at essentially the same optimization strategy as the "multi-core reuse" design, but with all of the planning logic under your direct control.
