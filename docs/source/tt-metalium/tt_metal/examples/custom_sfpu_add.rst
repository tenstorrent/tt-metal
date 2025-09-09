.. _Custom_SFPU_Add:

Vector addition using custom SFPU
=================================

The SFPU (Special Function Processing Unit) is a programmable vector engine designed for efficient computation of mathematical operations. Many functions in the compute API library, such as ``sin``, ``cos``, ``exp``, ``relu``, and ``tanh``, are implemented using the SFPU. By programming the SFPU directly, users can implement custom mathematical functions that are not available in the standard library, which is useful for specialized HPC workloads.

This example demonstrates how to program the vector engine to perform vector addition. This example serves as a starting point for users looking to implement custom SFPU operations.

We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/custom_sfpu_kernel_add`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_custom_sfpu_kernel_add`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/metal_example_custom_sfpu_kernel_add

Program setup
-------------

The host-side code sets up the device, buffers, and kernels. It's similar to other examples, but we'll highlight the key parts for this custom SFPU example.

First, we initialize the device and create DRAM buffers for two inputs (``src0``, ``src1``) and one output (``dst``).

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/custom_sfpu_kernel_add.cpp
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t n_tiles = 64;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    InterleavedBufferConfig config{
        .device = device,
        .size = n_tiles * tile_size_bytes,
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto src0_dram_buffer = CreateBuffer(config);
    auto src1_dram_buffer = CreateBuffer(config);
    auto dst_dram_buffer = CreateBuffer(config);

Next, we create circular buffers (CBs) for communication between the data movement and compute kernels on the Tensix core. We'll use two CBs for input (``cb_in0``, ``cb_in1``) and one for output (``cb_out0``).

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/custom_sfpu_kernel_add.cpp
    constexpr uint32_t tiles_per_cb = 2;
    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    CreateCircularBuffer(program, core, CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, tile_size_bytes));

    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    CreateCircularBuffer(program, core, CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, tile_size_bytes));

    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    CreateCircularBuffer(program, core, CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, tile_size_bytes));

Finally, we create the kernels. This example uses a reader, a writer, and a compute kernel.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/custom_sfpu_kernel_add.cpp
    auto reader = CreateKernel(
        program,
        "custom_sfpu_kernel_add/kernels/dataflow/read_tiles.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, ...});

    auto writer = CreateKernel(
        program,
        "custom_sfpu_kernel_add/kernels/dataflow/write_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, ...});

    auto compute = CreateKernel(
        program,
        "custom_sfpu_kernel_add/kernels/compute/tiles_add.cpp",
        core,
        ComputeConfig{});

The Kernels
-----------

Data Movement Kernels
~~~~~~~~~~~~~~~~~~~~~

The reader kernel reads tiles from two source DRAM buffers and pushes them into two separate input circular buffers.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        // ...
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(i, src0_dram_addr, l1_buffer_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);

            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(i, src1_dram_addr, l1_buffer_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }

The writer kernel is straightforward: it reads result tiles from the output circular buffer and writes them to the destination DRAM buffer.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        // ...
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            noc_async_write_tile(i, l1_buffer_addr, dst_dram_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }
    }

SFPU Compute Kernel
~~~~~~~~~~~~~~~~~~~

The compute kernel is where the custom SFPU logic resides. It waits for tiles from the input CBs, performs the addition using the SFPU, and pushes the result to the output CB.

The overall flow is:

1. Wait for input tiles to be available in ``cb_in0`` and ``cb_in1``.
2. Acquire destination registers. These registers will be used as a scratchpad for the computation.
3. Copy tiles from CBs to the destination registers.
4. Execute the custom SFPU addition function on the data in the destination registers.
5. Transfer the ownership of the destination registers to the packer
6. Reserve space in the output CB, pack the result tile, and push it.
7. Pop the input tiles from the input CBs.
8. Release the destination registers.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/kernels/compute/tiles_add.cpp
    namespace NAMESPACE {
    void MAIN {
        uint32_t n_tiles = get_arg_val<uint32_t>(0);

        constexpr auto cb_in0 = tt::CBIndex::c_0;
        constexpr auto cb_in1 = tt::CBIndex::c_1;
        constexpr auto cb_out0 = tt::CBIndex::c_16;

        init_sfpu(cb_in0, cb_out0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            tile_regs_acquire();
            copy_tile(cb_in0, 0, 0);
            copy_tile(cb_in1, 0, 1);

            my_add_tiles(0, 1, 0);

            tile_regs_commit();

            cb_reserve_back(cb_out0, 1);
            pack_tile(0, cb_out0);
            cb_push_back(cb_out0, 1);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
            tile_regs_release();
        }
    }
    } // namespace NAMESPACE

Custom SFPU Implementation
--------------------------

The core of this example is the custom SFPU function ``my_add_tiles``. It's implemented in a layered way, which is a common pattern for SFPU programming.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/kernels/compute/tiles_add.cpp
    #ifdef TRISC_MATH

    // Low-level function operating on a tile face
    void add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
        constexpr uint32_t n_vector_in_face = 32;

        // Calculate base indices for each tile in the Dst register array.
        // Each tile occupies 32 consecutive Dst registers (n_vector_in_face) in WH and BH
        // For example: tile 0 uses dst_reg[0-31], tile 1 uses dst_reg[32-63], etc.
        const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_face;
        const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_face;
        const uint32_t out_base_idx = dst_index_out * n_vector_in_face;

        // Process one face of the tile (8 SIMD operations covering 256 elements).
        // Each iteration processes 32 elements, so 8 iterations = 256 elements = one 16x16 face.
        for (size_t i = 0; i < 8; i++) {
            vFloat a = dst_reg[in0_base_idx + i];
            vFloat b = dst_reg[in1_base_idx + i];
            dst_reg[out_base_idx + i] = a + b;
        }
    }

    // LLK wrapper
    inline void my_add_tile_internal(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
        _llk_math_eltwise_binary_sfpu_params_<false>(add_tile_face, idx_dst0, idx_dst1, idx_dst0);
    }

    #endif // TRISC_MATH

    // High-level API function
    inline void my_add_tiles(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
        MATH(my_add_tile_internal(idx_dst0, idx_dst1, idx_out0));
    }


Here's a breakdown of the layers. Note that ``add_tile_face`` and ``my_add_tile_internal`` must be inside a ``#ifdef TRISC_MATH`` block, as they contain code that is specific to the math thread and will not compile for other RISC-V cores.

1.  **`my_add_tiles`**: This is the high-level, user-facing function that the main compute kernel calls. It wraps the internal function with the ``MATH()`` macro, which ensures the code is only compiled and executed on the math thread of the Tensix core.

2.  **`my_add_tile_internal`**: This function acts as a wrapper. ``_llk_math_eltwise_binary_sfpu_params_`` is an internal API of the Metalium kernel libraries. This helper automatically handles setting up SFPU for operation, iterating over all the faces of a tile, calling our ``add_tile_face`` function for each one then teardown the operation in prepration for the next one. This abstracts away the complexity of manual setup and state managment.

3.  **`add_tile_face`**: This is the lowest-level function and where the actual computation happens. It operates on a single *face* of a tile. A 32x32 tile is composed of four 16x16 faces. The SFPU processes data one face at a time. This function loads SIMD vectors (``vFloat``) from the destination registers, performs the addition, and stores the result back. The ``dst_reg`` is an array representing the SFPU's view of the destination registers.

    The function calculates base indices (``in0_base_idx``, ``in1_base_idx``, ``out_base_idx``) to map logical tile indices to physical SFPU register addresses. Since each tile occupies 32 consecutive registers, these base indices are computed by multiplying the tile index by 32. For example, if we're processing tiles at indices 0, 1, and 0 (for input0, input1, and output respectively), the base indices would be 0, 32, and 0, meaning the first input tile starts at ``dst_reg[0]``, the second input tile starts at ``dst_reg[32]``, and the output overwrites the first input tile starting at ``dst_reg[0]``.

This layered approach separates the high-level logic from the low-level, hardware-specific details, making the code cleaner and more maintainable.

.. warning::

    ``_llk_math_eltwise_binary_sfpu_params_`` and similar LLK helpers are internal APIs and may change in future releases. Tenstorrent does not guarantee backward compatibility for these internal functions. Users must keep them up to date with the latest Metalium releases.

Runtime Arguments and Execution
-------------------------------

Back on the host, we set the runtime arguments for the kernels. The reader and writer kernels need the DRAM buffer addresses, and all three kernels need to know the number of tiles to process.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/custom_sfpu_kernel_add.cpp
    SetRuntimeArgs(program, reader, core, {
        src0_dram_buffer->address(),
        src1_dram_buffer->address(),
        n_tiles
    });

    SetRuntimeArgs(program, writer, core, {
        dst_dram_buffer->address(),
        n_tiles
    });

    SetRuntimeArgs(program, compute, core, {
        n_tiles
    });

Finally, we enqueue the program for execution and read back the results from the destination DRAM buffer to verify correctness.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpu_kernel_add/custom_sfpu_kernel_add.cpp
    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    // Validation against golden output...

Conclusion
----------

This example demonstrated how to create a custom SFPU kernel for vector addition. Key takeaways include:

*   The layered approach to SFPU kernel development (high-level API, LLK wrapper, low-level face function).
*   The use of destination registers (``dst_reg``) for SFPU computations.
*   The role of the LLK API (e.g., ``_llk_math_eltwise_binary_sfpu_params_``) in simplifying SFPU programming by handling tile face iteration.
*   The standard pipeline of reader, compute, and writer kernels for processing data on Tensix cores.

By following this pattern, you can implement a wide variety of custom element-wise operations on the SFPU to accelerate your specific workloads.
