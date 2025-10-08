.. _Custom_SFPI_Add:

Vector addition using SFPI
==========================

This example demonstrates how to program the vector engine to perform the simplest operation - vector addition. This example serves as a starting point for users looking to implement custom operations using SFPI. We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/custom_sfpi_add`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_custom_sfpi_add`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/metal_example_custom_sfpi_add

.. warning::

    Tenstorrent does not guarantee backward compatibility for user-implemented SFPI functions. Keep your implementations up to date with the latest Metalium releases. APIs that call low-level SFPI functions may change without notice, and SFPI specifications may also change in future hardware versions.

Program setup
-------------

This example assumes familiarity with basic Metalium concepts like device initialization, buffer creation, circular buffers, and kernel setup. If you're new to these concepts, we recommend starting with the :ref:`Eltwise sfpu example<Eltwise sfpu example>` for a gentler introduction to programming Metalium kernels.

The host-side setup for this custom SFPI example follows the standard pattern: device initialization, DRAM buffer creation for two inputs and one output, circular buffer allocation for kernel communication, and kernel creation. The key difference from simpler examples is that we need two input circular buffers (``cb_in0``, ``cb_in1``) to handle the binary operation, plus the standard output buffer (``cb_out0``).

.. code-block:: cpp

    // Standard device and program setup
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // Submit work via the mesh command queue: uploads/downloads and program execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Allocate mesh buffers: two inputs + one output (replicated across mesh)
    auto src0_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    Program program = CreateProgram();

    // Create mesh workload for program execution across the mesh
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Circular buffers for kernel communication
    CreateCircularBuffer(program, core, /* cb_in0 config */);
    CreateCircularBuffer(program, core, /* cb_in1 config */);
    CreateCircularBuffer(program, core, /* cb_out0 config */);

    // Kernels: reader, writer, and custom SFPU compute
    auto reader = CreateKernel(program, "..../read_tiles.cpp", core, DataMovementConfig{...});
    auto writer = CreateKernel(program, "..../write_tile.cpp", core, DataMovementConfig{...});
    auto compute = CreateKernel(program, "..../tiles_add.cpp", core, ComputeConfig{});

The Kernels
-----------

Data Movement Kernels
~~~~~~~~~~~~~~~~~~~~~

The reader kernel reads tiles from two source DRAM buffers and pushes them into two separate input circular buffers.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        // ...
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_reserve_back(cb_in0, 1);
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);
            noc_async_read_tile(i, in0, cb_in0_addr);
            noc_async_read_tile(i, in1, cb_in1_addr);

            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }

The writer kernel is straightforward: it reads result tiles from the output circular buffer and writes them to the destination DRAM buffer.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        // ...
        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            uint32_t cb_out0_addr = get_read_ptr(cb_out0);
            noc_async_write_tile(i, out0, cb_out0_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }
    }

SFPI Compute Kernel
~~~~~~~~~~~~~~~~~~~

The compute kernel is where the custom SFPI logic resides. It waits for tiles from the input CBs, performs the addition using the SFPI, and pushes the result to the output CB.

The overall flow follows the same pattern as other compute kernels:

1. Wait for input tiles to be available in ``cb_in0`` and ``cb_in1``.
2. Acquire destination registers. These registers will be used as a scratchpad for the computation.
3. Copy tiles from CBs to the destination registers.
4. Execute the custom SFPI addition function on the data in the destination registers.
5. Transfer the ownership of the destination registers to the packer
6. Reserve space in the output CB, pack the result tile, and push it.
7. Pop the input tiles from the input CBs.
8. Release the destination registers.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp
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

            my_add_tiles(0, 1, 0); // <-- Call to custom SFPI addition function

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

Custom SFPI Implementation
--------------------------

The core of this example is the custom SFPI function ``my_add_tiles``. It's implemented in a layered way, which is a common pattern for SFPI programming to enable easy consumption and maintainability.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp
    #ifdef TRISC_MATH

    // Low-level function operating on a tile face
    void my_add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
        constexpr uint32_t n_vector_in_tile = 32;

        // Calculate base indices for each tile in the Dst register array.
        // Each tile occupies 32 consecutive Dst registers (n_vector_in_tile) in WH and BH
        // For example: tile 0 uses dst_reg[0-31], tile 1 uses dst_reg[32-63], etc.
        const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
        const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
        const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

        // Process one face of the tile (8 SIMD operations covering 256 elements).
        // Each iteration processes 32 elements, so 8 iterations = 256 elements = one 16x16 face.
        for (size_t i = 0; i < 8; i++) {
            vFloat a = dst_reg[in0_base_idx + i];
            vFloat b = dst_reg[in1_base_idx + i];
            dst_reg[out_base_idx + i] = a + b;
        }
    }
    #endif // TRISC_MATH

    // High-level API function
    void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
        MATH(_llk_math_eltwise_binary_sfpu_params_<false>(add_tile_face, idx_dst0, idx_dst1, idx_out0));
    }


Here's a breakdown of the layers. The ``add_tile_face`` must be inside a ``#ifdef TRISC_MATH`` block, since they use math-thread-specific code that will not compile for other RISC-V cores.

1.  **`my_add_tiles`**: This is the main function called by the compute kernel. It wraps the internal function with the ``MATH()`` macro, which ensures the code only runs on the math thread of the Tensix core.  ``_llk_math_eltwise_binary_sfpu_params_`` is an internal helper that sets up the SFPU, iterates over all faces of a tile, calls ``add_tile_face`` for each face, and then cleans up. This avoids manual setup and state management.

2.  **`add_tile_face`**: This is the most basic function, performing the actual addition on a single tile face. A 32x32 tile is divided into four 16x16 faces, and this function is called for each face. It uses the ``dst_reg`` array, which represents the SFPU's destination registers. The number of available ``dst_reg`` registers can be found in the :ref:`Compute Engines and Data Flow within Tensix<compute_engines_and_dataflow_within_tensix>` documentation.

    The function calculates base indices (``in0_base_idx``, ``in1_base_idx``, ``out_base_idx``) to map tile indices to register addresses within ``dst_reg``. Each tile occupies 32 registers; the base index is calculated by multiplying the tile index by 32 (refer to :ref:`Internal structure of a Tile<internal_structure_of_a_tile>` for more information on tile structure). For example, processing tiles at indices 0, 1, and 0 results in base indices of 0, 32, and 0, respectively. This means the first input tile starts at ``dst_reg[0]``, the second at ``dst_reg[32]``, and the output overwrites the first input tile at ``dst_reg[0]``.

    Within each face, the function loads SIMD vectors (``vFloat``) from the input registers, adds them, and writes the result back to the output registers.

    Each time the SFPI function is called, the helper automatically offsets ``dst_reg`` to point to the start of the current face. So, on the first call, ``dst_reg`` has an offset of 0; on the second, the offset is 8, and so on. The programmer does not need to manage this offset manually.

This layered structure keeps high-level logic separate from hardware-specific details, making the code easier to read and maintain.

.. warning::

    The value of ``n_vector_in_face`` is architecture dependent. The example above assumes a Tensix architecture where each vector is 32 wide. Which is true for currently shipping Tensix Processors (Wormhole and Blackhole). But may change in future versions. Users should verify this value against their target architecture specifications when adapting this example.

.. note::

    There are 3 internal APIs to invoke custom SFPI functions, depending on the number of input tiles. Please view the header file for the most up-to-date information.

    *  ``_llk_math_eltwise_unary_sfpu_params_``: For functions with one input tile (e.g., ``sin``, ``exp``).
    *  ``_llk_math_eltwise_binary_sfpu_params_``: For functions with two input tiles (e.g., ``add``, ``sub``, ``mul``, ``div``).
    *  ``_llk_math_eltwise_ternary_sfpu_params_``: For functions with three input tiles (e.g., ``where``).

.. warning::

    ``_llk_math_eltwise_binary_sfpu_params_`` and similar LLK helpers are internal APIs and may change in future releases. Tenstorrent does not guarantee backward compatibility for these internal functions. Users should keep their use up to date with the latest Metalium releases.

Runtime Arguments and Execution
-------------------------------

Back on the host, we set the runtime arguments for the kernels. The reader and writer kernels need the DRAM buffer addresses, and all three kernels need to know the number of tiles to process.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp
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

For mesh execution, we add the program to a mesh workload and enqueue it for execution across the mesh. We also upload input data using the mesh buffer API.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp

    // Upload input data to mesh buffers (non-blocking)
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_data, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b_data, /*blocking=*/false);

    // Add program to mesh workload and execute
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

Finally, we read back the results from the mesh buffer using the distributed API to verify correctness.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    // Validation against golden output...

Conclusion
----------

This example demonstrated how to create a custom SFPI kernel for vector addition using the Mesh API. Key takeaways include:

*   The layered approach to SFPI kernel development (high-level API, LLK wrapper, low-level face function).
*   The use of destination registers (``dst_reg``) for SFPU computations.
*   The role of the LLK API (e.g., ``_llk_math_eltwise_binary_sfpu_params_``) in simplifying SFPI programming by handling tile face iteration.
*   The standard pipeline of reader, compute, and writer kernels for processing data on Tensix cores.

By following this pattern, you can implement a wide variety of custom element-wise operations on the SFPU to accelerate your specific workloads while leveraging the distributed programming capabilities of the Mesh API.
