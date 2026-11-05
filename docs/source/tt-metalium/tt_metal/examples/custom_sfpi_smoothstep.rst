.. _Custom_SFPI_Smoothstep:

Smoothstep using SFPI
=====================

This document details the implementation of a custom SFPI kernel for the ``smoothstep`` function. It is intended for developers familiar with parallel programming concepts who are new to the Tenstorrent platform.

This example builds upon the :ref:`Vector addition using SFPI<Custom_SFPI_Add>` example, and introduces the following advanced SFPI concepts:

*   **Parameter Passing:** Passing scalar arguments to an SFPI kernel.
*   **Vector Predicates:** Performing element-wise conditional operations.

The ``smoothstep`` function is a non-linear interpolation function commonly used in graphics (see `GLSL smoothstep documentation <https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml>`_ for reference). It is defined as:

.. math::

    \operatorname{smoothstep}(e_0, e_1, x) =
    \begin{cases}
    0, & x \leq e_0, \\
    1, & x \geq e_1, \\
    \left( \dfrac{x - e_0}{e_1 - e_0} \right)^2 \bigl(3 - 2 \tfrac{x - e_0}{e_1 - e_0}\bigr),
    & e_0 < x < e_1 .
    \end{cases}

Although ``smoothstep`` is conceptually simple, its implementation is complex enough to demonstrate several advanced features of SFPI, such as parameter passing and vector predicates.

The full source code is available in the ``tt_metal/programming_examples/custom_sfpi_smoothstep`` directory.

Building and Running
--------------------

Build the example using the ``--build-programming-examples`` flag:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples

Run the example:

.. code-block:: bash

    ./build/programming_examples/metal_example_custom_sfpi_smoothstep

.. warning::

    Tenstorrent does not guarantee backward compatibility for user-implemented SFPI functions.

Program setup
-------------

The host-side setup for this custom SFPI example follows the standard pattern: mesh device initialization, DRAM tensor allocation, dataflow buffer descriptions for kernel communication, and kernel descriptions (``KernelSpec``) in a ``ProgramSpec``. Unlike binary operations that require two input tensors, smoothstep is a unary operation requiring only a single input tensor (``src0_dram_tensor``) plus the output tensor (``dst_dram_tensor``). Correspondingly, we need only one input dataflow buffer (``src0``, bound as ``dfb::in0`` in the kernels) and one output dataflow buffer (``dst``, bound as ``dfb::out0``).

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep.cpp
    // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // Submit work via the mesh command queue: uploads/downloads and program execution.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
    constexpr experimental::NodeCoord core = {0, 0};

    // ...

    // Configure mesh tensors. Use single-tile page size so transfers operate tile-by-tile: each row of a
    // row-major tensor is one page (number of bytes when round-robin between banks), so a row of one tile's
    // worth of elements gives a page size of one tile. The tensors live in DRAM.
    TensorSpec dram_tensor_spec(
        Shape({n_tiles, elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    // Allocate the tensors (replicated across mesh; on unit mesh ⇒ single device allocation)
    // src0 is input tensor; dst is output tensor
    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // ...

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName dst_dfb{"dst"};
    const experimental::TensorParamName src0_tensor_param{"src0"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // Dataflow buffers for kernel communication
    constexpr uint32_t tiles_per_cb = 2;
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = tile_size_bytes,                       // Since we will be sending one tile at a time, we set
                                                             // the entry size to the tile size
        .num_entries = tiles_per_cb,                         // The number of entries in the dataflow buffer
        .data_format_metadata = tt::DataFormat::Float16_b,   // The data format it'll hold
    };
    experimental::DataflowBufferSpec dst_dfb_spec{
        .unique_id = dst_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Kernels: reader, writer, and custom SFPU compute
    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0")},
        .tensor_bindings = {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {experimental::ConsumerOf(src0_dfb, "in0"), experimental::ProducerOf(dst_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = false, // We don't need the destination accumulator to be FP32 as input and output are BFP16
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "custom_sfpi_smoothstep",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units = {{.name = "custom_sfpi_smoothstep", .kernels = {reader, writer, compute}, .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

The Kernels
-----------

Data Movement Kernels
~~~~~~~~~~~~~~~~~~~~~

The reader kernel reads tiles from a single source DRAM tensor and pushes them into the input dataflow buffer. Since smoothstep is a unary operation, we only need to read from one source tensor. The kernel gets the tile count as the named runtime argument ``args::n_tiles`` instead of by index.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);
        DataflowBuffer dfb_in0(dfb::in0);
        ...
        const auto in0 = TensorAccessor(tensor::in0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.reserve_back(1);
            uint32_t dfb_in0_addr = dfb_in0.get_write_ptr();
            noc_async_read_page(i, in0, dfb_in0_addr);
            noc_async_read_barrier();
            dfb_in0.push_back(1);
        }
    }

The writer kernel is straightforward: it reads result tiles from the output dataflow buffer and writes them to the destination DRAM tensor.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);
        DataflowBuffer dfb_out0(dfb::out0);
        const auto out0 = TensorAccessor(tensor::out0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_out0.wait_front(1);
            uint32_t dfb_out0_addr = dfb_out0.get_read_ptr();
            noc_async_write_page(i, out0, dfb_out0_addr);
            noc_async_write_barrier();
            dfb_out0.pop_front(1);
        }
    }

SFPI Compute Kernel
~~~~~~~~~~~~~~~~~~~

The compute kernel is where the custom SFPI logic resides. It waits for tiles from the input dataflow buffer, performs the smoothstep operation using the SFPI, and pushes the result to the output dataflow buffer.

The overall flow follows the standard pattern for unary compute kernels:

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        // Input dataflow buffer for tiles
        DataflowBuffer dfb_in0(dfb::in0);
        // Output dataflow buffer for results
        DataflowBuffer dfb_out0(dfb::out0);

        // Initialize SFPU for computation using dfb_in0 and dfb_out0
        compute_kernel_hw_startup(dfb::in0, dfb::out0);
        copy_init(dfb::in0);

        // precompute inverse of (edge1 - edge0) for efficiency
        constexpr float edge0 = 0.0f;
        constexpr float edge1 = 1.0f;
        float inv_delta = 1.0f / (edge1 - edge0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.wait_front(1);
            tile_regs_acquire();
            copy_tile(dfb::in0, 0, 0);  // input x
            my_smoothstep_tiles(0, edge0, edge1, inv_delta);  // <-- Custom SFPU smoothstep
            tile_regs_commit();
            tile_regs_wait();
            dfb_out0.reserve_back(1);
            pack_tile(0, dfb::out0);
            dfb_out0.push_back(1);
            dfb_in0.pop_front(1);
            tile_regs_release();
        }
    }

Custom SFPI Implementation of Smoothstep
----------------------------------------

The ``my_smoothstep_tiles`` function uses the layered abstraction pattern shown in previous examples. This section focuses on the new concepts introduced in this kernel.

.. code-block:: cpp

    // tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_smoothstep.h

    #ifdef TRISC_MATH
    namespace ckernel::sfpu {

    // Low-level function operating on a tile face
    void smoothstep_tile_face(float edge0, [[maybe_unused]] float edge1, float inv_delta) {
        constexpr size_t vectors_per_face = 8;
        for (size_t i = 0; i < vectors_per_face; i++) {
            sfpi::vFloat x = sfpi::dst_reg[i];
            sfpi::vFloat t = (x - edge0) * inv_delta;
            v_if(t < 0.0f) { t = 0.0f; }
            v_elseif(t > 1.0f) { t = 1.0f; }
            v_endif;
            sfpi::vFloat result = t * t * (3.0f - 2.0f * t);
            sfpi::dst_reg[i] = result;
        }
    }

    }  // namespace ckernel::sfpu
    #endif // TRISC_MATH

    // High-level API function
    // Accepts `edge0`, `edge1` and `inv_delta` as parameters
    inline void my_smoothstep_tiles(uint32_t idx_dst0, float edge0, float edge1, float inv_delta) {
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            smoothstep_tile_face,
            idx_dst0,
            VectorMode::RC, // Apply on all 4 faces of the tile
            edge0,
            edge1,
            inv_delta));
    }

Parameter Passing
~~~~~~~~~~~~~~~~~

The `smoothstep` function needs two scalar parameters: ``edge0`` and ``edge1``. These are passed to the SFPI kernel through the ``SFPU_UNARY_CALL_NO_TEMPLATE_ARGS`` macro wrapper.

.. code-block:: cpp

    // Passes edge0 and edge1 as arguments to the SFPI kernel
    my_smoothstep_tiles(uint32_t idx_dst0, float edge0, float edge1, float inv_delta);
    // ↓
    // Use the parameters for all elements in the tile face
    smoothstep_tile_face(float edge0, [[maybe_unused]] float edge1, float inv_delta);

The macro wrapper takes the low-level face function, followed by the destination register index, vector mode, and any scalar parameters required by the face function. This approach makes it easy to pass constants or runtime values into the SFPI kernel.

Vector Predicates
~~~~~~~~~~~~~~~~~

The clamping of the intermediate value ``t`` to the [0, 1] range is implemented using vector predicates.

.. code-block:: cpp

    v_if(t < 0.0f) { t = 0.0f; }
    v_elseif(t > 1.0f) { t = 1.0f; }
    v_endif;

The ``v_if`` and ``v_elseif`` instructions perform element-wise conditional assignments on the ``vFloat`` vector ``t``. Each lane of the SIMD vector is evaluated independently. A ``v_endif`` is required to terminate the conditional block.

This is analogous to conditional execution in other parallel programming models, where a mask is used to control which processing elements are active.

Runtime Arguments and Execution
-------------------------------

Back on the host, we set the runtime arguments for the kernels with ``experimental::ProgramRunArgs`` and ``experimental::SetProgramRunArgs``. All three kernels get the named runtime argument ``n_tiles``. Since this is a unary operation, ``tensor_args`` binds only a single DRAM tensor to each of the reader and writer tensor parameters, so the kernels do not need the DRAM addresses as runtime arguments.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep.cpp
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = writer,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = compute,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
        .tensor_args =
            {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

Finally, we add the program to the mesh workload and enqueue it for execution, then read back the results from the destination DRAM tensor to verify correctness against the expected smoothstep function output.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep.cpp
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // Add the program to the workload for the mesh.
    workload.add_program(device_range, std::move(program));
    // Enqueue the workload for execution on the mesh (non-blocking) and wait for completion before reading back.
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    // Read the output buffer and compare it with the expected output.
    std::vector<bfloat16> result_vec =
        cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    for (size_t i = 0; i < result_vec.size(); ++i) {
        auto smoothstep = [](float edge0, float edge1, float x) {
            // Scale, bias and saturate x to 0..1 range
            x = (x - edge0) / (edge1 - edge0);
            x = std::clamp(x, 0.0f, 1.0f);
            // Evaluate polynomial
            return x * x * (3 - 2 * x);
        };
        const float expected = smoothstep(0.0f, 1.0f, static_cast<float>(a_data[i]));
        const float actual = static_cast<float>(result_vec[i]);
        // Check for match within tolerance...
    }

Conclusion
----------

This example demonstrates the implementation of a custom SFPI kernel with parameter passing and conditional logic. Key takeaways are:

*   **Parameter Passing:** The ``SFPU_UNARY_CALL_NO_TEMPLATE_ARGS`` macro wrapper is used to pass scalar arguments to a custom SFPI kernel.
*   **Vector Predicates:** The ``v_if``, ``v_elseif``, and ``v_endif`` instructions provide a mechanism for element-wise conditional logic within an SFPI kernel.
*   **Unary Operations:** Unary SFPI kernels can be implemented efficiently by performing the computation in-place in the destination registers.
