.. _MatMul_Multi_Core_Optimized_Data_Mcast_example:

Data Multicasting in `matmul_multicore_reuse_mcast`
===================================================

**Note**: This example only works on Grayskull.

Let's level up our code and show how you can leverage and fully customize METALIUM's core-to-core communication through a data broadcasting scheme. METALIUM offers you customizability for creating your very own compute fabric, allowing precise control over which cores disseminate, collect, or process segments of work. This example builds off of the data_reuse one, so we employ the same intermediate (partial) results handling scheme on-core.  However, rather than map tile-work statically to your coregrid, we map in0's rows and in1's columns to the coregrid's edges, and cascade work core-to-core dynamically.  A fun tidbit: "torrent" in Tenstorrent pays homage to this concept of tensor computation flowing like an ultra fast stream of water.


Additional Compile-Time Argument
--------------------------------

We introduced an out_block_num_tiles parameter in our reuse example, yet in our mcast example, we leverage it at compile time to navigate the complexity of multicasting partial results alongside fused operations -- namely the bias and activation functions -- which we will demonstrate further below.  Let's pass it as follows:

.. code-block:: cpp

    vector<uint32_t> compute_kernel_args = {
    ...
    out_block_tiles // out_block_num_tiles
    };
    log_info(tt::LogVerif, " -- out_block_tiles= {} --", out_block_tiles);

Configuring Core Ranges for Tile Distribution
---------------------------------------------

We can define our coregrid from any upper-left corner and define its range with any height and width that our problem calls for.  The cores that make up the grid can be relegated specific broadcast roles (sender or receiver cores) and compute workloads (only certain tiles of in0 and in1).  We start with the following initializations for the grid itself, upper-left corner, and grid edge vectors:

.. code-block:: cpp

    CoreCoord start_core = {0, 0};
    CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

.. code-block:: cpp

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

Next, we define the mcast role of the entire coregrid, the uppermost edge, and the leftmost edge vectors of cores like so:

- **``all_cores``**: Outlines the boundary of all inclusive cores for the matmul kernel.

    .. code-block:: cpp

            CoreRange all_cores(
                {(std::size_t)start_core_x, (std::size_t)start_core_y},
                {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

- **``left_column``**: Isolates the left column of cores for specialized tasks; in this case, multicasting in0 tiles.

    .. code-block:: cpp

            CoreRange left_column(
                {(std::size_t)start_core_x, (std::size_t)start_core_y},
                {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

- **``all_except_left_column``**: Designates which cores will perform their share of in0 and in1 tile work, and calculate their partial results.

    .. code-block:: cpp

            CoreRange all_except_left_column(
                {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
                {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

Then, we define the data flow framework.  The basic idea is that we initiate in0 row and in1 column data flow from the upper-left core of the coregrid (0,0) and designate this as our master sender core.

.. code-block:: cpp

    CoreRange in0_sender_in1_sender(
        {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});

Then we mcast send in0 rows of work vertically down the coregrid's left_column (from DRAM into each of these core's L1).  These left_column cores are responsible for disseminating the **same** in0 row tile data to each core, thereby leveraging the data reuse scheme as we mentioned in the last section.  We also ensure they are designated as receiver cores because they will also take on in1 column work.

.. code-block:: cpp

    CoreRange in0_sender_in1_receiver(
        {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

We also mcast send in1 columns of work horizontally across the coregrid (left to right) into left_column and all_except_left_column ranges of cores.  You can imagine the top row of our coregrid (minus the master sender core) will be responsible for disseminating all the **different** in1 columns of work.

.. code-block:: cpp

    CoreRange in0_receiver_in1_sender(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

The remaining tiles act as receivers for both in0 and in1 tile data.  Essentially we are computing output_tile work (partial results of our output matrix) on each core, wherein each core has been simultaneously mcasted a unique chunk of in0 and in1 tile data to compute on.

.. code-block:: cpp

    CoreRange in0_receiver_in1_receiver(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

This leaves each core with exactly the work it needs to compute its partial results of the output matrix.  We will end up using 4 dataflow kernels:

::

    in0 sender
    in0 receiver
    in1 sender+writer
    in1 receiver+writer

Dataflow Buffer Creation for CoreGrid
-------------------------------------

Recall in our data reuse example, we declared our L1 dataflow buffers with a ``DataflowBufferSpec`` like so:

.. code-block:: cpp

    // The output and the intermediate dataflow buffers share the same memory (aliased).
    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = out_CB_tiles,
        .data_format_metadata = cb_data_format,
        .advanced_options = {.alias_with = {interm0_dfb}},
    };

The mcast example declares its dataflow buffers in the same way.  A ``DataflowBufferSpec`` does not name any cores: METALIUM allocates one instance of the buffer on each core that runs a kernel bound to it.  Here, the work units of the four core groups below together cover ``all_cores``, so each core in the grid gets its own dataflow buffers.

Multicast Reader/Writer Kernel Setup
------------------------------------

In datareuse, one work unit ran the same reader and writer kernels on every core.  In mcast, we have designated core ranges (or more generally speaking, "groups"), and METALIUM gives us functionality to relegate a certain type of reader/writer kernel to a group: each group gets its own work unit in the ``ProgramSpec``.

Below, let's declare a ``KernelSpec`` for each specific sender-receiver kernel.  Take note that the ``hw_config`` of each kernel designates one of two data movement processors, NCRISC (loading data from DRAM to L1) or BRISC (storing data from L1 to DRAM), as defined in the ``$TT_METAL_HOME/tt_metal/api/tt-metalium/kernel_types.hpp`` file.  All four reader kernels share the same runtime argument names (``reader_runtime_arg_names``) and semaphore bindings, and each binds only the input tensors it reads from DRAM.

.. code-block:: cpp

    const tt_metal::experimental::TensorBinding in0_tensor_binding{
        .tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"};
    const tt_metal::experimental::TensorBinding in1_tensor_binding{
        .tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"};

    tt_metal::experimental::KernelSpec mm_reader_kernel_in0_sender_in1_sender_spec{
        .unique_id = mm_reader_kernel_in0_sender_in1_sender,
        .source = OVERRIDE_KERNEL_PREFIX
        "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .semaphore_bindings = reader_semaphore_bindings,
        .tensor_bindings = {in0_tensor_binding, in1_tensor_binding},
        .runtime_arg_schema = {.runtime_arg_names = reader_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt_metal::NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec mm_reader_kernel_in0_sender_in1_receiver_spec{
        .unique_id = mm_reader_kernel_in0_sender_in1_receiver,
        .source = OVERRIDE_KERNEL_PREFIX
        "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .semaphore_bindings = reader_semaphore_bindings,
        .tensor_bindings = {in0_tensor_binding},
        .runtime_arg_schema = {.runtime_arg_names = reader_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt_metal::NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec mm_reader_kernel_in0_receiver_in1_sender_spec{
        .unique_id = mm_reader_kernel_in0_receiver_in1_sender,
        .source = OVERRIDE_KERNEL_PREFIX
        "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .semaphore_bindings = reader_semaphore_bindings,
        .tensor_bindings = {in1_tensor_binding},
        .runtime_arg_schema = {.runtime_arg_names = reader_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt_metal::NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec mm_reader_kernel_in0_receiver_in1_receiver_spec{
        .unique_id = mm_reader_kernel_in0_receiver_in1_receiver,
        .source = OVERRIDE_KERNEL_PREFIX
        "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .semaphore_bindings = reader_semaphore_bindings,
        .runtime_arg_schema = {.runtime_arg_names = reader_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt_metal::NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec unary_writer_kernel_noc0_spec{
        .unique_id = unary_writer_kernel_noc0,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = writer_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt_metal::NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec unary_writer_kernel_noc1_spec{
        .unique_id = unary_writer_kernel_noc1,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = writer_runtime_arg_names},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt_metal::NOC::RISCV_1_default},
            },
    };

The work units then relegate each reader kernel, and the writer kernel for its column, to one core group:

.. code-block:: cpp

    .work_units =
        {{.name = "in0_sender_in1_sender",
          .kernels = {mm_reader_kernel_in0_sender_in1_sender, unary_writer_kernel_noc1, compute_kernel},
          .target_nodes = in0_sender_in1_sender},
         {.name = "in0_sender_in1_receiver",
          .kernels = {mm_reader_kernel_in0_sender_in1_receiver, unary_writer_kernel_noc1, compute_kernel},
          .target_nodes = in0_sender_in1_receiver},
         {.name = "in0_receiver_in1_sender",
          .kernels = {mm_reader_kernel_in0_receiver_in1_sender, unary_writer_kernel_noc0, compute_kernel},
          .target_nodes = in0_receiver_in1_sender},
         {.name = "in0_receiver_in1_receiver",
          .kernels = {mm_reader_kernel_in0_receiver_in1_receiver, unary_writer_kernel_noc0, compute_kernel},
          .target_nodes = in0_receiver_in1_receiver}},

If you are interested in further details on how these work, we implore you to check out the exact dataflow kernels located in the ``$TT_METAL_HOME/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow`` directory.  You can see there are many named runtime arguments with which to experiment with, such as mcast destination nocs.  You can imagine defining your own mcast scheme.

::

    in0_mcast_dest_noc_start_x
    in0_mcast_dest_noc_start_y
    in0_mcast_dest_noc_end_x
    in0_mcast_dest_noc_end_y

New Compute Kernel: Fused Bias Addition and Activation Functions
----------------------------------------------------------------

Like all the examples preceding, we declare our compute kernel with a ``KernelSpec`` as usual.  This example runs the same ``bmm_large_block_zm.cpp`` compute kernel as the data reuse example, and every work unit includes it:

.. code-block:: cpp

    // Create compute kernel. The compute kernel both produces and consumes the intermediate dataflow buffer.
    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0"),
             tt_metal::experimental::ProducerOf(interm0_dfb, "interm0"),
             tt_metal::experimental::ConsumerOf(interm0_dfb, "interm0")},
        .compile_time_args = compute_kernel_args,
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

A fused variant of this kernel, "bmm_large_block_zm_fused_bias_activation", is used by the ttnn matmul op (``ttnn/cpp/ttnn/operations/matmul/device/kernels/compute``).  It adds the following steps:

a. **Flow Control through Conditionals**

    - When bias fusion is enabled (`FUSE_BIAS`), intermediate results are directly packed and may not require reloading for subsequent operations within the same batch, indicated by `enable_reload = false`. We can employ this as a means of minimizing mem-to-mem operations.

    - For kernels without bias fusion or when the `PACKER_L1_ACC` is not defined, we determine whether intermediate results need to be reloaded based on the computation phase (ie. our `spill` condition and the current `block`). This ensures that for operations that accumulate results over multiple blocks, intermediate data is correctly managed across iterations.

b. **Bias Broadcasting Mechanism**

    .. code-block:: cpp

        add_bcast_rows_init();
        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
            uint32_t bcast_tile_idx = in1_index_subblock_offset;
            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
                bcast_tile_idx++;
            }
        }

c. **In-place Activation Function**

    .. code-block:: cpp

        #ifdef SFPU_OP_INIT_ACTIVATION
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
        #endif

d. **Handling Partial Results**

    .. code-block:: cpp

        if (enable_reload) {
            reload_from_cb_to_dst(in0_cb_id, in1_cb_id, mm_partials_cb_id, out_subblock_num_tiles, out_subblock_w, out_subblock_h, in0_block_w);
        }

Semaphores
----------

To cleanly coordinate the distribution and processing of in0 and in1 tiles in our mcast strategy, we should introduce semaphores. Without these, we run the risk of mcast sending data from one Tensix core to another too early (before the first Tensix core's dataflow buffer is fully populated), or mcast receiving too few packets of data and thus computing prematurely (before the second Tensix core's dataflow buffer is fully populated).  METALIUM makes this very simple, by allowing you to declare a ``SemaphoreSpec`` and simply passing the entire CoreGrid as its target nodes.  Therefore, we define our sender and receiver core semaphores as follows, to maintain synchronization of compute across the device.

.. code-block:: cpp

    /*
     * Semaphores used for the mcast handshake (initial value INVALID = 0)
     */
    tt_metal::experimental::SemaphoreSpec in0_mcast_sender_semaphore_spec{
        .unique_id = in0_mcast_sender_semaphore, .target_nodes = all_cores};
    tt_metal::experimental::SemaphoreSpec in0_mcast_receiver_semaphore_spec{
        .unique_id = in0_mcast_receiver_semaphore, .target_nodes = all_cores};
    tt_metal::experimental::SemaphoreSpec in1_mcast_sender_semaphore_spec{
        .unique_id = in1_mcast_sender_semaphore, .target_nodes = all_cores};
    tt_metal::experimental::SemaphoreSpec in1_mcast_receiver_semaphore_spec{
        .unique_id = in1_mcast_receiver_semaphore, .target_nodes = all_cores};

The reader kernels bind all four semaphores by name:

.. code-block:: cpp

    const tt_metal::experimental::Group<tt_metal::experimental::SemaphoreBinding> reader_semaphore_bindings = {
        {.semaphore_spec_name = in0_mcast_sender_semaphore, .accessor_name = "in0_mcast_sender"},
        {.semaphore_spec_name = in0_mcast_receiver_semaphore, .accessor_name = "in0_mcast_receiver"},
        {.semaphore_spec_name = in1_mcast_sender_semaphore, .accessor_name = "in1_mcast_sender"},
        {.semaphore_spec_name = in1_mcast_receiver_semaphore, .accessor_name = "in1_mcast_receiver"}};

Inside a reader kernel, each ``Semaphore`` is constructed from its accessor name, and a ``Noc`` object is used for the remote semaphore operations:

.. code-block:: cpp

    Semaphore in0_mcast_sender_semaphore(sem::in0_mcast_sender);
    Semaphore in0_mcast_receiver_semaphore(sem::in0_mcast_receiver);
    Semaphore in1_mcast_sender_semaphore(sem::in1_mcast_sender);
    Semaphore in1_mcast_receiver_semaphore(sem::in1_mcast_receiver);
    Noc noc;

A receiver core resets its receiver semaphore, increments the sender core's semaphore with ``up(...)``, and waits until the sender sets its receiver semaphore to ``VALID``:

.. code-block:: cpp

    // Set in0 semaphore value to INVALID
    in0_mcast_receiver_semaphore.set(INVALID);

    // Atomic increment source core counter
    in0_mcast_sender_semaphore.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);

    // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
    in0_mcast_receiver_semaphore.wait(VALID);

The sender core waits until all receivers are ready, multicasts the block, and then multicasts ``VALID`` to the receiver semaphores with ``set_multicast(...)``:

.. code-block:: cpp

    // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its value
    // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
    in0_mcast_sender_semaphore.wait(in0_mcast_num_dests);
    in0_mcast_sender_semaphore.set(0);

.. code-block:: cpp

    // We should also multicast the flag to destinations
    // num_dests must not include source, since we are NOT really doing a local copy!
    in0_mcast_receiver_semaphore.set_multicast(
        noc,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_num_dests);

Kernel Runtime Arguments
------------------------

Recall that we just designated NCRISCs to handle our DRAM->CoreGrid L1 data movement.  METALIUM lets us bind the input tensors to the reader kernels and dereference them with a stride by multiples of core coordinates.  ``AddRuntimeArgsForNode`` adds the named runtime args of one core to a kernel's ``KernelRunArgs``.

.. code-block:: cpp

    tt_metal::experimental::AddRuntimeArgsForNode(
        mm_reader_run_args->runtime_arg_values,
        core,
        {
            {"in0_tensor_start_tile_id", (std::uint32_t)Kt * per_core_M * core_idx_y},
            {"in0_tensor_stride_w", (std::uint32_t)1},
            {"in0_tensor_stride_h", (std::uint32_t)Kt},
            {"in0_tensor_next_block_stride", (std::uint32_t)in0_block_w},

            {"in0_block_w", (std::uint32_t)in0_block_w},
            {"in0_block_h", (std::uint32_t)per_core_M},
            {"in0_block_num_tiles", (std::uint32_t)in0_block_w * per_core_M},

            {"in1_tensor_start_tile_id", (std::uint32_t)per_core_N * core_idx_x},
            {"in1_tensor_stride_w", (std::uint32_t)1},
            {"in1_tensor_stride_h", (std::uint32_t)Nt},
            {"in1_tensor_next_block_stride", (std::uint32_t)in0_block_w * Nt},
            ...

For runtime, we need to set a few more IDs on corner cores of our CoreGrid, that will act solely as worker cores.

.. code-block:: cpp

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
            CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
            CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)core.y};
            CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
            CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
            CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_r - 1};

            auto left_core_physical = mesh_device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = mesh_device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = mesh_device->worker_core_from_logical_core(right_core);
            auto top_core_physical = mesh_device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = mesh_device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = mesh_device->worker_core_from_logical_core(bottom_core);

At this point we can specify exactly which worker core plays which role for mcasting in0 and in1 data.  Here we can map the physical core on device with:

.. code-block:: cpp

    {"in0_mcast_dest_noc_start_x", (std::uint32_t)right_core_physical.x},
    {"in0_mcast_dest_noc_start_y", (std::uint32_t)right_core_physical.y},
    {"in0_mcast_dest_noc_end_x", (std::uint32_t)left_core_plus_one_physical.x},
    {"in0_mcast_dest_noc_end_y", (std::uint32_t)left_core_plus_one_physical.y},
    {"in0_mcast_num_dests", (std::uint32_t)(num_cores_c - 1)},
    {"in0_mcast_sender_noc_x", (std::uint32_t)left_core_physical.x},
    {"in0_mcast_sender_noc_y", (std::uint32_t)left_core_physical.y},

    {"in1_mcast_dest_noc_start_x", (std::uint32_t)bottom_core_physical.x},
    {"in1_mcast_dest_noc_start_y", (std::uint32_t)bottom_core_physical.y},
    {"in1_mcast_dest_noc_end_x", (std::uint32_t)top_core_plus_one_physical.x},
    {"in1_mcast_dest_noc_end_y", (std::uint32_t)top_core_plus_one_physical.y},
    {"in1_mcast_num_dests", (std::uint32_t)(num_cores_r - 1)},
    {"in1_mcast_sender_noc_x", (std::uint32_t)top_core_physical.x},
    {"in1_mcast_sender_noc_y", (std::uint32_t)top_core_physical.y},
    ...


Finally, we target our NCRISC (RISCV_0) and BRISC (RISCV_1) processors: for each core, we pick the reader and writer kernels whose ``KernelRunArgs`` receive that core's runtime args. For our master send core (0,0), which initiates data movement for both matrices in0 and in1:

.. code-block:: cpp

    if (core_idx_x == 0 and core_idx_y == 0) {
        mm_reader_run_args = &mm_reader_kernel_in0_sender_in1_sender_run_args;  // RISCV_0_default
        writer_run_args = &unary_writer_kernel_noc1_run_args;                   // RISCV_1_default
    }

For the left_column cores, we task them with receiving in1 columns from the top and sending in0 rows to the right:

.. code-block:: cpp

    else if (core_idx_x == 0 and core_idx_y != 0) {
        mm_reader_run_args = &mm_reader_kernel_in0_sender_in1_receiver_run_args;  // RISCV_0_default
        writer_run_args = &unary_writer_kernel_noc1_run_args;                     // RISCV_1_default
    }

For the upper_row cores (minus the upper-left master send core), we task them with receiving matrix in0 rows from the left, and sending in1 columns upwards.


.. code-block:: cpp

    else if (core_idx_x != 0 and core_idx_y == 0) {
        mm_reader_run_args = &mm_reader_kernel_in0_receiver_in1_sender_run_args;  // RISCV_1_default
        writer_run_args = &unary_writer_kernel_noc0_run_args;                     // RISCV_0_default
    }

For all other cores (between the left_column and upper_row cores, minus the master send core), we task these with receiving in0 rows from the left and in1 columns from the top, thereby dividing work appropriately and commencing the partial results computation process.

.. code-block:: cpp

    else {
        mm_reader_run_args = &mm_reader_kernel_in0_receiver_in1_receiver_run_args;  // RISCV_1_default
        writer_run_args = &unary_writer_kernel_noc0_run_args;                       // RISCV_0_default
    }

After the loop, ``SetProgramRunArgs`` passes the runtime args of all six reader and writer kernels to the program, together with the DRAM tensors for the tensor parameters:

.. code-block:: cpp

    // Set the runtime args, and the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {mm_reader_kernel_in0_sender_in1_sender_run_args,
             mm_reader_kernel_in0_sender_in1_receiver_run_args,
             mm_reader_kernel_in0_receiver_in1_sender_run_args,
             mm_reader_kernel_in0_receiver_in1_receiver_run_args,
             unary_writer_kernel_noc0_run_args,
             unary_writer_kernel_noc1_run_args},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);
