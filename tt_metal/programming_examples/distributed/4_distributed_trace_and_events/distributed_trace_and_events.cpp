// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// The following is an advanced programming example that demonstrates:
//
// 1. Initializing a MeshDevice with 2 MeshCommandQueues and a dedicated memory region to store MeshWorkload Traces
// 2. Loading a SubDevice configuration on a Virtual Mesh, and how this configuration gets replicated across all
// physical devices
// 3. Allocating MeshBuffers in the distributed memory space exposed by the Virtual Mesh, to shard data across physical
// devices
// 4. Constructing programs targeting different SubDevices
// 5. Constructing homogeneous (same program dispatched to all physical devices) and heterogeneous (different programs
// dispatched
//    to physical different devices) MeshWorkloads from programs
// 6. Capturing the execution of MeshWorkloads inside a MeshTrace that gets loaded onto the Virtual Mesh
// 7. Performing IO and MeshTrace execution on different MeshCommandQueues and using MeshEvents for MeshCQ <--> MeshCQ
// synchronization

std::shared_ptr<Program> EltwiseBinaryProgramGenerator(
    MeshDevice& mesh_device,
    const std::shared_ptr<MeshBuffer>& src0_buf,
    const std::shared_ptr<MeshBuffer>& src1_buf,
    const std::shared_ptr<MeshBuffer>& output_buf,
    const CoreRangeSet& cores_for_program,
    uint32_t num_tiles,
    uint32_t single_tile_size,
    uint32_t eltwise_op_index) {
    // Program Generation helper function: Can be used to run addition, multiplication and subtraction
    // on a SubDevice.
    // Requires:
    // 1. The src (input) and output buffers
    // 2. The cores of the SubDevice being targeted
    // 3. The number of tiles that must be processed by the op
    // 4. The size of the tile in bytes
    // The op specifier: Addition (0), Multiplication (1), Subtraction (2)
    const std::vector<std::string> op_id_to_op_define = {"add_tiles", "mul_tiles", "sub_tiles"};
    const std::vector<std::string> op_id_to_op_type_define = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWMUL", "EltwiseBinaryType::ELWSUB"};

    const experimental::KernelSpecName binary_reader_kernel{"binary_reader"};
    const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
    const experimental::KernelSpecName eltwise_binary_kernel{"eltwise_binary"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName src1_dfb{"src1"};
    const experimental::DFBSpecName output_dfb{"output"};

    uint32_t num_input_tiles = 2;
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    uint32_t num_output_tiles = 2;
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src0_buf).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(src1_buf).append_to(reader_compile_time_args);
    experimental::KernelSpec binary_reader_kernel_spec{
        .unique_id = binary_reader_kernel,
        .source =
            "tt_metal/programming_examples/distributed/4_distributed_trace_and_events/kernels/"
            "reader_dual_8bank.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
        .runtime_arg_schema = {.runtime_arg_names = {"src0_addr", "src0_num_tiles", "src1_addr", "src1_num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
            },
    };
    // The buffers are mesh buffers (not tensors), so their accessor args are passed positionally.
    binary_reader_kernel_spec.advanced_options.compile_time_varargs = reader_compile_time_args;

    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(output_buf).append_to(writer_compile_time_args);
    experimental::KernelSpec unary_writer_kernel_spec{
        .unique_id = unary_writer_kernel,
        .source =
            "tt_metal/programming_examples/distributed/4_distributed_trace_and_events/kernels/"
            "writer_unary_8bank.cpp",
        .dfb_bindings = {experimental::ConsumerOf(output_dfb, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
            },
    };
    unary_writer_kernel_spec.advanced_options.compile_time_varargs = writer_compile_time_args;

    experimental::KernelSpec::CompilerOptions::Defines binary_defines = {
        {"ELTWISE_OP", op_id_to_op_define[eltwise_op_index]},
        {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op_index]}};
    experimental::KernelSpec eltwise_binary_kernel_spec{
        .unique_id = eltwise_binary_kernel,
        .source = "tt_metal/programming_examples/distributed/4_distributed_trace_and_events/kernels/eltwise_binary.cpp",
        .compiler_options = {.defines = binary_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"),
             experimental::ConsumerOf(src1_dfb, "in1"),
             experimental::ProducerOf(output_dfb, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt", "per_core_block_size"}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::ProgramSpec program_spec{
        .name = "eltwise_binary",
        .kernels = {binary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_binary_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
        .work_units =
            {{.name = "eltwise_binary",
              .kernels = {binary_reader_kernel, unary_writer_kernel, eltwise_binary_kernel},
              .target_nodes = cores_for_program}},
    };
    std::shared_ptr<Program> program =
        std::make_shared<Program>(experimental::MakeProgramFromSpec(mesh_device, program_spec));

    experimental::KernelRunArgs eltwise_binary_args{.kernel = eltwise_binary_kernel};
    experimental::KernelRunArgs unary_writer_args{.kernel = unary_writer_kernel};
    experimental::KernelRunArgs binary_reader_args{.kernel = binary_reader_kernel};
    for (const experimental::NodeCoord& core : experimental::node_range_to_nodes(cores_for_program)) {
        experimental::AddRuntimeArgsForNode(
            eltwise_binary_args.runtime_arg_values,
            core,
            {{"per_core_block_cnt", num_tiles}, {"per_core_block_size", 1u}});
        experimental::AddRuntimeArgsForNode(
            unary_writer_args.runtime_arg_values,
            core,
            {{"dst_addr", static_cast<uint32_t>(output_buf->address())}, {"num_tiles", num_tiles}});
        experimental::AddRuntimeArgsForNode(
            binary_reader_args.runtime_arg_values,
            core,
            {{"src0_addr", static_cast<uint32_t>(src0_buf->address())},
             {"src0_num_tiles", num_tiles},
             {"src1_addr", static_cast<uint32_t>(src1_buf->address())},
             {"src1_num_tiles", num_tiles}});
    }
    experimental::SetProgramRunArgs(
        *program,
        experimental::ProgramRunArgs{.kernel_run_args = {eltwise_binary_args, unary_writer_args, binary_reader_args}});

    return program;
}

int main() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    // Initialize constants used to define the workload
    constexpr uint32_t ADD_OP_ID = 0;
    constexpr uint32_t MULTIPLY_OP_ID = 1;
    constexpr uint32_t SUBTRACT_OP_ID = 2;
    // Create a 2x4 MeshDevice with 2 MeshCQs, 16MB allocated to the trace region and Ethernet Dispatch enabled
    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(MeshShape(2, 4)),  // Shape of MeshDevice
        0,  // l1 small size
        16 << 20, // trace region size
        2, // num MeshCQs
        DispatchCoreType::ETH /* Dispatch Configuration: 8 Chip Wormhole systems can only support 2 MeshCQs when Ethernet Dispatch is enabled */);

    // Initialize command queue ids used for data movement and workload dispatch
    constexpr uint8_t data_movement_cq_id = 1;
    constexpr uint8_t workload_cq_id = 0;
    auto& data_movement_cq = mesh_device->mesh_command_queue(data_movement_cq_id);
    auto& workload_cq = mesh_device->mesh_command_queue(workload_cq_id);

    // =========== Step 1: Initialize and load two SubDevices ===========
    // Each SubDevice contains a single core. This SubDevice configuration is loaded on each physical device
    // in the Virtual Mesh
    CoreRangeSet sub_device_1_cores(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet sub_device_2_cores(CoreRange({1, 1}, {1, 1}));
    SubDevice sub_device_1(std::array{sub_device_1_cores});
    SubDevice sub_device_2(std::array{sub_device_2_cores});
    auto sub_device_manager = mesh_device->create_sub_device_manager(
        {sub_device_1, sub_device_2}, 3200 /* size of L1 region allocated for the SubDevices */);
    mesh_device->load_sub_device_manager(sub_device_manager);

    // =========== Step 2: Initialize IO Buffers and Workload parameters ===========
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // Using bfloat16 in this example
    uint32_t num_tiles_per_device = 2048;  // Number of tiles sent to each physical device
    uint32_t num_tiles_in_mesh =
        num_tiles_per_device * mesh_device->num_devices();  // The total number of tiles in the distributed memory space

    // Specify data layout in distributed memory space - Data will be sharded in row major order across the Virtual Mesh
    tt::tt_metal::distributed::ShardedBufferConfig global_buffer_config{
        .global_size = single_tile_size * num_tiles_in_mesh,  // Total size of the sharded buffer
        .global_buffer_shape =
            {num_tiles_in_mesh * TILE_WIDTH, TILE_HEIGHT},  // Data represents horizontally concatenated tiles
        .shard_shape = {num_tiles_per_device * TILE_WIDTH, TILE_HEIGHT},  // Row major sharding
        .shard_orientation = ShardOrientation::ROW_MAJOR                  // Row major sharding
    };
    // Specify data layout on a single physical device
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = true};
    // Allocate buffers in distributed memory space for first MeshWorkload
    auto add_src0_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    auto add_src1_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    auto add_output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    // Allocate buffers in distributed memory space for second MeshWorkload
    auto mul_sub_src0_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    auto mul_sub_src1_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    auto mul_sub_output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());

    // =========== Step 3: Create Workloads to run on the Virtual Mesh ===========
    // Specify Device Ranges on which the Workloads will run
    MeshCoordinateRange all_devices(mesh_device->shape());
    MeshCoordinateRange top_row(MeshCoordinate{0, 0}, MeshCoordinate{0, mesh_device->num_cols() - 1});
    MeshCoordinateRange bottom_row(
        MeshCoordinate{mesh_device->num_rows() - 1, 0},
        MeshCoordinate{mesh_device->num_rows() - 1, mesh_device->num_cols() - 1});
    // Create three eltwise binary ops using a simple program generation function
    auto add_program = EltwiseBinaryProgramGenerator(
        *mesh_device,
        add_src0_buf,
        add_src1_buf,
        add_output_buf,
        sub_device_1_cores,  // Addition runs on the first SubDevice
        num_tiles_per_device,
        single_tile_size,
        ADD_OP_ID);
    auto multiply_program = EltwiseBinaryProgramGenerator(
        *mesh_device,
        mul_sub_src0_buf,
        mul_sub_src1_buf,
        mul_sub_output_buf,
        sub_device_2_cores,  // Multiplication runs on the second SubDevice
        num_tiles_per_device,
        single_tile_size,
        MULTIPLY_OP_ID);
    auto subtract_program = EltwiseBinaryProgramGenerator(
        *mesh_device,
        mul_sub_src0_buf,
        mul_sub_src1_buf,
        mul_sub_output_buf,
        sub_device_2_cores,  // Subtraction runs on the second SubDevice
        num_tiles_per_device,
        single_tile_size,
        SUBTRACT_OP_ID);
    // Create MeshWorkloads and add programs to them. A MeshWorkload allows a program to target
    // multiple Physical Devices in the Virtual Mesh.
    auto add_mesh_workload = MeshWorkload();
    auto multiply_and_subtract_mesh_workload = MeshWorkload();
    add_mesh_workload.add_program(
        all_devices, std::move(*add_program));  // Addition runs on the full grid (sub_device 1)
    multiply_and_subtract_mesh_workload.add_program(
        top_row,
        std::move(*multiply_program));  // Multiplication runs on the top row (sub_device 2)
    multiply_and_subtract_mesh_workload.add_program(
        bottom_row,
        std::move(*subtract_program));  // Subtraction runs on the bottom row (sub device 2)

    // =========== Step 4: Compile and Load Workloads on the Mesh ===========
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), add_mesh_workload, true);
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), multiply_and_subtract_mesh_workload, true);
    // =========== Step 5: Trace the MeshWorkloads using the Workload Dispatch CQ ===========
    auto trace_id = mesh_device->begin_mesh_trace(workload_cq);
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), add_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), multiply_and_subtract_mesh_workload, false);
    mesh_device->end_mesh_trace(workload_cq, trace_id);

    // =========== Step 6: Populate inputs ===========
    uint32_t workload_0_src0_val = 2;
    uint32_t workload_0_src1_val = 3;
    uint32_t workload_1_src0_val = 7;
    uint32_t workload_1_src1_val = 5;
    // Uniform values passed to the add operation
    std::vector<uint32_t> add_src0_vec = create_constant_vector_of_bfloat16(add_src0_buf->size(), workload_0_src0_val);
    std::vector<uint32_t> add_src1_vec = create_constant_vector_of_bfloat16(add_src1_buf->size(), workload_0_src1_val);
    // Uniform values passed to the multiply and subtract operations (the top row runs multiplication with subtraction
    // on the bottom row of the Virtual Mesh)
    std::vector<uint32_t> mul_sub_src0_vec =
        create_constant_vector_of_bfloat16(mul_sub_src0_buf->size(), workload_1_src0_val);
    std::vector<uint32_t> mul_sub_src1_vec =
        create_constant_vector_of_bfloat16(mul_sub_src1_buf->size(), workload_1_src1_val);

    // =========== Step 7: Write inputs on MeshCQ1 ===========
    // IO is done through MeshCQ1 and Workload dispatch is done through MeshCQ0. Use MeshEvents to synchronize the
    // independent MeshCQs.
    EnqueueWriteMeshBuffer(data_movement_cq, add_src0_buf, add_src0_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, add_src1_buf, add_src1_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, mul_sub_src0_buf, mul_sub_src0_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, mul_sub_src1_buf, mul_sub_src1_vec);
    // Synchronize
    MeshEvent write_event = data_movement_cq.enqueue_record_event();
    workload_cq.enqueue_wait_for_event(write_event);
    // =========== Step 8: Run MeshTrace on MeshCQ0 ===========
    mesh_device->replay_mesh_trace(workload_cq, trace_id, false);
    // Synchronize
    MeshEvent trace_event = workload_cq.enqueue_record_event();
    data_movement_cq.enqueue_wait_for_event(trace_event);
    // =========== Step 9: Read Outputs on MeshCQ1 ===========
    std::vector<bfloat16> add_dst_vec = {};
    std::vector<bfloat16> mul_sub_dst_vec = {};
    EnqueueReadMeshBuffer(data_movement_cq, add_dst_vec, add_output_buf);
    EnqueueReadMeshBuffer(data_movement_cq, mul_sub_dst_vec, mul_sub_output_buf);

    // =========== Step 10: Verify Outputs ===========
    bool pass = true;
    for (auto val : add_dst_vec) {
        pass &= (static_cast<float>(val) == workload_0_src0_val + workload_0_src1_val);
    }
    for (int i = 0; i < mul_sub_dst_vec.size(); i++) {
        if (i < mul_sub_dst_vec.size() / 2) {
            pass &= (static_cast<float>(mul_sub_dst_vec[i]) == workload_1_src0_val * workload_1_src1_val);
        } else {
            pass &= (static_cast<float>(mul_sub_dst_vec[i]) == workload_1_src0_val - workload_1_src1_val);
        }
    }
    mesh_device->release_mesh_trace(trace_id);
    if (pass) {
        std::cout << "Running EltwiseBinary MeshTraces on 2 MeshCQs Passed!" << std::endl;
        return 0;
    }
    std::cout << "Running EltwiseBinary MeshTraces on MeshCQs Failed with Incorrect Outputs!" << std::endl;
    return 1;
}
