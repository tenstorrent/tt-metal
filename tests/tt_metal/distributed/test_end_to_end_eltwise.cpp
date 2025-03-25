// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <functional>


#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/sub_device.hpp>

#include <tt-metalium/bfloat16.hpp>


#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmock/gmock-matchers.h>


#include ".cpmcache/googletest/96129d89f45386492ae46d6bb8c027bc3df5f949/googletest/include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "host_api.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

using ::testing::FloatNear;
using ::testing::Pointwise;

std::shared_ptr<Program> EltwiseBinaryProgramGenerator(
    const std::shared_ptr<MeshBuffer>& src0_buf,
    const std::shared_ptr<MeshBuffer>& src1_buf,
    const std::shared_ptr<MeshBuffer>& output_buf,
    uint32_t num_tiles,
    uint32_t single_tile_size,
    uint32_t eltwise_op_index,
    const std::optional<std::reference_wrapper<SubDevice>> sub_device_for_program = std::nullopt) {
    // Program Generation helper function: Can be used to run addition, multiplication and subtraction
    // on a SubDevice.
    // Requires:
    // 1. The src (input) and output buffers
    // 2. The SubDevice being targeted
    // 3. The number of tiles that must be processed by the op
    // 4. The size of the tile in bytes
    // The op specifier: Addition (0), Multiplication (1), Subtraction (2)
    const std::vector<std::string> op_id_to_op_define = {"add_tiles", "mul_tiles", "sub_tiles"};
    const std::vector<std::string> op_id_to_op_type_define = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWMUL", "EltwiseBinaryType::ELWSUB"};

    const auto cores_for_program = sub_device_for_program->get().cores(HalProgrammableCoreType::TENSIX);

    std::shared_ptr<Program> program = std::make_shared<Program>();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(*program, cores_for_program, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(*program, cores_for_program, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(*program, cores_for_program, cb_output_config);

    auto binary_reader_kernel = tt_metal::CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
        cores_for_program,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        cores_for_program,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {};

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    std::map<string, string> binary_defines = {
        {"ELTWISE_OP", op_id_to_op_define[eltwise_op_index]},
        {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op_index]}};
    auto eltwise_binary_kernel = tt_metal::CreateKernel(
        *program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        cores_for_program,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

    SetRuntimeArgs(*program, eltwise_binary_kernel, cores_for_program, {num_tiles, 1});

    const std::array<uint32_t, 7> reader_args = {
        src0_buf->address(), 0, num_tiles, src1_buf->address(), 0, num_tiles, 0};

    const std::array<uint32_t, 3> writer_args = {output_buf->address(), 0, num_tiles};

    SetRuntimeArgs(*program, unary_writer_kernel, cores_for_program, writer_args);
    SetRuntimeArgs(*program, binary_reader_kernel, cores_for_program, reader_args);

    return program;
}

namespace ttnn::distributed::test {

using DistributedEndToEndTests = T3000MeshDeviceFixture;

TEST_F(DistributedEndToEndTests, ProgramDispatchTest) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape(1, 2)});

    EXPECT_NE(mesh_device->get_devices().size(), 2);
    EXPECT_EQ(mesh_device->shape(), MeshShape(1, 2));

    auto& cq = mesh_device->mesh_command_queue();

    EXPECT_GE(cq.id(), 0);

    auto example_program = CreateProgram();

    auto target_tensix_cores = CoreRange{
        CoreCoord{0, 0} /* start_coord */, CoreCoord{0, 1} /* end_coord */
    };

    auto compute_kernel_id = CreateKernel(
        example_program,
        "tt_metal/programming_examples/distributed/1_distributed_program_dispatch/kernels/void_kernel.cpp",
        target_tensix_cores,
        ComputeConfig{.compile_args = {}});

    // Configure the runtime arguments for the kernel.
    auto runtime_args = std::vector<uint32_t>{};
    SetRuntimeArgs(example_program, compute_kernel_id, target_tensix_cores, runtime_args);

    // TODO: print this out and check the contents write a more thorough test
    auto rt_args_out = GetRuntimeArgs(example_program, compute_kernel_id);
    EXPECT_EQ(rt_args_out.size(), 2);

    // Instantiate a MeshWorkload and attach the example program. We'll broadcast
    // this program by enqueueing it across all devices in our 2x4 mesh.
    auto mesh_workload = CreateMeshWorkload();
    auto target_devices = MeshCoordinateRange(mesh_device->shape());

    auto trace_id = BeginTraceCapture(mesh_device.get(), workload_cq_id);

    AddProgramToMeshWorkload(mesh_workload, std::move(example_program), target_devices);
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);
    EndTraceCapture(mesh_device.get(), cq, trace_id);

    ReplayTrace(mesh_device.get(), cq, trace_id, false);

    ReleaseTrace(mesh_device.get(), trace_id);

    auto& program = mesh_workload.get_programs().at(target_devices);

    // Synchronize the mesh command queue to ensure the workload has completed.
    Finish(cq);

    EXPECT_EQ(program.get_last_used_command_queue()->id(), example_program.get_last_used_command_queue()->id());
}

TEST_F(DistributedEndToEndTests, BufferRoundtripTest) {
    using tt::tt_metal::distributed::ShardedBufferConfig;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape(1, 2)});

    EXPECT_NE(mesh_device->get_devices().size(), 2);
    EXPECT_EQ(mesh_device->shape(), MeshShape(1, 2));

    auto& cq = mesh_device->mesh_command_queue();

    EXPECT_GE(cq.id(), 0);

    // Define the shape of the shard and the distributed buffer.
    // We will create a distributed buffer with 2 shards of {32, 32} and distribute it across the devices in the mesh.
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape = Shape2D{32 * mesh_device->num_rows(), 32 * mesh_device->num_cols()};
    uint32_t tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::UInt32);

    uint32_t distributed_buffer_size_bytes =
        mesh_device->num_rows() * 32 * mesh_device->num_cols() * 32 * tile_size_bytes;

    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    auto distributed_buffer_config = ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape};

    // Allocate a distributed buffer in L1 memory, spanning devices in the mesh.
    auto mesh_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Enqueue a write to the distributed buffer (L1 banks across devices) with random data.
    std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(
        distributed_buffer_size_bytes, 1, std::chrono::system_clock::now().time_since_epoch().count());
    EnqueueWriteMeshBuffer(cq, mesh_buffer, src_data);

    // Enqueue a read from the distributed buffer (L1 banks across devices) to a local buffer.
    std::vector<uint32_t> read_back_data{};
    EnqueueReadMeshBuffer(cq, read_back_data, mesh_buffer, true /* blocking */);

    // Data read back across all devices in the mesh should match the original data.
    EXPECT_EQ(src_data, read_back_data);
}

TEST_F(DistributedEndToEndTests, EltwiseAddTests) {
    constexpr uint32_t ADD_OP_ID = 0;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape(1, 2)});

    // Define the global buffer shape and shard shape for distributed buffers
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device->num_rows(), shard_shape.width() * mesh_device->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device->num_rows() * mesh_device->num_cols() * tile_size_bytes;

    // Configure device-local buffer settings
    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    // Create distributed buffers for inputs and output
    auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Create and initialize source data
    constexpr float val_to_add = 0.5f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, val_to_add);

    // Write data to distributed buffers
    auto& cq = mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, a, a_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, b, b_data, false /* blocking */);

    // Create program for distributed computation
    auto program = EltwiseBinaryProgramGenerator(a, b, c, num_tiles, tile_size_bytes, ADD_OP_ID);

    // Create mesh workload and broadcast the program across all devices
    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(*program), device_range);
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    // Read back results
    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true /* blocking */);

    // Verify results
    auto transform_to_golden = [val_to_add](const bfloat16& a) { return bfloat16(a.to_float() + val_to_add); };
    std::vector<uint32_t> golden_data =
        pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden));

    bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(a_data.data());
    bfloat16* b_bf16 = reinterpret_cast<bfloat16*>(b_data.data());
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(result_data.data());
    bfloat16* golden_bf16 = reinterpret_cast<bfloat16*>(golden_data.data());

    auto total_values = result_data.size() * 2;

    for (int i = 0; i < total_values; i++) {
        EXPECT_FLOAT_EQ(c_bf16[i].to_float(), golden_bf16[i].to_float());
    }
}

}  // namespace ttnn::distributed::test
