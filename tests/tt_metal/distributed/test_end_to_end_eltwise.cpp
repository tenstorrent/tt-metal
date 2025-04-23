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

#include "host_api.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal {

std::shared_ptr<Program> EltwiseBinaryProgramGenerator(

    const std::shared_ptr<distributed::MeshBuffer>& src0_buf,
    const std::shared_ptr<distributed::MeshBuffer>& src1_buf,
    const std::shared_ptr<distributed::MeshBuffer>& output_buf,
    uint32_t num_tiles,
    uint32_t single_tile_size,
    uint8_t eltwise_op_index,
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

    const auto cores_for_program = sub_device_for_program.has_value()
                                       ? sub_device_for_program->get().cores(HalProgrammableCoreType::TENSIX)
                                       : CoreRange(CoreCoord{0, 0}, CoreCoord(2, 2)); /* end_coord */

    auto program = std::make_shared<Program>();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(*program, cores_for_program, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = CreateCircularBuffer(*program, cores_for_program, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(
            num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = CreateCircularBuffer(*program, cores_for_program, cb_output_config);

    auto binary_reader_kernel = CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
        cores_for_program,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        cores_for_program,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {};

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    std::map<string, string> binary_defines = {
        {"ELTWISE_OP", op_id_to_op_define[eltwise_op_index]},
        {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op_index]}};
    auto eltwise_binary_kernel = CreateKernel(
        *program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        cores_for_program,
        ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

    SetRuntimeArgs(*program, eltwise_binary_kernel, cores_for_program, {num_tiles, 1});

    const std::array<uint32_t, 7> reader_args = {
        src0_buf->address(), 0, num_tiles, src1_buf->address(), 0, num_tiles, 0};

    const std::array<uint32_t, 3> writer_args = {output_buf->address(), 0, num_tiles};

    SetRuntimeArgs(*program, unary_writer_kernel, cores_for_program, writer_args);
    SetRuntimeArgs(*program, binary_reader_kernel, cores_for_program, reader_args);

    return program;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed::test {

using MeshEndToEndT3kTests = T3000MeshDeviceFixture;
using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST_F(MeshEndToEndT3kTests, ProgramDispatchTest) {

    auto& cq = mesh_device_->mesh_command_queue();

    uint8_t cq_id = cq.id();

    EXPECT_GE(cq_id, 0);

    auto example_program = CreateProgram();

    auto target_tensix_cores = CoreRange{
        CoreCoord{0, 0} /* start_coord */, CoreCoord{1, 1} /* end_coord */
    };

    auto compute_kernel_id = CreateKernel(
        example_program,
        "tt_metal/programming_examples/distributed/1_distributed_program_dispatch/kernels/void_kernel.cpp",
        target_tensix_cores,
        ComputeConfig{.compile_args = {}});

    auto runtime_args = std::vector<uint32_t>{};
    SetRuntimeArgs(example_program, compute_kernel_id, target_tensix_cores, runtime_args);

    auto rt_args_out = GetRuntimeArgs(example_program, compute_kernel_id);
    EXPECT_EQ(rt_args_out.size(), 2);

    auto mesh_workload = CreateMeshWorkload();

    auto target_devices = MeshCoordinateRange(mesh_device_->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(example_program), target_devices);

    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    EXPECT_EQ(mesh_workload.get_last_used_command_queue()->id(), cq_id);

    Finish(cq);
}

TEST_F(MeshEndToEndT3kTests, BufferRoundtripTest) {
    using tt::tt_metal::distributed::ShardedBufferConfig;

    auto& cq = mesh_device_->mesh_command_queue();

    EXPECT_GE(cq.id(), 0);

    // Define the shape of the shard and the distributed buffer.
    // We will create a distributed buffer with 2 shards of {32, 32} and distribute it across the devices in the mesh.
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape = Shape2D{32 * mesh_device_->num_rows(), 32 * mesh_device_->num_cols()};
    uint32_t tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::UInt32);

    uint32_t distributed_buffer_size_bytes =
        mesh_device_->num_rows() * 32 * mesh_device_->num_cols() * 32 * tile_size_bytes;

    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    auto distributed_buffer_config = ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape};

    auto mesh_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(
        distributed_buffer_size_bytes, 1, std::chrono::system_clock::now().time_since_epoch().count());
    EnqueueWriteMeshBuffer(cq, mesh_buffer, src_data);

    std::vector<uint32_t> read_back_data{};
    EnqueueReadMeshBuffer(cq, read_back_data, mesh_buffer, true /* blocking */);

    EXPECT_THAT(read_back_data, Pointwise(Eq(), src_data));
}

TEST_F(MeshEndToEndT3kTests, UntracedEltwiseAddTest) {
    constexpr uint8_t kAddOpId = 0;

    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device_->num_rows(), shard_shape.width() * mesh_device_->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device_->num_rows() * mesh_device_->num_cols() * tile_size_bytes;

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

    auto a_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto b_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto out_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());

    constexpr float kValToAdd = 0.7f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, kValToAdd);

    auto& cq = mesh_device_->mesh_command_queue();

    EnqueueWriteMeshBuffer(cq, a_buffer, a_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, b_buffer, b_data, true /* blocking */);

    auto program = EltwiseBinaryProgramGenerator(a_buffer, b_buffer, out_buffer, num_tiles, tile_size_bytes, kAddOpId);

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(*program), device_range);
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, out_buffer, true /* blocking */);

    auto transform_to_golden = [kValToAdd](const bfloat16& a) { return bfloat16(a.to_float() + kValToAdd); };
    std::vector<bfloat16> result_vec = unpack_uint32_vec_into_bfloat16_vec(result_data, bfloat16_identity_transform);
    std::vector<bfloat16> golden_vec = unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden);

    ASSERT_EQ(result_vec.size(), golden_vec.size());
    for (std::size_t i = 0; i < result_vec.size(); i++) {
        EXPECT_TRUE(is_close(result_vec[i].to_float(), golden_vec[i].to_float()));
    }
}

class MeshEndToEndT3kTraceTests : public MeshDeviceFixtureBase {
protected:
    MeshEndToEndT3kTraceTests() :
        MeshDeviceFixtureBase(Config{
            .mesh_device_types = {MeshDeviceFixtureBase::MeshDeviceType::T3000},
            .num_cqs = 2,
            .trace_region_size = 3072,  // 1024 per workload necessary
        }) {}
};

TEST_F(MeshEndToEndT3kTraceTests, EltwiseAddTest) {
    constexpr uint8_t kAddOpId = 0;

    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device_->num_rows(), shard_shape.width() * mesh_device_->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device_->num_rows() * mesh_device_->num_cols() * tile_size_bytes;

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

    auto a_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto b_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto out_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());

    constexpr float kValToAdd = 0.7f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, kValToAdd);

    auto program = EltwiseBinaryProgramGenerator(a_buffer, b_buffer, out_buffer, num_tiles, tile_size_bytes, kAddOpId);

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(*program), device_range);

    auto& cq = mesh_device_->mesh_command_queue();

    EnqueueMeshWorkload(cq, mesh_workload, true /* blocking */);

    auto trace_id = BeginTraceCapture(mesh_device_.get(), cq.id());
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);
    EndTraceCapture(mesh_device_.get(), cq.id(), trace_id);

    EnqueueWriteMeshBuffer(cq, a_buffer, a_data, false /* blocking */);
    // Block to prevent wriitng during trace, which is illegal
    EnqueueWriteMeshBuffer(cq, b_buffer, b_data, true /* blocking */);

    ReplayTrace(mesh_device_.get(), cq.id(), trace_id, false);

    ReleaseTrace(mesh_device_.get(), trace_id);

    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, out_buffer, true /* blocking */);

    auto transform_to_golden = [kValToAdd](const bfloat16& a) { return bfloat16(a.to_float() + kValToAdd); };

    std::vector<bfloat16> result_vec = unpack_uint32_vec_into_bfloat16_vec(result_data, bfloat16_identity_transform);
    std::vector<bfloat16> golden_vec = unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden);

    ASSERT_EQ(result_vec.size(), golden_vec.size());
    for (std::size_t i = 0; i < result_vec.size(); i++) {
        EXPECT_TRUE(is_close(result_vec[i].to_float(), golden_vec[i].to_float()));
    }
}

TEST_F(MeshEndToEndT3kTraceTests, EltwiseMulTest) {
    constexpr uint8_t kMulOpId = 1;

    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device_->num_rows(), shard_shape.width() * mesh_device_->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device_->num_rows() * mesh_device_->num_cols() * tile_size_bytes;

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

    auto a_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto b_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());
    auto out_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device_.get());

    constexpr float kValToMul = 0.2f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, kValToMul);

    auto program = EltwiseBinaryProgramGenerator(a_buffer, b_buffer, out_buffer, num_tiles, tile_size_bytes, kMulOpId);

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device_->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(*program), device_range);

    auto& cq = mesh_device_->mesh_command_queue();

    EnqueueMeshWorkload(cq, mesh_workload, true /* blocking */);

    auto trace_id = BeginTraceCapture(mesh_device_.get(), cq.id());
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);
    EndTraceCapture(mesh_device_.get(), cq.id(), trace_id);

    EnqueueWriteMeshBuffer(cq, a_buffer, a_data, false /* blocking */);
    // Block to prevent wriitng during trace, which is illegal
    EnqueueWriteMeshBuffer(cq, b_buffer, b_data, true /* blocking */);

    ReplayTrace(mesh_device_.get(), cq.id(), trace_id, false);

    ReleaseTrace(mesh_device_.get(), trace_id);

    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, out_buffer, true /* blocking */);

    auto transform_to_golden = [kValToMul](const bfloat16 a) { return bfloat16(a * bfloat16(kValToMul)); };
    std::vector<bfloat16> result_vec = unpack_uint32_vec_into_bfloat16_vec(result_data, bfloat16_identity_transform);
    std::vector<bfloat16> golden_vec = unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden);

    ASSERT_EQ(result_vec.size(), golden_vec.size());
    for (std::size_t i = 0; i < result_vec.size(); i++) {
        EXPECT_TRUE(is_close(result_vec[i].to_float(), golden_vec[i].to_float()));
    }
}

MATCHER_P(Bfloat16Eq, calculated, "") { return arg.to_float() == calculated; }

TEST_F(MeshEndToEndT3kTraceTests, SimulEltwiseTest) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    constexpr uint8_t kAddOpId = 0;
    constexpr uint8_t kMulOpId = 1;
    constexpr uint8_t kSubOpId = 2;

    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {0, mesh_device_->num_cols() - 1}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(
        CoreRange({mesh_device_->num_rows() - 1, 0}, {mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1}))});
    auto sub_device_manager = mesh_device_->create_sub_device_manager(
        {sub_device_1, sub_device_2}, 3200 /* size of L1 region allocated for the SubDevices */);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    constexpr uint8_t kDataMovementCqID = 1;
    constexpr uint8_t kWorkloadCqId = 0;
    auto& data_movement_cq = mesh_device_->mesh_command_queue(kDataMovementCqID);
    auto& workload_cq = mesh_device_->mesh_command_queue(kWorkloadCqId);

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    uint32_t num_tiles_per_device = 2048;  // Number of tiles sent to each physical device
    uint32_t num_tiles_in_mesh =
        num_tiles_per_device * mesh_device_->num_devices();  // The total number of tiles in the distributed memory space

    tt::tt_metal::distributed::ShardedBufferConfig global_buffer_config{
        .global_size = single_tile_size * num_tiles_in_mesh,  // Total size of the sharded buffer
        .global_buffer_shape =
            {num_tiles_in_mesh * TILE_WIDTH, TILE_HEIGHT},  // Data represents horizontally concatenated tiles
        .shard_shape = {num_tiles_per_device * TILE_WIDTH, TILE_HEIGHT},  // Row major sharding
        .shard_orientation = ShardOrientation::ROW_MAJOR                  // Row major sharding
    };

    // Specify data layout on a single physical device
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    // Allocate buffers in distributed memory space for first MeshWorkload
    auto add_src0_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto add_src1_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto add_output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Allocate buffers in distributed memory space for second MeshWorkload
    auto mul_sub_src0_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto mul_sub_src1_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto mul_sub_output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    MeshCoordinateRange all_devices(mesh_device_->shape());
    MeshCoordinateRange top_row(MeshCoordinate{0, 0}, MeshCoordinate{0, mesh_device_->num_cols() - 1});
    MeshCoordinateRange bottom_row(
        MeshCoordinate{mesh_device_->num_rows() - 1, 0},
        MeshCoordinate{mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1});
    // Create three eltwise binary ops using a simple program generation function
    auto add_program = EltwiseBinaryProgramGenerator(
        add_src0_buf,
        add_src1_buf,
        add_output_buf,
        num_tiles_per_device,
        single_tile_size,
        kAddOpId,
        sub_device_1);  // Addition runs on the first SubDevice
    auto multiply_program = EltwiseBinaryProgramGenerator(
        mul_sub_src0_buf,
        mul_sub_src1_buf,
        mul_sub_output_buf,
        num_tiles_per_device,
        single_tile_size,
        kMulOpId,
        sub_device_2);  // Multiplication runs on the second SubDevice);
    auto subtract_program = EltwiseBinaryProgramGenerator(
        mul_sub_src0_buf,
        mul_sub_src1_buf,
        mul_sub_output_buf,
        num_tiles_per_device,
        single_tile_size,
        kSubOpId,
        sub_device_2);  // Subtraction runs on the second SubDevice

    auto add_mesh_workload = CreateMeshWorkload();
    auto multiply_and_subtract_mesh_workload = CreateMeshWorkload();
    AddProgramToMeshWorkload(
        add_mesh_workload, std::move(*add_program), all_devices);  // Addition runs on the full grid (sub_device 1)
    AddProgramToMeshWorkload(
        multiply_and_subtract_mesh_workload,
        std::move(*multiply_program),
        top_row);  // Multiplication runs on the top row (sub_device 2)
    AddProgramToMeshWorkload(
        multiply_and_subtract_mesh_workload,
        std::move(*subtract_program),
        bottom_row);  // Subtraction runs on the bottom row (sub device 2)

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), add_mesh_workload, true);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), multiply_and_subtract_mesh_workload, true);

    auto trace_id = BeginTraceCapture(mesh_device_.get(), kWorkloadCqId);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), add_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), multiply_and_subtract_mesh_workload, false);
    EndTraceCapture(mesh_device_.get(), kWorkloadCqId, trace_id);

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

    EnqueueWriteMeshBuffer(data_movement_cq, add_src0_buf, add_src0_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, add_src1_buf, add_src1_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, mul_sub_src0_buf, mul_sub_src0_vec);
    EnqueueWriteMeshBuffer(data_movement_cq, mul_sub_src1_buf, mul_sub_src1_vec);

    MeshEvent write_event = EnqueueRecordEvent(data_movement_cq);
    EnqueueWaitForEvent(workload_cq, write_event);

    ReplayTrace(mesh_device_.get(), kWorkloadCqId, trace_id, false);

    // Synchronize
    MeshEvent trace_event = EnqueueRecordEvent(workload_cq);
    EnqueueWaitForEvent(data_movement_cq, trace_event);

    std::vector<bfloat16> add_dst_vec = {};
    std::vector<bfloat16> mul_sub_dst_vec = {};
    EnqueueReadMeshBuffer(data_movement_cq, add_dst_vec, add_output_buf);
    EnqueueReadMeshBuffer(data_movement_cq, mul_sub_dst_vec, mul_sub_output_buf);

    EXPECT_THAT(add_dst_vec, Each(Bfloat16Eq(workload_0_src0_val + workload_0_src1_val)));

    std::size_t sub_or_mul_size = mul_sub_dst_vec.size() / 2;
    tt::stl::Span<bfloat16> mul_dst_span(mul_sub_dst_vec.data(), sub_or_mul_size);
    tt::stl::Span<bfloat16> sub_dst_span(mul_sub_dst_vec.data() + sub_or_mul_size, sub_or_mul_size);

    EXPECT_THAT(mul_dst_span, Each(Bfloat16Eq(workload_1_src0_val * workload_1_src1_val)));
    EXPECT_THAT(sub_dst_span, Each(Bfloat16Eq(workload_1_src0_val - workload_1_src1_val)));
}

}  // namespace ttnn::distributed::test
