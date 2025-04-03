// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <functional>


#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>


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

    const auto cores_for_program = sub_device_for_program.has_value()
                                       ? sub_device_for_program->get().cores(HalProgrammableCoreType::TENSIX)
                                       : CoreRange(CoreCoord{0, 0}); /* end_coord */

    std::shared_ptr<Program> program = std::make_shared<Program>();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(*program, cores_for_program, cb_src0_config);

   constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
   CircularBufferConfig cb_src1_config =
       CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
           .set_page_size(src1_cb_index, tile_size_bytes);
   CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_src1_config);


   constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
   constexpr uint32_t num_output_tiles = 1;
   CircularBufferConfig cb_output_config =
       CircularBufferConfig(num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
           .set_page_size(output_cb_index, tile_size_bytes);
   CBHandle cb_output = tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_output_config);


   // Add data movement kernels
   KernelHandle reader = CreateKernel(
       program,
       "tt_metal/programming_examples/contributed/vecadd/kernels/interleaved_tile_read.cpp",
       target_tensix_core,
       DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});


   KernelHandle writer = CreateKernel(
       program,
       "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
       target_tensix_core,
       DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


   // Create the eltwise binary kernel
   auto compute = CreateKernel(
       program,
       "tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp",
       target_tensix_core,
       ComputeConfig{
           .math_fidelity = MathFidelity::HiFi4,
           .fp32_dest_acc_en = false,
           .math_approx_mode = false,
           .compile_args = {},
           .defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}}});


   // Set runtime arguments for each device
   SetRuntimeArgs(program, reader, target_tensix_core, {a->address(), b->address(), num_tiles});
   SetRuntimeArgs(program, writer, target_tensix_core, {c->address(), num_tiles});
   SetRuntimeArgs(program, compute, target_tensix_core, {num_tiles});


   return program;
}


namespace ttnn::distributed::test {

using DistributedEndToEndTests = T3000MeshDeviceFixture;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST_F(DistributedEndToEndTests, ProgramDispatchTest) {
    auto mesh_device = DistributedEndToEndTests::mesh_device_;

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

    auto runtime_args = std::vector<uint32_t>{};
    SetRuntimeArgs(example_program, compute_kernel_id, target_tensix_cores, runtime_args);

    // TODO: print this out and check the contents write a more thorough test
    auto rt_args_out = GetRuntimeArgs(example_program, compute_kernel_id);
    EXPECT_EQ(rt_args_out.size(), 2);

    auto mesh_workload = CreateMeshWorkload();

    auto target_devices = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(example_program), target_devices);

    auto& program = mesh_workload.get_programs().at(target_devices);

    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    Finish(cq);
}


TEST_F(TensorDistributionTest, BufferRoundtripTest) {
   using tt::tt_metal::distributed::ShardedBufferConfig;

    auto mesh_device = DistributedEndToEndTests::mesh_device_;

   auto& cq = mesh_device->mesh_command_queue();


   EXPECT_GE(cq.id(), 0);


   // Define the shape of the shard and the distributed buffer.
   // We will create a distributed buffer with 2 shards of {32, 32} and distribute it across the devices in the mesh.
   auto shard_shape = Shape2D{32, 32};
   auto distributed_buffer_shape = Shape2D{32 * mesh_device->num_rows(), 32 * mesh_device->num_cols()};
   uint32_t tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::UInt32);


   uint32_t distributed_buffer_size_bytes = mesh_device->num_rows()*32 * mesh_device->num_cols()*32 * tile_size_bytes;


   auto local_buffer_config = DeviceLocalBufferConfig{
       .page_size = tile_size_bytes,
       .buffer_type = BufferType::L1,
       .buffer_layout = TensorMemoryLayout::INTERLEAVED,
       .bottom_up = false};
   auto distributed_buffer_config = ShardedBufferConfig{
       .global_size = distributed_buffer_size_bytes,
       .global_buffer_shape = distributed_buffer_shape,
       .shard_shape = shard_shape};

    auto mesh_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(
        distributed_buffer_size_bytes, 1, std::chrono::system_clock::now().time_since_epoch().count());
    EnqueueWriteMeshBuffer(cq, mesh_buffer, src_data);

    std::vector<uint32_t> read_back_data{};
    EnqueueReadMeshBuffer(cq, read_back_data, mesh_buffer, true /* blocking */);

    EXPECT_THAT(read_back_data, Pointwise(FloatEq(), src_data));
}

class DistributedEndToEndTraceTests : public MeshDeviceFixtureBase {
protected:
    DistributedEndToEndTraceTests() :
        MeshDeviceFixtureBase(Config{
            .mesh_device_types = {MeshDeviceFixtureBase::MeshDeviceType::T3000},
            .num_cqs = 1,
            .trace_region_size = 1024,
        }) {}
};

TEST_F(DistributedEndToEndTraceTests, EltwiseAddTests) {
    constexpr uint32_t ADD_OP_ID = 0;

    auto mesh_device = DistributedEndToEndTraceTests::mesh_device_;

    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device->num_rows(), shard_shape.width() * mesh_device->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device->num_rows() * mesh_device->num_cols() * tile_size_bytes;

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

    auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    constexpr float val_to_add = 0.7f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, val_to_add);

    auto program = EltwiseBinaryProgramGenerator(a, b, c, num_tiles, tile_size_bytes, ADD_OP_ID);

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(*program), device_range);

    auto& cq = mesh_device->mesh_command_queue();

    EnqueueMeshWorkload(cq, mesh_workload, true /* blocking */);

    auto trace_id = BeginTraceCapture(mesh_device.get(), cq.id());
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);
    EndTraceCapture(mesh_device.get(), cq.id(), trace_id);

    EnqueueWriteMeshBuffer(cq, a, a_data, false /* blocking */);
    // Block to prevent wriitng during trace, which is illegal
    EnqueueWriteMeshBuffer(cq, b, b_data, true /* blocking */);

    ReplayTrace(mesh_device.get(), cq.id(), trace_id, false);

    ReleaseTrace(mesh_device.get(), trace_id);

    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true /* blocking */);

    auto transform_to_golden = [val_to_add](const bfloat16& a) { return bfloat16(a.to_float() + val_to_add); };
    std::vector<uint32_t> golden_data =
        pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden));

    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(result_data.data());
    bfloat16* golden_bf16 = reinterpret_cast<bfloat16*>(golden_data.data());

   auto total_values = result_data.size() * 2;


    for (int i = 0; i < total_values; i++) {
        EXPECT_THAT(is_close(c_bf16[i].to_float(), golden_bf16[i].to_float()), true);
    }
}

