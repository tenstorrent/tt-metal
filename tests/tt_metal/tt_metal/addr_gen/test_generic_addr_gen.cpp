// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/hw/inc/dataflow_api_generic_addrgen.h"

namespace addr_gen_tests {

template <typename DSpecT>
struct AddrGenInputs {
    using dspec = DSpecT;
};

struct ExpectedDSpec {
    std::array<uint32_t, 2> tensor_strides;
    size_t tensor_volume;

    std::array<uint32_t, 2> shard_strides;
    size_t shard_volume;

    std::array<uint32_t, 2> shard_grid;
    std::array<uint32_t, 2> shard_grid_strides;
};

struct ExpectedBankAndOffset {
    size_t page_id;
    size_t bank_id;
    size_t bank_offset;
};

template <ExpectedDSpec ExpectedDSpecVal, ExpectedBankAndOffset... ExpectedBankAndOffsetVals>
struct AddrGenExpected {
    static constexpr auto dspec = ExpectedDSpecVal;
    static constexpr auto bank_and_offset =
        std::array<ExpectedBankAndOffset, sizeof...(ExpectedBankAndOffsetVals)>{ExpectedBankAndOffsetVals...};
};

template <typename Inputs, typename Expected>
struct AddrGenParams {
    using inputs = Inputs;
    using expected = Expected;
};

namespace addr_gen_test_params {

constexpr std::array<uint32_t, 2> tensor_shape_array = {2, 3};
constexpr std::array<uint32_t, 2> shard_shape_array = {1, 2};
USING_SHAPE_WRAPPER(tensor_shape_1, tensor_shape_array);
USING_SHAPE_WRAPPER(shard_shape_1, shard_shape_array);

using test_params_1 = AddrGenParams<
    AddrGenInputs<KernelDistributionSpec<tensor_shape_1, shard_shape_1, 4>>,
    AddrGenExpected<
        ExpectedDSpec{{3, 1}, 6, {2, 1}, 2, {2, 2}, {2, 1}},
        ExpectedBankAndOffset{0, 0, 0},
        ExpectedBankAndOffset{1, 0, 1},
        ExpectedBankAndOffset{2, 1, 0},
        ExpectedBankAndOffset{3, 2, 0},
        ExpectedBankAndOffset{4, 2, 1},
        ExpectedBankAndOffset{5, 3, 0}>>;

}  // namespace addr_gen_test_params

// TODO: Merge with test_buffer_distribution_spec.cpp?
struct KernelParams {
    tt::tt_metal::Shape physical_tensor_shape;
    tt::tt_metal::Shape2D page_shape;
    float bytes_per_element;

    struct ShardSpec {
        tt::tt_metal::Shape physical_shard_shape;
        tt::tt_metal::CoreRangeSet grid;
        tt::tt_metal::ShardOrientation shard_orientation;
        tt::tt_metal::BufferType buffer_type;
    };
    ShardSpec input_shard_spec;
    ShardSpec output_shard_spec;
};

std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 2>
create_replicated_input_and_output_mesh_buffers_from_inputs(
    const KernelParams& inputs, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    // These values would be passed from tensor correctly based on PageConfig
    const auto host_size_in_bytes = inputs.physical_tensor_shape.volume() * inputs.bytes_per_element;
    const auto page_size = inputs.page_shape.height() * inputs.page_shape.width() * inputs.bytes_per_element;

    // Mirrors allocate_mesh_buffer_on_device in ttnn
    const tt::tt_metal::distributed::ReplicatedBufferConfig mesh_buffer_config{.size = host_size_in_bytes};

    // Create input mesh buffer
    auto input_buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.input_shard_spec.physical_shard_shape,
        inputs.page_shape,
        inputs.input_shard_spec.grid,
        inputs.input_shard_spec.shard_orientation);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig input_device_local_config{
        .page_size = page_size,
        .buffer_type = inputs.input_shard_spec.buffer_type,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .buffer_distribution_spec = input_buffer_distribution_spec,
    };
    const auto input_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, input_device_local_config, mesh_device);

    // Create output mesh buffer
    auto output_buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.input_shard_spec.physical_shard_shape,
        inputs.page_shape,
        inputs.output_shard_spec.grid,
        inputs.output_shard_spec.shard_orientation);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig output_device_local_config{
        .page_size = page_size,
        .buffer_type = inputs.output_shard_spec.buffer_type,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .buffer_distribution_spec = output_buffer_distribution_spec,
    };
    const auto output_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, output_device_local_config, mesh_device);

    return {input_mesh_buffer, output_mesh_buffer};
}
// struct KernelParams {
//     BufferDistributionSpecInputs input_buffer_spec;
// }

}  // namespace addr_gen_tests

using namespace addr_gen_tests;
using namespace tt::tt_metal;

template <typename T>
class AddrGenTests : public ::testing::Test {};

TYPED_TEST_SUITE(AddrGenTests, ::testing::Types<addr_gen_test_params::test_params_1>);

TYPED_TEST(AddrGenTests, LookUp) {
    using dspec_t = TypeParam::inputs::dspec;
    constexpr auto dspec_val = dspec_t{};
    using expected = TypeParam::expected;

    // Create sharded accessor
    auto sharded_accessor = ShardedAccessor<dspec_t>{};

    // Check that the computed values in DSpec match the expected values
    ASSERT_EQ(dspec_val.tensor_strides, expected::dspec.tensor_strides);
    ASSERT_EQ(dspec_val.tensor_volume, expected::dspec.tensor_volume);
    ASSERT_EQ(dspec_val.shard_strides, expected::dspec.shard_strides);
    ASSERT_EQ(dspec_val.shard_volume, expected::dspec.shard_volume);
    ASSERT_EQ(dspec_val.shard_grid, expected::dspec.shard_grid);
    ASSERT_EQ(dspec_val.shard_grid_strides, expected::dspec.shard_grid_strides);

    // Check that the computed bank and offset values match the expected values
    for (const auto& expected_bank_and_offset : expected::bank_and_offset) {
        auto [bank_id, bank_offset] = sharded_accessor.get_bank_and_offset(expected_bank_and_offset.page_id);
        std::cout << "page_id: " << expected_bank_and_offset.page_id << std::endl;
        std::cout << "bank_id: " << bank_id << std::endl;
        std::cout << "bank_offset: " << bank_offset << std::endl;
        EXPECT_EQ(bank_id, expected_bank_and_offset.bank_id);
        EXPECT_EQ(bank_offset, expected_bank_and_offset.bank_offset);
    }
}

TEST(AddrGenTests, OutOfBoundsPageAccess) {
    using dspec_t = addr_gen_test_params::test_params_1::inputs::dspec;
    constexpr auto dspec_val = dspec_t{};

    // Create sharded accessor
    auto sharded_accessor = ShardedAccessor<dspec_t>{};

    // Out of bounds page id and page coord
    constexpr auto out_of_bounds_page_id = dspec_val.tensor_volume + 1;
    auto out_of_bounds_page_coord = dspec_val.tensor_shape;
    out_of_bounds_page_coord[0] = dspec_val.tensor_shape[0] + 1;

    EXPECT_THAT(
        std::function<void()>([sharded_accessor, dspec_val]() {
            // Out of bounds page id
            constexpr auto out_of_bounds_page_id = dspec_val.tensor_volume + 1;
            const auto _ = sharded_accessor.get_bank_and_offset(out_of_bounds_page_id);
        }),
        ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "Page id {} must be less than tensor volume {}!", out_of_bounds_page_id, dspec_val.tensor_volume))));

    EXPECT_THAT(
        std::function<void()>([sharded_accessor, dspec_val]() {
            // Out of bounds page coord
            auto out_of_bounds_page_coord = dspec_val.tensor_shape;
            out_of_bounds_page_coord[0] = dspec_val.tensor_shape[0] + 1;
            const auto _ = sharded_accessor.get_bank_and_offset(out_of_bounds_page_coord);
        }),
        ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "Page coord {} must be less than tensor shape {} at rank 0!",
            out_of_bounds_page_coord[0],
            dspec_val.tensor_shape[0]))));
}

class KernelAddrGenTests : public GenericMeshDeviceFixture, public ::testing::WithParamInterface<KernelParams> {};

TEST_P(KernelAddrGenTests, SingleCoreReshard) {
    const auto& params = GetParam();

    // Create input and output replicated mesh buffers across generic mesh device; tests will only use first device
    const auto [mesh_buffer, output_mesh_buffer] =
        create_replicated_input_and_output_mesh_buffers_from_inputs(params, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto shard_view = mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto local_device = shard_view->device();
    const auto host_size_in_bytes = mesh_buffer->device_local_size();
    const auto bank_base_address = mesh_buffer->address();
    const auto output_bank_base_address = output_mesh_buffer->address();

    // Initialize local device buffer to 0
    {
        std::vector<uint32_t> zeros_vector(shard_view->aligned_size_per_bank() / sizeof(uint32_t), 0);
        for (const auto& core : corerange_to_cores(params.input_shard_spec.grid)) {
            tt::tt_metal::detail::WriteToDeviceL1(
                local_device, core, bank_base_address, zeros_vector, shard_view->core_type());
        }
    }

    // Clear out command queue
    {
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(local_device->id());
        chip_id_t mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(local_device->id());
        uint32_t cq_size = local_device->sysmem_manager().get_cq_size();
        uint32_t cq_start = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
            CommandQueueHostAddrType::UNRESERVED);

        std::vector<uint32_t> cq_zeros((cq_size - cq_start) / sizeof(uint32_t), 0);

        tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
            cq_zeros.data(),
            (cq_size - cq_start),
            get_absolute_cq_offset(channel, 0, cq_size) + cq_start,
            mmio_device_id,
            channel);
    }

    // Create src vector
    const auto src =
        tt::test_utils::generate_uniform_random_vector<uint8_t>(0, UINT8_MAX, host_size_in_bytes / sizeof(uint8_t));

    {
        tt::log_info("Writing with: FDMeshCommandQueue enqueue_write_shards");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(src.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, shard_data_transfer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());
    }

    // Validate output
    {
        // Initialize dst vector
        std::vector<uint8_t> dst(host_size_in_bytes / sizeof(uint8_t), 0);
        tt::log_info("Reading with: FDMeshCommandQueue enqueue_read_shards");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(dst.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_read_shards(shard_data_transfer, mesh_buffer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());

        // Validate read results are correct
        EXPECT_EQ(src, dst);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AddrGenTests,
    KernelAddrGenTests,
    ::testing::Values(
        // BLOCK sharding; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        KernelParams{
            .physical_tensor_shape = tt::tt_metal::Shape{2, 64, 96},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,

            .input_shard_spec =
                KernelParams::ShardSpec{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                KernelParams::ShardSpec{
                    .physical_shard_shape = tt::tt_metal::Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        }));
