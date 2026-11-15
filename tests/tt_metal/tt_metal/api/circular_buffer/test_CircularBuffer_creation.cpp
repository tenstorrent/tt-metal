// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include "circular_buffer_test_utils.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

using std::vector;
using namespace tt::tt_metal;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace basic_tests::circular_buffer {

bool test_cb_config_written_to_core(
    distributed::MeshWorkload& workload,
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreRangeSet& /*cr_set*/,
    const std::map<uint8_t, std::vector<uint32_t>>& cb_config_per_buffer_index) {
    bool pass = true;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    vector<uint32_t> cb_config_vector;

    for (const auto& cb : program.circular_buffers()) {
        for (const CoreRange& core_range : cb->core_ranges().ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord core_coord(x, y);
                    uint32_t cb_config_buffer_size =
                        workload.get_cb_size(mesh_device, core_coord, tt::CoreType::WORKER);

                    auto sem_base_addr = workload.get_sem_base_addr(mesh_device, core_coord, tt::CoreType::WORKER);
                    slow_dispatch::ReadFromL1(
                        *mesh_device, core_coord, sem_base_addr, cb_config_buffer_size, cb_config_vector);

                    for (const auto& [buffer_index, golden_cb_config] : cb_config_per_buffer_index) {
                        auto base_index = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * buffer_index;
                        pass &= (golden_cb_config.at(0) == cb_config_vector.at(base_index));      // address
                        pass &= (golden_cb_config.at(1) == cb_config_vector.at(base_index + 1));  // size
                        pass &= (golden_cb_config.at(2) == cb_config_vector.at(base_index + 2));  // num pages
                    }
                }
            }
        }
    }

    return pass;
}

TEST_F(MeshDeviceFixture, TensixTestCreateCircularBufferAtValidIndices) {
    CBConfig cb_config;

    CoreRange cr({0, 0}, {0, 1});
    CoreRangeSet cr_set({cr});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram(*this->devices_[0]);
    initialize_program(program, cr_set);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    uint32_t l1_unreserved_base = devices_.at(0)->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::map<uint8_t, std::vector<uint32_t>> golden_cb_config = {
        {0, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {2, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {16, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {24, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}}};
    std::map<uint8_t, tt::DataFormat> data_format_spec = {
        {0, cb_config.data_format},
        {2, cb_config.data_format},
        {16, cb_config.data_format},
        {24, cb_config.data_format}};
    CircularBufferConfig expected_config = CircularBufferConfig(cb_config.page_size, data_format_spec)
                                               .set_page_size(tt::CBIndex::c_0, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_2, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_16, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_24, cb_config.page_size);

    CircularBufferConfig actual_config = CircularBufferConfig(cb_config.page_size);
    actual_config.index(tt::CBIndex::c_0).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_2).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_16).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_24).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);

    EXPECT_TRUE(actual_config == expected_config);

    CreateCircularBuffer(program_, cr_set, actual_config);

    for (auto& device : this->devices_) {
        distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
        EXPECT_TRUE(test_cb_config_written_to_core(workload, device, cr_set, golden_cb_config));
    }
}

TEST_F(MeshDeviceFixture, TestCreateCircularBufferAtInvalidIndex) {
    CBConfig cb_config;

    EXPECT_ANY_THROW(CircularBufferConfig(cb_config.page_size, {{NUM_CIRCULAR_BUFFERS, cb_config.data_format}}));

    // An index the config can store but the architecture does not support is rejected when added to a program.
    if (max_dfbs_ < NUM_CIRCULAR_BUFFERS) {
        CircularBufferConfig config(cb_config.page_size, {{max_dfbs_, cb_config.data_format}});
        config.set_page_size(max_dfbs_, cb_config.page_size);
        Program program = CreateProgram(*this->devices_[0]);
        EXPECT_THAT(
            [&] { CreateCircularBuffer(program, CoreCoord(0, 0), config); },
            ThrowsMessage<std::runtime_error>(HasSubstr("exceeds max number of circular buffers per core")));
        EXPECT_TRUE(program.circular_buffers().empty());
    }
}

TEST_F(MeshDeviceFixture, TestCreateCircularBufferWithMismatchingConfig) {
    Program program = CreateProgram(*this->devices_[0]);
    CBConfig cb_config;

    EXPECT_ANY_THROW(
        CircularBufferConfig(cb_config.page_size, {{0, cb_config.data_format}}).set_page_size(1, cb_config.page_size));
}

// Verifies that the DataType-based CircularBufferConfig constructor produces a config with identical state to the
// tt::DataFormat-based constructor when the spec entries map to each other via datatype_to_dataformat_converter.
// CircularBufferConfig::operator== covers total_size, globally_allocated_address, data_formats, page_sizes, tiles,
// and shadow_global_buffer.
TEST_F(MeshDeviceFixture, TestCircularBufferConfigConstructorWithDataTypeMatchesDataFormat) {
    CBConfig cb_config;

    const std::map<uint8_t, DataType> data_type_spec = {
        {0, DataType::BFLOAT16},
        {2, DataType::FLOAT32},
        {16, DataType::UINT32},
    };

    std::map<uint8_t, tt::DataFormat> data_format_spec;
    for (const auto& [idx, dtype] : data_type_spec) {
        data_format_spec[idx] = datatype_to_dataformat_converter(dtype);
    }

    CircularBufferConfig config_via_data_format(cb_config.page_size, data_format_spec);
    CircularBufferConfig config_via_data_type(cb_config.page_size, data_type_spec);

    EXPECT_EQ(config_via_data_format, config_via_data_type);
    EXPECT_EQ(config_via_data_format.total_size(), cb_config.page_size);
    EXPECT_EQ(config_via_data_type.total_size(), cb_config.page_size);
    EXPECT_EQ(config_via_data_format.data_formats(), config_via_data_type.data_formats());
}

TEST_F(MeshDeviceFixture, TensixTestCreateCircularBufferAtOverlappingIndex) {
    Program program = CreateProgram(*this->devices_[0]);
    CBConfig cb_config;

    CoreRange cr({0, 0}, {1, 1});
    CoreRangeSet cr_set({cr});

    std::map<uint8_t, tt::DataFormat> data_format_spec1 = {{0, cb_config.data_format}, {16, cb_config.data_format}};
    CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, data_format_spec1)
                                       .set_page_size(0, cb_config.page_size)
                                       .set_page_size(16, cb_config.page_size);

    std::map<uint8_t, tt::DataFormat> data_format_spec2 = {
        {1, cb_config.data_format}, {2, cb_config.data_format}, {16, cb_config.data_format}};
    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, data_format_spec2)
                                       .set_page_size(1, cb_config.page_size)
                                       .set_page_size(2, cb_config.page_size)
                                       .set_page_size(16, cb_config.page_size);

    CreateCircularBuffer(program, cr_set, config1);

    EXPECT_ANY_THROW(CreateCircularBuffer(program, cr_set, config2));
}

TEST_F(MeshDeviceFixture, TensixTestCreateCircularBufferWithTooManyPages) {
    Program program = CreateProgram(*this->devices_[0]);
    CBConfig cb_config;

    CoreRange cr({0, 0}, {1, 1});
    CoreRangeSet cr_set({cr});

    CircularBufferConfig config = CircularBufferConfig(cb_config.page_size * (1 << 16), {{0, cb_config.data_format}})
                                      .set_page_size(0, cb_config.page_size);

    EXPECT_ANY_THROW(CreateCircularBuffer(program, cr_set, config));
}

TEST_F(MeshDeviceFixture, TensixTestCreateCircularBufferOnOutOfRangeCores) {
    for (auto& device : this->devices_) {
        auto& cq = device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = CreateProgram(*device);
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        auto grid_size = device->compute_with_storage_grid_size();
        // Extend one column beyond the compute grid into dispatch core territory
        CoreRange cr({0, 0}, {grid_size.x, grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        CBConfig cb_config;
        CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, {{0, cb_config.data_format}})
                                          .set_page_size(0, cb_config.page_size);
        CreateCircularBuffer(program_, cr_set, config);

        EXPECT_ANY_THROW(distributed::EnqueueMeshWorkload(cq, workload, false));
    }
}

CBDescriptor make_cb_descriptor(const CoreRangeSet& cr_set, uint8_t buffer_index, uint32_t num_pages) {
    CBConfig cb_config;
    return CBDescriptor{
        .total_size = num_pages * cb_config.page_size,
        .core_ranges = cr_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = buffer_index,
            .data_format = cb_config.data_format,
            .page_size = cb_config.page_size,
        }}},
    };
}

// A program built from a ProgramDescriptor merges a CB's core ranges as CreateCircularBuffer does.
TEST_F(MeshDispatchFixture, TensixTestCircularBufferFromDescriptorMergesCoreRanges) {
    CBConfig cb_config;
    CoreRangeSet cr_set(std::vector<CoreRange>{CoreRange({0, 0}, {2, 0}), CoreRange({3, 0}, {4, 0})});

    ProgramDescriptor descriptor;
    descriptor.cbs.push_back(make_cb_descriptor(cr_set, 0, cb_config.num_pages));
    Program program_from_descriptor(descriptor);

    Program program = CreateProgram(*this->devices_[0]);
    CreateCircularBuffer(
        program,
        cr_set,
        CircularBufferConfig(cb_config.num_pages * cb_config.page_size, {{0, cb_config.data_format}})
            .set_page_size(0, cb_config.page_size));

    ASSERT_EQ(program_from_descriptor.circular_buffers().size(), 1);
    const CoreRangeSet& core_ranges = program_from_descriptor.circular_buffers()[0]->core_ranges();
    EXPECT_EQ(core_ranges, CoreRangeSet(CoreRange({0, 0}, {4, 0}))) << core_ranges.str();
    EXPECT_EQ(core_ranges, program.circular_buffers()[0]->core_ranges()) << core_ranges.str();
}

// TensixTestCircularBufferConfigDispatchWithOverlappingCoreRanges' layout, built from a ProgramDescriptor and run:
// CB index 0 has a different size on two adjacent core groups, and the merged range of CB index 1 overlaps both.
// Each core must still receive its own group's config for index 0.
TEST_F(MeshDispatchFixture, TensixTestCircularBufferFromDescriptorWithOverlappingCoreRanges) {
    CoreRangeSet cores0(CoreRange({0, 0}, {2, 0}));
    CoreRangeSet cores1(CoreRange({3, 0}, {4, 0}));
    CoreRangeSet all_cores(std::vector<CoreRange>{CoreRange({0, 0}, {2, 0}), CoreRange({3, 0}, {4, 0})});

    ProgramDescriptor descriptor;
    descriptor.cbs = {
        make_cb_descriptor(cores0, 0, 1), make_cb_descriptor(cores1, 0, 2), make_cb_descriptor(all_cores, 1, 1)};
    KernelDescriptor kernel;
    kernel.kernel_source = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.config = ReaderConfigDescriptor{};
    descriptor.kernels.push_back(kernel);

    CBConfig cb_config;
    for (auto& device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), Program(descriptor));
        this->RunProgram(device, workload);

        for (const CoreRange& core_range : all_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                std::vector<uint32_t> cb_config_vector;
                slow_dispatch::ReadFromL1(
                    *device,
                    core,
                    workload.get_sem_base_addr(device, core, tt::CoreType::WORKER),
                    workload.get_cb_size(device, core, tt::CoreType::WORKER),
                    cb_config_vector);
                const uint32_t cb0_num_pages = cores0.contains(core) ? 1 : 2;
                const uint32_t cb0_base = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * 0;
                const uint32_t cb1_base = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * 1;
                EXPECT_EQ(cb_config_vector.at(cb0_base + 1), cb0_num_pages * cb_config.page_size) << core.str();
                EXPECT_EQ(cb_config_vector.at(cb0_base + 2), cb0_num_pages) << core.str();
                EXPECT_EQ(cb_config_vector.at(cb1_base + 1), cb_config.page_size) << core.str();
            }
        }
    }
}

}  // end namespace basic_tests::circular_buffer
