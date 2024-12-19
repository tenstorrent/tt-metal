// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_workload.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"
namespace tt::tt_metal::distributed::test {

static inline void skip_test_if_not_t3000() {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const auto arch = tt::Cluster::instance().arch();
    const size_t num_devices = tt::Cluster::instance().number_of_devices();

    if (slow_dispatch) {
        GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
    }
    if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
    }
}
class MeshDevice_T3000 : public ::testing::Test {
protected:
    void SetUp() override {
        skip_test_if_not_t3000();
        this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

class MeshDevice_N300 : public ::testing::Test {
protected:
    void SetUp() override { this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2))); }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MeshDevice_N300, TestHomogenousMeshWorkload) {
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    // Create buffers
    std::vector<std::shared_ptr<Buffer>> input_buffers = {};
    std::vector<std::shared_ptr<Buffer>> output_buffers = {};
    for (auto device : mesh_device_->get_devices()) {
        InterleavedBufferConfig dram_config{
            .device = device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                input_buffers.push_back(CreateBuffer(dram_config));
                output_buffers.push_back(CreateBuffer(dram_config));
            }
        }
    }
    // Create MeshWorkload
    Program program = CreateProgram();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    auto reader_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        full_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    auto scaling_sem_idx = CreateSemaphore(program, full_grid, sem_scaling_factor);
    uint32_t scaling_height_toggle = 16;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    uint32_t add_factor = 64;
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            CoreCoord curr_core = {col_idx, row_idx};
            SetRuntimeArgs(
                program,
                reader_writer_kernel,
                curr_core,
                {input_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 output_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 0, /* src_bank_id */
                 0, /* dst_bank_id */
                 add_factor,
                 constants::TILE_HEIGHT,
                 constants::TILE_WIDTH,
                 scaling_sem_idx,
                 scaling_height_toggle});
            CBHandle cb_src0 = CreateCircularBuffer(program, curr_core, cb_src0_config);
        }
    }
    auto mesh_workload = MeshWorkload();
    LogicalDeviceRange devices = LogicalDeviceRange({0, 0}, {2, 1});
    mesh_workload.add_program(devices, program);
    std::size_t buffer_idx = 0;
    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteBuffer(device->command_queue(), input_buffers.at(buffer_idx), src_vec, false);
                buffer_idx++;
            }
        }
    }

    mesh_workload.enqueue(mesh_device_, 0, false);
    buffer_idx = 0;
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                std::vector<bfloat16> dst_vec = {};
                EnqueueReadBuffer(device->command_queue(), output_buffers.at(buffer_idx), dst_vec, true);
                buffer_idx++;
                for (int i = 0; i < dst_vec.size(); i++) {
                    float ref_val = std::pow(2, 1);
                    if (i >= 512) {
                        ref_val = std::pow(2, 2);
                    }
                    EXPECT_EQ(dst_vec[i].to_float(), ref_val);
                }
            }
        }
    }
}

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST(MeshDeviceSuite, Test1x1SystemMeshInitialize) {
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();

    auto config =
        tt::tt_metal::distributed::MeshDeviceConfig(MeshShape(1, 1), MeshOffset(0, 0), {}, MeshType::RowMajor);

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close_devices();
    });
}

}  // namespace tt::tt_metal::distributed::test
