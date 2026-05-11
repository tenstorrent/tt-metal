// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// Path 1: Program's static CB region (allocated upward from l1_unreserved_base)
// runs past the lowest L1 address occupied by an allocated buffer. The
// underlying detection is validate_circular_buffer_region in ProgramImpl; our
// wrapper in ConfigureDeviceWithProgram translates its TT_THROW into an
// ASAN-style abort for emule.
TEST_F(MeshDeviceFixture, Metadata_CB_Tensor_Clash_SanityCheck) {
    auto& mesh_device = this->devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    // Pin a low lowest_occupied_compute_l1_address by allocating a 1 MB L1
    // mesh buffer. The allocator places L1 buffers from the top of L1
    // downward, so this leaves only a small unreserved window below the
    // buffer for CBs to grow into.
    constexpr uint32_t l1_buffer_size = 1024 * 1024;  // 1 MB
    distributed::DeviceLocalBufferConfig l1_local_config{
        .page_size = l1_buffer_size, .buffer_type = BufferType::L1, .bottom_up = false};
    distributed::ReplicatedBufferConfig l1_buf_config{.size = l1_buffer_size};
    auto l1_buffer = distributed::MeshBuffer::create(l1_buf_config, l1_local_config, mesh_device.get());

    // CB sized to overrun the buffer's lower edge.
    Program program = CreateProgram();
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_num_pages = 512;
    constexpr uint32_t cb_page_size = 2048;
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_num_pages * cb_page_size, {{cb_id, tt::DataFormat::Float16_b}})
            .set_page_size(cb_id, cb_page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {}
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*\\[ASAN ERROR\\] Metadata Overflow.*");
}

// Path 2: Program's per-core-type metadata (rta + sem + cb + dfb + kernel_text)
// exceeds the reserved KERNEL_CONFIG window in L1, regardless of whether any
// tensors are allocated. The existing ProgramImpl validators can't catch this
// when no L1 tensor pins lowest_occupied_compute_l1_address. Our wrapper-side
// window check in ConfigureDeviceWithProgram is what fires here.
TEST_F(MeshDeviceFixture, Metadata_KernelConfig_Window_Overflow_SanityCheck) {
    auto& mesh_device = this->devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {}
    )";

    // Inflate the RTA section by placing one kernel on every compute core,
    // each with the max allowed runtime args. Each kernel forms its own
    // kernel group (one core per group), so finalize_rt_args sums them.
    //   per-kernel RTA section = max_runtime_args * sizeof(uint32_t) = 1364 B
    //   total RTA section = num_cores * 1364 B
    // Tensix KERNEL_CONFIG window is 69 KB on WH/BH, so >= 52 cores overflows.
    CoreCoord grid = device->compute_with_storage_grid_size();
    std::vector<uint32_t> max_rtas(max_runtime_args, 0);
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            CoreCoord core{x, y};
            auto kernel_id = CreateKernelFromString(
                program,
                kernel_src,
                core,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            SetRuntimeArgs(program, kernel_id, core, max_rtas);
        }
    }

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*\\[ASAN ERROR\\] Metadata Overflow.*");
}

}  // namespace tt::tt_metal
