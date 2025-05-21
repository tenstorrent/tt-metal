// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/tt_core_coordinates.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using namespace tt::tt_metal;

constexpr CoreCoord worker_core = {0, 0};
constexpr size_t cb_n_pages = 32;
constexpr size_t cb_page_size = 16;
constexpr size_t n_cbs = 32;
constexpr size_t data_buffer_size = cb_n_pages * cb_n_pages;

std::vector<std::shared_ptr<Buffer>> create_output_buffers(Program& program, IDevice* device) {
    std::vector<std::shared_ptr<Buffer>> output_buffers;
    output_buffers.reserve(n_cbs);
    for (size_t i = 0; i < n_cbs; i++) {
        // Bootleg way to put a single buffer on a single core
        auto const& buffer_config = ShardedBufferConfig{
            device,
            data_buffer_size,
            data_buffer_size,
            BufferType::L1,
            TensorMemoryLayout::WIDTH_SHARDED,
            ShardSpecBuffer(
                CoreRangeSet(CoreRange(worker_core)),
                {cb_n_pages, cb_n_pages},
                ShardOrientation::ROW_MAJOR,
                {cb_n_pages, cb_n_pages},
                {1, 1}),
        };
        output_buffers.push_back(CreateBuffer(buffer_config));
    }
    return output_buffers;
}

std::vector<uint32_t> generate_rt_args(
    uint32_t master_semaphore, uint32_t subordinate_semaphore, std::vector<std::shared_ptr<Buffer>> const& data_buffers) {
    std::vector<uint32_t> rt_args;
    rt_args.reserve(2 + n_cbs);
    rt_args.push_back(master_semaphore);
    rt_args.push_back(subordinate_semaphore);
    std::transform(data_buffers.begin(), data_buffers.end(), std::back_inserter(rt_args), [](auto const& buffer) {
        return buffer->address();
    });
    return rt_args;
}

TEST_F(DeviceFixture, TensixTestCircularBufferNonBlockingAPIs) {
    Program program;
    IDevice* device = devices_.at(0);

    auto const master_semaphore = CreateSemaphore(program, worker_core, 0, CoreType::WORKER);
    auto const subordinate_semaphore = CreateSemaphore(program, worker_core, 0, CoreType::WORKER);

    std::vector<CBHandle> cbs;
    cbs.reserve(n_cbs);
    for (size_t i = 0; i < n_cbs; i++) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_n_pages * cb_page_size, {{i, tt::DataFormat::Float16_b}})
                .set_page_size(i, cb_page_size);
        cbs.push_back(CreateCircularBuffer(program, worker_core, cb_config));
    }

    auto const& master_data_buffers = create_output_buffers(program, device);
    auto const& subordinate_data_buffers = create_output_buffers(program, device);

    std::vector<uint32_t> const& kernel_ct_args{n_cbs, cb_n_pages};

    auto const master_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_non_blocking_master_test_kernel.cpp",
        worker_core,
        tt::tt_metal::ReaderDataMovementConfig{kernel_ct_args});
    auto const subordinate_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_non_blocking_subordinate_test_kernel.cpp",
        worker_core,
        tt::tt_metal::WriterDataMovementConfig{kernel_ct_args});

    auto const& master_rt_args = generate_rt_args(master_semaphore, subordinate_semaphore, master_data_buffers);
    auto const& subordinate_rt_args = generate_rt_args(master_semaphore, subordinate_semaphore, subordinate_data_buffers);

    tt::tt_metal::SetRuntimeArgs(program, master_kernel_id, worker_core, master_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, subordinate_kernel_id, worker_core, subordinate_rt_args);

    tt::tt_metal::detail::CompileProgram(device, program);
    tt::tt_metal::detail::LaunchProgram(device, program, true);

    std::vector<uint32_t> out_buf(data_buffer_size);
    for (size_t i = 0; i < n_cbs; i++) {
        tt::tt_metal::detail::ReadFromBuffer(master_data_buffers[i], out_buf, false);

        uint8_t const* raw_data = reinterpret_cast<uint8_t*>(out_buf.data());
        for (size_t pages_pushed = 0; pages_pushed < cb_n_pages; pages_pushed++) {
            for (size_t requested_pages_free = 0; requested_pages_free < cb_n_pages; requested_pages_free++) {
                ASSERT_EQ(
                    static_cast<bool>(raw_data[pages_pushed * cb_n_pages + requested_pages_free]),
                    requested_pages_free <= (cb_n_pages - pages_pushed));
            }
        }
    }

    for (size_t i = 0; i < n_cbs; i++) {
        tt::tt_metal::detail::ReadFromBuffer(subordinate_data_buffers[i], out_buf, true);

        uint8_t const* raw_data = reinterpret_cast<uint8_t*>(out_buf.data());
        for (size_t pages_pushed = 0; pages_pushed < cb_n_pages; pages_pushed++) {
            for (size_t filled_pages_requested = 0; filled_pages_requested < cb_n_pages; filled_pages_requested++) {
                ASSERT_EQ(
                    static_cast<bool>(raw_data[pages_pushed * cb_n_pages + filled_pages_requested]),
                    filled_pages_requested <= pages_pushed);
            }
        }
    }
}
