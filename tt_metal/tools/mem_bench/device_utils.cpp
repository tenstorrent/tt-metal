// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <string>
#include <string_view>
#include <variant>

#include "context.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "device.hpp"
#include "device_utils.hpp"
#include "kernel_types.hpp"

namespace tt {
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::tools::mem_bench {

std::vector<uint32_t> read_cores(tt::tt_metal::IDevice* device, const CoreRange& cores, uint32_t addr) {
    std::vector<uint32_t> data;
    for (int xi = cores.start_coord.x; xi <= cores.end_coord.x; ++xi) {
        for (int yi = cores.start_coord.y; yi <= cores.end_coord.y; ++yi) {
            std::vector<uint32_t> single_data;
            tt::tt_metal::detail::ReadFromDeviceL1(device, CoreCoord{xi, yi}, addr, sizeof(uint32_t), single_data);
            data.push_back(single_data[0]);
        }
    }
    return data;
}

std::optional<CoreRange> configure_kernels(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const Context& context,
    uint32_t start_y,
    uint32_t num_kernels,
    bool is_writer,
    uint32_t pcie_size,
    uint32_t pcie_offset) {
    constexpr std::string_view k_PcieBenchKernel = "tt_metal/tools/mem_bench/kernels/mem_bench_kernel.cpp";
    const auto grid_size = device->logical_grid_size();
    const auto max_x = grid_size.x;
    const auto max_y = grid_size.y;
    uint32_t total_kernel_transfer = context.total_size;
    uint32_t kernel_transfer_size = context.page_size;

    if (!kernel_transfer_size) {
        kernel_transfer_size = total_kernel_transfer;
    } else if (!num_kernels) {
        return {};
    }

    // Number readers either less than one row
    // or a multiple of the rows
    CoreCoord start_coord{0, start_y};
    CoreCoord end_coord;
    if (num_kernels <= max_x) {
        end_coord.x = start_coord.x + num_kernels - 1;
        end_coord.y = start_coord.y;
    } else {
        const auto number_of_rows = num_kernels / max_x;
        const auto last_row_width = (num_kernels % max_x) ? num_kernels % max_x : max_x;
        end_coord.x = start_coord.x + last_row_width - 1;
        end_coord.y = number_of_rows - 1;
    }
    CoreRange core_range{start_coord, end_coord};

    std::vector<uint32_t> pcie_bench_compile_args(12, 0);
    if (is_writer) {
        pcie_bench_compile_args[5] = 0;                     // reserved_0
        pcie_bench_compile_args[6] = pcie_offset;           // pcie_wr_base
        pcie_bench_compile_args[7] = pcie_size;             // pcie_wr_size
        pcie_bench_compile_args[8] = kernel_transfer_size;  // pcie_wr_transfer_size
    } else {
        pcie_bench_compile_args[0] = context.device_address.unreserved;  // my_rd_dst_addr
        pcie_bench_compile_args[1] = pcie_offset;                        // pcie_rd_base
        pcie_bench_compile_args[2] = pcie_size;                          // pcie_rd_size
        pcie_bench_compile_args[3] = kernel_transfer_size;               // pcie_rd_transfer_size
    }
    pcie_bench_compile_args[4] = context.device_address.rd_bytes;  // my_bytes_rd_addr
    pcie_bench_compile_args[9] = context.device_address.wr_bytes;  // my_bytes_wr_addr
    pcie_bench_compile_args[10] = total_kernel_transfer;
    pcie_bench_compile_args[11] = context.device_address.cycles;

    [[maybe_unused]] auto kernel = tt::tt_metal::CreateKernel(
        program,
        std::string{k_PcieBenchKernel},
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC_0,
            .compile_args = pcie_bench_compile_args,
            .defines = {},
        });

    return core_range;
}

}  // namespace tt::tt_metal::tools::mem_bench
