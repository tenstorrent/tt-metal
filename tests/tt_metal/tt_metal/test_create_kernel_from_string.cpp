// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <unordered_set>

#include "core_coord.h"
#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "impl/device/device.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "impl/program/program.hpp"
#include "tt_cluster_descriptor_types.h"

using namespace tt;
using namespace tt::tt_metal;

class CompileProgramWithKernelCreatedFromString : public ::testing::Test {
   protected:
    void SetUp() override {
        vector<chip_id_t> device_ids;
        const uint32_t num_devices = GetNumAvailableDevices();
        for (chip_id_t i = 0; i < num_devices; i++) {
            device_ids.push_back(i);
        }
        this->devices_ = detail::CreateDevices(device_ids);
    }

    void TearDown() override { detail::CloseDevices(this->devices_); }

    std::map<chip_id_t, Device *> devices_;
};

TEST_F(CompileProgramWithKernelCreatedFromString, DataMovementKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const string &kernel_src_code = R"(
    #include <stdint.h>
    #include "dataflow_api.h"

    void kernel_main() {
        uint32_t src_addr = get_arg_val<uint32_t>(0);
        uint32_t src_noc_x = get_arg_val<uint32_t>(1);
        uint32_t src_noc_y = get_arg_val<uint32_t>(2);
        uint32_t num_tiles = get_arg_val<uint32_t>(3);

        constexpr uint32_t cb_id_in0 = 0;

        constexpr uint32_t ublock_size_tiles = 4;
        uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
            uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);

            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, ublock_size_tiles);
            src_addr += ublock_size_bytes;
        }
    }
    )";

    Program program = CreateProgram();
    tt_metal::CreateKernelFromString(
        program,
        kernel_src_code,
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    for (auto [_, device] : this->devices_) {
        detail::CompileProgram(device, program);
    };
}

TEST_F(CompileProgramWithKernelCreatedFromString, ComputeKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const string &kernel_src_code = R"(
#include "debug/dprint.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

    void MAIN {

        DPRINT_MATH(DPRINT << "Hello, I am running a void compute kernel." << ENDL());

    }

}
    )";

    Program program = CreateProgram();
    tt_metal::CreateKernelFromString(
        program,
        kernel_src_code,
        cores,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});

    for (auto [_, device] : this->devices_) {
        detail::CompileProgram(device, program);
    };
}

TEST_F(CompileProgramWithKernelCreatedFromString, EthernetKernel) {
    const string &kernel_src_code = R"(
#include <cstdint>
#include "debug/ring_buffer.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
    for (uint32_t idx = 0; idx < 40; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
}
#else
}
}
#endif
)";

    for (auto [device_id, device] : this->devices_) {
        const std::unordered_set<CoreCoord> &active_ethernet_cores = device->get_active_ethernet_cores(true);
        if (active_ethernet_cores.empty()) {
            log_info(LogTest, "Skipping this test on device {} because it has no active ethernet cores.", device_id);
            continue;
        }
        Program program = CreateProgram();
        tt_metal::CreateKernelFromString(
            program,
            kernel_src_code,
            *active_ethernet_cores.begin(),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});
        detail::CompileProgram(device, program);
    };
}
