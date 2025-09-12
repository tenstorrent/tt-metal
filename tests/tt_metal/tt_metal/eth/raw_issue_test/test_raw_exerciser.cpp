#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/hal.hpp>
#include "impl/context/metal_context.hpp"
#include <cstdint>
#include <stdexcept>

/**
 * @brief TT-Metal test program for raw issue exerciser with NOC noise.
 *
 * This test launches the raw_exerciser_kernel on an Ethernet core to test potential hardware bugs
 * in L1 writes, while simultaneously running the noc_noise_emitter_kernel on another core to
 * generate interfering NOC traffic to the same L1.
 *
 * Buffers and coordinates are obtained from Metalium device objects.
 * Number of attempts: 1,000,000
 * Exerciser write buffer: 128 bytes
 * NOC dest buffer: 10,000 bytes
 * Address step: 16
 * NOC write size: 128 bytes
 */

int main(int argc, char** argv) {
    bool pass = true;

    try {
        // Device setup
        tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(0);

        tt::tt_metal::Program program{};

        // Get HAL instance for L1 address queries
        auto& hal = tt::tt_metal::MetalContext::instance().hal();

        uint32_t unreserved_l1_start = hal.get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

        // Get active Ethernet cores
        auto active_eth_cores = device->get_active_ethernet_cores(false);
        if (active_eth_cores.empty()) {
            throw std::runtime_error("No active Ethernet cores found");
        }

        // Select core for exerciser
        CoreCoord logical_exerciser_core = *active_eth_cores.begin();
        CoreCoord physical_exerciser_core = device->ethernet_core_from_logical_core(logical_exerciser_core);

        // Select core for emitter (prefer another eth core, fallback to worker {0,0})
        CoreCoord logical_emitter_core;
        bool emitter_on_eth = active_eth_cores.size() >= 2;
        if (emitter_on_eth) {
            logical_emitter_core = *(++active_eth_cores.begin());
        } else {
            logical_emitter_core = {0, 0};
        }

        // Create kernels
        auto exerciser_kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/eth/raw_issue_test/raw_exerciser_kernel.cpp",
            logical_exerciser_core,
            tt::tt_metal::EthernetConfig{});

        auto emitter_kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/eth/raw_issue_test/noc_noise_emitter_kernel.cpp",
            logical_emitter_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

        // Runtime args for exerciser
        uint32_t range_start = unreserved_l1_start;
        uint32_t range_end = range_start + 128;
        uint32_t addr_step = 16;
        uint32_t num_attempts = 1000000;

        tt::tt_metal::SetRuntimeArgs(
            program, exerciser_kernel, logical_exerciser_core, {range_start, range_end, addr_step, num_attempts});

        // Runtime args for emitter
        uint64_t num_iters = 1000000ULL;
        uint32_t valid_write_range_start = range_end;
        uint32_t valid_write_range_end = valid_write_range_start + 10000;
        uint32_t dest_noc_x = physical_exerciser_core.x;
        uint32_t dest_noc_y = physical_exerciser_core.y;
        uint32_t noc_write_size = 128;

        uint32_t num_iters_hi = static_cast<uint32_t>(num_iters >> 32);
        uint32_t num_iters_lo = static_cast<uint32_t>(num_iters & 0xFFFFFFFFULL);

        tt::tt_metal::SetRuntimeArgs(
            program,
            emitter_kernel,
            logical_emitter_core,
            {num_iters_hi,
             num_iters_lo,
             valid_write_range_start,
             valid_write_range_end,
             dest_noc_x,
             dest_noc_y,
             noc_write_size});

        // Launch program
        tt::tt_metal::EnqueueProgram(device->command_queue(), program, true);

        // Close device
        pass &= tt::tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        log_error(tt::LogTest, "{}", e.what());
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
