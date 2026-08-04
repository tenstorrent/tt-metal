// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _DEPLOYMENT_COMMON_H
#define _DEPLOYMENT_COMMON_H

#include "tt_metal/api/tt-metalium/hal.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp"

#define TEST_PARAM(type, var, initial, envvar) \
    type var = (initial);                      \
    get_env((envvar), &var)

struct l1_allocator {
    uint32_t start;
    uint32_t end;
};

std::string pci_bdf_for_device_id(uint32_t device_id);
std::string trim_copy(std::string s);
std::string read_text_file_trimmed(const std::string& path);
std::string get_ubb_id_str(uint32_t chip_id);
std::vector<std::string> get_chip_physical_locations();

#define ROUND_UP(x, a) ((((x) + (a) - 1) / (a)) * (a))
#define ROUND_DOWN(x, a) (((x) / (a)) * (a))

#define ALIGNMENT 64  // TODO

static inline struct l1_allocator new_tensix_allocator() {
    using namespace tt::tt_metal;

    uint32_t start = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    return (struct l1_allocator){
        .start = start,
        .end = end,
    };
}

static inline struct l1_allocator new_erisc_allocator() {
    using namespace tt::tt_metal;

    uint32_t start =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    return (struct l1_allocator){
        .start = start,
        .end = end,
    };
}

[[maybe_unused]]
static inline uint32_t l1_alloc(struct l1_allocator* alloc, uint32_t size, uint32_t alignment = ALIGNMENT) {
    uint32_t start = ROUND_UP(alloc->start, alignment);

    TT_FATAL(start + size <= alloc->end, "Couldn't allocate in L1");

    alloc->start = start + size;

    return start;
}

[[maybe_unused]]
static uint32_t read_l1_u32(tt::tt_metal::IDevice* const device, const tt::tt_metal::CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->worker_core_from_logical_core(core), l1_addr, sizeof(uint32_t));

    return delta_vec[0];
}

[[maybe_unused]]
static uint64_t read_l1_u64(tt::tt_metal::IDevice* const device, const tt::tt_metal::CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->worker_core_from_logical_core(core), l1_addr, 2 * sizeof(uint32_t));

    return (uint64_t)delta_vec[0] | ((uint64_t)delta_vec[1] << 32);
}

[[maybe_unused]]
static uint32_t read_eth_l1_u32(tt::tt_metal::IDevice* const device, const tt::tt_metal::CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->ethernet_core_from_logical_core(core), l1_addr, sizeof(uint32_t));

    return delta_vec[0];
}

[[maybe_unused]]
static uint64_t read_eth_l1_u64(tt::tt_metal::IDevice* const device, const tt::tt_metal::CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->ethernet_core_from_logical_core(core), l1_addr, 2 * sizeof(uint32_t));

    return (uint64_t)delta_vec[0] | ((uint64_t)delta_vec[1] << 32);
}

extern std::atomic_bool g_stop_requested;
extern std::atomic_bool g_stop_message_printed;

void handle_sigint(int);

class SignalGuard {
private:
    sighandler_t prev;
    int signum;

public:
    SignalGuard(int sig, sighandler_t handler) {
        signum = sig;
        prev = signal(sig, handler);
    }
    ~SignalGuard() { signal(signum, prev); }
};

[[maybe_unused]]
static bool bandwidth_check(
    tt::tt_metal::IDevice* const send_device,
    const tt::tt_metal::CoreCoord& send_core,
    uint32_t send_delta_addr,
    uint64_t total_transferred,
    double threshold) {
    /* ==================== */
    uint64_t delta = read_l1_u64(send_device, send_core, send_delta_addr);
    double deltas = delta / 1.35e9; /* Assuming fixed max frequency */
    double bandwidth = total_transferred / 1e9 / deltas;
    log_info(tt::LogTest, "      Bandwidth {:.3f} GB/s, {:.3f} ms", bandwidth, deltas * 1000);

    bool pass = bandwidth >= threshold;
    if (!pass) {
        log_critical(tt::LogTest, "      Expected at least: {} GB/s, got {:.2f} GB/s", threshold, bandwidth);
    }

    return pass;
}

[[maybe_unused]]
static bool dram_data_check(
    tt::tt_metal::IDevice* const recv_device,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    std::vector<uint32_t>& inputs) {
    /* ==================== */
    uint64_t total_transferred = dram_end_addr - dram_start_addr;
    std::vector<uint32_t> outputs(total_transferred / sizeof(uint32_t));

    tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
        recv_device, dram_bank_id, dram_start_addr, total_transferred, outputs);
    log_info(tt::LogTest, "      Read {} bytes", outputs.size() * sizeof(uint32_t));
    TT_FATAL(inputs.size() == outputs.size(), "Input and output vector sizes must match");

    uint64_t total_mismatches = 0;
    for (long i = 0; i < inputs.size(); i++) {
        if (inputs[i] != outputs[i]) {
            if (!total_mismatches) {
                log_critical(
                    tt::LogTest,
                    "      Input and output data don't match starting at: {:x}",
                    dram_start_addr + i * sizeof(uint32_t));
            }
            total_mismatches++;
            // log_critical(tt::LogTest, "      Input and output data don't match at {:x}: {:x} {:x}", i, inputs[i],
            // 		outputs[i]);
        }
    }
    if (total_mismatches) {
        log_critical(tt::LogTest, "      Total mismatches: {} words", total_mismatches);
    }

    return !total_mismatches;
}

[[maybe_unused]]
static void get_env(const char* varname, uint32_t* var) {
    const char* p = std::getenv(varname);
    if (p) {
        std::string snum(p);
        *var = stoul(snum);
        log_info(tt::LogTest, "{} {}", varname, *var);
    }
}

#endif /* _DEPLOYMENT_COMMON_H */
