// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <dev_msgs.h>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <logger.hpp>
#include <rtoptions.hpp>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "hal_types.hpp"
#include "llrt.hpp"
#include "llrt/hal.hpp"
#include "metal_soc_descriptor.h"
// #include <umd/device/driver_atomics.h> - This should be included as it is used here, but the file is missing include
// guards
#include <umd/device/tt_core_coordinates.h>

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

ll_api::memory const& get_risc_binary(
    string const& path,
    ll_api::memory::Loading loading) {
    static struct {
      std::unordered_map<std::string, std::unique_ptr<ll_api::memory const>> map;
      std::mutex mutex;
      std::condition_variable cvar;
    } cache;

    std::unique_lock lock(cache.mutex);
    auto [slot, inserted] = cache.map.try_emplace(path);
    ll_api::memory const* ptr = nullptr;
    if (inserted) {
      // We're the first with PATH. Create and insert.
      lock.unlock();
      ptr = new ll_api::memory(path, loading);

      lock.lock();
      // maps have iterator stability, so SLOT is still valid.
      slot->second = decltype(slot->second)(ptr);
      // We can't wake just those waiting on this slot, so wake them
      // all. Should be a rare event anyway.
      cache.cvar.notify_all();
    } else {
        if (!slot->second) {
            // Someone else is creating the initial entry, wait for them.
            cache.cvar.wait(lock, [=] { return bool(slot->second); });
        }
        ptr = slot->second.get();
        TT_ASSERT(ptr->get_loading() == loading);
    }

    return *ptr;
}

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor

void write_hex_vec_to_core(chip_id_t chip, const CoreCoord &core, tt::stl::Span<const uint8_t> hex_vec, uint64_t addr, bool small_access) {
    // the API is named "write_core", and its overloaded variant is taking (chip, core) pair, ie. it can write to
    // core's L1
    tt::Cluster::instance().write_core(hex_vec.data(), hex_vec.size(), tt_cxy_pair(chip, core), addr, small_access);
}

std::vector<uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t sz_bytes) {
    std::vector<uint32_t> read_hex_vec;
    tt::Cluster::instance().read_core(read_hex_vec, sz_bytes, tt_cxy_pair(chip, core), addr);
    return read_hex_vec;
}

CoreCoord logical_core_from_ethernet_core(chip_id_t chip_id, const CoreCoord &ethernet_core) {
    return tt::Cluster::instance().get_logical_ethernet_core_from_virtual(chip_id, ethernet_core);
}

void write_launch_msg_to_core(chip_id_t chip, const CoreCoord core, launch_msg_t *msg, go_msg_t *go_msg,  uint64_t base_addr, bool send_go) {

    msg->kernel_config.mode = DISPATCH_MODE_HOST;

    uint64_t launch_addr = base_addr + offsetof(launch_msg_t, kernel_config);
    // TODO: Get this from the hal. Need to modify the write_launch_msg_to_core API to get the LM and Go signal addr from the hal.
    uint64_t go_addr = base_addr + sizeof(launch_msg_t) * launch_msg_buffer_num_entries;

    tt::Cluster::instance().write_core((void *)&msg->kernel_config, sizeof(kernel_config_msg_t), tt_cxy_pair(chip, core), launch_addr);
    tt_driver_atomics::sfence();
    if (send_go) {
        tt::Cluster::instance().write_core(go_msg, sizeof(go_msg_t), tt_cxy_pair(chip, core), go_addr);
    }
}

void print_worker_cores(chip_id_t chip_id) {
    std::cout << std::endl << "worker cores: " << std::endl;
    for (const CoreCoord& core :
         tt::Cluster::instance().get_soc_desc(chip_id).get_cores(CoreType::TENSIX, CoordSystem::PHYSICAL)) {
        std::cout << core.str() << " ";
    }
    std::cout << std::endl << std::endl;
}

ll_api::memory read_mem_from_core(chip_id_t chip, const CoreCoord &core, const ll_api::memory& mem, uint64_t local_init_addr) {

    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = tt::tt_metal::hal_ref.relocate_dev_addr(addr, local_init_addr);
        tt::Cluster::instance().read_core(&*mem_ptr, len * sizeof(uint32_t), tt_cxy_pair(chip, core), relo_addr);
    });
    return read_mem;
}

bool test_load_write_read_risc_binary(
    ll_api::memory const& mem,
    chip_id_t chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx) {
    assert(tt::Cluster::instance().is_worker_core(core, chip_id) or tt::Cluster::instance().is_ethernet_core(core, chip_id));

    uint64_t local_init_addr =
        tt::tt_metal::hal_ref.get_jit_build_config(core_type_idx, processor_class_idx, processor_type_idx)
            .local_init_addr;
    auto core_type = tt::tt_metal::hal_ref.get_programmable_core_type(core_type_idx);

    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size()*sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr = tt::tt_metal::hal_ref.relocate_dev_addr(addr, local_init_addr);

        tt::Cluster::instance().write_core(&*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), relo_addr);
    });

    log_debug(tt::LogLLRuntime, "wrote hex to core {}", core.str().c_str());

    if (std::getenv("TT_METAL_KERNEL_READBACK_ENABLE") != nullptr) {
        tt::Cluster::instance().l1_barrier(chip_id);
        ll_api::memory read_mem = read_mem_from_core(chip_id, core, mem, local_init_addr);
        log_debug(tt::LogLLRuntime, "read hex back from the core");
        return mem == read_mem;
    }

    return true;
}

void write_binary_to_address(ll_api::memory const& mem, chip_id_t chip_id, const CoreCoord& core, uint32_t address) {
    log_debug(tt::LogLLRuntime, "vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        tt::Cluster::instance().write_core(&*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), address);
    });
}

CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id) {
    return tt::Cluster::instance().get_soc_desc(chip_id).get_preferred_worker_core_for_dram_view(dram_channel_id);
}

namespace internal_ {

static bool check_if_riscs_on_specified_core_done(chip_id_t chip_id, const CoreCoord &core, int run_state) {
    bool is_eth_core = tt::Cluster::instance().is_ethernet_core(core, chip_id);
    bool is_active_eth_core = false;
    bool is_inactive_eth_core = false;

        // Determine whether an ethernet core is active or idle. Their host handshake interfaces are different.
    if (is_eth_core) {
        auto active_eth_cores =  tt::Cluster::instance().get_active_ethernet_cores(chip_id);
        auto inactive_eth_cores =  tt::Cluster::instance().get_inactive_ethernet_cores(chip_id);
        is_active_eth_core = active_eth_cores.find(logical_core_from_ethernet_core(chip_id, core)) != active_eth_cores.end();
        is_inactive_eth_core = inactive_eth_cores.find(logical_core_from_ethernet_core(chip_id, core)) != inactive_eth_cores.end();
        //we should not be operating on any reserved cores here.
        assert(is_active_eth_core or is_inactive_eth_core);
    }

    tt_metal::HalProgrammableCoreType dispatch_core_type =  is_active_eth_core ? tt_metal::HalProgrammableCoreType::ACTIVE_ETH :
        is_inactive_eth_core ? tt_metal::HalProgrammableCoreType::IDLE_ETH : tt_metal::HalProgrammableCoreType::TENSIX;
    uint64_t go_msg_addr = tt_metal::hal_ref.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

    auto get_mailbox_is_done = [&](uint64_t go_msg_addr) {
        constexpr int RUN_MAILBOX_BOGUS = 3;
        std::vector<uint32_t> run_mailbox_read_val = {RUN_MAILBOX_BOGUS};
        run_mailbox_read_val = read_hex_vec_from_core(chip_id, core, go_msg_addr & ~0x3, sizeof(uint32_t));
        go_msg_t* core_status = (go_msg_t*)(run_mailbox_read_val.data());
        uint8_t run = core_status->signal;
        if (run != run_state && run != RUN_MSG_DONE) {
            fprintf(
                stderr,
                "Read unexpected run_mailbox value: 0x%x (expected 0x%x or 0x%x)\n",
                run,
                run_state,
                RUN_MSG_DONE);
            TT_FATAL(
                run == run_state || run == RUN_MSG_DONE, "Read unexpected run_mailbox value from core {}", core.str());
        }

        return run == RUN_MSG_DONE;
    };
    return get_mailbox_is_done(go_msg_addr);
}

void wait_until_cores_done(
    chip_id_t device_id, int run_state, std::unordered_set<CoreCoord> &not_done_phys_cores, int timeout_ms) {
    // poll the cores until the set of not done cores is empty
    int loop_count = 1;
    auto start = std::chrono::high_resolution_clock::now();
    bool is_simulator = llrt::RunTimeOptions::get_instance().get_simulator_enabled();

    if (is_simulator) timeout_ms = 0;
    while (!not_done_phys_cores.empty()) {
        if (timeout_ms > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                std::string cores = fmt::format("{}", fmt::join(not_done_phys_cores, ", "));
                TT_THROW(
                    "Device {}: Timeout ({} ms) waiting for physical cores to finish: {}.",
                    device_id,
                    timeout_ms,
                    cores);
            }
        }

        // Print not-done cores
        if (loop_count % 1000 == 0) {
            log_debug(tt::LogMetal, "Not done phys cores: {}", fmt::join(not_done_phys_cores, " "));
            usleep(100000);
        }

        for (auto it = not_done_phys_cores.begin(); it != not_done_phys_cores.end(); ) {
            const auto &phys_core = *it;

            bool is_done = llrt::internal_::check_if_riscs_on_specified_core_done(device_id, phys_core, run_state);

            if (is_done) {
                log_debug(tt::LogMetal, "Phys cores just done: {}", phys_core.str());
                it = not_done_phys_cores.erase(it);
            } else {
                ++it;
            }
        }
        loop_count++;

        // Continuously polling cores on simulator can cause it to run much slower than real hardware.
        if (is_simulator)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Continuously polling cores here can cause other host-driven noc transactions (dprint, watcher) to drastically
        // slow down for remote devices. So when debugging with these features, add a small delay to allow other
        // host-driven transactions through.
        if (llrt::RunTimeOptions::get_instance().get_watcher_enabled() ||
            llrt::RunTimeOptions::get_instance().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace internal_

}  // namespace llrt

}  // namespace tt
