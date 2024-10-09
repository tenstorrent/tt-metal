// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include "llrt.hpp"
#include "hal.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"

#include "jit_build/settings.hpp"

#include "fmt/ranges.h"

#include <unordered_set>
#include <mutex>
#include "dev_msgs.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

ll_api::memory get_risc_binary(string const &path, uint32_t riscv_id, PackSpans pack_spans) {

    static const uint32_t processor_to_fw_base_addr[] = {
        MEM_BRISC_FIRMWARE_BASE,
        MEM_NCRISC_FIRMWARE_BASE,
        MEM_TRISC0_FIRMWARE_BASE,
        MEM_TRISC1_FIRMWARE_BASE,
        MEM_TRISC2_FIRMWARE_BASE,
        eth_l1_mem::address_map::FIRMWARE_BASE,
        MEM_IERISC_FIRMWARE_BASE,
    };

    static struct {
      std::unordered_map<std::string, std::unique_ptr<ll_api::memory>> map;
      std::mutex mutex;
      std::condition_variable cvar;
    } cache;

    std::unique_lock lock(cache.mutex);
    auto [slot, inserted] = cache.map.try_emplace(path);
    if (inserted) {
      // We're the first with PATH. Create and insert.
      lock.unlock();
      auto *ptr = new ll_api::memory(path);

      if (pack_spans == PackSpans::PACK) {
          uint64_t data_start = MEM_LOCAL_BASE;
          uint64_t text_start = processor_to_fw_base_addr[riscv_id];
          ptr->pack_data_into_text(text_start, data_start);
      }

      lock.lock();
      // maps have iterator stability, so SLOT is still valid.
      slot->second = decltype(slot->second)(ptr);
      // We can't wake just those waiting on this slot, so wake them
      // all. Should be a rare event anyway.
      cache.cvar.notify_all();
    } else if (!slot->second) {
        // Someone else is creating the initial entry, wait for them.
        cache.cvar.wait(lock, [=] { return bool(slot->second); });
    }

    return *slot->second.get();
}

// Return the code size in 16 byte units
// This matches what the fw needs for datamovement
// and...squeezes more data into the launch message (2^20=1M)
uint16_t get_binary_code_size16(const ll_api::memory& mem, int riscv_id) {

    uint64_t range_min, range_max;
    switch (riscv_id) {
        case 0:
            range_min = MEM_BRISC_FIRMWARE_BASE;
            range_max = MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE;
            break;
        case 1:
            range_min = MEM_NCRISC_FIRMWARE_BASE;
            range_max = MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE;
            break;
        case 2:
            range_min = MEM_TRISC0_FIRMWARE_BASE;
            range_max = MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC0_FIRMWARE_SIZE;
            break;
        case 3:
            range_min = MEM_TRISC1_FIRMWARE_BASE;
            range_max = MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC1_FIRMWARE_SIZE;
            break;
        case 4:
            range_min = MEM_TRISC2_FIRMWARE_BASE;
            range_max = MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC2_FIRMWARE_SIZE;
            break;
        case 5:
            range_min = eth_l1_mem::address_map::FIRMWARE_BASE;
            range_max = eth_l1_mem::address_map::FIRMWARE_BASE + eth_l1_mem::address_map::FIRMWARE_SIZE;
            break;
        case 6:
            range_min = MEM_IERISC_FIRMWARE_BASE;
            range_max = MEM_IERISC_FIRMWARE_BASE + MEM_IERISC_FIRMWARE_SIZE;
            break;
        default: TT_THROW("Bad riscv_id: {}", riscv_id);
    }

    uint64_t min = std::numeric_limits<decltype(min)>::max();
    uint64_t max = 0;
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {

        uint32_t len_bytes = len_words * sizeof(uint32_t);
        // Only use the addresses within the firmware code range
        if (addr >= range_min && addr + len_bytes <= range_max) {
            if (addr < min) {
                min = addr;
            }
            if (addr + len_bytes > max) {
                max = addr + len_bytes;
            }
        }
    });

    return (uint16_t)((max - min + 15) >> 4);
}

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor

void write_hex_vec_to_core(chip_id_t chip, const CoreCoord &core, const std::vector<uint32_t>& hex_vec, uint64_t addr, bool small_access) {
    // the API is named "write_core", and its overloaded variant is taking (chip, core) pair, ie. it can write to
    // core's L1
    tt::Cluster::instance().write_core(hex_vec.data(), hex_vec.size() * sizeof(uint32_t), tt_cxy_pair(chip, core), addr, small_access);
}

std::vector<uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t sz_bytes) {
    std::vector<uint32_t> read_hex_vec;
    tt::Cluster::instance().read_core(read_hex_vec, sz_bytes, tt_cxy_pair(chip, core), addr);
    return read_hex_vec;
}

CoreCoord logical_core_from_ethernet_core(chip_id_t chip_id, const CoreCoord &physical_core) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return soc_desc.get_logical_ethernet_core_from_physical(physical_core);
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

void launch_erisc_app_fw_on_core(chip_id_t chip, CoreCoord core) {
    llrt::write_hex_vec_to_core(chip, core, {0x1}, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
}

void print_worker_cores(chip_id_t chip_id) {
    std::cout << std::endl << "worker cores: " << std::endl;
    for (const CoreCoord &core : tt::Cluster::instance().get_soc_desc(chip_id).physical_workers) {
        std::cout << core.str() << " ";
    }
    std::cout << std::endl << std::endl;
}

ll_api::memory read_mem_from_core(chip_id_t chip, const CoreCoord &core, const ll_api::memory& mem, uint64_t local_init_addr) {

    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = relocate_dev_addr(addr, local_init_addr);
        tt::Cluster::instance().read_core(&*mem_ptr, len * sizeof(uint32_t), tt_cxy_pair(chip, core), relo_addr);
    });
    return read_mem;
}


uint32_t generate_risc_startup_addr(bool is_eth_core) {
    // Options for handling brisc fw not starting at mem[0]:
    // 1) Program the register for the start address out of reset
    // 2) Encode a jump in crt0 for mem[0]
    // 3) Write the jump to mem[0] here
    // This does #3.  #1 may be best, #2 gets messy (elf files
    // drop any section before .init, crt0 needs ifdefs, etc)
    constexpr uint32_t jal_opcode = 0x6f;
    constexpr uint32_t jal_max_offset = 0x0007ffff;
    uint32_t opcode = jal_opcode;
    uint32_t firmware_base = is_eth_core ? MEM_IERISC_FIRMWARE_BASE : MEM_BRISC_FIRMWARE_BASE;
    assert(firmware_base < jal_max_offset);
    // See riscv spec for offset encoding below
    uint32_t jal_offset_bit_20 = 0;
    uint32_t jal_offset_bits_10_to_1 = (firmware_base & 0x7fe) << 20;
    uint32_t jal_offset_bit_11 = (firmware_base & 0x800) << 9;
    uint32_t jal_offset_bits_19_to_12 = (firmware_base & 0xff000) << 0;
    uint32_t jal_offset =
        jal_offset_bit_20 |
        jal_offset_bits_10_to_1 |
        jal_offset_bit_11 |
        jal_offset_bits_19_to_12;

    return jal_offset | opcode;
}

void program_risc_startup_addr(chip_id_t chip_id, const CoreCoord &core) {
    vector<uint32_t> jump_to_fw;
    jump_to_fw.push_back(generate_risc_startup_addr(is_ethernet_core(core, chip_id)));
    write_hex_vec_to_core(chip_id, core, jump_to_fw, 0);
}

bool test_load_write_read_risc_binary(ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int riscv_id) {
    assert(is_worker_core(core, chip_id) or is_ethernet_core(core, chip_id));

    uint64_t local_init_addr;
    switch (riscv_id) {
        case 0: local_init_addr = MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH; break;
        case 1: local_init_addr = MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH; break;
        case 2: local_init_addr = MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH; break;
        case 3: local_init_addr = MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH; break;
        case 4: local_init_addr = MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH; break;
        case 5: local_init_addr = eth_l1_mem::address_map::FIRMWARE_BASE; break;
        case 6: local_init_addr = MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH; break;
    }

    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size()*sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr = relocate_dev_addr(addr, local_init_addr);

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

bool test_load_write_read_trisc_binary(ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int triscv_id) {

    assert(triscv_id >= 0 and triscv_id <= 2);
    return test_load_write_read_risc_binary(mem, chip_id, core, triscv_id + 2);
}

CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id) {
    return tt::Cluster::instance().get_soc_desc(chip_id).get_preferred_worker_core_for_dram_channel(dram_channel_id);
}

namespace internal_ {

static bool check_if_riscs_on_specified_core_done(chip_id_t chip_id, const CoreCoord &core, int run_state) {
    bool is_eth_core = is_ethernet_core(core, chip_id);
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
    uint64_t go_msg_addr = tt_metal::hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

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
                run == run_state || run == RUN_MSG_DONE,
                "Read unexpected run_mailbox value");
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
        // Continuously polling cores here can cause other host-driven noc transactions (dprint, watcher) to drastically
        // slow down for remote devices. So when debugging with these features, add a small delay to allow other
        // host-driven transactions through.
        if (llrt::OptionsG.get_watcher_enabled() ||
            llrt::OptionsG.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace internal_

}  // namespace llrt

}  // namespace tt
