// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"

#include "build_kernels_for_riscv/build_kernel_options.hpp"

#include <unordered_set>
#include <mutex>
#include <fmt/ranges.h>

#include "tools/cpuprof/cpuprof.h"
#include "dev_msgs.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

namespace fs = std::filesystem;

using std::endl;
using std::move;
using std::string;
using std::to_string;
using std::uint32_t;
using std::unordered_map;
using std::vector;

struct HexNameToMemVectorCache {
    using lock = std::unique_lock<std::mutex>;
    // maps from RisckCacheMapKey to hex file path
    static HexNameToMemVectorCache &inst() {
        static HexNameToMemVectorCache inst_;
        return inst_;
    }

    bool exists(const string &path) {
        lock l(mutex_);
        return cache_.find(path) != cache_.end();
    }
    ll_api::memory &get(const string &path) {
        lock l(mutex_);
        return cache_[path];
    }
    void add(const string &path, ll_api::memory &mem) {
        lock l(mutex_);
        cache_[path] = mem;
    }

    unordered_map<string, ll_api::memory> cache_;
    std::mutex mutex_;
};

// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
ll_api::memory get_risc_binary(string path, chip_id_t chip_id, bool fw_build) {

    std::string binary_key = fmt::format("{}_{}", chip_id, path);
    string path_to_bin = (fw_build ? get_firmware_compile_outpath(chip_id) : get_kernel_compile_outpath(chip_id)) + path;

    if (HexNameToMemVectorCache::inst().exists(binary_key)) {
        // std::cout << "-- HEX2MEM CACHE HIT FOR " << binary_key << std::endl;
        return HexNameToMemVectorCache::inst().get(binary_key);
    }

    fs::path bin_file(path_to_bin);
    if (!fs::exists(bin_file)) {
        string tt_metal_home = string(getenv("TT_METAL_HOME"));
        // try loading from home in case cwd isn't home
        path_to_bin = tt_metal_home + "/" + path_to_bin;
        fs::path bin_file_h(path_to_bin);
        if (!fs::exists(bin_file_h)) {
            std::cout << " Error: " << bin_file.c_str() << " doesn't exist" << endl;
            TT_ASSERT(false);
        }
    }

    std::ifstream hex_istream(path_to_bin);
    ll_api::memory mem(hex_istream);

    // add this path to binary cache
    HexNameToMemVectorCache::inst().add(binary_key, mem);

    return mem;
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
            range_min = MEM_TRISC0_BASE;
            range_max = MEM_TRISC0_BASE + MEM_TRISC0_SIZE;
            break;
        case 3:
            range_min = MEM_TRISC1_BASE;
            range_max = MEM_TRISC1_BASE + MEM_TRISC1_SIZE;
            break;
        case 4:
            range_min = MEM_TRISC2_BASE;
            range_max = MEM_TRISC2_BASE + MEM_TRISC2_SIZE;
            break;
        default: TT_ASSERT("Bad riscv_id: {}", riscv_id);
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

std::vector<std::uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t sz_bytes) {
    vector<std::uint32_t> read_hex_vec;
    tt::Cluster::instance().read_core(read_hex_vec, sz_bytes, tt_cxy_pair(chip, core), addr);
    return read_hex_vec;
}

void write_launch_msg_to_core(chip_id_t chip, CoreCoord core, launch_msg_t *msg) {
    msg->mode = DISPATCH_MODE_HOST;
    TT_ASSERT(sizeof(launch_msg_t) % sizeof(uint32_t) == 0);
    if (static_cast<bool>(msg->enable_erisc)) {
        llrt::write_hex_vec_to_core(chip, core, {0x1}, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
    } else {
        tt::Cluster::instance().write_core(
            (void *)msg, sizeof(launch_msg_t), tt_cxy_pair(chip, core), GET_MAILBOX_ADDRESS_HOST(launch));
    }
}

void print_worker_cores(chip_id_t chip_id) {
    std::cout << std::endl << "worker cores: " << std::endl;
    for (const CoreCoord &core : tt::Cluster::instance().get_soc_desc(chip_id).physical_workers) {
        std::cout << core.str() << " ";
    }
    std::cout << std::endl << std::endl;
}

bool is_worker_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_workers.begin(), soc_desc.physical_workers.end(), core) != soc_desc.physical_workers.end();
}
bool is_ethernet_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_ethernet_cores.begin(), soc_desc.physical_ethernet_cores.end(), core) !=
           soc_desc.physical_ethernet_cores.end();
}

CircularBufferConfigVec create_circular_buffer_config_vector() {
    CircularBufferConfigVec circular_buffer_config_vec(
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG, 0);  // init to 0's
    return circular_buffer_config_vec;
}

void set_config_for_circular_buffer(
    CircularBufferConfigVec &circular_buffer_config_vec,
    uint32_t circular_buffer_index,
    uint32_t addr_in_bytes,
    uint32_t size_in_bytes,
    uint32_t num_pages) {

    uint32_t page_size = size_in_bytes / num_pages;
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index) =
        addr_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 1) =
        size_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 2) = num_pages;
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 3) = page_size >> 4;
}

void write_circular_buffer_config_vector_to_core(chip_id_t chip, const CoreCoord &core, CircularBufferConfigVec circular_buffer_config_vec) {
    write_hex_vec_to_core(chip, core, circular_buffer_config_vec, CIRCULAR_BUFFER_CONFIG_BASE);
}

void write_graph_interpreter_op_info_to_core(chip_id_t chip, const CoreCoord &core, op_info_t op_info, int op_idx) {
    vector<uint32_t> op_info_vec = {
        op_info.op_code,
        op_info.cb_in0_id,
        op_info.cb_in1_id,
        op_info.cb_out_id,
        op_info.pop0,
        op_info.pop1,
        op_info.unary};
    uint32_t offset = op_info_vec.size() * sizeof(uint32_t) * op_idx;

    write_hex_vec_to_core(chip, core, op_info_vec, OP_INFO_BASE_ADDR + offset);
}

ll_api::memory read_mem_from_core(chip_id_t chip, const CoreCoord &core, const ll_api::memory& mem, uint64_t local_init_addr) {

    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = relocate_dev_addr(addr, local_init_addr);
        tt::Cluster::instance().read_core(&*mem_ptr, len * sizeof(uint32_t), tt_cxy_pair(chip, core), relo_addr);
    });
    return read_mem;
}

void program_brisc_startup_addr(chip_id_t chip_id, const CoreCoord &core) {
    // Options for handling brisc fw not starting at mem[0]:
    // 1) Program the register for the start address out of reset
    // 2) Encode a jump in crt0 for mem[0]
    // 3) Write the jump to mem[0] here
    // This does #3.  #1 may be best, #2 gets messy (elf files
    // drop any section before .init, crt0 needs ifdefs, etc)
    vector<uint32_t> jump_to_fw;
    constexpr uint32_t jal_opcode = 0x6f;
    constexpr uint32_t jal_max_offset = 0x0007ffff;
    uint32_t opcode = jal_opcode;
    assert(MEM_BRISC_FIRMWARE_BASE < jal_max_offset);
    // See riscv spec for offset encoding below
    uint32_t jal_offset_bit_20 = 0;
    uint32_t jal_offset_bits_10_to_1 = (MEM_BRISC_FIRMWARE_BASE & 0x7fe) << 20;
    uint32_t jal_offset_bit_11 = (MEM_BRISC_FIRMWARE_BASE & 0x800) << 9;
    uint32_t jal_offset_bits_19_to_12 = (MEM_BRISC_FIRMWARE_BASE & 0xff000) << 0;
    uint32_t jal_offset =
        jal_offset_bit_20 |
        jal_offset_bits_10_to_1 |
        jal_offset_bit_11 |
        jal_offset_bits_19_to_12;
    jump_to_fw.push_back(jal_offset | opcode);
    write_hex_vec_to_core(chip_id, core, jump_to_fw, 0);
}

bool test_load_write_read_risc_binary(ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int riscv_id) {
    assert(is_worker_core(core, chip_id) or is_ethernet_core(core, chip_id));

    uint64_t local_init_addr;
    switch (riscv_id) {
        case 0: local_init_addr = MEM_BRISC_INIT_LOCAL_L1_BASE; break;
        case 1: local_init_addr = MEM_NCRISC_INIT_LOCAL_L1_BASE; break;
        case 2: local_init_addr = MEM_TRISC0_INIT_LOCAL_L1_BASE; break;
        case 3: local_init_addr = MEM_TRISC1_INIT_LOCAL_L1_BASE; break;
        case 4: local_init_addr = MEM_TRISC2_INIT_LOCAL_L1_BASE; break;
        case 5: local_init_addr = eth_l1_mem::address_map::FIRMWARE_BASE; break;
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
    if (is_ethernet_core(core, chip_id)) {
        const auto &readback_vec =
            read_hex_vec_from_core(chip_id, core, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG, sizeof(uint32_t));
        return (readback_vec[0] == 0);
    } else {
        std::function<bool(uint64_t)> get_mailbox_is_done = [&](uint64_t run_mailbox_address) {
            constexpr int RUN_MAILBOX_BOGUS = 3;
            std::vector<uint32_t> run_mailbox_read_val = {RUN_MAILBOX_BOGUS};
            // read a single uint32_t even though launch.run is smaller than that
            run_mailbox_read_val = read_hex_vec_from_core(chip_id, core, run_mailbox_address & ~0x3, sizeof(uint32_t));
            uint8_t run = run_mailbox_read_val[0] >> (8 * (offsetof(launch_msg_t, run) & 3));
            if (run != run_state && run != RUN_MSG_DONE) {
                fprintf(
                    stderr,
                    "Read unexpected run_mailbox value: 0x%x (expected 0x%x or 0x%x)\n",
                    run,
                    run_state,
                    RUN_MSG_DONE);
                TT_FATAL(run_mailbox_read_val[0] == run_state || run_mailbox_read_val[0] == RUN_MSG_DONE);
            }

            return run == RUN_MSG_DONE;
        };

        return get_mailbox_is_done(GET_MAILBOX_ADDRESS_HOST(launch.run));
    }
}

void wait_until_cores_done(chip_id_t device_id,
                           int run_state,
                           std::unordered_set<CoreCoord>& not_done_phys_cores) {

    // poll the cores until the set of not done cores is empty
    int loop_count = 0;
    while (!not_done_phys_cores.empty()) {
        // Print not-done cores
        if (loop_count % 20 == 0) {
            string not_done_cores_str = "Not done phys cores: ";
            for (const auto &core : not_done_phys_cores) {
                not_done_cores_str += (core.str() + " ");
            }
            log_debug(tt::LogMetal, not_done_cores_str.c_str());
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
    }
}

}  // namespace internal_

}  // namespace llrt

}  // namespace tt
