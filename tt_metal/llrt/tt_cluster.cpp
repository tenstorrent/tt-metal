// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_cluster.hpp"
#include "third_party/umd/device/tt_silicon_driver_common.hpp"
#include "device_data.hpp"
#include <immintrin.h>
#include <string>
#include <iomanip>
#include <iostream>
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/debug_print_common.h"
#include "rtoptions.hpp"
#include "watcher.hpp"

using std::to_string;
using std::cout;
using std::endl;

#ifdef ARCH_GRAYSKULL
static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB  = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr unsigned int DYNAMIC_TLB_BASE_INDEX    = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;

#else
static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB  = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;
#endif


std::chrono::seconds tt_cluster::get_device_timeout() {
    int device_timeout = 3600; // seconds
    const char* timeout_override = std::getenv("TT_METAL_BACKEND_TIMEOUT");
    if (timeout_override) {
        device_timeout = atoi(timeout_override);
    }
    return std::chrono::seconds{device_timeout};
}

std::chrono::seconds tt_cluster::get_device_duration() {
    high_resolution_clock::time_point device_current_time;
    device_current_time = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(device_current_time - device_reset_time);
    return duration;
}

int tt_cluster::get_num_chips() {
    return get_all_chips().size();
}

std::unordered_set<chip_id_t> tt_cluster::get_all_chips() {
    return ndesc->get_all_chips();
}

std::set<chip_id_t> tt_cluster::get_all_mmio_chips() {
    return device->get_target_mmio_device_ids();
}

void tt_cluster::dump_wall_clock_mailbox(std::string output_dir) {
    bool is_output_dir_populated = output_dir.find("tt_build") != std::string::npos;
    if (is_output_dir_populated) {
        for (auto device_id : target_device_ids){
            string output_path = output_dir + "/wall_clock_device_";
            output_path += to_string(device_id) + ".yaml";
            tt::log_info(tt::LogLLRuntime, "Reading wall-clock mailbox for device {}, output yaml path {}", device_id, output_path);
            std::ofstream output_file(output_path);
            const int mailbox_base_addr = MEM_WALL_CLOCK_MAILBOX_ADDRESS;
            const int num_mailbox_32_regs = 4;
            const int mailbox_size = num_mailbox_32_regs * 4;
            for (auto &worker_core : get_soc_desc(device_id).workers) {
                int core_x = worker_core.x;
                int core_y = worker_core.y;
                std::string core_id = std::to_string(core_x) + "-" + std::to_string(core_y);
                output_file << core_id << ":" << std::endl;

                std::vector<uint32_t> mailbox_events;
                read_dram_vec(mailbox_events, tt_cxy_pair(device_id, core_x, core_y), mailbox_base_addr, mailbox_size);
                assert(mailbox_events.size() == num_mailbox_32_regs);
                uint64_t start_time = (uint64_t(mailbox_events[1]) << 32) + mailbox_events[0];
                uint64_t end_time = (uint64_t(mailbox_events[3]) << 32) + mailbox_events[2];
                output_file << "        " << std::left << std::setw(12) << "start: " << start_time << std::endl;
                output_file << "        " << std::left << std::setw(12) << "end: " << end_time << std::endl;
                output_file << "        " << std::left << std::setw(12) << "runtime: " << end_time - start_time
                            << std::endl;
            }
            output_file.close();
        }
    }
}

// clean up bad system resource state that may be carried over
void tt_cluster::clean_system_resources() {
    TT_ASSERT(device != nullptr ,  "Device not initialized, make sure compile is done before running!");
    device->clean_system_resources();
}

void tt_cluster::verify_eth_fw() {
    const std::unordered_set<chip_id_t> &all_chips = ndesc->get_all_chips();
    for (const chip_id_t &chip : all_chips) {
        std::vector<uint32_t> mem_vector;
        std::vector<uint32_t> fw_versions;

        for (CoreCoord &eth_core : get_soc_desc(chip).ethernet_cores) {
            read_dram_vec(mem_vector, tt_cxy_pair(chip, eth_core), eth_l1_mem::address_map::FW_VERSION_ADDR, 4);
            fw_versions.push_back(mem_vector.at(0));
        }
        verify_sw_fw_versions(chip, SW_VERSION, fw_versions);
    }
}

int extract_chip_id_from_sdesc_path(std::filesystem::path sdesc_path) {
    string file = sdesc_path.filename().string();
    return atoi(file.substr(0, file.find(".")).c_str());
}

std::unordered_map<chip_id_t, metal_SocDescriptor> get_metal_desc_from_tt_desc(const std::unordered_map<chip_id_t, tt_SocDescriptor>& input) {
    std::unordered_map<chip_id_t, metal_SocDescriptor> rval = {};
    for(const auto it : input) {
        rval.emplace(it.first, metal_SocDescriptor(it.second));
    }
    return rval;
}

void tt_cluster::open_device(
    const tt::ARCH &arch,
    const TargetDevice &target_type,
    const std::set<chip_id_t> &target_devices,
    const std::string &sdesc_path,
    const std::string &ndesc_path,
    const bool &skip_driver_allocs) {
#ifdef ARCH_GRAYSKULL
    tt::log_assert(arch == tt::ARCH::GRAYSKULL, "Arch={} doesn't match compile-time build for GRAYSKULL", get_string(arch));
#endif
#ifdef ARCH_WORMHOLE
    tt::log_assert((arch == tt::ARCH::WORMHOLE_B0) || (arch == tt::ARCH::WORMHOLE), "Arch={} doesn't match compile-time build for WORMHOLE", get_string(arch));
#endif
    target_device_ids = target_devices;

    if (ndesc_path == "") {
        ndesc = tt_cluster_description::create_for_grayskull_cluster(target_devices);
    } else {
        ndesc = tt_cluster_description::create_from_yaml(ndesc_path);
    }
    tt::log_info(tt::LogDevice, "Network descriptor loaded {}", ndesc_path);

    // TT_ASSERT(sdesc_per_chip.size());
    TT_ASSERT(ndesc != nullptr);

    if (target_type == TargetDevice::Silicon) {
        // This is the target/desired number of mem channels per arch/device. Silicon driver will attempt to open
        // this many hugepages as channels, and assert if workload uses more than available.
        uint32_t num_host_mem_ch_per_mmio_device = 1;
        std::unordered_map<std::string, std::int32_t> dynamic_tlb_config = {};
        // This will remove harvested rows from the soc descriptor
        const bool perform_harvesting = true;

        device = std::make_unique<tt_SiliconDevice>(sdesc_path, ndesc_path, target_device_ids, num_host_mem_ch_per_mmio_device, dynamic_tlb_config, skip_driver_allocs, perform_harvesting);

        device->set_driver_host_address_params(host_address_params);
        device->set_driver_eth_interface_params(eth_interface_params);
    } else if (target_type == TargetDevice::Versim) {
        device = std::make_unique<tt_VersimDevice>(sdesc_path, ndesc_path);
    }
    device->set_device_l1_address_params(l1_fw_params);
    type = target_type;
    TT_ASSERT(type == TargetDevice::Versim or type == TargetDevice::Silicon);
    sdesc_per_chip = get_metal_desc_from_tt_desc(device->get_virtual_soc_descriptors());

    if (device) {
        // if (get_num_chips() != tt::MAX_AVAILABLE_CHIPS) {
        //     if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        //         TT_ASSERT(device->get_number_of_chips_in_cluster() >= 1, "Must have at least one detected chip available on device!");
        //     } else {
        //         //TT_ASSERT(get_num_chips() <= device->get_number_of_chips_in_cluster(), "Requested number of chips through machine descriptor is bigger than number of chips available on device!");
        //     }
        // }
    }
}

std::map<int, int> tt_cluster::get_all_device_aiclks(){
    return device->get_clocks();
}

int tt_cluster::get_device_aiclk(const chip_id_t &chip_id) {
    if (target_device_ids.find(chip_id) != target_device_ids.end()) {
        return device->get_clocks().at(chip_id);
    }
    return 0;
}

// void tt_cluster::set_power_state(tt_DevicePowerState device_state) {
//     std::stringstream ss;
//     ss << "Setting silicon device power state to " << device_state;
//     tt::log_info("{}", ss.str());
//     device->set_power_state(device_state);
// }

void tt_cluster::reset_debug_print_server_buffers() {
    for (const int device_id : this->target_device_ids) {
        auto workers = get_soc_desc(device_id).workers;
        for (const CoreCoord &core : workers)
        for (int hart_id = 0; hart_id < 5; hart_id++) { // TODO(AP): must match DPRINT_NHARTS, magic
            // compute the buffer address for the requested hart
            uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

            // The way this works is we first initialize all dprint buffers
            // for all cores and all harts to this magic number.
            // This way the device DPRINT will know that this core hasn't been initialized
            // from tt_start_debug_print_server
            // If we didn't have this mechanism them DPRINT in the kernel would have no way of knowing
            // Whether the host is polling it's buffer and flushing it.
            // If kernel code (in debug_print.h) detects that this magic value is present
            // Then it simply skips the print entirely. It prevents the kernel code
            // from hanging in a stall waiting for the host to flush the buffer
            // and removes the requirement that the host must listen on device's buffer.
            vector<uint32_t> initbuf = { uint32_t(DEBUG_PRINT_SERVER_DISABLED_MAGIC) };
            write_dram_vec(initbuf, {uint32_t(device_id), core}, base_addr);
        }
    }
}

#ifdef ARCH_WORMHOLE
std::int32_t get_static_tlb_index(CoreCoord target) {
    bool is_eth_location = std::find(std::cbegin(DEVICE_DATA.ETH_LOCATIONS), std::cend(DEVICE_DATA.ETH_LOCATIONS), target) != std::cend(DEVICE_DATA.ETH_LOCATIONS);
    bool is_tensix_location = std::find(std::cbegin(DEVICE_DATA.T6_X_LOCATIONS), std::cend(DEVICE_DATA.T6_X_LOCATIONS), target.x) != std::cend(DEVICE_DATA.T6_X_LOCATIONS) &&
                              std::find(std::cbegin(DEVICE_DATA.T6_Y_LOCATIONS), std::cend(DEVICE_DATA.T6_Y_LOCATIONS), target.y) != std::cend(DEVICE_DATA.T6_Y_LOCATIONS);
    // implementation migrated from wormhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/wormhole.py` from tensix repo (t6py-wormhole-bringup branch)

    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }

    if (is_eth_location) {
        if (target.y == 6) {
            target.y = 1;
        }

        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        int flat_index = target.y * 8 + target.x;
        int tlb_index = flat_index;
        return tlb_index;

    } else if (is_tensix_location) {
        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        if (target.y >= 6) {
            target.y -= 1;
        }
        target.y -= 1;

        int flat_index = target.y * 8 + target.x;

        // All 80 get single 1MB TLB.
        int tlb_index = DEVICE_DATA.ETH_LOCATIONS.size() + flat_index;

        return tlb_index;
    } else {
        return -1;
    }
}
#endif

#ifdef ARCH_GRAYSKULL
std::int32_t get_static_tlb_index(CoreCoord target) {
    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }
    int flat_index = target.y * DEVICE_DATA.GRID_SIZE_X + target.x;
    if (flat_index == 0) {
        return -1;
    }
    return flat_index;
}
#endif

void tt_cluster::configure_static_tlbs(const std::uint32_t& chip) {
    auto sdesc = get_soc_desc(chip);
    auto statically_mapped_cores = sdesc.workers;
    statically_mapped_cores.insert(statically_mapped_cores.end(), sdesc.ethernet_cores.begin(), sdesc.ethernet_cores.end());
    std::int32_t address = 0;

    // Setup static TLBs for all worker cores
    for(auto& core : statically_mapped_cores) {
        auto tlb_index = get_static_tlb_index(core);
        device->configure_tlb(chip, core, tlb_index, address);
    }
    // Setup static TLBs for MMIO mapped data space
    uint64_t peer_dram_offset = DEVICE_DATA.DRAM_CHANNEL_0_PEER2PEER_REGION_START;
    for (uint32_t tlb_id = DYNAMIC_TLB_BASE_INDEX; tlb_id < DYNAMIC_TLB_BASE_INDEX + DYNAMIC_TLB_COUNT; tlb_id++) {
        device->configure_tlb(chip, CoreCoord(DEVICE_DATA.DRAM_CHANNEL_0_X, DEVICE_DATA.DRAM_CHANNEL_0_Y), tlb_id, peer_dram_offset);
        // Align address space of 16MB TLB to 16MB boundary
        peer_dram_offset += DEVICE_DATA.DYNAMIC_TLB_16M_SIZE;
    }
    device->setup_core_to_tlb_map([] (CoreCoord core) {return get_static_tlb_index(core);});
}

void tt_cluster::start_device(const tt_device_params &device_params) {
    TT_ASSERT(sdesc_per_chip.size(), "Descriptor must be loaded. Try open_device()");
    TT_ASSERT(device != nullptr ,  "Device not initialized, make sure compile is done before running!");

    if(type == TargetDevice::Silicon && device_params.init_device) {
        for(auto& device_id : device->get_target_mmio_device_ids()) {
            configure_static_tlbs(device_id);
        }
        //tt::tlb_config::activate_static_tlbs(device);
    }

    device->start_device(device_params);

    device_reset_time = high_resolution_clock::now();
}

void tt_cluster::close_device() {
    if (device) {
        device->close_device();
        device.reset();
    }
    sdesc_per_chip.clear();
}

void tt_cluster::assert_risc_reset(const chip_id_t &chip) {
    device->assert_risc_reset(chip);
}

void tt_cluster::set_remote_tensix_risc_reset(const tt_cxy_pair &core, const TensixSoftResetOptions &soft_resets) {
    if (type == TargetDevice::Silicon) {
        auto valid = soft_resets & ALL_TENSIX_SOFT_RESET;

        std::vector<uint32_t> vec = {(std::underlying_type<TensixSoftResetOptions>::type) valid};
        write_dram_vec(vec, core, 0xFFB121B0 /* Should get this value from the device */);
        _mm_sfence();
    } else {
        if ((soft_resets == TENSIX_DEASSERT_SOFT_RESET) or (soft_resets == TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER)) {
            device->deassert_risc_reset();
        } else if (soft_resets == TENSIX_ASSERT_SOFT_RESET) {
            device->assert_risc_reset();
        }
    }
}

void tt_cluster::deassert_risc_reset(const chip_id_t &target_device_id, bool start_stagger) {
    if (type == TargetDevice::Versim) {
        // Not running silicon multichip test
        device->deassert_risc_reset(*this->target_device_ids.begin());
    } else if (type == TargetDevice::Silicon) {
        log_debug(tt::LogLLRuntime, "Stagger start : {}", start_stagger);
        TT_ASSERT(not start_stagger, "UMD currently does not support staggered deassert of RISC reset");
        device->deassert_risc_reset(target_device_id);
    }
    device_reset_time = high_resolution_clock::now();
    deasserted_risc_reset = false;
}

inline uint64_t get_sys_addr(uint32_t chip_x, uint32_t chip_y, uint32_t noc_x, uint32_t noc_y, uint64_t offset) {
    uint64_t result = chip_y;
    uint64_t noc_addr_local_bits_mask = (1UL << NOC_ADDR_LOCAL_BITS) - 1;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= chip_x;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_y;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_x;
    result <<= NOC_ADDR_LOCAL_BITS;
    result |= (noc_addr_local_bits_mask & offset);
    return result;
}

void tt_cluster::write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access)
{
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    metal_SocDescriptor& desc_to_use = get_soc_desc(chip_id);
    tt::log_assert(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size()
    );
    TT_ASSERT(d_subchannel < desc_to_use.dram_cores.at(d_chan).size(), "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    write_dram_vec(vec, dram_core, addr + offset, small_access);
}

void tt_cluster::read_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, uint32_t size, bool small_access)
{
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    metal_SocDescriptor& desc_to_use = get_soc_desc(chip_id);
    tt::log_assert(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size()
    );
    TT_ASSERT(d_subchannel < desc_to_use.dram_cores.at(d_chan).size(), "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    read_dram_vec(vec, dram_core, addr + offset, size, small_access);
}

void tt_cluster::write_dram_vec(const std::uint32_t *mem_ptr, uint32_t len, tt_cxy_pair dram_core, uint64_t addr, bool small_access)
{
    int chip_id = dram_core.chip;
    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_write(get_soc_desc(chip_id), {dram_core.x, dram_core.y}, addr, len * sizeof(uint32_t));
    }
    device->write_to_device(mem_ptr, len, dram_core, addr, "LARGE_WRITE_TLB");
    if (device->get_target_remote_device_ids().find(dram_core.chip) != device->get_target_remote_device_ids().end()) {
        device->wait_for_non_mmio_flush();
    }
}

void tt_cluster::write_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, bool small_access)
{
    write_dram_vec(&vec[0], vec.size(), dram_core, addr, small_access);
}

void tt_cluster::read_dram_vec(std::uint32_t *mem_ptr, tt_cxy_pair dram_core, uint64_t addr, uint32_t size_in_bytes, bool small_access)
{
    int chip_id = dram_core.chip;

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_read(get_soc_desc(chip_id), {dram_core.x, dram_core.y}, addr, size_in_bytes);
    }
    device->read_from_device(mem_ptr, dram_core, addr, size_in_bytes, "LARGE_READ_TLB");
}

void tt_cluster::read_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size_in_bytes, bool small_access)
{
    vec.resize(size_in_bytes / sizeof(uint32_t));
    read_dram_vec(&vec[0], dram_core, addr, size_in_bytes, small_access);
}

void tt_cluster::write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id)
{
    // TODO: Uplift
    constexpr uint16_t channel = 0;
    device->write_to_sysmem(vec, addr, channel, src_device_id);
}

void tt_cluster::read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id)
{
    // TODO: Uplift
    constexpr uint16_t channel = 0;
    device->read_from_sysmem(vec, addr, channel, size, src_device_id);
}

void *tt_cluster::channel_0_address(std::uint32_t offset, std::uint32_t device_id) const {
    TT_ASSERT(ndesc->is_chip_mmio_capable(device_id), "Cannot call channel_0_address for non-MMIO device");
    return device->channel_0_address(offset, device_id);
}

// void *tt_cluster::host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const {
//     return device->host_dma_address(offset, src_device_id);
// }

void tt_cluster::verify_sw_fw_versions(
    int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) {
    tt_version sw(sw_version), fw_first_eth_core(fw_versions.at(0));
    tt::log_info(
        tt::LogDevice,
        "Software version {}, Ethernet FW version {} (Device {})",
        sw.str(),
        fw_first_eth_core.str(),
        device_id);
    for (std::uint32_t &fw_version : fw_versions) {
        tt_version fw(fw_version);

        TT_ASSERT(fw == fw_first_eth_core, "FW versions are not the same across different ethernet cores");
        TT_ASSERT(sw.major == fw.major, "SW/FW major version number out of sync");
        TT_ASSERT(sw.minor <= fw.minor, "SW version is newer than FW version");
    }
}

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram) << ", subchan = " << std::get<2>(dram);
    return os;
}

bool check_dram_core_exists(const std::vector<std::vector<CoreCoord>> &all_dram_cores, CoreCoord target_core) {
    bool dram_core_exists = false;
    for (const auto &dram_cores_in_channel : all_dram_cores) {
        for (auto dram_core : dram_cores_in_channel) {
            if (dram_core.x == target_core.x && dram_core.y == target_core.y) {
                return true;
            }
        }
    }
    return false;
}

void tt_cluster::on_destroy(tt_cluster_on_destroy_callback cb) {
    on_destroy_callbacks.push_back(cb);
}

void tt_cluster::on_close_device(tt_cluster_on_close_device_callback cb) {
    on_close_device_callbacks.push_back(cb);
}

// This barrier works given the assumption that static VCs are used (hardcoded true in UMD)
// TODO (abhullar): Add API to query whether static VCs are used
void tt_cluster::set_dram_barrier(chip_id_t chip_id, uint32_t barrier_value) {
    _mm_sfence(); // Flush any existing writes to PCIe
    // Set barrier value
    std::vector<uint32_t> barrier_vec = {barrier_value};
    for (int channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
        this->write_dram_vec(barrier_vec, tt_target_dram{chip_id, channel, 0}, DRAM_BARRIER_BASE);
    }

    // Ensure value has been propagated
    bool barrier_val_propagated = false;
    while (not barrier_val_propagated) {
        barrier_val_propagated = true;
        for (int channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
            vector<std::uint32_t> barrier_val;
            this->read_dram_vec(barrier_val, tt_target_dram{chip_id, channel, 0}, DRAM_BARRIER_BASE, sizeof(uint32_t));
            barrier_val_propagated &= (barrier_val[0] == barrier_value);
        }
    }
}

void tt_cluster::initialize_dram_barrier(chip_id_t chip_id) {
    this->set_dram_barrier(chip_id, BARRIER_RESET);
}

void tt_cluster::dram_barrier(chip_id_t chip_id) {
    this->set_dram_barrier(chip_id, BARRIER_SET);
    this->set_dram_barrier(chip_id, BARRIER_RESET);
}

void tt_cluster::set_l1_barrier(chip_id_t chip_id, uint32_t barrier_value) {
    _mm_sfence(); // Flush any existing writes to PCIe
    // TODO (abhullar): Can get rid of logic to skip harvested cores in uplifted UMD branch because descriptor.workers does not included harvested cores
    const tt_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);
    uint32_t harvested_noc_rows = this->type == tt::TargetDevice::Silicon ? this->get_harvested_rows(chip_id) : 0;
    std::vector<unsigned int> noc_row_harvested(soc_desc.grid_size.y, 0);
    uint32_t num_harvested_rows = 0;
    for (unsigned int r = 0; r < soc_desc.grid_size.y; r++) {
        bool row_harvested = (harvested_noc_rows>>r)&0x1;
        num_harvested_rows += row_harvested;
        noc_row_harvested[r] = row_harvested;
    }

    std::vector<CoreCoord> cores_written;
    std::vector<uint32_t> barrier_vec = {barrier_value};
    for (const CoreCoord &worker_core : soc_desc.workers) {
        unsigned int row = worker_core.y;
        if (not noc_row_harvested[row]) {
            this->write_dram_vec(barrier_vec, tt_cxy_pair(chip_id, worker_core), MEM_BARRIER_ADDRESS);
            cores_written.emplace_back(worker_core);
        }
    }

    // Ensure value has been propagated
    bool barrier_value_propagated = false;
    while (not barrier_value_propagated) {
        barrier_value_propagated = true;
        for (const CoreCoord &worker_core : cores_written) {
            vector<std::uint32_t> barrier_val;
            this->read_dram_vec(barrier_val, tt_cxy_pair(chip_id, worker_core), MEM_BARRIER_ADDRESS, sizeof(uint32_t));
            barrier_value_propagated &= (barrier_val[0] == barrier_value);
        }
    }
}

void tt_cluster::initialize_l1_barrier(chip_id_t chip_id) {
    this->set_l1_barrier(chip_id, BARRIER_RESET);
}

// This barrier works given the assumption that static VCs are used (hardcoded true in UMD)
// TODO (abhullar): Add API to query whether static VCs are used
void tt_cluster::l1_barrier(chip_id_t chip_id) {
    this->set_l1_barrier(chip_id, BARRIER_SET);
    this->set_l1_barrier(chip_id, BARRIER_RESET);
}
