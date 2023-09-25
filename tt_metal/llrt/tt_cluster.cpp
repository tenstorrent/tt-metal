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

// clean up bad system resource state that may be carried over
void tt_cluster::clean_system_resources() {
    TT_ASSERT(device != nullptr ,  "Device not initialized, make sure compile is done before running!");
    device->clean_system_resources();
}

void tt_cluster::verify_eth_fw() {
    const std::unordered_set<chip_id_t> &all_chips = device->get_all_chips_in_cluster();
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

std::unordered_map<chip_id_t, metal_SocDescriptor> get_metal_desc_from_tt_desc(
    const std::unordered_map<chip_id_t, tt_SocDescriptor>& input, const std::unordered_map<chip_id_t, uint32_t> &per_chip_id_harvesting_masks)
{
    std::unordered_map<chip_id_t, metal_SocDescriptor> rval = {};
    for(const auto it : input) {
        chip_id_t id = it.first;
        rval.emplace(id, metal_SocDescriptor(it.second, per_chip_id_harvesting_masks.at(id)));
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
    sdesc_per_chip = get_metal_desc_from_tt_desc(device->get_virtual_soc_descriptors(), device->get_harvesting_masks_for_soc_descriptors());
}

int tt_cluster::get_device_aiclk(const chip_id_t &chip_id) {
    if (target_device_ids.find(chip_id) != target_device_ids.end()) {
        return device->get_clocks().at(chip_id);
    }
    return 0;
}

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
}

void tt_cluster::close_device() {
    for (auto cb: on_close_device_callbacks) {
        // presumably we will have multiple devices per cluster in the future
        // so we pass a device index here
        // currently this is only used for shutting down the debug print server
        cb(this, 0);
    }

    if (device) {
        device->close_device();
        device.reset();
    }
    sdesc_per_chip.clear();
}

void tt_cluster::assert_risc_reset(const chip_id_t &chip) {
    device->assert_risc_reset(chip);
}

void tt_cluster::deassert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) {
    tt_cxy_pair virtual_chip_coord = this->convert_physical_cxy_to_virtual(physical_chip_coord);
    device->deassert_risc_reset_at_core(virtual_chip_coord);
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

// UMD expects virtual NOC coordinates
tt_cxy_pair tt_cluster::convert_physical_cxy_to_virtual(const tt_cxy_pair &physical_cxy) {
    const metal_SocDescriptor& soc_desc = get_soc_desc(physical_cxy.chip);
    CoreCoord virtual_core({
            .x = static_cast<size_t>(soc_desc.physical_routing_to_virtual_routing_x.at(physical_cxy.x)),
            .y = static_cast<size_t>(soc_desc.physical_routing_to_virtual_routing_y.at(physical_cxy.y)),
    });
    return tt_cxy_pair(physical_cxy.chip, virtual_core);
}

void tt_cluster::write_dram_vec(const std::uint32_t *mem_ptr, uint32_t len, tt_cxy_pair dram_core, uint64_t addr, bool small_access)
{
    int chip_id = dram_core.chip;
    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_write(get_soc_desc(chip_id), {dram_core.x, dram_core.y}, addr, len * sizeof(uint32_t));
    }
    tt_cxy_pair virtual_dram_core = this->convert_physical_cxy_to_virtual(dram_core);
    device->write_to_device(mem_ptr, len, virtual_dram_core, addr, "LARGE_WRITE_TLB");
    if (device->get_target_remote_device_ids().find(virtual_dram_core.chip) != device->get_target_remote_device_ids().end()) {
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
    tt_cxy_pair virtual_dram_core = this->convert_physical_cxy_to_virtual(dram_core);
    device->read_from_device(mem_ptr, virtual_dram_core, addr, size_in_bytes, "LARGE_READ_TLB");
}

void tt_cluster::read_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size_in_bytes, bool small_access)
{
    vec.resize(size_in_bytes / sizeof(uint32_t));
    read_dram_vec(&vec[0], dram_core, addr, size_in_bytes, small_access);
}

void tt_cluster::write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id)
{
    constexpr uint16_t channel = 0;
    device->write_to_sysmem(vec, addr, channel, src_device_id);
}

void tt_cluster::read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id)
{
    // TODO: Uplift
    constexpr uint16_t channel = 0;
    device->read_from_sysmem(vec, addr, channel, size, src_device_id);
}

void tt_cluster::verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) {
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

void tt_cluster::on_destroy(tt_cluster_on_destroy_callback cb) {
    on_destroy_callbacks.push_back(cb);
}

void tt_cluster::on_close_device(tt_cluster_on_close_device_callback cb) {
    on_close_device_callbacks.push_back(cb);
}

// This barrier mimics a DRAM flush since there is no flush available for PCIe device
// It is used to ensure all previous writes to DRAM have been posted by writing to a specific address in DRAM and then polling until the written value is successfully readback
void tt_cluster::set_dram_barrier(chip_id_t chip_id, uint32_t barrier_value) {
    tt_driver_atomics::sfence(); // Flush any existing writes to PCIe

    // Write a value to reserved address in DRAM that is used for host-to-device synchronization
    std::vector<uint32_t> barrier_vec = {barrier_value};
    for (int channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
        this->write_dram_vec(barrier_vec, tt_target_dram{chip_id, channel, 0}, DRAM_BARRIER_BASE);
    }

    // sfence is sufficient to flush WC buffers, ensures the reads from barrier are not just hitting the cache
    tt_driver_atomics::sfence();

    // Loop until value written is readback from each DRAM bank
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

// Set DRAM barrier address to a known value
void tt_cluster::initialize_dram_barrier(chip_id_t chip_id) {
    this->set_dram_barrier(chip_id, BARRIER_RESET);
}

// DRAM barrier is used to implement host-to-device synchronization and should be used when all previous writes to DRAM need to be flushed
// This is needed because writes to device are not blocking unless strict TLB ordering is used (default ordering is posted)
// This barrier is intended to prevent races caused by out of order writes, specifically to ensure metadata and data to compute on are committed before launching kernels
void tt_cluster::dram_barrier(chip_id_t chip_id) {
    this->set_dram_barrier(chip_id, BARRIER_SET);
    this->set_dram_barrier(chip_id, BARRIER_RESET);
}

// This barrier mimics a L1 flush since there is no flush available for PCIe device
// It is used to ensure all previous writes to L1 have been posted by writing to a specific address in L1 and then polling until the written value is successfully readback
void tt_cluster::set_l1_barrier(chip_id_t chip_id, uint32_t barrier_value) {
    tt_driver_atomics::sfence(); // Flush any existing writes to PCIe
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    // Write a value to mailbox in L1 that is exclusively used for host-to-device synchronization
    std::vector<CoreCoord> cores_written;
    std::vector<uint32_t> barrier_vec = {barrier_value};
    for (const CoreCoord &physical_worker_core : soc_desc.physical_workers) {
        this->write_dram_vec(barrier_vec, tt_cxy_pair(chip_id, physical_worker_core), MEM_BARRIER_ADDRESS);
    }

    // sfence is sufficient to flush WC buffers, ensures the reads from barrier are not just hitting the cache
    tt_driver_atomics::sfence();

    // Loop until value written to L1 mailbox is readback from each L1
    bool barrier_value_propagated = false;
    while (not barrier_value_propagated) {
        barrier_value_propagated = true;
        for (const CoreCoord &physical_worker_core : soc_desc.physical_workers) {
            vector<std::uint32_t> barrier_val;
            this->read_dram_vec(barrier_val, tt_cxy_pair(chip_id, physical_worker_core), MEM_BARRIER_ADDRESS, sizeof(uint32_t));
            barrier_value_propagated &= (barrier_val[0] == barrier_value);
        }
    }
}

// Set L1 barrier mailbox to a known value
void tt_cluster::initialize_l1_barrier(chip_id_t chip_id) {
    this->set_l1_barrier(chip_id, BARRIER_RESET);
}

// L1 barrier is used to implement host-to-device synchronization and should be used when all previous writes to L1 need to be flushed
// This is needed because writes to device are not blocking unless strict TLB ordering is used (default ordering is posted)
// This barrier is intended to prevent races caused by out of order writes, specifically to ensure binaries, metadata, and data to compute on are committed before launching kernels
void tt_cluster::l1_barrier(chip_id_t chip_id) {
    this->set_l1_barrier(chip_id, BARRIER_SET);
    // Resets L1 mailbox to a known value
    this->set_l1_barrier(chip_id, BARRIER_RESET);
}

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram) << ", subchan = " << std::get<2>(dram);
    return os;
}
