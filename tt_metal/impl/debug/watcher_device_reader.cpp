// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <circular_buffer_constants.h>  // For NUM_CIRCULAR_BUFFERS
#include <core_coord.hpp>
#include <ctype.h>
#include "dev_msgs.h"
#include <fmt/base.h>
#include <tt-logger/tt-logger.hpp>
#include <metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core_descriptor.hpp"
#include "debug_helpers.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "hw/inc/debug/ring_buffer.h"
#include "llrt.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/xy_pair.h>
#include "watcher_device_reader.hpp"

using namespace tt::tt_metal;
using std::string;

#define NOC_MCAST_ADDR_START_X(addr) (MetalContext::instance().hal().get_noc_mcast_addr_start_x(addr))
#define NOC_MCAST_ADDR_START_Y(addr) (MetalContext::instance().hal().get_noc_mcast_addr_start_y(addr))
#define NOC_MCAST_ADDR_END_X(addr) (MetalContext::instance().hal().get_noc_mcast_addr_end_x(addr))
#define NOC_MCAST_ADDR_END_Y(addr) (MetalContext::instance().hal().get_noc_mcast_addr_end_y(addr))
#define NOC_UNICAST_ADDR_X(addr) (MetalContext::instance().hal().get_noc_ucast_addr_x(addr))
#define NOC_UNICAST_ADDR_Y(addr) (MetalContext::instance().hal().get_noc_ucast_addr_y(addr))
#define NOC_LOCAL_ADDR(addr) (MetalContext::instance().hal().get_noc_local_addr(addr))
#define NOC_OVERLAY_START_ADDR (MetalContext::instance().hal().get_noc_overlay_start_addr())
#define NOC_STREAM_REG_SPACE_SIZE (MetalContext::instance().hal().get_noc_stream_reg_space_size())
#define STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX \
    (MetalContext::instance().hal().get_noc_stream_remote_dest_buf_size_reg_index())
#define STREAM_REMOTE_DEST_BUF_START_REG_INDEX \
    (MetalContext::instance().hal().get_noc_stream_remote_dest_buf_start_reg_index())

namespace {  // Helper functions

// Helper function to get string rep of riscv type
const char* get_riscv_name(const CoreCoord& core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return " brisc";
        case DebugNCrisc: return "ncrisc";
        case DebugErisc: return "erisc";
        case DebugSubordinateErisc: return "subordinate_erisc";
        case DebugIErisc: return "ierisc";
        case DebugSubordinateIErisc: return "subordinate_ierisc";
        case DebugTrisc0: return "trisc0";
        case DebugTrisc1: return "trisc1";
        case DebugTrisc2: return "trisc2";
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

// Helper function to determine core type from virtual coord. TODO: Remove this once we fix code types.
CoreType core_type_from_virtual_core(chip_id_t device_id, const CoreCoord& virtual_coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(virtual_coord, device_id)) {
        return CoreType::WORKER;
    } else if (tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_coord, device_id)) {
        return CoreType::ETH;
    }

    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

    const std::vector<tt::umd::CoreCoord>& translated_dram_cores =
        soc_desc.get_cores(CoreType::DRAM, CoordSystem::TRANSLATED);
    if (std::find(translated_dram_cores.begin(), translated_dram_cores.end(), virtual_coord) !=
        translated_dram_cores.end()) {
        return CoreType::DRAM;
    }

    CoreType core_type =
        soc_desc.translate_coord_to(virtual_coord, CoordSystem::PHYSICAL, CoordSystem::PHYSICAL).core_type;
    if (core_type == CoreType::TENSIX) {
        core_type = CoreType::WORKER;
    }
    return core_type;
}

// Helper function to convert noc coord -> virtual coord. TODO: Remove this once we fix code types.
CoreCoord virtual_noc_coordinate(chip_id_t device_id, uint8_t noc_index, CoreCoord coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
        return coord;
    }
    auto grid_size = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        // Coordinate passed in can be NOC0 or NOC1. The noc_index corresponds to
        // the system this coordinate belongs to.
        // Use this to convert to NOC0 coordinates and then derive Virtual Coords from it.
        CoreCoord physical_coord = {
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
        return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
            device_id, physical_coord);
    }
}

// Helper function to get string rep of noc target.
string get_noc_target_str(
    chip_id_t device_id, CoreDescriptor& core, int noc, const debug_sanitize_noc_addr_msg_t* san) {
    auto get_core_and_mem_type = [](chip_id_t device_id, CoreCoord& noc_coord, int noc) -> std::pair<string, string> {
        // Get the virtual coord from the noc coord
        CoreCoord virtual_core = virtual_noc_coordinate(device_id, noc, noc_coord);
        CoreType core_type;
        try {
            core_type = core_type_from_virtual_core(device_id, virtual_core);
        } catch (std::runtime_error& e) {
            // We may not be able to get a core type if the virtual coords are bad.
            return {"Unknown", ""};
        }
        switch (core_type) {
            case CoreType::DRAM: return {"DRAM", "DRAM"};
            case CoreType::ETH: return {"Ethernet", "L1"};
            case CoreType::PCIE: return {"PCIe", "PCIE"};
            case CoreType::WORKER: return {"Tensix", "L1"};
            default: return {"Unknown", ""};
        }
    };
    string out = fmt::format(
        "{} using noc{} tried to {} {} {} bytes {} ",
        get_riscv_name(core.coord, san->which_risc),
        noc,
        string(san->is_multicast ? "multicast" : "unicast"),
        string(san->is_write ? "write" : "read"),
        san->len,
        string(san->is_write ? "from" : "to"));
    out += fmt::format("local L1[{:#08x}] {} ", san->l1_addr, string(san->is_write ? "to" : "from"));

    if (san->is_multicast) {
        CoreCoord target_virtual_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san->noc_addr), NOC_MCAST_ADDR_START_Y(san->noc_addr)};
        CoreCoord target_virtual_noc_core_end = {
            NOC_MCAST_ADDR_END_X(san->noc_addr), NOC_MCAST_ADDR_END_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ virtual coords {}-{} {}",
            type_and_mem.first,
            target_virtual_noc_core_start.str(),
            target_virtual_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_virtual_noc_core = {NOC_UNICAST_ADDR_X(san->noc_addr), NOC_UNICAST_ADDR_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core, noc);
        out += fmt::format(
            "{} core w/ virtual coords {} {}", type_and_mem.first, target_virtual_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x}]", NOC_LOCAL_ADDR(san->noc_addr));
    return out;
}
const launch_msg_t* get_valid_launch_message(const mailboxes_t* mbox_data) {
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    if (mbox_data->launch[launch_msg_read_ptr].kernel_config.enables == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    }
    return &mbox_data->launch[launch_msg_read_ptr];
}
}  // anonymous namespace

namespace tt::tt_metal {

WatcherDeviceReader::WatcherDeviceReader(FILE* f, chip_id_t device_id, const std::vector<string>& kernel_names) :
    f(f), device_id(device_id), kernel_names(kernel_names) {
    // On init, read out eth link retraining register so that we can see if retraining has occurred. WH only for now.
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        std::vector<uint32_t> read_data;
        for (const CoreCoord& eth_core :
             tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, eth_core, CoreType::ETH);
            read_data = tt::llrt::read_hex_vec_from_core(
                device_id,
                virtual_core,
                MetalContext::instance().hal().get_dev_addr(
                    HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::RETRAIN_COUNT),
                sizeof(uint32_t));
            logical_core_to_eth_link_retraining_count[eth_core] = read_data[0];
        }
    }
}

WatcherDeviceReader::~WatcherDeviceReader() {
    // On close, read out eth link retraining register so that we can see if retraining has occurred.
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        std::vector<uint32_t> read_data;
        for (const CoreCoord& eth_core :
             tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, eth_core, CoreType::ETH);
            read_data = tt::llrt::read_hex_vec_from_core(
                device_id,
                virtual_core,
                MetalContext::instance().hal().get_dev_addr(
                    HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::RETRAIN_COUNT),
                sizeof(uint32_t));
            uint32_t num_events = read_data[0] - logical_core_to_eth_link_retraining_count[eth_core];
            if (num_events > 0) {
                log_warning(
                    tt::LogMetal,
                    "Device {} virtual ethernet core {}: Watcher detected {} link retraining events.",
                    device_id,
                    virtual_core,
                    num_events);
            }
            if (f) {
                fprintf(
                    f,
                    "%s\n",
                    fmt::format(
                        "\tDevice {} Ethernet Core {} retraining events: {}", device_id, virtual_core, num_events)
                        .c_str());
            }
        }
    }
}

void WatcherDeviceReader::Dump(FILE* file) {
    // If specified, override the existing file destination
    if (file != nullptr) {
        this->f = file;
    }

    // At this point, file should be valid.
    TT_ASSERT(this->f != nullptr);

    if (f != stdout && f != stderr) {
        log_info(tt::LogMetal, "Watcher checking device {}", device_id);
    }

    // Clear per-dump info
    paused_cores.clear();
    highest_stack_usage.clear();
    used_kernel_names.clear();

    // Ignore storage-only cores
    std::unordered_set<CoreCoord> storage_only_cores;
    uint8_t num_hw_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    DispatchCoreConfig dispatch_core_config =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    for (auto core_coord : tt::get_logical_storage_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        storage_only_cores.insert(core_coord);
    }

    // Dump worker cores
    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreDescriptor logical_core = {{x, y}, CoreType::WORKER};
            if (storage_only_cores.find(logical_core.coord) == storage_only_cores.end()) {
                DumpCore(logical_core, false);
            }
        }
    }

    // Dump eth cores
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        CoreDescriptor logical_core = {eth_core, CoreType::ETH};
        DumpCore(logical_core, true);
    }
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        CoreDescriptor logical_core = {eth_core, CoreType::ETH};
        DumpCore(logical_core, false);
    }

    for (auto k_id : used_kernel_names) {
        fprintf(f, "k_id[%3d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
    }

    // Print stack usage report for this device/dump
    if (!highest_stack_usage.empty()) {
        fprintf(f, "Stack usage summary:");
        for (auto& risc_id_and_stack_info : highest_stack_usage) {
            stack_usage_info_t& info = risc_id_and_stack_info.second;
            const char* riscv_name = get_riscv_name(info.core.coord, risc_id_and_stack_info.first);
            // Threshold of free space for warning.
            constexpr uint32_t min_threshold = 64;
            fprintf(
                f,
                "\n\t%s highest stack usage: %u bytes free, on core %s, running kernel %s",
                riscv_name,
                info.stack_free,
                info.core.coord.str().c_str(),
                kernel_names[info.kernel_id].c_str());
            if (info.stack_free == 0) {
                // We had no free stack, this probably means we
                // overflowed, but it could be a remarkable coincidence.
                fprintf(f, " (OVERFLOW)");
                log_fatal(
                    tt::LogMetal,
                    "Watcher detected stack overflow on Device {} Core {}: "
                    "{}! Kernel {} uses (at least) all of the stack.",
                    device_id,
                    info.core.coord.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str());
            } else if (info.stack_free < min_threshold) {
                fprintf(f, " (Close to overflow)");
                log_warning(
                    tt::LogMetal,
                    "Watcher detected stack had fewer than {} bytes free on Device {} Core {}: "
                    "{}! Kernel {} leaves {} bytes unused.",
                    min_threshold,
                    device_id,
                    info.core.coord.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_free);
            }
        }
        fprintf(f, "\n");
    }

    // Handle any paused cores, wait for user input.
    if (!paused_cores.empty()) {
        string paused_cores_str = "Paused cores: ";
        for (auto& core_and_risc : paused_cores) {
            paused_cores_str += fmt::format(
                "{}:{}, ", core_and_risc.first.str(), get_riscv_name(core_and_risc.first, core_and_risc.second));
        }
        paused_cores_str += "\n";
        fprintf(f, "%s", paused_cores_str.c_str());
        log_info(tt::LogMetal, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_auto_unpause()) {
            while (std::cin.get() != '\n') {
                ;
            }
        }

        // Clear all pause flags
        for (auto& core_and_risc : paused_cores) {
            const CoreCoord& virtual_core = core_and_risc.first;
            riscv_id_t risc_id = core_and_risc.second;

            uint64_t addr = MetalContext::instance().hal().get_dev_addr(
                                get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::WATCHER) +
                            offsetof(watcher_msg_t, pause_status);

            // Clear only the one flag that we saved, in case another one was raised on device
            auto pause_data =
                tt::llrt::read_hex_vec_from_core(device_id, virtual_core, addr, sizeof(debug_pause_msg_t));
            auto pause_msg = reinterpret_cast<debug_pause_msg_t*>(&(pause_data[0]));
            pause_msg->flags[risc_id] = 0;
            tt::llrt::write_hex_vec_to_core(device_id, virtual_core, pause_data, addr);
        }
    }
    fflush(f);
}

void WatcherDeviceReader::DumpCore(CoreDescriptor& logical_core, bool is_active_eth_core) {
    // Watcher only treats ethernet + worker cores.
    bool is_eth_core = (logical_core.type == CoreType::ETH);
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    CoreDescriptor virtual_core;
    virtual_core.coord =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
    virtual_core.type = logical_core.type;

    // Print device id, core coords (logical)
    string core_type = is_eth_core ? (is_active_eth_core ? "acteth" : "idleth") : "worker";
    string core_coord_str = fmt::format(
        "core(x={:2},y={:2}) virtual(x={:2},y={:2})",
        logical_core.coord.x,
        logical_core.coord.y,
        virtual_core.coord.x,
        virtual_core.coord.y);
    if (rtoptions.get_watcher_phys_coords()) {
        CoreCoord phys_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_coordinate_from_logical_coordinates(
                device_id, logical_core.coord, logical_core.type, true);
        core_coord_str += fmt::format(" phys(x={:2},y={:2})", phys_core.x, phys_core.y);
    }
    string core_str = fmt::format("Device {} {} {}", device_id, core_type, core_coord_str);
    fprintf(f, "%s: ", core_str.c_str());

    // Ethernet cores have a different mailbox base addr
    uint64_t mailbox_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX);
    if (is_eth_core) {
        if (is_active_eth_core) {
            mailbox_addr = MetalContext::instance().hal().get_dev_addr(
                HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::MAILBOX);
        } else {
            mailbox_addr = MetalContext::instance().hal().get_dev_addr(
                HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::MAILBOX);
        }
    }

    constexpr uint32_t mailbox_read_size = offsetof(mailboxes_t, watcher) + sizeof(watcher_msg_t);
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device_id, virtual_core.coord, mailbox_addr, mailbox_read_size);
    mailboxes_t* mbox_data = (mailboxes_t*)(&data[0]);
    // Get the launch message buffer read pointer.
    // For more accurate reporting of launch messages and running kernel ids, dump data from the previous valid
    // program (one entry before), if the current program is invalid (enables == 0)
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    if (launch_msg_read_ptr > launch_msg_buffer_num_entries) {
        TT_THROW(
            "Watcher read invalid launch_msg_read_ptr on {}: read {}, max valid {}!",
            core_str,
            launch_msg_read_ptr,
            launch_msg_buffer_num_entries);
    }
    if (mbox_data->launch[launch_msg_read_ptr].kernel_config.enables == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    }
    // Validate these first since they are used in diagnostic messages below.
    ValidateKernelIDs(virtual_core, &(mbox_data->launch[launch_msg_read_ptr]));

    // Whether or not watcher data is available depends on a flag set on the device.
    bool enabled = (mbox_data->watcher.enable == WatcherEnabled);

    if (enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (!rtoptions.watcher_status_disabled()) {
            DumpWaypoints(virtual_core, mbox_data, false);
        }
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core) {
            DumpL1Status(virtual_core, &mbox_data->launch[launch_msg_read_ptr]);
        }
        if (!rtoptions.watcher_noc_sanitize_disabled()) {
            const auto NUM_NOCS_ = tt::tt_metal::MetalContext::instance().hal().get_num_nocs();
            for (uint32_t noc = 0; noc < NUM_NOCS_; noc++) {
                DumpNocSanitizeStatus(virtual_core, core_str, mbox_data, noc);
            }
        }
        if (!rtoptions.watcher_assert_disabled()) {
            DumpAssertStatus(virtual_core, core_str, mbox_data);
        }
        if (!rtoptions.watcher_pause_disabled()) {
            DumpPauseStatus(virtual_core, core_str, mbox_data);
        }
    }

    // Dump state always available
    DumpLaunchMessage(virtual_core, mbox_data);
    // Ethernet cores don't use the sync reg
    if (!is_eth_core && rtoptions.get_watcher_dump_all()) {
        // Reading registers while running can cause hangs, only read if
        // requested explicitly
        DumpSyncRegs(virtual_core);
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(
            f,
            "k_id:%3d",
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]);
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
            fprintf(
                f,
                "|%3d",
                mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]);
        }
    } else {
        fprintf(
            f,
            "k_ids:%3d|%3d|%3d",
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0],
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1],
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]);

        if (rtoptions.get_watcher_text_start()) {
            uint32_t kernel_config_base = mbox_data->launch[launch_msg_read_ptr].kernel_config.kernel_config_base[0];
            fprintf(f, " text_start:");
            for (size_t i = 0; i < NUM_PROCESSORS_PER_CORE_TYPE; i++) {
                const char* separator = (i > 0) ? "|" : "";
                fprintf(
                    f,
                    "%s0x%x",
                    separator,
                    kernel_config_base + mbox_data->launch[launch_msg_read_ptr].kernel_config.kernel_text_offset[i]);
            }
        }
    }

    // Ring buffer at the end because it can print a bunch of data, same for stack usage
    if (enabled) {
        if (!rtoptions.watcher_stack_usage_disabled()) {
            DumpStackUsage(virtual_core, mbox_data);
        }
        if (!rtoptions.watcher_ring_buffer_disabled()) {
            DumpRingBuffer(virtual_core, mbox_data, false);
        }
    }

    fprintf(f, "\n");

    fflush(f);
}

void WatcherDeviceReader::DumpL1Status(CoreDescriptor& core, const launch_msg_t* launch_msg) {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device_id, core.coord, HAL_MEM_L1_BASE, sizeof(uint32_t));
    TT_ASSERT(core.type == CoreType::WORKER);
    uint32_t core_type_idx =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    auto fw_launch_value =
        MetalContext::instance().hal().get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr_value;
    if (data[0] != fw_launch_value) {
        LogRunningKernels(core, launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.coord.str(), data[0]);
    }
}

void WatcherDeviceReader::DumpNocSanitizeStatus(
    CoreDescriptor& core, const string& core_str, const mailboxes_t* mbox_data, int noc) {
    const launch_msg_t* launch_msg = get_valid_launch_message(mbox_data);
    const debug_sanitize_noc_addr_msg_t* san = &mbox_data->watcher.sanitize_noc[noc];
    string error_msg;
    string error_reason;

    switch (san->return_code) {
        case DebugSanitizeNocOK:
            if (san->noc_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
                san->l1_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 || san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
                san->which_risc != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                san->is_multicast != DEBUG_SANITIZE_NOC_SENTINEL_OK_8 ||
                san->is_write != DEBUG_SANITIZE_NOC_SENTINEL_OK_8 ||
                san->is_target != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                error_msg = fmt::format(
                    "Watcher unexpected noc debug state on core {}, reported valid got noc{}{{0x{:08x}, {} }}",
                    core.coord.str().c_str(),
                    san->which_risc,
                    san->noc_addr,
                    san->len);
                error_msg += " (corrupted noc sanitization state - sanitization memory overwritten)";
            }
            break;
        case DebugSanitizeNocAddrUnderflow:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address underflow).";
            break;
        case DebugSanitizeNocAddrOverflow:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address overflow).";
            break;
        case DebugSanitizeNocAddrZeroLength:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (zero length transaction).";
            break;
        case DebugSanitizeNocTargetInvalidXY:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).";
            break;
        case DebugSanitizeNocMulticastNonWorker:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (multicast to non-worker core).";
            break;
        case DebugSanitizeNocMulticastInvalidRange:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (multicast invalid range).";
            break;
        case DebugSanitizeNocAlignment:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (invalid address alignment in NOC transaction).";
            break;
        case DebugSanitizeNocMixedVirtualandPhysical:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (mixing virtual and virtual coordinates in Mcast).";
            break;
        case DebugSanitizeInlineWriteDramUnsupported:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += " (inline dw writes do not support DRAM destination addresses).";
            break;
        case DebugSanitizeNocAddrMailbox:
            error_msg = get_noc_target_str(device_id, core, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " overwrites mailboxes).";
            break;
        default:
            error_msg = fmt::format(
                "Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}",
                core.coord.str(),
                san->return_code);
            error_msg += " (corrupted noc sanitization state - unknown failure code).";
    }

    // If we logged an error, print to stdout and throw.
    if (!error_msg.empty()) {
        log_warning(tt::LogMetal, "Watcher detected NOC error and stopped device:");
        log_warning(tt::LogMetal, "{}: {}", core_str, error_msg);
        DumpWaypoints(core, mbox_data, true);
        DumpRingBuffer(core, mbox_data, true);
        LogRunningKernels(core, launch_msg);
        // Save the error string for checking later in unit tests.
        MetalContext::instance().watcher_server()->set_exception_message(fmt::format("{}: {}", core_str, error_msg));
        TT_THROW("{}: {}", core_str, error_msg);
    }
}

void WatcherDeviceReader::DumpAssertStatus(CoreDescriptor& core, const string& core_str, const mailboxes_t* mbox_data) {
    const launch_msg_t* launch_msg = get_valid_launch_message(mbox_data);
    const debug_assert_msg_t* assert_status = &mbox_data->watcher.assert_status;
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            // TODO: Get rid of this once #6098 is implemented.
            const string line_num_warning =
                "Note that file name reporting is not yet implemented, and the reported line number for the assert may "
                "be from a different file.";
            const string error_msg = fmt::format(
                "{}: {} tripped an assert on line {}. Current kernel: {}. {}",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                assert_status->line_num,
                GetKernelName(core, launch_msg, assert_status->which).c_str(),
                line_num_warning.c_str());
            this->DumpAssertTrippedDetails(core, error_msg, mbox_data);
            break;
        }
        case DebugAssertNCriscNOCReadsFlushedTripped: {
            const string error_msg = fmt::format(
                "{}: {} detected an inter-kernel data race due to kernel completing with pending "
                "NOC transactions (missing NOC reads flushed barrier). Current kernel: {}.",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                GetKernelName(core, launch_msg, assert_status->which).c_str());
            this->DumpAssertTrippedDetails(core, error_msg, mbox_data);
            break;
        }
        case DebugAssertNCriscNOCNonpostedWritesSentTripped: {
            const string error_msg = fmt::format(
                "{}: {} detected an inter-kernel data race due to kernel completing with pending "
                "NOC transactions (missing NOC non-posted writes sent barrier). Current kernel: {}.",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                GetKernelName(core, launch_msg, assert_status->which).c_str());
            this->DumpAssertTrippedDetails(core, error_msg, mbox_data);
            break;
        }
        case DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped: {
            const string error_msg = fmt::format(
                "{}: {} detected an inter-kernel data race due to kernel completing with pending "
                "NOC transactions (missing NOC non-posted atomics flushed barrier). Current kernel: {}.",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                GetKernelName(core, launch_msg, assert_status->which).c_str());
            this->DumpAssertTrippedDetails(core, error_msg, mbox_data);
            break;
        }
        case DebugAssertNCriscNOCPostedWritesSentTripped: {
            const string error_msg = fmt::format(
                "{}: {} detected an inter-kernel data race due to kernel completing with pending "
                "NOC transactions (missing NOC posted writes sent barrier). Current kernel: {}.",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                GetKernelName(core, launch_msg, assert_status->which).c_str());
            this->DumpAssertTrippedDetails(core, error_msg, mbox_data);
            break;
        }
        case DebugAssertOK: {
            if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                TT_THROW(
                    "Watcher unexpected assert state on core {}, reported OK but got risc {}, line {}.",
                    core.coord.str(),
                    assert_status->which,
                    assert_status->line_num);
            }
            break;
        }
        default:
            LogRunningKernels(core, launch_msg);
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                core.coord.str(),
                assert_status->tripped);
    }
}

void WatcherDeviceReader::DumpAssertTrippedDetails(
    CoreDescriptor& core, const string& error_msg, const mailboxes_t* mbox_data) {
    log_warning(tt::LogMetal, "Watcher stopped the device due to tripped assert, see watcher log for more details");
    log_warning(tt::LogMetal, "{}", error_msg);
    DumpWaypoints(core, mbox_data, true);
    DumpRingBuffer(core, mbox_data, true);
    const launch_msg_t* launch_msg = get_valid_launch_message(mbox_data);
    LogRunningKernels(core, launch_msg);
    MetalContext::instance().watcher_server()->set_exception_message(error_msg);
    TT_THROW("Watcher detected tripped assert and stopped device.");
}

void WatcherDeviceReader::DumpPauseStatus(CoreDescriptor& core, const string& core_str, const mailboxes_t* mbox_data) {
    const debug_pause_msg_t* pause_status = &mbox_data->watcher.pause_status;
    // Just record which cores are paused, printing handled at the end.
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        auto pause = pause_status->flags[risc_id];
        if (pause == 1) {
            paused_cores.insert({core.coord, static_cast<riscv_id_t>(risc_id)});
        } else if (pause > 1) {
            string error_reason = fmt::format(
                "Watcher data corruption, pause state on core {} unknown code: {}.\n", core.coord.str(), pause);
            log_warning(tt::LogMetal, "{}: {}", core_str, error_reason);
            DumpWaypoints(core, mbox_data, true);
            DumpRingBuffer(core, mbox_data, true);
            LogRunningKernels(core, get_valid_launch_message(mbox_data));
            // Save the error string for checking later in unit tests.
            MetalContext::instance().watcher_server()->set_exception_message(
                fmt::format("{}: {}", core_str, error_reason));
            TT_THROW("{}", error_reason);
        }
    }
}

void WatcherDeviceReader::DumpRingBuffer(CoreDescriptor& /*core*/, const mailboxes_t* mbox_data, bool to_stdout) {
    const debug_ring_buf_msg_t* ring_buf_data = &mbox_data->watcher.debug_ring_buf;
    string out = "";
    if (ring_buf_data->current_ptr != DEBUG_RING_BUFFER_STARTING_INDEX) {
        // Latest written idx is one less than the index read out of L1.
        out += "\n\tdebug_ring_buffer=\n\t[";
        int curr_idx = ring_buf_data->current_ptr;
        for (int count = 1; count <= DEBUG_RING_BUFFER_ELEMENTS; count++) {
            out += fmt::format("0x{:08x},", ring_buf_data->data[curr_idx]);
            if (count % 8 == 0) {
                out += "\n\t ";
            }
            if (curr_idx == 0) {
                if (ring_buf_data->wrapped == 0) {
                    break;  // No wrapping, so no extra data available
                } else {
                    curr_idx = DEBUG_RING_BUFFER_ELEMENTS - 1;  // Loop
                }
            } else {
                curr_idx--;
            }
        }
        // Remove the last comma
        out.pop_back();
        out += "]";
    }

    // This function can either dump to stdout or the log file.
    if (to_stdout) {
        if (!out.empty()) {
            out = string("Last ring buffer status: ") + out;
            log_info(tt::LogMetal, "{}", out);
        }
    } else {
        fprintf(f, "%s", out.c_str());
    }
}

void WatcherDeviceReader::DumpRunState(CoreDescriptor& core, const launch_msg_t* launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT) {
        code = 'I';
    } else if (state == RUN_MSG_GO) {
        code = 'G';
    } else if (state == RUN_MSG_DONE) {
        code = 'D';
    } else if (state == RUN_MSG_RESET_READ_PTR) {
        code = 'R';
    } else if (state == RUN_SYNC_MSG_LOAD) {
        code = 'L';
    } else if (state == RUN_SYNC_MSG_WAITING_FOR_RESET) {
        code = 'W';
    } else if (state == RUN_SYNC_MSG_INIT_SYNC_REGISTERS) {
        code = 'S';
    }
    if (code == 'U') {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {}, {}, {}, {}, or {})",
            core.coord.str(),
            state,
            RUN_MSG_INIT,
            RUN_MSG_GO,
            RUN_MSG_DONE,
            RUN_SYNC_MSG_LOAD,
            RUN_SYNC_MSG_WAITING_FOR_RESET);
    } else {
        fprintf(f, "%c", code);
    }
}

void WatcherDeviceReader::DumpLaunchMessage(CoreDescriptor& core, const mailboxes_t* mbox_data) {
    bool is_eth = (core.type == CoreType::ETH);
    const launch_msg_t* launch_msg = get_valid_launch_message(mbox_data);
    const subordinate_sync_msg_t* subordinate_sync = &mbox_data->subordinate_sync;
    fprintf(f, "rmsg:");
    if (launch_msg->kernel_config.mode == DISPATCH_MODE_DEV) {
        fprintf(f, "D");
    } else if (launch_msg->kernel_config.mode == DISPATCH_MODE_HOST) {
        fprintf(f, "H");
    } else {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
            core.coord.str(),
            launch_msg->kernel_config.mode,
            DISPATCH_MODE_DEV,
            DISPATCH_MODE_HOST);
    }

    if (launch_msg->kernel_config.brisc_noc_id == 0 || launch_msg->kernel_config.brisc_noc_id == 1) {
        fprintf(f, "%d", launch_msg->kernel_config.brisc_noc_id);
    } else {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
            core.coord.str(),
            launch_msg->kernel_config.brisc_noc_id);
    }
    DumpRunState(core, launch_msg, mbox_data->go_message.signal);

    fprintf(f, "|");
    if (launch_msg->kernel_config.enables &
        ~(DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0 | DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1 | DISPATCH_CLASS_MASK_ETH_DM0 | DISPATCH_CLASS_MASK_ETH_DM1 |
          DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE)) {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected kernel enable on core {}: {} (expected only low bits set)",
            core.coord.str(),
            launch_msg->kernel_config.enables);
    }

    // TODO(#17275): Generalize and pull risc data out of HAL
    if (!is_eth) {
        if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0) {
            fprintf(f, "B");
        } else {
            fprintf(f, "b");
        }

        if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1) {
            fprintf(f, "N");
        } else {
            fprintf(f, "n");
        }

        if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE) {
            fprintf(f, "T");
        } else {
            fprintf(f, "t");
        }
    } else {
        if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_ETH_DM0) {
            fprintf(f, "E");
        } else {
            fprintf(f, "e");
        }
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
            if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_ETH_DM1) {
                fprintf(f, "E");
            } else {
                fprintf(f, "e");
            }
        }
    }

    fprintf(f, " h_id:%3d ", launch_msg->kernel_config.host_assigned_id);

    if (!is_eth) {
        fprintf(f, "smsg:");
        DumpRunState(core, launch_msg, subordinate_sync->dm1);
        DumpRunState(core, launch_msg, subordinate_sync->trisc0);
        DumpRunState(core, launch_msg, subordinate_sync->trisc1);
        DumpRunState(core, launch_msg, subordinate_sync->trisc2);
        fprintf(f, " ");
    } else if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        fprintf(f, "smsg:");
        DumpRunState(core, launch_msg, subordinate_sync->dm1);
        fprintf(f, " ");
    }
}

void WatcherDeviceReader::DumpWaypoints(CoreDescriptor& core, const mailboxes_t* mbox_data, bool to_stdout) {
    const launch_msg_t* launch_msg = get_valid_launch_message(mbox_data);
    const debug_waypoint_msg_t* debug_waypoint = mbox_data->watcher.debug_waypoint;
    string out;

    for (int cpu = 0; cpu < MAX_RISCV_PER_CORE; cpu++) {
        string risc_status;
        for (int byte = 0; byte < num_waypoint_bytes_per_riscv; byte++) {
            char v = ((char*)&debug_waypoint[cpu])[byte];
            if (v == 0) {
                break;
            }
            if (isprint(v)) {
                risc_status += v;
            } else {
                LogRunningKernels(core, launch_msg);
                TT_THROW(
                    "Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                    core.coord.str(),
                    (int)v);
            }
        }
        // Pad risc status to 4 chars for alignment
        string pad(4 - risc_status.length(), ' ');
        out += (pad + risc_status);
        if (cpu != MAX_RISCV_PER_CORE - 1) {
            out += ',';
        }
    }

    out += " ";

    // This function can either log the waypoint to the log or stdout.
    if (to_stdout) {
        out = string("Last waypoint: ") + out;
        log_info(tt::LogMetal, "{}", out);
    } else {
        fprintf(f, "%s ", out.c_str());
    }
}

void WatcherDeviceReader::DumpSyncRegs(CoreDescriptor& core) {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device_id, core.coord, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device_id, core.coord, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

void WatcherDeviceReader::DumpStackUsage(CoreDescriptor& core, const mailboxes_t* mbox_data) {
    const debug_stack_usage_t* stack_usage_mbox = &mbox_data->watcher.stack_usage;
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        const auto &usage = stack_usage_mbox->cpu[risc_id];
        if (usage.min_free) {
            auto &slot = highest_stack_usage[static_cast<riscv_id_t>(risc_id)];
            if (usage.min_free <= slot.stack_free) {
                slot = {core, usage.min_free - 1, stack_usage_mbox->cpu[risc_id].watcher_kernel_id};
            }
        }
    }
}

void WatcherDeviceReader::ValidateKernelIDs(CoreDescriptor& core, const launch_msg_t* launch) {
    if (core.type == CoreType::ETH) {
        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected erisc0 kernel id on Device {} core {}: {} (last valid {})",
                device_id,
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]] = true;

        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1];
            TT_THROW(
                "Watcher data corruption, unexpected erisc1 kernel id on Device {} core {}: {} (last valid {})",
                device_id,
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]] = true;
    } else {
        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected brisc kernel id on Device {} core {}: {} (last valid {})",
                device_id,
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]] = true;

        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1];
            TT_THROW(
                "Watcher data corruption, unexpected ncrisc kernel id on Device {} core {}: {} (last valid {})",
                device_id,
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]] = true;

        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE];
            TT_THROW(
                "Watcher data corruption, unexpected trisc kernel id on Device {} core {}: {} (last valid {})",
                device_id,
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]] = true;
    }
}

void WatcherDeviceReader::LogRunningKernels(CoreDescriptor& core, const launch_msg_t* launch_msg) {
    log_info(tt::LogMetal, "While running kernels:");
    if (core.type == CoreType::ETH) {
        log_info(
            tt::LogMetal,
            " erisc : {}",
            kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]]);
        log_info(
            tt::LogMetal,
            " erisc : {}",
            kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]]);
    } else {
        log_info(
            tt::LogMetal,
            " brisc : {}",
            kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]]);
        log_info(
            tt::LogMetal,
            " ncrisc: {}",
            kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]]);
        log_info(
            tt::LogMetal,
            " triscs: {}",
            kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]]);
    }
}

string WatcherDeviceReader::GetKernelName(CoreDescriptor& core, const launch_msg_t* launch_msg, uint32_t type) {
    switch (type) {
        case DebugBrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]];
        case DebugErisc:
        case DebugIErisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]];
        case DebugSubordinateErisc:
        case DebugSubordinateIErisc:
            return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]];
        case DebugNCrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]];
        case DebugTrisc0:
        case DebugTrisc1:
        case DebugTrisc2:
            return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]];
        default:
            LogRunningKernels(core, launch_msg);
            TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.coord.str(), type);
    }
    return "";
}

}  // namespace tt::tt_metal
