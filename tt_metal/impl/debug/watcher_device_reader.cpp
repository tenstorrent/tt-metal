// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>

#include "common/core_coord.h"
#include "hw/inc/debug/ring_buffer.h"
#include "hw/inc/dev_msgs.h"
#include "impl/device/device.hpp"
#include "llrt/rtoptions.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"

#include "watcher_device_reader.hpp"

using std::string;
namespace { // Helper functions

// Helper function to get string rep of riscv type
static const char *get_riscv_name(const CoreCoord &core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return "brisc";
        case DebugNCrisc: return "ncrisc";
        case DebugErisc: return "erisc";
        case DebugIErisc: return "ierisc";
        case DebugTrisc0: return "trisc0";
        case DebugTrisc1: return "trisc1";
        case DebugTrisc2: return "trisc2";
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

// Helper function to get stack size by riscv core type
static uint32_t get_riscv_stack_size(const CoreDescriptor &core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return MEM_BRISC_STACK_SIZE;
        case DebugNCrisc: return MEM_NCRISC_STACK_SIZE;
        case DebugErisc: return 0; // Not managed/checked by us.
        case DebugIErisc: return MEM_BRISC_STACK_SIZE;
        case DebugTrisc0: return MEM_TRISC0_STACK_SIZE;
        case DebugTrisc1: return MEM_TRISC1_STACK_SIZE;
        case DebugTrisc2: return MEM_TRISC2_STACK_SIZE;
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.coord.str(), type);
    }
    return 0;
}

// Helper function to get string rep of noc target.
static string get_noc_target_str(Device *device, CoreDescriptor &core, int noc, const debug_sanitize_noc_addr_msg_t *san) {
    auto get_core_and_mem_type = [](Device *device, CoreCoord &noc_coord, int noc) -> std::pair<string, string> {
        // Get the physical coord from the noc coord
        const metal_SocDescriptor &soc_d = tt::Cluster::instance().get_soc_desc(device->id());
        CoreCoord phys_core = {
            NOC_0_X(noc, soc_d.grid_size.x, noc_coord.x), NOC_0_Y(noc, soc_d.grid_size.y, noc_coord.y)};

        CoreType core_type;
        try {
            core_type = device->core_type_from_physical_core(phys_core);
        } catch (std::runtime_error &e) {
            // We may not be able to get a core type if the physical coords are bad.
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
        CoreCoord target_phys_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san->noc_addr), NOC_MCAST_ADDR_START_Y(san->noc_addr)};
        CoreCoord target_phys_noc_core_end = {NOC_MCAST_ADDR_END_X(san->noc_addr), NOC_MCAST_ADDR_END_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ physical coords {}-{} {}",
            type_and_mem.first,
            target_phys_noc_core_start.str(),
            target_phys_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_phys_noc_core = {NOC_UNICAST_ADDR_X(san->noc_addr), NOC_UNICAST_ADDR_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core, noc);
        out += fmt::format(
            "{} core w/ physical coords {} {}", type_and_mem.first, target_phys_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x}]", NOC_LOCAL_ADDR(san->noc_addr));
    return out;
}
const launch_msg_t* get_valid_launch_message(const mailboxes_t *mbox_data) {
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    if (mbox_data->launch[launch_msg_read_ptr].kernel_config.enables == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    }
    return &mbox_data->launch[launch_msg_read_ptr];
}
} // anonymous namespace

namespace tt::watcher {

WatcherDeviceReader::WatcherDeviceReader(
    FILE *f, Device *device, vector<string> &kernel_names, void (*set_watcher_exception_message)(const string &)) :
    f(f), device(device), kernel_names(kernel_names), set_watcher_exception_message(set_watcher_exception_message) {
    // On init, read out eth link retraining register so that we can see if retraining has occurred. WH only for now.
    if (device->arch() == ARCH::WORMHOLE_B0 && tt::llrt::OptionsG.get_watcher_enabled()) {
        vector<uint32_t> read_data;
        for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(eth_core);
            read_data = tt::llrt::read_hex_vec_from_core(
                device->id(), phys_core, eth_l1_mem::address_map::RETRAIN_COUNT_ADDR, sizeof(uint32_t));
            logical_core_to_eth_link_retraining_count[eth_core] = read_data[0];
        }
    }
}

WatcherDeviceReader::~WatcherDeviceReader() {
    // On close, read out eth link retraining register so that we can see if retraining has occurred.
    if (device->arch() == ARCH::WORMHOLE_B0 && tt::llrt::OptionsG.get_watcher_enabled()) {
        vector<uint32_t> read_data;
        for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(eth_core);
            read_data = tt::llrt::read_hex_vec_from_core(
                device->id(), phys_core, eth_l1_mem::address_map::RETRAIN_COUNT_ADDR, sizeof(uint32_t));
            uint32_t num_events = read_data[0] - logical_core_to_eth_link_retraining_count[eth_core];
            if (num_events > 0) {
                log_warning(
                    "Device {} physical ethernet core {}: Watcher detected {} link retraining events.",
                    device->id(),
                    phys_core,
                    num_events);
            }
            fprintf(
                f,
                "%s\n",
                fmt::format("\tDevice {} Ethernet Core {} retraining events: {}", device->id(), phys_core, num_events)
                    .c_str());
    }
    }
}

void WatcherDeviceReader::Dump(FILE *file) {
    // If specified, override the existing file destination
    if (file != nullptr) {
        this->f = file;
    }

    // At this point, file should be valid.
    TT_ASSERT(this->f != nullptr);

    if (f != stdout && f != stderr) {
        log_info(LogLLRuntime, "Watcher checking device {}", device->id());
    }

    // Clear per-dump info
    paused_cores.clear();
    highest_stack_usage.clear();
    used_kernel_names.clear();

    // Dump worker cores
    CoreCoord grid_size = device->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreDescriptor logical_core = {{x, y}, CoreType::WORKER};
            if (device->storage_only_cores().find(logical_core.coord) == device->storage_only_cores().end()) {
                DumpCore(logical_core, false);
            }
        }
    }

    // Dump eth cores
    for (const CoreCoord &eth_core : device->ethernet_cores()) {
        CoreDescriptor logical_core = {eth_core, CoreType::ETH};
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        if (device->is_active_ethernet_core(eth_core)) {
            DumpCore(logical_core, true);
        } else if (device->is_inactive_ethernet_core(eth_core)) {
            DumpCore(logical_core, false);
        } else {
            continue;
        }
    }

    for (auto k_id : used_kernel_names) {
        fprintf(f, "k_id[%d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
    }

    // Print stack usage report for this device/dump
    if (!highest_stack_usage.empty()) {
        fprintf(f, "Stack usage summary:");
        for (auto &risc_id_and_stack_info : highest_stack_usage) {
            stack_usage_info_t &info = risc_id_and_stack_info.second;
            const char *riscv_name = get_riscv_name(info.core.coord, risc_id_and_stack_info.first);
            uint16_t stack_size = get_riscv_stack_size(info.core, risc_id_and_stack_info.first);
            fprintf(
                f,
                "\n\t%s highest stack usage: %d/%d, on core %s, running kernel %s",
                riscv_name,
                info.stack_usage,
                stack_size,
                info.core.coord.str().c_str(),
                kernel_names[info.kernel_id].c_str());
            if (info.stack_usage >= stack_size) {
                fprintf(f, " (OVERFLOW)");
                log_fatal(
                    "Watcher detected stack overflow on Device {} Core {}: {}! Kernel {} uses {}/{} of the stack.",
                    device->id(),
                    info.core.coord.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_usage,
                    stack_size);
            } else if (stack_size - info.stack_usage <= std::min(32, stack_size / 10)) {
                fprintf(f, " (Close to overflow)");
                log_warning(
                    "Watcher detected stack usage within 10\% of max on Device {} Core {}: {}! Kernel {} uses "
                    "{}/{} of the stack.",
                    device->id(),
                    info.core.coord.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_usage,
                    stack_size);
            }
        }
        fprintf(f, "\n");
    }

    // Handle any paused cores, wait for user input.
    if (!paused_cores.empty()) {
        string paused_cores_str = "Paused cores: ";
        for (auto &core_and_risc : paused_cores) {
            paused_cores_str += fmt::format(
                "{}:{}, ", core_and_risc.first.str(), get_riscv_name(core_and_risc.first, core_and_risc.second));
        }
        paused_cores_str += "\n";
        fprintf(f, "%s", paused_cores_str.c_str());
        log_info(LogLLRuntime, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
        if (!tt::llrt::OptionsG.get_watcher_auto_unpause()) {
            while (std::cin.get() != '\n') {
                ;
            }
        }

        // Clear all pause flags
        for (auto &core_and_risc : paused_cores) {
            const CoreCoord &phys_core = core_and_risc.first;
            riscv_id_t risc_id = core_and_risc.second;

            uint64_t addr = GET_WATCHER_DEV_ADDR_FOR_CORE(device, phys_core, pause_status);

            // Clear only the one flag that we saved, in case another one was raised on device
            auto pause_data =
                tt::llrt::read_hex_vec_from_core(device->id(), phys_core, addr, sizeof(debug_pause_msg_t));
            auto pause_msg = reinterpret_cast<debug_pause_msg_t *>(&(pause_data[0]));
            pause_msg->flags[risc_id] = 0;
            tt::llrt::write_hex_vec_to_core(device->id(), phys_core, pause_data, addr);
        }
    }
    fflush(f);
}

void WatcherDeviceReader::DumpCore(CoreDescriptor &logical_core, bool is_active_eth_core) {
    // Watcher only treats ethernet + worker cores.
    bool is_eth_core = (logical_core.type == CoreType::ETH);
    CoreDescriptor core;
    core.coord = device->physical_core_from_logical_core(logical_core.coord, logical_core.type);
    core.type = logical_core.type;

    // Print device id, core coords (logical)
    string core_type = is_eth_core ? "ethnet" : "worker";
    string core_str = fmt::format(
        "Device {} {} core(x={:2},y={:2}) phys(x={:2},y={:2})",
        device->id(),
        core_type,
        logical_core.coord.x,
        logical_core.coord.y,
        core.coord.x,
        core.coord.y);
    fprintf(f, "%s: ", core_str.c_str());

    // Ethernet cores have a different mailbox base addr
    uint64_t mailbox_addr = MEM_MAILBOX_BASE;
    if (is_eth_core) {
        if (is_active_eth_core) {
            mailbox_addr = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
        }
        else {
            mailbox_addr = MEM_IERISC_MAILBOX_BASE;
        }
    }

    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core.coord, mailbox_addr, sizeof(mailboxes_t));
    mailboxes_t *mbox_data = (mailboxes_t *)(&data[0]);
    // Get the launch message buffer read pointer.
    // For more accurate reporting of launch messages and running kernel ids, dump data from the previous valid
    // program (one entry before), if the current program is invalid (enables == 0)
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    if (mbox_data->launch[launch_msg_read_ptr].kernel_config.enables == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    }
    // Validate these first since they are used in diagnostic messages below.
    ValidateKernelIDs(core, &(mbox_data->launch[launch_msg_read_ptr]));

    // Whether or not watcher data is available depends on a flag set on the device.
    bool enabled = (mbox_data->watcher.enable == WatcherEnabled);

    if (enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (!tt::llrt::OptionsG.watcher_status_disabled())
            DumpWaypoints(core, mbox_data, false);
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core)
            DumpL1Status(core, &mbox_data->launch[launch_msg_read_ptr]);
        if (!tt::llrt::OptionsG.watcher_noc_sanitize_disabled()) {
            for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                DumpNocSanitizeStatus(core, core_str, mbox_data, noc);
            }
        }
        if (!tt::llrt::OptionsG.watcher_assert_disabled())
            DumpAssertStatus(core, core_str, mbox_data);
        if (!tt::llrt::OptionsG.watcher_pause_disabled())
            DumpPauseStatus(core, core_str, mbox_data);
    }

    // Ethernet cores don't use the launch message/sync reg
    if (!is_eth_core) {
        // Dump state always available
        DumpLaunchMessage(core, mbox_data);
        if (tt::llrt::OptionsG.get_watcher_dump_all()) {
            // Reading registers while running can cause hangs, only read if
            // requested explicitly
            DumpSyncRegs(core);
        }
    } else {
        fprintf(f, "rmsg:");
        DumpRunState(core, &mbox_data->launch[launch_msg_read_ptr], mbox_data->go_message.signal);
        fprintf(f, " ");
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(f, "k_id:%d", mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]);
    } else {
        fprintf(
            f,
            "k_ids:%d|%d|%d",
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0],
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1],
            mbox_data->launch[launch_msg_read_ptr].kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]);
    }

    // Ring buffer at the end because it can print a bunch of data, same for stack usage
    if (enabled) {
        if (!tt::llrt::OptionsG.watcher_stack_usage_disabled())
            DumpStackUsage(core, mbox_data);
        if (!tt::llrt::OptionsG.watcher_ring_buffer_disabled())
            DumpRingBuffer(core, mbox_data, false);
    }

    fprintf(f, "\n");

    fflush(f);
}

void WatcherDeviceReader::DumpL1Status(CoreDescriptor &core, const launch_msg_t *launch_msg) {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core.coord, MEM_L1_BASE, sizeof(uint32_t));
    if (data[0] != llrt::generate_risc_startup_addr(false)) {
        LogRunningKernels(core, launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.coord.str(), data[0]);
    }
}

void WatcherDeviceReader::DumpNocSanitizeStatus(
    CoreDescriptor &core, const string &core_str, const mailboxes_t *mbox_data, int noc) {
    const launch_msg_t *launch_msg = get_valid_launch_message(mbox_data);
    const debug_sanitize_noc_addr_msg_t *san = &mbox_data->watcher.sanitize_noc[noc];
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
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address underflow).";
            break;
        case DebugSanitizeNocAddrOverflow:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address overflow).";
            break;
        case DebugSanitizeNocAddrZeroLength:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += " (zero length transaction).";
            break;
        case DebugSanitizeNocTargetInvalidXY:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += " (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).";
            break;
        case DebugSanitizeNocMulticastNonWorker:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += " (multicast to non-worker core).";
            break;
        case DebugSanitizeNocMulticastInvalidRange:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += " (multicast invalid range).";
            break;
        case DebugSanitizeNocAlignment:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += " (invalid address alignment in NOC transaction).";
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
        log_warning("Watcher detected NOC error and stopped device:");
        log_warning("{}: {}", core_str, error_msg);
        DumpWaypoints(core, mbox_data, true);
        DumpRingBuffer(core, mbox_data, true);
        LogRunningKernels(core, launch_msg);
        // Save the error string for checking later in unit tests.
        set_watcher_exception_message(fmt::format("{}: {}", core_str, error_msg));
        TT_THROW("{}: {}", core_str, error_msg);
    }
}

void WatcherDeviceReader::DumpAssertStatus(CoreDescriptor &core, const string &core_str, const mailboxes_t *mbox_data) {
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    const launch_msg_t *launch_msg = get_valid_launch_message(mbox_data);
    const debug_assert_msg_t *assert_status = &mbox_data->watcher.assert_status;
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            // TODO: Get rid of this once #6098 is implemented.
            std::string line_num_warning =
                "Note that file name reporting is not yet implemented, and the reported line number for the assert may "
                "be from a different file.";
            string error_msg = fmt::format(
                "{}: {} tripped an assert on line {}. Current kernel: {}. {}",
                core_str,
                get_riscv_name(core.coord, assert_status->which),
                assert_status->line_num,
                GetKernelName(core, launch_msg, assert_status->which).c_str(),
                line_num_warning.c_str());
            log_warning("Watcher stopped the device due to tripped assert, see watcher log for more details");
            log_warning(error_msg.c_str());
            DumpWaypoints(core, mbox_data, true);
            DumpRingBuffer(core, mbox_data, true);
            LogRunningKernels(core, launch_msg);
            set_watcher_exception_message(error_msg);
            TT_THROW("Watcher detected tripped assert and stopped device.");
            break;
        }
        case DebugAssertOK:
            if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                TT_THROW(
                    "Watcher unexpected assert state on core {}, reported OK but got risc {}, line {}.",
                    core.coord.str(),
                    assert_status->which,
                    assert_status->line_num);
            }
            break;
        default:
            LogRunningKernels(core, launch_msg);
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                core.coord.str(),
                assert_status->tripped);
    }
}

void WatcherDeviceReader::DumpPauseStatus(CoreDescriptor &core, const string &core_str, const mailboxes_t *mbox_data) {
    const debug_pause_msg_t *pause_status = &mbox_data->watcher.pause_status;
    // Just record which cores are paused, printing handled at the end.
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        auto pause = pause_status->flags[risc_id];
        if (pause == 1) {
            paused_cores.insert({core.coord, static_cast<riscv_id_t>(risc_id)});
        } else if (pause > 1) {
            string error_reason = fmt::format("Watcher data corruption, pause state on core {} unknown code: {}.\n", core.coord.str(), pause);
            log_warning(error_reason.c_str());
            log_warning("{}: {}", core_str, error_reason);
            DumpWaypoints(core, mbox_data, true);
            DumpRingBuffer(core, mbox_data, true);
            LogRunningKernels(core, get_valid_launch_message(mbox_data));
            // Save the error string for checking later in unit tests.
            set_watcher_exception_message(fmt::format("{}: {}", core_str, error_reason));
            TT_THROW("{}", error_reason);
        }
    }
}

void WatcherDeviceReader::DumpRingBuffer(CoreDescriptor &core, const mailboxes_t *mbox_data, bool to_stdout) {
    const debug_ring_buf_msg_t *ring_buf_data = &mbox_data->watcher.debug_ring_buf;
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
                if (ring_buf_data->wrapped == 0)
                    break;  // No wrapping, so no extra data available
                else
                    curr_idx = DEBUG_RING_BUFFER_ELEMENTS - 1;  // Loop
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
            log_info(out.c_str());
        }
    } else {
        fprintf(f, "%s", out.c_str());
    }
}

void WatcherDeviceReader::DumpRunState(CoreDescriptor &core, const launch_msg_t *launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT)
        code = 'I';
    else if (state == RUN_MSG_GO)
        code = 'G';
    else if (state == RUN_MSG_DONE)
        code = 'D';
    else if (state == RUN_MSG_RESET_READ_PTR)
        code = 'R';
    if (code == 'U') {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {} or {} or {})",
            core.coord.str(),
            state,
            RUN_MSG_INIT,
            RUN_MSG_GO,
            RUN_MSG_DONE);
    } else {
        fprintf(f, "%c", code);
    }
}

void WatcherDeviceReader::DumpLaunchMessage(CoreDescriptor &core, const mailboxes_t *mbox_data) {
    const launch_msg_t *launch_msg = get_valid_launch_message(mbox_data);
    const slave_sync_msg_t *slave_sync = &mbox_data->slave_sync;
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
    if (launch_msg->kernel_config.enables & ~(DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0 |
                                            DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1 |
                                            DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE)) {
        LogRunningKernels(core, launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected kernel enable on core {}: {} (expected only low bits set)",
            core.coord.str(),
            launch_msg->kernel_config.enables);
    }

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

    fprintf(f, " ");

    fprintf(f, "smsg:");
    DumpRunState(core, launch_msg, slave_sync->ncrisc);
    DumpRunState(core, launch_msg, slave_sync->trisc0);
    DumpRunState(core, launch_msg, slave_sync->trisc1);
    DumpRunState(core, launch_msg, slave_sync->trisc2);

    fprintf(f, " ");
}

void WatcherDeviceReader::DumpWaypoints(CoreDescriptor &core, const mailboxes_t *mbox_data, bool to_stdout) {
    const launch_msg_t *launch_msg = get_valid_launch_message(mbox_data);
    const debug_waypoint_msg_t *debug_waypoint = mbox_data->watcher.debug_waypoint;
    string out;

    for (int cpu = 0; cpu < MAX_RISCV_PER_CORE; cpu++) {
        string risc_status;
        for (int byte = 0; byte < num_waypoint_bytes_per_riscv; byte++) {
            char v = ((char *)&debug_waypoint[cpu])[byte];
            if (v == 0)
                break;
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
        if (cpu != MAX_RISCV_PER_CORE - 1)
            out += ',';
    }

    out += " ";

    // This function can either log the waypoint to the log or stdout.
    if (to_stdout) {
        out = string("Last waypoint: ") + out;
        log_info(out.c_str());
    } else {
        fprintf(f, "%s ", out.c_str());
    }
}

void WatcherDeviceReader::DumpSyncRegs(CoreDescriptor &core) {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core.coord, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core.coord, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

void WatcherDeviceReader::DumpStackUsage(CoreDescriptor &core, const mailboxes_t *mbox_data) {
    const debug_stack_usage_t *stack_usage_mbox = &mbox_data->watcher.stack_usage;
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        uint16_t stack_usage = stack_usage_mbox->max_usage[risc_id];
        if (stack_usage != watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
            if (stack_usage > highest_stack_usage[static_cast<riscv_id_t>(risc_id)].stack_usage) {
                highest_stack_usage[static_cast<riscv_id_t>(risc_id)] = {
                    core, stack_usage, stack_usage_mbox->watcher_kernel_id[risc_id]};
            }
        }
    }
}

void WatcherDeviceReader::ValidateKernelIDs(CoreDescriptor &core, const launch_msg_t *launch) {
    if (core.type == CoreType::ETH) {
        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected erisc kernel id on Device {} core {}: {} (last valid {})",
                device->id(),
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]] = true;
    } else {
        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected brisc kernel id on Device {} core {}: {} (last valid {})",
                device->id(),
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]] = true;

        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1];
            TT_THROW(
                "Watcher data corruption, unexpected ncrisc kernel id on Device {} core {}: {} (last valid {})",
                device->id(),
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]] = true;

        if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE] >= kernel_names.size()) {
            uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE];
            TT_THROW(
                "Watcher data corruption, unexpected trisc kernel id on Device {} core {}: {} (last valid {})",
                device->id(),
                core.coord.str(),
                watcher_kernel_id,
                kernel_names.size());
        }
        used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]] = true;
    }
}

void WatcherDeviceReader::LogRunningKernels(CoreDescriptor &core, const launch_msg_t *launch_msg) {
    log_info("While running kernels:");
    if (core.type == CoreType::ETH) {
        log_info(" erisc : {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]]);
    } else {
        log_info(" brisc : {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]]);
        log_info(" ncrisc: {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]]);
        log_info(
            " triscs: {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]]);
    }
}

string WatcherDeviceReader::GetKernelName(CoreDescriptor &core, const launch_msg_t *launch_msg, uint32_t type) {
    switch (type) {
        case DebugBrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]];
        case DebugErisc:
        case DebugIErisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]];
        case DebugNCrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]];
        case DebugTrisc0:
        case DebugTrisc1:
        case DebugTrisc2: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]];
        default:
            LogRunningKernels(core, launch_msg);
            TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.coord.str(), type);
    }
    return "";
}

}  // namespace tt::watcher
