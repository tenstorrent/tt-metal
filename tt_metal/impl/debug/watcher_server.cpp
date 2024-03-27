// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <filesystem>

#include "llrt/llrt.hpp"
#include "watcher_server.hpp"
#include "llrt/rtoptions.hpp"
#include "dev_mem_map.h"
#include "dev_msgs.h"

#include "debug/sanitize_noc.h"

#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"

#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/debug_ring_buffer_common.h"

namespace tt {
namespace watcher {

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8  = 0xda;

static std::atomic<bool> enabled = false;
static std::atomic<bool> server_running = false;
static std::mutex watch_mutex;
static std::unordered_set<Device *> devices;
static string logfile_path = "generated/watcher/";
static string logfile_name = "watcher.log";
static FILE *logfile = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();
static std::vector<string> kernel_names;

// Flag to signal whether the watcher server has been killed due to a thrown exception.
static std::atomic<bool> watcher_killed_due_to_error = false;

static double get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return elapsed_secs.count();
}

static FILE * create_file() {

    FILE *f;

    const char *fmode = tt::llrt::OptionsG.get_watcher_append()? "a" : "w";
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    std::filesystem::create_directories(output_dir);
    string fname = output_dir.string() + watcher::logfile_name;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create log file\n");
    }
    log_info(LogLLRuntime, "Watcher log file: {}", fname);

    fprintf(f, "At %.3lfs starting\n", watcher::get_elapsed_secs());
    fprintf(f, "Legend:\n");
    fprintf(f, "\tComma separated list specifices waypoint for BRISC,NCRISC,TRISC0,TRISC1,TRISC2\n");
    fprintf(f, "\tI=initialization sequence\n");
    fprintf(f, "\tW=wait (top of spin loop)\n");
    fprintf(f, "\tR=run (entering kernel)\n");
    fprintf(f, "\tD=done (finished spin loop)\n");
    fprintf(f, "\tX=host written value prior to fw launch\n");
    fprintf(f, "\n");
    fprintf(f, "\tA single character status is in the FW, other characters clarify where, eg:\n");
    fprintf(f, "\t\tNRW is \"noc read wait\"\n");
    fprintf(f, "\t\tNWD is \"noc write done\"\n");
    fprintf(f, "\tnoc<n>:<risc>{a, l}=an L1 address used by NOC<n> by <riscv> (eg, local src address)\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x,y), a, l}=NOC<n> unicast address used by <riscv>\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x1,y1)-(x2,y2), a, l}=NOC<n> multicast address used by <riscv>\n");
    fprintf(f, "\trmsg:<c>=brisc host run message, D/H device/host dispatch; brisc NOC ID; I/G/D init/go/done; | separator; B/b enable/disable brisc; N/n enable/disable ncrisc; T/t enable/disable TRISC\n");
    fprintf(f, "\tsmsg:<c>=slave run message, I/G/D for NCRISC, TRISC0, TRISC1, TRISC2\n");
    fprintf(f, "\tk_ids:<brisc id>|<ncrisc id>|<trisc id> (ID map to file at end of section)\n");
    fprintf(f, "\n");

    return f;
}

static void log_running_kernels(const launch_msg_t *launch_msg) {
    log_info("While running kernels:");
    log_info(" brisc : {}", kernel_names[launch_msg->brisc_watcher_kernel_id]);
    log_info(" ncrisc: {}", kernel_names[launch_msg->ncrisc_watcher_kernel_id]);
    log_info(" triscs: {}", kernel_names[launch_msg->triscs_watcher_kernel_id]);
}

static void dump_l1_status(FILE *f, Device *device, CoreCoord core, const launch_msg_t *launch_msg) {

    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, MEM_L1_BASE, sizeof(uint32_t));
    // XXXX TODO(pgk): get this const from llrt (jump to fw insn)
    if (data[0] != 0x2010006f) {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.str(), data[0]);
    }
}

static const char * get_riscv_name(CoreCoord core, uint32_t type)
{
    switch (type) {
    case DebugBrisc:
        return "brisc";
    case DebugNCrisc:
        return "ncrisc";
    case DebugErisc:
        return "erisc";
    case DebugTrisc0:
        return "trisc0";
    case DebugTrisc1:
        return "trisc1";
    case DebugTrisc2:
        return "trisc2";
    default:
        TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

static string get_kernel_name(CoreCoord core, const launch_msg_t *launch_msg, uint32_t type)
{
    switch (type) {
        case DebugBrisc:
        case DebugErisc:
            return kernel_names[launch_msg->brisc_watcher_kernel_id];
        case DebugNCrisc:
            return kernel_names[launch_msg->ncrisc_watcher_kernel_id];
        case DebugTrisc0:
        case DebugTrisc1:
        case DebugTrisc2:
            return kernel_names[launch_msg->triscs_watcher_kernel_id];
        default:
            log_running_kernels(launch_msg);
            TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return "";
}

static string get_debug_status(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {

    string out;

    for (int cpu = 0; cpu < num_riscv_per_core; cpu++) {
        for (int byte = 0; byte < num_status_bytes_per_riscv; byte++) {
            char v = ((char *)&debug_status[cpu])[byte];
            if (v == 0) break;
            if (isprint(v)) {
                out += v;
            } else {
                log_running_kernels(launch_msg);
                TT_THROW("Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                          core.str(), (int)v);
            }
        }
        if (cpu != num_riscv_per_core - 1) out += ',';
    }

    out += " ";
    return out;
}

static void log_waypoint(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {
    string out = get_debug_status(core, launch_msg, debug_status);
    out = string("Last waypoint: ") + out;
    log_info(out.c_str());
}

static void dump_noc_sanity_status(FILE *f,
                                   CoreCoord core,
                                   const launch_msg_t *launch_msg,
                                   int noc,
                                   const debug_sanitize_noc_addr_msg_t* san,
                                   const debug_status_msg_t *debug_status) {

    char buf[256];

    switch (san->invalid) {
    case DebugSanitizeNocInvalidOK:
        if (san->addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
            san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
            san->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
            log_running_kernels(launch_msg);
            log_waypoint(core, launch_msg, debug_status);
            snprintf(buf,sizeof(buf),
                     "Watcher unexpected noc debug state on core %s, reported valid got noc%d{0x%08lx, %d}",
                     core.str().c_str(), san->which, san->addr, san->len);
            TT_THROW(buf);
        }
        break;
    case DebugSanitizeNocInvalidL1:
        fprintf(f, "%s using noc%d reading L1[addr=0x%08lx,len=%d]\n", get_riscv_name(core, san->which), noc, san->addr, san->len);
        fflush(f);
        log_running_kernels(launch_msg);
        log_warning("Watcher stopped the device due to bad NOC L1/reg address");
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d reading L1[addr=0x%08lx,len=%d]",
                 core.str().c_str(), get_riscv_name(core, san->which), noc, san->addr, san->len);
        TT_THROW(buf);
        break;
    case DebugSanitizeNocInvalidUnicast:
        fprintf(f, "%s using noc%d tried to access core (%02ld,%02ld) L1[addr=0x%08lx,len=%d]\n",
                get_riscv_name(core, san->which),
                noc,
                NOC_UNICAST_ADDR_X(san->addr),
                NOC_UNICAST_ADDR_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        fflush(f);
        log_warning("Watcher stopped the device due to bad NOC unicast transaction");
        log_running_kernels(launch_msg);
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d tried to accesss core (%02ld,%02ld) L1[addr=0x%08lx,len=%d]",
                 core.str().c_str(),
                 get_riscv_name(core, san->which),
                 noc,
                 NOC_UNICAST_ADDR_X(san->addr),
                 NOC_UNICAST_ADDR_Y(san->addr),
                 NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        TT_THROW(buf);
        break;
    case DebugSanitizeNocInvalidMulticast:
        fprintf(f, "%s using noc%d tried to access core range (%02ld,%02ld)-(%02ld,%02ld) L1[addr=0x%08lx,len=%d]\n",
                get_riscv_name(core, san->which),
                noc,
                NOC_MCAST_ADDR_START_X(san->addr),
                NOC_MCAST_ADDR_START_Y(san->addr),
                NOC_MCAST_ADDR_END_X(san->addr),
                NOC_MCAST_ADDR_END_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        fflush(f);
        log_warning("Watcher stopped the device due to bad NOC multicast transaction");
        log_running_kernels(launch_msg);
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d tried to access core range (%02ld,%02ld)-(%02ld,%02ld) L1[addr=0x%08lx,len=%d]}",
                 core.str().c_str(),
                 get_riscv_name(core, san->which),
                 noc,
                 NOC_MCAST_ADDR_START_X(san->addr),
                 NOC_MCAST_ADDR_START_Y(san->addr),
                 NOC_MCAST_ADDR_END_X(san->addr),
                 NOC_MCAST_ADDR_END_Y(san->addr),
                 NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        TT_THROW(buf);
        break;
    default:
        log_running_kernels(launch_msg);
        TT_THROW("Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}\n",
                  core.str(), san->invalid);
    }
}

static void dump_noc_sanity_status(FILE *f,
                                   CoreCoord core,
                                   const launch_msg_t *launch_msg,
                                   const debug_sanitize_noc_addr_msg_t *san,
                                   const debug_status_msg_t *debug_status) {

    for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
        dump_noc_sanity_status(f, core, launch_msg, noc, &san[noc], debug_status);
    }
}

static void dump_assert_status(
    FILE *f,
    CoreCoord core,
    const launch_msg_t *launch_msg,
    const debug_assert_msg_t *assert_status,
    const debug_status_msg_t *debug_status
) {
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            // TODO: Get rid of this once #6098 is implemented.
            std::string line_num_warning = "Note that file name reporting is not yet implemented, and the reported line number for the assert may be from a different file.";
            fprintf(
                f, "%s tripped assert on line %d. Current kernel: %s. %s",
                get_riscv_name(core, assert_status->which),
                assert_status->line_num,
                get_kernel_name(core, launch_msg, assert_status->which).c_str(),
                line_num_warning.c_str()
            );
            fflush(f);
            log_running_kernels(launch_msg);
            log_waypoint(core, launch_msg, debug_status);
            log_info(LogLLRuntime, "Watcher stopped the device due to tripped assert.");
            TT_THROW(
                "Watcher detected an assert: core {}, riscv {}, line {}. Current kernel: {}. {}",
                core.str(),
                get_riscv_name(core, assert_status->which),
                assert_status->line_num,
                get_kernel_name(core, launch_msg, assert_status->which),
                line_num_warning
            );
            break;
        }
        case DebugAssertOK:
            if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                TT_THROW(
                    "Watcher unexpected assert state on core {}, reported OK but got risc {}, line {}.",
                    assert_status->which,
                    assert_status->line_num
                );
            }
            break;
        default:
            log_running_kernels(launch_msg);
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                core.str(),
                assert_status->tripped
            );
    }
}

static void dump_pause_status(
    CoreCoord core,
    const debug_pause_msg_t *pause_status,
    std::set<std::pair<CoreCoord, riscv_id_t>> &paused_cores
) {
    // Just record which cores are paused, printing handled at the end.
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        auto pause = pause_status->flags[risc_id];
        if (pause == 1) {
            paused_cores.insert({core, static_cast<riscv_id_t>(risc_id)});
        } else if (pause > 1) {
            TT_THROW(
                "Watcher data corruption, pause state on core {} unknown code: {}.\n",
                core.str(),
                pause
            );
        }
    }
}

static void dump_ring_buffer(FILE *f, Device *device, CoreCoord core) {
    uint64_t buf_addr = RING_BUFFER_ADDR;
    if (tt::llrt::is_ethernet_core(core, device->id()))
        buf_addr = eth_l1_mem::address_map::ERISC_RING_BUFFER_ADDR;
    auto from_dev = tt::llrt::read_hex_vec_from_core(
        device->id(),
        core,
        buf_addr,
        RING_BUFFER_SIZE
    );
    DebugRingBufMemLayout *ring_buf_data = reinterpret_cast<DebugRingBufMemLayout *>(&(from_dev[0]));
    if (ring_buf_data->current_ptr == DEBUG_RING_BUFFER_STARTING_INDEX)
        return;

    // Latest written idx is one less than the index read out of L1.
    string out = "\n\tdebug_ring_buffer=\n\t[";
    int curr_idx = ring_buf_data->current_ptr;
    for (int count = 1; count <= RING_BUFFER_ELEMENTS; count++) {
        out += fmt::format("0x{:08x},", ring_buf_data->data[curr_idx]);
        if (count % 8 == 0) {
            out += "\n\t ";
        }
        if (curr_idx == 0) {
            if (ring_buf_data->wrapped == 0)
                break; // No wrapping, so no extra data available
            else
                curr_idx = RING_BUFFER_ELEMENTS-1; // Loop
        } else {
            curr_idx--;
        }
    }
    // Remove the last comma
    out.pop_back();
    out += "]";
    fprintf(f, "%s", out.c_str());
}

static void dump_run_state(FILE *f, CoreCoord core, const launch_msg_t *launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT) code = 'I';
    else if (state == RUN_MSG_GO) code = 'G';
    else if (state == RUN_MSG_DONE) code = 'D';
    if (code == 'U') {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected run state on core{}: {} (expected {} or {} or {})",
                  core.str(), state, RUN_MSG_INIT, RUN_MSG_GO, RUN_MSG_DONE);
    } else {
        fprintf(f, "%c", code);
    }
}

static void dump_run_mailboxes(FILE *f,
                               CoreCoord core,
                               const launch_msg_t *launch_msg,
                               const slave_sync_msg_t *slave_sync) {

    fprintf(f, "rmsg:");

    if (launch_msg->mode == DISPATCH_MODE_DEV) {
        fprintf(f, "D");
    } else if (launch_msg->mode == DISPATCH_MODE_HOST) {
        fprintf(f, "H");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
                  core.str(), launch_msg->mode, DISPATCH_MODE_DEV, DISPATCH_MODE_HOST);
    }

    if (launch_msg->brisc_noc_id == 0 || launch_msg->brisc_noc_id == 1) {
        fprintf(f, "%d", launch_msg->brisc_noc_id);
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
                  core.str(), launch_msg->brisc_noc_id);
    }

    dump_run_state(f, core, launch_msg, launch_msg->run);

    fprintf(f, "|");

    if (launch_msg->enable_brisc == 1) {
        fprintf(f, "B");
    } else if (launch_msg->enable_brisc == 0) {
        fprintf(f, "b");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected brisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_brisc);
    }

    if (launch_msg->enable_ncrisc == 1) {
        fprintf(f, "N");
    } else if (launch_msg->enable_ncrisc == 0) {
        fprintf(f, "n");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected ncrisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_ncrisc);
    }

    if (launch_msg->enable_triscs == 1) {
        fprintf(f, "T");
    } else if (launch_msg->enable_triscs == 0) {
        fprintf(f, "t");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected trisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_triscs);
    }

    fprintf(f, " ");

    fprintf(f, "smsg:");
    dump_run_state(f, core, launch_msg, slave_sync->ncrisc);
    dump_run_state(f, core, launch_msg, slave_sync->trisc0);
    dump_run_state(f, core, launch_msg, slave_sync->trisc1);
    dump_run_state(f, core, launch_msg, slave_sync->trisc2);

    fprintf(f, " ");
}

static void dump_debug_status(FILE *f, CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {

    string out = get_debug_status(core, launch_msg, debug_status);
    fprintf(f, "%s ", out.c_str());
}

static void dump_sync_regs(FILE *f, Device *device, CoreCoord core) {

    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

static void validate_kernel_ids(FILE *f,
                                std::map<int, bool>& used_kernel_names,
                                CoreCoord core,
                                const launch_msg_t *launch) {

    if (launch->brisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected brisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->brisc_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->brisc_watcher_kernel_id] = true;

    if (launch->ncrisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected ncrisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->ncrisc_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->ncrisc_watcher_kernel_id] = true;

    if (launch->triscs_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected trisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->triscs_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->triscs_watcher_kernel_id] = true;
}

static void dump_core(
    FILE *f,
    std::map<int, bool>& used_kernel_names,
    Device *device,
    CoreCoord core,
    bool dump_all,
    std::set<std::pair<CoreCoord, riscv_id_t>> &paused_cores
) {

    string pad(11 - core.str().length(), ' ');
    fprintf(f, "Device %i, ", device->id());
    fprintf(f, "Core %s:%s  ", core.str().c_str(), pad.c_str());

    // Ethernet cores have a different mailbox base addr
    bool is_eth_core = tt::llrt::is_ethernet_core(core, device->id());
    uint64_t mailbox_addr = MEM_MAILBOX_BASE;
    if (is_eth_core) {
        mailbox_addr = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
    }

    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, mailbox_addr, sizeof(mailboxes_t));
    mailboxes_t *mbox_data = (mailboxes_t *)(&data[0]);

    // Validate these first since they are used in diagnostic messages below.
    validate_kernel_ids(f, used_kernel_names, core, &mbox_data->launch);

    auto &disabled_features_set = tt::llrt::OptionsG.get_watcher_disabled_features();
    if (watcher::enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (disabled_features_set.find("STATUS") == disabled_features_set.end())
            dump_debug_status(f, core, &mbox_data->launch, mbox_data->debug_status);
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core)
            dump_l1_status(f, device, core,  &mbox_data->launch);
        if (disabled_features_set.find("NOC_SANITIZE") == disabled_features_set.end())
            dump_noc_sanity_status(f, core, &mbox_data->launch, mbox_data->sanitize_noc, mbox_data->debug_status);
        if (disabled_features_set.find("ASSERT") == disabled_features_set.end())
            dump_assert_status(f, core, &mbox_data->launch, &mbox_data->assert_status, mbox_data->debug_status);
        if (disabled_features_set.find("PAUSE") == disabled_features_set.end())
            dump_pause_status(core, &mbox_data->pause_status, paused_cores);
    }

    // Ethernet cores don't use the launch message/sync reg
    if (!is_eth_core) {
        // Dump state always available
        dump_run_mailboxes(f, core, &mbox_data->launch, &mbox_data->slave_sync);
        if (dump_all || tt::llrt::OptionsG.get_watcher_dump_all()) {
            // Reading registers while running can cause hangs, only read if
            // requested explicitly
            dump_sync_regs(f, device, core);
        }
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(f, "k_id:%d", mbox_data->launch.brisc_watcher_kernel_id);
    } else {
        fprintf(f, "k_ids:%d|%d|%d",
            mbox_data->launch.brisc_watcher_kernel_id,
            mbox_data->launch.ncrisc_watcher_kernel_id,
            mbox_data->launch.triscs_watcher_kernel_id);
    }

    // Ring buffer at the end because it can print a bunch of data
    if (watcher::enabled) {
        if (disabled_features_set.find("RING_BUFFER") == disabled_features_set.end())
            dump_ring_buffer(f, device, core);
    }

    fprintf(f, "\n");

    fflush(f);
}

// noinline so that this fn exists to be called from dgb
static void  __attribute__((noinline)) dump(FILE *f, bool dump_all) {
    for (Device* device : devices) {
        if (f != stdout && f != stderr) {
            log_info(LogLLRuntime, "Watcher checking device {}", device->id());
        }

        std::set<std::pair<CoreCoord, riscv_id_t>> paused_cores;
        std::map<int, bool> used_kernel_names;
        CoreCoord grid_size = device->logical_grid_size();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord worker_core = device->worker_core_from_logical_core(logical_core);
                if (device->storage_only_cores().find(logical_core) == device->storage_only_cores().end()) {
                    dump_core(f, used_kernel_names, device, worker_core, dump_all, paused_cores);
                }
            }
        }

        for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
            CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
            dump_core(f, used_kernel_names, device, physical_core, dump_all, paused_cores);
        }

        for (auto k_id : used_kernel_names) {
            fprintf(f, "k_id[%d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
        }

        // Handle any paused cores, wait for user input.
        if (!paused_cores.empty()) {
            string paused_cores_str = "Paused cores: ";
            for (auto &core_and_risc : paused_cores) {
                paused_cores_str += fmt::format(
                    "{}:{}, ",
                    core_and_risc.first.str(),
                    get_riscv_name(core_and_risc.first, core_and_risc.second)
                );
            }
            paused_cores_str += "\n";
            fprintf(f, "%s", paused_cores_str.c_str());
            log_info(LogLLRuntime, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
            if (!tt::llrt::OptionsG.get_watcher_auto_unpause()) {
                while (std::cin.get() != '\n') { ; }
            }

            // Clear all pause flags
            for (auto &core_and_risc : paused_cores) {
                const CoreCoord &core = core_and_risc.first;
                riscv_id_t risc_id = core_and_risc.second;
                uint64_t addr = tt::llrt::is_ethernet_core(core, device->id())?
                    GET_ETH_MAILBOX_ADDRESS_HOST(pause_status) :
                    GET_MAILBOX_ADDRESS_HOST(pause_status);
                auto pause_data = tt::llrt::read_hex_vec_from_core(
                    device->id(),
                    core,
                    addr,
                    sizeof(debug_pause_msg_t)
                );
                auto pause_msg = reinterpret_cast<debug_pause_msg_t *>(&(pause_data[0]));
                pause_msg->flags[risc_id] = 0;
                tt::llrt::write_hex_vec_to_core(device->id(), core, pause_data, addr);
            }
        }
    }
}

static void watcher_loop(int sleep_usecs) {
    TT_ASSERT(watcher::server_running == false);
    watcher::server_running = true;
    int count = 0;

    // Print to the user which features are disabled via env vars.
    string disabled_features = "";
    auto &disabled_features_set = tt::llrt::OptionsG.get_watcher_disabled_features();
    if (!disabled_features_set.empty()) {
        for (auto &feature : disabled_features_set) {
            disabled_features += feature + ",";
        }
        disabled_features.pop_back();
    } else {
        disabled_features = "None";
    }
    log_info(LogLLRuntime, "Watcher server initialized, disabled features: {}", disabled_features);

    double last_elapsed_time = watcher::get_elapsed_secs();
    while (true) {
        // Delay an amount such that we wait a minimum of the set sleep_usecs between polls.
        while ((watcher::get_elapsed_secs() - last_elapsed_time) < ((double) sleep_usecs) / 1000000.) {
            // Odds are this thread will be killed during the usleep, the kill signal is
            // watcher::enabled = false from the main thread.
            if (!watcher::enabled)
                break;
            usleep(1);
        }
        count++;
        last_elapsed_time = watcher::get_elapsed_secs();

        {
            const std::lock_guard<std::mutex> lock(watch_mutex);

            // If all devices are detached, we can turn off the server, it will be turned back on
            // when a new device is attached.
            if (!watcher::enabled)
                break;

            fprintf(logfile, "-----\n");
            fprintf(logfile, "Dump #%d at %.3lfs\n", count, watcher::get_elapsed_secs());

            if (devices.size() == 0) {
                fprintf(logfile, "No active devices\n");
            }

            try {
                dump(logfile, false);
            } catch (std::runtime_error& e) {
                // Depending on whether test mode is enabled, catch and stop server, or re-throw.
                if (tt::llrt::OptionsG.get_test_mode_enabled()) {
                    watcher::watcher_killed_due_to_error = true;
                    watcher::enabled = false;
                    break;
                } else {
                    throw e;
                }
            }

            fprintf(logfile, "Dump #%d completed at %.3lfs\n", count, watcher::get_elapsed_secs());
        }
    }

    log_info(LogLLRuntime, "Watcher thread stopped watching...");
    watcher::server_running = false;
}

} // namespace watcher

void watcher_init(Device *device) {

    // Initialize debug status values to "unknown"
    std::vector<uint32_t> debug_status_init_val = { 'X', 'X', 'X', 'X', 'X' };

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    std::vector<uint32_t> debug_sanity_init_val;
    debug_sanity_init_val.resize(NUM_NOCS * sizeof(debug_sanitize_noc_addr_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
    debug_sanitize_noc_addr_msg_t *data = reinterpret_cast<debug_sanitize_noc_addr_msg_t *>(&(debug_sanity_init_val[0]));
    for (int i = 0; i < NUM_NOCS; i++) {
        data[i].addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data[i].len = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data[i].which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data[i].invalid = DebugSanitizeNocInvalidOK;
    }

    // Initialize debug asserts to not tripped.
    std::vector<uint32_t> debug_assert_init_val;
    debug_assert_init_val.resize(sizeof(debug_assert_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_assert_msg_t) % sizeof(uint32_t) == 0);
    debug_assert_msg_t *assert_data = reinterpret_cast<debug_assert_msg_t *>(&(debug_assert_init_val[0]));
    assert_data->line_num = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
    assert_data->tripped = DebugAssertOK;
    assert_data->which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_8;

    // Initialize pause flags to 0
    std::vector<uint32_t> debug_pause_init_val;
    debug_pause_init_val.resize(sizeof(debug_pause_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_pause_msg_t) % sizeof(uint32_t) == 0);
    debug_pause_msg_t *pause_data = reinterpret_cast<debug_pause_msg_t *>(&(debug_pause_init_val[0]));
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++) {
        pause_data->flags[idx] = 0;
    }

    // Initialize debug ring buffer to a known init val, we'll check against this to see if any
    // data has been written.
    std::vector<uint32_t> debug_ring_buf_init_val(RING_BUFFER_SIZE / sizeof(uint32_t), 0);
    DebugRingBufMemLayout *ring_buf_data = reinterpret_cast<DebugRingBufMemLayout *>(&(debug_ring_buf_init_val[0]));
    ring_buf_data->current_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
    ring_buf_data->wrapped = 0;

    // Initialize worker cores debug values
    CoreCoord grid_size = device->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = device->worker_core_from_logical_core(logical_core);
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_status_init_val, GET_MAILBOX_ADDRESS_HOST(debug_status));
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_sanity_init_val, GET_MAILBOX_ADDRESS_HOST(sanitize_noc));
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_assert_init_val, GET_MAILBOX_ADDRESS_HOST(assert_status));
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_pause_init_val, GET_MAILBOX_ADDRESS_HOST(pause_status));
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_ring_buf_init_val, RING_BUFFER_ADDR);
        }
    }

    // Initialize ethernet cores debug values
    for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_status_init_val,
            GET_ETH_MAILBOX_ADDRESS_HOST(debug_status)
        );
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_sanity_init_val,
            GET_ETH_MAILBOX_ADDRESS_HOST(sanitize_noc)
        );
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_assert_init_val,
            GET_ETH_MAILBOX_ADDRESS_HOST(assert_status)
        );
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_pause_init_val,
            GET_ETH_MAILBOX_ADDRESS_HOST(pause_status)
        );
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_ring_buf_init_val,
            eth_l1_mem::address_map::ERISC_RING_BUFFER_ADDR
        );
    }

    log_debug(LogLLRuntime, "Watcher initialized device {}", device->id());
}

void watcher_attach(Device *device) {

    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::enabled && tt::llrt::OptionsG.get_watcher_enabled()) {

        watcher::logfile = watcher::create_file();
        watcher::watcher_killed_due_to_error = false;

        watcher::kernel_names.clear();
        watcher::kernel_names.push_back("blank");
        watcher::enabled = true;

        int sleep_usecs = tt::llrt::OptionsG.get_watcher_interval() * 1000;
        std::thread watcher_thread = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher_thread.detach();
    }

    if (watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %.3lfs attach device %d\n", watcher::get_elapsed_secs(), device->id());
    }

    if (watcher::enabled) {
        log_info(LogLLRuntime, "Watcher attached device {}", device->id());
    }

    // Always register the device w/ watcher, even if disabled
    // This allows dump() to be called from debugger
    watcher::devices.insert(device);
}

void watcher_detach(Device *old) {

    {
        const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

        TT_ASSERT(watcher::devices.find(old) != watcher::devices.end());
        if (watcher::enabled && watcher::logfile != nullptr) {
            log_info(LogLLRuntime, "Watcher detached device {}", old->id());
            fprintf(watcher::logfile, "At %.3lfs detach device %d\n", watcher::get_elapsed_secs(), old->id());
        }
        watcher::devices.erase(old);
        if (watcher::enabled && watcher::devices.empty()) {
            // If no devices remain, shut down the watcher server.
            watcher::enabled = false;
            if (watcher::logfile != nullptr) {
                std::fclose(watcher::logfile);
                watcher::logfile = nullptr;
            }
        }
    }

    // If we shut down the watcher server, wait until it finishes up. Do this without holding the
    // lock because the watcher server may be waiting on it before it does its exit check.
    if (watcher::devices.empty())
        while (watcher::server_running) { ; }
}

int watcher_register_kernel(const string& name) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    watcher::kernel_names.push_back(name);

    return watcher::kernel_names.size() - 1;
}

bool watcher_server_killed_due_to_error() {
    return watcher::watcher_killed_due_to_error;
}

void watcher_server_set_error_flag(bool val) {
    watcher::watcher_killed_due_to_error = val;
}

void watcher_clear_log() {
    watcher::logfile = watcher::create_file();
}

string watcher_get_log_file_name() {
    return tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path + watcher::logfile_name;
}

} // namespace tt
