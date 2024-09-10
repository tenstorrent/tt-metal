// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools.hpp"
namespace internal {

using namespace tt::tt_metal;

// force cast a reference to solid value. Works around binding packed references
template <typename T>
static T val(T v) {
    return v;
}

void match_device_program_data_with_host_program_data(const char* host_file, const char* device_file) {



    std::ifstream host_dispatch_dump_file;
    std::ifstream device_dispatch_dump_file;

    host_dispatch_dump_file.open(host_file);
    device_dispatch_dump_file.open(device_file);

    vector<pair<string, vector<string>>> host_map;


    string line;
    string type;

    while (std::getline(host_dispatch_dump_file, line)) {

        if (line.find("*") != string::npos) {
            continue;
        } else if (line.find("BINARY SPAN") != string::npos or line.find("SEM") != string::npos or line.find("CB") != string::npos) {
            type = line;
        } else {
            vector<string> host_data = {line};
            while (std::getline(host_dispatch_dump_file, line) and (line.find("*") == string::npos)) {
                host_data.push_back(line);
            }
            host_map.push_back(make_pair(type, std::move(host_data)));
        }
    }

    vector<vector<string>> device_map;
    vector<string> device_data;
    while (std::getline(device_dispatch_dump_file, line) and line != "EXIT_CONDITION") {
        if (line == "CHUNK") {
            if (not device_data.empty()) {
                device_map.push_back(device_data);
            }
            device_data.clear();
        } else {
            device_data.push_back(line);
        }
    }
    std::getline(device_dispatch_dump_file, line);
    device_map.push_back(device_data);

    bool all_match = true;
    for (const auto& [type, host_data] : host_map) {
        bool match = false;

        for (const vector<string>& device_data : device_map) {
            if (host_data == device_data) {
                tt::log_info("Matched on {}", type);
                match = true;
                break;
            }
        }

        if (not match) {
            tt::log_info("Mismatch between host and device program data on {}", type);
        }
        all_match &= match;
    }

    host_dispatch_dump_file.close();
    device_dispatch_dump_file.close();

    if (all_match) {
        tt::log_info("Full match between host and device program data");
    }
}

void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(
    const char* DISPATCH_MAP_DUMP, Device* device) {
    std::string device_dispatch_dump_file_name = "device_" + std::string(DISPATCH_MAP_DUMP);
    while (true) {
        std::ifstream device_dispatch_dump_file;
        device_dispatch_dump_file.open(device_dispatch_dump_file_name);
        std::string line;
        while (!device_dispatch_dump_file.eof()) {
            std::getline(device_dispatch_dump_file, line);

            if (line.find("EXIT_CONDITION") != string::npos) {
                device_dispatch_dump_file.close();

                match_device_program_data_with_host_program_data(
                    DISPATCH_MAP_DUMP, device_dispatch_dump_file_name.c_str());
                CloseDevice(device);
                exit(0);
            }
        }
    }
}

// Returns the number of bytes taken up by this dispatch command (including header).
uint32_t dump_dispatch_cmd(CQDispatchCmd *cmd, uint32_t cmd_addr, std::ofstream &cq_file) {
    uint32_t stride = sizeof(CQDispatchCmd);  // Default stride is just the command
    CQDispatchCmdId cmd_id = cmd->base.cmd_id;

    if (cmd_id < CQ_DISPATCH_CMD_MAX_COUNT) {
        cq_file << fmt::format("{:#010x}: {}", cmd_addr, cmd_id);
        switch (cmd_id) {
            case CQ_DISPATCH_CMD_WRITE_LINEAR:
            case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
                cq_file << fmt::format(
                    " (num_mcast_dests={}, noc_xy_addr={:#010x}, addr={:#010x}, length={:#010x})",
                    val(cmd->write_linear.num_mcast_dests),
                    val(cmd->write_linear.noc_xy_addr),
                    val(cmd->write_linear.addr),
                    val(cmd->write_linear.length));
                stride += cmd->write_linear.length;
                break;
            case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
                if (cmd->write_linear_host.is_event) {
                    uint32_t *event_ptr = (uint32_t *)(cmd + 1);
                    cq_file << fmt::format(" (completed_event_id={})", *event_ptr);
                } else {
                    cq_file << fmt::format(" (length={:#010x})", val(cmd->write_linear_host.length));
                }
                stride += cmd->write_linear_host.length;
                break;
            case CQ_DISPATCH_CMD_WRITE_PAGED:
                cq_file << fmt::format(
                    " (is_dram={}, start_page={}, base_addr={:#010x}, page_size={:#010x}, pages={})",
                    val(cmd->write_paged.is_dram),
                    val(cmd->write_paged.start_page),
                    val(cmd->write_paged.base_addr),
                    val(cmd->write_paged.page_size),
                    val(cmd->write_paged.pages));
                stride += cmd->write_paged.pages * cmd->write_paged.page_size;
                break;
            case CQ_DISPATCH_CMD_WRITE_PACKED:
                cq_file << fmt::format(
                    " (flags={:#02x}, count={}, addr={:#010x}, size={:04x})",
                    val(cmd->write_packed.flags),
                    val(cmd->write_packed.count),
                    val(cmd->write_packed.addr),
                    val(cmd->write_packed.size));
                // TODO: How does the page count for for packed writes?
                break;
            case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:
                cq_file << fmt::format(
                    " (count={}, alignment={})", val(cmd->write_packed_large.count), val(cmd->write_packed_large.alignment));
                break;
            case CQ_DISPATCH_CMD_WAIT:
                cq_file << fmt::format(
                    " (barrier={}, notify_prefetch={}, clear_count=(), wait={}, addr={:#010x}, "
                    "count = {})",
                    val(cmd->wait.barrier),
                    val(cmd->wait.notify_prefetch),
                    val(cmd->wait.clear_count),
                    val(cmd->wait.wait),
                    val(cmd->wait.addr),
                    val(cmd->wait.count));
                break;
            case CQ_DISPATCH_CMD_DEBUG:
                cq_file << fmt::format(
                    " (pad={}, key={}, checksum={:#010x}, size={}, stride={})",
                    val(cmd->debug.pad),
                    val(cmd->debug.key),
                    val(cmd->debug.checksum),
                    val(cmd->debug.size),
                    val(cmd->debug.stride));
                break;
            case CQ_DISPATCH_CMD_DELAY: cq_file << fmt::format(" (delay={})", val(cmd->delay.delay)); break;
            // These commands don't have any additional data to dump.
            case CQ_DISPATCH_CMD_ILLEGAL: break;
            case CQ_DISPATCH_CMD_GO: break;
            case CQ_DISPATCH_CMD_SINK: break;
            case CQ_DISPATCH_CMD_EXEC_BUF_END: break;
            case CQ_DISPATCH_CMD_REMOTE_WRITE: break;
            case CQ_DISPATCH_CMD_TERMINATE: break;
            default: TT_FATAL(false, "Unrecognized dispatch command: {}", cmd_id); break;
        }
    }
    return stride;
}

// Returns the number of bytes taken up by this prefetch command (including header).
uint32_t dump_prefetch_cmd(CQPrefetchCmd *cmd, uint32_t cmd_addr, std::ofstream &iq_file) {
    uint32_t stride = dispatch_constants::ISSUE_Q_ALIGNMENT;  // Default stride matches alignment.
    CQPrefetchCmdId cmd_id = cmd->base.cmd_id;

    if (cmd_id < CQ_PREFETCH_CMD_MAX_COUNT) {
        iq_file << fmt::format("{:#010x}: {}", cmd_addr, cmd_id);
        switch (cmd_id) {
            case CQ_PREFETCH_CMD_RELAY_LINEAR:
                iq_file << fmt::format(
                    " (noc_xy_addr={:#010x}, addr={:#010x}, length={:#010x})",
                    val(cmd->relay_linear.noc_xy_addr),
                    val(cmd->relay_linear.addr),
                    val(cmd->relay_linear.length));
                break;
            case CQ_PREFETCH_CMD_RELAY_PAGED:
                iq_file << fmt::format(
                    " (packed_page_flags={:#02x}, length_adjust={:#x}, base_addr={:#010x}, page_size={:#010x}, "
                    "pages={:#010x})",
                    val(cmd->relay_paged.packed_page_flags),
                    val(cmd->relay_paged.length_adjust),
                    val(cmd->relay_paged.base_addr),
                    val(cmd->relay_paged.page_size),
                    val(cmd->relay_paged.pages));
                break;
            case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED:
                iq_file << fmt::format(
                    " (count={}, total_length={:#010x}, stride={:#010x})",
                    val(cmd->relay_paged_packed.count),
                    val(cmd->relay_paged_packed.total_length),
                    val(cmd->relay_paged_packed.stride));
                stride = cmd->relay_paged_packed.stride;
                break;
            case CQ_PREFETCH_CMD_RELAY_INLINE:
            case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            case CQ_PREFETCH_CMD_EXEC_BUF_END:
                iq_file << fmt::format(
                    " (length={:#010x}, stride={:#010x})", val(cmd->relay_inline.length), val(cmd->relay_inline.stride));
                stride = cmd->relay_inline.stride;
                break;
            case CQ_PREFETCH_CMD_EXEC_BUF:
                iq_file << fmt::format(
                    " (base_addr={:#010x}, log_page_size={}, pages={})",
                    val(cmd->exec_buf.base_addr),
                    val(cmd->exec_buf.log_page_size),
                    val(cmd->exec_buf.pages));
                break;
            case CQ_PREFETCH_CMD_DEBUG:
                iq_file << fmt::format(
                    " (pad={}, key={}, checksum={:#010x}, size={}, stride={})",
                    val(cmd->debug.pad),
                    val(cmd->debug.key),
                    val(cmd->debug.checksum),
                    val(cmd->debug.size),
                    val(cmd->debug.stride));
                stride = cmd->debug.stride;
                break;
            case CQ_PREFETCH_CMD_WAIT_FOR_EVENT:
                iq_file << fmt::format(
                    " (sync_event={:#08x}, sync_event_addr={:#08x})",
                    val(cmd->event_wait.sync_event),
                    val(cmd->event_wait.sync_event_addr));
                stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE + sizeof(CQPrefetchHToPrefetchDHeader);
                break;
            // These commands don't have any additional data to dump.
            case CQ_PREFETCH_CMD_ILLEGAL: break;
            case CQ_PREFETCH_CMD_STALL: break;
            case CQ_PREFETCH_CMD_TERMINATE: break;
            default: break;
        }
    }
    return stride;
}

void print_progress_bar(float progress, bool init = false) {
    if (progress > 1.0)
        progress = 1.0;
    static int prev_bar_position = -1;
    if (init)
        prev_bar_position = -1;
    int progress_bar_width = 80;
    int bar_position = static_cast<int>(progress * progress_bar_width);
    if (bar_position > prev_bar_position) {
        std::cout << "[";
        std::cout << string(bar_position, '=') << string(progress_bar_width - bar_position, ' ');
        std::cout << "]" << int(progress * 100.0) << " %\r" << std::flush;
        prev_bar_position = bar_position;
    }
}

void dump_completion_queue_entries(
    std::ofstream &cq_file, SystemMemoryManager &sysmem_manager, SystemMemoryCQInterface &cq_interface) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(sysmem_manager.get_device_id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(sysmem_manager.get_device_id());
    uint32_t completion_write_ptr =
        get_cq_completion_wr_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
        << 4;
    uint32_t completion_read_ptr =
        get_cq_completion_rd_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
        << 4;
    uint32_t completion_q_bytes = cq_interface.completion_fifo_size << 4;
    TT_ASSERT(completion_q_bytes % dispatch_constants::TRANSFER_PAGE_SIZE == 0);
    uint32_t base_addr = (cq_interface.issue_fifo_limit << 4);

    // Read out in pages, this is fine since all completion Q entries are page aligned.
    vector<uint8_t> read_data;
    read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
    tt::log_info("Reading Device {} CQ {}, Completion Queue...", sysmem_manager.get_device_id(), cq_interface.id);
    cq_file << fmt::format(
        "Device {}, CQ {}, Completion Queue: write_ptr={:#010x}, read_ptr={:#010x}\n",
        sysmem_manager.get_device_id(),
        cq_interface.id,
        completion_write_ptr,
        completion_read_ptr);
    uint32_t last_span_start;
    bool last_span_invalid = false;
    print_progress_bar(0.0, true);
    for (uint32_t page_offset = 0; page_offset < completion_q_bytes;) {  // page_offset increment at end of loop
        uint32_t page_addr = base_addr + page_offset;
        tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), page_addr, mmio_device_id, channel);

        // Check if this page starts with a valid command id
        CQDispatchCmd *cmd = (CQDispatchCmd *)read_data.data();
        if (cmd->base.cmd_id < CQ_DISPATCH_CMD_MAX_COUNT && cmd->base.cmd_id > CQ_DISPATCH_CMD_ILLEGAL) {
            if (last_span_invalid) {
                if (page_addr == last_span_start + dispatch_constants::TRANSFER_PAGE_SIZE) {
                    cq_file << fmt::format("{:#010x}: No valid dispatch commands detected.", last_span_start);
                } else {
                    cq_file << fmt::format(
                        "{:#010x}-{:#010x}: No valid dispatch commands detected.",
                        last_span_start,
                        page_addr - dispatch_constants::TRANSFER_PAGE_SIZE);
                }
                last_span_invalid = false;
                if (last_span_start <= (completion_write_ptr) &&
                    page_addr - dispatch_constants::TRANSFER_PAGE_SIZE >= (completion_write_ptr)) {
                    cq_file << fmt::format(" << write_ptr (0x{:08x})", completion_write_ptr);
                }
                if (last_span_start <= (completion_read_ptr) &&
                    page_addr - dispatch_constants::TRANSFER_PAGE_SIZE >= (completion_read_ptr)) {
                    cq_file << fmt::format(" << read_ptr (0x{:08x})", completion_read_ptr);
                }
                cq_file << std::endl;
            }
            uint32_t stride = dump_dispatch_cmd(cmd, page_addr, cq_file);
            // Completion Q is page-aligned
            uint32_t cmd_pages =
                (stride + dispatch_constants::TRANSFER_PAGE_SIZE - 1) / dispatch_constants::TRANSFER_PAGE_SIZE;
            page_offset += cmd_pages * dispatch_constants::TRANSFER_PAGE_SIZE;
            if (page_addr == completion_write_ptr)
                cq_file << fmt::format(" << write_ptr (0x{:08x})", completion_write_ptr);
            if (page_addr == completion_read_ptr)
                cq_file << fmt::format(" << read_ptr (0x{:08x})", completion_read_ptr);
            cq_file << std::endl;

            // Show which pages have data if present.
            if (cmd_pages > 2) {
                cq_file << fmt::format(
                    "{:#010x}-{:#010x}: Data pages\n",
                    page_addr + dispatch_constants::TRANSFER_PAGE_SIZE,
                    page_addr + (cmd_pages - 1) * dispatch_constants::TRANSFER_PAGE_SIZE);
            } else if (cmd_pages == 2) {
                cq_file << fmt::format("{:#010x}: Data page\n", page_addr + dispatch_constants::TRANSFER_PAGE_SIZE);
            }
        } else {
            // If no valid command, just move on and try the next page
            // cq_file << fmt::format("{:#010x}: No valid dispatch command", page_addr) << std::endl;
            if (!last_span_invalid)
                last_span_start = page_addr;
            last_span_invalid = true;
            page_offset += dispatch_constants::TRANSFER_PAGE_SIZE;
        }

        print_progress_bar((float)page_offset / completion_q_bytes + 0.005);
    }
    if (last_span_invalid) {
        cq_file << fmt::format(
            "{:#010x}-{:#010x}: No valid dispatch commands detected.", last_span_start, base_addr + completion_q_bytes);
    }
    std::cout << std::endl;
}

void dump_issue_queue_entries(
    std::ofstream &iq_file, SystemMemoryManager &sysmem_manager, SystemMemoryCQInterface &cq_interface) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(sysmem_manager.get_device_id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(sysmem_manager.get_device_id());
    // TODO: Issue Q read ptr is not prefetcly updated 0 try to read it out from chip on dump?
    uint32_t issue_read_ptr =
        get_cq_issue_rd_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size()) << 4;
    uint32_t issue_write_ptr =
        get_cq_issue_wr_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size()) << 4;
    uint32_t issue_q_bytes = cq_interface.issue_fifo_size << 4;
    uint32_t issue_q_base_addr = cq_interface.offset + CQ_START;

    // Read out in 4K pages, could do ISSUE_Q_ALIGNMENT chunks to match the entries but this is ~2x faster.
    vector<uint8_t> read_data;
    read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
    tt::log_info("Reading Device {} CQ {}, Issue Queue...", sysmem_manager.get_device_id(), cq_interface.id);
    iq_file << fmt::format(
        "Device {}, CQ {}, Issue Queue: write_ptr={:#010x}, read_ptr={:#010x} (read_ptr not currently implemented)\n",
        sysmem_manager.get_device_id(),
        cq_interface.id,
        issue_write_ptr,
        issue_read_ptr);
    uint32_t last_span_start;
    bool last_span_invalid = false;
    print_progress_bar(0.0, true);
    uint32_t first_page_addr = issue_q_base_addr - (issue_q_base_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
    uint32_t end_of_curr_page =
        first_page_addr + dispatch_constants::TRANSFER_PAGE_SIZE - 1;  // To track offset of latest page read out
    tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), first_page_addr, mmio_device_id, channel);
    for (uint32_t offset = 0; offset < issue_q_bytes;) {  // offset increments at end of loop
        uint32_t curr_addr = issue_q_base_addr + offset;
        uint32_t page_offset = curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE;

        // Check if we need to read a new page
        if (curr_addr > end_of_curr_page) {
            uint32_t page_base = curr_addr - (curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
            tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), page_base, mmio_device_id, channel);
            end_of_curr_page = page_base + dispatch_constants::TRANSFER_PAGE_SIZE - 1;
        }

        // Check for a valid command id
        CQPrefetchCmd *cmd = (CQPrefetchCmd *)(read_data.data() + page_offset);
        if (cmd->base.cmd_id < CQ_PREFETCH_CMD_MAX_COUNT && cmd->base.cmd_id != CQ_PREFETCH_CMD_ILLEGAL) {
            if (last_span_invalid) {
                if (curr_addr == last_span_start + dispatch_constants::ISSUE_Q_ALIGNMENT) {
                    iq_file << fmt::format("{:#010x}: No valid prefetch command detected.", last_span_start);
                } else {
                    iq_file << fmt::format(
                        "{:#010x}-{:#010x}: No valid prefetch commands detected.",
                        last_span_start,
                        curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT);
                }
                last_span_invalid = false;
                if (last_span_start <= (issue_write_ptr) &&
                    curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT >= (issue_write_ptr)) {
                    iq_file << fmt::format(" << write_ptr (0x{:08x})", issue_write_ptr);
                }
                if (last_span_start <= (issue_read_ptr) &&
                    curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT >= (issue_read_ptr)) {
                    iq_file << fmt::format(" << read_ptr (0x{:08x})", issue_read_ptr);
                }
                iq_file << std::endl;
            }

            uint32_t cmd_stride = dump_prefetch_cmd(cmd, curr_addr, iq_file);

            // Check for a bad stride (happen to have a valid cmd_id, overwritten values, etc.)
            if (cmd_stride + offset >= issue_q_bytes || cmd_stride == 0 ||
                cmd_stride % dispatch_constants::ISSUE_Q_ALIGNMENT != 0) {
                cmd_stride = dispatch_constants::ISSUE_Q_ALIGNMENT;
                iq_file << " (bad stride)";
            }

            if (curr_addr == issue_write_ptr)
                iq_file << fmt::format(" << write_ptr (0x{:08x})", issue_write_ptr);
            if (curr_addr == issue_read_ptr)
                iq_file << fmt::format(" << read_ptr (0x{:08x})", issue_read_ptr);
            iq_file << std::endl;

            // If it's a RELAY_INLINE command, then the data inside is dispatch commands, show them.
            if ((cmd->base.cmd_id == CQ_PREFETCH_CMD_RELAY_INLINE ||
                 cmd->base.cmd_id == CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH) &&
                cmd_stride > dispatch_constants::ISSUE_Q_ALIGNMENT) {
                uint32_t dispatch_offset = offset + sizeof(CQPrefetchCmd);
                uint32_t dispatch_curr_addr = issue_q_base_addr + dispatch_offset;
                while (dispatch_offset < offset + cmd_stride) {
                    // Check if we need to read a new page
                    if (dispatch_curr_addr > end_of_curr_page) {
                        uint32_t page_base =
                            dispatch_curr_addr - (dispatch_curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
                        tt::Cluster::instance().read_sysmem(
                            read_data.data(), read_data.size(), page_base, mmio_device_id, channel);
                        end_of_curr_page = page_base + dispatch_constants::TRANSFER_PAGE_SIZE - 1;
                    }

                    // Read the dispatch command
                    uint32_t dispatch_page_offset = dispatch_curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE;
                    CQDispatchCmd *dispatch_cmd = (CQDispatchCmd *)(read_data.data() + dispatch_page_offset);
                    if (dispatch_cmd->base.cmd_id < CQ_DISPATCH_CMD_MAX_COUNT) {
                        iq_file << "  ";
                        uint32_t dispatch_cmd_stride =
                            dump_dispatch_cmd(dispatch_cmd, issue_q_base_addr + dispatch_offset, iq_file);
                        dispatch_offset += dispatch_cmd_stride;
                        iq_file << std::endl;
                    } else {
                        dispatch_offset += sizeof(CQDispatchCmd);
                    }
                }
                offset += cmd_stride;
            } else {
                offset += cmd_stride;
            }
        } else {
            // If not a valid command, just move on and try the next.
            if (!last_span_invalid)
                last_span_start = curr_addr;
            last_span_invalid = true;
            offset += dispatch_constants::ISSUE_Q_ALIGNMENT;
        }
        print_progress_bar((float)offset / issue_q_bytes + 0.005);
    }
    if (last_span_invalid) {
        iq_file << fmt::format(
            "{:#010x}-{:#010x}: No valid prefetch commands detected.",
            last_span_start,
            issue_q_base_addr + issue_q_bytes);
    }
    std::cout << std::endl;
}

// Define a queue type, for when they're interchangeable.
typedef enum e_cq_queue_t {
    CQ_COMPLETION_QUEUE = 0,
    CQ_ISSUE_QUEUE      = 1
} cq_queue_t;

void dump_command_queue_raw_data(
    std::ofstream &out_file,
    SystemMemoryManager &sysmem_manager,
    SystemMemoryCQInterface &cq_interface,
    cq_queue_t queue_type) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(sysmem_manager.get_device_id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(sysmem_manager.get_device_id());

    // The following variables depend on completion Q vs issue Q
    uint32_t write_ptr, read_ptr, base_addr, bytes_to_read = 0;
    string queue_type_name;
    if (queue_type == CQ_COMPLETION_QUEUE) {
        write_ptr = get_cq_completion_wr_ptr<true>(
                        sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
                    << 4;
        read_ptr = get_cq_completion_rd_ptr<true>(
                       sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
                   << 4;
        bytes_to_read = cq_interface.completion_fifo_size << 4;  // Page-aligned, Issue Q is not.
        TT_ASSERT(bytes_to_read % dispatch_constants::TRANSFER_PAGE_SIZE == 0);
        base_addr = cq_interface.issue_fifo_limit << 4;
        queue_type_name = "Completion";
    } else if (queue_type == CQ_ISSUE_QUEUE) {
        write_ptr =
            get_cq_issue_wr_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
            << 4;
        read_ptr =
            get_cq_issue_rd_ptr<true>(sysmem_manager.get_device_id(), cq_interface.id, sysmem_manager.get_cq_size())
            << 4;
        bytes_to_read = cq_interface.issue_fifo_size << 4;
        base_addr = cq_interface.offset + CQ_START;
        queue_type_name = "Issue";
    } else {
        TT_FATAL(false, "Unrecognized CQ type: {}", queue_type);
    }

    // Read out in pages
    vector<uint8_t> read_data;
    read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
    out_file << std::endl;
    out_file << fmt::format(
                    "Device {}, CQ {}, {} Queue Raw Data:\n",
                    sysmem_manager.get_device_id(),
                    cq_interface.id,
                    queue_type_name)
             << std::hex;
    tt::log_info(
        "Reading Device {} CQ {}, {} Queue Raw Data...",
        sysmem_manager.get_device_id(),
        cq_interface.id,
        queue_type_name);
    print_progress_bar(0.0, true);
    for (uint32_t page_offset = 0; page_offset < bytes_to_read; page_offset += dispatch_constants::TRANSFER_PAGE_SIZE) {
        uint32_t page_addr = base_addr + page_offset;
        print_progress_bar((float)page_offset / bytes_to_read + 0.005);

        // Print in 16B per line
        tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), page_addr, mmio_device_id, channel);
        TT_ASSERT(read_data.size() % 16 == 0);
        for (uint32_t line_offset = 0; line_offset < read_data.size(); line_offset += 16) {
            uint32_t line_addr = page_addr + line_offset;

            // Issue Q may not be divisible by page size, so break early if we go past the end.
            if (queue_type == CQ_ISSUE_QUEUE) {
                if (line_addr + 16 >= base_addr + bytes_to_read) {
                    break;
                }
            }

            out_file << "0x" << std::setfill('0') << std::setw(8) << line_addr << ": ";
            for (uint32_t idx = 0; idx < 16; idx++) {
                uint8_t val = read_data[line_offset + idx];
                out_file << " " << std::setfill('0') << std::setw(2) << +read_data[line_offset + idx];
            }
            if (line_addr == write_ptr)
                out_file << fmt::format(" << write_ptr (0x{:08x})", write_ptr);
            if (line_addr == read_ptr)
                out_file << fmt::format(" << read_ptr (0x{:08x})", read_ptr);
            out_file << std::endl;
        }
    }
    std::cout << std::endl;
}

void dump_cqs(std::ofstream &cq_file, std::ofstream &iq_file, SystemMemoryManager &sysmem_manager, bool dump_raw_data) {
    for (SystemMemoryCQInterface &cq_interface : sysmem_manager.get_cq_interfaces()) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(sysmem_manager.get_device_id());
        // Dump completion queue + issue queue
        dump_completion_queue_entries(cq_file, sysmem_manager, cq_interface);
        dump_issue_queue_entries(iq_file, sysmem_manager, cq_interface);

        // This is really slow, so don't do it by default. It's sometimes helpful to read the raw bytes though.
        if (dump_raw_data) {
            dump_command_queue_raw_data(cq_file, sysmem_manager, cq_interface, CQ_COMPLETION_QUEUE);
            dump_command_queue_raw_data(iq_file, sysmem_manager, cq_interface, CQ_ISSUE_QUEUE);
        }
    }
}

}  // end namespace internal
