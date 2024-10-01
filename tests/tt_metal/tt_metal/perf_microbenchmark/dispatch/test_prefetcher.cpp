// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "assert.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "common.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_test.hpp"


constexpr uint32_t DEFAULT_TEST_TYPE = 0;
constexpr uint32_t DEVICE_DATA_SIZE = 768 * 1024;
constexpr uint32_t MAX_PAGE_SIZE = 256 * 1024; // bigger than scratch_db_page_size
constexpr uint32_t DRAM_PAGE_SIZE_DEFAULT = 1024;
constexpr uint32_t DRAM_PAGES_TO_READ_DEFAULT = 16;

constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_BASE_ADDR = 0x1f400000; // magic, half of dram
constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE = 10;
constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_PAGE_SIZE = 1 << DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE;

constexpr uint32_t DEFAULT_HUGEPAGE_ISSUE_BUFFER_SIZE = 256 * 1024 * 1024;
constexpr uint32_t DEFAULT_HUGEPAGE_COMPLETION_BUFFER_SIZE = 256 * 1024 * 1024;
constexpr uint32_t DEFAULT_PREFETCH_Q_ENTRIES = 1024;
constexpr uint32_t DEFAULT_CMDDAT_Q_SIZE = 128 * 1024 + 2 * sizeof(CQPrefetchCmd) + 2 * sizeof(CQDispatchCmd);
constexpr uint32_t DEFAULT_SCRATCH_DB_SIZE = 16 * 1024;

constexpr uint32_t DEFAULT_ITERATIONS = 10000;

constexpr uint32_t DEFAULT_PACKETIZED_PATH_TIMEOUT_EN = 0;

// constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
// constexpr uint32_t PCIE_ALIGNMENT = ((1 << PREFETCH_Q_LOG_MINSIZE) > CQ_PREFETCH_CMD_BARE_MIN_SIZE) ? (1 << PREFETCH_Q_LOG_MINSIZE) : CQ_PREFETCH_CMD_BARE_MIN_SIZE;

constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);
constexpr uint32_t DRAM_DATA_BASE_ADDR = 1024 * 1024;

constexpr uint32_t PCIE_TRANSFER_SIZE_DEFAULT = 4096;

constexpr uint32_t dev_hugepage_base_g = 2 * (CQ_START * sizeof(uint32_t)); // HOST_CQ uses some at the start address

constexpr uint32_t host_data_dirty_pattern = 0xbaadf00d;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;

bool packetized_path_timeout_en_g;
bool packetized_path_en_g;

bool warmup_g = false;
bool debug_g;
uint32_t max_prefetch_command_size_g;

uint32_t dispatch_buffer_page_size_g = 1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
uint32_t prefetch_q_entries_g;
uint32_t hugepage_issue_buffer_size_g;
void * host_hugepage_completion_buffer_base_g;
uint32_t cmddat_q_size_g;
uint32_t scratch_db_size_g;
uint32_t num_dram_banks_g;
bool use_coherent_data_g;
uint32_t test_type_g;
bool readback_every_iteration_g;
uint32_t big_g;
uint32_t pcie_transfer_size_g;
uint32_t dram_page_size_g;
uint32_t dram_pages_to_read_g;

uint32_t bytes_of_data_g = 0;
bool initialize_device_g = true;
uint32_t dispatch_wait_addr_g;
bool split_prefetcher_g;
bool split_dispatcher_g;
uint32_t prefetch_d_buffer_size_g;
bool use_dram_exec_buf_g = false;
bool relay_max_packed_paged_submcds = false;
uint32_t exec_buf_log_page_size_g;

CoreCoord first_worker_g = { 0, 1 };
CoreRange all_workers_g = {
    first_worker_g,
    {first_worker_g.x + 1, first_worker_g.y + 1},
};
CoreCoord phys_prefetch_core_g;

bool send_to_all_g = false;
bool perf_test_g = false;

uint32_t max_xfer_size_bytes_g = dispatch_buffer_page_size_g;
uint32_t min_xfer_size_bytes_g = 4;
uint32_t l1_buf_base_g;
uint32_t test_device_id_g = 0;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -t: test type: 0:Terminate 1:Smoke 2:Random 3:PCIe 4:DRAM-read 5:DRAM-write-read 6:Host 7:Packed-read (default {})", DEFAULT_TEST_TYPE);
        log_info(LogTest, "  -w: warm-up before starting timer (default disabled)");
        log_info(LogTest, "  -i: host iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, " -wx: right-most worker in grid (default {})", all_workers_g.end_coord.x);
        log_info(LogTest, " -wy: bottom-most worker in grid (default {})", all_workers_g.end_coord.y);
        log_info(LogTest, "  -b: run a \"big\" test (fills memory w/ fewer transactions) (default false)", DEFAULT_TEST_TYPE);
        log_info(LogTest, " -rb: gen data, readback and test every iteration - disable for perf measurements (default true)");
        log_info(LogTest, "  -c: use coherent data as payload (default false)");
        log_info(LogTest, "  -d: wrap all commands in debug commands and clear DRAM to known state (default disabled)");
        log_info(LogTest, "  -hp: host huge page issue buffer size (default {})", DEFAULT_HUGEPAGE_ISSUE_BUFFER_SIZE);
        log_info(LogTest, "  -pq: prefetch queue entries (default {})", DEFAULT_PREFETCH_Q_ENTRIES);
        log_info(LogTest, "  -cs: cmddat q size (default {})", DEFAULT_CMDDAT_Q_SIZE);
        log_info(LogTest, "-pdcs: prefetch_d cmddat cb size (default {})", dispatch_constants::get(CoreType::WORKER).prefetch_d_buffer_size());
        log_info(LogTest, "  -ss: scratch cb size (default {})", DEFAULT_SCRATCH_DB_SIZE);
        log_info(LogTest, " -pcies: size of data to transfer in pcie bw test type (default: {})", PCIE_TRANSFER_SIZE_DEFAULT);
        log_info(LogTest, " -dpgs: dram page size in dram bw test type (default: {})", DRAM_PAGE_SIZE_DEFAULT);
        log_info(LogTest, " -dpgr: dram pages to read in dram bw test type (default: {})", DRAM_PAGES_TO_READ_DEFAULT);
        log_info(LogTest, " -spre: split prefetcher into H and D variants (default not split)");
        log_info(LogTest, " -sdis: split dispatcher into H and the other thing variants (default not split)");
        log_info(LogTest, "  -packetized_timeout_en: packetized path timeout enabled (default false)");
        log_info(LogTest, "  -packetized_en: packetized path enabled (default false)");
        log_info(LogTest, "  -device_id: Device on which the test will be run, default = 0");
        log_info(LogTest, "  -x: execute commands from dram exec_buf (default 0)");
        log_info(LogTest, "  -mpps: give prefetcher the maximum number of packed data submcds to relay to dispatcher");
        log_info(LogTest, "-xpls: execute buffer log dram page size (default {})", DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE);
        log_info(LogTest, "  -s: seed for randomized tests (default 1)");
        exit(0);
    }

    warmup_g = test_args::has_command_option(input_args, "-w");
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    hugepage_issue_buffer_size_g = test_args::get_command_option_uint32(input_args, "-hp", DEFAULT_HUGEPAGE_ISSUE_BUFFER_SIZE);
    prefetch_q_entries_g = test_args::get_command_option_uint32(input_args, "-hq", DEFAULT_PREFETCH_Q_ENTRIES);
    cmddat_q_size_g = test_args::get_command_option_uint32(input_args, "-cs", DEFAULT_CMDDAT_Q_SIZE);
    max_prefetch_command_size_g = cmddat_q_size_g / 2;  // note: half this for best perf
    scratch_db_size_g = test_args::get_command_option_uint32(input_args, "-ss", DEFAULT_SCRATCH_DB_SIZE);
    use_coherent_data_g = test_args::has_command_option(input_args, "-c");
    readback_every_iteration_g = !test_args::has_command_option(input_args, "-rb");
    pcie_transfer_size_g = test_args::get_command_option_uint32(input_args, "-pcies", PCIE_TRANSFER_SIZE_DEFAULT);
    dram_page_size_g = test_args::get_command_option_uint32(input_args, "-dpgs", DRAM_PAGE_SIZE_DEFAULT);
    dram_pages_to_read_g = test_args::get_command_option_uint32(input_args, "-dpgr", DRAM_PAGES_TO_READ_DEFAULT);
    prefetch_d_buffer_size_g = test_args::get_command_option_uint32(input_args, "-pdcs", dispatch_constants::get(CoreType::WORKER).prefetch_d_buffer_size());

    test_type_g = test_args::get_command_option_uint32(input_args, "-t", DEFAULT_TEST_TYPE);
    all_workers_g.end_coord.x = test_args::get_command_option_uint32(input_args, "-wx", all_workers_g.end_coord.x);
    all_workers_g.end_coord.y = test_args::get_command_option_uint32(input_args, "-wy", all_workers_g.end_coord.y);
    split_prefetcher_g = test_args::has_command_option(input_args, "-spre");
    split_dispatcher_g = test_args::has_command_option(input_args, "-sdis");
    use_dram_exec_buf_g = test_args::has_command_option(input_args, "-x");
    relay_max_packed_paged_submcds = test_args::has_command_option(input_args, "-mpps");
    exec_buf_log_page_size_g = test_args::get_command_option_uint32(input_args, "-xpls", DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE);

    packetized_path_en_g = test_args::has_command_option(input_args, "-packetized_en");
    packetized_path_timeout_en_g = test_args::has_command_option(input_args, "-packetized_timeout_en");
    test_device_id_g = test_args::get_command_option_uint32(input_args, "-device_id", 0);

    uint32_t seed = test_args::get_command_option_uint32(input_args, "-s", 1);
    std::srand(seed);
    big_g = test_args::has_command_option(input_args, "-b");
    debug_g = test_args::has_command_option(input_args, "-d");

    if (debug_g && use_dram_exec_buf_g) {
        tt::log_fatal("Exec buf is not supported with debug commands");
        exit(0);
    }

    if (packetized_path_en_g && !(split_prefetcher_g && split_dispatcher_g)) {
        tt::log_fatal("Packetized path requires split prefetcher and dispatcher");
        exit(0);
    }

    if (!packetized_path_en_g && packetized_path_timeout_en_g) {
        tt::log_fatal("Packetized path timeout specified without enabling the packetized path");
        exit(0);
    }

    if (test_device_id_g != 0 && !packetized_path_en_g) {
        tt::log_fatal("Split device requires packetized path and split prefetcher/dispatcher");
        exit(0);
    }
}

void dirty_host_completion_buffer(uint32_t *host_hugepage_completion_buffer) {

    for (int i = 0; i < DEFAULT_HUGEPAGE_COMPLETION_BUFFER_SIZE / sizeof(uint32_t); i++) {
        host_hugepage_completion_buffer[i] = host_data_dirty_pattern;
    }
    tt_driver_atomics::sfence();
}

uint32_t round_cmd_size_up(uint32_t size) {
    uint32_t align_mask = hal.get_alignment(HalMemType::HOST) - 1;

    return (size + align_mask) & ~align_mask;
}

void add_bare_prefetcher_cmd(vector<uint32_t>& cmds,
                             CQPrefetchCmd& cmd,
                             bool pad = false) {

    uint32_t *ptr = (uint32_t *)&cmd;
    for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
        cmds.push_back(*ptr++);
    }

    if (pad) {
        // Pad debug cmd to always be the alignment size
        for (uint32_t i = 0; i < (CQ_PREFETCH_CMD_BARE_MIN_SIZE - sizeof(CQPrefetchCmd)) / sizeof(uint32_t); i++) {
            cmds.push_back(std::rand());
        }
    }
}

void add_prefetcher_packed_paged_read_cmd(vector<uint32_t>& cmds,
                                          vector<CQPrefetchRelayPagedPackedSubCmd>& sub_cmds,
                                          uint32_t length) {

    CQPrefetchCmd cmd{};
    cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED_PACKED;

    uint32_t stride = sub_cmds.size() * sizeof(CQPrefetchRelayPagedPackedSubCmd) + sizeof(CQPrefetchCmd);
    uint32_t aligned_stride = round_cmd_size_up(stride);
    cmd.relay_paged_packed.total_length = length;
    cmd.relay_paged_packed.stride = aligned_stride;
    cmd.relay_paged_packed.count = sub_cmds.size();

    uint32_t *ptr = (uint32_t *)&cmd;
    for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
        cmds.push_back(*ptr++);
    }

    constexpr size_t uint32_count = sizeof(CQPrefetchRelayPagedPackedSubCmd) / sizeof(uint32_t);
    std::vector<uint32_t> tmp_buffer(uint32_count);

    for (const auto& sub_cmd : sub_cmds) {
        std::memcpy(tmp_buffer.data(), &sub_cmd, sizeof(CQPrefetchRelayPagedPackedSubCmd));
        cmds.insert(cmds.end(), tmp_buffer.begin(), tmp_buffer.end());
    }

    for (int i = 0; i < (aligned_stride - stride) / sizeof(uint32_t); i++) {
        cmds.push_back(0);
    }
}

void add_prefetcher_paged_read_cmd(vector<uint32_t>& cmds,
                             vector<uint32_t>& sizes,
                             uint32_t start_page,
                             uint32_t base_addr,
                             uint32_t page_size,
                             uint32_t pages,
                             bool is_dram,
                             uint32_t length_adjust) {

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;

    cmd.relay_paged.packed_page_flags =
        (is_dram << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT) |
        (start_page << CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT);
    cmd.relay_paged.length_adjust = length_adjust;
    cmd.relay_paged.base_addr = base_addr;
    cmd.relay_paged.page_size = page_size;
    cmd.relay_paged.pages = pages;
    log_debug(tt::LogTest, "Generating CQ_PREFETCH_CMD_RELAY_PAGED w/ is_dram: {} start_page: {} base_addr: {} page_size: {} pages: {}",
        is_dram, start_page, base_addr, page_size, pages);

    add_bare_prefetcher_cmd(cmds, cmd, true);
}

void add_prefetcher_linear_read_cmd(Device *device,
                                    vector<uint32_t>& cmds,
                                    vector<uint32_t>& sizes,
                                    CoreCoord worker_core,
                                    uint32_t addr,
                                    uint32_t length) {

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;

    cmd.relay_linear.pad1 = 0;
    cmd.relay_linear.pad2 = 0;
    cmd.relay_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.relay_linear.addr = addr;
    cmd.relay_linear.length = length;

    add_bare_prefetcher_cmd(cmds, cmd, true);
}

void add_prefetcher_debug_prologue(vector<uint32_t>& cmds) {
    if (debug_g) {
        CQPrefetchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_PREFETCH_CMD_DEBUG;
        add_bare_prefetcher_cmd(cmds, debug_cmd, true);
    }
}

void add_prefetcher_debug_epilogue(vector<uint32_t>& cmds,
                                   size_t prior_end) {
    if (debug_g) {
        CQPrefetchCmd* debug_cmd_ptr;
        debug_cmd_ptr = (CQPrefetchCmd *)&cmds[prior_end];
        debug_cmd_ptr->debug.size = (cmds.size() - prior_end) * sizeof(uint32_t) - sizeof(CQPrefetchCmd);
        debug_cmd_ptr->debug.stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE;
        uint32_t checksum = 0;
        for (uint32_t i = prior_end + sizeof(CQPrefetchCmd) / sizeof(uint32_t); i < cmds.size(); i++) {
            checksum += cmds[i];
        }
        debug_cmd_ptr->debug.checksum = checksum;
    }
}

void add_prefetcher_cmd_to_hostq(vector<uint32_t>& cmds,
                                 vector<uint32_t>& sizes,
                                 const std::vector<uint32_t>& payload,
                                 size_t prior_end) {
    uint32_t cmd_size_bytes = (cmds.size() - prior_end) * sizeof(uint32_t);
    for (int i = 0; i < payload.size(); i++) {
        cmds.push_back(payload[i]);
    }
    uint32_t payload_length_bytes = payload.size() * sizeof(uint32_t);
    uint32_t pad_size_bytes = round_cmd_size_up(cmd_size_bytes + payload_length_bytes) - payload_length_bytes - cmd_size_bytes;
    for (int i = 0; i < pad_size_bytes / sizeof(uint32_t); i++) {
        cmds.push_back(0);
    }
    uint32_t new_size = (cmds.size() - prior_end) * sizeof(uint32_t);
    TT_FATAL(new_size <= max_prefetch_command_size_g, "Generated prefetcher command {} exceeds max command size {}", new_size, max_prefetch_command_size_g);
    TT_FATAL((new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    sizes.push_back(new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE);
}

void add_prefetcher_cmd(vector<uint32_t>& cmds,
                        vector<uint32_t>& sizes,
                        CQPrefetchCmd cmd) {

    vector<uint32_t> empty_payload;
    vector<uint32_t> data;

    auto prior_end = cmds.size();

    add_prefetcher_debug_prologue(cmds);
    add_bare_prefetcher_cmd(cmds, cmd);
    add_prefetcher_cmd_to_hostq(cmds, sizes, empty_payload, prior_end);
    add_prefetcher_debug_epilogue(cmds, prior_end);
}

void add_prefetcher_cmd(vector<uint32_t>& cmds,
                        vector<uint32_t>& sizes,
                        CQPrefetchCmdId id,
                        vector<uint32_t>& payload) {

    vector<uint32_t> data;

    auto prior_end = cmds.size();

    add_prefetcher_debug_prologue(cmds);

    CQPrefetchCmd cmd;
    memset(&cmd, 0, sizeof(CQPrefetchCmd));
    cmd.base.cmd_id = id;

    uint32_t payload_length_bytes = payload.size() * sizeof(uint32_t);
    switch (id) {
    case CQ_PREFETCH_CMD_RELAY_PAGED:
        TT_ASSERT(false);
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE:
    case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
    case CQ_PREFETCH_CMD_EXEC_BUF_END:
        cmd.relay_inline.length = payload_length_bytes;
        cmd.relay_inline.stride = round_cmd_size_up(payload_length_bytes + sizeof(CQPrefetchCmd));
        break;

    case CQ_PREFETCH_CMD_STALL:
        break;

    case CQ_PREFETCH_CMD_DEBUG:
        {
            static uint32_t key = 0;
            cmd.debug.key = ++key;
            cmd.debug.size = payload_length_bytes;
            cmd.debug.stride = round_cmd_size_up(payload_length_bytes + sizeof(CQPrefetchCmd));
            uint32_t checksum = 0;
            for (uint32_t i = 0; i < payload.size(); i++) {
                checksum += payload[i];
            }
            cmd.debug.checksum = checksum;
        }
        break;

    case CQ_PREFETCH_CMD_TERMINATE:
        break;

    default:
        fprintf(stderr, "Invalid prefetcher command: %d\n", id);
        break;
    }

    add_bare_prefetcher_cmd(cmds, cmd);
    add_prefetcher_cmd_to_hostq(cmds, sizes, payload, prior_end);
    add_prefetcher_debug_epilogue(cmds, prior_end);
}

// Model a paged read by updating worker data with interleaved/paged DRAM data, for validation later.
void add_paged_dram_data_to_device_data(Device *device,
                                  const CoreRange& workers,
                                  DeviceData& device_data,
                                  uint32_t start_page,
                                  uint32_t base_addr,
                                  uint32_t page_size,
                                  uint32_t pages,
                                  uint32_t length_adjust) {

    uint32_t base_addr_words = base_addr / sizeof(uint32_t);
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t length_adjust_words = length_adjust / sizeof(uint32_t);

    // Get data from DRAM map, add to all workers, but only set valid for cores included in workers range.
    TT_ASSERT(start_page < num_dram_banks_g);
    uint32_t last_page = start_page + pages;
    for (uint32_t page_idx = start_page; page_idx < last_page; page_idx++) {

        uint32_t dram_bank_id = page_idx % num_dram_banks_g;
        auto dram_channel = device->dram_channel_from_bank_id(dram_bank_id);
        CoreCoord bank_core = device->dram_core_from_dram_channel(dram_channel);
        uint32_t bank_offset = base_addr_words + page_size_words * (page_idx / num_dram_banks_g);

        if (page_idx == last_page - 1) page_size_words -= length_adjust_words;
        for (uint32_t j = 0; j  < page_size_words; j++) {
            uint32_t datum = device_data.at(bank_core, dram_bank_id, bank_offset + j);
            device_data.push_range(workers, datum, false);
        }
    }
}

// Packed page read from dram to linear write to worker
void gen_dram_packed_read_cmd(Device *device,
                              vector<uint32_t>& prefetch_cmds,
                              vector<uint32_t>& cmd_sizes,
                              DeviceData& device_data,
                              CoreCoord worker_core,
                              uint32_t log_page_size,
                              vector<uint32_t>& lengths) {

    vector<uint32_t> dispatch_cmds;
    vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds;

    uint32_t total_length = 0;
    for (auto l : lengths) {
        total_length += l;
    }
    gen_bare_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, total_length);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    uint32_t page_size = 1 << log_page_size;
    int count = 0;
    for (auto length : lengths) {
        TT_ASSERT(length <= num_dram_banks_g * page_size);
        TT_ASSERT((length & (hal.get_alignment(HalMemType::DRAM) - 1)) == 0);
        CQPrefetchRelayPagedPackedSubCmd sub_cmd;
        sub_cmd.start_page = 0; // TODO: randomize?
        sub_cmd.log_page_size = log_page_size;
        sub_cmd.base_addr = DRAM_DATA_BASE_ADDR + count * page_size;
        sub_cmd.length = length;
        sub_cmds.push_back(sub_cmd);
        count++;

        // Model the packed paged read in this function by updating worker data with interleaved/paged DRAM data, for validation later.
        uint32_t length_words = length / sizeof(uint32_t);
        uint32_t base_addr_words = (sub_cmd.base_addr - DRAM_DATA_BASE_ADDR) / sizeof(uint32_t);
        uint32_t page_size_words = page_size / sizeof(uint32_t);

        // Get data from DRAM map, add to all workers, but only set valid for cores included in workers range.
        uint32_t page_idx = sub_cmd.start_page;
        for (uint32_t i = 0; i < length_words; i += page_size_words) {
            uint32_t dram_bank_id = page_idx % num_dram_banks_g;
            auto dram_channel = device->dram_channel_from_bank_id(dram_bank_id);
            CoreCoord bank_core = device->dram_core_from_dram_channel(dram_channel);
            uint32_t bank_offset = base_addr_words + page_size_words * (page_idx / num_dram_banks_g);

            uint32_t words = (page_size_words > length_words - i) ? length_words - i : page_size_words;
            for (uint32_t j = 0; j  < words; j++) {
                uint32_t datum = device_data.at(bank_core, dram_bank_id, bank_offset + j);
                device_data.push_one(worker_core, datum);
            }

            page_idx++;
        }
    }

    auto prior_end = prefetch_cmds.size();
    add_prefetcher_packed_paged_read_cmd(prefetch_cmds, sub_cmds, total_length);
    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE);
}

// Interleaved/Paged Read of DRAM to Worker L1
void gen_dram_read_cmd(Device *device,
                       vector<uint32_t>& prefetch_cmds,
                       vector<uint32_t>& cmd_sizes,
                       DeviceData& device_data,
                       CoreCoord worker_core,
                       uint32_t start_page,
                       uint32_t base_addr,
                       uint32_t page_size,
                       uint32_t pages,
                       uint32_t length_adjust) {

    vector<uint32_t> dispatch_cmds;
    const bool is_dram = true;

    // Device code assumes padding to 32 bytes
    TT_ASSERT((length_adjust & 0x1f) == 0);

    uint32_t device_data_size = page_size * pages - length_adjust;
    log_trace(tt::LogTest, "Starting {} with worker_core: {} start_page: {} base_addr: 0x{:x} page_size: {} pages: {}. device_data_size: 0x{:x}",
        __FUNCTION__, worker_core.str(), start_page, base_addr, page_size, pages, device_data_size);

    gen_bare_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, device_data_size);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    auto prior_end = prefetch_cmds.size();
    add_prefetcher_paged_read_cmd(prefetch_cmds, cmd_sizes, start_page, base_addr + DRAM_DATA_BASE_ADDR, page_size, pages, is_dram, length_adjust);

    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE);

    // Model the paged read in this function by updating worker data with interleaved/paged DRAM data, for validation later.
    add_paged_dram_data_to_device_data(device, worker_core, device_data, start_page, base_addr, page_size, pages, length_adjust);
}

// Interleaved/Paged Write to DRAM.
void gen_dram_write_cmd(Device *device,
                    vector<uint32_t>& prefetch_cmds,
                    vector<uint32_t>& cmd_sizes,
                    DeviceData& device_data,
                    uint32_t start_page,
                    uint32_t page_size,
                    uint32_t pages) {

    vector<uint32_t> dispatch_cmds;
    const bool is_dram = true;
    log_trace(tt::LogTest, "Starting {} with is_dram: {} start_page: {} page_size: {} pages: {}", __FUNCTION__, is_dram, start_page, page_size, pages);

    gen_dispatcher_paged_write_cmd(device, dispatch_cmds, device_data, is_dram, start_page, page_size, pages);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
}

void gen_wait_and_stall_cmd(Device *device,
                            vector<uint32_t>& prefetch_cmds,
                            vector<uint32_t>& cmd_sizes) {

    vector<uint32_t> dispatch_cmds;

    CQDispatchCmd wait;
    wait.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
    wait.wait.barrier = true;
    wait.wait.notify_prefetch = true;
    wait.wait.wait = true;
    wait.wait.addr = dispatch_wait_addr_g;
    wait.wait.count = 0;
    add_bare_dispatcher_cmd(dispatch_cmds, wait);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_STALL, empty_payload);
}

// This is pretty much a blit: copies from worker core's start of data back to the end of data
void gen_linear_read_cmd(Device *device,
                         vector<uint32_t>& prefetch_cmds,
                         vector<uint32_t>& cmd_sizes,
                         DeviceData& device_data,
                         CoreCoord worker_core,
                         uint32_t length,
                         uint32_t offset = 0) {

    vector<uint32_t> dispatch_cmds;
    const uint32_t bank_id = 0; // No interleaved pages here.

    // Stall because we are reading data that was previously written
    gen_wait_and_stall_cmd(device, prefetch_cmds, cmd_sizes);

    gen_bare_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, length);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    auto prior_end = prefetch_cmds.size();
    uint32_t addr = device_data.get_base_result_addr(device_data.get_core_type(worker_core));
    add_prefetcher_linear_read_cmd(device, prefetch_cmds, cmd_sizes, worker_core, addr + offset * sizeof(uint32_t), length);
    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE);

    // Add linear data to worker data:
    uint32_t length_words = length / sizeof(uint32_t);
    for (uint32_t i = 0; i < length_words; i++) {
        device_data.push_one(worker_core, device_data.at(worker_core, bank_id, offset + i));
    }
    device_data.pad(worker_core, bank_id, hal.get_alignment(HalMemType::L1));
}

void gen_dispatcher_delay_cmd(Device *device,
                              vector<uint32_t>& prefetch_cmds,
                              vector<uint32_t>& cmd_sizes,
                              uint32_t count) {

    vector<uint32_t> dispatch_cmds;

    CQDispatchCmd delay;
    delay.base.cmd_id = CQ_DISPATCH_CMD_DELAY;
    delay.delay.delay = count;
    add_bare_dispatcher_cmd(dispatch_cmds, delay);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
}

void gen_paged_read_dram_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint32_t>& cmd_sizes,
                   DeviceData& device_data,
                   CoreCoord worker_core) {

    uint32_t pages_read = 0;
    bool finished = false;

    log_info(tt::LogTest, "Running Paged Read DRAM test with num_pages: {} page_size: {} to worker_core: {}", dram_pages_to_read_g, dram_page_size_g, worker_core.str());
    while (!finished) {

        uint32_t start_page = pages_read % num_dram_banks_g;
        uint32_t base_addr = (pages_read / num_dram_banks_g) * dram_page_size_g;

        gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                          start_page, base_addr, dram_page_size_g, dram_pages_to_read_g, 0);

        bytes_of_data_g += dram_page_size_g * dram_pages_to_read_g;
        pages_read += dram_pages_to_read_g;

        uint32_t all_valid_device_data_size = device_data.size() * sizeof(uint32_t);
        finished = all_valid_device_data_size >= DEVICE_DATA_SIZE;
    }
}

// End-To-End Paged/Interleaved Write+Read test that does the following:
//  1. Paged Write of host data to DRAM banks by dispatcher cmd, followed by stall to avoid RAW hazard
//  2. Paged Read of DRAM banks by prefetcher, relay data to dispatcher for linear write to L1.
//  3. Do previous 2 steps in a loop, reading and writing new data until DEVICE_DATA_SIZE bytes is written to worker core.
void gen_paged_write_read_dram_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint32_t>& cmd_sizes,
                   DeviceData& device_data,
                   CoreCoord worker_core,
                   uint32_t dst_addr) {

    // Keep a running total, to correctly calculate start page, base addr, write addr based on num banks.
    uint32_t pages_written = 0;
    bool finished = false;

    log_info(tt::LogTest, "Running Paged Write+Read DRAM test with num_pages: {} page_size: {} to worker_core: {}", dram_pages_to_read_g, dram_page_size_g, worker_core.str());
    while (!finished) {

        // TODO: writes/reads are inconsistent on using page or page % num_dram_banks
        uint32_t start_page = pages_written % num_dram_banks_g;
        uint32_t base_addr = (pages_written / num_dram_banks_g) * dram_page_size_g;     // For paged read/write

        gen_dram_write_cmd(device, prefetch_cmds, cmd_sizes, device_data,
                           pages_written, dram_page_size_g, dram_pages_to_read_g);
        gen_wait_and_stall_cmd(device, prefetch_cmds, cmd_sizes);
        gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                          start_page, base_addr, dram_page_size_g, dram_pages_to_read_g, 0);
        bytes_of_data_g += dram_page_size_g * dram_pages_to_read_g;
        pages_written += dram_pages_to_read_g;

        uint32_t all_valid_device_data_size = device_data.size() * sizeof(uint32_t);
        finished = all_valid_device_data_size >= DEVICE_DATA_SIZE;

        log_debug(tt::LogTest, "{} - Finished gen cmds w/ device_data_size: 0x{:x} finished: {} pages_written: {} num_banks: {} (start_page: {} base_addr: 0x{:x}",
            __FUNCTION__, all_valid_device_data_size, finished, pages_written, num_dram_banks_g, start_page, base_addr);
    }
}

void gen_pcie_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint32_t>& cmd_sizes,
                   DeviceData& device_data,
                   CoreCoord worker_core) {

    vector<uint32_t> dispatch_cmds;

    while (device_data.size() * sizeof(uint32_t) < DEVICE_DATA_SIZE) {
        dispatch_cmds.resize(0);
        gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, pcie_transfer_size_g);
        add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
        bytes_of_data_g += pcie_transfer_size_g;
    }
}

static void pad_host_data(DeviceData& device_data) {

    one_core_data_t& host_data = device_data.get_data()[device_data.get_host_core()][0];

    int pad = dispatch_buffer_page_size_g - ((host_data.data.size() * sizeof(uint32_t)) % dispatch_buffer_page_size_g);
    pad = pad % dispatch_buffer_page_size_g;
    for (int i = 0; i < pad / sizeof(uint32_t); i++) {
        device_data.push_one(device_data.get_host_core(), 0, host_data_dirty_pattern);
    }

    if (host_data.data.size() * sizeof(uint32_t) > hugepage_issue_buffer_size_g) {
        TT_THROW("Host test hugepage data wrap not (yet) supported, reduce test size");
    }
}

void gen_host_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint32_t>& cmd_sizes,
                   DeviceData& device_data) {

    constexpr uint32_t max_data_size = DEVICE_DATA_SIZE;

    // Read data from a worker so we can get reasonable BW measurements
    // TODO: extend the DRAM mechanism for pre-fill to workers
    vector<uint32_t>data;
    for (uint32_t i = 0; i < max_data_size / sizeof(uint32_t); i++) {
        data.push_back(i);
    }
    CoreCoord phys_worker_core = device->worker_core_from_logical_core(first_worker_g);
    llrt::write_hex_vec_to_core(device->id(), phys_worker_core, data, l1_buf_base_g);
    tt::Cluster::instance().l1_barrier(device->id());

    for (int count = 1; count < 100; count++) {
        uint32_t data_size_words = std::rand() % ((max_data_size / 100 / sizeof(uint32_t)) * count) + 1;
        uint32_t data_size_bytes = data_size_words * sizeof(uint32_t);

        std::vector<uint32_t> dispatch_cmds;
        gen_bare_dispatcher_host_write_cmd(dispatch_cmds, data_size_bytes);
        add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);
        auto prior_end = prefetch_cmds.size();
        add_prefetcher_linear_read_cmd(device, prefetch_cmds, cmd_sizes, first_worker_g, l1_buf_base_g, data_size_bytes);
        uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
        cmd_sizes.push_back(new_size >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE);

        // write host writes the command back to the host
        for (auto datum : dispatch_cmds) {
            device_data.push_one(device_data.get_host_core(), 0, datum);
        }
        for (int i = 0; i < data_size_words; i++) {
            uint32_t datum = data[i];
            device_data.push_one(device_data.get_host_core(), 0, datum);
        }
        pad_host_data(device_data);
    }
}

void gen_rnd_linear_cmd(Device *device,
                        vector<uint32_t>& prefetch_cmds,
                        vector<uint32_t>& cmd_sizes,
                        DeviceData& device_data,
                        CoreCoord worker_core) {

    vector<uint32_t> dispatch_cmds;

    // Hmm, how big a size to test?
    int max_linear_cmd_read_size = 20 * dispatch_buffer_page_size_g; // XXXXX 10 *
    uint32_t size = std::rand() % max_linear_cmd_read_size;
    size &= ~(sizeof(uint32_t) - 1);
    uint32_t offset = std::rand() % dispatch_buffer_page_size_g;
    offset = (offset >> 2) << 2;
    device_data.relevel(CoreType::WORKER); // XXXXX shouldn't be needed
    if (device_data.size_at(worker_core, 0) * sizeof(uint32_t) < max_linear_cmd_read_size + offset) {
        // Not enough data yet, just bail on this cmd
        return;
    }
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, size, offset);
}

void gen_rnd_dram_paged_cmd(Device *device,
                            vector<uint32_t>& prefetch_cmds,
                            vector<uint32_t>& cmd_sizes,
                            DeviceData& device_data,
                            CoreCoord worker_core) {

    vector<uint32_t> dispatch_cmds;

    const uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
    uint32_t start_page = std::rand() % num_dram_banks_g;
    uint32_t max_page_size = big_g ? MAX_PAGE_SIZE : 4096;
    uint32_t page_size = (std::rand() % (max_page_size + 1)) & ~(dram_alignment - 1);
    if (page_size < dram_alignment) page_size = dram_alignment;
    uint32_t max_data = big_g ? DEVICE_DATA_SIZE : DEVICE_DATA_SIZE / 8;
    uint32_t max = DEVICE_DATA_SIZE - device_data.size() * sizeof(uint32_t);
    max = (max > max_data) ? max_data : max;
    // start in bottom half of valid dram data to not worry about overflowing valid data
    uint32_t base_addr = (std::rand() % (DRAM_DATA_SIZE_BYTES / 2)) & ~(dram_alignment - 1);
    uint32_t size = std::rand() % max;
    if (size < page_size) size = page_size;
    uint32_t pages = size / page_size;
    TT_ASSERT(base_addr + (start_page * page_size + pages * page_size / num_dram_banks_g) < DRAM_DATA_SIZE_BYTES);

    uint32_t length_adjust = std::rand() % page_size;
    length_adjust = (length_adjust >> 5) << 5;
    if (length_adjust >= 64 * 1024) length_adjust = 63 * 1024;

    if (device_data.size() * sizeof(uint32_t) + page_size * pages - length_adjust + l1_buf_base_g >=
        device->l1_size_per_core()) {
        // try try again
        return;
    }

    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      start_page, base_addr, page_size, pages, length_adjust);
}

void gen_rnd_inline_cmd(Device *device,
                        vector<uint32_t>& prefetch_cmds,
                        vector<uint32_t>& cmd_sizes,
                        DeviceData& device_data,
                        CoreCoord worker_core) {

    vector<uint32_t> dispatch_cmds;

    // Randomize the dispatcher command we choose to relay
    // XXXX revisit the randomness of this, these commands get under-weighted

    uint32_t which_cmd = std::rand() % 2;
    switch (which_cmd) {
    case 0:
        // unicast write
        {
            uint32_t cmd_size_bytes = CQ_PREFETCH_CMD_BARE_MIN_SIZE;
            if (debug_g) {
                cmd_size_bytes += sizeof(CQDispatchCmd);
            }
            uint32_t max_size = big_g ? max_prefetch_command_size_g : max_prefetch_command_size_g / 16;
            uint32_t max_xfer_size_16b = (max_size - cmd_size_bytes) >> 4;
            uint32_t xfer_size_16B = (std::rand() & (max_xfer_size_16b - 1));
            // Note: this may overflow the DEVICE_DATA_SIZE, but by little enough that it won't overflow L1
            uint32_t xfer_size_bytes = xfer_size_16B << 4;

            gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data,
                                             xfer_size_bytes);
        }
        break;
    case 1:
        // packed unicast write
        gen_rnd_dispatcher_packed_write_cmd(device, dispatch_cmds, device_data);
        break;
    }

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
}

void gen_rnd_debug_cmd(vector<uint32_t>& prefetch_cmds,
                       vector<uint32_t>& cmd_sizes,
                       DeviceData& device_data) {

    vector<uint32_t> rnd_payload;
    generate_random_payload(rnd_payload, std::rand() % (dispatch_buffer_page_size_g - sizeof(CQDispatchCmd)) + 1);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, rnd_payload);
}

void gen_packed_read_test(Device *device,
                          vector<uint32_t>& prefetch_cmds,
                          vector<uint32_t>& cmd_sizes,
                          DeviceData& device_data) {

    bool done = false;
    while (!done) {
        uint32_t packed_read_page_size = std::rand() % 3 + 9; // 512, 1024, 2048
        uint32_t n_sub_cmds =  relay_max_packed_paged_submcds ?  CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS : (std::rand() % 7) + 1;

        vector<uint32_t> lengths;
        uint32_t total_length = 0;
        for (uint32_t i = 0; i < n_sub_cmds; i++) {
            uint32_t max_size128b = (scratch_db_size_g / 2) >> 7;
            uint32_t length = (std::rand() % max_size128b) << 7;
            if (length < 128) length = 128;
            total_length += length;
            lengths.push_back(length);
        }

        if (device_data.size() * sizeof(uint32_t) + total_length > DEVICE_DATA_SIZE) {
            // got close-ish to the end anyway...
            done = true;
        } else {
            gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, first_worker_g, packed_read_page_size, lengths);
        }
    }
}

void gen_rnd_test(Device *device,
                  vector<uint32_t>& prefetch_cmds,
                  vector<uint32_t>& cmd_sizes,
                  DeviceData& device_data) {

    while (device_data.size() * sizeof(uint32_t) < DEVICE_DATA_SIZE) {
        // Assumes terminate is the last command...
        uint32_t cmd = std::rand() % CQ_PREFETCH_CMD_TERMINATE;
        uint32_t x = rand() % (all_workers_g.end_coord.x - first_worker_g.x);
        uint32_t y = rand() % (all_workers_g.end_coord.y - first_worker_g.y);

        CoreCoord worker_core(first_worker_g.x + x, first_worker_g.y + y);

        switch (cmd) {
        case CQ_PREFETCH_CMD_RELAY_LINEAR:
            // TODO: disabled for now
            // test issue w/ handling re-leveling of results data after paged commands
            //gen_rnd_linear_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core);
            break;
        case CQ_PREFETCH_CMD_RELAY_PAGED:
            gen_rnd_dram_paged_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core);
            break;
        case CQ_PREFETCH_CMD_RELAY_INLINE:
            gen_rnd_inline_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core);
            break;
        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            break;
        case CQ_PREFETCH_CMD_STALL:
            break;
        case CQ_PREFETCH_CMD_DEBUG:
            if (!use_dram_exec_buf_g) {
                // Splitting debug cmds not implemented for exec_bufs (yet)
                gen_rnd_debug_cmd(prefetch_cmds, cmd_sizes, device_data);
            }
            break;
        }
    }
}

void gen_prefetcher_exec_buf_cmd_and_write_to_dram(Device *device,
                                                   vector<uint32_t>& prefetch_cmds,
                                                   vector<uint32_t> exec_buf_cmds,
                                                   vector<uint32_t>& cmd_sizes) {

    vector<uint32_t> empty_payload; // don't give me grief, it is just a test

    // Add the semaphore release for prefetch_h
    CQDispatchCmd dcmd;
    memset(&dcmd, 0, sizeof(CQDispatchCmd));

    // cmddat_q in prefetch_d is re-used for exec_buf
    // prefetch_h stalls at start of exec_buf by removing its downstream credits
    // This command releases prefetch_h from the stall by restoring credits
    dcmd.base.cmd_id = CQ_DISPATCH_CMD_EXEC_BUF_END;

    vector<uint32_t> dispatch_cmds;
    vector<uint32_t> empty_sizes; // unused for the exec_buf but call below needs it
    add_bare_dispatcher_cmd(dispatch_cmds, dcmd);

    add_prefetcher_cmd(exec_buf_cmds, empty_sizes, CQ_PREFETCH_CMD_EXEC_BUF_END, dispatch_cmds);

    // writes cmds to dram
    num_dram_banks_g = device->num_banks(BufferType::DRAM);

    uint32_t page_size = 1 << exec_buf_log_page_size_g;

    uint32_t length = exec_buf_cmds.size() * sizeof(uint32_t);
    length +=
        (page_size - (length & (page_size - 1))) &
        (page_size - 1); // rounded up to full pages

    uint32_t pages = length / page_size;
    uint32_t index = 0;
    for (uint32_t page_id = 0; page_id < pages; page_id++) {
        uint32_t bank_id = page_id % num_dram_banks_g;
        auto offset = device->bank_offset(BufferType::DRAM, bank_id);
        auto dram_channel = device->dram_channel_from_bank_id(bank_id);
        auto bank_core = device->dram_core_from_dram_channel(dram_channel);

        tt::Cluster::instance().write_core(static_cast<const void*>(&exec_buf_cmds[index / sizeof(uint32_t)]),
            page_size, tt_cxy_pair(device->id(), bank_core), DRAM_EXEC_BUF_DEFAULT_BASE_ADDR + offset + (page_id / num_dram_banks_g) * page_size);

        index += page_size;
    }
    tt::Cluster::instance().dram_barrier(device->id());

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF;
    cmd.exec_buf.pad1 = 0;
    cmd.exec_buf.pad2 = 0;
    cmd.exec_buf.base_addr = DRAM_EXEC_BUF_DEFAULT_BASE_ADDR;
    cmd.exec_buf.log_page_size = exec_buf_log_page_size_g;
    cmd.exec_buf.pages = pages;

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, cmd);

    // CQ_PREFETCH_CMD_EXEC_BUF command requires stall_prefetcher flag to be set. This is MSB on FetchQ entry
    // Hacky, but set it here, on the last cmd_size (FetchQ entry write, later)
    const bool stall_prefetcher = true;
    cmd_sizes[cmd_sizes.size() - 1] |= (stall_prefetcher << ((sizeof(dispatch_constants::prefetch_q_entry_type) * 8) - 1));

}

void gen_smoke_test(Device *device,
                    vector<uint32_t>& prefetch_cmds,
                    vector<uint32_t>& cmd_sizes,
                    DeviceData& device_data,
                    CoreCoord worker_core,
                    CoreCoord another_worker_core) {

    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    vector<uint32_t> dispatch_cmds;
    const uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);

    if (!use_dram_exec_buf_g) {
        add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, empty_payload);

        vector<uint32_t> rnd_payload;
        generate_random_payload(rnd_payload, 17);
        add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, rnd_payload);
    }

    // Write to worker
    dispatch_cmds.resize(0);
    device_data.reset();
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 32);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 1026);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 8448);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Check some hard page alignment sizes
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, dispatch_buffer_page_size_g);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, dispatch_buffer_page_size_g - sizeof(CQDispatchCmd));
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data,  2 * dispatch_buffer_page_size_g);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data,  2 * dispatch_buffer_page_size_g - sizeof(CQDispatchCmd));
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Merge 4 commands in the FetchQ
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 112);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 608);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 64);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 96);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Tests sending multiple commands w/ 1 FetchQ entry
    uint32_t combined_size =
        cmd_sizes[cmd_sizes.size() - 1] + cmd_sizes[cmd_sizes.size() - 2] +
        cmd_sizes[cmd_sizes.size() - 3] + cmd_sizes[cmd_sizes.size() - 4];

    cmd_sizes[cmd_sizes.size() - 4] = combined_size;
    cmd_sizes.pop_back();
    cmd_sizes.pop_back();
    cmd_sizes.pop_back();

    // Use another_worker_core as worker_core is filling up
    // Do this before dram paged commands below which level up the results across cores
    vector<uint32_t> lengths;
    constexpr uint32_t packed_read_page_size = 10;
    lengths.push_back(256);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    lengths.push_back(512);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    lengths.resize(0);
    lengths.push_back(1024);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);
    lengths.push_back(2048);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    lengths.resize(0);
    uint32_t length_to_read = align(2080, dram_alignment);
    lengths.push_back(length_to_read);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    lengths.push_back(length_to_read);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size + 1, lengths);

    lengths.resize(0);
    lengths.push_back(scratch_db_size_g / 8);
    lengths.push_back(scratch_db_size_g / 8);
    lengths.push_back(scratch_db_size_g / 8);
    lengths.push_back(scratch_db_size_g / 4);  // won't fit in first pass
    lengths.push_back(scratch_db_size_g / 2);  // won't fit in second pass
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    lengths.resize(0);
    lengths.push_back(scratch_db_size_g / 4 + 2 * 1024 + 32);
    lengths.push_back(scratch_db_size_g / 4 + 3 * 1024 + 32);
    lengths.push_back(scratch_db_size_g / 2);
    lengths.push_back(scratch_db_size_g / 8 + 5 * 1024 + 96);
    gen_dram_packed_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, another_worker_core, packed_read_page_size, lengths);

    // Read from dram, write to worker
    // start_page, base addr, page_size, pages
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, dram_alignment, num_dram_banks_g, 0);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, dram_alignment, num_dram_banks_g, 0);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      4, dram_alignment, dram_alignment * 2, num_dram_banks_g, 0);

    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, 128, 128, 0);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      4, dram_alignment, 2048, num_dram_banks_g + 4, 0);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      5, dram_alignment, 2048, num_dram_banks_g * 3 + 1, 0);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      3, align(128, dram_alignment), 6144, num_dram_banks_g - 1, 0);

    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, 128, 128, 32);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      4, dram_alignment, 2048, num_dram_banks_g * 2, 1536);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      5, dram_alignment, 2048, num_dram_banks_g * 2 + 1, 256);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      3, align(128, dram_alignment), 6144, num_dram_banks_g - 1, 640);

    // Large pages
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, scratch_db_size_g / 2 + dram_alignment, 2, 128); // just a little larger than the scratch_db, length_adjust backs into prior page
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      0, 0, scratch_db_size_g, 2, 0); // exactly the scratch db size

    // Forces length_adjust to back into prior read.  Device reads pages, shouldn't be a problem...
    uint32_t page_size = 256 + dram_alignment;
    uint32_t length = scratch_db_size_g / 2 / page_size * page_size + page_size;
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core,
                      3, 128, page_size, length / page_size, 160);

    // Send inline data to (maybe) multiple cores
    dispatch_cmds.resize(0);
    vector<CoreCoord> worker_cores;
    worker_cores.push_back(first_worker_g);
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, device_data, 4);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    worker_cores.resize(0);
    for (uint32_t y = all_workers_g.start_coord.y; y <= all_workers_g.end_coord.y; y++) {
        for (uint32_t x = all_workers_g.start_coord.x; x <= all_workers_g.end_coord.x; x++) {
            CoreCoord worker_core(x, y);
            worker_cores.push_back(worker_core);
        }
    }
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, device_data, 12);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, device_data, 12, true);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    worker_cores.resize(0);
    worker_cores.push_back(first_worker_g);
    worker_cores.push_back(all_workers_g.end_coord);
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, device_data, 156);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // These tests copy data from earlier tests so can't run first
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 32);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 65 * 1024);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, dispatch_buffer_page_size_g - sizeof(CQDispatchCmd));
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, dispatch_buffer_page_size_g);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 2 * dispatch_buffer_page_size_g - sizeof(CQDispatchCmd));
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 2 * dispatch_buffer_page_size_g);

    // Test wait/stall
    gen_dispatcher_delay_cmd(device, prefetch_cmds, cmd_sizes, 1024 * 1024);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, device_data, 1024);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    gen_wait_and_stall_cmd(device, prefetch_cmds, cmd_sizes);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 32, device_data.size_at(worker_core, 0) - 32 / sizeof(uint32_t));

    // Touch test write offset
    // Making sure this really works by doing a write would break lots of
    // existing test infra, so not tested yet. TODO
    dispatch_cmds.resize(0);
    uint32_t write_offset = 48;
    gen_dispatcher_set_write_offset_cmd(dispatch_cmds, write_offset);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    write_offset = 0;
    gen_dispatcher_set_write_offset_cmd(dispatch_cmds, write_offset);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Test host
    if (!use_dram_exec_buf_g) {
        for (int multiplier = 1; multiplier <= 3; multiplier++) {
            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * 32);
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * 36);
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * 1024);
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * dispatch_buffer_page_size_g - 2 * sizeof(CQDispatchCmd));
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * dispatch_buffer_page_size_g - sizeof(CQDispatchCmd));
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * dispatch_buffer_page_size_g);
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * dispatch_buffer_page_size_g + sizeof(CQDispatchCmd));
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);

            dispatch_cmds.resize(0);
            gen_dispatcher_host_write_cmd(dispatch_cmds, device_data, multiplier * dispatch_buffer_page_size_g + sizeof(CQDispatchCmd));
            add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
            pad_host_data(device_data);
        }
    }

    // Test Paged DRAM Write and Read. FIXME - Needs work - hits asserts.
    // gen_dram_write_cmd(device, prefetch_cmds, cmd_sizes, device_data, 0, 32, 64, 128);
    // gen_wait_and_stall_cmd(device, prefetch_cmds, cmd_sizes);
    // gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, device_data, worker_core, 0, 32, 64, 128);
}

void gen_prefetcher_cmds(Device *device,
                         vector<uint32_t>& prefetch_cmds,
                         vector<uint32_t>& cmd_sizes,
                         DeviceData& device_data,
                         uint32_t dst_addr) {

    switch (test_type_g) {
    case 0:
        // No cmds, tests terminating - true smoke test
        break;
    case 1:
        gen_smoke_test(device, prefetch_cmds, cmd_sizes, device_data, first_worker_g, all_workers_g.end_coord);
        break;
    case 2:
        gen_rnd_test(device, prefetch_cmds, cmd_sizes, device_data);
        break;
    case 3:
        gen_pcie_test(device, prefetch_cmds, cmd_sizes, device_data, first_worker_g);
        break;
    case 4:
        gen_paged_read_dram_test(device, prefetch_cmds, cmd_sizes, device_data, first_worker_g);
        break;
    case 5:
        gen_paged_write_read_dram_test(device, prefetch_cmds, cmd_sizes, device_data, first_worker_g, dst_addr);
        break;
    case 6:
        gen_host_test(device, prefetch_cmds, cmd_sizes, device_data);
        break;
    case 7:
        gen_packed_read_test(device, prefetch_cmds, cmd_sizes, device_data);
        break;
    default:
        log_fatal("Unknown test: {}", test_type_g);
        exit(0);
        break;
    }

    device_data.overflow_check(device);
}

void gen_terminate_cmds(vector<uint32_t>& prefetch_cmds,
                        vector<uint32_t>& cmd_sizes) {
    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    vector<uint32_t> dispatch_cmds;

    // Terminate dispatcher
    dispatch_cmds.resize(0);
    gen_dispatcher_terminate_cmd(dispatch_cmds);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Terminate prefetcher
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_TERMINATE, empty_payload);
}

// Ideally would work by cachelines, but the min size is less than that
void nt_memcpy(uint8_t *__restrict dst, const uint8_t * __restrict src, size_t n)
{
    size_t num_lines = n / CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(__m128i); j++) {
            // __m128i blk = _mm_stream_load_si128((__m128i *)src);
            __m128i blk = _mm_loadu_si128((const __m128i *)src);
            /* non-temporal store */
            _mm_stream_si128((__m128i *)dst, blk);
            src += sizeof(__m128i);
            dst += sizeof(__m128i);
        }
        n -= CQ_PREFETCH_CMD_BARE_MIN_SIZE;
    }

    if (num_lines > 0)
        tt_driver_atomics::sfence();
}

void write_prefetcher_cmd(Device *device,
                          vector<uint32_t>& cmds,
                          uint32_t& cmd_offset,
                          dispatch_constants::prefetch_q_entry_type cmd_size16b,
                          uint32_t*& host_mem_ptr,
                          uint32_t& prefetch_q_dev_ptr,
                          uint32_t& prefetch_q_dev_fence,
                          uint32_t prefetch_q_base,
                          uint32_t prefetch_q_rd_ptr_addr,
                          CoreCoord phys_prefetch_core,
                          tt::Writer& prefetch_q_writer) {

    static vector<uint32_t> read_vec;  // static to avoid realloc

    // wait for space
    while (prefetch_q_dev_ptr == prefetch_q_dev_fence) {
        tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), prefetch_q_rd_ptr_addr);
        prefetch_q_dev_fence = read_vec[0];
    }

    // wrap
    if (prefetch_q_dev_ptr == prefetch_q_base + prefetch_q_entries_g * sizeof(dispatch_constants::prefetch_q_entry_type)) {
        prefetch_q_dev_ptr = prefetch_q_base;

        while (prefetch_q_dev_ptr == prefetch_q_dev_fence) {
            tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), prefetch_q_rd_ptr_addr);
            prefetch_q_dev_fence = read_vec[0];
        }
    }

    constexpr uint32_t prefetch_q_msb_mask = (1 << ((sizeof(dispatch_constants::prefetch_q_entry_type) * 8) - 1));
    uint32_t cmd_size_bytes = (cmd_size16b & ~prefetch_q_msb_mask) << dispatch_constants::PREFETCH_Q_LOG_MINSIZE;
    uint32_t cmd_size_words = cmd_size_bytes / sizeof(uint32_t);

    nt_memcpy((uint8_t *)host_mem_ptr, (uint8_t *)&cmds[cmd_offset], cmd_size_bytes);
    cmd_offset += cmd_size_words;
    host_mem_ptr += cmd_size_words;

    // This updates FetchQ where each entry of type prefetch_q_entry_type is size in 16B.
    prefetch_q_writer.write(prefetch_q_dev_ptr, cmd_size16b);

    prefetch_q_dev_ptr += sizeof(dispatch_constants::prefetch_q_entry_type);
}

void write_prefetcher_cmds(uint32_t iterations,
                           Device *device,
                           vector<uint32_t> prefetch_cmds, // yes copy for dram_exec_buf
                           vector<uint32_t>& cmd_sizes,
                           void * host_hugepage_base,
                           uint32_t dev_hugepage_base,
                           uint32_t prefetch_q_base,
                           uint32_t prefetch_q_rd_ptr_addr,
                           CoreCoord phys_prefetch_core,
                           tt::Writer& prefetch_q_writer,
                           bool is_control_only) {

    static uint32_t *host_mem_ptr;
    static uint32_t prefetch_q_dev_ptr;
    static uint32_t prefetch_q_dev_fence;

    if (!is_control_only && use_dram_exec_buf_g) {
        // Write cmds to DRAM, generate a new command to execute those commands
        cmd_sizes.resize(0);
        vector<uint32_t> exec_buf_cmds = prefetch_cmds;
        prefetch_cmds.resize(0);
        gen_prefetcher_exec_buf_cmd_and_write_to_dram(device, prefetch_cmds, exec_buf_cmds, cmd_sizes);
    }

    if (initialize_device_g) {
        vector<uint32_t> prefetch_q(DEFAULT_PREFETCH_Q_ENTRIES, 0);
        vector<uint32_t> prefetch_q_rd_ptr_addr_data;

        prefetch_q_rd_ptr_addr_data.push_back(prefetch_q_base + prefetch_q_entries_g * sizeof(dispatch_constants::prefetch_q_entry_type));
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, prefetch_q_rd_ptr_addr_data, prefetch_q_rd_ptr_addr);
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, prefetch_q, prefetch_q_base);

        host_mem_ptr = (uint32_t *)host_hugepage_base;
        prefetch_q_dev_ptr = prefetch_q_base;
        prefetch_q_dev_fence = prefetch_q_base + prefetch_q_entries_g * sizeof(dispatch_constants::prefetch_q_entry_type);
        initialize_device_g = false;
    }

    for (uint32_t i = 0; i < iterations; i++) {
        uint32_t cmd_ptr = 0;
        for (uint32_t j = 0; j < cmd_sizes.size(); j++) {
            uint32_t cmd_size_words = ((uint32_t)cmd_sizes[j] << dispatch_constants::PREFETCH_Q_LOG_MINSIZE) / sizeof(uint32_t);
            uint32_t space_at_end_for_wrap_words = CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t);
            if ((void *)(host_mem_ptr + cmd_size_words) > (void *)((uint8_t *)host_hugepage_base + hugepage_issue_buffer_size_g)) {
                // Wrap huge page
                uint32_t offset = 0;
                host_mem_ptr = (uint32_t *)host_hugepage_base;
            }

            write_prefetcher_cmd(device, prefetch_cmds, cmd_ptr, cmd_sizes[j],
                                 host_mem_ptr, prefetch_q_dev_ptr, prefetch_q_dev_fence, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core, prefetch_q_writer);
        }
    }
}

// Clear DRAM (helpful for paged write to DRAM debug to have a fresh slate)
void initialize_dram_banks(Device *device)
{

    auto num_banks = device->num_banks(BufferType::DRAM);
    auto bank_size = DRAM_DATA_SIZE_WORDS * sizeof(uint32_t); // device->bank_size(BufferType::DRAM);
    auto fill = std::vector<uint32_t>(bank_size / sizeof(uint32_t), 0xBADDF00D);

    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        auto offset = device->bank_offset(BufferType::DRAM, bank_id);
        auto dram_channel = device->dram_channel_from_bank_id(bank_id);
        auto bank_core = device->dram_core_from_dram_channel(dram_channel);

        log_info(tt::LogTest, "Initializing DRAM {} bytes for bank_id: {} core: {} at addr: 0x{:x}", bank_size, bank_id, bank_core.str(), offset);
        tt::Cluster::instance().write_core(static_cast<const void*>(fill.data()), fill.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), bank_core), offset);
    }
}

std::chrono::duration<double> run_test(uint32_t iterations,
                                       Device *device,
                                       Program& program,
                                       Device *device_r,
                                       Program& program_r,
                                       vector<uint32_t>& cmd_sizes,
                                       vector<uint32_t>& terminate_sizes,
                                       vector<uint32_t>& cmds,
                                       vector<uint32_t>& terminate_cmds,
                                       void * host_hugepage_base,
                                       uint32_t dev_hugepage_base,
                                       uint32_t prefetch_q_base,
                                       uint32_t prefetch_q_rd_ptr_addr,
                                       CoreCoord phys_prefetch_core,
                                       tt::Writer& prefetch_q_writer) {

    auto start = std::chrono::system_clock::now();

    std::thread t1 ([&]() {
        write_prefetcher_cmds(iterations, device, cmds, cmd_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core, prefetch_q_writer, false);
        write_prefetcher_cmds(1, device, terminate_cmds, terminate_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core, prefetch_q_writer, true);
    });
    tt_metal::detail::LaunchProgram(device, program, false);
    if (test_device_id_g != 0) {
        tt_metal::detail::LaunchProgram(device_r, program_r, false);
    }
    tt_metal::detail::WaitProgramDone(device, program);
    if (test_device_id_g != 0) {
        tt_metal::detail::WaitProgramDone(device_r, program_r);
    }
    t1.join();

    auto end = std::chrono::system_clock::now();

    return end-start;
}

void configure_for_single_chip(Device *device,
                               Program& program,
                               void*& host_hugepage_base,
                               uint32_t prefetch_q_base,
                               uint32_t prefetch_q_rd_ptr_addr,
                               CoreCoord& phys_prefetch_relay_mux_core,
                               CoreCoord& phys_prefetch_relay_demux_core,
                               CoreCoord& phys_dispatch_relay_mux_core,
                               CoreCoord& phys_dispatch_relay_demux_core,
                               uint32_t& packetized_path_test_results_addr,
                               uint32_t packetized_path_test_results_size) {

    const CoreType dispatch_core_type = CoreType::WORKER;
    uint32_t dispatch_buffer_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_block_size_pages() * dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
    uint32_t num_compute_cores = device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;

    CoreCoord prefetch_core = {0, 0};
    CoreCoord prefetch_d_core = {3, 0};
    CoreCoord dispatch_core = {4, 0};
    CoreCoord dispatch_h_core = {7, 0};

    phys_prefetch_core_g = device->worker_core_from_logical_core(prefetch_core);
    CoreCoord phys_prefetch_d_core = device->worker_core_from_logical_core(prefetch_d_core);
    CoreCoord phys_dispatch_core = device->worker_core_from_logical_core(dispatch_core);
    CoreCoord phys_dispatch_h_core = device->worker_core_from_logical_core(dispatch_h_core);

    // Packetized relay nodes - instantiated only if packetized_path_en_g is set
    CoreCoord prefetch_relay_mux_core = {1, 0};
    CoreCoord prefetch_relay_demux_core = {2, 0};
    CoreCoord dispatch_relay_mux_core = {5, 0};
    CoreCoord dispatch_relay_demux_core = {6, 0};

    phys_prefetch_relay_mux_core = device->worker_core_from_logical_core(prefetch_relay_mux_core);
    phys_prefetch_relay_demux_core = device->worker_core_from_logical_core(prefetch_relay_demux_core);
    phys_dispatch_relay_mux_core = device->worker_core_from_logical_core(dispatch_relay_mux_core);
    phys_dispatch_relay_demux_core = device->worker_core_from_logical_core(dispatch_relay_demux_core);

    // Packetized components will write their status + a few debug values here:
    packetized_path_test_results_addr = L1_UNRESERVED_BASE;

    // Want different buffers on each core, instead use big buffer and self-manage it
    uint32_t l1_unreserved_base_aligned = align(L1_UNRESERVED_BASE + packetized_path_test_results_size, (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE)); // Was not aligned, lately.
    TT_ASSERT((l1_buf_base_g & ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) == 0);

    uint32_t dispatch_buffer_base = l1_buf_base_g;
    uint32_t prefetch_d_buffer_base = l1_buf_base_g;
    uint32_t prefetch_d_buffer_pages = prefetch_d_buffer_size_g >> dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
    dispatch_wait_addr_g = l1_unreserved_base_aligned + hal.get_alignment(HalMemType::L1);
    vector<uint32_t>zero_data(0);
    llrt::write_hex_vec_to_core(device->id(), phys_dispatch_core, zero_data, dispatch_wait_addr_g);

    uint32_t prefetch_q_size = prefetch_q_entries_g * sizeof(dispatch_constants::prefetch_q_entry_type);
    uint32_t noc_read_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmddat_q_base = prefetch_q_base + ((prefetch_q_size + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);

    // Implementation syncs w/ device on prefetch_q but not on hugepage, ie, assumes we can't run
    // so far ahead in the hugepage that we overright un-read commands since we'll first
    // stall on the prefetch_q
    TT_ASSERT(hugepage_issue_buffer_size_g > prefetch_q_entries_g * max_prefetch_command_size_g, "Shrink the max command size or grow the hugepage buffer size or shrink the prefetch_q size");
    TT_ASSERT(cmddat_q_size_g >= 2 * max_prefetch_command_size_g);

    // NOTE: this test hijacks hugepage
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    host_hugepage_base = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
    host_hugepage_base = (void *)((uint8_t *)host_hugepage_base + dev_hugepage_base_g);
    host_hugepage_completion_buffer_base_g = (void *)((uint8_t *)host_hugepage_base + hugepage_issue_buffer_size_g);
    uint32_t dev_hugepage_completion_buffer_base = dev_hugepage_base_g + hugepage_issue_buffer_size_g;
    uint32_t* host_hugepage_completion_buffer = (uint32_t *)host_hugepage_completion_buffer_base_g;
    vector<uint32_t> tmp = {dev_hugepage_completion_buffer_base >> 4};
    CoreCoord phys_dispatch_host_core = split_dispatcher_g ? phys_dispatch_h_core : phys_dispatch_core;
    tt::llrt::write_hex_vec_to_core(device->id(), phys_dispatch_host_core, tmp, CQ_COMPLETION_WRITE_PTR);
    tt::llrt::write_hex_vec_to_core(device->id(), phys_dispatch_host_core, tmp, CQ_COMPLETION_READ_PTR);
    dirty_host_completion_buffer(host_hugepage_completion_buffer);

    const uint32_t prefetch_core_sem_0_id = tt_metal::CreateSemaphore(program, {prefetch_core}, 0);
    const uint32_t prefetch_d_core_sem_0_id = tt_metal::CreateSemaphore(program, {prefetch_d_core}, 0);
    TT_ASSERT(prefetch_core_sem_0_id == prefetch_d_core_sem_0_id);
    if (packetized_path_en_g) {
        const uint32_t prefetch_relay_mux_core_sem_0_id = tt_metal::CreateSemaphore(program, {prefetch_relay_mux_core}, 0); // unused
        const uint32_t prefetch_relay_demux_core_sem_0_id = tt_metal::CreateSemaphore(program, {prefetch_relay_demux_core}, 0); // unused
        TT_ASSERT(prefetch_relay_mux_core_sem_0_id == prefetch_relay_demux_core_sem_0_id);
        TT_ASSERT(prefetch_relay_mux_core_sem_0_id == prefetch_core_sem_0_id);
    }
    const uint32_t prefetch_sync_sem = prefetch_core_sem_0_id;

    const uint32_t prefetch_downstream_buffer_pages = split_prefetcher_g ? prefetch_d_buffer_pages : dispatch_buffer_pages;
    const uint32_t prefetch_core_sem_1_id = tt_metal::CreateSemaphore(program, {prefetch_core}, prefetch_downstream_buffer_pages);
    if (packetized_path_en_g) {
        // for the unpacketize stage, we use rptr/wptr for flow control, and poll semaphore
        // value only to update the rptr:
        const uint32_t prefetch_relay_demux_core_sem_1_id = tt_metal::CreateSemaphore(program, {prefetch_relay_demux_core}, 0);
        TT_ASSERT(prefetch_core_sem_1_id == prefetch_relay_demux_core_sem_1_id);
    }
    const uint32_t prefetch_downstream_cb_sem = prefetch_core_sem_1_id;

    const uint32_t prefetch_d_core_sem_1_id = tt_metal::CreateSemaphore(program, {prefetch_d_core}, 0);
    const uint32_t prefetch_d_core_sem_2_id = tt_metal::CreateSemaphore(program, {prefetch_d_core}, dispatch_buffer_pages);
    if (packetized_path_en_g) {
        const uint32_t prefetch_relay_mux_core_sem_1_id = tt_metal::CreateSemaphore(program, {prefetch_relay_mux_core}, 0);
        TT_ASSERT(prefetch_d_core_sem_1_id == prefetch_relay_mux_core_sem_1_id);
    }
    const uint32_t prefetch_d_upstream_cb_sem = prefetch_d_core_sem_1_id;
    const uint32_t prefetch_d_downstream_cb_sem = prefetch_d_core_sem_2_id;

    const uint32_t dispatch_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_core}, 0);
    const uint32_t dispatch_relay_demux_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0); // unused
    TT_ASSERT(dispatch_core_sem_0_id == dispatch_relay_demux_core_sem_0_id);
    const uint32_t dispatch_sync_sem = dispatch_core_sem_0_id;

    const uint32_t dispatch_core_sem_1_id = tt_metal::CreateSemaphore(program, {dispatch_core}, 0); // 1
    const uint32_t dispatch_relay_demux_core_sem_1_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0); // unused
    TT_ASSERT(dispatch_core_sem_1_id == dispatch_relay_demux_core_sem_1_id);
    const uint32_t dispatch_cb_sem = dispatch_core_sem_1_id;

    const uint32_t dispatch_core_sem_2_id = tt_metal::CreateSemaphore(program, {dispatch_core}, dispatch_buffer_pages);
    // for the unpacketize stage, we use rptr/wptr for flow control, and poll semaphore
    // value only to update the rptr:
    const uint32_t dispatch_relay_demux_core_sem_2_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0);
    TT_ASSERT(dispatch_core_sem_2_id == dispatch_relay_demux_core_sem_2_id);
    const uint32_t dispatch_downstream_cb_sem = dispatch_core_sem_2_id;

    const uint32_t dispatch_h_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_h_core}, 0);
    const uint32_t dispatch_relay_mux_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_relay_mux_core}, 0);
    TT_ASSERT(dispatch_h_core_sem_0_id == dispatch_relay_mux_core_sem_0_id);
    const uint32_t dispatch_h_cb_sem = dispatch_h_core_sem_0_id;

    std::vector<uint32_t> prefetch_compile_args = {
        dispatch_buffer_base, // overridden below for prefetch_h
        dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE, // overridden below for prefetch_h
        dispatch_buffer_pages, // overridden below for prefetch_h
        prefetch_downstream_cb_sem, // overridden below for prefetch_d
        dispatch_cb_sem, // overridden below for prefetch_h
        dev_hugepage_base_g,
        hugepage_issue_buffer_size_g,
        prefetch_q_base,
        prefetch_q_entries_g * (uint32_t)sizeof(dispatch_constants::prefetch_q_entry_type),
        prefetch_q_rd_ptr_addr,
        prefetch_q_rd_ptr_addr + sizeof(uint32_t),
        cmddat_q_base, // overridden for split below
        cmddat_q_size_g, // overridden for split below
        0, // scratch_db_base filled in below if used
        scratch_db_size_g,
        prefetch_sync_sem,
        prefetch_d_buffer_pages, // prefetch_d only
        prefetch_d_upstream_cb_sem, // prefetch_d only
        prefetch_downstream_cb_sem, // prefetch_d only
        dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
        dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
    };

    constexpr NOC my_noc_index = NOC::NOC_0;
    constexpr NOC dispatch_upstream_noc_index = NOC::NOC_1;

    if (split_prefetcher_g) {

        log_info(LogTest, "split prefetcher test, packetized_path_en={}", packetized_path_en_g);

        // prefetch_d
        uint32_t scratch_db_base = prefetch_d_buffer_base + (((prefetch_d_buffer_pages << dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE) +
                                                              noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_db_base < 1024 * 1024); // L1 size

        prefetch_compile_args[3] = prefetch_d_downstream_cb_sem;
        prefetch_compile_args[11] = prefetch_d_buffer_base;
        prefetch_compile_args[12] = prefetch_d_buffer_pages * (1 << dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
        prefetch_compile_args[13] = scratch_db_base;

        CoreCoord phys_prefetch_d_upstream_core =
            packetized_path_en_g ? phys_prefetch_relay_demux_core : phys_prefetch_core_g;
        configure_kernel_variant<true, false>(program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_d_core,
            phys_prefetch_d_core,
            phys_prefetch_d_upstream_core,
            phys_dispatch_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);

        // prefetch_h
        prefetch_compile_args[0] = prefetch_d_buffer_base;
        prefetch_compile_args[1] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        prefetch_compile_args[2] = prefetch_d_buffer_pages;
        prefetch_compile_args[3] = prefetch_downstream_cb_sem;
        prefetch_compile_args[4] = prefetch_d_upstream_cb_sem;
        prefetch_compile_args[11] = cmddat_q_base;
        prefetch_compile_args[12] = cmddat_q_size_g;
        prefetch_compile_args[13] = 0;

        CoreCoord phys_prefetch_h_downstream_core =
            packetized_path_en_g ? phys_prefetch_relay_mux_core : phys_prefetch_d_core;
        configure_kernel_variant<false, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_core,
            phys_prefetch_core_g,
            {0xffffffff, 0xffffffff}, // upstream core unused
            phys_prefetch_h_downstream_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);

        if (packetized_path_en_g) {

            uint32_t prefetch_relay_mux_queue_start_addr = prefetch_d_buffer_base;
            uint32_t prefetch_relay_mux_queue_size_bytes = prefetch_d_buffer_size_g;

            // Packetized path buffer, can be at any available address.
            constexpr uint32_t prefetch_relay_demux_queue_start_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t prefetch_relay_demux_queue_size_bytes = 0x10000;

            // For tests with checkers enabled, packetized path may time out and
            // cause the test to fail.
            // To save inner loop cycles, presently the packetized components have
            // a 32-bit timeout cycle counter so 4K cycles is the maximum timeout.
            // Setting this to 0 disables the timeout.
            uint32_t timeout_mcycles = packetized_path_timeout_en_g ? 4000 : 0;

            // These could start from 0, but we assign values that are easy to
            // identify for debug.
            constexpr uint32_t src_endpoint_start_id = 0xaa;
            constexpr uint32_t dest_endpoint_start_id = 0xbb;

            constexpr uint32_t num_src_endpoints = 1;
            constexpr uint32_t num_dest_endpoints = 1;

            std::vector<uint32_t> prefetch_relay_mux_compile_args =
            {
                0, // 0: reserved
                (prefetch_relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (prefetch_relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)phys_prefetch_core_g.x,
                                    (uint32_t)phys_prefetch_core_g.y,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (prefetch_relay_demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (prefetch_relay_demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)phys_prefetch_relay_demux_core.x, // 10: remote_tx_x
                (uint32_t)phys_prefetch_relay_demux_core.y, // 11: remote_tx_y
                0, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                packetized_path_test_results_addr, // 14: test_results_addr
                packetized_path_test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0x0,// 17: output_depacketize
                0x0,// 18: output_depacketize info
                // 19: input 0 packetize info:
                packet_switch_4B_pack(0x1,
                                    dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                    prefetch_downstream_cb_sem, // upstream sem
                                    prefetch_d_upstream_cb_sem), // local sem
                packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
                packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
                packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
            };

            log_info(LogTest, "run prefetch relay mux at x={},y={}", prefetch_relay_mux_core.x, prefetch_relay_mux_core.y);

            std::map<string, string> defines = {
                {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
            };

            auto mux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                {prefetch_relay_mux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = prefetch_relay_mux_compile_args,
                    .defines = defines,
                }
            );

            uint32_t dest_map_array[4] = {0, 1, 2, 3};
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            std::vector<uint32_t> demux_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    (prefetch_relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (prefetch_relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    num_dest_endpoints, // 3: demux_fan_out
                    packet_switch_4B_pack(phys_prefetch_d_core.x,
                                        phys_prefetch_d_core.y,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                    (prefetch_d_buffer_base >> 4), // 8: remote_tx_queue_start_addr_words 0
                    prefetch_d_buffer_size_g >> 4, // 9: remote_tx_queue_size_words 0
                    0, // 10: remote_tx_queue_start_addr_words 1
                    0, // 11: remote_tx_queue_size_words 1
                    0, // 12: remote_tx_queue_start_addr_words 2
                    0, // 13: remote_tx_queue_size_words 2
                    0, // 14: remote_tx_queue_start_addr_words 3
                    0, // 15: remote_tx_queue_size_words 3
                    (uint32_t)phys_prefetch_relay_mux_core.x, // 16: remote_rx_x
                    (uint32_t)phys_prefetch_relay_mux_core.y, // 17: remote_rx_y
                    num_dest_endpoints, // 18: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                    (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                    (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                    packetized_path_test_results_addr, // 22: test_results_addr
                    packetized_path_test_results_size, // 23: test_results_size
                    timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                    0x1, // 25: output_depacketize_mask
                    // 26: output 0 packetize info:
                    packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                        prefetch_d_upstream_cb_sem, // downstream sem
                                        prefetch_downstream_cb_sem, // local sem
                                        0), // remove header
                    packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
                };

            log_info(LogTest, "run prefetch relay demux at x={},y={}", prefetch_relay_demux_core.x, prefetch_relay_demux_core.y);
            auto demux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                {prefetch_relay_demux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = demux_compile_args,
                    .defines = defines,
                }
            );
        }

    } else {
        uint32_t scratch_db_base = cmddat_q_base + ((cmddat_q_size_g + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_db_base < 1024 * 1024); // L1 size
        prefetch_compile_args[13] = scratch_db_base;

        configure_kernel_variant<true, true>(
            program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_core,
            phys_prefetch_core_g,
            {0xffffffff, 0xffffffff}, // upstream core unused
            phys_dispatch_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);
    }

    std::vector<uint32_t> dispatch_compile_args = {
         dispatch_buffer_base,
         dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
         dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS * dispatch_constants::get(dispatch_core_type).dispatch_buffer_block_size_pages(),
         dispatch_cb_sem, // overridden below for h
         split_prefetcher_g ? prefetch_d_downstream_cb_sem : prefetch_downstream_cb_sem, // overridden below for dispatch_h
         dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
         prefetch_sync_sem,
         0, // true base of hugepage
         dev_hugepage_completion_buffer_base,
         DEFAULT_HUGEPAGE_COMPLETION_BUFFER_SIZE,
         dispatch_buffer_base,
         (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages,
         0, // unused on hd, filled in below for h and d
         0, // unused on hd, filled in below for h and d
         0, // unused unless tunneler is between h and d
         split_prefetcher_g,
         NOC_XY_ENCODING(phys_prefetch_core_g.x, phys_prefetch_core_g.y),
         prefetch_downstream_cb_sem,
         prefetch_downstream_buffer_pages,
         num_compute_cores // max_write_packed_cores
    };

    CoreCoord phys_upstream_from_dispatch_core = split_prefetcher_g ? phys_prefetch_d_core : phys_prefetch_core_g;
    if (split_dispatcher_g) {

        log_info(LogTest, "split dispatcher test, packetized_path_en={}", packetized_path_en_g);

        // dispatch_hd and dispatch_d
        uint32_t dispatch_d_preamble_size =
            packetized_path_en_g ? sizeof(dispatch_packet_header_t) : 0;
        dispatch_compile_args[12] = dispatch_downstream_cb_sem;
        dispatch_compile_args[13] = dispatch_h_cb_sem;
        dispatch_compile_args[14] = dispatch_d_preamble_size;
        CoreCoord phys_dispatch_d_downstream_core =
            packetized_path_en_g ? phys_dispatch_relay_mux_core : phys_dispatch_h_core;
        configure_kernel_variant<true, false>(program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_core,
            phys_dispatch_core,
            phys_upstream_from_dispatch_core,
            phys_dispatch_d_downstream_core,
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);

        // dispatch_h
        dispatch_compile_args[3] = dispatch_h_cb_sem;
        dispatch_compile_args[4] = dispatch_downstream_cb_sem;
        dispatch_compile_args[12] = dispatch_h_cb_sem;
        dispatch_compile_args[13] = dispatch_downstream_cb_sem;
        dispatch_compile_args[14] = 0; // preamble size
        CoreCoord phys_dispatch_h_upstream_core =
            packetized_path_en_g ? phys_dispatch_relay_demux_core : phys_dispatch_core;
        configure_kernel_variant<false, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_h_core,
            phys_dispatch_h_core,
            phys_dispatch_h_upstream_core,
            {0xffffffff,0xffffffff},
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);

        if (packetized_path_en_g) {

            uint32_t dispatch_relay_mux_queue_start_addr = dispatch_buffer_base;
            uint32_t dispatch_relay_mux_queue_size_bytes = dispatch_buffer_page_size_g*dispatch_buffer_pages;

            // Packetized path buffer, can be at any available address.
            constexpr uint32_t dispatch_relay_demux_queue_start_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t dispatch_relay_demux_queue_size_bytes = 0x10000;

            // For tests with checkers enabled, packetized path may time out and
            // cause the test to fail.
            // To save inner loop cycles, presently the packetized components have
            // a 32-bit timeout cycle counter so 4K cycles is the maximum timeout.
            // Setting this to 0 disables the timeout.
            uint32_t timeout_mcycles = packetized_path_timeout_en_g ? 4000 : 0;

            // These could start from 0, but we assign values that are easy to
            // identify for debug.
            constexpr uint32_t src_endpoint_start_id = 0xcc;
            constexpr uint32_t dest_endpoint_start_id = 0xdd;

            constexpr uint32_t num_src_endpoints = 1;
            constexpr uint32_t num_dest_endpoints = 1;

            std::vector<uint32_t> dispatch_relay_mux_compile_args =
            {
                0, // 0: reserved
                (dispatch_relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (dispatch_relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)phys_dispatch_core.x,
                                      (uint32_t)phys_dispatch_core.y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (dispatch_relay_demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (dispatch_relay_demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)phys_dispatch_relay_demux_core.x, // 10: remote_tx_x
                (uint32_t)phys_dispatch_relay_demux_core.y, // 11: remote_tx_y
                0, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                packetized_path_test_results_addr, // 14: test_results_addr
                packetized_path_test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0x0,// 17: output_depacketize
                0x0,// 18: output_depacketize info
                // 19: input 0 packetize info:
                packet_switch_4B_pack(0x1,
                                      dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                      dispatch_downstream_cb_sem, // upstream sem
                                      dispatch_h_cb_sem), // local sem
                packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
                packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
                packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
            };

            log_info(LogTest, "run dispatch relay mux at x={},y={}", dispatch_relay_mux_core.x, dispatch_relay_mux_core.y);

            std::map<string, string> defines = {
                {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
            };

            auto mux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                {dispatch_relay_mux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = dispatch_relay_mux_compile_args,
                    .defines = defines,
                }
            );

            uint32_t dest_map_array[4] = {0, 1, 2, 3};
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            std::vector<uint32_t> demux_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    (dispatch_relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (dispatch_relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    num_dest_endpoints, // 3: demux_fan_out
                    packet_switch_4B_pack(phys_dispatch_h_core.x,
                                          phys_dispatch_h_core.y,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                    packet_switch_4B_pack(0,
                                          0,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                    packet_switch_4B_pack(0,
                                          0,
                                        0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                    packet_switch_4B_pack(0,
                                          0,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                    (dispatch_buffer_base >> 4), // 8: remote_tx_queue_start_addr_words 0
                    (dispatch_buffer_page_size_g*dispatch_buffer_pages) >> 4, // 9: remote_tx_queue_size_words 0
                    0, // 10: remote_tx_queue_start_addr_words 1
                    0, // 11: remote_tx_queue_size_words 1
                    0, // 12: remote_tx_queue_start_addr_words 2
                    0, // 13: remote_tx_queue_size_words 2
                    0, // 14: remote_tx_queue_start_addr_words 3
                    0, // 15: remote_tx_queue_size_words 3
                    (uint32_t)phys_dispatch_relay_mux_core.x, // 16: remote_rx_x
                    (uint32_t)phys_dispatch_relay_mux_core.y, // 17: remote_rx_y
                    num_dest_endpoints, // 18: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                    (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                    (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                    packetized_path_test_results_addr, // 22: test_results_addr
                    packetized_path_test_results_size, // 23: test_results_size
                    timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                    0x1, // 25: output_depacketize_mask
                    // 26: output 0 packetize info:
                    packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                          dispatch_h_cb_sem, // downstream sem
                                          dispatch_downstream_cb_sem, // local sem
                                          1), // remove header
                    packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
                };

            log_info(LogTest, "run dispatch relay demux at x={},y={}", dispatch_relay_demux_core.x, dispatch_relay_demux_core.y);
            auto demux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                {dispatch_relay_demux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = demux_compile_args,
                    .defines = defines,
                }
            );
        }

    } else {
        configure_kernel_variant<true, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_core,
            phys_dispatch_core,
            phys_upstream_from_dispatch_core,
            {0xffffffff,0xffffffff},
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);
    }
}

// This is, sadly, copied and modified from above
// TODO: clean up, maybe leverage runtime structures
void configure_for_multi_chip(Device *device,
                              Program& program,
                              Device *device_r,
                              Program& program_r,
                              int device_id_l,
                              int device_id_r,
                              void*& host_hugepage_base,
                              uint32_t prefetch_q_base,
                              uint32_t prefetch_q_rd_ptr_addr,
                              CoreCoord& phys_prefetch_relay_mux_core,
                              CoreCoord& phys_prefetch_relay_demux_core,
                              CoreCoord& phys_dispatch_relay_mux_core,
                              CoreCoord& phys_dispatch_relay_demux_core,
                              uint32_t& packetized_path_test_results_addr,
                              uint32_t packetized_path_test_results_size) {

    const CoreType dispatch_core_type = CoreType::WORKER;
    uint32_t dispatch_buffer_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_block_size_pages() * dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
    uint32_t num_compute_cores = device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    TT_ASSERT(num_compute_cores == (device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y));

    CoreCoord prefetch_core = {0, 0};
    CoreCoord prefetch_d_core = {3, 0};
    CoreCoord dispatch_core = {4, 0};
    CoreCoord dispatch_h_core = {7, 0};

    phys_prefetch_core_g = device->worker_core_from_logical_core(prefetch_core);
    CoreCoord phys_prefetch_d_core = device_r->worker_core_from_logical_core(prefetch_d_core);
    CoreCoord phys_dispatch_core = device_r->worker_core_from_logical_core(dispatch_core);
    CoreCoord phys_dispatch_h_core = device->worker_core_from_logical_core(dispatch_h_core);

    // Packetized relay nodes - instantiated only if packetized_path_en_g is set
    CoreCoord prefetch_relay_mux_core = {1, 0};
    CoreCoord prefetch_relay_demux_core = {2, 0};
    CoreCoord dispatch_relay_mux_core = {5, 0};
    CoreCoord dispatch_relay_demux_core = {6, 0};

    phys_prefetch_relay_mux_core = device->worker_core_from_logical_core(prefetch_relay_mux_core);
    phys_prefetch_relay_demux_core = device_r->worker_core_from_logical_core(prefetch_relay_demux_core);
    phys_dispatch_relay_mux_core = device_r->worker_core_from_logical_core(dispatch_relay_mux_core);
    phys_dispatch_relay_demux_core = device->worker_core_from_logical_core(dispatch_relay_demux_core);
    CoreCoord tunneler_logical_core = device->get_ethernet_sockets(device_id_r)[0];
    CoreCoord tunneler_phys_core = device->ethernet_core_from_logical_core(tunneler_logical_core);
    CoreCoord r_tunneler_logical_core = device_r->get_ethernet_sockets(device_id_l)[0];
    CoreCoord r_tunneler_phys_core = device_r->ethernet_core_from_logical_core(r_tunneler_logical_core);
    log_info(LogTest, "Left Tunneler = {}", tunneler_logical_core.str());
    log_info(LogTest, "Right Tunneler = {}", r_tunneler_logical_core.str());

    // Packetized components will write their status + a few debug values here:
    packetized_path_test_results_addr = L1_UNRESERVED_BASE;
    uint32_t tunneler_queue_start_addr = 0x19000;
    uint32_t tunneler_queue_size_bytes = 0x10000;
    uint32_t tunneler_test_results_addr = 0x39000;
    uint32_t tunneler_test_results_size = 0x7000;

    // Want different buffers on each core, instead use big buffer and self-manage it
    uint32_t l1_unreserved_base_aligned = align(L1_UNRESERVED_BASE + packetized_path_test_results_size, (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE)); // Was aligned, lately.
    l1_buf_base_g = l1_unreserved_base_aligned + (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE); // Reserve a page.
    TT_ASSERT((l1_buf_base_g & ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) == 0);

    uint32_t dispatch_buffer_base = l1_buf_base_g;
    uint32_t prefetch_d_buffer_base = l1_buf_base_g;
    uint32_t prefetch_d_buffer_pages = prefetch_d_buffer_size_g >> dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
    prefetch_q_base = l1_buf_base_g;
    prefetch_q_rd_ptr_addr = l1_unreserved_base_aligned;
    dispatch_wait_addr_g = l1_unreserved_base_aligned + hal.get_alignment(HalMemType::L1);
    vector<uint32_t>zero_data(0);
    llrt::write_hex_vec_to_core(device_r->id(), phys_dispatch_core, zero_data, dispatch_wait_addr_g);

    uint32_t prefetch_q_size = prefetch_q_entries_g * sizeof(dispatch_constants::prefetch_q_entry_type);
    uint32_t noc_read_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmddat_q_base = prefetch_q_base + ((prefetch_q_size + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);

    // Implementation syncs w/ device on prefetch_q but not on hugepage, ie, assumes we can't run
    // so far ahead in the hugepage that we overright un-read commands since we'll first
    // stall on the prefetch_q
    TT_ASSERT(hugepage_issue_buffer_size_g > prefetch_q_entries_g * max_prefetch_command_size_g, "Shrink the max command size or grow the hugepage buffer size or shrink the prefetch_q size");
    TT_ASSERT(cmddat_q_size_g >= 2 * max_prefetch_command_size_g);

    // NOTE: this test hijacks hugepage
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    host_hugepage_base = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
    host_hugepage_base = (void *)((uint8_t *)host_hugepage_base + dev_hugepage_base_g);
    host_hugepage_completion_buffer_base_g = (void *)((uint8_t *)host_hugepage_base + hugepage_issue_buffer_size_g);
    uint32_t dev_hugepage_completion_buffer_base = dev_hugepage_base_g + hugepage_issue_buffer_size_g;
    uint32_t* host_hugepage_completion_buffer = (uint32_t *)host_hugepage_completion_buffer_base_g;
    vector<uint32_t> tmp = {dev_hugepage_completion_buffer_base >> 4};
    CoreCoord phys_dispatch_host_core = split_dispatcher_g ? phys_dispatch_h_core : phys_dispatch_core;
    tt::llrt::write_hex_vec_to_core(device->id(), phys_dispatch_host_core, tmp, CQ_COMPLETION_WRITE_PTR);
    tt::llrt::write_hex_vec_to_core(device->id(), phys_dispatch_host_core, tmp, CQ_COMPLETION_READ_PTR);
    dirty_host_completion_buffer(host_hugepage_completion_buffer);

    const uint32_t prefetch_core_sem_0_id = tt_metal::CreateSemaphore(program, {prefetch_core}, 0);
    const uint32_t prefetch_d_core_sem_0_id = tt_metal::CreateSemaphore(program_r, {prefetch_d_core}, 0);
    TT_ASSERT(prefetch_core_sem_0_id == prefetch_d_core_sem_0_id);
    if (packetized_path_en_g) {
        const uint32_t prefetch_relay_mux_core_sem_0_id =
            tt_metal::CreateSemaphore(program, {prefetch_relay_mux_core}, 0);  // unused
        const uint32_t prefetch_relay_demux_core_sem_0_id =
            tt_metal::CreateSemaphore(program_r, {prefetch_relay_demux_core}, 0);  // unused
        TT_ASSERT(prefetch_relay_mux_core_sem_0_id == prefetch_relay_demux_core_sem_0_id);
        TT_ASSERT(prefetch_relay_mux_core_sem_0_id == prefetch_core_sem_0_id);
    }
    const uint32_t prefetch_sync_sem = prefetch_core_sem_0_id;

    const uint32_t prefetch_downstream_buffer_pages = split_prefetcher_g ? prefetch_d_buffer_pages : dispatch_buffer_pages;
    const uint32_t prefetch_core_sem_1_id = tt_metal::CreateSemaphore(program, {prefetch_core}, prefetch_downstream_buffer_pages);
    if (packetized_path_en_g) {
        // for the unpacketize stage, we use rptr/wptr for flow control, and poll semaphore
        // value only to update the rptr:
        const uint32_t prefetch_relay_demux_core_sem_1_id = tt_metal::CreateSemaphore(program_r, {prefetch_relay_demux_core}, 0);
        TT_ASSERT(prefetch_core_sem_1_id == prefetch_relay_demux_core_sem_1_id);
    }
    const uint32_t prefetch_downstream_cb_sem = prefetch_core_sem_1_id;

    const uint32_t prefetch_d_core_sem_1_id = tt_metal::CreateSemaphore(program_r, {prefetch_d_core}, 0);
    const uint32_t prefetch_d_core_sem_2_id =
        tt_metal::CreateSemaphore(program_r, {prefetch_d_core}, dispatch_buffer_pages);
    if (packetized_path_en_g) {
        const uint32_t prefetch_relay_mux_core_sem_1_id =
            tt_metal::CreateSemaphore(program, {prefetch_relay_mux_core}, 0);
        TT_ASSERT(prefetch_d_core_sem_1_id == prefetch_relay_mux_core_sem_1_id);
    }
    const uint32_t prefetch_d_upstream_cb_sem = prefetch_d_core_sem_1_id;
    const uint32_t prefetch_d_downstream_cb_sem = prefetch_d_core_sem_2_id;

    const uint32_t dispatch_core_sem_0_id = tt_metal::CreateSemaphore(program_r, {dispatch_core}, 0);
    const uint32_t dispatch_relay_demux_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0); // unused
    TT_ASSERT(dispatch_core_sem_0_id == dispatch_relay_demux_core_sem_0_id);
    const uint32_t dispatch_sync_sem = dispatch_core_sem_0_id;

    const uint32_t dispatch_core_sem_1_id =tt_metal::CreateSemaphore(program_r, {dispatch_core}, 0);
    const uint32_t dispatch_relay_demux_core_sem_1_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0); // unused
    TT_ASSERT(dispatch_core_sem_1_id == dispatch_relay_demux_core_sem_1_id);
    const uint32_t dispatch_cb_sem = dispatch_core_sem_1_id;

    const uint32_t dispatch_core_sem_2_id = tt_metal::CreateSemaphore(program_r, {dispatch_core}, dispatch_buffer_pages);
    // for the unpacketize stage, we use rptr/wptr for flow control, and poll semaphore
    // value only to update the rptr:
    const uint32_t dispatch_relay_demux_core_sem_2_id = tt_metal::CreateSemaphore(program, {dispatch_relay_demux_core}, 0);
    TT_ASSERT(dispatch_core_sem_2_id == dispatch_relay_demux_core_sem_2_id);
    const uint32_t dispatch_downstream_cb_sem = dispatch_core_sem_2_id;

    const uint32_t dispatch_h_core_sem_0_id = tt_metal::CreateSemaphore(program, {dispatch_h_core}, 0);
    const uint32_t dispatch_relay_mux_core_sem_0_id = tt_metal::CreateSemaphore(program_r, {dispatch_relay_mux_core}, 0);
    TT_ASSERT(dispatch_h_core_sem_0_id == dispatch_relay_mux_core_sem_0_id);
    const uint32_t dispatch_h_cb_sem = dispatch_h_core_sem_0_id;

    std::vector<uint32_t> prefetch_compile_args = {
        dispatch_buffer_base, // overridden below for prefetch_h
        dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE, // overridden below for prefetch_h
        dispatch_buffer_pages, // overridden below for prefetch_h
        prefetch_downstream_cb_sem, // overridden below for prefetch_d
        dispatch_cb_sem, // overridden below for prefetch_h
        dev_hugepage_base_g,
        hugepage_issue_buffer_size_g,
        prefetch_q_base,
        prefetch_q_entries_g * (uint32_t)sizeof(dispatch_constants::prefetch_q_entry_type),
        prefetch_q_rd_ptr_addr,
        prefetch_q_rd_ptr_addr + sizeof(uint32_t),
        cmddat_q_base, // overridden for split below
        cmddat_q_size_g, // overridden for split below
        0, // scratch_db_base filled in below if used
        scratch_db_size_g,
        prefetch_sync_sem,
        prefetch_d_buffer_pages, // prefetch_d only
        prefetch_d_upstream_cb_sem, // prefetch_d only
        prefetch_downstream_cb_sem, // prefetch_d only
        dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
        dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
    };

    constexpr NOC my_noc_index = NOC::NOC_0;
    constexpr NOC dispatch_upstream_noc_index = NOC::NOC_1;

    if (split_prefetcher_g) {

        log_info(LogTest, "split prefetcher test, packetized_path_en={}", packetized_path_en_g);

        // prefetch_d
        uint32_t scratch_db_base = prefetch_d_buffer_base + (((prefetch_d_buffer_pages << dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE) +
                                                              noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_db_base < 1024 * 1024); // L1 size

        prefetch_compile_args[3] = prefetch_d_downstream_cb_sem;
        prefetch_compile_args[11] = prefetch_d_buffer_base;
        prefetch_compile_args[12] = prefetch_d_buffer_pages * (1 << dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
        prefetch_compile_args[13] = scratch_db_base;

        CoreCoord phys_prefetch_d_upstream_core =
            packetized_path_en_g ? phys_prefetch_relay_demux_core : phys_prefetch_core_g;
        configure_kernel_variant<true, false>(program_r,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_d_core,
            phys_prefetch_d_core,
            phys_prefetch_d_upstream_core,
            phys_dispatch_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);

        // prefetch_h
        prefetch_compile_args[0] = prefetch_d_buffer_base;
        prefetch_compile_args[1] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        prefetch_compile_args[2] = prefetch_d_buffer_pages;
        prefetch_compile_args[3] = prefetch_downstream_cb_sem;
        prefetch_compile_args[4] = prefetch_d_upstream_cb_sem;
        prefetch_compile_args[11] = cmddat_q_base;
        prefetch_compile_args[12] = cmddat_q_size_g;
        prefetch_compile_args[13] = 0;

        CoreCoord phys_prefetch_h_downstream_core =
            packetized_path_en_g ? phys_prefetch_relay_mux_core : phys_prefetch_d_core;
        configure_kernel_variant<false, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_core,
            phys_prefetch_core_g,
            {0xffffffff, 0xffffffff}, // upstream core unused
            phys_prefetch_h_downstream_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);

        if (packetized_path_en_g) {

            uint32_t prefetch_relay_mux_queue_start_addr = prefetch_d_buffer_base;
            uint32_t prefetch_relay_mux_queue_size_bytes = prefetch_d_buffer_size_g;

            // Packetized path buffer, can be at any available address.
            constexpr uint32_t prefetch_relay_demux_queue_start_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t prefetch_relay_demux_queue_size_bytes = 0x10000;

            // For tests with checkers enabled, packetized path may time out and
            // cause the test to fail.
            // To save inner loop cycles, presently the packetized components have
            // a 32-bit timeout cycle counter so 4K cycles is the maximum timeout.
            // Setting this to 0 disables the timeout.
            uint32_t timeout_mcycles = packetized_path_timeout_en_g ? 4000 : 0;

            // These could start from 0, but we assign values that are easy to
            // identify for debug.
            constexpr uint32_t src_endpoint_start_id = 0xaa;
            constexpr uint32_t dest_endpoint_start_id = 0xbb;

            constexpr uint32_t num_src_endpoints = 1;
            constexpr uint32_t num_dest_endpoints = 1;

            std::vector<uint32_t> prefetch_relay_mux_compile_args =
            {
                0, // 0: reserved
                (prefetch_relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (prefetch_relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)phys_prefetch_core_g.x,
                                    (uint32_t)phys_prefetch_core_g.y,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (tunneler_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)tunneler_phys_core.x, // 10: remote_tx_x
                (uint32_t)tunneler_phys_core.y, // 11: remote_tx_y
                0, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                packetized_path_test_results_addr, // 14: test_results_addr
                packetized_path_test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0x0,// 17: output_depacketize
                0x0,// 18: output_depacketize info
                // 19: input 0 packetize info:
                packet_switch_4B_pack(0x1,
                                    dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                    prefetch_downstream_cb_sem, // upstream sem
                                    prefetch_d_upstream_cb_sem), // local sem
                packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
                packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
                packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
            };

            log_info(LogTest, "run prefetch relay mux at x={},y={}", prefetch_relay_mux_core.x, prefetch_relay_mux_core.y);

            std::map<string, string> defines = {
                {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
            };

            auto mux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                {prefetch_relay_mux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = prefetch_relay_mux_compile_args,
                    .defines = defines,
                }
            );

            std::vector<uint32_t> tunneler_l_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    2, // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                    (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                    (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                    packet_switch_4B_pack(r_tunneler_phys_core.x,
                                        r_tunneler_phys_core.y,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::ETH), // 4: remote_receiver_0_info
                    packet_switch_4B_pack(phys_dispatch_relay_demux_core.x,
                                        phys_dispatch_relay_demux_core.y,
                                        num_dest_endpoints,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_receiver_1_info
                    tunneler_queue_start_addr >> 4, // 6: remote_receiver_queue_start_addr_words 0
                    tunneler_queue_size_bytes >> 4, // 7: remote_receiver_queue_size_words 0
                    (prefetch_relay_demux_queue_start_addr >> 4), // 8: remote_receiver_queue_start_addr_words 1
                    (prefetch_relay_demux_queue_size_bytes >> 4), // 9: remote_receiver_queue_size_words 1
                    packet_switch_4B_pack(phys_prefetch_relay_mux_core.x,
                                        phys_prefetch_relay_mux_core.y,
                                        num_dest_endpoints,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 10: remote_sender_0_info
                    packet_switch_4B_pack(r_tunneler_phys_core.x,
                                        r_tunneler_phys_core.y,
                                        3,
                                        (uint32_t)DispatchRemoteNetworkType::ETH), // 11: remote_sender_1_info
                    tunneler_test_results_addr, // 12: test_results_addr
                    tunneler_test_results_size, // 13: test_results_size
                    timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
                };

            auto tunneler_l_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
                tunneler_logical_core,
                tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0,
                    .compile_args = tunneler_l_compile_args
                }
            );

            std::vector<uint32_t> tunneler_r_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    2,  // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                    (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                    (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                    packet_switch_4B_pack(phys_prefetch_relay_demux_core.x,
                                          phys_prefetch_relay_demux_core.y,
                                          num_dest_endpoints,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_receiver_0_info
                    packet_switch_4B_pack(tunneler_phys_core.x,
                                          tunneler_phys_core.y,
                                          1,
                                          (uint32_t)DispatchRemoteNetworkType::ETH), // 5: remote_receiver_1_info
                    (prefetch_relay_demux_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
                    (prefetch_relay_demux_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
                    (tunneler_queue_start_addr + tunneler_queue_size_bytes) >> 4, // 8: remote_receiver_queue_start_addr_words 1
                    tunneler_queue_size_bytes >> 4, // 9: remote_receiver_queue_size_words 1
                    packet_switch_4B_pack(tunneler_phys_core.x,
                                        tunneler_phys_core.y,
                                        2,
                                        (uint32_t)DispatchRemoteNetworkType::ETH), // 10: remote_sender_0_info
                    packet_switch_4B_pack(phys_dispatch_relay_mux_core.x,
                                        phys_dispatch_relay_mux_core.y,
                                        num_dest_endpoints,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 11: remote_sender_1_info
                    tunneler_test_results_addr, // 12: test_results_addr
                    tunneler_test_results_size, // 13: test_results_size
                    timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
                };

            auto tunneler_r_kernel = tt_metal::CreateKernel(
                program_r,
                "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
                r_tunneler_logical_core,
                tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0,
                    .compile_args = tunneler_r_compile_args
                }
            );

            uint32_t dest_map_array[4] = {0, 1, 2, 3};
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            std::vector<uint32_t> demux_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    (prefetch_relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (prefetch_relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    num_dest_endpoints, // 3: demux_fan_out
                    packet_switch_4B_pack(phys_prefetch_d_core.x,
                                        phys_prefetch_d_core.y,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                    packet_switch_4B_pack(0,
                                        0,
                                        0,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                    (prefetch_d_buffer_base >> 4), // 8: remote_tx_queue_start_addr_words 0
                    prefetch_d_buffer_size_g >> 4, // 9: remote_tx_queue_size_words 0
                    0, // 10: remote_tx_queue_start_addr_words 1
                    0, // 11: remote_tx_queue_size_words 1
                    0, // 12: remote_tx_queue_start_addr_words 2
                    0, // 13: remote_tx_queue_size_words 2
                    0, // 14: remote_tx_queue_start_addr_words 3
                    0, // 15: remote_tx_queue_size_words 3
                    //(uint32_t)phys_relay_mux_core.x, // 16: remote_rx_x
                    //(uint32_t)phys_relay_mux_core.y, // 17: remote_rx_y
                    //num_dest_endpoints, // 18: remote_rx_queue_id
                    (uint32_t)r_tunneler_phys_core.x, // 16: remote_rx_x
                    (uint32_t)r_tunneler_phys_core.y, // 17: remote_rx_y
                    2, // 18: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                    (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                    (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                    packetized_path_test_results_addr, // 22: test_results_addr
                    packetized_path_test_results_size, // 23: test_results_size
                    timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                    0x1, // 25: output_depacketize_mask
                    // 26: output 0 packetize info:
                    packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                        prefetch_d_upstream_cb_sem, // downstream sem
                                        prefetch_downstream_cb_sem, // local sem
                                        0), // remove header
                    packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
                };

            log_info(LogTest, "run prefetch relay demux at x={},y={}", prefetch_relay_demux_core.x, prefetch_relay_demux_core.y);
            auto demux_kernel = tt_metal::CreateKernel(
                program_r,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                {prefetch_relay_demux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = demux_compile_args,
                    .defines = defines,
                }
            );
        }

    } else {
        uint32_t scratch_db_base = cmddat_q_base + ((cmddat_q_size_g + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_db_base < 1024 * 1024); // L1 size
        prefetch_compile_args[13] = scratch_db_base;

        configure_kernel_variant<true, true>(
            program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_compile_args,
            prefetch_core,
            phys_prefetch_core_g,
            {0xffffffff, 0xffffffff}, // upstream core unused
            phys_dispatch_core,
            device,
            my_noc_index,
            my_noc_index,
            my_noc_index);
    }

    std::vector<uint32_t> dispatch_compile_args = {
         dispatch_buffer_base,
         dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
         dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS * dispatch_constants::get(dispatch_core_type).dispatch_buffer_block_size_pages(),
         dispatch_cb_sem, // overridden below for h
         split_prefetcher_g ? prefetch_d_downstream_cb_sem : prefetch_downstream_cb_sem,
         dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
         prefetch_sync_sem,
         0, // true base of hugepage
         dev_hugepage_completion_buffer_base,
         DEFAULT_HUGEPAGE_COMPLETION_BUFFER_SIZE,
         dispatch_buffer_base,
         (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages,
         0, // unused on hd, filled in below for h and d
         0, // unused on hd, filled in below for h and d
         0, // unused unless tunneler is between h and d
         split_prefetcher_g,
         NOC_XY_ENCODING(phys_prefetch_core_g.x, phys_prefetch_core_g.y),
         prefetch_downstream_cb_sem,
         prefetch_downstream_buffer_pages,
         num_compute_cores
    };

    CoreCoord phys_upstream_from_dispatch_core = split_prefetcher_g ? phys_prefetch_d_core : phys_prefetch_core_g;
    if (split_dispatcher_g) {

        log_info(LogTest, "split dispatcher test, packetized_path_en={}", packetized_path_en_g);

        // dispatch_hd and dispatch_d
        uint32_t dispatch_d_preamble_size =
            packetized_path_en_g ? sizeof(dispatch_packet_header_t) : 0;
        dispatch_compile_args[12] = dispatch_downstream_cb_sem;
        dispatch_compile_args[13] = dispatch_h_cb_sem;
        dispatch_compile_args[14] = dispatch_d_preamble_size;
        CoreCoord phys_dispatch_d_downstream_core =
            packetized_path_en_g ? phys_dispatch_relay_mux_core : phys_dispatch_h_core;
        configure_kernel_variant<true, false>(program_r,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_core,
            phys_dispatch_core,
            phys_upstream_from_dispatch_core,
            phys_dispatch_d_downstream_core,
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);

        // dispatch_h
        dispatch_compile_args[3] = dispatch_h_cb_sem;
        dispatch_compile_args[12] = dispatch_h_cb_sem;
        dispatch_compile_args[13] = dispatch_downstream_cb_sem;
        dispatch_compile_args[14] = 0; // preamble size
        CoreCoord phys_dispatch_h_upstream_core =
            packetized_path_en_g ? phys_dispatch_relay_demux_core : phys_dispatch_core;
        configure_kernel_variant<false, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_h_core,
            phys_dispatch_h_core,
            phys_dispatch_h_upstream_core,
            {0xffffffff,0xffffffff},
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);

        if (packetized_path_en_g) {

            uint32_t dispatch_relay_mux_queue_start_addr = dispatch_buffer_base;
            uint32_t dispatch_relay_mux_queue_size_bytes = dispatch_buffer_page_size_g*dispatch_buffer_pages;

            // Packetized path buffer, can be at any available address.
            constexpr uint32_t dispatch_relay_demux_queue_start_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t dispatch_relay_demux_queue_size_bytes = 0x10000;

            // For tests with checkers enabled, packetized path may time out and
            // cause the test to fail.
            // To save inner loop cycles, presently the packetized components have
            // a 32-bit timeout cycle counter so 4K cycles is the maximum timeout.
            // Setting this to 0 disables the timeout.
            uint32_t timeout_mcycles = packetized_path_timeout_en_g ? 4000 : 0;

            // These could start from 0, but we assign values that are easy to
            // identify for debug.
            constexpr uint32_t src_endpoint_start_id = 0xcc;
            constexpr uint32_t dest_endpoint_start_id = 0xdd;

            constexpr uint32_t num_src_endpoints = 1;
            constexpr uint32_t num_dest_endpoints = 1;

            std::vector<uint32_t> dispatch_relay_mux_compile_args =
            {
                0, // 0: reserved
                (dispatch_relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (dispatch_relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)phys_dispatch_core.x,
                                      (uint32_t)phys_dispatch_core.y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack(0,
                                      0,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                //(dispatch_relay_demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                //(dispatch_relay_demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                //(uint32_t)phys_dispatch_relay_demux_core.x, // 10: remote_tx_x
                //(uint32_t)phys_dispatch_relay_demux_core.y, // 11: remote_tx_y
                //num_dest_endpoints, // 12: remote_tx_queue_id
                ((tunneler_queue_start_addr + tunneler_queue_size_bytes) >> 4), // 8: remote_tx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)r_tunneler_phys_core.x, // 10: remote_tx_x
                (uint32_t)r_tunneler_phys_core.y, // 11: remote_tx_y
                1, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                packetized_path_test_results_addr, // 14: test_results_addr
                packetized_path_test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0x0,// 17: output_depacketize
                0x0,// 18: output_depacketize info
                // 19: input 0 packetize info:
                packet_switch_4B_pack(0x1,
                                      dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                      dispatch_downstream_cb_sem, // upstream sem
                                      dispatch_h_cb_sem), // local sem
                packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
                packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
                packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
                packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
            };

            log_info(LogTest, "run dispatch relay mux at x={},y={}", dispatch_relay_mux_core.x, dispatch_relay_mux_core.y);

            std::map<string, string> defines = {
                {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
            };

            auto mux_kernel = tt_metal::CreateKernel(
                program_r,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                {dispatch_relay_mux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = dispatch_relay_mux_compile_args,
                    .defines = defines,
                }
            );

            uint32_t dest_map_array[4] = {0, 1, 2, 3};
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            std::vector<uint32_t> demux_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    (dispatch_relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (dispatch_relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    num_dest_endpoints, // 3: demux_fan_out
                    packet_switch_4B_pack(phys_dispatch_h_core.x,
                                          phys_dispatch_h_core.y,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                    packet_switch_4B_pack(0,
                                          0,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                    packet_switch_4B_pack(0,
                                          0,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                    packet_switch_4B_pack(0,
                                          0,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                    (dispatch_buffer_base >> 4), // 8: remote_tx_queue_start_addr_words 0
                    (dispatch_buffer_page_size_g*dispatch_buffer_pages) >> 4, // 9: remote_tx_queue_size_words 0
                    0, // 10: remote_tx_queue_start_addr_words 1
                    0, // 11: remote_tx_queue_size_words 1
                    0, // 12: remote_tx_queue_start_addr_words 2
                    0, // 13: remote_tx_queue_size_words 2
                    0, // 14: remote_tx_queue_start_addr_words 3
                    0, // 15: remote_tx_queue_size_words 3
                    //(uint32_t)phys_dispatch_relay_mux_core.x, // 16: remote_rx_x
                    //(uint32_t)phys_dispatch_relay_mux_core.y, // 17: remote_rx_y
                    //num_dest_endpoints, // 18: remote_rx_queue_id
                    (uint32_t)tunneler_phys_core.x,
                    (uint32_t)tunneler_phys_core.y,
                    3,
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                    (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                    (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                    packetized_path_test_results_addr, // 22: test_results_addr
                    packetized_path_test_results_size, // 23: test_results_size
                    timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                    0x1, // 25: output_depacketize_mask
                    // 26: output 0 packetize info:
                    packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                          dispatch_h_cb_sem, // downstream sem
                                          dispatch_downstream_cb_sem, // local sem
                                          1), // remove header
                    packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
                    packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
                };

            log_info(LogTest, "run dispatch relay demux at x={},y={}", dispatch_relay_demux_core.x, dispatch_relay_demux_core.y);
            auto demux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                {dispatch_relay_demux_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = demux_compile_args,
                    .defines = defines,
                }
            );
        }

    } else {
        configure_kernel_variant<true, true>(program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_compile_args,
            dispatch_core,
            phys_dispatch_core,
            phys_upstream_from_dispatch_core,
            {0xffffffff,0xffffffff},
            device,
            my_noc_index,
            dispatch_upstream_noc_index,
            my_noc_index);
    }
}

int main(int argc, char **argv) {

    log_info(tt::LogTest, "test_prefetcher.cpp - Test Start");
    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    // TODO(abhullar): Have to initialize the HAL explicitly here because it is accessed before Device initializes it
    tt::ARCH arch = tt::Cluster::instance().arch();
    hal.initialize(arch);
    init(argc, argv);

    bool pass = true;
    try {
        int num_devices = tt_metal::GetNumAvailableDevices();
        if (test_device_id_g >= num_devices) {
            log_info(LogTest,
                "Device {} is not valid. Highest valid device id = {}.",
                test_device_id_g, num_devices-1);
            throw std::runtime_error("Invalid Device Id.");
        }
        int device_id_l = test_device_id_g;
        int device_id_r = -1;

        tt_metal::Device *device = tt_metal::CreateDevice(test_device_id_g);
        tt_metal::Device *device_r = nullptr;
        if (test_device_id_g == 0) {
            device_r = device;
        } else {
            auto const& device_active_eth_cores = device->get_active_ethernet_cores();
            auto remote_chips = tt::Cluster::instance().get_devices_controlled_by_mmio_device(device_id_l);
            //remove mmio chip from the set. get_devices_controlled_by_mmio_device() returns a set that
            //holds mmio chips as well as remote chips accessed through that mmmio chip.
            remote_chips.erase(device_id_l);
            if (device_active_eth_cores.size() == 0) {
                log_info(LogTest,
                         "Device {} does not have enough active cores. Need 1 active ethernet core for this test.",
                         device_id_l);
                tt_metal::CloseDevice(device);
                throw std::runtime_error("Test cannot run on specified device.");
            }

            device_id_r = test_device_id_g;
            for (auto eth_core : device_active_eth_cores) {
                auto [connected_device_id, eth_receiver_core] = device->get_connected_ethernet_core(eth_core);
                if (remote_chips.find(connected_device_id) != remote_chips.end()) {
                    device_id_r = connected_device_id;
                    break;
                }
            }
            if (device_id_r == device_id_l) {
                log_info(LogTest,
                         "Device {} does not have a remote device connected to it.",
                         device_id_l);
                tt_metal::CloseDevice(device);
                throw std::runtime_error("Test cannot run on specified device.");
            }

            device_r = tt_metal::CreateDevice(device_id_r);
        }

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::Program program_r = tt_metal::CreateProgram();

        void* host_hugepage_base;
        uint32_t packetized_path_test_results_size = 1024;
        uint32_t l1_unreserved_base_aligned = align(L1_UNRESERVED_BASE + packetized_path_test_results_size, (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE)); // Was not aligned, lately.
        l1_buf_base_g = l1_unreserved_base_aligned + (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE); // Reserve a page.
        uint32_t prefetch_q_base = l1_buf_base_g;
        uint32_t prefetch_q_rd_ptr_addr = l1_unreserved_base_aligned;
        CoreCoord phys_prefetch_relay_mux_core;
        CoreCoord phys_prefetch_relay_demux_core;
        CoreCoord phys_dispatch_relay_mux_core;
        CoreCoord phys_dispatch_relay_demux_core;
        uint32_t packetized_path_test_results_addr;
        if (test_device_id_g == 0) {
            configure_for_single_chip(device, program,
                                      host_hugepage_base,
                                      prefetch_q_base, prefetch_q_rd_ptr_addr,
                                      phys_prefetch_relay_mux_core, phys_prefetch_relay_demux_core,
                                      phys_dispatch_relay_mux_core, phys_dispatch_relay_demux_core,
                                      packetized_path_test_results_addr, packetized_path_test_results_size);
        } else {
            configure_for_multi_chip(device, program,
                                     device_r, program_r,
                                     device_id_l, device_id_r,
                                     host_hugepage_base,
                                     prefetch_q_base, prefetch_q_rd_ptr_addr,
                                     phys_prefetch_relay_mux_core, phys_prefetch_relay_demux_core,
                                     phys_dispatch_relay_mux_core, phys_dispatch_relay_demux_core,
                                     packetized_path_test_results_addr, packetized_path_test_results_size);
        }

        if ((1 << exec_buf_log_page_size_g) * device->num_banks(BufferType::DRAM) > cmddat_q_size_g) {
            log_fatal("Exec buffer must fit in cmddat_q, page size too large ({})", 1 << exec_buf_log_page_size_g);
            exit(0);
        }

        log_info(LogTest, "Hugepage buffer size {}", std::to_string(hugepage_issue_buffer_size_g));
        log_info(LogTest, "Prefetch prefetch_q entries {}", std::to_string(prefetch_q_entries_g));
        log_info(LogTest, "CmdDat buffer size {}", std::to_string(cmddat_q_size_g));
        log_info(LogTest, "Prefetch scratch buffer size {}", std::to_string(scratch_db_size_g));
        log_info(LogTest, "Max command size {}", std::to_string(max_prefetch_command_size_g));
        if (test_type_g >= 2) {
            perf_test_g = true;
        }
        if (test_type_g == 3) {
            perf_test_g = true;
            log_info(LogTest, "PCIE transfer size {}", std::to_string(pcie_transfer_size_g));
        }
        if (test_type_g == 4) {
            perf_test_g = true;
            log_info(LogTest, "DRAM page size {}", std::to_string(dram_page_size_g));
            log_info(LogTest, "DRAM pages to read {}", std::to_string(dram_pages_to_read_g));
        }
        if (use_dram_exec_buf_g) {
            log_info(LogTest, "Exec commands in DRAM exec_buf w/ page_size={}", 1 << exec_buf_log_page_size_g);
        }
        if (debug_g) {
            log_info(LogTest, "Debug mode enabled");
        }
        log_info(LogTest, "Iterations: {}", iterations_g);

        tt::Writer prefetch_q_writer =  tt::Cluster::instance().get_static_tlb_writer(tt_cxy_pair(device->id(), phys_prefetch_core_g));

        vector<uint32_t> cmds, terminate_cmds;
        vector<uint32_t> cmd_sizes, terminate_sizes;
        DeviceData device_data(device, all_workers_g, l1_buf_base_g, DRAM_DATA_BASE_ADDR, (uint32_t*)host_hugepage_completion_buffer_base_g, false, DRAM_DATA_SIZE_WORDS);
        num_dram_banks_g = device->num_banks(BufferType::DRAM);

        if (debug_g) {
            initialize_dram_banks(device);
        }

        tt::Cluster::instance().l1_barrier(device->id());
        tt::Cluster::instance().dram_barrier(device->id());
        if (test_device_id_g != 0) {
            tt::Cluster::instance().l1_barrier(device_r->id());
            tt::Cluster::instance().dram_barrier(device_r->id());
        }

        // Cache stuff
        gen_terminate_cmds(terminate_cmds, terminate_sizes);
        gen_prefetcher_cmds(device_r, cmds, cmd_sizes, device_data, l1_buf_base_g);
        if (warmup_g) {
            log_info(tt::LogTest, "Warming up cache now...");
            run_test(1, device, program, device_r, program_r, cmd_sizes, terminate_sizes, cmds, terminate_cmds, host_hugepage_base, dev_hugepage_base_g, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core_g, prefetch_q_writer);
            initialize_device_g = true;
        }

        log_info(tt::LogTest, "Generating cmds and running {} iterations (readback_every_iter: {}) now...", iterations_g, readback_every_iteration_g);
        if (readback_every_iteration_g) {
            for (int i = 0; i < iterations_g; i++) {
                log_info(LogTest, "Iteration: {}", std::to_string(i));
                initialize_device_g = true;
                cmds.resize(0);
                cmd_sizes.resize(0);
                device_data.reset();
                gen_prefetcher_cmds(device_r, cmds, cmd_sizes, device_data, l1_buf_base_g);
                run_test(1, device, program, device_r, program_r, cmd_sizes, terminate_sizes, cmds, terminate_cmds, host_hugepage_base, dev_hugepage_base_g, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core_g, prefetch_q_writer);
                pass &= device_data.validate(device_r);
                if (!pass) {
                    break;
                }
            }
        } else {
            auto elapsed_seconds = run_test(iterations_g, device, program, device_r, program_r, cmd_sizes, terminate_sizes, cmds, terminate_cmds, host_hugepage_base, dev_hugepage_base_g, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core_g, prefetch_q_writer);

            log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
            log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);
            log_warning(LogTest, "Performance mode, not validating results");
            if (test_type_g == 2 || test_type_g == 3) {
                float bw = (long int)bytes_of_data_g * iterations_g / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << bw;
                log_info(LogTest, "Sent {} bytes", bytes_of_data_g * iterations_g);
                log_info(LogTest, "BW: {} GB/s", ss.str());
            }
        }

        if (packetized_path_en_g) {
            vector<uint32_t> prefetch_relay_mux_results =
                tt::llrt::read_hex_vec_from_core(
                    device_r->id(), phys_prefetch_relay_mux_core, packetized_path_test_results_addr, packetized_path_test_results_size);
            log_info(LogTest, "prefetch relay mux status = {}", packet_queue_test_status_to_string(prefetch_relay_mux_results[PQ_TEST_STATUS_INDEX]));
            pass &= (prefetch_relay_mux_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

            vector<uint32_t> prefetch_relay_demux_results =
                tt::llrt::read_hex_vec_from_core(
                    device_r->id(), phys_prefetch_relay_demux_core, packetized_path_test_results_addr, packetized_path_test_results_size);
            log_info(LogTest, "prefetch relay demux status = {}", packet_queue_test_status_to_string(prefetch_relay_demux_results[PQ_TEST_STATUS_INDEX]));
            pass &= (prefetch_relay_demux_results[0] == PACKET_QUEUE_TEST_PASS);

            vector<uint32_t> dispatch_relay_mux_results =
                tt::llrt::read_hex_vec_from_core(
                    device->id(), phys_dispatch_relay_mux_core, packetized_path_test_results_addr, packetized_path_test_results_size);
            log_info(LogTest, "dispatch relay mux status = {}", packet_queue_test_status_to_string(dispatch_relay_mux_results[PQ_TEST_STATUS_INDEX]));
            pass &= (dispatch_relay_mux_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

            vector<uint32_t> dispatch_relay_demux_results =
                tt::llrt::read_hex_vec_from_core(
                    device->id(), phys_dispatch_relay_demux_core, packetized_path_test_results_addr, packetized_path_test_results_size);
            log_info(LogTest, "dispatch relay demux status = {}", packet_queue_test_status_to_string(dispatch_relay_demux_results[PQ_TEST_STATUS_INDEX]));
            pass &= (dispatch_relay_demux_results[0] == PACKET_QUEUE_TEST_PASS);
        }

        pass &= tt_metal::CloseDevice(device);
        if (test_device_id_g != 0) {
            pass &= tt_metal::CloseDevice(device_r);
        }
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "test_prefetcher.cpp - Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "test_prefetcher.cpp - Test Failed\n");
        return 1;
    }
}
