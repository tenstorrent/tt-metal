// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "common.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_test.hpp"


constexpr uint32_t DEFAULT_TEST_TYPE = 0;
constexpr uint32_t WORKER_DATA_SIZE = 768 * 1024;
constexpr uint32_t MAX_PAGE_SIZE = 64 * 1024;
constexpr uint32_t MAX_DRAM_READ_SIZE = 256 * 1024;
constexpr uint32_t DRAM_PAGE_SIZE_DEFAULT = 1024;
constexpr uint32_t DRAM_PAGES_TO_READ_DEFAULT = 16;

constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
// 764 to make this not divisible by 3 so we can test wrapping of dispatch buffer
constexpr uint32_t DISPATCH_BUFFER_BLOCK_SIZE_PAGES = 764 * 1024 / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;

constexpr uint32_t DEFAULT_HUGEPAGE_BUFFER_SIZE = 256 * 1024 * 1024;
constexpr uint32_t DEFAULT_PREFETCH_Q_ENTRIES = 128;
constexpr uint32_t DEFAULT_MAX_PREFETCH_COMMAND_SIZE = 64 * 1024;
constexpr uint32_t DEFAULT_CMDDAT_Q_SIZE = 128 * 1024;
constexpr uint32_t DEFAULT_SCRATCH_DB_SIZE = 128 * 1024;

constexpr uint32_t DEFAULT_ITERATIONS = 10000;

constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
constexpr uint32_t HUGEPAGE_ALIGNMENT = ((1 << PREFETCH_Q_LOG_MINSIZE) > CQ_PREFETCH_CMD_BARE_MIN_SIZE) ? (1 << PREFETCH_Q_LOG_MINSIZE) : CQ_PREFETCH_CMD_BARE_MIN_SIZE;

constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);
constexpr uint32_t DRAM_HACKED_BASE_ADDR = 1024 * 1024;  // don't interfere w/ fast dispatch storing our kernels...
constexpr uint32_t DRAM_DATA_ALIGNMENT = 32;

constexpr uint32_t PCIE_TRANSFER_SIZE_DEFAULT = 4096;


//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
bool warmup_g = false;
bool debug_g;
uint32_t max_prefetch_command_size_g;

uint32_t dispatch_buffer_page_size_g = 4096;
uint32_t prefetch_q_entries_g;
uint32_t hugepage_buffer_size_g;
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

CoreCoord first_worker_g = { 0, 1 };
CoreRange all_workers_g = {
    first_worker_g,
    {first_worker_g.x + 1, first_worker_g.y + 1},
};

bool send_to_all_g = false;
bool perf_test_g = false;

uint32_t max_xfer_size_bytes_g = 0xFFFFFFFF;
uint32_t min_xfer_size_bytes_g = 0;


constexpr uint32_t tx_gen_x = 3;
constexpr uint32_t tx_gen_y = 0;
constexpr uint32_t rx_gen_x = 3;
constexpr uint32_t rx_gen_y = 3;

constexpr uint32_t relay_mux_x = 3;
constexpr uint32_t relay_mux_y = 1;
constexpr uint32_t relay_demux_x = 3;
constexpr uint32_t relay_demux_y = 2;

constexpr uint32_t prng_seed = 0x100;
constexpr uint32_t data_kb_per_tx = 16*1024;
constexpr uint32_t max_packet_size_words = 0x100;

constexpr uint32_t tx_queue_start_addr = 0x80000;
constexpr uint32_t tx_queue_size_bytes = 0x10000;
constexpr uint32_t rx_queue_start_addr = 0xa0000;
constexpr uint32_t rx_queue_size_bytes = 0x20000;
constexpr uint32_t relay_mux_queue_start_addr = 0x80000;
constexpr uint32_t relay_mux_queue_size_bytes = 0x10000;
constexpr uint32_t relay_demux_queue_start_addr = 0x90000;
constexpr uint32_t relay_demux_queue_size_bytes = 0x20000;

constexpr uint32_t test_results_addr = 0x100000;
constexpr uint32_t test_results_size = 1024;

constexpr uint32_t timeout_mcycles = 1000;
constexpr uint32_t rx_disable_data_check = 0;

constexpr uint32_t src_endpoint_start_id = 0xaa;
constexpr uint32_t dest_endpoint_start_id = 0xbb;

constexpr uint32_t num_src_endpoints = 1;
constexpr uint32_t num_dest_endpoints = 1;


void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up before starting timer (default disabled)");
        log_info(LogTest, "  -i: host iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -t: test type: 0:Smoke 1:Random 2:PCIe, 3:DRAM (default {})", DEFAULT_TEST_TYPE);
        log_info(LogTest, " -wx: right-most worker in grid (default {})", all_workers_g.end.x);
        log_info(LogTest, " -wy: bottom-most worker in grid (default {})", all_workers_g.end.y);
        log_info(LogTest, "  -b: run a \"big\" test (fills memory w/ fewer transactions) (default false)", DEFAULT_TEST_TYPE);
        log_info(LogTest, " -rb: gen data, readback and test every iteration - disable for perf measurements (default true)");
        log_info(LogTest, "  -d: wrap all commands in debug commands (default disabled)");
        log_info(LogTest, "  -hp: host huge page buffer size (default {})", DEFAULT_HUGEPAGE_BUFFER_SIZE);
        log_info(LogTest, "  -pq: prefetch queue entries (default {})", DEFAULT_PREFETCH_Q_ENTRIES);
        log_info(LogTest, "  -cs: max cmddat q size (default {})", DEFAULT_CMDDAT_Q_SIZE);
        log_info(LogTest, "  -ss: max scratch cb size (default {})", DEFAULT_SCRATCH_DB_SIZE);
        log_info(LogTest, "  -mc: max command size (default {})", DEFAULT_MAX_PREFETCH_COMMAND_SIZE);
        log_info(LogTest, "  -pcies: size of data to transfer in pcie bw test type (default )", PCIE_TRANSFER_SIZE_DEFAULT);
        log_info(LogTest, "  -dpgs: dram page size in dram bw test type (default )", DRAM_PAGE_SIZE_DEFAULT);
        log_info(LogTest, "  -dpgr: dram pages to read in dram bw test type (default )", DRAM_PAGES_TO_READ_DEFAULT);
        log_info(LogTest, "  -c: use coherent data as payload (default false)");
        log_info(LogTest, "  -s: seed for randomized tests (default 1)");
        exit(0);
    }

    warmup_g = test_args::has_command_option(input_args, "-w");
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    hugepage_buffer_size_g = test_args::get_command_option_uint32(input_args, "-hp", DEFAULT_HUGEPAGE_BUFFER_SIZE);
    prefetch_q_entries_g = test_args::get_command_option_uint32(input_args, "-hq", DEFAULT_PREFETCH_Q_ENTRIES);
    cmddat_q_size_g = test_args::get_command_option_uint32(input_args, "-cs", DEFAULT_CMDDAT_Q_SIZE);
    scratch_db_size_g = test_args::get_command_option_uint32(input_args, "-ss", DEFAULT_SCRATCH_DB_SIZE);
    max_prefetch_command_size_g = test_args::get_command_option_uint32(input_args, "-mc", DEFAULT_MAX_PREFETCH_COMMAND_SIZE);
    use_coherent_data_g = test_args::has_command_option(input_args, "-c");
    readback_every_iteration_g = !test_args::has_command_option(input_args, "-rb");
    pcie_transfer_size_g = test_args::get_command_option_uint32(input_args, "-pcies", PCIE_TRANSFER_SIZE_DEFAULT);
    pcie_transfer_size_g = test_args::get_command_option_uint32(input_args, "-pcies", PCIE_TRANSFER_SIZE_DEFAULT);
    dram_page_size_g = test_args::get_command_option_uint32(input_args, "-dpgs", DRAM_PAGE_SIZE_DEFAULT);
    dram_pages_to_read_g = test_args::get_command_option_uint32(input_args, "-dpgr", DRAM_PAGES_TO_READ_DEFAULT);

    test_type_g = test_args::get_command_option_uint32(input_args, "-t", DEFAULT_TEST_TYPE);
    all_workers_g.end.x = test_args::get_command_option_uint32(input_args, "-wx", all_workers_g.end.x);
    all_workers_g.end.y = test_args::get_command_option_uint32(input_args, "-wy", all_workers_g.end.y);

    uint32_t seed = test_args::get_command_option_uint32(input_args, "-s", 1);
    std::srand(seed);
    big_g = test_args::has_command_option(input_args, "-b");
    debug_g = test_args::has_command_option(input_args, "-d");
}

uint32_t round_cmd_size_up(uint32_t size) {
    constexpr uint32_t align_mask = HUGEPAGE_ALIGNMENT - 1;

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

void add_prefetcher_dram_cmd(vector<uint32_t>& cmds,
                             vector<uint16_t>& sizes,
                             uint32_t start_page,
                             uint32_t base_addr,
                             uint32_t page_size,
                             uint32_t pages) {

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;

    cmd.relay_paged.is_dram = true;
    cmd.relay_paged.start_page = start_page;
    cmd.relay_paged.base_addr = base_addr;
    cmd.relay_paged.page_size = page_size;
    cmd.relay_paged.pages = pages;

    add_bare_prefetcher_cmd(cmds, cmd, true);
}

void add_prefetcher_linear_read_cmd(Device *device,
                                    vector<uint32_t>& cmds,
                                    vector<uint16_t>& sizes,
                                    CoreCoord worker_core,
                                    uint32_t addr,
                                    uint32_t length) {

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;

    cmd.relay_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.relay_linear.addr = addr;
    cmd.relay_linear.length = length;

    add_bare_prefetcher_cmd(cmds, cmd, true);
}

void add_prefetcher_cmd(vector<uint32_t>& cmds,
                        vector<uint16_t>& sizes,
                        CQPrefetchCmdId id,
                        vector<uint32_t>& payload) {

    vector<uint32_t> data;

    auto prior_end = cmds.size();

    if (debug_g) {
        CQPrefetchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_PREFETCH_CMD_DEBUG;
        add_bare_prefetcher_cmd(cmds, debug_cmd, true);
    }

    CQPrefetchCmd cmd;
    cmd.base.cmd_id = id;

    uint32_t payload_length_bytes = payload.size() * sizeof(uint32_t);
    switch (id) {
    case CQ_PREFETCH_CMD_RELAY_PAGED:
        TT_ASSERT(false);
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE:
    case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
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
    uint32_t cmd_size_bytes = (cmds.size() - prior_end) * sizeof(uint32_t);
    for (int i = 0; i < payload.size(); i++) {
        cmds.push_back(payload[i]);
    }
    uint32_t pad_size_bytes = round_cmd_size_up(cmd_size_bytes + payload_length_bytes) - payload_length_bytes - cmd_size_bytes;
    for (int i = 0; i < pad_size_bytes / sizeof(uint32_t); i++) {
        cmds.push_back(0);
    }
    uint32_t new_size = (cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    sizes.push_back(new_size >> PREFETCH_Q_LOG_MINSIZE);

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

// Model a paged read by updating worker data with interleaved/paged DRAM data, for validation later.
void add_paged_dram_data_to_worker_data(const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                                  const CoreRange& workers,
                                  worker_data_t& worker_data,
                                  uint32_t start_page,
                                  uint32_t base_addr,
                                  uint32_t page_size,
                                  uint32_t pages) {

    uint32_t base_addr_words = base_addr / sizeof(uint32_t);
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    // Get data from DRAM map, add to all workers, but only set valid for cores included in workers range.
    TT_ASSERT(start_page < num_dram_banks_g);
    for (uint32_t page_idx = start_page; page_idx < start_page + pages; page_idx++) {

        uint32_t dram_bank_id = page_idx % num_dram_banks_g;
        uint32_t bank_offset = base_addr_words + page_size_words * (page_idx / num_dram_banks_g);

        for (uint32_t j = 0; j  < page_size_words; j++) {
            for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
                for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                    CoreCoord core(x, y);
                    worker_data[core].data.push_back(dram_data_map.at(dram_bank_id)[bank_offset + j]);
                    worker_data[core].valid.push_back(workers.contains(core));
                }
            }
        }
    }
}

void gen_dram_read_cmd(Device *device,
                       vector<uint32_t>& prefetch_cmds,
                       vector<uint16_t>& cmd_sizes,
                       const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                       worker_data_t& worker_data,
                       CoreCoord worker_core,
                       uint32_t dst_addr,
                       uint32_t start_page,
                       uint32_t base_addr,
                       uint32_t page_size,
                       uint32_t pages) {

    vector<uint32_t> dispatch_cmds;

    uint32_t worker_data_size = page_size * pages;
    gen_bare_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, worker_data_size);

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    auto prior_end = prefetch_cmds.size();
    add_prefetcher_dram_cmd(prefetch_cmds, cmd_sizes, start_page, base_addr + DRAM_HACKED_BASE_ADDR, page_size, pages);

    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> PREFETCH_Q_LOG_MINSIZE);

    // Model the paged read in this function by updating worker data with interleaved/paged DRAM data, for validation later.
    add_paged_dram_data_to_worker_data(dram_data_map, worker_core, worker_data, start_page, base_addr, page_size, pages);
}

// This is pretty much a blit: copies from worker core's start of data back to the end of data
void gen_linear_read_cmd(Device *device,
                         vector<uint32_t>& prefetch_cmds,
                         vector<uint16_t>& cmd_sizes,
                         const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                         worker_data_t& worker_data,
                         CoreCoord worker_core,
                         uint32_t addr,
                         uint32_t length,
                         uint32_t offset = 0) {

    vector<uint32_t> dispatch_cmds;

    gen_bare_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, addr, length);

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    auto prior_end = prefetch_cmds.size();
    add_prefetcher_linear_read_cmd(device, prefetch_cmds, cmd_sizes, worker_core, addr + offset * sizeof(uint32_t), length);

    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> PREFETCH_Q_LOG_MINSIZE);

    // Add linear data to worker data:
    uint32_t length_words = length / sizeof(uint32_t);
    for (uint32_t i = 0; i < length_words; i++) {
        for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                CoreCoord core(x, y);
                if (core == worker_core) {
                    uint32_t datum = worker_data[core].data[offset + i];
                    worker_data[core].data.push_back(datum);
                    worker_data[core].valid.push_back(true);
                } else {
                    worker_data[core].data.push_back(0);
                    worker_data[core].valid.push_back(false);
                }
            }
        }
    }
}

void gen_wait_and_stall_cmd(Device *device,
                            vector<uint32_t>& prefetch_cmds,
                            vector<uint16_t>& cmd_sizes) {

    vector<uint32_t> dispatch_cmds;

    CQDispatchCmd wait;
    wait.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
    wait.wait.barrier = true;
    wait.wait.notify_prefetch = true;
    wait.wait.addr = dispatch_wait_addr_g;
    wait.wait.count = 0;
    add_bare_dispatcher_cmd(dispatch_cmds, wait);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_STALL, empty_payload);
}

void gen_dispatcher_delay_cmd(Device *device,
                              vector<uint32_t>& prefetch_cmds,
                              vector<uint16_t>& cmd_sizes,
                              uint32_t count) {

    vector<uint32_t> dispatch_cmds;

    CQDispatchCmd delay;
    delay.base.cmd_id = CQ_DISPATCH_CMD_DELAY;
    delay.delay.delay = count;
    add_bare_dispatcher_cmd(dispatch_cmds, delay);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
}

void gen_dram_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint16_t>& cmd_sizes,
                   const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                   worker_data_t& worker_data,
                   CoreCoord worker_core,
                   uint32_t dst_addr) {

    vector<uint32_t> dispatch_cmds;

    while (worker_data_size(worker_data) * sizeof(uint32_t) < WORKER_DATA_SIZE) {
        dispatch_cmds.resize(0);
        uint32_t start_page = 0;
        uint32_t base_addr = 0;
        gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                          start_page, base_addr, dram_page_size_g, dram_pages_to_read_g);

        bytes_of_data_g += dram_page_size_g * dram_pages_to_read_g;
    }
}

void gen_pcie_test(Device *device,
                   vector<uint32_t>& prefetch_cmds,
                   vector<uint16_t>& cmd_sizes,
                   const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                   worker_data_t& worker_data,
                   CoreCoord worker_core,
                   uint32_t dst_addr) {

    vector<uint32_t> dispatch_cmds;

    while (worker_data_size(worker_data) * sizeof(uint32_t) < WORKER_DATA_SIZE) {
        dispatch_cmds.resize(0);
        gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, pcie_transfer_size_g);
        add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
        bytes_of_data_g += pcie_transfer_size_g;
    }
}

void gen_rnd_dram_paged_cmd(Device *device,
                            vector<uint32_t>& prefetch_cmds,
                            vector<uint16_t>& cmd_sizes,
                            const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                            worker_data_t& worker_data,
                            CoreCoord worker_core,
                            uint32_t dst_addr) {

    vector<uint32_t> dispatch_cmds;

    uint32_t start_page = std::rand() % num_dram_banks_g;
    uint32_t max_page_size = big_g ? MAX_PAGE_SIZE : 4096;
    uint32_t page_size = (std::rand() % (max_page_size + 1)) & ~(DRAM_DATA_ALIGNMENT - 1);
    if (page_size == 0) page_size = DRAM_DATA_ALIGNMENT;
    uint32_t max_data = big_g ? WORKER_DATA_SIZE : WORKER_DATA_SIZE / 8;
    uint32_t max = WORKER_DATA_SIZE - worker_data_size(worker_data) * sizeof(uint32_t);
    max = (max > max_data) ? max_data : max;
    // start in bottom half of valid dram data to not worry about overflowing valid data
    uint32_t base_addr = (std::rand() % (DRAM_DATA_SIZE_BYTES / 2)) & ~(DRAM_DATA_ALIGNMENT - 1);
    uint32_t size = std::rand() % max;
    if (size < page_size) size = page_size;
    uint32_t pages = size / page_size;
    TT_ASSERT(base_addr + (start_page * page_size + pages * page_size / num_dram_banks_g) < DRAM_DATA_SIZE_BYTES);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      start_page, base_addr, page_size, pages);
}

void gen_rnd_inline_cmd(Device *device,
                        vector<uint32_t>& prefetch_cmds,
                        vector<uint16_t>& cmd_sizes,
                        worker_data_t& worker_data,
                        CoreCoord worker_core,
                        uint32_t dst_addr) {

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
            uint32_t max_size = big_g ? DEFAULT_MAX_PREFETCH_COMMAND_SIZE : DEFAULT_MAX_PREFETCH_COMMAND_SIZE / 16;
            uint32_t max_xfer_size_16b = (max_size - cmd_size_bytes) >> 4;
            uint32_t xfer_size_16B = (std::rand() & (max_xfer_size_16b - 1));
            // Note: this may overflow the WORKER_DATA_SIZE, but by little enough that it won't overflow L1
            uint32_t xfer_size_bytes = xfer_size_16B << 4;

            gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data,
                                             dst_addr, xfer_size_bytes);
        }
        break;
    case 1:
        // packed unicast write
        gen_rnd_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_data, dst_addr);
        break;
    }

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
}

void gen_rnd_debug_cmd(vector<uint32_t>& prefetch_cmds,
                       vector<uint16_t>& cmd_sizes,
                       worker_data_t& worker_data,
                       uint32_t dst_addr) {

    vector<uint32_t> rnd_payload;
    generate_random_payload(rnd_payload, std::rand() % (dispatch_buffer_page_size_g - sizeof(CQDispatchCmd)) + 1);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, rnd_payload);
}

void gen_rnd_test(Device *device,
                  vector<uint32_t>& prefetch_cmds,
                  vector<uint16_t>& cmd_sizes,
                  const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                  worker_data_t& worker_data,
                  uint32_t dst_addr) {

    while (worker_data_size(worker_data) * sizeof(uint32_t) < WORKER_DATA_SIZE) {
        // Assumes terminate is the last command...
        uint32_t cmd = std::rand() % CQ_PREFETCH_CMD_TERMINATE;
        uint32_t x = rand() % (all_workers_g.end.x - first_worker_g.x);
        uint32_t y = rand() % (all_workers_g.end.y - first_worker_g.y);

        CoreCoord worker_core(first_worker_g.x + x, first_worker_g.y + y);

        switch (cmd) {
        case CQ_PREFETCH_CMD_RELAY_PAGED:
            gen_rnd_dram_paged_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr);
            break;
        case CQ_PREFETCH_CMD_RELAY_INLINE:
            gen_rnd_inline_cmd(device, prefetch_cmds, cmd_sizes, worker_data, worker_core, dst_addr);
            break;
        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            break;
        case CQ_PREFETCH_CMD_STALL:
            break;
        case CQ_PREFETCH_CMD_DEBUG:
            gen_rnd_debug_cmd(prefetch_cmds, cmd_sizes, worker_data, dst_addr);
            break;
        }
    }
}

void gen_smoke_test(Device *device,
                    vector<uint32_t>& prefetch_cmds,
                    vector<uint16_t>& cmd_sizes,
                    const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                    worker_data_t& worker_data,
                    CoreCoord worker_core,
                    uint32_t dst_addr) {

    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    vector<uint32_t> dispatch_cmds;

    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, empty_payload);

    vector<uint32_t> rnd_payload;
    generate_random_payload(rnd_payload, 17);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_DEBUG, rnd_payload);

    // Write to worker
    dispatch_cmds.resize(0);
    reset_worker_data(worker_data);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 2048);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 1026);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 8448);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Merge 4 commands in the FetchQ
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 112);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 608);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 64);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 96);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Tests sending multiple commands w/ 1 FetchQ entry
    uint32_t combined_size =
        cmd_sizes[cmd_sizes.size() - 1] + cmd_sizes[cmd_sizes.size() - 2] +
        cmd_sizes[cmd_sizes.size() - 3] + cmd_sizes[cmd_sizes.size() - 4];

    cmd_sizes[cmd_sizes.size() - 4] = combined_size;
    cmd_sizes.pop_back();
    cmd_sizes.pop_back();
    cmd_sizes.pop_back();

    // Read from dram, write to worker
    // start_page, base addr, page_size, pages
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      0, 0, 32, num_dram_banks_g);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      0, 0, 32, num_dram_banks_g);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      4, 32, 64, num_dram_banks_g);

    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      0, 0, 128, 128);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      4, 32, 2048, num_dram_banks_g * 8);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      5, 32, 2048, num_dram_banks_g * 8 + 1);
    gen_dram_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr,
                      3, 128, 6144, num_dram_banks_g * 8 + 7);

    // Send inline data to (maybe) multiple cores
    dispatch_cmds.resize(0);
    vector<CoreCoord> worker_cores;
    worker_cores.push_back(first_worker_g);
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, worker_data, dst_addr, 4);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    worker_cores.resize(0);
    for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
        for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
            CoreCoord worker_core(x, y);
            worker_cores.push_back(worker_core);
        }
    }
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, worker_data, dst_addr, 12);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    dispatch_cmds.resize(0);
    worker_cores.resize(0);
    worker_cores.push_back(first_worker_g);
    worker_cores.push_back({first_worker_g.x + 3, first_worker_g.y});
    gen_dispatcher_packed_write_cmd(device, dispatch_cmds, worker_cores, worker_data, dst_addr, 156);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // These tests copy data from earlier tests so can't run first
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr, 32);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr, 65 * 1024);

    // Test wait/stall
    gen_dispatcher_delay_cmd(device, prefetch_cmds, cmd_sizes, 1024 * 1024);
    dispatch_cmds.resize(0);
    gen_dispatcher_unicast_write_cmd(device, dispatch_cmds, worker_core, worker_data, dst_addr, 2048);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);
    gen_wait_and_stall_cmd(device, prefetch_cmds, cmd_sizes);
    gen_linear_read_cmd(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, worker_core, dst_addr, 32, worker_data[worker_core].data.size() - 32 / sizeof(uint32_t));
}

void gen_prefetcher_cmds(Device *device,
                         vector<uint32_t>& prefetch_cmds,
                         vector<uint16_t>& cmd_sizes,
                         const unordered_map<uint32_t, vector<uint32_t>>& dram_data_map,
                         worker_data_t& worker_data,
                         uint32_t dst_addr) {

    switch (test_type_g) {
    case 0:
        gen_smoke_test(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, first_worker_g, dst_addr);
        break;
    case 1:
        gen_rnd_test(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, dst_addr);
        break;
    case 2:
        gen_pcie_test(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, first_worker_g, dst_addr);
        break;
    case 3:
        gen_dram_test(device, prefetch_cmds, cmd_sizes, dram_data_map, worker_data, first_worker_g, dst_addr);
        break;
    }
}

void gen_terminate_cmds(vector<uint32_t>& prefetch_cmds,
                        vector<uint16_t>& cmd_sizes) {
    vector<uint32_t> empty_payload; // don't give me grief, it is just a test
    vector<uint32_t> dispatch_cmds;

    // Terminate dispatcher
    dispatch_cmds.resize(0);
    gen_dispatcher_terminate_cmd(dispatch_cmds);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Terminate prefetcher
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_TERMINATE, empty_payload);
}

void write_prefetcher_cmd(Device *device,
                          vector<uint32_t>& cmds,
                          uint32_t& cmd_offset,
                          uint16_t cmd_size16b,
                          uint32_t*& host_mem_ptr,
                          uint32_t& prefetch_q_dev_ptr,
                          uint32_t& prefetch_q_dev_fence,
                          uint32_t prefetch_q_base,
                          uint32_t prefetch_q_rd_ptr_addr,
                          CoreCoord phys_prefetch_core) {

    static vector<uint32_t> read_vec;  // static to avoid realloc

    uint32_t cmd_size_bytes = (uint32_t)cmd_size16b << PREFETCH_Q_LOG_MINSIZE;
    uint32_t cmd_size_words = cmd_size_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < cmd_size_words; i++) {
        *host_mem_ptr = cmds[cmd_offset];
        host_mem_ptr++;
        cmd_offset++;
    }
    // Not clear to me if this is needed, writing to cached PCIe memory
    tt_driver_atomics::sfence();

    // wait for space
    while (prefetch_q_dev_ptr == prefetch_q_dev_fence) {
        tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), prefetch_q_rd_ptr_addr);
        prefetch_q_dev_fence = read_vec[0];
    }

    // wrap
    if (prefetch_q_dev_ptr == prefetch_q_base + prefetch_q_entries_g * sizeof(uint16_t)) {
        prefetch_q_dev_ptr = prefetch_q_base;

        while (prefetch_q_dev_ptr == prefetch_q_dev_fence) {
            tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), prefetch_q_rd_ptr_addr);
            prefetch_q_dev_fence = read_vec[0];
        }
    }

    tt::Cluster::instance().write_core((void *)&cmd_size16b, sizeof(uint16_t), tt_cxy_pair(device->id(), phys_prefetch_core), prefetch_q_dev_ptr, true);

    prefetch_q_dev_ptr += sizeof(uint16_t);
}

void write_prefetcher_cmds(uint32_t iterations,
                           Device *device,
                           vector<uint32_t>& prefetch_cmds,
                           vector<uint16_t>& cmd_sizes,
                           void * host_hugepage_base,
                           uint32_t dev_hugepage_base,
                           uint32_t prefetch_q_base,
                           uint32_t prefetch_q_rd_ptr_addr,
                           CoreCoord phys_prefetch_core) {

    static uint32_t *host_mem_ptr;
    static uint32_t prefetch_q_dev_ptr;
    static uint32_t prefetch_q_dev_fence;

    if (initialize_device_g) {
        vector<uint32_t> prefetch_q(DEFAULT_PREFETCH_Q_ENTRIES, 0);
        vector<uint32_t> prefetch_q_rd_ptr_addr_data;

        prefetch_q_rd_ptr_addr_data.push_back(prefetch_q_base + prefetch_q_entries_g * sizeof(uint16_t));
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, prefetch_q_rd_ptr_addr_data, prefetch_q_rd_ptr_addr);
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, prefetch_q, prefetch_q_base);

        host_mem_ptr = (uint32_t *)host_hugepage_base;
        prefetch_q_dev_ptr = prefetch_q_base;
        prefetch_q_dev_fence = prefetch_q_base + prefetch_q_entries_g * sizeof(uint16_t);
        initialize_device_g = false;
    }

    for (uint32_t i = 0; i < iterations; i++) {
        uint32_t cmd_ptr = 0;
        for (uint32_t j = 0; j < cmd_sizes.size(); j++) {
            uint32_t cmd_size_words = ((uint32_t)cmd_sizes[j] << PREFETCH_Q_LOG_MINSIZE) / sizeof(uint32_t);
            uint32_t space_at_end_for_wrap_words = CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t);
            if ((void *)(host_mem_ptr + cmd_size_words) > (void *)((uint8_t *)host_hugepage_base + hugepage_buffer_size_g)) {
                // Wrap huge page
                uint32_t offset = 0;
                host_mem_ptr = (uint32_t *)host_hugepage_base;
            }
            write_prefetcher_cmd(device, prefetch_cmds, cmd_ptr, cmd_sizes[j],
                                 host_mem_ptr, prefetch_q_dev_ptr, prefetch_q_dev_fence, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
        }
    }
}

// Populate interleaved DRAM with data for later readback.  Can we extended to L1 if needed.
void populate_interleaved_dram(Device *device, unordered_map<uint32_t, vector<uint32_t>>& dram_data_map)
{

    num_dram_banks_g = device->num_banks(BufferType::DRAM);;

    for (int bank_id = 0; bank_id < num_dram_banks_g; bank_id++) {
        auto offset = device->dram_bank_offset_from_bank_id(bank_id);
        auto dram_channel = device->dram_channel_from_bank_id(bank_id);
        auto bank_core = device->core_from_dram_channel(dram_channel);

        // Generate random or coherent data per bank of specific size.
        for (uint32_t i = 0; i < DRAM_DATA_SIZE_WORDS; i++) {
            uint32_t datum = (use_coherent_data_g) ? (((bank_id & 0xFF) << 24) | i) : std::rand();
            dram_data_map[bank_id].push_back(datum);
            if (i < 10) {
                log_debug(tt::LogTest, "{} - bank_id: {:2d} core: {} offset: 0x{:08x} using i: {:2d} datum: 0x{:08x}",
                    __FUNCTION__, bank_id, bank_core.str(), offset, i, datum);
            }
        }

        // Write to device once per bank (appropriate core and offset)
        tt::Cluster::instance().write_core(static_cast<const void*>(dram_data_map[bank_id].data()),
            dram_data_map[bank_id].size() * sizeof(uint32_t), tt_cxy_pair(device->id(), bank_core), DRAM_HACKED_BASE_ADDR + offset);
    }
}

std::chrono::duration<double> run_test(uint32_t iterations,
                                       Device *device,
                                       Program& program,
                                       vector<uint16_t>& cmd_sizes,
                                       vector<uint16_t>& terminate_sizes,
                                       vector<uint32_t>& cmds,
                                       vector<uint32_t>& terminate_cmds,
                                       void * host_hugepage_base,
                                       uint32_t dev_hugepage_base,
                                       uint32_t prefetch_q_base,
                                       uint32_t prefetch_q_rd_ptr_addr,
                                       CoreCoord phys_prefetch_core) {

    auto start = std::chrono::system_clock::now();

    std::thread t1 ([&]() {
        write_prefetcher_cmds(iterations, device, cmds, cmd_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
        write_prefetcher_cmds(1, device, terminate_cmds, terminate_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
    });
    tt_metal::detail::LaunchProgram(device, program);
    t1.join();

    auto end = std::chrono::system_clock::now();

    return end-start;
}

int main(int argc, char **argv) {
    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    init(argc, argv);

    uint32_t dispatch_buffer_pages = DISPATCH_BUFFER_BLOCK_SIZE_PAGES * DISPATCH_BUFFER_SIZE_BLOCKS;

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord prefetch_core = {0, 0};
        CoreCoord dispatch_core = {4, 0};

        CoreCoord phys_prefetch_core = device->worker_core_from_logical_core(prefetch_core);
        CoreCoord phys_dispatch_core = device->worker_core_from_logical_core(dispatch_core);

        ////

        CoreCoord tx_gen_core = {tx_gen_x, tx_gen_y};
        CoreCoord rx_gen_core = {rx_gen_x, rx_gen_y};
        CoreCoord relay_mux_core = {relay_mux_x, relay_mux_y};
        CoreCoord relay_demux_core = {relay_demux_x, relay_demux_y};

        CoreCoord phys_tx_gen_core = device->worker_core_from_logical_core(tx_gen_core);
        CoreCoord phys_rx_gen_core = device->worker_core_from_logical_core(rx_gen_core);
        CoreCoord phys_relay_mux_core = device->worker_core_from_logical_core(relay_mux_core);
        CoreCoord phys_relay_demux_core = device->worker_core_from_logical_core(relay_demux_core);

        std::vector<uint32_t> tx_gen_compile_args =
        {
            src_endpoint_start_id, // 0: src_endpoint_id
            num_dest_endpoints, // 1: num_dest_endpoints
            (tx_queue_start_addr >> 4), // 2: queue_start_addr_words
            (tx_queue_size_bytes >> 4), // 3: queue_size_words
            (relay_mux_queue_start_addr >> 4), // 4: remote_rx_queue_start_addr_words
            (relay_mux_queue_size_bytes >> 4), // 5: remote_rx_queue_size_words
            (uint32_t)phys_relay_mux_core.x, // 6: remote_rx_x
            (uint32_t)phys_relay_mux_core.y, // 7: remote_rx_y
            0, // 8: remote_rx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
            test_results_addr, // 10: test_results_addr
            test_results_size, // 11: test_results_size
            prng_seed, // 12: prng_seed
            data_kb_per_tx, // 13: total_data_kb
            max_packet_size_words, // 14: max_packet_size_words
            src_endpoint_start_id, // 15: src_endpoint_start_id
            dest_endpoint_start_id, // 16: dest_endpoint_start_id
            timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
        };

        log_info(LogTest, "run traffic_gen_tx at x={},y={}", tx_gen_core.x, tx_gen_core.y);
        auto tx_gen_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_tx.cpp",
            {tx_gen_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = tx_gen_compile_args,
                .defines = {}
            }
        );

        std::vector<uint32_t> rx_gen_compile_args =
        {
            dest_endpoint_start_id, // 0: dest_endpoint_id
            num_src_endpoints, // 1: num_src_endpoints
            num_dest_endpoints, // 2: num_dest_endpoints
            (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
            (rx_queue_size_bytes >> 4), // 4: queue_size_words
            (uint32_t)phys_relay_demux_core.x, // 5: remote_tx_x
            (uint32_t)phys_relay_demux_core.y, // 6: remote_tx_y
            0, // 7: remote_tx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
            test_results_addr, // 9: test_results_addr
            test_results_size, // 10: test_results_size
            prng_seed, // 11: prng_seed
            0, // 12: reserved
            max_packet_size_words, // 13: max_packet_size_words
            rx_disable_data_check, // 14: disable data check
            src_endpoint_start_id, // 15: src_endpoint_start_id
            dest_endpoint_start_id, // 16: dest_endpoint_start_id
            timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
        };

        log_info(LogTest, "run traffic_gen_rx at x={},y={}", rx_gen_core.x, rx_gen_core.y);
        auto rx_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_rx.cpp",
            {rx_gen_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = rx_gen_compile_args,
                .defines = {}
            }
        );

        std::vector<uint32_t> relay_mux_compile_args =
        {
            0, // 0: reserved
            (relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
            (relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
            num_src_endpoints, // 3: mux_fan_in
            packet_switch_4B_pack((uint32_t)phys_tx_gen_core.x,
                                    (uint32_t)phys_tx_gen_core.y,
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
            (relay_demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
            (relay_demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
            (uint32_t)phys_relay_demux_core.x, // 10: remote_tx_x
            (uint32_t)phys_relay_demux_core.y, // 11: remote_tx_y
            num_dest_endpoints, // 12: remote_tx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
            test_results_addr, // 14: test_results_addr
            test_results_size, // 15: test_results_size
            timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
        };

        log_info(LogTest, "run relay mux at x={},y={}", relay_mux_core.x, relay_mux_core.y);
        auto mux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            {relay_mux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = relay_mux_compile_args,
                .defines = {}
            }
        );

        uint32_t dest_map_array[4] = {0, 1, 2, 3};
        uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
        std::vector<uint32_t> demux_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                (relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_dest_endpoints, // 3: demux_fan_out
                packet_switch_4B_pack(phys_rx_gen_core.x,
                                      phys_rx_gen_core.y,
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
                (rx_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words 0
                (rx_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words 0
                (rx_queue_start_addr >> 4), // 10: remote_tx_queue_start_addr_words 1
                (rx_queue_size_bytes >> 4), // 11: remote_tx_queue_size_words 1
                (rx_queue_start_addr >> 4), // 12: remote_tx_queue_start_addr_words 2
                (rx_queue_size_bytes >> 4), // 13: remote_tx_queue_size_words 2
                (rx_queue_start_addr >> 4), // 14: remote_tx_queue_start_addr_words 3
                (rx_queue_size_bytes >> 4), // 15: remote_tx_queue_size_words 3
                (uint32_t)phys_relay_mux_core.x, // 16: remote_rx_x
                (uint32_t)phys_relay_mux_core.y, // 17: remote_rx_y
                num_dest_endpoints, // 18: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                test_results_addr, // 22: test_results_addr
                test_results_size, // 23: test_results_size
                timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
            };

        log_info(LogTest, "run demux at x={},y={}", relay_demux_core.x, relay_demux_core.y);
        auto demux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {relay_demux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_compile_args,
                .defines = {}
            }
        );

        ////

        // Want different buffers on each core, instead use big buffer and self-manage it
        uint32_t l1_unreserved_base_aligned = align(L1_UNRESERVED_BASE, (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE)); // Was not aligned, lately.
        uint32_t l1_buf_base = l1_unreserved_base_aligned + (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE); // Reserve a page.
        TT_ASSERT((l1_buf_base & ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) == 0);

        uint32_t dispatch_buffer_base = l1_buf_base;
        uint32_t dev_hugepage_base = 0;
        uint32_t prefetch_q_base = l1_buf_base;
        uint32_t prefetch_q_rd_ptr_addr = l1_unreserved_base_aligned;
        dispatch_wait_addr_g = l1_unreserved_base_aligned + 16;
        vector<uint32_t>zero_data(0);
        llrt::write_hex_vec_to_core(device->id(), phys_dispatch_core, zero_data, dispatch_wait_addr_g);

        uint32_t prefetch_q_size = prefetch_q_entries_g * sizeof(uint16_t);
        uint32_t noc_read_alignment = 32;
        uint32_t cmddat_q_base = prefetch_q_base + ((prefetch_q_size + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        uint32_t scratch_db_base = cmddat_q_base + ((cmddat_q_size_g + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_db_base < 1024 * 1024); // L1 size

        // Implementation syncs w/ device on prefetch_q but not on hugepage, ie, assumes we can't run
        // so far ahead in the hugepage that we overright un-read commands since we'll first
        // stall on the prefetch_q
        TT_ASSERT(hugepage_buffer_size_g > prefetch_q_entries_g * max_prefetch_command_size_g, "Shrink the max command size or grow the hugepage buffer size or shrink the prefetch_q size");
        TT_ASSERT(cmddat_q_size_g >= 2 * max_prefetch_command_size_g);
        TT_ASSERT(scratch_db_size_g >= MAX_PAGE_SIZE);

        // NOTE: this test hijacks hugepage
        void *host_hugepage_base;
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        host_hugepage_base = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        host_hugepage_base = (void *)((uint8_t *)host_hugepage_base + dev_hugepage_base);

        vector<uint32_t> cmds, terminate_cmds;
        vector<uint16_t> cmd_sizes, terminate_sizes;
        worker_data_t worker_data;
        for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                one_worker_data_t one;
                worker_data.insert({CoreCoord(x, y), one});
            }
        }

        // Model of interleaved DRAM memory by bank id
        unordered_map<uint32_t, vector<uint32_t>> dram_data_map;
        populate_interleaved_dram(device, dram_data_map);

        tt::Cluster::instance().l1_barrier(device->id());
        tt::Cluster::instance().dram_barrier(device->id());
        gen_prefetcher_cmds(device, cmds, cmd_sizes, dram_data_map, worker_data, l1_buf_base);
        gen_terminate_cmds(terminate_cmds, terminate_sizes);

        std::map<string, string> defines = {
            {"PREFETCH_NOC_X", std::to_string(phys_prefetch_core.x)},
            {"PREFETCH_NOC_Y", std::to_string(phys_prefetch_core.y)},
            {"DISPATCH_NOC_X", std::to_string(phys_dispatch_core.x)},
            {"DISPATCH_NOC_Y", std::to_string(phys_dispatch_core.y)},
        };

        constexpr uint32_t dispatch_cb_sem = 0;
        tt_metal::CreateSemaphore(program, {prefetch_core}, dispatch_buffer_pages);
        tt_metal::CreateSemaphore(program, {dispatch_core}, 0);

        constexpr uint32_t prefetch_sync_sem = 1;
        tt_metal::CreateSemaphore(program, {prefetch_core}, 0);
        tt_metal::CreateSemaphore(program, {dispatch_core}, 0);

        std::vector<uint32_t> dispatch_compile_args = {
             dispatch_buffer_base,
             DISPATCH_BUFFER_LOG_PAGE_SIZE,
             DISPATCH_BUFFER_SIZE_BLOCKS * DISPATCH_BUFFER_BLOCK_SIZE_PAGES,
             dispatch_cb_sem,
             DISPATCH_BUFFER_SIZE_BLOCKS,
             prefetch_sync_sem,
             // Hugepage compile args aren't used in this test since WriteHost is not tested here
             0,
             0,
        };

        std::vector<uint32_t> prefetch_compile_args = {
             dispatch_buffer_base,
             DISPATCH_BUFFER_LOG_PAGE_SIZE,
             dispatch_buffer_pages,
             dispatch_cb_sem,
             dev_hugepage_base,
             hugepage_buffer_size_g,
             prefetch_q_base,
             prefetch_q_entries_g * (uint32_t)sizeof(uint16_t),
             prefetch_q_rd_ptr_addr,
             cmddat_q_base,
             cmddat_q_size_g,
             scratch_db_base,
             scratch_db_size_g,
             prefetch_sync_sem,
        };

        auto sp1 = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch_hd.cpp",
            {prefetch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = prefetch_compile_args,
                .defines = defines
            }
        );

        auto d1 = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            {dispatch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = dispatch_compile_args,
                .defines = defines
            }
        );

        log_info(LogTest, "Hugepage buffer size {}", std::to_string(hugepage_buffer_size_g));
        log_info(LogTest, "Prefetch prefetch_q entries {}", std::to_string(prefetch_q_entries_g));
        log_info(LogTest, "CmdDat buffer size {}", std::to_string(cmddat_q_size_g));
        log_info(LogTest, "Prefetch scratch buffer size {}", std::to_string(scratch_db_size_g));
        log_info(LogTest, "Max command size {}", std::to_string(max_prefetch_command_size_g));
        if (test_type_g >= 2) {
            perf_test_g = true;
        }
        if (test_type_g == 2) {
            perf_test_g = true;
            log_info(LogTest, "PCIE transfer size {}", std::to_string(pcie_transfer_size_g));
        }
        if (test_type_g == 3) {
            perf_test_g = true;
            log_info(LogTest, "DRAM page size {}", std::to_string(dram_page_size_g));
            log_info(LogTest, "DRAM pages to read {}", std::to_string(dram_pages_to_read_g));
        }
        if (debug_g) {
            log_info(LogTest, "Debug mode enabled");
        }
        log_info(LogTest, "Iterations: {}", iterations_g);

        // Cache stuff
        if (warmup_g) {
            std::thread t1 ([&]() {
                write_prefetcher_cmds(1, device, cmds, cmd_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
                write_prefetcher_cmds(1, device, terminate_cmds, terminate_sizes, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
            });
            tt_metal::detail::LaunchProgram(device, program);
            t1.join();
            initialize_device_g = true;
        }

        if (readback_every_iteration_g) {
            for (int i = 0; i < iterations_g; i++) {
                log_info(LogTest, "Iteration: {}", std::to_string(i));
                initialize_device_g = true;
                cmds.resize(0);
                cmd_sizes.resize(0);
                reset_worker_data(worker_data);
                gen_prefetcher_cmds(device, cmds, cmd_sizes, dram_data_map, worker_data, l1_buf_base);
                run_test(1, device, program, cmd_sizes, terminate_sizes, cmds, terminate_cmds, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);
                pass &= validate_results(device, all_workers_g, worker_data, l1_buf_base);
                if (!pass) {
                    break;
                }
            }
        } else {
            auto elapsed_seconds = run_test(iterations_g, device, program, cmd_sizes, terminate_sizes, cmds, terminate_cmds, host_hugepage_base, dev_hugepage_base, prefetch_q_base, prefetch_q_rd_ptr_addr, phys_prefetch_core);

            log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
            log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);
            log_warning(LogTest, "Performance mode, not validating results");
            if (test_type_g == 2 || test_type_g == 3) {
                float bw = bytes_of_data_g * iterations_g / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << bw;
                log_info(LogTest, "BW: {} GB/s", ss.str());
            }
        }

        vector<uint32_t> tx_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_tx_gen_core, test_results_addr, test_results_size);
        log_info(LogTest, "TX status = {}", packet_queue_test_status_to_string(tx_results[PQ_TEST_STATUS_INDEX]));
        pass &= (tx_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        vector<uint32_t> rx_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_rx_gen_core, test_results_addr, test_results_size);
        log_info(LogTest, "RX status = {}", packet_queue_test_status_to_string(rx_results[PQ_TEST_STATUS_INDEX]));
        pass &= (rx_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        vector<uint32_t> mux_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_relay_mux_core, test_results_addr, test_results_size);
        log_info(LogTest, "relay mux status = {}", packet_queue_test_status_to_string(mux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (mux_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        vector<uint32_t> demux_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_relay_demux_core, test_results_addr, test_results_size);
        log_info(LogTest, "relay demux status = {}", packet_queue_test_status_to_string(demux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (demux_results[0] == PACKET_QUEUE_TEST_PASS);

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
