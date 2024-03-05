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

constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
// 764 to make this not divisible by 3 so we can test wrapping of dispatch buffer
constexpr uint32_t DISPATCH_BUFFER_BLOCK_SIZE_PAGES = 764 * 1024 / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;

constexpr uint32_t DEFAULT_HUGEPAGE_BUFFER_SIZE = 256 * 1024;
constexpr uint32_t DEFAULT_HOST_Q_ENTRIES = 128;
constexpr uint32_t DEFAULT_MAX_PREFETCH_COMMAND_SIZE = 64 * 1024;
constexpr uint32_t DEFAULT_CMDDAT_Q_SIZE = 64 * 1024;
constexpr uint32_t DEFAULT_SCRATCH_CB_SIZE = 128 * 1024;

constexpr uint32_t DEFAULT_ITERATIONS = 10000;

constexpr uint32_t HOST_Q_LOG_MINSIZE = 4;
constexpr uint32_t HUGEPAGE_ALIGNMENT = ((1 << HOST_Q_LOG_MINSIZE) > CQ_PREFETCH_CMD_BARE_MIN_SIZE) ? (1 << HOST_Q_LOG_MINSIZE) : CQ_PREFETCH_CMD_BARE_MIN_SIZE;

constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);
constexpr uint32_t DRAM_HACKED_BASE_ADDR = 1024 * 1024;  // don't interfere w/ fast dispatch storing our kernels...

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
bool warmup_g = false;
uint32_t prefetcher_iterations_g = 1;
bool debug_g;
uint32_t max_prefetch_command_size_g;

uint32_t dispatch_buffer_page_size_g = 4096;
uint32_t host_q_entries_g;
uint32_t hugepage_buffer_size_g;
uint32_t cmddat_q_size_g;
uint32_t scratch_cb_size_g;
uint32_t num_dram_banks_g;
bool use_coherent_data_g;

static bool initialize_device_g = true;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up before starting timer (default disabled)");
        log_info(LogTest, "  -i: host iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -d: wrap all commands in debug commands (default disabled)");
        log_info(LogTest, "  -hp: host huge page buffer size (default {})", DEFAULT_HUGEPAGE_BUFFER_SIZE);
        log_info(LogTest, "  -hq: host queue entries (default {})", DEFAULT_HOST_Q_ENTRIES);
        log_info(LogTest, "  -cs: max cmddat q size (default {})", DEFAULT_CMDDAT_Q_SIZE);
        log_info(LogTest, "  -ss: max scratch cb size (default {})", DEFAULT_SCRATCH_CB_SIZE);
        log_info(LogTest, "  -mc: max command size (default {})", DEFAULT_MAX_PREFETCH_COMMAND_SIZE);
        log_info(LogTest, "  -c: use coherent data as payload (default false)");
        exit(0);
    }

    warmup_g = test_args::has_command_option(input_args, "-w");
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    hugepage_buffer_size_g = test_args::get_command_option_uint32(input_args, "-hp", DEFAULT_HUGEPAGE_BUFFER_SIZE);
    host_q_entries_g = test_args::get_command_option_uint32(input_args, "-hq", DEFAULT_HOST_Q_ENTRIES);
    cmddat_q_size_g = test_args::get_command_option_uint32(input_args, "-cs", DEFAULT_CMDDAT_Q_SIZE);
    scratch_cb_size_g = test_args::get_command_option_uint32(input_args, "-ss", DEFAULT_SCRATCH_CB_SIZE);
    max_prefetch_command_size_g = test_args::get_command_option_uint32(input_args, "-mc", DEFAULT_MAX_PREFETCH_COMMAND_SIZE);
    use_coherent_data_g = test_args::has_command_option(input_args, "-c");

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
    TT_ASSERT((new_size >> HOST_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    sizes.push_back(new_size >> HOST_Q_LOG_MINSIZE);

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

void add_dram_data_to_worker_data(const vector<uint32_t>& dram_data,
                                  vector<uint32_t>& worker_data,
                                  uint32_t start_page,
                                  uint32_t base_addr,
                                  uint32_t page_size,
                                  uint32_t pages) {

    uint32_t base_addr_words = base_addr / sizeof(uint32_t);
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    TT_ASSERT(start_page < num_dram_banks_g);
    uint32_t page = start_page;
    for (uint32_t i = 0; i < pages; i++) {
        uint32_t index = base_addr_words + page * DRAM_DATA_SIZE_WORDS;
        for (uint32_t j = 0; j  < page_size_words; j++) {
            worker_data.push_back(dram_data[index + j]);
        }
        page++;
        if (page == num_dram_banks_g) {
            base_addr_words += page_size_words;
            page = 0;
        }
    }
}

void gen_dram_read_cmd(vector<uint32_t>& prefetch_cmds,
                       vector<uint16_t>& cmd_sizes,
                       const vector<uint32_t>& dram_data,
                       vector<uint32_t>& worker_data,
                       CoreCoord worker_core,
                       uint32_t dst_addr,
                       uint32_t start_page,
                       uint32_t base_addr,
                       uint32_t page_size,
                       uint32_t pages) {

    vector<uint32_t> dispatch_cmds;

    uint32_t worker_data_size = page_size * pages;
    gen_bare_dispatcher_write_cmd(dispatch_cmds, worker_data, worker_core, dst_addr, worker_data_size);

    // XXXX update cmd_sizes to pull the 2 commands at once!!!
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH, dispatch_cmds);

    auto prior_end = prefetch_cmds.size();
    add_prefetcher_dram_cmd(prefetch_cmds, cmd_sizes, start_page, base_addr + DRAM_HACKED_BASE_ADDR, page_size, pages);
    uint32_t new_size = (prefetch_cmds.size() - prior_end) * sizeof(uint32_t);
    TT_ASSERT(new_size <= max_prefetch_command_size_g, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((new_size >> HOST_Q_LOG_MINSIZE) < 0xFFFF, "HostQ command too large to represent");
    cmd_sizes.push_back(new_size >> HOST_Q_LOG_MINSIZE);

    add_dram_data_to_worker_data(dram_data, worker_data, start_page, base_addr, page_size, pages);
}

void gen_prefetcher_cmds(vector<uint32_t>& prefetch_cmds,
                         vector<uint16_t>& cmd_sizes,
                         const vector<uint32_t>& dram_data,
                         vector<uint32_t>& worker_data,
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
    worker_data.resize(0);
    gen_dispatcher_write_cmd(dispatch_cmds, worker_data, worker_core, dst_addr, 2048);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_write_cmd(dispatch_cmds, worker_data, worker_core, dst_addr, 1026);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Write to worker
    dispatch_cmds.resize(0);
    gen_dispatcher_write_cmd(dispatch_cmds, worker_data, worker_core, dst_addr, 8448);
    add_prefetcher_cmd(prefetch_cmds, cmd_sizes, CQ_PREFETCH_CMD_RELAY_INLINE, dispatch_cmds);

    // Read from dram, write to worker
    // start_page, base addr, page_size, pages
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      0, 0, 32, num_dram_banks_g);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      0, 0, 32, num_dram_banks_g);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      4, 32, 64, num_dram_banks_g);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      0, 0, 128, 128);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      4, 32, 2048, num_dram_banks_g * 8);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      5, 32, 2048, num_dram_banks_g * 8 + 1);
    gen_dram_read_cmd(prefetch_cmds, cmd_sizes, dram_data, worker_data, worker_core, dst_addr,
                      3, 128, 6144, num_dram_banks_g * 8 + 7);
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
                          uint32_t& host_q_dev_ptr,
                          uint32_t& host_q_dev_fence,
                          uint32_t host_q_base,
                          uint32_t host_q_rd_ptr_addr,
                          CoreCoord phys_prefetch_core) {

    static vector<uint32_t> read_vec;  // static to avoid realloc

    uint32_t cmd_size_bytes = (uint32_t)cmd_size16b << HOST_Q_LOG_MINSIZE;
    uint32_t cmd_size_words = cmd_size_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < cmd_size_words; i++) {
        *host_mem_ptr = cmds[cmd_offset];
        host_mem_ptr++;
        cmd_offset++;
    }
    // Not clear to me if this is needed, writing to cached PCIe memory
    tt_driver_atomics::sfence();

    // wait for space
    while (host_q_dev_ptr == host_q_dev_fence) {
        tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), host_q_rd_ptr_addr);
        host_q_dev_fence = read_vec[0];
    }

    // wrap
    if (host_q_dev_ptr == host_q_base + host_q_entries_g * sizeof(uint16_t)) {
        host_q_dev_ptr = host_q_base;

        while (host_q_dev_ptr == host_q_dev_fence) {
            tt::Cluster::instance().read_core(read_vec, sizeof(uint32_t), tt_cxy_pair(device->id(), phys_prefetch_core), host_q_rd_ptr_addr);
            host_q_dev_fence = read_vec[0];
        }
    }

    tt::Cluster::instance().write_core((void *)&cmd_size16b, sizeof(uint16_t), tt_cxy_pair(device->id(), phys_prefetch_core), host_q_dev_ptr, true);

    host_q_dev_ptr += sizeof(uint16_t);
}

void write_prefetcher_cmds(uint32_t iterations,
                           Device *device,
                           vector<uint32_t>& prefetch_cmds,
                           vector<uint16_t>& cmd_sizes,
                           void * host_hugepage_base,
                           uint32_t dev_hugepage_base,
                           uint32_t host_q_base,
                           uint32_t host_q_rd_ptr_addr,
                           CoreCoord phys_prefetch_core) {

    static uint32_t *host_mem_ptr;
    static uint32_t host_q_dev_ptr;
    static uint32_t host_q_dev_fence;

    if (initialize_device_g) {
        host_mem_ptr = (uint32_t *)host_hugepage_base;
        host_q_dev_ptr = host_q_base;
        host_q_dev_fence = host_q_base + host_q_entries_g * sizeof(uint16_t);
        initialize_device_g = false;
    }

    for (uint32_t i = 0; i < iterations; i++) {
        uint32_t cmd_ptr = 0;
        for (uint32_t j = 0; j < cmd_sizes.size(); j++) {
            uint32_t cmd_size_words = ((uint32_t)cmd_sizes[j] << HOST_Q_LOG_MINSIZE) / sizeof(uint32_t);
            uint32_t space_at_end_for_wrap_words = CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t);
            if ((void *)(host_mem_ptr + cmd_size_words) > (void *)((uint8_t *)host_hugepage_base + hugepage_buffer_size_g)) {
                // Wrap huge page
                uint32_t offset = 0;
                host_mem_ptr = (uint32_t *)host_hugepage_base;
            }
            write_prefetcher_cmd(device, prefetch_cmds, cmd_ptr, cmd_sizes[j],
                                 host_mem_ptr, host_q_dev_ptr, host_q_dev_fence, host_q_base, host_q_rd_ptr_addr, phys_prefetch_core);
        }
    }
}

void populate_dram(Device *device, vector<uint32_t>& dram_data)
{
    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
    vector<CoreCoord> dram_cores = soc_d.get_dram_cores();
    num_dram_banks_g = dram_cores.size();

    for (uint32_t i = 0; i < DRAM_DATA_SIZE_WORDS * num_dram_banks_g; i++) {
        uint32_t datum = (use_coherent_data_g) ? i : std::rand();
        dram_data.push_back(datum);
    }

    uint32_t offset = 0;
    for (auto core : dram_cores) {
        tt::Cluster::instance().write_core((void *)&dram_data[offset], DRAM_DATA_SIZE_BYTES, tt_cxy_pair(device->id(), core), DRAM_HACKED_BASE_ADDR);
        offset += DRAM_DATA_SIZE_WORDS;
    }
}

int main(int argc, char **argv) {
    init(argc, argv);

    uint32_t dispatch_buffer_pages = DISPATCH_BUFFER_BLOCK_SIZE_PAGES * DISPATCH_BUFFER_SIZE_BLOCKS;

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord prefetch_core = {0, 0};
        CoreCoord dispatch_core = {0, 1};
        CoreCoord worker_core = {0, 2};

        CoreCoord phys_prefetch_core = device->worker_core_from_logical_core(prefetch_core);
        CoreCoord phys_dispatch_core = device->worker_core_from_logical_core(dispatch_core);
        CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

        // Want different buffers on each core, instead use big buffer and self-manage it
        uint32_t l1_buf_base = L1_UNRESERVED_BASE;
        TT_ASSERT((l1_buf_base & ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) == 0);

        uint32_t dispatch_buffer_base = l1_buf_base;
        uint32_t dev_hugepage_base = 1 * 1024 * 1024 * 1024 - hugepage_buffer_size_g; // XXXXX HACK
        uint32_t host_q_base = l1_buf_base;
        uint32_t host_q_rd_ptr_addr = l1_buf_base - 4; // XXXXX hacks and hacks and hacks
        uint32_t host_q_size = host_q_entries_g * sizeof(uint16_t);
        uint32_t noc_read_alignment = 32;
        uint32_t cmddat_q_base = host_q_base + ((host_q_size + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        uint32_t scratch_cb_base = cmddat_q_base + ((cmddat_q_size_g + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
        TT_ASSERT(scratch_cb_base < 1024 * 1024); // L1 size

        // Implementation syncs w/ device on host_q but not on hugepage, ie, assumes we can't run
        // so far ahead in the hugepage that we overright un-read commands since we'll first
        // stall on the host_q
        TT_ASSERT(hugepage_buffer_size_g > host_q_entries_g * max_prefetch_command_size_g, "Shrink the max command size or grow the hugepage buffer size or shrink the host_q size");
        TT_ASSERT(cmddat_q_size_g > 2 * max_prefetch_command_size_g);

        void *host_hugepage_base;
        {
            // XXXXX HACKS - fix w/ allocator changes
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
            host_hugepage_base = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
            host_hugepage_base = (void *)((uint8_t *)host_hugepage_base + dev_hugepage_base);
        }

        vector<uint32_t> cmds, terminate_cmds;
        vector<uint16_t> cmd_sizes, terminate_sizes;
        vector<uint32_t> worker_data;
        vector<uint32_t> host_q(DEFAULT_HOST_Q_ENTRIES, 0);
        vector<uint32_t> host_q_rd_ptr_addr_data;

        host_q_rd_ptr_addr_data.push_back(host_q_base + host_q_entries_g * sizeof(uint16_t));
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, host_q_rd_ptr_addr_data, host_q_rd_ptr_addr);
        llrt::write_hex_vec_to_core(device->id(), phys_prefetch_core, host_q, host_q_base);

        vector<uint32_t> dram_data;
        populate_dram(device, dram_data);

        tt::Cluster::instance().l1_barrier(device->id());
        tt::Cluster::instance().dram_barrier(device->id());
        gen_prefetcher_cmds(cmds, cmd_sizes, dram_data, worker_data, phys_worker_core, l1_buf_base);
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

        std::vector<uint32_t> dispatch_compile_args = {
             dispatch_buffer_base,
             DISPATCH_BUFFER_LOG_PAGE_SIZE,
             DISPATCH_BUFFER_SIZE_BLOCKS * DISPATCH_BUFFER_BLOCK_SIZE_PAGES,
             dispatch_cb_sem,
             DISPATCH_BUFFER_SIZE_BLOCKS,
        };

        std::vector<uint32_t> prefetch_compile_args = {
             dispatch_buffer_base,
             DISPATCH_BUFFER_LOG_PAGE_SIZE,
             dispatch_buffer_pages,
             dispatch_cb_sem,
             dev_hugepage_base,
             hugepage_buffer_size_g,
             host_q_base,
             host_q_entries_g * (uint32_t)sizeof(uint16_t),
             host_q_rd_ptr_addr,
             cmddat_q_base,
             cmddat_q_size_g,
             scratch_cb_base,
             scratch_cb_size_g
        };

        auto sp1 = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/cq_prefetch_hd.cpp",
            {prefetch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = prefetch_compile_args,
                .defines = defines
            }
        );
        vector<uint32_t> args;
        args.push_back(prefetcher_iterations_g);
        tt_metal::SetRuntimeArgs(program, sp1, prefetch_core, args);

        auto d1 = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            {dispatch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = dispatch_compile_args,
                .defines = defines
            }
        );

        log_info(LogTest, "Prefetch host_q entries {}", std::to_string(host_q_entries_g));

        // Cache stuff
        if (warmup_g) {
            EnqueueProgram(cq, program, false);
            write_prefetcher_cmds(1, device, cmds, cmd_sizes, host_hugepage_base, dev_hugepage_base, host_q_base, host_q_rd_ptr_addr, phys_prefetch_core);
            write_prefetcher_cmds(1, device, terminate_cmds, terminate_sizes, host_hugepage_base, dev_hugepage_base, host_q_base, host_q_rd_ptr_addr, phys_prefetch_core);
            Finish(cq);
            initialize_device_g = true;
        }

        auto start = std::chrono::system_clock::now();
        EnqueueProgram(cq, program, false);
        write_prefetcher_cmds(iterations_g, device, cmds, cmd_sizes, host_hugepage_base, dev_hugepage_base, host_q_base, host_q_rd_ptr_addr, phys_prefetch_core);
        write_prefetcher_cmds(1, device, terminate_cmds, terminate_sizes, host_hugepage_base, dev_hugepage_base, host_q_base, host_q_rd_ptr_addr, phys_prefetch_core);
        Finish(cq);

        auto end = std::chrono::system_clock::now();

        pass &= validate_results(device, phys_worker_core, worker_data, l1_buf_base);

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
        log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);
        if (iterations_g == 1) {
            float bw = (float)worker_data.size() * sizeof(uint32_t) * prefetcher_iterations_g / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << bw;
            log_info(LogTest, "BW: {} GB/s", ss.str());
        } else {
            log_info(LogTest, "BW: -- GB/s (use -i 1 to report bandwidth)");
        }
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
