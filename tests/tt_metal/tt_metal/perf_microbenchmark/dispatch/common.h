// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include "core_coord.h"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "noc/noc_parameters.h"

extern bool debug_g;
extern bool use_coherent_data_g;
extern uint32_t dispatch_buffer_page_size_g;
extern CoreCoord first_worker_g;
extern CoreRange all_workers_g;
extern uint32_t min_xfer_size_bytes_g;
extern uint32_t max_xfer_size_bytes_g;
extern bool send_to_all_g;
extern bool perf_test_g;

struct one_worker_data_t {
    vector<bool> valid;
    vector<uint32_t> data;
};

typedef unordered_map<CoreCoord, unordered_map<uint32_t, one_worker_data_t>> worker_data_t;

inline void reset_worker_data(worker_data_t& awd) {
    for (auto& it : awd) {
        for (auto &wd : it.second) {
            wd.second.valid.resize(0);
            wd.second.data.resize(0);
        }
    }
}

// Return number of words in worker-data. Either in legacy mode which arbitrarily just looks at first worker and bank
// and does not consider valid words, or in improved mode that iterates over all workers and counts only valid words
// which could be slow, but was required for paged write+read test unless a global counter of bytes written is used.
inline uint32_t worker_data_size(worker_data_t& awd, bool all_workers_valid_only = false) {
    TT_ASSERT(awd.size() > 0, "Worker data is empty, not expected");
    uint32_t size = 0;

    if (all_workers_valid_only) {
        for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                CoreCoord core(x, y);
                const vector<bool>& dev_valid = awd.at(core).at(0).valid;
                // const vector<uint32_t>& dev_data = awd.at(core).at(0).data;
                for (int i = 0; i < dev_valid.size(); i++) {
                    if (dev_valid[i]){
                        size++;
                    }
                }
            }
        }
    } else {
        auto first_worker = awd.begin()->first;
        TT_ASSERT(awd[first_worker].size() > 0, "Worker data for core: {} is empty, not expected.", first_worker.str());
        auto first_bank_id = awd[first_worker].begin()->first;
        size = awd[first_worker_g][first_bank_id].data.size();
    }

    log_debug(tt::LogTest, "{} - all_workers_valid_only: {} returning size: {} words", __FUNCTION__, all_workers_valid_only, size);
    return size;
}

inline uint32_t padded_size(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

// Return a vector of core coordinates that are used as interleaved paged banks (L1 or DRAM)
inline std::vector<CoreCoord> get_cores_per_bank_id(Device *device, bool is_dram){
    uint32_t num_banks = device->num_banks(is_dram ? BufferType::DRAM : BufferType::L1);
    std::vector<CoreCoord> bank_cores;

    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        auto core = is_dram ?
            device->core_from_dram_channel(device->dram_channel_from_bank_id(bank_id)) :
            device->logical_core_from_bank_id(bank_id);
        bank_cores.push_back(core);
    }

    return bank_cores;
}

// Specific to this test. This test doesn't use Buffers, and for Storage cores in L1 that have 2 banks, they are intended
// to be allocated top-down and carry "negative" offsets via bank_to_l1_offset for cores that have 2 banks. This function
// will scan through all banks bank_to_l1_offset and return the minimum required buffer addr to avoid bank_to_l1_offset
// being applied and underflowing.  In GS this is basically 512B or half the L1 Bank size.
inline uint32_t get_min_required_buffer_addr(Device *device, bool is_dram){

    int32_t smallest_offset = std::numeric_limits<int32_t>::max();
    uint32_t num_banks = device->num_banks(is_dram ? BufferType::DRAM : BufferType::L1);

    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        int32_t offset = is_dram ? device->dram_bank_offset_from_bank_id(bank_id) : device->l1_bank_offset_from_bank_id(bank_id);
        smallest_offset = offset < smallest_offset ? offset : smallest_offset;
    }

    // If negative, flip it and this becomes the min required positive offset for a buffer in bank.
    uint32_t min_required_positive_offset = smallest_offset < 0 ? 0 - smallest_offset : 0;
    log_debug(tt::LogTest, "{} - smallest_offset: {} min_required_positive_offset: {}", __FUNCTION__, smallest_offset, min_required_positive_offset);

    return min_required_positive_offset;
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    uint32_t length) {

    for (uint32_t i = 0; i < length; i++) {
        uint32_t datum = (use_coherent_data_g) ? i : std::rand();
        cmds.push_back(datum);
    }
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    const CoreRange& workers,
                                    worker_data_t& data,
                                    uint32_t length_words) {

    static uint32_t coherent_count = 0;
    const uint32_t bank_id = 0; // No interleaved pages here.

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t i = 0; i < length_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? coherent_count++ : std::rand();
        cmds.push_back(datum);
        for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                CoreCoord core(x, y);
                data[core][bank_id].data.push_back(datum);
                data[core][bank_id].valid.push_back(workers.contains(core));
            }
        }
    }
}

// Generate a random payload for a paged write command. Note: Doesn't currently support using the base_addr here.
inline void generate_random_paged_payload(Device *device,
                                          CQDispatchCmd cmd,
                                          vector<uint32_t>& cmds,
                                          worker_data_t& data,
                                          uint32_t start_page,
                                          bool is_dram) {

    static uint32_t coherent_count = 0x100; // Abitrary starting value, avoid 0x0 since matches with DRAM prefill.
    auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    uint32_t num_banks = device->num_banks(buf_type);
    uint32_t words_per_page = cmd.write_paged.page_size / sizeof(uint32_t);
    log_debug(tt::LogTest, "Starting {} w/ is_dram: {} start_page: {} words_per_page: {}", __FUNCTION__, is_dram, start_page, words_per_page);

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t page_id = start_page; page_id < start_page + cmd.write_paged.pages; page_id++) {

        // 32B alignment taken from InterleavedAddrGen. If 16B page size, pad to 32B.
        const uint32_t page_size_alignment_bytes = 32;
        CoreCoord bank_core;
        uint32_t bank_id = page_id % num_banks;
        uint32_t bank_offset = align(cmd.write_paged.page_size, page_size_alignment_bytes) * (page_id / num_banks);

        if (is_dram) {
            auto dram_channel = device->dram_channel_from_bank_id(bank_id);
            bank_core = device->core_from_dram_channel(dram_channel);
        } else {
            bank_core = device->logical_core_from_bank_id(bank_id);
        }

        // Generate data and add to cmd for sending to device, and worker_data for correctness checking.
        for (uint32_t i = 0; i < words_per_page; i++) {
            uint32_t datum = (use_coherent_data_g) ? (((page_id & 0xFF) << 24) | coherent_count++) : std::rand();
            log_debug(tt::LogTest, "{} - Setting {} page_id: {} word: {} on core: {} (bank_id: {} bank_offset: {}) => datum: 0x{:x}",
                __FUNCTION__, is_dram ? "DRAM" : "L1", page_id, i, bank_core.str(), bank_id, bank_offset, datum);
            cmds.push_back(datum); // Push to device.
            data[bank_core][bank_id].data.push_back(datum); // Checking
            data[bank_core][bank_id].valid.push_back(true);
        }

        data[bank_core][bank_id].data.resize(padded_size(data[bank_core][bank_id].data.size(), page_size_alignment_bytes / sizeof(uint32_t)));
        data[bank_core][bank_id].valid.resize(padded_size(data[bank_core][bank_id].valid.size(), page_size_alignment_bytes / sizeof(uint32_t)));

    }
}

inline void generate_random_packed_payload(vector<uint32_t>& cmds,
                                           vector<CoreCoord>& worker_cores,
                                           worker_data_t& data,
                                           uint32_t size_words) {

    static uint32_t coherent_count = 0;
    const uint32_t bank_id = 0; // No interleaved pages here.

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
        for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
            CoreCoord core(x, y);
            for (uint32_t i = 0; i < size_words; i++) {
                bool contains = false;
                for (CoreCoord worker_core : worker_cores) {
                    if (core == worker_core) {
                        contains = true;
                        break;
                    }
                }
                if (contains) {
                    uint32_t datum = (use_coherent_data_g) ? ((x << 16) | (y << 24) | coherent_count++) : std::rand();

                    cmds.push_back(datum);
                    data[core][bank_id].data.push_back(datum);
                    data[core][bank_id].valid.push_back(true);
                } else {
                    data[core][bank_id].data.push_back(0xbaadf00d);
                    data[core][bank_id].valid.push_back(false);
                }
            }

            cmds.resize(padded_size(cmds.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
            data[core][bank_id].data.resize(padded_size(data[core][bank_id].data.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
            data[core][bank_id].valid.resize(padded_size(data[core][bank_id].valid.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
        }
    }
}

inline void add_bare_dispatcher_cmd(vector<uint32_t>& cmds,
                                    CQDispatchCmd cmd) {

    uint32_t *ptr = (uint32_t *)&cmd;
    for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
        cmds.push_back(*ptr++);
    }
}

inline size_t debug_prologue(vector<uint32_t>& cmds) {
    size_t prior = cmds.size();

    if (debug_g) {
        CQDispatchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_DISPATCH_CMD_DEBUG;
        add_bare_dispatcher_cmd(cmds, debug_cmd);
    }

    return prior;
}

inline void debug_epilogue(vector<uint32_t>& cmds,
                           size_t prior_end) {
    if (debug_g) {
        // Doing a checksum on the full command length is problematic in the kernel
        // as it requires the debug code to pull all the pages in before the actual
        // command is processed.  So, limit this to doing a checksum on the first page
        // (which is disappointing).  Any other value requires the checksum code to handle
        // buffer wrap which then messes up the routines w/ the embedded insn - not worth it
        CQDispatchCmd* debug_cmd_ptr;
        debug_cmd_ptr = (CQDispatchCmd *)&cmds[prior_end];
        uint32_t full_size = (cmds.size() - prior_end) * sizeof(uint32_t) - sizeof(CQDispatchCmd);
        uint32_t max_size = dispatch_buffer_page_size_g - sizeof(CQDispatchCmd);
        uint32_t size = (full_size > max_size) ? max_size : full_size;
        debug_cmd_ptr->debug.size = size;
        debug_cmd_ptr->debug.stride = sizeof(CQDispatchCmd);
        uint32_t checksum = 0;
        uint32_t start = prior_end + sizeof(CQDispatchCmd) / sizeof(uint32_t);
        for (uint32_t i = start; i < start + size / sizeof(uint32_t); i++) {
            checksum += cmds[i];
        }
        debug_cmd_ptr->debug.checksum = checksum;
    }
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, length_words);

    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               const CoreRange& workers,
                               worker_data_t& worker_data,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, workers, worker_data, length_words);

    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_paged_cmd(Device *device,
                                     vector<uint32_t>& cmds,
                                     worker_data_t& worker_data,
                                     CQDispatchCmd cmd,
                                     uint32_t start_page,
                                     bool is_dram) {

    size_t prior_end = debug_prologue(cmds);
    add_bare_dispatcher_cmd(cmds, cmd);
    generate_random_paged_payload(device, cmd, cmds, worker_data, start_page, is_dram);
    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_packed_cmd(Device *device,
                                      vector<uint32_t>& cmds,
                                      vector<CoreCoord>& worker_cores,
                                      worker_data_t& worker_data,
                                      CQDispatchCmd cmd,
                                      uint32_t size_words) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    for (CoreCoord core : worker_cores) {
        CoreCoord phys_worker_core = device->worker_core_from_logical_core(core);
        cmds.push_back(NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y));
    }
    cmds.resize(padded_size(cmds.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)

    generate_random_packed_payload(cmds, worker_cores, worker_data, size_words);

    debug_epilogue(cmds, prior_end);
}

// bare: doesn't generate random payload data, for use w/ eg, dram reads
inline void gen_bare_dispatcher_unicast_write_cmd(Device *device,
                                                  vector<uint32_t>& cmds,
                                                  CoreCoord worker_core,
                                                  worker_data_t& worker_data,
                                                  uint32_t dst_addr,
                                                  uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = dst_addr + worker_data_size(worker_data) * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    add_bare_dispatcher_cmd(cmds, cmd);
}

inline void gen_dispatcher_unicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreCoord worker_core,
                                             worker_data_t& worker_data,
                                             uint32_t dst_addr,
                                             uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);
    const uint32_t bank_id = 0; // No interleaved pages here.

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = dst_addr + worker_data[worker_core][bank_id].data.size() * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    add_dispatcher_cmd(cmds, worker_core, worker_data, cmd, length);
}

inline void gen_dispatcher_multicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreRange worker_core_range,
                                             worker_data_t& worker_data,
                                             uint32_t dst_addr,
                                             uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord physical_start = device->physical_core_from_logical_core(worker_core_range.start, CoreType::WORKER);
    CoreCoord physical_end = device->physical_core_from_logical_core(worker_core_range.end, CoreType::WORKER);
    const uint32_t bank_id = 0; // No interleaved pages here.

    // Sanity check that worker_data covers all cores being targeted.
    for (uint32_t y = worker_core_range.start.y; y <= worker_core_range.end.y; y++) {
        for (uint32_t x = worker_core_range.start.x; x <= worker_core_range.end.x; x++) {
            CoreCoord worker(x, y);
            TT_ASSERT(worker_data.find(worker) != worker_data.end(), "Worker core x={},y={} missing in worker_data", x, y);
        }
    }

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);
    cmd.write_linear.addr = dst_addr + worker_data[worker_core_range.start][bank_id].data.size() * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = worker_core_range.size();
    log_debug(tt::LogTest, "{} Setting addr: {} from dst_addr: {} and worker_data[{}][{}].data.size(): {}",
        __FUNCTION__, (uint32_t) cmd.write_linear.addr, dst_addr, worker_core_range.start.str(), bank_id,
        worker_data[worker_core_range.start][bank_id].data.size() * sizeof(uint32_t));

    add_dispatcher_cmd(cmds, worker_core_range, worker_data, cmd, length);
}

inline void gen_dispatcher_paged_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             worker_data_t& worker_data,
                                             bool is_dram,
                                             uint32_t start_page,
                                             uint32_t dst_addr,
                                             uint32_t page_size,
                                             uint32_t pages) {

    const uint32_t page_size_alignment_bytes = 32;
    uint32_t num_banks = device->num_banks(is_dram ? BufferType::DRAM : BufferType::L1);

    // Not safe to mix paged L1 and paged DRAM writes currently in this test since same book-keeping.
    static uint32_t prev_is_dram = -1;
    TT_ASSERT(prev_is_dram == -1 || prev_is_dram == is_dram, "Mixing paged L1 and paged DRAM writes not supported in this test.");
    prev_is_dram = is_dram;

    // Assumption embedded in this function (seems reasonable, true with a single buffer) that paged size will never change.
    static uint32_t prev_page_size = -1;
    TT_ASSERT(prev_page_size == -1 || prev_page_size == page_size, "Page size changed between calls to gen_dispatcher_paged_write_cmd - not supported.");
    prev_page_size = page_size;

    // For the CMD generation, start_page is 8 bits, so much wrap around, and increase base_addr instead based on page size,
    // which assumes page size never changed between calls to this function (checked above).
    uint32_t bank_offset = align(page_size, page_size_alignment_bytes) * (start_page / num_banks);
    uint32_t base_addr = dst_addr + bank_offset;
    uint8_t start_page_cmd = start_page % num_banks;

    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
    cmd.write_paged.is_dram = is_dram;
    cmd.write_paged.start_page = start_page_cmd;
    cmd.write_paged.base_addr = base_addr;
    cmd.write_paged.page_size = page_size;
    cmd.write_paged.pages = pages;

    log_debug(tt::LogTest, "Adding CQ_DISPATCH_CMD_WRITE_PAGED - is_dram: {} start_page: {} start_page_cmd: {} base_addr: 0x{:x} bank_offset: 0x{:x} page_size: {} pages: {})",
        is_dram, start_page, start_page_cmd, base_addr, bank_offset, page_size, pages);

    add_dispatcher_paged_cmd(device, cmds, worker_data, cmd, start_page, is_dram);
}


inline void gen_dispatcher_packed_write_cmd(Device *device,
                                            vector<uint32_t>& cmds,
                                            vector<CoreCoord>& worker_cores,
                                            worker_data_t& worker_data,
                                            uint32_t dst_addr,
                                            uint32_t size_words) {

    CQDispatchCmd cmd;

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
    cmd.write_packed.is_multicast = 0;
    cmd.write_packed.count = worker_cores.size();
    cmd.write_packed.addr = dst_addr + worker_data_size(worker_data) * sizeof(uint32_t);
    cmd.write_packed.size = size_words * sizeof(uint32_t);

    add_dispatcher_packed_cmd(device, cmds, worker_cores, worker_data, cmd, size_words);
}

inline uint32_t gen_rnd_dispatcher_packed_write_cmd(Device *device,
                                                    vector<uint32_t>& cmds,
                                                    worker_data_t& worker_data,
                                                    uint32_t dst_addr) {

    // Note: this cmd doesn't clamp to a max size which means it can overflow L1 buffer
    // However, this cmd doesn't send much data and the L1 buffer is < L1 limit, so...

    uint32_t xfer_size_words = (std::rand() % (dispatch_buffer_page_size_g / sizeof(uint32_t))) + 1;
    uint32_t xfer_size_bytes = xfer_size_words * sizeof(uint32_t);
    if (perf_test_g) {
        TT_ASSERT(max_xfer_size_bytes_g < dispatch_buffer_page_size_g);
        if (xfer_size_bytes > max_xfer_size_bytes_g) xfer_size_bytes = max_xfer_size_bytes_g;
        if (xfer_size_bytes < min_xfer_size_bytes_g) xfer_size_bytes = min_xfer_size_bytes_g;
    }

    vector<CoreCoord> gets_data;
    for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
        for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
            if (send_to_all_g || std::rand() % 2) {
                gets_data.push_back({x, y});
            }
        }
    }
    if (gets_data.size() == 0) {
        gets_data.push_back({all_workers_g.start.x, all_workers_g.start.y});
    }

    gen_dispatcher_packed_write_cmd(device, cmds, gets_data, worker_data,
                                    dst_addr, xfer_size_bytes / sizeof(uint32_t));

    return xfer_size_bytes;
}

inline void gen_dispatcher_host_write_cmd(vector<uint32_t>& cmds, uint32_t length) {

    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_HOST;
    // Include cmd in transfer
    cmd.write_linear_host.length = length + sizeof(CQDispatchCmd);

    add_dispatcher_cmd(cmds, cmd, length);
}

inline void gen_dispatcher_terminate_cmd(vector<uint32_t>& cmds) {

    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    add_dispatcher_cmd(cmds, cmd, 0);
}


// Validate a single core's worth of results vs expected. bank_id is purely informational, can pass -1 for non-paged testing.
inline bool validate_core_data(std::unordered_set<CoreCoord> &validated_cores, Device *device, const worker_data_t& worker_data, CoreCoord core, uint32_t base_addr, bool is_paged, uint32_t bank_id, bool is_dram) {

    int fail_count = 0;
    const vector<uint32_t>& dev_data = worker_data.at(core).at(bank_id).data;
    const vector<bool>& dev_valid = worker_data.at(core).at(bank_id).valid;
    uint32_t size_bytes =  dev_data.size() * sizeof(uint32_t);

    // If we had SOC desc, could get core type this way. Revisit this.
    // soc_desc.cores.at(core).type
    auto core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;
    CoreCoord phys_core;
    int32_t bank_offset = 0;

    if (is_paged){
        if (is_dram) {
            auto channel = device->dram_channel_from_bank_id(bank_id);
            phys_core = device->core_from_dram_channel(channel);
            bank_offset = device->dram_bank_offset_from_bank_id(bank_id);
        } else {
            phys_core = device->physical_core_from_logical_core(core, core_type);
            bank_offset = device->l1_bank_offset_from_bank_id(bank_id);
        }

        log_debug(tt::LogTest, "Paged-{} for bank_id: {} has base_addr: (0x{:x}) and DRAM bank offset: {:x})",
            is_dram ? "DRAM" : "L1", bank_id, base_addr, bank_offset);

        base_addr += bank_offset;
    } else {
        TT_ASSERT(core_type == CoreType::WORKER, "Non-Paged write tests expected to target L1 only for now and not DRAM.");
        phys_core = device->physical_core_from_logical_core(core, core_type);
    }

    // Read results from device and compare to expected for this core.
    vector<uint32_t> results = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, size_bytes);

    log_info(tt::LogTest, "Validating {} bytes from {} (paged bank_id: {}) from logical_core: {} / physical: {} at addr: 0x{:x}",
        size_bytes, is_dram ? "DRAM" : "L1", bank_id, core.str(), phys_core.str(), base_addr);

    for (int i = 0; i < dev_data.size(); i++) {
        if (!dev_valid[i]) continue;
        validated_cores.insert(core);

        if (results[i] != dev_data[i]) {
            if (!fail_count) {
                log_fatal(tt::LogTest, "Data mismatch - First 20 failures for logical_core: {} (physical: {})", core.str(), phys_core.str());
            }
            log_fatal(tt::LogTest, "[{:02d}] (Fail) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[i], (unsigned int)results[i]);
            if (fail_count++ > 20) {
                break;
            }
        } else {
            log_debug(tt::LogTest, "[{:02d}] (Pass) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[i], (unsigned int)results[i]);
        }
    }
    return fail_count;
}

inline bool validate_results(Device *device, CoreRange workers, const worker_data_t& worker_data, uint64_t l1_buf_base) {

    bool failed = false;
    std::unordered_set<CoreCoord> validated_cores;
    for (uint32_t y = workers.start.y; y <= workers.end.y; y++) {
        for (uint32_t x = workers.start.x; x <= workers.end.x; x++) {
            CoreCoord worker(x, y);
            failed |= validate_core_data(validated_cores, device, worker_data, worker, l1_buf_base, false, 0, false);
        }
    }

    log_info(tt::LogTest, "Validated {} cores total.", validated_cores.size());
    return !failed;
}

// Validate paged writes to DRAM or L1 by iterating over just the banks/cores that are valid for the worker date based on prior writes.
inline bool validate_results_paged(Device *device, const worker_data_t& worker_data, uint64_t l1_buf_base, bool is_dram) {

    // Get banks, to be able to iterate over them in order here, since worker_data is umap.
    auto bank_cores = get_cores_per_bank_id(device, is_dram);
    std::unordered_set<CoreCoord> validated_cores;
    bool failed = false;

    for (int bank_id = 0; bank_id < bank_cores.size(); bank_id++) {
        auto bank_core = bank_cores.at(bank_id);
        if ((worker_data.find(bank_core) == worker_data.end()) ||
            (worker_data.at(bank_core).find(bank_id) == worker_data.at(bank_core).end())) {
            log_debug(tt::LogTest, "Skipping bank_id: {} bank_core: {} as it has no data", bank_id, bank_core.str());
            continue;
        }

        failed |= validate_core_data(validated_cores, device, worker_data, bank_core, l1_buf_base, true, bank_id, is_dram);
    }

    log_info(tt::LogTest, "{} - Validated {} cores total.", __FUNCTION__, validated_cores.size());
    return !failed;
}
