// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "noc/noc_parameters.h"

extern bool debug_g;
extern bool use_coherent_data_g;
extern uint32_t dispatch_buffer_page_size_g;
extern CoreCoord first_worker_g;
extern CoreRange all_workers_g;

struct one_worker_data_t {
    vector<bool> valid;
    vector<uint32_t> data;
};

typedef unordered_map<CoreCoord, one_worker_data_t> worker_data_t;

void reset_worker_data(worker_data_t& awd) {
    for (auto& wd : awd) {
        wd.second.valid.resize(0);
        wd.second.data.resize(0);
    }
}

uint32_t worker_data_size(worker_data_t& awd) {
    return awd[first_worker_g].data.size();

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

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t i = 0; i < length_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? i : std::rand();
        cmds.push_back(datum);
        for (uint32_t y = all_workers_g.start.y; y < all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x < all_workers_g.end.x; x++) {
                CoreCoord core(x, y);
                data[core].data.push_back(datum);
                data[core].valid.push_back(workers.contains(core));
            }
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

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               const CoreRange& workers,
                               worker_data_t& worker_data,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    auto prior_end = cmds.size();

    if (debug_g) {
        CQDispatchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_DISPATCH_CMD_DEBUG;
        add_bare_dispatcher_cmd(cmds, debug_cmd);
    }

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, workers, worker_data, length_words);

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

// bare: doesn't generate random payload data, for use w/ eg, dram reads
inline void gen_bare_dispatcher_unicast_write_cmd(Device *device,
                                                  vector<uint32_t>& cmds,
                                                  CoreCoord worker_core,
                                                  worker_data_t& worker_data,
                                                  uint32_t dst_addr,
                                                  uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE;
    cmd.write.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write.addr = dst_addr + worker_data[worker_core].data.size() * sizeof(uint32_t);
    cmd.write.length = length;

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

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE;
    cmd.write.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write.addr = dst_addr + worker_data[worker_core].data.size() * sizeof(uint32_t);
    cmd.write.length = length;

    add_dispatcher_cmd(cmds, worker_core, worker_data, cmd, length);
}

inline void gen_dispatcher_terminate_cmd(vector<uint32_t>& cmds) {

    worker_data_t dummy_data;
    CoreCoord worker_dummy;
    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    add_dispatcher_cmd(cmds, worker_dummy, dummy_data, cmd, 0);
}

inline bool validate_results(Device *device, CoreRange workers, const worker_data_t& worker_data, uint64_t l1_buf_base) {

    bool failed = false;
    for (uint32_t y = workers.start.y; y < workers.end.y; y++) {
        for (uint32_t x = workers.start.x; x < workers.end.x; x++) {
            CoreCoord worker(x, y);
            CoreCoord phys_worker = device->worker_core_from_logical_core(worker);

            const vector<uint32_t>& dev_data = worker_data.at(worker).data;
            const vector<bool>& dev_valid = worker_data.at(worker).valid;
            log_info(tt::LogTest, "Validating {} bytes for core {}", dev_data.size() * sizeof(uint32_t), worker.str());

            vector<uint32_t> results =
                tt::llrt::read_hex_vec_from_core(device->id(), phys_worker, l1_buf_base, dev_data.size() * sizeof(uint32_t));

            int fail_count = 0;

            for (int i = 0; i < dev_data.size(); i++) {
                if (dev_valid[i] && results[i] != dev_data[i]) {
                    if (!failed) {
                        tt::log_fatal("Data mismatch");
                        fprintf(stderr, "First 10 failures for each core: [idx] expected->read\n");
                    }
                    if (fail_count == 0) {
                        fprintf(stderr, "Failures logical core: (%ld,%ld)\n", worker.x, worker.y);
                    }

                    fprintf(stderr, "  [%02d] 0x%08x->0x%08x\n", i, (unsigned int)dev_data[i], (unsigned int)results[i]);

                    failed = true;
                    fail_count++;
                    if (fail_count > 10) {
                        break;
                    }
                }
            }
        }
    }

    return !failed;
}
