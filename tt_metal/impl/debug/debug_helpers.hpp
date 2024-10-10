// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include "hostdevcommon/dprint_common.h"
#include "tt_metal/impl/device/device.hpp"

// Helper function for comparing CoreDescriptors for using in sets.
struct CoreDescriptorComparator {
    bool operator()(const CoreDescriptor &x, const CoreDescriptor &y) const {
        if (x.coord == y.coord) {
            return x.type < y.type;
        } else {
            return x.coord < y.coord;
        }
    }
};
#define CoreDescriptorSet std::set<CoreDescriptor, CoreDescriptorComparator>

// Helper function to get CoreDescriptors for all debug-relevant cores on device.
static CoreDescriptorSet GetAllCores(Device *device) {
    CoreDescriptorSet all_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size = device->logical_grid_size();
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            all_cores.insert({{x, y}, CoreType::WORKER});
        }
    }
    for (const auto& logical_core : device->get_active_ethernet_cores()) {
        all_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core : device->get_inactive_ethernet_cores()) {
        all_cores.insert({logical_core, CoreType::ETH});
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
static CoreDescriptorSet GetDispatchCores(Device* device) {
    CoreDescriptorSet dispatch_cores;
    unsigned num_cqs = device->num_hw_cqs();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    tt::log_warning("Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(device->id(), num_cqs, dispatch_core_type)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

inline uint64_t GetDprintBufAddr(Device *device, const CoreCoord &phys_core, int risc_id) {

    dprint_buf_msg_t *buf = device->get_dev_addr<dprint_buf_msg_t *>(phys_core, HalL1MemAddrType::DPRINT);
    return reinterpret_cast<uint64_t>(buf->data[risc_id]);
}

inline int GetNumRiscs(const CoreDescriptor &core) {
    return (core.type == CoreType::ETH)? DPRINT_NRISCVS_ETH : DPRINT_NRISCVS;
}
