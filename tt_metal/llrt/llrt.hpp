// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

// clang-format off
#include "llrt/tt_cluster.hpp"
#include "tensix.h"
#include "tt_metal/third_party/umd/device/device_api_metal.h"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"
#include "llrt_common/tiles.hpp"
#include "llrt/tt_memory.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "jit_build/build.hpp"
#include "dev_msgs.h"
// clang-format on

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using RamSrcAddr = unsigned int;
using RamDstAddr = unsigned int;
using SrcL1Core = CoreCoord;
using SrcL1Cores = std::vector<SrcL1Core>;
using DstL1Core = CoreCoord;
using DstL1Cores = std::vector<DstL1Core>;
using SrcChannelId = int;
using DstChannelId = int;
using DramBufferSize = unsigned int;
using DramSrcAddr = unsigned int;
using DramDstAddr = unsigned int;
using L1Addr = std::uint32_t;
using SrcAddr = std::uint32_t;
using DestAddr = std::uint32_t;
using LoadFirmwareFlag = bool;
using CountOffset = unsigned int;
using NCHW = std::array<std::uint32_t, 4>;
using RSUV = std::array<std::uint32_t, 4>;
using BYTES_PER_DATUM = std::uint32_t;
using TRANSACTION_SIZE = std::uint32_t;
using NUM_TRANSACTIONS = std::uint32_t;
using NUM_REPETITIONS = std::uint32_t;

using WorkerCore = tt_cxy_pair;
using WorkerCores = std::vector<WorkerCore>;

ll_api::memory get_risc_binary(string path);
uint16_t get_binary_code_size16(const ll_api::memory &mem, int riscv_id);

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
void write_hex_vec_to_core(
    chip_id_t chip,
    const CoreCoord &core,
    const std::vector<uint32_t> &hex_vec,
    uint64_t addr,
    bool small_access = false);

std::vector<std::uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t size);

CoreCoord logical_core_from_ethernet_core(chip_id_t chip_id, CoreCoord &physical_core);

void write_launch_msg_to_core(chip_id_t chip, CoreCoord core, launch_msg_t *msg, uint64_t addr, bool send_go = true);

void launch_erisc_app_fw_on_core(chip_id_t chip, CoreCoord core);

void print_worker_cores(chip_id_t chip_id = 0);

inline bool is_worker_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_workers.begin(), soc_desc.physical_workers.end(), core) !=
           soc_desc.physical_workers.end();
}

inline bool is_ethernet_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_ethernet_cores.begin(), soc_desc.physical_ethernet_cores.end(), core) !=
           soc_desc.physical_ethernet_cores.end();
}

uint32_t generate_risc_startup_addr(bool is_eth_core);
void program_risc_startup_addr(chip_id_t chip_id, const CoreCoord &core);

bool test_load_write_read_risc_binary(ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int riscv_id);

bool test_load_write_read_trisc_binary(ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int triscv_id);

// subchannel hard-coded to 0 for now
CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id = 0);

namespace internal_ {

void wait_until_cores_done(
    chip_id_t device_id, int run_state, std::unordered_set<CoreCoord> &not_done_phys_cores, int timeout_ms = 0);

}  // namespace internal_

inline uint64_t relocate_dev_addr(uint64_t addr, uint64_t local_init_addr) {
    uint64_t relo_addr;
    if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
        // Move addresses in the local memory range to l1 (copied by kernel)
        relo_addr = (addr & ~MEM_LOCAL_BASE) + local_init_addr;
    }
#ifdef NCRISC_HAS_IRAM
    else if ((addr & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
        // Move addresses in the trisc memory range to l1 (copied by kernel)
        relo_addr = (addr & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
    }
#endif
    else {
        relo_addr = addr;
    }
    return relo_addr;
}

}  // namespace llrt

}  // namespace tt
