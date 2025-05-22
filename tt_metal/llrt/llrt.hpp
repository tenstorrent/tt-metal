// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core_coord.hpp"
#include <tt_stl/span.hpp>
// clang-format off
#include "impl/context/metal_context.hpp"
#include "tt_memory.h"
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include "utils.hpp"

struct go_msg_t;
struct launch_msg_t;
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

// Return a reference to a potentially shared binary image.
// The images are cached by path name.
const ll_api::memory& get_risc_binary(
    std::string_view path, ll_api::memory::Loading loading = ll_api::memory::Loading::DISCRETE);

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
template <typename DType>
void write_hex_vec_to_core(
    chip_id_t chip,
    const CoreCoord& core,
    const std::vector<DType>& hex_vec,
    uint64_t addr,
    bool small_access = false) {
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(chip, core), addr, small_access);
}
template <typename DType>
void write_hex_vec_to_core(
    chip_id_t chip,
    const CoreCoord& core,
    tt::stl::Span<const DType> hex_vec,
    uint64_t addr,
    bool small_access = false) {
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(chip, core), addr, small_access);
}

std::vector<std::uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord& core, uint64_t addr, uint32_t size);

CoreCoord logical_core_from_ethernet_core(chip_id_t chip_id, CoreCoord& ethernet_core);

void write_launch_msg_to_core(
    chip_id_t chip, CoreCoord core, launch_msg_t* msg, go_msg_t* go_msg, uint64_t addr, bool send_go = true);

bool test_load_write_read_risc_binary(
    const ll_api::memory& mem,
    chip_id_t chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx);
void write_binary_to_address(const ll_api::memory& mem, chip_id_t chip_id, const CoreCoord& core, uint32_t address);

// subchannel hard-coded to 0 for now
CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id = 0);

namespace internal_ {

void wait_until_cores_done(
    chip_id_t device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms = 0);

}  // namespace internal_

}  // namespace llrt

}  // namespace tt
