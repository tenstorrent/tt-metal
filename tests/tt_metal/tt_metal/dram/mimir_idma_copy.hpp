// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

// Host side of the cce_idma_copy kernel ABI, for the dram/ suites and context/test_mimir_emu.cpp. The context target
// links no dram/ source file and builds without Grendel JTAG, so this header stays inline and includes no header that
// needs either (mimir_access.hpp, mimir_hw_state.hpp, mimir_fw_loader.hpp, mimir_snoop_ctl.hpp).
namespace tt::tt_metal::mimir_test {

inline constexpr const char* kIdmaCopyKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp";
inline constexpr CceStatusLine kIdmaCopyStatusLine{
    .done_magic = cce_idma_copy_abi::kDoneMagic,
    .num_words = cce_idma_copy_abi::kStatusWords,
    .poison = cce_idma_copy_abi::kStatusPoison};

// One source or destination as the kernel takes it: a kKind*, its selector and its offset.
struct IdmaOperand {
    uint32_t kind = 0;
    uint32_t selector = 0;
    uint64_t offset = 0;
};

// One launch's runtime args, a field per kArg* in cce_idma_copy_abi.h order. A field left at its default sends 0.
struct IdmaCopyArgs {
    uint32_t mode = 0;
    uint32_t backend_vc = 0;
    uint64_t status_offset = 0;
    uint32_t len_bytes = 0;
    uint64_t poll_budget_cycles = 0;
    IdmaOperand source;
    IdmaOperand destination;
    uint32_t expected_xor = 0;
    uint32_t num_elements = 0;
    uint32_t source_stride_bytes = 0;
    uint32_t trid = 0;
    bool snoop = false;
    uint32_t destination_stride_bytes = 0;
    uint32_t trid_stride = 0;
    uint64_t trace_offset = 0;
    uint64_t command_table_offset = 0;
    uint64_t stop_offset = 0;
    uint64_t progress_offset = 0;
    uint32_t seed = 0;
    uint32_t accumulate_format = 0;
    uint32_t accumulate_op = 0;
    uint32_t accumulate_disable_saturation = 0;
    uint32_t expected_lo = 0;
    uint32_t expected_hi = 0;
    uint64_t peer_flag_offset = 0;
    bool host_verified = false;
    uint32_t command_buffer = 0;
    uint64_t poll_region_offset = 0;
    uint32_t poll_region_bytes = 0;
    bool check_at_ack = false;
};

inline std::vector<uint32_t> idma_copy_args(const IdmaCopyArgs& copy) {
    namespace abi = cce_idma_copy_abi;
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = copy.mode;
    args[abi::kArgBackendVc] = copy.backend_vc;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(copy.status_offset);
    args[abi::kArgLenBytes] = copy.len_bytes;
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(copy.poll_budget_cycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(copy.poll_budget_cycles >> 32);
    args[abi::kArgSrcKind] = copy.source.kind;
    args[abi::kArgSrcSelector] = copy.source.selector;
    args[abi::kArgSrcOffsetLo] = static_cast<uint32_t>(copy.source.offset);
    args[abi::kArgSrcOffsetHi] = static_cast<uint32_t>(copy.source.offset >> 32);
    args[abi::kArgDstKind] = copy.destination.kind;
    args[abi::kArgDstSelector] = copy.destination.selector;
    args[abi::kArgDstOffsetLo] = static_cast<uint32_t>(copy.destination.offset);
    args[abi::kArgDstOffsetHi] = static_cast<uint32_t>(copy.destination.offset >> 32);
    args[abi::kArgExpectedXor] = copy.expected_xor;
    args[abi::kArgNumElements] = copy.num_elements;
    args[abi::kArgSourceStrideBytes] = copy.source_stride_bytes;
    args[abi::kArgTrid] = copy.trid;
    args[abi::kArgSnoop] = copy.snoop ? 1 : 0;
    args[abi::kArgDestinationStrideBytes] = copy.destination_stride_bytes;
    args[abi::kArgTridStride] = copy.trid_stride;
    args[abi::kArgTraceOffset] = static_cast<uint32_t>(copy.trace_offset);
    args[abi::kArgCmdTableOffset] = static_cast<uint32_t>(copy.command_table_offset);
    args[abi::kArgStopOffset] = static_cast<uint32_t>(copy.stop_offset);
    args[abi::kArgProgressOffset] = static_cast<uint32_t>(copy.progress_offset);
    args[abi::kArgSeed] = copy.seed;
    args[abi::kArgAccumulateFormat] = copy.accumulate_format;
    args[abi::kArgAccumulateOp] = copy.accumulate_op;
    args[abi::kArgAccumulateDisableSaturation] = copy.accumulate_disable_saturation;
    args[abi::kArgExpectedLo] = copy.expected_lo;
    args[abi::kArgExpectedHi] = copy.expected_hi;
    args[abi::kArgPeerFlagOffset] = static_cast<uint32_t>(copy.peer_flag_offset);
    args[abi::kArgHostVerified] = copy.host_verified ? 1 : 0;
    args[abi::kArgCommandBuffer] = copy.command_buffer;
    args[abi::kArgPollRegionOffset] = static_cast<uint32_t>(copy.poll_region_offset);
    args[abi::kArgPollRegionBytes] = copy.poll_region_bytes;
    args[abi::kArgCheckAtAck] = copy.check_at_ack ? 1 : 0;
    return args;
}

// MIM-407 backend of a write into GDDR: GDDR0 is the low 4 GiB tile of a Mimir, GDDR1 the high one.
inline uint32_t gddr_backend_for(uint64_t chiplet_offset) {
    return mimir_channel_location(chiplet_offset).tile == 0 ? cce_idma_copy_abi::kBackendGddr0
                                                            : cce_idma_copy_abi::kBackendGddr1;
}

// Where an iDMA source or destination lives. Tl1 kinds name the owning CCE and a TL1 offset; Gddr names a DRAM view
// and an offset in it.
enum class EndpointKind { OwnTl1, SramSpa, Gddr };

struct Endpoint {
    EndpointKind kind = EndpointKind::OwnTl1;
    uint32_t cce_index = 0;
    uint32_t view = 0;
    uint64_t offset = 0;
};

inline Endpoint leading_guard_of(const Endpoint& endpoint) {
    Endpoint guard = endpoint;
    guard.offset -= kGuardBytes;
    return guard;
}

}  // namespace tt::tt_metal::mimir_test
