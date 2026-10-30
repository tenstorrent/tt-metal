// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

// The Glean MM-4 plan names existing DRAM tests to port: dram_unary, api/test_dram, api/test_direct, the DRAM
// loopback and single-DM L1-write tests, and dram_sharded. Their kernels reach DRAM through the NoC dataflow API
// (Noc::async_read/async_write with AllocatorBank<DRAM>), which no Mimir CCE kernel has run yet. So these tests
// move the data with the CCE's iDMA, through cce_idma_copy.cpp's kModeTridBatch, until that is proven.

namespace tt::tt_metal::mimir_test::local_dram {

namespace abi = cce_idma_copy_abi;

// Destination slots hold [guard | region | guard]. A slot is one 4 KiB region plus both guards, so slots
// never overlap and every slot offset is 64-byte aligned.
constexpr uint32_t kMaxSlotRegionBytes = 4096;
constexpr uint64_t kSlotStrideBytes = kMaxSlotRegionBytes + 2 * kGuardBytes;
constexpr uint32_t kDestinationPoisonWord = 0xDEADBEEFu;
constexpr uint64_t kCopyTestBytes = 4 * kKiB;
constexpr const char* kIdmaCopyKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp";
// A launch shares one poll budget across its waits, so one budget plus launch.
constexpr std::chrono::seconds kHostDeadline{300};
// Own-TL1 destinations of a launch in which every hart copies into TL1 sit back to back from here, in hart 0's data
// area above every hart's sources. Status lines (tl1_done_block) sit below every data area.
constexpr uint64_t kSharedDestinationOffsetFromData = 64 * kKiB;

std::string topology_skip_reason(uint32_t cce_index) {
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (cce_index >= topology.num_cces) {
        return fmt::format("Descriptor has {} CCEs; CCE {} is not present.", topology.num_cces, cce_index);
    }
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    if (std::find(topology.view_mimir.begin(), topology.view_mimir.end(), local_mimir) == topology.view_mimir.end()) {
        return fmt::format("Descriptor has no DRAM view on Mimir {} for CCE {}.", local_mimir, cce_index);
    }
    return "";
}

// Lowest offset the test may use in the anchor: the DRAM UNRESERVED offset. Reserved DRAM sits below it.
uint64_t window_begin_of(const MimirAnchor& anchor, uint64_t unreserved) {
    return std::max(anchor.view_offset, unreserved);
}

uint64_t window_end_of(const MimirAnchor& anchor) { return anchor.view_offset + kAnchorBytes; }

// True when the anchor window, inside the view and at or above UNRESERVED, holds `bytes` contiguous bytes.
bool window_holds_bytes(
    const metal_SocDescriptor& soc, const MimirAnchor& anchor, uint64_t unreserved, uint64_t bytes) {
    const uint64_t window_begin = window_begin_of(anchor, unreserved);
    const uint64_t window_end = window_end_of(anchor);
    return window_end <= soc.dram_view_size && window_end > window_begin && window_end - window_begin >= bytes;
}

// First anchor of the view whose window holds `bytes`. Anchors that cannot hold them are logged and passed over.
const MimirAnchor* first_anchor_holding(
    const metal_SocDescriptor& soc, const std::vector<MimirAnchor>& anchors, uint64_t unreserved, uint64_t bytes) {
    for (const MimirAnchor& anchor : anchors) {
        if (window_holds_bytes(soc, anchor, unreserved, bytes)) {
            return &anchor;
        }
        log_info(
            tt::LogTest,
            "Skipping anchor {} of view {}: window cannot hold {} bytes.",
            anchor.name,
            anchor.view,
            bytes);
    }
    return nullptr;
}

// Slots pack upward from the window start. For "below_" anchors they pack downward from the window end, so the
// last slot's guard-after ends exactly at the window end. Hart 7 has the 4 KiB region, so that guard ends there.
uint64_t slot_block_begin(const MimirAnchor& anchor, uint64_t unreserved, uint32_t slot, uint32_t slot_count) {
    if (anchor.name.starts_with("below_")) {
        return window_end_of(anchor) - (slot_count - slot) * kSlotStrideBytes;
    }
    return window_begin_of(anchor, unreserved) + slot * kSlotStrideBytes;
}

// View offset of the region in slot `slot` of `slot_count` slots; the slot's guard before starts at its block.
uint64_t slot_data_offset(const MimirAnchor& anchor, uint64_t unreserved, uint32_t slot, uint32_t slot_count) {
    return slot_block_begin(anchor, unreserved, slot, slot_count) + kGuardBytes;
}

uint32_t region_bytes_for_hart(uint32_t hart) { return 512u << (hart % 4); }

// AddressDerived and Prng carry the test data: a constant pattern cannot catch a word-address error.
MimirPattern pattern_for(uint32_t index) {
    constexpr std::array<MimirPattern, 2> kDataPatterns = {MimirPattern::AddressDerived, MimirPattern::Prng};
    return kDataPatterns[index % kDataPatterns.size()];
}

// make_pattern with the Mimir and view folded in. AddressDerived words and the Prng seed take only the low 32 bits of
// the address, so without this a read that drops bit 32 (the other view) or goes to the other Mimir finds the same
// words there.
std::vector<uint32_t> make_located_pattern(
    MimirPattern pattern, size_t num_words, uint64_t base_address, uint32_t seed, uint32_t mimir_index, uint32_t view) {
    const uint32_t location_salt = 0x9E3779B9u * (((mimir_index << 8) | view) + 1);
    std::vector<uint32_t> words = make_pattern(pattern, num_words, base_address, seed ^ location_salt);
    if (pattern == MimirPattern::AddressDerived) {
        for (uint32_t& word : words) {
            word ^= location_salt;
        }
    }
    return words;
}

std::vector<uint32_t> poison_payload(size_t num_words) {
    return std::vector<uint32_t>(num_words, kDestinationPoisonWord);
}

// [guard | payload | guard] with guard tags 2n and 2n+1, so every guard is unique within a test.
std::vector<uint32_t> with_guards(uint32_t guard_index, const std::vector<uint32_t>& payload) {
    std::vector<uint32_t> block = make_guard(2u * guard_index);
    block.insert(block.end(), payload.begin(), payload.end());
    const std::vector<uint32_t> guard_after = make_guard(2u * guard_index + 1u);
    block.insert(block.end(), guard_after.begin(), guard_after.end());
    return block;
}

void write_guarded_dram(
    distributed::MeshDevice& unit_mesh,
    uint32_t view,
    uint64_t data_offset,
    uint32_t guard_index,
    const std::vector<uint32_t>& payload) {
    mimir_contract::write_gddr_view(unit_mesh, view, data_offset - kGuardBytes, with_guards(guard_index, payload));
}

void read_guarded_dram(
    distributed::MeshDevice& unit_mesh,
    uint32_t view,
    uint64_t data_offset,
    uint32_t payload_bytes,
    std::vector<uint32_t>& block) {
    mimir_contract::read_gddr_view(unit_mesh, view, data_offset - kGuardBytes, payload_bytes + 2 * kGuardBytes, block);
}

std::string describe_location(
    IDevice* device,
    const metal_SocDescriptor& soc,
    uint32_t cce_index,
    uint32_t hart,
    uint32_t view,
    const MimirAnchor& anchor,
    uint64_t data_offset,
    uint32_t size_bytes) {
    return fmt::format(
        "cce {} hart {} {} {}",
        cce_index,
        hart,
        anchor.name,
        mimir_dram_address(soc, device->id(), view, data_offset, size_bytes).host_string());
}

uint64_t shared_destination_tl1() { return tl1_data(0) + kSharedDestinationOffsetFromData; }

// Hart-private TL1 source for the shared-destination case. It sits above the disjoint-case source at tl1_data(hart),
// which later alignments reuse.
uint64_t shared_source_tl1(uint32_t hart) { return tl1_data(hart) + kEmuPageBytes; }

// The shared block must hold, at every word, the value of some hart's source: a lost or misplaced write matches none.
// Each word is attributed to the first hart that matches it, so the block can mix harts' words word by word.
::testing::AssertionResult shared_block_matches(
    const std::string& where,
    const std::vector<std::vector<uint32_t>>& hart_payloads,
    uint32_t num_harts,
    const std::vector<uint32_t>& block) {
    std::vector<uint32_t> picked = hart_payloads[0];
    for (size_t index = 0; index < picked.size(); ++index) {
        const size_t block_index = kGuardWords + index;
        if (block_index >= block.size()) {
            break;
        }
        for (uint32_t hart = 0; hart < num_harts; ++hart) {
            if (hart_payloads[hart][index] == block[block_index]) {
                picked[index] = hart_payloads[hart][index];
                break;
            }
        }
    }
    return compare_words(where, with_guards(0, picked), block);
}

// Disjoint blocks of size_bytes + both guards, placed back to back from first_block_begin, one per hart.
// Returns 0 when the first block does not fit below window_end. A 32 KiB region runs on hart 0 only.
uint32_t hart_capacity(uint32_t size_bytes, uint64_t first_block_begin, uint64_t window_end) {
    const uint64_t block_bytes = size_bytes + 2ull * kGuardBytes;
    if (window_end <= first_block_begin || window_end - first_block_begin < block_bytes) {
        return 0;
    }
    if (size_bytes >= kEmuPageBytes) {
        return 1;
    }
    const uint64_t blocks_that_fit = (window_end - first_block_begin) / block_bytes;
    return static_cast<uint32_t>(std::min<uint64_t>(cce_hart_count(), blocks_that_fit));
}

// One hart's cce_idma_copy kModeTridBatch launch between its own TL1 and its CCE's local GDDR (a chiplet offset), on
// trid `hart`, in commands of at most kMaxBytesInPacket. A GDDR destination is kernel-verified: after the ack the hart
// polls it against the source. host_verified leaves only the ack, for a block other harts write too. An own-TL1
// destination is always host-verified, and the hart polls the launch's poll region.
struct HartCopy {
    uint32_t hart = 0;
    bool reads_gddr = false;
    uint64_t tl1_offset = 0;
    uint64_t chiplet_offset = 0;
    uint32_t num_bytes = 0;
    bool host_verified = false;
};

// Every own-TL1 destination of one launch, as back-to-back guarded blocks. Each hart polls the XOR of the whole region
// before its status store, so no hart stores to this TL1 while another hart's iDMA may still be writing it (MIM-560).
struct PollRegion {
    uint64_t tl1_offset = 0;
    uint32_t num_bytes = 0;
    uint32_t expected_xor = 0;
};

// The poll region of back-to-back guarded blocks from `tl1_offset`, which must end up holding `final_blocks`.
PollRegion poll_region_over(uint64_t tl1_offset, const std::vector<std::vector<uint32_t>>& final_blocks) {
    PollRegion region{.tl1_offset = tl1_offset};
    for (const std::vector<uint32_t>& block : final_blocks) {
        region.num_bytes += static_cast<uint32_t>(block.size() * sizeof(uint32_t));
        region.expected_xor ^= xor_of(block);
    }
    return region;
}

uint32_t command_bytes(const HartCopy& copy) { return std::min(copy.num_bytes, abi::kMaxBytesInPacket); }

bool is_host_verified(const HartCopy& copy) { return copy.reads_gddr || copy.host_verified; }

// MIM-407: a copy from own TL1 uses the GDDR tile's backend (GDDR0 is the low 4 GiB of a Mimir, GDDR1 the high), any
// other copy kBackendFromRemote.
uint32_t backend_for(const HartCopy& copy) {
    if (copy.reads_gddr) {
        return abi::kBackendFromRemote;
    }
    return mimir_channel_location(copy.chiplet_offset).tile == 0 ? abi::kBackendGddr0 : abi::kBackendGddr1;
}

std::vector<uint32_t> copy_args(uint32_t cce_index, const HartCopy& copy, const PollRegion& poll_region) {
    const uint32_t len_bytes = command_bytes(copy);
    TT_FATAL(copy.num_bytes % len_bytes == 0, "{} B is not a whole number of {} B commands", copy.num_bytes, len_bytes);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t source_offset = copy.reads_gddr ? copy.chiplet_offset : copy.tl1_offset;
    const uint64_t destination_offset = copy.reads_gddr ? copy.tl1_offset : copy.chiplet_offset;
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = abi::kModeTridBatch;
    args[abi::kArgBackendVc] = backend_for(copy);
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(copy.hart));
    args[abi::kArgLenBytes] = len_bytes;
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgSrcKind] = copy.reads_gddr ? abi::kKindGddr : abi::kKindOwnTl1;
    args[abi::kArgSrcSelector] = copy.reads_gddr ? local_mimir : cce_index;
    args[abi::kArgSrcOffsetLo] = static_cast<uint32_t>(source_offset);
    args[abi::kArgSrcOffsetHi] = static_cast<uint32_t>(source_offset >> 32);
    args[abi::kArgDstKind] = copy.reads_gddr ? abi::kKindOwnTl1 : abi::kKindGddr;
    args[abi::kArgDstSelector] = copy.reads_gddr ? cce_index : local_mimir;
    args[abi::kArgDstOffsetLo] = static_cast<uint32_t>(destination_offset);
    args[abi::kArgDstOffsetHi] = static_cast<uint32_t>(destination_offset >> 32);
    args[abi::kArgExpectedXor] = poll_region.expected_xor;
    args[abi::kArgNumElements] = copy.num_bytes / len_bytes;
    args[abi::kArgSourceStrideBytes] = len_bytes;
    args[abi::kArgTrid] = copy.hart;
    args[abi::kArgDestinationStrideBytes] = len_bytes;
    args[abi::kArgHostVerified] = is_host_verified(copy) ? 1 : 0;
    args[abi::kArgPollRegionOffset] = static_cast<uint32_t>(poll_region.tl1_offset);
    args[abi::kArgPollRegionBytes] = poll_region.num_bytes;
    return args;
}

// Poisons every copy's status line, runs them all on `cce_index` in one Program and returns their status lines in copy
// order; nullopt on the host deadline.
std::optional<std::vector<std::vector<uint32_t>>> run_hart_copies(
    IDevice* device, uint32_t cce_index, const std::vector<HartCopy>& copies, const PollRegion& poll_region) {
    std::vector<CceKernelLaunch> launches;
    for (const HartCopy& copy : copies) {
        write_cce_tl1(
            device, cce_index, tl1_done_block(copy.hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
        launches.push_back(
            {.cce_index = cce_index,
             .kernel_path = kIdmaCopyKernel,
             .runtime_args = copy_args(cce_index, copy, poll_region),
             .hart = copy.hart});
    }
    Program program = CreateProgram();
    return run_cce_kernels_with_deadline(
        device,
        program,
        launches,
        std::vector<uint32_t>(launches.size(), abi::kDoneMagic),
        abi::kStatusWords,
        kHostDeadline);
}

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeLanded: return "landed";
        case abi::kCodeAcked: return "acked";
        case abi::kCodeVcSpaceTimeout: return "vc_space_timeout";
        case abi::kCodeAckTimeout: return "ack_timeout";
        case abi::kCodeDataTimeout: return "data_timeout";
        case abi::kCodeBadArgs: return "bad_args";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string poll_name(uint32_t poll) {
    switch (poll) {
        case abi::kPollNone: return "none";
        case abi::kPollVcSpace: return "vc_space";
        case abi::kPollAck: return "ack";
        case abi::kPollData: return "data";
    }
    return fmt::format("unknown_poll_{}", poll);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    const uint32_t flags = status[abi::kStatusFlags];
    return fmt::format(
        "kernel code {} flags [vc_space {} acked {} landed {}], {} commands issued, last poll {} {}, cycles vc_space "
        "{} ack {} data {}, pending high water {}, kernel mismatches {} (first word {} observed 0x{:08x} expected "
        "0x{:08x})",
        code_name(status[abi::kStatusCode]),
        (flags & abi::kFlagVcSpace) != 0,
        (flags & abi::kFlagAcked) != 0,
        (flags & abi::kFlagLanded) != 0,
        status[abi::kStatusElementsIssued],
        poll_name(status[abi::kStatusLastPoll]),
        status[abi::kStatusLastPollReady] != 0 ? "done" : "timed out",
        status[abi::kStatusVcSpaceCycles],
        status[abi::kStatusAckCycles],
        status[abi::kStatusDataCycles],
        status[abi::kStatusPendingHighWater],
        status[abi::kStatusMismatchedWords],
        status[abi::kStatusFirstMismatchWord],
        status[abi::kStatusObserved],
        status[abi::kStatusExpected]);
}

// A timeout code may leave a command in flight, so the next run needs a fresh session (MIM-601).
std::string code_note(uint32_t code) {
    return code == abi::kCodeVcSpaceTimeout || code == abi::kCodeAckTimeout || code == abi::kCodeDataTimeout
               ? kFreshSessionNote
               : "";
}

// Every copy of a launch ended with its success code after issuing every command. A pending high water of 0 is logged,
// not failed: the count can clear before the kernel reads it.
::testing::AssertionResult copies_completed(
    const std::vector<HartCopy>& copies, const std::vector<std::vector<uint32_t>>& statuses, const std::string& label) {
    std::string failures;
    for (size_t index = 0; index < copies.size(); ++index) {
        const HartCopy& copy = copies[index];
        const std::vector<uint32_t>& status = statuses[index];
        const uint32_t expected_code = is_host_verified(copy) ? abi::kCodeAcked : abi::kCodeLanded;
        const uint32_t expected_commands = copy.num_bytes / command_bytes(copy);
        if (status[abi::kStatusCode] != expected_code || status[abi::kStatusElementsIssued] != expected_commands) {
            failures += fmt::format(
                "{} hart {}: expected {} after {} commands; {}{}\n",
                label,
                copy.hart,
                code_name(expected_code),
                expected_commands,
                status_detail(status),
                code_note(status[abi::kStatusCode]));
        }
        if (status[abi::kStatusPendingHighWater] == 0) {
            log_info(tt::LogTest, "MM4 {} hart {}: pending high water 0", label, copy.hart);
        }
    }
    if (failures.empty()) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << failures;
}

// Word-level model of the copies, keyed by (is_dram, byte address). DRAM keys are chiplet offsets.
struct HostCopyModel {
    std::map<std::pair<bool, uint64_t>, uint32_t> words;

    void write(bool is_dram, uint64_t address, const std::vector<uint32_t>& data) {
        for (size_t index = 0; index < data.size(); ++index) {
            words[{is_dram, address + 4ull * index}] = data[index];
        }
    }

    std::vector<uint32_t> read(bool is_dram, uint64_t address, size_t num_words) const {
        std::vector<uint32_t> data(num_words);
        for (size_t index = 0; index < num_words; ++index) {
            data[index] = words.at({is_dram, address + 4ull * index});
        }
        return data;
    }

    void apply(const HartCopy& copy) {
        const uint64_t source = copy.reads_gddr ? copy.chiplet_offset : copy.tl1_offset;
        const uint64_t destination = copy.reads_gddr ? copy.tl1_offset : copy.chiplet_offset;
        for (uint32_t index = 0; index < copy.num_bytes / 4; ++index) {
            words[{!copy.reads_gddr, destination + 4ull * index}] = words.at({copy.reads_gddr, source + 4ull * index});
        }
    }
};

class MimirLocalDram : public ::testing::TestWithParam<uint32_t> {};

// CCE c writes its own TL1 to every local DRAM view, from all 8 harts, at every anchor.
TEST_P(MimirLocalDram, CceWritesTl1ToDram) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const std::string skip_reason = topology_skip_reason(cce_index);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice session;
    IDevice* device = session.device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
            const MimirAnchor& anchor = anchors[anchor_index];
            if (!window_holds_bytes(soc, anchor, unreserved, cce_hart_count() * kSlotStrideBytes)) {
                log_info(
                    tt::LogTest,
                    "Skipping {} of view {}: window cannot hold {} slots.",
                    anchor.name,
                    view,
                    cce_hart_count());
                continue;
            }

            std::vector<HartCopy> copies;
            std::vector<std::vector<uint32_t>> sources(cce_hart_count());
            std::vector<uint64_t> data_offsets(cce_hart_count());
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t data_offset = slot_data_offset(anchor, unreserved, hart, cce_hart_count());
                sources[hart] =
                    make_pattern(pattern_for(hart + anchor_index), region_words, tl1_data(hart), mimir_test_seed());
                write_cce_tl1(device, cce_index, tl1_data(hart), sources[hart]);
                write_guarded_dram(session.mesh(), view, data_offset, hart, poison_payload(region_words));
                data_offsets[hart] = data_offset;
                copies.push_back(HartCopy{
                    .hart = hart,
                    .tl1_offset = tl1_data(hart),
                    .chiplet_offset = chiplet_offset_of(soc, view, data_offset),
                    .num_bytes = region_bytes});
            }

            const std::string label = fmt::format("cce {} view {} {}", cce_index, view, anchor.name);
            const std::optional<std::vector<std::vector<uint32_t>>> statuses =
                run_hart_copies(device, cce_index, copies, PollRegion{});
            ASSERT_TRUE(statuses.has_value())
                << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string where = describe_location(
                    device, soc, cce_index, hart, view, anchor, data_offsets[hart], region_bytes_for_hart(hart));
                std::vector<uint32_t> block;
                read_guarded_dram(session.mesh(), view, data_offsets[hart], region_bytes_for_hart(hart), block);
                EXPECT_TRUE(compare_words(where, with_guards(hart, sources[hart]), block));
            }
            ASSERT_TRUE(copies_completed(copies, *statuses, label));
        }
    }
}

// Host writes DRAM, CCE copies it into its own TL1 at each hart's guarded block, host reads TL1 back. The blocks of
// all harts sit back to back in one poll region.
TEST_P(MimirLocalDram, CceReadsDramToTl1) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const std::string skip_reason = topology_skip_reason(cce_index);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice session;
    IDevice* device = session.device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
            const MimirAnchor& anchor = anchors[anchor_index];
            if (!window_holds_bytes(soc, anchor, unreserved, cce_hart_count() * kSlotStrideBytes)) {
                log_info(
                    tt::LogTest,
                    "Skipping {} of view {}: window cannot hold {} slots.",
                    anchor.name,
                    view,
                    cce_hart_count());
                continue;
            }

            std::vector<HartCopy> copies;
            std::vector<uint64_t> data_offsets(cce_hart_count());
            std::vector<uint64_t> block_offsets(cce_hart_count());
            std::vector<std::vector<uint32_t>> expected_blocks;
            uint64_t block_offset = shared_destination_tl1();
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t data_offset = slot_data_offset(anchor, unreserved, hart, cce_hart_count());
                const uint64_t chiplet_offset = chiplet_offset_of(soc, view, data_offset);
                data_offsets[hart] = data_offset;
                block_offsets[hart] = block_offset;
                const std::vector<uint32_t> source = make_located_pattern(
                    pattern_for(hart + anchor_index),
                    region_words,
                    chiplet_offset,
                    mimir_test_seed(),
                    local_mimir,
                    view);
                mimir_contract::write_gddr_view(session.mesh(), view, data_offset, source);
                // The TL1 destination is the payload of a guarded block.
                write_cce_tl1(device, cce_index, block_offset, with_guards(hart, poison_payload(region_words)));
                expected_blocks.push_back(with_guards(hart, source));
                copies.push_back(HartCopy{
                    .hart = hart,
                    .reads_gddr = true,
                    .tl1_offset = block_offset + kGuardBytes,
                    .chiplet_offset = chiplet_offset,
                    .num_bytes = region_bytes});
                block_offset += region_bytes + 2ull * kGuardBytes;
            }

            const std::string label = fmt::format("cce {} view {} {}", cce_index, view, anchor.name);
            const std::optional<std::vector<std::vector<uint32_t>>> statuses =
                run_hart_copies(device, cce_index, copies, poll_region_over(shared_destination_tl1(), expected_blocks));
            ASSERT_TRUE(statuses.has_value())
                << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string where = describe_location(
                    device, soc, cce_index, hart, view, anchor, data_offsets[hart], region_bytes_for_hart(hart));
                const std::vector<uint32_t> block = read_cce_tl1(
                    device,
                    cce_index,
                    block_offsets[hart],
                    static_cast<uint32_t>(expected_blocks[hart].size() * sizeof(uint32_t)));
                EXPECT_TRUE(compare_words(where, expected_blocks[hart], block));
            }
            ASSERT_TRUE(copies_completed(copies, *statuses, label));
        }
    }
}

// DRAM -> TL1 -> a different DRAM slot, per hart. The second slot of each hart sits 8 slots up in the same anchor.
// The two copies are separate launches (no kernel mode waits on one copy before issuing the next), so the host orders
// them and checks the intermediate TL1 in between; the CCE's own ordering is not tested.
TEST_P(MimirLocalDram, LoopbackChecksIntermediate) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const std::string skip_reason = topology_skip_reason(cce_index);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice session;
    IDevice* device = session.device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
            const MimirAnchor& anchor = anchors[anchor_index];
            if (!window_holds_bytes(soc, anchor, unreserved, 2 * cce_hart_count() * kSlotStrideBytes)) {
                log_info(
                    tt::LogTest,
                    "Skipping {} of view {}: window cannot hold {} slots.",
                    anchor.name,
                    view,
                    2 * cce_hart_count());
                continue;
            }

            std::vector<HartCopy> reads;
            std::vector<HartCopy> writes;
            std::vector<std::vector<uint32_t>> sources(cce_hart_count());
            std::vector<uint64_t> source_offsets(cce_hart_count());
            std::vector<uint64_t> destination_offsets(cce_hart_count());
            std::vector<uint64_t> block_offsets(cce_hart_count());
            std::vector<std::vector<uint32_t>> intermediate_blocks;
            uint64_t block_offset = shared_destination_tl1();
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t source_offset = slot_data_offset(anchor, unreserved, hart, 2 * cce_hart_count());
                const uint64_t destination_offset =
                    slot_data_offset(anchor, unreserved, cce_hart_count() + hart, 2 * cce_hart_count());
                const uint64_t source_chiplet = chiplet_offset_of(soc, view, source_offset);
                sources[hart] = make_located_pattern(
                    pattern_for(hart + anchor_index),
                    region_words,
                    source_chiplet,
                    mimir_test_seed(),
                    local_mimir,
                    view);
                source_offsets[hart] = source_offset;
                destination_offsets[hart] = destination_offset;
                block_offsets[hart] = block_offset;
                mimir_contract::write_gddr_view(session.mesh(), view, source_offset, sources[hart]);
                write_guarded_dram(
                    session.mesh(), view, destination_offset, cce_hart_count() + hart, poison_payload(region_words));
                write_cce_tl1(device, cce_index, block_offset, with_guards(hart, poison_payload(region_words)));
                intermediate_blocks.push_back(with_guards(hart, sources[hart]));

                reads.push_back(HartCopy{
                    .hart = hart,
                    .reads_gddr = true,
                    .tl1_offset = block_offset + kGuardBytes,
                    .chiplet_offset = source_chiplet,
                    .num_bytes = region_bytes});
                writes.push_back(HartCopy{
                    .hart = hart,
                    .tl1_offset = block_offset + kGuardBytes,
                    .chiplet_offset = chiplet_offset_of(soc, view, destination_offset),
                    .num_bytes = region_bytes});
                block_offset += region_bytes + 2ull * kGuardBytes;
            }

            const std::string label = fmt::format("cce {} view {} {}", cce_index, view, anchor.name);
            const std::optional<std::vector<std::vector<uint32_t>>> read_statuses = run_hart_copies(
                device, cce_index, reads, poll_region_over(shared_destination_tl1(), intermediate_blocks));
            ASSERT_TRUE(read_statuses.has_value())
                << label << " read: no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string source_where = describe_location(
                    device, soc, cce_index, hart, view, anchor, source_offsets[hart], region_bytes_for_hart(hart));
                const std::vector<uint32_t> intermediate = read_cce_tl1(
                    device,
                    cce_index,
                    block_offsets[hart],
                    static_cast<uint32_t>(intermediate_blocks[hart].size() * sizeof(uint32_t)));
                EXPECT_TRUE(compare_words(source_where + " intermediate TL1", intermediate_blocks[hart], intermediate));
            }
            ASSERT_TRUE(copies_completed(reads, *read_statuses, label + " read"));

            const std::optional<std::vector<std::vector<uint32_t>>> write_statuses =
                run_hart_copies(device, cce_index, writes, PollRegion{});
            ASSERT_TRUE(write_statuses.has_value())
                << label << " write: no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const std::string source_where =
                    describe_location(device, soc, cce_index, hart, view, anchor, source_offsets[hart], region_bytes);
                const std::string destination_where = describe_location(
                    device, soc, cce_index, hart, view, anchor, destination_offsets[hart], region_bytes);

                std::vector<uint32_t> source_readback;
                mimir_contract::read_gddr_view(
                    session.mesh(), view, source_offsets[hart], region_bytes, source_readback);
                EXPECT_TRUE(compare_words(source_where + " source", sources[hart], source_readback));

                const std::vector<uint32_t> intermediate = read_cce_tl1(
                    device,
                    cce_index,
                    block_offsets[hart],
                    static_cast<uint32_t>(intermediate_blocks[hart].size() * sizeof(uint32_t)));
                EXPECT_TRUE(compare_words(source_where + " intermediate TL1", intermediate_blocks[hart], intermediate));

                std::vector<uint32_t> destination_block;
                read_guarded_dram(session.mesh(), view, destination_offsets[hart], region_bytes, destination_block);
                EXPECT_TRUE(compare_words(
                    destination_where + " destination",
                    with_guards(cce_hart_count() + hart, sources[hart]),
                    destination_block));
            }
            ASSERT_TRUE(copies_completed(writes, *write_statuses, label + " write"));
        }
    }
}

// Every hart copies a region at each size and alignment in the matrix. Destinations are disjoint blocks, one per hart,
// and then shared: all harts write the same block, each from its own TL1 source. A region above kMaxBytesInPacket is
// several commands of that size.
TEST_P(MimirLocalDram, SizeAlignmentMatrix) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const std::string skip_reason = topology_skip_reason(cce_index);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice session;
    IDevice* device = session.device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    // kEmuPageBytes (32 KiB) is assumed to be the page size for the page-aligned offsets below. Page alignment is in
    // chiplet offset space, so the emulator may map these addresses to different pages.
    const std::vector<uint64_t> alignments = {0, 4, 60, 124, kEmuPageBytes - 4};

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        // The matrix reaches about 99 KiB past the window start (page slack, alignment, 8 blocks), so the anchor must
        // be a full window.
        const MimirAnchor* fitting_anchor = first_anchor_holding(soc, anchors, unreserved, kAnchorBytes);
        ASSERT_NE(fitting_anchor, nullptr) << "view " << view;
        const MimirAnchor& anchor = *fitting_anchor;
        const uint64_t window_begin = window_begin_of(anchor, unreserved);
        const uint64_t window_end = window_end_of(anchor);
        // The guard before the first destination must stay inside the window.
        const uint64_t page_base = tt::align(window_begin + kGuardBytes, kEmuPageBytes);

        for (uint32_t size_bytes = 4; size_bytes <= kEmuPageBytes; size_bytes *= 2) {
            const uint32_t size_words = size_bytes / 4;
            const uint32_t source_harts = hart_capacity(size_bytes, page_base - kGuardBytes, window_end);
            ASSERT_GE(source_harts, 1u) << "size " << size_bytes;
            std::vector<std::vector<uint32_t>> sources(source_harts);
            for (uint32_t hart = 0; hart < source_harts; ++hart) {
                sources[hart] = make_pattern(pattern_for(hart), size_words, tl1_data(hart), mimir_test_seed());
                write_cce_tl1(device, cce_index, tl1_data(hart), sources[hart]);
            }
            std::vector<std::vector<uint32_t>> shared_sources(source_harts);
            if (source_harts >= 2) {
                for (uint32_t hart = 0; hart < source_harts; ++hart) {
                    shared_sources[hart] = make_seed_pattern(hart, size_words);
                    write_cce_tl1(device, cce_index, shared_source_tl1(hart), shared_sources[hart]);
                }
            }

            for (uint64_t alignment : alignments) {
                const uint64_t first_data_offset = page_base + alignment;
                const uint32_t harts = hart_capacity(size_bytes, first_data_offset - kGuardBytes, window_end);
                const std::string case_label = fmt::format("size {} alignment {}", size_bytes, alignment);
                ASSERT_GE(harts, 1u) << case_label;

                std::vector<HartCopy> copies;
                std::vector<uint64_t> data_offsets(harts);
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    data_offsets[hart] = first_data_offset + hart * (size_bytes + 2ull * kGuardBytes);
                    write_guarded_dram(session.mesh(), view, data_offsets[hart], hart, poison_payload(size_words));
                    copies.push_back(HartCopy{
                        .hart = hart,
                        .tl1_offset = tl1_data(hart),
                        .chiplet_offset = chiplet_offset_of(soc, view, data_offsets[hart]),
                        .num_bytes = size_bytes});
                }
                const std::string label = fmt::format("cce {} view {} disjoint {}", cce_index, view, case_label);
                const std::optional<std::vector<std::vector<uint32_t>>> statuses =
                    run_hart_copies(device, cce_index, copies, PollRegion{});
                ASSERT_TRUE(statuses.has_value())
                    << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    const std::string where =
                        describe_location(device, soc, cce_index, hart, view, anchor, data_offsets[hart], size_bytes) +
                        " disjoint " + case_label;
                    std::vector<uint32_t> block;
                    read_guarded_dram(session.mesh(), view, data_offsets[hart], size_bytes, block);
                    EXPECT_TRUE(compare_words(where, with_guards(hart, sources[hart]), block));
                }
                ASSERT_TRUE(copies_completed(copies, *statuses, label));

                if (harts < 2) {
                    continue;
                }
                write_guarded_dram(session.mesh(), view, first_data_offset, 0, poison_payload(size_words));
                const uint64_t shared_chiplet_offset = chiplet_offset_of(soc, view, first_data_offset);
                // Host-verified: the other harts overwrite the block, so a hart's own data poll could time out.
                std::vector<HartCopy> shared_copies;
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    shared_copies.push_back(HartCopy{
                        .hart = hart,
                        .tl1_offset = shared_source_tl1(hart),
                        .chiplet_offset = shared_chiplet_offset,
                        .num_bytes = size_bytes,
                        .host_verified = true});
                }
                const std::string shared_label = fmt::format("cce {} view {} shared {}", cce_index, view, case_label);
                const std::optional<std::vector<std::vector<uint32_t>>> shared_statuses =
                    run_hart_copies(device, cce_index, shared_copies, PollRegion{});
                ASSERT_TRUE(shared_statuses.has_value())
                    << shared_label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
                std::vector<uint32_t> shared_block;
                read_guarded_dram(session.mesh(), view, first_data_offset, size_bytes, shared_block);
                EXPECT_TRUE(shared_block_matches(
                    describe_location(device, soc, cce_index, 0, view, anchor, first_data_offset, size_bytes) +
                        " shared " + case_label,
                    shared_sources,
                    harts,
                    shared_block));
                ASSERT_TRUE(copies_completed(shared_copies, *shared_statuses, shared_label));
            }
        }
    }
}

// Every hart runs the same overlapping ops, each in its own DRAM span and TL1 blocks, three repeats. Op 0 writes A to
// DRAM [R, R+4 KiB), op 1 writes B to [R+2 KiB, R+6 KiB) over it, and ops 2 and 3 read [R, R+4 KiB) and
// [R+2 KiB, R+6 KiB) back into TL1 C and D. Each op is one launch of all harts, so the ops run in program order and
// every op's destination is checked against the host model before the next op.
TEST_P(MimirLocalDram, RepeatedAndOverlappingPerHart) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const std::string skip_reason = topology_skip_reason(cce_index);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice session;
    IDevice* device = session.device();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t local_mimir = mimir_index_for_cce(cce_index);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    constexpr uint32_t kCopyWords = kCopyTestBytes / 4;
    constexpr uint64_t kOverlapShiftBytes = kCopyTestBytes / 2;
    constexpr uint64_t kSpanBytes = kCopyTestBytes + kOverlapShiftBytes;
    constexpr uint32_t kSpanWords = kSpanBytes / 4;
    constexpr uint64_t kHartStrideBytes = kSpanBytes + 2 * kGuardBytes;
    constexpr uint64_t kResultBlockBytes = kCopyTestBytes + 2 * kGuardBytes;
    // The C blocks of every hart, then the D blocks, each set one poll region.
    const uint64_t result_c_region = shared_destination_tl1();
    const uint64_t result_d_region = result_c_region + cce_hart_count() * kResultBlockBytes;

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        const MimirAnchor* fitting_anchor =
            first_anchor_holding(soc, anchors, unreserved, cce_hart_count() * kHartStrideBytes);
        ASSERT_NE(fitting_anchor, nullptr) << "view " << view;
        const MimirAnchor& anchor = *fitting_anchor;

        for (uint32_t repeat = 0; repeat < 3; ++repeat) {
            std::vector<uint64_t> region_offsets(cce_hart_count());
            std::vector<uint64_t> region_chiplets(cce_hart_count());
            std::vector<HostCopyModel> models(cce_hart_count());
            std::array<std::vector<HartCopy>, 4> ops;
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint64_t source_a_tl1 = tl1_data(hart);
                const uint64_t source_b_tl1 = tl1_data(hart) + kCopyTestBytes;
                const uint64_t result_c_tl1 = result_c_region + hart * kResultBlockBytes + kGuardBytes;
                const uint64_t result_d_tl1 = result_d_region + hart * kResultBlockBytes + kGuardBytes;
                region_offsets[hart] = window_begin_of(anchor, unreserved) + hart * kHartStrideBytes + kGuardBytes;
                region_chiplets[hart] = chiplet_offset_of(soc, view, region_offsets[hart]);

                // Each repeat has its own Prng seed, so the data differs between repeats.
                const std::vector<uint32_t> source_a = make_located_pattern(
                    pattern_for(repeat + hart),
                    kCopyWords,
                    source_a_tl1,
                    mimir_test_seed() ^ repeat,
                    local_mimir,
                    view);
                const std::vector<uint32_t> source_b = make_located_pattern(
                    pattern_for(repeat + hart + 1),
                    kCopyWords,
                    source_b_tl1,
                    mimir_test_seed() ^ repeat,
                    local_mimir,
                    view);
                write_cce_tl1(device, cce_index, source_a_tl1, source_a);
                write_cce_tl1(device, cce_index, source_b_tl1, source_b);
                write_guarded_dram(session.mesh(), view, region_offsets[hart], 3 * hart, poison_payload(kSpanWords));
                write_cce_tl1(
                    device,
                    cce_index,
                    result_c_tl1 - kGuardBytes,
                    with_guards(3 * hart + 1, poison_payload(kCopyWords)));
                write_cce_tl1(
                    device,
                    cce_index,
                    result_d_tl1 - kGuardBytes,
                    with_guards(3 * hart + 2, poison_payload(kCopyWords)));

                models[hart].write(false, source_a_tl1, source_a);
                models[hart].write(false, source_b_tl1, source_b);
                models[hart].write(true, region_chiplets[hart], poison_payload(kSpanWords));
                models[hart].write(false, result_c_tl1, poison_payload(kCopyWords));
                models[hart].write(false, result_d_tl1, poison_payload(kCopyWords));

                ops[0].push_back(HartCopy{
                    .hart = hart,
                    .tl1_offset = source_a_tl1,
                    .chiplet_offset = region_chiplets[hart],
                    .num_bytes = kCopyTestBytes});
                ops[1].push_back(HartCopy{
                    .hart = hart,
                    .tl1_offset = source_b_tl1,
                    .chiplet_offset = region_chiplets[hart] + kOverlapShiftBytes,
                    .num_bytes = kCopyTestBytes});
                ops[2].push_back(HartCopy{
                    .hart = hart,
                    .reads_gddr = true,
                    .tl1_offset = result_c_tl1,
                    .chiplet_offset = region_chiplets[hart],
                    .num_bytes = kCopyTestBytes});
                ops[3].push_back(HartCopy{
                    .hart = hart,
                    .reads_gddr = true,
                    .tl1_offset = result_d_tl1,
                    .chiplet_offset = region_chiplets[hart] + kOverlapShiftBytes,
                    .num_bytes = kCopyTestBytes});
            }

            for (uint32_t op = 0; op < ops.size(); ++op) {
                const bool reads_gddr = ops[op].front().reads_gddr;
                // A write op's destination is the hart's guarded DRAM span; a read op's is its TL1 C block (guard
                // 3h+1) or D block (guard 3h+2).
                std::vector<std::vector<uint32_t>> expected_blocks;
                for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                    const HartCopy& copy = ops[op][hart];
                    models[hart].apply(copy);
                    expected_blocks.push_back(
                        reads_gddr
                            ? with_guards(
                                  3 * hart + (op == 2 ? 1 : 2), models[hart].read(false, copy.tl1_offset, kCopyWords))
                            : with_guards(3 * hart, models[hart].read(true, region_chiplets[hart], kSpanWords)));
                }
                const PollRegion poll_region =
                    reads_gddr ? poll_region_over(ops[op].front().tl1_offset - kGuardBytes, expected_blocks)
                               : PollRegion{};
                const std::string label =
                    fmt::format("cce {} view {} {} repeat {} op {}", cce_index, view, anchor.name, repeat, op);
                const std::optional<std::vector<std::vector<uint32_t>>> statuses =
                    run_hart_copies(device, cce_index, ops[op], poll_region);
                ASSERT_TRUE(statuses.has_value())
                    << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
                for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                    const std::string where =
                        describe_location(
                            device, soc, cce_index, hart, view, anchor, region_offsets[hart], kSpanBytes) +
                        fmt::format(" repeat {} op {}", repeat, op);
                    std::vector<uint32_t> block;
                    if (reads_gddr) {
                        block =
                            read_cce_tl1(device, cce_index, ops[op][hart].tl1_offset - kGuardBytes, kResultBlockBytes);
                    } else {
                        read_guarded_dram(session.mesh(), view, region_offsets[hart], kSpanBytes, block);
                    }
                    EXPECT_TRUE(compare_words(where, expected_blocks[hart], block));
                }
                ASSERT_TRUE(copies_completed(ops[op], *statuses, label));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM4, MimirLocalDram, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::local_dram
