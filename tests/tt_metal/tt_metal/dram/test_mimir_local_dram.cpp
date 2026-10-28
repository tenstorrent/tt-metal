// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

// The Glean MM-4 plan names existing DRAM tests to port: dram_unary, api/test_dram, api/test_direct, the DRAM
// loopback and single-DM L1-write tests, and dram_sharded. Their kernels reach DRAM through the NoC dataflow API
// (Noc::async_read/async_write with AllocatorBank<DRAM>), which no Mimir CCE kernel has run yet. So these tests
// use the experimental/cce_gddr.h path through cce_copy_ops.cpp until that is proven.

namespace tt::tt_metal::mimir_test::local_dram {

// Destination slots hold [guard | region | guard]. A slot is one 4 KiB region plus both guards, so slots
// never overlap and every slot offset is 64-byte aligned.
constexpr uint32_t kMaxSlotRegionBytes = 4096;
constexpr uint64_t kSlotStrideBytes = kMaxSlotRegionBytes + 2 * kGuardBytes;
constexpr uint32_t kDestinationPoisonWord = 0xDEADBEEFu;
constexpr uint64_t kCopyTestBytes = 4 * kKiB;

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

CceCopyLaunch single_op_launch(uint32_t cce_index, uint32_t hart, const CopyOp& op) {
    CopyOpTable table;
    table.add(op);
    return CceCopyLaunch{.cce_index = cce_index, .hart = hart, .table = table};
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

// Word-level model of the copy ops, keyed by (is_dram, byte address). DRAM keys are chiplet offsets.
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

    // Applies the ops in program order and returns the XOR of every source word read, as the kernel does.
    uint32_t apply(const std::vector<CopyOp>& ops) {
        uint32_t signature = 0;
        for (const CopyOp& op : ops) {
            for (uint32_t index = 0; index < op.num_words; ++index) {
                const uint32_t value = words.at({op.source.kind == CopyKind::Gddr, op.source.offset + 4ull * index});
                signature ^= value;
                words[{op.destination.kind == CopyKind::Gddr, op.destination.offset + 4ull * index}] = value;
            }
        }
        return signature;
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

            std::vector<CceCopyLaunch> launches;
            std::vector<std::vector<uint32_t>> sources(cce_hart_count());
            std::vector<uint64_t> data_offsets(cce_hart_count());
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t data_offset = slot_data_offset(anchor, unreserved, hart, cce_hart_count());
                const uint64_t chiplet_offset = chiplet_offset_of(soc, view, data_offset);
                sources[hart] =
                    make_pattern(pattern_for(hart + anchor_index), region_words, tl1_data(hart), mimir_test_seed());
                write_cce_tl1(device, cce_index, tl1_data(hart), sources[hart]);
                write_guarded_dram(session.mesh(), view, data_offset, hart, poison_payload(region_words));
                data_offsets[hart] = data_offset;
                launches.push_back(single_op_launch(
                    cce_index,
                    hart,
                    CopyOp{
                        .source = own_tl1(tl1_data(hart)),
                        .destination = gddr(local_mimir, chiplet_offset),
                        .num_words = region_words}));
            }

            std::vector<CceCopyResult> results;
            ASSERT_TRUE(run_cce_copy_ops(device, launches, results));
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string where = describe_location(
                    device, soc, cce_index, hart, view, anchor, data_offsets[hart], region_bytes_for_hart(hart));
                expect_copy_done(results[hart], xor_of(sources[hart]), where);
                std::vector<uint32_t> block;
                read_guarded_dram(session.mesh(), view, data_offsets[hart], region_bytes_for_hart(hart), block);
                EXPECT_TRUE(compare_words(where, with_guards(hart, sources[hart]), block));
            }
        }
    }
}

// Host writes DRAM, CCE copies it into its own TL1 at each hart's guarded block, host reads TL1 back.
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

            std::vector<CceCopyLaunch> launches;
            std::vector<std::vector<uint32_t>> sources(cce_hart_count());
            std::vector<uint64_t> data_offsets(cce_hart_count());
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t data_offset = slot_data_offset(anchor, unreserved, hart, cce_hart_count());
                const uint64_t chiplet_offset = chiplet_offset_of(soc, view, data_offset);
                data_offsets[hart] = data_offset;
                sources[hart] =
                    make_pattern(pattern_for(hart + anchor_index), region_words, chiplet_offset, mimir_test_seed());
                mimir_contract::write_gddr_view(session.mesh(), view, data_offset, sources[hart]);
                // The TL1 destination is the payload of a guarded block at tl1_data(hart).
                write_cce_tl1(device, cce_index, tl1_data(hart), with_guards(hart, poison_payload(region_words)));
                launches.push_back(single_op_launch(
                    cce_index,
                    hart,
                    CopyOp{
                        .source = gddr(local_mimir, chiplet_offset),
                        .destination = own_tl1(tl1_data(hart) + kGuardBytes),
                        .num_words = region_words}));
            }

            std::vector<CceCopyResult> results;
            ASSERT_TRUE(run_cce_copy_ops(device, launches, results));
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string where = describe_location(
                    device, soc, cce_index, hart, view, anchor, data_offsets[hart], region_bytes_for_hart(hart));
                expect_copy_done(results[hart], xor_of(sources[hart]), where);
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const std::vector<uint32_t> block =
                    read_cce_tl1(device, cce_index, tl1_data(hart), region_bytes + 2 * kGuardBytes);
                EXPECT_TRUE(compare_words(where, with_guards(hart, sources[hart]), block));
            }
        }
    }
}

// DRAM -> TL1 -> a different DRAM slot, per hart. The second slot of each hart sits 8 slots up in the same anchor.
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

            std::vector<CceCopyLaunch> launches;
            std::vector<std::vector<uint32_t>> sources(cce_hart_count());
            std::vector<uint64_t> source_offsets(cce_hart_count());
            std::vector<uint64_t> destination_offsets(cce_hart_count());
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const uint32_t region_words = region_bytes / 4;
                const uint64_t source_offset = slot_data_offset(anchor, unreserved, hart, 2 * cce_hart_count());
                const uint64_t destination_offset =
                    slot_data_offset(anchor, unreserved, cce_hart_count() + hart, 2 * cce_hart_count());
                const uint64_t source_chiplet = chiplet_offset_of(soc, view, source_offset);
                const uint64_t destination_chiplet = chiplet_offset_of(soc, view, destination_offset);
                sources[hart] =
                    make_pattern(pattern_for(hart + anchor_index), region_words, source_chiplet, mimir_test_seed());
                source_offsets[hart] = source_offset;
                destination_offsets[hart] = destination_offset;
                mimir_contract::write_gddr_view(session.mesh(), view, source_offset, sources[hart]);
                write_guarded_dram(
                    session.mesh(), view, destination_offset, cce_hart_count() + hart, poison_payload(region_words));
                write_cce_tl1(device, cce_index, tl1_data(hart), with_guards(hart, poison_payload(region_words)));

                CopyOpTable table;
                table.add(CopyOp{
                    .source = gddr(local_mimir, source_chiplet),
                    .destination = own_tl1(tl1_data(hart) + kGuardBytes),
                    .num_words = region_words});
                table.add(CopyOp{
                    .source = own_tl1(tl1_data(hart) + kGuardBytes),
                    .destination = gddr(local_mimir, destination_chiplet),
                    .num_words = region_words});
                launches.push_back(CceCopyLaunch{.cce_index = cce_index, .hart = hart, .table = table});
            }

            std::vector<CceCopyResult> results;
            ASSERT_TRUE(run_cce_copy_ops(device, launches, results));
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint32_t region_bytes = region_bytes_for_hart(hart);
                const std::string source_where =
                    describe_location(device, soc, cce_index, hart, view, anchor, source_offsets[hart], region_bytes);
                const std::string destination_where = describe_location(
                    device, soc, cce_index, hart, view, anchor, destination_offsets[hart], region_bytes);
                // Op 1 re-reads what op 0 wrote, so the XOR of its source words cancels to zero. It is not checked.
                EXPECT_TRUE(results[hart].done) << source_where;

                std::vector<uint32_t> source_readback;
                mimir_contract::read_gddr_view(
                    session.mesh(), view, source_offsets[hart], region_bytes, source_readback);
                EXPECT_TRUE(compare_words(source_where + " source", sources[hart], source_readback));

                const std::vector<uint32_t> intermediate =
                    read_cce_tl1(device, cce_index, tl1_data(hart), region_bytes + 2 * kGuardBytes);
                EXPECT_TRUE(
                    compare_words(source_where + " intermediate TL1", with_guards(hart, sources[hart]), intermediate));

                std::vector<uint32_t> destination_block;
                read_guarded_dram(session.mesh(), view, destination_offsets[hart], region_bytes, destination_block);
                EXPECT_TRUE(compare_words(
                    destination_where + " destination",
                    with_guards(cce_hart_count() + hart, sources[hart]),
                    destination_block));
            }
        }
    }
}

// Every hart copies a 4 KiB-or-smaller region at each size and alignment in the matrix. Destinations are
// disjoint blocks, one per hart, and then shared: all harts write the same block, each from its own TL1 source.
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

                std::vector<CceCopyLaunch> launches;
                std::vector<uint64_t> data_offsets(harts);
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    data_offsets[hart] = first_data_offset + hart * (size_bytes + 2ull * kGuardBytes);
                    const uint64_t chiplet_offset = chiplet_offset_of(soc, view, data_offsets[hart]);
                    write_guarded_dram(session.mesh(), view, data_offsets[hart], hart, poison_payload(size_words));
                    launches.push_back(single_op_launch(
                        cce_index,
                        hart,
                        CopyOp{
                            .source = own_tl1(tl1_data(hart)),
                            .destination = gddr(local_mimir, chiplet_offset),
                            .num_words = size_words}));
                }
                std::vector<CceCopyResult> results;
                ASSERT_TRUE(run_cce_copy_ops(device, launches, results));
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    const std::string where =
                        describe_location(device, soc, cce_index, hart, view, anchor, data_offsets[hart], size_bytes) +
                        " disjoint " + case_label;
                    expect_copy_done(results[hart], xor_of(sources[hart]), where);
                    std::vector<uint32_t> block;
                    read_guarded_dram(session.mesh(), view, data_offsets[hart], size_bytes, block);
                    EXPECT_TRUE(compare_words(where, with_guards(hart, sources[hart]), block));
                }

                if (harts < 2) {
                    continue;
                }
                write_guarded_dram(session.mesh(), view, first_data_offset, 0, poison_payload(size_words));
                const uint64_t shared_chiplet_offset = chiplet_offset_of(soc, view, first_data_offset);
                std::vector<CceCopyLaunch> shared_launches;
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    shared_launches.push_back(single_op_launch(
                        cce_index,
                        hart,
                        CopyOp{
                            .source = own_tl1(shared_source_tl1(hart)),
                            .destination = gddr(local_mimir, shared_chiplet_offset),
                            .num_words = size_words}));
                }
                std::vector<CceCopyResult> shared_results;
                ASSERT_TRUE(run_cce_copy_ops(device, shared_launches, shared_results));
                for (uint32_t hart = 0; hart < harts; ++hart) {
                    const std::string where =
                        describe_location(device, soc, cce_index, hart, view, anchor, first_data_offset, size_bytes) +
                        " shared " + case_label;
                    expect_copy_done(shared_results[hart], xor_of(shared_sources[hart]), where);
                }
                std::vector<uint32_t> shared_block;
                read_guarded_dram(session.mesh(), view, first_data_offset, size_bytes, shared_block);
                EXPECT_TRUE(shared_block_matches(
                    describe_location(device, soc, cce_index, 0, view, anchor, first_data_offset, size_bytes) +
                        " shared " + case_label,
                    shared_sources,
                    harts,
                    shared_block));
            }
        }
    }
}

// Every hart runs the same overlapping op table at once, each in its own DRAM span and TL1 regions, three repeats.
// In a hart's op table, op 1 writes DRAM [R+2 KiB, R+6 KiB) over op 0's [R, R+4 KiB). Ops 2 and 3 read back what the
// earlier ops left in DRAM. Overlaps are within one hart only, so the expected result is the host model in that hart's
// program order.
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

    for (uint32_t view : dram_views_for_channel(soc, local_mimir)) {
        const std::vector<MimirAnchor> anchors =
            mimir_dram_anchors(soc, view, unreserved, /*include_reserved_start=*/false);
        const MimirAnchor* fitting_anchor =
            first_anchor_holding(soc, anchors, unreserved, cce_hart_count() * kHartStrideBytes);
        ASSERT_NE(fitting_anchor, nullptr) << "view " << view;
        const MimirAnchor& anchor = *fitting_anchor;

        for (uint32_t repeat = 0; repeat < 3; ++repeat) {
            std::vector<CceCopyLaunch> launches;
            std::vector<uint64_t> region_offsets(cce_hart_count());
            std::vector<uint32_t> expected_signatures(cce_hart_count());
            std::vector<std::vector<uint32_t>> expected_drams(cce_hart_count());
            std::vector<std::vector<uint32_t>> expected_cs(cce_hart_count());
            std::vector<std::vector<uint32_t>> expected_ds(cce_hart_count());
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const uint64_t source_a_tl1 = tl1_data(hart);
                const uint64_t source_b_tl1 = tl1_data(hart) + kCopyTestBytes;
                const uint64_t result_c_tl1 = tl1_data(hart) + 16 * kKiB;
                const uint64_t result_d_tl1 = tl1_data(hart) + 32 * kKiB;
                region_offsets[hart] = window_begin_of(anchor, unreserved) + hart * kHartStrideBytes + kGuardBytes;
                const uint64_t region_chiplet = chiplet_offset_of(soc, view, region_offsets[hart]);

                // Each repeat has its own Prng seed, so the data differs between repeats.
                const std::vector<uint32_t> source_a =
                    make_pattern(pattern_for(repeat + hart), kCopyWords, source_a_tl1, mimir_test_seed() ^ repeat);
                const std::vector<uint32_t> source_b =
                    make_pattern(pattern_for(repeat + hart + 1), kCopyWords, source_b_tl1, mimir_test_seed() ^ repeat);
                write_cce_tl1(device, cce_index, source_a_tl1, source_a);
                write_cce_tl1(device, cce_index, source_b_tl1, source_b);
                write_guarded_dram(session.mesh(), view, region_offsets[hart], 3 * hart, poison_payload(kSpanWords));
                write_cce_tl1(device, cce_index, result_c_tl1, with_guards(3 * hart + 1, poison_payload(kCopyWords)));
                write_cce_tl1(device, cce_index, result_d_tl1, with_guards(3 * hart + 2, poison_payload(kCopyWords)));

                const std::vector<CopyOp> ops = {
                    CopyOp{
                        .source = own_tl1(source_a_tl1),
                        .destination = gddr(local_mimir, region_chiplet),
                        .num_words = kCopyWords},
                    CopyOp{
                        .source = own_tl1(source_b_tl1),
                        .destination = gddr(local_mimir, region_chiplet + kOverlapShiftBytes),
                        .num_words = kCopyWords},
                    CopyOp{
                        .source = gddr(local_mimir, region_chiplet),
                        .destination = own_tl1(result_c_tl1 + kGuardBytes),
                        .num_words = kCopyWords},
                    CopyOp{
                        .source = gddr(local_mimir, region_chiplet + kOverlapShiftBytes),
                        .destination = own_tl1(result_d_tl1 + kGuardBytes),
                        .num_words = kCopyWords}};

                HostCopyModel model;
                model.write(false, source_a_tl1, source_a);
                model.write(false, source_b_tl1, source_b);
                expected_signatures[hart] = model.apply(ops);
                expected_drams[hart] = model.read(true, region_chiplet, kSpanWords);
                expected_cs[hart] = model.read(false, result_c_tl1 + kGuardBytes, kCopyWords);
                expected_ds[hart] = model.read(false, result_d_tl1 + kGuardBytes, kCopyWords);

                CceCopyLaunch launch{.cce_index = cce_index, .hart = hart, .table = CopyOpTable{}};
                for (const CopyOp& op : ops) {
                    launch.table.add(op);
                }
                launches.push_back(launch);
            }

            std::vector<CceCopyResult> results;
            ASSERT_TRUE(run_cce_copy_ops(device, launches, results));
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                const std::string where =
                    describe_location(device, soc, cce_index, hart, view, anchor, region_offsets[hart], kSpanBytes) +
                    fmt::format(" repeat {}", repeat);
                expect_copy_done(results[hart], expected_signatures[hart], where);

                std::vector<uint32_t> dram_block;
                read_guarded_dram(session.mesh(), view, region_offsets[hart], kSpanBytes, dram_block);
                EXPECT_TRUE(compare_words(where + " DRAM", with_guards(3 * hart, expected_drams[hart]), dram_block));
                const std::vector<uint32_t> c_block =
                    read_cce_tl1(device, cce_index, tl1_data(hart) + 16 * kKiB, kCopyTestBytes + 2 * kGuardBytes);
                EXPECT_TRUE(compare_words(where + " TL1 C", with_guards(3 * hart + 1, expected_cs[hart]), c_block));
                const std::vector<uint32_t> d_block =
                    read_cce_tl1(device, cce_index, tl1_data(hart) + 32 * kKiB, kCopyTestBytes + 2 * kGuardBytes);
                EXPECT_TRUE(compare_words(where + " TL1 D", with_guards(3 * hart + 2, expected_ds[hart]), d_block));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM4, MimirLocalDram, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::local_dram
