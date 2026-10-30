// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_remote_test.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::tl1 {

// The Glean MM-5 plan names the DM-suite tests (one_to_one, core_bidirectional, ...) for reuse. Their host side is
// bound to worker cores and fast dispatch, and their kernels use the NoC dataflow API, which no Quasar CCE kernel
// has run. So these tests reach TL1 from CCE kernels through experimental::cce_sram_address loads and stores.

// Read-back destination: 32 KiB into the hart's data area, so it never overlaps the source data.
inline constexpr uint64_t kReadbackOffset = 0x8000;
inline constexpr uint64_t kControlSourceOffset = 0x4000;
inline constexpr uint64_t kSiblingSourceOffset = 0x2000;
inline constexpr uint64_t kSiblingReadOffset = 0xC000;
inline constexpr uint32_t kLocalBytesPerHart = 256;
static_assert(kSiblingReadOffset + kLocalBytesPerHart <= kTl1HartDataBytes);
// Per-hart sizes for the write and read-back pass. Every size is at most 4 KiB, so the hart's data area and its
// 16 KiB landing slot both hold it with room for guards.
inline constexpr std::array<uint32_t, 8> kWriteReadSizesBytes = {64, 128, 256, 512, 1024, 2048, 3072, 4096};
// Offsets 0, 4, 60 and 124 match SizeAlignmentMatrix in test_mimir_local_dram.cpp; 12 adds the 16 B edge. Sizes are
// capped at 4 KiB to fit a landing slot.
inline constexpr std::array<uint32_t, 7> kSizeAlignmentSizesBytes = {4, 16, 60, 64, 128, 1024, 4096};
inline constexpr std::array<uint32_t, 5> kSizeAlignmentOffsetsBytes = {0, 4, 12, 60, 124};
static_assert(
    kReadbackOffset + kSizeAlignmentOffsetsBytes.back() + kSizeAlignmentSizesBytes.back() <= kTl1HartDataBytes);

struct LocalPair {
    uint32_t source_cce = 0;
    uint32_t destination_cce = 0;
};

struct RemotePair {
    uint32_t source_mimir = 0;
    uint32_t source_location = 0;
    uint32_t destination_mimir = 0;
    uint32_t destination_location = 0;

    uint32_t source_cce() const { return cce_index_for(source_mimir, source_location); }
    uint32_t destination_cce() const { return cce_index_for(destination_mimir, destination_location); }
};

struct RemoteTransfer {
    uint32_t source_cce = 0;
    uint32_t hart = 0;
    uint32_t destination_cce = 0;
    uint32_t size_bytes = 0;
    uint32_t alignment = 0;
    uint32_t salt = 0;
    MimirPattern pattern = MimirPattern::Prng;
};

struct PreparedTransfer {
    RemoteTransfer transfer;
    uint64_t source_offset = 0;
    uint64_t landing_offset = 0;
    uint64_t readback_offset = 0;
    uint32_t tag = 0;
    std::vector<uint32_t> expected;
    MimirTransferRecord write_record;
    MimirTransferRecord read_record;
};

std::string remote_pair_label(const RemotePair& pair) {
    return fmt::format(
        "M{}Cce{}ToM{}Cce{}",
        pair.source_mimir,
        pair.source_location,
        pair.destination_mimir,
        pair.destination_location);
}

std::string local_pair_name(const ::testing::TestParamInfo<LocalPair>& info) {
    return fmt::format("Cce{}To{}", info.param.source_cce, info.param.destination_cce);
}

std::string remote_pair_name(const ::testing::TestParamInfo<RemotePair>& info) { return remote_pair_label(info.param); }

std::vector<LocalPair> same_chiplet_pairs() {
    if (mimir_contract::selected_soc_desc_path() == nullptr) {
        return {{0, 0}};
    }
    std::vector<LocalPair> pairs;
    for (uint32_t cce_index : mimir_contract::cce_indices()) {
        pairs.push_back({cce_index, sibling_cce(cce_index)});
    }
    return pairs;
}

std::vector<RemotePair> direct_remote_pairs() { return {{0, 1, 1, 0}, {1, 0, 0, 1}}; }

std::vector<RemotePair> routed_remote_pairs() {
    return {{0, 0, 1, 0}, {0, 1, 1, 1}, {1, 0, 0, 0}, {1, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 0, 0}};
}

std::string local_skip_reason(const LocalPair& pair) {
    const std::string watcher_reason = mimir_watcher_skip_reason();
    if (!watcher_reason.empty()) {
        return watcher_reason;
    }
    if (pair.source_cce == pair.destination_cce) {
        return "Descriptor has one CCE per Mimir; local TL1 needs two.";
    }
    return "";
}

std::string remote_skip_reason(const RemotePair& pair) {
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_mimirs < 2) {
        return "Remote TL1 tests need a descriptor with two Mimirs.";
    }
    const uint32_t cces_per_mimir = locations_per_channel(configured_soc_desc());
    const uint32_t highest_location = std::max(pair.source_location, pair.destination_location);
    if (highest_location >= cces_per_mimir) {
        return fmt::format(
            "Descriptor has {} CCEs per Mimir; this pair needs location {}.", cces_per_mimir, highest_location);
    }
    return "";
}

uint32_t transfer_seed(uint32_t source_cce, uint32_t hart, uint32_t salt) {
    return mimir_test_seed() ^ ((source_cce * cce_hart_count() + hart + 1) * 0x9E3779B9u) ^ (salt * 0x85EBCA6Bu);
}

uint32_t guard_tag(uint32_t source_cce, uint32_t hart) { return source_cce * cce_hart_count() + hart + 1; }

std::string endpoint_name(uint32_t cce_index, uint64_t tl1_offset) {
    return fmt::format("M{}.CCE{} TL1 0x{:x}", mimir_index_for_cce(cce_index), location_for_cce(cce_index), tl1_offset);
}

void write_guard(IDevice* device, uint32_t cce_index, uint64_t tl1_offset, uint32_t tag) {
    write_cce_tl1(device, cce_index, tl1_offset, make_guard(tag));
}

void write_guards_except(IDevice* device, uint32_t num_cces, uint32_t excluded_cce, uint64_t tl1_offset, uint32_t tag) {
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (cce_index != excluded_cce) {
            write_guard(device, cce_index, tl1_offset, tag);
        }
    }
}

void expect_guard_intact(
    IDevice* device, uint32_t cce_index, uint64_t tl1_offset, uint32_t tag, const std::string& what) {
    EXPECT_TRUE(compare_words(
        fmt::format("{}: guard in CCE {} TL1 0x{:x}", what, cce_index, tl1_offset),
        make_guard(tag),
        read_cce_tl1(device, cce_index, tl1_offset, kGuardBytes)));
}

void expect_guards_intact_except(
    IDevice* device,
    uint32_t num_cces,
    uint32_t excluded_cce,
    uint64_t tl1_offset,
    uint32_t tag,
    const std::string& what) {
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (cce_index != excluded_cce) {
            expect_guard_intact(device, cce_index, tl1_offset, tag, what);
        }
    }
}

std::vector<MimirRegister> registers_on_mimir(const std::vector<MimirRegister>& registers, uint32_t mimir_index) {
    std::vector<MimirRegister> filtered;
    for (const MimirRegister& reg : registers) {
        if (reg.mimir_index == mimir_index) {
            filtered.push_back(reg);
        }
    }
    return filtered;
}

std::vector<std::string> debug_last_changes(
    const std::vector<MimirRegister>& before, const std::vector<MimirRegister>& after) {
    std::vector<std::string> changes;
    if (before.size() != after.size()) {
        changes.push_back(fmt::format("register count {} -> {}", before.size(), after.size()));
        return changes;
    }
    for (size_t index = 0; index < before.size(); ++index) {
        if (before[index].value != after[index].value) {
            changes.push_back(fmt::format(
                "{} on M{}: 0x{:x} -> 0x{:x}",
                before[index].name,
                before[index].mimir_index,
                before[index].value,
                after[index].value));
        }
    }
    return changes;
}

// A launch that throws is reported with its transfer records, then the caller returns.
void report_launch_exception(
    IDevice* device, const std::exception& error, const std::vector<MimirTransferRecord>& records) {
    dump_debug_last(MetalContext::instance().get_cluster(), soc_desc_of(device), device->id());
    ADD_FAILURE() << "cce_copy_ops launch threw: " << error.what();
    for (const MimirTransferRecord& record : records) {
        ADD_FAILURE() << record.to_string();
    }
}

// The read-back runs on the next hart of the same CCE, so the hart that wrote the data does not serve its own read.
uint32_t readback_hart(uint32_t writer_hart) { return (writer_hart + 1) % cce_hart_count(); }

// The issuing CCE writes to and reads from the TL1 of a sibling CCE on the same Mimir through SPA, which the DMRISC
// remap sends to the sibling's SRAM on the same chiplet. No ATT entry may be used, so the DEBUG_LAST registers must
// not change.
void run_local_copy(IDevice* device, const LocalPair& pair) {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const ChipId chip = device->id();
    const uint32_t num_cces = num_cces_from_soc(soc);
    const uint32_t source = pair.source_cce;
    const uint32_t destination = pair.destination_cce;
    const std::string label = fmt::format("Cce{}To{} SameChipletControl", source, destination);
    const uint32_t words = kLocalBytesPerHart / 4;
    const uint32_t source_mimir = mimir_index_for_cce(source);

    std::vector<std::vector<uint32_t>> expected(cce_hart_count());
    std::vector<std::vector<uint32_t>> control_expected(cce_hart_count());
    std::vector<std::vector<uint32_t>> sibling_expected(cce_hart_count());
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        const uint64_t source_offset = tl1_data(hart);
        const uint64_t control_source_offset = tl1_data(hart) + kControlSourceOffset;
        const uint64_t landing_offset = tl1_remote_landing(source, hart);
        const uint32_t tag = guard_tag(source, hart);
        expected[hart] = make_pattern(MimirPattern::Prng, words, source_offset, transfer_seed(source, hart, 0));
        control_expected[hart] =
            make_pattern(MimirPattern::Prng, words, control_source_offset, transfer_seed(source, hart, 1));
        sibling_expected[hart] = make_pattern(
            MimirPattern::Prng, words, landing_offset + kSiblingSourceOffset, transfer_seed(source, hart, 2));
        write_cce_tl1(device, source, source_offset, expected[hart]);
        write_cce_tl1(device, source, control_source_offset, control_expected[hart]);
        write_cce_tl1(device, destination, landing_offset + kSiblingSourceOffset, sibling_expected[hart]);
        write_cce_tl1(device, source, tl1_data(hart) + kSiblingReadOffset, make_seed_pattern(tag, words));
        write_cce_tl1(device, destination, landing_offset, make_seed_pattern(tag, words));
        write_guards_except(device, num_cces, destination, landing_offset, tag);
    }

    const D2dErrorSnapshot errors_before = snapshot_d2d_errors(cluster, soc, chip);
    const std::vector<MimirRegister> debug_before =
        registers_on_mimir(read_debug_last(cluster, soc, chip), source_mimir);

    std::vector<CceCopyLaunch> launches;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        const uint64_t landing_offset = tl1_remote_landing(source, hart);
        CopyOpTable table;
        table.add(
            CopyOp{own_tl1(tl1_data(hart) + kControlSourceOffset), own_tl1(tl1_data(hart) + kReadbackOffset), words});
        table.add(CopyOp{own_tl1(tl1_data(hart)), sram_spa(destination, landing_offset), words});
        table.add(CopyOp{
            sram_spa(destination, landing_offset + kSiblingSourceOffset),
            own_tl1(tl1_data(hart) + kSiblingReadOffset),
            words});
        launches.push_back(CceCopyLaunch{source, hart, table});
    }
    std::vector<CceCopyResult> results;
    try {
        run_cce_copy_ops(device, launches, results);
    } catch (const std::exception& error) {
        report_launch_exception(device, error, {});
        return;
    }

    const std::vector<MimirRegister> debug_after =
        registers_on_mimir(read_debug_last(cluster, soc, chip), source_mimir);
    for (const std::string& change : debug_last_changes(debug_before, debug_after)) {
        ADD_FAILURE() << label << ": ATT debug register changed, so the ATT may have been used: " << change;
    }
    expect_no_new_d2d_error_bits(errors_before, snapshot_d2d_errors(cluster, soc, chip), label);

    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        const uint64_t landing_offset = tl1_remote_landing(source, hart);
        const std::string what = fmt::format("{} hart {}", label, hart);
        expect_copy_done(
            results[hart],
            xor_of(expected[hart]) ^ xor_of(control_expected[hart]) ^ xor_of(sibling_expected[hart]),
            what);
        EXPECT_TRUE(compare_words(
            what + ", own TL1 control copy",
            control_expected[hart],
            read_cce_tl1(device, source, tl1_data(hart) + kReadbackOffset, kLocalBytesPerHart)));
        EXPECT_TRUE(
            compare_words(what, expected[hart], read_cce_tl1(device, destination, landing_offset, kLocalBytesPerHart)));
        EXPECT_TRUE(compare_words(
            what + ", sibling TL1 read",
            sibling_expected[hart],
            read_cce_tl1(device, source, tl1_data(hart) + kSiblingReadOffset, kLocalBytesPerHart)));
        expect_guards_intact_except(device, num_cces, destination, landing_offset, guard_tag(source, hart), what);
    }
    if (::testing::Test::HasFailure()) {
        dump_debug_last(cluster, soc, chip);
    }
}

// Shared body for the remote WriteRead, SizeAlignment, Bidir and FanOutFanIn tests. Each transfer is one hart of one
// source CCE writing a window in a destination CCE. With read_back, the next hart of the source CCE reads the same
// window back into its own TL1 with a second launch. Each direction of a transfer is checked on its own, with a record
// in every message.
void run_remote_transfers(
    IDevice* device, const std::vector<RemoteTransfer>& transfers, bool read_back, const std::string& label) {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const ChipId chip = device->id();
    const uint32_t num_cces = num_cces_from_soc(soc);
    std::vector<CceSpaRoutes> routes_by_cce;
    for (uint32_t cce = 0; cce < num_cces; ++cce) {
        routes_by_cce.push_back(mimir_contract::read_cce_spa_routes(device, cce));
    }

    std::vector<PreparedTransfer> prepared;
    for (const RemoteTransfer& transfer : transfers) {
        PreparedTransfer item;
        item.transfer = transfer;
        const uint32_t words = transfer.size_bytes / 4;
        item.source_offset = tl1_data(transfer.hart) + transfer.alignment;
        item.landing_offset = tl1_remote_landing(transfer.source_cce, transfer.hart) + transfer.alignment;
        item.readback_offset = tl1_data(transfer.hart) + kReadbackOffset + transfer.alignment;
        item.tag = guard_tag(transfer.source_cce, transfer.hart);
        const uint32_t seed = transfer_seed(transfer.source_cce, transfer.hart, transfer.salt);
        item.expected = make_pattern(transfer.pattern, words, item.source_offset, seed);
        item.write_record = make_sram_record(
            soc,
            routes_by_cce.at(transfer.source_cce),
            fmt::format(
                "{} hart {} {}",
                endpoint_name(transfer.source_cce, item.source_offset),
                transfer.hart,
                pattern_name(transfer.pattern)),
            endpoint_name(transfer.destination_cce, item.landing_offset),
            transfer.destination_cce,
            item.landing_offset,
            transfer.size_bytes,
            seed);
        item.read_record = make_sram_record(
            soc,
            routes_by_cce.at(transfer.source_cce),
            endpoint_name(transfer.destination_cce, item.landing_offset),
            fmt::format(
                "{} hart {} {}",
                endpoint_name(transfer.source_cce, item.readback_offset),
                readback_hart(transfer.hart),
                pattern_name(transfer.pattern)),
            transfer.destination_cce,
            item.landing_offset,
            transfer.size_bytes,
            seed);

        write_cce_tl1(device, transfer.source_cce, item.source_offset, item.expected);
        if (read_back) {
            write_cce_tl1(device, transfer.source_cce, item.readback_offset, make_seed_pattern(item.tag, words));
        }
        // Missing or double translation is caught by the guards in the other CCEs (below) and by the seeded landing
        // slot. The exact addresses such a translation would hit are not seeded.
        write_cce_tl1(device, transfer.destination_cce, item.landing_offset, make_seed_pattern(item.tag, words));
        write_cce_tl1(
            device, transfer.destination_cce, item.landing_offset + transfer.size_bytes, make_guard(item.tag));
        write_guards_except(device, num_cces, transfer.destination_cce, item.landing_offset, item.tag);
        prepared.push_back(std::move(item));
    }

    const D2dErrorSnapshot errors_before = snapshot_d2d_errors(cluster, soc, chip);
    std::vector<CceCopyLaunch> write_launches;
    for (const PreparedTransfer& item : prepared) {
        CopyOpTable table;
        table.add(CopyOp{
            own_tl1(item.source_offset),
            sram_spa(item.transfer.destination_cce, item.landing_offset),
            item.transfer.size_bytes / 4});
        write_launches.push_back(CceCopyLaunch{item.transfer.source_cce, item.transfer.hart, table});
        log_transfer(item.write_record);
    }
    std::vector<CceCopyResult> write_results;
    try {
        run_cce_copy_ops(device, write_launches, write_results);
    } catch (const std::exception& error) {
        std::vector<MimirTransferRecord> records;
        for (const PreparedTransfer& item : prepared) {
            records.push_back(item.write_record);
        }
        report_launch_exception(device, error, records);
        return;
    }

    for (size_t index = 0; index < prepared.size(); ++index) {
        const PreparedTransfer& item = prepared[index];
        const std::string text = item.write_record.to_string();
        expect_copy_done(write_results[index], xor_of(item.expected), text);
        EXPECT_TRUE(compare_words(
            text,
            item.expected,
            read_cce_tl1(device, item.transfer.destination_cce, item.landing_offset, item.transfer.size_bytes)));
    }

    if (read_back) {
        std::vector<CceCopyLaunch> read_launches;
        for (const PreparedTransfer& item : prepared) {
            CopyOpTable table;
            table.add(CopyOp{
                sram_spa(item.transfer.destination_cce, item.landing_offset),
                own_tl1(item.readback_offset),
                item.transfer.size_bytes / 4});
            read_launches.push_back(CceCopyLaunch{item.transfer.source_cce, readback_hart(item.transfer.hart), table});
            log_transfer(item.read_record);
        }
        std::vector<CceCopyResult> read_results;
        try {
            run_cce_copy_ops(device, read_launches, read_results);
        } catch (const std::exception& error) {
            std::vector<MimirTransferRecord> records;
            for (const PreparedTransfer& item : prepared) {
                records.push_back(item.read_record);
            }
            report_launch_exception(device, error, records);
            return;
        }

        for (size_t index = 0; index < prepared.size(); ++index) {
            const PreparedTransfer& item = prepared[index];
            const std::string text = item.read_record.to_string();
            expect_copy_done(read_results[index], xor_of(item.expected), text);
            EXPECT_TRUE(compare_words(
                text,
                item.expected,
                read_cce_tl1(device, item.transfer.source_cce, item.readback_offset, item.transfer.size_bytes)));
        }
    }

    for (const PreparedTransfer& item : prepared) {
        const std::string text = item.write_record.to_string();
        expect_guard_intact(
            device, item.transfer.destination_cce, item.landing_offset + item.transfer.size_bytes, item.tag, text);
        expect_guards_intact_except(
            device, num_cces, item.transfer.destination_cce, item.landing_offset, item.tag, text);
    }
    expect_no_new_d2d_error_bits(errors_before, snapshot_d2d_errors(cluster, soc, chip), label);
    if (::testing::Test::HasFailure()) {
        dump_debug_last(cluster, soc, chip);
    }
}

// Every hart of the source CCE writes its own window (64 B to 4 KiB) to the destination, then the next hart of
// that CCE reads it back.
void run_remote_write_read(IDevice* device, const std::vector<RemotePair>& pairs, const std::string& label) {
    std::vector<RemoteTransfer> transfers;
    for (const RemotePair& pair : pairs) {
        for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
            transfers.push_back(RemoteTransfer{
                .source_cce = pair.source_cce(),
                .hart = hart,
                .destination_cce = pair.destination_cce(),
                .size_bytes = kWriteReadSizesBytes[hart % kWriteReadSizesBytes.size()],
                .pattern = kReadLegPatterns[(pair.source_cce() + hart) % kReadLegPatterns.size()],
            });
        }
    }
    run_remote_transfers(device, transfers, /*read_back=*/true, label);
}

// The host fills a window in the remote CCE, and the source CCE copies it into its own TL1. The copy and
// the done-block XOR are both checked against the host data.
void run_remote_host_init_read(IDevice* device, const RemotePair& pair) {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const ChipId chip = device->id();
    const uint32_t num_cces = num_cces_from_soc(soc);
    const uint32_t source = pair.source_cce();
    const uint32_t destination = pair.destination_cce();
    const std::string label = fmt::format("{} HostInitRemoteRead", remote_pair_label(pair));

    struct HostFilledRead {
        uint32_t hart = 0;
        uint32_t size_bytes = 0;
        uint64_t landing_offset = 0;
        uint64_t copy_offset = 0;
        uint32_t tag = 0;
        std::vector<uint32_t> expected;
        MimirTransferRecord record;
    };

    const CceSpaRoutes source_routes = mimir_contract::read_cce_spa_routes(device, source);
    std::vector<HostFilledRead> reads;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        HostFilledRead read;
        read.hart = hart;
        read.size_bytes = kWriteReadSizesBytes[hart % kWriteReadSizesBytes.size()];
        read.landing_offset = tl1_remote_landing(source, hart);
        read.copy_offset = tl1_data(hart) + kReadbackOffset;
        read.tag = guard_tag(source, hart);
        const uint32_t seed = transfer_seed(source, hart, 1);
        read.expected = make_pattern(MimirPattern::Prng, read.size_bytes / 4, read.landing_offset, seed);
        read.record = make_sram_record(
            soc,
            source_routes,
            endpoint_name(destination, read.landing_offset),
            fmt::format("{} hart {}", endpoint_name(source, read.copy_offset), hart),
            destination,
            read.landing_offset,
            read.size_bytes,
            seed);
        write_cce_tl1(device, destination, read.landing_offset, read.expected);
        write_cce_tl1(device, source, read.copy_offset, make_seed_pattern(read.tag, read.size_bytes / 4));
        write_guard(device, source, read.copy_offset + read.size_bytes, read.tag);
        write_guards_except(device, num_cces, destination, read.landing_offset, read.tag);
        reads.push_back(std::move(read));
    }

    const D2dErrorSnapshot errors_before = snapshot_d2d_errors(cluster, soc, chip);
    std::vector<CceCopyLaunch> launches;
    for (const HostFilledRead& read : reads) {
        CopyOpTable table;
        table.add(CopyOp{sram_spa(destination, read.landing_offset), own_tl1(read.copy_offset), read.size_bytes / 4});
        launches.push_back(CceCopyLaunch{source, read.hart, table});
        log_transfer(read.record);
    }
    std::vector<CceCopyResult> results;
    try {
        run_cce_copy_ops(device, launches, results);
    } catch (const std::exception& error) {
        std::vector<MimirTransferRecord> records;
        for (const HostFilledRead& read : reads) {
            records.push_back(read.record);
        }
        report_launch_exception(device, error, records);
        return;
    }

    for (size_t index = 0; index < reads.size(); ++index) {
        const HostFilledRead& read = reads[index];
        const std::string text = read.record.to_string();
        expect_copy_done(results[index], xor_of(read.expected), text);
        EXPECT_TRUE(
            compare_words(text, read.expected, read_cce_tl1(device, source, read.copy_offset, read.size_bytes)));
        expect_guard_intact(device, source, read.copy_offset + read.size_bytes, read.tag, text);
        expect_guards_intact_except(device, num_cces, destination, read.landing_offset, read.tag, text);
    }
    expect_no_new_d2d_error_bits(errors_before, snapshot_d2d_errors(cluster, soc, chip), label);
    if (::testing::Test::HasFailure()) {
        dump_debug_last(cluster, soc, chip);
    }
}

void run_remote_size_alignment(IDevice* device, const RemotePair& pair) {
    constexpr size_t combination_count = kSizeAlignmentSizesBytes.size() * kSizeAlignmentOffsetsBytes.size();
    for (size_t first = 0; first < combination_count; first += cce_hart_count()) {
        std::vector<RemoteTransfer> round;
        for (uint32_t hart = 0; hart < cce_hart_count() && first + hart < combination_count; ++hart) {
            const size_t combination = first + hart;
            round.push_back(RemoteTransfer{
                .source_cce = pair.source_cce(),
                .hart = hart,
                .destination_cce = pair.destination_cce(),
                .size_bytes = kSizeAlignmentSizesBytes[combination / kSizeAlignmentOffsetsBytes.size()],
                .alignment = kSizeAlignmentOffsetsBytes[combination % kSizeAlignmentOffsetsBytes.size()],
                .salt = static_cast<uint32_t>(combination),
            });
        }
        run_remote_transfers(
            device,
            round,
            /*read_back=*/false,
            fmt::format("{} SizeAlignment round {}", remote_pair_label(pair), first / cce_hart_count()));
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

class MimirLocalTl1 : public ::testing::TestWithParam<LocalPair> {};

TEST_P(MimirLocalTl1, SameChipletControl) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const LocalPair pair = GetParam();
    if (const std::string skip_reason = local_skip_reason(pair); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    run_local_copy(device, pair);
}

INSTANTIATE_TEST_SUITE_P(MM5, MimirLocalTl1, ::testing::ValuesIn(same_chiplet_pairs()), local_pair_name);

class MimirRemoteTl1 : public MimirRemoteParamTest<RemotePair> {};

TEST_P(MimirRemoteTl1, WriteRead) {
    const RemotePair pair = GetParam();
    if (const std::string skip_reason = remote_skip_reason(pair); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    IDevice* device = open_remote_device();
    run_remote_write_read(device, {pair}, fmt::format("{} WriteRead", remote_pair_label(pair)));
}

TEST_P(MimirRemoteTl1, HostInitRemoteRead) {
    const RemotePair pair = GetParam();
    if (const std::string skip_reason = remote_skip_reason(pair); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    IDevice* device = open_remote_device();
    run_remote_host_init_read(device, pair);
}

TEST_P(MimirRemoteTl1, SizeAlignment) {
    const RemotePair pair = GetParam();
    if (const std::string skip_reason = remote_skip_reason(pair); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    IDevice* device = open_remote_device();
    run_remote_size_alignment(device, pair);
}

INSTANTIATE_TEST_SUITE_P(MM5Direct, MimirRemoteTl1, ::testing::ValuesIn(direct_remote_pairs()), remote_pair_name);

INSTANTIATE_TEST_SUITE_P(MM5Routed, MimirRemoteTl1, ::testing::ValuesIn(routed_remote_pairs()), remote_pair_name);

class MimirRemoteTl1Bidir : public MimirRemoteTest {};

TEST_F(MimirRemoteTl1Bidir, Concurrent) {
    const std::vector<RemotePair> pairs = direct_remote_pairs();
    if (const std::string skip_reason = remote_skip_reason(pairs.front()); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    IDevice* device = open_remote_device();
    run_remote_write_read(device, pairs, "MimirRemoteTl1Bidir.Concurrent");
}

// One-to-many from M0.CCE1 (to M1.CCE0 direct and M1.CCE1 one hop) and its mirror from M1.CCE0 (to M0.CCE1 direct
// and M0.CCE0 one hop). Many-to-one into M1.CCE1 (from M0.CCE0 two hops and M0.CCE1 one hop) and its mirror into
// M0.CCE0 (from M1.CCE1 two hops and M1.CCE0 one hop). All in one program, and each written window is read back by
// the next hart of its source CCE. The windows are disjoint because each landing slot is keyed by source CCE and
// hart. Each destination CCE is treated as the excluded CCE for its own transfers.
TEST_F(MimirRemoteMixed, FanOutFanIn) {
    if (const std::string skip_reason = remote_skip_reason(RemotePair{0, 0, 1, 1}); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    IDevice* device = open_remote_device();
    const uint32_t mimir0_cce0 = cce_index_for(0, 0);
    const uint32_t mimir0_cce1 = cce_index_for(0, 1);
    const uint32_t mimir1_cce0 = cce_index_for(1, 0);
    const uint32_t mimir1_cce1 = cce_index_for(1, 1);
    const std::vector<RemoteTransfer> transfers = {
        {.source_cce = mimir0_cce1, .hart = 0, .destination_cce = mimir1_cce0, .size_bytes = 1024},
        {.source_cce = mimir0_cce1, .hart = 1, .destination_cce = mimir1_cce1, .size_bytes = 2048},
        {.source_cce = mimir0_cce0, .hart = 0, .destination_cce = mimir1_cce1, .size_bytes = 512},
        {.source_cce = mimir0_cce1, .hart = 2, .destination_cce = mimir1_cce1, .size_bytes = 4096},
        {.source_cce = mimir1_cce0, .hart = 0, .destination_cce = mimir0_cce1, .size_bytes = 1024},
        {.source_cce = mimir1_cce0, .hart = 1, .destination_cce = mimir0_cce0, .size_bytes = 2048},
        {.source_cce = mimir1_cce1, .hart = 0, .destination_cce = mimir0_cce0, .size_bytes = 512},
        {.source_cce = mimir1_cce0, .hart = 2, .destination_cce = mimir0_cce0, .size_bytes = 4096},
    };
    run_remote_transfers(device, transfers, /*read_back=*/true, "MimirRemoteMixed.FanOutFanIn");
}

}  // namespace tt::tt_metal::mimir_test::tl1
