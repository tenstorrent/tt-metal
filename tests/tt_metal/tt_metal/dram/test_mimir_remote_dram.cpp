// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/arch.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

// The Glean MM-6 plan names dram_neighbour, dram_sharded, one_packet, interleaved, direct_write and the PCIe-style
// tests to adapt. Their host sides use fast-dispatch fixtures on Tensix or worker cores, and their kernels use the NoC
// dataflow API, which no Mimir CCE kernel has run yet. Mimir also has no PCIe. So these tests use the
// experimental/cce_gddr.h path through cce_copy_ops.cpp.

namespace tt::tt_metal::mimir_test::remote_dram {

inline constexpr uint32_t kEveryAnchorBytes = 4 * 1024;
inline constexpr uint32_t kEveryAnchorWords = static_cast<uint32_t>(kEveryAnchorBytes / sizeof(uint32_t));
inline constexpr uint32_t kLargeTransferBytes = 32 * 1024;
inline constexpr uint64_t kRemoteGuardDistance = 4 * kGiB;
inline constexpr uint32_t kMixedSlotBytes = 32 * 1024;
inline constexpr uint32_t kMixedRemoteSlotBytes = 8 * 1024;
inline constexpr uint32_t kTrafficReadBackTl1Offset = 64 * 1024;
inline constexpr std::array<uint32_t, 6> kTrafficSizeBytes = {4, 64, 124, 512, 4 * 1024, 32 * 1024};
inline constexpr std::array<uint32_t, 5> kTrafficAlignments = {0, 4, 60, 124, static_cast<uint32_t>(kEmuPageBytes - 4)};
// Test kinds salt the seeds and the pre-fill tags. AddressDerived and the constant patterns ignore the seed, so data
// another test left in DRAM can match; the decoys from add_transfer_guards catch a read that lands there.
inline constexpr uint32_t kTestKindWriteFromTl1 = 1;
inline constexpr uint32_t kTestKindReadToTl1 = 2;
inline constexpr uint32_t kTestKindTl1RoundTrip = 3;
// Constant patterns could match an unwritten page after a mis-translated read, so ReadToTl1 uses only these.
inline constexpr std::array<MimirPattern, 2> kReadToTl1Patterns = {MimirPattern::AddressDerived, MimirPattern::Prng};

// The TrafficMatrix test has its own fixture: gtest takes the suite name from the fixture class, and one fixture
// has one parameter type.
struct RemoteDramViewParam {
    uint32_t source_mimir = 0;
    uint32_t source_location = 0;
    uint32_t view = 0;
};

struct RemoteDramSourceParam {
    uint32_t source_mimir = 0;
    uint32_t source_location = 0;
};

std::string remote_dram_view_name(const ::testing::TestParamInfo<RemoteDramViewParam>& info) {
    return fmt::format("SrcM{}Cce{}View{}", info.param.source_mimir, info.param.source_location, info.param.view);
}

std::string remote_dram_source_name(const ::testing::TestParamInfo<RemoteDramSourceParam>& info) {
    return fmt::format("SrcM{}Cce{}", info.param.source_mimir, info.param.source_location);
}

std::vector<RemoteDramViewParam> remote_dram_view_params() {
    std::vector<RemoteDramViewParam> params;
    if (mimir_contract::selected_soc_desc_path() == nullptr) {
        params.push_back(RemoteDramViewParam{});
        return params;
    }
    const metal_SocDescriptor& soc = configured_soc_desc();
    const uint32_t mimirs = static_cast<uint32_t>(soc.get_num_dram_channels());
    const uint32_t locations = locations_per_channel(soc);
    const uint32_t views = static_cast<uint32_t>(soc.get_num_dram_views());
    for (uint32_t source_mimir = 0; source_mimir < mimirs; ++source_mimir) {
        for (uint32_t source_location = 0; source_location < locations; ++source_location) {
            for (uint32_t view = 0; view < views; ++view) {
                params.push_back(RemoteDramViewParam{source_mimir, source_location, view});
            }
        }
    }
    return params;
}

std::vector<RemoteDramSourceParam> remote_dram_source_params() {
    std::vector<RemoteDramSourceParam> params;
    if (mimir_contract::selected_soc_desc_path() == nullptr) {
        params.push_back(RemoteDramSourceParam{});
        return params;
    }
    const metal_SocDescriptor& soc = configured_soc_desc();
    const uint32_t mimirs = static_cast<uint32_t>(soc.get_num_dram_channels());
    const uint32_t locations = locations_per_channel(soc);
    for (uint32_t source_mimir = 0; source_mimir < mimirs; ++source_mimir) {
        for (uint32_t source_location = 0; source_location < locations; ++source_location) {
            params.push_back(RemoteDramSourceParam{source_mimir, source_location});
        }
    }
    return params;
}

// Decided from the descriptor before any device is opened.
std::string source_skip_reason(const MimirTopology& topology, uint32_t source_mimir, uint32_t source_location) {
    if (topology.num_mimirs < 2) {
        return "Remote DRAM needs two Mimirs; the descriptor has one.";
    }
    if (source_mimir >= topology.num_mimirs) {
        return fmt::format("Descriptor has no Mimir {}.", source_mimir);
    }
    const uint32_t source_cce = cce_index_for(source_mimir, source_location);
    if (source_cce >= topology.num_cces) {
        return fmt::format("Descriptor does not expose CCE {}.", source_cce);
    }
    return "";
}

std::string view_skip_reason(const MimirTopology& topology, const RemoteDramViewParam& param) {
    const std::string source_reason = source_skip_reason(topology, param.source_mimir, param.source_location);
    if (!source_reason.empty()) {
        return source_reason;
    }
    if (param.view >= topology.num_views) {
        return fmt::format("Descriptor has no DRAM view {}.", param.view);
    }
    if (topology.view_mimir[param.view] == param.source_mimir) {
        return fmt::format("DRAM view {} is on the source Mimir, not a remote view.", param.view);
    }
    return "";
}

struct RemoteDramRun {
    distributed::MeshDevice& unit_mesh;
    IDevice* device;
    const metal_SocDescriptor& soc;
    Cluster& cluster;
    ChipId chip;
    uint32_t source_mimir;
    uint32_t source_location;
    uint32_t source_cce;
    uint32_t dest_mimir;
    uint32_t dest_view;
    uint64_t unreserved;
    CceSpaRoutes source_routes;
};

RemoteDramRun make_run(
    distributed::MeshDevice& unit_mesh,
    IDevice* device,
    uint32_t source_mimir,
    uint32_t source_location,
    uint32_t dest_view) {
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    return RemoteDramRun{
        unit_mesh,
        device,
        soc,
        MetalContext::instance().get_cluster(),
        device->id(),
        source_mimir,
        source_location,
        cce_index_for(source_mimir, source_location),
        static_cast<uint32_t>(soc.get_channel_for_dram_view(static_cast<int>(dest_view))),
        dest_view,
        unreserved,
        mimir_contract::read_cce_spa_routes(device, cce_index_for(source_mimir, source_location))};
}

const MimirAnchor& anchor_named(const std::vector<MimirAnchor>& anchors, const std::string& name) {
    for (const MimirAnchor& anchor : anchors) {
        if (anchor.name == name) {
            return anchor;
        }
    }
    TT_FATAL(false, "No DRAM anchor named {}", name);
    return anchors.front();
}

std::string source_tl1_label(uint32_t source_mimir, uint32_t source_location, uint32_t hart, uint64_t tl1_offset) {
    return fmt::format("M{}.CCE{} hart {} TL1 0x{:x}", source_mimir, source_location, hart, tl1_offset);
}

std::string dram_label(uint32_t dest_mimir, uint32_t dest_view, uint64_t view_offset) {
    return fmt::format("M{} view {} offset 0x{:x}", dest_mimir, dest_view, view_offset);
}

MimirTransferRecord make_gddr_record(
    const CceSpaRoutes& source_routes,
    const metal_SocDescriptor& soc,
    uint32_t source_mimir,
    uint32_t dest_mimir,
    std::string source,
    std::string destination,
    uint64_t chiplet_offset,
    uint32_t size_bytes,
    uint32_t seed) {
    const std::optional<AttEndpointPrediction> endpoint =
        att_endpoint_for_remote_gddr(source_mimir, source_routes.gddr_d2d[dest_mimir]);
    MimirTransferRecord record;
    record.source = std::move(source);
    record.destination = std::move(destination);
    record.requested_address = mimir_contract::cce_gddr_address(source_routes, dest_mimir, chiplet_offset);
    record.destination_local_address = tt::umd::mimir_local_address_windows(soc).dram_base + chiplet_offset;
    record.predicted_stages.emplace_back("issued SPA", record.requested_address);
    if (endpoint.has_value()) {
        record.predicted_stages.emplace_back(
            fmt::format("ATT endpoint {} (expected entry word 0x{:x})", endpoint->entry_index, endpoint->value),
            endpoint->entry_index);
    }
    record.predicted_stages.emplace_back("after SS_BAR2 (destination-local)", record.destination_local_address);
    record.length = size_bytes;
    record.alignment = static_cast<uint32_t>(record.requested_address % 64);
    record.crosses_d2d = source_mimir != dest_mimir;
    record.seed = seed;
    return record;
}

struct ChipletLocation {
    uint32_t view = 0;
    uint64_t view_offset = 0;
};

// False when [chiplet_offset, +size) is not inside an exposed view of that Mimir.
bool resolve_chiplet_location(
    const metal_SocDescriptor& soc,
    uint32_t mimir_index,
    uint64_t chiplet_offset,
    uint32_t size_bytes,
    ChipletLocation& location) {
    for (uint32_t view = 0; view < static_cast<uint32_t>(soc.get_num_dram_views()); ++view) {
        if (static_cast<uint32_t>(soc.get_channel_for_dram_view(static_cast<int>(view))) != mimir_index) {
            continue;
        }
        const uint64_t view_base = static_cast<uint64_t>(soc.get_address_offset(static_cast<int>(view)));
        if (chiplet_offset >= view_base && chiplet_offset + size_bytes <= view_base + soc.dram_view_size) {
            location.view = view;
            location.view_offset = chiplet_offset - view_base;
            return true;
        }
    }
    return false;
}

// Anchors are taken without the reserved start, so every anchor starts at or above UNRESERVED.
bool inside_an_anchor(const RemoteDramRun& run, uint32_t view, uint64_t view_offset, uint64_t size_bytes) {
    for (const MimirAnchor& anchor : mimir_dram_anchors(run.soc, view, run.unreserved, false)) {
        if (view_offset >= anchor.view_offset && view_offset + size_bytes <= anchor.view_offset + kAnchorBytes) {
            return true;
        }
    }
    return false;
}

// A guard is a region that no transfer may touch. Its seed words are unique, so a stray write shows on readback.
struct GuardRegion {
    std::string owner;
    std::string label;
    uint32_t view = 0;
    uint64_t view_offset = 0;
    std::vector<uint32_t> expected;
};

void add_guard(
    const RemoteDramRun& run,
    std::vector<GuardRegion>& guards,
    const std::string& owner,
    const std::string& label,
    uint32_t mimir_index,
    uint64_t chiplet_offset,
    uint32_t size_bytes,
    uint32_t& next_guard_tag) {
    TT_FATAL(
        size_bytes != 0 && size_bytes % sizeof(uint32_t) == 0,
        "Guard size {} must be a nonzero multiple of 4 bytes",
        size_bytes);
    ChipletLocation location;
    if (!resolve_chiplet_location(run.soc, mimir_index, chiplet_offset, size_bytes, location)) {
        log_info(
            tt::LogTest,
            "Guard {} skipped: M{} chiplet offset 0x{:x} is not exposed.",
            label,
            mimir_index,
            chiplet_offset);
        return;
    }
    if (!inside_an_anchor(run, location.view, location.view_offset, size_bytes)) {
        log_info(
            tt::LogTest,
            "Guard {} skipped: M{} view {} offset 0x{:x} is outside every anchor.",
            label,
            mimir_index,
            location.view,
            location.view_offset);
        return;
    }
    for (const GuardRegion& existing : guards) {
        if (existing.view == location.view && existing.view_offset == location.view_offset) {
            return;
        }
    }
    GuardRegion guard;
    guard.owner = owner;
    guard.label = label;
    guard.view = location.view;
    guard.view_offset = location.view_offset;
    guard.expected = make_seed_pattern(next_guard_tag++, size_bytes / static_cast<uint32_t>(sizeof(uint32_t)));
    guards.push_back(std::move(guard));
}

// A guard just outside a traffic window.
void add_window_guard(
    const std::string& owner,
    const std::string& label,
    uint32_t view,
    uint64_t view_offset,
    uint32_t tag,
    std::vector<GuardRegion>& guards) {
    GuardRegion guard;
    guard.owner = owner;
    guard.label = label;
    guard.view = view;
    guard.view_offset = view_offset;
    guard.expected = make_guard(tag, kGuardWords);
    guards.push_back(std::move(guard));
}

bool expect_words(
    const std::string& what,
    const std::vector<uint32_t>& expected,
    const std::vector<uint32_t>& actual,
    const std::string& context) {
    const ::testing::AssertionResult result = compare_words(what, expected, actual);
    if (!result) {
        ADD_FAILURE() << context << "\n" << result.message();
        return false;
    }
    return true;
}

void write_dram(
    distributed::MeshDevice& unit_mesh, uint32_t view, uint64_t view_offset, const std::vector<uint32_t>& data) {
    mimir_contract::write_gddr_view(unit_mesh, view, view_offset, data);
}

bool read_dram_and_expect(
    distributed::MeshDevice& unit_mesh,
    uint32_t view,
    uint64_t view_offset,
    const std::vector<uint32_t>& expected,
    const std::string& context) {
    std::vector<uint32_t> actual;
    mimir_contract::read_gddr_view(
        unit_mesh, view, view_offset, static_cast<uint32_t>(expected.size() * sizeof(uint32_t)), actual);
    return expect_words("DRAM readback", expected, actual, context);
}

void write_guards(const RemoteDramRun& run, const std::vector<GuardRegion>& guards) {
    for (const GuardRegion& guard : guards) {
        write_dram(run.unit_mesh, guard.view, guard.view_offset, guard.expected);
    }
}

bool check_guards(const RemoteDramRun& run, const std::vector<GuardRegion>& guards) {
    bool ok = true;
    for (const GuardRegion& guard : guards) {
        ok =
            read_dram_and_expect(
                run.unit_mesh, guard.view, guard.view_offset, guard.expected, guard.owner + "\nguard " + guard.label) &&
            ok;
    }
    return ok;
}

// Passed and Failed are results the test reports on. Threw means the launch raised an exception, so the caller
// returns without touching the device again.
enum class RunOutcome { Passed, Failed, Threw };

// All launches of one call go in one Program, whose launches are unordered. Only the done flags and XOR signatures
// are checked here; the data checks follow.
RunOutcome run_launches(
    IDevice* device,
    const std::vector<CceCopyLaunch>& launches,
    std::vector<CceCopyResult>& results,
    const std::vector<MimirTransferRecord>& records,
    const std::vector<uint32_t>& expected_xors) {
    TT_FATAL(expected_xors.size() == launches.size(), "One expected XOR per launch is required");
    try {
        const ::testing::AssertionResult launched = run_cce_copy_ops(device, launches, results);
        if (!launched) {
            ADD_FAILURE() << launched.message();
            return RunOutcome::Failed;
        }
    } catch (const std::exception& e) {
        std::string record_lines;
        for (const MimirTransferRecord& record : records) {
            record_lines += "\n" + record.to_string();
        }
        dump_debug_last(MetalContext::instance().get_cluster(), soc_desc_of(device), device->id());
        ADD_FAILURE() << "cce_copy_ops threw: " << e.what() << record_lines;
        return RunOutcome::Threw;
    }
    bool ok = true;
    for (size_t i = 0; i < results.size(); ++i) {
        const CceCopyResult& result = results[i];
        const std::string where = fmt::format(
            "cce_copy_ops on CCE {} hart {}: a missing done magic means the kernel returned early on an unknown "
            "endpoint kind, or host and kernel disagree on the copy ABI",
            result.cce_index,
            result.hart);
        expect_copy_done(result, expected_xors[i], where);
        if (!result.done || result.xor_signature != expected_xors[i]) {
            ok = false;
        }
    }
    return ok ? RunOutcome::Passed : RunOutcome::Failed;
}

// One transfer per hart of one anchor. The destination is dest_view at anchor + hart * 4 KiB.
struct HartTransfer {
    uint32_t hart = 0;
    uint32_t read_back_hart = 0;
    uint64_t view_offset = 0;
    uint64_t chiplet_offset = 0;
    uint64_t tl1_offset = 0;
    uint32_t seed = 0;
    uint32_t prefill_tag = 0;
    std::vector<uint32_t> pattern;
    MimirTransferRecord record;
    MimirTransferRecord read_back_record;
};

std::vector<HartTransfer> plan_hart_transfers(
    const RemoteDramRun& run, const MimirAnchor& anchor, uint32_t anchor_index, uint32_t test_kind) {
    std::vector<HartTransfer> transfers;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        HartTransfer transfer;
        transfer.hart = hart;
        transfer.read_back_hart = (hart + 1) % cce_hart_count();
        transfer.view_offset = anchor.view_offset + static_cast<uint64_t>(hart) * kEveryAnchorBytes;
        transfer.chiplet_offset = chiplet_offset_of(run.soc, run.dest_view, transfer.view_offset);
        transfer.tl1_offset = tl1_data(hart);
        transfer.seed = mimir_test_seed() ^ (test_kind << 24) ^ (run.source_cce << 16) ^ (anchor_index << 8) ^ hart;
        transfer.prefill_tag =
            (test_kind << 12) | (run.source_cce << 8) | ((anchor_index * cce_hart_count() + hart) & 0xFFu);
        const uint32_t pattern_index = hart + anchor_index + test_kind + run.source_cce;
        const MimirPattern pattern = test_kind == kTestKindReadToTl1
                                         ? kReadToTl1Patterns[pattern_index % kReadToTl1Patterns.size()]
                                         : kAllMimirPatterns[pattern_index % kAllMimirPatterns.size()];
        transfer.pattern = make_pattern(pattern, kEveryAnchorWords, transfer.chiplet_offset, transfer.seed);
        transfers.push_back(std::move(transfer));
    }
    return transfers;
}

// Decoys where a mis-translated write would land or a mis-translated read would read: the same offset on the source
// Mimir, +/-4 GiB on the destination Mimir, and the other remote views of the destination Mimir at the same offset.
void add_alias_decoys(
    const RemoteDramRun& run,
    const std::string& owner,
    const std::string& label_prefix,
    uint64_t chiplet_offset,
    uint32_t size_bytes,
    std::vector<GuardRegion>& guards,
    uint32_t& next_guard_tag) {
    add_guard(
        run,
        guards,
        owner,
        label_prefix + " source Mimir same offset",
        run.source_mimir,
        chiplet_offset,
        size_bytes,
        next_guard_tag);
    add_guard(
        run,
        guards,
        owner,
        label_prefix + " destination +4 GiB",
        run.dest_mimir,
        chiplet_offset + kRemoteGuardDistance,
        size_bytes,
        next_guard_tag);
    if (chiplet_offset >= kRemoteGuardDistance) {
        add_guard(
            run,
            guards,
            owner,
            label_prefix + " destination -4 GiB",
            run.dest_mimir,
            chiplet_offset - kRemoteGuardDistance,
            size_bytes,
            next_guard_tag);
    }
    const uint64_t dest_view_offset =
        chiplet_offset - static_cast<uint64_t>(run.soc.get_address_offset(static_cast<int>(run.dest_view)));
    for (uint32_t view = 0; view < static_cast<uint32_t>(run.soc.get_num_dram_views()); ++view) {
        const bool other_remote_view =
            view != run.dest_view &&
            static_cast<uint32_t>(run.soc.get_channel_for_dram_view(static_cast<int>(view))) == run.dest_mimir;
        if (!other_remote_view) {
            continue;
        }
        const uint64_t other_view_chiplet =
            static_cast<uint64_t>(run.soc.get_address_offset(static_cast<int>(view))) + dest_view_offset;
        add_guard(
            run,
            guards,
            owner,
            fmt::format("{} other remote view {}", label_prefix, view),
            run.dest_mimir,
            other_view_chiplet,
            size_bytes,
            next_guard_tag);
    }
}

void add_transfer_guards(
    const RemoteDramRun& run,
    const HartTransfer& transfer,
    std::vector<GuardRegion>& guards,
    uint32_t& next_guard_tag) {
    add_alias_decoys(
        run,
        transfer.record.to_string(),
        fmt::format("hart {}", transfer.hart),
        transfer.chiplet_offset,
        kEveryAnchorBytes,
        guards,
        next_guard_tag);
}

// Every hart copies its own TL1 data to remote DRAM. Each destination is pre-filled with a seed pattern so a copy
// that never lands fails. Checks the destination, the guards and the D2D debug and AXI error bits across the program.
RunOutcome write_from_tl1_anchor(
    const RemoteDramRun& run, const MimirAnchor& anchor, uint32_t anchor_index, uint32_t& next_guard_tag) {
    std::vector<HartTransfer> transfers = plan_hart_transfers(run, anchor, anchor_index, kTestKindWriteFromTl1);
    std::vector<GuardRegion> guards;
    std::vector<CceCopyLaunch> launches;
    std::vector<MimirTransferRecord> records;
    std::vector<uint32_t> expected_xors;
    for (HartTransfer& transfer : transfers) {
        transfer.record = make_gddr_record(
            run.source_routes,
            run.soc,
            run.source_mimir,
            run.dest_mimir,
            source_tl1_label(run.source_mimir, run.source_location, transfer.hart, transfer.tl1_offset),
            dram_label(run.dest_mimir, run.dest_view, transfer.view_offset),
            transfer.chiplet_offset,
            kEveryAnchorBytes,
            transfer.seed);
        log_transfer(transfer.record);
        write_dram(
            run.unit_mesh,
            run.dest_view,
            transfer.view_offset,
            make_seed_pattern(transfer.prefill_tag, kEveryAnchorWords));
        write_cce_tl1(run.device, run.source_cce, transfer.tl1_offset, transfer.pattern);
        add_transfer_guards(run, transfer, guards, next_guard_tag);

        CceCopyLaunch launch;
        launch.cce_index = run.source_cce;
        launch.hart = transfer.hart;
        launch.table.add(
            CopyOp{own_tl1(transfer.tl1_offset), gddr(run.dest_mimir, transfer.chiplet_offset), kEveryAnchorWords});
        launches.push_back(std::move(launch));
        records.push_back(transfer.record);
        expected_xors.push_back(xor_of(transfer.pattern));
    }

    write_guards(run, guards);
    bool ok = true;

    const D2dErrorSnapshot before = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    std::vector<CceCopyResult> results;
    const RunOutcome launched = run_launches(run.device, launches, results, records, expected_xors);
    if (launched == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    ok = launched == RunOutcome::Passed && ok;
    const D2dErrorSnapshot after = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    ok = expect_no_new_d2d_error_bits(before, after, "Anchor " + anchor.name) && ok;

    for (const HartTransfer& transfer : transfers) {
        ok = read_dram_and_expect(
                 run.unit_mesh, run.dest_view, transfer.view_offset, transfer.pattern, transfer.record.to_string()) &&
             ok;
    }
    ok = check_guards(run, guards) && ok;
    if (!ok) {
        dump_debug_last(run.cluster, run.soc, run.chip);
    }
    return ok ? RunOutcome::Passed : RunOutcome::Failed;
}

// The host writes remote DRAM and every hart copies it into its own TL1 window.
RunOutcome read_to_tl1_anchor(
    const RemoteDramRun& run, const MimirAnchor& anchor, uint32_t anchor_index, uint32_t& next_guard_tag) {
    std::vector<HartTransfer> transfers = plan_hart_transfers(run, anchor, anchor_index, kTestKindReadToTl1);
    std::vector<GuardRegion> guards;
    std::vector<CceCopyLaunch> launches;
    std::vector<MimirTransferRecord> records;
    std::vector<uint32_t> expected_xors;
    for (HartTransfer& transfer : transfers) {
        transfer.record = make_gddr_record(
            run.source_routes,
            run.soc,
            run.source_mimir,
            run.dest_mimir,
            dram_label(run.dest_mimir, run.dest_view, transfer.view_offset),
            source_tl1_label(run.source_mimir, run.source_location, transfer.hart, transfer.tl1_offset),
            transfer.chiplet_offset,
            kEveryAnchorBytes,
            transfer.seed);
        add_transfer_guards(run, transfer, guards, next_guard_tag);
        log_transfer(transfer.record);
        write_dram(run.unit_mesh, run.dest_view, transfer.view_offset, transfer.pattern);
        write_cce_tl1(run.device, run.source_cce, transfer.tl1_offset, make_guard(next_guard_tag++, kEveryAnchorWords));

        CceCopyLaunch launch;
        launch.cce_index = run.source_cce;
        launch.hart = transfer.hart;
        launch.table.add(
            CopyOp{gddr(run.dest_mimir, transfer.chiplet_offset), own_tl1(transfer.tl1_offset), kEveryAnchorWords});
        launches.push_back(std::move(launch));
        records.push_back(transfer.record);
        expected_xors.push_back(xor_of(transfer.pattern));
    }

    write_guards(run, guards);

    const D2dErrorSnapshot before = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    std::vector<CceCopyResult> results;
    const RunOutcome launched = run_launches(run.device, launches, results, records, expected_xors);
    if (launched == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    bool ok = launched == RunOutcome::Passed;
    const D2dErrorSnapshot after = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    ok = expect_no_new_d2d_error_bits(before, after, "Anchor " + anchor.name) && ok;
    for (const HartTransfer& transfer : transfers) {
        const std::vector<uint32_t> actual =
            read_cce_tl1(run.device, run.source_cce, transfer.tl1_offset, kEveryAnchorBytes);
        ok = expect_words("TL1 readback", transfer.pattern, actual, transfer.record.to_string()) && ok;
    }
    ok = check_guards(run, guards) && ok;
    if (!ok) {
        dump_debug_last(run.cluster, run.soc, run.chip);
    }
    return ok ? RunOutcome::Passed : RunOutcome::Failed;
}

// Each hart copies TL1 A to remote DRAM, and then the same data is copied from remote DRAM to TL1 B by the next
// hart. Launches in one program are unordered, so the read-back is a second program that starts after the write.
RunOutcome round_trip_anchor(
    const RemoteDramRun& run, const MimirAnchor& anchor, uint32_t anchor_index, uint32_t& next_guard_tag) {
    std::vector<HartTransfer> transfers = plan_hart_transfers(run, anchor, anchor_index, kTestKindTl1RoundTrip);
    std::vector<GuardRegion> guards;
    std::vector<CceCopyLaunch> write_launches;
    std::vector<CceCopyLaunch> read_back_launches;
    std::vector<MimirTransferRecord> write_records;
    std::vector<MimirTransferRecord> read_back_records;
    std::vector<uint32_t> expected_xors;
    for (HartTransfer& transfer : transfers) {
        const uint64_t tl1_b_offset = transfer.tl1_offset + kEveryAnchorBytes;
        transfer.record = make_gddr_record(
            run.source_routes,
            run.soc,
            run.source_mimir,
            run.dest_mimir,
            source_tl1_label(run.source_mimir, run.source_location, transfer.hart, transfer.tl1_offset),
            dram_label(run.dest_mimir, run.dest_view, transfer.view_offset),
            transfer.chiplet_offset,
            kEveryAnchorBytes,
            transfer.seed);
        transfer.read_back_record = make_gddr_record(
            run.source_routes,
            run.soc,
            run.source_mimir,
            run.dest_mimir,
            dram_label(run.dest_mimir, run.dest_view, transfer.view_offset),
            source_tl1_label(run.source_mimir, run.source_location, transfer.read_back_hart, tl1_b_offset),
            transfer.chiplet_offset,
            kEveryAnchorBytes,
            transfer.seed);
        add_transfer_guards(run, transfer, guards, next_guard_tag);
        log_transfer(transfer.record);
        log_transfer(transfer.read_back_record);
        write_dram(
            run.unit_mesh,
            run.dest_view,
            transfer.view_offset,
            make_seed_pattern(transfer.prefill_tag, kEveryAnchorWords));
        write_cce_tl1(run.device, run.source_cce, transfer.tl1_offset, transfer.pattern);
        write_cce_tl1(run.device, run.source_cce, tl1_b_offset, make_guard(next_guard_tag++, kEveryAnchorWords));

        CceCopyLaunch write_launch;
        write_launch.cce_index = run.source_cce;
        write_launch.hart = transfer.hart;
        write_launch.table.add(
            CopyOp{own_tl1(transfer.tl1_offset), gddr(run.dest_mimir, transfer.chiplet_offset), kEveryAnchorWords});
        write_launches.push_back(std::move(write_launch));
        write_records.push_back(transfer.record);

        CceCopyLaunch read_back_launch;
        read_back_launch.cce_index = run.source_cce;
        read_back_launch.hart = transfer.read_back_hart;
        read_back_launch.table.add(
            CopyOp{gddr(run.dest_mimir, transfer.chiplet_offset), own_tl1(tl1_b_offset), kEveryAnchorWords});
        read_back_launches.push_back(std::move(read_back_launch));
        read_back_records.push_back(transfer.read_back_record);
        expected_xors.push_back(xor_of(transfer.pattern));
    }

    write_guards(run, guards);

    const D2dErrorSnapshot before = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    std::vector<CceCopyResult> write_results;
    const RunOutcome write_outcome =
        run_launches(run.device, write_launches, write_results, write_records, expected_xors);
    if (write_outcome == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    bool ok = write_outcome == RunOutcome::Passed;
    for (const HartTransfer& transfer : transfers) {
        const std::string write_context = transfer.record.to_string();
        const std::vector<uint32_t> tl1_a =
            read_cce_tl1(run.device, run.source_cce, transfer.tl1_offset, kEveryAnchorBytes);
        ok = expect_words("TL1 A", transfer.pattern, tl1_a, write_context) && ok;
        ok =
            read_dram_and_expect(run.unit_mesh, run.dest_view, transfer.view_offset, transfer.pattern, write_context) &&
            ok;
    }
    if (write_outcome != RunOutcome::Passed) {
        // The write did not finish on every hart, so the read-back would not be ordered after it.
        const D2dErrorSnapshot after_write = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
        expect_no_new_d2d_error_bits(before, after_write, "Anchor " + anchor.name);
        check_guards(run, guards);
        dump_debug_last(run.cluster, run.soc, run.chip);
        return RunOutcome::Failed;
    }

    std::vector<CceCopyResult> read_back_results;
    const RunOutcome read_back_outcome =
        run_launches(run.device, read_back_launches, read_back_results, read_back_records, expected_xors);
    if (read_back_outcome == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    ok = read_back_outcome == RunOutcome::Passed && ok;
    for (const HartTransfer& transfer : transfers) {
        const std::vector<uint32_t> tl1_b =
            read_cce_tl1(run.device, run.source_cce, transfer.tl1_offset + kEveryAnchorBytes, kEveryAnchorBytes);
        ok = expect_words("TL1 B", transfer.pattern, tl1_b, transfer.read_back_record.to_string()) && ok;
    }
    const D2dErrorSnapshot after = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    ok = expect_no_new_d2d_error_bits(before, after, "Anchor " + anchor.name) && ok;
    ok = check_guards(run, guards) && ok;
    if (!ok) {
        dump_debug_last(run.cluster, run.soc, run.chip);
    }
    return ok ? RunOutcome::Passed : RunOutcome::Failed;
}

// One copy to remote DRAM at a size, alignment or window-edge case, then a read of it back into TL1. The 64 B guards
// sit just before and just after the window. Alias decoys of up to 4 KiB sit where a mis-translated copy would land,
// so a misrouted write corrupts one and a misrouted read returns its data. The read leg runs only if the write leg
// passed. It reads into TL1 B on the next hart, with a 64 B guard after it.
RunOutcome check_traffic_case(
    const RemoteDramRun& run,
    uint64_t view_offset,
    uint32_t size_bytes,
    uint32_t case_index,
    uint32_t& next_guard_tag) {
    const uint32_t hart = size_bytes >= kLargeTransferBytes ? 0 : case_index % cce_hart_count();
    const uint32_t words = size_bytes / static_cast<uint32_t>(sizeof(uint32_t));
    const uint64_t chiplet_offset = chiplet_offset_of(run.soc, run.dest_view, view_offset);
    const uint32_t seed = mimir_test_seed() ^ case_index;
    const std::vector<uint32_t> pattern =
        make_pattern(kAllMimirPatterns[case_index % kAllMimirPatterns.size()], words, chiplet_offset, seed);
    const uint64_t tl1_offset = tl1_data(hart);
    const MimirTransferRecord record = make_gddr_record(
        run.source_routes,
        run.soc,
        run.source_mimir,
        run.dest_mimir,
        source_tl1_label(run.source_mimir, run.source_location, hart, tl1_offset),
        dram_label(run.dest_mimir, run.dest_view, view_offset),
        chiplet_offset,
        size_bytes,
        seed);
    log_transfer(record);

    std::vector<GuardRegion> guards;
    add_window_guard(
        record.to_string(), "before window", run.dest_view, view_offset - kGuardBytes, 2 * case_index + 1, guards);
    add_window_guard(
        record.to_string(), "after window", run.dest_view, view_offset + size_bytes, 2 * case_index + 2, guards);
    add_alias_decoys(
        run,
        record.to_string(),
        "traffic",
        chiplet_offset,
        std::min(size_bytes, kEveryAnchorBytes),
        guards,
        next_guard_tag);

    write_cce_tl1(run.device, run.source_cce, tl1_offset, pattern);
    write_dram(run.unit_mesh, run.dest_view, view_offset, make_guard(case_index + 1, words));
    write_guards(run, guards);

    CceCopyLaunch launch;
    launch.cce_index = run.source_cce;
    launch.hart = hart;
    launch.table.add(CopyOp{own_tl1(tl1_offset), gddr(run.dest_mimir, chiplet_offset), words});
    std::vector<CceCopyLaunch> launches{launch};
    std::vector<CceCopyResult> results;
    std::vector<MimirTransferRecord> records{record};
    const std::vector<uint32_t> expected_xors{xor_of(pattern)};
    const D2dErrorSnapshot before = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    const RunOutcome launched = run_launches(run.device, launches, results, records, expected_xors);
    if (launched == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    bool ok = launched == RunOutcome::Passed;
    const D2dErrorSnapshot after = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    ok = expect_no_new_d2d_error_bits(before, after, record.to_string()) && ok;
    ok = read_dram_and_expect(run.unit_mesh, run.dest_view, view_offset, pattern, record.to_string()) && ok;
    ok = check_guards(run, guards) && ok;
    if (!ok) {
        dump_debug_last(run.cluster, run.soc, run.chip);
        return RunOutcome::Failed;
    }

    const uint32_t read_hart = size_bytes >= kLargeTransferBytes ? 0 : (hart + 1) % cce_hart_count();
    const uint64_t tl1_b_offset = tl1_data(read_hart) + kTrafficReadBackTl1Offset;
    const std::vector<uint32_t> tl1_guard = make_guard(next_guard_tag++, kGuardWords);
    write_cce_tl1(run.device, run.source_cce, tl1_b_offset, make_guard(next_guard_tag++, words));
    write_cce_tl1(run.device, run.source_cce, tl1_b_offset + size_bytes, tl1_guard);
    const MimirTransferRecord read_record = make_gddr_record(
        run.source_routes,
        run.soc,
        run.source_mimir,
        run.dest_mimir,
        dram_label(run.dest_mimir, run.dest_view, view_offset),
        source_tl1_label(run.source_mimir, run.source_location, read_hart, tl1_b_offset),
        chiplet_offset,
        size_bytes,
        seed);
    log_transfer(read_record);

    CceCopyLaunch read_launch;
    read_launch.cce_index = run.source_cce;
    read_launch.hart = read_hart;
    read_launch.table.add(CopyOp{gddr(run.dest_mimir, chiplet_offset), own_tl1(tl1_b_offset), words});
    std::vector<CceCopyLaunch> read_launches{read_launch};
    std::vector<CceCopyResult> read_results;
    std::vector<MimirTransferRecord> read_records{read_record};
    const D2dErrorSnapshot read_before = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    const RunOutcome read_launched = run_launches(run.device, read_launches, read_results, read_records, expected_xors);
    if (read_launched == RunOutcome::Threw) {
        return RunOutcome::Threw;
    }
    ok = read_launched == RunOutcome::Passed;
    const D2dErrorSnapshot read_after = snapshot_d2d_errors(run.cluster, run.soc, run.chip);
    ok = expect_no_new_d2d_error_bits(read_before, read_after, read_record.to_string()) && ok;
    ok = expect_words(
             "TL1 B",
             pattern,
             read_cce_tl1(run.device, run.source_cce, tl1_b_offset, size_bytes),
             read_record.to_string()) &&
         ok;
    ok = expect_words(
             "TL1 B guard",
             tl1_guard,
             read_cce_tl1(run.device, run.source_cce, tl1_b_offset + size_bytes, kGuardBytes),
             read_record.to_string()) &&
         ok;
    if (!ok) {
        dump_debug_last(run.cluster, run.soc, run.chip);
    }
    return ok ? RunOutcome::Passed : RunOutcome::Failed;
}

struct MixedTransfer {
    enum class Target { LocalDram, RemoteDram, RemoteTl1 };
    Target target = Target::LocalDram;
    uint32_t source_cce = 0;
    uint32_t hart = 0;
    uint32_t dest_mimir = 0;
    uint32_t dest_view = 0;
    uint64_t dest_view_offset = 0;
    uint64_t chiplet_offset = 0;
    uint32_t dest_cce = 0;
    uint64_t dest_tl1_offset = 0;
    uint64_t source_tl1_offset = 0;
    std::vector<uint32_t> pattern;
    MimirTransferRecord record;
};

// Per CCE: hart 1 local DRAM, hart 2 remote DRAM, hart 3 remote TL1 into the other Mimir's CCE with the same location.
// Every DRAM region is inside the first anchor of its view: offsets are below 100 KiB from UNRESERVED. Local DRAM
// slots are kMixedSlotBytes apart by location. Remote DRAM slots start after them and are kMixedRemoteSlotBytes apart
// by source Mimir and location, so remote writes from different Mimirs never share an offset and the "source Mimir
// same offset" decoy of one sender is not another sender's destination.
std::vector<MixedTransfer> plan_mixed_transfers(
    const metal_SocDescriptor& soc,
    const std::vector<CceSpaRoutes>& routes_by_cce,
    uint64_t unreserved,
    uint32_t num_mimirs,
    uint32_t num_cces) {
    std::vector<MixedTransfer> transfers;
    for (uint32_t cce = 0; cce < num_cces; ++cce) {
        const uint32_t source_mimir = mimir_index_for_cce(cce);
        const uint32_t location = location_for_cce(cce);
        const uint32_t dest_mimir = (source_mimir + 1) % num_mimirs;
        const uint32_t dest_cce = cce_index_for(dest_mimir, location);
        const uint32_t local_view = local_dram_view_for_cce(soc, cce);
        const uint32_t remote_view = base_dram_view_for_mimir(soc, dest_mimir);
        const uint64_t local_offset = unreserved + static_cast<uint64_t>(location) * kMixedSlotBytes;
        const uint64_t remote_offset =
            unreserved + locations_per_channel(soc) * static_cast<uint64_t>(kMixedSlotBytes) +
            static_cast<uint64_t>(cce_index_for(source_mimir, location)) * kMixedRemoteSlotBytes;

        for (uint32_t hart = 1; hart <= 3; ++hart) {
            MixedTransfer transfer;
            transfer.source_cce = cce;
            transfer.hart = hart;
            transfer.source_tl1_offset = tl1_data(hart);
            if (hart == 1) {
                transfer.target = MixedTransfer::Target::LocalDram;
                transfer.dest_mimir = source_mimir;
                transfer.dest_view = local_view;
                transfer.dest_view_offset = local_offset;
                transfer.chiplet_offset = chiplet_offset_of(soc, local_view, local_offset);
            } else if (hart == 2) {
                transfer.target = MixedTransfer::Target::RemoteDram;
                transfer.dest_mimir = dest_mimir;
                transfer.dest_view = remote_view;
                transfer.dest_view_offset = remote_offset;
                transfer.chiplet_offset = chiplet_offset_of(soc, remote_view, remote_offset);
            } else {
                transfer.target = MixedTransfer::Target::RemoteTl1;
                transfer.dest_mimir = dest_mimir;
                transfer.dest_cce = dest_cce;
                transfer.dest_tl1_offset = tl1_remote_landing(cce, hart);
            }
            const uint64_t pattern_base = transfer.target == MixedTransfer::Target::RemoteTl1 ? transfer.dest_tl1_offset
                                                                                              : transfer.chiplet_offset;
            const uint32_t seed = mimir_test_seed() ^ (cce << 8) ^ hart;
            transfer.pattern = make_pattern(
                kAllMimirPatterns[(cce * cce_hart_count() + hart) % kAllMimirPatterns.size()],
                kEveryAnchorWords,
                pattern_base,
                seed);
            if (transfer.target == MixedTransfer::Target::RemoteDram) {
                transfer.record = make_gddr_record(
                    routes_by_cce.at(cce),
                    soc,
                    source_mimir,
                    dest_mimir,
                    source_tl1_label(source_mimir, location, hart, transfer.source_tl1_offset),
                    dram_label(dest_mimir, remote_view, remote_offset),
                    transfer.chiplet_offset,
                    kEveryAnchorBytes,
                    seed);
            } else if (transfer.target == MixedTransfer::Target::RemoteTl1) {
                transfer.record = make_sram_record(
                    soc,
                    routes_by_cce.at(cce),
                    source_tl1_label(source_mimir, location, hart, transfer.source_tl1_offset),
                    fmt::format("M{}.CCE{} TL1 0x{:x}", dest_mimir, location, transfer.dest_tl1_offset),
                    dest_cce,
                    transfer.dest_tl1_offset,
                    kEveryAnchorBytes,
                    seed);
            }
            transfers.push_back(std::move(transfer));
        }
    }
    return transfers;
}

// Local DRAM, remote DRAM and remote TL1 writes on different harts, in both directions, in one program. The remote
// DRAM writes have alias decoys. A second program then reads every destination back into TL1 B.
TEST(MimirRemoteMixed, Concurrent) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_mimirs < 2) {
        GTEST_SKIP() << "Remote DRAM needs two Mimirs; the descriptor has one.";
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const metal_SocDescriptor& soc = soc_desc_of(device);
    Cluster& cluster = MetalContext::instance().get_cluster();
    const ChipId chip = device->id();
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    std::vector<CceSpaRoutes> routes_by_cce;
    for (uint32_t cce = 0; cce < topology.num_cces; ++cce) {
        routes_by_cce.push_back(mimir_contract::read_cce_spa_routes(device, cce));
    }
    std::vector<MixedTransfer> transfers =
        plan_mixed_transfers(soc, routes_by_cce, unreserved, topology.num_mimirs, topology.num_cces);

    uint32_t next_guard_tag = 1;
    std::vector<GuardRegion> decoys;
    std::vector<CceCopyLaunch> launches;
    std::vector<MimirTransferRecord> records;
    std::vector<uint32_t> expected_xors;
    for (MixedTransfer& transfer : transfers) {
        write_cce_tl1(device, transfer.source_cce, transfer.source_tl1_offset, transfer.pattern);
        const std::vector<uint32_t> guard = make_guard(next_guard_tag++, kEveryAnchorWords);
        if (transfer.target == MixedTransfer::Target::RemoteTl1) {
            write_cce_tl1(device, transfer.dest_cce, transfer.dest_tl1_offset, guard);
        } else {
            mimir_contract::write_gddr_view(device_closer.mesh(), transfer.dest_view, transfer.dest_view_offset, guard);
        }
        if (transfer.target != MixedTransfer::Target::LocalDram) {
            log_transfer(transfer.record);
            records.push_back(transfer.record);
        }
        if (transfer.target == MixedTransfer::Target::RemoteDram) {
            const RemoteDramRun run = make_run(
                device_closer.mesh(),
                device,
                mimir_index_for_cce(transfer.source_cce),
                location_for_cce(transfer.source_cce),
                transfer.dest_view);
            add_alias_decoys(
                run,
                transfer.record.to_string(),
                fmt::format("CCE {} hart {}", transfer.source_cce, transfer.hart),
                transfer.chiplet_offset,
                kEveryAnchorBytes,
                decoys,
                next_guard_tag);
        }

        CceCopyLaunch launch;
        launch.cce_index = transfer.source_cce;
        launch.hart = transfer.hart;
        if (transfer.target == MixedTransfer::Target::RemoteTl1) {
            launch.table.add(CopyOp{
                own_tl1(transfer.source_tl1_offset),
                sram_spa(transfer.dest_cce, transfer.dest_tl1_offset),
                kEveryAnchorWords});
        } else {
            launch.table.add(CopyOp{
                own_tl1(transfer.source_tl1_offset),
                gddr(transfer.dest_mimir, transfer.chiplet_offset),
                kEveryAnchorWords});
        }
        launches.push_back(std::move(launch));
        expected_xors.push_back(xor_of(transfer.pattern));
    }

    for (const GuardRegion& decoy : decoys) {
        for (const MixedTransfer& transfer : transfers) {
            if (transfer.target == MixedTransfer::Target::RemoteTl1) {
                continue;
            }
            const bool overlaps =
                decoy.view == transfer.dest_view && decoy.view_offset < transfer.dest_view_offset + kEveryAnchorBytes &&
                transfer.dest_view_offset < decoy.view_offset + decoy.expected.size() * sizeof(uint32_t);
            TT_FATAL(!overlaps, "Decoy {} overlaps the destination of {}", decoy.label, transfer.record.to_string());
        }
    }
    if (!decoys.empty()) {
        const RemoteDramRun decoy_run = make_run(device_closer.mesh(), device, 0, 0, decoys.front().view);
        write_guards(decoy_run, decoys);
    }

    const D2dErrorSnapshot before = snapshot_d2d_errors(cluster, soc, chip);
    std::vector<CceCopyResult> results;
    const RunOutcome launched = run_launches(device, launches, results, records, expected_xors);
    if (launched == RunOutcome::Threw) {
        return;
    }
    bool ok = launched == RunOutcome::Passed;
    const D2dErrorSnapshot after = snapshot_d2d_errors(cluster, soc, chip);
    ok = expect_no_new_d2d_error_bits(before, after, "MimirRemoteMixed.Concurrent") && ok;
    for (const MixedTransfer& transfer : transfers) {
        if (transfer.target == MixedTransfer::Target::RemoteTl1) {
            const std::vector<uint32_t> actual =
                read_cce_tl1(device, transfer.dest_cce, transfer.dest_tl1_offset, kEveryAnchorBytes);
            ok = expect_words("remote TL1", transfer.pattern, actual, transfer.record.to_string()) && ok;
        } else {
            const std::string context =
                transfer.target == MixedTransfer::Target::LocalDram
                    ? fmt::format("local DRAM from CCE {} hart {}", transfer.source_cce, transfer.hart)
                    : transfer.record.to_string();
            ok = read_dram_and_expect(
                     device_closer.mesh(), transfer.dest_view, transfer.dest_view_offset, transfer.pattern, context) &&
                 ok;
        }
    }
    if (!decoys.empty()) {
        const RemoteDramRun decoy_run = make_run(device_closer.mesh(), device, 0, 0, decoys.front().view);
        ok = check_guards(decoy_run, decoys) && ok;
    }
    if (!ok) {
        dump_debug_last(cluster, soc, chip);
        return;
    }

    std::vector<CceCopyLaunch> read_launches;
    std::vector<MimirTransferRecord> read_records;
    std::vector<uint32_t> read_expected_xors;
    for (const MixedTransfer& transfer : transfers) {
        const uint64_t tl1_b_offset = transfer.source_tl1_offset + kEveryAnchorBytes;
        const uint32_t source_mimir = mimir_index_for_cce(transfer.source_cce);
        const uint32_t source_location = location_for_cce(transfer.source_cce);
        write_cce_tl1(device, transfer.source_cce, tl1_b_offset, make_guard(next_guard_tag++, kEveryAnchorWords));

        CceCopyLaunch launch;
        launch.cce_index = transfer.source_cce;
        launch.hart = transfer.hart;
        if (transfer.target == MixedTransfer::Target::RemoteTl1) {
            launch.table.add(CopyOp{
                sram_spa(transfer.dest_cce, transfer.dest_tl1_offset), own_tl1(tl1_b_offset), kEveryAnchorWords});
            const MimirTransferRecord read_record = make_sram_record(
                soc,
                routes_by_cce.at(transfer.source_cce),
                fmt::format("M{}.CCE{} TL1 0x{:x}", transfer.dest_mimir, source_location, transfer.dest_tl1_offset),
                source_tl1_label(source_mimir, source_location, transfer.hart, tl1_b_offset),
                transfer.dest_cce,
                transfer.dest_tl1_offset,
                kEveryAnchorBytes,
                mimir_test_seed() ^ (transfer.source_cce << 8) ^ transfer.hart);
            log_transfer(read_record);
            read_records.push_back(read_record);
        } else {
            launch.table.add(
                CopyOp{gddr(transfer.dest_mimir, transfer.chiplet_offset), own_tl1(tl1_b_offset), kEveryAnchorWords});
            if (transfer.target == MixedTransfer::Target::RemoteDram) {
                const MimirTransferRecord read_record = make_gddr_record(
                    routes_by_cce.at(transfer.source_cce),
                    soc,
                    source_mimir,
                    transfer.dest_mimir,
                    dram_label(transfer.dest_mimir, transfer.dest_view, transfer.dest_view_offset),
                    source_tl1_label(source_mimir, source_location, transfer.hart, tl1_b_offset),
                    transfer.chiplet_offset,
                    kEveryAnchorBytes,
                    mimir_test_seed() ^ (transfer.source_cce << 8) ^ transfer.hart);
                log_transfer(read_record);
                read_records.push_back(read_record);
            }
        }
        read_launches.push_back(std::move(launch));
        read_expected_xors.push_back(xor_of(transfer.pattern));
    }

    const D2dErrorSnapshot read_before = snapshot_d2d_errors(cluster, soc, chip);
    std::vector<CceCopyResult> read_results;
    const RunOutcome read_launched =
        run_launches(device, read_launches, read_results, read_records, read_expected_xors);
    if (read_launched == RunOutcome::Threw) {
        return;
    }
    ok = read_launched == RunOutcome::Passed;
    const D2dErrorSnapshot read_after = snapshot_d2d_errors(cluster, soc, chip);
    ok = expect_no_new_d2d_error_bits(read_before, read_after, "MimirRemoteMixed.Concurrent read-back") && ok;
    for (const MixedTransfer& transfer : transfers) {
        const std::vector<uint32_t> actual = read_cce_tl1(
            device, transfer.source_cce, transfer.source_tl1_offset + kEveryAnchorBytes, kEveryAnchorBytes);
        ok = expect_words(
                 "TL1 B",
                 transfer.pattern,
                 actual,
                 fmt::format("read-back on CCE {} hart {}", transfer.source_cce, transfer.hart)) &&
             ok;
    }
    if (!ok) {
        dump_debug_last(cluster, soc, chip);
    }
}

class MimirRemoteDram : public ::testing::TestWithParam<RemoteDramViewParam> {};

TEST_P(MimirRemoteDram, WriteFromTl1) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const RemoteDramViewParam param = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const std::string skip_reason = view_skip_reason(topology, param);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const RemoteDramRun run =
        make_run(device_closer.mesh(), device, param.source_mimir, param.source_location, param.view);
    const std::vector<MimirAnchor> anchors = mimir_dram_anchors(run.soc, run.dest_view, run.unreserved, false);
    uint32_t next_guard_tag = 1;
    for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
        if (write_from_tl1_anchor(run, anchors[anchor_index], anchor_index, next_guard_tag) == RunOutcome::Threw) {
            return;
        }
    }
}

TEST_P(MimirRemoteDram, ReadToTl1) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const RemoteDramViewParam param = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const std::string skip_reason = view_skip_reason(topology, param);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const RemoteDramRun run =
        make_run(device_closer.mesh(), device, param.source_mimir, param.source_location, param.view);
    const std::vector<MimirAnchor> anchors = mimir_dram_anchors(run.soc, run.dest_view, run.unreserved, false);
    uint32_t next_guard_tag = 1;
    for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
        if (read_to_tl1_anchor(run, anchors[anchor_index], anchor_index, next_guard_tag) == RunOutcome::Threw) {
            return;
        }
    }
}

TEST_P(MimirRemoteDram, Tl1RoundTrip) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const RemoteDramViewParam param = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const std::string skip_reason = view_skip_reason(topology, param);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const RemoteDramRun run =
        make_run(device_closer.mesh(), device, param.source_mimir, param.source_location, param.view);
    const std::vector<MimirAnchor> anchors = mimir_dram_anchors(run.soc, run.dest_view, run.unreserved, false);
    uint32_t next_guard_tag = 1;
    for (uint32_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
        if (round_trip_anchor(run, anchors[anchor_index], anchor_index, next_guard_tag) == RunOutcome::Threw) {
            return;
        }
    }
}

class MimirRemoteDramTrafficMatrix : public ::testing::TestWithParam<RemoteDramSourceParam> {};

// One anchor per remote view, every size and alignment case. Each window starts and ends at least 64 B inside the
// anchor, so its guards are inside the anchor too.
TEST_P(MimirRemoteDramTrafficMatrix, TrafficMatrix) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const RemoteDramSourceParam param = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const std::string skip_reason = source_skip_reason(topology, param.source_mimir, param.source_location);
    if (!skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    uint32_t case_index = 0;
    uint32_t next_guard_tag = 1;
    for (uint32_t view = 0; view < topology.num_views; ++view) {
        if (topology.view_mimir[view] == param.source_mimir) {
            continue;
        }
        const RemoteDramRun run =
            make_run(device_closer.mesh(), device, param.source_mimir, param.source_location, view);
        const std::vector<MimirAnchor> anchors = mimir_dram_anchors(run.soc, view, run.unreserved, false);
        const MimirAnchor& anchor = anchor_named(anchors, "unres_g0");
        for (uint32_t size_bytes : kTrafficSizeBytes) {
            for (uint32_t alignment : kTrafficAlignments) {
                if (check_traffic_case(
                        run, anchor.view_offset + kGuardBytes + alignment, size_bytes, case_index++, next_guard_tag) ==
                    RunOutcome::Threw) {
                    return;
                }
            }
            if (check_traffic_case(
                    run,
                    anchor.view_offset + kAnchorBytes - kGuardBytes - size_bytes,
                    size_bytes,
                    case_index++,
                    next_guard_tag) == RunOutcome::Threw) {
                return;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM6, MimirRemoteDram, ::testing::ValuesIn(remote_dram_view_params()), remote_dram_view_name);
INSTANTIATE_TEST_SUITE_P(
    MM6, MimirRemoteDramTrafficMatrix, ::testing::ValuesIn(remote_dram_source_params()), remote_dram_source_name);

}  // namespace tt::tt_metal::mimir_test::remote_dram
