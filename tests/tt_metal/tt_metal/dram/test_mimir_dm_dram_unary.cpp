// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 port of data_movement/dram_unary (D2): a reader hart copies GDDR into a TL1 staging ring and a writer hart on
// the same CCE copies it back out to GDDR, both with SPA loads and stores, in one Program. Chunks pass between the
// harts through two DRAM-core semaphores read and written through the uncached alias. The host seeds the source and
// fills both semaphore words with a nonzero value, waits for both status lines with a wall-clock deadline, and checks
// the destination, the staging ring, the guards on both sides of each, and the semaphores: both harts must resolve the
// runtime's semaphore addresses, see them initialised to 0 at launch, and leave them at the chunk count.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_dram_unary_abi.h"

namespace tt::tt_metal::mimir_test::dm_dram_unary {

namespace abi = cce_dram_unary_abi;

inline constexpr const char* kReaderKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_dram_unary_reader.cpp";
inline constexpr const char* kWriterKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_dram_unary_writer.cpp";
inline constexpr uint32_t kReaderHart = 0;
inline constexpr uint32_t kWriterHart = 1;
// Upstream's Quasar directed case is 4 transactions; two slots make the reader wait on drained for chunks 2 and 3.
inline constexpr uint32_t kChunkBytes = 1024;
inline constexpr uint32_t kNumChunks = 4;
inline constexpr uint32_t kNumSlots = 2;
inline constexpr uint32_t kTotalBytes = kChunkBytes * kNumChunks;
inline constexpr uint32_t kStagingBytes = kChunkBytes * kNumSlots;
// Each chunk costs at most one wait per hart, so the harts finish within kNumChunks * 2 budgets; the slack covers
// launch and the copies.
inline constexpr std::chrono::seconds kLaunchSlack{60};
inline constexpr std::chrono::seconds kHostDeadline = kNumChunks * 2 * kPollBudgetHostTime + kLaunchSlack;
// Above DRAM UNRESERVED in the CCE's local view.
inline constexpr uint64_t kSourceViewOffset = 0x10000;
inline constexpr uint64_t kDestinationViewOffset = 0x20000;
// Written to both semaphore words before launch; the runtime must overwrite it with the initial value 0.
inline constexpr uint32_t kSemaphorePrefill = 0x5E3A5E3Au;
inline constexpr uint32_t kLeadingGuardTag = 0x800;

static_assert(kChunkBytes % abi::kLineBytes == 0 && kChunkBytes <= abi::kMaxChunkBytes);
static_assert(kNumSlots <= abi::kMaxSlots && kNumChunks <= abi::kMaxChunks);
static_assert(kSourceViewOffset + kTotalBytes <= kDestinationViewOffset);

std::string dram_unary_skip_reason(uint32_t cce_index) {
    if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
        return watcher_reason;
    }
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (cce_index >= topology.num_cces) {
        return fmt::format("Descriptor has {} CCEs; CCE {} is not present.", topology.num_cces, cce_index);
    }
    return "";
}

struct DramUnaryLayout {
    uint32_t cce_index = 0;
    uint32_t local_mimir = 0;
    uint32_t view = 0;
    uint64_t source_view_offset = 0;
    uint64_t destination_view_offset = 0;
    uint64_t source_chiplet_offset = 0;
    uint64_t destination_chiplet_offset = 0;
    uint64_t staging_offset = 0;
};

DramUnaryLayout layout_for(const metal_SocDescriptor& soc, uint32_t cce_index) {
    const uint64_t dram_unreserved = dram_unreserved_base();
    DramUnaryLayout layout;
    layout.cce_index = cce_index;
    layout.local_mimir = mimir_index_for_cce(cce_index);
    layout.view = local_dram_view_for_cce(soc, cce_index);
    layout.source_view_offset = dram_unreserved + kSourceViewOffset;
    layout.destination_view_offset = dram_unreserved + kDestinationViewOffset;
    layout.source_chiplet_offset = chiplet_offset_of(soc, layout.view, layout.source_view_offset);
    layout.destination_chiplet_offset = chiplet_offset_of(soc, layout.view, layout.destination_view_offset);
    layout.staging_offset = tl1_data(kReaderHart);
    return layout;
}

// Semaphore ids for filled and drained, one line apart. Ids are L1-alignment slots, so the ids between them are created
// as spacers.
struct SemaphoreIds {
    uint32_t filled = 0;
    uint32_t drained = 0;
};

uint32_t semaphore_ids_per_line() {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    TT_FATAL(
        l1_alignment != 0 && abi::kLineBytes % l1_alignment == 0,
        "L1 alignment {} does not divide the {} B line",
        l1_alignment,
        abi::kLineBytes);
    return abi::kLineBytes / l1_alignment;
}

std::vector<uint32_t> kernel_args(
    const DramUnaryLayout& layout, const SemaphoreIds& semaphores, uint64_t status_offset) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(status_offset);
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgLocalMimir] = layout.local_mimir;
    args[abi::kArgSourceOffsetLo] = static_cast<uint32_t>(layout.source_chiplet_offset);
    args[abi::kArgSourceOffsetHi] = static_cast<uint32_t>(layout.source_chiplet_offset >> 32);
    args[abi::kArgDestinationOffsetLo] = static_cast<uint32_t>(layout.destination_chiplet_offset);
    args[abi::kArgDestinationOffsetHi] = static_cast<uint32_t>(layout.destination_chiplet_offset >> 32);
    args[abi::kArgStagingOffset] = static_cast<uint32_t>(layout.staging_offset);
    args[abi::kArgChunkBytes] = kChunkBytes;
    args[abi::kArgNumChunks] = kNumChunks;
    args[abi::kArgNumSlots] = kNumSlots;
    args[abi::kArgFilledSemaphoreId] = semaphores.filled;
    args[abi::kArgDrainedSemaphoreId] = semaphores.drained;
    return args;
}

// Creates the two DRAM-core semaphores (initial value 0) on the CCE in `program`.
SemaphoreIds create_semaphores(IDevice* device, Program& program, const DramUnaryLayout& layout) {
    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), layout.cce_index);
    const CoreRange core_range(logical_core, logical_core);
    std::vector<uint32_t> semaphore_ids;
    for (uint32_t index = 0; index <= semaphore_ids_per_line(); ++index) {
        semaphore_ids.push_back(CreateSemaphore(program, core_range, 0, CoreType::DRAM));
    }
    const SemaphoreIds semaphores{.filled = semaphore_ids.front(), .drained = semaphore_ids.back()};
    TT_FATAL(
        semaphores.drained - semaphores.filled == semaphore_ids_per_line(),
        "semaphore ids {} and {} are not one line apart",
        semaphores.filled,
        semaphores.drained);
    return semaphores;
}

std::vector<CceKernelLaunch> launches_for(const DramUnaryLayout& layout, const SemaphoreIds& semaphores) {
    return {
        CceKernelLaunch{
            .cce_index = layout.cce_index,
            .kernel_path = kReaderKernel,
            .runtime_args = kernel_args(layout, semaphores, tl1_done_block(kReaderHart)),
            .hart = kReaderHart},
        CceKernelLaunch{
            .cce_index = layout.cce_index,
            .kernel_path = kWriterKernel,
            .runtime_args = kernel_args(layout, semaphores, tl1_done_block(kWriterHart)),
            .hart = kWriterHart}};
}

// Where the runtime writes the semaphore's initial value at launch (ProgramImpl::init_semaphores). Valid once the
// Program's offsets are finalized.
uint64_t semaphore_tl1_offset(Program& program, uint32_t semaphore_id) {
    const Hal& hal = MetalContext::instance().hal();
    const uint32_t dram_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM);
    return hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG) +
           program.impl().get_program_config(dram_index).sem_offset +
           static_cast<uint64_t>(semaphore_id) * hal.get_alignment(HalMemType::L1);
}

uint32_t read_semaphore(IDevice* device, uint32_t cce_index, uint64_t tl1_offset) {
    return read_cce_tl1(device, cce_index, tl1_offset, sizeof(uint32_t), Tl1Range::WholeTl1).front();
}

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeDone: return "done";
        case abi::kCodeWaitTimeout: return "wait_timeout";
        case abi::kCodeBadArgs: return "bad_args";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    return fmt::format(
        "kernel code {}, chunks done {}/{}, timed-out chunk {} (semaphore observed {}), max wait cycles {}, semaphores "
        "filled 0x{:x} drained 0x{:x}, at start filled {} drained {}",
        code_name(status[abi::kStatusCode]),
        status[abi::kStatusChunksDone],
        kNumChunks,
        status[abi::kStatusTimedOutChunk],
        status[abi::kStatusSemaphoreObserved],
        status[abi::kStatusMaxWaitCycles],
        status[abi::kStatusFilledSemaphoreAddress],
        status[abi::kStatusDrainedSemaphoreAddress],
        status[abi::kStatusFilledAtStart],
        status[abi::kStatusDrainedAtStart]);
}

std::string gddr_label(
    const CceSpaRoutes& routes, const DramUnaryLayout& layout, uint64_t view_offset, uint64_t chiplet_offset) {
    return fmt::format(
        "GDDR view {} (M{}) +0x{:x} (chiplet 0x{:x}, SPA 0x{:x})",
        layout.view,
        layout.local_mimir,
        view_offset,
        chiplet_offset,
        mimir_contract::cce_gddr_address(routes, layout.local_mimir, chiplet_offset));
}

std::string staging_label(const DramUnaryLayout& layout) {
    return fmt::format(
        "TL1 staging 0x{:x}, {} slots of {} B (CCE{})",
        layout.staging_offset,
        kNumSlots,
        kChunkBytes,
        layout.cce_index);
}

// The ring after the last chunk: slot s holds the last chunk c with c % kNumSlots == s.
std::vector<uint32_t> expected_staging(const std::vector<uint32_t>& source) {
    const uint32_t chunk_words = kChunkBytes / sizeof(uint32_t);
    std::vector<uint32_t> staging(kStagingBytes / sizeof(uint32_t));
    for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
        std::copy_n(
            source.begin() + chunk * chunk_words, chunk_words, staging.begin() + (chunk % kNumSlots) * chunk_words);
    }
    return staging;
}

std::vector<uint32_t> read_destination(
    distributed::MeshDevice& unit_mesh, const DramUnaryLayout& layout, uint64_t byte_offset, uint32_t bytes) {
    std::vector<uint32_t> words;
    mimir_contract::read_gddr_view(unit_mesh, layout.view, layout.destination_view_offset + byte_offset, bytes, words);
    return words;
}

class MimirDmDramUnary : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirDmDramUnary, ReaderWriterHandshake) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    if (const std::string skip_reason = dram_unary_skip_reason(cce_index); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();

    const metal_SocDescriptor& soc = soc_desc_of(device);
    record_descriptor_properties(soc);

    const DramUnaryLayout layout = layout_for(soc, cce_index);
    const uint32_t seed = mimir_test_seed() ^ (cce_index * 0x9E3779B9u);
    const std::vector<uint32_t> source =
        make_pattern(MimirPattern::Prng, kTotalBytes / sizeof(uint32_t), layout.source_chiplet_offset, seed);
    const std::vector<uint32_t> destination_guard = make_guard(2 * cce_index);
    const std::vector<uint32_t> staging_guard = make_guard(2 * cce_index + 1);
    const std::vector<uint32_t> destination_leading_guard = make_guard(kLeadingGuardTag + 2 * cce_index);
    const std::vector<uint32_t> staging_leading_guard = make_guard(kLeadingGuardTag + 2 * cce_index + 1);
    std::vector<uint32_t> destination_fill = make_seed_pattern(2 * cce_index, kTotalBytes / sizeof(uint32_t));
    destination_fill.insert(destination_fill.end(), destination_guard.begin(), destination_guard.end());
    std::vector<uint32_t> staging_fill = make_seed_pattern(2 * cce_index + 1, kStagingBytes / sizeof(uint32_t));
    staging_fill.insert(staging_fill.end(), staging_guard.begin(), staging_guard.end());
    mimir_contract::write_gddr_view(device_closer.mesh(), layout.view, layout.source_view_offset, source);
    mimir_contract::write_gddr_view(
        device_closer.mesh(), layout.view, layout.destination_view_offset, destination_fill);
    mimir_contract::write_gddr_view(
        device_closer.mesh(), layout.view, layout.destination_view_offset - kGuardBytes, destination_leading_guard);
    write_cce_tl1(device, cce_index, layout.staging_offset, staging_fill);
    write_cce_tl1(device, cce_index, layout.staging_offset - kGuardBytes, staging_leading_guard);
    for (const uint32_t hart : {kReaderHart, kWriterHart}) {
        write_cce_tl1(
            device, cce_index, tl1_done_block(hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
    }

    const CceSpaRoutes routes = mimir_contract::read_cce_spa_routes(device, layout.cce_index);
    const std::string source_label =
        gddr_label(routes, layout, layout.source_view_offset, layout.source_chiplet_offset);
    const std::string destination_label =
        gddr_label(routes, layout, layout.destination_view_offset, layout.destination_chiplet_offset);
    const CoreCoord translated_core = translated_dram_core(soc, cce_index);
    const std::string label = fmt::format(
        "CCE{} reader hart {} / writer hart {}: {} chunks of {} B, {} -> {} -> {}",
        cce_index,
        kReaderHart,
        kWriterHart,
        kNumChunks,
        kChunkBytes,
        source_label,
        staging_label(layout),
        destination_label);
    for (const auto& [leg, hart, leg_source, leg_destination, kernel] :
         {std::tuple{"reader", kReaderHart, source_label, staging_label(layout), kReaderKernel},
          std::tuple{"writer", kWriterHart, staging_label(layout), destination_label, kWriterKernel}}) {
        record_leg(leg, "source", leg_source);
        record_leg(leg, "destination", leg_destination);
        record_leg(
            leg,
            "core",
            fmt::format("CCE{} hart {} translated ({}, {})", cce_index, hart, translated_core.x, translated_core.y));
        record_leg(leg, "length", fmt::format("{} chunks of {} B", kNumChunks, kChunkBytes));
        record_leg(
            leg,
            "alignment",
            fmt::format(
                "source {} destination {} staging {}",
                layout.source_chiplet_offset % abi::kLineBytes,
                layout.destination_chiplet_offset % abi::kLineBytes,
                layout.staging_offset % abi::kLineBytes));
        record_leg(
            leg, "mode", fmt::format("SPA loads/stores, {} slots, DRAM-core semaphores via uncached alias", kNumSlots));
        record_leg(leg, "seed", fmt::format("0x{:08x}", seed));
        record_leg(leg, "kernel", kernel);
    }

    Program program = CreateProgram();
    const SemaphoreIds semaphores = create_semaphores(device, program, layout);
    const std::vector<CceKernelLaunch> launches = launches_for(layout, semaphores);
    add_cce_kernels(device, program, launches);
    program.impl().compile(device, /*force_slow_dispatch=*/true);
    program.impl().finalize_offsets(device);
    const uint64_t filled_offset = semaphore_tl1_offset(program, semaphores.filled);
    const uint64_t drained_offset = semaphore_tl1_offset(program, semaphores.drained);
    for (const uint64_t semaphore_offset : {filled_offset, drained_offset}) {
        write_cce_tl1(device, cce_index, semaphore_offset, {kSemaphorePrefill}, Tl1Range::WholeTl1);
    }
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = launch_and_wait_for_statuses(
        device, program, launches, {abi::kReaderDoneMagic, abi::kWriterDoneMagic}, abi::kStatusWords, kHostDeadline);
    if (!statuses.has_value()) {
        const std::vector<uint32_t> reader_line =
            read_cce_tl1(device, cce_index, tl1_done_block(kReaderHart), abi::kStatusWords * sizeof(uint32_t));
        const std::vector<uint32_t> writer_line =
            read_cce_tl1(device, cce_index, tl1_done_block(kWriterHart), abi::kStatusWords * sizeof(uint32_t));
        const std::string timeout = fmt::format(
            "no status after {} s (host deadline): reader magic 0x{:08x} ({}); writer magic 0x{:08x} ({}); semaphores "
            "now filled 0x{:x} drained 0x{:x}",
            kHostDeadline.count(),
            reader_line[abi::kStatusMagic],
            status_detail(reader_line),
            writer_line[abi::kStatusMagic],
            status_detail(writer_line),
            read_semaphore(device, cce_index, filled_offset),
            read_semaphore(device, cce_index, drained_offset));
        record_leg("reader", "status", timeout);
        record_leg("writer", "status", timeout);
        ADD_FAILURE() << label << ": " << timeout << kFreshSessionNote;
        return;
    }

    const std::vector<uint32_t>& reader_status = statuses->at(0);
    const std::vector<uint32_t>& writer_status = statuses->at(1);
    const std::string reader_detail = status_detail(reader_status);
    const std::string writer_detail = status_detail(writer_status);
    record_leg("reader", "status", reader_detail);
    record_leg("writer", "status", writer_detail);
    log_info(tt::LogTest, "MM8 dram_unary {}: reader {}; writer {}", label, reader_detail, writer_detail);

    EXPECT_EQ(reader_status[abi::kStatusCode], abi::kCodeDone) << label << ": reader " << reader_detail;
    EXPECT_EQ(writer_status[abi::kStatusCode], abi::kCodeDone) << label << ": writer " << writer_detail;
    EXPECT_EQ(reader_status[abi::kStatusChunksDone], kNumChunks) << label << ": reader " << reader_detail;
    EXPECT_EQ(writer_status[abi::kStatusChunksDone], kNumChunks) << label << ": writer " << writer_detail;
    EXPECT_EQ(reader_status[abi::kStatusFilledAtStart], 0u) << label << ": filled was not 0 at launch";
    EXPECT_EQ(reader_status[abi::kStatusDrainedAtStart], 0u) << label << ": drained was not 0 at launch";
    EXPECT_EQ(writer_status[abi::kStatusDrainedAtStart], 0u) << label << ": drained was not 0 at launch";
    for (const auto& [hart_name, status] : {std::pair{"reader", &reader_status}, std::pair{"writer", &writer_status}}) {
        EXPECT_EQ((*status)[abi::kStatusFilledSemaphoreAddress], static_cast<uint32_t>(filled_offset))
            << label << ": the " << hart_name << " resolved filled away from the runtime's semaphore";
        EXPECT_EQ((*status)[abi::kStatusDrainedSemaphoreAddress], static_cast<uint32_t>(drained_offset))
            << label << ": the " << hart_name << " resolved drained away from the runtime's semaphore";
    }
    EXPECT_EQ(read_semaphore(device, cce_index, filled_offset), kNumChunks) << label << ": filled after the run";
    EXPECT_EQ(read_semaphore(device, cce_index, drained_offset), kNumChunks) << label << ": drained after the run";
    EXPECT_TRUE(compare_with_recheck(
        "dram_unary " + label + " destination",
        "writer",
        source,
        [&device_closer, &layout](uint64_t byte_offset, uint32_t bytes) {
            return read_destination(device_closer.mesh(), layout, byte_offset, bytes);
        }))
        << writer_detail;
    EXPECT_EQ(read_destination(device_closer.mesh(), layout, kTotalBytes, kGuardBytes), destination_guard)
        << label << ": guard after the destination was overwritten";
    std::vector<uint32_t> destination_leading;
    mimir_contract::read_gddr_view(
        device_closer.mesh(),
        layout.view,
        layout.destination_view_offset - kGuardBytes,
        kGuardBytes,
        destination_leading);
    EXPECT_EQ(destination_leading, destination_leading_guard)
        << label << ": guard before the destination was overwritten";
    const std::vector<uint32_t> staging = read_cce_tl1(device, cce_index, layout.staging_offset, kStagingBytes);
    EXPECT_TRUE(compare_words(label + " staging ring", expected_staging(source), staging)) << reader_detail;
    EXPECT_EQ(read_cce_tl1(device, cce_index, layout.staging_offset + kStagingBytes, kGuardBytes), staging_guard)
        << label << ": guard after the staging ring was overwritten";
    EXPECT_EQ(read_cce_tl1(device, cce_index, layout.staging_offset - kGuardBytes, kGuardBytes), staging_leading_guard)
        << label << ": guard before the staging ring was overwritten";
}

INSTANTIATE_TEST_SUITE_P(MM8, MimirDmDramUnary, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::dm_dram_unary
