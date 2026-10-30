// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "dev_mem_map.h"
#include "hostdev/cce_spa_routes.h"
#include "umd/device/utils/error_detail.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_fw_loader.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::host_tl1 {

constexpr std::array<uint32_t, 5> kVisibilitySizes = {4, 16, 64, 4096, 32768};
constexpr uint32_t kVisibilityPoisonWord = 0xBADC0DE5;
constexpr uint32_t kBoundaryWindowBytes = 16;
constexpr uint32_t kBoundaryTagsPerCce = 256;
constexpr uint32_t kVisibilityGuardTagBase = 0x400;
constexpr uint32_t kRejectSentinelTagBase = 0x800;

uint64_t cce_tl1_begin() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
}

uint64_t cce_tl1_end() {
    const Hal& hal = MetalContext::instance().hal();
    return hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE) +
           hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
}

struct Tl1Region {
    std::string name;
    uint64_t begin;
    uint64_t size;
};

struct HalDramL1Region {
    std::string name;
    HalL1MemAddrType type;
    uint64_t begin;
    uint64_t size;
};

// The CCE TL1 block in address order, from the HAL where it records the region and from dev_mem_map.h for the sizes it
// does not. Used for the overlap and containment checks.
std::vector<Tl1Region> cce_tl1_regions() {
    const Hal& hal = MetalContext::instance().hal();
    const uint32_t harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    const HalJitBuildConfig& jit_build_config =
        hal.get_jit_build_config(hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM), 0, 0);
    return {
        {"reserved (cached soft semaphores)", cce_tl1_begin(), MEM_CCE_RESERVED_SIZE},
        {"mailbox",
         hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX),
         hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX)},
        {"firmware image", jit_build_config.fw_base_addr, MEM_CCE_FIRMWARE_SIZE},
        {"global (firmware and per-hart kernel data)",
         MEM_CCE_GLOBAL_BASE,
         static_cast<uint64_t>(MEM_CCE_GLOBAL_SIZE) * (harts + 1)},
        {"local (per-hart TLS)", MEM_CCE_LOCAL_BASE, static_cast<uint64_t>(MEM_CCE_LOCAL_SIZE) * harts},
        {"init local scratch", jit_build_config.local_init_addr, MEM_CCE_INIT_LOCAL_L1_SCRATCH_SIZE},
        {"bank-to-NOC scratch",
         hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH),
         hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH)},
        {"CCE SPA routes", MEM_CCE_SPA_ROUTES_BASE, MEM_CCE_SPA_ROUTES_SIZE},
        {"kernel config",
         hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG),
         hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG)},
        {"kernel", MEM_CCE_KERNEL_BASE, static_cast<uint64_t>(MEM_CCE_KERNEL_SIZE) * harts},
    };
}

// The DRAM-core L1 types the HAL defines, read from MetalContext without initialize().
std::vector<HalDramL1Region> dram_hal_l1_regions(const Hal& hal) {
    const std::vector<std::pair<std::string, HalL1MemAddrType>> types = {
        {"BASE", HalL1MemAddrType::BASE},
        {"BARRIER", HalL1MemAddrType::BARRIER},
        {"MAILBOX", HalL1MemAddrType::MAILBOX},
        {"LAUNCH", HalL1MemAddrType::LAUNCH},
        {"WATCHER", HalL1MemAddrType::WATCHER},
        {"DPRINT_BUFFERS", HalL1MemAddrType::DPRINT_BUFFERS},
        {"PROFILER", HalL1MemAddrType::PROFILER},
        {"KERNEL_CONFIG", HalL1MemAddrType::KERNEL_CONFIG},
        {"UNRESERVED", HalL1MemAddrType::UNRESERVED},
        {"CORE_INFO", HalL1MemAddrType::CORE_INFO},
        {"GO_MSG", HalL1MemAddrType::GO_MSG},
        {"GO_MSG_INDEX", HalL1MemAddrType::GO_MSG_INDEX},
        {"LAUNCH_MSG_BUFFER_RD_PTR", HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR},
        {"BANK_TO_NOC_SCRATCH", HalL1MemAddrType::BANK_TO_NOC_SCRATCH},
    };
    std::vector<HalDramL1Region> regions;
    for (const std::pair<std::string, HalL1MemAddrType>& type : types) {
        regions.push_back(
            {type.first,
             type.second,
             hal.get_dev_addr(HalProgrammableCoreType::DRAM, type.second),
             hal.get_dev_size(HalProgrammableCoreType::DRAM, type.second)});
    }
    return regions;
}

const HalDramL1Region& hal_region_of(const std::vector<HalDramL1Region>& regions, HalL1MemAddrType type) {
    for (const HalDramL1Region& region : regions) {
        if (region.type == type) {
            return region;
        }
    }
    throw std::logic_error("HAL DRAM L1 region missing from the table");
}

class MimirHostTl1Cce : public ::testing::TestWithParam<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(MM1, MimirHostTl1Cce, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

struct BoundaryWindow {
    uint32_t cce_index;
    uint64_t tl1_offset;
    std::vector<uint32_t> expected;
};

struct PreflightChunk {
    std::string name;
    uint64_t tl1_offset;
    uint32_t size_bytes;
    uint32_t alignment;
    bool go_message = false;
};

constexpr uint8_t kPreflightPoisonByte = 0xE5;
// Fills the aligned word around a sub-word chunk, so a write that zero-fills the rest of the word fails.
constexpr uint32_t kPreflightNeighbourWord = 0x5A5A5A5Au;

std::vector<uint32_t> words_of_bytes(const std::vector<uint8_t>& bytes) {
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), words.size() * sizeof(uint32_t));
    return words;
}

// TL1 windows the runtime writes during CCE firmware init: the exact firmware ELF spans and the HAL addresses from
// build_cce_fw_manifest and the HAL WATCHER address, sizes from the dev_msgs factory. Address, size and alignment only;
// no runtime values.
std::vector<PreflightChunk> firmware_load_chunks(const CceFwManifest& manifest, const dev_msgs::Factory& factory) {
    std::vector<PreflightChunk> chunks;
    for (const CceFwSpan& span : manifest.spans) {
        chunks.push_back(
            {fmt::format("firmware span at TL1 0x{:x}", span.tl1_offset),
             span.tl1_offset,
             static_cast<uint32_t>(span.words.size() * sizeof(uint32_t)),
             4});
    }
    chunks.push_back(
        {"watcher init",
         MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::WATCHER),
         static_cast<uint32_t>(factory.size_of<dev_msgs::watcher_msg_t>()),
         4});
    chunks.push_back(
        {"core_info", manifest.core_info, static_cast<uint32_t>(factory.size_of<dev_msgs::core_info_msg_t>()), 4});

    constexpr uint32_t kBankToNocParts = 4;
    const uint64_t bank_to_noc_part_bytes = (manifest.bank_to_noc_size / 4) & ~uint64_t{3};
    for (uint32_t part = 0; part < kBankToNocParts; ++part) {
        const uint64_t part_bytes = part + 1 == kBankToNocParts
                                        ? manifest.bank_to_noc_size - (kBankToNocParts - 1) * bank_to_noc_part_bytes
                                        : bank_to_noc_part_bytes;
        chunks.push_back(
            {fmt::format("bank-to-NOC scratch part {}", part),
             manifest.bank_to_noc_scratch + part * bank_to_noc_part_bytes,
             static_cast<uint32_t>(part_bytes),
             4});
    }
    chunks.push_back(
        {"CCE SPA routes",
         manifest.bank_to_noc_scratch + manifest.bank_to_noc_size,
         static_cast<uint32_t>(sizeof(CceSpaRoutes)),
         4});

    chunks.push_back(
        {"launch message buffer",
         manifest.launch,
         static_cast<uint32_t>(dev_msgs::launch_msg_buffer_num_entries * factory.size_of<dev_msgs::launch_msg_t>()),
         4});
    const size_t go_message_bytes = factory.size_of<dev_msgs::go_msg_t>();
    TT_FATAL(
        go_message_bytes == sizeof(uint32_t),
        "go message is {} bytes; the preflight treats it as one aligned word",
        go_message_bytes);
    chunks.push_back({"go message", manifest.go_msg, static_cast<uint32_t>(go_message_bytes), 4, true});
    chunks.push_back({"launch_msg_rd_ptr", manifest.launch_msg_rd_ptr, 4, 4});
    chunks.push_back({"go_message_index", manifest.go_msg_index, 4, 4});
    chunks.push_back({"subordinate_sync", manifest.subordinate_sync, 8, 4});
    chunks.push_back({"fw_shared_globals_ready", manifest.fw_globals_ready, 1, 1});
    return chunks;
}

std::vector<uint8_t> preflight_pattern(const PreflightChunk& chunk, uint32_t cce_index) {
    const uint32_t seed = mimir_test_seed() ^ (0x9E3779B9u * (cce_index + 1));
    const size_t num_words = (chunk.size_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    const std::vector<uint32_t> words = make_pattern(MimirPattern::Prng, num_words, chunk.tl1_offset, seed);
    std::vector<uint8_t> bytes(chunk.size_bytes);
    std::memcpy(bytes.data(), words.data(), bytes.size());
    for (uint8_t& byte : bytes) {
        if (byte == kPreflightPoisonByte) {
            byte ^= 0xFF;
        }
    }
    return bytes;
}

// Whole-word chunks go through write_cce_tl1/read_cce_tl1 with Tl1Range::WholeTl1. The 1-byte chunk cannot; it uses
// Cluster on translated_dram_core(soc, cce) at manifest.l1_noc_offset + tl1_offset.
void write_preflight_chunk(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    uint32_t cce_index,
    const CceFwManifest& manifest,
    const PreflightChunk& chunk,
    const std::vector<uint8_t>& bytes) {
    if (chunk.size_bytes % sizeof(uint32_t) == 0) {
        write_cce_tl1(cluster, chip, cce_index, chunk.tl1_offset, words_of_bytes(bytes), Tl1Range::WholeTl1);
        return;
    }
    const tt_cxy_pair core(chip, translated_dram_core(soc, cce_index));
    cluster.write_core(
        bytes.data(), static_cast<uint32_t>(bytes.size()), core, manifest.l1_noc_offset + chunk.tl1_offset);
}

std::vector<uint8_t> read_preflight_chunk(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    uint32_t cce_index,
    const CceFwManifest& manifest,
    const PreflightChunk& chunk) {
    if (chunk.size_bytes % sizeof(uint32_t) == 0) {
        const std::vector<uint32_t> words =
            read_cce_tl1(cluster, chip, cce_index, chunk.tl1_offset, chunk.size_bytes, Tl1Range::WholeTl1);
        std::vector<uint8_t> bytes(chunk.size_bytes);
        std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }
    const tt_cxy_pair core(chip, translated_dram_core(soc, cce_index));
    std::vector<uint8_t> bytes(chunk.size_bytes);
    cluster.read_core(
        bytes.data(), static_cast<uint32_t>(bytes.size()), core, manifest.l1_noc_offset + chunk.tl1_offset);
    return bytes;
}

TEST(MimirHostTl1, Tl1MapLayout) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }
    MIMIR_REQUIRE_IDENTITY();

    constexpr ChipId chip = 0;
    Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, /*require_uncore_released=*/true);

    const Hal& hal = MetalContext::instance().hal();
    const std::vector<Tl1Region> regions = cce_tl1_regions();
    uint64_t previous_end = cce_tl1_begin();
    for (const Tl1Region& region : regions) {
        EXPECT_GT(region.size, 0u) << region.name << " is empty";
        EXPECT_GE(region.begin, previous_end) << region.name << " overlaps or precedes the region before it";
        EXPECT_LE(region.begin + region.size, cce_tl1_end()) << region.name << " does not fit in CCE TL1";
        previous_end = region.begin + region.size;
    }
    EXPECT_GE(tl1_test_base(), previous_end) << "Test TL1 layout starts inside the firmware-reserved range";
    EXPECT_LE(tl1_test_end(), cce_tl1_end()) << "Test TL1 layout runs past the end of CCE TL1";
    EXPECT_EQ(tl1_test_base() % hal.get_alignment(HalMemType::DRAM), 0u);

    const std::vector<HalDramL1Region> hal_regions = dram_hal_l1_regions(hal);
    const HalDramL1Region& barrier_region = hal_region_of(hal_regions, HalL1MemAddrType::BARRIER);
    EXPECT_GE(barrier_region.begin, regions.front().begin);
    EXPECT_LE(barrier_region.begin + barrier_region.size, regions.front().begin + regions.front().size);
    // Each is a field of the mailbox, so it must lie inside the HAL MAILBOX region.
    const HalDramL1Region& mailbox_region = hal_region_of(hal_regions, HalL1MemAddrType::MAILBOX);
    const uint64_t mailbox_end = mailbox_region.begin + mailbox_region.size;
    for (HalL1MemAddrType type :
         {HalL1MemAddrType::LAUNCH,
          HalL1MemAddrType::WATCHER,
          HalL1MemAddrType::DPRINT_BUFFERS,
          HalL1MemAddrType::PROFILER,
          HalL1MemAddrType::GO_MSG,
          HalL1MemAddrType::GO_MSG_INDEX,
          HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR}) {
        const HalDramL1Region& region = hal_region_of(hal_regions, type);
        EXPECT_GT(region.size, 0u) << region.name;
        EXPECT_GE(region.begin, mailbox_region.begin) << region.name;
        EXPECT_LE(region.begin + region.size, mailbox_end) << region.name;
    }
    // The HAL records no size for CORE_INFO, so bound it by the message it holds.
    const uint64_t core_info_begin = hal_region_of(hal_regions, HalL1MemAddrType::CORE_INFO).begin;
    const uint64_t core_info_size =
        hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).size_of<dev_msgs::core_info_msg_t>();
    EXPECT_GE(core_info_begin, mailbox_region.begin);
    EXPECT_LE(core_info_begin + core_info_size, mailbox_end);

    for (uint32_t cce_index = 0; cce_index < topology.num_cces; ++cce_index) {
        for (size_t index = 0; index < regions.size(); ++index) {
            const Tl1Region& region = regions[index];
            const MimirCceL1AddressInfo info = mimir_cce_l1_address(soc, chip, cce_index, region.begin, 4);
            log_info(tt::LogTest, "{} {}", region.name, info.host_string());

            // The tag is the same on every run, so poison the word first or a dropped write passes on a rerun.
            const std::vector<uint32_t> poison = {kVisibilityPoisonWord};
            write_cce_tl1(cluster, chip, cce_index, region.begin, poison, Tl1Range::WholeTl1);
            ASSERT_TRUE(compare_words(
                fmt::format("CCE{} {} poison", cce_index, region.name),
                poison,
                read_cce_tl1(cluster, chip, cce_index, region.begin, 4, Tl1Range::WholeTl1)));

            const uint32_t tag = cce_index * kBoundaryTagsPerCce + static_cast<uint32_t>(index);
            const std::vector<uint32_t> expected = make_guard(tag, 1);
            write_cce_tl1(cluster, chip, cce_index, region.begin, expected, Tl1Range::WholeTl1);
            const std::vector<uint32_t> actual =
                read_cce_tl1(cluster, chip, cce_index, region.begin, 4, Tl1Range::WholeTl1);
            EXPECT_TRUE(compare_words(fmt::format("CCE{} {} first word", cce_index, region.name), expected, actual));
        }
    }
}

TEST_P(MimirHostTl1Cce, PerCceVisibility) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (cce_index >= topology.num_cces) {
        GTEST_SKIP() << "Descriptor exposes only " << topology.num_cces << " CCEs.";
    }
    MIMIR_REQUIRE_IDENTITY();

    constexpr ChipId chip = 0;
    Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, /*require_uncore_released=*/true);

    const uint64_t tl1_begin = cce_tl1_begin();
    const uint64_t tl1_end = cce_tl1_end();
    std::set<uint64_t> offsets = {
        tl1_begin, tl1_end - 4, tl1_end - 16, tl1_end - 64, tl1_end - 4096, tl1_begin + 2 * kMiB};
    for (const Tl1Region& region : cce_tl1_regions()) {
        offsets.insert(region.begin);
    }

    const uint64_t l1_noc_offset = MetalContext::instance().hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    size_t case_index = 0;
    uint32_t guard_tag = kVisibilityGuardTagBase;
    for (uint64_t offset : offsets) {
        for (uint32_t size : kVisibilitySizes) {
            if (offset + size > tl1_end) {
                continue;
            }
            const MimirPattern pattern = kAllMimirPatterns[case_index % kAllMimirPatterns.size()];
            ++case_index;
            const uint64_t cluster_address = l1_noc_offset + offset;
            const std::vector<uint32_t> expected =
                make_pattern(pattern, size / sizeof(uint32_t), cluster_address, mimir_test_seed());
            std::vector<std::pair<uint64_t, uint32_t>> guards;
            if (offset >= tl1_begin + kGuardBytes) {
                guards.push_back({offset - kGuardBytes, guard_tag++});
            }
            if (offset + size + kGuardBytes <= tl1_end) {
                guards.push_back({offset + size, guard_tag++});
            }
            for (const std::pair<uint64_t, uint32_t>& guard : guards) {
                write_cce_tl1(cluster, chip, cce_index, guard.first, make_guard(guard.second), Tl1Range::WholeTl1);
            }
            // Poison the target first, so a dropped write reads back the poison rather than a stale pattern.
            const std::vector<uint32_t> poison(size / sizeof(uint32_t), kVisibilityPoisonWord);
            write_cce_tl1(cluster, chip, cce_index, offset, poison, Tl1Range::WholeTl1);
            write_cce_tl1(cluster, chip, cce_index, offset, expected, Tl1Range::WholeTl1);
            const std::vector<uint32_t> actual =
                read_cce_tl1(cluster, chip, cce_index, offset, size, Tl1Range::WholeTl1);
            EXPECT_TRUE(compare_words(
                fmt::format("CCE{} TL1 0x{:x} +{} B {}", cce_index, offset, size, pattern_name(pattern)),
                expected,
                actual));
            for (const std::pair<uint64_t, uint32_t>& guard : guards) {
                const std::vector<uint32_t> guard_actual =
                    read_cce_tl1(cluster, chip, cce_index, guard.first, kGuardBytes, Tl1Range::WholeTl1);
                EXPECT_TRUE(compare_words(
                    fmt::format(
                        "CCE{} guard {} 0x{:x} of payload 0x{:x} +{} B",
                        cce_index,
                        guard.first < offset ? "below" : "above",
                        guard.first,
                        offset,
                        size),
                    make_guard(guard.second),
                    guard_actual));
            }
        }
    }
}

TEST(MimirHostTl1, BoundaryAndIsolation) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }
    MIMIR_REQUIRE_IDENTITY();

    constexpr ChipId chip = 0;
    Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, /*require_uncore_released=*/true);

    std::set<uint64_t> edges;
    for (const Tl1Region& region : cce_tl1_regions()) {
        edges.insert(region.begin);
        edges.insert(region.begin + region.size);
    }
    // Each window is 16 B and centred on its edge; the two fixed windows straddle the 16 B and 64 B boundaries.
    std::vector<uint64_t> window_offsets;
    for (uint64_t edge : edges) {
        window_offsets.push_back(edge >= 8 ? edge - 8 : 0);
    }
    window_offsets.push_back(0x1008);
    window_offsets.push_back(0x1038);

    std::vector<BoundaryWindow> windows;
    for (uint32_t cce_index = 0; cce_index < topology.num_cces; ++cce_index) {
        for (size_t index = 0; index < window_offsets.size(); ++index) {
            const uint32_t tag = cce_index * kBoundaryTagsPerCce + static_cast<uint32_t>(index);
            windows.push_back({cce_index, window_offsets[index], make_guard(tag, kBoundaryWindowBytes / 4)});
        }
    }

    // A host-only process inherits these windows from the last one, so poison every window first or a dropped write
    // passes on a rerun.
    const std::vector<uint32_t> poison(kBoundaryWindowBytes / sizeof(uint32_t), kVisibilityPoisonWord);
    for (const BoundaryWindow& window : windows) {
        write_cce_tl1(cluster, chip, window.cce_index, window.tl1_offset, poison, Tl1Range::WholeTl1);
    }
    for (const BoundaryWindow& window : windows) {
        ASSERT_TRUE(compare_words(
            fmt::format("CCE{} TL1 0x{:x} boundary window poison", window.cce_index, window.tl1_offset),
            poison,
            read_cce_tl1(
                cluster, chip, window.cce_index, window.tl1_offset, kBoundaryWindowBytes, Tl1Range::WholeTl1)));
    }

    // Every CCE is written before any window is read, so a window that aliases another CCE or another window shows up.
    for (const BoundaryWindow& window : windows) {
        write_cce_tl1(cluster, chip, window.cce_index, window.tl1_offset, window.expected, Tl1Range::WholeTl1);
    }
    for (const BoundaryWindow& window : windows) {
        const std::vector<uint32_t> actual =
            read_cce_tl1(cluster, chip, window.cce_index, window.tl1_offset, kBoundaryWindowBytes, Tl1Range::WholeTl1);
        EXPECT_TRUE(compare_words(
            fmt::format("CCE{} TL1 0x{:x} boundary window", window.cce_index, window.tl1_offset),
            window.expected,
            actual));
    }
}

TEST(MimirHostTl1, HostSideRejections) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }
    MIMIR_REQUIRE_IDENTITY();

    constexpr ChipId chip = 0;
    Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, /*require_uncore_released=*/true);

    const uint64_t l1_noc_offset = MetalContext::instance().hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t first_address_past_cce = l1_noc_offset + cce_tl1_end();
    const uint64_t last_word_of_cce = first_address_past_cce - 4;
    const uint64_t unaligned_address = l1_noc_offset + 0x1002;
    constexpr uint64_t kAlignedWordOffset = 0x1000;
    uint32_t word = 0;
    std::array<uint32_t, 2> pair = {0, 0};

    for (uint32_t cce_index = 0; cce_index < topology.num_cces; ++cce_index) {
        const tt_cxy_pair core(chip, cce_virtual_core(cluster, chip, cce_index));
        const uint32_t tag_base = kRejectSentinelTagBase + cce_index * 16;
        const uint64_t last_word_offset = cce_tl1_end() - 4;
        write_cce_tl1(cluster, chip, cce_index, last_word_offset, make_guard(tag_base, 1), Tl1Range::WholeTl1);
        write_cce_tl1(cluster, chip, cce_index, kAlignedWordOffset, make_guard(tag_base + 1, 1), Tl1Range::WholeTl1);
        for (uint32_t other_cce_index = 0; other_cce_index < topology.num_cces; ++other_cce_index) {
            if (other_cce_index != cce_index) {
                write_cce_tl1(
                    cluster,
                    chip,
                    other_cce_index,
                    cce_tl1_begin(),
                    make_guard(tag_base + 2 + other_cce_index, 1),
                    Tl1Range::WholeTl1);
            }
        }

        EXPECT_THROW(
            cluster.write_core(&word, sizeof(word), core, first_address_past_cce), tt::umd::error::UmdBaseException);
        EXPECT_THROW(
            cluster.read_core(&word, sizeof(word), core, first_address_past_cce), tt::umd::error::UmdBaseException);
        EXPECT_THROW(
            cluster.write_core(pair.data(), sizeof(pair), core, last_word_of_cce), tt::umd::error::UmdBaseException);
        EXPECT_THROW(
            cluster.read_core(pair.data(), sizeof(pair), core, last_word_of_cce), tt::umd::error::UmdBaseException);
        EXPECT_THROW(cluster.write_core(&word, sizeof(word), core, unaligned_address), std::invalid_argument);
        EXPECT_THROW(cluster.read_core(&word, sizeof(word), core, unaligned_address), std::invalid_argument);

        EXPECT_TRUE(compare_words(
            fmt::format("CCE{} TL1 last word untouched by the past-end and straddling rejections", cce_index),
            make_guard(tag_base, 1),
            read_cce_tl1(cluster, chip, cce_index, last_word_offset, 4, Tl1Range::WholeTl1)));
        EXPECT_TRUE(compare_words(
            fmt::format("CCE{} TL1 word 0x{:x} untouched by the unaligned rejections", cce_index, kAlignedWordOffset),
            make_guard(tag_base + 1, 1),
            read_cce_tl1(cluster, chip, cce_index, kAlignedWordOffset, 4, Tl1Range::WholeTl1)));
        for (uint32_t other_cce_index = 0; other_cce_index < topology.num_cces; ++other_cce_index) {
            if (other_cce_index != cce_index) {
                EXPECT_TRUE(compare_words(
                    fmt::format(
                        "CCE{} TL1 first word untouched by the past-end rejections on CCE{}",
                        other_cce_index,
                        cce_index),
                    make_guard(tag_base + 2 + other_cce_index, 1),
                    read_cce_tl1(cluster, chip, other_cce_index, cce_tl1_begin(), 4, Tl1Range::WholeTl1)));
            }
        }
    }
}

TEST(MimirFwPreflight, FirmwareLoadPreflight) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }
    MIMIR_REQUIRE_IDENTITY();

    constexpr ChipId chip = 0;
    Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, /*require_uncore_released=*/true);

    const CceFwManifest& manifest = build_cce_fw_manifest(MetalContext::instance(), chip);
    const dev_msgs::Factory& factory =
        MetalContext::instance().hal().get_dev_msgs_factory(HalProgrammableCoreType::DRAM);
    const std::vector<PreflightChunk> chunks = firmware_load_chunks(manifest, factory);

    for (uint32_t cce_index = 0; cce_index < topology.num_cces; ++cce_index) {
        std::vector<std::vector<uint8_t>> expected_bytes;
        for (const PreflightChunk& chunk : chunks) {
            ASSERT_GT(chunk.size_bytes, 0u) << chunk.name;
            ASSERT_EQ(chunk.tl1_offset % chunk.alignment, 0u) << chunk.name;
            if (chunk.size_bytes % sizeof(uint32_t) != 0) {
                const uint64_t word_offset = chunk.tl1_offset & ~uint64_t{sizeof(uint32_t) - 1};
                ASSERT_LE(chunk.tl1_offset - word_offset + chunk.size_bytes, sizeof(uint32_t)) << chunk.name;
                const std::vector<uint32_t> neighbour_word = {kPreflightNeighbourWord};
                write_cce_tl1(cluster, chip, cce_index, word_offset, neighbour_word, Tl1Range::WholeTl1);
                ASSERT_TRUE(compare_words(
                    fmt::format("CCE{} word around {} before the write", cce_index, chunk.name),
                    neighbour_word,
                    read_cce_tl1(cluster, chip, cce_index, word_offset, sizeof(uint32_t), Tl1Range::WholeTl1)));
            }
            write_preflight_chunk(
                cluster,
                soc,
                chip,
                cce_index,
                manifest,
                chunk,
                std::vector<uint8_t>(chunk.size_bytes, kPreflightPoisonByte));
            expected_bytes.push_back(preflight_pattern(chunk, cce_index));
            write_preflight_chunk(cluster, soc, chip, cce_index, manifest, chunk, expected_bytes.back());
        }
        for (size_t index = 0; index < chunks.size(); ++index) {
            const PreflightChunk& chunk = chunks[index];
            const std::vector<uint8_t>& expected = expected_bytes[index];
            const std::vector<uint8_t> actual = read_preflight_chunk(cluster, soc, chip, cce_index, manifest, chunk);
            if (chunk.size_bytes % sizeof(uint32_t) == 0) {
                EXPECT_TRUE(compare_words(
                    fmt::format("CCE{} {}", cce_index, chunk.name), words_of_bytes(expected), words_of_bytes(actual)));
            } else {
                const uint64_t word_offset = chunk.tl1_offset & ~uint64_t{sizeof(uint32_t) - 1};
                uint32_t expected_word = kPreflightNeighbourWord;
                std::memcpy(
                    reinterpret_cast<uint8_t*>(&expected_word) + (chunk.tl1_offset - word_offset),
                    expected.data(),
                    expected.size());
                EXPECT_TRUE(compare_words(
                    fmt::format("CCE{} {} and the rest of its word", cce_index, chunk.name),
                    {expected_word},
                    read_cce_tl1(cluster, chip, cce_index, word_offset, sizeof(uint32_t), Tl1Range::WholeTl1)));
            }
            if (chunk.go_message) {
                dev_msgs::go_msg_t expected_go_message = factory.create<dev_msgs::go_msg_t>();
                dev_msgs::go_msg_t actual_go_message = factory.create<dev_msgs::go_msg_t>();
                std::memcpy(expected_go_message.data(), expected.data(), expected_go_message.size());
                std::memcpy(actual_go_message.data(), actual.data(), actual_go_message.size());
                EXPECT_EQ(
                    static_cast<uint32_t>(actual_go_message.view().signal()),
                    static_cast<uint32_t>(expected_go_message.view().signal()))
                    << "CCE" << cce_index << " " << chunk.name << " signal";
            }
        }

        // The runtime writes the reset vector once, to hart 0's register, and relies on it reaching all eight harts.
        ASSERT_NO_FATAL_FAILURE(poison_cce_reset_vectors(cluster, chip, cce_index));
        const Hal& hal = MetalContext::instance().hal();
        const uint32_t fw_launch_addr_value =
            hal.get_jit_build_config(hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM), 0, 0)
                .fw_launch_addr_value;
        cluster.write_reg(
            &fw_launch_addr_value,
            tt_cxy_pair(chip, cce_virtual_core(cluster, chip, cce_index)),
            manifest.fw_launch_addr);
        const std::vector<uint64_t> reset_vectors = read_cce_reset_vectors(cluster, soc, chip, cce_index);
        for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
            EXPECT_EQ(reset_vectors[hart], fw_launch_addr_value) << "CCE" << cce_index << " hart " << hart;
        }
    }
}

}  // namespace tt::tt_metal::mimir_test::host_tl1
