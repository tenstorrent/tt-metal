// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Which NOC path to the tile's local DRAM sees the X280 caches (README § 4, Host
// -> X280 NOC writes; § 5). A Tensix kernel reads and writes a region of the
// tile's local DRAM bank while firmware-coherent-alias.bin loads and stores the
// same bytes through its cached Memory Port, with no flush between them:
//
//   (tile) X280_LOCAL_DDR_TILE_BASE + offset        coherent in both directions
//   (tile) X280_SYSTEM_PORT_DDR_TILE_BASE + offset  stale in both directions
//   DRAM bank, bank address offset                  stale in both directions
//
// The Memory Port alias is a 47-bit address, so the kernel writes NOC_*_ADDR_MID
// itself. A flushed line read through the DRAM bank closes the loop: the region
// is the tile's own DRAM. Each observation repeats kRounds times on fresh lines.
// The image flushes the whole region before and after, and the region is a
// buffer this test allocates, so the case is reversible.

#include "x280_probe_support.hpp"

#include "x280.h"

#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace x280;
using namespace std::chrono_literals;
namespace distributed = tt::tt_metal::distributed;
using Line = std::array<uint32_t, 16>;

constexpr uint64_t kRun = 0xC0E4E7A1A5000001ull;
constexpr uint32_t kRounds = 20;
constexpr uint32_t kRegionBytes = 32 * 1024;
constexpr uint32_t kRegionLines = kRegionBytes / X280_PROBE_LINE_BYTES;
constexpr uint32_t kFirstLine = 16;  // keep clear of the region's edges
constexpr uint32_t kLinesPerRound = 8;
static_assert(kFirstLine + kRounds * kLinesPerRound + kFirstLine <= kRegionLines);

// tt-metal DRAM bank of each hardware L2CPU's local DRAM (README § 1).
constexpr std::array<uint32_t, 4> kLocalBank = {5, 6, 7, 7};

constexpr auto kCommandTimeout = 1s;

struct Tally {
    const char* path;
    uint32_t same = 0;
    uint32_t rounds = 0;
    void add(bool equal) {
        same += equal ? 1 : 0;
        rounds++;
    }
};

class X280CoherentAliasHw : public x280::test::X280ProbeHwBase {
protected:
    void command(uint64_t op, uint32_t first, uint32_t count, uint32_t seed = 0) {
        ASSERT_EQ(block_->command(op, first, count, seed, kCommandTimeout), X280_PROBE_STATUS_OK)
            << "image command " << op << " on lines " << first << "+" << count << " failed or timed out";
    }

    // Cached load of one line by the X280; the copy it made in LIM.
    Line x280_read(uint32_t line) {
        command(X280_PROBE_OP_READ, line, 1);
        return block_->data_line(0);
    }
};

}  // namespace

TEST_F(X280CoherentAliasHw, MemoryPortAliasIsCoherentAndTheLowAliasesAreStale) {
    const uint32_t hw = static_cast<uint32_t>(tile_->id().hw_index);
    ASSERT_LT(hw, kLocalBank.size());
    const uint32_t bank = kLocalBank[hw];
    const auto& allocator = mesh_->allocator();
    const uint32_t banks = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
    if (bank >= banks) {
        GTEST_SKIP() << "the tile's local DRAM bank " << bank << " is not among the " << banks << " DRAM banks";
    }

    // One page per bank: page `bank` of an interleaved buffer is the region, at
    // the buffer's address in that bank.
    auto region = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(kRegionBytes) * banks},
        distributed::DeviceLocalBufferConfig{.page_size = kRegionBytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_.get());
    const uint32_t bank_addr = static_cast<uint32_t>(region->address());
    const uint64_t offset = region->address() + allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bank);
    const uint64_t memory_port = X280_LOCAL_DDR_TILE_BASE + offset;
    const uint64_t system_port = X280_SYSTEM_PORT_DDR_TILE_BASE + offset;
    const TranslatedCoord at = runtime_->virtual_core_from_tile(tile_->id(), x280::test::NocProbe::kNoc);
    printf(
        "\nL2CPU %u at (%u,%u), DRAM bank %u, offset 0x%llx, LIM ways %u (WayEnable %u)\n",
        hw,
        at.x(),
        at.y(),
        bank,
        static_cast<unsigned long long>(offset),
        static_cast<unsigned>(runtime_->lim_shape().lim_ways),
        15u - runtime_->lim_shape().lim_ways);

    block_->reset(kRun, offset, kRegionLines);
    const SessionId id =
        enqueue(x280::test::test_image("X280_COHERENT_ALIAS_FW_IMAGE", "firmware-coherent-alias.bin"), kRun);
    const auto running = runtime_->session(id).wait_until_running(3s);
    ASSERT_EQ(running.outcome, WaitOutcome::Running) << describe(running);

    // Nothing of an earlier run stays cached, then the DRAM starts at zero.
    command(X280_PROBE_OP_FLUSH, 0, kRegionLines);
    std::vector<uint32_t> zeros(region->size() / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(mesh_->mesh_command_queue(), region, zeros, /*blocking=*/true);

    auto line_addr = [](uint64_t base, uint32_t line) { return base + static_cast<uint64_t>(line) * 64; };
    Tally to_x280{"Tensix write via the Memory Port alias -> X280 cached read"};
    Tally from_x280{"X280 cached write -> Tensix read via the Memory Port alias"};
    Tally sys_read{"X280 cached write -> Tensix read via the System Port alias"};
    Tally dram_read{"X280 cached write -> Tensix read via the DRAM bank"};
    Tally sys_write{"Tensix write via the System Port alias -> X280 cached read"};
    Tally dram_write{"Tensix write via the DRAM bank -> X280 cached read"};
    Tally flushed{"X280 cached write + CCACHE0_FLUSH64 -> Tensix read via the DRAM bank"};

    for (uint32_t round = 0; round < kRounds && !HasFatalFailure(); ++round) {
        const uint32_t base = kFirstLine + round * kLinesPerRound;
        const uint32_t seed = 0x5EED0000u + round * 16;

        // Coherent, NOC -> X280: the line is in the X280's cache before the write.
        const uint32_t a = base + 0;
        (void)x280_read(a);
        probe_->write_raw(at, line_addr(memory_port, a), 1, a, seed + 0);
        to_x280.add(x280_read(a) == x280::test::pattern_line(seed + 0, a));

        // Coherent, X280 -> NOC: the line is dirty in the X280's cache.
        const uint32_t b = base + 1;
        command(X280_PROBE_OP_WRITE, b, 1, seed + 1);
        from_x280.add(probe_->read_raw(at, line_addr(memory_port, b), 1)[0] == x280::test::pattern_line(seed + 1, b));

        // Stale, X280 -> NOC.
        const uint32_t c = base + 2;
        command(X280_PROBE_OP_WRITE, c, 1, seed + 2);
        const Line want_c = x280::test::pattern_line(seed + 2, c);
        sys_read.add(probe_->read_raw(at, line_addr(system_port, c), 1)[0] == want_c);
        dram_read.add(probe_->read_bank(bank, line_addr(bank_addr, c), 1)[0] == want_c);

        // Stale, NOC -> X280, each with the line already cached.
        const uint32_t d = base + 3;
        (void)x280_read(d);
        probe_->write_raw(at, line_addr(system_port, d), 1, d, seed + 3);
        sys_write.add(x280_read(d) == x280::test::pattern_line(seed + 3, d));

        const uint32_t e = base + 4;
        (void)x280_read(e);
        probe_->write_bank(bank, line_addr(bank_addr, e), 1, e, seed + 4);
        dram_write.add(x280_read(e) == x280::test::pattern_line(seed + 4, e));

        // The DRAM bank is the memory behind the alias: a flushed line is there.
        const uint32_t f = base + 5;
        command(X280_PROBE_OP_WRITE, f, 1, seed + 5);
        command(X280_PROBE_OP_FLUSH, f, 1);
        flushed.add(probe_->read_bank(bank, line_addr(bank_addr, f), 1)[0] == x280::test::pattern_line(seed + 5, f));
    }

    // Leave no line of the buffer in the X280's caches, dirty or clean.
    command(X280_PROBE_OP_FLUSH, 0, kRegionLines);
    ASSERT_EQ(block_->command(X280_PROBE_OP_EXIT, 0, 0, 0, kCommandTimeout), X280_PROBE_STATUS_OK);
    const auto cleaned = runtime_->cleanup(id, 3s);
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
    EXPECT_TRUE(active_all_idle(cleaned.hartstates, kRun)) << describe(cleaned);

    printf("observed the written data, of %u rounds:\n", kRounds);
    for (const Tally* t : {&to_x280, &from_x280, &sys_read, &dram_read, &sys_write, &dram_write, &flushed}) {
        printf("  %2u  %s\n", t->same, t->path);
    }
    for (const Tally* t : {&to_x280, &from_x280, &flushed}) {
        EXPECT_EQ(t->same, t->rounds) << t->path;
    }
    for (const Tally* t : {&sys_read, &dram_read, &sys_write, &dram_write}) {
        EXPECT_EQ(t->same, 0u) << t->path;
    }
}
