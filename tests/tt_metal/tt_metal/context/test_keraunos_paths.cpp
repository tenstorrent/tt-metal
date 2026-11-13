// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1 of keraunos_plan.md: a ladder of read and write probes from Keraunos CCEs, ordered from the
// paths already known to work to the ones that hung before, so one batch on the emulator shows where a
// remote read first fails. Every probe checks its data from the host.
//
// Addresses a CCE uses (package SPA, see tt::umd::GrendelAddressWindows and keraunos_mimir_gddr_read.md):
//   HSIO n SRAM          0x12_0680_0000 + n * tile_stride + eth_cce * 4 MiB    (identity DMRISC windows 6-9)
//   Mimir CCE c SRAM     0x12_8000_0000 + c * 4 MiB                       (K D2D BAR1)
//   Mimir m GDDR         0x1_0000_0000_0000 + (16 + m) * 8 GiB            (the D2D1 lane: K reaches M1 directly)

#include "keraunos_cce_fixture.hpp"

#include <tt-umd/coordinates/grendel_noc_address_resolver.hpp>

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

class KeraunosPaths : public KeraunosCceTest {
protected:
    // Global index of the first CCE on Mimir `mimir`, in the numbering the SRAM SPA window uses.
    std::optional<uint32_t> first_cce_on_mimir(uint32_t mimir) const {
        for (uint32_t mimir_cce = 0; mimir_cce < num_mimir_cces_on(device_); ++mimir_cce) {
            if (mimir_index_for_cce(soc_desc_of(device_), mimir_cce) == mimir) {
                return mimir_cce;
            }
        }
        return std::nullopt;
    }

    static uint64_t mimir_cce_sram_spa(uint32_t mimir_cce, uint64_t offset) {
        const tt::umd::GrendelAddressWindows w{};
        return w.dram_l1_base + static_cast<uint64_t>(mimir_cce) * w.dram_l1_stride + offset;
    }

    // Keraunos sits on the D2D1 lane of the package ATT, 16 instances above the D2D0 lane.
    static uint64_t mimir_gddr_spa_from_keraunos(uint32_t mimir, uint64_t offset) {
        const tt::umd::GrendelAddressWindows w{};
        constexpr uint64_t kD2d1LaneInstances = 16;  // MIMIR_CCE_GDDR_D2D_LANE_INSTANCES
        return w.dram_base + (kD2d1LaneInstances + mimir) * w.dram_stride + offset;
    }

    uint32_t mimir_cce_offset() const {
        return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }
    uint64_t mimir_cce_noc_addr() const {
        return MetalContext::instance().hal().get_dev_noc_addr(
            HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }

    void add_cpu_copy(Program& program, const Hsio& c, uint64_t src, uint64_t dst, uint32_t words, uint32_t local_flags) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_cpu_copy.cpp",
            c.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        std::vector<uint32_t> args = split(src);
        const auto d = split(dst);
        args.insert(args.end(), d.begin(), d.end());
        args.insert(args.end(), {words, unreserved_ + kStatusOffset, local_flags});
        SetRuntimeArgs(program, k, c.logical, args);
    }

    void expect_done(const Hsio& c) {
        const auto status = read_words(c, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    }

    // Copies a block between two places with either engine, checking the block lands where `read_dst`
    // looks. The source has already been written by the caller.
    enum class Engine { Cpu, Idma };
    // With the CPU engine `dst_local` means `dst` is a core-local offset (the kernel uses the uncached alias).
    void run_copy(Engine engine, const Hsio& c, uint64_t src, uint64_t dst, uint32_t bytes, bool dst_local = false) {
        write_words(c, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        if (engine == Engine::Cpu) {
            add_cpu_copy(program, c, src, dst, bytes / sizeof(uint32_t), dst_local ? 2u : 0u);
        } else {
            add_idma_copy(program, c, {.src = src, .dst = dst, .len = bytes});
        }
        launch(program, c);
        expect_done(c);
    }

    static std::vector<uint32_t> pattern(uint32_t tag, uint32_t bytes) {
        std::vector<uint32_t> p(bytes / sizeof(uint32_t));
        for (uint32_t i = 0; i < p.size(); i++) {
            p[i] = (tag << 16) | i;
        }
        return p;
    }

    // HSIO `from` reads HSIO `to`'s SRAM into its own (read) or writes its own into `to`'s (write).
    void expect_tile_to_tile(Engine engine, uint32_t from, uint32_t to, bool read, uint32_t tag) {
        const auto a = hsio_core(from);
        const auto b = hsio_core(to);
        ASSERT_TRUE(a.has_value() && b.has_value());
        constexpr uint32_t kBytes = 256;
        const auto data = pattern(tag, kBytes);
        const Hsio& src_core = read ? *b : *a;
        const Hsio& dst_core = read ? *a : *b;
        write_words(src_core, kSrcOffset, data);
        write_words(dst_core, kDstOffset, std::vector<uint32_t>(data.size(), 0));
        const bool dst_local = engine == Engine::Cpu && read;
        const uint64_t dst = dst_local ? local_addr(kDstOffset) : global_addr(dst_core, kDstOffset);
        run_copy(engine, *a, global_addr(src_core, kSrcOffset), dst, kBytes, dst_local);
        EXPECT_EQ(read_words(dst_core, kDstOffset, data.size()), data);
    }

    // HSIO 2 CCE 0 (the core that owns D2D2) and a Mimir CCE's SRAM.
    void expect_mimir_cce_sram(Engine engine, uint32_t mimir, bool read, uint32_t tag) {
        if (num_mimirs_on(device_) <= mimir) {
            GTEST_SKIP() << "Descriptor has no Mimir " << mimir;
        }
        const auto k = hsio_core(2);
        const auto mimir_cce = first_cce_on_mimir(mimir);
        ASSERT_TRUE(k.has_value() && mimir_cce.has_value());
        const CoreCoord mimir_core = logical_mimir_cce_core(soc_desc_of(device_), *mimir_cce);
        constexpr uint32_t kBytes = 256;
        const auto data = pattern(tag, kBytes);
        const uint64_t remote = mimir_cce_sram_spa(*mimir_cce, mimir_cce_offset());
        if (read) {
            write_mimir_cce_sram(device_, mimir_core, mimir_cce_noc_addr(), data.data(), kBytes);
            write_words(*k, kDstOffset, std::vector<uint32_t>(data.size(), 0));
            const bool dst_local = engine == Engine::Cpu;
            const uint64_t dst = dst_local ? local_addr(kDstOffset) : global_addr(*k, kDstOffset);
            run_copy(engine, *k, remote, dst, kBytes, dst_local);
            EXPECT_EQ(read_words(*k, kDstOffset, data.size()), data);
        } else {
            write_words(*k, kSrcOffset, data);
            const std::vector<uint32_t> zero(data.size(), 0);
            write_mimir_cce_sram(device_, mimir_core, mimir_cce_noc_addr(), zero.data(), kBytes);
            run_copy(engine, *k, global_addr(*k, kSrcOffset), remote, kBytes);
            std::vector<uint32_t> got(data.size(), 0);
            read_mimir_cce_sram(device_, mimir_core, mimir_cce_noc_addr(), got.data(), kBytes);
            EXPECT_EQ(got, data);
        }
    }

    // HSIO 2 CCE 0 and Mimir `mimir`'s GDDR.
    void expect_mimir_gddr(Engine engine, uint32_t mimir, bool read, uint32_t tag) {
        if (num_mimirs_on(device_) <= mimir) {
            GTEST_SKIP() << "Descriptor has no Mimir " << mimir;
        }
        const auto k = hsio_core(2);
        ASSERT_TRUE(k.has_value());
        const auto& hal = MetalContext::instance().hal();
        const uint32_t gddr_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const uint32_t bytes = hal.get_alignment(HalMemType::DRAM);
        const auto data = pattern(tag, bytes);
        const uint64_t remote = mimir_gddr_spa_from_keraunos(mimir, gddr_offset);
        GddrBackingScope gddr_backing;
        if (read) {
            write_mimir_gddr(*mesh_, device_, mimir, gddr_offset, data);
            write_words(*k, kDstOffset, std::vector<uint32_t>(data.size(), 0));
            const bool dst_local = engine == Engine::Cpu;
            const uint64_t dst = dst_local ? local_addr(kDstOffset) : global_addr(*k, kDstOffset);
            run_copy(engine, *k, remote, dst, bytes, dst_local);
            EXPECT_EQ(read_words(*k, kDstOffset, data.size()), data);
        } else {
            write_words(*k, kSrcOffset, data);
            write_mimir_gddr(*mesh_, device_, mimir, gddr_offset, std::vector<uint32_t>(data.size(), 0));
            run_copy(engine, *k, global_addr(*k, kSrcOffset), remote, bytes);
            std::vector<uint32_t> got;
            read_mimir_gddr(*mesh_, device_, mimir, gddr_offset, bytes, got);
            EXPECT_EQ(got, data);
        }
    }
};

// --- HSIO SRAM ---------------------------------------------------------------------------------------
TEST_F(KeraunosPaths, CpuWritesOtherTileSram) { expect_tile_to_tile(Engine::Cpu, 0, 1, /*read=*/false, 0xB1); }
TEST_F(KeraunosPaths, IdmaWritesOtherTileSram) { expect_tile_to_tile(Engine::Idma, 0, 1, /*read=*/false, 0xB2); }
TEST_F(KeraunosPaths, IdmaReadsOtherTileSram) { expect_tile_to_tile(Engine::Idma, 0, 1, /*read=*/true, 0xB3); }
TEST_F(KeraunosPaths, CpuReadsOtherTileSram) { expect_tile_to_tile(Engine::Cpu, 0, 1, /*read=*/true, 0xB4); }

// --- Mimir CCE SRAM over D2D2 (SysEng's ker_cce_dma_m2k_remap path) ------------------------------------
TEST_F(KeraunosPaths, IdmaWritesM1MimirCceSram) { expect_mimir_cce_sram(Engine::Idma, 1, /*read=*/false, 0xB5); }
TEST_F(KeraunosPaths, IdmaReadsM1MimirCceSram) { expect_mimir_cce_sram(Engine::Idma, 1, /*read=*/true, 0xB6); }

// --- Mimir GDDR over D2D2 ------------------------------------------------------------------------------
TEST_F(KeraunosPaths, IdmaWritesM1Gddr) { expect_mimir_gddr(Engine::Idma, 1, /*read=*/false, 0xB7); }
TEST_F(KeraunosPaths, IdmaReadsM1Gddr) { expect_mimir_gddr(Engine::Idma, 1, /*read=*/true, 0xB8); }

// --- CPU loads across D2D: the original hang. Last, because a stuck load can leave the hart mid-transaction.
TEST_F(KeraunosPaths, CpuReadsM1MimirCceSram) { expect_mimir_cce_sram(Engine::Cpu, 1, /*read=*/true, 0xB9); }
TEST_F(KeraunosPaths, CpuReadsM1Gddr) { expect_mimir_gddr(Engine::Cpu, 1, /*read=*/true, 0xBA); }

}  // namespace
}  // namespace tt::tt_metal
