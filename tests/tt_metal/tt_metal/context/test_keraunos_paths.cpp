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
    // The CCE of each HSIO tile that issues the traffic: 0 for the KeraunosPaths suite, 1 for KeraunosPathsCce1.
    uint32_t k_cce_ = 0;

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
        const auto a = hsio_core(from, k_cce_);
        const auto b = hsio_core(to, k_cce_);
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
        const auto k = hsio_core(2, k_cce_);
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

    // HSIO 2 CCE 0 and one Mimir CCE's SRAM, named by its global CCE index (the SRAM SPA window numbering).
    void expect_mimir_cce_index(Engine engine, uint32_t mimir_cce, bool read, uint32_t tag) {
        const auto k = hsio_core(2, k_cce_);
        ASSERT_TRUE(k.has_value());
        const CoreCoord mimir_core = logical_mimir_cce_core(soc_desc_of(device_), mimir_cce);
        constexpr uint32_t kBytes = 256;
        const auto data = pattern(tag, kBytes);
        const uint64_t remote = mimir_cce_sram_spa(mimir_cce, mimir_cce_offset());
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

    // Every Mimir CCE (CCE0 and CCE1 of M0 and M1), one after another.
    void expect_every_mimir_cce(Engine engine, bool read, uint32_t tag_base) {
        const uint32_t n = num_mimir_cces_on(device_);
        if (n == 0) {
            GTEST_SKIP() << "Descriptor has no Mimir CCE";
        }
        for (uint32_t mimir_cce = 0; mimir_cce < n; ++mimir_cce) {
            SCOPED_TRACE(fmt::format(
                "Mimir {} CCE {} (global {})",
                mimir_index_for_cce(soc_desc_of(device_), mimir_cce),
                mimir_cce,
                mimir_cce));
            expect_mimir_cce_index(engine, mimir_cce, read, tag_base + mimir_cce);
            if (HasFatalFailure()) {
                return;
            }
        }
    }

    // HSIO 2 CCE 0 and Mimir `mimir`'s GDDR.
    void expect_mimir_gddr(Engine engine, uint32_t mimir, bool read, uint32_t tag) {
        if (num_mimirs_on(device_) <= mimir) {
            GTEST_SKIP() << "Descriptor has no Mimir " << mimir;
        }
        const auto k = hsio_core(2, k_cce_);
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

    // Number of HSIO tiles in the descriptor (CCE0 of each).
    uint32_t hsio_count() const {
        uint32_t n = 0;
        for (const auto& core : soc_desc_of(device_).get_arch_descriptor().get_keraunos().cores) {
            n += core.eth_cce == 0 ? 1 : 0;
        }
        return n;
    }

    // Every ordered pair of distinct HSIO tiles, plus each tile through its own global window.
    void expect_tile_matrix(Engine engine, bool read, uint32_t tag_base) {
        const uint32_t n = hsio_count();
        ASSERT_GE(n, 2u);
        for (uint32_t from = 0; from < n; ++from) {
            for (uint32_t to = 0; to < n; ++to) {
                SCOPED_TRACE(fmt::format("HSIO {} {} HSIO {}", from, read ? "reads" : "writes", to));
                expect_tile_to_tile(engine, from, to, read, tag_base + from * n + to);
                if (HasFatalFailure()) {
                    return;
                }
            }
        }
    }

    // One GDDR copy from HSIO 2 CCE 0 to `mimir`'s DRAM view `view` at `extra` bytes into it. The package address
    // of the view's start is its address_offset; the host reaches the same bytes through the view.
    void expect_mimir_gddr_at(Engine engine, uint32_t mimir, uint32_t view, uint64_t extra, bool read, uint32_t tag) {
        const auto k = hsio_core(2, k_cce_);
        ASSERT_TRUE(k.has_value());
        const auto& hal = MetalContext::instance().hal();
        const metal_SocDescriptor& soc = soc_desc_of(device_);
        const uint64_t host_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED) + extra;
        const uint32_t bytes = hal.get_alignment(HalMemType::DRAM);
        const auto data = pattern(tag, bytes);
        const uint64_t remote =
            mimir_gddr_spa_from_keraunos(mimir, static_cast<uint64_t>(soc.get_address_offset(static_cast<int>(view))) + host_offset);
        GddrBackingScope gddr_backing;
        if (read) {
            write_gddr_view(*mesh_, device_, view, host_offset, data);
            write_words(*k, kDstOffset, std::vector<uint32_t>(data.size(), 0));
            const bool dst_local = engine == Engine::Cpu;
            const uint64_t dst = dst_local ? local_addr(kDstOffset) : global_addr(*k, kDstOffset);
            run_copy(engine, *k, remote, dst, bytes, dst_local);
            EXPECT_EQ(read_words(*k, kDstOffset, data.size()), data);
        } else {
            write_words(*k, kSrcOffset, data);
            write_gddr_view(*mesh_, device_, view, host_offset, std::vector<uint32_t>(data.size(), 0));
            run_copy(engine, *k, global_addr(*k, kSrcOffset), remote, bytes);
            std::vector<uint32_t> got;
            read_gddr_view(*mesh_, device_, view, host_offset, bytes, got);
            EXPECT_EQ(got, data);
        }
    }

    // Every DRAM view of `mimir` at several offsets: the first bytes, a page in, 1 MiB in, and the view's midpoint.
    void expect_mimir_gddr_views(Engine engine, uint32_t mimir, bool read, uint32_t tag_base) {
        if (num_mimirs_on(device_) <= mimir) {
            GTEST_SKIP() << "Descriptor has no Mimir " << mimir;
        }
        const metal_SocDescriptor& soc = soc_desc_of(device_);
        const auto views = dram_views_for_channel(soc, mimir);
        ASSERT_FALSE(views.empty());
        const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        uint32_t tag = tag_base;
        for (uint32_t view : views) {
            const uint64_t view_size = soc.dram_view_size;
            std::vector<uint64_t> extras = {0, 4096, 1ull << 20};
            if (view_size > 2 * (unreserved + (1ull << 20))) {
                extras.push_back(view_size / 2);
            }
            for (uint64_t extra : extras) {
                if (unreserved + extra + 64 > view_size) {
                    continue;
                }
                SCOPED_TRACE(fmt::format("Mimir {} view {} +{:#x}", mimir, view, extra));
                expect_mimir_gddr_at(engine, mimir, view, extra, read, tag++);
                if (HasFatalFailure()) {
                    return;
                }
            }
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


// --- Phase 2: the full matrix --------------------------------------------------------------------------
// Every HSIO tile to every other, and to itself through its own global window.
TEST_F(KeraunosPaths, IdmaWritesEveryTileSram) { expect_tile_matrix(Engine::Idma, /*read=*/false, 0x10); }
TEST_F(KeraunosPaths, IdmaReadsEveryTileSram) { expect_tile_matrix(Engine::Idma, /*read=*/true, 0x30); }
TEST_F(KeraunosPaths, CpuWritesEveryTileSram) { expect_tile_matrix(Engine::Cpu, /*read=*/false, 0x50); }
TEST_F(KeraunosPaths, CpuReadsEveryTileSram) { expect_tile_matrix(Engine::Cpu, /*read=*/true, 0x70); }

// CPU stores across D2D (loads are covered above).
TEST_F(KeraunosPaths, CpuWritesM1MimirCceSram) { expect_mimir_cce_sram(Engine::Cpu, 1, /*read=*/false, 0xC1); }
TEST_F(KeraunosPaths, CpuWritesM1Gddr) { expect_mimir_gddr(Engine::Cpu, 1, /*read=*/false, 0xC2); }

// M1 GDDR: every DRAM view, several offsets, both engines, both directions.
TEST_F(KeraunosPaths, IdmaWritesM1GddrViews) { expect_mimir_gddr_views(Engine::Idma, 1, /*read=*/false, 0xD0); }
TEST_F(KeraunosPaths, IdmaReadsM1GddrViews) { expect_mimir_gddr_views(Engine::Idma, 1, /*read=*/true, 0xD1); }
TEST_F(KeraunosPaths, CpuWritesM1GddrViews) { expect_mimir_gddr_views(Engine::Cpu, 1, /*read=*/false, 0xD2); }
TEST_F(KeraunosPaths, CpuReadsM1GddrViews) { expect_mimir_gddr_views(Engine::Cpu, 1, /*read=*/true, 0xD3); }

// M0 GDDR from Keraunos: the 0x1_0020_... route, M0's D2D1 behind M1 (two hops).
TEST_F(KeraunosPaths, IdmaWritesM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Idma, 0, /*read=*/false, 0xE0); }
TEST_F(KeraunosPaths, IdmaReadsM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Idma, 0, /*read=*/true, 0xE1); }
TEST_F(KeraunosPaths, CpuWritesM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Cpu, 0, /*read=*/false, 0xE2); }
TEST_F(KeraunosPaths, CpuReadsM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Cpu, 0, /*read=*/true, 0xE3); }


// --- Phase 3: K CCE <-> Mimir CCE SRAM over BAR1, every Mimir CCE --------------------------------------
TEST_F(KeraunosPaths, IdmaWritesEveryMimirCceSram) { expect_every_mimir_cce(Engine::Idma, /*read=*/false, 0x90); }
TEST_F(KeraunosPaths, IdmaReadsEveryMimirCceSram) { expect_every_mimir_cce(Engine::Idma, /*read=*/true, 0xA0); }
TEST_F(KeraunosPaths, CpuWritesEveryMimirCceSram) { expect_every_mimir_cce(Engine::Cpu, /*read=*/false, 0xB0); }
TEST_F(KeraunosPaths, CpuReadsEveryMimirCceSram) { expect_every_mimir_cce(Engine::Cpu, /*read=*/true, 0xC0); }


// --- Phase 3: Mimir CCE -> Keraunos HSIO SRAM (worker -> router direction) -------------------------------
// A Mimir CCE kernel loads from or stores to a Keraunos HSIO SRAM through the package address the Keraunos
// CCEs use for it. M1 is the Mimir on the D2D link to K0.
class KeraunosFromMimir : public KeraunosPaths {
protected:
    void expect_mimir_to_hsio(uint32_t mimir, uint32_t hsio, bool read, uint32_t tag) {
        if (num_mimirs_on(device_) <= mimir) {
            GTEST_SKIP() << "Descriptor has no Mimir " << mimir;
        }
        const auto k = hsio_core(hsio, k_cce_);
        const auto mimir_cce = first_cce_on_mimir(mimir);
        ASSERT_TRUE(k.has_value() && mimir_cce.has_value());
        const CoreCoord mimir_core = logical_mimir_cce_core(soc_desc_of(device_), *mimir_cce);
        constexpr uint32_t kWords = 64;
        const auto data = pattern(tag, kWords * sizeof(uint32_t));
        const uint32_t local_offset = mimir_cce_offset() + 0x1000;
        const uint32_t status_offset = mimir_cce_offset();
        const uint64_t noc = mimir_cce_noc_addr();
        const std::vector<uint32_t> zero_status(8, 0);
        write_mimir_cce_sram(device_, mimir_core, noc, zero_status.data(), zero_status.size() * sizeof(uint32_t));
        if (read) {
            write_words(*k, kSrcOffset, data);
            const std::vector<uint32_t> zero(kWords, 0);
            write_mimir_cce_sram(device_, mimir_core, noc + 0x1000, zero.data(), kWords * sizeof(uint32_t));
        } else {
            write_words(*k, kDstOffset, std::vector<uint32_t>(kWords, 0));
            write_mimir_cce_sram(device_, mimir_core, noc + 0x1000, data.data(), kWords * sizeof(uint32_t));
        }
        const uint64_t remote = global_addr(*k, read ? kSrcOffset : kDstOffset);
        Program program = CreateProgram();
        const KernelHandle kernel = CreateKernel(
            program, std::string(kKernelDir) + "mimir_cce_ker_sram_copy.cpp", mimir_core, DramConfig{.noc = NOC::NOC_0});
        std::vector<uint32_t> args = split(remote);
        args.insert(args.end(), {kWords, local_offset, read ? 1u : 0u, status_offset});
        SetRuntimeArgs(program, kernel, mimir_core, args);
        try {
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
            distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/true);
        } catch (const std::exception& e) {
            std::vector<uint32_t> status(8, 0);
            read_mimir_cce_sram(device_, mimir_core, noc, status.data(), status.size() * sizeof(uint32_t));
            FAIL() << "launch failed: " << e.what() << "\nstatus: " << fmt::format("{:#010x}", fmt::join(status, " "));
        }
        std::vector<uint32_t> status(8, 0);
        read_mimir_cce_sram(device_, mimir_core, noc, status.data(), status.size() * sizeof(uint32_t));
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        if (read) {
            std::vector<uint32_t> got(kWords, 0);
            read_mimir_cce_sram(device_, mimir_core, noc + 0x1000, got.data(), kWords * sizeof(uint32_t));
            EXPECT_EQ(got, data);
        } else {
            EXPECT_EQ(read_words(*k, kDstOffset, kWords), data);
        }
    }
};

TEST_F(KeraunosFromMimir, M1CceWritesHsio2Sram) { expect_mimir_to_hsio(1, 2, /*read=*/false, 0xF1); }
TEST_F(KeraunosFromMimir, M1CceReadsHsio2Sram) { expect_mimir_to_hsio(1, 2, /*read=*/true, 0xF2); }
TEST_F(KeraunosFromMimir, M1CceWritesHsio0Sram) { expect_mimir_to_hsio(1, 0, /*read=*/false, 0xF3); }
TEST_F(KeraunosFromMimir, M1CceReadsHsio0Sram) { expect_mimir_to_hsio(1, 0, /*read=*/true, 0xF4); }
// M0 is two hops from K0, behind M1.
TEST_F(KeraunosFromMimir, M0CceWritesHsio2Sram) { expect_mimir_to_hsio(0, 2, /*read=*/false, 0xF5); }
TEST_F(KeraunosFromMimir, M0CceReadsHsio2Sram) { expect_mimir_to_hsio(0, 2, /*read=*/true, 0xF6); }


// --- Phase 4: the same paths issued from CCE1 of each tile ----------------------------------------------
// CCE1 has its own DMRISC remapper and command buffers, so each path is repeated with CCE1 as the issuer, and
// as the target of the Mimir CCE's stores.
class KeraunosPathsCce1 : public KeraunosPaths {
protected:
    void SetUp() override {
        KeraunosPaths::SetUp();
        k_cce_ = 1;
    }
};

TEST_F(KeraunosPathsCce1, IdmaWritesEveryTileCce1Sram) { expect_tile_matrix(Engine::Idma, /*read=*/false, 0x110); }
TEST_F(KeraunosPathsCce1, IdmaReadsEveryTileCce1Sram) { expect_tile_matrix(Engine::Idma, /*read=*/true, 0x130); }
TEST_F(KeraunosPathsCce1, CpuWritesEveryTileCce1Sram) { expect_tile_matrix(Engine::Cpu, /*read=*/false, 0x150); }
TEST_F(KeraunosPathsCce1, CpuReadsEveryTileCce1Sram) { expect_tile_matrix(Engine::Cpu, /*read=*/true, 0x170); }

TEST_F(KeraunosPathsCce1, IdmaWritesEveryMimirCceSram) { expect_every_mimir_cce(Engine::Idma, /*read=*/false, 0x190); }
TEST_F(KeraunosPathsCce1, IdmaReadsEveryMimirCceSram) { expect_every_mimir_cce(Engine::Idma, /*read=*/true, 0x1A0); }
TEST_F(KeraunosPathsCce1, CpuWritesEveryMimirCceSram) { expect_every_mimir_cce(Engine::Cpu, /*read=*/false, 0x1B0); }
TEST_F(KeraunosPathsCce1, CpuReadsEveryMimirCceSram) { expect_every_mimir_cce(Engine::Cpu, /*read=*/true, 0x1C0); }

TEST_F(KeraunosPathsCce1, IdmaWritesM1GddrViews) { expect_mimir_gddr_views(Engine::Idma, 1, /*read=*/false, 0x1D0); }
TEST_F(KeraunosPathsCce1, IdmaReadsM1GddrViews) { expect_mimir_gddr_views(Engine::Idma, 1, /*read=*/true, 0x1D8); }
TEST_F(KeraunosPathsCce1, CpuWritesM1GddrViews) { expect_mimir_gddr_views(Engine::Cpu, 1, /*read=*/false, 0x1E0); }
TEST_F(KeraunosPathsCce1, CpuReadsM1GddrViews) { expect_mimir_gddr_views(Engine::Cpu, 1, /*read=*/true, 0x1E8); }
TEST_F(KeraunosPathsCce1, IdmaWritesM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Idma, 0, /*read=*/false, 0x1F0); }
TEST_F(KeraunosPathsCce1, IdmaReadsM0GddrThroughM1) { expect_mimir_gddr_views(Engine::Idma, 0, /*read=*/true, 0x1F8); }

class KeraunosFromMimirCce1 : public KeraunosFromMimir {
protected:
    void SetUp() override {
        KeraunosFromMimir::SetUp();
        k_cce_ = 1;
    }
};

TEST_F(KeraunosFromMimirCce1, M1CceWritesHsio2Cce1Sram) { expect_mimir_to_hsio(1, 2, /*read=*/false, 0x1A1); }
TEST_F(KeraunosFromMimirCce1, M1CceReadsHsio2Cce1Sram) { expect_mimir_to_hsio(1, 2, /*read=*/true, 0x1A2); }

// IDMA L1 accumulate (format 4, INT32_NORMAL, operation 9) across chiplets and on Mimir CCEs. The accumulate adds when
// the destination is reached through the issuing tile's SoC-local window; these cases find out what the D2D and
// package routes do with it.
class KeraunosMimirAccumulate : public KeraunosPaths {
protected:
    static constexpr uint32_t kWords = 16;
    static constexpr uint32_t kRepeats = 3;
    static constexpr uint32_t kStart = 100;

    std::vector<uint32_t> start_words() const {
        std::vector<uint32_t> s(kWords);
        for (uint32_t i = 0; i < kWords; i++) {
            s[i] = kStart + i;
        }
        return s;
    }
    std::vector<uint32_t> added_words() const {
        auto s = start_words();
        for (auto& v : s) {
            v += kRepeats;
        }
        return s;
    }

    std::vector<uint32_t> accum_args(uint64_t src, uint64_t dst, uint32_t status_offset) const {
        std::vector<uint32_t> args = split(src);
        const auto d = split(dst);
        args.insert(args.end(), d.begin(), d.end());
        args.insert(args.end(), {kWords * 4, kRepeats, /*fmt=*/4, /*op=*/9, /*flags=*/0, status_offset});
        return args;
    }

    void launch_mesh(Program& program) {
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/true);
    }

    // The Mimir CCE runs the accumulate; src and dst are addresses as that CCE issues them. Source and destination
    // blocks live in the CCE's own SRAM at the unreserved base + 0x1000 / + 0x2000; the status block at the base.
    std::vector<uint32_t> run_on_mimir(uint32_t mimir_cce, uint64_t src, uint64_t dst, bool dst_is_own) {
        const CoreCoord core = logical_mimir_cce_core(soc_desc_of(device_), mimir_cce);
        const uint64_t noc = mimir_cce_noc_addr();
        const std::vector<uint32_t> ones(kWords, 1);
        const auto start = start_words();
        std::vector<uint32_t> zero(8, 0);
        write_mimir_cce_sram(device_, core, noc, zero.data(), zero.size() * 4);
        write_mimir_cce_sram(device_, core, noc + 0x1000, ones.data(), kWords * 4);
        if (dst_is_own) {
            write_mimir_cce_sram(device_, core, noc + 0x2000, start.data(), kWords * 4);
        }
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program, std::string(kKernelDir) + "keraunos_cce_idma_accum_cfg.cpp", core, DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(program, k, core, accum_args(src, dst, mimir_cce_offset()));
        launch_mesh(program);
        std::vector<uint32_t> status(8, 0);
        read_mimir_cce_sram(device_, core, noc, status.data(), status.size() * 4);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        std::vector<uint32_t> got(kWords, 0);
        if (dst_is_own) {
            read_mimir_cce_sram(device_, core, noc + 0x2000, got.data(), kWords * 4);
        }
        return got;
    }

    void report(const char* what, uint64_t dst, const std::vector<uint32_t>& got) {
        std::cout << fmt::format("{} dst {:#x}: {}", what, dst, fmt::join(got, " ")) << std::endl;
        EXPECT_EQ(got, added_words()) << (got[0] == 1 ? "acts as a copy" : "neither a copy nor an add");
    }
};

// HSIO 2 CCE 0 (on D2D2) accumulates into a Mimir CCE's SRAM through its package address.
// Disabled on mmk (2026-11-20): the D2D route copies instead of adding.
TEST_F(KeraunosMimirAccumulate, DISABLED_KeraunosToMimirCceSram) {
    const auto k = hsio_core(2, 0);
    const auto mimir_cce = first_cce_on_mimir(1);
    ASSERT_TRUE(k.has_value() && mimir_cce.has_value());
    const CoreCoord core = logical_mimir_cce_core(soc_desc_of(device_), *mimir_cce);
    write_words(*k, kSrcOffset, std::vector<uint32_t>(kWords, 1));
    write_words(*k, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    const auto start = start_words();
    write_mimir_cce_sram(device_, core, mimir_cce_noc_addr() + 0x2000, start.data(), kWords * 4);
    const uint64_t dst = mimir_cce_sram_spa(*mimir_cce, mimir_cce_offset() + 0x2000);
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "keraunos_cce_idma_accum_cfg.cpp",
        k->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
    SetRuntimeArgs(program, kernel, k->logical, accum_args(global_addr(*k, kSrcOffset), dst, unreserved_ + kStatusOffset));
    launch(program, *k);
    std::vector<uint32_t> got(kWords, 0);
    read_mimir_cce_sram(device_, core, mimir_cce_noc_addr() + 0x2000, got.data(), kWords * 4);
    report("K->Mimir", dst, got);
}

// A Mimir CCE accumulates into its own SRAM through the route table's address (the package form).
TEST_F(KeraunosMimirAccumulate, MimirOwnSramRouteForm) {
    const auto c = first_cce_on_mimir(1);
    ASSERT_TRUE(c.has_value());
    const uint64_t base = mimir_cce_sram_spa(*c, mimir_cce_offset());
    report("Mimir own (route)", base + 0x2000, run_on_mimir(*c, base + 0x1000, base + 0x2000, true));
}

// A Mimir CCE accumulates into its own SRAM through cce-dv's Mimir-local window (SRAM_LBASE 0x1280000000).
// Disabled on mmk (2026-11-20): 0x1280000000 is not this CCE's SRAM; the route-table form is the one that works.
TEST_F(KeraunosMimirAccumulate, DISABLED_MimirOwnSramLocalForm) {
    const auto c = first_cce_on_mimir(1);
    ASSERT_TRUE(c.has_value());
    const uint64_t base = 0x1280000000ULL + mimir_cce_offset();
    report("Mimir own (local)", base + 0x2000, run_on_mimir(*c, base + 0x1000, base + 0x2000, true));
}

// A Mimir CCE accumulates into HSIO 2 CCE 0's SRAM through the Keraunos package address.
// Disabled on mmk (2026-11-20): the D2D route copies instead of adding.
TEST_F(KeraunosMimirAccumulate, DISABLED_MimirToKeraunosSram) {
    const auto k = hsio_core(2, 0);
    const auto c = first_cce_on_mimir(1);
    ASSERT_TRUE(k.has_value() && c.has_value());
    write_words(*k, kDstOffset, start_words());
    const uint64_t base = mimir_cce_sram_spa(*c, mimir_cce_offset());
    const uint64_t dst = global_addr(*k, kDstOffset);
    run_on_mimir(*c, base + 0x1000, dst, false);
    report("Mimir->K", dst, read_words(*k, kDstOffset, kWords));
}

// CCE1 accumulating to CCE0's SoC address left CCE0 unchanged; see whether it landed in CCE1's own SRAM instead.
// Disabled on mmk (2026-11-20): from CCE1, CCE0's SoC address is not reached.
TEST_F(KeraunosMimirAccumulate, DISABLED_Cce1ToCce0SocWhereItLands) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    const uint64_t soc0 = 0x22800000ULL;
    const uint64_t soc1 = 0x22C00000ULL;
    write_words(*b, kSrcOffset, std::vector<uint32_t>(kWords, 1));
    write_words(*a, kDstOffset, start_words());
    write_words(*b, kDstOffset, start_words());
    write_words(*b, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "keraunos_cce_idma_accum_cfg.cpp",
        b->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
    SetRuntimeArgs(
        program,
        kernel,
        b->logical,
        accum_args(soc1 + unreserved_ + kSrcOffset, soc0 + unreserved_ + kDstOffset, unreserved_ + kStatusOffset));
    launch(program, *b);
    const auto on0 = read_words(*a, kDstOffset, kWords);
    const auto on1 = read_words(*b, kDstOffset, kWords);
    std::cout << fmt::format("CCE1 -> {:#x}: CCE0 dst {} | CCE1 dst {}", soc0, fmt::join(on0, " "), fmt::join(on1, " "))
              << std::endl;
    const auto status = read_words(*b, kStatusOffset, kStatusWords);
    std::cout << fmt::format("status: {:#010x}", fmt::join(status, " ")) << std::endl;
    EXPECT_EQ(on0, added_words());
}

}  // namespace
}  // namespace tt::tt_metal
