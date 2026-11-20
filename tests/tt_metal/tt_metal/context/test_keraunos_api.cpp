// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 5 of keraunos_plan.md: the device API in experimental/keraunos_cce.h and the host-written route table
// behind it. Each test runs a kernel that builds its addresses with the API, then checks the data from the host,
// on both CCEs of the issuing tile.

#include "keraunos_cce_fixture.hpp"

#include <tt-umd/coordinates/grendel_noc_address_resolver.hpp>

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

constexpr uint32_t kIdentityStatusOffset = 0x100;  // relative to the unreserved base

class KeraunosApi : public KeraunosCceTest {
protected:
    enum class Kind : uint32_t { Hsio = 0, MimirCceSram = 1, MimirGddr = 2 };

    uint32_t mimir_cce_offset() const {
        return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }
    uint64_t mimir_cce_noc_addr() const {
        return MetalContext::instance().hal().get_dev_noc_addr(
            HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }
    std::optional<uint32_t> first_cce_on_mimir(uint32_t mimir) const {
        for (uint32_t c = 0; c < num_mimir_cces_on(device_); ++c) {
            if (mimir_index_for_cce(soc_desc_of(device_), c) == mimir) {
                return c;
            }
        }
        return std::nullopt;
    }
    static std::vector<uint32_t> pattern(uint32_t tag, uint32_t bytes) {
        std::vector<uint32_t> p(bytes / sizeof(uint32_t));
        for (uint32_t i = 0; i < p.size(); i++) {
            p[i] = (tag << 16) | i;
        }
        return p;
    }

    // What the identity kernel reports for HSIO `hsio` CCE `cce`.
    void expect_identity(uint32_t hsio, uint32_t cce) {
        const auto c = hsio_core(hsio, cce);
        ASSERT_TRUE(c.has_value());
        write_words(*c, kIdentityStatusOffset, std::vector<uint32_t>(16, 0));
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_api_identity.cpp",
            c->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        SetRuntimeArgs(program, k, c->logical, {unreserved_ + kIdentityStatusOffset});
        launch(program, *c, kIdentityStatusOffset);
        const auto w = read_words(*c, kIdentityStatusOffset, 16);
        EXPECT_EQ(w[4], kAcked) << fmt::format("status: {:#x}", fmt::join(w, " "));
        EXPECT_EQ(w[0], cce);
        EXPECT_EQ(w[1], hsio);
        EXPECT_EQ(w[2], 0u);  // hart 0
        EXPECT_EQ(w[3], hsio_count());
        const uint64_t own = global_addr(*c, 0) - unreserved_;
        EXPECT_EQ((static_cast<uint64_t>(w[7]) << 32) | w[6], own);
        const tt::umd::GrendelAddressWindows win{};
        EXPECT_EQ((static_cast<uint64_t>(w[9]) << 32) | w[8], win.dram_base + (16 + 1) * win.dram_stride);
        EXPECT_EQ((static_cast<uint64_t>(w[11]) << 32) | w[10], win.dram_l1_base + 1 * win.dram_l1_stride);
        EXPECT_EQ(w[12], cce == 0 ? 0x800000u : 0xC00000u);  // own uncached alias
        EXPECT_EQ(w[13], cce == 0 ? 0xC00000u : 0x800000u);  // the sibling's
    }

    uint32_t hsio_count() const {
        uint32_t n = 0;
        for (const auto& core : soc_desc_of(device_).get_arch_descriptor().get_keraunos().cores) {
            n += core.eth_cce == 0 ? 1 : 0;
        }
        return n;
    }

    // One transfer from HSIO 2's CCE `cce` to a target the API addresses, checked from the host.
    void expect_transfer(Kind kind, uint32_t index, uint32_t target_cce, uint32_t cce, bool cpu, bool read, uint32_t tag) {
        const auto k = hsio_core(2, cce);
        ASSERT_TRUE(k.has_value());
        constexpr uint32_t kBytes = 256;
        const auto data = pattern(tag, kBytes);
        const uint32_t local_offset = unreserved_ + (read ? kDstOffset : kSrcOffset);
        uint64_t remote_offset = 0;
        GddrBackingScope gddr_backing;
        std::optional<Hsio> remote_hsio;
        std::optional<CoreCoord> mimir_core;
        uint32_t gddr_offset = 0;
        switch (kind) {
            case Kind::Hsio:
                remote_hsio = hsio_core(index, target_cce);
                ASSERT_TRUE(remote_hsio.has_value());
                remote_offset = unreserved_ + kDstOffset + 0x1000;
                break;
            case Kind::MimirCceSram: {
                const auto mimir_cce = first_cce_on_mimir(1);
                ASSERT_TRUE(mimir_cce.has_value());
                mimir_core = logical_mimir_cce_core(soc_desc_of(device_), index);
                remote_offset = mimir_cce_offset();
                break;
            }
            case Kind::MimirGddr:
                gddr_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
                remote_offset = gddr_offset;
                break;
        }
        const std::vector<uint32_t> zero(data.size(), 0);
        auto host_write = [&](const std::vector<uint32_t>& v) {
            switch (kind) {
                case Kind::Hsio: write_words(*remote_hsio, remote_offset - unreserved_, v); break;
                case Kind::MimirCceSram:
                    write_mimir_cce_sram(device_, *mimir_core, mimir_cce_noc_addr(), v.data(), kBytes);
                    break;
                case Kind::MimirGddr: write_mimir_gddr(*mesh_, device_, index, gddr_offset, v); break;
            }
        };
        auto host_read = [&]() {
            std::vector<uint32_t> got(data.size(), 0);
            switch (kind) {
                case Kind::Hsio: got = read_words(*remote_hsio, remote_offset - unreserved_, data.size()); break;
                case Kind::MimirCceSram:
                    read_mimir_cce_sram(device_, *mimir_core, mimir_cce_noc_addr(), got.data(), kBytes);
                    break;
                case Kind::MimirGddr: read_mimir_gddr(*mesh_, device_, index, gddr_offset, kBytes, got); break;
            }
            return got;
        };
        write_words(*k, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        if (read) {
            host_write(data);
            write_words(*k, kDstOffset, zero);
        } else {
            write_words(*k, kSrcOffset, data);
            host_write(zero);
        }
        Program program = CreateProgram();
        const KernelHandle kernel = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_api_transfer.cpp",
            k->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        std::vector<uint32_t> args = {
            static_cast<uint32_t>(kind), index, target_cce, static_cast<uint32_t>(remote_offset),
            static_cast<uint32_t>(remote_offset >> 32), kBytes, local_offset, read ? 1u : 0u,
            unreserved_ + kStatusOffset, cpu ? 1u : 0u};
        SetRuntimeArgs(program, kernel, k->logical, args);
        launch(program, *k);
        const auto status = read_words(*k, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        if (read) {
            EXPECT_EQ(read_words(*k, kDstOffset, data.size()), data);
        } else {
            EXPECT_EQ(host_read(), data);
        }
    }

    // Both engines, both directions, from `cce`.
    void expect_every_engine(Kind kind, uint32_t index, uint32_t target_cce, uint32_t cce, uint32_t tag) {
        for (bool cpu : {false, true}) {
            for (bool read : {false, true}) {
                SCOPED_TRACE(fmt::format("cce {} {} {}", cce, cpu ? "cpu" : "idma", read ? "read" : "write"));
                expect_transfer(kind, index, target_cce, cce, cpu, read, tag + (cpu ? 2 : 0) + (read ? 1 : 0));
                if (HasFatalFailure()) {
                    return;
                }
            }
        }
    }
};

// The identity and the route table the host wrote, on both CCEs of every tile.
TEST_F(KeraunosApi, IdentityAndRoutesOnEveryCce) {
    for (uint32_t hsio = 0; hsio < hsio_count(); ++hsio) {
        for (uint32_t cce = 0; cce < 2; ++cce) {
            SCOPED_TRACE(fmt::format("HSIO {} CCE {}", hsio, cce));
            expect_identity(hsio, cce);
            if (HasFatalFailure()) {
                return;
            }
        }
    }
}

TEST_F(KeraunosApi, Cce0TransfersToOtherTileSram) { expect_every_engine(Kind::Hsio, 1, 0, 0, 0x210); }
TEST_F(KeraunosApi, Cce0TransfersToOtherTileCce1Sram) { expect_every_engine(Kind::Hsio, 3, 1, 0, 0x220); }
TEST_F(KeraunosApi, Cce1TransfersToOtherTileSram) { expect_every_engine(Kind::Hsio, 1, 0, 1, 0x230); }
TEST_F(KeraunosApi, Cce1TransfersToOtherTileCce1Sram) { expect_every_engine(Kind::Hsio, 3, 1, 1, 0x240); }

TEST_F(KeraunosApi, Cce0TransfersToMimirCceSram) {
    for (uint32_t c = 0; c < num_mimir_cces_on(device_); ++c) {
        SCOPED_TRACE(fmt::format("Mimir CCE {}", c));
        expect_every_engine(Kind::MimirCceSram, c, 0, 0, 0x250 + c * 4);
        if (HasFatalFailure()) {
            return;
        }
    }
}
TEST_F(KeraunosApi, Cce1TransfersToMimirCceSram) {
    for (uint32_t c = 0; c < num_mimir_cces_on(device_); ++c) {
        SCOPED_TRACE(fmt::format("Mimir CCE {}", c));
        expect_every_engine(Kind::MimirCceSram, c, 0, 1, 0x270 + c * 4);
        if (HasFatalFailure()) {
            return;
        }
    }
}

TEST_F(KeraunosApi, Cce0TransfersToM1Gddr) { expect_every_engine(Kind::MimirGddr, 1, 0, 0, 0x290); }
TEST_F(KeraunosApi, Cce0TransfersToM0Gddr) { expect_every_engine(Kind::MimirGddr, 0, 0, 0, 0x2A0); }
TEST_F(KeraunosApi, Cce1TransfersToM1Gddr) { expect_every_engine(Kind::MimirGddr, 1, 0, 1, 0x2B0); }
TEST_F(KeraunosApi, Cce1TransfersToM0Gddr) { expect_every_engine(Kind::MimirGddr, 0, 0, 1, 0x2C0); }

// CCE0 and CCE1 of a tile hand a counter back and forth with peer_signal / peer_wait.
TEST_F(KeraunosApi, PeerSignalAndWait) {
    const auto cce0 = hsio_core(0, 0);
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce0.has_value() && cce1.has_value());
    constexpr uint32_t kRounds = 64;
    constexpr uint32_t kFlagA = 0x400;
    constexpr uint32_t kFlagB = 0x440;
    write_words(*cce0, kFlagA, {0});
    write_words(*cce0, kFlagB, {0});
    write_words(*cce0, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    write_words(*cce1, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    uint32_t role = 0;
    for (const Hsio* core : {&*cce0, &*cce1}) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_api_peer_sync.cpp",
            core->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            program, k, core->logical,
            {role++, kRounds, unreserved_ + kFlagA, unreserved_ + kFlagB, unreserved_ + kStatusOffset});
    }
    launch(program, *cce0);
    for (const Hsio* core : {&*cce0, &*cce1}) {
        const auto status = read_words(*core, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[1], kRounds) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(status[4], kAcked);
    }
    EXPECT_EQ(read_words(*cce0, kFlagA, 1), std::vector<uint32_t>{kRounds});
    EXPECT_EQ(read_words(*cce0, kFlagB, 1), std::vector<uint32_t>{kRounds});
}

}  // namespace
}  // namespace tt::tt_metal
