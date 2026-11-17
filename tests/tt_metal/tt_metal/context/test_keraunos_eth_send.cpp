// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Keraunos Ethernet send ordering on one mmk package, set up like the KeraunosEmu cases (test_mimir_emu.cpp): a unit
// MeshDevice, IDLE_ETH kernels on DM0 through EthernetConfig. The Ethernet loopback HSIO0.eth0 <-> HSIO1.eth0 joins
// tile 0 CCE0 and tile 1 CCE0; the host writes each core's Ethernet-link slot before launch.
//
// KeraunosEthSend/Ordering.*: payload-then-credit on one TX queue (CM-087), the credit through
// eth_write_remote_reg() (CmdBuf1) or cce_eth::send_dword() directly (Credit8).
//
// Run (after bringup_mmk.py has brought the loopback up in packet mode):
//   export TT_METAL_EMU_SERVER=<host>:<port>
//   export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/mmk.yaml
//   export TT_METAL_SLOW_DISPATCH_MODE=1 KERAUNOS_ETH_LOOPBACK=1
//   ./build/test/tt_metal/unit_tests_context --gtest_filter='KeraunosEthSend/Ordering.Run/*'
// Optional: KERAUNOS_E6_ITERATIONS (1000), KERAUNOS_E6_PAYLOAD_BYTES (4096), KERAUNOS_ETH_TXQ (0),
// KERAUNOS_ETH_POLLS (per device wait, 2000000), KERAUNOS_ETH_HOST_TIMEOUT_S (1800).
// A host timeout means a kernel is stuck in an unbounded wait (a send's queue-space spin): the test prints the result
// blocks and exits the process, and the emulator job needs a reset before the next run.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include "env_lib.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "keraunos_session.hpp"
#include "mimir_contract.hpp"
#include "tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering.h"

namespace tt::tt_metal::keraunos_eth_send_test {
namespace {

using namespace ::tt::tt_metal::mimir_contract;
using namespace ::tt::tt_metal::keraunos_session;
using namespace ::keraunos_eth_ordering;

constexpr const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering_sender.cpp";
constexpr const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering_receiver.cpp";
// The loopback's ends.
constexpr uint32_t kSenderHsio = 0;
constexpr uint32_t kReceiverHsio = 1;
// Global HSIO 0 CCE SRAM window; tile j is tile_stride above it (as in KeraunosEmu.EthernetKernelWritesOtherTileSram).
constexpr uint64_t kHsioGlobalSramBase = 0x1206800000ULL;

#define skip_unless_keraunos_loopback()                                                              \
    do {                                                                                             \
        skip_unless_ethernet_package();                                                              \
        if (std::getenv("KERAUNOS_ETH_LOOPBACK") == nullptr) {                                       \
            GTEST_SKIP() << "Set KERAUNOS_ETH_LOOPBACK=1 once bringup_mmk.py has the loopback up.";  \
        }                                                                                            \
    } while (0)

struct OrderingVariant {
    const char* name;
    std::map<std::string, std::string> defines;
};

std::string format_sender(const SenderResult& r) {
    return fmt::format(
        "sender: status 0x{:08x} sent {} ack_timeout_iter {} idma_outstanding {}",
        r.status, r.iterations_sent, r.ack_timeout_iteration, r.idma_outstanding_at_end);
}

std::string format_receiver(const ReceiverResult& r) {
    return fmt::format(
        "receiver: status 0x{:08x} seen {} last_credit {} mismatches {} (resolved later {}) first: iter {} word {} got "
        "0x{:08x} want 0x{:08x} | credit skips {} first {} -> {} | timeout waiting for {}",
        r.status, r.iterations_seen, r.last_credit, r.mismatch_count, r.mismatch_resolved_count,
        r.first_mismatch_iteration, r.first_mismatch_word, r.first_mismatch_got, r.first_mismatch_want,
        r.credit_skips, r.first_skip_from, r.first_skip_to, r.timeout_waiting_for);
}

class Ordering : public ::testing::TestWithParam<OrderingVariant> {};

TEST_P(Ordering, Run) {
    skip_unless_keraunos_loopback();
    const OrderingVariant& variant = GetParam();

    Session s(kPayloadOffset + kMaxPayloadBytes);
    const auto tx = s.core(kSenderHsio);
    const auto rx = s.core(kReceiverHsio);
    if (!tx || !rx) {
        GTEST_SKIP() << "Descriptor does not list CCE0 of HSIO tiles 0 and 1.";
    }

    const uint32_t base = s.base();
    const uint32_t iterations = tt::parse_env<uint32_t>("KERAUNOS_E6_ITERATIONS", 1000);
    const uint32_t payload_bytes = tt::parse_env<uint32_t>("KERAUNOS_E6_PAYLOAD_BYTES", 4096);
    const uint32_t txq = tt::parse_env<uint32_t>("KERAUNOS_ETH_TXQ", 0);
    const uint32_t polls = tt::parse_env<uint32_t>("KERAUNOS_ETH_POLLS", 2'000'000);
    ASSERT_TRUE(payload_bytes > 0 && payload_bytes % 16 == 0 && payload_bytes <= kMaxPayloadBytes)
        << "KERAUNOS_E6_PAYLOAD_BYTES " << payload_bytes << " must be a non-zero multiple of 16 up to "
        << kMaxPayloadBytes;

    const std::vector<uint32_t> zero_result(kResultSize / sizeof(uint32_t), 0);
    s.write_eth_link(*tx, *rx);
    s.write_eth_link(*rx, *tx);
    s.write(*tx, base + kResultOffset, zero_result);
    s.write(*tx, base + kAckOffset, {0});
    s.write(*rx, base + kResultOffset, zero_result);
    s.write(*rx, base + kCreditOffset, {0, 0, 0, 0});
    s.write(*rx, base + kPayloadOffset, std::vector<uint32_t>(payload_bytes / sizeof(uint32_t), kCanary));

    const uint64_t ack_global = kHsioGlobalSramBase + static_cast<uint64_t>(tx->hsio) * s.tile_stride() + base +
                                kAckOffset;

    Program program = CreateProgram();
    const KernelHandle sender = CreateKernel(
        program,
        kSenderKernel,
        tx->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .defines = variant.defines});
    const KernelHandle receiver = CreateKernel(
        program, kReceiverKernel, rx->logical, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
    SetRuntimeArgs(program, sender, tx->logical, {base, iterations, payload_bytes, txq, polls});
    SetRuntimeArgs(
        program,
        receiver,
        rx->logical,
        {base,
         iterations,
         payload_bytes,
         polls,
         static_cast<uint32_t>(ack_global),
         static_cast<uint32_t>(ack_global >> 32)});

    auto dump = [&] {
        fmt::print(stderr, "{}\n", format_sender(s.read_struct<SenderResult>(*tx, base + kResultOffset)));
        fmt::print(stderr, "{}\n", format_receiver(s.read_struct<ReceiverResult>(*rx, base + kResultOffset)));
    };
    s.run(
        program,
        {{*tx, base + kResultOffset}, {*rx, base + kResultOffset}},
        kStatusRunning,
        std::chrono::seconds(tt::parse_env<uint32_t>("KERAUNOS_ETH_HOST_TIMEOUT_S", 1800)),
        dump);

    const auto tx_result = s.read_struct<SenderResult>(*tx, base + kResultOffset);
    const auto rx_result = s.read_struct<ReceiverResult>(*rx, base + kResultOffset);
    const std::string trace = format_sender(tx_result) + "\n" + format_receiver(rx_result);
    log_info(tt::LogTest, "Ordering {}: {}", variant.name, trace);

    // Every credit found its payload in place, none skipped, all iterations through.
    EXPECT_EQ(tx_result.status, kStatusDone) << trace;
    EXPECT_EQ(rx_result.status, kStatusDone) << trace;
    EXPECT_EQ(rx_result.iterations_seen, iterations) << trace;
    EXPECT_EQ(rx_result.mismatch_count, 0u) << trace;
    EXPECT_EQ(rx_result.credit_skips, 0u) << trace;
    EXPECT_EQ(tx_result.idma_outstanding_at_end, 0u) << trace;
    EXPECT_EQ(s.read(*rx, base + kCreditOffset, 1)[0], iterations) << trace;
    const auto payload = s.read(*rx, base + kPayloadOffset, payload_bytes / sizeof(uint32_t));
    uint32_t payload_errors = 0;
    for (uint32_t j = 0; j < payload.size(); j++) {
        payload_errors += payload[j] != payload_word(iterations, j);
    }
    EXPECT_EQ(payload_errors, 0u) << "final payload words not iteration " << iterations << "'s pattern";
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosEthSend,
    Ordering,
    ::testing::Values(
        // Sends on CMDBUF_1 as eth_cce_send.h does; the credit through eth_write_remote_reg().
        OrderingVariant{"CmdBuf1", {}},
        // The credit as compat's 8-byte inline update, cce_eth::send_dword() called directly.
        OrderingVariant{"Credit8", {{"KERAUNOS_CREDIT_SEND_DWORD", "1"}}}),
    [](const ::testing::TestParamInfo<OrderingVariant>& info) { return std::string(info.param.name); });

}  // namespace
}  // namespace tt::tt_metal::keraunos_eth_send_test
