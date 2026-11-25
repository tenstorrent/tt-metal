// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Keraunos Ethernet send ordering on one mmk package, on the Keraunos CCE fixture (keraunos_cce_fixture.hpp): IDLE_ETH
// kernels on DM0 through EthernetConfig. The Ethernet loopback HSIO0.eth0 <-> HSIO1.eth0 joins
// tile 0 CCE0 and tile 1 CCE0; the host writes each core's Ethernet-link slot before launch.
//
// KeraunosEthSend/Ordering.*: payload-then-credit on one TX queue, the credit through
// eth_write_remote_reg() (CmdBuf1) or cce_eth::send_dword() directly (Credit8).
//
// Run (once the HSIO0.eth0 <-> HSIO1.eth0 loopback is up in packet mode):
//   export TT_METAL_EMU_SERVER=<host>:<port>
//   export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/descriptors/soc_descs/mmk.yaml
//   export TT_METAL_SLOW_DISPATCH_MODE=1 TT_KERAUNOS_ETH_LOOPBACK=1
//   ./build/test/tt_metal/unit_tests_context --gtest_filter='KeraunosEthSend/Ordering.Run/*'
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
#include <thread>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include "hostdev/eth_link.h"
#include "env_lib.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "keraunos_cce_fixture.hpp"
#include "mimir_contract.hpp"
#include "tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering.h"

namespace tt::tt_metal::keraunos_eth_send_test {
namespace {

using namespace ::tt::tt_metal::mimir_contract;
using namespace ::keraunos_eth_ordering;

constexpr const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering_sender.cpp";
constexpr const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering_receiver.cpp";
// The loopback's ends.
constexpr uint32_t kSenderHsio = 0;
constexpr uint32_t kReceiverHsio = 1;
constexpr uint32_t kIterations = 8;
constexpr uint32_t kPayloadBytes = 256;
static_assert(
    kPayloadBytes > 0 && kPayloadBytes % 16 == 0 && kPayloadBytes <= kMaxPayloadBytes,
    "the payload is a non-zero multiple of 16 bytes that fits the receiver's block");
constexpr uint32_t kTxq = 0;
constexpr uint32_t kPolls = 2'000'000;  // per device wait
constexpr std::chrono::seconds kHostTimeout{1800};

// TT_KERAUNOS_ETH_LOOPBACK=1: the loopback is up. Unset or 0 skips the suite; any other value throws.
bool keraunos_loopback_up() {
    const auto value = tt::parse_env<std::string>("TT_KERAUNOS_ETH_LOOPBACK", "0");
    TT_FATAL(value == "0" || value == "1", "TT_KERAUNOS_ETH_LOOPBACK must be 0 or 1, got '{}'", value);
    return value == "1";
}

#define skip_unless_keraunos_loopback()                                                                \
    do {                                                                                               \
        skip_unless_ethernet_package();                                                                \
        if (!keraunos_loopback_up()) {                                                                 \
            GTEST_SKIP() << "Set TT_KERAUNOS_ETH_LOOPBACK=1 once tiles 0 and 1 have the loopback up."; \
        }                                                                                              \
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

// The shared Keraunos CCE fixture with the test's blocks above the Ethernet-link slot, which sits inside IDLE_ETH's
// unreserved region. Offsets passed to the fixture are from its unreserved base; the kernels take L1 addresses.
class Ordering : public keraunos_test::KeraunosCceTest, public ::testing::WithParamInterface<OrderingVariant> {
protected:
    // Opening the device costs minutes on the emulator: not without the loopback.
    static void SetUpTestSuite() {
        if (keraunos_loopback_up()) {
            KeraunosCceTest::SetUpTestSuite();
        }
    }

    void SetUp() override {
        skip_unless_keraunos_loopback();
        KeraunosCceTest::SetUp();
        if (IsSkipped() || HasFatalFailure()) {
            return;
        }
        base_ = (MEM_AERISC_UNRESERVED_BASE + 63) & ~63u;
        ASSERT_TRUE(base_ >= unreserved_ && base_ + kPayloadOffset + kMaxPayloadBytes <= MEM_IERISC_L1_SIZE)
            << fmt::format("Test base 0x{:x} does not fit the IDLE_ETH unreserved region", base_);
    }

    // The fixture's offset of the test's block at @p offset.
    uint32_t at(uint32_t offset) const { return base_ - unreserved_ + offset; }

    template <typename T>
    T read_struct(const keraunos_test::Hsio& core, uint32_t offset) {
        T value{};
        read_eth_sram(device_, core.logical, unreserved_noc_ + at(offset), &value, sizeof(value));
        return value;
    }

    void write_eth_link(const keraunos_test::Hsio& core, const keraunos_test::Hsio& peer) {
        const EthLink link{.peer_tile = peer.hsio, .valid = ETH_LINK_VALID};
        write_eth_sram(
            device_,
            core.logical,
            unreserved_noc_ + MEM_IERISC_FABRIC_ETH_LINK_BASE - unreserved_,
            &link,
            sizeof(link));
    }

    // Launches without blocking, polls each core's status word until it leaves kStatusRunning or the host timeout
    // passes, then waits for the program. On a host timeout the cores are hung, so @p dump prints what they recorded
    // and the process exits: nothing after it could reach the device.
    void run(Program& program, const std::vector<keraunos_test::Hsio>& watched, const std::function<void()>& dump) {
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/false);

        const auto deadline = std::chrono::steady_clock::now() + kHostTimeout;
        for (const keraunos_test::Hsio& core : watched) {
            for (uint32_t status = read_words(core, at(kResultOffset), 1)[0]; status == kStatusRunning || status == 0;
                 status = read_words(core, at(kResultOffset), 1)[0]) {
                if (std::chrono::steady_clock::now() > deadline) {
                    fmt::print(stderr, "Keraunos Ethernet send: host timeout; device state follows\n");
                    dump();
                    fmt::print(stderr, "Exiting: the kernels are stuck. Reset the emulator job before the next run.\n");
                    std::fflush(stderr);
                    std::_Exit(3);
                }
                // A device read takes seconds on the emulator; a shorter interval is back-to-back reads.
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        distributed::Finish(mesh_->mesh_command_queue());
    }

    uint32_t base_ = 0;  // L1 address of the test's blocks
};

TEST_P(Ordering, Run) {
    const OrderingVariant& variant = GetParam();

    const auto tx = hsio_core(kSenderHsio);
    const auto rx = hsio_core(kReceiverHsio);
    if (!tx || !rx || !device_->ethernet_cores().contains(tx->logical) ||
        !device_->ethernet_cores().contains(rx->logical)) {
        GTEST_SKIP() << "Descriptor does not list CCE0 of HSIO tiles 0 and 1.";
    }

    const uint32_t base = base_;

    const std::vector<uint32_t> zero_result(kResultSize / sizeof(uint32_t), 0);
    write_eth_link(*tx, *rx);
    write_eth_link(*rx, *tx);
    write_words(*tx, at(kResultOffset), zero_result);
    write_words(*tx, at(kAckOffset), {0});
    write_words(*rx, at(kResultOffset), zero_result);
    write_words(*rx, at(kCreditOffset), {0, 0, 0, 0});
    write_words(*rx, at(kPayloadOffset), std::vector<uint32_t>(kPayloadBytes / sizeof(uint32_t), kCanary));

    const uint64_t ack_global = global_addr(*tx, at(kAckOffset));

    // Both kernels reach the NOC through cce_noc.h.
    std::map<std::string, std::string> sender_defines = variant.defines;
    sender_defines.emplace("KERAUNOS_CCE_NOC", "1");
    Program program = CreateProgram();
    const KernelHandle sender = CreateKernel(
        program,
        kSenderKernel,
        tx->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .defines = sender_defines});
    const KernelHandle receiver = CreateKernel(
        program,
        kReceiverKernel,
        rx->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .defines = {{"KERAUNOS_CCE_NOC", "1"}}});
    SetRuntimeArgs(program, sender, tx->logical, {base, kIterations, kPayloadBytes, kTxq, kPolls});
    SetRuntimeArgs(
        program,
        receiver,
        rx->logical,
        {base,
         kIterations,
         kPayloadBytes,
         kPolls,
         static_cast<uint32_t>(ack_global),
         static_cast<uint32_t>(ack_global >> 32)});

    auto dump = [&] {
        fmt::print(stderr, "{}\n", format_sender(read_struct<SenderResult>(*tx, kResultOffset)));
        fmt::print(stderr, "{}\n", format_receiver(read_struct<ReceiverResult>(*rx, kResultOffset)));
    };
    run(program, {*tx, *rx}, dump);

    const auto tx_result = read_struct<SenderResult>(*tx, kResultOffset);
    const auto rx_result = read_struct<ReceiverResult>(*rx, kResultOffset);
    const std::string trace = format_sender(tx_result) + "\n" + format_receiver(rx_result);
    log_info(tt::LogTest, "Ordering {}: {}", variant.name, trace);

    // Every credit found its payload in place, none skipped, all iterations through.
    EXPECT_EQ(tx_result.status, kStatusDone) << trace;
    EXPECT_EQ(rx_result.status, kStatusDone) << trace;
    EXPECT_EQ(rx_result.iterations_seen, kIterations) << trace;
    EXPECT_EQ(rx_result.mismatch_count, 0u) << trace;
    EXPECT_EQ(rx_result.credit_skips, 0u) << trace;
    EXPECT_EQ(tx_result.idma_outstanding_at_end, 0u) << trace;
    EXPECT_EQ(read_words(*rx, at(kCreditOffset), 1)[0], kIterations) << trace;
    const auto payload = read_words(*rx, at(kPayloadOffset), kPayloadBytes / sizeof(uint32_t));
    uint32_t payload_errors = 0;
    for (uint32_t j = 0; j < payload.size(); j++) {
        payload_errors += payload[j] != payload_word(kIterations, j);
    }
    EXPECT_EQ(payload_errors, 0u) << "final payload words not iteration " << kIterations << "'s pattern";
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosEthSend,
    Ordering,
    ::testing::Values(
        // Sends on CMDBUF_1 as eth_cce_send.h does; the credit through eth_write_remote_reg().
        OrderingVariant{"CmdBuf1", {}},
        // The credit as an 8-byte inline update, cce_eth::send_dword() called directly.
        OrderingVariant{"Credit8", {{"KERAUNOS_CREDIT_SEND_DWORD", "1"}}}),
    [](const ::testing::TestParamInfo<OrderingVariant>& info) { return std::string(info.param.name); });

}  // namespace
}  // namespace tt::tt_metal::keraunos_eth_send_test
