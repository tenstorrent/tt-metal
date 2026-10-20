// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet TX queue probe for the fabric throughput study (fabric_exp/RESULTS.md §16). With channel trimming, 2D
// fabric writes stop at 187.6 cycles/packet once a packet no longer fits one Ethernet packet, and one-way neighbor
// exchange stops at ~130 below that. The explanation so far: the TXQs run in resend mode, a queue's sequence acks
// come back only with traffic on the same queue in the other direction (else every LOCAL_SEQ_UPDATE_TIMEOUT cycles),
// so a one-way queue sends ~128 Ethernet packets per timeout. This probe tests that on a bare link, no fabric, and
// measures what splitting the packets over two queues gives.
//
// One kernel on ERISC0 at each end of one link between two chips. The sender sends packets as the router does
// (payload into a landing slot, then a pointer update on the same queue), on one queue or round-robin over TXQ0 and
// TXQ1; the receiver takes them in order (merging the two queues) and may send small messages back. Variants:
//
//   1  data on TXQ0, nothing back                         -> the one-way floor, with no router in the way
//   2  data round-robin on TXQ0 + TXQ1, nothing back      -> the split: two sequence windows?
//   3  data on TXQ0, 16 B back on TXQ1 every packet       -> today's two-ERISC fabric (credits on TXQ1)
//   4  data on TXQ0, 16 B back on TXQ0 every 4 packets    -> credits on the data queue (the queue swap)
//   5  as 3, and the sender waits on the counts as credits for its slots
//   6  as 4, gated the same way
//   7  as 1, with the receiver's TXQ0 keepalive period set to each of TXQ_PROBE_SEQ_TIMEOUTS -> the timeout fix
//   8  as 5 (today's fabric), with the same keepalive periods
//
// Nothing gates the sender in 1-4: the receiver only counts packets and checks each landing slot holds that packet
// (or a later one for the same slot). Cycles are the cores' mcycle, which runs at the Ethernet clock (1.35 GHz).
//
// Needs slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1) and fabric off, so ERISC0 is free. Environment:
// TXQ_PROBE_CHIPS ("0,1": sender, receiver), TXQ_PROBE_LINK (0: which link between them), TXQ_PROBE_VARIANTS
// ("1,2,3,4,5,6"; variant 2 drives two queues from one core, which the router hung on in tt-metal#29073; here it ran
// cleanly), TXQ_PROBE_PAYLOADS ("2048,4000,4176": 4176 is a 4096 B payload plus an 80 B 2D header),
// TXQ_PROBE_PACKETS (50000), TXQ_PROBE_WARMUP (5000), TXQ_PROBE_SLOTS (16), TXQ_PROBE_SEQ_TIMEOUTS
// ("4000,2000,1000,500": keepalive periods in cycles for variants 7 and 8; firmware sets 8000), TXQ_PROBE_CSV.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/tt_fabric/eth_txq_probe/kernels/eth_txq_probe_common.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace eth_txq_probe;

constexpr const char* kKernel = "tests/tt_metal/tt_fabric/eth_txq_probe/kernels/eth_txq_probe.cpp";
constexpr double kClockGhz = 1.35;

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : fallback;
}

std::vector<uint32_t> env_list(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    std::stringstream ss(v ? v : fallback);
    std::vector<uint32_t> out;
    for (std::string item; std::getline(ss, item, ',');) {
        out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

struct Variant {
    uint32_t id;
    const char* name;
    uint32_t data_queues;
    uint32_t reverse_queue;
    uint32_t reverse_every;
    bool gated;
    bool sweeps_seq_timeout;  // run once per TXQ_PROBE_SEQ_TIMEOUTS entry
};

constexpr Variant kVariants[] = {
    {1, "TXQ0, nothing back", 1, kNoQueue, 0, false, false},
    {2, "TXQ0+TXQ1 split, nothing back", 2, kNoQueue, 0, false, false},
    {3, "TXQ0, back on TXQ1 every pkt", 1, 1, 1, false, false},
    {4, "TXQ0, back on TXQ0 every 4", 1, 0, 4, false, false},
    {5, "TXQ0, credits on TXQ1 every pkt", 1, 1, 1, true, false},
    {6, "TXQ0, credits on TXQ0 every 4", 1, 0, 4, true, false},
    {7, "TXQ0, nothing back", 1, kNoQueue, 0, false, true},
    {8, "TXQ0, credits on TXQ1 every pkt", 1, 1, 1, true, true},
};

const char* status_name(uint32_t s) {
    switch (s) {
        case kRunning: return "still running";
        case kDone: return "done";
        case kErrSenderTxq: return "sender: TXQ stayed busy";
        case kErrSenderCredits: return "sender: no credits";
        case kErrSenderDone: return "sender: no done flag";
        case kErrReceiverQ0: return "receiver: TXQ0 packet never arrived";
        case kErrReceiverQ1: return "receiver: TXQ1 packet never arrived";
        case kErrReceiverTxq: return "receiver: TXQ stayed busy";
        default: return "unknown";
    }
}

}  // namespace

TEST(EthTxqProbe, OneWayLinkThroughput) {
    if (MetalContext::instance().get_cluster().arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole only";
    }
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "needs TT_METAL_SLOW_DISPATCH_MODE=1 (fast dispatch cannot launch active Ethernet kernels)";
    }

    const std::vector<uint32_t> chips = env_list("TXQ_PROBE_CHIPS", "0,1");
    ASSERT_EQ(chips.size(), 2u);
    const uint32_t link_index = env_u32("TXQ_PROBE_LINK", 0);
    const std::vector<uint32_t> variant_ids = env_list("TXQ_PROBE_VARIANTS", "1,2,3,4,5,6");
    const std::vector<uint32_t> payloads = env_list("TXQ_PROBE_PAYLOADS", "2048,4000,4176");
    const uint32_t num_packets = env_u32("TXQ_PROBE_PACKETS", 50000);
    const uint32_t warmup = env_u32("TXQ_PROBE_WARMUP", 5000);
    const uint32_t slots = env_u32("TXQ_PROBE_SLOTS", 16);
    const std::vector<uint32_t> seq_timeouts = env_list("TXQ_PROBE_SEQ_TIMEOUTS", "4000,2000,1000,500");
    ASSERT_LT(warmup + 1, num_packets);
    ASSERT_TRUE(slots > 0 && (slots & (slots - 1)) == 0) << "TXQ_PROBE_SLOTS must be a power of two";
    for (uint32_t p : payloads) {
        ASSERT_TRUE(p > 0 && p <= kSlotBytes && p % 16 == 0) << "payload " << p << " B";
    }

    auto meshes = distributed::MeshDevice::create_unit_meshes({static_cast<int>(chips[0]), static_cast<int>(chips[1])});
    if (MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::DISABLED) {
        GTEST_FAIL() << "fabric was enabled; its routers would occupy the ERISCs";
    }
    auto sender_mesh = meshes.at(chips[0]);
    auto receiver_mesh = meshes.at(chips[1]);
    IDevice* sender_device = sender_mesh->get_devices()[0];
    IDevice* receiver_device = receiver_mesh->get_devices()[0];

    // The link: active Ethernet cores of the sender chip connected to the receiver chip, in a stable order.
    std::vector<std::pair<CoreCoord, CoreCoord>> links;
    for (const auto& core : sender_device->get_active_ethernet_cores(true)) {
        const auto [peer_chip, peer_core] = sender_device->get_connected_ethernet_core(core);
        if (peer_chip == receiver_device->id()) {
            links.emplace_back(core, peer_core);
        }
    }
    std::sort(links.begin(), links.end());
    ASSERT_LT(link_index, links.size()) << "chips " << chips[0] << " and " << chips[1] << " have " << links.size()
                                        << " links";
    const auto [sender_logical, receiver_logical] = links[link_index];
    const CoreCoord sender_virtual = sender_mesh->virtual_core_from_logical_core(sender_logical, CoreType::ETH);
    const CoreCoord receiver_virtual = receiver_mesh->virtual_core_from_logical_core(receiver_logical, CoreType::ETH);

    const auto& hal = MetalContext::instance().hal();
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t used = kSlotsOffset + slots * kSlotBytes;
    ASSERT_LE(used, size) << "TXQ_PROBE_SLOTS too large for Ethernet L1";

    log_info(
        LogTest,
        "Sender chip {} Ethernet core logical ({},{}) virtual ({},{}) -> receiver chip {} logical ({},{}) virtual "
        "({},{}); link {} of {}; {} packets ({} warm-up), {} slots of {} B",
        chips[0],
        sender_logical.x,
        sender_logical.y,
        sender_virtual.x,
        sender_virtual.y,
        chips[1],
        receiver_logical.x,
        receiver_logical.y,
        receiver_virtual.x,
        receiver_virtual.y,
        link_index,
        links.size(),
        num_packets,
        warmup,
        slots,
        kSlotBytes);

    std::ofstream csv;
    if (const char* path = std::getenv("TXQ_PROBE_CSV")) {
        csv.open(path);
        csv << "variant,name,seq_timeout,seq_timeout_read_back,seq_timeout_firmware,payload,status_tx,status_rx,"
               "rx_cyc_per_pkt,tx_cyc_per_pkt,rx_all_cyc_per_pkt,gbps,merge_errors,tx_rxq0_drops,tx_rxq1_drops,"
               "rx_rxq0_drops,rx_rxq1_drops";
        for (const char* side : {"tx", "rx"}) {
            for (uint32_t q = 0; q < kNumQueues; q++) {
                for (const char* counter : {"transfers", "pkts_started", "pkts_ended", "words96", "resends"}) {
                    csv << "," << side << "_q" << q << "_" << counter;
                }
            }
        }
        csv << "\n";
    }

    auto read_results = [&](IDevice* device, const CoreCoord& core) {
        std::vector<uint32_t> words(R_COUNT);
        cluster.read_core(
            words.data(), R_COUNT * sizeof(uint32_t), tt_cxy_pair(device->id(), core), base + kResultsOffset);
        return words;
    };
    auto delta = [](const std::vector<uint32_t>& r, uint32_t q, uint32_t c) {
        return r[R_TXQ_AFTER + q * kTxqCounters + c] - r[R_TXQ_BEFORE + q * kTxqCounters + c];
    };
    auto drops = [](const std::vector<uint32_t>& r, uint32_t q) {
        return r[R_RXQ_DROPS_AFTER + q] - r[R_RXQ_DROPS_BEFORE + q];
    };

    bool failed = false;
    for (uint32_t id : variant_ids) {
        const Variant* v = nullptr;
        for (const Variant& candidate : kVariants) {
            if (candidate.id == id) {
                v = &candidate;
            }
        }
        ASSERT_NE(v, nullptr) << "no variant " << id;
        if (v->gated) {
            ASSERT_GT(slots, v->reverse_every) << "gated variants need more slots than packets per credit";
        }
        const std::vector<uint32_t> timeouts = v->sweeps_seq_timeout ? seq_timeouts : std::vector<uint32_t>{0};
        for (uint32_t timeout : timeouts) {
            for (uint32_t payload : payloads) {
                const std::vector<uint32_t> zeros(used / sizeof(uint32_t), 0);
                cluster.write_core(zeros.data(), used, tt_cxy_pair(sender_device->id(), sender_virtual), base);
                cluster.write_core(zeros.data(), used, tt_cxy_pair(receiver_device->id(), receiver_virtual), base);
                cluster.l1_barrier(sender_device->id());
                cluster.l1_barrier(receiver_device->id());

                const std::vector<uint32_t> ct_args = {v->data_queues, v->reverse_queue, v->reverse_every, v->gated};
                auto make_workload = [&](const CoreCoord& core, bool is_sender) {
                    Program program = CreateProgram();
                    const KernelHandle kernel = CreateKernel(
                        program,
                        kKernel,
                        core,
                        EthernetConfig{
                            .noc = NOC::NOC_0,
                            .processor = DataMovementProcessor::RISCV_0,
                            .compile_args = ct_args,
                            .opt_level = KernelBuildOptLevel::O3});
                    SetRuntimeArgs(
                        program,
                        kernel,
                        core,
                        {base, is_sender ? 1u : 0u, payload, num_packets, warmup, slots, timeout});
                    distributed::MeshWorkload workload;
                    workload.add_program(
                        distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
                    return workload;
                };
                distributed::MeshWorkload sender_workload = make_workload(sender_logical, true);
                distributed::MeshWorkload receiver_workload = make_workload(receiver_logical, false);
                // Slow dispatch blocks until a program finishes, and each kernel waits for the other in its handshake.
                auto run = [](const std::shared_ptr<distributed::MeshDevice>& mesh, distributed::MeshWorkload& w) {
                    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), w, false);
                    distributed::Finish(mesh->mesh_command_queue());
                };
                std::thread rx([&] { run(receiver_mesh, receiver_workload); });
                std::thread tx([&] { run(sender_mesh, sender_workload); });
                tx.join();
                rx.join();

                const std::vector<uint32_t> s = read_results(sender_device, sender_virtual);
                const std::vector<uint32_t> r = read_results(receiver_device, receiver_virtual);
                const uint32_t measured = num_packets - 1 - warmup;
                const double rx_cyc = static_cast<double>(r[R_T_LAST] - r[R_T_WARM]) / measured;
                const double rx_all = static_cast<double>(r[R_T_LAST] - r[R_T_FIRST]) / (num_packets - 1);
                const double tx_cyc = static_cast<double>(s[R_T_LAST] - s[R_T_WARM]) / (num_packets - warmup);
                const double gbps = rx_cyc > 0 ? payload / rx_cyc * kClockGhz : 0.0;
                const uint32_t sent_q1 = v->data_queues == 2 ? num_packets / 2 : 0;
                const uint32_t sent_q0 = num_packets - sent_q1;
                const uint32_t sent[kNumQueues] = {sent_q0, sent_q1};

                std::string queues;
                for (uint32_t q = 0; q < kNumQueues; q++) {
                    // Counters: TRANSFER_CNT, PKT_START_CNT, PKT_END_CNT, WORD_CNT (96 B units), RESEND_CNT. The packet
                    // counts read up to ~20% low at 4000 B and above (exact at 2048 B); TRANSFER_CNT is exact.
                    if (delta(s, q, 0) == 0 && delta(r, q, 0) == 0) {
                        continue;
                    }
                    queues += fmt::format(
                        " | q{} tx {:.2f} eth/pkt {} resends, rx {} transfers {} eth pkts",
                        q,
                        sent[q] ? static_cast<double>(delta(s, q, 1)) / sent[q] : 0.0,
                        delta(s, q, 4),
                        delta(r, q, 0),
                        delta(r, q, 1));
                }
                const std::string keepalive = timeout == 0 ? std::string()
                                                           : fmt::format(
                                                                 ", keepalive {} (read back {}, firmware {})",
                                                                 timeout,
                                                                 r[R_SEQ_TIMEOUT_USED],
                                                                 r[R_SEQ_TIMEOUT_FIRMWARE]);
                log_info(
                    LogTest,
                    "{} {:32s}{} {:5d} B: rx {:6.1f} cyc/pkt ({:5.1f} incl. warm-up), tx {:6.1f}, {:5.1f} GB/s, "
                    "merge errors {}, RX drops tx side {}/{} rx side {}/{}{}{}",
                    v->id,
                    v->name,
                    keepalive,
                    payload,
                    rx_cyc,
                    rx_all,
                    tx_cyc,
                    gbps,
                    r[R_MERGE_ERRORS],
                    drops(s, 0),
                    drops(s, 1),
                    drops(r, 0),
                    drops(r, 1),
                    queues,
                    (s[R_STATUS] == kDone && r[R_STATUS] == kDone) ? std::string()
                                                                   : fmt::format(
                                                                         " | FAILED tx: {} at {}, rx: {} at {}",
                                                                         status_name(s[R_STATUS]),
                                                                         s[R_ERR_INDEX],
                                                                         status_name(r[R_STATUS]),
                                                                         r[R_ERR_INDEX]));
                if (csv.is_open()) {
                    csv << v->id << ",\"" << v->name << "\"," << timeout << "," << r[R_SEQ_TIMEOUT_USED] << ","
                        << r[R_SEQ_TIMEOUT_FIRMWARE] << "," << payload << "," << s[R_STATUS] << "," << r[R_STATUS]
                        << "," << rx_cyc << "," << tx_cyc << "," << rx_all << "," << gbps << "," << r[R_MERGE_ERRORS]
                        << "," << drops(s, 0) << "," << drops(s, 1) << "," << drops(r, 0) << "," << drops(r, 1);
                    for (const auto* res : {&s, &r}) {
                        for (uint32_t q = 0; q < kNumQueues; q++) {
                            for (uint32_t c = 0; c < kTxqCounters; c++) {
                                csv << "," << delta(*res, q, c);
                            }
                        }
                    }
                    csv << "\n";
                }
                failed |= s[R_STATUS] != kDone || r[R_STATUS] != kDone || r[R_MERGE_ERRORS] != 0;

                if (payload == payloads.front() && timeout == timeouts.front()) {
                    for (const auto& [side, res] : {std::pair{"sender", &s}, std::pair{"receiver", &r}}) {
                        for (uint32_t q = 0; q < kNumQueues; q++) {
                            const uint32_t* c = &(*res)[R_TXQ_CONFIG + q * kTxqConfig];
                            log_info(
                                LogTest,
                                "{} TXQ{} config: ctrl {:#x}, max pkt {} B, remote seq timeout {}, local seq update "
                                "timeout {}, accept ahead {}",
                                side,
                                q,
                                c[0],
                                c[1],
                                c[2],
                                c[3],
                                c[4]);
                        }
                    }
                }
            }
        }
    }
    EXPECT_FALSE(failed);
}
