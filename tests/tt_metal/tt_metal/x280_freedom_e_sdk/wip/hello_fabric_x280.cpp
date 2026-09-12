// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// WIP: freedom-metal + tt-fabric device headers (no NOC). Does not build —
// see README.md. Exercises packet headers, 1D routing encode, ChannelBufferPointer.

#include <cstdint>
#include <cstddef>
#include <cstdio>

// --- freedom-e-sdk / freedom-metal ------------------------------------------
#include <metal/cpu.h>

// --- tt-metal fabric device code (unmodified, compiled from the tt-metal tree)
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"

using namespace tt::tt_fabric;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what, unsigned long long got, unsigned long long want) {
    ++g_checks;
    if (ok) {
        printf("  [ ok ] %-46s = %llu\n", what, got);
    } else {
        ++g_failures;
        printf("  [FAIL] %-46s = %llu (expected %llu)\n", what, got, want);
    }
}

void check_eq(const char* what, unsigned long long got, unsigned long long want) {
    check(got == want, what, got, want);
}

void check_hex(const char* what, uint32_t got, uint32_t want) {
    ++g_checks;
    if (got == want) {
        printf("  [ ok ] %-46s = 0x%08lx\n", what, (unsigned long)got);
    } else {
        ++g_failures;
        printf("  [FAIL] %-46s = 0x%08lx (expected 0x%08lx)\n", what, (unsigned long)got, (unsigned long)want);
    }
}

// -- 1. freedom-metal -------------------------------------------------------
void report_platform() {
    printf("[1] freedom-metal / platform\n");

    const int hartid = metal_cpu_get_current_hartid();
    struct metal_cpu* cpu = metal_cpu_get(hartid);

    printf("  hart id                                        = %d\n", hartid);
    printf(
        "  sizeof(void*)                                  = %u bytes (XLEN=%u)\n",
        (unsigned)sizeof(void*),
        (unsigned)(sizeof(void*) * 8));

    if (cpu != nullptr) {
        const unsigned long long timebase = metal_cpu_get_timebase(cpu);
        const unsigned long long t0 = metal_cpu_get_timer(cpu);
        printf("  metal_cpu_get_timebase()                       = %llu Hz\n", timebase);
        printf("  metal_cpu_get_timer()                          = %llu ticks\n", t0);
    } else {
        printf("  metal_cpu_get() returned NULL (no CPU node in BSP)\n");
    }
    printf("\n");
}

// -- 2. Packet header layout (sizes are wire-load-bearing) ------------------
void check_packet_header_layout() {
    printf("[2] fabric packet header layout (tt::tt_fabric)\n");

    check_eq("sizeof(RoutingFields)", sizeof(RoutingFields), 1);
    check_eq("sizeof(MulticastRoutingCommandHeader)", sizeof(MulticastRoutingCommandHeader), 2);
    check_eq("sizeof(NocCommandFields)", sizeof(NocCommandFields), 40);

    // LowLatencyPacketHeaderT<0> targets 48B (16 hops);
    // LowLatencyPacketHeaderT<1> targets 64B (32 hops).
    check_eq("sizeof(LowLatencyPacketHeaderT<0>)", sizeof(LowLatencyPacketHeaderT<0>), 48);
    check_eq("sizeof(LowLatencyPacketHeaderT<1>)", sizeof(LowLatencyPacketHeaderT<1>), 64);

    // payload_size_bytes must stay 4B-aligned: PackedPayloadAndSendType::load()
    // does a single 4B load at that offset instead of two uncached L1 reads.
    check_eq(
        "offsetof(payload_size_bytes) % 4",
        offsetof(PacketHeaderBase<LowLatencyPacketHeaderT<1>>, payload_size_bytes) % 4,
        0);
    printf("\n");
}

// -- 3. 1D routing encoder (patterns from fabric_common.h) ------------------
void check_routing_encoder() {
    printf("[3] fabric 1D routing encoder (routing_encoding)\n");

    // Unicast, 3 hops, 1 word:
    //   hop0 FORWARD_ONLY 0b10, hop1 FORWARD_ONLY 0b10, hop2 WRITE_ONLY 0b01
    //   => 0b01'10'10 == 0x1A
    uint32_t uni[1] = {0};
    routing_encoding::encode_1d_unicast(/*num_hops=*/3, uni, /*num_words=*/1);
    check_hex("encode_1d_unicast(hops=3)", uni[0], 0x1A);

    // Self-route encodes to zero.
    uint32_t self_route[1] = {0xdeadbeef};
    routing_encoding::encode_1d_unicast(/*num_hops=*/0, self_route, /*num_words=*/1);
    check_hex("encode_1d_unicast(hops=0) [self]", self_route[0], 0x0);

    // Multicast, start_hop=3, range_hops=2:
    //   hop0 FWD 0b10, hop1 FWD 0b10, hop2 WRITE_AND_FWD 0b11, hop3 WRITE 0b01
    //   => 0b01'11'10'10 == 0x7A
    uint32_t mcast[1] = {0};
    routing_encoding::encode_1d_multicast(/*start_hop=*/3, /*range_hops=*/2, mcast, /*num_words=*/1);
    check_hex("encode_1d_multicast(start=3, range=2)", mcast[0], 0x7A);

    // Now drive the same encoder through the packet header's own helper, which
    // is what a fabric sender kernel actually calls.
    LowLatencyPacketHeader hdr{};
    hdr.to_chip_unicast(3);
    check_hex("hdr.to_chip_unicast(3).routing_fields.value", hdr.routing_fields.value, 0x1A);

    LowLatencyPacketHeader mcast_hdr{};
    mcast_hdr.to_chip_multicast(MulticastRoutingCommandHeader{/*start_distance_in_hops=*/3, /*range_hops=*/2});
    check_hex("hdr.to_chip_multicast(3,2).routing_fields.value", mcast_hdr.routing_fields.value, 0x7A);
    printf("\n");
}

// -- 4. ChannelBufferPointer credit/wrap arithmetic -------------------------
void check_flow_control() {
    printf("[4] fabric flow control (ChannelBufferPointer)\n");

    constexpr uint8_t kNumBuffers = 8;
    using Ptr = ChannelBufferPointer<kNumBuffers>;

    check_eq("Ptr::ptr_wrap_size", Ptr::ptr_wrap_size, 2 * kNumBuffers);
    check_eq("Ptr::is_size_pow2", Ptr::is_size_pow2, 1);

    Ptr wrptr;
    Ptr rdptr;
    check_eq("fresh channel: rdptr caught up to wrptr", rdptr.is_caught_up_to(wrptr), 1);

    // Fill the channel completely.
    for (uint8_t i = 0; i < kNumBuffers; ++i) {
        wrptr.increment();
    }
    check_eq("after 8 writes: raw wrptr", wrptr.get_ptr().get(), 8);
    check_eq("after 8 writes: buffer index", wrptr.get_buffer_index().get(), 0);
    check_eq("after 8 writes: reader still behind", rdptr.is_caught_up_to(wrptr), 0);
    check_eq("after 8 writes: distance_behind", rdptr.distance_behind(wrptr), 8);

    // Drain it.
    rdptr.increment_n(kNumBuffers);
    check_eq("after 8 reads: reader caught up", rdptr.is_caught_up_to(wrptr), 1);

    // Full lap: 2*NUM_BUFFERS increments must return to the raw origin.
    Ptr lap;
    for (uint8_t i = 0; i < Ptr::ptr_wrap_size; ++i) {
        lap.increment();
    }
    check_eq("full lap (16 increments): raw ptr back to 0", lap.get_ptr().get(), 0);
    printf("\n");
}

}  // namespace

int main() {
    printf("\n");
    printf("=====================================================================\n");
    printf(" hello_fabric_x280 -- tt-metal fabric code on freedom-e-sdk\n");
    printf("=====================================================================\n");
    printf("\n");

    report_platform();
    check_packet_header_layout();
    check_routing_encoder();
    check_flow_control();

    printf("---------------------------------------------------------------------\n");
    if (g_failures == 0) {
        printf(" PASS -- %d/%d checks passed.\n", g_checks, g_checks);
        printf(" tt-metal fabric packet headers, routing encoder and flow control\n");
        printf(" all ran on a SiFive RISC-V core under freedom-metal.\n");
    } else {
        printf(" FAIL -- %d of %d checks failed.\n", g_failures, g_checks);
    }
    printf("---------------------------------------------------------------------\n");
    printf("\n");

    return g_failures == 0 ? 0 : 1;
}
