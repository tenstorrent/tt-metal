// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/packet_injection/erisc_producer.hpp
//
// Idle-ERISC binding of the packet-injection Producer: its transport and
// core-sync transport. The host seeds every staged header before launch, so the
// producer re-sends that record and constructs nothing.

#ifndef X280_KERNELS_PACKET_INJECTION_ERISC_PRODUCER_HPP
#define X280_KERNELS_PACKET_INJECTION_ERISC_PRODUCER_HPP

#include <stdint.h>
#include <stddef.h>

#include "packet_injection_abi.h"
#include "packet_injection_slot.hpp"
#include "metal_producer_transport.hpp"
#include "producer.hpp"
#include "../shared/core_sync_transport_metal.hpp"

#if defined(__riscv)
#include "internal/ethernet/tt_eth_api.h"
#endif

namespace packet_injection {

// The ERISC publishes ready completed-packet headers: throughput is the one
// measurement it has, and creation is outside the measured operation.
static_assert(
    PACKET_INJECTION_MODE == (uint32_t)PI_MODE_THROUGHPUT && PACKET_INJECTION_INCLUDE_CREATION == 0,
    "an ERISC producer image must be PI_MODE_THROUGHPUT with creation excluded");

/* Production idle-ERISC hardware transport */
struct DeviceEriscTransport : MetalProducerTransport {
    /** One producer per Ethernet tile, so every worker is its tile's first. */
    static constexpr uint32_t kRisc = 0u;

    /** The NoC the host creates the kernel on (EthernetConfig::noc). */
    static constexpr uint32_t kNoc = 0u;
#if defined(NOC_INDEX)
    static_assert(NOC_INDEX == kNoc, "the ERISC producer kernel must be created on NOC 0");
#endif

    // A worker's tile span must end inside the idle-ERISC unreserved window,
    // whose end Metal's HAL takes from MEM_ERISC_MAX_SIZE. Only a device build
    // sees that macro; a host build mirrors the value so that it accepts and
    // rejects exactly what silicon does, and the assert fails the device build
    // if the two ever diverge.
#if defined(MEM_ERISC_MAX_SIZE)
    static constexpr uint32_t kL1SpanLimit = MEM_ERISC_MAX_SIZE;
    static_assert(kL1SpanLimit == 455264u, "the mirrored host L1 bound below no longer matches MEM_ERISC_MAX_SIZE");
#else
    static constexpr uint32_t kL1SpanLimit = 455264u;
#endif

    /** The failure actor for this image. */
    static constexpr pi_actor_t kActor = PI_ACTOR_ERISC;

    /** Does nothing: the host seeded the staged header before launch. */
    static inline void prepare_header(
        volatile fs_hybrid_mesh_packet_header_t* header, uint16_t dst_chip, uint16_t dst_mesh) {
        (void)header;
        (void)dst_chip;
        (void)dst_mesh;
    }

    /**
     * Reads the Ethernet tile's 64-bit wall clock into `*ticks` and returns 1;
     * returns zero on a host build. The read has no retry path, so
     * `max_attempts` is unused.
     */
    static inline int read_timer(uint32_t max_attempts, uint64_t* ticks) {
        (void)max_attempts;
#if defined(__riscv)
        *ticks = eth_read_wall_clock();
        return 1;
#else
        (void)ticks;
        return 0;
#endif
    }
};

using EriscWorkerTransport = core_sync::MetalTransport<ProgrammableCoreType::IDLE_ETH, core_sync::ArrivalKind::Word>;

#if X280_CORE_SYNC_METAL_TRANSPORT
inline pi_error_t run_producer(const pi_producer_args_t& args) {
    Producer<PACKET_INJECTION_MODE, PACKET_INJECTION_INCLUDE_CREATION, DeviceEriscTransport, EriscWorkerTransport>
        producer(args);
    return producer.run();
}
#endif

}  // namespace packet_injection

#endif  // X280_KERNELS_PACKET_INJECTION_ERISC_PRODUCER_HPP
