// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/packet_injection/tensix_producer.hpp
//
// Tensix (BRISC / NCRISC) binding of the packet-injection Producer: its
// transport, RISC identity and core-sync transport.

#ifndef X280_KERNELS_PACKET_INJECTION_TENSIX_PRODUCER_HPP
#define X280_KERNELS_PACKET_INJECTION_TENSIX_PRODUCER_HPP

#include <stdint.h>
#include <stddef.h>

#include "packet_injection_abi.h"
#include "packet_injection_slot.hpp"
#include "metal_producer_transport.hpp"
#include "producer.hpp"
#include "../shared/core_sync_transport_metal.hpp"
#include "../shared/tensix_live_timer.h"

namespace packet_injection {

/* Production Tensix hardware transport */
struct DeviceTensixTransport : MetalProducerTransport {
    // Which RISC of its tile this image is. A host build has no processor identity
    // and stands in as the first.
#if defined(__riscv)
#if defined(COMPILE_FOR_BRISC) || defined(RISCV_0)
    static constexpr uint32_t kRisc = 0u;
#elif defined(COMPILE_FOR_NCRISC) || defined(RISCV_1)
    static constexpr uint32_t kRisc = 1u;
#else
#error "A Tensix producer image must be compiled for BRISC or NCRISC"
#endif
#else
    static constexpr uint32_t kRisc = 0u;
#endif

    // BRISC drives NOC 0 and NCRISC NOC 1, which is the DataMovementConfig the host
    // creates each producer kernel with rather than anything hardware imposes.
    static constexpr uint32_t kNoc = kRisc;

    // A worker's tile span must fit the core's own L1. MEM_L1_SIZE is the
    // authority and only a device build sees it; a host build mirrors the value so
    // that it accepts and rejects exactly what silicon does, and the assert fails
    // the device build if the two ever diverge.
#if defined(MEM_L1_SIZE)
    static constexpr uint32_t kL1SpanLimit = MEM_L1_SIZE;
    static_assert(kL1SpanLimit == 1536u * 1024u, "the mirrored host L1 bound below no longer matches MEM_L1_SIZE");
#else
    static constexpr uint32_t kL1SpanLimit = 1536u * 1024u;
#endif

    /** The failure actor for this image's RISC. */
    static constexpr pi_actor_t kActor = kRisc == 0u ? PI_ACTOR_BRISC : PI_ACTOR_NCRISC;

    /** Seeds `header` as a zero-payload unicast to `dst_chip`/`dst_mesh`. */
    static inline void prepare_header(
        volatile fs_hybrid_mesh_packet_header_t* header, uint16_t dst_chip, uint16_t dst_mesh) {
        fs_seed_unicast_header(header, dst_chip, dst_mesh);
    }

    static inline int read_timer(uint32_t max_attempts, uint64_t* ticks) {
        return tensix_live_timer_read(TENSIX_TIMER_LOCAL_LOW_ADDR, max_attempts, ticks);
    }
};

using PacketWorkerTransport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX, core_sync::ArrivalKind::Word>;

#if X280_CORE_SYNC_METAL_TRANSPORT
inline pi_error_t run_producer(const pi_producer_args_t& args) {
    Producer<PACKET_INJECTION_MODE, PACKET_INJECTION_INCLUDE_CREATION, DeviceTensixTransport, PacketWorkerTransport>
        producer(args);
    return producer.run();
}
#endif

}  // namespace packet_injection

#endif  // X280_KERNELS_PACKET_INJECTION_TENSIX_PRODUCER_HPP
