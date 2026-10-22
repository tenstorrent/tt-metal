// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define TT_METAL_GLOBAL_SEMAPHORE_H_

#if defined(COMPILE_FOR_TRISC)
#error "GlobalSemaphore is only available to data-movement kernels"
#endif
#if defined(ARCH_QUASAR)
#error "GlobalSemaphore is not supported on Quasar yet"
#endif

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore_binding_token.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/semaphore_dm_impl.h"
#include "internal/risc_attribs.h"

#if defined(GLOBAL_SEMAPHORE_WITH_FABRIC)
// Off-chip targets for the fabric overloads below. The fabric topology the kernel is built for picks
// the addressing: hop counts on a 1D line or ring, device and mesh ids on a 2D mesh.
#if defined(FABRIC_2D)
struct ToChip {
    uint8_t dst_dev_id;
    uint16_t dst_mesh_id;
};
struct ToChips {
    uint8_t dst_dev_id;
    uint16_t dst_mesh_id;
    uint8_t e;
    uint8_t w;
    uint8_t n;
    uint8_t s;
};
#else
struct ToChip {
    uint8_t num_hops;
};
struct ToChips {
    uint8_t start_distance;
    uint8_t range;
};
#endif

namespace global_semaphore_detail {
template <typename T>
inline constexpr bool is_fabric_sender_v = tt::tt_fabric::common::experimental::is_edm_sender_v<T> ||
                                           tt::tt_fabric::common::experimental::is_mux_sender_v<T> ||
                                           tt::tt_fabric::common::experimental::is_fabric_mux_v2_sender_v<T>;

template <typename T>
inline constexpr bool is_chip_target_v = std::is_same_v<T, ToChip> || std::is_same_v<T, ToChips>;

#if defined(FABRIC_2D)
namespace fabric_api = tt::tt_fabric::mesh::experimental;
#else
namespace fabric_api = tt::tt_fabric::linear::experimental;
#endif
}  // namespace global_semaphore_detail
#endif

/**
 * @brief A semaphore that lives across programs, and across chips.
 *
 * Construct from a GlobalSemaphoreBindingToken: `GlobalSemaphore s(token);`. The constructor reads the
 * semaphore's address from the common runtime arg word the token names. The semaphore has the same address on
 * every core and device it is allocated on, so the class fills it in for NoC and fabric targets alike.
 *
 * up() and down() go through the NoC-atomic path, so a word incremented by other cores, or other chips, is
 * never updated with a plain read-modify-write. value() and wait() are plain loads and set() is a plain store;
 * none of them is atomic with respect to in-flight increments.
 *
 * Rules:
 *  - down() is single-consumer: the consumer runs on the core that owns the semaphore, and remote cores only up().
 *  - A semaphore's protocol uses either down() or wait() + set(0), never both: set() is a plain store and can
 *    overwrite increments still in flight.
 *
 * The off-chip (fabric) overloads are available when the kernel includes api/dataflow/global_semaphore_fabric.h.
 */
class GlobalSemaphore {
public:
    explicit FORCE_INLINE GlobalSemaphore(const GlobalSemaphoreBindingToken& token) :
        l1_addr_(get_common_arg_val<uint32_t>(token.crta_offset_)) {}

    // ---- This core's copy ----

    // Fresh read. For polling loops and stop flags.
    FORCE_INLINE uint32_t value() const { return semaphore_detail::load(l1_addr_, SCOPE); }

    // Block until value == v.
    FORCE_INLINE void wait(uint32_t v) const { semaphore_detail::wait<CORE_TYPE>(l1_addr_, SCOPE, v); }

    // Block until value >= v.
    FORCE_INLINE void wait_min(uint32_t v) const { semaphore_detail::wait_min<CORE_TYPE>(l1_addr_, SCOPE, v); }

    // Plain store: reset to 0, sentinels, run counters.
    FORCE_INLINE void set(uint32_t v) { semaphore_detail::set<CORE_TYPE>(l1_addr_, SCOPE, v); }

    // Atomic increment.
    FORCE_INLINE void up(uint32_t v = 1) { semaphore_detail::up<CORE_TYPE>(l1_addr_, SCOPE, v); }

    // Single consumer: wait until value >= v, then atomic subtract.
    FORCE_INLINE void down(uint32_t v = 1) { semaphore_detail::down<CORE_TYPE>(l1_addr_, SCOPE, v); }

    // ---- Another core on the same chip (NoC) ----

    template <bool posted = false>
    FORCE_INLINE void up(
        const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t v = 1, uint8_t vc = NOC_UNICAST_WRITE_VC) {
        const uint64_t dst = ::get_noc_addr(noc_x, noc_y, l1_addr_, noc.get_noc_id());
        noc_semaphore_inc<posted>(dst, v, noc.get_noc_id(), vc);
    }

    // ---- A rectangle of cores on the same chip (NoC multicast) ----

    // Copies this core's current value to every core in the rectangle (set() it first).
    // NocOptions::MCAST_INCL_SRC includes the sender.
    template <NocOptions opts = NocOptions::DEFAULT>
    FORCE_INLINE void set_multicast(
        const Noc& noc,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        uint32_t num_dests,
        bool linked = false) {
        semaphore_detail::set_multicast<opts>(l1_addr_, noc, x_start, y_start, x_end, y_end, num_dests, linked);
    }

    FORCE_INLINE void inc_multicast(
        const Noc& noc,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        uint32_t v,
        uint32_t num_dests) {
        semaphore_detail::inc_multicast(l1_addr_, noc, x_start, y_start, x_end, y_end, v, num_dests);
    }

#if defined(GLOBAL_SEMAPHORE_WITH_FABRIC)
    // ---- Another chip (fabric) ----
    // Same names as the NoC calls above; the parameters pick on-chip vs off-chip. These overloads only accept
    // fabric sender types (or a connection manager), so a NoC call can never resolve to them by mistake.
    // ChipTarget picks unicast (ToChip, one chip) or multicast (ToChips, many chips); on every target chip the
    // same core (noc_x, noc_y) is incremented. noc: which NoC the destination core uses.

    template <
        typename FabricSender,
        typename ChipTarget,
        typename = std::enable_if_t<
            global_semaphore_detail::is_fabric_sender_v<FabricSender> &&
            global_semaphore_detail::is_chip_target_v<ChipTarget>>>
    FORCE_INLINE void up(
        FabricSender& sender,
        volatile PACKET_HEADER_TYPE* hdr,
        ChipTarget chip,
        uint32_t noc_x,
        uint32_t noc_y,
        uint32_t v = 1,
        bool flush = true,
        uint8_t noc = 0) {
        const tt::tt_fabric::NocUnicastAtomicIncCommandHeader cmd{fabric_noc_addr(noc_x, noc_y, noc), v, flush};
#if defined(FABRIC_2D)
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc(
                &sender, hdr, chip.dst_dev_id, chip.dst_mesh_id, cmd);
        } else {
            tt::tt_fabric::mesh::experimental::fabric_multicast_noc_unicast_atomic_inc(
                &sender, hdr, chip.dst_dev_id, chip.dst_mesh_id, mcast_range(chip), cmd);
        }
#else
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
                &sender, hdr, cmd, chip.num_hops);
        } else {
            tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
                &sender, hdr, cmd, chip.start_distance, chip.range);
        }
#endif
    }

#if defined(FABRIC_2D)
    // Sends on every connection in a route; each connection already knows its destination chip.
    FORCE_INLINE void up(
        tt::tt_fabric::RoutingPlaneConnectionManager& mgr,
        uint8_t route_id,
        uint32_t noc_x,
        uint32_t noc_y,
        uint32_t v = 1,
        bool flush = true,
        uint8_t noc = 0) {
        tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc(
            mgr,
            route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_noc_addr(noc_x, noc_y, noc), v, flush});
    }
#else
    // Sends on every connection in a route; num_hops[i] is the hop count for connection i.
    FORCE_INLINE void up(
        tt::tt_fabric::RoutingPlaneConnectionManager& mgr,
        uint8_t route_id,
        uint8_t* num_hops,
        uint32_t noc_x,
        uint32_t noc_y,
        uint32_t v = 1,
        bool flush = true,
        uint8_t noc = 0) {
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            mgr,
            route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_noc_addr(noc_x, noc_y, noc), v, flush},
            num_hops);
    }
#endif

    // One packet: write size_bytes from src (a local L1 object, e.g. a CoreLocalMem or Scratchpad) to dst on the
    // target chip, then increment this semaphore on core (noc_x, noc_y) there; the data lands first. src and dst
    // take the same endpoint types and args as Noc::async_write. noc: which NoC the destination core uses.
    template <
        typename FabricSender,
        typename ChipTarget,
        typename Src,
        typename Dst,
        typename = std::enable_if_t<
            global_semaphore_detail::is_fabric_sender_v<FabricSender> &&
            global_semaphore_detail::is_chip_target_v<ChipTarget>>>
    FORCE_INLINE void write_and_up(
        FabricSender& sender,
        volatile PACKET_HEADER_TYPE* hdr,
        ChipTarget chip,
        const Noc& noc,
        const Src& src,
        const typename noc_traits_t<Src>::src_args_type& src_args,
        const Dst& dst,
        const typename noc_traits_t<Dst>::dst_args_type& dst_args,
        uint32_t size_bytes,
        uint32_t noc_x,
        uint32_t noc_y,
        uint32_t v = 1,
        bool flush = true) {
        const auto src_l1_addr =
            static_cast<uint32_t>(noc_traits_t<Src>::template src_addr<Noc::AddressType::LOCAL_L1>(src, noc, src_args));
        const uint64_t payload_noc_addr =
            noc_traits_t<Dst>::template dst_addr<Noc::AddressType::NOC>(dst, noc, dst_args);
        const tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader cmd{
            payload_noc_addr, fabric_noc_addr(noc_x, noc_y, noc.get_noc_id()), v, flush};
#if defined(FABRIC_2D)
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &sender, hdr, chip.dst_dev_id, chip.dst_mesh_id, src_l1_addr, size_bytes, cmd);
        } else {
            tt::tt_fabric::mesh::experimental::fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &sender, hdr, chip.dst_dev_id, chip.dst_mesh_id, mcast_range(chip), src_l1_addr, size_bytes, cmd);
        }
#else
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &sender, hdr, src_l1_addr, size_bytes, cmd, chip.num_hops);
        } else {
            tt::tt_fabric::linear::experimental::fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &sender, hdr, src_l1_addr, size_bytes, cmd, chip.start_distance, chip.range);
        }
#endif
    }

    // Fast path for hot loops that signal many times: set up the packet header once, then each send only
    // submits it. Same as the fabric *_set_state / *_with_state calls, but the class fills in the address.
    template <typename ChipTarget, typename = std::enable_if_t<global_semaphore_detail::is_chip_target_v<ChipTarget>>>
    FORCE_INLINE void up_set_state(
        volatile PACKET_HEADER_TYPE* hdr,
        ChipTarget chip,
        uint32_t noc_x,
        uint32_t noc_y,
        uint32_t v = 1,
        bool flush = true,
        uint8_t noc = 0) {
        constexpr auto mask = UnicastAtomicIncUpdateMask::All;
        const tt::tt_fabric::NocUnicastAtomicIncCommandHeader cmd{fabric_noc_addr(noc_x, noc_y, noc), v, flush};
#if defined(FABRIC_2D)
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc_set_state<mask>(
                hdr, chip.dst_dev_id, chip.dst_mesh_id, cmd);
        } else {
            tt::tt_fabric::mesh::experimental::fabric_multicast_noc_unicast_atomic_inc_set_state<mask>(
                hdr, chip.dst_dev_id, chip.dst_mesh_id, mcast_range(chip), cmd);
        }
#else
        if constexpr (std::is_same_v<ChipTarget, ToChip>) {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc_set_state<mask>(
                hdr, chip.num_hops, cmd);
        } else {
            tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc_set_state<mask>(
                hdr, chip.start_distance, chip.range, cmd);
        }
#endif
    }

    // Submits a header prepared by up_set_state().
    template <
        typename FabricSender,
        typename = std::enable_if_t<global_semaphore_detail::is_fabric_sender_v<FabricSender>>>
    FORCE_INLINE void up_with_state(FabricSender& sender, volatile PACKET_HEADER_TYPE* hdr) {
        global_semaphore_detail::fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::None>(&sender, hdr);
    }

    // Same, but changes the target core on each send.
    template <
        typename FabricSender,
        typename = std::enable_if_t<global_semaphore_detail::is_fabric_sender_v<FabricSender>>>
    FORCE_INLINE void up_with_state(
        FabricSender& sender, volatile PACKET_HEADER_TYPE* hdr, uint32_t noc_x, uint32_t noc_y, uint8_t noc = 0) {
        global_semaphore_detail::fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr>(
            &sender, hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_noc_addr(noc_x, noc_y, noc), 0});
    }
#endif

private:
    static constexpr ProgrammableCoreType CORE_TYPE = ProgrammableCoreType::TENSIX;
    static constexpr SemScope SCOPE = SemScope::EXTERNAL;

#if defined(GLOBAL_SEMAPHORE_WITH_FABRIC)
#if defined(FABRIC_2D)
    static FORCE_INLINE tt::tt_fabric::mesh::experimental::MeshMcastRange mcast_range(const ToChips& chip) {
        return tt::tt_fabric::mesh::experimental::MeshMcastRange{chip.e, chip.w, chip.n, chip.s};
    }
#endif
    FORCE_INLINE uint64_t fabric_noc_addr(uint32_t noc_x, uint32_t noc_y, uint8_t noc) const {
        return safe_get_noc_addr(noc_x, noc_y, l1_addr_, noc);
    }
#endif
    uint32_t l1_addr_;
};
