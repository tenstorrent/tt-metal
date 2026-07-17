// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

// Generic real-time profiler packet pipeline. See tools/x280_bm/PROFILER_PACKET_PIPELINE.md.
//
// Two representations:
//   * WIRE packet  — POD, packed, <=64B, PacketHeader{type} at offset 0. What the X280
//                    presenter emits and what the host decodes off the D2H socket.
//   * ENRICHED packet — host-built, unconstrained, callback-facing. Fully resolved
//                    (names deciphered, coordinates translated) so subscribers never see
//                    ciphered data and no two subscribers redo the same translation.
//
// The host is decode -> enrich -> dispatch: it reads the wire type, does per-type
// enrichment ONCE, then invokes the callbacks registered for that packet type.

namespace tt::tt_metal::experimental {

// Discriminator carried at offset 0 of every wire packet and used to route enriched
// packets to their subscribers. One value per distinct packet kind (see Option B in the
// design doc): ZONE_START/ZONE_END share the WorkerZone kind (start/end is an inner field).
enum class ProfilerPacketType : uint16_t {
    WorkerZone = 0,
    // Future: Event, Payload, Sync, ProgramZone, ...
};

// ---------------------------------------------------------------------------------------
// Wire packets (POD, packed, <=64B). Layout is shared by convention with the X280 presenter
// (tools/x280_bm/src/profzone.c), which writes these fields as raw 32-bit words; keep the two
// in sync. Little-endian on both ends, so a 32-bit store of `type` populates the low 16 bits.
// ---------------------------------------------------------------------------------------
#pragma pack(push, 1)
struct WirePacketHeader {
    uint16_t type;  // ProfilerPacketType
    uint16_t reserved;
};

struct WorkerZoneWire {
    WirePacketHeader header;  // offset 0
    uint32_t core_x;          // 4  — virtual coord (what the X280 NoC addresses)
    uint32_t core_y;          // 8  — virtual coord
    uint32_t risc;            // 12 — 0=BRISC 1=NCRISC 2/3/4=TRISC_0/1/2
    uint32_t timer_id;        // 16 — packet-type bits ((id>>16)&0x7) + 16-bit name hash
    uint32_t time_hi;         // 20 — timestamp high (12 valid bits)
    uint32_t time_lo;         // 24 — timestamp low
};
#pragma pack(pop)
static_assert(sizeof(WorkerZoneWire) == 28, "WorkerZoneWire must match the X280 presenter layout");

// Tier-1 compact on-wire record (X280 -> host). 16 B, 4 per 64B page. The X280 collect hart ships the
// raw 2-word marker (w0,w1) VERBATIM plus a packed identity word, doing only 3 stores/marker and no
// field extraction; the host expands each record into a WorkerZoneWire below. This is the transported
// format; WorkerZoneWire remains the in-memory struct the ring + consumer use. Keep in sync with
// tools/x280_bm/src/profzone.c (w_rec / PACK_IDENT). Little-endian both ends.
//   ident = core_x[0:9] | core_y[10:19] | risc[20:23]
//   w0    = raw marker word0 = 0x80000000 | (timer_id<<12) | time_hi(12b)   (bit31 set => valid)
//   w1    = raw marker word1 = time_lo
#pragma pack(push, 1)
struct WorkerZoneWireCompact {
    uint32_t ident;  // 0  — packed structural identity (from the mirror index)
    uint32_t w0;     // 4  — raw marker word0 (bit31 = valid sentinel)
    uint32_t w1;     // 8  — raw marker word1 (time_lo)
    uint32_t pad;    // 12 — unwritten by the FW; ignored by the host
};
#pragma pack(pop)
static_assert(sizeof(WorkerZoneWireCompact) == 16, "compact record must be 16 B (4 per 64B page)");

// ---------------------------------------------------------------------------------------
// Enriched packets (callback-facing). Not size- or POD-constrained; never transported.
// Each exposes a static kType so a single Register(callback) can auto-file it.
// ---------------------------------------------------------------------------------------
struct WorkerZonePacket {
    static constexpr ProfilerPacketType kType = ProfilerPacketType::WorkerZone;

    uint32_t chip_id;
    uint32_t core_virtual_x;  // as relayed by the X280
    uint32_t core_virtual_y;
    uint32_t core_noc0_x;  // translated — matches the standard DeviceProfiler / DRAM view
    uint32_t core_noc0_y;
    uint32_t risc;          // 0=BRISC 1=NCRISC 2/3/4=TRISC_0/1/2
    uint32_t timer_id;      // 16-bit zone-name hash
    std::string_view name;  // deciphered zone name; stable for the profiler session
    uint64_t timestamp;     // raw device ticks
    bool is_start;          // true = ZONE_START, false = ZONE_END
};

// ---------------------------------------------------------------------------------------
// Registration + dispatch.
// ---------------------------------------------------------------------------------------
using ProfilerPacketCallbackHandle = uint64_t;
// Type-erased callback stored in the registry; receives a pointer to the enriched packet.
using RawProfilerPacketCallback = std::function<void(const void* enriched_packet)>;

// Internal: register/dispatch by explicit type with a type-erased callback. Prefer the
// typed RegisterProfilerPacketCallback below.
ProfilerPacketCallbackHandle RegisterProfilerPacketCallbackRaw(
    ProfilerPacketType type, RawProfilerPacketCallback callback);
void UnregisterProfilerPacketCallback(ProfilerPacketCallbackHandle handle);
// Invoke every callback registered for `type`, passing the enriched packet by address.
// Called from the real-time profiler receiver thread.
void InvokeProfilerPacketCallbacks(ProfilerPacketType type, const void* enriched_packet);

namespace detail {
// Extract the (single) argument type of a callable with one `const T&` parameter.
template <typename T>
struct callback_arg;
template <typename R, typename C, typename A>
struct callback_arg<R (C::*)(const A&) const> {
    using type = A;
};
template <typename R, typename C, typename A>
struct callback_arg<R (C::*)(const A&)> {
    using type = A;
};
template <typename F>
using callback_arg_t = typename callback_arg<decltype(&std::remove_reference_t<F>::operator())>::type;
}  // namespace detail

// Register a subscriber for a single packet kind. The packet type is deduced from the
// callback's argument (auto-detect), or given explicitly as RegisterProfilerPacketCallback<T>(cb)
// for cases deduction can't handle (e.g. generic lambdas). Type-safe: the callback receives the
// enriched packet by const-ref. Returns a handle for UnregisterProfilerPacketCallback.
template <typename T = void, typename F>
ProfilerPacketCallbackHandle RegisterProfilerPacketCallback(F&& callback) {
    using Packet = std::conditional_t<std::is_same_v<T, void>, detail::callback_arg_t<F>, T>;
    auto fn = std::function<void(const Packet&)>(std::forward<F>(callback));
    return RegisterProfilerPacketCallbackRaw(
        Packet::kType, [fn = std::move(fn)](const void* p) { fn(*static_cast<const Packet*>(p)); });
}

}  // namespace tt::tt_metal::experimental
