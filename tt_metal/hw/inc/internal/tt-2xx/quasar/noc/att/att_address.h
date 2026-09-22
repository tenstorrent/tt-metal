// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "internal/tt-2xx/quasar/noc/att/att.h"

/**
 * @file
 * @brief The typed ATT resolution layer: how an intent ("worker (x,y)", "this
 * initiator's L1", "logical DRAM bank") becomes the one translated 64-bit
 * operand the NOC V3 transport writes whole into a command-buffer register.
 *
 *   Address::worker(x, y, offset).encode<MAP>()
 *     -> resolve(MAP, address)     : identity -> {window, selector}   (map data only)
 *     -> Window::make_address()    : window bits | selector | offset  (att.h window math)
 *
 * A map is pure data: a MapData aggregate declared by its configuration header
 * alongside the windows and selector tables it points at. One shared resolver
 * interprets every map, so map-specific logic cannot exist - adding a map is
 * adding data. This header names no product configuration; the ATT enablement
 * selects the active MapData at compile time, and the host tests instantiate
 * the resolver once per product map.
 */
namespace noc_att {

using NocAddress = std::uint64_t;

struct NocMulticastAddress {
    NocAddress start_address;       // operand of the rectangle's start tile
    std::uint32_t extent_xy;        // (height << 6) | width, for the operand + extent register form
    std::uint32_t rectangle_count;  // tiles in the rectangle; 0 = invalid rectangle
    NocAddress end_address;         // operand of the rectangle's end tile
    std::uint32_t start_node_xy;    // start tile's endpoint word ((y << 6) | x), for the coordinate register form
    std::uint32_t end_node_xy;      // end tile's endpoint word
};

/// @brief The roles a window can play in a map. A map assigns a Window to each
/// role (several roles may share one Window, as on maps with a single remote
/// window). Add values before Count; Count doubles as the array extent and
/// Invalid marks "no role" in ResolvedTile.
enum class WindowClass : std::uint8_t {
    LoopbackScratch = 0,  // optional selector-free, pass-through self aperture
    Worker,
    Dram,
    FullTile,
    Local,

    // Add values before this line.
    Count,
    Invalid = Count,
};

inline constexpr std::size_t WINDOW_CLASS_COUNT = static_cast<std::size_t>(WindowClass::Count);

/// @brief A resolved ATT identity: which window and selector reach a tile.
/// Members sorted by size so padding stays minimal.
struct ResolvedTile {
    std::uint32_t selector;
    std::uint32_t noc_x;
    std::uint32_t noc_y;
    bool valid;
    WindowClass window;
};

static_assert(sizeof(ResolvedTile) == 16, "keep ResolvedTile at two words");

inline constexpr ResolvedTile INVALID_TILE{0, 0, 0, false, WindowClass::Invalid};

/// @brief A constexpr view of one transcribed table: pointer plus element
/// count, deduced from the array so lengths are never spelled by hand. The
/// minimal C++17 stand-in for std::span (device JIT builds are C++17). Empty
/// tables are expressed by size, never by pointer checks (nullptr is a valid
/// address on this hardware).
template <typename T>
struct Table {
    const T* data = nullptr;
    std::uint32_t count = 0;

    constexpr Table() = default;
    template <std::size_t N>
    constexpr Table(const T (&array)[N]) : data(array), count(N) {}

    constexpr const T& operator[](std::uint32_t index) const { return data[index]; }
    constexpr std::uint32_t size() const { return count; }
    constexpr bool empty() const { return count == 0; }
};

/// @brief The window a map assigns to roles it does not support: its compare
/// value cannot match any real device address (the same convention the
/// bring-up replay uses to park unused mask-table slots), so every identity
/// resolved onto it fails validation.
inline constexpr Window NO_WINDOW{~std::uint64_t{0}, 0, 0, 0, 0, false};

constexpr bool is_no_window(const Window& window) { return window.compare == NO_WINDOW.compare; }

/// @brief The declarative description of one product map. Every field is
/// transcribed configuration data; the resolver below is the only code that
/// interprets it.
///
/// Each configuration header declares exactly one inline constexpr instance,
/// consumed only as a `const MapData&` template argument - everything
/// constant-folds, so the instance is never odr-used at runtime and
/// contributes no storage to the device binary.
struct MapData {
    /// Window per role, by value, indexed by WindowClass. Roles the map does
    /// not support hold NO_WINDOW.
    std::array<Window, WINDOW_CLASS_COUNT> windows;

    /// The local (self) identity: this window class at selector 0 reaches the
    /// boot-patched per-initiator endpoint.
    WindowClass local_window_class;

    /// The worker grid in the coordinate frame kernels receive (today: the soc
    /// descriptor's physical functional_workers positions, since Quasar NOC
    /// translation is the identity placeholder - see tenstorrent/tt-umd#2494).
    /// Selector values come from the transcribed table, never from arithmetic.
    std::uint32_t worker_origin_x;
    std::uint32_t worker_origin_y;
    std::uint32_t worker_grid_x;
    std::uint32_t worker_grid_y;
    Table<std::uint8_t> worker_selectors;  // worker_grid_x * worker_grid_y entries, row-major

    /// Offset from host coordinates (the soc descriptor's frame, which is what
    /// kernels are given) to hardware coordinates (the NOC_NODE_ID frame, which
    /// the endpoint words below and my_x/my_y use). Add it to a host coordinate
    /// before looking it up in these tables or comparing it with my_x/my_y.
    /// Same value as UMD's package offset; zero where the two frames coincide.
    std::uint32_t node_id_offset_x;
    std::uint32_t node_id_offset_y;

    /// Endpoint words ((y << 6) | x, hardware frame) by selector, the inverse
    /// of the hardware's table, used to find this core's own selector from its
    /// NOC_NODE_ID coordinate.
    Table<std::uint16_t> worker_endpoint_words;
    Table<std::uint16_t> full_tile_endpoint_words;
    Table<std::uint16_t> dram_endpoint_words;

    /// Logical DRAM bank -> selector. Empty = no binding: every Dram identity
    /// resolves invalid.
    Table<std::uint8_t> dram_selectors;

    /// Dispatch-capable tiles by coordinate. Empty = no binding.
    struct DispatchEntry {
        std::uint16_t x;
        std::uint16_t y;
        std::uint8_t selector;
        WindowClass window;
    };
    Table<DispatchEntry> dispatch_entries;
};

constexpr const Window& map_window(const MapData& map, WindowClass window_class) {
    return map.windows[static_cast<std::size_t>(window_class)];
}

/// @brief Find the role of a matching, supported window. Shared windows can
/// occupy several roles; their address arithmetic and endpoint rows agree.
constexpr WindowClass matching_window_class(const MapData& map, NocAddress address) {
    constexpr WindowClass candidates[] = {
        WindowClass::Worker,
        WindowClass::Dram,
        WindowClass::FullTile,
        WindowClass::Local,
        WindowClass::LoopbackScratch};
    for (WindowClass window_class : candidates) {
        const Window& window = map_window(map, window_class);
        if (!is_no_window(window) && window.matches(address)) {
            return window_class;
        }
    }
    return WindowClass::Invalid;
}

/// @brief Every operand for this initiator's own L1 is
/// local_window_base(map) | local_address: the local window's selector-0 base
/// (which coincides with that window's compare on the current maps). Derived
/// from the map so it cannot drift; the ATT enablement surfaces the same value
/// as the NOC_ATT_LOCAL_WINDOW_BASE macro the V3 header requires at preprocess
/// time, static_asserted against this.
constexpr std::uint64_t local_window_base(const MapData& map) {
    return map_window(map, map.local_window_class).make_address(/*selector*/ 0, /*local*/ 0);
}

/// @brief One typed NoC target: the tile kind, its per-kind identity, and the
/// local byte offset within it. Factories are the only way to build one, and
/// encode() is the single resolution + encoding step: it maps the identity
/// through the map's transcribed tables and checks the complete transfer
/// against the selected window.
class Address {
public:
    /// Identities are stored as 16 bits; values that do not fit clamp to
    /// 0xFFFF, which no map can resolve (endpoint coordinates and bank counts
    /// are far smaller), so an oversized identity is rejected at resolution
    /// instead of silently wrapping onto a valid one.
    static constexpr std::uint16_t narrow_identity(std::uint32_t value) {
        return value > 0xFFFF ? std::uint16_t{0xFFFF} : static_cast<std::uint16_t>(value);
    }

    enum class Kind : std::uint8_t {
        Local,            // this initiator, via the boot-patched self endpoint
        Worker,           // worker coordinate in the kernel-visible frame
        Dram,             // logical DRAM bank
        Dispatch,         // dispatch virtual coordinate
        LoopbackScratch,  // QSR1 boot-owned absolute aperture [0x100000, 0x200000)
    };

    static constexpr Address local(std::uint64_t offset) { return {Kind::Local, 0, 0, 0, offset}; }
    static constexpr Address worker(std::uint32_t x, std::uint32_t y, std::uint64_t offset) {
        return {Kind::Worker, narrow_identity(x), narrow_identity(y), 0, offset};
    }
    static constexpr Address dram(std::uint32_t logical_bank, std::uint64_t offset) {
        return {Kind::Dram, 0, 0, narrow_identity(logical_bank), offset};
    }
    static constexpr Address dispatch(std::uint32_t virtual_x, std::uint32_t virtual_y, std::uint64_t offset) {
        return {Kind::Dispatch, narrow_identity(virtual_x), narrow_identity(virtual_y), 0, offset};
    }
    static constexpr Address loopback_scratch(std::uint64_t absolute_l1_address) {
        return {Kind::LoopbackScratch, 0, 0, 0, absolute_l1_address};
    }

    constexpr Kind kind() const { return kind_; }
    constexpr std::uint64_t offset() const { return offset_; }

    // Per-kind identity accessors: kind() says which fields are meaningful,
    // and every field is always valid to read (unused ones are zero), so there
    // is no active-member question.
    constexpr std::uint32_t x() const { return x_; }
    constexpr std::uint32_t y() const { return y_; }
    constexpr std::uint32_t bank() const { return bank_; }

    /// @brief Resolve and encode this target through @p Map; empty when the
    /// identity is out of map or the complete transfer does not fit.
    template <const MapData& Map>
    constexpr std::optional<NocAddress> encode(std::uint64_t size = 1) const;

private:
    constexpr Address(Kind kind, std::uint16_t x, std::uint16_t y, std::uint16_t bank, std::uint64_t offset) :
        offset_(offset), x_(x), y_(y), bank_(bank), kind_(kind) {}

    // Members sorted by size so padding stays minimal (one byte).
    std::uint64_t offset_;
    std::uint16_t x_;
    std::uint16_t y_;
    std::uint16_t bank_;
    Kind kind_;
};

static_assert(sizeof(Address) == 16, "keep Address at two words; padding grows silently otherwise");

/// @brief The one resolver: identity -> {window class, selector}, interpreting
/// only the map's data.
constexpr ResolvedTile resolve(const MapData& map, Address address) {
    switch (address.kind()) {
        // Selector 0 of the map's local window reaches the boot-patched
        // per-initiator self endpoint; the local identity is a constant.
        case Address::Kind::Local: return {0, 0, 0, true, map.local_window_class};
        case Address::Kind::Worker: {
            const std::uint32_t x = address.x();
            const std::uint32_t y = address.y();
            if (x < map.worker_origin_x || y < map.worker_origin_y) {
                return INVALID_TILE;
            }
            const std::uint32_t logical_x = x - map.worker_origin_x;
            const std::uint32_t logical_y = y - map.worker_origin_y;
            if (logical_x >= map.worker_grid_x || logical_y >= map.worker_grid_y) {
                return INVALID_TILE;
            }
            const std::uint32_t index = logical_y * map.worker_grid_x + logical_x;
            return {map.worker_selectors[index], x, y, true, WindowClass::Worker};
        }
        case Address::Kind::Dram: {
            const std::uint32_t bank = address.bank();
            if (bank >= map.dram_selectors.size()) {
                return INVALID_TILE;
            }
            return {map.dram_selectors[bank], bank, 0, true, WindowClass::Dram};
        }
        case Address::Kind::Dispatch: {
            for (std::uint32_t i = 0; i < map.dispatch_entries.size(); ++i) {
                const MapData::DispatchEntry& entry = map.dispatch_entries[i];
                if (entry.x == address.x() && entry.y == address.y()) {
                    return {entry.selector, entry.x, entry.y, true, entry.window};
                }
            }
            return INVALID_TILE;
        }
        default: return INVALID_TILE;
    }
}

/// @brief This initiator's identity from its NOC_NODE_ID coordinates, via the
/// generated inverse endpoint tables (physical layout is irregular; this is a
/// table lookup by nature).
constexpr ResolvedTile resolve_current(const MapData& map, std::uint32_t noc_x, std::uint32_t noc_y) {
    const std::uint16_t endpoint = static_cast<std::uint16_t>((noc_y << 6) | noc_x);
    for (std::uint32_t selector = 0; selector < map.worker_endpoint_words.size(); ++selector) {
        if (map.worker_endpoint_words[selector] == endpoint) {
            return {selector, noc_x, noc_y, true, WindowClass::Worker};
        }
    }
    for (std::uint32_t selector = 0; selector < map.full_tile_endpoint_words.size(); ++selector) {
        if (map.full_tile_endpoint_words[selector] == endpoint) {
            return {selector, noc_x, noc_y, true, WindowClass::FullTile};
        }
    }
    for (std::uint32_t selector = 0; selector < map.dram_endpoint_words.size(); ++selector) {
        if (map.dram_endpoint_words[selector] != 0 && map.dram_endpoint_words[selector] == endpoint) {
            return {selector, noc_x, noc_y, true, WindowClass::Dram};
        }
    }
    return INVALID_TILE;
}

// Resolve a coordinate in the kernel-visible (host/descriptor) frame.
constexpr ResolvedTile resolve_host_coordinate(const MapData& map, std::uint32_t x, std::uint32_t y) {
    return resolve_current(map, x + map.node_id_offset_x, y + map.node_id_offset_y);
}

// Whether a kernel-visible (host/descriptor frame) coordinate names the initiator whose NOC_NODE_ID
// coordinates are (@p noc_x, @p noc_y).
constexpr bool host_coordinate_is_current(
    const MapData& map, std::uint32_t x, std::uint32_t y, std::uint32_t noc_x, std::uint32_t noc_y) {
    return x + map.node_id_offset_x == noc_x && y + map.node_id_offset_y == noc_y;
}

template <const MapData& Map>
constexpr std::optional<NocAddress> Address::encode(std::uint64_t size) const {
    if (size == 0) {
        return std::nullopt;
    }
    if (kind_ == Kind::LoopbackScratch) {
        // Scratch operands are absolute addresses in the map's boot-owned
        // pass-through aperture. An absent aperture must not emit raw operands.
        const Window& scratch = map_window(Map, WindowClass::LoopbackScratch);
        if (!is_no_window(scratch) && !scratch.translate_address && scratch.endpoint_size == 0 &&
            scratch.matches(offset_) && scratch.transfer_supported(scratch.local_address(offset_), size)) {
            return offset_;
        }
        return std::nullopt;
    }
    const ResolvedTile resolved = resolve(Map, *this);
    if (!resolved.valid) {
        return std::nullopt;
    }
    const Window& window = map_window(Map, resolved.window);
    if (is_no_window(window) || !window.selector_supported(resolved.selector) ||
        !window.transfer_supported(offset_, size)) {
        return std::nullopt;
    }
    return window.make_address(resolved.selector, offset_);
}

/// @brief Validate a complete NoC operand at the point where its transfer size
/// is known. Address generators often cannot do this because they return a
/// base address before the caller has selected the transaction length.
template <const MapData& Map>
constexpr bool transfer_supported(NocAddress address, std::uint64_t size) {
    if (size == 0) {
        return false;
    }
    const WindowClass window_class = matching_window_class(Map, address);
    if (window_class == WindowClass::Invalid) {
        return false;
    }
    const Window& window = map_window(Map, window_class);
    return window.transfer_supported(window.local_address(address), size);
}

template <const MapData& Map>
constexpr bool is_self_address(NocAddress address, ResolvedTile current_tile) {
    const WindowClass window_class = matching_window_class(Map, address);
    if (window_class == WindowClass::Invalid) {
        return false;
    }
    const std::uint16_t endpoint = map_window(Map, window_class).endpoint_index(address);
    // Every alias of the boot-patched endpoint is self, including scratch.
    // Only this row is unconditional: a local window can select other tiles.
    const Window& self_window = map_window(Map, Map.local_window_class);
    if (!is_no_window(self_window) && endpoint == self_window.endpoint_table_offset) {
        return true;
    }
    if (!current_tile.valid) {
        return false;
    }
    // Compare physical endpoint coordinates, not window classes: one tile can
    // have worker and full-tile aliases, and windows can share endpoint rows.
    const std::uint32_t current_word = (current_tile.noc_y << 6) | current_tile.noc_x;
    constexpr WindowClass endpoint_windows[] = {WindowClass::Worker, WindowClass::FullTile};
    for (WindowClass endpoint_window : endpoint_windows) {
        const Window& window = map_window(Map, endpoint_window);
        const Table<std::uint16_t>& words =
            endpoint_window == WindowClass::Worker ? Map.worker_endpoint_words : Map.full_tile_endpoint_words;
        if (!is_no_window(window) && endpoint >= window.endpoint_table_offset) {
            const std::uint32_t selector = endpoint - window.endpoint_table_offset;
            if (selector < words.size()) {
                return words[selector] == current_word;
            }
        }
    }
    return false;
}

template <const MapData& Map>
constexpr std::optional<NocAddress> extract_local_address(NocAddress address) {
    const WindowClass window_class = matching_window_class(Map, address);
    if (window_class == WindowClass::Invalid) {
        return std::nullopt;
    }
    const Window& window = map_window(Map, window_class);
    // A pass-through window preserves the absolute address delivered by ATT.
    // Translating windows in these maps have BAR 0 and strip the window bits.
    return window.translate_address ? window.local_address(address) : address;
}

/// The packed software multicast-descriptor layout (36-bit local address,
/// four 6-bit rectangle coordinates). This is a software container format the
/// shared get_noc_multicast_addr path produces; it is decoded back into worker
/// coordinates before any ATT resolution, so it carries no XY hardware meaning
/// under ATT.
inline constexpr std::uint32_t DESCRIPTOR_LOCAL_BITS = 36;
inline constexpr std::uint32_t DESCRIPTOR_NODE_BITS = 6;
inline constexpr std::uint64_t DESCRIPTOR_LOCAL_LIMIT = std::uint64_t{1} << DESCRIPTOR_LOCAL_BITS;
inline constexpr std::uint32_t DESCRIPTOR_NODE_LIMIT = 1u << DESCRIPTOR_NODE_BITS;
inline constexpr std::uint32_t DESCRIPTOR_BITS = DESCRIPTOR_LOCAL_BITS + 4 * DESCRIPTOR_NODE_BITS;
// A valid packed descriptor uses only 60 bits. Keep the integer interface while
// carrying packing failure to the resolver in a reserved bit.
inline constexpr NocAddress INVALID_MULTICAST_DESCRIPTOR = NocAddress{1} << DESCRIPTOR_BITS;

/// @brief Multicast is worker-rectangle-only, so it works directly in worker
/// coordinates. The result carries both register forms the Quasar multicast
/// paths need (rectangle_count 0 = invalid rectangle): the start operand plus
/// the extent for regular writes and atomics, whose destination the NOC
/// translates with the rectangle math, and the end operand plus the start and
/// end tiles' endpoint words for inline writes, whose destination field the
/// NOC translates without it (see noc_nonblocking_api_v3.h).
template <const MapData& Map>
constexpr NocMulticastAddress make_worker_multicast(
    std::uint32_t start_x,
    std::uint32_t start_y,
    std::uint32_t end_x,
    std::uint32_t end_y,
    std::uint64_t offset,
    std::uint64_t size = 1) {
    if (end_x < start_x || end_y < start_y) {
        return {};
    }
    const Address start_tile = Address::worker(start_x, start_y, offset);
    const Address end_tile = Address::worker(end_x, end_y, offset);
    const std::optional<NocAddress> start = start_tile.template encode<Map>(size);
    const std::optional<NocAddress> end = end_tile.template encode<Map>(size);
    if (!start.has_value() || !end.has_value()) {
        return {};
    }
    // encode() succeeded, so both tiles resolve to worker selectors; the
    // endpoint table gives their NOC node words.
    const std::uint32_t start_selector = resolve(Map, start_tile).selector;
    const std::uint32_t end_selector = resolve(Map, end_tile).selector;
    if (start_selector >= Map.worker_endpoint_words.size() || end_selector >= Map.worker_endpoint_words.size()) {
        return {};
    }
    const std::uint32_t width = end_x - start_x + 1;
    const std::uint32_t height = end_y - start_y + 1;
    return {
        *start,
        (height << DESCRIPTOR_NODE_BITS) | width,
        width * height,
        *end,
        Map.worker_endpoint_words[start_selector],
        Map.worker_endpoint_words[end_selector]};
}

constexpr NocAddress make_multicast_descriptor(
    std::uint32_t start_x,
    std::uint32_t start_y,
    std::uint32_t end_x,
    std::uint32_t end_y,
    std::uint64_t local_address) {
    if (start_x >= DESCRIPTOR_NODE_LIMIT || start_y >= DESCRIPTOR_NODE_LIMIT || end_x >= DESCRIPTOR_NODE_LIMIT ||
        end_y >= DESCRIPTOR_NODE_LIMIT || local_address >= DESCRIPTOR_LOCAL_LIMIT) {
        return INVALID_MULTICAST_DESCRIPTOR;
    }
    return (std::uint64_t{start_y} << (DESCRIPTOR_LOCAL_BITS + 3 * DESCRIPTOR_NODE_BITS)) |
           (std::uint64_t{start_x} << (DESCRIPTOR_LOCAL_BITS + 2 * DESCRIPTOR_NODE_BITS)) |
           (std::uint64_t{end_y} << (DESCRIPTOR_LOCAL_BITS + DESCRIPTOR_NODE_BITS)) |
           (std::uint64_t{end_x} << DESCRIPTOR_LOCAL_BITS) | local_address;
}

/// @brief The fields of a software multicast descriptor. valid is false when the
/// value is not a descriptor (INVALID_MULTICAST_DESCRIPTOR, or any bit above
/// DESCRIPTOR_BITS set); the fields are then zero.
struct MulticastDescriptorFields {
    std::uint32_t start_x;
    std::uint32_t start_y;
    std::uint32_t end_x;
    std::uint32_t end_y;
    std::uint64_t local_address;
    bool valid;
};

constexpr MulticastDescriptorFields decode_multicast_descriptor(NocAddress descriptor) {
    if ((descriptor >> DESCRIPTOR_BITS) != 0) {
        return {};
    }
    constexpr std::uint32_t node_mask = DESCRIPTOR_NODE_LIMIT - 1;
    return {
        static_cast<std::uint32_t>(descriptor >> (DESCRIPTOR_LOCAL_BITS + 2 * DESCRIPTOR_NODE_BITS)) & node_mask,
        static_cast<std::uint32_t>(descriptor >> (DESCRIPTOR_LOCAL_BITS + 3 * DESCRIPTOR_NODE_BITS)) & node_mask,
        static_cast<std::uint32_t>(descriptor >> DESCRIPTOR_LOCAL_BITS) & node_mask,
        static_cast<std::uint32_t>(descriptor >> (DESCRIPTOR_LOCAL_BITS + DESCRIPTOR_NODE_BITS)) & node_mask,
        descriptor & (DESCRIPTOR_LOCAL_LIMIT - 1),
        true};
}

template <const MapData& Map>
constexpr NocMulticastAddress resolve_worker_multicast(NocAddress descriptor, std::uint64_t size = 1) {
    const MulticastDescriptorFields rect = decode_multicast_descriptor(descriptor);
    if (!rect.valid) {
        return {};
    }
    return make_worker_multicast<Map>(rect.start_x, rect.start_y, rect.end_x, rect.end_y, rect.local_address, size);
}


// ---------------------------------------------------------------------------
// Operand diagnostics: the inverse of encode(). Given a complete operand and
// the map alone, say what it reaches. Used by the watcher's NoC sanitizer on
// the device and by the watcher's report on the host; never on an issue path.
// ---------------------------------------------------------------------------

/// @brief Dram target whose selector no logical bank is bound to.
inline constexpr std::uint32_t DRAM_BANK_UNKNOWN = ~std::uint32_t{0};

/// @brief Where a complete operand lands.
struct OperandTarget {
    enum class Kind : std::uint8_t {
        Invalid,   ///< matches no window, or names a selector the map does not list
        Self,      ///< this initiator's own L1: the local window at selector 0, or the pass-through scratch aperture
        Worker,    ///< a worker tile's L1 through the worker window
        FullTile,  ///< a tile's address space through the full-tile window (workers, dispatch tiles, perimeter)
        Dram,      ///< a DRAM channel through the DRAM window
    };
    Kind kind;
    /// The window the operand matched; Invalid when none did.
    WindowClass window;
    /// The selector field of that window (0 when the window is selector-free).
    std::uint32_t selector;
    /// The byte address the target sees: the window's local field, or the whole
    /// operand through a selector-free pass-through aperture.
    std::uint64_t local_address;
    /// Whether endpoint_word names the target tile.
    bool endpoint_known;
    /// The target tile as (y << 6) | x in the NOC_NODE_ID frame, when known.
    std::uint32_t endpoint_word;
    /// Dram only: the logical bank bound to the selector, else DRAM_BANK_UNKNOWN.
    std::uint32_t bank;
};

constexpr bool same_window(const Window& a, const Window& b) {
    return a.compare == b.compare && a.mask_bits == b.mask_bits && a.endpoint_shift == b.endpoint_shift &&
           a.endpoint_size == b.endpoint_size && a.endpoint_table_offset == b.endpoint_table_offset;
}

/// @brief Classify a complete unicast operand. Multicast descriptors are a
/// different container (decode_multicast_descriptor). The checks run in the
/// order the endpoint tables overlap: Self first (on a map whose local window
/// is a shared window, selector 0 is the boot-patched self row), then Dram
/// (on a map with one shared remote window only the selector tells DRAM from
/// tiles, and DRAM selectors also appear in the full-tile table), then Worker,
/// then FullTile.
constexpr OperandTarget classify_operand(const MapData& map, NocAddress address) {
    using Kind = OperandTarget::Kind;
    const auto in_window = [address](const Window& window) {
        return !is_no_window(window) && window.matches(address);
    };
    const Window& local = map_window(map, map.local_window_class);
    if (in_window(local) && local.selector(address) == 0) {
        return {Kind::Self, map.local_window_class, 0, local.local_address(address), false, 0, DRAM_BANK_UNKNOWN};
    }
    const Window& scratch = map_window(map, WindowClass::LoopbackScratch);
    if (in_window(scratch)) {
        // Selector-free and pass-through: the operand itself is the absolute L1 address.
        const std::uint64_t seen = scratch.translate_address ? scratch.local_address(address) : address;
        return {Kind::Self, WindowClass::LoopbackScratch, 0, seen, false, 0, DRAM_BANK_UNKNOWN};
    }
    const Window& dram = map_window(map, WindowClass::Dram);
    const Window& tile = map_window(map, WindowClass::FullTile);
    if (in_window(dram)) {
        const std::uint32_t selector = dram.selector(address);
        std::uint32_t bank = DRAM_BANK_UNKNOWN;
        for (std::uint32_t b = 0; b < map.dram_selectors.size(); ++b) {
            if (map.dram_selectors[b] == selector) {
                bank = b;
                break;
            }
        }
        // The DRAM tile's word: its own table (0 = row not programmed), or the
        // full-tile table when the map's DRAM tiles live there behind a shared window.
        bool known = selector < map.dram_endpoint_words.size() && map.dram_endpoint_words[selector] != 0;
        std::uint32_t word = known ? map.dram_endpoint_words[selector] : 0;
        if (!known && bank != DRAM_BANK_UNKNOWN && !is_no_window(tile) && same_window(dram, tile) &&
            selector < map.full_tile_endpoint_words.size()) {
            known = true;
            word = map.full_tile_endpoint_words[selector];
        }
        if (bank != DRAM_BANK_UNKNOWN || known) {
            return {Kind::Dram, WindowClass::Dram, selector, dram.local_address(address), known, word, bank};
        }
    }
    const Window& worker = map_window(map, WindowClass::Worker);
    if (in_window(worker)) {
        const std::uint32_t selector = worker.selector(address);
        if (selector < map.worker_endpoint_words.size()) {
            return {
                Kind::Worker,
                WindowClass::Worker,
                selector,
                worker.local_address(address),
                true,
                map.worker_endpoint_words[selector],
                DRAM_BANK_UNKNOWN};
        }
    }
    if (in_window(tile)) {
        const std::uint32_t selector = tile.selector(address);
        if (selector < map.full_tile_endpoint_words.size()) {
            return {
                Kind::FullTile,
                WindowClass::FullTile,
                selector,
                tile.local_address(address),
                true,
                map.full_tile_endpoint_words[selector],
                DRAM_BANK_UNKNOWN};
        }
    }
    const WindowClass matched = matching_window_class(map, address);
    const std::uint32_t selector = matched == WindowClass::Invalid ? 0 : map_window(map, matched).selector(address);
    return {Kind::Invalid, matched, selector, 0, false, 0, DRAM_BANK_UNKNOWN};
}
}  // namespace noc_att
