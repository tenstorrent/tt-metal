// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Typed read: every region a router's manifest entry names, read out of its snapshot into its registered C++ type.
// This is the only part of decode that reads captured bytes; everything after it works on the values.

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_snapshot.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::decode {

static_assert(std::endian::native == std::endian::little, "Decode copies the device's little-endian bytes as they are");

// One read of a region: which of capture's passes made it, when, and the region's elements, read as T. A stream
// register has one element, its word under the manifest's read_mask: the value the router reads.
template <typename T>
struct RegionRead {
    // The round that read it. Null for the full read: the L1 read for an L1 region, or the register's one read.
    std::optional<uint32_t> round;
    // When the read started, on capture's clock.
    uint64_t t_ns = 0;
    // Where the region's bytes start in the .bin: inside the full read's L1 read for an L1 region, or the read's own
    // bytes for a round read or a stream register.
    uint64_t bin_offset = 0;
    // A stream register's whole word, before it's masked to read_mask, so bits the router doesn't read still show.
    // Null for an L1 region.
    std::optional<uint32_t> word;
    std::vector<T> elements;
};

// Every read of one region, in the order capture made them: one per round that read it, for a region the manifest
// marks multi_sampled, then its full read, unless capture didn't make it.
template <typename T>
using RegionReads = std::vector<RegionRead<T>>;

// An inactive L1 region's bytes, in the full read's L1 read.
struct InactiveL1 {
    std::span<const std::byte> bytes;
    // Where they start in the .bin.
    uint64_t bin_offset = 0;
};

// An inactive stream register's word, from the full read.
struct InactiveRegister {
    uint32_t word = 0;
    // Where it is in the .bin.
    uint64_t bin_offset = 0;
};

// A region the manifest marks inactive: allocated, but nothing on this run writes it.
using InactiveRegion = std::variant<InactiveL1, InactiveRegister>;

namespace detail {

// std::variant<Wrap<T>...> over the types in a TypeList.
// ie. if TypeList is (T1, T2, T3), and Wrap = vector, then the type is
// std::variant<vector<T1>, vector<T2>, vector<T3>>.
template <template <typename> class Wrap, typename List>
struct variant_of_each;

template <template <typename> class Wrap, typename... Ts>
struct variant_of_each<Wrap, manifest::TypeList<Ts...>> {
    using type = std::variant<Wrap<Ts>...>;
};

}  // namespace detail

// A region's reads, as whichever registered type the manifest names for it.
using TypedRegionReads = typename detail::variant_of_each<RegionReads, manifest::RegisteredTypes>::type;

// The path of the arch area `name` (for_each_arch_area's name) below a router's node, where each router core's copy
// of it is: "l1_areas/<name>", such as "l1_areas/heartbeat".
inline std::string make_arch_area_path(std::string_view name) { return fmt::format("l1_areas/{}", name); }

// Every region one router's manifest entry names, read into its registered type, by path. A region has at least one
// read, or is inactive, or is Unread, never empty. Paths are for_each_router_region's, below the router's node; the
// arch areas, which every router core has its own copy of, are at "l1_areas/<name>" (for_each_arch_area's name). Valid
// while the snapshot is.
class RouterValues {
public:
    // Reads every region out of the snapshot. Checks each L1 region's bounds once (inside the captured L1, and an
    // element at least as large as its type), then bit_casts each element.
    RouterValues(const RouterSnapshot& snapshot, const manifest::Arch& arch);

    const RouterSnapshot& get_snapshot() const { return *snapshot_; }

    // Every read of the region at `path`. Null when there's no region there, or it's inactive or unread. Fails when
    // the manifest names another type than T there: analysis and the manifest then disagree, a decode bug.
    template <typename T>
    const RegionReads<T>* find_reads(std::string_view path) const;

    // The region's read in `round`, or its full read when `round` is null. Null when it has none there. Fails as
    // find_reads does.
    template <typename T>
    const RegionRead<T>* find_read(std::string_view path, std::optional<uint32_t> round) const;

    // What capture read of the inactive region at `path`. Null when the region there is active, or there's none.
    const InactiveRegion* find_inactive(std::string_view path) const;

    // Why the region at `path` has no bytes. Null when it was read, or there's no region there.
    const Unread* find_unread(std::string_view path) const;

    // How many of the router's regions, the arch areas' copies included, were read, are inactive, or are unread. Each
    // region is exactly one of the three.
    size_t get_num_read() const { return reads_.size(); }
    size_t get_num_inactive() const { return inactive_.size(); }
    size_t get_num_unread() const { return unread_.size(); }
    // Every unread region's reason, by path.
    const std::map<std::string, Unread, std::less<>>& get_unread_regions() const { return unread_; }

private:
    const RouterSnapshot* snapshot_;
    std::map<std::string, TypedRegionReads, std::less<>> reads_;
    std::map<std::string, InactiveRegion, std::less<>> inactive_;
    std::map<std::string, Unread, std::less<>> unread_;
};

template <typename T>
const RegionReads<T>* RouterValues::find_reads(std::string_view path) const {
    const auto found = reads_.find(path);
    if (found == reads_.end()) {
        return nullptr;
    }
    const RegionReads<T>* reads = std::get_if<RegionReads<T>>(&found->second);
    TT_FATAL(
        reads != nullptr,
        "Fabric decode: analysis reads {} as {}, but the manifest names another type there",
        path,
        manifest::manifest_type_name<T>);
    return reads;
}

template <typename T>
const RegionRead<T>* RouterValues::find_read(std::string_view path, std::optional<uint32_t> round) const {
    const RegionReads<T>* reads = find_reads<T>(path);
    if (reads == nullptr) {
        return nullptr;
    }
    const auto read = std::ranges::find(*reads, round, &RegionRead<T>::round);
    return read == reads->end() ? nullptr : &*read;
}

}  // namespace tt::tt_fabric::decode
