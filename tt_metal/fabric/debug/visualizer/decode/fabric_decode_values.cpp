// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"

namespace tt::tt_fabric::decode {

namespace {

// Why a region has no bytes: the snapshot's reason when the router wasn't read in full, else `otherwise`, a reason
// of the region's own.
Unread get_missing_bytes_reason(const RouterSnapshot& snapshot, Unread otherwise) {
    return snapshot.get_unread_reason().value_or(std::move(otherwise));
}

// Element `index` of a region: the sizeof(T) bytes at index * element_size, copied into a T. T is trivially copyable
// (the registry checks), and needn't be default constructible.
template <typename T>
T read_element(std::span<const std::byte> region_bytes, uint32_t element_size, uint32_t index) {
    std::array<std::byte, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), region_bytes.data() + (static_cast<size_t>(index) * element_size), sizeof(T));
    return std::bit_cast<T>(bytes);
}

// Where a read's bytes start in the .bin. Every read the snapshot keeps worked, and a read that worked has an offset.
uint64_t get_read_bin_offset(const capture::ReadRecord& read) {
    TT_FATAL(
        read.bin_offset.has_value(),
        "Fabric decode: the capture record's read at {:#x} didn't fail, but has no .bin offset",
        read.address);
    return *read.bin_offset;
}

// What reading one region gives: its reads, its bytes if it's inactive, or why it has none.
using RegionResult = std::variant<TypedRegionReads, InactiveRegion, Unread>;

// A region's elements, each the sizeof(T) bytes at index * element_size of `region_bytes`, which hold the whole region.
template <typename T>
std::vector<T> read_elements(std::span<const std::byte> region_bytes, const manifest::content::L1& region) {
    std::vector<T> elements;
    elements.reserve(region.num_elements);
    for (uint32_t index = 0; index < region.num_elements; ++index) {
        elements.push_back(read_element<T>(region_bytes, region.element_size, index));
    }
    return elements;
}

// The L1 region's reads: one per round that read it, when the manifest marks it multi_sampled, then its read in the
// full read's L1 read. Or its bytes, when it's inactive. Or why it has none: an element is smaller than T, the region
// isn't inside the captured L1, or capture read none of it. The bounds are checked once, so each element is read
// without checks.
template <typename T>
RegionResult read_l1_region(const RouterSnapshot& snapshot, const manifest::content::L1& region) {
    const uint64_t region_size_bytes = static_cast<uint64_t>(region.num_elements) * region.element_size;
    const capture::ReadRecord* l1_read = snapshot.find_l1_read();
    std::span<const std::byte> full_read_bytes;
    if (l1_read != nullptr) {
        const uint64_t l1_end = l1_read->address + l1_read->size_bytes;
        if (region.address < l1_read->address || region.address > l1_end ||
            region_size_bytes > l1_end - region.address) {
            return Unread{
                .code = UnreadCode::OUTSIDE_CAPTURED_L1,
                .message = fmt::format(
                    "{:#x} to {:#x} isn't inside the captured L1, {:#x} to {:#x}",
                    region.address,
                    region.address + region_size_bytes,
                    l1_read->address,
                    l1_end)};
        }
        full_read_bytes = snapshot.get_l1_bytes().subspan(region.address - l1_read->address, region_size_bytes);
    }

    // Nothing writes an inactive region, so its bytes aren't values: they're kept as they are. The manifest's reader
    // refuses an inactive region that's multi_sampled, so it has no round reads.
    if (!region.active) {
        if (l1_read == nullptr) {
            return get_missing_bytes_reason(
                snapshot, Unread{.code = UnreadCode::OUTSIDE_CAPTURED_L1, .message = "capture didn't read the L1"});
        }
        return InactiveRegion{
            .bytes = full_read_bytes,
            .bin_offset = get_read_bin_offset(*l1_read) + (region.address - l1_read->address)};
    }

    if (sizeof(T) > region.element_size) {
        return Unread{
            .code = UnreadCode::TYPE_LARGER_THAN_ELEMENT,
            .message = fmt::format(
                "{} is {} bytes, but the region's elements are {} bytes apart",
                region.type,
                sizeof(T),
                region.element_size)};
    }

    RegionReads<T> reads;
    if (region.multi_sampled.has_value()) {
        for (const RoundRead& round_read : snapshot.find_round_reads(region.address)) {
            TT_FATAL(
                round_read.bytes.size() == region_size_bytes,
                "Fabric decode: round {} read {} bytes at {:#x}, but the manifest's region there is {} bytes",
                round_read.round,
                round_read.bytes.size(),
                region.address,
                region_size_bytes);
            reads.push_back(RegionRead<T>{
                .round = round_read.round,
                .t_ns = round_read.read->t_ns,
                .bin_offset = get_read_bin_offset(*round_read.read),
                .elements = read_elements<T>(round_read.bytes, region)});
        }
    }
    if (l1_read != nullptr) {
        reads.push_back(RegionRead<T>{
            .t_ns = l1_read->t_ns,
            .bin_offset = get_read_bin_offset(*l1_read) + (region.address - l1_read->address),
            .elements = read_elements<T>(full_read_bytes, region)});
    }
    if (reads.empty()) {
        return get_missing_bytes_reason(
            snapshot, Unread{.code = UnreadCode::OUTSIDE_CAPTURED_L1, .message = "capture didn't read the L1"});
    }
    return TypedRegionReads{std::in_place_type<RegionReads<T>>, std::move(reads)};
}

// A stream register's value: the word's low value_bits bits. The bits above can hold other state (Blackhole keeps
// more above BUF_SPACE_AVAILABLE's count).
constexpr uint32_t mask_stream_register_word(uint32_t word, uint32_t value_bits) {
    return value_bits >= 32 ? word : word & ((uint32_t{1} << value_bits) - 1);
}

// A stream register's value read as T, which fits in a word.
template <typename T>
T read_stream_register_value(uint32_t word, uint32_t value_bits) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    const auto value =
        std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(mask_stream_register_word(word, value_bits));
    std::array<std::byte, sizeof(T)> bytes{};
    std::copy_n(value.begin(), sizeof(T), bytes.begin());
    return std::bit_cast<T>(bytes);
}

// A round read's word: the read's four bytes.
uint32_t read_round_word(const RoundRead& round_read, uint64_t address) {
    TT_FATAL(
        round_read.bytes.size() == sizeof(uint32_t),
        "Fabric decode: round {} read {} bytes of the register at {:#x}, not a word",
        round_read.round,
        round_read.bytes.size(),
        address);
    std::array<std::byte, sizeof(uint32_t)> word{};
    std::ranges::copy(round_read.bytes, word.begin());
    return std::bit_cast<uint32_t>(word);
}

// The stream register's reads, each its value as T: one per round that read it, when the manifest marks it
// multi_sampled, then its read in the full read. Or why capture read none.
template <typename T>
RegionResult read_stream_register(const RouterSnapshot& snapshot, const manifest::content::StreamReg& stream_register) {
    if constexpr (sizeof(T) > sizeof(uint32_t)) {
        TT_THROW(
            "Fabric decode: the manifest names {} for the stream register at {:#x}, but it's larger than a word",
            stream_register.type,
            stream_register.address);
    } else {
        RegionReads<T> reads;
        if (stream_register.multi_sampled.has_value()) {
            for (const RoundRead& round_read : snapshot.find_round_reads(stream_register.address)) {
                const uint32_t word = read_round_word(round_read, stream_register.address);
                reads.push_back(RegionRead<T>{
                    .round = round_read.round,
                    .t_ns = round_read.read->t_ns,
                    .bin_offset = get_read_bin_offset(*round_read.read),
                    .word = word,
                    .elements = {read_stream_register_value<T>(word, stream_register.value_bits)}});
            }
        }
        if (const std::optional<RegisterRead> register_read = snapshot.find_register_read(stream_register.address)) {
            reads.push_back(RegionRead<T>{
                .t_ns = register_read->read->t_ns,
                .bin_offset = get_read_bin_offset(*register_read->read),
                .word = register_read->word,
                .elements = {read_stream_register_value<T>(register_read->word, stream_register.value_bits)}});
        }
        if (reads.empty()) {
            return get_missing_bytes_reason(
                snapshot,
                Unread{
                    .code = UnreadCode::REGISTER_NOT_CAPTURED,
                    .message = fmt::format("capture didn't read the register at {:#x}", stream_register.address)});
        }
        return TypedRegionReads{std::in_place_type<RegionReads<T>>, std::move(reads)};
    }
}

// Calls read(std::type_identity<T>{}) for the registered type T named `type`, and returns its result. The parser
// refuses a manifest naming a type this build doesn't register, so failing here is a decode bug.
template <typename Read>
RegionResult read_as_named_type(std::string_view type, std::string_view path, const Read& read) {
    std::optional<RegionResult> result;
    const bool registered =
        manifest::visit_registered_type(type, [&](auto type_identity) { result.emplace(read(type_identity)); });
    TT_FATAL(registered, "Fabric decode: {} names type {}, which this build doesn't register", path, type);
    return std::move(*result);
}

// Keeps what reading the region at `path` gave: its reads in `reads`, its bytes in `inactive`, or why it's unread in
// `unread`.
void keep_read_result(
    std::string_view path,
    RegionResult result,
    std::map<std::string, TypedRegionReads, std::less<>>& reads,
    std::map<std::string, InactiveRegion, std::less<>>& inactive,
    std::map<std::string, Unread, std::less<>>& unread) {
    const bool kept = std::visit(
        ttsl::overloaded{
            [&](TypedRegionReads& region_reads) { return reads.emplace(path, std::move(region_reads)).second; },
            [&](InactiveRegion& region) { return inactive.emplace(path, region).second; },
            [&](Unread& reason) { return unread.emplace(path, std::move(reason)).second; },
        },
        result);
    TT_FATAL(kept, "Fabric decode: the router names a region at {} twice", path);
}

}  // namespace

RouterValues::RouterValues(const RouterSnapshot& snapshot, const manifest::Arch& arch) : snapshot_(&snapshot) {
    const auto read_l1_region_at = [&](std::string_view path, const manifest::content::L1& region) {
        keep_read_result(
            path,
            read_as_named_type(
                region.type,
                path,
                [&]<typename T>(std::type_identity<T>) { return read_l1_region<T>(snapshot, region); }),
            reads_,
            inactive_,
            unread_);
    };
    const auto read_stream_register_at = [&](std::string_view path,
                                             const manifest::content::StreamReg& stream_register) {
        keep_read_result(
            path,
            read_as_named_type(
                stream_register.type,
                path,
                [&]<typename T>(std::type_identity<T>) { return read_stream_register<T>(snapshot, stream_register); }),
            reads_,
            inactive_,
            unread_);
    };

    // Name the lambda types so overloaded holds copies; deduction would make it hold const references.
    manifest::for_each_router_region(
        *snapshot.get_router().router,
        ttsl::overloaded<decltype(read_l1_region_at), decltype(read_stream_register_at)>{
            read_l1_region_at, read_stream_register_at});
    // Every router core has its own copy of the areas the architecture fixes.
    manifest::for_each_arch_area(arch, [&](std::string_view name, const manifest::content::L1& area) {
        read_l1_region_at(make_arch_area_path(name), area);
    });
}

const InactiveRegion* RouterValues::find_inactive(std::string_view path) const {
    const auto region = inactive_.find(path);
    return region == inactive_.end() ? nullptr : &region->second;
}

const Unread* RouterValues::find_unread(std::string_view path) const {
    const auto reason = unread_.find(path);
    return reason == unread_.end() ? nullptr : &reason->second;
}

}  // namespace tt::tt_fabric::decode
