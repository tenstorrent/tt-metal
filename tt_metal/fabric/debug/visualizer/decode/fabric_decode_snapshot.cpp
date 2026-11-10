// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_snapshot.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::decode {

namespace {

// Why capture didn't read the router: the record has no entry for it, its chip wasn't found or is another
// architecture, or its core wasn't found. Null when capture read it.
std::optional<Unread> find_not_read_reason(const CapturedRouter& router) {
    if (router.record == nullptr) {
        return Unread{.code = UnreadCode::NOT_IN_RECORD, .message = "the capture record has no entry for the router"};
    }
    switch (router.chip_record->location) {
        case capture::ChipLocation::FOUND: break;
        case capture::ChipLocation::MISSING:
            return Unread{
                .code = UnreadCode::CHIP_MISSING,
                .message = fmt::format("the reader found no chip with asic_id {}", router.chip_record->asic_id)};
        case capture::ChipLocation::ARCH_MISMATCH:
            return Unread{
                .code = UnreadCode::CHIP_ARCH_MISMATCH, .message = "the chip isn't the manifest's architecture"};
    }
    const capture::RouterTarget& target = router.record->target;
    TT_FATAL(
        target.location.has_value(),
        "Fabric decode: the capture record found {}'s chip, but has no location for the router",
        router.path);
    switch (*target.location) {
        case capture::CoreLocation::FOUND:
        case capture::CoreLocation::CORE_MISMATCH: return std::nullopt;
        case capture::CoreLocation::CHANNEL_NOT_FOUND:
            return Unread{
                .code = UnreadCode::CHANNEL_NOT_FOUND,
                .message = fmt::format("the chip has no Ethernet core for channel {}", target.router.eth_chan)};
        case capture::CoreLocation::CORE_NOT_FOUND:
            return Unread{
                .code = UnreadCode::CORE_NOT_FOUND,
                .message = fmt::format(
                    "the manifest's core ({}, {}) isn't an Ethernet core of the chip",
                    target.router.virtual_core.x,
                    target.router.virtual_core.y)};
    }
    TT_THROW("Fabric decode: unknown core location {}", static_cast<int>(*target.location));
}

// Why a router is PARTIAL, by the step of the read that failed.
UnreadCode get_failed_read_code(capture::ReadStep step) {
    switch (step) {
        case capture::ReadStep::REGISTERS_BEFORE:
        case capture::ReadStep::REGISTERS_AFTER: return UnreadCode::REGISTER_READ_FAILED;
        case capture::ReadStep::L1: return UnreadCode::L1_READ_FAILED;
    }
    TT_THROW("Fabric decode: unknown read step {}", static_cast<int>(step));
}

// A register read's word, from the .bin.
uint32_t read_register_word(const capture::LoadedCapture& capture, const capture::ReadRecord& read) {
    const std::span<const std::byte> bytes = capture.get_read_bytes(read);
    TT_FATAL(
        bytes.size() == sizeof(uint32_t),
        "Fabric decode: the capture record's register read at {:#x} is {} bytes, not a word",
        read.address,
        bytes.size());
    std::array<std::byte, sizeof(uint32_t)> word{};
    std::ranges::copy(bytes, word.begin());
    return std::bit_cast<uint32_t>(word);
}

// The first of the router's reads that `matches`, or null. A router with no record entry has no reads.
template <typename Matches>
const capture::ReadRecord* find_read(const CapturedRouter& router, Matches matches) {
    if (router.record == nullptr) {
        return nullptr;
    }
    const auto read = std::ranges::find_if(router.record->reads, matches);
    return read == router.record->reads.end() ? nullptr : &*read;
}

}  // namespace

RouterSnapshot::RouterSnapshot(const CapturedRouter& router, const capture::LoadedCapture& capture) : router_(router) {
    unread_reason_ = find_not_read_reason(router_);
    if (unread_reason_.has_value()) {
        return;
    }
    // Capture read the core in ReadStep order, and stopped at the first read that failed.
    bool read_l1 = false;
    for (const capture::ReadRecord& read : router_.record->reads) {
        if (read.error.has_value()) {
            status_ = RouterReadStatus::PARTIAL;
            unread_reason_ = Unread{.code = get_failed_read_code(read.step), .message = *read.error};
            return;
        }
        switch (read.step) {
            case capture::ReadStep::REGISTERS_BEFORE:
                register_words_[read.address].before = read_register_word(capture, read);
                break;
            case capture::ReadStep::L1:
                l1_bytes_ = capture.get_read_bytes(read);
                read_l1 = true;
                break;
            case capture::ReadStep::REGISTERS_AFTER:
                register_words_[read.address].after = read_register_word(capture, read);
                break;
        }
    }
    TT_FATAL(
        read_l1,
        "Fabric decode: the capture record has no L1 read for {}, though none of its reads failed",
        router_.path);
    status_ = RouterReadStatus::COMPLETE;
}

const capture::ReadRecord* RouterSnapshot::find_failed_read() const {
    return find_read(router_, [](const capture::ReadRecord& read) { return read.error.has_value(); });
}

const capture::ReadRecord* RouterSnapshot::find_l1_read() const {
    return find_read(router_, [](const capture::ReadRecord& read) {
        return read.step == capture::ReadStep::L1 && !read.error.has_value();
    });
}

RegisterWords RouterSnapshot::find_register_words(uint64_t address) const {
    const auto words = register_words_.find(address);
    return words == register_words_.end() ? RegisterWords{} : words->second;
}

std::vector<RouterSnapshot> make_router_snapshots(const RouterPairing& pairing, const capture::LoadedCapture& capture) {
    std::vector<RouterSnapshot> snapshots;
    snapshots.reserve(pairing.routers.size());
    for (const CapturedRouter& router : pairing.routers) {
        snapshots.emplace_back(router, capture);
    }
    return snapshots;
}

}  // namespace tt::tt_fabric::decode
