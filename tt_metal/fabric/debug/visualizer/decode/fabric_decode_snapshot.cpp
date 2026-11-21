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

// Whether `read` is the full read's L1 read: the manifest's whole Ethernet core L1. Every other read of the full read
// is a register's word.
bool is_l1_read(const capture::ReadRecord& read, const manifest::EthCoreL1& l1) {
    return read.address == l1.address && read.size_bytes == l1.size_bytes;
}

// A register read, with its word from the .bin.
RegisterRead make_register_read(const capture::LoadedCapture& capture, const capture::ReadRecord& read) {
    const std::span<const std::byte> bytes = capture.get_read_bytes(read);
    TT_FATAL(
        bytes.size() == sizeof(uint32_t),
        "Fabric decode: the capture record's register read at {:#x} is {} bytes, not a word",
        read.address,
        bytes.size());
    std::array<std::byte, sizeof(uint32_t)> word{};
    std::ranges::copy(bytes, word.begin());
    return RegisterRead{.read = &read, .word = std::bit_cast<uint32_t>(word)};
}

}  // namespace

RouterSnapshot::RouterSnapshot(
    const CapturedRouter& router, const capture::LoadedCapture& capture, const manifest::EthCoreL1& l1) :
    router_(router) {
    unread_reason_ = find_not_read_reason(router_);
    if (unread_reason_.has_value()) {
        return;
    }

    // The rounds' reads that worked, whatever the full read did. A failed round read has no bytes, and is a finding
    // rather than a gap in the router's bytes.
    for (const capture::RoundRecord& round : router_.record->rounds) {
        for (const capture::ReadRecord& read : round.reads) {
            if (!read.error.has_value()) {
                round_reads_[read.address].push_back(
                    RoundRead{.round = round.round, .read = &read, .bytes = capture.get_read_bytes(read)});
            }
        }
    }

    // The full read: the core registers, the whole L1, then each stream register. Capture stopped at its first read
    // that failed.
    for (const capture::ReadRecord& read : router_.record->reads) {
        const bool l1_read = is_l1_read(read, l1);
        if (read.error.has_value()) {
            status_ = RouterReadStatus::PARTIAL;
            unread_reason_ = Unread{
                .code = l1_read ? UnreadCode::L1_READ_FAILED : UnreadCode::REGISTER_READ_FAILED,
                .message = *read.error};
            return;
        }
        if (l1_read) {
            l1_read_ = &read;
            l1_bytes_ = capture.get_read_bytes(read);
            continue;
        }
        const bool first_read = register_reads_.emplace(read.address, make_register_read(capture, read)).second;
        TT_FATAL(
            first_read,
            "Fabric decode: the capture record reads the register at {:#x} of {} twice in its full read",
            read.address,
            router_.path);
    }
    TT_FATAL(
        l1_read_ != nullptr,
        "Fabric decode: the capture record has no L1 read for {}, though none of its reads failed",
        router_.path);
    status_ = RouterReadStatus::COMPLETE;
}

const capture::ReadRecord* RouterSnapshot::find_failed_read() const {
    if (router_.record == nullptr) {
        return nullptr;
    }
    const auto read = std::ranges::find_if(
        router_.record->reads, [](const capture::ReadRecord& read) { return read.error.has_value(); });
    return read == router_.record->reads.end() ? nullptr : &*read;
}

std::optional<RegisterRead> RouterSnapshot::find_register_read(uint64_t address) const {
    const auto read = register_reads_.find(address);
    return read == register_reads_.end() ? std::nullopt : std::optional(read->second);
}

std::span<const RoundRead> RouterSnapshot::find_round_reads(uint64_t address) const {
    const auto reads = round_reads_.find(address);
    return reads == round_reads_.end() ? std::span<const RoundRead>{} : std::span<const RoundRead>(reads->second);
}

std::vector<RouterSnapshot> make_router_snapshots(
    const RouterPairing& pairing, const capture::LoadedCapture& capture, const manifest::EthCoreL1& l1) {
    std::vector<RouterSnapshot> snapshots;
    snapshots.reserve(pairing.routers.size());
    for (const CapturedRouter& router : pairing.routers) {
        snapshots.emplace_back(router, capture, l1);
    }
    return snapshots;
}

}  // namespace tt::tt_fabric::decode
