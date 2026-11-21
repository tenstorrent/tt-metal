// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_load.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_inputs.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// How much of a router's full read capture made. Its rounds don't count: a failed round read is a finding, not a gap
// in the router's bytes.
enum class RouterReadStatus {
    // The core registers, the whole L1 and every stream register: all read.
    COMPLETE,
    // A read of the full read failed, and nothing after it was read (capture stops a core at its first failure).
    PARTIAL,
    // Capture read nothing of the router.
    NOT_READ,
};

// Reasons for why something has no bytes: a whole router, or one region of it.
// The decoded file writes it by lower_enum_name.
enum class UnreadCode {
    // -- Why a router is NOT_READ or PARTIAL. --

    // The capture record has no entry for the router.
    NOT_IN_RECORD,
    // The reader found no chip with the manifest's asic_id (capture::ChipLocation::MISSING).
    CHIP_MISSING,
    // The chip isn't the manifest's architecture (capture::ChipLocation::ARCH_MISMATCH).
    CHIP_ARCH_MISMATCH,
    // The chip has no Ethernet core for the router's channel (capture::CoreLocation::CHANNEL_NOT_FOUND).
    CHANNEL_NOT_FOUND,
    // The manifest's core isn't an Ethernet core of the chip (capture::CoreLocation::CORE_NOT_FOUND).
    CORE_NOT_FOUND,
    // A register read of the full read, a core register or a stream register, failed.
    REGISTER_READ_FAILED,
    // The full read's L1 read failed.
    L1_READ_FAILED,

    // -- Why one region of a router that was read has no bytes. --

    // The manifest names a stream register capture didn't read.
    REGISTER_NOT_CAPTURED,
    // The region isn't inside the L1 capture read.
    OUTSIDE_CAPTURED_L1,
    // One element of the region is smaller than its type, so the region's layout isn't this build's.
    TYPE_LARGER_THAN_ELEMENT,
};

// Why something has no bytes. Recorded where its value would be, never thrown.
struct Unread {
    UnreadCode code = UnreadCode::NOT_IN_RECORD;
    // Capture's error text where there is one, else what decode found.
    std::string message;
};

// One read of a register in the full read: the read in the record (its address and t_ns), and the word it read, from
// the .bin.
struct RegisterRead {
    const capture::ReadRecord* read = nullptr;
    uint32_t word = 0;
};

// One read of a multi-sampled region in one of capture's rounds: the round, the read in the record (its address, size
// and t_ns), and its bytes, from the .bin.
struct RoundRead {
    uint32_t round = 0;
    const capture::ReadRecord* read = nullptr;
    std::span<const std::byte> bytes;
};

// One router's capture: how much of it was read, and the bytes that were. It keeps only what the record doesn't
// have, which is the read status and the bytes from the .bin. Its getters look everything else up in the record. The
// pointers and spans it returns are into the record and the mapped .bin, and are valid while the DecodeInputs are.
class RouterSnapshot {
public:
    // Works out the router's read status from its full read, and finds its bytes in the .bin, its rounds' and its full
    // read's. `l1` is the manifest's Ethernet core L1, which the full read's L1 read covers. Copies no bytes.
    RouterSnapshot(const CapturedRouter& router, const capture::LoadedCapture& capture, const manifest::EthCoreL1& l1);

    // The router, its manifest entry, and its record entry, which has its rounds and where its core was found.
    const CapturedRouter& get_router() const { return router_; }
    RouterReadStatus get_read_status() const { return status_; }
    // Why the router is NOT_READ or PARTIAL. Null when it's COMPLETE.
    const std::optional<Unread>& get_unread_reason() const { return unread_reason_; }
    // The full read's read that failed: its address and error. Null when none did.
    const capture::ReadRecord* find_failed_read() const;
    // The full read's L1 read: its address, size and .bin offset. Null when the L1 wasn't read.
    const capture::ReadRecord* find_l1_read() const { return l1_read_; }
    // The L1 read's bytes. Empty when the L1 wasn't read.
    std::span<const std::byte> get_l1_bytes() const { return l1_bytes_; }
    // The full read's read of the register at `address`. Null when capture read no register there.
    std::optional<RegisterRead> find_register_read(uint64_t address) const;
    // Every read that worked of the multi-sampled region at `address` in capture's rounds, in round order. Empty when
    // the region isn't multi-sampled, or no round read it.
    std::span<const RoundRead> find_round_reads(uint64_t address) const;

private:
    CapturedRouter router_;
    RouterReadStatus status_ = RouterReadStatus::NOT_READ;
    std::optional<Unread> unread_reason_;
    const capture::ReadRecord* l1_read_ = nullptr;
    std::span<const std::byte> l1_bytes_;
    // The full read's register reads, core registers and stream registers, by address.
    std::unordered_map<uint64_t, RegisterRead> register_reads_;
    // The rounds' reads that worked, by address.
    std::unordered_map<uint64_t, std::vector<RoundRead>> round_reads_;
};

// One snapshot per paired router, in the pairing's order. `l1` is the manifest's Ethernet core L1.
std::vector<RouterSnapshot> make_router_snapshots(
    const RouterPairing& pairing, const capture::LoadedCapture& capture, const manifest::EthCoreL1& l1);

}  // namespace tt::tt_fabric::decode
