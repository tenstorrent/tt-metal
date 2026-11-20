// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared by the two kernels of the bank-local frame copy (kv_copy_reader.cpp, kv_copy_writer.cpp).
//
// The job: for every (source frame, destination frame) pair of the PLAN, every layer slot of every kind, copy the
// frame's ONE 32-row shard in each DRAM bank from the source tensor to the destination tensor: same chip, same
// bank, another address (fused/kv_store/frames.py: a frame keeps chip, row and bank). Nothing crosses chips.
//
// A copier core serves one bank as lane `lane` of `lanes`. Both of its data-movement processors enumerate the
// same packets in the same order (for_each_packet) and meet in a circular buffer: the reader fills pages from
// DRAM, the writer empties them into DRAM. A shard is cut into `packets` NoC packets (at most 16,384 bytes each);
// packet number n of the bank goes to lane n % lanes.
//
// The plan (uint32, one page of PLAN_WORDS words of a device tensor, so a traced program follows it):
//     [0] pairs   [1] 0   [2 + 2 i] source frame   [3 + 2 i] destination frame
// A frame number is what the tensor's geometry multiplies: the pool frame, or the frame position in a slot-major
// cache (frames.Geometry). A frame must not be a source and a destination in one plan (the lanes interleave).
#pragma once

#include <cstdint>

namespace kv_copy {

constexpr uint32_t PLAN_WORDS = 512;
constexpr uint32_t PLAN_BYTES = PLAN_WORDS * sizeof(uint32_t);
constexpr uint32_t PLAN_HEADER = 2;
constexpr uint32_t MAX_PAIRS = (PLAN_WORDS - PLAN_HEADER) / 2;  // 255 frames = 2,088,960 tokens per launch
constexpr uint32_t MAX_KINDS = 2;

// Measurement switches (compile-time `mode`).
constexpr uint32_t MODE_NO_WRITE = 1;  // pages are read and dropped: the read side alone
constexpr uint32_t MODE_NO_READ = 2;   // pages are written as they lie in L1: the write side alone
constexpr uint32_t MODE_RETURN = 4;    // both kernels return at once: the launch floor

// One kind of storage in one (source, destination) pair of tensors; all strides in bytes inside a bank.
struct Kind {
    uint32_t shard_bytes;  // one layer slot of one frame in one bank
    uint32_t packets;      // NoC packets a shard is cut into
    uint32_t slots;        // layer slots per frame
    uint32_t src_frame;    // source: bytes from one frame to the next, and from one slot to the next
    uint32_t src_slot;
    uint32_t dst_frame;
    uint32_t dst_slot;
    constexpr uint32_t packet_bytes() const { return shard_bytes / packets; }
};

// f(source offset, destination offset) for every packet of `lane`, in the bank's packet order.
template <typename Plan, typename F>
inline __attribute__((always_inline)) void for_each_packet(
    const Kind& kd, Plan pairs, uint32_t count, uint32_t lane, uint32_t lanes, F&& f) {
    const uint32_t pkt = kd.packet_bytes();
    uint32_t turn = 0;
    for (uint32_t j = 0; j < count; ++j) {
        uint32_t s = pairs[2 * j] * kd.src_frame;
        uint32_t d = pairs[2 * j + 1] * kd.dst_frame;
        for (uint32_t l = 0; l < kd.slots; ++l) {
            for (uint32_t h = 0; h < kd.packets; ++h) {
                if (turn == lane) {
                    f(s + h * pkt, d + h * pkt);
                }
                turn = turn + 1 == lanes ? 0 : turn + 1;
            }
            s += kd.src_slot;
            d += kd.dst_slot;
        }
    }
}

}  // namespace kv_copy
