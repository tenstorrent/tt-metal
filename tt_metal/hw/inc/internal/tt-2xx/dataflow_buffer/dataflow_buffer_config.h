// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dfb_access.h"

// Transaction-id space is [0, HW_TXN_ID_MAX]. Id 0 is NOC_OVERLAY_TRID_STATIC (untagged).
// Quasar-only pool split:
//   user / kernel : [0, USER_TXN_ID_MAX]
//   DFB / runtime : [DFB_TXN_ID_BASE, HW_TXN_ID_MAX]  (allocated top-down)
constexpr uint8_t HW_TXN_ID_MAX = 31;
constexpr uint8_t USER_TXN_ID_MAX = 7;
constexpr uint8_t DFB_TXN_ID_BASE = USER_TXN_ID_MAX + 1;                       // 8
constexpr uint8_t NUM_DFB_POOL_TXN_IDS = HW_TXN_ID_MAX - DFB_TXN_ID_BASE + 1;  // 24
static_assert(USER_TXN_ID_MAX >= 1);
static_assert(DFB_TXN_ID_BASE > USER_TXN_ID_MAX);

namespace dfb {

constexpr uint8_t NUM_DFBS = 32;
// Pack TRISC stores only active logical DFBs in a compact local array. Its TC state is allocated
// separately from a shared pool, so common one-TC DFBs do not each reserve the six-slot worst case.
constexpr uint8_t MAX_ACTIVE_DFBS_PACK = 24;
constexpr uint8_t MAX_PACK_TC_SLOTS = 32;
static_assert(MAX_PACK_TC_SLOTS >= MAX_ACTIVE_DFBS_PACK);

constexpr uint8_t NUM_TENSIX = 4;
constexpr uint8_t NUM_TILE_COUNTERS_PER_TENSIX = 32;
constexpr uint8_t NUM_TENSIX_TILE_COUNTERS_FOR_DM = 16;
// First TC ID in the default Tensix-only pool (not accessible by DM); used for intra/inter-tensix DFBs.
// Note: The Remapper can be programmed to expose these TCs to DMs.
constexpr uint8_t TC_TENSIX_POOL_START = NUM_TENSIX_TILE_COUNTERS_FOR_DM;  // = 16
// Size of the Tensix-only pool [TC_TENSIX_POOL_START, NUM_TILE_COUNTERS_PER_TENSIX), per Neo. This is a
// counter budget, not a DFB budget: how many TCs a DFB draws depends on its access pattern and endpoint
// count, so only the intra-tensix cost below is fixed.
constexpr uint8_t NUM_TENSIX_ONLY_TILE_COUNTERS = NUM_TILE_COUNTERS_PER_TENSIX - TC_TENSIX_POOL_START;  // = 16
// Intra-tensix HW workaround: a T6 update to a Tensix-only TC aliases into overlay TCs 0-15 unless the
// remapper routes that TC somewhere. Each intra-tensix DFB therefore burns two Tensix-only TCs on its
// Neo — the ClientL TC that packer and unpacker both drive, plus a sacrificial ClientR shadow that
// absorbs the HW update copy. ClientL and ClientR need not be adjacent; remapper maps any pair.
constexpr uint8_t TILE_COUNTERS_PER_INTRA_TENSIX_DFB = 2;

constexpr uint8_t NUM_REMAPPER_PAIRINGS = 64;
// Only pairs [0, 16) can express a 1-to-many (grouped fan-out) mapping; [16, 64) are 1-to-1 only.
// Allocate by fan-out so single-consumer mappings don't consume the scarce fan-out pairs.
// Packer (intra-tensix) and DM1 1-to-1 share [16, 64): before finalizing a core, host reserves an
// exact contiguous block per Neo from pair 63 downward; DM1 allocates from pair 16 upward. This
// keeps packer pairs above DM1's high-watermark clear while leaving unused capacity available.
constexpr uint8_t NUM_REMAPPER_ONE_TO_MANY_PAIRINGS = 16;
constexpr uint8_t REMAPPER_ONE_TO_ONE_PAIR_START = NUM_REMAPPER_ONE_TO_MANY_PAIRINGS;
// Max txn ids assigned to one DFB producer or consumer side
constexpr uint8_t NUM_TXN_IDS = 4;
static_assert(NUM_DFB_POOL_TXN_IDS >= NUM_TXN_IDS);
// Max TCs a single risc RR-walks for one DFB; also the per-DFB producer signal-slot
// stride. Quasar reserves DM0 (ISR) and DM1 (remapper), so at most DM2–7 (=6) can
// participate as producers/consumers — keep these limits identical.
constexpr uint8_t MAX_NUM_TILE_COUNTERS_TO_RR = 6;
constexpr uint8_t MAX_PRODUCERS_PER_DFB = MAX_NUM_TILE_COUNTERS_TO_RR;
// DM0 blob constants
constexpr uint8_t MAX_DM0_REMAPPER_SLOTS = 8;  // max DM producer RISCs
constexpr uint8_t MAX_CLIENT_RS = 4;           // max consumers per remapper slot (4 Tensix or 4 DM clientR IDs)
// Must match TxnDFBDescriptor::tile_counters[18] (32-byte ISR blob slot).
// Worst consumer case: 4 ALL DMs × 4 producer TCs = 16; worst producer case: 6 DM producers × 1 TC.
constexpr uint8_t MAX_TCS_PER_TXN = 18;
// Alias used by TxnDFBDescriptor / main-line overflow guards. Keep identical to MAX_TCS_PER_TXN.
constexpr uint8_t MAX_TILE_COUNTERS_PER_SIDE = MAX_TCS_PER_TXN;

constexpr uint16_t TENSIX_RISC_OFFSET = 8;  // First 8 represent DMs
// Hartids 0-7 = DM0-7, 8-11 = Neo0-3 (TRISC init uses 8 + neo_id).
constexpr uint8_t NUM_PARTICIPATING_HARTIDS = 12;

using PackedTileCounter = uint8_t;  // bits 5-6: tensix_id (2 bits), bits 0-4: counter_id (5 bits)

// PackedTileCounter bit layout constants
constexpr uint8_t PACKED_TC_COUNTER_ID_BITS = 5;  // Number of bits for counter_id
constexpr uint8_t PACKED_TC_COUNTER_ID_MASK =
    (1 << PACKED_TC_COUNTER_ID_BITS) - 1;                                 // 0x1F - mask for 5-bit counter_id (0-31)
constexpr uint8_t PACKED_TC_TENSIX_ID_SHIFT = PACKED_TC_COUNTER_ID_BITS;  // 5 - shift to access tensix_id
constexpr uint8_t PACKED_TC_TENSIX_ID_BITS = 2;                           // Number of bits for tensix_id
constexpr uint8_t PACKED_TC_TENSIX_ID_MASK =
    (1 << PACKED_TC_TENSIX_ID_BITS) - 1;  // 0x03 - mask for 2-bit tensix_id (0-3)

// NOLINTBEGIN(readability-redundant-inline-specifier)
inline __attribute__((always_inline)) constexpr uint8_t get_tensix_id(PackedTileCounter p) {
    return (p >> PACKED_TC_TENSIX_ID_SHIFT) & PACKED_TC_TENSIX_ID_MASK;
}

inline __attribute__((always_inline)) constexpr uint8_t get_counter_id(PackedTileCounter p) {
    return p & PACKED_TC_COUNTER_ID_MASK;
}
// NOLINTEND(readability-redundant-inline-specifier)

}  // namespace dfb

/*
    DFB config region layout (Quasar / tt-2xx). The host writes it snoopable, so the device reads
    it through the cache with no L2 invalidates. Every region below starts on a 64B line.

    [dfb_config_base]:
      dfb_global_header_t (120B): one 8B dfb_hart_desc_t per hart, the region offsets, DM1's slot
      word (dm1_slots), DM0's prologue word (dm0_packed) and the has_dm0_isr / dm0_isr_ready flags.

    [dfb_config_base + dm1_remapper_blob_offset]:
      DM1 remapper blob — core-wide, read by DM1:
        [dfb_dm1_remapper_core_header_t (4B)][pad to 64B][dfb_dm1_remapper_slot_t x num_slots]

    [dfb_config_base + dm0_isr_blob_offset]:
      DM0 ISR blob — core-wide, read by DM0:
        [dfb_dm0_isr_blob_core_header_t (8B)][pad to 64B]
        [txn_desc_pool: one descriptor per used txn id, ascending id order, each slot 8, 16 or 32B
                        (the smallest that fits every descriptor on the core), threshold in its
                        last byte]

    [dfb_config_base + hart_desc[h].blob_start]:
      Per-hart sequential init blob, hart_desc[h].num_entries entries, one per DFB the hart
      touches, padded to a whole number of 64B lines. DM entries are a control prefix plus a
      ready-made LocalDFBInterface image; TRISC entries are a dfb_hart_init_entry_t plus a packed
      TC tail.

    [dfb_config_base + dfb_signal_region_off]:
      uint8_t  dfb_signal[NUM_DFBS * MAX_PRODUCERS_PER_DFB] — 192B; producer hart h writes byte 1
                                                              to dfb_signal[h] once, after its init.
      uint32_t dfb_expected_signal[NUM_DFBS]                — 128B; mask of the harts that produce
                                                              DFB d.
      Consumers poll dfb_signal[h] for every h set in dfb_expected_signal[d].
*/

// Everything setup_local_dfb_interfaces() needs from the global header before it can start
// walking, in ONE naturally-aligned 8B record per hart, so the prologue makes one uncached load.
//  * num_entries, not participation_mask: the device only ever needed the mask's popcount.
//  * signal_region_off is one global value replicated into every hart's record, so reading it is
//    not a second uncached round trip.
struct dfb_hart_desc_t {
    uint8_t num_entries;         // popcount(participation_mask[h])
    uint8_t flags;               // DFB_HART_DESC_FLAG_*
    uint16_t blob_start;         // offset of hart h's init blob from the region base
    uint16_t blob_len;           // bytes
    uint16_t signal_region_off;  // mirrors dfb_signal_region_off
};
static_assert(sizeof(dfb_hart_desc_t) == 8, "dfb_hart_desc_t must be exactly 8B for the single ld");

// hart_desc[h].flags: hart h produces at least one DFB, so it publishes dfb_signal[h] once.
constexpr uint8_t DFB_HART_DESC_FLAG_PRODUCER = 0x01u;

// Fixed header at the start of the DFB config region. hart_desc sits at offset 0 so the device
// prologue's single uncached load is config_base + hart*8. DM1 and DM0 each start with one
// uncached load as well:
//   * dm1_slots holds DM1's slot-array offset and slot count.
//   * dm0_packed holds the DM0 ISR blob offset, the descriptor stride and both txn-id masks (see
//     dfb_dm0_packed_*). It is 0 exactly when has_dm0_isr is 0. has_dm0_isr stays because DM
//     consumers read it.
// The participation masks stay host-side (dataflow_buffer.cpp): the device never reads them.
struct dfb_global_header_t {
    dfb_hart_desc_t hart_desc[dfb::NUM_PARTICIPATING_HARTIDS];  // 96B at offset 0

    uint16_t dm1_remapper_blob_offset;  // -> DM1 remapper blob
    uint16_t dm0_isr_blob_offset;       // -> DM0 ISR blob (core header + descriptor pool)
    uint16_t dfb_signal_region_off;     // -> signal region: per-hart byte slots then dfb_expected_signal[NUM_DFBS]
    uint8_t num_dfbs;
    uint8_t dm0_isr_ready;  // cleared by host; set by DM0 when ISR is armed
    uint8_t has_dm0_isr;    // 1 if any DFB uses implicit sync
    uint8_t _rsvd[3];       // pad 105 -> 108
    uint32_t dm1_slots;     // [15:0] DM1 slot array offset from region base, [31:16] num_slots
    uint64_t dm0_packed;    // DM0 prologue word (dfb_dm0_packed_*); 0 = no DM0 ISR
};

// dm0_packed layout: [15:0] DM0 ISR blob offset (64B aligned; bits [1:0] carry the descriptor
// stride code), [39:16] producer txn-id mask >> DFB_TXN_ID_BASE, [63:40] consumer txn-id mask >>
// DFB_TXN_ID_BASE. DFB txn ids are confined to [DFB_TXN_ID_BASE, HW_TXN_ID_MAX] = [8, 31], so each
// mask fits in 24 bits; the host TT_FATALs on any bit below DFB_TXN_ID_BASE.
static_assert(HW_TXN_ID_MAX - DFB_TXN_ID_BASE + 1 <= 24, "DFB txn-id masks no longer fit dm0_packed");
constexpr uint32_t DFB_DM0_PACKED_MASK_BITS = 24u;
inline uint64_t dfb_dm0_packed_make(uint32_t blob_offset, uint32_t producer_mask, uint32_t consumer_mask) {
    return static_cast<uint64_t>(blob_offset & 0xFFFFu) |
           (static_cast<uint64_t>(producer_mask >> DFB_TXN_ID_BASE) << 16) |
           (static_cast<uint64_t>(consumer_mask >> DFB_TXN_ID_BASE) << (16 + DFB_DM0_PACKED_MASK_BITS));
}
inline uint32_t dfb_dm0_packed_blob_offset(uint64_t w) { return static_cast<uint32_t>(w) & 0xFFC0u; }
inline uint32_t dfb_dm0_packed_producer_mask(uint64_t w) {
    return (static_cast<uint32_t>(w >> 16) & ((1u << DFB_DM0_PACKED_MASK_BITS) - 1u)) << DFB_TXN_ID_BASE;
}
inline uint32_t dfb_dm0_packed_consumer_mask(uint64_t w) {
    return static_cast<uint32_t>(w >> (16 + DFB_DM0_PACKED_MASK_BITS)) << DFB_TXN_ID_BASE;
}

// DM1/DM0 blobs begin immediately after the header; no prefix tables.
inline uint32_t dfb_config_header_size() { return sizeof(dfb_global_header_t); }

// Number of init/wait entries in a hart blob (= popcount of participation_mask[h]).
inline uint8_t dfb_hart_participation_count(uint32_t participation_mask) {
    return static_cast<uint8_t>(__builtin_popcount(participation_mask));
}

// Flag bits for dfb_hart_init_entry_t::flags
constexpr uint8_t DFB_HART_FLAG_IS_PRODUCER = (1u << 7);
// DM1 owns this producer's remapper pair; the producer must wait for it before touching its TCs.
constexpr uint8_t DFB_HART_FLAG_REMAPPER_WAIT_DM1 = (1u << 6);
constexpr uint8_t DFB_HART_FLAG_BROADCAST_TC = (1u << 5);
// This Neo's packer programs its own remapper pair (intra-tensix alias). DM1 cannot do it: the
// Tensix-only TC pool is invisible to DM, and one Neo cannot see another Neo's TCs.
// Mutually exclusive with DFB_HART_FLAG_REMAPPER_WAIT_DM1.
constexpr uint8_t DFB_HART_FLAG_REMAPPER_SELF_PROG = (1u << 4);
constexpr uint8_t DFB_HART_FLAG_TRISC_MASK = 0x0Fu;  // bits 3:0 = tensix_trisc_mask (which TRISC(s) run DFB ops)

// Layout: dfb_blob_tc_pair_t[num_tcs] immediately after the 32B header, followed by
// uint8_t packed_tile_counter[num_tcs] padded to the next 4B boundary.
// This keeps base_addr and limit for the same slot adjacent (8B apart, same cache line).
// Total TC section = num_tcs*9B rounded up to 4B.
struct dfb_blob_tc_pair_t {
    uint32_t base_addr;  // TRISC: tile units (host >> cb_addr_shift); DM: raw byte addresses
    uint32_t limit;      // TRISC: tile units (host >> cb_addr_shift); DM: raw byte addresses
};
static_assert(sizeof(dfb_blob_tc_pair_t) == 8, "dfb_blob_tc_pair_t must be 8B");

// Per-(hart, DFB) init entry in this hart's sequential blob.
// Fixed 32B header, followed by dfb_blob_tc_pair_t[num_tcs] (8B each), then
// uint8_t packed_tile_counter[num_tcs] padded to 4B.
// Total entry size = 32 + ceil9(num_tcs) where ceil9(n) = (n*9 + 3) & ~3.
struct dfb_hart_init_entry_t {
    uint8_t logical_dfb_id;
    uint8_t num_tcs;
    uint8_t flags;        // DFB_HART_FLAG_* bits above; bits3:0 = tensix_trisc_mask
    uint8_t split_tc;     // byte 3: 1 = each transaction is split across all tile counters
                          // (a share of the credits to each)
    uint32_t entry_size;  // raw bytes; device applies >> cb_addr_shift
    // Host precomputes the ready-to-copy stride_size per hart type:
    //   DM harts:    stride_size_precomp = entry_size_raw * stride_in_entries  (raw bytes)
    //   TRISC harts: stride_size_precomp = (entry_size_raw >> cb_addr_shift) * stride_in_entries  (tile units)
    uint32_t stride_size_precomp;
    uint8_t stride_size_tiles;              // byte 12; stride in entries
    uint8_t num_txn_ids;                    // byte 13
    uint8_t threshold;                      // byte 14
    uint8_t num_entries_per_txn_id;         // byte 15
    uint8_t num_entries_per_txn_id_per_tc;  // byte 16
    uint8_t producer_signal_bit;            // byte 17
    uint8_t txn_ids[dfb::NUM_TXN_IDS];      // bytes 18-21
    uint8_t remapper_pair_index;            // byte 22
    uint8_t intra_shadow_tc_id;             // byte 23: intra-tensix ClientR shadow TC id, 0xFF if unused.
    uint16_t num_entries;                   // bytes 24-25; ring entry count (main update_size path)
    uint16_t capacity;                      // bytes 26-27; producer: TC capacity; consumer: 0
    uint16_t block_size;                    // bytes 28-29; how many tiles exist in one block. 1 unless the ring is
                                            // BLOCKED, otherwise block size.
    uint16_t entries_to_jump;               // bytes 30-31; how many tiles the counter's cursor jumps on each
                                            // rotation of tile counters.
} __attribute__((packed));
static_assert(sizeof(dfb_hart_init_entry_t) == 32, "dfb_hart_init_entry_t must be 32B");
static_assert(offsetof(dfb_hart_init_entry_t, capacity) == 26, "capacity must occupy former pad bytes 26-27");
static_assert(offsetof(dfb_hart_init_entry_t, num_entries) == 24, "num_entries must stay at bytes 24-25");

// Returns total serialized bytes for one dfb_hart_init_entry_t with num_tcs TC slots.
// num_tcs pairs (8B each) + num_tcs ptc bytes, rounded up to 4B.
// = sizeof(header) + ((num_tcs * 9 + 3) & ~3).
constexpr uint32_t dfb_hart_init_entry_byte_size(uint32_t num_tcs) {
    const uint32_t tc_bytes = num_tcs * 9u;
    return static_cast<uint32_t>(sizeof(dfb_hart_init_entry_t)) + ((tc_bytes + 3u) & ~3u);
}

// ---------------------------------------------------------------------------------------------
// DM init via a host-built LocalDFBInterface image.
//
// The host already knows every value that ends up in a DM's LocalDFBInterface, so it emits the
// finished interface bytes and the device copies them: per entry, a control read plus a 64-bit
// word copy, with no unpack, no per-field stores and no multiply to find the interface slot.
//
// DM entry layout:
//   [0] u16 iface_byte_off        logical_dfb_id * sizeof(LocalDFBInterface), premultiplied
//   [2] u8  flags
//   [3] u8  _rsvd
//   [4] u16 capacity
//   [6] u8  remapper_pair_index
//   [7] u8  entry_bytes           finished size of this entry, so the device advances with an add
//   [8] LocalDFBInterface image: the 32B scalar head + one 16B DFBTCSlot per TC
//
// The 8B control prefix holds exactly what is not part of the interface: these fields drive
// branches, the remapper wait and the tile-counter register writes. num_tcs is not in the prefix:
// the device reads it from the image's own num_tcs_to_rr.
//
// TRISC keeps the dfb_hart_init_entry_t header + packed TC tail.

// DM LocalDFBInterface geometry. dataflow_buffer_init.h static_asserts these against the real
// type; they are mirrored here so this header stays independent of the device interface header.
constexpr uint32_t DFB_DM_IFACE_SCALAR_BYTES = 32u;  // offsetof(LocalDFBInterface, tc_slots)
constexpr uint32_t DFB_DM_IFACE_SLOT_BYTES = 16u;    // sizeof(DFBTCSlot)
constexpr uint32_t DFB_DM_IFACE_SIZE = 128u;         // sizeof(LocalDFBInterface)
constexpr uint32_t DFB_IFACE_IMAGE_CTRL_BYTES = 8u;
// offsetof(LocalDFBInterface, num_tcs_to_rr)
constexpr uint32_t DFB_DM_IFACE_NUM_TCS_BYTE_OFF = 8u;

// Entry = 8 ctrl + 32 scalars + 16 * num_tcs: a multiple of 8 for every num_tcs, so every image
// starts 8B-aligned and the device copies it with ld/sd. Max 136, so entry_bytes fits its byte.
constexpr uint32_t dfb_dm_image_entry_byte_size(uint32_t num_tcs) {
    return DFB_IFACE_IMAGE_CTRL_BYTES + DFB_DM_IFACE_SCALAR_BYTES + DFB_DM_IFACE_SLOT_BYTES * num_tcs;
}
static_assert(dfb_dm_image_entry_byte_size(dfb::MAX_NUM_TILE_COUNTERS_TO_RR) <= 0xFFu, "entry_bytes must fit a byte");

// Entry size for a given hart class.
constexpr uint32_t dfb_hart_init_entry_byte_size_for(uint32_t num_tcs, bool is_dm) {
    return is_dm ? dfb_dm_image_entry_byte_size(num_tcs) : dfb_hart_init_entry_byte_size(num_tcs);
}

inline void dfb_put_u16_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}
inline void dfb_put_u32_le(uint8_t* p, uint32_t v) {
    dfb_put_u16_le(p, v);
    dfb_put_u16_le(p + 2, v >> 16);
}

// Build one DM entry: control prefix + the finished LocalDFBInterface bytes. Byte offsets mirror
// LocalDFBInterface / DFBTCSlot in dataflow_buffer_interface.h (DM variant); the device
// static_asserts the geometry.
inline void dfb_write_dm_image_entry(
    uint8_t* entry,
    uint16_t iface_byte_off,
    uint8_t flags,
    uint16_t capacity,
    uint8_t remapper_pair_index,
    uint8_t num_tcs,
    uint32_t entry_size,
    uint32_t stride_size,
    const uint8_t txn_ids[dfb::NUM_TXN_IDS],
    uint8_t threshold,
    uint8_t num_entries_per_txn_id,
    uint8_t num_entries_per_txn_id_per_tc,
    uint8_t num_txn_ids,
    uint8_t broadcast_tc,
    uint16_t num_entries,
    uint16_t block_size,
    uint16_t split_tc,
    uint32_t jump_bytes,
    const uint32_t* tc_base_addrs,
    const uint32_t* tc_limits,
    const uint8_t* tc_ptcs) {
    const uint32_t entry_bytes = dfb_dm_image_entry_byte_size(num_tcs);
    for (uint32_t i = 0; i < entry_bytes; i++) {
        entry[i] = 0u;
    }
    dfb_put_u16_le(entry + 0, iface_byte_off);
    entry[2] = flags;
    dfb_put_u16_le(entry + 4, capacity);
    entry[6] = remapper_pair_index;
    entry[7] = static_cast<uint8_t>(entry_bytes);

    uint8_t* img = entry + DFB_IFACE_IMAGE_CTRL_BYTES;
    dfb_put_u32_le(img + 0, entry_size);   // entry_size (cb_addr_shift == 0 on DM)
    dfb_put_u32_le(img + 4, stride_size);  // stride_size
    img[8] = num_tcs;                      // num_tcs_to_rr
    img[9] = 0u;                           // tc_idx
    img[10] = txn_ids[0];
    img[11] = txn_ids[1];
    img[12] = txn_ids[2];
    img[13] = txn_ids[3];
    img[14] = threshold;
    img[15] = num_entries_per_txn_id;
    img[16] = num_entries_per_txn_id_per_tc;
    img[17] = num_txn_ids;
    img[18] = broadcast_tc;
    dfb_put_u16_le(img + 20, num_entries);
    dfb_put_u16_le(img + 22, block_size);
    dfb_put_u16_le(img + 24, split_tc);
    dfb_put_u32_le(img + 28, jump_bytes);  // jump: a DM cursor is a byte address

    for (uint32_t t = 0; t < num_tcs; t++) {
        uint8_t* s = img + DFB_DM_IFACE_SCALAR_BYTES + DFB_DM_IFACE_SLOT_BYTES * t;
        dfb_put_u32_le(s + 0, tc_base_addrs[t]);  // ptr == base_addr at init
        dfb_put_u32_le(s + 4, tc_base_addrs[t]);  // base_addr
        dfb_put_u32_le(s + 8, tc_limits[t]);      // limit
        s[12] = tc_ptcs[t];                       // packed_tile_counter
    }
}

struct dfb_txn_id_descriptor_t {
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t num_entries_to_process_threshold;  // entries each txn ID tracks before posting/acking
    uint8_t num_txn_ids;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
} __attribute__((packed));

// Every field is naturally aligned at its current offset:
// entry_size/stride_in_entries (u32) at 0/4, capacity/num_entries (u16) at 8/10,
// risc_mask_bits (u16 bitfield) at 12, producer/consumer txn descriptors (packed, alignment 1) at 14/22,
// trailing u8 fields at 30-33 (struct size 36 with tail padding).
struct dfb_initializer_t {
    uint32_t entry_size;
    uint32_t stride_in_entries;
    uint16_t capacity;
    uint16_t num_entries;
    struct {
        uint16_t dm_mask : 8;            // bits 0-7: DM RISC mask
        uint16_t tensix_mask : 4;        // bits 8-11: Neo RISC mask
        uint16_t tensix_trisc_mask : 4;  // bits 12-15: which TRISC(s) on the Neo run DFB ops (see dataflow_buffer.inl)
    } risc_mask_bits;
    // Participant mask (DM/Neo hartids, per_risc layout, popcount): dm_mask | (tensix_mask << 8).
    // tensix_trisc_mask is separate — TRISC-side only, not OR'd into that mask.
    // For DM-to-DM DFBs, producer and consumer would have different set of transaction ids
    dfb_txn_id_descriptor_t producer_txn_descriptor;
    dfb_txn_id_descriptor_t consumer_txn_descriptor;
    uint8_t num_producers;
    uint8_t _pad[3];  // reserved (was dm0_blob_size; DM0 blob is now a separate global region)
};
static_assert(sizeof(dfb_initializer_t) == 36, "dfb_initializer_t size changed — check field alignment");

// Core-wide header for the DM1 remapper blob.
// Followed by dfb_dm1_remapper_slot_t[num_slots] aggregated in ascending DFB id order.
struct dfb_dm1_remapper_core_header_t {  // 4 bytes
    uint16_t num_slots;
    uint8_t _pad[2];
} __attribute__((packed));

// Core-wide header at dm0_isr_blob_offset (before the descriptor pool).
// Host ORs txn ids across all DFBs on this core; DM0 loads once for CMDBUF IE programming.
struct dfb_dm0_isr_blob_core_header_t {
    uint32_t producer_txn_id_mask;
    uint32_t consumer_txn_id_mask;
};

// Txn descriptor layout, shared by DM0's init copy and the ISR. Hot fields first: num_counters and
// tiles_to_post_or_ack in bytes 0-1 and counters from byte 2, so the first 8B carry everything for
// <= 6 counters.
//
// The CMDBUF tiles-to-process threshold travels in the descriptor (byte 20 of the full 32B image)
// rather than in a separate threshold pool, so DM0 programs each txn id in the same walk that copies
// its descriptor. The ISR never reads it.
//
// In the config, each descriptor slot is only as wide as the core needs: 8B (<= 5 counters), 16B
// (<= 13) or 32B, the same for every txn id on the core, with the threshold in the slot's last byte
// (dfb_dm0_desc_stride_for / dfb_dm0_desc_threshold_offset). DM0's D$ is blocking, so every pool line
// it reads is a full stall. The destination g_txn_dfb_descriptor keeps its 32B stride; the extra byte
// copied there lands in a counter slot the ISR never reads.

// 32-byte image matching TxnDFBDescriptor layout in dataflow_buffer_interface.h.
struct dfb_dm0_txn_descriptor_image_t {
    uint8_t num_counters;
    uint8_t tiles_to_post_or_ack;  // union post/ack share offset in TxnDFBDescriptor
    uint8_t tile_counters[18];
    uint8_t threshold;  // DM0 init only; lands in TxnDFBDescriptor padding
    uint8_t _pad[11];
};

// One entry per producer RISC that uses the remapper.
// 12 bytes: pair_index + pre-computed clientR/clientL register values.
//
// clientR_val / clientL_val are pre-computed on the host using the same bitfield layout
// as tClientR_Config_Reg_u / tClientL_Config_Reg_u in remapper_common.hpp:
//
//   clientR_val  [31:0]:
//     slot r occupies bits [r*8+7 : r*8]:  id_r[2:0] at bit r*8, cnt_sel_r[4:0] at bit r*8+3
//     for each consumer r: clientR_val |= (id_R & 0x7) << (r*8) | (tc_R & 0x1F) << (r*8 + 3)
//
//   clientL_val  [31:0]:
//     [2:0]  = producer_client_type (id_L)
//     [7:3]  = tc_id (cnt_sel_L)
//     [11:8] = (1 << num_clientRs) - 1  (valid mask)
//     [12]   = 1  (clientl_is_producer, always 1 for DFB producers)
//     [13]   = 1  (clientr_group, always 1 for DFB fan-out)
//     [14]   = 0  (distribute, always 0)
//
// Device side: setup_dfb_remapper() writes clientR_val/clientL_val directly to remapper HW
// registers (no staging through g_remapper_configurator arrays).
//
// aligned(4): `packed` alone gives the struct alignment 1, so the device would read each uint32_t
// field with four byte loads plus shifts, even though the fields sit at 4-byte offsets and the
// array starts on a 64B line. aligned(4) keeps the 12B layout and lets the compiler use word loads.
struct dfb_dm1_remapper_slot_t {
    uint8_t pair_index;    // remapper pair index for this producer
    uint8_t _pad[3];       // pad to 8 bytes
    uint32_t clientR_val;  // pre-computed ClientR config register value
    uint32_t clientL_val;  // pre-computed ClientL config register value
} __attribute__((packed, aligned(4)));
static_assert(sizeof(dfb_dm1_remapper_slot_t) == 12, "dfb_dm1_remapper_slot_t must be 12 bytes");

static_assert(sizeof(dfb_dm0_isr_blob_core_header_t) == 8, "dfb_dm0_isr_blob_core_header_t must be 8 bytes");

// Where a blob's CACHED payload starts, relative to the blob base. The DM1 slot array and the DM0
// descriptor pool are walked through the cached view, but each sits behind a small core header.
// Starting the payload right after the header would start the cached walk 4 or 8 bytes into a line
// (a 192B slot array would then occupy 4 lines instead of 3), so the payload starts on the next
// line. The host emitter and both device readers must use these, not sizeof(the core header).
constexpr uint32_t DFB_DM1_CACHED_PAYLOAD_OFFSET = 64u;
constexpr uint32_t DFB_DM0_CACHED_PAYLOAD_OFFSET = 64u;
static_assert(
    sizeof(dfb_dm1_remapper_core_header_t) <= DFB_DM1_CACHED_PAYLOAD_OFFSET,
    "DM1 remapper core header no longer fits before the cached payload");
static_assert(
    sizeof(dfb_dm0_isr_blob_core_header_t) <= DFB_DM0_CACHED_PAYLOAD_OFFSET,
    "DM0 ISR core header no longer fits before the cached payload");
static_assert(sizeof(dfb_dm0_txn_descriptor_image_t) == 32, "dfb_dm0_txn_descriptor_image_t must be 32 bytes");
// DM0 takes the threshold out of the source slot's byte 20 at the full 32B stride.
static_assert(offsetof(dfb_dm0_txn_descriptor_image_t, threshold) == 20, "threshold must be byte 20 of the image");

// Smallest source stride holding num_counters counters (bytes 2..) plus the threshold in the last byte.
inline uint32_t dfb_dm0_desc_stride_for(uint32_t max_counters) {
    return max_counters <= 5u ? 8u : (max_counters <= 13u ? 16u : 32u);
}
inline uint32_t dfb_dm0_desc_threshold_offset(uint32_t stride) {
    return stride == 32u ? static_cast<uint32_t>(offsetof(dfb_dm0_txn_descriptor_image_t, threshold)) : stride - 1u;
}
// dm0_packed bits [1:0]: 0 = 32 B, 1 = 16 B, 2 = 8 B.
inline uint32_t dfb_dm0_desc_stride_code(uint32_t stride) { return stride == 8u ? 2u : (stride == 16u ? 1u : 0u); }
inline uint32_t dfb_dm0_packed_desc_stride(uint64_t w) { return 32u >> (static_cast<uint32_t>(w) & 3u); }

// The descriptor pool holds one slot per txn id actually in use, in ascending id order, so their size
// depends on how many ids a core uses and not on where those ids sit in [0, HW_TXN_ID_MAX].
// DFB ids are allocated from the top of the pool, so indexing by raw txn id would make even a
// single-DFB core emit (and invalidate) a full 32-slot table.
inline uint32_t dm0_isr_txn_slot_count(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return static_cast<uint32_t>(__builtin_popcount(producer_txn_id_mask | consumer_txn_id_mask));
}

// Dense slot for txn_id: how many used ids precede it. txn_id must be set in all_mask.
inline uint32_t dm0_isr_txn_slot_index(uint32_t all_mask, uint32_t txn_id) {
    return static_cast<uint32_t>(__builtin_popcount(all_mask & ((1u << txn_id) - 1u)));
}

inline uint32_t dm0_isr_txn_desc_pool_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return dm0_isr_txn_slot_count(producer_txn_id_mask, consumer_txn_id_mask) * sizeof(dfb_dm0_txn_descriptor_image_t);
}

inline uint32_t dm0_isr_blob_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return DFB_DM0_CACHED_PAYLOAD_OFFSET + dm0_isr_txn_desc_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
}

static_assert(sizeof(dfb_global_header_t) == 120, "dfb_global_header_t size changed — check field alignment");
static_assert(offsetof(dfb_global_header_t, hart_desc) == 0, "hart_desc[] must be at offset 0");
static_assert(offsetof(dfb_global_header_t, dm1_slots) == 108, "dm1_slots must be 4B aligned at 108");
static_assert(offsetof(dfb_global_header_t, dm0_packed) == 112, "dm0_packed must be 8B aligned at 112");
static_assert(sizeof(dfb_dm1_remapper_core_header_t) == 4, "dfb_dm1_remapper_core_header_t must be 4 bytes");
static_assert(sizeof(dfb_initializer_t) == 36, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_hart_init_entry_t) == 32, "dfb_hart_init_entry_t must be 32B");

namespace dfb {

// ---------------------------------------------------------------------------
// DFB init timing scratch (written by device during setup_*; host reads after benchmarks).
// Layout: 16 fixed slots × 16 uint32 words (64 B each), 1024 B total in cached L1
// (tail of the 4 MiB region so host watcher reads succeed and DFB config is not clobbered).
//
// Slot order:
//   0-7:  DM0-DM7
//   8-15: Neo0 unpack, Neo0 pack, Neo1 unpack, Neo1 pack, Neo2 unpack, Neo2 pack,
//         Neo3 unpack, Neo3 pack
//
// Per-role metrics (A..J = METRIC_A..METRIC_J):
//   DM0_ISR: A=pre_loop_sw B=subpassB_desc C=hw_reg_write_cycles D=subpassB_l1_read
//            E=subpassB_rocc_issue F=first_ie_rmw G=second_ie_rmw H=isr_enable
//            I=hw_reg_writes J=subpassB_hw (D and E are 0: thresholds are written in the descriptor walk)
//   DM1_RMP: A=blob_l1_read_sw B=blob_loop_ovhd C=pairs_reg_hw D=enable_remapper_hw
//            E=first_pair_clientR_hw F=first_pair_clientL_hw G=last_pair_hw
//            H=hw_reg_write_cycles I=hw_reg_writes J=pairs_slots_written
//   DM_LOCAL/TRISC: A=merged_sw B=remapper_spin C=tc_hw D=hw_reg_writes E=tc_reset_hw
//                   F=tc_capacity_hw G=pre_loop H=entry_hdr I=tc_slots J=sig_write
// Only e2e, start, end and the DM1 slot counts are written by a plain timed build; every other metric
// needs DFB_INIT_INNER_TIMING_ENABLED (TT_METAL_MEASURE_DFB_INIT_INNER_TIME=1) and is 0 otherwise.
// ---------------------------------------------------------------------------
constexpr uint8_t DFB_INIT_TIMING_NUM_SLOTS = 16;
constexpr uint8_t DFB_INIT_TIMING_WORDS_PER_SLOT = 16;
constexpr uint32_t DFB_INIT_TIMING_REGION_BYTES = static_cast<uint32_t>(DFB_INIT_TIMING_NUM_SLOTS) *
                                                  static_cast<uint32_t>(DFB_INIT_TIMING_WORDS_PER_SLOT) *
                                                  sizeof(uint32_t);
// Cached L1 byte offset for host reads (Tensix L1 window is [0, 4 MiB)).
// Device writes use MEM_L1_UNCACHED_BASE + this offset so TL1 is updated without L2 flush.
constexpr uint32_t DFB_INIT_TIMING_L1_BYTE_OFFSET = (4u * 1024u * 1024u) - DFB_INIT_TIMING_REGION_BYTES;

constexpr uint32_t DFB_INIT_TIMING_MAGIC = 0xDFB07100u;

enum DfbInitTimingRole : uint8_t {
    DFB_INIT_TIMING_ROLE_DM0_ISR = 0,
    DFB_INIT_TIMING_ROLE_DM1_RMP = 1,
    DFB_INIT_TIMING_ROLE_DM_LOCAL = 2,
    DFB_INIT_TIMING_ROLE_TRISC_LOCAL = 3,
};

enum DfbInitTimingWord : uint8_t {
    DFB_INIT_TIMING_W_MAGIC = 0,
    DFB_INIT_TIMING_W_VALID = 1,
    DFB_INIT_TIMING_W_ROLE = 2,
    DFB_INIT_TIMING_W_E2E = 3,
    DFB_INIT_TIMING_W_METRIC_A = 4,
    DFB_INIT_TIMING_W_METRIC_B = 5,
    DFB_INIT_TIMING_W_METRIC_C = 6,
    DFB_INIT_TIMING_W_METRIC_D = 7,
    DFB_INIT_TIMING_W_METRIC_E = 8,
    DFB_INIT_TIMING_W_METRIC_F = 9,
    DFB_INIT_TIMING_W_START = 10,
    DFB_INIT_TIMING_W_END = 11,
    DFB_INIT_TIMING_W_METRIC_G = 12,
    DFB_INIT_TIMING_W_METRIC_H = 13,
    DFB_INIT_TIMING_W_METRIC_I = 14,
    DFB_INIT_TIMING_W_METRIC_J = 15,
};

}  // namespace dfb

// Return value of setup_local_dfb_interfaces()
// Contiguous packer remapper pairs programmed this launch ([lo, hi); lo==0xFF if none).
// Pack trisc.cc tears this range down after the kernel. DM/unpack ignore the return.
struct DfbPackerRemapperRange {
    uint8_t lo = 0xFFu;
    uint8_t hi = 0;
};
