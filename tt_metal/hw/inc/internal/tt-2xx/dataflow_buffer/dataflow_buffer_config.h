// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

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

enum AccessPattern : uint8_t {
    STRIDED,
    ALL,
    UNKNOWN,
};

constexpr uint8_t NUM_DFBS = 32;
// Pack TRISC stores only active logical DFBs in a compact local array to reduce local-memory pressure.
constexpr uint8_t MAX_ACTIVE_DFBS_PACK = 16;

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
constexpr uint8_t MAX_CLIENT_RS          = 4;   // max consumers per remapper slot (4 Tensix or 4 DM clientR IDs)
// Must match TxnDFBDescriptor::tile_counters[18] (32-byte ISR blob slot).
// Worst consumer case: 4 ALL DMs × 4 producer TCs = 16; worst producer case: 6 DM producers × 1 TC.
constexpr uint8_t MAX_TCS_PER_TXN        = 18;
// Alias used by TxnDFBDescriptor / main-line overflow guards. Keep identical to MAX_TCS_PER_TXN.
constexpr uint8_t MAX_TILE_COUNTERS_PER_SIDE = MAX_TCS_PER_TXN;

constexpr uint16_t TENSIX_RISC_OFFSET = 8; // First 8 represent DMs
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
    DFB config region layout (Quasar / tt-2xx):

    [dfb_config_base]:
      dfb_global_header_t (96B) — fixed-size; DM1/DM0 blob offsets stored inside.

    [dfb_config_base + dm1_remapper_blob_offset]:
      DM1 remapper blob — core-wide flat layout, read by DM1:
        [dfb_dm1_remapper_core_header_t(4B) + dfb_dm1_remapper_slot_t × num_slots]

    [dfb_config_base + dm0_isr_blob_offset]:
      DM0 ISR blob — core-wide, read by DM0:
        [dfb_dm0_isr_blob_core_header_t(8B): precomputed producer/consumer txn IE masks]
        [txn_threshold_pool: dfb_dm0_isr_txn_threshold_t, one slot per used txn id (see
                             dm0_isr_txn_slot_index), ascending id order]
        [txn_desc_pool: dfb_dm0_txn_descriptor_image_t, same dense slot order, contiguous after txn_hw_pool]
        ...

    [dfb_config_base + ghdr->hart_blob_offset[h]]:
      Per-hart sequential init blob (one per participating hartid):
        dfb_hart_init_entry_t[num_entries]  — one per DFB this hart participates in
        (4B-padded end)
      Entry count is NOT stored in the blob; device derives it from participation_mask[h].
      hart_blob_offset[h] points at the first init entry (4B-aligned).
      Non-participating harts have a minimal 4-byte {0,0,0,0} blob.

    [dfb_config_base + dfb_signal_region_off]:
      uint8_t  dfb_signal[NUM_DFBS * MAX_PRODUCERS_PER_DFB]  — 192B; producer i of DFB d writes
                                                                       byte 1 to slot [d*MAX_PRODUCERS_PER_DFB+i].
      uint32_t dfb_expected_signal[NUM_DFBS]                        — 128B; host-computed bitmask of which
                                                                       producer bits are active per DFB.
      Producers use a plain volatile store + fence (no AMO). Consumers iterate over bits in
      dfb_expected_signal[d] and poll each producer's byte slot in dfb_signal.

    DM1 reads linearly through only remapper slot data (unchanged).
    DM0 reads linearly through only ISR txn data (unchanged).
    DM2-7 + TRISC each walk their own sequential init blob — no pointer-table indirection.

    Memory (worst case 4Sx4A, 5 riscs, 4 rmp slots, 8 DFBs):
      96 + (4+4*16) + (8+20)*8 + per_hart_blobs(~3.2KB) + signal_region(256B) ≈ 3.7KB
*/

// Everything setup_local_dfb_interfaces() needs from the global header before it can start
// walking, in ONE naturally-aligned 8B record per hart -- so the prologue makes one uncached load
// where it used to make four.
//
// Two of these fields are stored differently from the canonical header above, because the device's
// access pattern differs from the host's:
//  * num_entries, not participation_mask. The device reads the 32-bit mask solely to popcount it;
//    storing the count instead costs 1 byte instead of 4, frees the room that lets
//    signal_region_off join the record, and deletes the cpopw.
//  * signal_region_off is replicated into every hart's record. It is one global value, but reading
//    it from its own header slot is a second uncached round trip; 2 bytes x 12 harts buys it.
struct dfb_hart_desc_t {
    uint8_t num_entries;  // popcount(participation_mask[h])
    uint8_t _rsvd;
    uint16_t blob_start;         // mirrors hart_blob_offset[h]
    uint16_t blob_len;           // bytes, so the device needs no end-start subtract
    uint16_t signal_region_off;  // mirrors dfb_signal_region_off
};
static_assert(sizeof(dfb_hart_desc_t) == 8, "dfb_hart_desc_t must be exactly 8B for the single ld");

// Fixed header at the start of the DFB config region.
//
// Repacked. hart_desc[] now supersedes two arrays the device no longer reads:
//   * participation_mask[12] (48B) -- the device read it only to popcount; hart_desc carries the
//     count. The mask is still needed by the HOST emitter and its validation, so it moved to a
//     host-local array in dataflow_buffer.cpp rather than into device-visible memory.
//   * hart_blob_offset[12] (24B)  -- fully duplicated by hart_desc[h].blob_start.
// The three region offsets were uint32_t; every offset in this region already fits uint16_t
// (hart_blob_offset always was one), so they narrow with no loss of range.
//
// hart_desc sits at offset 0 deliberately: the device prologue's single uncached load becomes
// config_base + hart*8 with no constant to add.
struct dfb_global_header_t {
    dfb_hart_desc_t hart_desc[dfb::NUM_PARTICIPATING_HARTIDS];  // 96B at offset 0

    uint16_t dm1_remapper_blob_offset;  // -> DM1 remapper blob
    uint16_t dm0_isr_blob_offset;       // -> DM0 ISR blob (core header + txn pools)
    uint16_t dfb_signal_region_off;     // -> signal region; still read by DM0 and dm.cc
    uint8_t  num_dfbs;
    uint8_t  dm0_isr_ready;             // cleared by host; set by DM0 when ISR is armed
    uint8_t has_dm0_isr;                // 1 if any DFB uses implicit sync
    uint8_t _rsvd[7];                   // pad 105 -> 112 (multiple of alignof(dfb_hart_desc_t))
};

// DM1/DM0 blobs begin immediately after the header; no prefix tables.
inline uint32_t dfb_config_header_size() { return sizeof(dfb_global_header_t); }

// Number of init/wait entries in a hart blob (= popcount of participation_mask[h]).
inline uint8_t dfb_hart_participation_count(uint32_t participation_mask) {
    return static_cast<uint8_t>(__builtin_popcount(participation_mask));
}

// Flag bits for dfb_hart_init_entry_t::flags
constexpr uint8_t DFB_HART_FLAG_IS_PRODUCER  = (1u << 7);
// DM1 owns this producer's remapper pair; the producer must wait for it before touching its TCs.
constexpr uint8_t DFB_HART_FLAG_REMAPPER_WAIT_DM1 = (1u << 6);
constexpr uint8_t DFB_HART_FLAG_BROADCAST_TC = (1u << 5);
// This Neo's packer programs its own remapper pair (intra-tensix alias). DM1 cannot do it: the
// Tensix-only TC pool is invisible to DM, and one Neo cannot see another Neo's TCs.
// Mutually exclusive with DFB_HART_FLAG_REMAPPER_WAIT_DM1.
constexpr uint8_t DFB_HART_FLAG_REMAPPER_SELF_PROG = (1u << 4);
constexpr uint8_t DFB_HART_FLAG_TRISC_MASK   = 0x0Fu;  // bits 3:0 = tensix_trisc_mask (which TRISC(s) run DFB ops)

// Layout: dfb_blob_tc_pair_t[num_tcs] immediately after the 28B header, followed by
// uint8_t packed_tile_counter[num_tcs] padded to the next 4B boundary.
// This keeps base_addr and limit for the same slot adjacent (8B apart, same cache line).
// Total TC section = num_tcs*9B rounded up to 4B.
struct dfb_blob_tc_pair_t {
    uint32_t base_addr;  // TRISC: tile units (host >> cb_addr_shift); DM: raw byte addresses
    uint32_t limit;      // TRISC: tile units (host >> cb_addr_shift); DM: raw byte addresses
};
static_assert(sizeof(dfb_blob_tc_pair_t) == 8, "dfb_blob_tc_pair_t must be 8B");

// Per-(hart, DFB) init entry in this hart's sequential blob.
// Fixed 28B header, followed by dfb_blob_tc_pair_t[num_tcs] (8B each), then
// uint8_t packed_tile_counter[num_tcs] padded to 4B.
// Total entry size = 28 + ceil9(num_tcs) where ceil9(n) = (n*9 + 3) & ~3.
struct dfb_hart_init_entry_t {
    uint8_t  logical_dfb_id;
    uint8_t  num_tcs;
    uint8_t  flags;                          // DFB_HART_FLAG_* bits above; bits3:0 = tensix_trisc_mask
    uint8_t _reserved0;                      // kept zeroed for 28B layout stability
    uint32_t entry_size;                     // raw bytes; device applies >> cb_addr_shift
    // Host precomputes the ready-to-copy stride_size per hart type:
    //   DM harts:    stride_size_precomp = entry_size_raw * stride_in_entries  (raw bytes)
    //   TRISC harts: stride_size_precomp = (entry_size_raw >> cb_addr_shift) * stride_in_entries  (tile units)
    uint32_t stride_size_precomp;
    uint8_t  stride_size_tiles;              // TRISC: stride in entries; DM: unused (see dm scalar pack below)
    uint8_t  num_txn_ids;                    // TRISC layout byte 13; DM pack uses byte 21 (see below)
    uint8_t  threshold;                      // TRISC layout; DM pack byte 18
    uint8_t  num_entries_per_txn_id;         // TRISC layout; DM pack byte 19
    uint8_t  num_entries_per_txn_id_per_tc;  // TRISC layout byte 16; DM pack byte 20
    uint8_t  producer_signal_bit;            // TRISC layout byte 17; DM pack byte 13 (transport)
    uint8_t  txn_ids[dfb::NUM_TXN_IDS];     // TRISC layout bytes 18-21; DM pack bytes 14-17
    uint8_t  remapper_pair_index;            // TRISC layout byte 22; DM pack remapper at byte 23
    uint8_t intra_shadow_tc_id;              // TRISC byte 23: intra-tensix ClientR shadow TC id, 0xFF if unused.
                                             // Intra-tensix never targets a DM hart, so DM pack p[11]
                                             // reclaims this byte for remapper_pair_index.
    uint16_t num_entries;                    // bytes 24-25; ring entry count (main update_size path)
    uint16_t capacity;  // bytes 26-27; producer: TC capacity; consumer: 0
} __attribute__((packed));
static_assert(sizeof(dfb_hart_init_entry_t) == 28, "dfb_hart_init_entry_t must be 28B");
static_assert(offsetof(dfb_hart_init_entry_t, capacity) == 26, "capacity must occupy former pad bytes 26-27");
static_assert(offsetof(dfb_hart_init_entry_t, num_entries) == 24, "num_entries must stay at bytes 24-25");

// DM only: bytes [12,24) of the init entry header mirror LocalDFBInterface bytes [8,20)
// (num_tcs_to_rr through _tc_align_pad). Host writes this 12B span in DTCM order; device
// unpacks w3–w5 in dfb_unpack_entry_header_dm and stores via dfb_write_dm_iface_scalars_from_hdr.
// Byte 13 carries producer_signal_bit for init decode (device sets tc_idx=0 after scalar write).
constexpr uint32_t DFB_INIT_ENTRY_DM_SCALAR_PACK_BYTE_OFF = 12u;
constexpr uint32_t DFB_INIT_ENTRY_DM_SCALAR_PACK_BYTES = 12u;
constexpr uint32_t DFB_INIT_ENTRY_DM_PRODUCER_SIGNAL_BYTE_OFF = 13u;

inline void dfb_write_dm_scalar_pack_to_blob(
    uint8_t* entry_bytes,
    uint8_t num_tcs,
    uint8_t producer_signal_bit,
    const uint8_t txn_ids[dfb::NUM_TXN_IDS],
    uint8_t threshold,
    uint8_t num_entries_per_txn_id,
    uint8_t num_entries_per_txn_id_per_tc,
    uint8_t num_txn_ids,
    uint8_t broadcast_tc,
    uint8_t remapper_pair_index) {
    uint8_t* const p = entry_bytes + DFB_INIT_ENTRY_DM_SCALAR_PACK_BYTE_OFF;
    p[0] = num_tcs;
    p[1] = producer_signal_bit;
    p[2] = txn_ids[0];
    p[3] = txn_ids[1];
    p[4] = txn_ids[2];
    p[5] = txn_ids[3];
    p[6] = threshold;
    p[7] = num_entries_per_txn_id;
    p[8] = num_entries_per_txn_id_per_tc;
    p[9] = num_txn_ids;
    p[10] = broadcast_tc;
    p[11] = remapper_pair_index;
}

inline uint8_t dfb_read_init_entry_producer_signal_bit(const uint8_t* entry_bytes, bool is_dm_hart) {
    if (is_dm_hart) {
        return entry_bytes[DFB_INIT_ENTRY_DM_PRODUCER_SIGNAL_BYTE_OFF];
    }
    return reinterpret_cast<const dfb_hart_init_entry_t*>(entry_bytes)->producer_signal_bit;
}

// Returns total serialized bytes for one dfb_hart_init_entry_t with num_tcs TC slots.
// num_tcs pairs (8B each) + num_tcs ptc bytes, rounded up to 4B.
// = sizeof(header) + ((num_tcs * 9 + 3) & ~3).
inline constexpr uint32_t dfb_hart_init_entry_byte_size(uint32_t num_tcs) {
    const uint32_t tc_bytes = num_tcs * 9u;
    return static_cast<uint32_t>(sizeof(dfb_hart_init_entry_t)) + ((tc_bytes + 3u) & ~3u);
}

// ---------------------------------------------------------------------------------------------
// DM init via a host-built LocalDFBInterface image.
//
// The host already knows every value that ends up in a DM's LocalDFBInterface, so it emits the
// finished interface bytes and the device copies them. The DM's per-entry work becomes a control
// read plus a word copy: no unpack, no bit extraction, no per-field or per-TC-slot stores, and no
// multiply to find the interface slot.
//
// DM entry layout:
//   [0] u16 iface_byte_off        logical_dfb_id * sizeof(LocalDFBInterface), premultiplied
//   [2] u8  flags
//   [3] u8  sig_slot              precomputed dfb_signal[] index, 0xFF = consumer
//   [4] u16 capacity
//   [6] u8  remapper_pair_index
//   [7] u8  num_tcs
//   [8] LocalDFBInterface image, 24 + 20*num_tcs bytes, copied verbatim
//
// The 8B control prefix holds exactly what cannot be part of a copy: these five drive branches,
// the remapper spin and tile-counter HW register writes rather than landing in the interface.
//
// Cost: the image materializes rd_ptr/wr_ptr/base_addr (the device used to store one loaded value
// three times) plus slot padding, so a 1-TC entry grows 40B -> 52B. That enlarges the config
// region, which DM0's central invalidate covers.
//
// TRISC is unaffected and keeps the 28B header + packed TC tail.

// DM0 invalidates the whole DFB config region once, before releasing the subordinates, instead of
// every worker invalidating its own blob. The single range is a superset of the per-worker ones.
// Lives in this header so the worker (dataflow_buffer_init.h) and DM0 (dm.cc) agree on it.

// DM LocalDFBInterface geometry. dataflow_buffer_init.h static_asserts these against the real
// type; they are mirrored here so this header stays independent of the device interface header.
constexpr uint32_t DFB_DM_IFACE_SCALAR_BYTES = 32u;  // offsetof(LocalDFBInterface, tc_slots)
constexpr uint32_t DFB_DM_IFACE_SLOT_BYTES = 16u;    // sizeof(DFBTCSlot)
constexpr uint32_t DFB_DM_IFACE_SIZE = 128u;         // sizeof(LocalDFBInterface)
constexpr uint32_t DFB_IFACE_IMAGE_CTRL_BYTES = 8u;

// Accessors for the DM image entry's 8B control prefix and the image behind it.
//
// The layout above lived only in a comment, so readers hand-rolled their casts. The config gtest
// read DM entries as a classic dfb_hart_init_entry_t -- correct until DFB_IFACE_IMAGE, and
// silently wrong after: entry_size came back as 822018064 (which is capacity | rmp<<16 |
// stride<<24) and nobody noticed because the test was already red for an unrelated reason.
// Byte-wise so these are safe on an unaligned pointer.
inline uint16_t dfb_dm_image_entry_iface_byte_off(const uint8_t* e) {
    return static_cast<uint16_t>(e[0] | (e[1] << 8));
}
inline uint8_t dfb_dm_image_entry_flags(const uint8_t* e) { return e[2]; }
// Precomputed dfb_signal[] index; DFB_SIG_SLOT_NONE (0xFF) means this hart is not a producer.
inline uint8_t dfb_dm_image_entry_sig_slot(const uint8_t* e) { return e[3]; }
inline uint16_t dfb_dm_image_entry_capacity(const uint8_t* e) { return static_cast<uint16_t>(e[4] | (e[5] << 8)); }
inline uint8_t dfb_dm_image_entry_remapper_pair_index(const uint8_t* e) { return e[6]; }
// Byte 7 is the finished entry stride when DFB_PRECOMP_ENTRY_BYTES is on, else num_tcs.
inline uint8_t dfb_dm_image_entry_byte7(const uint8_t* e) { return e[7]; }
// entry_size is not in the control prefix -- it is the first word of the interface image.
inline uint32_t dfb_dm_image_entry_size(const uint8_t* e) {
    const uint8_t* img = e + DFB_IFACE_IMAGE_CTRL_BYTES;
    return static_cast<uint32_t>(img[0]) | (static_cast<uint32_t>(img[1]) << 8) |
           (static_cast<uint32_t>(img[2]) << 16) | (static_cast<uint32_t>(img[3]) << 24);
}
constexpr uint8_t DFB_SIG_SLOT_NONE = 0xFFu;
// offsetof(LocalDFBInterface, num_tcs_to_rr) -- byte 0 of the third image word.
constexpr uint32_t DFB_DM_IFACE_NUM_TCS_BYTE_OFF = 8u;

// Control byte 7 carries the finished entry stride instead of num_tcs, so the device advances to
// the next entry with an add instead of the multiply-add that sits on the pointer's dependency
// chain. Fits a byte (32 + 20*6 = 152), and num_tcs is recovered from the image's own
// num_tcs_to_rr, which the copy loads anyway -- so the blob does not grow.
// Defaults on: it has no effect unless DFB_IFACE_IMAGE is also on (both the host store and the
// device read are gated on it), and when the image is on it is strictly better -- it removes the
// num_tcs duplication rather than adding a field, so it costs no blob bytes, and measured
// -27 retired instructions and -1..-2 D$ misses.

// A blob TC slot is NOT a full DFBTCSlot image. At init rd_ptr == wr_ptr == base_addr, so storing
// the interface slot verbatim would carry base three times -- 8 redundant bytes per slot and two
// extra loads. The blob stores {base, limit, ptc_word} and the device fans base out to the three
// pointers from one register: 12 B and 3 loads per slot instead of 20 B and 5.
//
// ptc keeps its own word rather than being packed across slots the way the classic tail does.
// That packing is what forces the guarded ptc_w0/ptc_w1 preloads plus a per-slot
// select/variable-shift/mask, measured at +230 retired instructions -- far more than the 3 bytes
// per slot it saves.
constexpr uint32_t DFB_DM_BLOB_SLOT_BYTES = 8u;  // {base, limit}; ptc moved to the scalar head
// Entry = 8 ctrl + 32 scalars + 8*num_tcs -> a multiple of 8 for EVERY num_tcs, so the whole
// blob stays 8B-aligned and GCC can pair lw->ld on the copy.

inline constexpr uint32_t dfb_dm_iface_image_bytes(uint32_t num_tcs) {
    return DFB_DM_IFACE_SCALAR_BYTES + DFB_DM_BLOB_SLOT_BYTES * num_tcs;
}
inline constexpr uint32_t dfb_dm_image_entry_byte_size(uint32_t num_tcs) {
    // Rounded up to 8 so every entry -- and therefore the image inside it -- starts 8B-aligned.
    // Without this the device copy is stuck at lw/sw pairs: a uint32_t* only promises 4B, so GCC
    // cannot merge to ld/sd however the data actually lands.
    const uint32_t raw = DFB_IFACE_IMAGE_CTRL_BYTES + dfb_dm_iface_image_bytes(num_tcs);
    return (raw + 7u) & ~7u;
}

// Entry size for a given hart class. TRISC always keeps the classic layout.
inline constexpr uint32_t dfb_hart_init_entry_byte_size_for(uint32_t num_tcs, [[maybe_unused]] bool is_dm) {
    return is_dm ? dfb_dm_image_entry_byte_size(num_tcs) : dfb_hart_init_entry_byte_size(num_tcs);
}

// dfb_signal[] index for a producer, folded on the host: logical_dfb_id * MAX_PRODUCERS_PER_DFB +
// bit. Max is 31*6+5 = 191 against a 192B signal region, so 0xFF stays free as the consumer
// sentinel and the whole thing fits in one byte.
inline uint8_t dfb_precomp_signal_slot(uint8_t logical_dfb_id, uint8_t producer_signal_bit) {
    if (producer_signal_bit == DFB_SIG_SLOT_NONE) {
        return DFB_SIG_SLOT_NONE;
    }
    return static_cast<uint8_t>(
        logical_dfb_id * static_cast<uint8_t>(dfb::MAX_PRODUCERS_PER_DFB) + producer_signal_bit);
}

inline void dfb_put_u32_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

// Build one DM entry: control prefix + the finished LocalDFBInterface bytes. Field placement
// mirrors, byte for byte, what the device used to store one field at a time.
inline void dfb_write_dm_image_entry(
    uint8_t* entry,
    uint16_t iface_byte_off,
    uint8_t flags,
    uint8_t sig_slot,
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
    const uint32_t* tc_base_addrs,
    const uint32_t* tc_limits,
    const uint8_t* tc_ptcs) {
    for (uint32_t i = 0; i < dfb_dm_image_entry_byte_size(num_tcs); i++) {
        entry[i] = 0u;
    }
    entry[0] = static_cast<uint8_t>(iface_byte_off);
    entry[1] = static_cast<uint8_t>(iface_byte_off >> 8);
    entry[2] = flags;
    entry[3] = sig_slot;
    entry[4] = static_cast<uint8_t>(capacity);
    entry[5] = static_cast<uint8_t>(capacity >> 8);
    entry[6] = remapper_pair_index;
    // Finished stride, so the device does not multiply. Max 40 + 8*6 = 88, fits a byte.
    entry[7] = static_cast<uint8_t>(dfb_dm_image_entry_byte_size(num_tcs));

    uint8_t* img = entry + DFB_IFACE_IMAGE_CTRL_BYTES;
    dfb_put_u32_le(img + 0, entry_size);   // iface.entry_size  (cb_addr_shift == 0 on DM)
    dfb_put_u32_le(img + 4, stride_size);  // iface.stride_size
    img[8] = num_tcs;                      // iface.num_tcs_to_rr
    img[9] = 0u;                           // iface.tc_idx
    img[10] = txn_ids[0];
    img[11] = txn_ids[1];
    img[12] = txn_ids[2];
    img[13] = txn_ids[3];
    img[14] = threshold;
    img[15] = num_entries_per_txn_id;
    img[16] = num_entries_per_txn_id_per_tc;
    img[17] = num_txn_ids;
    img[18] = broadcast_tc;
    img[19] = 0u;  // iface._tc_align_pad
    img[20] = static_cast<uint8_t>(num_entries);
    img[21] = static_cast<uint8_t>(num_entries >> 8);
    // iface.ptc[]: hoisted out of DFBTCSlot, so it is part of the scalar head and rides the
    // straight-line copy instead of costing a per-slot store on the device.
    for (uint32_t t = 0; t < num_tcs; t++) {
        img[22 + t] = tc_ptcs[t];
    }
    // img[22+num_tcs, 32) is iface.ptc[] tail + _pad1; already zero.

    for (uint32_t t = 0; t < num_tcs; t++) {
        uint8_t* s = img + DFB_DM_IFACE_SCALAR_BYTES + DFB_DM_BLOB_SLOT_BYTES * t;
        dfb_put_u32_le(s + 0, tc_base_addrs[t]);  // base -> rd_ptr, wr_ptr and base_addr on device
        dfb_put_u32_le(s + 4, tc_limits[t]);      // limit
    }
}

struct dfb_txn_id_descriptor_t {
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t num_entries_to_process_threshold; // entries each txn ID tracks before posting/acking
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
        uint16_t dm_mask : 8;             // bits 0-7: DM RISC mask
        uint16_t tensix_mask : 4;         // bits 8-11: Neo RISC mask
        uint16_t tensix_trisc_mask : 4;   // bits 12-15: which TRISC(s) on the Neo run DFB ops (see dataflow_buffer.inl)
    } risc_mask_bits;
    // Participant mask (DM/Neo hartids, per_risc layout, popcount): dm_mask | (tensix_mask << 8).
    // tensix_trisc_mask is separate — TRISC-side only, not OR'd into that mask.
    // For DM-to-DM DFBs, producer and consumer would have different set of transaction ids
    dfb_txn_id_descriptor_t producer_txn_descriptor;
    dfb_txn_id_descriptor_t consumer_txn_descriptor;
    uint8_t num_producers;
    uint8_t _pad[3];                  // reserved (was dm0_blob_size; DM0 blob is now a separate global region)
};
static_assert(sizeof(dfb_initializer_t) == 36, "dfb_initializer_t size changed — check field alignment");

// Core-wide header for the DM1 remapper blob.
// Followed by dfb_dm1_remapper_slot_t[num_slots] aggregated in ascending DFB id order.
struct dfb_dm1_remapper_core_header_t {  // 4 bytes
    uint16_t num_slots;
    uint8_t _pad[2];
} __attribute__((packed));

// Core-wide header at dm0_isr_blob_offset (before txn threshold + descriptor pools).
// Host ORs txn ids across all DFBs on this core; DM0 loads once for CMDBUF IE programming.
struct dfb_dm0_isr_blob_core_header_t {
    uint32_t producer_txn_id_mask;
    uint32_t consumer_txn_id_mask;
};

// CMDBUF threshold for one txn id (role/path implied by producer/consumer masks in core_hdr).
// 4B stride so the DM0 ISR blob + following hart blobs stay 4B-aligned.
struct dfb_dm0_isr_txn_threshold_t {
    uint8_t threshold;
    uint8_t _pad[3];
};

// 32-byte image matching TxnDFBDescriptor layout in dataflow_buffer_interface.h.
struct dfb_dm0_txn_descriptor_image_t {
    uint8_t num_counters;
    uint8_t tile_counters[18];
    uint8_t tiles_to_post_or_ack;  // union post/ack share offset in TxnDFBDescriptor
    uint8_t _pad[12];
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
struct dfb_dm1_remapper_slot_t {
    uint8_t  pair_index;   // remapper pair index for this producer
    uint8_t  _pad[3];      // pad to 8 bytes
    uint32_t clientR_val;  // pre-computed ClientR config register value
    uint32_t clientL_val;  // pre-computed ClientL config register value
} __attribute__((packed));
static_assert(sizeof(dfb_dm1_remapper_slot_t) == 12, "dfb_dm1_remapper_slot_t must be 12 bytes");

static_assert(sizeof(dfb_dm0_isr_blob_core_header_t) == 8, "dfb_dm0_isr_blob_core_header_t must be 8 bytes");
static_assert(sizeof(dfb_dm0_txn_descriptor_image_t) == 32, "dfb_dm0_txn_descriptor_image_t must be 32 bytes");
static_assert(sizeof(dfb_dm0_isr_txn_threshold_t) == 4, "dfb_dm0_isr_txn_threshold_t must be 4 bytes");

// Both pools hold one slot per txn id actually in use, in ascending id order, so their size
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

inline uint32_t dm0_isr_txn_hw_pool_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return dm0_isr_txn_slot_count(producer_txn_id_mask, consumer_txn_id_mask) * sizeof(dfb_dm0_isr_txn_threshold_t);
}

inline uint32_t dm0_isr_txn_desc_pool_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return dm0_isr_txn_slot_count(producer_txn_id_mask, consumer_txn_id_mask) * sizeof(dfb_dm0_txn_descriptor_image_t);
}

inline uint32_t dm0_isr_blob_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return sizeof(dfb_dm0_isr_blob_core_header_t) + dm0_isr_txn_hw_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask) +
           dm0_isr_txn_desc_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
}

static_assert(sizeof(dfb_global_header_t) == 112, "dfb_global_header_t size changed — check field alignment");
static_assert(
    offsetof(dfb_global_header_t, hart_desc) == 0,
    "hart_desc[] must be at offset 0 so the prologue address is config_base + hart*8");
static_assert(sizeof(dfb_dm1_remapper_core_header_t) == 4, "dfb_dm1_remapper_core_header_t must be 4 bytes");
static_assert(sizeof(dfb_initializer_t) == 36, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_hart_init_entry_t) == 28, "dfb_hart_init_entry_t must be 28B");

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
//            I=unused J=subpassB_hw
//   DM1_RMP: A=blob_l1_read_sw B=blob_loop_ovhd C=pairs_reg_hw D=enable_remapper_hw
//            E=first_pair_clientR_hw F=first_pair_clientL_hw G=last_pair_hw
//            H=hw_reg_write_cycles I=hw_reg_writes J=pairs_slots_written
//   DM_LOCAL/TRISC: A=merged_sw B=remapper_spin C=tc_hw D=hw_reg_writes E=tc_reset_hw
//                   F=tc_capacity_hw G=pre_loop H=entry_hdr I=tc_slots J=sig_write
// ---------------------------------------------------------------------------
constexpr uint8_t DFB_INIT_TIMING_NUM_SLOTS = 16;
constexpr uint8_t DFB_INIT_TIMING_WORDS_PER_SLOT = 16;
constexpr uint32_t DFB_INIT_TIMING_REGION_BYTES =
    static_cast<uint32_t>(DFB_INIT_TIMING_NUM_SLOTS) * static_cast<uint32_t>(DFB_INIT_TIMING_WORDS_PER_SLOT) *
    sizeof(uint32_t);
// Cached L1 byte offset for host reads (Tensix L1 window is [0, 4 MiB)).
// Device writes use MEM_L1_UNCACHED_BASE + this offset so TL1 is updated without L2 flush.
constexpr uint32_t DFB_INIT_TIMING_L1_BYTE_OFFSET =
    (4u * 1024u * 1024u) - DFB_INIT_TIMING_REGION_BYTES;

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
