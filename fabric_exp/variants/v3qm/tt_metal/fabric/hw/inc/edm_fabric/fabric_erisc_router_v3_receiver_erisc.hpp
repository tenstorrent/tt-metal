// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Part of fabric_erisc_router_v3.cpp, which includes it once, at file scope: it uses that file's definitions and is
// not a standalone header.

// ============================================================================
// receiver_erisc: ERISC 1 of a two-ERISC router (receiver channels only)
// ============================================================================
//
// When `runs`, ERISC 1 runs its own kernel, run_kernel: the receiver side of kernel_main's init, its own downstream
// router connections, the main loop below, and teardown. Any build it does not cover runs kernel_main and the generic
// loop.
//
// Per pass, per serviced receiver channel (VC0, then VC1):
//   1. Packets known present: the packets-received stream register is read only when the count from its last read is
//      used up. The receiver still decrements the register once per retired packet, so the register stays the source
//      of truth and the kept count is a lower bound of it: a packet is never seen before it arrived.
//   2. Header: payload size, send type and src_ch_id in one word, then the route byte(s), read at the slot address
//      plus this router's coordinate (decode_action). The slot address is a running pointer over the channel's
//      compile-time ring, not a table.
//   3. Admission: local-only packets need nothing. A forward tests a count of known-free slots per live downstream and
//      re-reads that downstream's local free-slots register only when the count is zero. Each forward still decrements
//      the register, so it stays the source of truth and the count is a lower bound of it.
//   4. Deferred flushed atomic (VC0, 2D): drained or bypassed (can_defer_local_atomic_flush), before any NoC command
//      of this packet.
//   5. Forward: one cursor per downstream connection (running slot address, compile-time command buffers and local
//      stream ids). Single-live E/W and 1D forwards use the stateful NoC API, as in the generic loop.
//   6. Local delivery: plain writes, scatter writes, atomics and fused write + atomics are issued directly (the NoC
//      commands of execute_chip_unicast_to_local_chip_impl's arms); every other send type goes through it, out of
//      line (local_delivery_shared). Local writes use one command buffer whose write state is set once
//      (local_write_init_state, local_write_with_state).
//      1D decides once from the route field: deliver, forward, or forward and deliver.
//   7. Retire: the slot's record (src_ch_id and the command buffers that read the slot), packets-received register -1,
//      next slot.
//   8. Completions: one credit frame per batch of up to four slots, as in the generic loop (send_completions).
//
// Receiver channel 0 always uses the inline compile-time forms below, also on VC1 routers; only the VC1 receiver
// channel goes through the shared out-of-line bodies (VC1 code size). Grouped reads: see "Grouped reads" above.
//
// The main loop (run_main_loop) is a function of its own, so its code does not depend on init, handshake or teardown
// code, and the work it does once per round (termination test, heartbeat, context-switch handshake) is out of line,
// so the function holds only per-packet code (see run_main_loop).
namespace receiver_erisc {

// Whether this ERISC runs this program: ERISC 1 of a two-ERISC Blackhole router that services only receiver channels,
// all with counter credits and forwarding, without the speedy path, deadlock avoidance, first-level ack, UDM,
// src_ch_id-less packets, context switching or per-transaction flush checks, and with local and downstream writes on
// this ERISC's NoC (the completion check reads that NoC's per-trid counters for both).
constexpr bool runs = [] {
#if defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)
    if (!(NUM_ACTIVE_ERISCS == 2 && MY_ERISC_ID == 1)) {
        return false;
    }
    bool ok = !any_sender_channel_serviced && is_receiver_channel_serviced[0] && !super_speedy_mode &&
              multi_txq_enabled && !enable_deadlock_avoidance && !ENABLE_FIRST_LEVEL_ACK_VC0 &&
              !ENABLE_FIRST_LEVEL_ACK_VC1 && !udm_mode && !skip_src_ch_id_update && !enable_context_switch &&
              !enable_trid_flush_check_on_noc_txn && !FORCE_ALL_PATHS_TO_USE_SAME_NOC &&
              tt::tt_fabric::local_chip_noc_equals_downstream_noc && tt::tt_fabric::edm_to_local_chip_noc == noc_index &&
              (is_2d_fabric || std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>);
    for (size_t ch = 0; ch < NUM_RECEIVER_CHANNELS; ch++) {
        if (is_receiver_channel_serviced[ch]) {
            ok = ok && receiver_channel_uses_counter_credits(ch) && !is_receiver_channel_forwarding_disabled[ch];
        }
    }
    return ok;
#else
    return false;
#endif
}();

#if defined(ARCH_BLACKHOLE)
template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t read_stream_count() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) & stream_count_mask;
}

// On an intermesh router with VC0-to-VC1 crossover, receiver channel 0 forwards into the VC1 downstream connections, and
// kernel_main's VC0 adapters credit the downstream's VC1 sender channels too.
constexpr bool vc0_crosses_to_vc1 =
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    true;
#else
    false;
#endif

// Compile-time facts of the downstream router connections of VC `VC`, as kernel_main's adapters (RouterToRouterSender)
// are built for it: count (compact slots), slots per downstream ring, forwarding command buffers, and per compact slot
// the downstream sender channel's free-slots stream (its credit register) and this router's local copy of it.
template <size_t VC>
struct DownstreamConnections {
    static constexpr size_t size = VC == 0 ? VC0_DOWNSTREAM_EDM_SIZE : VC1_DOWNSTREAM_EDM_SIZE;
    static constexpr size_t num_slots = VC == 0 ? DOWNSTREAM_SENDER_NUM_BUFFERS_VC0 : DOWNSTREAM_SENDER_NUM_BUFFERS_VC1;
    static constexpr uint8_t data_cmd_buf = receiver_channel_forwarding_data_cmd_buf_ids[VC];
    static constexpr uint8_t sync_cmd_buf = receiver_channel_forwarding_sync_cmd_buf_ids[VC];

    template <size_t COMPACT>
    static constexpr uint32_t local_credit_stream() {
        if constexpr (VC == 0) {
            return vc0_local_free_slots_stream_id<COMPACT>();
        } else {
            return vc1_local_free_slots_stream_id<COMPACT>();
        }
    }

    static constexpr std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> remote_credit_streams = [] {
        std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> ids{};
        for (uint32_t i = 0; i < MAX_DOWNSTREAM_EDM_COUNT; i++) {
#if defined(FABRIC_2D)
            ids[i] = (VC == 0 && !vc0_crosses_to_vc1) ? get_vc0_downstream_sender_channel_free_slots_stream_id(i)
                                                          : get_vc1_downstream_sender_channel_free_slots_stream_id(i);
#else
            ids[i] = sender_channel_1_free_slots_stream_id;
#endif
        }
        return ids;
    }();
    static constexpr std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> local_credit_streams = [] {
        std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> ids{};
        for (uint32_t i = 0; i < MAX_DOWNSTREAM_EDM_COUNT; i++) {
#if defined(FABRIC_2D)
            ids[i] = VC == 0 ? vc0_local_free_slots_stream_id(i) : vc1_local_free_slots_stream_id(i);
#else
            ids[i] = vc0_local_free_slots_stream_id(0);
#endif
        }
        return ids;
    }();

    static_assert(size > 0 && size <= MAX_DOWNSTREAM_EDM_COUNT);
    static_assert(num_slots <= stream_count_mask);
};

// Compile-time facts of receiver channel CH. It forwards into the downstream connections of its own VC, or of VC1 for
// VC0 with crossover.
template <size_t CH>
struct RxChannel {
    static constexpr size_t ds_vc = (CH == 0 && vc0_crosses_to_vc1) ? 1 : CH;
    using DS = DownstreamConnections<ds_vc>;
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t slot_bytes = channel_buffer_size;
    static constexpr uint32_t first_slot = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t pkts_stream = to_receiver_packets_sent_streams[CH];
    // Completion credits go back in batches of a quarter of the ring, at most 4 (one word of slot records); rings under
    // 8 slots credit every slot on its own. So a channel never holds back more than a quarter of its slots
    // (send_completions).
    static constexpr uint32_t batch_target = num_slots < 8 ? 1 : (num_slots / 4 < 4 ? num_slots / 4 : 4);
    static constexpr size_t ds_size = DS::size;
    static constexpr uint8_t data_cmd_buf = DS::data_cmd_buf;
    static constexpr uint8_t sync_cmd_buf = DS::sync_cmd_buf;
    // Flushed atomics may be deferred past later packets (VC0 only; see can_defer_local_atomic_flush).
    static constexpr bool defers_flushed_atomics = CH == 0 && can_defer_local_atomic_flush<false>();
    // The VC1 receiver channel delivers, admits and forwards through shared out-of-line bodies (code size); receiver
    // channel 0 always uses the inline compile-time forms, so VC0 traffic costs the same with or without VC1.
    static constexpr bool uses_shared_bodies = CH == 1;

    template <size_t COMPACT>
    static constexpr uint32_t local_credit_stream() {
        return DS::template local_credit_stream<COMPACT>();
    }

    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= stream_count_mask);
};

// Per-slot record of a receiver channel, one byte per slot: the sender channel the packet came from (src_ch_id, for the
// completion credit) in bits 0-5, and the command buffers whose NoC commands read the slot's data in bits 6-7:
// SLOT_READ_LOCAL for a local write (local-write command buffer), SLOT_READ_FWD for a forward (forwarding data
// command buffer). Local atomics and inline writes carry their values in command-buffer registers and do not read the
// slot; neither does a packet that is neither delivered nor forwarded. A record is written when its slot is retired
// (after the packet's NoC commands). Records are kept as words, so the four slots of a completion batch are one word
// load (two when the batch straddles a word).
constexpr uint32_t SLOT_SRC_MASK = 0x3F;
constexpr uint32_t SLOT_READ_LOCAL = 0x40;
constexpr uint32_t SLOT_READ_FWD = 0x80;
constexpr uint32_t SLOT_READS = SLOT_READ_LOCAL | SLOT_READ_FWD;
static_assert(MAX_NUM_SENDER_CHANNELS <= SLOT_SRC_MASK + 1);
// Local send types whose NoC commands read the slot: write, fused write + atomic, scatter, sparse multicast.
constexpr uint32_t LOCAL_TYPES_READING_SLOT = (1u << tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE);

static_assert(tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST < 32);

FORCE_INLINE uint32_t local_read_bits(tt::tt_fabric::NocSendType noc_send_type) {
    return ((LOCAL_TYPES_READING_SLOT >> (static_cast<uint32_t>(noc_send_type) & 31u)) & 0x1u) * SLOT_READ_LOCAL;
}

// Slot `base + n` of a ring of N slots (n < N). Unlike wrap_increment_n, also right for N = 2.
template <uint32_t N>
FORCE_INLINE uint8_t slot_add(uint32_t base, uint32_t n) {
    const uint32_t i = base + n;
    if constexpr ((N & (N - 1)) == 0) {
        return static_cast<uint8_t>(i & (N - 1));
    } else {
        return static_cast<uint8_t>(i >= N ? i - N : i);
    }
}

// The records of a ring of N slots: byte k of word w is slot 4w + k (little-endian, so `four_from` can shift).
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
template <uint32_t N>
struct SlotRecords {
    static constexpr uint32_t num_words = (N + 3) / 4;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint8_t idx, uint32_t record) {
        reinterpret_cast<uint8_t*>(words.data())[idx] = static_cast<uint8_t>(record);
    }
    FORCE_INLINE uint32_t get(uint8_t idx) const { return reinterpret_cast<const uint8_t*>(words.data())[idx]; }
    // Records of slots base .. base + 3 (ring order), slot base in the low byte.
    FORCE_INLINE uint32_t four_from(uint32_t base) const {
        if constexpr (N % 4 == 0) {
            const uint32_t w = base >> 2;
            const uint32_t shift = (base & 3) * 8;
            uint32_t four = words[w];
            if (shift != 0) {
                const uint32_t next = w + 1 == num_words ? 0 : w + 1;
                four = (four >> shift) | (words[next] << (32 - shift));
            }
            return four;
        } else {
            uint32_t four = 0;
            uint8_t idx = static_cast<uint8_t>(base);
            for (uint32_t k = 0; k < 4; k++) {
                four |= get(idx) << (8 * k);
                idx = tt::tt_fabric::wrap_increment<N>(idx);
            }
            return four;
        }
    }
};

// Loop-carried state of one receiver channel.
struct RxChannelState {
    uint32_t avail;           // packets known to be in the ring and not yet retired (lower bound of the register)
    uint32_t slot_addr;       // header address of the next slot to retire
    uint32_t uncompleted;     // slots retired (NoC commands issued) but not yet credited back to the far sender
    uint32_t idle_polls;      // consecutive passes without progress while a partial batch waits
    uint32_t pending_atomic;  // deferred flushed atomic (encoding: can_defer_local_atomic_flush); VC0, 2D only
    uint8_t idx;              // buffer index of the next slot to retire
    // STABILITY_PROBE_STATE
};

template <size_t CH>
FORCE_INLINE void init_rx_channel_state(RxChannelState& st) {
    st.avail = 0;
    st.slot_addr = RxChannel<CH>::first_slot;
    st.uncompleted = 0;
    st.idle_polls = 0;
    st.pending_atomic = 0;
    st.idx = 0;
}

// One downstream router connection as this ERISC forwards to it: the next slot in the downstream sender channel's ring,
// the ring bounds, the downstream's NoC coordinate and credit register (stateless sends), the update address of the
// local copy of its free slots (for runtime-indexed slots; its read address is a fixed offset away), and a count of
// free slots known from the last read of that copy. This ERISC sets the connections up itself (init_downstreams,
// open_downstreams, setup_downstream_cmd_bufs) with the same writes, in the same order, as kernel_main's
// RouterToRouterSender adapters: a change to the adapter's constructor, open() or setup_edm_noc_cmd_buf() must be made
// here too.
struct DownstreamCursor {
    uint32_t slot_addr;
    uint32_t first_slot_addr;
    uint32_t last_slot_addr;
    uint32_t free_slots;
    uint32_t noc_coord;
    uint32_t local_credit_update_addr;
    uint64_t credit_noc_addr;
};
// A power of two, so that a runtime index into the downstream connections is a shift (see run_main_loop).
static_assert(sizeof(DownstreamCursor) == 32);

// From a stream's free-slots update register to its free-slots (space available) register.
constexpr uint32_t stream_credit_read_offset = get_stream_reg_read_addr<0>() - get_stream_reg_write_addr<0>();
static_assert(get_stream_reg_read_addr<1>() - get_stream_reg_write_addr<1>() == stream_credit_read_offset);

// Has the downstream a free slot? Re-reads the local copy register only when the known count is used up.
template <uint32_t LOCAL_CREDIT_STREAM>
FORCE_INLINE bool downstream_has_space(DownstreamCursor& c) {
    if (c.free_slots == 0) {
        c.free_slots = read_stream_count<LOCAL_CREDIT_STREAM>();
    }
    return c.free_slots != 0;
}

FORCE_INLINE bool downstream_has_space_rt(DownstreamCursor& c) {
    if (c.free_slots == 0) {
        c.free_slots = *reinterpret_cast<volatile uint32_t*>(c.local_credit_update_addr + stream_credit_read_offset) &
                       stream_count_mask;
    }
    return c.free_slots != 0;
}

// The forward: the whole packet into the downstream's next slot (tagged with the slot's trid), the downstream's
// free-slots register -1 (its doorbell), and the local copy -1. The same NoC commands, in the same order, as
// RouterToRouterSender::send_payload_non_blocking_from_address_with_trid without deadlock avoidance. A stateless
// forward's credit write uses write_at_cmd_buf (noc_inline_dw_write), so it must not bypass a deferred atomic.
template <bool STATEFUL, uint8_t DATA_CMD_BUF, uint8_t SYNC_CMD_BUF>
FORCE_INLINE void forward_payload(DownstreamCursor& c, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;
    const uint32_t dst_addr = c.slot_addr;
    send_chunk_from_address_with_trid<STATEFUL>(src_addr, 1, size_bytes, c.noc_coord, dst_addr, trid, noc, DATA_CMD_BUF);
    c.slot_addr = dst_addr == c.last_slot_addr ? c.first_slot_addr : dst_addr + channel_buffer_size;
    if constexpr (STATEFUL) {
        noc_inline_dw_write_with_state<false, false, true, false, false, InlineWriteDst::REG>(0, 0, SYNC_CMD_BUF, noc);
    } else {
        noc_inline_dw_write<InlineWriteDst::REG>(
            c.credit_noc_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xf, noc);
    }
    c.free_slots--;
}

#if defined(FABRIC_2D_VC1_ACTIVE)
// The VC1 receiver channel (channel 1) delivers, admits and forwards through one out-of-line body each, which walk the
// downstream slots an action uses at run time, so the second receiver channel adds little code (VC1 routers must fit
// both ERISCs in the kernel config buffer). Receiver channel 0 keeps the inline compile-time forms. VC1 forwards are
// stateless.
template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) bool admit_slots_shared(uint32_t slots, DownstreamCursor* ds) {
    bool ok = true;
    for (uint32_t i = 0; slots != 0; i++, slots >>= 1) {
        if (slots & 0x1) {
            ok = downstream_has_space_rt(ds[i]) && ok;
        }
    }
    return ok;
}

template <uint8_t DATA_CMD_BUF>
__attribute__((noinline)) void forward_slots_shared(
    uint32_t slots, DownstreamCursor* ds, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    for (uint32_t i = 0; slots != 0; i++, slots >>= 1) {
        if (slots & 0x1) {
            forward_payload<false, DATA_CMD_BUF, DATA_CMD_BUF>(ds[i], src_addr, size_bytes, trid);
            *reinterpret_cast<volatile uint32_t*>(ds[i].local_credit_update_addr) =
                pack_value_for_inc_on_write_stream_reg_write(-1);
        }
    }
}
#endif

// Local delivery through execute_chip_unicast_to_local_chip_impl, out of line: the VC1 receiver channel's whole local
// delivery (VC1 code size), and the send types local_delivery does not issue itself (inline writes, multicast, sparse
// multicast), which keeps their dispatch out of the main loop. With DEFERS_FLUSHED_ATOMICS it takes and returns the
// channel's deferred-atomic state (can_defer_local_atomic_flush).
template <bool DEFERS_FLUSHED_ATOMICS>
__attribute__((noinline)) uint32_t local_delivery_shared(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint8_t rx_channel_id,
    uint32_t pending_atomic_flush) {
    execute_chip_unicast_to_local_chip_impl(
        packet_start,
        payload_size_bytes,
        noc_send_type,
        transaction_id,
        rx_channel_id,
        DEFERS_FLUSHED_ATOMICS ? &pending_atomic_flush : nullptr);
    return pending_atomic_flush;
}

// Forward to the downstream in compact slot COMPACT of channel CH (compile-time slot; not used with VC1).
template <size_t CH, size_t COMPACT, bool STATEFUL>
FORCE_INLINE void forward_to(DownstreamCursor& c, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    using C = RxChannel<CH>;
    forward_payload<STATEFUL, C::data_cmd_buf, C::sync_cmd_buf>(c, src_addr, size_bytes, trid);
    increment_local_update_ptr_val<C::template local_credit_stream<COMPACT>()>(-1);
}

// Forward to a downstream picked at run time (the intermesh exit of an edge router); always stateless.
template <size_t CH>
FORCE_INLINE void forward_to_rt(
    std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds,
    uint32_t compact_index,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    using C = RxChannel<CH>;
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (C::uses_shared_bodies) {
        forward_slots_shared<C::data_cmd_buf>(1u << compact_index, ds.data(), src_addr, size_bytes, trid);
        return;
    }
#endif
    DownstreamCursor& c = ds[compact_index];
    forward_payload<false, C::data_cmd_buf, C::sync_cmd_buf>(c, src_addr, size_bytes, trid);
    *reinterpret_cast<volatile uint32_t*>(c.local_credit_update_addr) = pack_value_for_inc_on_write_stream_reg_write(-1);
}

// Local writes. Every local write of this program (plain, fused and scatter writes, and the writes of the out-of-line
// deliveries) is a unicast non-posted write on forward_and_local_write_noc_vc from local_chip_data_cmd_buf, and nothing
// else on this ERISC uses that command buffer while the main loop runs: the forwards use their own data and sync
// command buffers, atomics and inline writes use write_at_cmd_buf (checked below). So the main loop sets the command
// buffer's write state once, with local_write_init_state (the NoC API's noc_write_init_state), and each write sets only
// what differs per write, with local_write_with_state: its transaction ID (ncrisc_noc_set_transaction_id), then its
// source, destination and length, and sends (noc_write_with_state). The out-of-line deliveries' writes
// (execute_chip_unicast_to_local_chip_impl, through ncrisc_noc_fast_write) program the same state. The forwarding
// command buffers keep their programmed state across the main loop in the same way (setup_downstream_cmd_bufs). With
// DPRINT or the watcher, whose NoC traffic shares command buffers, every write programs its whole state instead
// (noc_async_write_one_packet_with_trid / ncrisc_noc_fast_write), as the forwards then use the stateless API.
constexpr bool local_write_state_preset =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    true;
#else
    false;
#endif

constexpr bool local_write_cmd_buf_is_exclusive = [] {
    for (size_t ch = 0; ch < NUM_RECEIVER_CHANNELS; ch++) {
        if (receiver_channel_forwarding_data_cmd_buf_ids[ch] == tt::tt_fabric::local_chip_data_cmd_buf ||
            receiver_channel_forwarding_sync_cmd_buf_ids[ch] == tt::tt_fabric::local_chip_data_cmd_buf) {
            return false;
        }
    }
    return tt::tt_fabric::local_chip_data_cmd_buf != write_at_cmd_buf;
}();
static_assert(!runs || local_write_cmd_buf_is_exclusive);

FORCE_INLINE void local_write_init_state() {
    if constexpr (local_write_state_preset) {
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        noc_write_init_state<cmd_buf, CQ_NOC_mkp>(noc, tt::tt_fabric::forward_and_local_write_noc_vc);
    }
}

// One local write of `size` bytes, tagged with `trid`. COUNTED adds it to the software write counters
// (noc_nonposted_writes_num_issued/acked, as noc_async_write_one_packet_with_trid always does on Blackhole). A scatter
// packet's writes are not counted one by one: the packet adds them once, after its last write, since nothing reads the
// counters in between. The router uses DM_DEDICATED_NOC (noc_mode), so these are the counters the API updates.
template <bool COUNTED>
FORCE_INLINE void local_write_with_state(uint32_t src_addr, uint64_t dst_noc_addr, uint32_t size, uint32_t trid) {
    constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size, -1, false, noc);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_addr, size);
    if constexpr (local_write_state_preset) {
        ncrisc_noc_set_transaction_id(noc, cmd_buf, trid);  // waits for the command buffer
        noc_write_with_state<
            noc_mode,
            cmd_buf,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            /*update_counter=*/COUNTED,
            /*posted=*/false>(noc, src_addr, dst_noc_addr, size);
    } else {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_write<noc_mode, /*use_trid=*/true, /*update_counter=*/COUNTED>(
            noc,
            cmd_buf,
            src_addr,
            dst_noc_addr,
            size,
            tt::tt_fabric::forward_and_local_write_noc_vc,
            false,  // mcast
            false,  // linked
            1,      // num_dests
            true,   // multicast_path_reserve
            false,  // posted
            trid);
    }
    WAYPOINT("NWPD");
}

// Local delivery for receiver channel CH. A plain write (the common case), a scatter write, an atomic increment and a
// fused write + atomic increment are issued here, with the same NoC commands as the matching arm of
// execute_chip_unicast_to_local_chip_impl but without its send-type dispatch (sparse test, range test, jump table). A
// scatter write adds its writes to the software write counters once, before its optional semaphore increment, whose
// flush test reads them. Every other send type goes through execute_chip_unicast_to_local_chip_impl. A flushed atomic
// that is deferred is always the packet's last NoC command (can_defer_local_atomic_flush). Adds the slot's read class
// to `record`. The VC1 receiver channel keeps the shared out-of-line body (VC1 code size).
template <size_t CH>
FORCE_INLINE void local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t trid,
    [[maybe_unused]] uint32_t* pending_atomic_flush,
    uint32_t& record) {
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (RxChannel<CH>::uses_shared_bodies) {
        record |= local_read_bits(noc_send_type);
        local_delivery_shared<false>(packet_start, payload_size_bytes, noc_send_type, trid, CH, 0);
        return;
    }
#endif
    using tt::tt_fabric::NocSendType;
    const uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);
    if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
        record |= SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        local_write_with_state</*COUNTED=*/true>(
            payload_start_address, packet_start->command_fields.unicast_write.noc_address, payload_size_bytes, trid);
    } else if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        record |= SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
        const auto& scatter = packet_start->command_fields.unicast_scatter_write;
        const uint8_t chunk_count = scatter.chunk_count;
        uint8_t packet_encoding = scatter.chunk_encoding;
        // Up to three unicast writes, then the last chunk: a unicast write of the rest of the payload, or a semaphore
        // increment (its value carried in the chunk size).
        uint32_t offset = scatter.chunk_size[0];
        uint32_t num_write_chunks = 1;
        local_write_with_state</*COUNTED=*/false>(payload_start_address, scatter.noc_address[0], offset, trid);
        shift_to_next_chunk(packet_encoding);
        if (chunk_count > 2) {
            const uint32_t chunk_size_1 = scatter.chunk_size[1];
            local_write_with_state</*COUNTED=*/false>(
                payload_start_address + offset, scatter.noc_address[1], chunk_size_1, trid);
            offset += chunk_size_1;
            shift_to_next_chunk(packet_encoding);
            num_write_chunks++;
            if (chunk_count == 4) {
                const uint32_t chunk_size_2 = scatter.chunk_size[2];
                local_write_with_state</*COUNTED=*/false>(
                    payload_start_address + offset, scatter.noc_address[2], chunk_size_2, trid);
                offset += chunk_size_2;
                shift_to_next_chunk(packet_encoding);
                num_write_chunks++;
            }
        }
        constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;
        const ChunkEncoding chunk_encoding = static_cast<ChunkEncoding>(packet_encoding & CHUNK_ENCODING_MASK);
        const uint8_t last_chunk_index = chunk_count - 1;
        const uint64_t final_destination_noc_address = scatter.noc_address[last_chunk_index];
        if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            local_write_with_state</*COUNTED=*/false>(
                payload_start_address + offset,
                final_destination_noc_address,
                static_cast<uint16_t>(payload_size_bytes - offset),
                trid);
            num_write_chunks++;
        }
        noc_nonposted_writes_num_issued[noc] += num_write_chunks;
        noc_nonposted_writes_acked[noc] += num_write_chunks;
        if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {
                execute_local_semaphore_inc(
                    final_destination_noc_address,
                    scatter.chunk_size[last_chunk_index],
                    chunk_encoding == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH
                        ? static_cast<uint8_t>(
                              tt::tt_fabric::NocFlushType::FLUSH_STRICT + ((packet_encoding >> 2) & 0x1u))
                        : static_cast<uint8_t>(tt::tt_fabric::NocFlushType::FLUSH_NONE),
                    CH,
                    pending_atomic_flush,
                    trid);
            } else {
                ASSERT(false);
            }
        }
    } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        execute_local_semaphore_inc(
            packet_start->command_fields.unicast_seminc.noc_address,
            packet_start->command_fields.unicast_seminc.val,
            packet_start->command_fields.unicast_seminc.flush,
            CH,
            pending_atomic_flush,
            static_cast<uint8_t>((trid - 1u) & 0xFu));
    } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
        record |= SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        local_write_with_state</*COUNTED=*/true>(
            payload_start_address,
            packet_start->command_fields.unicast_seminc_fused.noc_address,
            payload_size_bytes,
            trid);
        execute_local_semaphore_inc(
            packet_start->command_fields.unicast_seminc_fused.semaphore_noc_address,
            packet_start->command_fields.unicast_seminc_fused.val,
            packet_start->command_fields.unicast_seminc_fused.flush,
            CH,
            pending_atomic_flush,
            trid);
    } else {
        record |= local_read_bits(noc_send_type);
        if constexpr (RxChannel<CH>::defers_flushed_atomics) {
            *pending_atomic_flush = local_delivery_shared<true>(
                packet_start, payload_size_bytes, noc_send_type, trid, CH, *pending_atomic_flush);
        } else {
            local_delivery_shared<false>(packet_start, payload_size_bytes, noc_send_type, trid, CH, 0);
        }
    }
}

#if defined(FABRIC_2D)
// Does the action use compact downstream slot I of channel CH (a LIVE direction whose action bit is set)? Bits outside
// LIVE are ignored, as in admit_2d_dispatch / forward_2d_dispatch.
template <size_t CH, size_t I>
FORCE_INLINE bool action_uses(uint8_t action) {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
        return (action & Routing2DCodec::action_bit(dirs[I])) != 0;
    } else {
        return false;
    }
}

// The action bits of channel CH's LIVE directions: an action forwards (from this router) iff it has one of them.
template <size_t CH>
constexpr uint8_t live_forward_action_bits() {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    uint8_t bits = 0;
    for (size_t i = 0; i < 4; i++) {
        if ((LIVE >> i) & 1) {
            bits |= Routing2DCodec::action_bit(dirs[i]);
        }
    }
    return bits;
}

// Compact downstream slots the action uses (bit i = compact slot i).
template <size_t CH>
FORCE_INLINE uint32_t action_slots(uint8_t action) {
    return (action_uses<CH, 0>(action) ? 0x1u : 0u) | (action_uses<CH, 1>(action) ? 0x2u : 0u) |
           (action_uses<CH, 2>(action) ? 0x4u : 0u) | (action_uses<CH, 3>(action) ? 0x8u : 0u);
}

// Known-free count of compact slot I is used up (only for LIVE slots the action uses).
template <size_t CH, size_t I>
FORCE_INLINE bool needs_credit_reload(uint8_t action, std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds) {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return action_uses<CH, I>(action) && ds[I].free_slots == 0;
    } else {
        return false;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE uint32_t read_live_credit() {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return NOC_STREAM_READ_REG(
            RxChannel<CH>::template local_credit_stream<I>(), STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    } else {
        return 0;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE void set_live_credit(std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds, uint32_t raw) {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        ds[I].free_slots = raw & stream_count_mask;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE bool has_credit_if_used(uint8_t action, std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds) {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return !action_uses<CH, I>(action) || ds[I].free_slots != 0;
    } else {
        return true;
    }
}

// Admission for the forwarding part of a 2D action: every LIVE direction the action uses needs a free slot downstream.
// Local delivery needs nothing (no UDM here). The known-free counts are tested first; only when a needed one is used
// up are the live local copy registers re-read, as one group.
template <size_t CH>
FORCE_INLINE bool admit_2d(uint8_t action, std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds) {
    using C = RxChannel<CH>;
    constexpr uint8_t LIVE = live_eth_mask_2d<C::ds_size>();
    if constexpr (LIVE == 0) {
        return true;
    } else if constexpr (LIVE == 0b0001) {
        if (action_uses<CH, 0>(action)) {
            return downstream_has_space<C::template local_credit_stream<0>()>(ds[0]);
        }
        return true;
    } else {
        if ((action & Routing2DCodec::ACTION_ETH_MASK) == 0) {
            return true;
        }
        const bool reload = needs_credit_reload<CH, 0>(action, ds) || needs_credit_reload<CH, 1>(action, ds) ||
                            needs_credit_reload<CH, 2>(action, ds) || needs_credit_reload<CH, 3>(action, ds);
        if (reload) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const uint32_t raw0 = read_live_credit<CH, 0>();
            const uint32_t raw1 = read_live_credit<CH, 1>();
            const uint32_t raw2 = read_live_credit<CH, 2>();
            const uint32_t raw3 = read_live_credit<CH, 3>();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            set_live_credit<CH, 0>(ds, raw0);
            set_live_credit<CH, 1>(ds, raw1);
            set_live_credit<CH, 2>(ds, raw2);
            set_live_credit<CH, 3>(ds, raw3);
        }
        return has_credit_if_used<CH, 0>(action, ds) && has_credit_if_used<CH, 1>(action, ds) &&
               has_credit_if_used<CH, 2>(action, ds) && has_credit_if_used<CH, 3>(action, ds);
    }
}

template <size_t CH, size_t I>
FORCE_INLINE uint8_t forward_if_used(
    uint8_t action,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid,
    std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds,
    uint32_t& record) {
    constexpr uint8_t LIVE = live_eth_mask_2d<RxChannel<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        if (action_uses<CH, I>(action)) {
            constexpr bool stateful = use_2d_single_live_stateful_noc<RxChannel<CH>::ds_size>();
            forward_to<CH, I, stateful>(ds[I], src_addr, size_bytes, trid);
            record |= SLOT_READ_FWD;
            return static_cast<uint8_t>(1u << I);
        }
    }
    return 0;
}

// One unmodified full-packet copy per LIVE direction the action uses, in compact slot order (local delivery is the
// caller's). Returns the mask of compact slots forwarded to, for channel-trimming capture; adds the forward read class to
// `record` when it forwards.
template <size_t CH>
FORCE_INLINE uint8_t forward_2d(
    uint8_t action,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid,
    std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds,
    uint32_t& record) {
    uint8_t forwarded_slot_mask = forward_if_used<CH, 0>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= forward_if_used<CH, 1>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= forward_if_used<CH, 2>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= forward_if_used<CH, 3>(action, src_addr, size_bytes, trid, ds, record);
    return forwarded_slot_mask;
}
#endif

// Completion credits. A slot's credit goes back to the sender channel its packet came from once every NoC command that
// reads the slot has left the NIU (the far sender may then overwrite it). Credits go back in batches, one Ethernet frame
// per batch: each frame costs the far ERISC 0 a credit return to its own upstream (about 35 cycles, RESULTS §56 and
// §72), so batching saves that on three packets of four on steady streams.
//
// When a batch goes out:
//   - while packets keep arriving: once more than batch_target slots wait, so the newest slot (which may still carry a
//     deferred atomic) is never in it;
//   - on a pass without a packet: a full batch at once, a partial one after credit_release_idle_passes such passes.
// So a channel holds back at most batch_target - 1 slots whose writes have left (a quarter of its ring at most, none on
// rings under 8 slots), and only until its next packet or a few passes without one (V3E1_NOTES §12.20).
//
// Proving a batch's slots free takes one NoC status read in the usual case, not one per slot (reading every slot costs
// E1 about 10 cycles per packet: RESULTS §98, v3p). It rests on two facts:
//   - two command buffers read slots: the local-write buffer (local delivery) and the forwarding data buffer
//     (forwards), and each slot's record says which of them read it (SLOT_READ_LOCAL, SLOT_READ_FWD);
//   - the commands of one command buffer leave the NIU in issue order, so when the newest slot a buffer read has no
//     outgoing write, neither has any older slot on that buffer.
// So a batch needs a status read of its newest slot on each buffer it used: one read, or two (as one group) when the
// batch mixes local writes and forwards; none when no slot was read (atomics, inline writes). A transaction ID shared
// with newer slots only makes the check wait longer. The TX queue is read after the transaction IDs, as in the generic
// loop. A send type that read a slot on a third command buffer would break this.
constexpr uint32_t credit_release_idle_passes = 4;

// A full batch whose newest write is still in the NIU waits for it to leave, then is credited, rather than being left to
// a later pass. Under load the next pass handles the next packet first, so the credit would go a packet (~100 cycles)
// later (V3E1_NOTES §12.16). The wait is short: the newest slot was retired at least a pass earlier, and a 4352 B write
// leaves the NIU ~75 cycles after issue. It is bounded, at ~8 cycles per status read; past the bound the batch is left
// to a later pass as before.
constexpr uint32_t completion_wait_reads = 32;

// A batch's records are its slots' record bytes in one word, the oldest slot in the low byte (SlotRecords::four_from),
// so a test can look at every slot of the batch at once:
//   - in_each_byte(field): the mask of `field` in every slot's byte;
//   - slot k and slot k + 1 agree on a field when byte k of (records XOR records >> 8) has none of its bits;
//   - newest_slot_in(bits): the batch position (0 = oldest) of the newest slot whose byte in `bits` is not zero.
constexpr uint32_t in_each_byte(uint32_t field) { return field * 0x01010101u; }
FORCE_INLINE uint32_t newest_slot_in(uint32_t bits) { return (31 - static_cast<uint32_t>(__builtin_clz(bits))) >> 3; }

// Outgoing writes of transaction ID `trid` on this ERISC's NoC (the completion check's status read).
FORCE_INLINE uint32_t writes_outgoing(uint32_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    volatile uint32_t* const outgoing = reinterpret_cast<volatile uint32_t*>(
        (static_cast<uint32_t>(noc) << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(0)));
    return outgoing[trid];
}

// The transaction ID of the slot at `position` in the batch that starts at slot `base`.
template <uint32_t N, typename WriteTridTracker>
FORCE_INLINE uint32_t batch_slot_trid(const WriteTridTracker& trid_tracker, uint32_t base, uint32_t position) {
    return trid_tracker.get_buffer_slot_trid(tt::tt_fabric::BufferIndex{slot_add<N>(base, position)});
}

template <size_t CH, typename WriteTridTracker, typename CreditSender>
FORCE_INLINE void send_completions(
    RxChannelState& st,
    SlotRecords<RxChannel<CH>::num_slots>& records,
    WriteTridTracker& trid_tracker,
    CreditSender& credit_sender,
    bool progress) {
    constexpr uint32_t T = RxChannel<CH>::batch_target;
    constexpr uint32_t N = RxChannel<CH>::num_slots;
    static_assert(T <= 4);
    const uint32_t uncompleted = st.uncompleted;
    bool batch_ready;
    if (progress) {
        st.idle_polls = 0;
        batch_ready = uncompleted > T;
    } else {
        batch_ready = uncompleted >= T || (uncompleted != 0 && ++st.idle_polls >= credit_release_idle_passes);
    }
    if (!batch_ready) {
        return;
    }
    const uint32_t batch = uncompleted > T ? T : uncompleted;
    // The oldest uncredited slot: `uncompleted` slots behind the next one to retire.
    uint32_t base = static_cast<uint32_t>(st.idx) + N - uncompleted;
    if constexpr ((N & (N - 1)) == 0) {
        base &= N - 1;
    } else {
        base = base >= N ? base - N : base;
    }
    const uint8_t base_idx = static_cast<uint8_t>(base);

    // The batch's records; bytes past the batch cleared.
    uint32_t recs = records.four_from(base);
    if (batch < 4) {
        recs &= (1u << (8 * batch)) - 1;
    }
    // Byte k: where slot k's record differs from slot k + 1's, for the T - 1 pairs of a full batch.
    constexpr uint32_t pairs = (1u << (8 * (T - 1))) - 1;
    const uint32_t changes = recs ^ (recs >> 8);
    const bool full_batch = batch == T;
    const bool one_sender = full_batch && (changes & pairs & in_each_byte(SLOT_SRC_MASK)) == 0;
    const bool reads_alike = full_batch && (changes & pairs & in_each_byte(SLOT_READS)) == 0;

    if (reads_alike) {
        // Every slot reads its data the same way: the newest slot proves them all, or no slot is read.
        if (((recs >> (8 * (T - 1))) & SLOT_READS) != 0) {
            const uint32_t newest_trid = batch_slot_trid<N>(trid_tracker, base, T - 1);
            if (writes_outgoing(newest_trid) != 0) {
                // Wait for the write to leave, then credit (bounded; see completion_wait_reads).
                uint32_t reads = 1;
                while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(
                    tt::tt_fabric::edm_to_local_chip_noc, newest_trid)) {
                    if (++reads == completion_wait_reads) {
                        return;
                    }
                }
            }
        }
    } else {
        // The slots each command buffer read (their record bytes).
        const uint32_t local_reads = recs & in_each_byte(SLOT_READ_LOCAL);
        const uint32_t fwd_reads = recs & in_each_byte(SLOT_READ_FWD);
        if ((local_reads | fwd_reads) != 0) {
            // The newest slot read on any buffer; the other buffer's newest slot, when it is older.
            const uint32_t newest = newest_slot_in(local_reads | fwd_reads);
            const uint32_t newest_trid = batch_slot_trid<N>(trid_tracker, base, newest);
            const uint32_t other_reads = local_reads == 0 || fwd_reads == 0      ? 0u
                                         : newest_slot_in(local_reads) == newest ? fwd_reads
                                                                                 : local_reads;
            if (other_reads == 0 || newest_slot_in(other_reads) == newest) {
                if (writes_outgoing(newest_trid) != 0) {
                    return;
                }
            } else {
                const uint32_t older_trid = batch_slot_trid<N>(trid_tracker, base, newest_slot_in(other_reads));
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const uint32_t newest_out = writes_outgoing(newest_trid);
                const uint32_t older_out = writes_outgoing(older_trid);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                if ((newest_out | older_out) != 0) {
                    return;
                }
            }
        }
    }
    if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        if (internal_::eth_txq_is_busy(receiver_txq_id)) {
            return;
        }
    }

    st.idle_polls = 0;
    // Credits: a full batch from one sender channel (the usual case) is one credit of T to it; any other batch is one
    // credit per slot. The counters end the same either way; the frame is sent once, below.
    if (one_sender) {
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>(recs & SLOT_SRC_MASK), T);
    } else if (T == 4 && full_batch) {
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>(recs & SLOT_SRC_MASK), 1);
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>((recs >> 8) & SLOT_SRC_MASK), 1);
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>((recs >> 16) & SLOT_SRC_MASK), 1);
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>((recs >> 24) & SLOT_SRC_MASK), 1);
    } else {
        for (uint32_t k = 0; k < batch; k++) {
            credit_sender.accumulate_completion_credit(static_cast<uint8_t>((recs >> (8 * k)) & SLOT_SRC_MASK), 1);
        }
    }
    if constexpr (!(WriteTridTracker::BOTH_PARAMS_ARE_EQUAL || WriteTridTracker::BOTH_PARAMS_ARE_POW2)) {
        // Transaction IDs assigned per slot: forget the batch's.
        uint8_t idx = base_idx;
        for (uint32_t k = 0; k < batch; k++) {
            trid_tracker.clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex{idx});
            idx = tt::tt_fabric::wrap_increment<N>(idx);
        }
    }
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    }
    credit_sender.flush_completion_credits();
    st.uncompleted = uncompleted - batch;
}

// Retire the slot just handled: packets-received register -1 (after every NoC command of the packet), next slot.
template <size_t CH>
FORCE_INLINE void retire_slot(RxChannelState& st) {
    using C = RxChannel<CH>;
    st.avail--;
    increment_local_update_ptr_val<C::pkts_stream>(-1);
    st.uncompleted++;
    const uint8_t next_idx = tt::tt_fabric::wrap_increment<C::num_slots>(st.idx);
    st.idx = next_idx;
    st.slot_addr = next_idx == 0 ? C::first_slot : st.slot_addr + C::slot_bytes;
}

#if defined(FABRIC_2D)
// The compact downstream slot of the intermesh exit toward mesh dst_mesh_id (edge routers), or DS_SIZE when that
// boundary direction is not addressable from this router. Out of line: rare work (exit packets of edge routers).
template <size_t DS_SIZE>
__attribute__((noinline)) uint32_t intermesh_egress_slot(
    const tt::tt_fabric::routing_l1_info_t& routing_table, uint16_t dst_mesh_id) {
    const auto boundary_dir =
        static_cast<eth_chan_directions>(routing_table.inter_mesh_direction_table.get_original_direction(dst_mesh_id));
    const bool boundary_dir_is_addressable = boundary_dir < eth_chan_directions::COUNT &&
                                             boundary_dir != static_cast<eth_chan_directions>(my_direction) &&
                                             get_downstream_edm_interface_index(boundary_dir) < DS_SIZE;
    return boundary_dir_is_addressable ? static_cast<uint32_t>(get_downstream_edm_interface_index(boundary_dir))
                                       : static_cast<uint32_t>(DS_SIZE);
}
#endif

#if defined(FABRIC_2D)
// The packet's action byte, as Routing2DCodec decodes it (decode_action<my_direction>, or decode_action_y_first on an
// intermesh ingress channel after its landing intercept): the X byte route_buffer[MESH_Y_SIZE + x] on an E/W-facing
// router; otherwise the Y byte route_buffer[y] when it is not 0, else the X byte. Each byte is indexed in the header
// directly (the slot address plus the router's coordinate, with the byte's fixed offset in the load). The codec reads
// them through one shared volatile pointer to the route buffer, which -Os keeps in a register, adding MESH_Y_SIZE to
// the X coordinate per packet.
// - E/W-facing: one byte decides, read as a header field like the rest of the header this ERISC reads.
// - Y first: both bytes are read volatile and unconditionally, as the codec reads them, so the two loads issue
//   together. A destination reached on the Y leg decides on the X byte (its Y byte is 0); read as plain fields, the X
//   load could be placed behind the Y byte's test, and it would wait for the Y load.
template <size_t CH>
FORCE_INLINE uint8_t decode_action(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_header, uint8_t my_mesh_coord_y, uint8_t my_mesh_coord_x) {
    constexpr auto dir = static_cast<eth_chan_directions>(my_direction);
    if constexpr (
        !receiver_channel_is_intermesh_ingress[CH] &&
        (dir == eth_chan_directions::EAST || dir == eth_chan_directions::WEST)) {
        return packet_header->route_buffer[MESH_Y_SIZE + my_mesh_coord_x];
    } else {
        const volatile PACKET_HEADER_TYPE* const header = packet_header;
        const uint8_t action_y = header->route_buffer[my_mesh_coord_y];
        const uint8_t action_x = header->route_buffer[MESH_Y_SIZE + my_mesh_coord_x];
        return action_y != 0 ? action_y : action_x;
    }
}
#endif

// One pass over receiver channel CH: at most one packet, then the deferred-atomic idle drain and completions.
template <size_t CH, typename WriteTridTracker, typename CreditSender, typename LocalTelemetryT>
FORCE_INLINE bool rx_channel_step(
    RxChannelState& st,
    SlotRecords<RxChannel<CH>::num_slots>& records,
    std::array<DownstreamCursor, RxChannel<CH>::ds_size>& ds,
    WriteTridTracker& trid_tracker,
    CreditSender& credit_sender,
    [[maybe_unused]] const tt::tt_fabric::routing_l1_info_t& routing_table,
    [[maybe_unused]] uint16_t my_mesh_id,
    [[maybe_unused]] uint8_t my_mesh_coord_y,
    [[maybe_unused]] uint8_t my_mesh_coord_x,
    [[maybe_unused]] LocalTelemetryT& local_fabric_telemetry) {
    using C = RxChannel<CH>;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (st.avail == 0) {
        st.avail = read_stream_count<C::pkts_stream>();
    }

    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_forward_timer;
    receiver_forward_timer.set_should_dump(st.avail != 0);
    receiver_forward_timer.open();

    bool progress = false;
    if (st.avail != 0) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(st.slot_addr);
#if defined(FABRIC_2D)
        const auto hdr_w40 = PackedHeaderWord40::load(packet_header);
        uint32_t record = hdr_w40.src_ch_id;  // the slot's record (SlotRecords), stored when it is retired
        // Same decode as run_receiver_channel_step_impl: landing intercept (intermesh ingress) -> action decode ->
        // intermesh-exit intercept (edge routers) -> admission on the action bits.
        uint8_t action;
        bool local_deliver = false;
        bool intermesh_egress = false;
        uint8_t intermesh_egress_index = 0;
        [[maybe_unused]] uint32_t fwd_slots = 0;
        bool admitted;
        if constexpr (receiver_channel_is_intermesh_ingress[CH]) {
            fabric_set_2d_intermesh_landing_route(packet_header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
        }
        action = decode_action<CH>(packet_header, my_mesh_coord_y, my_mesh_coord_x);
        bool intermesh_exit = false;
        if constexpr (is_intramesh_router_on_edge) {
            intermesh_exit =
                Routing2DCodec::action_is_intermesh_exit(action, packet_header->dst_start_mesh_id, my_mesh_id);
        }
        if (intermesh_exit) {
            const uint32_t egress_index =
                intermesh_egress_slot<C::ds_size>(routing_table, packet_header->dst_start_mesh_id);
            if (egress_index >= C::ds_size) {
                ASSERT(false);
                admitted = false;
            } else {
                intermesh_egress = true;
                intermesh_egress_index = static_cast<uint8_t>(egress_index);
#if defined(FABRIC_2D_VC1_ACTIVE)
                if constexpr (C::uses_shared_bodies) {
                    admitted = admit_slots_shared(1u << intermesh_egress_index, ds.data());
                } else
#endif
                {
                    admitted = downstream_has_space_rt(ds[intermesh_egress_index]);
                }
            }
        } else {
            local_deliver = (action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
#if defined(FABRIC_2D_VC1_ACTIVE)
            if constexpr (C::uses_shared_bodies) {
                fwd_slots = action_slots<CH>(action);
                admitted = fwd_slots == 0 || admit_slots_shared(fwd_slots, ds.data());
            } else
#endif
            {
                admitted = (action & live_forward_action_bits<CH>()) == 0 || admit_2d<CH>(action, ds);
            }
        }
#else
        const ROUTING_FIELDS_TYPE cached_routing_fields = packet_header->routing_fields;
        uint32_t record = packet_header->src_ch_id;  // the slot's record (SlotRecords), stored when it is retired
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        const bool admitted = routing == LowLatencyFields::WRITE_ONLY ||
                              downstream_has_space<C::template local_credit_stream<0>()>(ds[0]);
#endif
        if (admitted) {
            progress = true;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
            const uint8_t trid = trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                tt::tt_fabric::BufferIndex{st.idx});
#if defined(FABRIC_2D)
            if constexpr (C::defers_flushed_atomics) {
                // A deferred flushed atomic is pending. 1: strict (waits for every earlier write on the NoC).
                // Relaxed: 0x10 | the trid of its own packet, then 0x100 | (trid_b << 4) | trid_a once the one or two
                // trids it still waits for are known. Resolved here, before this packet issues any NoC command.
                uint32_t& pending_atomic_flush = st.pending_atomic;
                if (pending_atomic_flush) {
                    if (pending_atomic_flush == 1u) {
                        complete_pending_atomic_flush(CH);
                        pending_atomic_flush = 0;
                    } else {
                        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
                        const uint32_t trid_s = pending_atomic_flush & 0xFu;
                        const bool can_bypass =
                            hdr_w40.noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE &&
                            (use_2d_single_live_stateful_noc<C::ds_size>() ||
                             action == Routing2DCodec::ACTION_LOCAL_DELIVER) &&
                            !intermesh_egress && ((trid_s - trid) & 0xFu) > 1u;
                        if (!can_bypass) {
                            drain_pending_atomic_flush(CH, pending_atomic_flush);
                        } else if (pending_atomic_flush < 0x100u) {
                            // First write past the atomic: nothing has been issued since its packet, so every
                            // outstanding write is older. Read global outstanding first, then trid_s and
                            // trid_p = (trid_s - 1) & 0xF. Since counts only fall, g_out == s_out + p_out proves no
                            // write is outstanding on any other trid (regardless of ack order).
                            const uint32_t trid_p = (trid_s - 1u) & 0xFu;
                            const uint32_t acked_target = noc_nonposted_writes_acked[noc];
                            uint32_t g_out;
                            uint32_t s_out;
                            uint32_t p_out;
                            do {
                                const uint32_t ack_rec = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
                                s_out = NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid_s));
                                p_out = NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid_p));
                                std::atomic_signal_fence(std::memory_order_seq_cst);
                                g_out = acked_target - ack_rec;
                            } while (g_out != s_out + p_out);
                            if ((s_out | p_out) == 0) {
                                noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(noc);
                                pending_atomic_flush = 0;
                            } else {
                                const uint32_t wait_a = s_out != 0 ? trid_s : trid_p;
                                const uint32_t wait_b = p_out != 0 ? trid_p : trid_s;
                                pending_atomic_flush = 0x100u | (wait_b << 4) | wait_a;
                            }
                        } else {
                            const uint32_t trid_b = (pending_atomic_flush >> 4) & 0xFu;
                            if (ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_s)) {
                                if (trid_b == trid_s ||
                                    ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_b)) {
                                    noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(noc);
                                    pending_atomic_flush = 0;
                                } else {
                                    // trid_s is done: trid_b is the one left (0x100 | (trid_b << 4) | trid_b).
                                    pending_atomic_flush = (pending_atomic_flush & ~0xFu) | trid_b;
                                }
                            }
                        }
                    }
                }
            }
            const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);
            const uint32_t size_bytes = hdr_w40.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
            if (intermesh_egress) {
                // The exit chip's only output is the INTERMESH egress, sent as-is.
                forward_to_rt<CH>(ds, intermesh_egress_index, src_addr, size_bytes, trid);
                record |= SLOT_READ_FWD;
                record_2d_forwarded_slots<CH>(1u << intermesh_egress_index);
            } else {
                uint32_t forwarded_slot_mask;
#if defined(FABRIC_2D_VC1_ACTIVE)
                if constexpr (C::uses_shared_bodies) {
                    if (fwd_slots != 0) {
                        forward_slots_shared<C::data_cmd_buf>(fwd_slots, ds.data(), src_addr, size_bytes, trid);
                        record |= SLOT_READ_FWD;
                    }
                    forwarded_slot_mask = fwd_slots;
                } else
#endif
                {
                    // One test for a packet that is only delivered here; the forward tests each direction it uses.
                    forwarded_slot_mask = (action & live_forward_action_bits<CH>()) == 0
                                              ? 0u
                                              : forward_2d<CH>(action, src_addr, size_bytes, trid, ds, record);
                }
                record_2d_forwarded_slots<CH>(static_cast<uint8_t>(forwarded_slot_mask));
                if (local_deliver) {
                    local_delivery<CH>(
                        packet_header,
                        hdr_w40.payload_size_bytes,
                        hdr_w40.noc_send_type,
                        trid,
                        C::defers_flushed_atomics ? &st.pending_atomic : nullptr,
                        record);
                }
            }
#else
            constexpr bool stateful_1d =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
                !FORCE_ALL_PATHS_TO_USE_SAME_NOC;
#else
                false;
#endif
            // 1D: WRITE_ONLY delivers locally; FORWARD_ONLY forwards; WRITE_AND_FORWARD forwards and then delivers (a
            // sparse multicast write delivers first: see below); NOOP does nothing. Decided once, from the route field.
            const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);
            const uint16_t payload_size_bytes = packed.payload_size_bytes;
            const tt::tt_fabric::NocSendType noc_send_type = packed.noc_send_type;
            ASSERT(routing != LowLatencyFields::NOOP);
            const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);
            const uint32_t size_bytes = payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
            auto forward = [&]() __attribute__((always_inline)) {
                update_packet_header_for_next_hop(packet_header, cached_routing_fields);
                forward_to<CH, 0, stateful_1d>(ds[0], src_addr, size_bytes, trid);
                record |= SLOT_READ_FWD;
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(CH, 1);
            };
            if (routing == LowLatencyFields::WRITE_ONLY) {
                local_delivery<CH>(packet_header, payload_size_bytes, noc_send_type, trid, nullptr, record);
            } else if (routing == LowLatencyFields::FORWARD_ONLY) {
                forward();
            } else if (routing == LowLatencyFields::WRITE_AND_FORWARD) {
                if (noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
                    // This chip writes its own page group, then write_idx/chip_idx advance so the downstream chip picks
                    // the next group; the forward sends the advanced header.
                    local_delivery_shared<false>(packet_header, payload_size_bytes, noc_send_type, trid, CH, 0);
                    auto& sparse = packet_header->command_fields.sparse_mcast_write;
                    sparse.write_idx += sparse.counts[sparse.chip_idx];
                    sparse.chip_idx++;
                    forward();
                    record |= SLOT_READ_LOCAL;
                } else {
                    forward();
                    local_delivery<CH>(packet_header, payload_size_bytes, noc_send_type, trid, nullptr, record);
                }
            }
#endif
            records.set(st.idx, record);
            retire_slot<CH>(st);
        }
    }

    if constexpr (C::defers_flushed_atomics) {
        if (!progress && st.pending_atomic) {
            drain_pending_atomic_flush(CH, st.pending_atomic);
        }
    }
    receiver_forward_timer.close();

    send_completions<CH>(st, records, trid_tracker, credit_sender, progress);
    return progress;
}

// This ERISC's downstream connections, set up as kernel_main's RouterToRouterSender adapters are for a receiver ERISC,
// from the same runtime args (per VC: has-downstream mask with one bit per compact slot; ring bases, NoC x/y,
// registration and location-info addresses and teardown semaphores by dense index), VC0 first, then VC1:
//   init_downstreams (at adapter construction, before the Ethernet handshake): per connected downstream, zero its
//     teardown semaphore and set the local copy of its free slots to its slot count; fill the cursor (slot i of the
//     downstream ring is base + i * channel_buffer_size; stateless sends go to get_noc_addr(x, y, 0) >>
//     NOC_ADDR_COORD_SHIFT, stateless credits to the downstream's free-slots update register).
//   open_downstreams (after the handshake, VCs whose receiver channel is serviced): connection::open_start, then
//     open_finish, per connected downstream.
//   setup_downstream_cmd_bufs (after the next sync, same VCs): pre-program the data and sync command buffers per
//     connected downstream, then VC0's downstream 0 again when the stateful single-live forward uses it.
// Each runs once, out of line, over a small table of the VCs (one body for all of them).
struct DownstreamSetup {
    DownstreamCursor* cursors;
    uint32_t size;  // compact slots
    uint32_t has_downstream_mask;
    const uint32_t* ring_bases;
    const uint32_t* location_info_addrs;
    const uint32_t* registration_addrs;
    const size_t* teardown_sem_addrs;
    uint32_t noc_x_packed;
    uint32_t noc_y_packed;
    const uint32_t* remote_credit_streams;  // by compact slot
    const uint32_t* local_credit_streams;   // by compact slot
    uint32_t num_slots;
    uint8_t data_cmd_buf;
    uint8_t sync_cmd_buf;
    bool receiver_serviced;
};

template <size_t VC>
FORCE_INLINE DownstreamSetup downstream_setup(
    std::array<DownstreamCursor, DownstreamConnections<VC>::size>& ds,
    uint32_t has_downstream_edm_mask,
    const uint32_t* ring_bases,
    const uint32_t* location_info_addrs,
    const uint32_t* registration_addrs,
    const size_t* teardown_sem_addrs,
    uint32_t noc_x_packed,
    uint32_t noc_y_packed) {
    using D = DownstreamConnections<VC>;
    return DownstreamSetup{
        ds.data(),
        D::size,
        has_downstream_edm_mask & 0xF,
        ring_bases,
        location_info_addrs,
        registration_addrs,
        teardown_sem_addrs,
        noc_x_packed,
        noc_y_packed,
        D::remote_credit_streams.data(),
        D::local_credit_streams.data(),
        D::num_slots,
        D::data_cmd_buf,
        D::sync_cmd_buf,
        is_receiver_channel_serviced[VC]};
}

template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) void init_downstreams(const DownstreamSetup* vcs, uint32_t num_vcs) {
    for (uint32_t v = 0; v < num_vcs; v++) {
        const DownstreamSetup& vc = vcs[v];
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            DownstreamCursor& c = vc.cursors[compact_index];
            c.free_slots = 0;
            c.local_credit_update_addr = get_stream_reg_write_addr(vc.local_credit_streams[compact_index]);
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                *reinterpret_cast<volatile uint32_t*>(vc.teardown_sem_addrs[dense_index]) = 0;
                init_ptr_val(vc.local_credit_streams[compact_index], vc.num_slots);
                const uint32_t ring_base = vc.ring_bases[dense_index];
                const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
                const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
                c.slot_addr = ring_base;
                c.first_slot_addr = ring_base;
                c.last_slot_addr = ring_base + (vc.num_slots - 1) * channel_buffer_size;
                c.noc_coord = static_cast<uint32_t>(get_noc_addr(noc_x, noc_y, 0) >> NOC_ADDR_COORD_SHIFT);
                c.credit_noc_addr = get_noc_addr(
                    noc_x,
                    noc_y,
                    get_stream_reg_write_addr(vc.remote_credit_streams[compact_index]),
                    tt::tt_fabric::edm_to_downstream_noc);
                dense_index++;
            } else {
                c.slot_addr = 0;
                c.first_slot_addr = 0;
                c.last_slot_addr = 0;
                c.noc_coord = 0;
                c.credit_noc_addr = 0;
            }
        }
    }
}

template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) void open_downstreams(const DownstreamSetup* vcs, uint32_t num_vcs) {
    // Blackhole opens router-to-router connections with non-posted writes (kernel_main's
    // use_posted_writes_for_connection_open).
    constexpr bool posted = false;
    for (uint32_t v = 0; v < num_vcs; v++) {
        const DownstreamSetup& vc = vcs[v];
        if (!vc.receiver_serviced) {
            continue;
        }
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
                const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
                tt::tt_fabric::connection::open_start<false, posted, tt::tt_fabric::worker_handshake_noc>(
                    vc.location_info_addrs[dense_index],
                    vc.cursors[compact_index].local_credit_update_addr,
                    vc.teardown_sem_addrs[dense_index],
                    noc_x,
                    noc_y);
                tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
                    vc.registration_addrs[dense_index],
                    reinterpret_cast<volatile uint32_t*>(vc.teardown_sem_addrs[dense_index]),
                    noc_x,
                    noc_y);
                dense_index++;
            }
        }
    }
}

template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) void setup_downstream_cmd_bufs(
    const DownstreamSetup* vcs, uint32_t num_vcs, bool vc0_downstream0_again) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;
    constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
    auto setup = [](const DownstreamSetup& vc, uint32_t compact_index, uint32_t dense_index) {
        const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
        const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
        const uint64_t edm_noc_addr = get_noc_addr(noc_x, noc_y, 0, noc);
        noc_async_write_one_packet_with_trid_set_state<true>(edm_noc_addr, vc.data_cmd_buf, noc, noc_vc);
        const uint64_t noc_sem_addr =
            get_noc_addr(noc_x, noc_y, get_stream_reg_write_addr(vc.remote_credit_streams[compact_index]), noc);
        noc_inline_dw_write_set_state<true, true>(
            noc_sem_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xF, vc.sync_cmd_buf, noc, noc_vc);
    };
    for (uint32_t v = 0; v < num_vcs; v++) {
        const DownstreamSetup& vc = vcs[v];
        if (!vc.receiver_serviced) {
            continue;
        }
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                setup(vc, compact_index, dense_index);
                dense_index++;
            }
        }
        if (v == 0 && vc0_downstream0_again && (vc.has_downstream_mask & 0x1)) {
            setup(vc, 0, 0);
        }
    }
}

// The main loop's once-per-round work that is not about packets, out of line (see run_main_loop): the termination
// test (the termination signal, or the launch mailbox's exit flag), and at the end of a round the heartbeat and the
// context-switch handshake (this ERISC is never its master).
__attribute__((noinline)) bool termination_requested() {
    return got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr));
}

__attribute__((noinline)) void end_round(uint16_t& fabric_heartbeat_counter) {
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);
    if ((++fabric_heartbeat_counter & 0x3F) == 0) {
        *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
    }
    run_routing_without_noc_sync_coordinated_as_non_master(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr));
}

// The transaction ID tracker of receiver channel CH.
template <size_t CH>
using RxTridTracker = WriteTransactionIdTracker<
    RECEIVER_NUM_BUFFERS_ARRAY[CH],
    NUM_TRANSACTION_IDS,
    RX_CH_TRID_STARTS[CH],
    edm_to_local_chip_noc,
    edm_to_downstream_noc>;

// The main loop: every iterations_between_ctx_switch_and_teardown_checks passes, check termination, drain a deferred
// atomic, update telemetry and the heartbeat, and follow the context-switch handshake (this ERISC is never its
// master).
//
// A function of its own, like sender_erisc::run_main_loop: its code depends only on this body, not on run_kernel's
// init, handshake and teardown code or on the firmware code the kernel is inlined into. Its only inputs are the
// downstream connections init sets up at run time; everything else is compile-time or its own state (channel state,
// slot records, transaction ID trackers, credit senders). Work done once per round or rarely (termination test,
// heartbeat, context-switch handshake, intermesh exit lookup, uncommon send types) is in small functions of its own, so
// this function holds only per-packet code: with that work inlined here, cold edits elsewhere re-laid out the whole
// loop (RESULTS §97).
template <
    typename DownstreamsVC0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    typename DownstreamsVC1
#endif
    >
__attribute__((noinline)) void run_main_loop(
    [[maybe_unused]] DownstreamsVC0& ds_vc0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    [[maybe_unused]] DownstreamsVC1& ds_vc1
#endif
) {
    // Telemetry's TX heartbeat reads the sender channels' free-slot registers (fabric-flat table).
    [[maybe_unused]] const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids =
        sender_channel_free_slots_stream_ids;
    RxTridTracker<0> trid_tracker_ch0;
    trid_tracker_ch0.init();
#if defined(FABRIC_2D_VC1_ACTIVE)
    RxTridTracker<1> trid_tracker_ch1;
    trid_tracker_ch1.init();
#endif
    FabricTelemetry local_fabric_telemetry{};
    [[maybe_unused]] auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetry*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);

    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;
    const uint16_t my_mesh_id = routing_table.my_mesh_id;

    auto credit_senders = init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();

#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    auto& ds_ch0 = ds_vc1;
#else
    auto& ds_ch0 = ds_vc0;
#endif
#if defined(FABRIC_2D_VC1_SERVICED)
    auto& ds_ch1 = ds_vc1;
#endif
    RxChannelState st_ch0;
    init_rx_channel_state<0>(st_ch0);
    SlotRecords<RxChannel<0>::num_slots> records_ch0{};
#if defined(FABRIC_2D_VC1_SERVICED)
    RxChannelState st_ch1;
    init_rx_channel_state<1>(st_ch1);
    SlotRecords<RxChannel<1>::num_slots> records_ch1{};
#endif

    uint16_t fabric_heartbeat_counter = 0;

    local_write_init_state();
    // STABILITY_PROBE_PRELOOP
    while (!termination_requested()) {
        uint32_t rx_progress = 0;
        uint64_t loop_start_cycles;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            loop_start_cycles = get_timestamp();
        }
        for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
            // STABILITY_PROBE_LOOP
            rx_progress |= rx_channel_step<0>(
                st_ch0,
                records_ch0,
                ds_ch0,
                trid_tracker_ch0,
                credit_senders.template get<0>(),
                routing_table,
                my_mesh_id,
                my_mesh_coord_y,
                my_mesh_coord_x,
                local_fabric_telemetry);
#if defined(FABRIC_2D_VC1_SERVICED)
            if constexpr (is_receiver_channel_serviced[1]) {
                rx_progress |= rx_channel_step<1>(
                    st_ch1,
                    records_ch1,
                    ds_ch1,
                    trid_tracker_ch1,
                    credit_senders.template get<1>(),
                    routing_table,
                    my_mesh_id,
                    my_mesh_coord_y,
                    my_mesh_coord_x,
                    local_fabric_telemetry);
            }
#endif
        }
        if constexpr (RxChannel<0>::defers_flushed_atomics) {
            if (st_ch0.pending_atomic) {
                drain_pending_atomic_flush(0, st_ch0.pending_atomic);
            }
        }

        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            uint64_t loop_end_cycles = get_timestamp();
            uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
            update_bw_cycles(loop_delta_cycles, /*tx_progress=*/false, rx_progress, local_fabric_telemetry);
        }
        if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
            update_telemetry(
                local_sender_channel_free_slots_stream_ids,
                /*tx_progress=*/false,
                rx_progress,
                local_fabric_telemetry,
                fabric_telemetry);
        }

        end_round(fabric_heartbeat_counter);
    }
}
#endif  // ARCH_BLACKHOLE

}  // namespace receiver_erisc
