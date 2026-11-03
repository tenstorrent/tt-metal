// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

constexpr uint8_t SENDER_TXQ = sender_txq_id;
// The NoC of the credits this router writes to its producers (workers and upstream routers).
constexpr uint8_t WORKER_CREDIT_NOC = tt::tt_fabric::worker_handshake_noc;

// Injection channels on a deadlock-avoiding ring or torus send only with two free far slots (bubble flow control): the
// worker channel, and on a torus a channel fed from the other axis (a turn). (It reads the untrimmed array: GCC rejects
// the trimmed std::array<bool, N>'s operator[] in this constant expression as "used before its definition".)
constexpr uint32_t remote_free_slots_to_send_by(size_t ch) {
    return (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel_all_[ch])
               ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
               : 1;
}

// The host's layout of the sender channels: VC0's, then VC1's, then VC2's (ct args).
constexpr size_t first_sender_channel_of(size_t vc) {
    return vc == 0 ? 0 : vc == 1 ? VC1_LOCAL_CHANNEL_START : VC2_LOCAL_CHANNEL_START;
}
constexpr size_t sender_channel_count_of(size_t vc) {
    return vc == 0 ? ACTUAL_VC0_SENDER_CHANNELS : vc == 1 ? ACTUAL_VC1_SENDER_CHANNELS : ACTUAL_VC2_SENDER_CHANNELS;
}
constexpr size_t vc_of_sender_channel(size_t ch) {
    return ch < VC1_LOCAL_CHANNEL_START ? 0 : ch < VC2_LOCAL_CHANNEL_START ? 1 : 2;
}

// A worker channel: the first sender channel of a VC whose routing mode has a worker (VC0's; VC2's, neighbour exchange;
// VC1 is router to router), as the host marks it (erisc_datamover_builder.cpp). For every channel position, serviced or
// not; the others are forwarding channels, which the routers on this chip that face other directions write into.
constexpr bool is_worker_channel(size_t ch) {
    return vc_of_sender_channel(ch) != 1 && ch == first_sender_channel_of(vc_of_sender_channel(ch));
}

// The serviced sender channels among the `count` channels from `first`, one bit each.
constexpr uint32_t channels_in(size_t first, size_t count) {
    uint32_t mask = 0;
    for (size_t ch = first; ch < first + count && ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= is_sender_channel_serviced[ch] ? (1u << ch) : 0u;
    }
    return mask;
}

// Sender channel CH.
template <size_t CH>
struct SenderChannel {
    static constexpr size_t vc = vc_of_sender_channel(CH);
    static constexpr bool worker = is_worker_channel(CH);
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far router's receiver writes over the link.
    static constexpr uint32_t completion_counter_addr = to_sender_remote_completion_counters_base_address + CH * 4;
    static constexpr uint32_t first_level_ack_counter_addr = to_sender_remote_ack_counters_base_address + CH * 4;
    // A worker channel's connection semaphore (compile time; the worker interface holds the same address).
    static constexpr uint32_t connection_semaphore_addr = sender_connection_semaphore_addrs[CH];
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[vc];
    static constexpr uint32_t remote_free_slots_to_send = remote_free_slots_to_send_by(CH);
    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
};

// VC's sender features: its serviced sender channels, its worker channel (where serviced) and its forwarding channels,
// whether its packets are first-level acked, and where its free-slots registers are read: in the pass's first read
// group when the VC has a worker channel position, whose connection semaphore that group reads anyway; right before
// its own visits otherwise, so a packet landing during the earlier VCs' visits does not wait a pass
// (V4_SENDER_SHAPE.md).
template <size_t VC>
struct SenderVc {
    static constexpr uint32_t sender_channels = channels_in(first_sender_channel_of(VC), sender_channel_count_of(VC));
    static constexpr uint32_t worker_channels =
        is_worker_channel(first_sender_channel_of(VC)) ? sender_channels & (1u << first_sender_channel_of(VC)) : 0u;
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[VC];
    static constexpr bool reads_free_slots_at_pass_start = is_worker_channel(first_sender_channel_of(VC));
};

// Every VC's serviced channels, by kind; the VCs this router sends on (bit VC: VC has a serviced sender channel); the
// channels whose free-slots registers the pass's first read group reads.
static_assert(MAX_NUM_VCS == 3, "the sender's features and its pass name VCs 0, 1 and 2");
constexpr uint32_t sender_channels =
    SenderVc<0>::sender_channels | SenderVc<1>::sender_channels | SenderVc<2>::sender_channels;
constexpr uint32_t worker_channels =
    SenderVc<0>::worker_channels | SenderVc<1>::worker_channels | SenderVc<2>::worker_channels;
constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
constexpr uint32_t sender_vcs = (SenderVc<0>::sender_channels != 0 ? 1u : 0u) |
                                (SenderVc<1>::sender_channels != 0 ? 2u : 0u) |
                                (SenderVc<2>::sender_channels != 0 ? 4u : 0u);
constexpr uint32_t channels_read_at_pass_start =
    (SenderVc<0>::reads_free_slots_at_pass_start ? SenderVc<0>::sender_channels : 0u) |
    (SenderVc<1>::reads_free_slots_at_pass_start ? SenderVc<1>::sender_channels : 0u) |
    (SenderVc<2>::reads_free_slots_at_pass_start ? SenderVc<2>::sender_channels : 0u);
// In Tensix mux mode the mux is the only producer: no router forwards into this one. (Channel trimming may leave the
// mux's worker channel unserviced too.)
static_assert(!tensix_mux_mode || forwarding_channels == 0);

// A forwarding channel's index among the serviced forwarding channels (its producer's address at init).
template <size_t CH>
constexpr size_t forwarding_index() {
    static_assert(((forwarding_channels >> CH) & 1u) != 0, "not a serviced forwarding channel");
    return popcount(forwarding_channels & ((1u << CH) - 1u));
}

// The far router's receiver channel of VC: its slots and its packets-received register (on the far router).
template <size_t VC>
struct RemoteReceiverChannel {
    static constexpr size_t rx =  // the host's index of the channel
        VC == 0 ? VC0_RECEIVER_CHANNEL : VC == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[rx];
    static constexpr uint32_t first_slot_addr =
        eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[rx]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[rx];
};

// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

// This router's progress into a remote receiver channel: its slots known free, and the next slot to fill.
struct RemoteReceiverChannelProgress {
    uint32_t free_slots;
    uint32_t next_slot_addr;
};

// A remote receiver channel's progress before the first send: every slot free.
template <size_t VC>
constexpr RemoteReceiverChannelProgress remote_receiver_channel_progress_start() {
    return {RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr};
}

// A sender channel's progress through its slots: its per-packet state.
struct SenderChannelProgress {
    uint32_t next_slot_addr;     // the next slot to send from
    uint32_t sent;               // packets sent
    uint32_t completed;          // completions taken
    uint32_t first_level_acked;  // first-level acks taken (first-level-ack VCs only)
};

// A sender channel's state: its progress, and its producer's link: a worker channel's connection state (the value its
// connection semaphore holds while nothing changes), a forwarding channel's upstream router's free-slots register
// address (cached when the upstream router connected).
template <size_t CH>
struct SenderChannelState {
    SenderChannelProgress progress = {SenderChannel<CH>::first_slot_addr, 0, 0, 0};
    [[no_unique_address]] Present<SenderChannel<CH>::worker, uint32_t, 0> connection_state =
        present<SenderChannel<CH>::worker, 0>(tt::tt_fabric::connection_interface::unused_connection_value);
    [[no_unique_address]] Present<!SenderChannel<CH>::worker, uint64_t, 1> upstream_free_slots_addr{};
};

// VC's send state: its remote receiver channel's progress, and its worker channel's state where the VC has a serviced
// worker channel. (Its forwarding channels' states are SenderChannelStates::forwarding.)
template <size_t VC>
struct SenderVcState {
    RemoteReceiverChannelProgress remote = remote_receiver_channel_progress_start<VC>();
    [[no_unique_address]] Present<SenderVc<VC>::worker_channels != 0, SenderChannelState<first_sender_channel_of(VC)>, 2>
        worker{};
};

