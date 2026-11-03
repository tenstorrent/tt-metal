// ---------------------------------------------------------------------------------------------------------------------
// Producers
// ---------------------------------------------------------------------------------------------------------------------

// Every sender channel position's library interface: its producer's connection (connection info, connection semaphore,
// the producer's credit address, cached when it connects). At a fixed location, not in run()'s frame: the out-of-line
// connection path reaches them by channel. Storage and an explicit start: the kernel runs no global constructors.
struct SenderChannelInterfaces {
    using All = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<
                         tt::tt_fabric::worker_handshake_noc,
                         SENDER_NUM_BUFFERS_ARRAY>::make(std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

    // Channel `ch`'s read counter in its connection info: the credits returned so far, which a worker reads when it
    // connects, and the source word of each credit write to it.
    static constexpr uint32_t read_counter_addr(size_t ch) {
        return sender_connection_info_addrs[ch] +
               offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
    }

    // Builds every position's interface, serviced or not (once, at init, on an ERISC that serves sender channels): its
    // connection info, its connection semaphore (the runtime args'), its starting credits (0 for a worker channel, whose
    // credits are counts; the slot count for a forwarding channel) and a worker channel's credit source word (its read
    // counter).
    __attribute__((noinline)) static void start(
        const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& connection_semaphores) {
        new (storage) All{};
        for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
            constexpr size_t CH = decltype(i)::value;
            constexpr bool worker = is_sender_channel_worker[CH];  // the host's role, serviced or not
            auto* info =
                reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(sender_connection_info_addrs[CH]);
            info->edm_read_counter = 0;
            ASSERT(!worker || connection_semaphores[CH] == sender_connection_semaphore_addrs[CH]);
            new (&of<CH>()) tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<
                tt::tt_fabric::worker_handshake_noc,
                SENDER_NUM_BUFFERS_ARRAY[CH]>(
                info,
                0,
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_semaphores[CH]),
                CommandBuffers::worker_credits[CH],
                worker ? 0 : SENDER_NUM_BUFFERS_ARRAY[CH],
                worker ? read_counter_addr(CH) : notify_worker_of_read_counter_update_src_address);
        });
    }

    // Channel CH's interface (after start).
    template <size_t CH>
    static FORCE_INLINE auto& of() {
        return std::launder(reinterpret_cast<All*>(storage))->template get<CH>();
    }

private:
    alignas(All) static inline unsigned char storage[sizeof(All)];
};

// What a worker channel's connection-semaphore change does, the same for every worker channel (one copy: the connection
// type does not depend on the channel's slot count). Opening caches the worker's credit address; closing writes the
// credits returned so far (`returned`) into the worker's connection info and signals its teardown semaphore. Returns
// the value the semaphore holds while nothing changes.
__attribute__((noinline)) uint32_t open_or_close_worker_connection(
    tt::tt_fabric::EdmChannelWorkerConnection<tt::tt_fabric::worker_handshake_noc>& connection,
    uint32_t semaphore,
    uint32_t expected,
    uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            connection.cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, MY_ETH_CHANNEL>();
            return open_connection_value;
        }
        return unused_connection_value;
    }
    if (semaphore == close_connection_request_value) {
        connection.close_connection<true, ENABLE_RISC_CPU_DATA_CACHE>(returned);
        return unused_connection_value;
    }
    return open_connection_value;
}

// A worker channel: a worker on this chip connects at run time (its connection semaphore, checked every pass) and is
// credited with the absolute count of slots returned. The count goes into the channel's read counter, which therefore
// always holds it; while a worker is connected, a 4-byte posted NoC write copies that word into the worker's L1 on the
// channel's own credit command buffer (CommandBuffers). Where command buffers keep presets, that buffer stays programmed
// for the connection (at the connection, and again after the base firmware ran), so a credit only issues the write;
// elsewhere every credit is a whole command. The credit and the connection event are out of line: a credit goes back
// once per completion group from the far router (every few packets), or under first-level acks once per ack frame, and
// its constants are built then instead of being held in the sender loop's registers for its whole run (RESULTS §98).
template <size_t CH>
struct WorkerChannel {
    static_assert(SenderChannel<CH>::worker);
    static_assert(!sender_ch_live_check_skip[CH], "the host checks every worker channel's connection");
    // Unflushed: the read counter may be rewritten before an earlier write read it; the worker then sees the newer,
    // larger count.
    static_assert(!enable_read_counter_update_noc_flush);

    // Its connection semaphore (compile time; the interface holds the same address), its read counter and its credit
    // command buffer.
    static constexpr uint32_t connection_semaphore_addr = sender_connection_semaphore_addrs[CH];
    static constexpr uint32_t read_counter_addr = SenderChannelInterfaces::read_counter_addr(CH);
    static constexpr uint8_t credit_cmd_buf = CommandBuffers::worker_credits[CH];

    static FORCE_INLINE bool connected(const SenderChannelState<CH>& state) {
        return state.connection_state != tt::tt_fabric::connection_interface::unused_connection_value;
    }

    // The connection, every pass (by rule), from the semaphore the pass's read group read: a change (open or close
    // request) is handled out of line. Returns whether it changed.
    static FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, uint32_t semaphore) {
        if (semaphore != state.connection_state) {
            const uint32_t returned =
                SenderChannel<CH>::first_level_ack ? state.progress.first_level_acked : state.progress.completed;
            state.connection_state = connection_event(semaphore, state.connection_state, returned);
            return true;
        }
        return false;
    }

    // `returned` slots in all: into the read counter and, while a worker is connected, from there into the worker's L1.
    __attribute__((noinline)) static void credit(uint32_t returned, bool connected) {
        constexpr uint32_t source = read_counter_addr;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source) = returned;
        if (connected) {
            if constexpr (CommandBuffers::presets) {
                ncrisc_noc_write_with_state<noc_mode, /*posted=*/true, /*update_counter=*/true, /*one_packet=*/true>(
                    SenderChannel<CH>::producer_noc,
                    credit_cmd_buf,
                    source,
                    static_cast<uint32_t>(interface().cached_worker_semaphore_address));
            } else {
                while (!noc_cmd_buf_ready(SenderChannel<CH>::producer_noc, credit_cmd_buf)) {
                }
                ncrisc_noc_fast_write<noc_mode>(
                    SenderChannel<CH>::producer_noc,
                    credit_cmd_buf,
                    source,
                    interface().cached_worker_semaphore_address,
                    sizeof(uint32_t),
                    NOC_UNICAST_WRITE_VC,
                    /*mcast=*/false,
                    /*linked=*/false,
                    /*num_dests=*/1,
                    /*multicast_path_reserve=*/true,
                    /*posted=*/true);
            }
        }
    }

    // The credit command buffer programmed for the connected worker, where command buffers keep presets.
    __attribute__((noinline)) static void preset_credit() {
        if constexpr (CommandBuffers::presets) {
            ncrisc_noc_write_set_state</*posted=*/true, /*one_packet=*/true>(
                SenderChannel<CH>::producer_noc,
                credit_cmd_buf,
                interface().cached_worker_semaphore_address,
                sizeof(uint32_t),
                NOC_UNICAST_WRITE_VC);
        }
    }

private:
    static FORCE_INLINE auto& interface() { return SenderChannelInterfaces::of<CH>(); }

    // The shared event, then, where it opened the connection, the credit command buffer preset and the credits
    // returned so far sent. Its own function, so the pass calls one function per channel.
    __attribute__((noinline)) static uint32_t connection_event(uint32_t semaphore, uint32_t expected, uint32_t returned) {
        const uint32_t state = open_or_close_worker_connection(interface(), semaphore, expected, returned);
        if (expected == tt::tt_fabric::connection_interface::unused_connection_value &&
            state != tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_credit();
            credit(returned, true);
        }
        return state;
    }
};

// A forwarding channel: a router on this chip facing another direction writes into it. That upstream router connects
// once at init and is credited +n on its free-slots register.
template <size_t CH>
struct ForwardingChannel {
    static_assert(!SenderChannel<CH>::worker);
    // Whether init waits for its upstream router to connect (SENDER_CH_<i>_WAIT_STATIC_CONNECTION; a channel a trimming
    // override forced on has none).
    static constexpr bool waits_for_upstream_router = sender_ch_wait_static_connection[CH];

    // The upstream router's free-slots register, as the interface cached it when that router connected.
    static FORCE_INLINE uint64_t upstream_free_slots_addr() {
        return SenderChannelInterfaces::of<CH>().cached_worker_semaphore_address;
    }

    // +n on the upstream router's free-slots register.
    static FORCE_INLINE void credit(uint64_t upstream_free_slots_addr, uint32_t n) {
        noc_inline_dw_write<InlineWriteDst::REG, true>(
            upstream_free_slots_addr,
            pack_value_for_inc_on_write_stream_reg_write(n),
            INLINE_WRITE_ALL_BYTES,
            SenderChannel<CH>::producer_noc);
    }
};

