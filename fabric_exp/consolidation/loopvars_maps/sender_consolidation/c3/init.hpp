// ---------------------------------------------------------------------------------------------------------------------
// Init (out of line, once)
// ---------------------------------------------------------------------------------------------------------------------

// Waits until an upstream router asks to connect on `connection_semaphore`, or termination. One body for every channel.
__attribute__((noinline)) void wait_for_connect_request(volatile tt_l1_ptr uint32_t* connection_semaphore) {
    uint32_t polls = 0;
    while (!connect_is_requested(*connection_semaphore) && !terminating()) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (polls++ == 100000) {
            polls = 0;
            run_routing();
        }
    }
}

// Waits for each forwarding channel's upstream router to connect where the host has it wait, and caches that router's
// free-slots register address in the channel's interface (after SenderChannelInterfaces::start).
__attribute__((noinline)) void wait_for_upstream_routers() {
    for_each_index<NUM_SENDER_CHANNELS>([](auto i) {
        constexpr size_t CH = decltype(i)::value;
        if constexpr (((SenderChannels::forwarding_channels >> CH) & 1u) != 0) {
            if constexpr (ForwardingChannel<CH>::waits_for_upstream_router) {
                auto& interface = SenderChannelInterfaces::of<CH>();
                wait_for_connect_request(interface.connection_live_semaphore);
                interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
            }
        }
    });
}

