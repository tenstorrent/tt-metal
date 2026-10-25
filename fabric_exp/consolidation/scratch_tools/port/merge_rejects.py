import sys
p = sys.argv[1] + "/fabric_erisc_router_v3.cpp"
s = open(p).read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
# Rejected hunk 1 (v3e1c): the uncounted-write helper and local_delivery's comment.
sub("""// Local delivery. A plain unicast write, the common case, is issued here: the same NoC command as
// execute_chip_unicast_to_local_chip_impl's write arm, without the send-type dispatch (sparse test, range test, jump
// table). Every other send type goes through that dispatch. Adds the slot's local read class to `record`.
template <size_t CH>
FORCE_INLINE void local_delivery(""",
"""// One write of a scatter packet without the software write counters (noc_nonposted_writes_num_issued/acked):
// noc_async_write_one_packet_with_trid's command with update_counter = false. The caller adds the packet's writes to
// the counters once, after its last write; nothing reads them in between. The router uses DM_DEDICATED_NOC, so these
// are the counters ncrisc_noc_fast_write would have updated.
FORCE_INLINE void local_write_uncounted(uint32_t src_addr, uint64_t dst_noc_addr, uint32_t size, uint32_t trid) {
    constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size, -1, false, noc);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_addr, size);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write<noc_mode, /*use_trid=*/true, /*update_counter=*/false>(
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
FORCE_INLINE void local_delivery(""")
# Rejected hunk 4 (v3l3): the yield flag's comment, and the TX-queue spin inline again.
sub("""    bool attempted;                                   // a send was attempted this round (the idle context switch test)
    Telemetry telemetry;
};

// The TX queue was busy after a data command (never seen in practice): spin until it is idle. Out of line: one copy for
// every send site.
template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) void wait_for_sender_txq_idle() {""",
"""    bool attempted;                                   // a send attempt yielded this round (the TX queue was busy)
    Telemetry telemetry;
};

// The TX queue was busy after a data command (never seen in practice): spin until it is idle.
FORCE_INLINE void wait_for_sender_txq_idle() {""")
# The receiver overview's step 6 follows the new local delivery.
sub("""//   6. Local delivery: a plain write is issued directly (the write arm's NoC command); every other send type goes
//      through execute_chip_unicast_to_local_chip_impl.""",
"""//   6. Local delivery: plain writes, scatter writes, atomics and fused write + atomics are issued directly (the NoC
//      commands of execute_chip_unicast_to_local_chip_impl's arms); every other send type goes through it.""")
open(p, "w").write(s)
print("merged", p)
