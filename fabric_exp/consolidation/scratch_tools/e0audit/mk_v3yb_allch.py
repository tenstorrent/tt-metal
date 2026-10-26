#!/usr/bin/env python3
"""mk_v3yb.py SENDER_HPP: v3y -> v3yb: a channel that has a packet but no far room takes the completions reported so far
(and returns their credit) before its send decision, so that room they free is used in the same pass. The rest of the
visit is v3y's. Every anchor must match exactly once."""
import sys
p = sys.argv[1]
s = open(p).read()


def sub(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, n, a[:100])
    s = s.replace(a, b)


sub("""// Counters the far receiver writes over Ethernet, one word per sender channel.
FORCE_INLINE constexpr uint32_t completion_counter_addr(uint32_t ch) {
    return to_sender_remote_completion_counters_base_address + ch * sizeof(uint32_t);
}
""", """// Counters the far receiver writes over Ethernet, one word per sender channel.
FORCE_INLINE constexpr uint32_t completion_counter_addr(uint32_t ch) {
    return to_sender_remote_completion_counters_base_address + ch * sizeof(uint32_t);
}
// Completions the far receiver has reported for channel `ch` beyond `completed`.
FORCE_INLINE int32_t new_completions(uint32_t ch, uint32_t completed) {
    return static_cast<int32_t>(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(ch)) - completed);
}
""")

sub("""// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
""", """// A channel that has a packet but no far room takes its completions (and returns their credit) before the send decision,
// so that room they free is used in the same pass, not the next: how soon a freed slot is used then does not depend on
// how long the rest of the pass is.
// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
""")

# forwarding visit
sub("""// channel's bit in `in_flight` equal to "packets not yet completed (or acked)". Inline and specialized for CH: slot
""", """// channel's bit in `in_flight` equal to "packets not yet completed (or acked)". A packet that waits for far room takes
// the completions first (see the top of this file). Inline and specialized for CH: slot
""")
sub("""    uint32_t sent = channel.sent;
    uint32_t completed = channel.completed;
    int32_t completions = 0;
    bool sent_packet = false;
    if (far_has_room<RX, bit>(far, CH) && has_unsent_packet) {
        uint32_t slot = channel.slot;""", """    uint32_t sent = channel.sent;
    uint32_t completed = channel.completed;
    if (has_unsent_packet && !far_has_room<RX, bit>(far, CH) && sent != completed) {
        const int32_t freed = new_completions(CH, completed);
        if (freed) {
            far.free_slots += freed;
            completed += freed;
            channel.completed = completed;
            if constexpr (!FIRST_LEVEL_ACK) {
                credit_upstream_router(channel, freed);
            }
        }
    }
    int32_t completions = 0;
    bool sent_packet = false;
    if (far_has_room<RX, bit>(far, CH) && has_unsent_packet) {
        uint32_t slot = channel.slot;""")

# worker visit
sub("""                    const bool has_unsent_packet = (free_slots_raw[0] & stream_count_mask) != num_slots;
                    int32_t completions = 0;
                    bool sent_packet = false;
""", """                    const bool has_unsent_packet = (free_slots_raw[0] & stream_count_mask) != num_slots;
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    // A packet that waits for far room takes the completions first (see the top of this file).
                    if (has_unsent_packet && !far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) &&
                        worker_sent != worker_completed) {
                        const int32_t freed = new_completions(0, worker_completed);
                        if (freed) {
                            far_vc0.free_slots += freed;
                            worker_completed += freed;
                            if constexpr (!worker_first_level_ack) {
                                credit_worker(worker, worker_completed, connected);
                            }
                        }
                    }
                    int32_t completions = 0;
                    bool sent_packet = false;
""")
sub("""                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = static_cast<int32_t>(
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(0)) -
                            worker_completed);
                    }
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    if (completions) {""", """                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = new_completions(0, worker_completed);
                    }
                    if (completions) {""")
sub("""    if (!sent_packet && sent != completed) {
        completions = static_cast<int32_t>(
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(CH)) - completed);
    }""", """    if (!sent_packet && sent != completed) {
        completions = new_completions(CH, completed);
    }""")
open(p, "w").write(s)
print("ok")
