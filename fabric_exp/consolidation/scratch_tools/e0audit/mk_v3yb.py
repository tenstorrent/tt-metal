#!/usr/bin/env python3
"""mk_v3yb.py SENDER_HPP: v3y -> v3yb: when the worker channel has a packet but the far ring has no room, the worker visit
takes the completions reported so far (and returns their credit) before its send decision, so that room they free is
used in the same pass. The rest of the visit, and the forwarding visits, are v3y's. Every anchor must match once."""
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
""", """// When the worker channel has a packet but the far ring has no room, the worker visit takes the completions reported so
// far (and returns their credit) before its send decision, so that room they free is used in the same pass, not the
// next: how soon a freed slot is used then does not depend on how long the rest of the pass is. Only the worker visit
// does this: the builder gives the worker channel the spare slots, so it is the channel that ends up waiting on far
// room, while a forwarded flow waits on its own channel's credits (forwarding visits keep the plain order).
// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
""")

# worker visit (the early take and the send nested under one test of has_unsent_packet)
sub("""                    const bool has_unsent_packet = (free_slots_raw[0] & stream_count_mask) != num_slots;
                    int32_t completions = 0;
                    bool sent_packet = false;
                    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && has_unsent_packet) {
                        sent_packet = send_packet<free_slots_stream<0>, VC0_RECEIVER_CHANNEL>(
                            0,
                            worker_slot,
                            ChannelSlots<0>::first,
                            ChannelSlots<0>::last,
                            far_vc0,
                            completion_counter_addr(0),
                            worker_sent,
                            worker_completed,
                            completions,
                            state.attempted,
                            state.telemetry);
                    }
                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = static_cast<int32_t>(
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(0)) -
                            worker_completed);
                    }
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    if (completions) {""", """                    const bool has_unsent_packet = (free_slots_raw[0] & stream_count_mask) != num_slots;
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    int32_t completions = 0;
                    bool sent_packet = false;
                    if (has_unsent_packet) {
                        // Waiting for far room: take the completions first (see the top of this file).
                        if (!far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && worker_sent != worker_completed) {
                            const int32_t freed = new_completions(0, worker_completed);
                            if (freed) {
                                far_vc0.free_slots += freed;
                                worker_completed += freed;
                                if constexpr (!worker_first_level_ack) {
                                    credit_worker(worker, worker_completed, connected);
                                }
                            }
                        }
                        if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0)) {
                            sent_packet = send_packet<free_slots_stream<0>, VC0_RECEIVER_CHANNEL>(
                                0,
                                worker_slot,
                                ChannelSlots<0>::first,
                                ChannelSlots<0>::last,
                                far_vc0,
                                completion_counter_addr(0),
                                worker_sent,
                                worker_completed,
                                completions,
                                state.attempted,
                                state.telemetry);
                        }
                    }
                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = new_completions(0, worker_completed);
                    }
                    if (completions) {""")
open(p, "w").write(s)
print("ok")
