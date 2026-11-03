"""packed_uses.py VARIANT_DIR: rewrites the uses of Position / CompletionCounts to the packed Cursor (both bases)."""
import sys
p = sys.argv[1] + "/tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
s = open(p).read()
def sub(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit(f"{s.count(old)}x: {old[:70]!r}")
    s = s.replace(old, new)

# first-level acks
sub("""FORCE_INLINE bool accumulate_first_level_acks(Position& position, Credits& credits, uint32_t arrived) {""",
    """FORCE_INLINE bool accumulate_first_level_acks(Cursor& cursor, Credits& credits, uint32_t arrived) {""")
sub("""    uint32_t index = ring_add<Channel<CH>::num_slots>(position.index, position.ready);""",
    """    uint32_t index = ring_add<Channel<CH>::num_slots>(cursor.index(), cursor.ready());""")
sub("""    position.ready += arrived;
    return true;""", """    cursor.add_ready(arrived);
    return true;""")
# retire
sub("""    Position& position = cursor.position;
    sources.set(position.index, source);
    cursor.counts.uncredited++;
    position.ready--;
    if constexpr (!Channel<CH>::first_level_ack) {
        increment_local_update_ptr_val<Channel<CH>::packets_received_stream>(-1);
    }
    advance<CH>(position);""", """    sources.set(cursor.index(), source);
    if constexpr (!Channel<CH>::first_level_ack) {
        increment_local_update_ptr_val<Channel<CH>::packets_received_stream>(-1);
    }
    retire_slot<CH>(cursor);""")
# handle
sub("""    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(cursor.position.slot_addr);""",
    """    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(cursor.slot_addr);""")
sub("""Channel<CH>::CompletionTrids::of_slot(cursor.position.index)""", """Channel<CH>::CompletionTrids::of_slot(cursor.index())""", 2)
sub("""                cursor.position.slot_addr,
                size,
                trid);""", """                cursor.slot_addr,
                size,
                trid);""")
sub("""routers.forward_all(route.forward_to(), cursor.position.slot_addr, size, trid);""",
    """routers.forward_all(route.forward_to(), cursor.slot_addr, size, trid);""")
sub("""downstreams.relay.forward(cursor.position.slot_addr, size, trid);""",
    """downstreams.relay.forward(cursor.slot_addr, size, trid);""")
# receive
sub("""    Position& position = cursor.position;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (position.ready == 0) {
        if constexpr (Channel<CH>::first_level_ack) {
            return false;  // nothing acked to process
        } else {
            position.ready = arrived;  // the register's count, from the pass's grouped reads
            if (position.ready == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(position.slot_addr);""",
"""    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (cursor.ready() == 0) {
        if constexpr (Channel<CH>::first_level_ack) {
            return false;  // nothing acked to process
        } else {
            cursor.set_ready(arrived);  // the register's count, from the pass's grouped reads
            if (cursor.ready() == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(cursor.slot_addr);""")
# slots_due_for_credit
sub("""FORCE_INLINE bool slots_due_for_credit(CompletionCounts& counts, uint32_t next_index, bool progress, uint32_t& count) {
    const uint32_t waiting = counts.uncredited;""", """FORCE_INLINE bool slots_due_for_credit(Cursor& cursor, bool progress, uint32_t& count) {
    const uint32_t waiting = cursor.uncredited();""")
sub("""            counts.idle_passes_with_partial_group = 0;
            is_due = waiting > GROUP_SIZE;""", """            cursor.clear_idle_passes();
            is_due = waiting > GROUP_SIZE;""")
sub("""            is_due = waiting >= GROUP_SIZE ||
                     (waiting != 0 && ++counts.idle_passes_with_partial_group >= idle_passes_before_partial_credit);""",
"""            if (waiting != 0 && waiting < GROUP_SIZE) {
                cursor.count_idle_pass();
            }
            is_due = waiting >= GROUP_SIZE ||
                     (waiting != 0 && cursor.idle_passes() >= idle_passes_before_partial_credit);""")
sub("""        const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
        const uint32_t group_end""", """        const uint32_t first = oldest_uncredited<NUM_SLOTS>(cursor.index(), waiting);
        const uint32_t group_end""")
# credit_completions
sub("""// `next_index` is the next slot to retire; `progress` says whether this pass handled a packet.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool credit_completions(
    CompletionCounts& counts,
    const SlotSourcesOf<CH>& sources,
    uint32_t next_index,
    bool progress,
    CreditSender& credits) {""", """// `progress` says whether this pass handled a packet.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool credit_completions(
    Cursor& cursor,
    const SlotSourcesOf<CH>& sources,
    bool progress,
    CreditSender& credits) {""")
sub("""    const uint32_t waiting = counts.uncredited;
    uint32_t count;
    if (!slots_due_for_credit<completion_group_size, num_slots>(counts, next_index, progress, count)) {""",
"""    const uint32_t waiting = cursor.uncredited();
    uint32_t count;
    if (!slots_due_for_credit<completion_group_size, num_slots>(cursor, progress, count)) {""")
sub("""    const uint32_t first = oldest_uncredited<num_slots>(next_index, waiting);
    if (!trid_writes_sent_within""", """    const uint32_t first = oldest_uncredited<num_slots>(cursor.index(), waiting);
    if (!trid_writes_sent_within""")
sub("""    if constexpr (completion_group_size > 1) {
        counts.idle_passes_with_partial_group = 0;
    }
    accumulate_completion_credits<completion_group_size>(sources, first, count, credits);
    counts.uncredited = waiting - count;""", """    if constexpr (completion_group_size > 1) {
        cursor.clear_idle_passes();
    }
    accumulate_completion_credits<completion_group_size>(sources, first, count, credits);
    cursor.credit(count);""")
# step
sub("""        credit_frame_due |= accumulate_first_level_acks<CH>(cursor.position, credits, arrived);""",
    """        credit_frame_due |= accumulate_first_level_acks<CH>(cursor, credits, arrived);""")
sub("""    timer.set_should_dump(cursor.position.ready != 0);""", """    timer.set_should_dump(cursor.ready() != 0);""")
sub("""    credit_frame_due |= credit_completions<CH>(cursor.counts, sources, cursor.position.index, progress, credits);""",
    """    credit_frame_due |= credit_completions<CH>(cursor, sources, progress, credits);""")
open(p, "w").write(s)
print("uses ok")
