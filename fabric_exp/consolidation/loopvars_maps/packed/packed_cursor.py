"""packed_cursor.py VARIANT_DIR: the packed cursor (one counts word + the slot address) in place of Position and
CompletionCounts, in router_receiver.hpp; shared by v4rxb2p (B2) and v4rxp2p (P2)."""
import sys
p = sys.argv[1] + "/tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
s = open(p).read()
def sub(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit(f"{s.count(old)}x: {old[:70]!r}")
    s = s.replace(old, new)

sub("""// Where a receiver channel is in its ring.
struct Position {
    uint32_t ready;      // packets this channel may process (present, and acked under first-level acks), not retired
    uint32_t slot_addr;  // header address of the next slot
    uint32_t index;      // ring index of the next slot
};

// Moves `position` to the next slot of channel CH's ring.
template <size_t CH>
FORCE_INLINE void advance(Position& position) {
    position.index = Channel<CH>::next_index(position.index);
    position.slot_addr =
        position.index == 0 ? Channel<CH>::first_slot_addr : position.slot_addr + channel_buffer_size;
}
""", """// A receiver channel's cursor: the next slot's header address, and the channel's counts packed in one word. Each count
// is bounded by the ring's slot count (at most 255, Channel) or by idle_passes_before_partial_credit, so a byte holds
// it; the operations below keep each count in its byte (no carry or borrow crosses into the next).
class Cursor {
public:
    uint32_t slot_addr = 0;  // header address of the next slot

    // Packets this channel may process (present, and acked under first-level acks), not retired.
    FORCE_INLINE uint32_t ready() const { return field(ready_shift); }
    // Ring index of the next slot.
    FORCE_INLINE uint32_t index() const { return counts_ >> index_shift; }
    // Slots retired (their NoC commands issued) but not yet credited.
    FORCE_INLINE uint32_t uncredited() const { return field(uncredited_shift); }
    // Groups of several slots: passes without a packet while a partial completion group waits.
    FORCE_INLINE uint32_t idle_passes() const { return field(idle_shift); }

    FORCE_INLINE void add_ready(uint32_t n) { counts_ += n << ready_shift; }
    FORCE_INLINE void set_ready(uint32_t n) { counts_ = (counts_ & ~(0xFFu << ready_shift)) | (n << ready_shift); }
    FORCE_INLINE void credit(uint32_t n) { counts_ -= n << uncredited_shift; }
    FORCE_INLINE void count_idle_pass() { counts_ += 1u << idle_shift; }
    FORCE_INLINE void clear_idle_passes() { counts_ &= ~(0xFFu << idle_shift); }
    // The next slot's packet retired: one fewer ready, one more uncredited, and the cursor on the slot after it.
    FORCE_INLINE void retire(uint32_t next_index, uint32_t next_slot_addr) {
        const uint32_t counts = counts_ - (1u << ready_shift) + (1u << uncredited_shift);
        counts_ = (counts & ~(0xFFu << index_shift)) | (next_index << index_shift);
        slot_addr = next_slot_addr;
    }

    // The counts word, for code that keeps it in memory and compares it.
    FORCE_INLINE uint32_t counts_word() const { return counts_; }
    FORCE_INLINE void set_counts_word(uint32_t counts) { counts_ = counts; }

private:
    // counts_: ready in bits 0-7, uncredited in 8-15, idle passes in 16-23, the ring index in 24-31.
    static constexpr uint32_t ready_shift = 0;
    static constexpr uint32_t uncredited_shift = 8;
    static constexpr uint32_t idle_shift = 16;
    static constexpr uint32_t index_shift = 24;
    FORCE_INLINE uint32_t field(uint32_t shift) const { return (counts_ >> shift) & 0xFF; }
    uint32_t counts_ = 0;
};
static_assert(idle_passes_before_partial_credit <= 0xFF);

// Retires the next slot of channel CH's ring (Cursor::retire) and moves the cursor to the slot after it.
template <size_t CH>
FORCE_INLINE void retire_slot(Cursor& cursor) {
    const uint32_t next = Channel<CH>::next_index(cursor.index());
    cursor.retire(next, next == 0 ? Channel<CH>::first_slot_addr : cursor.slot_addr + channel_buffer_size);
}
""")
sub("""// The completion counters of one receiver channel (loop locals).
struct CompletionCounts {
    uint32_t uncredited = 0;  // slots retired (NoC commands issued) but not yet credited
    // Groups of several slots: passes without a packet while a partial completion group waits.
    uint32_t idle_passes_with_partial_group = 0;
};

""", "")
open(p, "w").write(s)
print("cursor type ok")
