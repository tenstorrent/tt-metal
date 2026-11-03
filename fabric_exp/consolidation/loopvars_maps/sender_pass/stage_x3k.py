"""stage_x3k.py: v4txo3k = v4txo3 + each VC's remote receiver progress packed into one word (V4_SENDER_OWNERSHIP.md §8.4,
the user's decision): RemoteReceiverChannelProgress<VC>, a type with operations (has_room, next_slot_addr, take_slot,
add_free_slots), its encoding, constants and static_asserts inside, ASSERTs on the protocol invariants."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo3", f"{V}/v4txo3k"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()

def rep(o, n, count=1):
    global s
    assert s.count(o) == count, (s.count(o), o[:90])
    s = s.replace(o, n)

rep("""// This router's progress into a remote receiver channel: its slots known free, and the next slot to fill.
struct RemoteReceiverChannelProgress {
    uint32_t free_slots;
    uint32_t next_slot_addr;
};

// A remote receiver channel's progress before the first send: every slot free.
template <size_t VC>
constexpr RemoteReceiverChannelProgress remote_receiver_channel_progress_start() {
    return {RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr};
}
""", """// This router's progress into VC's remote receiver channel, in one word: the slots known free there (low 16 bits) and
// the next slot to fill, its address / 16 (high 16 bits). One word, so a VC kept in memory takes and puts back its
// progress with one load and one store (V4_SENDER_OWNERSHIP.md §8.4). The encoding is for ERISC L1 only (the remote
// receiver channel is always in the far ERISC's L1); a Tensix downstream must not reuse it.
template <size_t VC>
class RemoteReceiverChannelProgress {
public:
    // Every slot free; the next slot is the first.
    static constexpr RemoteReceiverChannelProgress start() {
        return RemoteReceiverChannelProgress(
            encode(RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr));
    }

    FORCE_INLINE bool has_room(uint32_t n) const { return free_slots() >= n; }

    // The next slot's address: the high half times 16, which is word >> 12 because the low half stays below 4096.
    FORCE_INLINE uint32_t next_slot_addr() const { return word >> 12; }

    // After a send into the next slot: the next slot advances (after the last, back to the first) and one free slot is
    // taken, in one add. The next slot is the last one exactly when the word is at least the last slot's high half.
    FORCE_INLINE void take_slot() {
        ASSERT(has_room(1));
        word += word >= last_slot_word ? wrap_add : advance_add;
    }

    // Slots the far router completed: free again.
    FORCE_INLINE void add_free_slots(uint32_t n) {
        ASSERT(free_slots() + n <= RemoteReceiverChannel<VC>::num_slots);
        word += n;
    }

    FORCE_INLINE bool operator!=(const RemoteReceiverChannelProgress& other) const { return word != other.word; }

private:
    static constexpr uint32_t encode(uint32_t free_slots, uint32_t slot_addr) { return (slot_addr / 16) << 16 | free_slots; }
    static constexpr uint32_t last_slot_word = encode(0, RemoteReceiverChannel<VC>::last_slot_addr);
    static constexpr uint32_t advance_add = encode(0, channel_buffer_size) - 1;
    static constexpr uint32_t wrap_add = encode(0, RemoteReceiverChannel<VC>::first_slot_addr) - last_slot_word - 1;  // mod 2^32
    static_assert(RemoteReceiverChannel<VC>::first_slot_addr % 16 == 0 && channel_buffer_size % 16 == 0);
    static_assert(RemoteReceiverChannel<VC>::last_slot_addr / 16 < (1u << 16), "a slot address / 16 fits in 16 bits");
    static_assert(RemoteReceiverChannel<VC>::num_slots < (1u << 12), "the free-slot count stays below bit 12");

    constexpr explicit RemoteReceiverChannelProgress(uint32_t w) : word(w) {}
    FORCE_INLINE uint32_t free_slots() const { return word & 0xFFFFu; }

    uint32_t word;
};
""")
rep("""    RemoteReceiverChannelProgress remote = remote_receiver_channel_progress_start<VC>();""",
    """    RemoteReceiverChannelProgress<VC> remote = RemoteReceiverChannelProgress<VC>::start();""")
# send
rep("""    RemoteReceiverChannelProgress& remote,
    uint32_t& sent,""", """    RemoteReceiverChannelProgress<SenderChannel<CH>::vc>& remote,
    uint32_t& sent,""")
rep("""    internal_::eth_send_packet_bytes_unsafe(SENDER_TXQ, slot_addr, remote.next_slot_addr, size);
    remote.next_slot_addr = remote.next_slot_addr == RemoteReceiverChannel<VC>::last_slot_addr
                                ? RemoteReceiverChannel<VC>::first_slot_addr
                                : remote.next_slot_addr + channel_buffer_size;
""", """    internal_::eth_send_packet_bytes_unsafe(SENDER_TXQ, slot_addr, remote.next_slot_addr(), size);
    remote.take_slot();
""")
rep("""    remote.free_slots--;
""", "")
# the forwarding channels' helper: the VC's progress type
i = s.index("FORCE_INLINE void visit_forwarding_channels(")
j = s.rindex("template <uint32_t MASK, typename Connections>", 0, i)
s = s[:j] + "template <uint32_t MASK, typename Connections, typename Remote>" + s[j + len("template <uint32_t MASK, typename Connections>"):]
rep("""    uint32_t& channels_in_flight,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    if constexpr (MASK != 0) {""", """    uint32_t& channels_in_flight,
    Remote& remote,
    bool& send_yielded) {
    if constexpr (MASK != 0) {""")
# visits
rep("""    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {""", """    RemoteReceiverChannelProgress<SenderChannel<CH>::vc>& remote,
    bool& send_yielded) {""", count=2)
rep("""remote.free_slots >= SenderChannel<CH>::remote_free_slots_to_send""", """remote.has_room(SenderChannel<CH>::remote_free_slots_to_send)""", count=2)
rep("""        remote.free_slots += completions;""", """        remote.add_free_slots(completions);""", count=2)
# the rule's put-back
rep("""        const RemoteReceiverChannelProgress& taken,
        const RemoteReceiverChannelProgress& remote) {""", """        const RemoteReceiverChannelProgress<VC>& taken,
        const RemoteReceiverChannelProgress<VC>& remote) {""")
rep("""            if (remote.free_slots != taken.free_slots) {
                send_state.remote.free_slots = remote.free_slots;
            }
            if (remote.next_slot_addr != taken.next_slot_addr) {
                send_state.remote.next_slot_addr = remote.next_slot_addr;
            }""", """            if (remote != taken) {
                send_state.remote = remote;
            }""")
rep("""        const RemoteReceiverChannelProgress taken = send_state.remote;
        RemoteReceiverChannelProgress remote = taken;""", """        const RemoteReceiverChannelProgress<VC> taken = send_state.remote;
        RemoteReceiverChannelProgress<VC> remote = taken;""")
open(p, "w").write(s)
print("made", dst)
