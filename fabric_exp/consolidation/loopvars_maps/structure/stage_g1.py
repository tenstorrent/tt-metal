"""stage_g1.py: v4rxg1 = v4rxf4 + channel-kind renames (receiver side), both static_asserts out, uint8_t source."""
import os, re, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4rxf4", f"{V}/v4rxg1"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"

def edit(name, subs=(), regex=()):
    p = f"{R}/{name}"
    s = open(p).read()
    for o, n in subs:
        assert s.count(o) == 1, (name, o[:80])
        s = s.replace(o, n)
    for o, n in regex:
        s, k = re.subn(o, n, s)
        assert k, (name, o)
    open(p, "w").write(s)

edit("router_receiver.hpp",
     subs=[
         ("""static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(MAX_NUM_SENDER_CHANNELS <= 0x100);  // a sender channel fits in a byte

""", ""),
         ("    FORCE_INLINE void set(uint32_t index, uint32_t source) { bytes()[index] = static_cast<uint8_t>(source); }",
          "    FORCE_INLINE void set(uint32_t index, uint8_t source) { bytes()[index] = source; }"),
         ("SlotSourcesOf<VC>& sources, uint32_t source) {", "SlotSourcesOf<VC>& sources, uint8_t source) {"),
     ],
     regex=[(r"\bRingProgress\b", "ReceiverChannelProgress"), (r"\bring_add\b", "index_add"),
            (r"\bcompletion_group_size_of_ring\b", "completion_group_size_for")])

edit("router_downstream.hpp",
     subs=[
         ("//   - a cursor: the next slot in the downstream sender channel's ring and the number of slots known free there;",
          "//   - its progress: the next slot in the downstream sender channel and the number of slots known free there;"),
         ("//   - the link: the ring's base and the downstream's NoC coordinate and credit register (from runtime args, set at init).",
          "//   - the link: the channel's first slot and the downstream's NoC coordinate and credit register (from runtime args,\n//     set at init)."),
         ("// The downstream connections of one VC as the receiver loop uses them: the cursors, and the links set at init. Holding\n// the links by value puts a forward's ring bounds at a fixed offset from its cursor, one load away.",
          "// The downstream connections of one VC as the receiver loop uses them: each downstream sender channel's progress, and\n// the links set at init. Holding the links by value puts a forward's slot bounds at a fixed offset from its progress,\n// one load away."),
         ("// links and the cursors.", "// links and the progress."),
         ("// the cursor filled.", "// its progress set."),
         ("// The slot after `slot` in a ring from `ring_base` whose last slot is at `last_slot_offset`.",
          "// The slot after `slot` in a channel whose slots start at `first_slot_addr` and end at `last_slot_offset` from it."),
         ("    uint32_t next_slot;   // the downstream ring's next slot\n",
          "    uint32_t next_slot_addr;  // the downstream sender channel's next slot\n"),
         ("    uint32_t free_slots;  // slots known free there (a lower bound of the local credit register)\n",
          "    uint32_t free_slots;      // slots known free there (a lower bound of the local credit register)\n"),
     ],
     regex=[(r"\bDownstreamCursor\b", "DownstreamSenderChannelProgress"), (r"\bnext_ring_slot\b", "slot_after"),
            (r"\bring_bases\b", "first_slot_addrs"), (r"\bring_base\b", "first_slot_addr"),
            (r"\.next_slot\b", ".next_slot_addr"), (r"\bcursors\b", "progress"), (r"\bcursor\b", "progress"),
            (r"ring base,", "first slot address,")])

edit("router_init.hpp",
     subs=[("// Before the Ethernet handshake, on the receiver ERISC: every downstream connection's links and cursors",
            "// Before the Ethernet handshake, on the receiver ERISC: every downstream connection's links and progress")],
     regex=[(r"\bring_bases\b", "first_slot_addrs"), (r"\bring_base\b", "first_slot_addr")])
print("made", dst)
