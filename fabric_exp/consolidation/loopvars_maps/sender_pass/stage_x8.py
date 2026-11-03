"""stage_x8.py SRC DST: the comment pass (V4_SENDER_OWNERSHIP.md §3 #9): no cycle numbers or variant names in the
sender's comments, one line per why with a doc pointer. Identity expected."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/{sys.argv[1]}", f"{V}/{sys.argv[2]}"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()
for o, n in [
    ("""// The channel enters only through four values of SenderChannel<CH>: its ring bounds (the slot advance), its completion
// counter, its free-slots register and its id (stamped into the header); the rest is the same for every channel. It is
// inlined once per channel with those values as immediates: one copy shared at run time cost forwarded packets 5 to 18
// cycles (RESULTS §98, v4s, v4s2).""",
     """// The channel enters only through four values of SenderChannel<CH>: its slot bounds (the slot advance), its completion
// counter, its free-slots register and its id (stamped into the header); the rest is the same for every channel. It is
// inlined once per channel with those values as immediates: one copy shared at run time is slower on forwarded packets
// (RESULTS §98)."""),
    ("""// sender loop's registers for its whole run (2D 1-hop writes 141.8 -> 130.5, RESULTS §98).""",
     """// sender loop's registers for its whole run (RESULTS §98)."""),
]:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
