"""stage_g3.py: v4rxg3 = v4rxg2 + loop_local_vc derived: the lowest VC whose receiver channel this router runs."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4rxg2", f"{V}/v4rxg3"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
s = open(p).read()
for o, n in [
("""    // The VC whose progress run() carries as a loop local.
    static constexpr size_t loop_local_vc = 0;
""", """    // The VC whose progress run() carries as a loop local: the lowest VC whose receiver channel this router runs (a
    // trimmed router can run VC1's and not VC0's, V4_TRIM.md §1). 0 where none runs: then nothing is stepped.
    static constexpr size_t loop_local_vc = receiver_vcs != 0 ? lowest_bit(receiver_vcs) : 0;
"""),
("""    // The loop-local VC's progress in place: what run() copies at its start; a one-ERISC router steps on it in place.
    // Where that VC's channel does not run, the states themselves: no step uses it.
""", """    // The loop-local VC's progress in place: what run() copies at its start; a one-ERISC router steps on it in place.
    // Without a receiver channel (receiver_vcs == 0), the states themselves: no step uses it.
"""),
("""    template <size_t VC, typename LoopCopy>
    FORCE_INLINE ReceiverChannelProgress& progress_for(LoopCopy& loop_copy) {""",
 """    template <size_t VC>
    FORCE_INLINE ReceiverChannelProgress& progress_for(ReceiverChannelProgress& loop_copy) {"""),
]:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
