"""stage_x3f.py: v4txo3f = v4txo3b + the decisions on send's two fences that close no read group (V4_SENDER_OWNERSHIP.md
§3 #7, Q3): the one after the header's source-channel stamp goes (removing it alone is byte-identical on every config,
X0f), the one after the free-slots register update stays as a documented primitive, order_point() (removing it changes
E0 code). Identity expected."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo3b", f"{V}/v4txo3f"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"
p = f"{R}/router_sender.hpp"
s = open(p).read()
for o, n in [
    ("""        header->src_ch_id = static_cast<uint8_t>(CH);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status)) {""", """        header->src_ch_id = static_cast<uint8_t>(CH);
    }
    if (txq_busy(status)) {"""),
    ("""    increment_local_update_ptr_val<SenderChannel<CH>::free_slots_stream>(1);
    std::atomic_signal_fence(std::memory_order_seq_cst);""", """    increment_local_update_ptr_val<SenderChannel<CH>::free_slots_stream>(1);
    order_point();  // the free-slots update stays before the TX-queue wait (V4_SENDER_OWNERSHIP.md §12)"""),
]:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
assert "atomic_signal_fence" not in s
open(p, "w").write(s)
p = f"{R}/router_common.hpp"
s = open(p).read()
o = """FORCE_INLINE void end_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }
"""
n = """FORCE_INLINE void end_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }

// An ordering point outside a read group (std::atomic_signal_fence): the same compiler barrier for memory, where code
// relies on the order of the memory accesses around it but they form no group.
FORCE_INLINE void order_point() { std::atomic_signal_fence(std::memory_order_seq_cst); }
"""
assert s.count(o) == 1
s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
