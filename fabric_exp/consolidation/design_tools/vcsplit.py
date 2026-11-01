import sys, re, collections
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
def cls(fns, e):
    s = " ".join(fns)
    if e == 0:
        if re.search(r"(visit_worker|send|credit_worker|worker_connection_event)<4u|holds_packet<4u", s): return "VC2 worker (visit, its free-slots test)"
        if "dw_inline" in s: return "VC0 worker credit (inline write)"
        if re.search(r"sender::send<0u", s): return "VC0 send"
        if re.search(r"visit_worker<0u", s): return "VC0 worker visit (checks, credit)"
        if re.search(r"read_free_slots_registers|channels_holding_packets|holds_packet|channels_with_packets|read_stream_raw", s): return "free-slots group (VC0 part)"
        if "visit_forwarding" in s: return "VC0 forwarding visit"
        return "loop, round, other"
    else:
        if re.search(r"receiver::(step|receive)<1u|arrival_register<1u", s): return "VC2 receiver step (incl. its register read)"
        if re.search(r"receiver::(step|receive|handle|retire)<0u|deliver<|local_write|release_completions<4ul, 16ul|release_completions<1ul, 16ul|acknowledge", s): return "VC0 receiver step"
        if re.search(r"send_credit_frame|send_packet|flush_completion", s): return "credit frame"
        if re.search(r"arrival_register|read_txq|read_stream_raw", s): return "pass register group"
        return "loop, round, other"
def split(path):
    lines = open(path).read().splitlines()
    twin = lines[0].split()[1]; e = 0 if " E0 " in lines[0] else 1
    rows = [(int(l[:8], 16), float(l.split()[1])) for l in lines[1:]]
    loc = ps.a2l(twin, [a for a, _ in rows])
    c = collections.Counter()
    for a, v in rows:
        c[cls([f for f, _, _ in loc.get(a, [])], e)] += v
    return c, e
for off, on in [(sys.argv[1], sys.argv[2])]:
    A, e = split(off); B, _ = split(on)
    print(f"  E{e}: {'section':45} {'off':>7} {'on':>7} {'on-off':>7}")
    for k in sorted(set(A) | set(B), key=lambda k: -(B[k] - A[k])):
        print(f"      {k:45} {A[k]:7.1f} {B[k]:7.1f} {B[k]-A[k]:+7.1f}")
    print(f"      {'total':45} {sum(A.values()):7.1f} {sum(B.values()):7.1f} {sum(B.values())-sum(A.values()):+7.1f}")
