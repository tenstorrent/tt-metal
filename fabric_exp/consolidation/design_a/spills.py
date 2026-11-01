#!/usr/bin/env python3
"""spills.py ELF...: per receiver ELF (rxlist.py's view), the loop function's instructions and sp accesses (fn / loop),
and the loop's named spills of interest: the scatter chunk encoding stored after its load (router_local_delivery.hpp:102),
`me` reloads (router_routing.hpp:84-88), the NoC status base reloaded for the completion proof
(noc_nonblocking_api.h:184), the TX status (send_credit_frame's register) stored, and the loop's sp accesses by
innermost source file."""
import collections, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rxcmp
HERE = os.path.dirname(os.path.abspath(__file__))
for e in sys.argv[1:]:
    out = subprocess.run([sys.executable, f"{HERE}/rxlist.py", e], capture_output=True, text=True).stdout.splitlines()
    loop_sp = [l for l in out if l[7:9] == "LS"]
    st = rxcmp.rx_stats(rxcmp.functions(e))
    c = lambda pat: sum(1 for l in loop_sp if re.search(pat, l))
    files = collections.Counter(l.split()[4 if len(l.split()) > 4 else -1].split(":")[0] for l in loop_sp)
    print(f"{e.split('/')[-4]:16s} fn {st[0]} insns {st[1] + st[2]} sp | loop {st[3]} insns {st[4] + st[5]} sp | "
          f"enc-spill {c(r'sw .*router_local_delivery.hpp:102')} me {c(r'lw .*router_routing.hpp:8[4-8]')} "
          f"status-base {c(r'lw .*noc_nonblocking_api.h:184')}")
