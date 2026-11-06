#!/usr/bin/env python3
"""arms.py TREE ARM: apply one placement-audit arm to the router sources of TREE (a worktree checked out at the base).
Each edit removes exact lines, checked against the base text first. ARM 'base' changes nothing."""
import sys

R = "tt_metal/fabric/hw/inc/edm_fabric/router/"
# arm -> list of (file, line number at the base 7cd3c13a0db, exact stripped text)
SITES = {
    "rg_s537": [("router_sender.hpp", 537, "begin_read_group();"), ("router_sender.hpp", 540, "end_read_group();")],
    "rg_s583": [("router_sender.hpp", 583, "begin_read_group();"), ("router_sender.hpp", 586, "end_read_group();")],
    "rg_s605": [("router_sender.hpp", 605, "begin_read_group();"), ("router_sender.hpp", 614, "end_read_group();")],
    "rg_s673": [("router_sender.hpp", 673, "begin_read_group();"), ("router_sender.hpp", 682, "end_read_group();")],
    "rg_r755": [("router_receiver.hpp", 755, "begin_read_group();"), ("router_receiver.hpp", 764, "end_read_group();")],
    "rg_d414": [("router_downstream.hpp", 414, "begin_read_group();"), ("router_downstream.hpp", 416, "end_read_group();")],
    "op_s617": [("router_sender.hpp", 617, "order_point();  // the free-slots update stays before the TX-queue wait")],
    "kim_s755": [("router_sender.hpp", 755, "keep_in_memory(state);")],
    "kim_s816": [("router_sender.hpp", 816, "keep_in_memory(send_state);")],
    "kim_r713": [("router_receiver.hpp", 713, "keep_in_memory(this->template get<VC>());")],
}
SITES["rg_all"] = sum((SITES[a] for a in ("rg_s537", "rg_s583", "rg_s605", "rg_s673", "rg_r755", "rg_d414")), [])

tree, arm = sys.argv[1], sys.argv[2]
if arm == "base":
    sys.exit(0)
by_file = {}
for f, n, text in SITES[arm]:
    by_file.setdefault(f, []).append((n, text))
for f, edits in by_file.items():
    p = f"{tree}/{R}{f}"
    lines = open(p).read().split("\n")
    for n, text in edits:
        assert lines[n - 1].strip() == text, (p, n, lines[n - 1])
    for n, _ in sorted(edits, reverse=True):
        del lines[n - 1]
    open(p, "w").write("\n".join(lines))
    print(f"{arm}: {f} -{len(edits)} lines")
