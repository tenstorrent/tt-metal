#!/usr/bin/env python3
"""gen_ring_yaml.py OUT: the VC2 ring tests on one Blackhole Galaxy torus (8x4: rows 0-7 are the N/S axis, a ring of 8
on a Y torus; columns 0-3 the E/W axis, a ring of 4 on an X torus). VC2_LINE_BRINGUP.md, phase 2.

Per torus config (X only, Y only, XY), every test data-checked with flow control on, VC2 in line mode (vc2_line):
  WrapUnicast    VC2 unicasts whose shortest path takes the wrap link, and one that does not (no flow is exactly
                 half a ring away: the test infra and the routing table may break that tie differently)
  FullRingMcast  VC2 multicast around a whole ring, from a chip whose ring wraps on the way
  AllChips       every chip of a ring multicasts around the whole ring, both directions at once (bidirectional; the
                 cyclic load bubble flow control is for)
  WithVC0        the same VC2 load next to VC0 torus traffic on the same rings (full-ring multicasts, wrapping and
                 turning unicasts), from other worker cores
  Line           on X-only and Y-only tori: VC2 lines on the axis that is not a ring
VC0 flows set vc_id 0 explicitly; VC2 flows are all straight (the test infra refuses a VC2 flow that turns).
"""
import sys

ROWS, COLS = 8, 4
X_RING, Y_RING = COLS, ROWS


def sender(row, col, vc, dest, ftype="unicast"):
    pattern = {"ftype": ftype}
    if "device" in dest:
        pattern["destination"] = {"device": [0, list(dest["device"])]}
    else:
        pattern["destination"] = {"hops": dest["hops"]}
    return {"device": [0, [row, col]], "vc_id": vc, "patterns": [pattern]}


def test(name, torus, senders, num_packets, sizes, ntypes=("unicast_write",)):
    return {
        "name": name,
        "enable_flow_control": True,
        "fabric_setup": {"topology": "Torus", "torus_config": torus, "vc2_line": True},
        "parametrization_params": {"size": list(sizes), "ntype": list(ntypes)},
        "defaults": {"ftype": "unicast", "num_packets": num_packets},
        "senders": senders,
    }


def x_ring_tests(torus, row_a=0, row_b=1):
    t = []
    t.append(test(f"VC2RingX_{torus}_WrapUnicast", torus, [
        sender(row_a, 3, 2, {"device": (row_a, 0)}),        # E 1 hop over the wrap link
        sender(row_b, 0, 2, {"device": (row_b, 3)}),        # W 1 hop over the wrap link
        sender(row_a + 2, 1, 2, {"device": (row_a + 2, 2)}),  # 1 hop, no wrap (no ties: half a ring away is one)
    ], 5000, (1024, 4096), ("unicast_write", "fused_atomic_inc")))
    t.append(test(f"VC2RingX_{torus}_FullRingMcast", torus, [
        sender(row_a, 1, 2, {"hops": {"E": X_RING - 1}}, "mcast"),
        sender(row_b, 2, 2, {"hops": {"W": X_RING - 1}}, "mcast"),
    ], 5000, (1024, 4096)))
    t.append(test(f"VC2RingX_{torus}_AllChips", torus,
                  [sender(row_a, c, 2, {"hops": {"E": X_RING - 1}}, "mcast") for c in range(COLS)] +
                  [sender(row_a, c, 2, {"hops": {"W": X_RING - 1}}, "mcast") for c in range(COLS)],
                  2000, (4096,)))
    t.append(test(f"VC2RingX_{torus}_WithVC0", torus,
                  [sender(row_b, c, 2, {"hops": {"E": X_RING - 1}}, "mcast") for c in range(COLS)] +
                  [sender(row_b, c, 0, {"hops": {"E": X_RING - 1}}, "mcast") for c in range(COLS)] +
                  [sender(row_b, 3, 0, {"device": (row_b, 0)}),                 # VC0 over the wrap link
                   sender(row_b, 0, 0, {"device": (row_b + 3, 2)})],            # VC0 turning
                  2000, (4096,)))
    return t


def y_ring_tests(torus, col_a=0, col_b=1):
    t = []
    t.append(test(f"VC2RingY_{torus}_WrapUnicast", torus, [
        sender(6, col_a, 2, {"device": (1, col_a)}),   # S 3 hops over the wrap link
        sender(1, col_b, 2, {"device": (6, col_b)}),   # N 3 hops over the wrap link
        sender(0, col_a + 2, 2, {"device": (2, col_a + 2)}),  # 2 hops, no wrap (no ties: half a ring away is one)
    ], 5000, (1024, 4096), ("unicast_write", "fused_atomic_inc")))
    t.append(test(f"VC2RingY_{torus}_FullRingMcast", torus, [
        sender(3, col_a, 2, {"hops": {"S": Y_RING - 1}}, "mcast"),
        sender(5, col_b, 2, {"hops": {"N": Y_RING - 1}}, "mcast"),
    ], 5000, (1024, 4096)))
    t.append(test(f"VC2RingY_{torus}_AllChips", torus,
                  [sender(r, col_a, 2, {"hops": {"S": Y_RING - 1}}, "mcast") for r in range(ROWS)] +
                  [sender(r, col_a, 2, {"hops": {"N": Y_RING - 1}}, "mcast") for r in range(ROWS)],
                  2000, (4096,)))
    t.append(test(f"VC2RingY_{torus}_WithVC0", torus,
                  [sender(r, col_b, 2, {"hops": {"S": Y_RING - 1}}, "mcast") for r in range(ROWS)] +
                  [sender(r, col_b, 0, {"hops": {"S": Y_RING - 1}}, "mcast") for r in range(ROWS)] +
                  [sender(6, col_b, 0, {"device": (1, col_b)}),                 # VC0 over the wrap link
                   sender(0, col_b, 0, {"device": (5, (col_b + 2) % COLS)})],   # VC0 turning
                  2000, (4096,)))
    return t


def line_tests(torus, axis):
    # The axis that is not a ring: VC2 goes straight from one end to the other, both ways.
    if axis == "Y":
        s = [sender(0, 0, 2, {"device": (ROWS - 1, 0)}), sender(ROWS - 1, 1, 2, {"device": (0, 1)}),
             sender(0, 2, 2, {"hops": {"S": ROWS - 1}}, "mcast")]
    else:
        s = [sender(0, 0, 2, {"device": (0, COLS - 1)}), sender(1, COLS - 1, 2, {"device": (1, 0)}),
             sender(2, 0, 2, {"hops": {"E": COLS - 1}}, "mcast")]
    return [test(f"VC2Line{axis}_{torus}", torus, s, 5000, (4096,))]


def main(out):
    import yaml
    tests = []
    tests += x_ring_tests("X") + line_tests("X", "Y")
    tests += y_ring_tests("Y") + line_tests("Y", "X")
    tests += x_ring_tests("XY") + y_ring_tests("XY")
    hdr = ("# Generated by gen_ring_yaml.py (VC2 rings on one Blackhole Galaxy torus, VC2_LINE_BRINGUP.md phase 2).\n"
           "# Galaxy 8x4: device [0, [row, col]]; rows 0-7 (N/S, a ring of 8 on a Y torus), columns 0-3 (E/W, a ring of 4\n"
           "# on an X torus). VC2 in line mode (vc2_line): rings on torus axes, lines on the other axis.\n")
    with open(out, "w") as f:
        f.write(hdr)
        yaml.safe_dump({"Tests": tests}, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"{len(tests)} tests -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
