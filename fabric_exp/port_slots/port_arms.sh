#!/bin/bash
# The port slot measurements' arms (SLOT_ALLOCATOR.md section 12.12): "SET ARM KIND SUITE ROW [ENV]" per line.
# KIND 2d / 1d (the row goes to TT_FABRIC_EXP_SLOTS_2D / _1D) or val (default slots, data-checked); ROW W,F,R or -;
# ENV extra NAME=VALUE pairs joined by ";". Suites in fabric_exp/port_slots/yamls (mk_port_yamls.py).
port_arms() {
    local mgd="TT_MESH_GRAPH_DESC_PATH=/home/aagarwal/tt-metal/fabric_exp/consolidation/slotalloc/knee/merge/p150_x8_4x2_mesh_mgd.textproto"
    # Data-checked sanity first (default slots: the port's tier table).
    echo "V val2d3k val pval2d_3072 - -"
    echo "V val1d3k val pval1d_3072 - -"
    echo "V val2d4k val pval2d_4352 - -"
    # Measurement 1: the VC0 forwarding knee at 3072 B slots. 2D: W 20, R 16, F 12 .. 26 (the 115-slot budget);
    # the forwarding rows on the 2x4 mesh, and N/S 3-hop on the 4x2 mesh. 1D: W 24, R 32, F 16 .. 56.
    for f in 12 14 16 18 20 22 24 26; do
        echo "F2 f$f 2d pf2d_3072 20,$f,16 -"
        echo "F2 f${f}ns 2d pf2dns_3072 20,$f,16 $mgd"
    done
    for f in 16 24 32 40 48 56; do echo "F1 f$f 1d pf1d_3072 24,$f,32 -"; done
    # Measurement 2: the same depths at 3072 and 4352 B slots, 256-2048 B packets (D2's pair for the receiver and
    # forwarding). Receiver: W 24, F 8 (A2), R 8 .. 24. Forwarding: W 20, R 16 (A3), F 8, 12, 15 (fit both budgets).
    for slot in 3072 4352; do
        for r in 8 12 16 24; do echo "S r${r}_$slot 2d pw2d_$slot 24,8,$r -"; done
        for f in 8 12 15; do echo "S f${f}_$slot 2d pf2ds_$slot 20,$f,16 -"; done
    done
}
