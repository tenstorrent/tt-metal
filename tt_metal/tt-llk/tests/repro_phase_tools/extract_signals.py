#!/usr/bin/env python3
"""Copy the ~60 signals the repro analysis needs out of a Versim VCD (several GB) into a small change list.

usage: extract_signals.py <1-1-core_dump.vcd> <out.txt>
Each output line is '<time> <label> <value>'; one clock cycle is 2 time units.
"""
import sys

P = "TOP_01_01.tt_tensix_with_l1.tensix."
TD = P + "instruction_thread0.instruction_issue.tdma"
SPECS = [
    ("clk", P + "trisc(2).u_trisc", "i_clk"),
    (
        "rden",
        TD,
        "tdma_dstac_regif_rden",
    ),  # DEST read request, one bit per packer (bit 0 = packer 0)
    (
        "ready",
        TD,
        "dstac_regif_tdma_reqif_ready",
    ),  # DEST read granted, one bit per packer
    ("dvld", TD, "i_dstac_regif_tdma_data_vld"),
]
SPECS += [
    (k, TD, k)
    for k in (
        "l1_arbiter_wrvalid",
        "l1_arbiter_win_vector",
        "l1_arbiter_packet_accept",
        "i_l1_tdma_pack_reqif_ready",
        "i_l1_tdma_reqif_ready",
    )
]
for k in range(4):
    pk = TD + f".gen_pack_instance({k}).packer"
    SPECS += [
        (f"p{k}_l1_wren", pk, "o_l1_wren"),
        (f"p{k}_l1_ready", pk, "i_l1_req_ready"),
        (f"busy{k}", pk, "o_busy"),
    ]
for t in range(3):  # TRISC0 unpack, TRISC1 math, TRISC2 pack
    s = P + f"trisc({t}).u_trisc"
    SPECS += [
        (f"t{t}_l1_rden", s, "o_l1_rden"),
        (f"t{t}_l1_wren", s, "o_l1_wren"),
        (f"t{t}_l1_addr", s, "o_l1_rdaddr"),
        (f"t{t}_l1_ready", s, "i_l1_req_ready"),
        (f"t{t}_ibuf_wren", s, "o_trisc_instrn_buf_wren"),
        (f"t{t}_ibuf_ready", s, "i_trisc_instrn_buf_req_ready"),
    ]

want = {(scope, var): label for label, scope, var in SPECS}
ids, stack = {}, []
with open(sys.argv[1], "rb") as f, open(sys.argv[2], "w") as w:
    for raw in f:
        line = raw.decode("latin1").strip()
        if line.startswith("$scope"):
            stack.append(line.split()[2])
        elif line.startswith("$upscope"):
            stack.pop()
        elif line.startswith("$var"):
            t = line.split()
            label = want.get((".".join(stack), t[4]))
            if label:
                ids.setdefault(t[3], []).append(label)
        elif line.startswith("$enddefinitions"):
            break
    found = {l for ls in ids.values() for l in ls}
    missing = sorted(set(want.values()) - found)
    if missing:
        sys.exit(f"signals not found in the VCD: {missing}")
    now = "0"
    for raw in f:
        c = raw[:1]
        if c == b"#":
            now = raw[1:].strip().decode()
        elif c == b"b":
            val, _, i = raw[1:].strip().partition(b" ")
            for label in ids.get(i.decode("latin1"), ()):
                w.write(f"{now} {label} {val.decode()}\n")
        elif c in (b"0", b"1", b"x", b"z"):
            for label in ids.get(raw[1:].strip().decode("latin1"), ()):
                w.write(f"{now} {label} {c.decode()}\n")
