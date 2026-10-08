#!/usr/bin/env python3
"""ext.py <vcd or -> <out.txt>: crossbar, packer L1, per-TRISC fetch and NoC L1 signals as '<time> <label> <value>' (1 cycle = 2 time units)."""
import sys

P = "TOP_01_01.tt_tensix_with_l1."
T = P + "tensix."
II = T + "instruction_thread0.instruction_issue"
TD = II + ".tdma"
NOC = P + "overlay_noc_nius_routers"
SPECS = [
    ("clk", T + "trisc(2).u_trisc", "i_clk"),
    ("xb_rden", II, "dma_rdrow_xbar_rden"),
    ("xb_ready", II, "dma_rdrow_xbar_ready"),
    ("xb_addr", II, "dma_rdrow_xbar_rd_addr"),
    ("noc_l1_rden", NOC, "o_mem_out_rden"),
    ("noc_l1_rdy", NOC, "i_mem_out_reqif_ready"),
    ("noc_l1_in_rden", NOC, "o_mem_in_rden"),
    ("noc_reg_rden", NOC, "noc_niu_reg_rd_en"),
]
for k in range(4):
    pk = TD + f".gen_pack_instance({k}).packer"
    SPECS += [
        (f"p{k}_l1_wren", pk, "o_l1_wren"),
        (f"p{k}_l1_ready", pk, "i_l1_req_ready"),
    ]
for t in range(3):
    U = T + f"trisc({t}).u_trisc"
    C = U + ".u_trisc_control"
    CPU = C + ".briscv"
    IF = CPU + ".ifetch"
    IC = C + ".u_icache"
    SPECS += [
        (f"t{t}_cmt_pc", CPU, "dbg_obs_cmt_pc"),
        (f"t{t}_mp", CPU, "ex_bp_mispredict"),
        (f"t{t}_ibuf_wren", U, "o_trisc_instrn_buf_wren"),
        (f"t{t}_ibuf_ready", U, "i_trisc_instrn_buf_req_ready"),
        (f"t{t}_l1_rden", U, "o_l1_rden"),
        (f"t{t}_l1_wren", U, "o_l1_wren"),
        (f"t{t}_l1_addr", U, "o_l1_rdaddr"),
        (f"t{t}_ic_rden", IC, "o_l1_rden"),
        (f"t{t}_ic_addr", IC, "o_l1_rdaddr"),
        (f"t{t}_ic_req", IC, "perf_cnt_req"),
        (f"t{t}_ic_hit", IC, "perf_cnt_hit"),
        (f"t{t}_ic_mshr", IC, "perf_cnt_mshr"),
        (f"t{t}_type_rand", IF + ".predict_instrn_type", "rand_addr"),
    ]
want = {(s, v): l for l, s, v in SPECS}
f = sys.stdin.buffer if sys.argv[1] == "-" else open(sys.argv[1], "rb")
ids, stack = {}, []
with open(sys.argv[2], "w") as w:
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
                ids.setdefault(t[3].encode("latin1"), []).append(label)
        elif line.startswith("$enddefinitions"):
            break
    missing = sorted(set(want.values()) - {l for ls in ids.values() for l in ls})
    if missing:
        print("missing:", missing, file=sys.stderr)
    now = 0
    for raw in f:
        c = raw[:1]
        if c == b"#":
            now = int(raw[1:])
            continue
        if c == b"b":
            val, _, i = raw[1:].strip().partition(b" ")
        elif c in b"01xz" and c:
            val, i = raw[:1], raw[1:].strip()
        else:
            continue
        for label in ids.get(i, ()):
            w.write(f"{now} {label} {val.decode()}\n")
