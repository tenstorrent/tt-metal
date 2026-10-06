#!/usr/bin/env python3
"""Copy the pack RISC-V (TRISC2) fetch, icache and branch predictor signals out of a Versim VCD.

usage: extract_fetch.py <vcd or - for stdin> <out.txt> [tmin tmax]
Each output line is '<time> <label> <value>'; one clock cycle is 2 time units.
Parameters (constant values) are written once with time 'param'.
"""
import sys

T2 = "TOP_01_01.tt_tensix_with_l1.tensix.trisc(2).u_trisc"
C = T2 + ".u_trisc_control"
CPU, IF, IC = C + ".briscv", C + ".briscv.ifetch", C + ".u_icache"
SPECS = [
    ("clk", T2, "i_clk"),
    ("ibuf_wren", T2, "o_trisc_instrn_buf_wren"),
    ("ibuf_ready", T2, "i_trisc_instrn_buf_req_ready"),
    ("cmt_pc", CPU, "dbg_obs_cmt_pc"),  # PC of the instruction that commits
    ("ex_mispredict", CPU, "ex_bp_mispredict"),
    ("ex_taken", CPU, "ex_branch_taken"),
    ("ex_branch_pc", CPU, "ex_branch_pc"),
    ("if_pc", IF, "pc"),
    ("if_mispredict", IF, "mispredict"),
    ("bp_taken", IF, "bp_taken"),
    ("bp_addr", IF, "bp_taken_addr"),
    ("req_stall", IF, "request_stall"),
    ("bp_stall", IF, "bp_fetch_stall"),
    ("ififo_empty", IF, "instrn_fifo_empty"),
    ("imem_req", C, "rv_out_imem_req"),
    ("imem_addr", C, "rv_out_imem_addr"),
    ("imem_gnt", C, "rv_in_imem_gnt"),
    ("imem_vld", C, "rv_in_imem_rvalid"),
    ("ic_l1_rden", IC, "o_l1_rden"),
    ("ic_l1_addr", IC, "o_l1_rdaddr"),
    ("ic_l1_ready", IC, "i_l1_req_ready"),
    ("ic_l1_vld", IC, "i_l1_data_valid"),
    ("ic_mshr_miss", IC, "mshr_miss"),
    ("ic_pf_stall", IC, "prefetch_stall"),
    ("ic_pf_vld", IC, "prefetch_vld"),
    ("ic_pf_addr", IC, "prefetch_addr"),
    ("ic_cnt_req", IC, "perf_cnt_req"),
    ("ic_cnt_hit", IC, "perf_cnt_hit"),
    ("ic_cnt_stall", IC, "perf_cnt_stall"),
    ("ic_cnt_mshr", IC, "perf_cnt_mshr"),
    ("bp_pred_pc", IF + ".bp", "bp_prediction_pc"),
    ("bp_cnt", IF + ".bp", "curr_cnt_val"),
    ("bp_upd", IF + ".bp", "i_ex_update_vld"),
    ("bp_lookup_pc", IF + ".bp", "i_pc_addr"),
    ("bp_lookup_req", IF + ".bp", "i_pc_req"),
    ("bp_vld", IF + ".bp", "o_pc_bp_vld"),
    ("type_hit", IF + ".predict_instrn_type", "hit"),
    ("type_rand", IF + ".predict_instrn_type", "rand_addr"),
    ("type_upd", IF + ".predict_instrn_type", "i_update_vld"),
    ("type_upd_pc", IF + ".predict_instrn_type", "i_update_pc"),
]
SPECS += [(f"bp{k}", IF + ".bp", f"bp_array({k})") for k in range(16)]
SPECS += [(f"tag{k}", IF + ".predict_instrn_type", f"tags({k})") for k in range(16)]
SPECS += [(f"deco{k}", IF + ".predict_instrn_type", f"deco({k})") for k in range(16)]
PARAMS = [(f"{s.rsplit('.', 1)[1]}.{v}", s, v) for s, v in [
    (IF + ".bp", "BP_DEPTH"), (IF + ".bp", "BP_ADDR_OFFSET_WIDTH"), (IF + ".bp", "BP_CNT_WIDTH"),
    (IF + ".bp", "BP_ENTRY_WIDTH"), (IF + ".bp", "ADDR_ITEM_WIDTH"), (IC, "MSHR_COUNT"),
    (IC, "LINE_WIDTH"), (IC, "LINE_ADDR_WIDTH"), (IC, "WAY_COUNT"), (C, "LARGE_TRISC_ICACHE"),
    (C, "i_disable_bp"), (C, "i_icache_prefetch_en"), (C, "i_prefetch_max_req"),
]]
want = {(s, v): l for l, s, v in SPECS + PARAMS}
params = {l for l, _, _ in PARAMS}
tmin, tmax = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1 << 62)
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
        sys.exit(f"signals not found in the VCD: {missing}")
    now, seen, last, started = 0, set(), {}, False
    for raw in f:
        c = raw[:1]
        if c == b"#":
            now = int(raw[1:])
            if not started and now >= tmin:
                started = True
                for label, v in sorted(last.items()):
                    w.write(f"{tmin} {label} {v}\n")
            if now > tmax and len(seen) == len(params):
                break
            continue
        if c == b"b":
            val, _, i = raw[1:].strip().partition(b" ")
        elif c in b"01xz" and c:
            val, i = raw[:1], raw[1:].strip()
        else:
            continue
        for label in ids.get(i, ()):
            if label in params:
                if label not in seen:
                    seen.add(label)
                    w.write(f"param {label} {int(val, 2) if val.isdigit() else val.decode()}\n")
            elif now < tmin:
                last[label] = val.decode()
            elif now <= tmax:
                w.write(f"{now} {label} {val.decode()}\n")
