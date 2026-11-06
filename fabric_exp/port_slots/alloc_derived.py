#!/usr/bin/env python3
"""alloc_derived.py NEW_DIR OLD_DIR [CASE ...]: the staged allocator's rows and the values derived from the receiver
counts, per mock case, for a new build against an old one (SLOT_ALLOCATOR.md section 12.14). Each DIR holds one
directory per case (rc_capture.sh's RC/port, or cov.py's COV/cases) with the slot log (TT_FABRIC_EXP_LOG_SLOTS=1) and the
router builds' named args. Per case and build: the rows (alloc_gate_rows.py's format), and over the case's router
builds the distinct derived values:
  T     BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS (deadlock-avoidance routers only; the host's
        max(2, R - 16))
  ids   NUM_TRANSACTION_IDS (this branch's uniform rule) and, per receiver channel, what the per-channel rule of
        aagarwal/fabric-router-mainline-trid gives (fabric_trid_allocator mirrored here)
  G     the completion group size per receiver channel (1 under first-level acks, else 1 / 2 / 4 for < 8 / 8-15 / 16+)
Prints the cases whose rows or derived values differ between the builds (all cases with --all)."""
import glob
import os
import re
import sys

import contextlib  # noqa: E402
import io  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):  # the sketch prints its table on import
    import alloc_gate_rows as gr  # noqa: E402

DA_ARGS = ("ENABLE_DEADLOCK_AVOIDANCE", "BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS", "NUM_TRANSACTION_IDS",
           "ENABLE_FIRST_LEVEL_ACK_VC0")


def group_size(slots, fla):
    return 1 if fla else (1 if slots < 8 else (2 if slots < 16 else 4))


def rule_ids(receivers):
    """The per-channel rule (fabric_trid_allocator.cpp): one ID per group if all fit in 16, else shrink the largest."""
    ids = [max(1, -(-s // group_size(s, fla))) for s, fla in receivers]
    while sum(ids) > 16:
        k = max(range(len(ids)), key=lambda i: ids[i])
        x = ids[k]
        ids[k] = x // 2 if x & (x - 1) == 0 else 1 << (x.bit_length() - 1)
    return ids


def derived(case_dir):
    out = set()
    rows = gr.rows(case_dir)
    main = [k for k in rows if not gr.is_mux(k)]
    for f in glob.glob(os.path.join(case_dir, "**", "kernels", "fabric_erisc_router", "*", "named_ct_arg_map_generated.h"),
                       recursive=True):
        n = dict((k, int(v)) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(f).read()))
        da = n.get("ENABLE_DEADLOCK_AVOIDANCE", 0)
        fla = n.get("ENABLE_FIRST_LEVEL_ACK_VC0", 0)
        t = n.get("BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS", "-") if da else "-"
        for call in main:
            rx = [(r, bool(fla) if vc == 0 else False) for vc, (tx, rxs, _) in call for r in rxs]
            gs = "/".join(str(group_size(s, a)) for s, a in rx)
            out.add(f"DA {da}: T {t}, uniform ids {n.get('NUM_TRANSACTION_IDS', '-')}, G {gs}, rule ids "
                    f"{'/'.join(map(str, rule_ids(rx)))}")
    return rows, sorted(out)


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    show_all = "--all" in sys.argv
    new, old = args[:2]
    want = args[2:]
    cases = sorted({os.path.basename(p) for d in (new, old) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)})
    print("| case | new rows | old rows | new derived | old derived |")
    print("| --- | --- | --- | --- | --- |")
    for c in cases:
        if want and c not in want:
            continue
        rn, dn = derived(os.path.join(new, c))
        ro, do = derived(os.path.join(old, c))
        fn = "; ".join(f"{gr.fmt(k)} x{v}" for k, v in sorted(rn.items()) if not gr.is_mux(k)) or "-"
        fo = "; ".join(f"{gr.fmt(k)} x{v}" for k, v in sorted(ro.items()) if not gr.is_mux(k)) or "-"
        if show_all or fn != fo or dn != do:
            print(f"| {c} | {fn} | {fo} | {'; '.join(dn)} | {'; '.join(do)} |")


if __name__ == "__main__":
    main()
