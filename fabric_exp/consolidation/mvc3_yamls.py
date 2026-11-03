#!/usr/bin/env python3
"""mvc3_yamls.py: the multi-VC throughput rows on the VC1 split (P150x8 as two 2x2 meshes; report multi-VC page,
2026-11-02), written to fabric_exp/vc1/mvc3/. Run mvc3_runs.sh to measure them.

One link inside mesh 1 next to the landing chip: [1,0] -> [1,1] (E/W). On a09u32 since the merge [0,0] lands on [1,0]
(SLOT_STUDY.md P0), so [0,0] -> [1,1] crosses the inter-mesh link to [1,0] and then one VC1 hop on this link.
  F0  VC0 from a local worker: [1,0] -> [1,1], vc_id 0 (rank 1)
  F1  VC1 forwarded in from mesh 0: [0,0] -> [1,1] (rank 0; VC0 in mesh 0, VC1 on [1,0] -> [1,1])
  F2  VC2 neighbour exchange from a local worker: [1,0] -> [1,1], vc_id 2 (rank 1)
Method (as consolidation/multivc_runs.sh): benchmark mode, sync off (each sender times its own send loop), per-sender
rates from the profiler's log lines. A VC's rate under contention comes from a row where that VC finishes first (its
competitors are given enough packets to outlast it), so every packet it times was sent while the others also sent.
Cross-rank start order: F1 runs in the other process, and without sync nothing orders the two processes' launches. The
rows are grouped so that the competitors always start first:
  base  rows with no cross-rank competitor (each VC alone, F0 + F2): no delay
  lag1  F0 or F2 measured, F1 competing: mesh 1's process launches D ms later (TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS;
        the standard rank bindings: every process but launch rank 0 waits)
  lag0  F1 measured, F0 and/or F2 competing: mesh 0's process launches D ms later (rank bindings with TT_RUN_RANK
        swapped, so diag_launch_delay picks mesh 0's process; TT_RUN_RANK is read only by it and by tt-triage)
Packet counts: the measured flow N_m = T_ALONE / a_m (about T_ALONE cycles alone); each competitor
N_c = MARGIN * (D_c + SLOWDOWN * N_m * a_m) / a_c, where a_* are fast estimates of each flow's rate alone (cycles per
packet; a competitor runs at most this fast), D_c the competitor's head start in cycles (D for a cross-rank competitor in
a lag file, else 0), SLOWDOWN a bound on the measured flow's slowdown under contention. Whether the measured flow did
finish first is checked from the logs (build_multivc3.py), not assumed.
Sets: A_d (new router v4rxg3v with VC2 compiled in, default allocation), B_main / B_new (VC0 + VC1 only). A_v8 (VC2 8 + 10)
was dropped: the host refused it at -O3 (job 479, 16 B over the kernel config buffer); user: default allocation only."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vc1", "mvc3")
SIZES = [256, 1024, 4352]
CLOCK = 1.35e9
D_MS = 200
D_CYC = D_MS * 1e-3 * CLOCK
T_ALONE = 0.4e9
SLOWDOWN = 3.0
MARGIN = 1.5
FLOWS = {  # flow: (device, vc_id or None)
    0: ([1, 0], 0),
    1: ([0, 0], None),
    2: ([1, 0], 2),
}
RANK = {0: 1, 1: 0, 2: 1}  # the MPI rank (mesh) each flow's sender runs in
# Fast estimates of each flow's rate alone (cycles per packet), per set (the lead's rows, jobs 323 / 376-388 / 469-472;
# census main_fw VC1 inter rows)
EST = {
    "A_d": {0: 130, 1: 400, 2: 740},
    "B_new": {0: 130, 1: 380},
    "B_main": {0: 180, 1: 900},
}


def rows(vcs):
    """(name, measured flow or None, flows, group) for the flows `vcs`"""
    out = [(f"V{v}", v, [v], "base") for v in vcs]
    pairs = [(a, b) for i, a in enumerate(vcs) for b in vcs[i + 1:]]
    for a, b in pairs:
        for m in (a, b):
            other = b if m == a else a
            cross = RANK[m] != RANK[other]
            grp = "base" if not cross else ("lag0" if RANK[m] == 0 else "lag1")
            out.append((f"P{a}{b}m{m}", m, [a, b], grp))
    if len(vcs) == 3:
        for m in vcs:
            out.append((f"T012m{m}", m, list(vcs), "lag0" if RANK[m] == 0 else "lag1"))
    return out


def counts(est, m, flows, grp):
    n_m = round(T_ALONE / est[m] / 1e4) * 1e4
    res = {m: int(n_m)}
    for c in flows:
        if c == m:
            continue
        head = D_CYC if (grp != "base" and RANK[c] != RANK[m]) else 0.0
        n_c = MARGIN * (head + SLOWDOWN * n_m * est[m]) / est[c]
        res[c] = int(math.ceil(n_c / 1e5) * 1e5)
    return res


def test_yaml(name, flows, cnt, vc2):
    lines = [f'  - name: "{name}"', "    benchmark_mode: true",
             "    sync: false  # each sender times its own send loop (mvc3_yamls.py)", "    fabric_setup:",
             "      topology: Mesh"]
    if vc2:
        lines.append("      use_vc2: true")
    lines += ["    parametrization_params:", "      ntype: [unicast_write]", f"      size: {SIZES}", "    defaults:",
              "      ftype: unicast", "    senders:"]
    for f in flows:
        dev, vc = FLOWS[f]
        lines.append(f"      - device: {dev}")
        if vc is not None:
            lines.append(f"        vc_id: {vc}")
        lines += ["        patterns:", "          - destination:", "              device: [1, 1]",
                  f"            num_packets: {cnt[f]}"]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    plan = []
    for s, est in EST.items():
        vcs = [0, 1, 2] if s.startswith("A") else [0, 1]
        groups = {}
        for name, m, flows, grp in rows(vcs):
            cnt = counts(est, m, flows, grp) if len(flows) > 1 else {m: int(round(T_ALONE / est[m] / 1e4) * 1e4)}
            groups.setdefault(grp, []).append(test_yaml(f"MVC3_{s}_{name}", flows, cnt, s.startswith("A")))
            longest = max(cnt[f] * est[f] for f in flows) + (D_CYC if grp != "base" else 0)
            plan.append(f"{s:7s} {grp:5s} {name:8s} measured F{m} " +
                        " ".join(f"F{f}={cnt[f]}" for f in flows) + f"  longest flow alone-est {longest / CLOCK:.2f} s")
        for grp, tests in groups.items():
            p = os.path.join(OUT, f"mvc3_{s}_{grp}.yaml")
            with open(p, "w") as f:
                f.write(f"# Generated by fabric_exp/consolidation/mvc3_yamls.py (set {s}, group {grp}; see its docstring).\n"
                        f"# Run via consolidation/mvc3_runs.sh. Group {grp}: " +
                        {"base": "no launch delay", "lag1": f"mesh 1 launches {D_MS} ms later (standard rank bindings)",
                         "lag0": f"mesh 0 launches {D_MS} ms later (rank bindings with TT_RUN_RANK swapped)"}[grp] +
                        ".\nallocation_policies:\n  receiver:\n    policy: RoundRobin\n    max_configs_per_core: 4\n"
                        "Tests:\n" + "\n\n".join(tests) + "\n")
    open(os.path.join(OUT, "plan.txt"), "w").write("\n".join(plan) + "\n")
    print("\n".join(plan))


if __name__ == "__main__":
    main()
