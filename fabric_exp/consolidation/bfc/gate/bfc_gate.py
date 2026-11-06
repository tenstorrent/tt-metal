#!/usr/bin/env python3
"""bfc_gate.py BASE_COV NEW_COV [CASE ...]: the JIT's own ELFs of two mock captures (cov.py mock, one per host build),
paired by their build inputs with the injection-threshold arg removed, compared on every loadable section.

A build's inputs: its kernel name and processor, its compile command's -D defines and flags (paths and the build hash
dropped), and its named compile-time args (named_ct_arg_map_generated.h) without
BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS. A build of one capture with no partner in the other means
the host changed something else (or the capture logged different builds): listed.

Per case: router images identical / differ, other images identical / differ, unpaired, and the new arg's values on the
new side (with the far VC0 receiver's slots, from the router's log line when the capture logged at debug level)."""
import collections, glob, json, os, re, sys

sys.path.insert(0, "/home/aagarwal/tt-metal-mainline/.scratch")
import cov  # noqa: E402

ARG = "BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS"


def builds(covdir, case):
    p = os.path.join(covdir, "cases", case, "cmds.json")
    if not os.path.exists(p):
        return None
    db = json.load(open(p))
    out = {}
    for key, cmds in db.items():
        link = cmds["link"][0]
        m = re.search(r"\s-o\s+(\S+\.elf)", link)
        if not m:
            continue
        elf = os.path.join(os.path.dirname(m.group(1)), os.path.basename(m.group(1)).split(".")[0] + ".elf")
        if not os.path.exists(elf):
            continue
        bdir = re.search(r"(\S*/" + re.escape(key) + r")/", link).group(1)  # .../kernels/<name>/<hash>/<proc>
        hdir = os.path.dirname(bdir)
        named = ""
        nf = os.path.join(hdir, "named_ct_arg_map_generated.h")
        if os.path.exists(nf):
            named = open(nf).read()
        value = None
        m2 = re.search(r'\{"' + ARG + r'",(\d+)\}', named)
        if m2:
            value = int(m2.group(1))
        stripped = re.sub(r',?\{"' + ARG + r'",\d+\}', "", named)
        comp = cmds["compile"][0]
        flags = []
        for x in comp.split(" "):
            if x.startswith("-DFULL_KERNEL_NAME=") or x.startswith("-include") or "/" in x:
                continue
            flags.append(x)
        name, proc = key.split("/")[1], key.split("/")[-1]
        k = (name, proc, " ".join(flags), stripped)
        out.setdefault(k, []).append((key, elf, value))
    return out


def main():
    base_cov, new_cov = sys.argv[1], sys.argv[2]
    cases = sys.argv[3:] or sorted(set(os.listdir(os.path.join(base_cov, "cases"))) &
                                   set(os.listdir(os.path.join(new_cov, "cases"))))
    tot = collections.Counter()
    diffs = []
    for case in cases:
        a, b = builds(base_cov, case), builds(new_cov, case)
        if a is None or b is None:
            print(f"{case:26s} missing capture ({'base' if a is None else 'new'})")
            continue
        per = collections.Counter()
        vals = collections.Counter()
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                per["unpaired " + ("base" if k in a else "new")] += 1
                continue
            ea, eb = a[k][0][1], b[k][0][1]
            kind = "router" if k[0] == "fabric_erisc_router" else "other"
            v = b[k][0][2]
            if v is not None and kind == "router":
                vals[(v, k[1])] += 1
            if a[k][0][2] is not None:
                per["base has the arg"] += 1
            same = cov.image(ea) == cov.image(eb)
            per[f"{kind} {'identical' if same else 'differ'}"] += 1
            if not same:
                diffs.append((case, k[1], v, ea, eb))
        tot.update(per)
        vs = ", ".join(f"T{v} x{n} ({'E0' if p == 'active_erisc' else 'E1'})" for (v, p), n in sorted(vals.items()))
        print(f"{case:26s} " + "  ".join(f"{x} {per[x]}" for x in sorted(per)) + (f"   new arg: {vs}" if vs else ""),
              flush=True)
    print("totals: " + "  ".join(f"{x} {tot[x]}" for x in sorted(tot)))
    with open(os.path.join(new_cov, "gate_diffs.json"), "w") as f:
        json.dump(diffs, f, indent=0)
    for d in diffs[:3]:
        print("  differs:", d[0], d[1], "T", d[2], d[4])


if __name__ == "__main__":
    main()
