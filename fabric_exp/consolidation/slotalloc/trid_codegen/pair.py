import glob, os, re, sys, subprocess, collections, json
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
C = sys.argv[1]
def named(h):
    return dict((k, int(v)) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(h).read()))
def builds(run):
    out = {}
    for elf in glob.glob(f"{C}/{run}/*/cache_*/*/kernels/fabric_erisc_router_v2/*/*/*.elf"):
        if elf.endswith(".xip.elf"): continue
        proc = os.path.basename(os.path.dirname(elf)); gen = os.path.dirname(os.path.dirname(elf))
        n = named(os.path.join(gen, "named_ct_arg_map_generated.h"))
        key = tuple(sorted((k, v) for k, v in n.items() if not re.match(r"RX_CH_\d_(NUM_TRIDS|TRID_START|COMPLETION_GROUP_SIZE)", k))) + (("proc", proc),)
        out.setdefault(key, []).append((elf, n))
    return out
def text(elf):
    o = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    return [re.sub(r" <[^>]*>", "", re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1])) for l in o.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
res = {}
for topo in ("t42", "t24"):
    a, b = builds(f"{topo}_idoff"), builds(f"{topo}_id")
    rows = []
    for k in sorted(a.keys() & b.keys()):
        ea, na = a[k][0]; eb, nb = b[k][0]
        ta, tb = text(ea), text(eb)
        d = dict(k)
        rows.append({"dir": "EWNS"[d["MY_DIRECTION"]] if d["MY_DIRECTION"] < 4 else str(d["MY_DIRECTION"]), "chan": d["MY_ETH_CHANNEL"],
                     "proc": d["proc"], "ids": (na["RX_CH_0_NUM_TRIDS"], nb["RX_CH_0_NUM_TRIDS"]),
                     "n": (len(ta), len(tb)), "same": ta == tb, "elfs": (ea, eb)})
    unp = len(a.keys() ^ b.keys())
    print(f"{topo}: {len(rows)} paired configs, {unp} unpaired")
    summ = collections.Counter((r["dir"], r["proc"], r["ids"], r["n"][1] - r["n"][0], r["same"]) for r in rows)
    for (dr, p, ids, dn, same), c in sorted(summ.items()):
        print(f"   dir {dr} {p:26s} ids {ids[0]}->{ids[1]}: {c} configs, insns {dn:+d}, identical .text {same}")
    res[topo] = rows
json.dump(res, open(f"{C}/pairs.json", "w"), indent=1)
