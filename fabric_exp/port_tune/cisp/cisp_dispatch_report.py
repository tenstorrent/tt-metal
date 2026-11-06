#!/usr/bin/env python3
"""cisp_dispatch_report.py RESULTS_DIR: the dispatch cores' PC samples of a cisp_dispatch.sh run (noc0 column x 16 on
chips 0, 3, 7), mapped to the cq_prefetch / cq_dispatch / cq_dispatch_subordinate images in the arm's cache through the
cores' launch slots (load = kcb + text_off[risc]); per active (core, RISC), the innermost function and source line of
its samples inside slow and normal NX unicast_write iterations (slow: the test's sender row below 97% of the off arm's
median), and in the off arm."""
import bisect, collections, csv, glob, json, os, re, statistics, struct, subprocess, sys

sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hang_workers as hw  # noqa: E402
import cisp_pcs2_report as P  # noqa: E402

D = sys.argv[1]
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
CACHE = {"on": "/home/aagarwal/.tt-metal-cache-cisp-on-ci", "off": "/home/aagarwal/.tt-metal-cache-cisp-off-ci"}
RISC_P = {"brisc": 0, "ncrisc": 1}


def images(arm):
    out = []
    for name in ("cq_prefetch", "cq_dispatch", "cq_dispatch_subordinate"):
        for e in glob.glob(f"{CACHE[arm]}/*/kernels/{name}/*/*/*.elf.xip.elf"):
            base, text = P.text_segment(e)
            out.append((name, e[: -len(".xip.elf")], base, text))
    return out


_c = {}


def where(elf, a):
    if (elf, a) not in _c:
        o = subprocess.run([A2L, "-f", "-C", "-e", elf, hex(a)], capture_output=True, text=True).stdout.splitlines()
        fn = re.sub(r"\(.*", "", o[0]).strip() if o else "?"
        ln = "/".join(o[1].split("/")[-2:]) if len(o) > 1 else "?"
        _c[(elf, a)] = f"{fn} {re.sub(r' .discriminator.*', '', ln)}"
    return _c[(elf, a)]


def run(arm, slow_of):
    pre = os.path.join(D, f"disp_{arm}_samples")
    meta = json.load(open(pre + ".json"))
    imgs = images(arm)
    snaps = collections.defaultdict(list)
    sraw = open(pre + ".snap.bin", "rb").read()
    step = 10 + 1488
    for o in range(0, len(sraw) - step + 1, step):
        t, k = struct.unpack_from("<dH", sraw, o)
        hd = hw.parse_head(sraw[o + 10: o + 10 + 1488])
        if hd:
            snaps[k].append((t + meta["t_start_epoch"], hd["slots"]))
    ci = {(c["chip"], c["core"]): k for k, c in enumerate(meta["snap_cores"])}
    win = P.windows(os.path.join(D, f"disp_{arm}.log"))
    starts = [w[2] for w in win]
    res = collections.defaultdict(collections.Counter)
    for t, i, pc in struct.iter_unpack("<dHI", open(pre + ".bin", "rb").read()):
        tg = meta["targets"][i]
        if not tg["chan"].startswith("f16-") or tg["erisc"] not in RISC_P:
            continue
        t += meta["t_start_epoch"]
        j = bisect.bisect_right(starts, t) - 1
        if j < 0 or t > win[j][3] or "unicast_write" not in win[j][0]:
            continue
        grp = "slow" if slow_of(j) else "normal"
        p = RISC_P[tg["erisc"]]
        k = ci[(tg["chip"], tg["chan"])]
        ts = [s[0] for s in snaps[k]]
        q = bisect.bisect_right(ts, t) - 1
        label = None
        for qq in (q, q - 1, q + 1):
            if 0 <= qq < len(snaps[k]):
                for sl in snaps[k][qq][1]:
                    load, size = sl["kcb"] + sl["text_off"][p], sl["text_size"][p]
                    if size and load <= pc < load + size:
                        m = min((im for im in imgs if abs(len(im[3]) - size) <= 256), key=lambda im: abs(len(im[3]) - size), default=None)
                        label = f"{m[0]}: {where(m[1], pc - load + m[2])}" if m else f"kernel image ({size} B)"
                        break
            if label:
                break
        res[(tg["chip"], tg["chan"], tg["erisc"], grp)][label or f"firmware/other pc {pc & ~0xff:#x}"] += 1
    return res


offm = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, "disp_off_bandwidth_results.csv"))):
    offm[r["test_name"]].append(float(r["bandwidth_GB_s"]))
onr = list(csv.DictReader(open(os.path.join(D, "disp_on_bandwidth_results.csv"))))
slow = [float(r["bandwidth_GB_s"]) < 0.97 * statistics.median(offm[r["test_name"]]) for r in onr]
for arm, sf in (("on", lambda j: j < len(slow) and slow[j]), ("off", lambda j: False)):
    res = run(arm, sf)
    print(f"\n== {arm}")
    for key in sorted(res):
        c = res[key]
        n = sum(c.values())
        if all(k.startswith("firmware") for k in c):
            continue
        print(f"   chip {key[0]} {key[1]} {key[2]} {key[3]}: {n} samples")
        for lab, v in c.most_common(6):
            print(f"      {100 * v / n:6.2f}%  {lab}")
