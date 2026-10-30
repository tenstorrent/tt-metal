#!/usr/bin/env python3
"""placement.py BUILD_A BUILD_B [--cases a,b] [--fn=run<]: where the compiler put each local of a router loop
function, per image, and whether two builds agree (V4_SENDER_SHAPE.md §0). From the DWARF of each image (coverage
builds keep it): for every local variable of the function's concrete instance, its location kind: "frame" (a frame
object, DW_OP_fbreg), "split" (a location list: scalarized into registers and spill slots), or "-" (optimized out).
Prints, per case and ERISC, the images whose kinds differ between the builds, and a summary."""
import collections, glob, os, re, subprocess, sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"
RE = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"


def kinds(elf, fn_prefix):
    t = subprocess.run([RE, "--debug-dump=info", elf], capture_output=True, text=True).stdout.splitlines()
    n = len(t)
    dies = []  # (line index, depth, offset, tag)
    for i, l in enumerate(t):
        m = re.search(r"<(\d+)><([0-9a-f]+)>: Abbrev Number: \d+ \((DW_TAG_\w+)\)", l)
        if m:
            dies.append((i, int(m.group(1)), m.group(2), m.group(3)))
    attrs = {}
    for k, (i, d, off, tag) in enumerate(dies):
        end = dies[k + 1][0] if k + 1 < len(dies) else n
        a = {}
        for l in t[i + 1:end]:
            mm = re.search(r"DW_AT_(\w+)\s*:\s*(.*)$", l)
            if mm:
                a[mm.group(1)] = mm.group(2)
        attrs[off] = (d, tag, a)
    def name_of(off):
        d, tag, a = attrs.get(off, (0, "", {}))
        if "name" in a:
            return a["name"].split(": ")[-1].strip()
        for key in ("specification", "abstract_origin"):
            if key in a:
                mm = re.search(r"<0x([0-9a-f]+)>", a[key])
                if mm:
                    return name_of(mm.group(1))
        return ""
    # abstract subprograms whose name starts with fn_prefix -> their child variables
    var_names = {}
    for k, (i, d, off, tag) in enumerate(dies):
        if tag == "DW_TAG_subprogram" and name_of(off).startswith(fn_prefix) and "low_pc" not in attrs[off][2]:
            for i2, d2, off2, tag2 in dies[k + 1:]:
                if d2 <= d:
                    break
                if tag2 == "DW_TAG_variable":
                    var_names[off2] = name_of(off2)
    out = {}
    for off, (d, tag, a) in attrs.items():
        if tag == "DW_TAG_variable" and "abstract_origin" in a:
            mm = re.search(r"<0x([0-9a-f]+)>", a["abstract_origin"])
            if mm and mm.group(1) in var_names:
                loc = a.get("location", "")
                out[var_names[mm.group(1)]] = "frame" if "fbreg" in loc else "split" if "location list" in loc else "-"
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    fn = opts.get("fn", "run<")
    cases = set(opts["cases"].split(",")) if "cases" in opts else None
    a, b = args
    diff, same = collections.Counter(), collections.Counter()
    for e in sorted(glob.glob(f"{COV}/{a}/*/*/*.elf")):
        rel = os.path.relpath(e, f"{COV}/{a}")
        case = rel.split("__")[0]
        if cases and case not in cases:
            continue
        o = f"{COV}/{b}/{rel}"
        if not os.path.exists(o):
            continue
        ka, kb = kinds(e, fn), kinds(o, fn)
        if not ka and not kb:
            continue
        if ka == kb:
            same[case] += 1
        else:
            diff[case] += 1
            if diff[case] <= 2:
                print(f"  {rel}: " + ", ".join(f"{v}: {ka.get(v)} -> {kb.get(v)}" for v in sorted(set(ka) | set(kb)) if ka.get(v) != kb.get(v)))
    print("same placement:", dict(same))
    print("different:", dict(diff))


if __name__ == "__main__":
    main()
