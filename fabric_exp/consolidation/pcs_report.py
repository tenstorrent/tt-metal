#!/usr/bin/env python3
"""pcs_report.py SAMPLES_PREFIX CACHE_DIR [--top N] [--min-kernel F]: summarize pc_sampler.py output.

Per target (chip, Ethernet channel, ERISC): the router build it runs, found by matching the kernel image head the
sampler read at the launch message's load address against the run's kernel cache (E0 = active_erisc, E1 =
subordinate_active_erisc; the relocated .xip.elf image). Kernels run from the kernel config window, so a PC in [load, load + text size) maps to the
ELF address pc - load + text start; any other PC is firmware (tt-metal's Ethernet firmware, or the base firmware the
idle context switch hands over to). Then, for the busiest targets (most samples in the kernel), the kernel samples by
function (nm symbol sizes) and the hottest ELF addresses."""
import collections
import glob
import json
import struct
import subprocess
import sys

NM = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
prefix, cache = sys.argv[1], sys.argv[2]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 4
meta = json.load(open(prefix + ".json"))
per = collections.defaultdict(list)
for t, i, pc in struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()):
    per[i].append(pc)


def text_segment(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<8I", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:  # PT_LOAD, executable
            return p_vaddr, d[p_off:p_off + p_filesz]
    return None, b""


def symbols(elf):
    out = subprocess.run([NM, "-S", "-C", elf], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        p = line.split(None, 3)
        if len(p) == 4 and p[2] in "tTwW":
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    return sorted(syms)


builds = {}
for proc in ("active_erisc", "subordinate_active_erisc"):
    builds[proc] = []
    # The image a core runs is the relocated ".xip.elf" one (byte-equal to L1); its addresses are the ELF's.
    for e in sorted(glob.glob(f"{cache}/**/fabric_erisc_router_v2/*/{proc}/{proc}.elf.xip.elf", recursive=True)):
        base, text = text_segment(e)
        builds[proc].append((e, base, text))
print(f"{meta['samples']} samples, {meta['duration_s']} s, {meta['rate_per_s']}/s; "
      f"{len(builds['active_erisc'])} E0 and {len(builds['subordinate_active_erisc'])} E1 builds in the cache")


def fn_of(syms, a):
    for s, z, n in syms:
        if s <= a < s + z:
            return n.split("(")[0][:70]
    return f"<no symbol {a:#x}>"


rows = []
for i, tgt in enumerate(meta["targets"]):
    pcs = per.get(i, [])
    lay = meta.get("layouts", {}).get(f"{tgt['chip']}/{tgt['chan']}", {})
    e = lay.get("erisc", {}).get(str(tgt["erisc"]), {})
    proc = "active_erisc" if tgt["erisc"] == 0 else "subordinate_active_erisc"
    head = bytes.fromhex(e.get("head", "")) if e else b""
    match = [(p, b) for p, b, t in builds[proc] if head and t[:len(head)] == head]
    load, size = e.get("load", 0), e.get("text_size", 0)
    kern = [pc - load + match[0][1] for pc in pcs if match and load <= pc < load + size]
    rows.append((len(kern), tgt, len(pcs), match, load, kern))

print("\nper target: chip/chan/ERISC: samples in the kernel / all, matching builds, load address")
for nk, tgt, n, match, load, kern in sorted(rows, key=lambda r: -r[0]):
    b = match[0][0].split("fabric_erisc_router_v2/")[-1].split("/")[0] if match else "-"
    print(f"  {tgt['chip']}/{tgt['chan']:>2}/E{tgt['erisc']}: {nk:5d} / {n:5d} in kernel ({100 * nk / max(n, 1):5.1f}%), "
          f"{len(match)} matching build(s) {b}, load {load:#x}")
for nk, tgt, n, match, load, kern in sorted(rows, key=lambda r: -r[0])[:top]:
    if not match:
        continue
    syms = symbols(match[0][0])
    print(f"\n== {tgt['chip']}/{tgt['chan']}/E{tgt['erisc']}: {nk} kernel samples of {n}")
    fns = collections.Counter(fn_of(syms, a) for a in kern)
    for fn, c in fns.most_common(8):
        print(f"   {100 * c / max(nk, 1):5.1f}%  {fn}")
    hot = collections.Counter(kern)
    print("   hottest addresses: " + ", ".join(f"{a:#x} {100 * c / max(nk, 1):.1f}%" for a, c in hot.most_common(8)))
