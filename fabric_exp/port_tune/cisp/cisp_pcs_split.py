#!/usr/bin/env python3
"""cisp_pcs_split.py SAMPLES_PREFIX CACHE_DIR [--erisc 0|1]: where each router ERISC's PC samples go, by inline chain
(cisp follow-up (a); a within-ERISC split, never a bound). Mapping as consolidation/pcs_lines.py: the target's build is
the cache image (<proc>.elf.xip.elf) whose text starts with the image head read at the launch message's load address;
kernel PCs map to pc - load + text start; addr2line -i on the cache's -g <proc>.elf gives the inline chain; the first
and last 5% of each target's kernel samples (by time) are dropped. Buckets, by the whole chain (first match wins):
link_sync (any frame in link_sync::), eth_ptp, kernel_profiler, end_round (the rest of end_round), telemetry
(telemetry:: outside end_round), passes (everything else in the kernel). Firmware = samples outside the kernel image.
Traffic-carrying targets: more than 15% of kernel samples in sender::send (E0) / receiver:: handle or forward (E1)."""
import collections, glob, json, re, struct, subprocess, sys
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
prefix, cache = sys.argv[1], sys.argv[2]
want_e = int(sys.argv[sys.argv.index("--erisc") + 1]) if "--erisc" in sys.argv else 0
meta = json.load(open(prefix + ".json"))
recs = list(struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()))
by_t = collections.defaultdict(list)
for t, i, pc in recs:
    by_t[i].append((t, pc))
def text_segment(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<IIIIIIII", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off:p_off + p_filesz]
    raise ValueError(path)
proc = "active_erisc" if want_e == 0 else "subordinate_active_erisc"
images = [(e, *text_segment(e)) for e in sorted(glob.glob(f"{cache}/**/fabric_erisc_router*/*/{proc}/{proc}.elf.xip.elf", recursive=True))]
chain_cache = {}
def chains_for(elf, addrs):
    todo = [a for a in addrs if (elf, a) not in chain_cache]
    for k0 in range(0, len(todo), 2000):
        part = todo[k0:k0 + 2000]
        out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf] + [hex(a) for a in part], capture_output=True, text=True).stdout.splitlines()
        cur, k = None, 0
        while k < len(out):
            if re.fullmatch(r"0x[0-9a-f]+", out[k]):
                cur = int(out[k], 16); chain_cache[(elf, cur)] = []; k += 1; continue
            fn = out[k]
            while re.search(r"<[^<>]*>", fn):
                fn = re.sub(r"<[^<>]*>", "", fn)
            fn = re.sub(r"\(.*", "", fn).split()[-1] if fn.strip() else "?"
            chain_cache[(elf, cur)].append(fn); k += 2
    return {a: chain_cache[(elf, a)] for a in addrs}
def bucket(ch):
    s = " ".join(ch)
    for name, pat in (("link_sync", "link_sync::"), ("eth_ptp", "eth_ptp::"), ("kernel_profiler", "kernel_profiler::")):
        if pat in s:
            return name
    if any(f.endswith("end_round") for f in ch):
        return "end_round"
    if "telemetry::" in s:
        return "telemetry"
    return "passes"
ORDER = ["passes", "end_round", "link_sync", "eth_ptp", "kernel_profiler", "telemetry"]
rows, tot = [], collections.Counter()
for idx, tg in enumerate(meta["targets"]):
    if tg["erisc"] != want_e or idx not in by_t:
        continue
    lay = meta["layouts"][f"{tg['chip']}/{tg['chan']}"]["erisc"][str(want_e)]
    head, load, size = bytes.fromhex(lay["head"]), lay["load"], lay["text_size"]
    m = next(((e, base) for e, base, text in images if head and text[:len(head)] == head), None)
    s = by_t[idx]
    if m is None:
        rows.append((tg, len(s), None, None)); continue
    elf, base = m[0][: -len(".xip.elf")], m[1]
    kern = sorted((t, pc - load + base) for t, pc in s if load <= pc < load + size)
    cut = len(kern) // 20
    kern = kern[cut: len(kern) - cut]
    ch = chains_for(elf, sorted({a for _, a in kern}))
    c = collections.Counter(bucket(ch[a]) for _, a in kern)
    send = sum(1 for _, a in kern if any(f.endswith("sender::send") for f in ch[a])) if want_e == 0 else \
        sum(1 for _, a in kern if any(f.split("::")[-1] in ("handle", "forward_all", "receive") for f in ch[a]))
    rows.append((tg, len(s), c, send / max(1, len(kern))))
print(f"{prefix}: ERISC{want_e}, {len(rows)} targets, {sum(r[1] for r in rows)} samples")
print("chip/chan   samples  kernel  busy   " + "  ".join(f"{b:>15s}" for b in ORDER))
for tg, ns, c, busy in rows:
    if c is None:
        print(f"{tg['chip']}/{tg['chan']:<4} {ns:9d}  (no matching build)"); continue
    n = sum(c.values())
    print(f"{tg['chip']}/{tg['chan']:<4} {ns:9d} {n:7d} {100 * busy:5.1f}%  " + "  ".join(f"{100 * c[b] / max(1, n):14.3f}%" for b in ORDER))
    if busy > 0.15:
        tot.update(c)
n = sum(tot.values())
print(f"traffic-carrying ERISC{want_e}s pooled ({n} kernel samples): " + ", ".join(f"{b} {100 * tot[b] / max(1, n):.3f}%" for b in ORDER))
