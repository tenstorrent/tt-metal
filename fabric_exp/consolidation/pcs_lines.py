#!/usr/bin/env python3
"""pcs_lines.py SAMPLES_PREFIX CACHE_DIR CHIP/CHAN/ERISC [--top N] [--depth D] [--dump F]: one router's PC samples
by source.

For one target of a pc_sampler.py run: its build (the run cache's relocated image equal to the image head read at the
launch message's load address, as pcs_report.py matches it), its kernel samples as ELF addresses, the first and last 5%
by time dropped (outside the traffic window), then through addr2line -i on the cache's own <proc>.elf (the JIT builds
with -g; its loadable code is the xip image's) to source. Prints the share of samples per innermost source line, and
per inline chain cut to D frames from the innermost (the helper and its callers), the top N of each. --dump writes
every window sample's address with its whole chain (JSON: [[count, [[function, file:line], ...], ELF address], ...]) to F."""
import collections, glob, json, os, re, struct, subprocess, sys

A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
prefix, cache, want = sys.argv[1], sys.argv[2], sys.argv[3]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
depth = int(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 2
chip, chan, erisc = (int(x) for x in want.replace("E", "").split("/"))
meta = json.load(open(prefix + ".json"))
idx = next(i for i, t in enumerate(meta["targets"]) if (t["chip"], t["chan"], t["erisc"]) == (chip, chan, erisc))
samples = [(t, pc) for t, i, pc in struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()) if i == idx]
proc = "active_erisc" if erisc == 0 else "subordinate_active_erisc"
lay = meta["layouts"][f"{chip}/{chan}"]["erisc"][str(erisc)]
head, load, size = bytes.fromhex(lay["head"]), lay["load"], lay["text_size"]


def text_segment(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<IIIIIIII", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off:p_off + p_filesz]
    raise ValueError(path)


build = None
for e in sorted(glob.glob(f"{cache}/**/fabric_erisc_router*/*/{proc}/{proc}.elf.xip.elf", recursive=True)):
    base, text = text_segment(e)
    if text[:len(head)] == head:
        build = (e[: -len(".xip.elf")], base)
        break
if build is None:
    sys.exit("no build in the cache matches the target's image")
elf, base = build
kern = sorted((t, pc - load + base) for t, pc in samples if load <= pc < load + size)
cut = len(kern) // 20
kern = kern[cut: len(kern) - cut]
addrs = sorted({a for _, a in kern})
# One addr2line call: -a prints each address, then its (function, file:line) pairs, innermost first.
out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf] + [hex(a) for a in addrs], capture_output=True,
                     text=True).stdout.splitlines()
chains, cur, k = {}, None, 0
while k < len(out):
    if re.fullmatch(r"0x[0-9a-f]+", out[k]):
        cur = int(out[k], 16)
        chains[cur] = []
        k += 1
        continue
    fn = out[k]
    while re.search(r"<[^<>]*>", fn):  # template arguments, innermost first
        fn = re.sub(r"<[^<>]*>", "", fn)
    fn = re.sub(r"\(.*", "", fn).replace("fabric_router::", "").split()[-1] if fn.strip() else "?"
    loc = re.sub(r" \(discriminator \d+\)", "", out[k + 1]).split("/tt_metal/")[-1].split("/fabric_exp/")[-1]
    chains[cur].append((fn, loc))
    k += 2
n = len(kern)
by_line = collections.Counter()
by_chain = collections.Counter()
for _, a in kern:
    f = chains[a]
    by_line[f"{f[0][1]}  ({f[0][0]})" if f else "?"] += 1
    by_chain[" < ".join(fn for fn, _ in f[:depth]) if f else "?"] += 1
if "--dump" in sys.argv:
    counts = collections.Counter(a for _, a in kern)
    json.dump([[c, chains[a], a] for a, c in counts.items()], open(sys.argv[sys.argv.index("--dump") + 1], "w"))
print(f"{want}: {len(samples)} samples, {n} kernel samples in the window; build {elf.split('/kernels/')[-1]}")
print(f"\nby innermost line (top {top}):")
for k, c in by_line.most_common(top):
    print(f"  {100 * c / n:5.1f}%  {k}")
print(f"\nby inline chain, {depth} frames (top {top}):")
for k, c in by_chain.most_common(top):
    print(f"  {100 * c / n:5.1f}%  {k}")
