"""phase_lines.py PREFIX CACHE CHIP/CHAN/ERISC T0_OVER_T2 [--top N]: one target's E0 samples in the overlap phase of a
vc02 row (both VCs sending): the traffic window is first..last sample inside sender::send; the overlap is its first
T0/T2 (VC0's cycles over VC2's). Per innermost source line and per (visit, innermost line): share of the overlap's
samples. Templates kept, so visit_worker<0> and visit_worker<CH2> stay apart."""
import collections, glob, json, re, struct, subprocess, sys
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
prefix, cache, want, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
chip, chan, erisc = (int(x) for x in want.split("/"))
meta = json.load(open(prefix + ".json"))
idx = next(i for i, t in enumerate(meta["targets"]) if (t["chip"], t["chan"], t["erisc"]) == (chip, chan, erisc))
samples = [(t, pc) for t, i, pc in struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()) if i == idx]
proc = "active_erisc" if erisc == 0 else "subordinate_active_erisc"
lay = meta["layouts"][f"{chip}/{chan}"]["erisc"][str(erisc)]
head, load, size = bytes.fromhex(lay["head"]), lay["load"], lay["text_size"]
def text_segment(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28); phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<IIIIIIII", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off:p_off + p_filesz]
build = None
for e in sorted(glob.glob(f"{cache}/**/fabric_erisc_router*/*/{proc}/{proc}.elf.xip.elf", recursive=True)):
    base, text = text_segment(e)
    if text[:len(head)] == head:
        build = (e[: -len(".xip.elf")], base); break
elf, base = build
kern = sorted((t, pc - load + base) for t, pc in samples if load <= pc < load + size)
addrs = sorted({a for _, a in kern})
out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf] + [hex(a) for a in addrs], capture_output=True, text=True).stdout.splitlines()
chains, cur, k = {}, None, 0
while k < len(out):
    if re.fullmatch(r"0x[0-9a-f]+", out[k]):
        cur = int(out[k], 16); chains[cur] = []; k += 1; continue
    fn = re.sub(r"\(.*", "", out[k]).replace("fabric_router::", "")
    fn = re.sub(r"<(\d+)u?l?,.*", r"<\1>", fn)
    fn = re.sub(r"^\S+ ", "", fn) if "<" in fn.split(" ")[0] or " " in fn else fn
    loc = re.sub(r" \(discriminator \d+\)", "", out[k + 1]).split("/tt_metal/")[-1].split("/fabric_exp/")[-1]
    loc = loc.replace("fabric/hw/inc/edm_fabric/router/", "")
    chains[cur].append((fn[:60], loc)); k += 2
in_send = [t for t, a in kern if any(f.startswith("sender::send<") for f, _ in chains[a])]
if "--hist" in sys.argv:
    lo, hi = kern[0][0], kern[-1][0]
    nb = 35
    import collections as C
    hs = C.Counter(int((t - lo) / (hi - lo + 1e-9) * nb) for t in in_send)
    hv = C.Counter(int((t - lo) / (hi - lo + 1e-9) * nb) for t, a in kern if any(f.startswith("sender::visit_worker<4") for f, _ in chains[a]))
    h0 = C.Counter(int((t - lo) / (hi - lo + 1e-9) * nb) for t, a in kern if any(f.startswith("sender::visit_worker<0") for f, _ in chains[a]))
    ha = C.Counter(int((t - lo) / (hi - lo + 1e-9) * nb) for t, a in kern)
    for b in range(nb):
        print(f"{lo + (hi-lo)*b/nb:8.3f} all {ha[b]:4d} send {hs[b]:4d} vw0 {h0[b]:4d} vw4 {hv[b]:4d}")
# The traffic window: 20 ms bins with at least 30% of the busiest bin's send samples (stray early samples dropped).
import collections as _C
_b = _C.Counter(int(t / 0.02) for t in in_send)
_thr = 0.3 * max(_b.values())
_hot = sorted(b for b, c in _b.items() if c >= _thr)
t0, t1 = _hot[0] * 0.02, (_hot[-1] + 1) * 0.02
tov = t0 + (t1 - t0) * frac
win = [(t, a) for t, a in kern if t0 <= t <= tov]
n = len(win)
print(f"{want}: build {elf.split('/kernels/')[-1]}; traffic {t1 - t0:.3f} s, overlap {tov - t0:.3f} s: {n} samples")
by_line, by_visit = collections.Counter(), collections.Counter()
for _, a in win:
    ch = chains[a]
    inner = ch[0][1] if ch else "?"
    by_line[inner] += 1
    visit = next((f for f, _ in ch if re.match(r"sender::(visit_\w+|check_\w+|channels_with_packets|pass|run|send)\b", f) and not f.startswith("sender::send")), "?")
    by_visit[visit] += 1
print(" by visit:")
for kk, c in by_visit.most_common(top):
    print(f"  {100*c/n:5.1f}%  {kk}")
print(" by innermost line:")
for kk, c in by_line.most_common(top):
    print(f"  {100*c/n:5.1f}%  {kk}")
json.dump({"n": n, "lines": by_line, "visits": by_visit}, open(sys.argv[sys.argv.index("--json") + 1], "w")) if "--json" in sys.argv else None
