# Instructions and sp accesses of sender::run by innermost inline function (and source line), two ELFs side by side.
import re, subprocess, sys, collections
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
def stats(elf, key):
    for l in subprocess.run([B + "nm", "-S", elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        if len(p) >= 4 and key in p[3]:
            lo, size = int(p[0], 16), int(p[1], 16); break
    dis = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", f"--start-address={lo}", f"--stop-address={lo+size}", elf], capture_output=True, text=True).stdout
    ins = [(int(m.group(1), 16), m.group(2).strip()) for m in (re.match(r"\s*([0-9a-f]+):\s+(.*)$", l) for l in dis.splitlines()) if m]
    loc = ps.a2l(elf, [a for a, _ in ins])
    c = collections.Counter(); s = collections.Counter()
    for a, t in ins:
        fr = loc.get(a, [])
        nm0 = fr[0][0] if fr else "?"; base = nm0.split("<")[0].split("::")[-1]; tmpl = re.search(r"<(\d+)u", nm0); name = base + ("<" + tmpl.group(1) + ">" if tmpl else "")
        line = fr[0][1].split("/")[-1] if fr else "?"
        k = f"{name} @ {line}"
        c[k] += 1
        if re.search(r"\(sp\)", t): s[k] += 1
    return c, s
a, b, key = sys.argv[1], sys.argv[2], sys.argv[3]
ca, sa = stats(a, key); cb, sb = stats(b, key)
rows = sorted(set(ca) | set(cb), key=lambda k: -abs(cb[k] - ca[k]) - abs(sb[k] - sa[k]))
print(f"total insns {sum(ca.values())} -> {sum(cb.values())}, sp {sum(sa.values())} -> {sum(sb.values())}")
for k in rows[:25]:
    if cb[k] != ca[k] or sb[k] != sa[k]:
        print(f"  {cb[k]-ca[k]:+4d} insns {sb[k]-sa[k]:+3d} sp   {k}")
