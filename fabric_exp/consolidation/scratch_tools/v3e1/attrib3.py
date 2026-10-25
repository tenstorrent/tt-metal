import sys, re, subprocess, collections
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
e = sys.argv[1]
out = subprocess.run([OD, "-d", "--section=.text", e], capture_output=True, text=True).stdout
addrs = [l.split(":")[0].strip() for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
a2l = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", e] + ["0x" + a for a in addrs], capture_output=True, text=True).stdout.split("\n")
recs = []; cur = None; i = 0
while i < len(a2l):
    l = a2l[i]
    if re.match(r"^0x[0-9a-f]+$", l):
        cur = [int(l, 16), []]; recs.append(cur); i += 1; continue
    if cur is not None and l and i + 1 < len(a2l):
        cur[1].append((l, a2l[i+1])); i += 2; continue
    i += 1
cnt = collections.Counter()
for a, fr in recs:
    names = [n for n, loc in fr]
    if len(names) and names[-1].startswith("kernel_main"):
        # second-outermost frame (what kernel_main called/inlined), or kernel_main's own line
        if len(fr) >= 2:
            n = re.sub(r"<.*", "", fr[-2][0].split("(")[0]).split(" ")[-1][:60]
            cnt[n] += 4
        else:
            ln = int(fr[-1][1].split(":")[-1].split()[0]) if ":" in fr[-1][1] else 0
            cnt[f"kernel_main:{ln//20*20}"] += 4
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:40]:
    print(f"{v:6d} {k}")
