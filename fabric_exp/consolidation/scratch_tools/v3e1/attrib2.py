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
        cur[1].append(l); i += 2; continue
    i += 1
cnt = collections.Counter()
keys = sys.argv[2].split(",")
for a, names in recs:
    tag = None
    for n in names:
        base = re.sub(r"<.*", "", n.split("(")[0]).split(" ")[-1]
        if base in keys:
            # channel template arg, if any
            m = re.search(r"<\(?(?:unsigned int|unsigned long)?\)?(\d)", n)
            tag = base + (f"<{m.group(1)}>" if m and base.startswith("e1_rx") else "")
            break
    if tag is None:
        tag = "other:" + (names[-1].split("(")[0][:50] if names else "?")
    cnt[tag] += 4
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"{v:6d} {k}")
print(sum(cnt.values()))
