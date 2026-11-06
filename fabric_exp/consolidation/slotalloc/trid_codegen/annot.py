import subprocess, re, sys, os
P = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
def sh(a, inp=None): return subprocess.run(a, capture_output=True, text=True, input=inp).stdout
def func_range(elf, want):
    for l in sh([P + "nm", "-C", "-S", elf]).splitlines():
        p = l.split(None, 3)
        if len(p) == 4 and re.search(want, p[3]): return int(p[0], 16), int(p[1], 16), p[3]
def annotate(elf, want=r"receiver::run[<(]"):
    lo, sz, name = func_range(elf, want)
    out = sh([P + "objdump", "-d", "--no-show-raw-insn", f"--start-address={lo}", f"--stop-address={lo+sz}", elf])
    ins = []
    for l in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+(.*)$", l)
        if m: ins.append((int(m.group(1), 16), re.sub(r"\s*#.*$", "", re.sub(r" <[^>]*>", "", m.group(2))).strip()))
    a2l = sh([P + "addr2line", "-f", "-i", "-C", "-a", "-e", elf], "\n".join(hex(a) for a, _ in ins) + "\n").splitlines()
    chains, cur, i = {}, None, 0
    while i < len(a2l):
        l = a2l[i]
        if l.startswith("0x"):
            cur = int(l, 16); chains[cur] = []; i += 1; continue
        fn = l; loc = a2l[i + 1] if i + 1 < len(a2l) else "?"
        chains[cur].append((re.sub(r"\(.*", "", fn).split("::")[-1], os.path.basename(loc.split(" ")[0])))
        i += 2
    return name, [(a, t, chains.get(a, [])) for a, t in ins]
if __name__ == "__main__":
    name, rows = annotate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else r"receiver::run[<(]")
    for a, t, c in rows:
        print(f"{a:6x}  {t:40s}  " + " < ".join(f"{f}@{l}" for f, l in c[:4]))
