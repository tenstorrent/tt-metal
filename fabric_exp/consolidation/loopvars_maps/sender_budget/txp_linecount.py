"""linecount.py ELF FUNC_SUBSTR: instructions of the (first) function whose mangled name contains FUNC_SUBSTR, counted by
innermost source line (router_sender.hpp lines and helpers), with the inline chain's visit (template args kept)."""
import re, subprocess, sys, collections
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
elf, fn = sys.argv[1], sys.argv[2]
syms = subprocess.run([B + "nm", "-S", elf], capture_output=True, text=True).stdout.split("\n")
s = next(l for l in syms if fn in l and l.split()[2] in "tT")
addr, size = int(s.split()[0], 16), int(s.split()[1], 16)
dis = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", f"--start-address={addr:#x}", f"--stop-address={addr+size:#x}", elf], capture_output=True, text=True).stdout
ins = [(int(m.group(1), 16), m.group(2)) for m in re.finditer(r"^\s*([0-9a-f]+):\s+(.*)$", dis, re.M)]
out = subprocess.run([B + "addr2line", "-a", "-f", "-i", "-C", "-e", elf] + [hex(a) for a, _ in ins], capture_output=True, text=True).stdout.splitlines()
ch, cur, k = {}, None, 0
while k < len(out):
    if re.fullmatch(r"0x[0-9a-f]+", out[k]):
        cur = int(out[k], 16); ch[cur] = []; k += 1; continue
    f = re.sub(r"\(.*", "", out[k]).replace("fabric_router::", "")
    f = re.sub(r"<(\d+)u?l?,.*", r"<\1>", f).split(" ")[-1]
    ch[cur].append((f[:40], out[k + 1].split("/")[-1].split(" ")[0])); k += 2
c = collections.Counter()
for a, t in ins:
    fr = ch.get(a, [])
    inner = fr[0] if fr else ("?", "?")
    visit = next((f for f, _ in fr if re.match(r"sender::(visit_worker|visit_kept_worker|visit_forwarding|pass)", f)), "?")
    c[(visit, inner[0], inner[1])] += 1
for k2, v in sorted(c.items(), key=lambda x: (x[0][0], x[0][2])):
    print(f"{v:4d}  {k2[0]:28s} {k2[1]:30s} {k2[2]}")
