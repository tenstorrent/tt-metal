import re, sys
# For each TXQ register read (lw rd,4|8(base) where base was set by lui base,0xffb90), report the distance in
# instructions to the first later instruction that reads or writes rd. Only the first ~N sends after the loop top.
L = open(sys.argv[1]).read().splitlines()
ins = []
for l in L:
    m = re.match(r"^\s+([0-9a-f]+):\t\S+\s*\t(\S+)\s*(.*)$", l)
    if m: ins.append((int(m.group(1), 16), m.group(2), re.sub(r"\s*#.*$|\s*<.*>$", "", m.group(3))))
top = next(ins[i-1][0] for i, x in enumerate(ins) if x[1] == "lw" and re.search(r",1188\(\w+\)", x[2]))
end = max(x[0] for x in ins if re.search(rf"\b{top:x}\b", x[2]) and x[1].startswith("b"))
txq = set()
out = []
for i, (a, op, args) in enumerate(ins):
    if a < top or a > end: continue
    if op == "lui" and args.endswith("0xffb90"): txq.add(args.split(",")[0])
    m = re.match(r"(\w+),(4|8)\((\w+)\)$", args)
    if op == "lw" and m and m.group(3) in txq:
        rd = m.group(1)
        for j in range(i + 1, min(i + 30, len(ins))):
            regs = re.findall(r"\b([as]\d+|t\d|ra|s1[01])\b", ins[j][2])
            if rd in regs:
                kind = "write" if regs and regs[0] == rd and ins[j][1] not in ("sw", "sb", "sh") and not ins[j][1].startswith("b") else "read"
                out.append(f"{a:x} {'cmd' if m.group(2)=='4' else 'status'} -> {ins[j][1]} {kind} after {j-i}")
                break
    # base reg overwritten?
    if args and op not in ("sw", "sb", "sh") and not op.startswith("b"):
        d = args.split(",")[0]
        if d in txq and not (op == "lui" and args.endswith("0xffb90")): txq.discard(d)
print("\n".join(out[:12]))
