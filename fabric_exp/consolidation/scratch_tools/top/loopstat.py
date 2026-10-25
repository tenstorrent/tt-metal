import re, subprocess, sys
# E0 main loop: from the first grouped free-slot read (lw rX,1188(rY) after a lui 0xffb4x) to the farthest backward
# branch/jump that targets an address at or before it; counts instructions and sp-relative loads/stores in between.
OD = "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
def stat(elf):
    ins = []
    for l in subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\t\S+\s+\t?(\S+)\s*(.*)", l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
    heads = [i for i, x in enumerate(ins) if x[1] == "lw" and ",1188(" in x[2]]
    h = heads[0]
    while h > 0 and not ins[h - 1][1] == "lui":
        h -= 1
    h -= 1
    head = ins[h][0]
    back = max(i for i, x in enumerate(ins) if x[1] in ("j", "bnez", "beqz", "bne", "beq", "bltz", "bgez", "blt", "bge", "bltu", "bgeu")
               and (m := re.search(r"\b([0-9a-f]+) <", x[2])) and int(m.group(1), 16) <= head and x[0] > head)
    body = ins[h : back + 1]
    ld = sum(1 for x in body if x[1] in ("lw", "lbu", "lhu", "lb", "lh") and "(sp)" in x[2])
    st = sum(1 for x in body if x[1] in ("sw", "sb", "sh") and "(sp)" in x[2])
    return f"loop {head:x}-{ins[back][0]:x}: {len(body)} insns, {ld} sp loads, {st} sp stores"
for e in sys.argv[1:]:
    print(e.split("/")[2], stat(e))
