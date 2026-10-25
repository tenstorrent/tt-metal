import re, subprocess, sys
# E0 main loop: head = the lui before the first grouped free-slot read (lw rX,1188(rY)); end = the last branch/jump
# whose target is within [head-16, head] (the exact back-edge). Prints counts, then every sp access / li / lui / call
# inside the loop with -v.
OD = "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
BR = ("j", "bnez", "beqz", "bne", "beq", "bltz", "bgez", "blt", "bge", "bltu", "bgeu", "blez", "bgtz")
def loop(elf):
    ins = []
    for l in subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\t\S+\s+\t?(\S+)\s*(.*)", l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
    h = next(i for i, x in enumerate(ins) if x[1] == "lw" and ",1188(" in x[2])
    while ins[h][1] != "lui":
        h -= 1
    head = ins[h][0]
    back = max(i for i, x in enumerate(ins) if x[1] in BR and x[0] > head
               and (m := re.search(r"\b([0-9a-f]+) <", x[2])) and head - 16 <= int(m.group(1), 16) <= head)
    return ins[h : back + 1]
if __name__ == "__main__":
    verbose = "-v" in sys.argv
    for e in [a for a in sys.argv[1:] if a != "-v"]:
        body = loop(e)
        ld = [x for x in body if x[1] in ("lw", "lbu", "lhu", "lb", "lh") and "(sp)" in x[2]]
        st = [x for x in body if x[1] in ("sw", "sb", "sh") and "(sp)" in x[2]]
        calls = [x for x in body if x[1] in ("jal", "call", "jalr")]
        print(f"{e}: loop {body[0][0]:x}-{body[-1][0]:x}: {len(body)} insns, {len(ld)} sp loads, {len(st)} sp stores, {len(calls)} calls")
        if verbose:
            for x in body:
                if "(sp)" in x[2] or x[1] in ("li", "lui", "jal", "call", "jalr") or re.search(r"\bandi\b", x[1]):
                    print(f"   {x[0]:x}: {x[1]} {x[2]}")
