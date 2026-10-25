import re, sys
# condense objdump -d -l output: one line per instruction, tagged with the last source file:line seen
src = "?"
for l in open(sys.argv[1]):
    l = l.rstrip("\n")
    m = re.match(r"^(/\S+):(\d+)", l)
    if m:
        src = m.group(1).split("/")[-1] + ":" + m.group(2)
        continue
    m = re.match(r"^\s+([0-9a-f]+):\t([0-9a-f]+)\s+\t?(.*)$", l)
    if m:
        print(f"{m.group(1):>6s}  {m.group(3):48s} {src}")
    elif re.match(r"^[0-9a-f]+ <", l) and not l.split("<")[1].startswith(".L"):
        print(l)
