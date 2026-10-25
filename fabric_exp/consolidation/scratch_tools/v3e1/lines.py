import re, sys
# print instruction ranges whose source line matches a pattern (file:line regex), grouped into contiguous runs
lines = open(sys.argv[1]).read().split("\n")
pat = re.compile(sys.argv[2])
cur = None; runs = []; last = None
for l in lines:
    m = re.match(r"^(/\S+):(\d+)", l)
    if m: cur = f"{m.group(1).split('/')[-1]}:{m.group(2)}"; continue
    m = re.match(r"^\s+([0-9a-f]+):\t", l)
    if m and cur and pat.search(cur):
        a = int(m.group(1), 16)
        if runs and a - runs[-1][1] <= 8: runs[-1][1] = a
        else: runs.append([a, a])
print(" ".join(f"{a:x}-{b:x}" for a, b in runs))
