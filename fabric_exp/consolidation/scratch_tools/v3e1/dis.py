import re, sys
lines=open(sys.argv[1]).read().split("\n")
lo, hi = int(sys.argv[2],16), int(sys.argv[3],16)
cur=None
for l in lines:
    m=re.match(r"^(/\S+):(\d+)",l)
    if m: cur=f"{m.group(1).split('/')[-1]}:{m.group(2)}"; continue
    m=re.match(r"^\s+([0-9a-f]+):\t(\S+)\s*(.*)$",l)
    if m:
        a=int(m.group(1),16)
        if lo<=a<hi:
            arg=re.sub(r"\s+#.*$","",m.group(3)); arg=re.sub(r" <[^>]*>","",arg)
            print(f"{a:5x}  {m.group(2):7s} {arg:28s} {cur}")
