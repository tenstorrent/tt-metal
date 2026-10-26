import re, difflib, sys, glob, os
S="/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v"
A, B, fn = sys.argv[1], sys.argv[2], sys.argv[3]
def sect(path):
    s=open(path, errors="replace").read()
    i=s.find(f";; Function {fn} ")
    if i < 0: return None
    j=s.find("\n;; Function", i+10)
    return s[i:j if j>0 else None].split("\n",1)[1].split("\n")
def norm(x):
    x = re.sub(r"/v3t/[^/]+/overlay/", "/X/overlay/", x)
    return re.sub(r"\d+", "#", re.sub(r"0x[0-9a-f]+", "#", x))
for f in sorted(glob.glob(f"{S}/{A}/*/subordinate_active_erisc/*ltrans0.ltrans.*r.*"), key=lambda p: int(re.search(r"ltrans\.(\d+)r", p).group(1))):
    suf = f.split("ltrans0.ltrans.")[1]
    g = glob.glob(f"{S}/{B}/*/subordinate_active_erisc/*ltrans0.ltrans.{suf}")
    if not g: continue
    a, b = sect(f), sect(g[0])
    if a is None or b is None: continue
    na=[norm(x) for x in a]; nb=[norm(x) for x in b]
    if na != nb:
        print("first differing pass (numbers ignored):", suf, len(a), len(b))
        sm=difflib.SequenceMatcher(None,na,nb,autojunk=False)
        k=0
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if tag!="equal":
                for x in a[max(0,i1-2):i2+1][:8]: print("  A|", x[:160])
                for x in b[max(0,j1-2):j2+1][:8]: print("  B|", x[:160])
                k+=1
                if k>=3: break
        break
    else:
        print("same (numbers ignored):", suf)
