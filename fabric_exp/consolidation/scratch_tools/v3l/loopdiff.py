import sys, difflib, re
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range, dis, stat
def body(elf, sub="e0::run_main_loop"):
    st = stat(elf, sub)
    a, b = fn_range(elf, sub)
    h, e = int(st['header'], 16), int(st['backedge'], 16)
    out = []
    names = {}
    def canon(m):
        r = m.group(0)
        if r in ('sp', 'zero', 'ra'):
            return r
        names.setdefault(r, f"r{len(names)}")
        return names[r]
    for ad, op, ar in dis(elf, a, b):
        if h <= ad <= e:
            ar = re.sub(r'\b[0-9a-f]{4,}\b', 'T', ar) if op.startswith(('b', 'j')) else ar
            if '-raw' not in sys.argv:
                ar = re.sub(r'\b(?:[ast][0-9]+|sp|zero|ra)\b', canon, ar)
                ar = re.sub(r'-?[0-9]+\(sp\)', 'N(sp)', ar)
            out.append(f"{op} {ar}")
    return out
A, B = body(sys.argv[1]), body(sys.argv[2])
for l in difflib.unified_diff(A, B, lineterm='', n=1):
    print(l)
