import sys, difflib, re
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range, dis, stat
def body(elf, sub="e0::run_main_loop"):
    st = stat(elf, sub); a, b = fn_range(elf, sub)
    h, e = int(st['header'], 16), int(st['backedge'], 16)
    out = []
    for ad, op, ar in dis(elf, a, b):
        if h <= ad <= e:
            ar = re.sub(r'\b[0-9a-f]{4,}\b', 'T', ar) if op.startswith(('b', 'j')) else ar
            ar = re.sub(r'\b(?:[ast][0-9]+|zero|ra)\b', 'R', ar)
            ar = re.sub(r'-?[0-9]+\(sp\)', 'N(sp)', ar)
            out.append(f"{op} {ar}")
    return out
A, B = body(sys.argv[1]), body(sys.argv[2])
sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
ch = [(t, A[i1:i2], B[j1:j2]) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal']
print(f"{len(A)} -> {len(B)} instructions; {sum(max(len(x),len(y)) for _,x,y in ch)} differ (registers and stack offsets ignored)")
for t, x, y in ch:
    for l in x: print('-', l)
    for l in y: print('+', l)
