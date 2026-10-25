"""ringworker.py A.elf B.elf: the ring E0 pass up to channel 1's test (worker visit + connection check), opcode diff."""
import sys, difflib
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from segdiff import segs
def worker(elf, ch1_test):
    a, _ = segs(elf)
    idx = [i for i, x in enumerate(a) if x == ch1_test]
    return a[:idx[0] if ch1_test == 'andi R,R,16' else idx[1]]
A = worker(sys.argv[1], 'andi R,R,16'); B = worker(sys.argv[2], 'andi R,R,2047')
sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
ops = [(t, A[i1:i2], B[j1:j2]) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal']
print(len(A), len(B), sum(1 for x in A if x.startswith('lw R,N(sp)')), sum(1 for x in B if x.startswith('lw R,N(sp)')), ops)
