"""segdiff.py A.elf B.elf: compare e0::run_main_loop's pass, split at the VC1 group read (lw ..,1188 of a stream
other than VC0's group), opcode-level (registers, stack offsets, branch targets ignored), and the whole function."""
import sys, difflib, re
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range, dis, stat
def norm(op, ar):
    ar = re.sub(r'\b[0-9a-f]{4,}\b', 'T', ar) if op.startswith(('b', 'j')) else ar
    ar = re.sub(r'\b(?:[ast][0-9]+|zero|ra)\b', 'R', ar)
    ar = re.sub(r'-?[0-9]+\(sp\)', 'N(sp)', ar)
    return f"{op} {ar}"
def segs(elf, sub="e0::run_main_loop"):
    st = stat(elf, sub); a, b = fn_range(elf, sub)
    h, e = int(st['header'], 16), int(st['backedge'], 16)
    ins = [(ad, op, ar) for ad, op, ar in dis(elf, a, b) if h <= ad <= e]
    # VC1 group: first lw ..,1188 after the first four grouped reads, whose base lui is not 0xffb40..0xffb4d range of VC0
    cut = len(ins)
    n1188 = 0
    for i, (ad, op, ar) in enumerate(ins):
        if op == 'lw' and ',1188(' in ar:
            n1188 += 1
            if n1188 > NVC0:
                # back up to the preceding lui block
                j = i
                while j > 0 and ins[j - 1][1] in ('lui', 'lw') and (ins[j-1][1] == 'lui' or ',1188(' in ins[j-1][2]):
                    j -= 1
                cut = j; break
    return [norm(op, ar) for _, op, ar in ins[:cut]], [norm(op, ar) for _, op, ar in ins[cut:]]
NVC0 = int(next((a for a in sys.argv[3:] if a.isdigit()), '4')) if __name__ == "__main__" else 2
def main():
    A0, A1 = segs(sys.argv[1]); B0, B1 = segs(sys.argv[2])
    for name, A, B in (("VC0 part", A0, B0), ("VC1 part", A1, B1)):
        sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
        d = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal')
        print(f"{name}: {len(A)} -> {len(B)} instructions, {d} differ")
        if '-v' in sys.argv:
            for t, i1, i2, j1, j2 in sm.get_opcodes():
                if t != 'equal':
                    print(f"  @{i1}/{j1}")
                    for l in A[i1:i2]: print('   -', l)
                    for l in B[j1:j2]: print('   +', l)
if __name__ == "__main__":
    main()
