#!/usr/bin/env python3
"""rml.py ELF [FUNC_SUBSTR] [--all]: compact listing of a function (default sender_erisc::run_main_loop); marks the
instructions of its largest strongly connected region (the inner pass loop) with '*'. Prints counts."""
import sys, re, subprocess
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit/tools')
import fncmp
NM = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
elf = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'sender_erisc::run_main_loop'
fs = fncmp.funcs(elf)
dem = {}
for l in subprocess.run([NM, '-C', elf], capture_output=True, text=True).stdout.splitlines():
    p = l.split(None, 2)
    if len(p) == 3:
        dem[int(p[0], 16)] = p[2]
cands = [(n, v) for n, v in fs.items() if sub in dem.get(v[0], '')]
n, (st, sz) = cands[0]
ins = fncmp.disasm(elf, st, st + sz)
reg = set(a for a, _, _ in fncmp.loop_region(ins, st, sz))
names = {v[0]: dem.get(v[0], k) for k, v in fs.items()}
for a, op, arg in ins:
    m = re.search(r'([0-9a-f]+)$', arg)
    tgt = ''
    if op in ('jal', 'j') or op.startswith('b'):
        try:
            t = int(arg.split(',')[-1], 16)
            if t in names and not (st <= t < st + sz):
                tgt = '  -> ' + names[t][:60]
        except ValueError:
            pass
    print(f"{'*' if a in reg else ' '} {a:5x}: {op:8s} {arg}{tgt}")
print(f"# {dem.get(st)[:80]}: {len(ins)} instructions, inner-loop region {len(reg)}", file=sys.stderr)
