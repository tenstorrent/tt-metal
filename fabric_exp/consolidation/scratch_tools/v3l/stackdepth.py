import sys, glob, os, re, subprocess, concurrent.futures
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range, dis
def frame(elf, sub):
    r = fn_range(elf, sub)
    if not r: return 0
    ins = dis(elf, r[0], min(r[1], r[0] + 64))
    return next((int(x[2].split(",")[-1].lstrip("-")) for x in ins if x[1] == "addi" and x[2].startswith("sp,sp,-")), 0)
def depth(elf):
    km = frame(elf, 'kernel_main')
    ml = frame(elf, 'e0::run_main_loop')
    vf = max(frame(elf, 'e0::visit_forwarding_channel'), frame(elf, 'e0::worker_connection_event'))
    return km, ml, vf, km + ml + vf
d = sys.argv[1]
elfs = sorted(glob.glob(f'{d}/*/active_erisc/active_erisc.elf'))
with concurrent.futures.ThreadPoolExecutor(48) as ex:
    res = list(ex.map(depth, elfs))
tot = [r[3] for r in res]
print(d.split('/')[-1], 'E0 configs', len(res), 'max depth (kernel_main+main loop+callee)', max(tot), 'median', sorted(tot)[len(tot)//2])
i = tot.index(max(tot)); print('  deepest', elfs[i].split('/')[-3], res[i])
