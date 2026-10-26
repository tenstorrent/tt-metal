#!/usr/bin/env python3
"""paths.py BUILD_PREFIX [BUILD_PREFIX...] [--levels O3,O2,Os]: E0 per-pass counts from the RV32 emulator (emu2.py) on the
five hot E0 configs. Per scenario: mean per pass over 64 passes (two rounds, so per-round work is amortized at 1/32),
as instructions / taken branches / loads and score = instructions + 2 x taken + loads (V3L_NOTES §11's proxy).
Scenarios: worker send (worker always has a packet), forwarded send (channel 1, or the first VC1 channel), both, idle,
and 'wait' (worker always has a packet, completions arrive every K passes so the far ring fills: reports whether the
pass that takes the completions also sends)."""
import subprocess, sys, re
E = '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit'
CFG = {
    'W':    ('10733501891889981293', '0:17,1:16,2:16,3:16', '0x15bb0', None, [('wrk', '0'), ('fwd', '1'), ('both', '0,1'), ('idle', '')], 20),
    'NS':   ('8414232161770541884', '0:17,1:16,2:16,3:16', '0x15bb0', None, [('wrk', '0'), ('fwd', '1'), ('both', '0,1'), ('idle', '')], 20),
    '1D':   ('10744374302904868627', '0:16,1:16', '0x15b90', None, [('wrk', '0'), ('fwd', '1'), ('both', '0,1'), ('idle', '')], 40),
    'RING': ('16671252331734229980', '0:16,1:16', '0x15b90', '0x15b80', [('wrk', '0'), ('fwd', '1'), ('both', '0,1'), ('idle', '')], 40),
    'VC1':  ('1526198486225049198', '0:51,1:4,2:4,3:4,14:2,15:2,16:2', '0x15bb0', None, [('wrk', '0'), ('fwd', '1'), ('vc1', '14'), ('idle', '')], 12),
}
def emu(elf, slots, base, ack, busy, extra=()):
    cmd = ['python3', f'{E}/tools/emu2.py', elf, '--func', 'sender_erisc::run_main_loop', '--slots', slots, '--comp-base', base,
           '--busy', busy, '--chmap', '14:4,15:5,16:6', *extra]
    if ack:
        cmd += ['--ack-base', ack]
    return subprocess.run(cmd, capture_output=True, text=True).stdout
def mean(out):
    m = re.search(r'insns ([\d.]+), taken ([\d.]+), loads ([\d.]+)', out)
    if not m:
        return None
    return tuple(map(float, m.groups()))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
levels = 'O3,O2,Os'
if '--levels' in sys.argv:
    levels = sys.argv[sys.argv.index('--levels') + 1]
    args = [a for a in args if a != levels]
for lvl in levels.split(','):
    for cname, (h, slots, base, ack, scen, kwait) in CFG.items():
        for pre in args:
            elf = f'{pre}_{lvl}/{h}/active_erisc/active_erisc.elf'
            cells = []
            for sname, busy in scen:
                r = mean(emu(elf, slots, base, ack, busy, ['--passes', '64', '--warm', '32']))
                cells.append(f'{sname} ' + ('n/a' if r is None else f'{r[0]:.1f}/{r[1]:.1f}/{r[2]:.1f} [{r[0] + 2 * r[1] + r[2]:.0f}]'))
            out = emu(elf, slots, base, ack, '0', ['--comp-every', str(kwait), '--passes', str(3 * kwait), '--warm', str(kwait), '--seq'])
            m = re.search(r'lag: (.*)', out)
            lags = sorted(set(m.group(1).split())) if m else []
            wait = 'wait: passes from credit frame to send ' + (','.join(lags) if lags else 'n/a')
            print(f'-{lvl} {cname:4s} {pre.split("/")[-1]:8s} ' + ' | '.join(cells) + ' | ' + wait)
