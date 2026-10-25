#!/usr/bin/env python3
"""emu.py ELF [options]: run an E0 main-loop function of a router ELF in a small RV32 emulator with a scripted
device model, and count instructions, taken branches and stack accesses per pass (pass = one visit of the loop
header, the grouped free-slots read).

Device model:
  - sender-channel free-slots registers (stream reg 0x4a4): constant per stream, from --raw S:V,... (default: the
    channel's slot count, i.e. idle; given as --busy S:N to read N-1);
  - local free-slots update writes (stream reg 0x438) count a send on that stream;
  - completion counters (L1 COMP_BASE + 4*ch): every --comp-every passes, set to that channel's send count;
  - TXQ, NoC command buffers: never busy; everything else is plain memory (zero unless written).
"""
import argparse, re, subprocess, struct, sys, collections

OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
NM = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
REGS = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2', 's0', 's1', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
        's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']
RI = {r: i for i, r in enumerate(REGS)}
RI['fp'] = 8
M32 = 0xffffffff


def sx(v, bits=32):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def load_prog(elf):
    out = subprocess.run([OD, '-d', '-M', 'no-aliases', '--no-show-raw-insn', elf], capture_output=True, text=True).stdout
    prog = {}
    for l in out.splitlines():
        m = re.match(r'\s+([0-9a-f]+):\t(\S+)\s*(.*)$', l)
        if m:
            ops = re.sub(r'\s*[#<].*$', '', m.group(3)).strip()
            prog[int(m.group(1), 16)] = (m.group(2), [o.strip() for o in ops.split(',')] if ops else [])
    return prog


def load_mem(elf):
    d = open(elf, 'rb').read()
    phoff, = struct.unpack_from('<I', d, 28)
    phentsize, phnum = struct.unpack_from('<HH', d, 42)
    mem = {}
    for i in range(phnum):
        t, off, va, pa, fs, ms, fl, al = struct.unpack_from('<8I', d, phoff + i * phentsize)
        if t == 1:
            for j in range(fs):
                mem[va + j] = d[off + j]
    return mem


def sym(elf, sub):
    for l in subprocess.run([NM, '-C', elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and sub in p[2] and p[1] in 'tT':
            return int(p[0], 16)
    raise SystemExit(f'no symbol {sub}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf')
    ap.add_argument('--func', default='e0::run_main_loop')
    ap.add_argument('--slots', default='0:17,1:16,2:16,3:16', help='stream:slot count')
    ap.add_argument('--busy', default='', help='streams that always hold a packet (raw = slots - 1)')
    ap.add_argument('--log-writes', default=None, help='file: every device-register write (address, value) in order')
    ap.add_argument('--chmap', default='', help='stream:channel for completion counters (default: same index)')
    ap.add_argument('--every', default='', help='S:K,... stream S holds a packet only on every K-th pass')
    ap.add_argument('--comp-base', type=lambda x: int(x, 0), default=0x15bb0)
    ap.add_argument('--comp-every', type=int, default=2)
    ap.add_argument('--passes', type=int, default=40)
    ap.add_argument('--warm', type=int, default=8)
    ap.add_argument('--header', type=lambda x: int(x, 16), default=None)
    ap.add_argument('--pass-reg', type=lambda x: int(x, 0), default=0xffb404a4)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--max', type=int, default=2_000_000)
    ap.add_argument('--skip', default='', help='comma list of function-name substrings to return from immediately')
    ap.add_argument('--poke', default='', help='ADDR=VAL,... initial words')
    ap.add_argument('--watch', type=lambda x: int(x, 16), default=None)
    ap.add_argument('--force', default='', help='ADDR=VAL,... words that always read as VAL')
    ap.add_argument('--stuck', type=int, default=200000, help='report the hottest pcs if no pass after this many steps')
    a = ap.parse_args()

    prog = load_prog(a.elf)
    mem = collections.defaultdict(int, load_mem(a.elf))
    slots = {int(k): int(v) for k, v in (x.split(':') for x in a.slots.split(',') if x)}
    busy = {int(x) for x in a.busy.split(',') if x}
    raw = {s: (n - 1 if s in busy else n) for s, n in slots.items()}
    every = {int(k): int(v) for k, v in (x.split(':') for x in a.every.split(',') if x)}
    sends = collections.Counter()
    stream_ch = {}  # stream -> channel index for completion counters: stream s is channel s (VC0) unless mapped
    for s in slots:
        stream_ch[s] = s
    for kv in [t for t in a.chmap.split(',') if t]:
        k, v = kv.split(':'); stream_ch[int(k)] = int(v)

    x = [0] * 32
    x[RI['sp']] = 0x10100000
    x[RI['ra']] = 0xdead0000
    x[RI['a0']] = 0x20000000  # channel worker interface tuple (zeroed)
    x[RI['a1']] = 0x20100000  # stream id array
    for i in range(16):
        struct.pack_into
        for b in range(4):
            mem[0x20100000 + 4 * i + b] = (i >> (8 * b)) & 0xff
    mem[0] = 1  # the worker connection semaphore (pointer 0 in the zeroed tuple): open
    pc = sym(a.elf, a.func)
    header = a.header
    skips = set()
    for sub in [t for t in a.skip.split(',') if t]:
        for l in subprocess.run([NM, '-C', a.elf], capture_output=True, text=True).stdout.splitlines():
            p = l.split(None, 2)
            if len(p) == 3 and sub in p[2] and p[1] in 'tT':
                skips.add(int(p[0], 16))
    for kv in [t for t in a.poke.split(',') if t]:
        k, v = kv.split('=')
        for b in range(4):
            mem[int(k, 0) + b] = (int(v, 0) >> (8 * b)) & 0xff
    hot = collections.Counter()
    last = collections.deque(maxlen=400)
    spin = collections.Counter()

    force = {int(k, 0): int(v, 0) for k, v in (t.split('=') for t in a.force.split(',') if t)}

    def rd(addr, n, signed=False):
        addr &= M32
        if addr in force:
            v = force[addr]
        elif 0xffb40000 <= addr < 0xffb80000 and (addr & 0xfff) == 0x4a4:
            v = raw.get((addr - 0xffb40000) >> 12, 0)
        elif 0xffb90000 <= addr < 0xffba0000 or 0xffb20000 <= addr < 0xffb30000:
            v = 0
        else:
            v = sum(mem[addr + i] << (8 * i) for i in range(n))
        return sx(v, 8 * n) & M32 if signed else v

    wlog = [] if a.log_writes else None

    def wr(addr, n, v):
        addr &= M32
        if wlog is not None and addr >= 0xffb20000 and cur is not None:
            wlog.append(f'{addr:#x} {v & M32:#x}')
        if 0xffb40000 <= addr < 0xffb80000 and (addr & 0xfff) == 0x438:
            sends[(addr - 0xffb40000) >> 12] += 1
            return
        if addr >= 0xffb20000 and not (0xffb00000 <= addr < 0xffb10000):
            return
        for i in range(n):
            mem[addr + i] = (v >> (8 * i)) & 0xff

    def r(name):
        return x[RI[name]]

    def w(name, v):
        i = RI[name]
        if i:
            x[i] = v & M32

    def memop(op):
        m = re.match(r'(-?\d+)\((\w+)\)', op)
        return (r(m.group(2)) + int(m.group(1))) & M32, m.group(2) == 'sp'

    passes = []
    cur = None
    npass = 0
    steps = 0
    while steps < a.max:
        steps += 1
        if pc == 0xdead0000:
            print('returned'); break
        if pc in skips:
            pc = r('ra'); continue
        if pc not in prog:
            raise SystemExit(f'pc {pc:#x} not in program; last: ' + ' '.join(f'{p:x}' for p in last))
        last.append(pc)
        if a.watch == pc:
            print('watch', ' '.join(f'{REGS[i]}={x[i]:#x}' for i in range(1, 32))); a.watch = None
        if cur is None:
            hot[pc] += 1
            if steps == a.stuck:
                print("no pass yet; last:", " ".join(f"{p:x}" for p in list(last)[-400::5])); print("hottest:", ', '.join(f'{p:x}:{c}' for p, c in hot.most_common(12)))
                raise SystemExit(1)
        mn, ops = prog[pc]
        is_start = False
        if header is not None:
            is_start = pc == header
        elif mn == 'lw' and ops[1].startswith('1188('):
            is_start = memop(ops[1])[0] == a.pass_reg  # the pass starts at the worker channel's free-slots read
        if is_start:
            if cur is not None:
                passes.append(cur)
            npass += 1
            for s_, k_ in every.items():
                raw[s_] = slots[s_] - 1 if npass % k_ == 0 else slots[s_]
            if npass % a.comp_every == 0:
                for s, c in sends.items():
                    ch = stream_ch.get(s, s)
                    v = c
                    for b in range(4):
                        mem[a.comp_base + 4 * ch + b] = (v >> (8 * b)) & 0xff
            if len(passes) >= a.warm + a.passes:
                break
            cur = collections.Counter()
            cur['sends'] = 0
            cur['_start_sends'] = sum(sends.values())
        if cur is not None:
            cur['insns'] += 1
            if a.trace and len(passes) >= a.warm and len(passes) < a.warm + 2:
                print(f'{pc:x}: {mn} {",".join(ops)}')
        npc = pc + 4
        if mn == 'lui':
            w(ops[0], int(ops[1], 0) << 12)
        elif mn == 'auipc':
            w(ops[0], pc + (int(ops[1], 0) << 12))
        elif mn in ('addi', 'xori', 'ori', 'andi', 'slti', 'sltiu', 'slli', 'srli', 'srai'):
            s1 = r(ops[1]); imm = int(ops[2], 0)
            v = {'addi': lambda: s1 + imm, 'xori': lambda: s1 ^ (imm & M32), 'ori': lambda: s1 | (imm & M32),
                 'andi': lambda: s1 & (imm & M32), 'slti': lambda: int(sx(s1) < imm),
                 'sltiu': lambda: int(s1 < (imm & M32)), 'slli': lambda: s1 << imm, 'srli': lambda: s1 >> imm,
                 'srai': lambda: sx(s1) >> imm}[mn]()
            w(ops[0], v)
        elif mn in ('add', 'sub', 'and', 'or', 'xor', 'sll', 'srl', 'sra', 'slt', 'sltu', 'sh1add', 'sh2add',
                    'sh3add', 'mul', 'andn', 'orn', 'min', 'max', 'minu', 'maxu', 'czero.eqz', 'czero.nez'):
            s1 = r(ops[1]); s2 = r(ops[2])
            v = {'add': lambda: s1 + s2, 'sub': lambda: s1 - s2, 'and': lambda: s1 & s2, 'or': lambda: s1 | s2,
                 'xor': lambda: s1 ^ s2, 'sll': lambda: s1 << (s2 & 31), 'srl': lambda: s1 >> (s2 & 31),
                 'sra': lambda: sx(s1) >> (s2 & 31), 'slt': lambda: int(sx(s1) < sx(s2)), 'sltu': lambda: int(s1 < s2),
                 'sh1add': lambda: (s1 << 1) + s2, 'sh2add': lambda: (s1 << 2) + s2, 'sh3add': lambda: (s1 << 3) + s2,
                 'mul': lambda: s1 * s2, 'andn': lambda: s1 & ~s2, 'orn': lambda: s1 | (~s2 & M32),
                 'min': lambda: min(sx(s1), sx(s2)), 'max': lambda: max(sx(s1), sx(s2)), 'minu': lambda: min(s1, s2),
                 'maxu': lambda: max(s1, s2), 'czero.eqz': lambda: 0 if s2 == 0 else s1,
                 'czero.nez': lambda: 0 if s2 != 0 else s1}[mn]()
            w(ops[0], v)
        elif mn in ('zext.h', 'zext.b', 'sext.h', 'sext.b'):
            s1 = r(ops[1])
            v = {'zext.h': s1 & 0xffff, 'zext.b': s1 & 0xff, 'sext.h': sx(s1, 16), 'sext.b': sx(s1, 8)}[mn]
            w(ops[0], v)
        elif mn in ('lw', 'lh', 'lhu', 'lb', 'lbu'):
            addr, spr = memop(ops[1])
            n = {'lw': 4, 'lh': 2, 'lhu': 2, 'lb': 1, 'lbu': 1}[mn]
            w(ops[0], rd(addr, n, signed=mn in ('lh', 'lb')))
            if cur is not None:
                cur['loads'] += 1
                if spr:
                    cur['sp_loads'] += 1
        elif mn in ('sw', 'sh', 'sb'):
            addr, spr = memop(ops[1])
            wr(addr, {'sw': 4, 'sh': 2, 'sb': 1}[mn], r(ops[0]))
            if cur is not None:
                cur['stores'] += 1
                if spr:
                    cur['sp_stores'] += 1
        elif mn in ('beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu'):
            s1 = r(ops[0]); s2 = r(ops[1])
            t = {'beq': s1 == s2, 'bne': s1 != s2, 'blt': sx(s1) < sx(s2), 'bge': sx(s1) >= sx(s2),
                 'bltu': s1 < s2, 'bgeu': s1 >= s2}[mn]
            tgt = int(ops[2], 16)
            if t and tgt <= pc and cur is None:  # init waits (barriers, handshakes): break out after a while
                spin[pc] += 1
                if spin[pc] > (100 if pc - 16 <= tgt else 400):
                    t = False
                    spin[pc] = 0
            if t:
                npc = tgt
                if cur is not None:
                    cur['taken'] += 1
        elif mn == 'jal':
            w(ops[0], pc + 4)
            npc = int(ops[1], 16)
            if cur is not None:
                cur['taken'] += 1
                if ops[0] == 'ra':
                    cur['calls'] += 1
        elif mn == 'jalr':
            m = re.match(r'(-?\d+)\((\w+)\)', ops[1])
            t = (r(m.group(2)) + int(m.group(1))) & ~1
            w(ops[0], pc + 4)
            npc = t if (t in prog or t == 0xdead0000) else pc + 4  # firmware entry points: treat as returning
            if cur is not None:
                cur['taken'] += 1
        elif mn.startswith('fence'):
            pass
        elif mn.startswith('csrr'):
            w(ops[0], steps)
        else:
            raise SystemExit(f'unhandled {mn} {ops} at {pc:#x}')
        pc = npc & M32
        if cur is not None:
            cur['sends'] = sum(sends.values()) - cur['_start_sends']
    if wlog is not None:
        open(a.log_writes, 'w').write('\n'.join(wlog) + '\n')
    ss = passes[a.warm:]
    if not ss:
        raise SystemExit(f'no passes measured (header {header and hex(header)}, steps {steps})')
    tot = collections.Counter()
    for p in ss:
        tot.update(p)
    keys = ['insns', 'taken', 'loads', 'sp_loads', 'stores', 'sp_stores', 'calls', 'sends']
    print(f'{len(ss)} passes; per pass: ' +
          ', '.join(f'{k} {tot[k] / len(ss):.2f}' for k in keys))
    pat = collections.Counter((p['insns'], p['taken'], p['loads'], p['sends']) for p in ss)
    print('pass shapes (insns, taken, loads, sends): ' + ', '.join(f'{k}x{v}' for k, v in sorted(pat.items())))


if __name__ == '__main__':
    main()
