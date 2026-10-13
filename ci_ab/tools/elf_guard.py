#!/usr/bin/env python3
"""elf_guard.py <objcopy> <builds root> <out>: per harness variant (build.h) and build, the sha of each TRISC ELF's
loadable sections; checks the A/A pair (m, ma) is identical, and that the new side (n) differs from the head (h) on every
UInt32 and four-tile row math ELF it changes (all but Int32 MAX/MIN and PACK_ISOLATE), and nowhere else. Exits 1 when a
check fails."""
import glob, hashlib, os, re, subprocess, sys, tempfile
OBJ, ROOT, OUT = sys.argv[1:4]
def key(bh):
    t = open(bh).read()
    g = lambda p, d='': (re.search(p, t).group(1) if re.search(p, t) else d)
    return ' '.join([g(r'MATH_FORMAT = ckernel::to_underlying\(DataFormat::(\w+)\)'), 'd32' if 'is_fp32_dest_acc_en = true' in t else 'd16',
                     g(r'ReduceDim::REDUCE_(\w+)'), g(r'PoolType::(\w+)'), 'ct' + g(r'BLOCK_CT_DIM = (\d+)'), 'rt' + g(r'BLOCK_RT_DIM = (\d+)'),
                     g(r'PERF_RUN_TYPE = PerfRunType::(\w+)', '-'), os.path.basename(os.path.dirname(os.path.dirname(bh)))])
def sha(e):
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile() as f:
        for s in ('.init', '.text', '.ldm_data'):
            if subprocess.run([OBJ, '-O', 'binary', '-j', s, e, f.name], capture_output=True).returncode == 0:
                h.update(s.encode() + open(f.name, 'rb').read())
    return h.hexdigest()[:16]
db = {}
for b in sorted(os.listdir(ROOT)):
    for bh in glob.glob(f'{ROOT}/{b}/tt-llk-build/sources/*/*/build.h'):
        k = key(bh)
        for e in sorted(glob.glob(os.path.dirname(bh) + '/elf/*.elf')):
            db.setdefault(b, {})[k + ' ' + os.path.basename(e)] = sha(e)
with open(OUT, 'w') as f:
    for b, m in db.items():
        for k, v in sorted(m.items()):
            f.write(f'{b} {k} {v}\n')
def cmp(a, b):
    ks = sorted(set(db.get(a, {})) & set(db.get(b, {})))
    return ks, [k for k in ks if db[a][k] != db[b][k]]
ok = True
for a, b in (('ra_m', 'ra_ma'),):
    ks, d = cmp(a, b)
    print(f'ELF A/A {a} vs {b}: {len(ks) - len(d)} of {len(ks)} identical'); ok &= bool(ks) and not d
for a, b in (('ra_h', 'ra_n'), ('ra_m', 'ra_n'), ('ra_m', 'ra_h')):
    ks, d = cmp(a, b)
    tgt = [k for k in ks if 'math.elf' in k and 'PACK_ISOLATE' not in k and
           ('UInt32' in k or ('ROW' in k and 'ct4' in k and not ('Int32' in k and ('MAX' in k or 'MIN' in k))))]
    td = [k for k in tgt if k in d]
    print(f'ELF {a} vs {b}: {len(d)} of {len(ks)} differ; UInt32 and four-tile row math ELFs differing {len(td)} of {len(tgt)}')
    if (a, b) == ('ra_h', 'ra_n'):
        other = [k for k in d if k not in tgt]
        ok &= bool(tgt) and len(td) == len(tgt) and not other
        print(f'  differing outside the targets: {len(other)}')
        for k in other[:20]:
            print('   ', k)
print('ELF GUARD ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
