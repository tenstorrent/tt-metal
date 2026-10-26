"""loadcmp.py A_DIR B_DIR PROC [PROC...]: compare every config's ELF loadable image (PT_LOAD segments: vaddr, memsz,
file bytes) between two build trees, for the given processor dirs (active_erisc = E0, subordinate_active_erisc = E1)."""
import struct, sys, os, glob
def loads(path):
    d = open(path, 'rb').read()
    assert d[:4] == b'\x7fELF' and d[4] == 1  # ELF32
    phoff, = struct.unpack_from('<I', d, 28); phentsize, phnum = struct.unpack_from('<HH', d, 42)
    segs = []
    for i in range(phnum):
        p_type, p_off, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align = struct.unpack_from('<8I', d, phoff + i * phentsize)
        if p_type == 1:
            segs.append((p_vaddr, p_paddr, p_memsz, d[p_off:p_off + p_filesz]))
    return segs
A, B = sys.argv[1], sys.argv[2]
for proc in sys.argv[3:]:
    same = diff = miss = 0; bad = []
    for e in sorted(glob.glob(f'{A}/*/{proc}/{proc}.elf')):
        h = e.split('/')[-3]; b = f'{B}/{h}/{proc}/{proc}.elf'
        if not os.path.exists(b): miss += 1; continue
        if loads(e) == loads(b): same += 1
        else: diff += 1; bad.append(h)
    print(f'{proc}: identical={same} differ={diff} missing_in_B={miss}' + (f' e.g. {bad[:3]}' if bad else ''))
