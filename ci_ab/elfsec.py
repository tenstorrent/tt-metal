#!/usr/bin/env python3
"""Loaded-image identity of ELF pairs: every SHF_ALLOC PROGBITS section's bytes and address (not .profiler_meta).
usage: elfsec.py <list file of 'a b' lines> -> prints same/differs per line"""
import struct, sys
def secs(p):
    d = open(p, 'rb').read()
    shoff, = struct.unpack_from('<I', d, 0x20); shentsize, shnum, shstrndx = struct.unpack_from('<HHH', d, 0x2e)
    sh = [struct.unpack_from('<IIIIIIIIII', d, shoff + i * shentsize) for i in range(shnum)]
    st = sh[shstrndx]; names = d[st[4]:st[4] + st[5]]
    out = {}
    for name, typ, flags, addr, off, size, *_ in sh:
        n = names[name:names.index(b'\0', name)].decode()
        if typ == 1 and flags & 2 and n != '.profiler_meta': out[n] = (addr, d[off:off + size])
    return out
for line in open(sys.argv[1]):
    a, b = line.split()
    print('same' if secs(a) == secs(b) else 'differs')
