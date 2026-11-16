#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""repack.py: build run-ttx-kmdtlb from the pinned run-ttx and kmdtlb.code, byte for byte the same every time.

run-ttx is a PyInstaller onefile binary: an ELF bootloader whose `pydata` section holds a CArchive (the files, a TOC,
an 88 byte cookie). The bootloader runs the archive's 's' (script) entries in TOC order before run-ttx's own main
script. This adds one: pyi_rth_kmdtlb (kmdtlb.code, kmdtlb.py marshalled by Python 3.7), just before `run-ttx`, so
run-ttx's TLB windows come from the driver (kmdtlb.py says how). Every other byte of the archive is kept; the ELF
section header table and .shstrtab move behind the bigger section.

  repack.py <run-ttx> <kmdtlb.code> <out>     write out (mode 0755); prints its sha256
  repack.py --list <binary>                   the TOC
"""

import hashlib
import os
import struct
import sys
import zlib

MAGIC = b"MEI\014\013\012\013\016"
COOKIE = struct.Struct("!8sIIII64s")
ENTRY = struct.Struct("!IIIIBc")
NAME = "pyi_rth_kmdtlb"


class Bad(Exception):
    pass


def elf_pydata(b):
    """(section header list, index of pydata, index of .shstrtab, e_shoff) of a little-endian ELF64"""
    if b[:4] != b"\x7fELF" or b[4] != 2 or b[5] != 1:
        raise Bad("not a little-endian ELF64 file")
    (shoff,) = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    if shentsize != 64:
        raise Bad("section header size %d" % shentsize)
    shdrs = [list(struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * 64)) for i in range(shnum)]
    so, ss = shdrs[shstrndx][4], shdrs[shstrndx][5]
    strtab = b[so : so + ss]

    def name(h):
        return strtab[h[0] : strtab.index(b"\0", h[0])].decode()

    idx = [i for i, h in enumerate(shdrs) if name(h) == "pydata"]
    if len(idx) != 1:
        raise Bad("no single pydata section")
    return shdrs, idx[0], shstrndx, shoff


def toc(arch):
    """(cookie fields, [(pos, clen, ulen, cflag, type, name, raw entry)]) of a CArchive"""
    i = arch.rfind(MAGIC)
    if i < 0 or i + COOKIE.size != len(arch):
        raise Bad("no CArchive cookie at the end of pydata")
    magic, ln, tocpos, toclen, pyver, pylib = COOKIE.unpack_from(arch, i)
    if ln != len(arch):
        raise Bad("archive length %d != pydata size %d" % (ln, len(arch)))
    t = arch[tocpos : tocpos + toclen]
    out, o = [], 0
    while o < len(t):
        el, pos, cl, ul, cf, ty = ENTRY.unpack_from(t, o)
        nm = t[o + ENTRY.size : o + el].split(b"\0")[0].decode()
        out.append((pos, cl, ul, cf, ty.decode(), nm, t[o : o + el]))
        o += el
    return (tocpos, toclen, pyver, pylib), out


def toc_entry(pos, cl, ul, cf, ty, name):
    n = name.encode() + b"\0"
    el = ENTRY.size + len(n)
    el += -el % 16  # PyInstaller pads TOC entries to 16 bytes
    return ENTRY.pack(el, pos, cl, ul, cf, ty.encode()) + n.ljust(el - ENTRY.size, b"\0")


def repack(exe, code):
    shdrs, pi, si, shoff = elf_pydata(exe)
    po, ps = shdrs[pi][4], shdrs[pi][5]
    so, ss = shdrs[si][4], shdrs[si][5]
    if not (po + ps <= so and so + ss <= shoff and shoff + len(shdrs) * 64 == len(exe)):
        raise Bad("pydata, .shstrtab and the section headers are not the last parts of the file, in that order")
    if any(h[4] >= po + ps and i not in (si,) and h[1] != 8 for i, h in enumerate(shdrs) if i != pi):
        raise Bad("a section lies behind pydata")
    arch = exe[po : po + ps]
    (tocpos, toclen, pyver, pylib), ents = toc(arch)
    if pyver != 307:
        raise Bad("run-ttx's Python is %d, kmdtlb.code is for 307" % pyver)
    if any(e[5] == NAME for e in ents):
        raise Bad("%s is in the archive already" % NAME)
    mains = [k for k, e in enumerate(ents) if e[4] == "s" and e[5] == "run-ttx"]
    if len(mains) != 1:
        raise Bad("no single run-ttx script entry")
    if code[:1] not in (b"c", b"\xe3"):
        raise Bad("kmdtlb.code is not a marshalled code object")
    data = zlib.compress(code, 9)
    pos = tocpos  # the new entry's data goes where the old TOC was; every other entry keeps its position
    newtoc = b""
    for k, e in enumerate(ents):
        if k == mains[0]:
            newtoc += toc_entry(pos, len(data), len(code), 1, "s", NAME)
        newtoc += e[6]
    tocpos2 = pos + len(data)
    body = arch[:tocpos] + data + newtoc
    arch2 = body + COOKIE.pack(MAGIC, len(body) + COOKIE.size, tocpos2, len(newtoc), pyver, pylib)
    strtab = exe[so : so + ss]
    so2 = po + len(arch2)
    shoff2 = so2 + ss
    shoff2 += -shoff2 % 8
    shdrs2 = [list(h) for h in shdrs]
    shdrs2[pi][5] = len(arch2)
    shdrs2[si][4] = so2
    head = bytearray(exe[:po])
    struct.pack_into("<Q", head, 0x28, shoff2)
    out = bytes(head) + arch2 + strtab + b"\0" * (shoff2 - so2 - ss)
    out += b"".join(struct.pack("<IIQQQQIIQQ", *h) for h in shdrs2)
    return out


def main(argv):
    if len(argv) == 3 and argv[1] == "--list":
        b = open(argv[2], "rb").read()
        shdrs, pi, _si, _o = elf_pydata(b)
        for e in toc(b[shdrs[pi][4] : shdrs[pi][4] + shdrs[pi][5]])[1]:
            print(e[4], e[1], e[2], e[5])
        return 0
    if len(argv) != 4:
        sys.stderr.write(__doc__)
        return 2
    try:
        out = repack(open(argv[1], "rb").read(), open(argv[2], "rb").read())
    except (Bad, OSError) as e:
        sys.stderr.write("repack.py: %s\n" % e)
        return 1
    tmp = argv[3] + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.chmod(tmp, 0o755)
    os.replace(tmp, argv[3])
    print("%s sha256=%s bytes=%d" % (os.path.basename(argv[3]), hashlib.sha256(out).hexdigest(), len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
