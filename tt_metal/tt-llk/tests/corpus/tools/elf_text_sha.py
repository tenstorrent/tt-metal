#!/usr/bin/env python3
"""laneMK — dependency-free .text-section sha256 of an ELF (RISCV32/ELF32/64, LE/BE).

The object-identity anchor is the math kernel's .text (laneKC: the whole ELF file is not
byte-reproducible across compiles — debug info / offsets vary — but .text is stable). This
extractor lets the exabox worker gate on .text without the toolchain (no objcopy). Pure
stdlib; matches `riscv-tt-elf-objcopy -O binary --only-section=.text elf | sha256sum`.
"""
import hashlib
import struct
import sys


def text_sha256(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF")
    ei_class = data[4]  # 1=32-bit, 2=64-bit
    ei_data = data[5]  # 1=LE, 2=BE
    en = "<" if ei_data == 1 else ">"
    if ei_class == 1:  # ELF32
        e_shoff = struct.unpack_from(en + "I", data, 0x20)[0]
        e_shentsize = struct.unpack_from(en + "H", data, 0x2E)[0]
        e_shnum = struct.unpack_from(en + "H", data, 0x30)[0]
        e_shstrndx = struct.unpack_from(en + "H", data, 0x32)[0]
        name_o, off_o, size_o, name_fmt, off_fmt = 0x00, 0x10, 0x14, "I", "I"
    else:  # ELF64
        e_shoff = struct.unpack_from(en + "Q", data, 0x28)[0]
        e_shentsize = struct.unpack_from(en + "H", data, 0x3A)[0]
        e_shnum = struct.unpack_from(en + "H", data, 0x3C)[0]
        e_shstrndx = struct.unpack_from(en + "H", data, 0x3E)[0]
        name_o, off_o, size_o, name_fmt, off_fmt = 0x00, 0x18, 0x20, "I", "Q"

    def sh(i):
        base = e_shoff + i * e_shentsize
        name = struct.unpack_from(en + name_fmt, data, base + name_o)[0]
        off = struct.unpack_from(en + off_fmt, data, base + off_o)[0]
        size = struct.unpack_from(en + off_fmt, data, base + size_o)[0]
        return name, off, size

    str_off = sh(e_shstrndx)[1]

    def sname(name_idx):
        end = data.index(b"\x00", str_off + name_idx)
        return data[str_off + name_idx : end].decode("ascii", "replace")

    for i in range(e_shnum):
        name, off, size = sh(i)
        if sname(name) == ".text":
            return hashlib.sha256(data[off : off + size]).hexdigest()
    raise ValueError("no .text section")


if __name__ == "__main__":
    print(text_sha256(sys.argv[1]))
