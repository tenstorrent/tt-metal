#!/usr/bin/env python3
"""Per-TRISC code fingerprints of every ELF in an LLK build tree.

Usage: hash_elfs.py <build dir>/sources <objdump> <helpers dir> <out.csv>

Columns per ELF (test_name, variant_id, thread):
  img    sha1 of every allocated PROGBITS section (name + bytes): the exact binary image, no debug info.
  text   sha1 of .text bytes only.
  shape  sha1 of the disassembly of every function, with branch targets inside a function written as
         instruction positions and every other number masked: the instruction sequence and control flow,
         independent of constants (loop counts, dims, addresses).
  window helpers/perf/window_hash.fingerprint: the TILE_LOOP zone, address independent, constants kept.
  text_size  bytes of .text.
"""
import csv, hashlib, os, re, struct, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC, OBJDUMP, HELPERS, OUT = sys.argv[1:5]
sys.path.insert(0, HELPERS)
from perf import window_hash  # noqa: E402

HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
TARGET = re.compile(r"\b([0-9a-f]+) <([^>+]+)(?:\+0x[0-9a-f]+)?>")
NUM = re.compile(r"(?<![A-Za-z_])(-?0x[0-9a-fA-F]+|-?\d+)\b")


def sections(data):
    if data[:4] != b"\x7fELF" or data[4] != 1:
        return []
    e_shoff, = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    hdrs = [struct.unpack_from("<IIIIIIIIII", data, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    stro = hdrs[e_shstrndx][4]
    out = []
    for name, typ, flags, _, off, size, *_ in hdrs:
        n = data[stro + name: data.index(b"\0", stro + name)].decode()
        out.append((n, typ, flags, data[off: off + size] if typ != 8 else b""))
    return out


def shape(text):
    funcs, cur = {}, None
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            cur = m.group(2); funcs[cur] = []; continue
        m = INSN.match(line)
        if m and cur is not None:
            funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))
    lines = []
    for f in sorted(funcs):
        body = funcs[f]
        pos = {a: k for k, (a, _, _) in enumerate(body)}
        lines.append(f"<{f}>")
        for _, mn, ops in body:
            def repl(m):
                a, sym = int(m.group(1), 16), m.group(2)
                return f"@{pos[a]}" if sym == f and a in pos else f"<{sym}>"
            ops = TARGET.sub(repl, ops)
            ops = NUM.sub("N", ops)
            lines.append(f"{mn} {ops}")
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def one(path):
    p = Path(path)
    data = p.read_bytes()
    secs = sections(data)
    img = hashlib.sha1()
    text = b""
    for n, typ, flags, b in secs:
        if flags & 2 and typ == 1:  # SHF_ALLOC, PROGBITS
            img.update(n.encode() + b"\0" + b)
        if n == ".text":
            text = b
    r = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", str(p)], capture_output=True, text=True)
    sh = shape(r.stdout) if r.returncode == 0 else "no-objdump"
    try:
        win = window_hash.fingerprint(OBJDUMP, p)
    except Exception as e:  # keep going; record the failure
        win = "err-" + type(e).__name__
    rel = p.relative_to(SRC).parts  # test_name, variant_id, elf, thread.elf
    return [rel[0], rel[1], rel[-1][:-4], img.hexdigest()[:12], hashlib.sha1(text).hexdigest()[:12], sh, win, len(text)]


if __name__ == "__main__":
    elfs = sorted(str(p) for p in Path(SRC).glob("*/*/elf/*.elf"))
    print(f"{len(elfs)} ELFs", flush=True)
    with ProcessPoolExecutor(os.cpu_count()) as ex, open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_name", "variant_id", "thread", "img", "text", "shape", "window", "text_size"])
        for row in ex.map(one, elfs, chunksize=64):
            w.writerow(row)
