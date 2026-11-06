#!/usr/bin/env python3
"""bfc_insn_diff.py BASE.elf NEW.elf: every instruction of the two images' .text that differs (same address, other
text), with its function and source line (addr2line inline chain, innermost frame), plus each image's loadable bytes
and .text size. Exit 0 when the images have the same instruction count at every address."""
import os, re, subprocess, sys

B = os.environ.get("ROUTER_TOOLS_TOOLCHAIN", "/home/aagarwal/tt-metal-mainline-bfc/runtime/sfpi/compiler/bin/riscv-tt-elf-")


def sh(a, inp=None):
    return subprocess.run(a, capture_output=True, text=True, input=inp).stdout


def dis(elf):
    out = {}
    fn = None
    for l in sh([B + "objdump", "-d", "--no-show-raw-insn", "-j", ".text", elf]).splitlines():
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", l)
        if m:
            fn = m.group(2)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+(\S.*)$", l)
        if m:
            out[int(m.group(1), 16)] = (fn, re.sub(r"\s+", " ", m.group(2).split("#")[0].split("<")[0]).strip())
    return out


def code_bytes(elf):
    out = sh([B + "readelf", "-lW", elf])
    return sum(int(m.group(1), 16) for m in re.finditer(r"^\s+LOAD\s+\S+\s+\S+\s+\S+\s+(0x[0-9a-f]+)", out, re.M))


def text_size(elf):
    for l in sh([B + "readelf", "-SW", elf]).splitlines():
        m = re.search(r"\s\.text\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
        if m:
            return int(m.group(1), 16)
    return 0


def where(elf, addrs):
    out = sh([B + "addr2line", "-i", "-f", "-C", "-a", "-e", elf], "\n".join(hex(a) for a in addrs)).splitlines()
    res, cur, i = {}, None, 0
    while i < len(out):
        if out[i].startswith("0x"):
            cur = int(out[i], 16)
            i += 1
            continue
        if cur not in res:
            fn = re.sub(r"\(.*", "", out[i]).split("::")[-1]
            res[cur] = f"{fn} {out[i + 1].split('/')[-1].split(' ')[0]}"
        i += 2
    return res


a, b = sys.argv[1], sys.argv[2]
da, db = dis(a), dis(b)
diff = [x for x in sorted(set(da) | set(db)) if da.get(x, (None, None))[1] != db.get(x, (None, None))[1]]
print(f"code bytes {code_bytes(a)} -> {code_bytes(b)}; .text {text_size(a)} -> {text_size(b)}; "
      f"instructions {len(da)} -> {len(db)}; differing addresses {len(diff)}")
w = where(b, diff[:40])
for x in diff[:40]:
    print(f"  {x:#07x} {db.get(x, ('?', '-'))[0][:60]:60s} {da.get(x, (None, '-'))[1]:28s} -> {db.get(x, (None, '-'))[1]:28s} {w.get(x, '')}")
sys.exit(0 if len(da) == len(db) and set(da) == set(db) else 1)
