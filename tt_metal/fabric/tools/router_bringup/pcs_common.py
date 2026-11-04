# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md). BLACKHOLE ONLY: readers of pc_sampler.py runs.

- load_run(prefix): the run's targets, layouts and samples.
- match_build(cache, erisc, layout, kernel): the kernel-cache ELF that a target ran (its image head equals the start of
  the cache image's loadable text) and that image's link base.
- kernel_addrs(samples, layout, base): the samples inside the kernel image, as ELF addresses, the first and last 5% by
  time dropped (outside the traffic window).
- a2l(elf, addrs): {addr: [(function, file, line)]}, innermost first (addr2line -a -f -i -C; the JIT builds with -g).
"""
import collections
import glob
import json
import os
import re
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_tools_common import toolchain_prefix  # noqa: E402

A2L = toolchain_prefix() + "addr2line"
# Blackhole: ERISC0 runs the active_erisc image, ERISC1 the subordinate_active_erisc image.
PROC = {0: "active_erisc", 1: "subordinate_active_erisc"}


def load_run(prefix):
    meta = json.load(open(prefix + ".json"))
    samples = collections.defaultdict(list)
    for t, i, pc in struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()):
        samples[i].append((t, pc))
    return meta, samples


def layout(meta, tgt):
    key = f"{tgt['chip']}/{tgt['chan']}"
    lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
    return lay.get("erisc", {}).get(str(tgt["erisc"]), {}) if isinstance(lay, dict) else {}


def text_segment(path):
    d = open(path, "rb").read()
    (phoff,) = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<IIIIIIII", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off : p_off + p_filesz]
    raise ValueError(path)


_XIPS = {}


def match_build(cache, erisc, lay, kernel="fabric_erisc_router*"):
    """(elf, link base, hash) of the cache build whose relocated image starts with the target's image head, or None."""
    proc = PROC[erisc]
    head = bytes.fromhex(lay.get("head", "")) if lay else b""
    if not head:
        return None
    key = (cache, proc, kernel)
    if key not in _XIPS:
        _XIPS[key] = [
            (e, *text_segment(e))
            for e in sorted(glob.glob(f"{cache}/**/kernels/{kernel}/*/{proc}/{proc}.elf.xip.elf", recursive=True))
        ]
    for e, base, text in _XIPS[key]:
        if text[: len(head)] == head:
            return e[: -len(".xip.elf")], base, os.path.basename(os.path.dirname(os.path.dirname(e)))
    return None


def kernel_addrs(samples, lay, base):
    load, size = lay["load"], lay["text_size"]
    kern = sorted((t, pc - load + base) for t, pc in samples if load <= pc < load + size)
    cut = len(kern) // 20
    return kern[cut : len(kern) - cut]


def a2l(elf, addrs):
    out = subprocess.run(
        [A2L, "-a", "-f", "-i", "-C", "-e", elf], input="\n".join(hex(a) for a in addrs), capture_output=True, text=True
    ).stdout.splitlines()
    res, cur, i = {}, None, 0
    while i < len(out):
        line = out[i]
        if re.fullmatch(r"0x[0-9a-f]+", line):
            cur = int(line, 16)
            res[cur] = []
            i += 1
            continue
        fn = line
        loc = re.sub(r" \(discriminator \d+\)", "", out[i + 1] if i + 1 < len(out) else "??:0")
        m = re.match(r"(.*):(\d+)", loc)
        res[cur].append((fn, os.path.basename(m.group(1)) if m else loc, int(m.group(2)) if m else 0))
        i += 2
    return res
