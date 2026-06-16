#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Statically build a *realizable* CFG-value palette by probing the LLK source.

Instead of running the suite and snapshotting (LLK_POLLUTE_CAPTURE), enumerate, per config
FIELD, the values the code can write, and compose realistic word values. Domains:
  - format fields            -> the DataFormat HW enum codes (tensix_types.h)
  - narrow fields (<=3 bits)  -> full enumeration 0..2^w-1 (flags, CR/Clear, fidelity)
  - addr-mod *Incr (6 bits)   -> the literal .incr values in addr_mod_t{} initializers
  - other wide fields         -> {0} + literal constants scanned at their write sites
                                 (address/dim regs stay ~{0} -> harmless; they're filtered live anyway)
Per live addr32, palette[addr32] = {0} U { single-field-set word values within the live bits }.
Output JSON {addr32: [values]} feeds LLK_POLLUTE_CFG=realizable (set LLK_POLLUTE_PALETTE to it).

This covers the discrete config space (formats/flags/addr-mod) deterministically with no device.
Composite struct-packed writes decompose into these same fields, so field-level coverage holds.
"""
import glob
import json
import re

BH = "/localdev/iklikovac/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole"
CFG = f"{BH}/cfg_defines.h"
TYPES = f"{BH}/tensix_types.h"
ROOT = "/localdev/iklikovac/tt-metal/tt_metal/tt-llk/tt_llk_blackhole"

# Bit-granular live surface (keep in sync with cfg_pollution._LIVE_MASK / gen_live_mask.py).
LIVE_MASK = {
    0: 0x0000FFFF, 1: 0xFFFFFFFF, 2: 0xFFFFFFFF, 5: 0x0000FFFF, 7: 0x0000FFFF, 12: 0xFFFFFFFF,
    13: 0xFFFFFFFF, 14: 0xFFFFFFFF, 15: 0xFFFFFFFF, 16: 0x0000FFFF, 17: 0xFFFFFFFF, 18: 0xFFFFFFFF,
    19: 0x0000FFFF, 20: 0xFFFFFFFF, 21: 0xFFFFFFFF, 24: 0xFFFFFFFF, 25: 0xFFFFFFFF, 28: 0x0000FFFF,
    29: 0x0000FFFF, 30: 0x0000FFFF, 31: 0x0000FFFF, 32: 0x0000FFFF, 33: 0x0000FFFF, 34: 0x0000FFFF,
    35: 0x0000FFFF, 37: 0x0000FFFF, 38: 0x0000FFFF, 39: 0x0000FFFF, 40: 0x0000FFFF, 41: 0x0000FFFF,
    47: 0x0000FFFF, 48: 0x0000FFFF, 49: 0x0000FFFF, 50: 0xFFFFFFFF, 51: 0x0000FFFF, 52: 0x0000FFFF,
    53: 0x0000FFFF, 54: 0x0000FFFF, 55: 0x0000FFFF, 56: 0xFFFFFFFF, 57: 0xFFFFFFFF, 59: 0xFFFFFFFF,
    64: 0xFFFF000F, 65: 0xFFFFFFFF, 68: 0xFFFFFFFF, 69: 0xFFFFFFFF, 70: 0xFFFFFFFF, 71: 0xFFC80000,
    72: 0xFFFFFFFF, 73: 0x00000030, 76: 0xFFFFFFFF, 77: 0xFFFFFFFF, 84: 0xFFFFFFFF, 86: 0xFFFFFFFF,
    92: 0xFFFFFFFF, 93: 0xFFFFFFFF, 112: 0xFFFF000F, 113: 0xFFFF0000, 119: 0x00400000, 120: 0x0000000F,
    124: 0xFFFFFFFF, 125: 0xFFFFFFFF, 140: 0xFFFFFFFF, 141: 0xFFFFFFFF, 180: 0xFFFFFFFF, 181: 0xFFFFFFFF,
    182: 0xFFFFFFFF, 183: 0xFFFFFFFF, 186: 0xFFFFFFFF, 209: 0xFFFFFFFF, 211: 0xFFFFFFFF, 220: 0x0000000B,
}


def data_format_codes():
    """Valid HW DataFormat enum codes (<=0x1F, excluding SW-only/Invalid)."""
    txt = open(TYPES).read()
    m = re.search(r"enum class DataFormat\s*:[^{]+\{(.*?)\}", txt, re.S)
    codes = set()
    for name, val in re.findall(r"(\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)", m.group(1)):
        v = int(val, 0)
        if v <= 0x1F and name != "Invalid":  # real HW encodings fit in 5 bits
            codes.add(v)
    return sorted(codes)


def parse_fields():
    """addr32 -> [(name, shamt, mask)] for every field defined in cfg_defines."""
    A, S, M = {}, {}, {}
    for line in open(CFG):
        for d, dct, conv in ((r"_ADDR32\s+(\d+)", A, int), (r"_SHAMT\s+(\d+)", S, int),
                             (r"_MASK\s+(0x[0-9A-Fa-f]+|\d+)", M, lambda x: int(x, 0))):
            m = re.match(r"#define\s+(\w+?)" + d + r"\b", line)
            if m:
                dct[m.group(1)] = conv(m.group(2))
    fields = {}
    for name, a in A.items():
        if name in S and name in M:
            fields.setdefault(a, []).append((name, S[name], M[name]))
    return fields


def incr_literals():
    """Literal .incr values used in addr_mod_t{} initializers across the LLK lib."""
    vals = {0}
    for f in glob.glob(ROOT + "/**/*.h", recursive=True):
        for m in re.finditer(r"\.incr\s*=\s*(-?\d+)", open(f, errors="ignore").read()):
            vals.add(int(m.group(1)) & 0x3F)
    return sorted(vals)


def literal_scan(name):
    """Literal values written to a field at cfg_reg_rmw_tensix<FIELD_RMW>(LIT) / SETC16 sites."""
    vals = set()
    pat = re.compile(r"cfg_reg_rmw_tensix<\s*" + re.escape(name) + r"_RMW\s*>\s*\(\s*(0x[0-9A-Fa-f]+|\d+)\s*\)")
    for f in glob.glob(ROOT + "/**/*.h", recursive=True):
        for m in pat.finditer(open(f, errors="ignore").read()):
            vals.add(int(m.group(1), 0))
    return vals


def rmw_written_fields():
    """Field names written FIELD-AWARE via cfg_reg_rmw_tensix<NAME_RMW>/<NAME_ADDR32,..>.

    A field's value is only realizably-non-default if some op writes THAT field (RMW) or it is
    an addr-mod sub-field (set() packs the section). Fields touched only by a whole-word WRCFG of
    an unrelated GPR (e.g. REG7 data-format bits, which get an offset GPR's zero) are NOT here, so
    their realizable value stays {0} — avoids inventing values no op actually produces.
    """
    s = set()
    for f in glob.glob(ROOT + "/**/*.h", recursive=True):
        txt = open(f, errors="ignore").read()
        s.update(re.findall(r"cfg_reg_rmw_tensix<\s*(\w+?)_RMW\b", txt))
        s.update(re.findall(r"cfg_reg_rmw_tensix<\s*(\w+?)_ADDR32\b", txt))
    return s


def is_format(name):
    return "data_format" in name.lower() or ("FORMAT_SPEC" in name and re.search(r"(SrcA|SrcB|Dstacc)", name))


def field_domain(name, mask, fmt, incr, written):
    w = bin(mask).count("1")
    fw = mask >> ((mask & -mask).bit_length() - 1)  # mask shifted to 0 = max field value
    is_addrmod = name.startswith("ADDR_MOD")
    if not is_addrmod and name not in written:
        return []  # not field-aware-written -> realizable value is just the default (0)
    if is_format(name):
        return [c for c in fmt if c <= fw]
    if is_addrmod and name.endswith("Incr") and w >= 4:
        return [v for v in incr if v <= fw]
    if w <= 3:
        return list(range(1, fw + 1))  # nonzero values of a narrow field (flags/CR/clear/fidelity)
    return [v for v in literal_scan(name) if v <= fw]  # wide non-format: only known literals


def main():
    fmt, incr, fields, written = data_format_codes(), incr_literals(), parse_fields(), rmw_written_fields()
    palette = {}
    for a, live in LIVE_MASK.items():
        vals = {0}
        for name, shamt, mask in fields.get(a, []):
            if not (mask & live):
                continue  # field not in the pollutable (live) bits
            for v in field_domain(name, mask, fmt, incr, written):
                vals.add(((v << shamt) & mask) & live)
        palette[a] = sorted(vals)
    out = "/tmp/palette_static.json"
    with open(out, "w") as f:
        json.dump({str(a): v for a, v in palette.items()}, f)
    tot = sum(len(v) for v in palette.values())
    print(f"format codes: {fmt}")
    print(f".incr literals: {incr}")
    print(f"palette: {len(palette)} words, {tot} total values -> {out}")
    for a in (16, 17, 49, 71, 72, 92):
        print(f"  addr32 {a:3d}: {[hex(x) for x in palette[a][:8]]}{' ...' if len(palette[a]) > 8 else ''}")


if __name__ == "__main__":
    main()
