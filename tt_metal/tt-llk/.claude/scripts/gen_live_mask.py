#!/usr/bin/env python3
"""Compute per-addr32 written-bit masks for the Blackhole LLK CFG-write surface.

For each CFG-write site, determine the target addr32 word(s) and which BITS it writes:
  - WRCFG_32b / WRCFG_128b / `cfg[...] =`  -> whole word(s) (0xFFFFFFFF)
  - SETC16 / addr_mod set                  -> low 16 bits (0xFFFF)
  - cfg_reg_rmw_tensix<FIELD_RMW> / RMWCIB -> that field's mask
The union per word = bits some op writes. Never-written fields (e.g. Downsample) stay 0.
Reachability caveat: PCK0_ADDR_BASE_REG_0's only writer (program_packer_dest_offset_registers)
is uncalled dead code -> excluded explicitly.
"""
import os
import re
import glob

ROOT = "/localdev/iklikovac/tt-metal/tt_metal/tt-llk/tt_llk_blackhole"
CFG = "/localdev/iklikovac/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h"

# cfg_defines: base-name -> addr32, base-name -> mask
ADDR32, MASK = {}, {}
for line in open(CFG):
    m = re.match(r"#define\s+(\w+?)_ADDR32\s+(\d+)\b", line)
    if m:
        ADDR32[m.group(1)] = int(m.group(2))
    m = re.match(r"#define\s+(\w+?)_MASK\s+(0x[0-9A-Fa-f]+|\d+)\b", line)
    if m:
        MASK[m.group(1)] = int(m.group(2), 0)

DEAD = {"PCK0_ADDR_BASE_REG_0_Base"}  # only writer is uncalled program_packer_dest_offset_registers

# collect write-site files
files = set()
for f in glob.glob(ROOT + "/**/*.h", recursive=True):
    txt = open(f, errors="ignore").read()
    if re.search(r"TTI?_WRCFG|TT_WRCFG|RMWCIB|SETC16|cfg_reg_rmw|cfg\w*\[", txt):
        files.add(f)

live = {}  # addr32 -> mask


def add(word, mask):
    if word is None:
        return
    live[word] = live.get(word, 0) | mask


def w(base, off=0):
    return ADDR32[base] + off if base in ADDR32 else None


for f in files:
    for line in open(f, errors="ignore"):
        line = re.sub(r"//.*", "", line)  # drop line comments (e.g. commented-out cfg[...] writes)
        # A) WRCFG: mode + target (whole word; 128b spans +0..+3)
        for mm in re.finditer(r"_WRCFG\(\s*[^,]+,\s*([^,]+),\s*(\w+?)_ADDR32\s*(?:\+\s*(\d+))?\s*\)", line):
            mode, base, off = mm.group(1), mm.group(2), int(mm.group(3) or 0)
            if base in DEAD:
                continue
            n = 4 if "128b" in mode else 1
            for k in range(n):
                add(w(base, off + k), 0xFFFFFFFF)
        # B) direct cfg[...] = (whole word)
        for mm in re.finditer(r"\[\s*(\w+?)_ADDR32\s*(?:\+\s*(\d+))?\s*\]\s*=", line):
            base, off = mm.group(1), int(mm.group(2) or 0)
            if base not in DEAD:
                add(w(base, off), 0xFFFFFFFF)
        # C) cfg_reg_rmw_tensix<FIELD_RMW> (field mask)
        for mm in re.finditer(r"cfg_reg_rmw_tensix<\s*(\w+?)_RMW\b", line):
            base = mm.group(1)
            if base not in DEAD:
                add(w(base), MASK.get(base, 0xFFFFFFFF))
        # D) cfg_reg_rmw_tensix<FIELD_ADDR32, shamt, mask>
        for mm in re.finditer(r"cfg_reg_rmw_tensix<\s*(\w+?)_ADDR32\s*(?:\+\s*(\d+))?\s*,\s*[^,]+,\s*([^>]+)>", line):
            base, off, mexpr = mm.group(1), int(mm.group(2) or 0), mm.group(3).strip()
            if base in DEAD:
                continue
            hexm = re.match(r"(0x[0-9A-Fa-f]+|\d+)$", mexpr)
            if hexm:
                add(w(base, off), int(hexm.group(1), 0))
            elif mexpr.endswith("_MASK") and mexpr[:-5] in MASK:
                add(w(base, off), MASK[mexpr[:-5]])
            else:
                # cfg_reg_rmw_tensix is always a FIELD rmw (never whole-word). Unresolvable mask
                # expr (a local constant) -> fall back to this field's own mask, not whole word.
                add(w(base, off), MASK.get(base, 0))
        # E) SETC16 literal target (16-bit)
        for mm in re.finditer(r"_SETC16\(\s*(\w+?)_ADDR32\b", line):
            base = mm.group(1)
            if base not in DEAD:
                add(w(base), 0xFFFF)

# addr_mod_t::set(IDX) / addr_mod_pack_t::set(IDX) write 16-bit ADDR_MOD section regs via
# array-indexed SETC16 (not literal _ADDR32 -> missed by the parser above). Only the sections in
# the reg-addr arrays (ckernel_addrmod.h) are addressable: AB/DST/BIAS SEC0-7, PACK SEC0-3.
# (ADDR_MOD_AB2 etc. are NOT in any array -> never written by set() -> NOT live.)
addr_mod_regs = (
    [f"ADDR_MOD_AB_SEC{n}_SrcAIncr" for n in range(8)]
    + [f"ADDR_MOD_DST_SEC{n}_DestIncr" for n in range(8)]
    + [f"ADDR_MOD_BIAS_SEC{n}_BiasIncr" for n in range(8)]
    + [f"ADDR_MOD_PACK_SEC{n}_YsrcIncr" for n in range(4)]
)
for base in addr_mod_regs:
    if base in ADDR32:
        live[ADDR32[base]] = live.get(ADDR32[base], 0) | 0xFFFF

# Report, restricted to the known live word set (sanity vs the 70-word list)
for a in sorted(live):
    fields = sorted(n for n, aa in ADDR32.items() if aa == a)
    print(f"  {a:3d}: 0x{live[a]:08X}")
print(f"\ntotal live words: {len(live)}")
print(f"word 71 mask = 0x{live.get(71,0):08X}  (Downsample bits 0-18 = 0x7FFFF should be CLEAR)")
print(f"   71 & Downsample(0x7FFFF) = 0x{live.get(71,0) & 0x7FFFF:X}  (expect 0)")
print(f"word 16 mask = 0x{live.get(16,0):08X}")
print(f"word 6 (DEST_REGW_BASE) present? {6 in live}")
# emit a python literal for cfg_pollution
items = ", ".join(f"{a}: 0x{live[a]:08X}" for a in sorted(live))
open("/tmp/live_mask.txt", "w").write("{" + items + "}")
