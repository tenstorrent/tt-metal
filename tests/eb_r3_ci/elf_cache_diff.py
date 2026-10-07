#!/usr/bin/env python3
"""Round 3 eltwise binary: every compute and data movement ELF of two JIT kernel caches, kernels matched by name and
defines_generated.h (EB_R3_ and BINARY_NG_BLOCK lines dropped); disassembly with addresses and branch targets masked.
usage: elf_cache_diff.py <cache A> <cache B>"""
import glob, os, re, subprocess, sys, collections

od = next((p for p in ("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump",) if os.path.exists(p)), None)
if od is None:
    sys.exit("ELFID no objdump")


def variants(root):
    out = {}
    for d in glob.glob(f"{root}/*/*/kernels/*/*/") + glob.glob(f"{root}/*/kernels/*/*/") + glob.glob(f"{root}/kernels/*/*/"):
        try:
            defs = "".join(open(d + f).read() for f in ("defines_generated.h", "kernel_args_generated.h", "named_args_generated.h", "chlkc_descriptors.h") if os.path.exists(d + f))
        except OSError:
            continue
        name = d.rstrip("/").split("/")[-2]
        out[(name, re.sub(r"(?m)^.*(EB_R3_|BINARY_NG_BLOCK).*\n", "", defs))] = d
    return out


def dis(elf):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout.splitlines()
    return [re.sub(r"\b[0-9a-f]{4,8}\b <[^>]*>", "T", m.group(1).split("#")[0]).strip() for m in (re.match(r"^\s*[0-9a-f]+:\s*(.*)$", l) for l in txt) if m]


va, vb = variants(sys.argv[1]), variants(sys.argv[2])
common = [k for k in va if k in vb]
print(f"ELFID variants A {len(va)} B {len(vb)} matched {len(common)}")
cnt = collections.Counter(); diffs = collections.Counter()
for k in common:
    for elf in sorted(glob.glob(va[k] + "*/*.elf")):
        rel = elf[len(va[k]):]
        eb = vb[k] + rel
        if not os.path.exists(eb):
            continue
        same = dis(elf) == dis(eb)
        cnt["identical" if same else "differ"] += 1
        if not same:
            diffs[(k[0], rel)] += 1
print(f"ELFID elfs identical {cnt['identical']} differ {cnt['differ']}")
for (n, rel), c in sorted(diffs.items()):
    print(f"ELFID differ {n} {rel} x{c}")
