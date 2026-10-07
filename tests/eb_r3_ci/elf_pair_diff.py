#!/usr/bin/env python3
"""Round 3 eltwise binary: compute ELFs of two JIT kernel caches, variants matched by defines_generated.h without the EB_R3
lines; per TRISC, the disassembly with addresses and branch targets masked, diffed. usage: elf_pair_diff.py <cache A> <cache B> <kernel name>..."""
import difflib, glob, os, re, shutil, subprocess, sys

od = shutil.which("riscv-tt-elf-objdump") or next(iter(glob.glob("/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump") + glob.glob("/**/riscv-tt-elf-objdump", recursive=True)), None)
A, B, names = sys.argv[1], sys.argv[2], sys.argv[3:]


def variants(root, name):
    out = {}
    for d in glob.glob(f"{root}/**/kernels/{name}/*/", recursive=True):
        try:
            defs = open(d + "defines_generated.h").read()
        except OSError:
            continue
        out[re.sub(r"(?m)^.*EB_R3_.*\n", "", defs)] = d
    return out


def dis(elf):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout.splitlines()
    lines = []
    for l in txt:
        m = re.match(r"^\s*[0-9a-f]+:\s*(.*)$", l)
        if not m:
            continue
        ins = re.sub(r"\b[0-9a-f]{4,8}\b <[^>]*>", "TARGET", m.group(1).split("#")[0]).strip()
        lines.append(ins)
    return lines


for name in names:
    va, vb = variants(A, name), variants(B, name)
    common = [k for k in va if k in vb]
    print(f"ELFDIFF {name}: variants A {len(va)} B {len(vb)} matched {len(common)}")
    for k in common:
        tag = ",".join(sorted(set(re.findall(r"#define (\w+)", k)) & {"PACK_RELU"} | set(re.findall(r"(GELU|SILU|SOFTPLUS|RELU)", k))))
        for t in ("trisc0", "trisc1", "trisc2"):
            ea, eb = f"{va[k]}{t}/{t}.elf", f"{vb[k]}{t}/{t}.elf"
            if not (os.path.exists(ea) and os.path.exists(eb)):
                continue
            la, lb = dis(ea), dis(eb)
            if la == lb:
                print(f"ELFDIFF {name} [{tag}] {t}: identical ({len(la)} instructions)")
                continue
            d = [x for x in difflib.unified_diff(la, lb, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
            print(f"ELFDIFF {name} [{tag}] {t}: {len(la)} vs {len(lb)} instructions, {len(d)} diff lines")
            for x in d[:24]:
                print(f"ELFDIFF   {x}")
