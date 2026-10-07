#!/usr/bin/env python3
"""Round 3 eltwise binary: two JIT kernel caches built by the same host, kernels matched by their cache path (kernel name and
hash of defines, arguments and config); every ELF's loaded sections hashed and compared. usage: elf_key_diff.py <A> <B>"""
import collections, glob, hashlib, os, subprocess, sys
OBJ = next((p for p in ("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objcopy",) if os.path.exists(p)), None)


def h(elf):
    out = subprocess.run([OBJ, "-O", "binary", "--only-section=.text", "--only-section=.data", "--only-section=.rodata", elf, "/dev/stdout"], capture_output=True).stdout
    return hashlib.sha1(out).hexdigest()


def keys(root):
    d = {}
    for e in glob.glob(f"{root}/**/kernels/*/*/*/*.elf", recursive=True):
        rel = e.split("/kernels/", 1)[1]
        d[rel] = e
    return d


A, B = keys(sys.argv[1]), keys(sys.argv[2])
common = sorted(set(A) & set(B))
diff = collections.Counter(); same = 0
for k in common:
    if h(A[k]) == h(B[k]):
        same += 1
    else:
        diff[k.split("/")[0] + " " + k.split("/")[-1]] += 1
print(f"ELFKEY A {len(A)} B {len(B)} matched {len(common)} identical {same} differ {sum(diff.values())}")
for k, c in sorted(diff.items()):
    print(f"ELFKEY differ {k} x{c}")
