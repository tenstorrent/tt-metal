import sys, glob, os, re, subprocess, collections
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
def text(e):
    out = subprocess.run([OD, "-d", "--section=.text", e], capture_output=True, text=True).stdout
    ins = [re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1]) for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
    return [re.sub(r" <[^>]*>", "", x) for x in ins]
def elfs(root):
    r = {}
    for e in glob.glob(root + "/*/kernels/fabric_erisc_router_v2/*/*/*.elf"):
        if e.endswith(".xip.elf"): continue
        h = e.split("/")[-3]; p = e.split("/")[-2]
        r[(h, p)] = e
    return r
a = elfs(sys.argv[1]); b = elfs(sys.argv[2])
res = collections.Counter(); diffs = []
for k in sorted(set(a) & set(b)):
    same = text(a[k]) == text(b[k])
    res[(k[1], same)] += 1
    if not same: diffs.append(k)
print(dict(res), "only_a", len(set(a)-set(b)), "only_b", len(set(b)-set(a)))
for k in diffs[:10]: print("diff", k)
