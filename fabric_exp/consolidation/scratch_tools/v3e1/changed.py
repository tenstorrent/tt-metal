import sys, glob, os, subprocess, re, json
sys.path.insert(0, ".")
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
def text(e):
    out = subprocess.run([OD, "-d", "--section=.text", e], capture_output=True, text=True).stdout
    ins = [re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1]) for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
    return [re.sub(r" <[^>]*>", "", x) for x in ins]
a, b = sys.argv[1], sys.argv[2]
import concurrent.futures
keys = [os.path.relpath(e, b) for e in glob.glob(f"{b}/*/*/*.elf")]
def cmp(k):
    return k, text(os.path.join(a, k)) == text(os.path.join(b, k))
with concurrent.futures.ThreadPoolExecutor(48) as ex:
    res = dict(ex.map(cmp, keys))
same = sorted(k.split("/")[0] + "/" + k.split("/")[1] for k, v in res.items() if v)
diff = sorted(k.split("/")[0] + "/" + k.split("/")[1] for k, v in res.items() if not v)
json.dump({"same": same, "diff": diff}, open(sys.argv[3], "w"))
print(len(same), len(diff))
