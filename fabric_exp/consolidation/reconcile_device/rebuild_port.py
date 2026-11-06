#!/usr/bin/env python3
"""rebuild_port.py CASE_DIR OUT [N]: rebuild the router E1 (subordinate_active_erisc) images a port coverage case logged,
with their debug info, into OUT (same command handling as the port's cov.py build_one_; the case's cache is read in
place, nothing is written outside OUT). Read-only on the port tree (the case root's symlinks)."""
import concurrent.futures, json, os, re, subprocess, sys
case_dir, out = sys.argv[1], sys.argv[2]
cmds = json.load(open(os.path.join(case_dir, "cmds.json")))
keys = [k for k in cmds if k.startswith("kernels/fabric_erisc_router/") and k.endswith("subordinate_active_erisc")]

def cache_dir_of(link, key):
    m = re.search(r"\s-o\s+(\S+)/" + re.escape(key) + r"/", link)
    return m.group(1) + "/" + key

def one(key):
    c = cmds[key]
    wd = os.path.join(out, key.split("/")[2])
    os.makedirs(wd, exist_ok=True)
    src = cache_dir_of(c["link"][0], key)
    objs = {}
    for cc in c["compile"]:
        a, o, i = cc.split(" "), [], 0
        while i < len(a):
            if a[i] in ("-o", "-MF"):
                p = os.path.join(wd, os.path.basename(a[i + 1]))
                if a[i] == "-o":
                    objs[a[i + 1]] = p
                o += [a[i], p]; i += 2; continue
            o.append(a[i]); i += 1
        r = subprocess.run(["nice", "-n", "10"] + o, cwd=src, capture_output=True, text=True)
        if r.returncode:
            return key, "compile failed: " + r.stderr[-500:]
    link = c["link"][0]
    for old, new in objs.items():
        link = link.replace(old, new)
    if len(objs) == 1:
        link = re.sub(r"\S+/" + re.escape(key) + r"/[^/\s]+\.o(?=\s)", next(iter(objs.values())), link)
    lm = re.search(r"\s-o\s+(\S+\.elf)", link)
    elf = os.path.join(wd, "subordinate_active_erisc.elf")
    link = link[: lm.start(1)] + elf + link[lm.end(1):]
    r = subprocess.run(link, shell=True, capture_output=True, text=True)
    for p in objs.values():
        if os.path.exists(p): os.remove(p)
    return key, ("link failed: " + r.stderr[-500:]) if r.returncode else "ok"

with concurrent.futures.ThreadPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 12) as ex:
    for k, s in ex.map(one, keys):
        print(k.split("/")[2], s)
