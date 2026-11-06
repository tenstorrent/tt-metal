#!/usr/bin/env python3
"""mkfarm_pr.py BASE DST REPO BASE_SHA SHA [KEEP...]: galaxy/pairs/mkfarm.py for the PR perf refresh: DST = BASE as
symlinks, with the files that differ between BASE_SHA and SHA as real copies at SHA; files SHA deletes are left out;
KEEP paths stay BASE's (a local patch that restores BASE's version, e.g. the telemetry switch in test_tt_fabric.cpp:
e7fe1899049 + local_patches/test_tt_fabric_no_bw_telemetry.patch == c3c1813c42a's file). Host binaries come from BASE,
so use it only when SHA changes no compiled host source except KEEP files."""
import os, subprocess, sys
base, dst, repo, base_sha, sha = sys.argv[1:6]
keep = set(sys.argv[6:])
st = subprocess.run(["git", "-C", repo, "diff", "--name-status", "--no-renames", base_sha, sha], capture_output=True,
                    text=True, check=True).stdout.splitlines()
changed, deleted = [], []
for line in st:
    s, f = line.split("\t", 1)
    if f in keep:
        continue
    (deleted if s == "D" else changed).append(f)
assert not os.path.exists(dst), dst
os.makedirs(dst)
real_dirs = {""}
for f in changed + deleted:
    parts = f.split("/")[:-1]
    for k in range(1, len(parts) + 1):
        real_dirs.add("/".join(parts[:k]))
gone = set(changed) | set(deleted)
def farm(rel):
    src_dir = os.path.join(base, rel)
    out_dir = os.path.join(dst, rel)
    os.makedirs(out_dir, exist_ok=True)
    for e in os.listdir(src_dir):
        r = f"{rel}/{e}" if rel else e
        if r in real_dirs:
            farm(r)
        elif r in gone:
            continue
        else:
            os.symlink(os.path.join(src_dir, e), os.path.join(out_dir, e))
farm("")
for f in changed:
    data = subprocess.run(["git", "-C", repo, "show", f"{sha}:{f}"], capture_output=True, check=True).stdout
    with open(os.path.join(dst, f), "wb") as fh:
        fh.write(data)
open(os.path.join(dst, ".tree_sha"), "w").write(
    f"{sha} (farm of {base} at {base_sha}; changed: {len(changed)}, deleted: {len(deleted)}, kept from base: {sorted(keep)})\n")
print(f"{dst}: {len(changed)} files at {sha}, {len(deleted)} deleted, kept {sorted(keep)}")
for f in changed:
    print("  ", f)
