#!/usr/bin/env python3
"""mkfarm.py BASE DST REPO SHA: a lean tree DST = BASE as symlinks, with the files that differ between BASE's commit
(BASE/.tree_sha or argv) and SHA in REPO as real copies at SHA. Real directories only along the changed files' paths
(every other entry a symlink to BASE's), so a few inodes. Usage: mkfarm.py BASE DST REPO BASE_SHA SHA."""
import os, subprocess, sys
base, dst, repo, base_sha, sha = sys.argv[1:6]
changed = subprocess.run(["git", "-C", repo, "diff", "--name-only", base_sha, sha], capture_output=True, text=True,
                         check=True).stdout.split()
assert not os.path.exists(dst), dst
os.makedirs(dst)
real_dirs = {""}
for f in changed:
    parts = f.split("/")[:-1]
    for k in range(1, len(parts) + 1):
        real_dirs.add("/".join(parts[:k]))
def farm(rel):
    src_dir = os.path.join(base, rel)
    out_dir = os.path.join(dst, rel)
    os.makedirs(out_dir, exist_ok=True)
    for e in os.listdir(src_dir):
        r = f"{rel}/{e}" if rel else e
        if r in real_dirs:
            farm(r)
        elif r in changed:
            continue
        else:
            os.symlink(os.path.join(src_dir, e), os.path.join(out_dir, e))
farm("")
for f in changed:
    data = subprocess.run(["git", "-C", repo, "show", f"{sha}:{f}"], capture_output=True, check=True).stdout
    with open(os.path.join(dst, f), "wb") as fh:
        fh.write(data)
open(os.path.join(dst, ".tree_sha"), "w").write(f"{sha} (farm of {base} at {base_sha}; changed: {len(changed)})\n")
print(f"{dst}: {len(changed)} files at {sha}, real dirs {len(real_dirs)}")
for f in changed:
    print("  ", f)
