#!/usr/bin/env python3
"""mk_rf_farm.py BASE DST WORKTREE BASE_SHA: a Galaxy tree for the report data refresh as a symlink farm (few inodes on
/data; as pr_perf/mkfarm_pr.py and the tune_e7fe farm): DST = BASE as symlinks, with
  - every file that differs between BASE_SHA and WORKTREE's HEAD (git diff --no-renames), and every file of WORKTREE's
    local diff, as a real copy of WORKTREE's working file (so local patches are in); files deleted there are left out;
  - the host binaries from WORKTREE/build_Release: tt_metal/libtt_metal.so, tt_stl/libtt_stl.so, the UMD and Tracy
    libraries, test_tt_fabric (also over BASE's build_Release/lib copies of the same libraries).
Writes DST/.tree_sha (the commit, the local diff's files, the binaries' md5)."""
import hashlib, os, shutil, subprocess, sys

base, dst, wt, base_sha = sys.argv[1:5]
git = lambda *a: subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True, check=True).stdout
head = git("rev-parse", "HEAD").strip()
changed, deleted = set(), set()
for line in git("diff", "--name-status", "--no-renames", base_sha, "HEAD").splitlines():
    s, f = line.split("\t", 1)
    (deleted if s == "D" else changed).add(f)
local = set(git("diff", "--name-only", "HEAD").split())
changed |= local
B = "build_Release"
bins = {  # DST path: WORKTREE path
    f"{B}/tt_metal/libtt_metal.so": f"{B}/tt_metal/libtt_metal.so",
    f"{B}/tt_stl/libtt_stl.so": f"{B}/tt_stl/libtt_stl.so",
    f"{B}/lib/libtt_metal.so": f"{B}/tt_metal/libtt_metal.so",
    f"{B}/lib/libtt_stl.so": f"{B}/tt_stl/libtt_stl.so",
    f"{B}/lib/libtracy.so.0.14.1": f"{B}/lib/libtracy.so.0.14.1",
    f"{B}/lib/libtt-umd.so.0.9.15": f"{B}/tt_metal/third_party/umd/lib/libtt-umd.so.0.9.15",
    f"{B}/tt_metal/third_party/umd/lib/libtt-umd.so.0.9.15": f"{B}/tt_metal/third_party/umd/lib/libtt-umd.so.0.9.15",
    f"{B}/test/tt_metal/tt_fabric/test_infra/test_tt_fabric": f"{B}/test/tt_metal/tt_fabric/test_infra/test_tt_fabric",
}
bins = {d: s for d, s in bins.items() if os.path.exists(os.path.join(base, d)) or d.endswith(("test_tt_fabric", "libtt_metal.so"))}
real = changed | set(bins)
assert not os.path.exists(dst), dst
real_dirs = {""}
for f in real | deleted:
    parts = f.split("/")[:-1]
    for k in range(1, len(parts) + 1):
        real_dirs.add("/".join(parts[:k]))


def farm(rel):
    src_dir, out_dir = os.path.join(base, rel), os.path.join(dst, rel)
    os.makedirs(out_dir, exist_ok=True)
    for e in os.listdir(src_dir):
        r = f"{rel}/{e}" if rel else e
        if r in real_dirs and os.path.isdir(os.path.join(src_dir, e)) and not os.path.islink(os.path.join(src_dir, e)):
            farm(r)
        elif r in real or r in deleted or r in ("generated", "build"):
            continue
        else:
            os.symlink(os.path.join(src_dir, e), os.path.join(out_dir, e))


farm("")
os.makedirs(os.path.join(dst, "generated"), exist_ok=True)
os.symlink("build_Release", os.path.join(dst, "build"))  # never BASE's build link (it would run BASE's binary)
for f in sorted(changed):
    src = os.path.join(wt, f)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
        shutil.copy2(src, os.path.join(dst, f))
md5 = {}
for d, s in bins.items():
    os.makedirs(os.path.dirname(os.path.join(dst, d)), exist_ok=True)
    shutil.copy2(os.path.join(wt, s), os.path.join(dst, d))
    md5[d] = hashlib.md5(open(os.path.join(dst, d), "rb").read()).hexdigest()
open(os.path.join(dst, ".tree_sha"), "w").write(
    f"{head} (farm of {base} at {base_sha}; {len(changed)} files from {wt} ({len(local)} of them its local diff: "
    f"{' '.join(sorted(local))}); {len(deleted)} deleted; host binaries from {wt}/{B}: "
    + " ".join(f"{d}={m}" for d, m in sorted(md5.items())) + ")\n")
print(f"{dst}: {len(changed)} files ({len(local)} local), {len(deleted)} deleted, {len(bins)} binaries; HEAD {head[:11]}")
