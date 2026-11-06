#!/usr/bin/env python3
"""mkroot.py SRC ROOT REL...: a runtime root that is SRC's symlink farm except the files REL (paths under SRC), which are
copied (to be edited): every directory on a REL's path is real, everything else a symlink into SRC (as cisp_nohook)."""
import os, shutil, sys
src, root, rels = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3:]
real_dirs = {""}
for r in rels:
    parts = r.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        real_dirs.add("/".join(parts[:i]))
def build(rel):
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    for name in sorted(os.listdir(os.path.join(src, rel))):
        r = f"{rel}/{name}" if rel else name
        if r in real_dirs:
            build(r)
        elif r in rels:
            shutil.copy2(os.path.join(src, r), os.path.join(root, r))
        elif not os.path.lexists(os.path.join(root, r)):
            os.symlink(os.path.join(src, r), os.path.join(root, r))
build("")
print(f"{root}: farm of {src}, real files: {rels}")
