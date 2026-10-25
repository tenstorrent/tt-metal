#!/usr/bin/env python3
"""Swap a router variant into the tree for a hardware run, and put the tree back.

  swap.py apply VARIANT_DIR BACKUP_DIR   back up every tree file the variant replaces (offline_build.resolve: mirrored
                                         paths, bare names, fabric_erisc_router_v3.cpp -> the v2 path), then copy the
                                         variant's files in; files the tree lacks are recorded and created
  swap.py restore BACKUP_DIR             copy the backups back, delete the created files, check every file matches
Only for broker job scripts; never while another job uses the tree.
"""
import filecmp, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_build import ROOT, resolve

cmd, args = sys.argv[1], sys.argv[2:]
if cmd == "apply":
    variant, bk = os.path.abspath(args[0]), os.path.abspath(args[1])
    if os.path.exists(os.path.join(bk, "manifest.json")):
        sys.exit(f"{bk} holds an unrestored swap")
    os.makedirs(bk, exist_ok=True)
    man = {"saved": [], "created": []}
    for src, dst in resolve(variant):
        t = os.path.join(ROOT, dst)
        if os.path.exists(t):
            os.makedirs(os.path.dirname(os.path.join(bk, dst)), exist_ok=True)
            shutil.copy2(t, os.path.join(bk, dst))
            man["saved"].append(dst)
        else:
            man["created"].append(dst)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copy(src, t)
    json.dump(man, open(os.path.join(bk, "manifest.json"), "w"), indent=1)
    print(f"== router: {os.path.basename(variant)} ({len(man['saved'])} replaced, {len(man['created'])} created)")
elif cmd == "restore":
    bk = os.path.abspath(args[0])
    mp = os.path.join(bk, "manifest.json")
    if not os.path.exists(mp):
        print("nothing to restore")
        sys.exit(0)
    man = json.load(open(mp))
    bad = 0
    for dst in man["saved"]:
        shutil.copy2(os.path.join(bk, dst), os.path.join(ROOT, dst))
        bad += not filecmp.cmp(os.path.join(bk, dst), os.path.join(ROOT, dst), shallow=False)
    for dst in man["created"]:
        if os.path.exists(os.path.join(ROOT, dst)):
            os.remove(os.path.join(ROOT, dst))
    if bad:
        sys.exit(f"!!! RESTORE FAILED for {bad} files")
    os.rename(mp, mp + ".done")
    print(f"tree restored ({len(man['saved'])} files, {len(man['created'])} removed)")
else:
    sys.exit(__doc__)
