#!/usr/bin/env python3
"""offline_build.py with the overlay taken from a frozen snapshot of tt_metal/fabric (HEAD + s79mc's v2) instead of
the live tree, which broker jobs swap variants into. Same CLI as offline_build.py."""
import os, shutil, sys

sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import offline_build as ob

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap")


def make_overlay(variant, out):
    ov = os.path.join(out, "overlay")
    if os.path.exists(ov):
        shutil.rmtree(ov)
    shutil.copytree(SNAP + "/tt_metal/fabric", ov + "/tt_metal/fabric", symlinks=True)
    for src, dst in ob.resolve(variant):
        os.makedirs(os.path.dirname(os.path.join(ov, dst)), exist_ok=True)
        shutil.copy(src, os.path.join(ov, dst))
    return ov


ob.make_overlay = make_overlay

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--variant", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--opt", default="O3")
    b.add_argument("--set-opt", default=None)
    b.add_argument("--hashes", default="all")
    b.add_argument("--flags", default="")
    b.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    sys.exit(ob.build(a))
