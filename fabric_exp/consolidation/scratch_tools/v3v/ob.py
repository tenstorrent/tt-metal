#!/usr/bin/env python3
# Wrapper around consolidation/offline_build.py: prep_headers picks a proc entry that has a compile command (some -Os
# entries in cmds.json are incomplete), otherwise identical.
import sys, importlib.util
spec = importlib.util.spec_from_file_location("ob", "/home/aagarwal/tt-metal/fabric_exp/consolidation/offline_build.py")
ob = importlib.util.module_from_spec(spec); spec.loader.exec_module(ob)
import json
_orig_build = ob.build
def build(args):
    db = json.load(open(ob.DB))[args.opt]
    hashes = list(db) if args.hashes == "all" else args.hashes.split(",")
    hashes = [h for h in hashes if any("compile" in c and "link" in c for c in db[h].values())]
    args.hashes = ",".join(hashes)
    orig_prep = ob.prep_headers
    def prep(h, cmd, ov, out):
        c = next(c["compile"] for c in db[h].values() if "compile" in c and "link" in c)
        return orig_prep(h, c, ov, out)
    ob.prep_headers = prep
    return _orig_build(args)
ob.build = build
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--variant", required=True); b.add_argument("--out", required=True)
    b.add_argument("--opt", default="O3"); b.add_argument("--set-opt", default=None)
    b.add_argument("--hashes", default="all"); b.add_argument("--flags", default=""); b.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    sys.exit(ob.build(a))
