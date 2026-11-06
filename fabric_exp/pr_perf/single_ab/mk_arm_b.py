#!/usr/bin/env python3
"""mk_arm_b.py BASE DST: arm B of the single-ERISC A/B (single_ab_runs.sh here; local, never committed).

DST = BASE as symlinks (nxab/mk_nxab_trees.py's farm), with tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp
a real copy of router_single.B.hpp (arm_b_inline.patch: SINGLE_ERISC_PASS FORCE_INLINE and inline_passes = true on
every arch). BASE's own router_single.hpp must be ba486eb70ba's (router_single.A.hpp), so B differs from A in that one
file only. DST must not exist, except a farm this script made (.tree_sha names it), which is rebuilt."""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nxab"))
from mk_nxab_trees import farm, stamp  # noqa: E402

REL = "tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    base, dst = (os.path.abspath(p) for p in sys.argv[1:3])
    a, b = os.path.join(HERE, "router_single.A.hpp"), os.path.join(HERE, "router_single.B.hpp")
    got = sha(os.path.join(base, REL))
    if got != sha(a):
        sys.exit(f"!!! {base}/{REL} is not ba486eb70ba's (sha256 {got[:12]}): not making arm B")
    if os.path.lexists(dst):
        t = os.path.join(dst, ".tree_sha")
        if not (os.path.isfile(t) and not os.path.islink(t) and open(t).read().startswith("single-ERISC A/B arm B")):
            sys.exit(f"!!! {dst} exists and is not an arm B farm: not touched")
        shutil.rmtree(dst)
    farm(base, dst, {REL: ("file", b)})
    # B's own: its run outputs (generated/) and the harness marker, never written through a link into A
    for e in ("generated", ".lb2_pr_owned"):
        if os.path.islink(os.path.join(dst, e)):
            os.unlink(os.path.join(dst, e))
    stamp(dst,f"single-ERISC A/B arm B: farm of {base}; {REL} = router_single.B.hpp (sha256 {sha(b)[:12]})\n")
    assert sha(os.path.join(dst, REL)) == sha(b) and not os.path.islink(os.path.join(dst, REL))
    print(f"{dst}: farm of {base}, {REL} = arm B (sha256 {sha(b)[:12]})")


if __name__ == "__main__":
    main()
