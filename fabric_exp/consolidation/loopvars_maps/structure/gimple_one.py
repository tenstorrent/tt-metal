"""gimple_one.py BUILD KEY [PROC] [FLAGS...]: rebuild one coverage-build image (BUILD e.g. v4rxf1_O3, KEY case__hash) in
the scratchpad with GCC dump flags added to the LTO link (where the code is generated); prints the work dir."""
import os, sys, json, glob, shutil, re
os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m")
os.environ.setdefault("OFFLINE_BASE", "/home/aagarwal/.tt-metal-p1-base-m")
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import coverage_rxo as C
import offline_build as ob
build, key = sys.argv[1], sys.argv[2]
proc = sys.argv[3] if len(sys.argv) > 3 else "subordinate_active_erisc"
flags = sys.argv[4:] or ["-fdump-tree-optimized"]
ob.EDM = C.ROOT + "/tt_metal/fabric/impl/kernels/edm_fabric"
B = f"{C.COV}/builds/{build}"
ov = f"{B}/overlay"
case, h = key.split("__")
c = json.load(open(f"{C.COV}/cases/{case}/cmds.json"))[h][proc]
cc = {k: v.replace(f"{C.KDIR}{h}/", f"{C.KDIR}{key}/").replace(C.TREE_EDM + " ", ob.EDM + " ") for k, v in c.items()}
out = f"/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/f1/gimple/{build}"
hd = f"{out}/{key}"
if os.path.exists(hd):
    shutil.rmtree(hd)
os.makedirs(hd)
for f in glob.glob(f"{B}/{key}/*.h*"):
    shutil.copy(f, hd)
cc["link"] = re.sub(r"(-flto-partition=one)", r"\1 " + " ".join(flags), cc["link"], count=1)
k, p, err = ob.build_one(key, proc, cc, ov, out, [], "O3")
print(err or "ok", f"{hd}/{proc}")
