# Round 3 reduce, issue verdicts: the outputs of the prof_users.py cases (default configs, one call each, fixed seed),
# saved to compare main (farm fm_main) with the branch (farm fm_v4) bit for bit.
# usage: python bitid_users.py <out.pt>      compare: python bitid_users.py --cmp <main.pt> <branch.pt>
import sys

import torch

if sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2], weights_only=True), torch.load(sys.argv[3], weights_only=True)
    for k in a:
        if k not in b:
            print(f"{k}: missing in branch")
            continue
        x, y = a[k].float(), b[k].float()
        same = torch.equal(a[k], b[k])
        d = (x - y).abs()
        d = d[torch.isfinite(d)]
        print(f"{k}: {'bit identical' if same else 'DIFFERENT'} max abs diff {d.max().item() if d.numel() else 0:.3e}")
    sys.exit(0)

import os

# 05 (12:35 UTC): the device only under scripts/hwlock.sh (it exports HWLOCK_HELD); the mock cluster (an existing descriptor) needs no lock
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")

import ttnn

sys.path.insert(0, "/localdev/mvlahovic/llk_analysis_builds/round3/reduce/recheck/tools")
from rc_users import CASES

device = ttnn.open_device(device_id=0, l1_small_size=32768)
torch.manual_seed(58707)
outs = {}
import os
import zlib

prefixes = tuple(os.environ.get("BITID_PREFIX", "").split(","))
for name, setup in CASES:
    if not name.startswith(prefixes):
        continue
    torch.manual_seed(zlib.crc32(name.encode()))
    try:
        run, keep = setup(device, "dflt")
        y = run()
        if y is None:
            y = keep[0]
        outs[name] = ttnn.to_torch(ttnn.from_device(y))
        if name.startswith("gn"):
            outs[name + "__in"], outs[name + "__w"], outs[name + "__b"] = keep[2], keep[3], keep[4]
            outs[name + "__g"] = torch.tensor(keep[5])
        print(f"CASE {name} ok", flush=True)
    except Exception as e:
        print(f"CASE {name} FAIL {str(e).splitlines()[0][:300]}", flush=True)
torch.save(outs, sys.argv[1])
ttnn.close_device(device)
