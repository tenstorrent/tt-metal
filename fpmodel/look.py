import sys, os, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
F = json.load(open("fitted.json"))
x = d[d.problem_id.str.endswith(sys.argv[1])].copy()
t, pr = predict(geometry(x), F[x.arch_.iloc[0]], parts=True)
x["pred_us"] = t / 1e3
x["us"] = x.device_ns / 1e3
for k in ("read", "comp", "reload", "write", "nK", "nob"):
    x[k] = np.round(pr[k], 0)
r = x.iloc[0]
print(
    r[
        [
            "M",
            "K",
            "N",
            "batch",
            "a_dtype",
            "b_dtype",
            "a_mem",
            "b_mem",
            "out_mem",
            "fidelity",
            "fp32_acc",
            "packer_l1_acc",
            "bias",
            "activation",
        ]
    ].to_dict()
)
cols = [
    "origin",
    "family",
    "grid_x",
    "grid_y",
    "per_core_M",
    "per_core_N",
    "out_block_h",
    "out_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "in0_block_w",
    "us",
    "pred_us",
    "read",
    "comp",
    "reload",
    "write",
    "nK",
    "nob",
]
print(x.sort_values("us")[cols].head(int(sys.argv[2]) if len(sys.argv) > 2 else 25).round(1).to_string())
