"""Pick with model7 (plain argmin) from an enumerated-candidates CSV, using frozen constants.
usage: pick7.py ENUM_CSV ARCH OUT_CSV [CONST_JSON (default fitted_v7_<arch>.json)]"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
from data import KEY, CAND

src, arch, out = sys.argv[1:4]
p = json.load(open(sys.argv[4] if len(sys.argv) > 4 else f"fitted_v7_{arch}.json"))
# frozen constants define the version: terms whose constants the file lacks (added later) are switched off
for name, (_, _, term) in M.CONSTANTS.items():
    if term and name not in p:
        M.OFF.add(term)
if M.OFF:
    print("terms off (no constants in the file):", sorted(M.OFF))
e = pd.read_csv(src, low_memory=False).drop_duplicates(KEY, keep="last")
e = e[e.origin.isin(CAND) & (e.status == "ok")].reset_index(drop=True)
e["arch_"] = arch
MC = "MatmulMultiCoreProgramConfig(allowed_worker_cores=std::nullopt)"


def multicore_candidates(e):
    """one MultiCore (non-reusing factory) candidate per problem it can run: interleaved inputs and output, no fused
    bias or activation. The enumerator never offers it, but legacy uses it, and on small batched problems it wins."""
    rows = []
    for case, g in e.groupby("case"):
        r = g.iloc[0]
        act = str(r.get("activation", "") or "")
        if r.a_mem not in ("dram", "l1") or r.b_mem not in ("dram", "l1") or r.out_mem not in ("dram", "l1"):
            continue
        if (r.get("bias", 0) == 1) or act not in ("", "nan", "None"):
            continue
        grid = (g.grid_x * g.grid_y).max()
        tiles = r.batch * -(-r.M // 32) * -(-r.N // 32)
        x = r.copy()
        x["family"], x["config"], x["origin"] = "multicore", MC, "enumerated"
        for k in (
            "per_core_M",
            "per_core_N",
            "out_block_h",
            "out_block_w",
            "in0_block_w",
            "out_subblock_h",
            "out_subblock_w",
        ):
            x[k] = np.nan
        x["fuse_batch"] = 0
        x["cores"] = min(grid, tiles)
        rows.append(x)
    return pd.DataFrame(rows)


if os.environ.get("NO_MC") != "1":  # v13 on
    mc = multicore_candidates(e)
    e = pd.concat([e, mc], ignore_index=True)
MAX_LOWP_SPILLS = 32


def precision_valid(e):
    """numerics, not speed: bfp8/bfp4 partials without fp32 or L1 accumulation lose precision with every spill round trip.
    Across the fresh and suite runs, pcc failures go from 1 in 128 configs at <= 32 round trips to 7 in 8 at 65-96 and
    24 in 24 above 96, so candidates over 32 are dropped when the problem has one at or under 32."""
    lowp = (
        e.out_dtype.fillna(e.a_dtype).isin(["bfp8", "bfp4"])
        & (e.fp32_acc.fillna(0) == 0)
        & (e.packer_l1_acc.fillna(0) == 0)
    )
    spills = np.ceil(np.ceil(e.K / 32) / e.in0_block_w.fillna(1)) - 1
    bad = lowp & (spills > MAX_LOWP_SPILLS) & (e.family != "multicore")
    has_ok = (~bad).groupby(e.case).transform("any")
    return e[~(bad & has_ok)]


if os.environ.get("NO_PRECISION_RULE") != "1":  # v14 on
    e = precision_valid(e).reset_index(drop=True)
e = M.annotate(e)
e["pred"] = M.predict(M.geometry(e), p)
r = e.loc[e.groupby("case").pred.idxmin(), ["case", "config", "family", "origin", "pred"]]
r["pred_us"] = (r.pred / 1e3).round(2)
r.drop(columns="pred").to_csv(out, index=False)
print(len(r), "picks; same as rules:", int((r.origin == "heuristic").sum()), r.family.value_counts().to_dict())
