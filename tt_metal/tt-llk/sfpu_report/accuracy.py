# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Before/after accuracy of the selected ops.

Each side runs ``test_sfpu_report_accuracy.py``, which saves the device result of
every variant over two stimuli (the whole input format, and the special values).
The comparison is done here, on the host, with the ULP helpers the nightly ULP
sweep uses (``helpers/ulp.py``, ``helpers/ulp_sweep.py``). No budget is involved:
both sides are measured on the same device against the same golden, and the
report shows what changed.
"""

import hashlib
import sys
from pathlib import Path

import runner

sys.path.insert(0, str(runner.PYTHON_TESTS))

DRIVER = "test_sfpu_report_accuracy.py"

#: Special-value classes, in report order.
CLASSES = ("nan", "inf", "zero", "subnormal", "extreme")


def measure(side, arch, ops, out_dir, log, jobs=8):
    """Dump the side's raw results for ``ops`` (MathOperation names) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"SFPU_REPORT_DUMP": str(out_dir), "SFPU_REPORT_OPS": ",".join(ops)}
    runner.produce_consume(side, arch, [DRIVER], env=env, log=log, producer_jobs=jobs)
    return out_dir


def _same(result, golden):
    """Lane-wise: identical value, NaN-ness and sign (so -0 != +0)."""
    import torch

    r, g = result.to(torch.float64), golden.to(torch.float64)
    both_nan = torch.isnan(r) & torch.isnan(g)
    return both_nan | ((r == g) & (torch.signbit(r) == torch.signbit(g)))


def _sweep_stats(d):
    import torch
    from helpers.format_config import DataFormat
    from helpers.llk_params import MathOperation
    from helpers.ulp import ulp_distance, ulp_stats
    from helpers.ulp_sweep import measurable_mask, nonfinite_failures

    in_fmt, out_fmt = DataFormat[d["in"]], DataFormat[d["out"]]
    src, golden, result = d["src"], torch.as_tensor(d["golden"]), d["result"]
    mask = measurable_mask(src, golden, result, in_fmt)
    distance = ulp_distance(golden, result)
    stats = ulp_stats(distance, mask)
    nonfinite = nonfinite_failures(
        MathOperation[d["op"]], src, golden, result, in_fmt, out_fmt
    )
    flat = distance.reshape(-1).to(torch.int64)
    measured = mask.reshape(-1) & (flat >= 0)
    le1 = float(((flat <= 1) & measured).sum()) / max(int(measured.sum()), 1)
    return {
        "lanes": stats["lanes"],
        "max": stats["max"],
        "mean": stats["mean"],
        "p99": stats["p99"],
        "exact": stats["exact_frac"],
        "le1": le1,
        "nonfinite": int(nonfinite.sum()),
        "nonfinite_examples": [
            (float(src[i]), float(golden[i]), float(result[i]))
            for i in nonfinite.nonzero().flatten()[:3].tolist()
        ],
        "worst_input": (
            float(src.reshape(-1)[stats["worst_index"]])
            if stats["worst_index"] is not None
            else None
        ),
        "_distance": flat,
        "_measured": measured,
    }


def _specials(d):
    import torch

    golden, result = torch.as_tensor(d["golden"]).reshape(-1), d["result"].reshape(-1)
    src = d["src"].reshape(-1)
    same = _same(result, golden)
    out = {}
    for cls in CLASSES:
        idx = [i for i, c in enumerate(d["classes"]) if c == cls]
        if not idx:
            continue
        # The special-values tile repeats its values; judge each distinct input once.
        seen, bad = set(), []
        for i in idx:
            key = int(src[i].to(torch.float32).view(torch.int32))  # bits: NaN != NaN
            if key in seen:
                continue
            seen.add(key)
            if not bool(same[i]):
                bad.append((float(src[i]), float(golden[i]), float(result[i])))
        out[cls] = {"inputs": len(seen), "failures": bad}
    return out


def _load(path):
    import torch

    return torch.load(path, weights_only=False)


def variant_key(d):
    return (d["op"], d["in"], d["out"], d["approx"], d["dest_acc"])


def compare(base_dir, head_dir):
    """Returns a list of per-variant records, base vs head."""
    import torch

    records = {}
    for kind in ("sweep", "specials"):
        for head_file in sorted(Path(head_dir).glob(f"*__{kind}.pt")):
            base_file = Path(base_dir) / head_file.name
            h = _load(head_file)
            rec = records.setdefault(variant_key(h), {"key": variant_key(h)})
            if not base_file.exists():
                rec.setdefault("missing", []).append(f"base {kind}")
                continue
            b = _load(base_file)
            if kind == "sweep":
                bs, hs = _sweep_stats(b), _sweep_stats(h)
                both = bs["_measured"] & hs["_measured"]
                rec["worse"] = int(((hs["_distance"] > bs["_distance"]) & both).sum())
                rec["better"] = int(((hs["_distance"] < bs["_distance"]) & both).sum())
                rec["head_digest"] = hashlib.sha256(
                    h["result"].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                rec["bit_identical"] = (
                    bool(torch.equal(b["result"].view(-1), h["result"].view(-1)))
                    if b["result"].dtype == h["result"].dtype
                    else False
                )
                for s in (bs, hs):
                    s.pop("_distance")
                    s.pop("_measured")
                rec["base"], rec["head"] = bs, hs
            else:
                rec["specials"] = {"base": _specials(b), "head": _specials(h)}
    return list(records.values())
