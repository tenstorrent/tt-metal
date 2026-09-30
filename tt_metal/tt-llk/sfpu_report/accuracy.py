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

#: Special-value classes of test_sfpu_report_accuracy.special_values().
CLASSES = ("nan", "inf", "zero", "subnormal", "extreme")


def measure(side, arch, ops, out_dir, log, jobs=8, formats=()):
    """Dump the side's raw results for ``ops`` (MathOperation names) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unary = [o for o in ops if not o.startswith("Sfpu")]
    binary = [o for o in ops if o.startswith("Sfpu")]
    env = {
        "SFPU_REPORT_DUMP": str(out_dir),
        "SFPU_REPORT_OPS": ",".join(unary),
        "SFPU_REPORT_BINARY_OPS": ",".join(binary),
        "SFPU_REPORT_FORMATS": ",".join(formats),
    }
    # A variant that fails on one side shows up as missing in the comparison.
    runner.produce_consume(
        side, arch, [DRIVER], env=env, log=log, producer_jobs=jobs, check=False
    )
    return out_dir


def _sweep_stats(d):
    import torch
    from helpers.format_config import DataFormat
    from helpers.llk_params import MathOperation
    from helpers.ulp import ulp_distance, ulp_stats
    from helpers.ulp_sweep import measurable_mask, nonfinite_failures

    in_fmt, out_fmt = DataFormat[d["in"]], DataFormat[d["out"]]
    src, golden, result = d["src"], torch.as_tensor(d["golden"]), d["result"]
    if d.get("exact"):
        return _exact_stats(d)
    from helpers.llk_params import DestAccumulation

    # The Dest width decides which inputs reach the SFPU as subnormals (flushed), so
    # the sweep's own masking needs it, as the nightly ULP sweep passes it.
    dest = DestAccumulation[d["dest_acc"]]
    mask = measurable_mask(
        src, golden, result, in_fmt, output_format=out_fmt, dest_acc=dest
    )
    if d.get("binary"):
        # A lane is measurable only if both operands are.
        mask = mask & measurable_mask(
            d["src_b"], golden, result, in_fmt, output_format=out_fmt, dest_acc=dest
        )
    distance = ulp_distance(golden, result)
    stats = ulp_stats(distance, mask)
    nonfinite = nonfinite_failures(
        MathOperation[d["op"]], src, golden, result, in_fmt, out_fmt, dest_acc=dest
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


def _same(result, golden):
    """Lane-wise: identical value, NaN-ness and sign (so -0 != +0)."""
    import torch

    r, g = result.to(torch.float64), torch.as_tensor(golden).to(torch.float64)
    both_nan = torch.isnan(r) & torch.isnan(g)
    return both_nan | ((r == g) & (torch.signbit(r) == torch.signbit(g)))


def _exact_stats(d):
    """Comparisons and integer ops: a lane is right or wrong, there is no ULP."""
    import torch

    golden = torch.as_tensor(d["golden"]).reshape(-1)
    wrong = ~_same(d["result"].reshape(-1), golden)
    lanes = int(wrong.numel())
    return {
        "metric": "exact",
        "lanes": lanes,
        "wrong": int(wrong.sum()),
        "wrong_examples": [
            (
                float(d["src"].reshape(-1)[i]),
                float(d["src_b"].reshape(-1)[i]) if "src_b" in d else None,
                float(golden[i]),
                float(d["result"].reshape(-1)[i]),
            )
            for i in wrong.nonzero().flatten()[:3].tolist()
        ],
        "_wrong": wrong,
    }


def _specials(d):
    """{input bits: (class, input, result)} for each distinct special input.

    Not judged against the golden: the harness golden models the unpacker and the
    Dest, so for NaN, signed zeros and subnormals it is not an IEEE reference (it
    maps a NaN input to inf, and -0 to +0). What a reviewer needs is what the
    hardware returns, and whether the PR changed it.
    """
    import torch

    src, result = d["src"].reshape(-1), d["result"].reshape(-1)
    bits = lambda t, i: int(
        t[i].to(torch.float64).view(torch.int64)
    )  # noqa: E731; NaN != NaN
    out = {}
    if d.get("binary"):
        src_b = d["src_b"].reshape(-1)
        for i in range(result.numel()):
            key = (bits(src, i), bits(src_b, i))
            if key not in out:
                out[key] = ("pair", (float(src[i]), float(src_b[i])), float(result[i]))
        return out
    for i, cls in enumerate(d["classes"]):
        key = bits(src, i)
        if key not in out:
            out[key] = (cls, float(src[i]), float(result[i]))
    return out


def _bits(x):
    import struct

    return struct.pack(">d", x)


def specials_diff(base, head):
    """Special inputs whose result changed, and NaN propagation on each side."""
    changed = []
    for key, (cls, x, new) in head.items():
        if key in base:
            old = base[key][2]
            if _bits(old) != _bits(new) and not (old != old and new != new):
                changed.append({"class": cls, "input": x, "old": old, "new": new})
    nan_ok = {
        side: all(r != r for c, _, r in spec.values() if c == "nan")
        for side, spec in (("base", base), ("head", head))
    }
    return {"changed": changed, "nan_propagates": nan_ok}


def _load(path):
    import torch

    return torch.load(path, weights_only=False)


def variant_key(d):
    return (d["op"], d["in"], d["out"], d["approx"], d["dest_acc"])


def compare(base_dir, head_dir):
    """Returns a list of per-variant records, base vs head."""
    import torch

    records = {}
    for kind in ("sweep", "random", "specials"):
        for head_file in sorted(Path(head_dir).glob(f"*__{kind}.pt")):
            base_file = Path(base_dir) / head_file.name
            h = _load(head_file)
            rec = records.setdefault(variant_key(h), {"key": variant_key(h)})
            if not base_file.exists():
                rec.setdefault("missing", []).append(f"base {kind}")
                continue
            b = _load(base_file)
            if kind in ("sweep", "random"):
                bs, hs = _sweep_stats(b), _sweep_stats(h)
                rec["kind"] = kind
                rec["binary"] = bool(h.get("binary"))
                rec["coverage"] = h.get("coverage")
                if bs.get("metric") == "exact":
                    rec["worse"] = int((hs["_wrong"] & ~bs["_wrong"]).sum())
                    rec["better"] = int((~hs["_wrong"] & bs["_wrong"]).sum())
                    bs.pop("_wrong")
                    hs.pop("_wrong")
                else:
                    both = bs["_measured"] & hs["_measured"]
                    rec["worse"] = int(
                        ((hs["_distance"] > bs["_distance"]) & both).sum()
                    )
                    rec["better"] = int(
                        ((hs["_distance"] < bs["_distance"]) & both).sum()
                    )
                    for side in (bs, hs):
                        side.pop("_distance")
                        side.pop("_measured")
                rec["head_digest"] = hashlib.sha256(
                    h["result"].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                rec["bit_identical"] = (
                    bool(torch.equal(b["result"].view(-1), h["result"].view(-1)))
                    if b["result"].dtype == h["result"].dtype
                    else False
                )
                rec["base"], rec["head"] = bs, hs
            else:
                rec["specials"] = specials_diff(_specials(b), _specials(h))
    return list(records.values())
