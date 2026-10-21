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
import math
import shutil
import sys
from pathlib import Path

import runner

sys.path.insert(0, str(runner.PYTHON_TESTS))

#: The driver lives next to this file. pytest applies the harness's conftest.py only to
#: files under tests/python_tests/, so each run copies it there and removes it after.
DRIVER = "test_sfpu_report_accuracy.py"
DRIVER_SOURCE = Path(__file__).resolve().parent / DRIVER


def measure(side, arch, ops, out_dir, log, jobs=8, formats=()):
    """Dump the side's raw results for ``ops`` (MathOperation names) to ``out_dir``.

    Returns the device run's pytest exit code: 0 or ``runner.NO_TESTS_COLLECTED`` is
    clean, anything else means some variants failed and left no results.
    """
    # compare() reads every result in the directory: a previous run's must not count.
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    unary = [o for o in ops if not o.startswith("Sfpu")]
    binary = [o for o in ops if o.startswith("Sfpu")]
    env = {
        "SFPU_REPORT_DUMP": str(out_dir),
        "SFPU_REPORT_OPS": ",".join(unary),
        "SFPU_REPORT_BINARY_OPS": ",".join(binary),
        "SFPU_REPORT_FORMATS": ",".join(formats),
    }
    installed = runner.PYTHON_TESTS / DRIVER
    shutil.copyfile(DRIVER_SOURCE, installed)
    try:
        # A variant that fails on one side shows up as missing in the comparison.
        return runner.produce_consume(side, arch, [DRIVER], env=env, log=log, producer_jobs=jobs, check=False)
    finally:
        installed.unlink(missing_ok=True)


def _sweep_stats(d):
    import torch
    from helpers.format_config import DataFormat
    from helpers.llk_params import ApproximationMode, MathOperation
    from helpers.ulp import FLUSH_SUBNORMAL_OUTPUTS, ulp_distance, ulp_stats
    from helpers.ulp_sweep import measurable_mask, nonfinite_failures

    in_fmt, out_fmt = DataFormat[d["in"]], DataFormat[d["out"]]
    src, golden, result = d["src"], torch.as_tensor(d["golden"]), d["result"]
    if d.get("exact"):
        return _exact_stats(d)
    from helpers.llk_params import DestAccumulation

    # The Dest width decides which inputs reach the SFPU as subnormals (flushed), so
    # the sweep's own masking needs it, as the nightly ULP sweep passes it.
    dest = DestAccumulation[d["dest_acc"]]
    mask = measurable_mask(src, golden, result, in_fmt, output_format=out_fmt, dest_acc=dest)
    if d.get("binary"):
        # A lane is measurable only if both operands are.
        mask = mask & measurable_mask(d["src_b"], golden, result, in_fmt, output_format=out_fmt, dest_acc=dest)
    # As test_unary_sfpu_ulp.py measures it: the Dest flushes subnormal results, and
    # the approx mode names the cell for the sweep's known non-finite lanes.
    distance = ulp_distance(golden, result, flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS)
    stats = ulp_stats(distance, mask)
    nonfinite = nonfinite_failures(
        MathOperation[d["op"]],
        src,
        golden,
        result,
        in_fmt,
        out_fmt,
        dest_acc=dest,
        approx_mode=ApproximationMode[d["approx"]],
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
            (float(src[i]), float(golden[i]), float(result[i])) for i in nonfinite.nonzero().flatten()[:3].tolist()
        ],
        "worst_input": (float(src.reshape(-1)[stats["worst_index"]]) if stats["worst_index"] is not None else None),
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


#: How close to an input a float result must be to be that input: the coarsest output
#: format (bf16, bfp8) keeps 8 significant bits, so rounding an fp32 input to it moves
#: the value by up to 2**-8 of itself.
SAME_AS_INPUT = 2.0**-7

#: Fewer compared lanes than this say nothing about a whole tile.
MIN_LANES = 64


def _returns_input(d, lanes=None, exact=False):
    """``"first"`` or ``"second"``: the operand the result is, unchanged, on every compared
    lane; None when it is neither. It is what comes back when no kernel ran over the
    tile: the unpacked input, still in Dest.

    ``lanes`` selects the lanes to compare (all when None); non-finite ones never count.
    A float result may differ from its input by the output format's rounding, an exact
    op's (comparisons, integers) not at all.
    """
    import torch

    result = d["result"].reshape(-1).to(torch.float64)
    for name, operand in (("first", "src"), ("second", "src_b")):
        if operand not in d:
            continue
        src = torch.as_tensor(d[operand]).reshape(-1).to(torch.float64)
        if src.numel() != result.numel():
            continue
        keep = torch.isfinite(src) & torch.isfinite(result)
        if lanes is not None:
            keep &= lanes.reshape(-1)
        if int(keep.sum()) < MIN_LANES:
            continue
        allowed = 0.0 if exact else SAME_AS_INPUT * src[keep].abs()
        if bool(((result[keep] - src[keep]).abs() <= allowed).all()):
            return name
    return None


#: The smallest normal magnitude of a float format: a nonzero input below it is subnormal.
_MIN_NORMAL = {"Float16": 2.0**-14}


def _min_normal(fmt):
    return _MIN_NORMAL.get(fmt, 2.0**-126 if fmt.startswith(("Float", "Bfp", "Tf32")) else 0.0)


def _golden_is_ieee(operands, min_normal):
    """Whether the harness golden is an IEEE reference for these inputs. It models the
    unpacker and the Dest, so it maps a NaN input to inf and -0 to +0, and flushes
    subnormals: for those it says what the hardware does, not what is right."""
    for v in operands:
        if v != v or (v == 0 and math.copysign(1.0, v) < 0) or (v != 0 and abs(v) < min_normal):
            return False
    return True


def _specials(d):
    """{input bits: (class, input, result, reference)} for each distinct special input.

    ``reference`` is the golden where it is an IEEE reference for the input, else None.
    What a reviewer needs is what the hardware returns and whether the PR changed it;
    the reference tells a change toward the right answer from a change away from it.
    """
    import torch

    src, result = d["src"].reshape(-1), d["result"].reshape(-1)
    golden = torch.as_tensor(d["golden"]).reshape(-1) if d.get("golden") is not None else None
    min_normal = _min_normal(d.get("in", ""))
    bits = lambda t, i: int(t[i].to(torch.float64).view(torch.int64))  # noqa: E731; NaN != NaN

    def ref(i, operands):
        if golden is None or i >= golden.numel() or not _golden_is_ieee(operands, min_normal):
            return None
        return float(golden[i])

    out = {}
    if d.get("binary"):
        src_b = d["src_b"].reshape(-1)
        for i in range(result.numel()):
            key = (bits(src, i), bits(src_b, i))
            if key not in out:
                pair = (float(src[i]), float(src_b[i]))
                out[key] = ("pair", pair, float(result[i]), ref(i, pair))
        return out
    for i, cls in enumerate(d["classes"]):
        key = bits(src, i)
        if key not in out:
            out[key] = (cls, float(src[i]), float(result[i]), ref(i, (float(src[i]),)))
    return out


def _bits(x):
    import struct

    return struct.pack(">d", x)


def specials_diff(base, head):
    """Special inputs whose result changed, and NaN propagation on each side."""
    changed = []
    for key, (cls, x, new, *ref) in head.items():
        if key in base:
            old = base[key][2]
            if _bits(old) != _bits(new) and not (old != old and new != new):
                changed.append({"class": cls, "input": x, "old": old, "new": new, "ref": ref[0] if ref else None})
    nan_ok = {
        side: all(v[2] != v[2] for v in spec.values() if v[0] == "nan")
        for side, spec in (("base", base), ("head", head))
    }
    return {"changed": changed, "nan_propagates": nan_ok}


def _load(path):
    import torch

    return torch.load(path, weights_only=False)


def variant_key(d):
    return (d["op"], d["in"], d["out"], d["approx"], d["dest_acc"])


def compare(base_dir, head_dir):
    """Returns a list of per-variant records, base vs head.

    A result on one side only is recorded under ``missing`` as the side without it and
    the stimulus (``"head sweep"``): the PR broke the variant, or made it build.
    """
    import torch

    records = {}
    for kind in ("sweep", "random", "specials"):
        names = sorted({p.name for d in (base_dir, head_dir) for p in Path(d).glob(f"*__{kind}.pt")})
        for name in names:
            base_file, head_file = Path(base_dir) / name, Path(head_dir) / name
            if not (base_file.exists() and head_file.exists()):
                one = _load(base_file if base_file.exists() else head_file)
                rec = records.setdefault(variant_key(one), {"key": variant_key(one)})
                rec.setdefault("missing", []).append(f"{'base' if head_file.exists() else 'head'} {kind}")
                continue
            h, b = _load(head_file), _load(base_file)
            rec = records.setdefault(variant_key(h), {"key": variant_key(h)})
            if kind in ("sweep", "random"):
                bs, hs = _sweep_stats(b), _sweep_stats(h)
                rec["kind"] = kind
                rec["binary"] = bool(h.get("binary"))
                rec["coverage"] = h.get("coverage")
                # A side that hands back an input unchanged ran no kernel over the tile.
                exact = bs.get("metric") == "exact"
                rec["returns_input"] = {
                    "base": _returns_input(b, bs.get("_measured"), exact),
                    "head": _returns_input(h, hs.get("_measured"), exact),
                }
                if exact:
                    rec["worse"] = int((hs["_wrong"] & ~bs["_wrong"]).sum())
                    rec["better"] = int((~hs["_wrong"] & bs["_wrong"]).sum())
                    bs.pop("_wrong")
                    hs.pop("_wrong")
                else:
                    both = bs["_measured"] & hs["_measured"]
                    rec["worse"] = int(((hs["_distance"] > bs["_distance"]) & both).sum())
                    rec["better"] = int(((hs["_distance"] < bs["_distance"]) & both).sum())
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
