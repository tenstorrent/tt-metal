"""Shared v3 accuracy contract. Tensor helpers require torch, not a device.

L2 values are percentages, not fractions or squared relative error. Acceptance
is separate from replay determinism, source/input integrity and performance.
"""

import math

RELATIVE_ALLOWANCE = 1.05
L2_FLOOR_PP = 0.0001
ZERO_REFERENCE_ABS_FLOOR = 1e-6


def acceptance(baseline, candidate):
    """Pure-scalar gate; undefined relative metrics never silently pass."""
    if not baseline["finite"] or not candidate["finite"]:
        return {"pass": False, "reason": "nonfinite output or reference"}
    if baseline["reference_is_zero"] != candidate["reference_is_zero"]:
        raise ValueError("Baseline and candidate do not share reference classification")
    if baseline["reference_is_zero"]:
        threshold = max(ZERO_REFERENCE_ABS_FLOOR, RELATIVE_ALLOWANCE * baseline["max_abs"])
        value = candidate["max_abs"]
        metric = "max_abs"
    else:
        if baseline["l2_pct"] is None or candidate["l2_pct"] is None:
            raise ValueError("Nonzero reference requires finite L2 metrics")
        threshold = RELATIVE_ALLOWANCE * baseline["l2_pct"] + L2_FLOOR_PP
        value = candidate["l2_pct"]
        metric = "l2_pct"
    passed = math.isfinite(value) and math.isfinite(threshold) and value <= threshold
    return {"pass": passed, "metric": metric, "value": value, "limit": threshold,
            "reason": "within budget" if passed else "numerical budget exceeded"}


def metrics(actual, reference):
    """All-output metrics, with explicit zero-reference and zero-row handling."""
    import torch

    if actual.shape != reference.shape:
        raise ValueError("Output/reference shapes differ")
    a, ref = actual.double(), reference.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(ref).all())
    if not bool(torch.isfinite(ref).all()):
        raise ValueError("Reference is nonfinite")
    ref_norm = ref.norm().item()
    result = {"finite": finite, "reference_is_zero": ref_norm == 0,
              "reference_norm": ref_norm, "reference_rms": ref.square().mean().sqrt().item(),
              "l2_pct": None, "pcc": None, "max_abs": None,
              "row_l2_pct_p95": None, "row_l2_pct_p99": None, "row_l2_pct_max": None,
              "zero_reference_rows": None, "zero_row_mismatches": None,
              "zero_row_max_abs": None}
    if not finite:
        return result
    delta = a - ref
    result["max_abs"] = delta.abs().max().item()
    if ref_norm:
        result["l2_pct"] = 100 * delta.norm().item() / ref_norm
    ac, rc = a.flatten() - a.mean(), ref.flatten() - ref.mean()
    an, rn = ac.norm().item(), rc.norm().item()
    if an > 0 and rn > 1e-12 * ref_norm:
        result["pcc"] = (ac @ rc).item() / (an * rn)
    row_ref = ref.norm(dim=-1).flatten()
    row_err = delta.norm(dim=-1).flatten()
    nonzero = row_ref != 0
    result["zero_reference_rows"] = int((~nonzero).sum().item())
    result["zero_row_mismatches"] = int(((~nonzero) & (row_err != 0)).sum().item())
    if bool((~nonzero).any()):
        result["zero_row_max_abs"] = delta.reshape(-1, delta.shape[-1])[~nonzero].abs().max().item()
    if bool(nonzero.any()):
        row_l2 = 100 * row_err[nonzero] / row_ref[nonzero]
        result.update(row_l2_pct_p95=torch.quantile(row_l2, 0.95).item(),
                      row_l2_pct_p99=torch.quantile(row_l2, 0.99).item(),
                      row_l2_pct_max=row_l2.max().item())
    return result


def compare(baseline, candidate, reference):
    """Neither similarity to baseline nor PCC substitutes for reference L2."""
    bm, cm = metrics(baseline, reference), metrics(candidate, reference)
    distance = None
    if bm["finite"] and cm["finite"] and bm["reference_norm"]:
        distance = 100 * (candidate.double() - baseline.double()).norm().item() / bm["reference_norm"]
    return {"baseline_metrics": bm, "candidate_metrics": cm,
            "acceptance": acceptance(bm, cm),
            "candidate_minus_baseline_l2_pct_of_reference": distance,
            "note": "Row metrics exclude exactly zero-reference rows, whose absolute errors are reported separately; row/PCC regressions require review even if global L2 passes."}


def scalar_self_test():
    def record(l2=None, absolute=0.0, zero=False, finite=True):
        return {"finite": finite, "reference_is_zero": zero, "l2_pct": l2, "max_abs": absolute}
    b = record(0.18)
    assert acceptance(b, record(0.18909))["pass"]
    assert not acceptance(b, record(0.18911))["pass"]
    assert acceptance(b, record(0.01))["pass"]  # Improvements are allowed.
    assert acceptance(record(0.0), record(0.00009))["pass"]
    assert not acceptance(record(0.0), record(0.00011))["pass"]
    assert acceptance(record(zero=True), record(zero=True, absolute=1e-6))["pass"]
    assert not acceptance(record(zero=True), record(zero=True, absolute=1.1e-6))["pass"]
    assert not acceptance(b, record(finite=False))["pass"]
    assert not acceptance(b, record(float("nan")))["pass"]
    try:
        acceptance(b, record(zero=True))
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched references were not rejected")
    print("PASS: 10 scalar acceptance self-tests")


def tensor_self_test():
    import torch

    ref = torch.tensor([[1.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    actual = ref.clone()
    actual[0] *= 1.01
    actual[1, 0] = 1e-7
    m = metrics(actual, ref)
    assert abs(m["l2_pct"] - 1.0) < 1e-8
    assert m["zero_reference_rows"] == 1 and m["zero_row_mismatches"] == 1
    assert m["zero_row_max_abs"] == 1e-7
    assert abs(m["row_l2_pct_max"] - 1.0) < 1e-8
    assert not compare(ref, actual, ref)["acceptance"]["pass"]
    assert compare(ref, ref.clone(), ref)["acceptance"]["pass"]
    z = torch.zeros_like(ref)
    assert metrics(z, z)["l2_pct"] is None
    assert metrics(torch.ones_like(z), torch.ones_like(z))["pcc"] is None
    assert compare(z, z + 1e-7, z)["acceptance"]["pass"]
    assert not compare(z, z + 1e-5, z)["acceptance"]["pass"]
    bad = ref.clone()
    bad[0, 0] = float("nan")
    assert not compare(ref, bad, ref)["acceptance"]["pass"]
    print("PASS: tensor metric/zero-reference/nonfinite self-tests")


if __name__ == "__main__":
    scalar_self_test()
    if "--tensor-self-test" in __import__("sys").argv:
        tensor_self_test()
