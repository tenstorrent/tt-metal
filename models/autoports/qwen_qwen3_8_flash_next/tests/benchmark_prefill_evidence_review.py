# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU review; never import TTNN or open devices."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(
    description="CPU-only independent raw tensor, hash and canary review; no TTNN imports."
)
parser.add_argument("--directory", type=Path, required=True)
args = parser.parse_args()
root = args.directory
if (root / "review.json").exists():
    raise FileExistsError("Preserve the existing review; choose a fresh evidence directory")
torch.set_num_threads(8)
d = json.loads((root / "summary.json").read_text())
assert all(r.get("mesh_closed") for r in d["layers"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pairs(a, b, path=""):
    if isinstance(a, torch.Tensor):
        yield path, a, b
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            yield from pairs(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from pairs(x, y, path + "/" + str(i))
    else:
        raise ValueError(type(a))


def canaries(x):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "nonzero":
                out.append(v == 0)
            elif k == "sentinel_nonzero":
                out.extend(n == 0 for n in v)
            elif k == "finite":
                out.append(v is True)
            else:
                out.extend(canaries(v))
    elif isinstance(x, list):
        for v in x:
            out.extend(canaries(v))
    return out


sources = []
for p, s in d["sources"].items():
    snap = root / (Path(p).name + ".txt")
    assert sha(snap) == s
    sources.append(dict(path=str(snap), sha256=s))
review = dict(
    status="completed_independent_numerical_review",
    reviewer_source_sha256=sha(__file__),
    summary_sha256=sha(root / "summary.json"),
    decision="Independent component evidence review; no model accuracy verdict. Timing includes host upload/readback and is not pure execution or full48 performance. Owner decides qualification from complete evidence.",
    raw_tensor_pairs=0,
    raw_hashes_verified=0,
    input_hashes_verified=0,
    source_snapshots_verified=len(sources),
    numeric_differences=[],
    canary_assertions=0,
    layers=[],
    trace_tracking=d["trace_tracking"],
    memory="Snapshots only; not peaks or proof full48 fits1GiB trace/ordinaryDRAM.",
    canary_coverage="GDN/PLE:31 unselected slots among first32; QSA:four unallocated sentinel pages. Active-slot state and129 mapped QSA pages are retained; not every unused physical page or remaining scratch/request slot.",
)
for layer in d["layers"]:
    assert sha(root / f"layer{layer['layer']}" / "inputs.pt") == layer["input_sha256"]
    review["input_hashes_verified"] += 1
    summary = dict(layer=layer["layer"], cases=[])
    for case in layer["cases"]:
        p = Path(case["raw"]["path"])
        assert sha(p) == case["raw"]["sha256"]
        review["raw_hashes_verified"] += 1
        raw = torch.load(p, map_location="cpu", weights_only=True)
        nonexact = []
        total = 0
        for name, a, b in pairs(raw["original"], raw["candidate"]):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()
            eq = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            total += 1
            if not eq:
                nonexact.append(
                    dict(
                        tensor=name,
                        dtype=str(a.dtype),
                        shape=list(a.shape),
                        max_abs=float((a.double() - b.double()).abs().max()),
                        different_values=int((a.double() != b.double()).count_nonzero()),
                        different_bytes=int(
                            (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).count_nonzero()
                        ),
                    )
                )
        cc = canaries([v[k] for v in case["variants"].values() for k in ("state_checks", "handoff_checks")])
        assert cc and all(cc)
        review["canary_assertions"] += len(cc)
        review["raw_tensor_pairs"] += total
        row = dict(
            length=case["length"],
            raw_pairs=total,
            all_raw_bytes_equal=not nonexact,
            nonexact=nonexact,
            baseline_median_seconds=case["variants"]["original"]["median_seconds"],
            candidate_median_seconds=case["variants"]["candidate"]["median_seconds"],
            median_speedup=case["median_speedup"],
        )
        summary["cases"].append(row)
        if nonexact:
            review["numeric_differences"].append(dict(layer=layer["layer"], **row))
        del raw
    review["layers"].append(summary)
review["state_and_handoff_bit_exact"] = all(
    all(x["tensor"].startswith("/output/") for x in row["nonexact"]) for row in review["numeric_differences"]
)
(root / "review.json").write_text(json.dumps(review, indent=2) + "\n")
print(json.dumps({k: v for k, v in review.items() if k not in ("layers", "numeric_differences")}, indent=2))
print(
    [
        (r["layer"], [(c["length"], c["median_speedup"]) for c in r["cases"] if c["length"] >= 1024])
        for r in review["layers"]
    ]
)
print("Nonexact cases", [(r["layer"], r["length"], len(r["nonexact"])) for r in review["numeric_differences"]])
