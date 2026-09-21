# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Numerics-relaxed FP32 PV/state experiments; immutable best C/D controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import ttnn

spec = importlib.util.spec_from_file_location("fp32_v3_core", Path(__file__).with_name("integrity_core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
metrics_spec = importlib.util.spec_from_file_location("fp32_v3_numerics", Path(__file__).parents[1] / "numerics.py")
numerics = importlib.util.module_from_spec(metrics_spec)
metrics_spec.loader.exec_module(numerics)


def accuracy(actual, ref, baseline):
    x, r, b = actual.double(), ref.double(), baseline.double()
    error, base_error = x - r, b - r
    norm = float(r.norm())
    row_norm = r.norm(dim=-1)
    row_l2 = 100 * error.norm(dim=-1) / row_norm.clamp_min(1e-300)
    max_abs = float(error.abs().max())
    base_max_abs = float(base_error.abs().max())
    l2 = 100 * float(error.norm()) / norm if norm else None
    base_l2 = 100 * float(base_error.norm()) / norm if norm else None
    threshold = 1.05 * base_l2 + 0.0001 if norm else max(1e-6, 1.05 * base_max_abs)
    finite = bool(torch.isfinite(x).all() and torch.isfinite(r).all())
    centered_x, centered_r = x.flatten() - x.mean(), r.flatten() - r.mean()
    pcc_denom = float(centered_x.norm() * centered_r.norm())
    near = r.abs() <= max(1e-12, float(r.square().mean().sqrt()) * 1e-3)
    centered_ref = r - r.mean(dim=-2, keepdim=True)
    centered_actual = x - x.mean(dim=-2, keepdim=True)
    centered_base = b - b.mean(dim=-2, keepdim=True)
    centered_norm = float(centered_ref.norm())
    return dict(
        l2_pct=l2, baseline_l2_pct=base_l2, numerical_threshold=threshold,
        numerical_pass=finite and (l2 if norm else max_abs) <= threshold,
        finite=finite, reference_norm=norm, max_abs_error=max_abs,
        baseline_max_abs_error=base_max_abs,
        pcc=float((centered_x * centered_r).sum()) / pcc_denom if pcc_denom else None,
        row_l2_p95_pct=float(torch.quantile(row_l2.flatten(), 0.95)) if norm else None,
        row_l2_p99_pct=float(torch.quantile(row_l2.flatten(), 0.99)) if norm else None,
        row_l2_worst_pct=float(row_l2.max()) if norm else None,
        baseline_distance_pct=100 * float((x - b).norm()) / float(b.norm()) if float(b.norm()) else None,
        baseline_max_abs_distance=float((x - b).abs().max()),
        near_zero_count=int(near.sum()),
        near_zero_max_abs_error=float(error[near].abs().max()) if near.any() else None,
        centered_query_variation_l2_pct=100 * float((centered_actual-centered_ref).norm()) / centered_norm if centered_norm > 1e-12 * norm else None,
        baseline_centered_query_variation_l2_pct=100 * float((centered_base-centered_ref).norm()) / centered_norm if centered_norm > 1e-12 * norm else None,
    )


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--variants", default="D,C")
    p.add_argument("--algorithms", default="baseline,full_pv")
    p.add_argument("--distributions", default="normal")
    p.add_argument("--k-chunks", type=int, default=3)
    p.add_argument("--q-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=1240)
    p.add_argument("--distinct-kv", action="store_true")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--iters", type=int, default=1)
    opts = p.parse_args()
    target = core.HERE / (opts.label + ".json")
    assert not target.exists()
    algorithms = opts.algorithms.split(",")
    assert algorithms[0] == "baseline"
    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for variant in opts.variants.split(","):
            for distribution in opts.distributions.split(","):
                oracle = None
                for round_index in range(opts.rounds):
                    for algorithm in algorithms if round_index % 2 == 0 else algorithms[::-1]:
                        args = argparse.Namespace(**vars(opts))
                        args.variant = variant
                        args.mode = "accurate" if variant == "D" else "qk4_pv2"
                        args.distribution = distribution
                        args.algorithm = algorithm
                        args.hybrid_block_pack = False
                        args.label = f"{opts.label}-{variant}-{distribution}-r{round_index}-{algorithm}"
                        record, actual, ref, inputs = core.run(device, args)
                        if oracle is None:
                            assert algorithm == "baseline"
                            oracle = actual.clone()
                        record.update(accuracy(actual, ref, oracle))
                        record.update(numerics.compare(oracle, actual, ref))
                        record["shared_metrics_sha256"] = hashlib.sha256(Path(numerics.__file__).read_bytes()).hexdigest()
                        record.update(round_index=round_index, raw_trace_equal=True,
                                      input_hashes=[digest(x) for x in inputs],
                                      private_source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                             for path in core.HERE.glob("*") if path.suffix in (".py", ".cpp", ".hpp")})
                        records.append(record)
                        torch.save(actual, core.HERE / (args.label + ".pt"))
                        target.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
                        print("RESULT " + json.dumps(record, allow_nan=False), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
