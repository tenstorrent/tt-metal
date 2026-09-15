# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate input/probability quantization on qualification's original inputs.

Diagnostic only, not a bit-accurate Tensix simulator. All modeled matmuls,
softmax and reductions are FP64. No kernel changes or qualification rescoring.
Select the worst sampled-row head per distribution/length from qualification;
this is deliberately failure-directed, not an unbiased accuracy estimate.
"""

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("qualification", ROOT / "experiments/sdpa-l2/qualification-v1/qualify.py")
QUAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUAL)


def quantize(x, fraction_bits, rounding=True):
    shift = 23 - fraction_bits
    bits = x.float().contiguous().view(torch.int32)
    if rounding:
        bits = bits + (1 << (shift - 1)) - 1 + ((bits >> shift) & 1)
    return (bits & ~((1 << shift) - 1)).view(torch.float32).double()


def statistics(actual, gold):
    delta = actual.double() - gold
    rms = gold.square().mean().sqrt()
    row_den = gold.norm(dim=-1).clamp_min(0.01 * rms * math.sqrt(gold.shape[-1]))
    rows = 100 * delta.norm(dim=-1) / row_den
    ac, gc = actual.double().flatten(), gold.flatten()
    ac, gc = ac - ac.mean(), gc - gc.mean()
    return dict(
        l2_pct=float(100 * delta.norm() / gold.norm()),
        pcc=float((ac @ gc) / (ac.norm() * gc.norm())),
        row_p99_pct=float(torch.quantile(rows, 0.99)),
        row_max_pct=float(rows.max()),
        worst_sample_index=int(rows.argmax()),
    )


def model(q, k, v, batch=32):
    q_original = q.double()
    q_effective = QUAL.repro.preprocess_query(q, 6, 1.0027, True).double() / 1.0027
    k, v = k.double(), v.double()
    outputs = {name: [] for name in ("gold", "q_only", "p6_only", "q_p6", "q_p6_cutoff", "linear_q")}
    diagnostics = []
    for start in range(0, q.shape[0], batch):
        s = q_original[start : start + batch] @ k.T / math.sqrt(q.shape[-1])
        sq = q_effective[start : start + batch] @ k.T / math.sqrt(q.shape[-1])
        ds = sq - s
        p = s.softmax(-1)
        pq = sq.softmax(-1)
        gold = p @ v
        outputs["gold"].append(gold)
        outputs["q_only"].append(pq @ v)
        # First-order sensitivity: dO = sum_j p_j dS_j (V_j - O).
        tangent = (p * ds) @ v - (p * ds).sum(-1, keepdim=True) * gold
        outputs["linear_q"].append(gold + tangent)
        for name, scores in (("p6_only", s), ("q_p6", sq)):
            centered = scores - scores.amax(-1, keepdim=True)
            weights = quantize(centered.exp(), 6)
            outputs[name].append((weights @ v) / weights.sum(-1, keepdim=True))
            if name == "q_p6":
                truncated = weights.masked_fill(centered < -21.45, 0)
                outputs["q_p6_cutoff"].append((truncated @ v) / truncated.sum(-1, keepdim=True))
        centered = sq - sq.amax(-1, keepdim=True)
        tail_mass = (pq * (centered < -21.45)).sum(-1)
        centered_ds = ds - (p * ds).sum(-1, keepdim=True)
        weighted_logit_std = (p * centered_ds.square()).sum(-1).sqrt()
        top = s.topk(2, dim=-1).values
        for i in range(s.shape[0]):
            diagnostics.append(
                dict(
                    sample_index=start + i,
                    entropy=float(-(p[i] * p[i].clamp_min(1e-300).log()).sum()),
                    max_probability=float(p[i].max()),
                    top_logit_gap=float(top[i, 0] - top[i, 1]),
                    q_max_abs=float(q_original[start + i].abs().max()),
                    weighted_logit_error_std=float(weighted_logit_std[i]),
                    cutoff_probability_mass=float(tail_mass[i]),
                )
            )
    return {name: torch.cat(parts) for name, parts in outputs.items()}, diagnostics


def self_test():
    assert torch.equal(
        quantize(torch.tensor([1.0, 1.0078125, 1.015625]), 6), torch.tensor([1.0, 1.0, 1.015625]).double()
    )
    q, k, v = QUAL.repro.make_inputs(1, 32, 64, 128, 17, "normal")
    out, _ = model(q[0, 0], k[0, 0], v[0, 0], batch=8)
    gold = QUAL.repro.reference(q, k, v)[0, 0]
    torch.testing.assert_close(out["gold"], gold, rtol=1e-12, atol=1e-12)
    assert (out["linear_q"] - out["q_only"]).norm() < 0.02 * (out["q_only"] - gold).norm()
    print("PASS: quantization, FP64 reference, and first-order sensitivity self-tests", flush=True)


def device_original_q(device, q, k, v, pos):
    """Original Q/scale; external diagnostic patch selects QK-only or both HiFi4."""
    tensors = [
        ttnn.from_torch(
            x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for x in (q, k, v)
    ]
    result = None
    try:
        result = ttnn.transformer.scaled_dot_product_attention(
            *tensors,
            is_causal=False,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=1024,
                exp_approx_mode=True,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False
            ),
            scale=1 / math.sqrt(128),
        )
        full = ttnn.to_torch(result)
        assert torch.isfinite(full).all()
        return full[..., pos, :].contiguous(), QUAL.tensor_hash(full)
    finally:
        for x in (result, *tensors):
            if x is not None:
                ttnn.deallocate(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="store_true")
    fidelity = parser.add_mutually_exclusive_group()
    fidelity.add_argument(
        "--experimental-qk-hifi4", action="store_true", help="Requires qk-hifi4.patch; not a production option"
    )
    fidelity.add_argument(
        "--experimental-both-hifi4",
        action="store_true",
        help="Requires both-hifi4.patch, including matched denominator",
    )
    parser.add_argument(
        "--original-q", action="store_true", help="Remove Q preprocessing and scale compensation in QK-HiFi4 diagnostic"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 262144])
    parser.add_argument("--distributions", nargs="+", default=["normal", "scaled_qk", "outliers"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    experimental = args.experimental_qk_hifi4 or args.experimental_both_hifi4
    if args.original_q and not experimental:
        parser.error("--original-q requires a HiFi4 diagnostic patch flag")
    torch.set_num_threads(args.threads)
    self_test()
    records = [
        json.loads(line)
        for line in (ROOT / "experiments/sdpa-l2/qualification-v1/accepted-results.jsonl").read_text().splitlines()
    ]
    selected = []
    for n in args.lengths:
        for dist in args.distributions:
            eligible = [
                r
                for r in records
                if r["mode"] == "accurate"
                and r["kv_len"] == n
                and r["distribution"] == dist
                and r["group"] in ("normal", "stress")
            ]
            record = max(eligible, key=lambda r: max(h["row_max_pct"] for h in r["per_head"]))
            head = max(record["per_head"], key=lambda h: h["row_max_pct"])["head"]
            selected.append((record, head))
    device = ttnn.open_device(device_id=0) if args.device else None
    try:
        with args.output.open("x") as stream:
            for record, head in selected:
                start = time.monotonic()
                print(f"START {record['id']} head={head}", flush=True)
                q, k, v = QUAL.inputs(record)
                input_hashes = [QUAL.tensor_hash(x) for x in (q, k, v)]
                assert input_hashes == record["input_sha256"], "Original input hash mismatch"
                pos = QUAL.positions(record["q_len"], record["seed"])
                assert pos.tolist() == record["positions"]
                actual = None
                full_hash = None
                if device is not None:
                    if args.original_q:
                        actual_all, full_hash = device_original_q(device, q, k, v, pos)
                    else:
                        actual_all, full_hash = QUAL.run_device(device, q, k, v, pos, "accurate")
                    if not experimental:
                        assert full_hash == record["full_output_sha256"], "Retained device output changed"
                    actual = actual_all[0, head].double()
                outputs, rows = model(q[0, head, pos], k[0, head], v[0, head])
                gold = outputs.pop("gold")
                metrics = {name: statistics(value.bfloat16(), gold) for name, value in outputs.items()}
                metrics["rounding_oracle"] = statistics(gold.bfloat16(), gold)
                metrics["cutoff_increment"] = statistics(outputs["q_p6_cutoff"], outputs["q_p6"])
                if actual is not None:
                    metrics["device"] = statistics(actual, gold)
                    metrics["device_vs_q_only"] = statistics(actual, outputs["q_only"])
                    metrics["device_vs_q_p6"] = statistics(actual, outputs["q_p6"])
                    error = actual - gold
                    qerror = outputs["q_only"] - gold
                    q_alignment = float((error * qerror).sum() / (error.norm() * qerror.norm()))
                else:
                    q_alignment = None
                for name, value in outputs.items():
                    row_errors = 100 * (value.bfloat16().double() - gold).norm(dim=-1) / gold.norm(dim=-1)
                    for row, error in zip(rows, row_errors):
                        row[name + "_l2_pct"] = float(error)
                if actual is not None:
                    for row, error in zip(rows, 100 * (actual - gold).norm(dim=-1) / gold.norm(dim=-1)):
                        row["device_l2_pct"] = float(error)
                result = dict(
                    id=record["id"],
                    head=head,
                    distribution=record["distribution"],
                    kv_len=record["kv_len"],
                    experimental_qk_hifi4=args.experimental_qk_hifi4,
                    experimental_both_hifi4=args.experimental_both_hifi4,
                    diagnostic_mode=os.environ.get("TT_SDPA_ACCURACY_DIAG"),
                    original_q=args.original_q,
                    full_output_sha256=full_hash,
                    input_sha256=input_hashes,
                    reference_positions=pos.tolist(),
                    metrics=metrics,
                    device_q_error_cosine=q_alignment,
                    qualification_head=record["per_head"][head],
                    rows=rows,
                    elapsed_s=time.monotonic() - start,
                )
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps({k: result[k] for k in ("id", "head", "metrics", "device_q_error_cosine", "elapsed_s")}),
                    flush=True,
                )
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
