# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-head qualification of the HiFi4 accuracy diagnostic, without fallback.

Uses qualification-v1's original inputs, reference sampling and stricter 0.5%
gates. Common Q/K/V and constant V are excluded as requested. No performance
measurement. Resume retains completed records and original raw gate results.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QUAL = module("qual", "experiments/sdpa-l2/qualification-v1/qualify.py")
ADJ = module("adj", "experiments/sdpa-l2/qualification-v1/adjudicate.py")


def verify_diagnostic_source():
    """Prevent silently labeling the retained/default build as a HiFi4 experiment."""
    manifest = ROOT / "experiments/sdpa-l2/accuracy-investigation/DIAGNOSTIC-SHA256.txt"
    for line in manifest.read_text().splitlines():
        expected, path = line.split(maxsplit=1)
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, f"Apply diagnostic.patch and rebuild first: source hash mismatch for {path}"


def invoke(device, q, k, v, pos, trace=False):
    shape = dict(q_len=q.shape[-2], kv_len=k.shape[-2])
    reason = QUAL.unsupported(shape, "accurate")
    if reason:
        raise ValueError("Fallback forbidden: " + reason)
    tensors = [
        ttnn.from_torch(
            x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for x in (q, k, v)
    ]
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=QUAL.k_chunk(shape, "accurate"),
        exp_approx_mode=True,
    )
    # Temporary factory emits HiFi4 while preserving the known streaming dispatch.
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def operation():
        return ttnn.transformer.scaled_dot_product_attention(
            *tensors, is_causal=False, program_config=program, compute_kernel_config=config, scale=1 / math.sqrt(128)
        )

    result = replay = trace_id = None
    try:
        result = operation()
        full = ttnn.to_torch(result)
        assert torch.isfinite(full).all(), "nonfinite full output"
        if trace:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            replay = operation()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            assert torch.equal(full, ttnn.to_torch(replay)), "full trace output mismatch"
        return full[..., pos, :].contiguous(), QUAL.tensor_hash(full)
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for x in (result, replay, *tensors):
            if x is not None:
                ttnn.deallocate(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diag-mode", type=int, choices=[3, 4], default=3)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--heads", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    os.environ["TT_SDPA_ACCURACY_DIAG"] = str(args.diag_mode)
    verify_diagnostic_source()
    torch.set_num_threads(16)
    cases = [
        c for c in QUAL.make_manifest() if c["distribution"] not in ("common_q", "common_k", "common_v", "constant_v")
    ]
    if args.holdout:
        template = [
            c
            for c in cases
            if c["heads"] == 5
            and c["seed"] == 1234
            and c["kv_len"] in (32768, 262144)
            and c["group"] in ("normal", "stress")
            and c["distribution"] != "scaled_low"
        ]
        cases = []
        for c in template:
            for seed in (1239, 1240):
                cases.append(dict(c, seed=seed, id=c["id"].replace("s1234", f"s{seed}")))
        assert len(cases) == 12
    cases = [
        c for c in cases if (not args.heads or c["heads"] in args.heads) and (not args.seeds or c["seed"] in args.seeds)
    ]
    old = {
        r["id"]: r
        for r in map(
            json.loads, (ROOT / "experiments/sdpa-l2/qualification-v1/accepted-results.jsonl").read_text().splitlines()
        )
        if r["mode"] == "accurate"
    }
    done = set()
    if args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text().splitlines()]
        assert all(r["diagnostic_mode"] == args.diag_mode for r in existing)
        done = {r["id"] for r in existing if r["status"] != "ERROR"}
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    device.enable_program_cache()
    try:
        with args.output.open("a") as log:
            for c in cases:
                if c["id"] in done:
                    continue
                start = time.monotonic()
                row = dict(
                    c,
                    mode="accurate",
                    diagnostic_mode=args.diag_mode,
                    q_preprocessing="none",
                    actual_math_fidelity="HiFi4",
                    fp32_streaming=True,
                    fallback_executed=False,
                    reference_precision="FP64",
                    holdout=args.holdout,
                )
                reason = QUAL.unsupported(c, "accurate")
                if reason:
                    row.update(status="UNSUPPORTED", reason=reason, fp32_streaming=False)
                else:
                    print("QUAL_START " + c["id"], flush=True)
                    q, k, v = QUAL.inputs(c)
                    hashes = [QUAL.tensor_hash(x) for x in (q, k, v)]
                    if not args.holdout:
                        assert hashes == old[c["id"]]["input_sha256"]
                    pos = QUAL.positions(c["q_len"], c["seed"])
                    if c["distribution"] in ("zero_v", "cancellation"):
                        gold = torch.zeros((1, c["heads"], len(pos), 128), dtype=torch.float64)
                    elif c["distribution"] == "uniform":
                        gold = v.double().mean(-2, keepdim=True).expand(1, c["heads"], len(pos), 128).contiguous()
                    else:
                        gold = QUAL.repro.reference(q[..., pos, :], k, v, block=4096)
                    trace = c["seed"] == (1239 if args.holdout else 1234)
                    actual, full_hash = invoke(device, q, k, v, pos, trace=trace)
                    per_head, failed = QUAL.assess(actual, gold, c, "accurate", v)
                    assert hashes == [QUAL.tensor_hash(x) for x in (q, k, v)]
                    row.update(
                        status="FAIL" if failed else "PASS",
                        per_head=per_head,
                        failed_gates=failed,
                        input_sha256=hashes,
                        positions=pos.tolist(),
                        reference_rows=len(pos),
                        full_output_sha256=full_hash,
                        sampled_output_sha256=QUAL.tensor_hash(actual),
                        trace_full_equality=trace,
                        k_chunk=QUAL.k_chunk(c, "accurate"),
                        q_chunk=128,
                    )
                    row = ADJ.adjudicate(row)
                    del q, k, v, gold, actual
                row["elapsed_s"] = time.monotonic() - start
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print("QUAL_RESULT " + json.dumps({k: row[k] for k in ("id", "status", "elapsed_s")}), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
