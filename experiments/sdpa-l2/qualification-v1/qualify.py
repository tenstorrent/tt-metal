# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numerical qualification only. Requires the qualification-only host guards.

Full device operations; FP64 reference on every row <=2048, otherwise 512
stratified rows/head. No latency measurements. Resume never overwrites results.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)
SEEDS = [1234, 1235, 1236, 1237, 1238]
LENGTHS = [2048, 8192, 25920, 32768, 65536, 75600, 131072, 262144]


def make_manifest():
    cases = []

    def add(group, n, h, seed, dist="normal", offset=0, q_len=None):
        c = dict(group=group, kv_len=n, q_len=q_len or n, heads=h, seed=seed, distribution=dist, offset=offset)
        c["id"] = f"{group}-n{n}-q{c['q_len']}-h{h}-s{seed}-{dist}-{offset}"
        cases.append(c)

    for n in LENGTHS:
        for h in (5, 10):
            for seed in SEEDS:
                add("normal", n, h, seed)
    # Stress cross product at the streaming activation boundary and maximum
    # context, both head counts, all five seeds. Other lengths covered above.
    distributions = [("scaled_low", 0), ("scaled_qk", 0), ("outliers", 0)]
    distributions += [(d, c) for d in ("common_q", "common_k", "common_v") for c in (-32, -8, 8, 32)]
    for n in (32768, 262144):
        for h in (5, 10):
            for dist, offset in distributions:
                for seed in SEEDS:
                    add("stress", n, h, seed, dist, offset)
    for n in (2048, 32768, 262144):
        for h in (5, 10):
            for dist in ("zero_v", "constant_v", "uniform", "cancellation"):
                for seed in SEEDS:
                    add("structural", n, h, seed, dist)
    # Tile (32), Q chunk (128), K chunk (512/1024), and FP32 dispatch edges.
    for n in (31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 32767, 32769, 33280):
        for h in (5, 10):
            for seed in SEEDS:
                add("boundary", n, h, seed)
    for h in (5, 10):
        for seed in SEEDS:
            add("structural", 1, h, seed, "single_key", q_len=128)
    return cases


def k_chunk(c, mode):
    return 1024 if mode == "accurate" and c["kv_len"] % 1024 == 0 else 512


def positions(n, seed):
    if n <= 2048:
        return torch.arange(n)
    selected = set(range(16)) | set(range(n - 16, n)) | set(range(n // 2 - 8, n // 2 + 8))
    for chunk in (128, 512, 1024):
        for anchor in (chunk, (n // (2 * chunk)) * chunk, ((n - 1) // chunk) * chunk):
            selected.update(x for x in (anchor - 1, anchor, anchor + 1) if 0 <= x < n)
    selected.update(torch.linspace(0, n - 1, 256).long().tolist())
    gen = torch.Generator().manual_seed(seed + 7000)
    while len(selected) < 512:
        selected.add(int(torch.randint(n, (1,), generator=gen)))
    assert len(selected) == 512
    return torch.tensor(sorted(selected))


def inputs(c):
    dist = c["distribution"]
    base = (
        dist
        if dist in ("scaled_qk", "outliers", "common_q", "common_k", "common_v", "constant_v", "uniform")
        else "normal"
    )
    q, k, v = repro.make_inputs(c["heads"], c["q_len"], c["kv_len"], 128, c["seed"], base, c["offset"])
    if dist == "scaled_low":
        q, k = (q.float() * 0.5).bfloat16(), (k.float() * 0.5).bfloat16()
    elif dist == "zero_v":
        v.zero_()
    elif dist == "cancellation":
        q.zero_()
        v[..., 1::2, :] = -v[..., ::2, :]
    return q, k, v


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def assess(actual, gold, c, mode, v):
    actual = actual.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(gold).all():
        return [], ["nonfinite_output_or_reference"]
    stress = c["group"] == "stress"
    limit = (5.0 if stress else 3.5) if mode == "fast" else 0.5
    pcc_limit = (0.998 if stress else 0.999) if mode == "fast" else 0.99998
    per_head, failures = [], []
    for h in range(c["heads"]):
        a, e = actual[0, h], gold[0, h]
        d = a - e
        rms = e.square().mean().sqrt()
        error_norm, reference_norm = d.norm(), e.norm()
        row_norm = e.square().mean(-1).sqrt().clamp_min(0.01 * rms)
        row_error = d.square().mean(-1).sqrt()
        row_pct = torch.where(row_norm > 0, 100 * row_error / row_norm, torch.zeros_like(row_error))
        raw_row = torch.where(e.norm(dim=-1) > 0, 100 * d.norm(dim=-1) / e.norm(dim=-1), torch.zeros_like(row_error))
        ac, ec = a.flatten() - a.mean(), e.flatten() - e.mean()
        pcc = (
            float((ac @ ec) / (ac.norm() * ec.norm())) if ec.norm() > 1e-12 * reference_norm and ac.norm() > 0 else None
        )
        zero_mismatch = int(((e == 0) & (d != 0)).sum())
        rel = 100 * d[e != 0].abs() / e[e != 0].abs()
        floor = (e.bfloat16().double() - e).norm()
        m = dict(
            head=h,
            l2_pct=float(100 * error_norm / reference_norm) if reference_norm else None,
            pcc=pcc,
            row_p99_pct=float(torch.quantile(row_pct, 0.99)),
            row_max_pct=float(row_pct.max()),
            raw_row_p99_pct=float(torch.quantile(raw_row, 0.99)),
            raw_row_max_pct=float(raw_row.max()),
            zero_reference_mismatches=zero_mismatch,
            max_abs=float(d.abs().max()),
            reference_rms=float(rms),
            max_relative_pct=None if zero_mismatch else (float(rel.max()) if rel.numel() else 0.0),
            rounding_floor_l2_pct=float(100 * floor / reference_norm) if reference_norm else None,
        )
        failed = []
        if reference_norm > 0:
            if m["l2_pct"] > limit:
                failed.append("l2")
            if pcc is None and ec.norm() > 1e-12 * reference_norm:
                failed.append("pcc_undefined_actual_constant")
            elif pcc is not None and pcc < pcc_limit:
                failed.append("pcc")
            if m["row_p99_pct"] > 2 * limit:
                failed.append("row_p99")
            if m["row_max_pct"] > 4 * limit:
                failed.append("row_max")
        if c["distribution"] == "zero_v":
            if torch.count_nonzero(a):
                failed.append("zero_v_exact")
        elif c["distribution"] == "cancellation":
            # Exact paired V and Q=0 give zero. Explicit absolute budget is
            # T times V RMS / sqrt(number of keys), not an epsilon denominator.
            absolute_budget = limit / 100 * float(v[0, h].double().square().mean().sqrt()) / math.sqrt(c["kv_len"])
            m["cancellation_absolute_budget"] = absolute_budget
            if m["max_abs"] > absolute_budget:
                failed.append("cancellation_absolute")
        elif reference_norm == 0:
            if torch.count_nonzero(a):
                failed.append("zero_reference_unexpected")
        if c["distribution"] in ("constant_v", "single_key"):
            target = torch.ones_like(a) if c["distribution"] == "constant_v" else v[0, h, 0].double().expand_as(a)
            b = target.bfloat16()
            up = torch.nextafter(b, torch.full_like(b, math.inf)).double() - target
            down = target - torch.nextafter(b, torch.full_like(b, -math.inf)).double()
            ulp = torch.maximum(up, down)
            m["structural_max_ulp"] = float(((a - target).abs() / ulp).max())
            if m["structural_max_ulp"] > 1:
                failed.append("structural_one_ulp")
        if c["distribution"] == "common_v":
            residual_norm = (e - c["offset"]).norm()
            budget = limit / 100 * residual_norm + 2 * floor
            m["common_v_budget_ratio"] = float(error_norm / budget) if budget else None
            if error_norm > budget:
                failed.append("common_v_residual")
        m["failed_gates"] = failed
        failures.extend(f"head{h}:{f}" for f in failed)
        per_head.append(m)
    return per_head, failures


def run_device(device, q, k, v, pos, mode, trace=False, pad_value=0.0):
    fp32 = mode == "accurate"
    device_q = repro.preprocess_query(q, 6, 1.0027, True) if fp32 else q
    tensors = [
        ttnn.from_torch(
            x,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=pad_value,
        )
        for x in (device_q, k, v)
    ]
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=k_chunk({"kv_len": k.shape[-2]}, mode),
        exp_approx_mode=True,
    )
    result = traced = None
    trace_id = None
    try:

        def invoke():
            return ttnn.transformer.scaled_dot_product_attention(
                *tensors,
                is_causal=False,
                program_config=program,
                compute_kernel_config=config,
                scale=1 / (math.sqrt(128) * (1.0027 if fp32 else 1)),
            )

        result = invoke()
        full = ttnn.to_torch(result)
        if not torch.isfinite(full).all():
            raise AssertionError("nonfinite_full_output")
        actual = full[..., pos, :].contiguous()
        full_hash = tensor_hash(full)
        if trace:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            traced = invoke()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            replay = ttnn.to_torch(traced)
            if not torch.equal(replay, full):
                raise AssertionError("trace_full_output_mismatch")
        return actual, full_hash
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for x in (result, traced, *tensors):
            if x is not None:
                ttnn.deallocate(x)


def unsupported(c, mode):
    if mode == "fast":
        return None
    if c["q_len"] % 128 or c["kv_len"] % 512:
        return "generated_padding_mask: current FP32 streaming excludes padded Q/K"
    if c["kv_len"] < 32768:
        return "K length below FP32 streaming minimum 32768"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", choices=["normal", "stress", "structural", "boundary"])
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--distributions", nargs="+")
    parser.add_argument("--modes", nargs="+", default=["fast", "accurate"], choices=["fast", "accurate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.self_test:
        repro.check_reference()
        c = dict(heads=1, group="normal", distribution="normal", kv_len=32)
        e = torch.randn(1, 1, 32, 128, dtype=torch.float64)
        _, failed = assess(e, e, c, "accurate", e)
        assert not failed
        _, failed = assess(e * 1.01, e, c, "accurate", e)
        assert "head0:l2" in failed
        c["distribution"] = "zero_v"
        _, failed = assess(torch.zeros_like(e), torch.zeros_like(e), c, "accurate", e)
        assert not failed
        _, failed = assess(torch.ones_like(e), torch.zeros_like(e), c, "accurate", e)
        assert "head0:zero_v_exact" in failed
        for n in LENGTHS:
            p = positions(n, 1234)
            assert len(p) == (n if n <= 2048 else 512)
        print("SELF_TEST_PASS", flush=True)
        return
    cases = make_manifest()
    for arg, key in (
        (args.group, "group"),
        (args.lengths, "kv_len"),
        (args.seeds, "seed"),
        (args.heads, "heads"),
        (args.distributions, "distribution"),
    ):
        if arg:
            cases = [c for c in cases if c[key] in arg]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        with args.output.open("x") as f:
            json.dump(cases, f, indent=2)
        print(f"MANIFEST {len(cases)} input cases, {len(cases) * len(args.modes)} mode cases")
        return
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "ERROR":
                done.add((row["id"], row["mode"]))
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    device.enable_program_cache()
    try:
        with args.output.open("a") as log:

            def emit(row):
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(
                    "QUAL_RESULT "
                    + json.dumps({k: row[k] for k in ("id", "mode", "status", "failed_gates") if k in row}),
                    flush=True,
                )

            for c in cases:
                pending = [mode for mode in args.modes if (c["id"], mode) not in done]
                if not pending:
                    continue
                supported = []
                for mode in pending:
                    reason = unsupported(c, mode)
                    if reason:
                        emit(dict(c, mode=mode, status="UNSUPPORTED", reason=reason, fallback_executed=False))
                    else:
                        supported.append(mode)
                if not supported:
                    continue
                started = time.monotonic()
                print("QUAL_START " + c["id"], flush=True)
                q, k, v = inputs(c)
                input_hashes = [tensor_hash(x) for x in (q, k, v)]
                pos = positions(c["q_len"], c["seed"])
                dist = c["distribution"]
                if dist in ("zero_v", "cancellation"):
                    gold = torch.zeros((1, c["heads"], len(pos), 128), dtype=torch.float64)
                elif dist == "constant_v":
                    gold = torch.ones((1, c["heads"], len(pos), 128), dtype=torch.float64)
                elif dist == "single_key":
                    gold = v[..., :1, :].double().expand(1, c["heads"], len(pos), 128).contiguous()
                elif dist == "uniform":
                    gold = v.double().mean(-2, keepdim=True).expand(1, c["heads"], len(pos), 128).contiguous()
                else:
                    gold = repro.reference(q[..., pos, :], k, v, block=1024)
                if dist == "common_k":
                    centered = k.double() - c["offset"]
                    centered_bf16 = centered.bfloat16()
                    centered_exact = torch.equal(centered_bf16.double(), centered)
                    if c["seed"] == SEEDS[0]:
                        identity_gold = repro.reference(q[..., pos, :], centered, v, block=1024)
                        torch.testing.assert_close(identity_gold, gold, rtol=1e-10, atol=1e-11)
                    centered_gold = (
                        gold if centered_exact else repro.reference(q[..., pos, :], centered_bf16, v, block=1024)
                    )
                for mode in supported:
                    row = dict(
                        c,
                        mode=mode,
                        harness_revision="v1.1_centering_quantization",
                        reference_rows=len(pos),
                        positions=pos.tolist(),
                        input_sha256=input_hashes,
                        full_device_operation=True,
                        q_preprocessing="bitceil6_scale1.0027" if mode == "accurate" else "none",
                        k_chunk=k_chunk(c, mode),
                        q_chunk=128,
                        fp32_streaming=mode == "accurate",
                        bf16_compensated=mode == "fast",
                    )
                    try:
                        # Trace correctness on the first seed of every geometry/distribution.
                        trace = c["seed"] == SEEDS[0]
                        actual, full_hash = run_device(device, q, k, v, pos, mode, trace=trace)
                        per_head, failed = assess(actual, gold, c, mode, v)
                        row.update(
                            per_head=per_head,
                            failed_gates=failed,
                            trace_full_equality=trace,
                            full_output_sha256=full_hash,
                            sampled_output_sha256=tensor_hash(actual),
                        )
                        # Perturb physical tile padding, not logical tokens; compare entire outputs.
                        if c["group"] == "boundary" and c["kv_len"] % 32:
                            _, poisoned_hash = run_device(device, q, k, v, pos, mode, pad_value=32.0)
                            row["padding_full_output_equal"] = poisoned_hash == full_hash
                            if poisoned_hash != full_hash:
                                failed.append("padding_invariance")
                        if dist == "common_k":
                            centered_actual, _ = run_device(device, q, centered_bf16, v, pos, mode)
                            center_metrics, center_failures = assess(centered_actual, centered_gold, c, mode, v)
                            row["common_k_centered_per_head"] = center_metrics
                            row["common_k_centering_exact_bf16"] = centered_exact
                            row["common_k_reference_shift_l2_pct"] = float(
                                100 * (gold - centered_gold).norm() / gold.norm()
                            )
                            failed.extend("centered:" + failure for failure in center_failures)
                            pair_errors = []
                            for head in range(c["heads"]):
                                pair_pct = float(
                                    100
                                    * (
                                        (actual[0, head].double() - gold[0, head])
                                        - (centered_actual[0, head].double() - centered_gold[0, head])
                                    ).norm()
                                    / gold[0, head].norm()
                                )
                                pair_errors.append(pair_pct)
                                pair_limit = (5.0 if mode == "fast" else 0.5) * (
                                    1 + float(centered_gold[0, head].norm() / gold[0, head].norm())
                                )
                                if pair_pct > pair_limit:
                                    failed.append(f"head{head}:common_k_pair")
                            row["common_k_pair_l2_pct"] = pair_errors
                        row["status"] = "FAIL" if failed else "PASS"
                    except Exception as exc:
                        message = str(exc)
                        row.update(
                            status="UNSUPPORTED" if "QUALIFICATION_UNSUPPORTED" in message else "ERROR",
                            error=message,
                            failed_gates=[],
                        )
                        if row["status"] == "UNSUPPORTED":
                            row["fallback_executed"] = False
                        emit(row)
                        if row["status"] == "ERROR":
                            raise  # Do not continue using a potentially faulted device.
                    else:
                        emit(row)
                assert [tensor_hash(x) for x in (q, k, v)] == input_hashes, "caller input mutation"
                print(f"QUAL_DONE {c['id']} wall_seconds={time.monotonic() - started:.2f}", flush=True)
                del q, k, v, gold
                if dist == "common_k":
                    del centered, centered_bf16, centered_gold
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
