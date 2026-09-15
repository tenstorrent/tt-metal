# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone BF16 SDPA accuracy experiment (run on an allocated device).

PYTHONPATH=ttnn:tools:. python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py

Relative L2 = ||actual-reference||_2 / ||reference||_2, not squared error.
Reference consumes the SAME already-rounded BF16 inputs, with FP64 arithmetic.
Short Q / long KV is noncausal attention, not a full causal Galaxy benchmark.
The exact_exp variant requests exp_approx_mode=False; unmodified main still
uses approximate exp for the logits. See experiments/sdpa-l2/REPORT.md.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import ttnn


def metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise AssertionError(
            f"SDPA output or reference contains nonfinite values: output {int((~torch.isfinite(a)).sum())}/{a.numel()}, "
            f"reference {int((~torch.isfinite(e)).sum())}/{e.numel()}; output first 16={a[:16].tolist()}"
        )
    if e.norm() == 0:
        raise AssertionError("Relative L2 is undefined for a zero-norm reference")
    delta = a - e
    ac, ec = a - a.mean(), e - e.mean()
    denom = ac.norm() * ec.norm()
    gain = (a @ e) / (e @ e)
    row_l2 = 100 * (actual.double() - expected).norm(dim=-1) / expected.norm(dim=-1)
    # No epsilon/clamping in the requested elementwise relative error. Equal
    # zeros contribute zero; a nonzero output at an exact reference zero is
    # unbounded (JSON null plus an explicit count, rather than nonstandard Inf).
    zero = e == 0
    zero_mismatches = int((zero & (delta != 0)).sum().item())
    relative = torch.zeros_like(e)
    relative[~zero] = 100 * delta[~zero].abs() / e[~zero].abs()
    worst = int((zero & (delta != 0)).nonzero()[0].item()) if zero_mismatches else int(relative.argmax().item())
    rms = e.square().mean().sqrt()
    significant = e.abs() >= 0.01 * rms
    return {
        "l2_pct": (100 * delta.norm() / e.norm()).item(),
        "max_relative_error_pct": None if zero_mismatches else relative[worst].item(),
        "zero_reference_mismatch_count": zero_mismatches,
        "max_relative_reference": e[worst].item(),
        "max_relative_actual": a[worst].item(),
        "reference_rms": rms.item(),
        "max_relative_error_above_1pct_rms_pct": relative[significant].max().item(),
        "pcc": (ac @ ec / denom).item() if ec.norm() > 1e-12 * e.norm() and ac.norm() > 0 else None,
        "gain": gain.item(),
        "gain_corrected_l2_pct": (100 * (a - gain * e).norm() / e.norm()).item(),
        "max_abs": delta.abs().max().item(),
        "row_l2_pct_median": row_l2.median().item(),
        "row_l2_pct_p95": torch.quantile(row_l2.flatten(), 0.95).item(),
    }


def reference(q, k, v, block=4096, causal=False, positions=None):
    """Stable online softmax in FP64, bounded memory even at 256K keys."""
    q = q.double()
    m = torch.full((*q.shape[:-1], 1), -torch.inf, dtype=torch.float64)
    total = torch.zeros_like(m)
    out = torch.zeros_like(q)
    for start in range(0, k.shape[-2], block):
        scores = q @ k[..., start : start + block, :].double().transpose(-1, -2) / math.sqrt(q.shape[-1])
        if causal:
            scores.masked_fill_(torch.arange(start, start + scores.shape[-1])[None, :] > positions[:, None], -torch.inf)
        new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
        correction = torch.exp(m - new_m)
        weights = torch.exp(scores - new_m)
        out = out * correction + weights @ v[..., start : start + block, :].double()
        total = total * correction + weights.sum(-1, keepdim=True)
        m = new_m
    return out / total


def make_inputs(heads, q_len, kv_len, dim, seed, distribution, common_mode=32.0):
    tensors = []
    for index, length in enumerate((q_len, kv_len, kv_len)):
        gen = torch.Generator().manual_seed(seed + index * 1000)
        shape = (1, heads, length, dim)
        x = torch.randn(shape, generator=gen)
        if distribution == "outliers":
            x += 10 * torch.randn(shape, generator=gen) * (torch.rand(shape, generator=gen) < 0.001)
        if distribution == "scaled_qk" and index < 2:
            x *= 2
        if distribution == "biased_v" and index == 2:
            x += 1
        if distribution == ("common_q", "common_k", "common_v")[index]:
            x += common_mode
        if distribution in ("uniform", "uniform_constant_v") and index == 0:
            x.zero_()
        if distribution in ("constant_v", "uniform_constant_v") and index == 2:
            x.fill_(1)
        tensors.append(x.bfloat16())
    return tensors


def preprocess_query(q, fraction_bits, prescale, bitceil=False):
    if bitceil:
        # Equivalent for finite normal BF16 and signed zero in the supported scale interval.
        # The synthetic distributions in this repro contain no subnormals or non-finite values.
        return ((q.view(torch.int16) + 1) & -2).view(torch.bfloat16)
    if fraction_bits is None:
        return q
    shift = 23 - fraction_bits
    bits = (q.float() * prescale).view(torch.int32)
    rounded = (bits + (1 << (shift - 1)) - 1 + ((bits >> shift) & 1)) & ~((1 << shift) - 1)
    return rounded.view(torch.float32).bfloat16()


def check_reference():
    q, k, v = make_inputs(2, 32, 96, 32, 17, "normal")
    positions = torch.arange(64, 96)
    for causal in (False, True):
        scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(32)
        if causal:
            scores.masked_fill_(torch.arange(96)[None, :] > positions[:, None], -torch.inf)
        dense = scores.softmax(-1) @ v.double()
        torch.testing.assert_close(
            reference(q, k, v, block=32, causal=causal, positions=positions), dense, rtol=1e-12, atol=1e-12
        )
    scale_error = metrics(dense * 2, dense)
    # Check the reference's common-mode identities using the already-quantized
    # shifted inputs; this avoids conflating input BF16 rounding with SDPA error.
    shifted_k = (k.float() + 32).bfloat16()
    torch.testing.assert_close(
        reference(q, shifted_k, v), reference(q, shifted_k.double() - 32, v), rtol=1e-12, atol=1e-12
    )
    shifted_v = (v.float() + 32).bfloat16()
    torch.testing.assert_close(
        reference(q, k, shifted_v) - 32, reference(q, k, shifted_v.double() - 32), rtol=1e-12, atol=1e-12
    )
    assert abs(scale_error["l2_pct"] - 100) < 1e-10 and abs(scale_error["pcc"] - 1) < 1e-10
    assert abs(scale_error["max_relative_error_pct"] - 100) < 1e-10
    zero_check = metrics(torch.tensor([[0.0, 2.0]]), torch.tensor([[0.0, 1.0]]))
    assert zero_check["max_relative_error_pct"] == 100 and zero_check["zero_reference_mismatch_count"] == 0
    zero_check = metrics(torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 1.0]]))
    assert zero_check["max_relative_error_pct"] is None and zero_check["zero_reference_mismatch_count"] == 1
    # Exhaust all finite normal BF16 bit patterns, plus signed zero.
    bits = torch.arange(65536, dtype=torch.int32)
    exponent = bits & 0x7F80
    selected = ((exponent != 0) & (exponent != 0x7F80)) | ((bits & 0x7FFF) == 0)
    all_normal = bits[selected].to(torch.int16).view(torch.bfloat16)
    for prescale in (1.001, 1.002, 1.0027, 1.0038):
        expected = preprocess_query(all_normal, 6, prescale).view(torch.int16)
        actual = preprocess_query(all_normal, 6, prescale, bitceil=True).view(torch.int16)
        assert torch.equal(actual, expected), "Bitwise preprocessing differs from float RNE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[4096, 32768, 131072, 262144])
    parser.add_argument("--q-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--q-round-bits", type=int, choices=[6, 7], help="RNE Q fraction bits; reference retains original BF16 Q"
    )
    parser.add_argument(
        "--q-prescale", type=float, default=1.0, help="Scale Q before rounding; compensate in attention scale"
    )
    parser.add_argument(
        "--q-bitceil",
        action="store_true",
        help="Fast BF16 bitwise implementation for normal Q and 1 < prescale < 1 + 1/256",
    )
    parser.add_argument(
        "--attention-scale-factor",
        type=float,
        default=1.0,
        help="Experimental temperature correction; reference unchanged",
    )
    parser.add_argument("--q-chunk", type=int, default=128)
    parser.add_argument("--grid", nargs=2, type=int, metavar=("X", "Y"), help="Optional compute grid")
    parser.add_argument(
        "--record-fp32-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Metadata only: record an FP32-streaming source build; does not select the kernel",
    )
    parser.add_argument(
        "--device-query-rows",
        type=int,
        default=0,
        help="Noncausal short-Q device length; --q-len still controls reference sampling",
    )
    parser.add_argument("--k-chunks", nargs="+", type=int, default=[512])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["hifi2", "hifi4", "exact_exp", "fp32"],
        choices=["hifi2", "hifi4", "exact_exp", "hifi4_exact", "fp32", "fp32_hifi2", "fp32_hifi4"],
    )
    parser.add_argument(
        "--distribution",
        choices=[
            "normal",
            "outliers",
            "uniform",
            "constant_v",
            "uniform_constant_v",
            "scaled_qk",
            "biased_v",
            "common_q",
            "common_k",
            "common_v",
        ],
        default="normal",
    )
    parser.add_argument("--common-mode", type=float, default=32.0, help="Offset for common_q/common_k/common_v")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--benchmark-iters", type=int, default=0, help="Warmed trace replays, excluding compilation and transfers"
    )
    parser.add_argument("--benchmark-warmup", type=int, default=2, help="Unmeasured trace replays before timing")
    parser.add_argument(
        "--causal", action="store_true", help="Full square causal attention; reference samples up to q-len rows"
    )
    parser.add_argument(
        "--full", action="store_true", help="Full square attention, including noncausal; reference samples q-len rows"
    )
    parser.add_argument("--query-sampling", choices=["tail", "spread"], default="tail")
    parser.add_argument(
        "--sampled-device",
        action="store_true",
        help="Noncausal accuracy diagnostic: generate full inputs but execute only the reference Q rows; not full-prefill timing",
    )
    parser.add_argument("--label", default="unmodified-main")
    parser.add_argument("--per-head-metrics", action="store_true")
    parser.add_argument("--check-full-output", action="store_true")
    parser.add_argument("--check-reference-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sdpa-l2.jsonl"))
    parser.add_argument(
        "--max-l2-pct", type=float, help="Optional accuracy acceptance threshold (failure after all cases)"
    )
    args = parser.parse_args()
    if args.sampled_device and (not args.full or args.causal):
        parser.error("--sampled-device requires noncausal --full")
    if args.device_query_rows and (args.full or args.causal or args.device_query_rows < args.q_len):
        parser.error("--device-query-rows requires short noncausal Q and at least --q-len rows")
    if args.q_prescale != 1.0 and args.q_round_bits is None:
        parser.error("--q-prescale requires --q-round-bits")
    if args.q_bitceil and not (args.q_round_bits == 6 and 1.0 < args.q_prescale < 1.0 + 1.0 / 256):
        parser.error("--q-bitceil requires --q-round-bits 6 and 1 < --q-prescale < 1 + 1/256")
    torch.set_num_threads(args.threads)
    check_reference()
    if args.check_reference_only:
        print("FP64 blockwise reference agrees with dense softmax; PCC/L2 scale-error check passed.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.benchmark_iters else 0)
    failures = []
    try:
        with args.output.open("w") as log:
            for length in args.kv_lens:
                q_len = length if args.causal or args.full else (args.device_query_rows or args.q_len)
                q, k, v = make_inputs(
                    args.heads, q_len, length, args.dim, args.seed, args.distribution, args.common_mode
                )
                if (args.causal or args.full) and args.query_sampling == "spread":
                    positions = torch.linspace(0, q_len - 1, min(q_len, args.q_len)).long()
                else:
                    positions = torch.arange(max(0, q_len - args.q_len), q_len)
                gold = reference(q[..., positions, :], k, v, causal=args.causal, positions=positions)
                floor = metrics(gold.bfloat16(), gold)["l2_pct"]
                preprocess_start = time.perf_counter()
                device_q = preprocess_query(q, args.q_round_bits, args.q_prescale, args.q_bitceil)
                preprocessing_seconds = time.perf_counter() - preprocess_start
                if args.sampled_device:
                    device_q = device_q[..., positions, :].contiguous()
                output_positions = torch.arange(len(positions)) if args.sampled_device else positions
                tq, tk, tv = [
                    ttnn.from_torch(
                        x,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for x in (device_q, k, v)
                ]
                try:
                    for chunk in args.k_chunks:
                        for variant in args.variants:
                            fp32 = variant.startswith("fp32")
                            fidelity = "HiFi4" if variant in ("hifi4", "hifi4_exact", "fp32", "fp32_hifi4") else "HiFi2"
                            approx = variant not in ("exact_exp", "hifi4_exact", "fp32")
                            config = ttnn.WormholeComputeKernelConfig(
                                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                                math_approx_mode=approx,
                                fp32_dest_acc_en=fp32,
                                packer_l1_acc=False,
                            )
                            program = ttnn.SDPAProgramConfig(
                                compute_with_storage_grid_size=(
                                    ttnn.CoreCoord(*args.grid) if args.grid else device.compute_with_storage_grid_size()
                                ),
                                q_chunk_size=args.q_chunk,
                                k_chunk_size=chunk,
                                exp_approx_mode=approx,
                            )
                            start = time.monotonic()
                            result = ttnn.transformer.scaled_dot_product_attention(
                                tq,
                                tk,
                                tv,
                                is_causal=args.causal,
                                program_config=program,
                                compute_kernel_config=config,
                                scale=args.attention_scale_factor / (math.sqrt(args.dim) * args.q_prescale),
                            )
                            actual_full = ttnn.to_torch(result)
                            if args.check_full_output:
                                assert torch.isfinite(actual_full).all(), "nonfinite full output"
                            actual = actual_full[..., output_positions, :]
                            elapsed = time.monotonic() - start
                            replay_ms = []
                            if args.benchmark_iters:
                                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                                traced_result = ttnn.transformer.scaled_dot_product_attention(
                                    tq,
                                    tk,
                                    tv,
                                    is_causal=args.causal,
                                    program_config=program,
                                    compute_kernel_config=config,
                                    scale=args.attention_scale_factor / (math.sqrt(args.dim) * args.q_prescale),
                                )
                                ttnn.end_trace_capture(device, trace_id, cq_id=0)
                                try:
                                    for iteration in range(args.benchmark_iters + args.benchmark_warmup):
                                        start = time.perf_counter()
                                        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                                        if iteration >= args.benchmark_warmup:
                                            replay_ms.append(1000 * (time.perf_counter() - start))
                                    traced_full = ttnn.to_torch(traced_result)
                                    if args.check_full_output:
                                        assert torch.equal(traced_full, actual_full), "full trace output mismatch"
                                    else:
                                        torch.testing.assert_close(
                                            traced_full[..., output_positions, :], actual, rtol=0, atol=0
                                        )
                                    del traced_full
                                finally:
                                    ttnn.release_trace(device, trace_id)
                                    ttnn.deallocate(traced_result)
                            row = dict(
                                label=args.label,
                                query_sampling=args.query_sampling,
                                kv_len=length,
                                q_len=device_q.shape[-2],
                                source_q_len=q_len,
                                sampled_device=args.sampled_device,
                                benchmark_warmup=args.benchmark_warmup,
                                reference_rows=len(positions),
                                full_output_finite_checked=args.check_full_output,
                                full_trace_equality=args.check_full_output and bool(args.benchmark_iters),
                                heads=args.heads,
                                dim=args.dim,
                                q_round_bits=args.q_round_bits,
                                q_bitceil=args.q_bitceil,
                                q_prescale=args.q_prescale,
                                attention_scale_factor=args.attention_scale_factor,
                                preprocessing_seconds=preprocessing_seconds,
                                seed=args.seed,
                                distribution=args.distribution,
                                common_mode=args.common_mode if args.distribution.startswith("common_") else None,
                                causal=args.causal,
                                q_chunk=args.q_chunk,
                                k_chunk=chunk,
                                kv_chunks=math.ceil(length / chunk),
                                variant=variant,
                                math_fidelity=fidelity,
                                accuracy_diagnostic_mode=os.environ.get("TT_SDPA_ACCURACY_DIAG"),
                                fp32_sub_batch=os.environ.get("TT_SDPA_FP32_SUB_BATCH", "1"),
                                fp32_fused_exp="TT_SDPA_FP32_FUSED_EXP" in os.environ,
                                fp32_cache_max="TT_SDPA_FP32_CACHE_MAX" in os.environ,
                                fp32_shadow_max="TT_SDPA_FP32_SHADOW_MAX" in os.environ,
                                fp32_paired_unpack="TT_SDPA_FP32_PAIRED_UNPACK" in os.environ,
                                fp32_paired_pack="TT_SDPA_FP32_PAIRED_PACK" in os.environ,
                                fp32_pipeline="TT_SDPA_FP32_PIPELINE" in os.environ,
                                fp32_l1_sub="TT_SDPA_FP32_L1_SUB" in os.environ,
                                fp32_l1_macro="TT_SDPA_FP32_L1_MACRO" in os.environ,
                                fp32_refine_macro="TT_SDPA_FP32_REFINE_MACRO" in os.environ,
                                fp32_l1_repeat="TT_SDPA_FP32_L1_REPEAT" in os.environ,
                                fp32_qk_width=os.environ.get("TT_SDPA_FP32_QK_WIDTH", "4"),
                                fp32_qk_height=os.environ.get("TT_SDPA_FP32_QK_HEIGHT", "1"),
                                fp32_reuse_exp="TT_SDPA_FP32_REUSE_EXP" in os.environ,
                                fp32_extra_const="TT_SDPA_FP32_EXTRA_CONST" in os.environ,
                                denominator_phases=os.environ.get("TT_SDPA_DENOM_PHASES", "4"),
                                experimental_math_fidelity_override=(
                                    "HiFi4" if "TT_SDPA_ACCURACY_DIAG" in os.environ else None
                                ),
                                math_approx_mode=approx,
                                exp_approx_mode=approx,
                                fp32_dest_acc_en=fp32,
                                packer_l1_acc=False,
                                streaming=(
                                    True
                                    if "TT_SDPA_ACCURACY_DIAG" in os.environ
                                    else (args.record_fp32_streaming if fp32 else True)
                                ),
                                compute_grid=args.grid
                                or [
                                    device.compute_with_storage_grid_size().x,
                                    device.compute_with_storage_grid_size().y,
                                ],
                                bf16_rounding_l2_pct=floor,
                                seconds_including_compile=elapsed,
                                trace_replay_ms=replay_ms,
                                trace_median_ms=statistics.median(replay_ms) if replay_ms else None,
                                sampled_output_sha256=hashlib.sha256(
                                    actual.float().contiguous().numpy().tobytes()
                                ).hexdigest(),
                                full_output_sha256=(
                                    hashlib.sha256(
                                        actual_full.contiguous().view(torch.uint16).numpy().tobytes()
                                    ).hexdigest()
                                    if args.check_full_output
                                    else None
                                ),
                                **metrics(actual, gold),
                            )
                            if args.per_head_metrics:
                                row["per_head"] = [metrics(actual[:, h], gold[:, h]) for h in range(args.heads)]
                            if args.distribution == "common_v":
                                # A large output DC level can hide errors in the
                                # small attention signal. Keep the same error
                                # numerator, but also normalize by gold - DC.
                                residual_norm = (gold - args.common_mode).norm()
                                row["l2_relative_to_v_residual_pct"] = (
                                    (100 * (actual.double() - gold).norm() / residual_norm).item()
                                    if residual_norm > 0
                                    else None
                                )
                            print(json.dumps(row), flush=True)
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            ttnn.deallocate(result)
                            if args.max_l2_pct is not None and row["l2_pct"] > args.max_l2_pct:
                                failures.append(row)
                finally:
                    for tensor in (tq, tk, tv):
                        ttnn.deallocate(tensor)
    finally:
        ttnn.close_device(device)
    if failures:
        raise AssertionError(f"{len(failures)} cases exceed relative L2 {args.max_l2_pct}% (see {args.output})")


if __name__ == "__main__":
    main()
