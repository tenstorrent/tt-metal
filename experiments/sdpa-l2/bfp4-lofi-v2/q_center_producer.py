# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-only one-Q128-block centering + HiFi4 mean-Q/original-K correction.

Producer qualification/timing, NOT an integrated LoFi attention result.
No host-generated bias or correction is uploaded. No full N-squared tensor.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

import center_mean
import q_center_preprocess as CENTER

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build(device, q, k, ncores=4, mean_mode="bf16_fpu"):
    """Return centered_Q, repeated_FP32_correction, invoke, info.

    Q BF16 [1,H,128,128], ORIGINAL K BF16 [1,H,N,128]. Correction shape
    [1,H,32,N] repeats one unique row and is UNSCALED. It must be added to
    the centered QK score before the attention scale and before row maxima.
    invoke recomputes every producer stage from the original device inputs.
    Keep invoke alive until its traces are released.
    """
    import ttnn

    assert q.dtype == k.dtype == ttnn.bfloat16
    assert tuple(q.shape) == (1, q.shape[1], 128, 128)
    assert len(k.shape) == 4 and tuple(k.shape)[:2] == tuple(q.shape)[:2]
    assert k.shape[2] > 0 and k.shape[2] % 32 == 0 and k.shape[-1] == 128
    bias, mean = center_mean.build(device, q, mean_mode)
    centered, center, actual_cores = CENTER.build(device, q, bias, ncores)
    correction = ttnn.allocate_tensor_on_device([1, q.shape[1], 32, k.shape[2]], ttnn.float32,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)

    def correct():
        # Do NOT substitute quantized K here. The omitted Q mean must interact
        # with original BF16 K at high precision; quantized K reintroduces
        # common-Q amplification of its quantization error.
        ttnn.matmul(bias, k, transpose_b=True, dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config,
            optional_output_tensor=correction)

    def invoke():
        mean()
        center()
        correct()

    invoke.bias = bias
    invoke.mean = mean
    invoke.center = center
    invoke.correct = correct
    info = dict(
        reference_input="original BF16 Q/K, never overwritten", mean=mean.precision,
        centered_q="FP32 SFPU (Q - actual BF16 bias), RNE7, BF16 storage",
        correction="actual BF16 bias @ original BF16 K^T; HiFi4, FP32 DST/output, unscaled",
        correction_shape=list(correction.shape), correction_row_repeats=32,
        correction_storage_bytes=q.shape[1] * 32 * k.shape[2] * 4,
        correction_physical_flops=2 * q.shape[1] * 32 * k.shape[2] * 128,
        correction_unique_row_flops=2 * q.shape[1] * k.shape[2] * 128,
        center_cores=actual_cores, correction_program="TTNN auto-selection; includes any required transpose",
        logical_mean_rounding_cancels="same rounded BF16 bias used in subtraction and correction",
    )
    return centered, correction, invoke, info


def metrics(actual, expected):
    delta = actual.double() - expected.double()
    norm = expected.double().norm()
    return dict(l2_pct=float(100 * delta.norm() / norm) if norm else None,
                max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(actual).all()))


def source_files():
    files = [Path(__file__).resolve(), HERE / "q_center_preprocess.py", HERE / "center_mean.py",
             HERE / "center_preprocess.py", HERE / "center_preprocess/reader.cpp",
             HERE / "center_preprocess/writer.cpp",
             ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"]
    files += sorted((HERE / "q_center_preprocess").glob("*.cpp"))
    files += sorted((HERE / "q_center_preprocess").glob("*.hpp"))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--distribution", choices=("normal", "common_q", "structured"), default="normal")
    parser.add_argument("--common-mode", type=float, default=32)
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--max-correction-l2", type=float, default=0.01)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 32 == 0 and args.heads > 0 and args.cores > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats >= 0
    assert Path(args.label).name == args.label
    output_path = HERE / (args.label + ".json")
    assert not output_path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("q_center_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    q, k, v = repro.make_inputs(args.heads, 128, args.length, 128, args.seed,
                               "common_q" if args.distribution == "common_q" else "normal", args.common_mode)
    if args.distribution == "structured":
        head = torch.arange(args.heads).reshape(1, args.heads, 1, 1)
        column = torch.arange(128).reshape(1, 1, 1, 128)
        q = (q.float() + (head + 1) * torch.sin(column.float() / 9) * args.common_mode).bfloat16()
    import ttnn

    files = source_files()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    device = ttnn.open_device(device_id=0, trace_region_size=8388608 if args.iters else 0)
    try:
        dq, dk = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k)]
        centered, correction, invoke, info = build(device, dq, dk, args.cores, args.mean_mode)
        invoke()
        actual_q, actual_c = ttnn.to_torch(centered).float(), ttnn.to_torch(correction).float()
        bias = ttnn.to_torch(invoke.bias).bfloat16()
        expected_q = CENTER.oracle(q, bias)
        q_mismatches = int((actual_q != expected_q).sum())
        assert q_mismatches == 0, f"Centered-RNE7 mismatch count {q_mismatches}"
        expected_c = bias.double() @ k.double().transpose(-1, -2)
        correction_metrics = metrics(actual_c, expected_c)
        print("CORRECTION_QUALIFICATION", json.dumps(correction_metrics), flush=True)
        assert correction_metrics["finite"]
        assert correction_metrics["l2_pct"] is not None and correction_metrics["l2_pct"] < args.max_correction_l2, correction_metrics
        assert torch.equal(actual_c, actual_c[:, :, :1].expand_as(actual_c)), "Correction rows differ"
        # Host diagnostic ONLY: exact matmul of device-produced centered Q and
        # ORIGINAL K, then device correction; not LoFi QK or integrated SDPA.
        scores = (actual_q.double() @ k.double().transpose(-1, -2) + actual_c[:, :, :1].double()) / (128**0.5)
        diagnostic_output = torch.softmax(scores, dim=-1) @ v.double()
        reference = repro.reference(q, k, v)
        diagnostic_metrics = repro.metrics(diagnostic_output, reference)
        reference_scores = q.double() @ k.double().transpose(-1, -2) / (128**0.5)
        score_metrics = metrics(scores, reference_scores)
        bias_metrics = metrics(bias[:, :, :1], q.double().mean(dim=2, keepdim=True))
        timings = {}
        for name, call in (("mean", invoke.mean), ("center_rne7", invoke.center),
                           ("correction", invoke.correct), ("producer_total", invoke)):
            times = center_mean.measure(device, call, args)
            timings[name] = dict(ms=times, median_ms=statistics.median(times) if times else None)
        assert torch.equal(actual_q, ttnn.to_torch(centered).float())
        assert torch.equal(actual_c, ttnn.to_torch(correction).float())
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in files)
        result = dict(**vars(args), **info, centered_q_mismatches=q_mismatches,
            correction_metrics=correction_metrics, bias_metrics=bias_metrics, score_metrics=score_metrics,
            host_exact_matmul_attention_diagnostic=diagnostic_metrics,
            diagnostic_warning="CPU exact centered-Q/original-K matmul + device correction; NOT integrated LoFi attention",
            timings=timings, source_sha256=hashes, trace_equal=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print("Q_CENTER_PRODUCER_RESULT", json.dumps(result), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
