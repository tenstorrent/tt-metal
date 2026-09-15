# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-derived common-V metrics from stored device L2 and original references.

No device execution, saved device-output tensor loading, or new replay checks.
The same per-head/channel mean(original BF16 V) is conceptually subtracted
from actual and reference, preserving their difference exactly.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPRO = ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
VARIANTS = ("lofi_native", "lofi_lut", "hi2_native", "hi2_lut")


def load_exact_cpu_functions():
    # Compile the original, unmodified function ASTs, excluding module imports
    # and the device main. This avoids importing TTNN for a CPU-only analysis.
    source = ast.parse(REPRO.read_text(), filename=str(REPRO))
    names = {"make_inputs", "reference", "metrics"}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPRO), "exec"), namespace)
    return namespace


def record_paths():
    return [
        HERE
        / (
            f"exp-stress-{variant}-common_v-s{seed}-v2.json"
            if length == 1024
            else f"exp-stress-long-{variant}-common_v-s{seed}-v1.json"
        )
        for length in (1024, 32768)
        for seed in (1240, 1241)
        for variant in VARIANTS
    ]


def source_pins(paths):
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(), REPRO, *paths]
    }


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = record_paths()
    records = [(path, json.loads(path.read_text())) for path in paths]
    pins = source_pins(paths)
    functions = load_exact_cpu_functions()
    repro_key = str(REPRO.relative_to(ROOT))
    for path, record in records:
        assert record["source_sha256"][repro_key] == pins[repro_key], path
        assert record["distribution"] == "common_v" and record["heads"] == 2
        assert record["fp32_dst"] and record["sample_rows"] == (1024 if record["length"] == 1024 else 128)
    started = time.perf_counter()
    with (HERE / (args.label + ".jsonl")).open("x") as stream:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                execution="CPU-derived metrics from stored hardware norms; no device output tensors",
                args=vars(args),
                source_sha256=pins,
                python=sys.version,
                torch_version=torch.__version__,
                platform=platform.platform(),
                cpu_threads=4,
                interop_threads=1,
                cpu_function_loading="Exact pinned REPRO AST functions make_inputs/reference/metrics; no device-module imports",
                reference_rms_tolerance=dict(rel_tol=1e-12, abs_tol=1e-12),
                assumptions=[
                    "Original BF16 QKV and original FP64 reference, no reference replacement",
                    "Common center is FP64 token mean of original BF16 V, per head and value channel",
                    "Actual centered L2 is algebraically derived from stored L2; no actual output reconstructed",
                    "BF16 nearest-output floor is independently regenerated from the original FP64 reference",
                    "Error/floor norm ratio is not an orthogonal decomposition of kernel and output-rounding error",
                    "HiFi2 original input hashes must match; LoFi records without hashes are linked by same generator/seed/shape/rows/reference RMS only",
                    "No device preprocessing, output finiteness, trace replay or device-input integrity is newly verified here",
                ],
            )
        )
        count = 0
        for length in (1024, 32768):
            for seed in (1240, 1241):
                inputs = functions["make_inputs"](2, length, length, 128, seed, "common_v")
                hashes = [tensor_hash(value) for value in inputs]
                rows = torch.linspace(0, length - 1, 1024 if length == 1024 else 128).long().unique()
                reference = functions["reference"](inputs[0][..., rows, :], inputs[1], inputs[2])
                mean_v = inputs[2].double().mean(dim=-2, keepdim=True)
                residual = reference - mean_v
                ref_norm, residual_norm = float(reference.norm()), float(residual.norm())
                ref_rms = float(reference.square().mean().sqrt())
                rounded = reference.bfloat16()
                floor_error_norm = float((rounded.double() - reference).norm())
                floor_metrics = functions["metrics"](rounded, reference)
                # The centered reference is also direct attention on centered V,
                # providing an independent algebraic oracle for this denominator.
                centered_direct = functions["reference"](
                    inputs[0][..., rows, :], inputs[1], inputs[2].double() - mean_v
                )
                assert torch.allclose(centered_direct, residual, rtol=1e-9, atol=1e-12)
                assert ref_norm > 0 and residual_norm > 0 and floor_error_norm > 0
                group = [
                    (path, record) for path, record in records if record["length"] == length and record["seed"] == seed
                ]
                assert len(group) == 4
                matched_hi2 = 0
                for path, record in group:
                    assert record["sampled_query_rows"] == rows.tolist()
                    assert math.isclose(record["accuracy"]["reference_rms"], ref_rms, rel_tol=1e-12, abs_tol=1e-12)
                    available_hashes = record.get("original_input_sha256")
                    if record.get("variant") == "hi2_fp32_bf16":
                        assert available_hashes == hashes, path
                        matched_hi2 += 1
                    elif available_hashes is not None:
                        assert available_hashes == hashes, path
                    accuracy = record["accuracy"]
                    error_norm = accuracy["l2_pct"] * ref_norm / 100.0
                    assert error_norm >= floor_error_norm * (
                        1 - 1e-9
                    ), "Stored BF16-output error below nearest-output floor"
                    assert source_pins(paths) == pins and [tensor_hash(x) for x in inputs] == hashes
                    emit(
                        dict(
                            kind="derived_result",
                            evidence=str(path.relative_to(ROOT)),
                            execution="CPU-derived, not a new device result",
                            length=length,
                            heads=2,
                            seed=seed,
                            sample_rows=len(rows),
                            sampled_query_rows=rows.tolist(),
                            variant=("hi2" if record.get("variant") == "hi2_fp32_bf16" else "lofi")
                            + ("_lut" if record["lut_exp"] else "_native"),
                            original_input_sha256=hashes,
                            original_reference_sha256=tensor_hash(reference),
                            reference_norm=ref_norm,
                            reference_rms=ref_rms,
                            residual_reference_norm=residual_norm,
                            residual_reference_rms=float(residual.square().mean().sqrt()),
                            center_mean_original_v_sha256=tensor_hash(mean_v),
                            stored_original_l2_pct=accuracy["l2_pct"],
                            stored_original_pcc=accuracy["pcc"],
                            derived_actual_error_norm=error_norm,
                            derived_actual_centered_l2_pct=100.0 * error_norm / residual_norm,
                            bf16_nearest_floor_original_l2_pct=floor_metrics["l2_pct"],
                            bf16_nearest_floor_original_pcc=floor_metrics["pcc"],
                            bf16_nearest_floor_error_norm=floor_error_norm,
                            bf16_nearest_floor_centered_l2_pct=100.0 * floor_error_norm / residual_norm,
                            actual_error_norm_over_floor=error_norm / floor_error_norm,
                            stored_reference_rms_matches=True,
                            recorded_original_input_hash_matches=True if available_hashes is not None else None,
                            centered_reference_algebra_passed=True,
                            centered_reference_algebra_max_abs=float((centered_direct - residual).abs().max()),
                            original_inputs_unchanged=True,
                            sources_and_evidence_unchanged=True,
                            new_device_output_or_replay_verification=False,
                        )
                    )
                    count += 1
                assert matched_hi2 == 2
        assert count == 16
        emit(
            dict(
                kind="complete",
                cases=count,
                sources_and_evidence_unchanged=source_pins(paths) == pins,
                cpu_wall_seconds=time.perf_counter() - started,
                device_jobs=0,
            )
        )


if __name__ == "__main__":
    main()
