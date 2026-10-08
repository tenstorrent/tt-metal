# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated accurate partial-tile attention experiment; native sources stay fixed."""

import hashlib
import json
import math
import statistics
from pathlib import Path

COMPUTE = Path("ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp")
COMMON = Path("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp")
PINNED = {
    str(COMPUTE): "c865353d07a55967ca6959e3b5efda744312009e8b998d7425edf84d4e18f23e",
    str(COMMON): "6ae905e3619ffc6cb940fc88985d64c798aed732eaadc67da1b593c45d50a2cd",
}
VARIANTS = ("native", "accurate_partial")
CONTEXTS = (1024, 32768, 131072, 262016)
HARDWARE_CASES = (
    (32768, 8),
    (32768, 16),
    (32768, 32),
    (16384, 8),
    (16384, 16),
    (16384, 32),
    (131072, 8),
    (131072, 16),
    (262016, 4),
    (262016, 8),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Missing or ambiguous partial-tile source anchor")
    return source.replace(before, after, 1)


def patch_common(source):
    # Restrict edits to this one helper; other exponentials/reduction stages
    # retain the pinned arithmetic. Removing the forced approximate branch alone
    # is insufficient: the full-tile scalar API also ignores vector_mode.
    begin = source.index("void sub_exp_block_bcast_cols_inplace(")
    end = source.index("\n}", begin) + 2
    body = source[begin:end]
    condition = "if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC)"
    if body.count(condition) != 2:
        raise ValueError("Missing or ambiguous partial-tile exponential branches")
    body = body.replace(condition, "if constexpr (EXP_APPROX_MODE)")
    body = replace_once(
        body,
        "Keep this path for partial faces.\n    // The accurate branch below handles full RC tiles.",
        "Accurate mode also supports partial faces.\n    // Preserve the requested vector region for both scaling and exponentiation.",
    )
    body = replace_once(
        body,
        "                    mul_unary_tile(j, scale_fp32);",
        """                    if constexpr (vector_mode == VectorMode::RC) {
                        mul_unary_tile(j, scale_fp32);
                    } else {
                        // Same FP32 scalar multiplication as mul_unary_tile,
                        // restricted to the valid faces of this partial tile.
                        MATH(SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_binop_with_scalar,
                            (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
                            j,
                            vector_mode,
                            scale_fp32));
                    }""",
    )
    return source[:begin] + body + source[end:]


def build_overlay(native, destination, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown partial-tile diagnostic variant")
    native, destination = Path(native).resolve(), Path(destination).resolve()
    if any(c in str(destination) for c in ('"', "\n", "\r", "\\")):
        raise ValueError("Unsafe overlay header path")
    for name, expected in PINNED.items():
        if sha(native / name) != expected:
            raise ValueError(f"Pinned attention dependency changed: {name}")
    source = (native / COMMON).read_text()
    header = patch_common(source) if variant == "accurate_partial" else source
    compute = replace_once(
        (native / COMPUTE).read_text(),
        '#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"',
        f'#include "{destination / COMMON}"',
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, contents in ((COMMON, header), (COMPUTE, compute)):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    manifest = dict(
        variant=variant,
        overlay=str(destination),
        native=str(native),
        native_sha256=PINNED,
        compute=str(destination / COMPUTE),
        common=str(destination / COMMON),
        overlay_sha256={str(p): sha(destination / p) for p in (COMPUTE, COMMON)},
        precision_change=False,
        promoted_to_model=False,
        changes="Honor accurate exp on partial query tiles, including FP32 scale over only the valid vector region",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_overlay(manifest, kernel_path, cwd):
    if Path(kernel_path).resolve() != Path(manifest["overlay"]):
        raise ValueError("Partial-tile override does not match manifest")
    if (Path(cwd) / COMPUTE).exists():
        raise ValueError("Working directory shadows the attention override")
    for name, expected in manifest["native_sha256"].items():
        if sha(Path(manifest["native"]) / name) != expected:
            raise ValueError("Native attention source changed")
    for name, expected in manifest["overlay_sha256"].items():
        if sha(Path(manifest["overlay"]) / name) != expected:
            raise ValueError("Partial-tile overlay changed")


def compilation_evidence(cache, manifest):
    rows = []
    groups = {}
    # Dataflow kernels use kernel_includes.hpp; this pinned runtime emits
    # compute includes directly into the generated TRISC source wrappers.
    for path in Path(cache).rglob("chlkc_*.cpp"):
        source = path.read_text()
        if COMPUTE.name not in source:
            continue
        if f'#include "{manifest["compute"]}"' not in source:
            raise ValueError("JIT compiled a different attention compute kernel")
        groups.setdefault(path.parent, set()).add(path.name)
        rows.append(dict(path=str(path), sha256=sha(path)))
    if not rows:
        raise ValueError("Missing compilation evidence for partial-tile overlay")
    required = {"chlkc_unpack.cpp", "chlkc_math.cpp", "chlkc_pack.cpp"}
    if any(not required.issubset(names) for names in groups.values()):
        raise ValueError("Incomplete TRISC compilation evidence for partial-tile overlay")
    return rows


def compare_simulator(reports):
    if [r["variant"] for r in reports] != list(VARIANTS):
        raise ValueError("Need native and candidate simulator receipts")
    if any(r.get("state") != "completed" or r.get("cleanup_completed") is not True for r in reports):
        raise ValueError("Both simulator processes must complete cleanly")
    expected = [(length, length - delta, heads) for length in CONTEXTS for delta in (0, 37) for heads in (32, 6)]
    if any([(c["context"], c["active_tokens"], c["query_heads"]) for c in r["cases"]] != expected for r in reports):
        raise ValueError("Incomplete simulator case coverage")
    rows = []
    for native, candidate in zip(reports[0]["cases"], reports[1]["cases"]):
        for key in ("context", "active_tokens", "query_heads", "operand_sha256", "reference_sha256"):
            if native[key] != candidate[key]:
                raise ValueError("Simulator comparisons have mismatched operands")
        if native["query_heads"] == 32 and (
            not native["accuracy"]["passed"] or native["output_sha256"] != candidate["output_sha256"]
        ):
            raise ValueError("Passing full-tile control must remain bit-identical")
        rows.append(
            dict(
                context=native["context"],
                active_tokens=native["active_tokens"],
                query_heads=native["query_heads"],
                native_accuracy=native["accuracy"],
                candidate_accuracy=candidate["accuracy"],
                output_bit_identical=native["output_sha256"] == candidate["output_sha256"],
            )
        )
    return dict(
        completed=True,
        candidate_accuracy_passed=all(r["candidate_accuracy"]["passed"] for r in rows),
        comparisons=rows,
        hardware_qualified=False,
        promoted_to_model=False,
        scope="One virtual-chip synthetic numerical screen; no hardware timing or model-eval qualification",
    )


def compare_hardware(cases):
    """Bracket each partial query with unchanged full-query kernel controls."""
    if len(cases) != 3:
        raise ValueError("Need full/partial/full hardware controls")
    before, partial, after = cases
    if [c["device_query_heads"] for c in cases] != [32, 6, 32]:
        raise ValueError("Wrong hardware query-head comparison")
    for key in (
        "input_tokens",
        "batch",
        "positions",
        "seed",
        "aligned_capacity",
        "native_chunk",
        "page_table_sha256",
        "query_bf16_sha256",
        "reference_fp32_sha256",
        "worker_grid",
    ):
        if before[key] != partial[key] or before[key] != after[key]:
            raise ValueError("Mismatched hardware geometry or operands")
    if any(len(c["candidates"]) != 1 for c in cases):
        raise ValueError("Need one fixed chunk per hardware case")
    rows = [c["candidates"][0] for c in cases]
    if any(
        len(c["candidates"][0]["traced_call_us"]) != 5
        or len(c["baseline_repeat_us"]) != 5
        or len(c["candidates"][0]["output_fp32_sha256_per_rank"]) != 4
        or len(c["baseline_repeat_output_fp32_sha256_per_rank"]) != 4
        for c in cases
    ):
        raise ValueError("Incomplete hardware samples or ranks")
    for index in (0, 2):
        if not cases[index]["passed"] or not rows[index]["accuracy_passed"]:
            raise ValueError("Full-query hardware control failed")
    if rows[0]["output_fp32_sha256_per_rank"] != rows[2]["output_fp32_sha256_per_rank"]:
        raise ValueError("Full-query control output changed")
    deterministic = all(
        c["candidates"][0]["output_fp32_sha256_per_rank"] == c["baseline_repeat_output_fp32_sha256_per_rank"]
        for c in cases
    )
    baseline_samples = [
        v for c in (before, after) for v in (*c["candidates"][0]["traced_call_us"], *c["baseline_repeat_us"])
    ]
    candidate_samples = [*rows[1]["traced_call_us"], *partial["baseline_repeat_us"]]
    if any(v <= 0 or not math.isfinite(v) for v in (*baseline_samples, *candidate_samples)):
        raise ValueError("Invalid hardware timing")
    baseline_us, candidate_us = statistics.median(baseline_samples), statistics.median(candidate_samples)
    # Compare the four full-query medians; include the within-case bracket.
    control_medians = [
        statistics.median(v)
        for c in (before, after)
        for v in (c["candidates"][0]["traced_call_us"], c["baseline_repeat_us"])
    ]
    drift = max(control_medians) / min(control_medians) - 1
    candidate_drift = abs(statistics.median(partial["baseline_repeat_us"]) / rows[1]["median_traced_call_us"] - 1)
    passed = partial["passed"] and rows[1]["accuracy_passed"]
    qualified = passed and deterministic and max(drift, candidate_drift) <= 0.03
    return dict(
        input_tokens=before["input_tokens"],
        batch=before["batch"],
        full_query_us=baseline_us,
        partial_query_us=candidate_us,
        candidate_accuracy_passed=passed,
        deterministic_replay=deterministic,
        full_query_drift_fraction=drift,
        partial_query_drift_fraction=candidate_drift,
        timing_comparison_qualified=qualified,
        qualified_speedup=baseline_us / candidate_us if qualified else None,
        output_bit_identical=rows[0]["output_fp32_sha256_per_rank"] == rows[1]["output_fp32_sha256_per_rank"],
        promoted_to_model=False,
        scope="Synthetic TP4 kernel comparison; full-query controls use unchanged arithmetic in the candidate overlay",
    )
