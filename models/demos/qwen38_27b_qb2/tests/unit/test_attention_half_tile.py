# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from models.demos.qwen38_27b_qb2.tests.attention_half_tile import (
    COMMON,
    COMPUTE,
    CONTEXTS,
    VARIANTS,
    build_overlay,
    compare_hardware,
    compare_simulator,
    compilation_evidence,
    patch_common,
    sha,
    verify_overlay,
)

MODULE = "models.demos.qwen38_27b_qb2.tests.attention_half_tile"
ROOT = Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[5]))


def test_exact_native_patch_changes_only_the_partial_accurate_region(expect_error):
    source = (ROOT / COMMON).read_text()
    result = patch_common(source)
    begin = source.index("void sub_exp_block_bcast_cols_inplace(")
    end = source.index("\n}", begin) + 2
    new_end = result.index("\n}", begin) + 2
    assert source[:begin] == result[:begin] and source[end:] == result[new_end:]
    body = result[begin:new_end]
    assert "EXP_APPROX_MODE || vector_mode" not in body
    assert body.count("if constexpr (EXP_APPROX_MODE)") == 2
    assert (
        "if constexpr (vector_mode == VectorMode::RC) {\n                        mul_unary_tile(j, scale_fp32);" in body
    )
    assert "                            vector_mode,\n                            scale_fp32" in body
    assert "exp_tile<false, false, InputClamping::ClampToNegative, iterations>(j, vector_mode_exp)" in body
    for changed in (
        source.replace("if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC)", "changed", 1),
        source.replace("                    mul_unary_tile(j, scale_fp32);", "renamed"),
    ):
        with expect_error(ValueError, "Missing or ambiguous"):
            patch_common(changed)


@pytest.mark.parametrize("variant", VARIANTS)
def test_overlay_is_immutable_and_does_not_edit_native_sources(tmp_path, variant, expect_error):
    # Actual pinned sources exercise include relocation and every edit anchor.
    native = tmp_path / "native"
    for name in (COMMON, COMPUTE):
        p = native / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    hashes = {str(p): sha(native / p) for p in (COMMON, COMPUTE)}
    with patch(MODULE + ".PINNED", hashes):
        result = build_overlay(native, tmp_path / "overlay", variant)
        verify_overlay(result, tmp_path / "overlay", tmp_path / "empty")
        assert hashes == {str(p): sha(native / p) for p in (COMMON, COMPUTE)}
        assert f'#include "{tmp_path/"overlay"/COMMON}"' in (tmp_path / "overlay" / COMPUTE).read_text()
        if variant == "native":
            assert (tmp_path / "overlay" / COMMON).read_bytes() == (ROOT / COMMON).read_bytes()
        with expect_error(FileExistsError, "File exists"):
            build_overlay(native, tmp_path / "overlay", variant)
        with expect_error(ValueError, "shadows"):
            verify_overlay(result, tmp_path / "overlay", native)
        (tmp_path / "overlay" / COMMON).write_text("changed")
        with expect_error(ValueError, "overlay changed"):
            verify_overlay(result, tmp_path / "overlay", tmp_path / "empty")


def test_jit_evidence_requires_the_exact_attention_override(tmp_path, expect_error):
    manifest = dict(compute="/overlay/" + str(COMPUTE))
    with expect_error(ValueError, "Missing compilation"):
        compilation_evidence(tmp_path, manifest)
    p = tmp_path / "chlkc_math.cpp"
    p.write_text(f'#include "{manifest["compute"]}"\n')
    with expect_error(ValueError, "Incomplete TRISC"):
        compilation_evidence(tmp_path, manifest)
    for name in ("chlkc_unpack.cpp", "chlkc_pack.cpp"):
        (tmp_path / name).write_text(p.read_text())
    assert len(compilation_evidence(tmp_path, manifest)) == 3
    p.write_text(f'#include "/native/{COMPUTE}"\n')
    with expect_error(ValueError, "different attention"):
        compilation_evidence(tmp_path, manifest)


def reports():
    return [
        dict(
            variant=variant,
            state="completed",
            cleanup_completed=True,
            cases=[
                dict(
                    context=length,
                    active_tokens=length - delta,
                    query_heads=heads,
                    operand_sha256={"query": "same"},
                    reference_sha256="reference",
                    output_sha256="full-control" if heads == 32 else variant,
                    accuracy=dict(passed=heads == 32 or variant == "accurate_partial"),
                )
                for length in CONTEXTS
                for delta in (0, 37)
                for heads in (32, 6)
            ],
        )
        for variant in VARIANTS
    ]


def test_native_partial_failure_is_preserved_and_candidate_is_not_model_qualified():
    report = compare_simulator(reports())
    assert report["candidate_accuracy_passed"]
    assert not report["hardware_qualified"] and not report["promoted_to_model"]
    assert sum(not r["native_accuracy"]["passed"] for r in report["comparisons"]) == 8
    bad = reports()
    bad[1]["cases"][-1]["accuracy"]["passed"] = False
    assert not compare_simulator(bad)["candidate_accuracy_passed"]


@pytest.mark.parametrize("failure", ["missing", "changed_full_control", "changed_operands", "unclean"])
def test_partial_evidence_cannot_qualify_the_candidate(failure, expect_error):
    data = copy.deepcopy(reports())
    if failure == "missing":
        data[1]["cases"].pop()
    elif failure == "changed_full_control":
        data[1]["cases"][0]["output_sha256"] = "different"
    elif failure == "changed_operands":
        data[1]["cases"][-1]["operand_sha256"] = {"query": "other"}
    else:
        data[1]["cleanup_completed"] = False
    with expect_error(ValueError, "Incomplete|bit-identical|mismatched|cleanly"):
        compare_simulator(data)


def hardware_cases():
    cases = []
    for heads, us in ((32, 100), (6, 80), (32, 100)):
        cases.append(
            dict(
                device_query_heads=heads,
                input_tokens=32768,
                batch=16,
                positions=list(range(16)),
                seed=123,
                aligned_capacity=33280,
                native_chunk=256,
                page_table_sha256="pages",
                query_bf16_sha256="query",
                reference_fp32_sha256="reference",
                worker_grid=[12, 10],
                passed=True,
                candidates=[
                    dict(
                        traced_call_us=[us] * 5,
                        median_traced_call_us=us,
                        accuracy_passed=True,
                        output_fp32_sha256_per_rank=["output"] * 4,
                    )
                ],
                baseline_repeat_us=[us] * 5,
                baseline_repeat_output_fp32_sha256_per_rank=["output"] * 4,
            )
        )
    return cases


def test_hardware_speedup_requires_same_operands_and_both_controls():
    result = compare_hardware(hardware_cases())
    assert result["timing_comparison_qualified"] and result["qualified_speedup"] == 1.25
    assert not result["promoted_to_model"]


@pytest.mark.parametrize("failure", ["operands", "control", "hash", "missing_repeat", "missing_rank", "nonfinite"])
def test_invalid_hardware_comparison_is_rejected(failure, expect_error):
    data = hardware_cases()
    if failure == "operands":
        data[1]["reference_fp32_sha256"] = "changed"
    elif failure == "control":
        data[2]["passed"] = False
    elif failure == "hash":
        data[2]["candidates"][0]["output_fp32_sha256_per_rank"][0] = "changed"
    elif failure == "missing_repeat":
        data[1]["baseline_repeat_us"].pop()
    elif failure == "missing_rank":
        data[1]["candidates"][0]["output_fp32_sha256_per_rank"].pop()
    else:
        data[1]["baseline_repeat_us"][0] = float("nan")
    with expect_error(ValueError, "Mismatched|failed|changed|Incomplete|Invalid"):
        compare_hardware(data)


@pytest.mark.parametrize("failure", ["numerical", "drift", "nondeterministic"])
def test_failing_or_unstable_partial_query_is_retained_without_speedup_claim(failure):
    data = hardware_cases()
    if failure == "numerical":
        data[1]["passed"] = False
        data[1]["candidates"][0]["accuracy_passed"] = False
    elif failure == "drift":
        data[2]["baseline_repeat_us"] = [104] * 5
    else:
        data[1]["baseline_repeat_output_fp32_sha256_per_rank"][0] = "changed"
    result = compare_hardware(data)
    assert not result["timing_comparison_qualified"] and result["qualified_speedup"] is None
