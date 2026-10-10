# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify page-table row routing and causal continuation under batching."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.qwen38_27b_qb2.tests.prefill_attention_batch import CASES, validate_report


def load_boundary(ops):
    path = Path(__file__).resolve().parents[2] / "tt/prefill_attention.py"
    parsed = ast.parse(path.read_text())
    function = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "fill_and_attend"
    )
    namespace = {"ttnn": ops}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fill_and_attend"]


@pytest.mark.parametrize("batch,start,length", [(1, 0, 64), (2, 32, 64), (3, 64, 65), (16, 0, 128)])
def test_batching_preserves_prefix_inactive_pages_and_causal_output(batch, start, length):
    torch.manual_seed(20261010)
    pages = (start + length + 31) // 32 + 1
    # The table is already sliced to nonzero global slots; local row zero is
    # deliberately not global request zero or physical block zero.
    table = torch.randperm((batch + 2) * pages).reshape(batch + 2, pages)[1 : batch + 1]
    q = torch.randn(batch, 2, length, 4)
    k, v = [torch.randn(batch, 1, length, 4) for _ in range(2)]
    seeds = [torch.randn((batch + 2) * pages, 1, 32, 4).half() for _ in range(2)]
    results, caches, counts = [], [], []
    for batched in (False, True):
        key, value = [seed.clone() for seed in seeds]
        calls = dict(fill=0, attention=0)

        def fill(cache, update, chunk_table, *, batch_idx=0, batch_idx_tensor=None):
            calls["fill"] += 1
            assert cache.dtype == update.dtype
            rows = [batch_idx] if batch_idx_tensor is None else batch_idx_tensor.tolist()
            assert len(rows) == update.shape[0]
            padded = torch.nn.functional.pad(update, (0, 0, 0, (-length) % 32))
            for user, row in enumerate(rows):
                for page, physical in enumerate(chunk_table[row]):
                    cache[physical] = padded[user, :, page * 32 : (page + 1) * 32]

        def attend(query, key_cache, value_cache, full_table, position, *, scale, program_config):
            calls["attention"] += 1
            assert position == start and program_config == "same"
            outputs = []
            for user in range(query.shape[0]):
                kk, vv = [
                    cache[full_table[user]].permute(1, 0, 2, 3).flatten(1, 2).float()
                    for cache in (key_cache, value_cache)
                ]
                score = query[user] @ kk.transpose(-1, -2) * scale
                allowed = torch.arange(pages * 32)[None, :] <= start + torch.arange(length)[:, None]
                outputs.append(torch.softmax(score.masked_fill(~allowed, float("-inf")), dim=-1) @ vv)
            return torch.stack(outputs)

        boundary = load_boundary(
            SimpleNamespace(
                typecast=lambda x, dtype: x.to(dtype),
                concat=torch.cat,
                experimental=SimpleNamespace(paged_fill_cache=fill),
                transformer=SimpleNamespace(chunked_scaled_dot_product_attention=attend),
            )
        )
        results.append(
            boundary(
                q,
                k,
                v,
                key,
                value,
                table,
                start,
                page_size=32,
                scale=0.5,
                program_config="same",
                batch_indices=torch.arange(batch) if batched else None,
            )
        )
        caches.append((key, value))
        counts.append(calls)
        touched = table[:, start // 32 : (start + length + 31) // 32].flatten()
        untouched = torch.ones(key.shape[0], dtype=torch.bool)
        untouched[touched] = False
        for actual, seed in zip((key, value), seeds):
            assert torch.equal(actual[untouched], seed[untouched])
    assert torch.equal(results[0], results[1])
    assert all(torch.equal(a, b) for a, b in zip(*caches))
    assert counts == [dict(fill=2 * batch, attention=batch), dict(fill=2, attention=1)]


@pytest.mark.parametrize("defect", ["table_batch", "indices", "unaligned", "capacity"])
def test_rejects_invalid_boundary_before_device_operations(defect, expect_error):
    q = torch.zeros(2, 2, 32, 4)
    k = torch.zeros(2, 1, 32, 4)
    table = torch.zeros(2, 1, dtype=torch.int32)
    indices, start = torch.arange(2), 0
    if defect == "table_batch":
        table = table[:1]
    elif defect == "indices":
        indices = indices[:1]
    elif defect == "unaligned":
        start = 1
    else:
        start = 32
    with expect_error(ValueError, ".+"):
        load_boundary(SimpleNamespace())(
            q, k, k, None, None, table, start, page_size=32, scale=0.5, program_config=None, batch_indices=indices
        )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_compute_configuration_is_forwarded_identically_without_changing_defaults(batched, override):
    calls = []
    choice = object() if override else None
    values = torch.zeros(2, 1, 32, 4)

    def attend(q, *args, **kwargs):
        calls.append(kwargs)
        return q

    boundary = load_boundary(
        SimpleNamespace(
            experimental=SimpleNamespace(paged_fill_cache=lambda *a, **k: None),
            transformer=SimpleNamespace(chunked_scaled_dot_product_attention=attend),
            concat=torch.cat,
        )
    )
    boundary(
        values,
        values,
        values,
        values,
        values,
        torch.zeros(2, 1, dtype=torch.int32),
        0,
        page_size=32,
        scale=0.5,
        program_config="fixed",
        batch_indices=torch.arange(2) if batched else None,
        compute_kernel_config=choice,
    )
    assert len(calls) == (1 if batched else 2)
    for call in calls:
        assert call["program_config"] == "fixed"
        if override:
            assert call["compute_kernel_config"] is choice
        else:
            assert "compute_kernel_config" not in call


def receipt():
    cases = []
    for batch, start, length in CASES:
        cases.append(
            dict(
                batch=batch,
                start_pos=start,
                chunk_tokens=length,
                total_context=start + length,
                expected_cache_sha256=["k", "v"],
                arms=[
                    dict(
                        name=name,
                        wall_ms=[duration] * 5,
                        median_ms=duration,
                        output_sha256_per_rank=["same"] * 4,
                        cache_sha256_per_rank=[["k"] * 4, ["v"] * 4],
                        accuracy_per_rank=[
                            dict(passed=True, pcc_per_user=[1.0] * batch, relative_rms_per_user=[0.0] * batch)
                        ]
                        * 4,
                    )
                    for name, duration in (("before", 10), ("batched", 8), ("after", 10))
                ],
            )
        )
    return dict(state="completed", passed=True, cleanup_completed=True, cases=cases)


def test_boundary_gain_never_implies_full_model_or_gpqa_qualification():
    rows = validate_report(receipt())
    assert len(rows) == len(CASES)
    assert all(row["speedup"] == 1.25 and not row["full_model_measured"] and not row["gpqa_qualified"] for row in rows)


@pytest.mark.parametrize("defect", ["coverage", "dirty", "missing_user", "cache", "output", "accounting"])
def test_incomplete_or_bad_device_evidence_is_not_a_passing_boundary(defect, expect_error):
    report = copy.deepcopy(receipt())
    arm = report["cases"][0]["arms"][1]
    if defect == "coverage":
        report["cases"].pop()
    elif defect == "dirty":
        report["cleanup_completed"] = False
    elif defect == "missing_user":
        arm["accuracy_per_rank"][0]["pcc_per_user"].pop()
    elif defect == "cache":
        arm["cache_sha256_per_rank"][0][0] = "wrong-page"
    elif defect == "output":
        arm["output_sha256_per_rank"] = ["different"] * 4
    else:
        arm["median_ms"] = 1
    with expect_error(ValueError, ".+"):
        validate_report(report)
