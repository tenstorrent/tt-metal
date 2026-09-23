# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import statistics
import os
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare, reference
from .test_sdpa_recipe_model_capture import model_capture


def config(grid):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512)


def options(variant, grid):
    return dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=config(grid),
    )


def upload(device, host, variant, pad_value=None):
    return prepare(
        [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, pad_value=pad_value) for x in host], variant
    )


def joint(inputs, variant, grid):
    return ttnn.transformer.joint_scaled_dot_product_attention(
        *inputs[0], *inputs[1], joint_strategy="rear", **options(variant, grid)
    )


def joined(outputs):
    return torch.cat([ttnn.to_torch(output) for output in outputs], dim=2)


@pytest.fixture(
    scope="module",
    params=[
        (384, 128, 1, (1, 1), "normal"),
        (32, 480, 1, (1, 1), "common_q"),
        (512, 512, 1, (2, 1), "common_k"),
        (1280, 256, 3, (4, 3), "changed_max"),
        (3072, 512, 1, (3, 1), "uniform"),
        (4096, 512, 4, (4, 4), "normal"),
    ],
)
def joint_case(request):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    n, j, heads, grid, distribution = request.param
    host = make_inputs(n + j, distribution, q_length=n + j, heads=heads)
    segments = [[x[..., start:end, :].contiguous() for x in host] for start, end in ((0, n), (n, n + j))]
    expected = torch.cat(
        [reference(host[0][..., start : start + 256, :], *host[1:]) for start in range(0, n + j, 256)], dim=2
    )
    return host, segments, expected, grid, distribution


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_recipe_matches_dense(device, joint_case, variant, record_property):
    host, segments, expected, grid, distribution = joint_case
    dense_inputs = upload(device, host, variant)
    inputs = [upload(device, segment, variant) for segment in segments]
    original = [digest(ttnn.to_torch(x)) for segment in inputs for x in segment]
    baseline = ttnn.transformer.scaled_dot_product_attention(*dense_inputs, is_causal=False, **options(variant, grid))
    actual = joined(joint(inputs, variant, grid))
    dense = ttnn.to_torch(baseline)
    assert digest(actual) == digest(dense)
    observed, baseline_metrics = metrics(actual, expected), metrics(dense, expected)
    assert observed["l2_pct"] <= 1.05 * baseline_metrics["l2_pct"] + 0.0001
    record_property("variant", variant)
    record_property("distribution", distribution)
    record_property("dense_output_equal", True)
    for key, value in observed.items():
        record_property(key, value)
    assert original == [digest(ttnn.to_torch(x)) for segment in inputs for x in segment]


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_recipe_model_capture(device, model_capture, variant, record_property):
    name, host, expected = model_capture
    grid = (4, 4)
    dense = ttnn.transformer.scaled_dot_product_attention(
        *upload(device, host, variant), is_causal=False, **options(variant, grid)
    )
    segments = [
        upload(device, [x[..., start:end, :].contiguous() for x in host], variant)
        for start, end in ((0, 4096), (4096, 4608))
    ]
    actual = joined(joint(segments, variant, grid))
    assert digest(actual) == digest(ttnn.to_torch(dense))
    observed = metrics(actual, expected)
    assert observed["pcc"] > 0.9
    record_property("variant", variant)
    record_property("capture", name)
    record_property("dense_output_equal", True)
    for key, value in observed.items():
        record_property(key, value)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lengths", [(384, 128), (320, 96)])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_joint_recipe_cache_trace(device, variant, lengths):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    grid = (2, 1)
    outputs = []
    retained_inputs = []
    primary, total = lengths[0], sum(lengths)
    for seed in (20260919, 20260920):
        host = make_inputs(total, "normal", q_length=total, seed=seed)
        inputs = [
            upload(device, [x[..., start:end, :].contiguous() for x in host], variant)
            for start, end in ((0, primary), (primary, total))
        ]
        retained_inputs.append(inputs)
        actual = joint(inputs, variant, grid)
        outputs.append(joined(actual))
        if len(outputs) == 1:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries
            assert digest(outputs[0]) != digest(outputs[1])
    if total == 512:
        expected = ttnn.transformer.scaled_dot_product_attention(
            *upload(device, host, variant), is_causal=False, **options(variant, grid)
        )
        assert digest(outputs[-1]) == digest(ttnn.to_torch(expected))
    other_split = [
        upload(device, [x[..., start:end, :].contiguous() for x in host], variant)
        for start, end in ((0, 256), (256, total))
    ]
    for index in range(3):
        assert digest(joined([segment[index] for segment in other_split])) == digest(
            joined([segment[index] for segment in retained_inputs[-1]])
        )
    repartitioned = joined(joint(other_split, variant, grid))
    assert digest(repartitioned) == digest(outputs[-1]), metrics(repartitioned, outputs[-1])
    assert digest(joined(joint(retained_inputs[-1], variant, grid))) == digest(outputs[-1])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = joint(retained_inputs[-1], variant, grid)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(joined(traced)) == digest(outputs[-1])
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "q_lengths,k_lengths,heads,grid,distribution",
    [
        ((32, 32), (32, 32), 1, (1, 1), "normal"),
        ((384, 128), (32, 224), 1, (2, 1), "uniform"),
        ((32, 256), (384, 128), 2, (4, 1), "normal"),
        ((512, 256), (512, 256), 2, (4, 1), "normal"),
        ((768, 32), (1024, 32), 2, (6, 1), "changed_max"),
        ((32, 32), (32, 512), 1, (1, 1), "constant_v"),
        ((512, 32), (32, 224), 1, (2, 1), "zero_v"),
    ],
)
def test_joint_recipe_chunk_tails(device, variant, q_lengths, k_lengths, heads, grid, distribution, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(sum(k_lengths), distribution, q_length=sum(q_lengths), heads=heads)
    segments = [
        [x[..., : lengths[0], :].contiguous() for x, lengths in zip(host, (q_lengths, k_lengths, k_lengths))],
        [x[..., lengths[0] :, :].contiguous() for x, lengths in zip(host, (q_lengths, k_lengths, k_lengths))],
    ]
    inputs = [upload(device, segment, variant) for segment in segments]
    hashes = [digest(ttnn.to_torch(x)) for segment in inputs for x in segment]
    outputs = joint(inputs, variant, grid)
    assert [tuple(x.shape) for x in outputs] == [tuple(segment[0].shape) for segment in segments]
    actual = joined(outputs)
    observed = metrics(actual, reference(*host))
    record_property("variant", variant)
    record_property("distribution", distribution)
    for key, value in observed.items():
        record_property(key, value)
    if distribution == "zero_v":
        assert observed["max_abs"] <= 1e-6
    else:
        limits = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}
        if distribution == "constant_v":
            limits = {key: 0.4 if key in ("C", "D") else 1 for key in limits}
        assert observed["l2_pct"] < limits[variant]
        if distribution == "uniform":
            # Duplicating every key/value leaves exact attention unchanged and
            # gives an aligned diagnostic. Finite-precision reductions differ,
            # so equal attention does not imply equal error for this comparison.
            dense_host = [host[0], *(x.repeat(1, 1, 2, 1) for x in host[1:])]
            dense = ttnn.transformer.scaled_dot_product_attention(
                *upload(device, dense_host, variant), is_causal=False, **options(variant, grid)
            )
            baseline = metrics(ttnn.to_torch(dense), reference(*host))
            record_property("duplicate_kv_l2_pct", baseline["l2_pct"])
            if variant in ("A", "B"):
                legacy = ttnn.transformer.scaled_dot_product_attention(
                    *upload(device, host, variant),
                    is_causal=False,
                    program_config=config(grid),
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False
                    ),
                )
                legacy_metrics = metrics(ttnn.to_torch(legacy), reference(*host))
                record_property("legacy_tail_l2_pct", legacy_metrics["l2_pct"])
                assert observed["l2_pct"] <= 1.05 * legacy_metrics["l2_pct"] + 0.0001
    assert hashes == [digest(ttnn.to_torch(x)) for segment in inputs for x in segment]


@pytest.mark.parametrize("invalid", ["joint_heads", "joint_dtype", "strategy", "unprepared", "compute"])
def test_joint_recipe_rejects_unsupported(device, invalid):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    n, j = 384, 128
    segments = [make_inputs(length, "normal", q_length=length) for length in (n, j)]
    if invalid == "joint_heads":
        segments[1] = [x.repeat(1, 2, 1, 1) for x in segments[1]]
    inputs = [upload(device, segment, "D") for segment in segments]
    kwargs = options("D", (2, 1))
    strategy = "front" if invalid == "strategy" else "rear"
    if invalid == "joint_dtype":
        inputs[1][1] = ttnn.from_torch(segments[1][1], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    elif invalid == "unprepared":
        kwargs["precision"] = ttnn.SDPAPrecision.LOW_PRECISION
    elif invalid == "compute":
        kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
    entries = device.num_program_cache_entries()
    with pytest.raises(RuntimeError, match="SDPA|recipe|precision"):
        ttnn.transformer.joint_scaled_dot_product_attention(*inputs[0], *inputs[1], joint_strategy=strategy, **kwargs)
    assert device.num_program_cache_entries() == entries


@pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_PERF") != "1", reason="Opt-in performance benchmark")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_joint_recipe_throughput(device, variant, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(4608, "normal", q_length=4608, heads=4)
    grid = (4, 4)
    dense = upload(device, host, variant)
    segments = [
        upload(device, [x[..., start:end, :].contiguous() for x in host], variant)
        for start, end in ((0, 4096), (4096, 4608))
    ]
    calls = {
        "dense": lambda: (
            ttnn.transformer.scaled_dot_product_attention(*dense, is_causal=False, **options(variant, grid)),
        ),
        "joint": lambda: joint(segments, variant, grid),
    }
    record_property("variant", variant)
    expected = None
    for mode, invoke in calls.items():
        output = joined(invoke())
        if expected is None:
            expected = digest(output)
        assert digest(output) == expected
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        traced = invoke()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            times = []
            for iteration in range(12):
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 3:
                    times.append(elapsed)
            assert digest(joined(traced)) == expected
            record_property(f"{mode}_trace_wall_ms", statistics.median(times))
        finally:
            ttnn.release_trace(device, trace)
