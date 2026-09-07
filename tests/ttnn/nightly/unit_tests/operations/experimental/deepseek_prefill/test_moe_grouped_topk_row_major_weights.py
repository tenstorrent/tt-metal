# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct row-major weights must match tiled weights and preserve cached bindings."""

import pytest
import torch

import ttnn


def _inputs(device, shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(shape, generator=generator)
    bias = torch.randn(shape, generator=generator) * 0.1
    return tuple(ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT) for x in (scores, bias))


def _gate(inputs, k, groups, memory_config, layout, **kwargs):
    return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        *inputs,
        n_groups=groups,
        summed_experts_per_group=2 if groups == 8 else 1,
        topk_groups=4 if groups == 8 else 1,
        n_activated_experts=k,
        route_scale=2.5,
        stable_sort=True,
        memory_config=memory_config,
        weights_layout=layout,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
@pytest.mark.parametrize(
    "shape,k,groups,score_func",
    [
        ((1, 1, 1, 256), 1, 1, "sigmoid"),
        ((1, 1, 31, 256), 8, 8, "sigmoid"),
        ((1, 1, 32, 256), 16, 1, "sigmoid"),
        ((1, 1, 33, 256), 17, 1, "sigmoid"),
        ((2, 1, 33, 256), 32, 1, "sigmoid"),
        ((1, 2, 65, 256), 8, 8, "sigmoid"),
        ((1, 1, 640, 256), 8, 1, "sigmoid"),
        ((1, 1, 65, 384), 8, 1, "sigmoid"),
        ((1, 1, 33, 256), 8, 8, "sqrtsoftplus"),
    ],
    ids=["single", "face_low", "tile_full", "face_cross", "batched_tail", "multi_batch", "moe", "kimi", "sqrtsoftplus"],
)
def test_row_major_weights(device, shape, k, groups, score_func, memory_config, dtype):
    inputs = _inputs(device, shape, dtype, 1049)
    tiled = _gate(inputs, k, groups, memory_config, ttnn.TILE_LAYOUT, score_func=score_func)
    row_major = _gate(inputs, k, groups, memory_config, ttnn.ROW_MAJOR_LAYOUT, score_func=score_func)
    assert row_major[0].layout == ttnn.ROW_MAJOR_LAYOUT
    assert row_major[1].layout == ttnn.TILE_LAYOUT
    for expected, actual in zip(tiled, row_major):
        assert torch.equal(ttnn.to_torch(expected), ttnn.to_torch(actual))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
def test_row_major_weights_fresh_bindings_and_old_trace(device, memory_config):
    device.enable_program_cache()
    shape = (1, 1, 64, 256)
    retained = []

    def make_args(seed, real_tokens, pad_side):
        inputs = _inputs(device, shape, ttnn.float32, seed)
        padding = ttnn.from_torch(
            torch.tensor([[real_tokens, pad_side]], dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        debug = ttnn.from_torch(torch.zeros(shape), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        retained.extend([*inputs, padding, debug])
        return inputs, dict(padding_config=padding, biased_scores=debug)

    original, options = make_args(2026, 41, 0)
    reference = _gate(original, 8, 8, memory_config, ttnn.TILE_LAYOUT, **options)
    expected = [ttnn.to_torch(x).clone() for x in reference]
    warm = _gate(original, 8, 8, memory_config, ttnn.ROW_MAJOR_LAYOUT, **options)
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    captured = _gate(original, 8, 8, memory_config, ttnn.ROW_MAJOR_LAYOUT, **options)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    retained.extend([*reference, *warm, *captured])
    try:
        for iteration in range(3):
            fresh, fresh_options = make_args(2027 + iteration, 31 + iteration, iteration % 2)
            fresh_reference = _gate(fresh, 8, 8, memory_config, ttnn.TILE_LAYOUT, **fresh_options)
            before = device.num_program_cache_entries()
            actual = _gate(fresh, 8, 8, memory_config, ttnn.ROW_MAJOR_LAYOUT, **fresh_options)
            assert device.num_program_cache_entries() == before
            for a, b in zip(fresh_reference, actual):
                assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
            retained.extend([*fresh_reference, *actual])

            # Clear the captured weights so a replay using the wrong output binding cannot pass.
            zeros = ttnn.from_torch(torch.zeros_like(expected[0]), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(zeros, captured[0])
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            for a, b in zip(expected, captured):
                assert torch.equal(a, ttnn.to_torch(b))
    finally:
        ttnn.release_trace(device, trace)


def test_row_major_weights_rejects_unsupported_width(device, expect_error):
    inputs = _inputs(device, (1, 1, 32, 256), ttnn.float32, 1049)
    with expect_error(RuntimeError, "at most 32 activated experts"):
        _gate(inputs, 64, 1, ttnn.L1_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT)
