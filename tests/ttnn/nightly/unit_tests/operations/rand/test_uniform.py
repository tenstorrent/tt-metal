# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import time
import pytest
import numpy as np
import ttnn
from collections import Counter
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from enum import Enum


class TestMode(Enum):
    VALIDATE = 0
    BENCHMARK = 1


def check_torch_uniform_bfloat16():
    input = torch.zeros(10, 10, dtype=torch.bfloat16).uniform_(2.1, 2.11)
    logger.info(input)


# With small tensor ttnn might be slower than torch
def benchmark_uniform(cpu_input, npu_input, rand_from, rand_to):
    iter_num = 10

    cpu_total_time = 0
    for i in range(iter_num + 1):
        cpu_start_time = time.time_ns()
        cpu_input.uniform_(rand_from, rand_to)
        cpu_end_time = time.time_ns()
        if i > 0:
            cpu_total_time += cpu_end_time - cpu_start_time
    logger.info(f"CPU avg time: {cpu_total_time / iter_num}ns")

    npu_total_time = 0
    for i in range(iter_num + 1):
        npu_start_time = time.time_ns()
        ttnn.uniform(npu_input, rand_from, rand_to)
        npu_end_time = time.time_ns()
        if i > 0:
            npu_total_time += npu_end_time - npu_start_time
    logger.info(f"NPU avg time: {npu_total_time / iter_num}ns")


def validate_uniform(npu_input, shape, rand_from, rand_to, seed, dtype, compute_kernel_config):
    ttnn.uniform(npu_input, rand_from, rand_to, seed, compute_kernel_config=compute_kernel_config)
    tt_input = ttnn.to_torch(npu_input).reshape(shape)
    elem_cnt = Counter(tt_input.flatten().tolist())

    expected_mean, expected_var = (rand_from + rand_to) / 2, pow(rand_to - rand_from, 2) / 12
    npu_mean, npu_var = torch.mean(tt_input).item(), torch.var(tt_input).item()
    min_val, max_val = torch.min(tt_input).item(), torch.max(tt_input).item()

    logger.info(f"Distinct elements: {len(elem_cnt.keys())}")
    if max_val == rand_to:
        logger.info(f"Count max_val: {elem_cnt[max_val]}")
    logger.info(f"Min val: {min_val}, Max val: {max_val}")
    logger.info(f"Expected mean: {expected_mean}, NPU mean: {npu_mean}")
    logger.info(f"Expected var: {expected_var}, NPU var: {npu_var}")

    assert torch.tensor(rand_from, dtype=get_lib_dtype(torch, dtype)) <= min_val and max_val < torch.tensor(
        rand_to, dtype=get_lib_dtype(torch, dtype)
    )
    assert np.allclose(npu_mean, expected_mean, rtol=0.5)
    assert np.allclose(npu_var, expected_var, rtol=0.5)


# Due to the issue with tensix instruction to generated pseudo-random numbers: #13904, the seed is temporarily fixed to make the test result consistent.
def run_uniform(shape, rand_range, dtype, device, seed=0, compute_kernel_options=None, mode=TestMode.VALIDATE):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    rand_from, rand_to = rand_range[0], rand_range[1]
    cpu_input = torch.ones(shape, dtype=get_lib_dtype(torch, dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, dtype), layout=ttnn.TILE_LAYOUT)

    if mode == TestMode.BENCHMARK:
        benchmark_uniform(cpu_input=cpu_input, npu_input=npu_input, rand_from=rand_from, rand_to=rand_to)
    else:
        validate_uniform(
            npu_input=npu_input,
            shape=shape,
            rand_from=rand_from,
            rand_to=rand_to,
            seed=seed,
            dtype=dtype,
            compute_kernel_config=compute_kernel_config,
        )


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],
        [64, 64],
        [1, 512, 2, 256],
        [512, 512],
        [1024, 1024],
    ],
)
@pytest.mark.parametrize("rand_range", [[0, 1], [2.1, 9], [-5.1, 1.2]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("seed", [2024, 19, 522021])
def test_uniform(shape, rand_range, dtype, seed, device):
    torch.manual_seed(seed)
    run_uniform(shape, rand_range, dtype, device, seed=seed)


@pytest.mark.parametrize(
    "shape",
    [[2, 32, 32, 16]],
)
@pytest.mark.parametrize("rand_range", [[-3, 4]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("seed", [0])
def test_uniform_callback(shape, rand_range, dtype, seed, device):
    torch.manual_seed(seed)
    num_program_cache_entries_list = []
    for _ in range(2):
        run_uniform(shape, rand_range, dtype, device, seed=seed)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

        # Cache must hit when we change seed and seed runtime arg is overrode
        seed = seed + 1

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


def test_uniform_seed_distinguishes_cache_entries(device):
    """Regression guard for uniform's seed static/dynamic contract.

    uniform writes in-place into its input tensor (output aliases input). `seed` is excluded from
    compute_program_hash (so calls differing only in seed cache-hit) but must be re-applied to the
    cached program on every dispatch. Pins both halves:
      * different seed must NOT grow the cache  -> guards against re-adding seed to the hash.
      * different seed must change the output    -> guards against the frozen-runtime-arg bug on
        the in-place fast path (only reachable now that resolve_bindings allows input==output).
      * same seed must reproduce the output      -> guards against wrongly re-randomizing.
      * different from/to range on a hit         -> must NOT grow the cache and must be re-applied
        (from/to are also hash-excluded and re-derived by override_runtime_arguments).
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    npu = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    ttnn.uniform(npu, 0.0, 1.0, 1234)
    out_a = ttnn.to_torch(npu).float().clone()
    entries_a = device.num_program_cache_entries()

    ttnn.uniform(npu, 0.0, 1.0, 1234)
    out_a2 = ttnn.to_torch(npu).float().clone()
    entries_b = device.num_program_cache_entries()

    ttnn.uniform(npu, 0.0, 1.0, 5678)
    out_c = ttnn.to_torch(npu).float().clone()
    entries_c = device.num_program_cache_entries()

    ttnn.uniform(npu, 5.0, 10.0, 1234)
    out_d = ttnn.to_torch(npu).float().clone()
    entries_d = device.num_program_cache_entries()

    assert entries_b == entries_a, "same seed must reuse the cached program"
    assert entries_c == entries_a, "a different seed must NOT add a cache entry -- seed is dynamic, not hashed"
    assert (
        entries_d == entries_a
    ), "a different from/to range must NOT add a cache entry -- range is dynamic, not hashed"
    assert torch.equal(out_a, out_a2), "same seed must reproduce identical output"
    assert not torch.equal(out_a, out_c), "a different seed must change the output (seed re-patched on the fast path)"
    assert out_d.min() >= 5.0 and out_d.max() < 10.0, "a different from/to range must be re-applied on the cache hit"

    device.disable_and_clear_program_cache()


@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 2, 4, 70, 40]],
)
@pytest.mark.parametrize("seed", [1408])
@pytest.mark.parametrize("rand_range", [[0, 1]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_uniform_with_compute_kernel_options(shape, seed, rand_range, dtype, device, compute_kernel_options):
    torch.manual_seed(seed)
    run_uniform(shape, rand_range, dtype, device, seed=seed, compute_kernel_options=compute_kernel_options)
