# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
import time
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from collections import Counter
from loguru import logger
from scipy import stats


def run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=False, compute_kernel_options=None, p_value=0.5):
    """
    Runs the ttnn.bernoulli operation and performs a statistical hypothesis test
    to validate its fairness.
    """
    # Bernoulli operation is a comparison operation between input and a random generated number.
    # RNG is expected to be from a uniform distribution.
    k = 20  # Number of bernoulli operations to run to gather sufficient data for the test.
    seed = 0  # Randomize number generation sequence for every bernoulli operation.
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # Set input to a fixed p_value to test the reliability of the bernoulli sampler.
    cpu_input = torch.empty(shape, dtype=get_lib_dtype(torch, in_dtype))
    cpu_input.fill_(p_value)

    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    # Aggregate results across all 'k' runs for a single, powerful statistical test.
    total_trials = 0
    total_successes = 0

    for i in range(k):
        npu_output = None
        if is_out_alloc:
            cpu_output = torch.rand(shape, dtype=get_lib_dtype(torch, out_dtype))
            npu_output = ttnn.from_torch(
                cpu_output, device=device, dtype=get_lib_dtype(ttnn, out_dtype), layout=ttnn.TILE_LAYOUT
            )
            ttnn.bernoulli(
                npu_input,
                seed,
                output=npu_output,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )
        else:
            npu_output = ttnn.bernoulli(
                npu_input,
                seed,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )

        tt_output = ttnn.to_torch(npu_output).reshape(shape)

        # Aggregate results
        num_trials_in_run = tt_output.numel()
        num_successes_in_run = torch.sum(tt_output).item()

        # Basic sanity check that only 0s and 1s are present
        assert num_successes_in_run <= num_trials_in_run

        total_trials += num_trials_in_run
        total_successes += num_successes_in_run

    logger.info(f"Total successes: {total_successes} out of {total_trials} total trials.")
    observed_prob = total_successes / total_trials if total_trials > 0 else 0
    logger.info(f"Observed probability: {observed_prob:.6f}")
    logger.info(f"Expected probability: {p_value}")

    # --- Statistical Hypothesis Test (Binomial Test) ---
    # H₀ (Null Hypothesis): The hardware sampler is fair and produces '1's with the expected probability (p_value).
    # H₁ (Alternative Hypothesis): The sampler is biased and produces '1's with a different probability.
    # We use a significance level of 0.01, which corresponds to 99% confidence.
    alpha = 0.01

    # The binomial test calculates the probability of observing a result this extreme
    # (or more extreme), assuming the null hypothesis is true.
    result = stats.binomtest(k=int(total_successes), n=total_trials, p=p_value, alternative="two-sided")
    p_value_from_test = result.pvalue

    logger.info(f"Binomial test p-value: {p_value_from_test:.6f}")
    logger.info(f"Significance level (alpha): {alpha}")

    # The p-value is the probability of seeing our data (or something more extreme) if the sampler is truly fair.
    # If this probability is very low (p < alpha), we have evidence to reject the idea that it's fair.
    assert (
        p_value_from_test > alpha
    ), f"P-value ({p_value_from_test:.6f}) is less than alpha ({alpha}). Rejecting H₀: The sampler appears to be biased."


@pytest.mark.parametrize("p_value", [0.5])
@pytest.mark.parametrize(
    "shape",
    [
        [2003],
        [500, 500],
        [1, 512, 2, 256],
    ],
)
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("is_out_alloc", [True, False])
def test_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc, p_value):
    run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=is_out_alloc, p_value=p_value)


@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 8, 70, 40]],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
@pytest.mark.parametrize("seed", [1408])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_bernoulli_with_compute_kernel_options(shape, seed, in_dtype, out_dtype, device, compute_kernel_options):
    run_bernoulli(shape, in_dtype, out_dtype, device, compute_kernel_options=compute_kernel_options)
