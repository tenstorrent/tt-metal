# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for the RunningStatistics device op, exercised via
ttnn.batch_norm(training=True) (which updates running_mean/running_var through it).

Pins the cache-keying granularity so it can be verified BEFORE and AFTER removing
RunningStatistics::compute_program_hash. The custom hash keyed on the whole attributes struct
(momentum/memory_config/compute_kernel_config/input_dtype/dtype) + each tensor's TensorSpec plus a
redundant padded_shape (derived from tensor_spec) and a redundant optional re-encoding. The framework
default hashes the whole attrs struct + tensor_args, i.e. the same distinctions and the same
logical-shape keying, so the TOTAL program-cache entry count for a batch_norm(training=True) call is
unchanged.

- Same config twice -> reuse (no new entries).
- Different input shape / momentum -> extra entries (RunningStatistics re-keys).
"""

import pytest
import torch

import ttnn


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_bn_training(device, input_shape, momentum, seed=0):
    """ttnn.batch_norm(training=True) inside the cache counter (updates running stats via RunningStatistics)."""
    torch.manual_seed(seed)
    channels = input_shape[1]
    inp = ttnn.from_torch(torch.rand(input_shape, dtype=torch.bfloat16) * 5 + 5, device=device, layout=ttnn.TILE_LAYOUT)
    mean = ttnn.from_torch(
        torch.rand(channels, dtype=torch.bfloat16).view(1, channels, 1, 1) * 6 + 4,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    var = ttnn.from_torch(
        torch.rand(channels, dtype=torch.bfloat16).view(1, channels, 1, 1) * 16 + 4,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    with device.cache_entries_counter.measure():
        out = ttnn.batch_norm(inp, running_mean=mean, running_var=var, training=True, eps=1e-5, momentum=momentum)
    return ttnn.to_torch(out)


def test_bn_cache_reuse_same_config(device, isolate_program_cache):
    """Same config twice -> the second run adds no new program-cache entries."""
    run_bn_training(device, (1, 4, 32, 32), momentum=0.1, seed=0)
    n_after_first = device.cache_entries_counter.total
    run_bn_training(device, (1, 4, 32, 32), momentum=0.1, seed=1)
    assert device.cache_entries_counter.total == n_after_first  # reuse: no growth


def test_bn_cache_miss_different_shape(device, isolate_program_cache):
    """Different input shape -> more entries than a pure reuse."""
    run_bn_training(device, (1, 4, 32, 32), momentum=0.1)
    n_after_first = device.cache_entries_counter.total
    run_bn_training(device, (1, 4, 64, 64), momentum=0.1)
    assert device.cache_entries_counter.total > n_after_first


def test_bn_cache_miss_different_momentum(device, isolate_program_cache):
    """momentum is a hashed RunningStatistics attribute -> at least one extra entry."""
    run_bn_training(device, (1, 4, 32, 32), momentum=0.1)
    n_after_first = device.cache_entries_counter.total
    run_bn_training(device, (1, 4, 32, 32), momentum=0.5)
    assert device.cache_entries_counter.total > n_after_first
