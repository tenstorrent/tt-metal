# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache keying tests for ttnn.batch_norm(training=True).

In training mode batch_norm drives the RunningStatistics device op, which updates
running_mean/running_var in place. These tests pin two invariants the suite must uphold:

  * Cache-key granularity of RunningStatistics. Runs that share its program key (same
    per-channel tensor specs, momentum, and optional-stat presence) must reuse a cached
    program, while runs that differ in any keyed attribute must compile a distinct one. This
    guards against under-keying (a cache hit reusing a program compiled for a different buffer
    configuration) and over-keying (spurious recompiles).
  * The running-statistics side effect. Each call must write the updated running_mean/
    running_var for that call's own tensors, so a cache hit cannot silently leave a call's
    buffers un-updated (or update someone else's). Every run therefore reads the stats back
    and compares them against a torch reference.

The RunningStatistics device op has no direct Python binding, so it is exercised through
ttnn.batch_norm(training=True), which is its only public entry point.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

# (N, C, H, W); H and W are tile-aligned and C > 1 so the per-channel running-stat tensors
# are non-degenerate.
BASE_SHAPE = (1, 4, 32, 32)

# The initial running-stat values sit well above the batch statistics of the input (which
# lie in [0, 1)). A momentum-weighted update therefore moves each value away from its initial
# value by a clearly observable amount, so a cache hit that leaves a call's buffers un-updated
# (i.e. still at the initial value) is detectable.
INIT_RUNNING_MEAN = 10.0
INIT_RUNNING_VAR = 10.0


def _stat_tensor(value, channels, device, torch_dtype=torch.bfloat16):
    return ttnn.from_torch(
        torch.full((1, channels, 1, 1), value, dtype=torch_dtype),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )


def _assert_running_stats_updated(input_torch, channels, momentum, eps, updated_mean, updated_var):
    """Assert the read-back running stats match a torch reference update from the known initial
    values, i.e. the op wrote this call's buffers instead of leaving them at their initial value."""
    # torch.nn.functional.batch_norm updates the running tensors in place. PyTorch requires both
    # to be provided together, so a neutral stand-in is supplied for an absent stat and only the
    # requested stats are asserted.
    ref_mean = torch.full((channels,), INIT_RUNNING_MEAN, dtype=torch.float32)
    ref_var = torch.full((channels,), INIT_RUNNING_VAR, dtype=torch.float32)
    F.batch_norm(
        input_torch.float(),
        running_mean=ref_mean,
        running_var=ref_var,
        training=True,
        momentum=momentum,
        eps=eps,
    )
    if updated_mean is not None:
        # The update must have moved the value off its initial state (catches a stale-binding
        # cache hit that never wrote this call's buffer).
        assert not torch.allclose(updated_mean, torch.full_like(updated_mean, INIT_RUNNING_MEAN), atol=0.1)
        assert_numeric_metrics(
            ref_mean.view(1, channels, 1, 1),
            updated_mean,
            rtol=0.05,
            atol=0.1,
            frobenius_threshold=0.05,
            check_pcc=False,
        )
    if updated_var is not None:
        assert not torch.allclose(updated_var, torch.full_like(updated_var, INIT_RUNNING_VAR), atol=0.1)
        assert_numeric_metrics(
            ref_var.view(1, channels, 1, 1),
            updated_var,
            rtol=0.05,
            atol=0.1,
            frobenius_threshold=0.05,
            check_pcc=False,
        )


def _run_bn_training(
    device, input_shape, momentum, *, with_mean=True, with_var=True, seed=0, eps=1e-5, torch_dtype=torch.bfloat16
):
    """Run ttnn.batch_norm(training=True) under the program-cache entry counter, then read back
    and validate the in-place running-stat update for this call."""
    torch.manual_seed(seed)
    channels = input_shape[1]
    input_torch = torch.rand(input_shape, dtype=torch_dtype)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT)
    mean_tt = _stat_tensor(INIT_RUNNING_MEAN, channels, device, torch_dtype) if with_mean else None
    var_tt = _stat_tensor(INIT_RUNNING_VAR, channels, device, torch_dtype) if with_var else None

    with device.cache_entries_counter.measure():
        ttnn.batch_norm(
            input_tt,
            running_mean=mean_tt,
            running_var=var_tt,
            training=True,
            eps=eps,
            momentum=momentum,
        )

    updated_mean = ttnn.to_torch(mean_tt) if with_mean else None
    updated_var = ttnn.to_torch(var_tt) if with_var else None
    _assert_running_stats_updated(input_torch, channels, momentum, eps, updated_mean, updated_var)


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_bn_cache_reuse_same_config(device, isolate_program_cache):
    """Identical config across calls must reuse cached programs: the second run adds no entries."""
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, seed=0)
    n_after_first = device.cache_entries_counter.total
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, seed=1)
    assert device.cache_entries_counter.total == n_after_first  # reuse: no growth


def test_bn_cache_miss_different_channels(device, isolate_program_cache):
    """Changing the channel count changes the RunningStatistics per-channel tensor specs, which
    must key a distinct program (new entries). H/W alone would not re-key RunningStatistics."""
    _run_bn_training(device, (1, 4, 32, 32), momentum=0.1)
    n_after_first = device.cache_entries_counter.total
    _run_bn_training(device, (1, 8, 32, 32), momentum=0.1)
    assert device.cache_entries_counter.total > n_after_first


def test_bn_cache_miss_different_momentum(device, isolate_program_cache):
    """momentum is a hashed RunningStatistics attribute, so changing it must key a distinct program."""
    _run_bn_training(device, BASE_SHAPE, momentum=0.1)
    n_after_first = device.cache_entries_counter.total
    _run_bn_training(device, BASE_SHAPE, momentum=0.5)
    assert device.cache_entries_counter.total > n_after_first


def test_bn_cache_miss_different_dtype(device, isolate_program_cache):
    """dtype drives the RunningStatistics compile-time `any_float32` path, which selects a different
    compute kernel (sfpu vs non-sfpu). A bfloat16 run and a float32 run must therefore key distinct
    programs, and a repeated dtype must reuse. This pins the dtype keying (input_dtype/dtype) that
    the default hash must preserve now that the op no longer keys on it explicitly."""
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, torch_dtype=torch.bfloat16, seed=0)
    n_bf16 = device.cache_entries_counter.total
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, torch_dtype=torch.float32, seed=1)
    n_fp32 = device.cache_entries_counter.total
    assert n_fp32 > n_bf16  # dtype flip -> distinct program
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, torch_dtype=torch.float32, seed=2)
    assert device.cache_entries_counter.total == n_fp32  # same dtype -> reuse


def test_bn_cache_optional_stat_presence(device, isolate_program_cache):
    """running_mean/running_var presence is baked into the RunningStatistics program, so each
    present/absent combination must key a distinct program while a repeated combination reuses it.
    Only RunningStatistics depends on this presence, so the entry growth is attributable to it
    (the upstream reduction/batch-norm programs are identical across combinations)."""
    # Both stats present.
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=True, with_var=True, seed=0)
    n_both = device.cache_entries_counter.total
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=True, with_var=True, seed=1)
    assert device.cache_entries_counter.total == n_both  # same combo -> reuse

    # running_var absent -> distinct presence bits -> new program.
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=True, with_var=False, seed=2)
    n_mean_only = device.cache_entries_counter.total
    assert n_mean_only > n_both
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=True, with_var=False, seed=3)
    assert device.cache_entries_counter.total == n_mean_only  # same combo -> reuse

    # running_mean absent (var present) -> another distinct combination -> new program.
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=False, with_var=True, seed=4)
    n_var_only = device.cache_entries_counter.total
    assert n_var_only > n_mean_only
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=False, with_var=True, seed=5)
    assert device.cache_entries_counter.total == n_var_only  # same combo -> reuse

    # Neither stat present -> a fourth distinct combination. In training mode RunningStatistics
    # still runs (it just has no buffers to update), so this presence combo keys its own program.
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=False, with_var=False, seed=6)
    n_neither = device.cache_entries_counter.total
    assert n_neither > n_var_only
    _run_bn_training(device, BASE_SHAPE, momentum=0.1, with_mean=False, with_var=False, seed=7)
    assert device.cache_entries_counter.total == n_neither  # same combo -> reuse
