# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
import numpy as np


def test_dopout(device):
    t = torch.ones(
        (
            4,
            1,
            32,
            64,
        )
    )
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_ratios = []
    s = 124
    prob = 0.2
    for _ in range(1000):
        output = ttnn.experimental.dropout(t_tt, probability=prob, scale=1.0 / (1.0 - prob), seed=s + 1)
        s = s + 1
        output_torch = ttnn.to_torch(output)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        tt_ratios.append(r)

    mean = np.mean(tt_ratios)
    std = np.std(tt_ratios)
    # current dropout has pretty high variance so we just checking with some reasonable nubmers
    assert np.allclose(mean, prob, rtol=0.02)
    assert std < prob


def test_dropout_output_tensor(device):
    """Test that dropout correctly uses the output_tensor parameter for in-place operation.

    This is a regression test for https://github.com/tenstorrent/tt-metal/issues/30284
    """
    t = torch.ones((1, 1, 32, 64))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Call dropout with output_tensor=tensor (in-place)
    output = ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=12345, output_tensor=tensor)

    # Both tensor and output should reference the same underlying data
    # After the operation, the original tensor should be modified
    tensor_torch = ttnn.to_torch(tensor)
    output_torch = ttnn.to_torch(output)

    # The tensors should have the same values (both should show the dropout result)
    assert torch.allclose(
        tensor_torch, output_torch
    ), "output_tensor parameter not working: input tensor was not modified in place"


def test_dropout_seed_distinguishes_cache_entries(device):
    """Regression: descriptor framework only re-patches Buffer* slots on cache hit.
    `seed` is a non-Buffer RTA scalar; if it's omitted from compute_program_hash,
    two same-shape calls with different seeds collide on the cached program and
    the second uses the first call's seed (identical dropout mask)."""
    t = torch.ones((1, 1, 32, 64))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    out_a = ttnn.to_torch(ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=1234))
    cache_after_a = device.num_program_cache_entries()

    ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=1234)
    cache_after_b = device.num_program_cache_entries()

    out_c = ttnn.to_torch(ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=5678))
    cache_after_c = device.num_program_cache_entries()

    assert cache_after_b == cache_after_a, "same seed must reuse the cached program"
    assert cache_after_c > cache_after_b, (
        "different seed must produce a distinct cache entry — otherwise the second call "
        "silently reuses the first call's seed (identical dropout mask bug)"
    )
    assert not torch.equal(out_a, out_c), "different seeds must produce different dropout masks"
