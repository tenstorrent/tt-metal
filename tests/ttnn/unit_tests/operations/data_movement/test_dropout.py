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


@pytest.mark.parametrize("in_place", [False, True])
def test_dropout_seed_distinguishes_cache_entries(device, in_place):
    """Regression guard for dropout's seed static/dynamic contract, on both the distinct-output and
    in-place (output_tensor==input) fast paths.

    `seed` is excluded from compute_program_hash (so calls differing only in seed cache-hit) but
    must be re-applied to the cached program on every dispatch. Pins both halves:
      * different seed must NOT grow the cache  -> guards against re-adding seed to the hash.
      * different seed must change the dropout mask -> guards against the frozen-seed bug on the
        fast path (the in_place case is only reachable since resolve_bindings allows input==output).
    """
    t = torch.ones((1, 1, 32, 64))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    device.enable_program_cache()
    device.clear_program_cache()

    def run(seed):
        out_kwargs = {"output_tensor": tensor} if in_place else {}
        out = ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=seed, **out_kwargs)
        return ttnn.to_torch(out).float().clone()

    out_a = run(1234)
    entries_a = device.num_program_cache_entries()
    run(1234)
    entries_b = device.num_program_cache_entries()
    out_c = run(5678)
    entries_c = device.num_program_cache_entries()

    assert entries_a == 1, "the first dispatch must create exactly one cache entry"
    assert entries_b == entries_a, "same seed must reuse the cached program"
    assert entries_c == entries_a, "a different seed must NOT add a cache entry -- seed is dynamic, not hashed"
    assert not torch.equal(out_a, out_c), "a different seed must change the dropout mask (seed re-patched on fast path)"

    device.disable_and_clear_program_cache()


def test_dropout_cache_hit_rederives_buffer_address(device):
    """On a cache hit the override must re-derive the input/output buffer addresses from the current
    tensors (not reuse the first-miss addresses). Reallocate a fresh input with different VALUES between
    two same-shape dispatches: the kept (non-dropped) outputs must reflect the second input, proving the
    program read the new buffer address."""
    device.enable_program_cache()
    device.clear_program_cache()

    scale = 2.0
    shape = (1, 1, 32, 64)

    a = ttnn.from_torch(torch.ones(shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.to_torch(ttnn.experimental.dropout(a, probability=0.5, scale=scale, seed=1234))  # miss

    b = ttnn.from_torch(torch.full(shape, 3.0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out_b = ttnn.to_torch(ttnn.experimental.dropout(b, probability=0.5, scale=scale, seed=1234)).float()

    assert device.num_program_cache_entries() == 1, "same-shape dispatch must reuse the cached program"
    nonzero = out_b[out_b != 0.0]
    assert nonzero.numel() > 0, "expected some elements to survive dropout"
    assert torch.allclose(
        nonzero, torch.full_like(nonzero, 3.0 * scale)
    ), "kept outputs must be 3.0*scale -- proves the hit re-derived the new input buffer address"

    device.disable_and_clear_program_cache()
