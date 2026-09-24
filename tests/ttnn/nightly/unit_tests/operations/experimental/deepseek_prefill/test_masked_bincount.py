# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Cache-hit coverage for ttnn.experimental.deepseek_prefill.masked_bincount.

The histogram must follow the buffers of the current call. A program-cache hit that
kept the first call's addresses would count the wrong expert ids or the wrong mask.
"""

import pytest
import torch

import ttnn


def _golden(indices, mask, n_routed_experts, num_experts_per_token):
    counts = torch.zeros(n_routed_experts, dtype=torch.int64)
    selected = indices[:, :num_experts_per_token].reshape(-1)
    for expert_idx in selected.tolist():
        if expert_idx < n_routed_experts and mask[expert_idx] >= 0:
            counts[expert_idx] += 1
    return counts


def _run(device, indices, mask, n_routed_experts, num_experts_per_token):
    tt_indices = ttnn.from_torch(indices.to(torch.int16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.experimental.deepseek_prefill.masked_bincount(
        tt_indices, tt_mask, n_routed_experts, num_experts_per_token
    )
    return [tt_indices, tt_mask, output], ttnn.to_torch(output).reshape(-1).to(torch.int64)


def test_masked_bincount_program_cache_hit_patches_addresses(device):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("masked_bincount places work on a fixed 8x8 core grid")

    device.enable_program_cache()
    n_routed_experts = 8
    num_experts_per_token = 6
    tokens = 64
    mask = torch.tensor([0, 0, -1, 1, -1, 2, 0, 3], dtype=torch.int32)
    retained = []
    histograms = []
    entries_after_miss = None
    # Column 0 is present, column 2 is masked. The last column is the sentinel index and must not be counted.
    patterns = (0, 2)

    for fill in patterns:
        indices = torch.full((tokens, num_experts_per_token), fill, dtype=torch.int64)
        indices[:, -1] = n_routed_experts
        before = device.num_program_cache_entries()
        buffers, actual = _run(device, indices, mask, n_routed_experts, num_experts_per_token)
        if entries_after_miss is None:
            entries_after_miss = device.num_program_cache_entries()
            assert entries_after_miss > before
        else:
            assert device.num_program_cache_entries() == entries_after_miss
        retained.extend(buffers)
        expected = _golden(indices, mask, n_routed_experts, num_experts_per_token)
        assert torch.equal(actual, expected)
        histograms.append(actual.clone())

    assert not torch.equal(histograms[0], histograms[1])
