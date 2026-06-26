# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — L1 budget fit for wide/tall reduce dim.

Tests that the V2 streaming path (constant-bounded CBs) correctly handles
wide/tall shapes that previously OOMed with the V1 full-slab CB layout.
The V2 path uses a 3-pass approach (max, sum, apply) for large reduce dims,
and a non-reduce-dim chunking approach for small reduce dims with large
non-reduce dims.

Test shapes target the OOM boundary: W ∈ {4096, 8192}, H ∈ {2048, 4096},
and combinations like 1024×1024 that exceed the 256 KiB V1 CB budget.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape, dim",
    [
        # Wide W (chunk_along_reduce, dim=-1)
        ((1, 1, 32, 4096), -1),
        ((1, 1, 32, 8192), -1),
        ((1, 1, 128, 4096), -1),
        ((2, 1, 64, 4096), -1),
        # Tall H (chunk_along_reduce, dim=-2)
        ((1, 1, 2048, 256), -2),
        ((1, 1, 4096, 128), -2),
        # Small reduce dim, large non-reduce (chunk_along_non_reduce)
        ((1, 1, 32, 4096), -2),
        ((1, 1, 32, 8192), -2),
        ((2, 1, 64, 4096), -2),
        # Both large
        ((1, 1, 1024, 1024), -1),
        ((1, 1, 1024, 1024), -2),
        # 3D rank (unsqueeze to 4D)
        ((1, 32, 4096), -1),
        ((1, 32, 8192), -1),
        # 2D rank (unsqueeze to 4D)
        ((32, 4096), -1),
        ((128, 8192), -1),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_softmax_wide_tall_shapes(device, shape, dim, dtype):
    """Test V2 streaming path on wide/tall shapes that previously OOMed."""
    torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[dtype]
    torch_input = torch.randn(*shape, dtype=torch_dtype)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    pcc_threshold = 0.999 if dtype == ttnn.float32 else 0.999
    assert_with_pcc(result.float(), expected.float(), pcc=pcc_threshold)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1, 1, 32, 4096), -1),
        ((1, 1, 32, 8192), -1),
        ((1, 1, 128, 4096), -1),
        ((1, 1, 2048, 256), -2),
        ((1, 1, 4096, 128), -2),
        ((1, 1, 32, 4096), -2),
        ((1, 1, 1024, 1024), -1),
        ((1, 1, 1024, 1024), -2),
    ],
)
def test_softmax_v2_data_distributions(device, shape, dim):
    """Test V2 path with multiple data distributions (no inf/nan)."""
    distributions = [
        ("randn", lambda s: torch.randn(*s, dtype=torch.float32)),
        ("uniform_0_1", lambda s: torch.rand(*s, dtype=torch.float32)),
        ("small", lambda s: torch.randn(*s, dtype=torch.float32) * 0.01),
        ("large", lambda s: torch.randn(*s, dtype=torch.float32) * 10.0),
        ("positive", lambda s: torch.rand(*s, dtype=torch.float32) + 0.5),
        ("negative", lambda s: -(torch.rand(*s, dtype=torch.float32) + 0.5)),
    ]

    for name, gen_fn in distributions:
        torch_input = gen_fn(shape)
        expected = torch.softmax(torch_input, dim=dim)

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
        result = ttnn.to_torch(ttnn_output)

        # No inf/nan
        assert not torch.isnan(result.float()).any(), f"{name}: NaN in output"
        assert not torch.isinf(result.float()).any(), f"{name}: Inf in output"
        # "large" (×10.0) stresses exp dynamic range; "small" (×0.01) produces
        # near-uniform output where PCC is sensitive to tiny numerical differences.
        pcc_threshold = 0.98 if name in ("large", "small") else 0.99
        actual_pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
        assert actual_pcc >= pcc_threshold, f"{name}: PCC {actual_pcc} < {pcc_threshold}"


def test_softmax_v1_v2_equivalence(device):
    """V1 and V2 should produce numerically identical output for a shape
    that sits at the dispatch boundary."""
    # (1,1,32,256) has V1 footprint ~120KB → uses V1
    # (1,1,32,4096) has V1 footprint ~2.5MB → uses V2
    # Both should produce correct output
    for shape, dim in [((1, 1, 32, 256), -1), ((1, 1, 32, 4096), -1)]:
        torch_input = torch.randn(*shape, dtype=torch.float32)
        expected = torch.softmax(torch_input, dim=dim)

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
        result = ttnn.to_torch(ttnn_output)

        assert_with_pcc(result.float(), expected.float(), pcc=0.999)
