# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Host-only test for patch_merger.

The patch merger is pure spatial rearrangement (no compute) on host,
so it should match the HF reference bit-exactly. We also confirm the
flatten variant produces the right shape for the projector input.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_patch_merger.py -v
"""
from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.patch_merger import patch_merger, patch_merger_per_image


@torch.no_grad()
@pytest.mark.parametrize(
    "grid_hws",
    [
        [[16, 16]],  # single image, 256 tokens.
        [[32, 24]],  # asymmetric.
        [[16, 16], [32, 24]],  # multi-image packed.
    ],
)
def test_patch_merger_matches_hf(model_args, grid_hws):
    """Per-image output matches HF; concatenated output matches HF cat."""
    pcc_threshold = 0.9999

    ref_fn = model_args.reference_patch_merger()  # the HF patch_merger free function.

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())
    D = model_args.hidden_size

    torch.manual_seed(0)
    x_pt = torch.randn(L, D, dtype=torch.bfloat16)

    # HF reference returns a list of (L_new_i, kh*kw, D).
    ref_list = ref_fn(x_pt, grid_tensor, merge_kernel_size=model_args.merge_kernel_size)

    # Per-image variant matches HF list element-wise.
    ours_list = patch_merger_per_image(x_pt, grid_tensor, model_args.merge_kernel_size)
    assert len(ours_list) == len(ref_list)
    for i, (r, o) in enumerate(zip(ref_list, ours_list)):
        assert r.shape == o.shape, f"image {i}: ref shape {r.shape} != ours {o.shape}"
        # Pure reshape on identical data — should be byte-identical.
        assert torch.equal(r, o), f"image {i}: ref vs ours not byte-identical"

    # Concat-then-flatten variant has expected output shape.
    flat = patch_merger(x_pt, grid_tensor, model_args.merge_kernel_size, flatten=True)
    expected_l_new = sum(h // 2 * (w // 2) for h, w in grid_hws)
    assert flat.shape == (expected_l_new, model_args.merge_dim)

    # PCC against concat of HF list (both flattened the same way).
    ref_cat = torch.cat(ref_list, dim=0)
    ref_flat = ref_cat.reshape(expected_l_new, -1)
    passing, pcc_msg = comp_pcc(ref_flat.float(), flat.float(), pcc_threshold)
    logger.info(f"[grid_hws={grid_hws}] {comp_allclose(ref_flat, flat)} {pcc_msg}")
    assert passing, f"PCC mismatch for grid_hws={grid_hws}: {pcc_msg}"
