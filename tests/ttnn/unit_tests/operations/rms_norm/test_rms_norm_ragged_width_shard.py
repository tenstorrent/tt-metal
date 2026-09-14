# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests for WIDTH_SHARDED inputs whose shard grid does not divide W evenly.

The last shard is padded (its tile count exceeds the tiles left of W). The op must normalize over
the LOGICAL width: the padded tiles are poisoned here so any leak into the sum of squares is a
gross error, not a rounding one. Two geometries come from the production sharded test corpus
(W=96 over 2x64, W=224 over 3x96); the third is the eval harness's auto shard config for a
Llama-FFN width (W=11008 -> 115 cores x 3 tiles on a 13-wide grid, non-rectangular bounding box
with passive cores).
"""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

PAD_POISON = 1000.0
TILE = 32


def _reference(x_logical, gamma, eps):
    xf = x_logical.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma.float().reshape(-1)


def _run(device, tt_x, shape, gamma_torch, eps, dtype):
    w = shape[-1]
    tt_x = ttnn.fill_implicit_tile_padding(tt_x, PAD_POISON)
    tt_g = ttnn.from_torch(gamma_torch.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(tt_x, gamma=tt_g, epsilon=eps, memory_config=tt_x.memory_config())
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    x_logical = ttnn.to_torch(ttnn.to_memory_config(tt_x, ttnn.L1_MEMORY_CONFIG))[..., :w]
    got = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)).float()[..., :w]
    ref = _reference(x_logical, gamma_torch, eps)
    assert_with_pcc(ref, got, 0.999 if dtype == ttnn.float32 else 0.995)
    rel_rms = torch.sqrt(((got - ref) ** 2).mean()) / torch.sqrt((ref**2).mean())
    assert rel_rms.item() < (0.02 if dtype == ttnn.float32 else 0.04), f"rel RMS {rel_rms.item()} (padding leak?)"


@pytest.mark.parametrize(("w", "num_cores_w"), [(96, 2), (224, 3)], ids=["w96_c2", "w224_c3"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_uneven_width_shards(device, w, num_cores_w, dtype):
    torch.manual_seed(0)
    shape = (1, 1, 32, w)
    wt = -(-w // TILE)
    shard_wt = -(-wt // num_cores_w)
    assert (num_cores_w - 1) * shard_wt < wt
    x = torch.randn(shape)
    g = torch.randn(w).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    cfg = ttnn.create_sharded_memory_config(
        shape=(32, shard_wt * TILE),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_x = ttnn.to_memory_config(tt_x, memory_config=cfg)
    _run(device, tt_x, shape, g, 1e-5, dtype)


@pytest.mark.parametrize("shape", [(1, 1, 160, 11008), (1, 224, 11008)], ids=["4d_llama_ffn", "3d_llama_ffn"])
def test_rms_norm_auto_shard_padded_last_shard(device, shape):
    """auto_shard_config(PAD): Wt=344 -> ceil-split over the grid leaves a padded last shard and a
    non-rectangular shard grid (passive bounding-box cores in the multicast rectangle)."""
    torch.manual_seed(0)
    cfg = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ncores = len(ttnn.corerange_to_cores(cfg.shard_spec.grid, None, True))
    if ncores * cfg.shard_spec.shape[1] == shape[-1]:
        pytest.skip("this grid divides W evenly; the padded-last-shard regime is not reached")
    x = torch.randn(shape)
    g = torch.randn(shape[-1]).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    _run(device, tt_x, shape, g, 1e-6, ttnn.bfloat16)
