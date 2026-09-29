# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The MiniMax-H3 per-head QK-norm split into groups of heads.

The fused per-head norm keeps the whole per-device row resident in L1. At TP=4 that is 14 heads
(1792 columns) and it fits; at TP=1 all 56 heads land on one chip, the row is 7168 columns, and the
op raises before its block-major layout can recover:

    Statically allocated circular buffers on core range [0-0 - 10-1] grow to 1862724 B which is
    beyond max L1 size of 1572864 B

Splitting the call into groups of heads is the same arithmetic on a narrower row -- a per-head norm
reduces inside one head and the RoPE tables are head_dim-wide, so nothing crosses a group boundary.
The device row below is the one that matters: it asserts the grouped result is what the ungrouped
one would have been, at a width where both still run.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.normalization import DistributedRMSNorm
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import (
    _PER_HEAD_NORM_MAX_COLS,
    per_head_norm_groups,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor

from .common import SMALL_LINE_PARALLEL

HEAD_DIM = 128
H3_HEADS = 56


def test_a_shard_that_already_fits_is_never_split():
    """Every mesh wide enough to shard the heads keeps exactly the call it had before."""
    for tp in (4, 8, 32):
        assert per_head_norm_groups(H3_HEADS // tp, HEAD_DIM) == 1
    assert per_head_norm_groups(1, HEAD_DIM) == 1


def test_a_one_chip_shard_is_split_to_the_width_a_galaxy_already_runs():
    groups = per_head_norm_groups(H3_HEADS, HEAD_DIM)
    assert groups == 4
    assert H3_HEADS % groups == 0
    assert H3_HEADS // groups * HEAD_DIM == _PER_HEAD_NORM_MAX_COLS


@pytest.mark.parametrize("heads", [56, 28, 14, 12, 7])
def test_the_split_is_always_the_fewest_groups_that_fit(heads):
    groups = per_head_norm_groups(heads, HEAD_DIM)
    assert heads % groups == 0, (heads, groups)
    assert heads // groups * HEAD_DIM <= _PER_HEAD_NORM_MAX_COLS, (heads, groups)
    # Fewest: no smaller group count would have fit.
    for fewer in range(1, groups):
        assert heads % fewer or heads // fewer * HEAD_DIM > _PER_HEAD_NORM_MAX_COLS


@SMALL_LINE_PARALLEL
def test_head_groups_match_the_ungrouped_norm(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp
):
    """Grouped == ungrouped, on device, at a width where the ungrouped call still runs.

    8 heads of 128 is 1024 columns, well inside L1 either way, so both legs are real and the
    comparison is of the split alone. `head_groups=1` is the pre-existing path by construction
    (the grouping is gated on `> 1`), which makes this a genuine before/after.
    """
    del sp_axis, device_params, is_fsdp
    heads, seq = 8, 64
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if heads % tp_factor:
        pytest.skip(f"{heads} heads do not divide TP={tp_factor}")
    heads_per_device = heads // tp_factor
    inner_dim = heads * HEAD_DIM

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    norm = DistributedRMSNorm(
        embedding_dim=inner_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        mesh_axis=tp_axis,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )
    torch.manual_seed(7)
    norm.load_torch_state_dict({"weight": torch.randn(inner_dim) * 0.1 + 1.0})

    # bf16_tensor requires mesh_axis and shard_dim to be given together or not at all; at TP=1 the
    # row is already whole, so neither applies.
    shard = {"mesh_axis": tp_axis, "shard_dim": -1} if tp_factor > 1 else {}
    x = bf16_tensor(torch.randn(1, 1, seq, inner_dim // tp_factor), device=mesh_device, **shard)
    kwargs = dict(num_heads_per_device=heads_per_device, per_head_norm=True)
    plain = ttnn.to_torch(ttnn.get_device_tensors(norm(x, **kwargs))[0]).to(torch.float32)

    for groups in (g for g in (2, 4) if heads_per_device % g == 0):
        grouped = ttnn.to_torch(ttnn.get_device_tensors(norm(x, head_groups=groups, **kwargs))[0]).to(torch.float32)
        assert grouped.shape == plain.shape, (groups, grouped.shape, plain.shape)
        rel = float(torch.linalg.vector_norm(grouped - plain) / torch.linalg.vector_norm(plain))
        logger.info(f"head_groups={groups}: relative RMSE vs the ungrouped norm {rel:.2e}")
        # Same op, same data, just fewer columns per launch -- this is exactness, not agreement.
        assert rel == 0.0, f"head_groups={groups} changed the result (relative RMSE {rel:.2e})"
