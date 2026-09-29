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


@pytest.mark.parametrize("heads", [56, 28, 14, 12, 7, 4, 3, 2])
def test_the_split_is_always_legal_and_the_fewest_that_fit(heads):
    groups = per_head_norm_groups(heads, HEAD_DIM)
    assert heads % groups == 0, (heads, groups)
    # Legal first: the op refuses a single-head call, so a group must carry at least two heads.
    # This is not hypothetical -- an earlier version fell back to one group per head and the 1x4
    # row turned an L1 overflow into "per_head_norm requires num_heads_per_device > 1".
    if groups > 1:
        assert heads // groups >= 2, (heads, groups)
    # Fewest that fit, among the legal counts.
    for fewer in range(1, groups):
        if heads % fewer == 0 and (fewer == 1 or heads // fewer >= 2):
            assert heads // fewer * HEAD_DIM > _PER_HEAD_NORM_MAX_COLS, (heads, groups, fewer)


def test_a_width_no_legal_split_can_fix_stays_legal():
    """If nothing legal gets under the budget, the answer must still be something the op accepts.

    Two heads of a very wide head_dim cannot be split at all (one head per group is refused), so
    the only correct answer is 1 -- an overflow the op may still handle, rather than a call it is
    guaranteed to reject.
    """
    assert per_head_norm_groups(2, _PER_HEAD_NORM_MAX_COLS) == 1
    # Three heads can only split 3 ways, which is one head each, so it cannot split either.
    assert per_head_norm_groups(3, _PER_HEAD_NORM_MAX_COLS) == 1


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
    # 8 heads PER DEVICE on every row, so the per-device row is the same 1024 columns whatever the
    # mesh and both the 2- and 4-way splits leave at least two heads in a group (the op refuses a
    # single-head call). A fixed total head count would give the 1x4 row only 2 heads per device,
    # where no legal split exists at all.
    seq = 64
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    heads_per_device = 8
    heads = heads_per_device * tp_factor
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

    # The tensor handed in is the FULL inner dim and the mesh mapper does the fracturing, so the
    # per-device row is inner_dim // tp_factor -- which is what the norm's own check expects. At
    # TP=1 there is nothing to fracture, and bf16_tensor wants mesh_axis and shard_dim given
    # together or not at all.
    shard = {"mesh_axis": tp_axis, "shard_dim": -1} if tp_factor > 1 else {}
    x = bf16_tensor(torch.randn(1, 1, seq, inner_dim), device=mesh_device, **shard)
    kwargs = dict(num_heads_per_device=heads_per_device, per_head_norm=True)
    plain = ttnn.to_torch(ttnn.get_device_tensors(norm(x, **kwargs))[0]).to(torch.float32)

    # Groups of at least two heads: the op rejects a single-head call, which is also why
    # per_head_norm_groups never produces one.
    tried = [g for g in (2, 4) if heads_per_device % g == 0 and heads_per_device // g >= 2]
    assert tried, f"no legal split to test at {heads_per_device} heads per device"

    for groups in tried:
        grouped = ttnn.to_torch(ttnn.get_device_tensors(norm(x, head_groups=groups, **kwargs))[0]).to(torch.float32)
        assert grouped.shape == plain.shape, (groups, grouped.shape, plain.shape)
        rel = float(torch.linalg.vector_norm(grouped - plain) / torch.linalg.vector_norm(plain))
        logger.info(f"head_groups={groups}: relative RMSE vs the ungrouped norm {rel:.2e}")
        # Same op, same data, just fewer columns per launch -- this is exactness, not agreement.
        assert rel == 0.0, f"head_groups={groups} changed the result (relative RMSE {rel:.2e})"
