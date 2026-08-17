# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement, composition and one block's reads — what every device gate here needs
before it can measure anything.

The shape is fixed rather than parametrized. `d = 7168` at 640 rows per chip is what
prefill runs — it chunks 5120 tokens across an 8-deep sequence axis on the Galaxy — and
the op's cost and its collective's choice of algorithm both turn on that row count, so a
gate parametrized at 64 tokens exercises a reduction the model never issues.

Inputs are drawn from a caller-supplied generator rather than a seed, so a file that
draws several things from one stream keeps its own draw order and stays reproducible
across changes here.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

HIDDEN_SIZE = 7168
PER_CHIP_TOKENS = 640

# On a unit-RMS stream the scores are `N(0, ‖q‖₂²)`, so the folded query's norm is the
# softmax's temperature and the only thing about the weights this op can be sensitive to.
# This lands at `‖q‖₂ ≈ 1.7`; K3's own query weights run 0.07 to 0.23 over a block, which
# is a near-uniform softmax and a milder shift for the online rescale to carry. The scale
# here is kept above the checkpoint's deliberately — it is the harder of the two.
PROJ_STD = 0.02

# `ttnn.all_reduce` and the gather kernel's own fabric writes both need an initialized
# fabric context on a real mesh; without it they die in the control plane rather than
# returning wrong numbers. 2D is what the rest of this model runs on and what the op's
# own unit test pins, and the two must agree — the op picks its route from the config.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}

# The op was brought up and measured only on Blackhole, and its mixture runs on
# `ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`, which has no Wormhole coverage.
blackhole_only = pytest.mark.skipif(not is_blackhole(), reason="Kimi K3 AttnRes is brought up on Blackhole only")


def generator(seed=0):
    return torch.Generator().manual_seed(seed)


def random_hidden(rng, num_tokens):
    return torch.randn(num_tokens, HIDDEN_SIZE, generator=rng)


def random_case(rng, num_tokens, num_sealed):
    """One read's inputs: the live stream and `num_sealed` frozen snapshots."""
    return random_hidden(rng, num_tokens), torch.randn(num_tokens, num_sealed, HIDDEN_SIZE, generator=rng)


def random_queries(rng, count):
    """`count` folded queries, each a norm weight times a projection row."""
    randn = lambda: torch.randn(HIDDEN_SIZE, generator=rng)
    return [(1.0 + 0.1 * randn()) * (PROJ_STD * randn()) for _ in range(count)]


def place(op, tensor, mesh_mapper=None):
    """One host tensor onto the mesh, in the stream's dtype and layout."""
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=op.mesh_device,
        mesh_mapper=op.stream_mapper if mesh_mapper is None else mesh_mapper,
    )


def place_case(op, running_sum, block_residual, mesh_mapper=None):
    """The live stream as `[1, 1, N, d]` and the sealed set as `[1, S, N, d]`.

    The reference holds the sealed set `[N, S, d]` and the read batches over `S`, so the
    two leading axes swap on the way across. An empty sealed set has no device tensor at
    all — `merge` takes None there rather than a zero-wide operand.
    """
    stream = place(op, running_sum.unsqueeze(0).unsqueeze(0), mesh_mapper)
    sealed = place(op, block_residual.permute(1, 0, 2).unsqueeze(0), mesh_mapper) if block_residual.shape[1] else None
    return stream, sealed


def compose(op, tensor):
    """A device tensor back on host as `[rows, d]`, its shards joined."""
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)


def read_block(op, tt_block, tt_prefix, tt_queries):
    """Every read site of one block on host, in the order the walk issues them.

    One `inter_block` over the sealed set, then a `merge` per site: the sealed half is
    loop-invariant across a block and only the live half is not. Each site is freed as
    soon as it is composed, so the block holds one output at a time — a caller that needs
    them together builds its own list.
    """
    partials, shifts, masses = op.inter_block(tt_block, tt_queries)
    try:
        for site, tt_query in enumerate(tt_queries):
            merged = op.merge(partials, shifts, masses, tt_prefix, tt_query, site)
            yield compose(op, merged)
            ttnn.deallocate(merged)
    finally:
        for tensor in (partials, shifts, masses):
            ttnn.deallocate(tensor)
