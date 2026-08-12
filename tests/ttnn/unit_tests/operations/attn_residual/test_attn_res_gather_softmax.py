# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC for the AttnRes read: statistics, the cross-rank gather, and the fold, in one op.

The whole read is one program, so nothing inside it is separately observable and the
gate is fp32 torch over the unsharded `d`.

Sharded the way the model shards, at the shape the model runs: 1280 tokens over a
2-deep sequence axis and `d` over a 4-deep tensor-parallel axis, so every chip holds 640
rows of a 1792-wide shard. `tp_factor == 1` would make the gather an identity and certify
nothing, so there is no single-device arm, and the row count is not swept — the op's cost
and its collective's algorithm both turn on it.

The `settle` arms hand in a deferred residual write, which the op folds by distributing
the row weight over the two addends rather than by summing them first. The stream it
hands back has to equal the `ttnn.add` it saves the caller *exactly*, not approximately,
because the walk chains 93 of them — so that half is gated against `ttnn.add` itself.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="attn_res_gather_softmax has only been brought up on Blackhole"
)

# bfloat16 in and out with one rounding at the pack.
PCC = 0.9999

HIDDEN_SIZE = 7168
INV_HIDDEN_SIZE = 1.0 / HIDDEN_SIZE
EPS = 1e-6
PER_CHIP_TOKENS = 640

TP_AXIS = 1
SP_AXIS = 0

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _oracle(partial, prefix_sum, shift, mass, query):
    """The whole read in fp32, taken against the unsharded `d`."""
    partial, prefix_sum = partial.float(), prefix_sum.float()
    shift, mass, query = shift.float(), mass.float(), query.float()

    sum_squares = (prefix_sum * prefix_sum).sum(dim=-1, keepdim=True)
    dots = (prefix_sum * query).sum(dim=-1, keepdim=True)
    live_scores = dots * torch.rsqrt(sum_squares * INV_HIDDEN_SIZE + EPS)

    merged_shift = torch.maximum(shift, live_scores)
    rescale = torch.exp(shift - merged_shift)
    live_weight = torch.exp(live_scores - merged_shift)
    return (partial * rescale + prefix_sum * live_weight) / (mass * rescale + live_weight)


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((2, 4), FABRIC, id="mesh-2x4")], indirect=True)
@pytest.mark.parametrize("fuse_add", [False, True], ids=["plain", "settle"])
def test_matches_torch(mesh_device, device_params, fuse_add):
    torch.manual_seed(2026)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    num_tokens = PER_CHIP_TOKENS * sp_factor

    stream_dims, vector_dims, scalar_dims = [None, None], [None, None], [None, None]
    stream_dims[SP_AXIS], stream_dims[TP_AXIS] = 2, 3
    vector_dims[TP_AXIS] = 3
    scalar_dims[SP_AXIS] = 2

    stream_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=stream_dims, mesh_shape=mesh_shape)
    vector_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=vector_dims, mesh_shape=mesh_shape)
    scalar_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=scalar_dims, mesh_shape=mesh_shape)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_shape)

    shape = [1, 1, num_tokens, HIDDEN_SIZE]
    partial = torch.randn(shape, dtype=torch.bfloat16)
    # The live stream is scaled so the sum of squares over the full `d` lands near
    # `HIDDEN_SIZE`, which is the range `inv_hidden_size` normalizes against.
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    pending = torch.randn(shape, dtype=torch.bfloat16) if fuse_add else None
    # What the op scores and folds against: `prefix_sum` alone, or the sum it settles.
    stream = prefix_sum.float() + pending.float() if fuse_add else prefix_sum.float()
    query = torch.randn([1, 1, 1, HIDDEN_SIZE], dtype=torch.bfloat16) * 0.05
    # `mass` is a sum of exponentials against a running maximum, so it is at least
    # one; drawn around zero it would put the denominator near zero and make the gate
    # measure cancellation instead of the op.
    shift = torch.randn([1, 1, num_tokens, 1]) * 2.0
    mass = torch.rand([1, 1, num_tokens, 1]) * 7.0 + 1.0

    to_dev = lambda t, mapper, dtype=ttnn.bfloat16: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )
    tt_partial = to_dev(partial, stream_mapper)
    tt_prefix = to_dev(prefix_sum, stream_mapper)
    tt_pending = to_dev(pending, stream_mapper) if fuse_add else None
    tt_query = to_dev(query, vector_mapper)
    tt_shift = to_dev(shift, scalar_mapper, ttnn.float32)
    tt_mass = to_dev(mass, scalar_mapper, ttnn.float32)

    # Caller-allocated exchange scratch: one sum-of-squares and one dots plane per
    # rank, replicated across the tensor-parallel axis so a page has the same address
    # on every chip of it.
    tt_stats = to_dev(
        torch.zeros([1, 2 * tp_factor, num_tokens, 1]),
        scalar_mapper,
        ttnn.float32,
    )

    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        mesh_device.compute_with_storage_grid_size().x - 1,
                        mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            ]
        ),
        0,
    )

    fused = ttnn.experimental.attn_res_gather_softmax(
        tt_partial,
        tt_prefix,
        tt_shift,
        tt_mass,
        tt_query,
        tt_stats,
        semaphore,
        cluster_axis=TP_AXIS,
        inv_hidden_size=INV_HIDDEN_SIZE,
        eps=EPS,
        pending=tt_pending,
    )
    assert len(fused) == (2 if fuse_add else 1), f"{len(fused)} outputs at fuse_add={fuse_add}"

    # Every chip of a tensor-parallel row holds a different `d` shard and every chip
    # of a sequence column a different token block, so the composer's concat is the
    # whole tensor back — no slicing.
    got = ttnn.to_torch(fused[0], mesh_composer=composer)
    want = _oracle(partial, stream, shift, mass, query)

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    _, vs_torch = assert_with_pcc(want, got.float(), PCC)
    logger.info(f"fused vs torch: {vs_torch}")

    if fuse_add:
        # The stream the caller carries forward, against the dispatch it saves them.
        tt_added = ttnn.add(tt_prefix, tt_pending)
        settled = ttnn.to_torch(fused[1], mesh_composer=composer)
        added = ttnn.to_torch(tt_added, mesh_composer=composer)
        assert torch.equal(settled, added), f"max|settled - add| = {(settled.float() - added.float()).abs().max():.6e}"
        _, vs_torch_stream = assert_with_pcc(stream, settled.float(), PCC)
        logger.info(f"settled stream vs torch: {vs_torch_stream}, bit-identical to ttnn.add")


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((2, 4), FABRIC, id="mesh-2x4")], indirect=True)
def test_rejects_a_site_past_the_batch_on_a_cache_hit(mesh_device, device_params, expect_error):
    """`site` shapes no kernel and is kept out of the program hash, so the second call
    below is a cache hit and never reaches the validation the first one passed. Without
    a check on that path the factory turns the bad site into page offsets and reads past
    the operands it was handed. Numerics are gated above; this one only needs a batch,
    so it runs at the smallest shape that still shards four ways."""
    torch.manual_seed(2026)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    num_tokens, hidden, sites = 32 * sp_factor, 32 * tp_factor, 2

    stream_dims, vector_dims, scalar_dims = [None, None], [None, None], [None, None]
    stream_dims[SP_AXIS], stream_dims[TP_AXIS] = 2, 3
    vector_dims[TP_AXIS] = 3
    scalar_dims[SP_AXIS] = 2

    to_dev = lambda t, dims, dtype=ttnn.bfloat16: ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape),
    )
    stream_shape, scalar_shape = [sites, 1, num_tokens, hidden], [sites, 1, num_tokens, 1]
    tt_partial = to_dev(torch.randn(stream_shape, dtype=torch.bfloat16), stream_dims)
    tt_prefix = to_dev(torch.randn([1, 1, num_tokens, hidden], dtype=torch.bfloat16), stream_dims)
    tt_query = to_dev(torch.randn([1, 1, 1, hidden], dtype=torch.bfloat16), vector_dims)
    tt_shift = to_dev(torch.randn(scalar_shape), scalar_dims, ttnn.float32)
    tt_mass = to_dev(torch.rand(scalar_shape) + 1.0, scalar_dims, ttnn.float32)
    tt_stats = to_dev(torch.zeros([1, 2 * tp_factor, num_tokens, 1]), scalar_dims, ttnn.float32)

    grid = mesh_device.compute_with_storage_grid_size()
    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
        0,
    )

    read = lambda site: ttnn.experimental.attn_res_gather_softmax(
        tt_partial,
        tt_prefix,
        tt_shift,
        tt_mass,
        tt_query,
        tt_stats,
        semaphore,
        cluster_axis=TP_AXIS,
        inv_hidden_size=INV_HIDDEN_SIZE,
        eps=EPS,
        site=site,
    )

    entries_before = mesh_device.num_program_cache_entries()
    read(sites - 1)
    assert mesh_device.num_program_cache_entries() > entries_before, "the first read did not populate the cache"

    with expect_error(RuntimeError, f"site {sites} is past partial's dim 0 of {sites}"):
        read(sites)
