# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC for the AttnRes read: statistics, the cross-rank gather, and the fold, in one op.

The whole read is one program, so nothing inside it is separately observable and the
gate is fp32 torch over the unsharded `d`.

Sharded the way the model shards, at the shape the model runs: 1280 tokens over a
2-deep sequence axis and `d` over a tensor-parallel axis, so every chip holds 640 rows.
The 2x4 arm gathers at the model's own width, 1792 columns per chip. The gather is a
fabric collective, so at `tp_factor == 1` there are no peers and half the op never
runs — there is no
single-device arm, and the row count is not swept, since the op's cost and its
collective's algorithm both turn on it.

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
# `rms_norm_eps` from the K3 checkpoint's own config, so the op is exercised at the
# magnitude the model hands it rather than at one that never reaches the kernel.
EPS = 1e-5
PER_CHIP_TOKENS = 640

TP_AXIS = 1
SP_AXIS = 0

# Each arm names the whole box it runs on, so the one that does not match this machine has
# to drop out rather than open a submesh and stall in the ethernet handshake.
EXACT_BOX = {"require_exact_physical_num_devices": True}

FABRIC_2D = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, **EXACT_BOX}

# Galaxy's production fabric wraps both axes, and a wrapped axis is the only place where
# the direction a peer's route takes and the direction its rank suggests come apart: the
# shorter way round to a higher rank runs the other way. The gather has to open its
# connection on the route's direction, and this is the only arm that can tell whether it
# did -- an unwrapped mesh agrees with rank order and passes either way.
TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    **EXACT_BOX,
}

# Wrapping only the TP axis is the smallest configuration that can expose a wrap-direction
# bug, because that is the axis the collective runs on -- the SP axis stays a line and
# contributes no wrap link the gather could take. Keeping it alongside the both-axes profile
# is what pins that the SP wrap is not what carries the result: routing is dimension-ordered,
# so a peer with no SP displacement takes its first hop along TP whatever the SP axis does,
# and the invariant test below holds that by sweeping every same-row route for one that left.
TORUS_X = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    **EXACT_BOX,
}

# Every test derives its shape from the fixture's mesh, and a mesh narrower than the box it
# opens on is a submesh, which fabric does not come up on -- so each box contributes its
# own full-width arm rather than a slice. Galaxy runs both fabrics: unwrapped 2D widens the
# TP axis without wrapping it, which separates a width bug from a wrap bug.
#
# The fabric is chosen once per program run, before the mesh opens, so the op runs under
# whichever one the enclosing transformer picked.
MESH_ARMS = [
    pytest.param((2, 4), FABRIC_2D, id="fabric2d-mesh-2x4"),
    pytest.param((8, 4), FABRIC_2D, id="fabric2d-mesh-8x4"),
    pytest.param((8, 4), TORUS_X, id="torusx-mesh-8x4"),
    pytest.param((8, 4), TORUS_XY, id="torusxy-mesh-8x4"),
]


def _oracle(partial, running_sum, shift, mass, query):
    """The whole read in fp32, taken against the unsharded `d`."""
    partial, running_sum = partial.float(), running_sum.float()
    shift, mass, query = shift.float(), mass.float(), query.float()

    sum_squares = (running_sum * running_sum).sum(dim=-1, keepdim=True)
    dots = (running_sum * query).sum(dim=-1, keepdim=True)
    live_scores = dots * torch.rsqrt(sum_squares * INV_HIDDEN_SIZE + EPS)

    merged_shift = torch.maximum(shift, live_scores)
    rescale = torch.exp(shift - merged_shift)
    live_weight = torch.exp(live_scores - merged_shift)
    return (partial * rescale + running_sum * live_weight) / (mass * rescale + live_weight)


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
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
    running_sum = torch.randn(shape, dtype=torch.bfloat16)
    pending = torch.randn(shape, dtype=torch.bfloat16) if fuse_add else None
    # What the op scores and folds against: `running_sum` alone, or the sum it settles.
    stream = running_sum.float() + pending.float() if fuse_add else running_sum.float()
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
    tt_prefix = to_dev(running_sum, stream_mapper)
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

    fused = ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
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


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
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

    read = lambda site: ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
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


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
def test_every_site_reads_its_own_plane_from_one_cached_program(mesh_device, device_params):
    """A walk issues 186 reads and every one of them is a cache hit after the first, so
    the site has to reach the kernels as a patched runtime arg. A site left at the value
    the program was built with returns the wrong plane and still returns it silently —
    the shapes match and nothing traps. Each site is checked against its own plane, and
    the cache is checked to have grown once across the whole sweep."""
    torch.manual_seed(2026)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    num_tokens, sites = PER_CHIP_TOKENS * sp_factor, 3

    stream_dims, vector_dims, scalar_dims = [None, None], [None, None], [None, None]
    stream_dims[SP_AXIS], stream_dims[TP_AXIS] = 2, 3
    vector_dims[TP_AXIS] = 3
    scalar_dims[SP_AXIS] = 2

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_shape)
    to_dev = lambda t, dims, dtype=ttnn.bfloat16: ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape),
    )

    # Every plane is drawn independently, so a read that returns a neighbour's plane
    # misses by the full scale of the data rather than by a roundoff.
    partial = torch.randn([sites, 1, num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16)
    shift = torch.randn([sites, 1, num_tokens, 1]) * 2.0
    mass = torch.rand([sites, 1, num_tokens, 1]) * 7.0 + 1.0
    running_sum = torch.randn([1, 1, num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16)
    query = torch.randn([1, 1, 1, HIDDEN_SIZE], dtype=torch.bfloat16) * 0.05

    tt_partial = to_dev(partial, stream_dims)
    tt_prefix = to_dev(running_sum, stream_dims)
    tt_query = to_dev(query, vector_dims)
    tt_shift = to_dev(shift, scalar_dims, ttnn.float32)
    tt_mass = to_dev(mass, scalar_dims, ttnn.float32)
    tt_stats = to_dev(torch.zeros([1, 2 * tp_factor, num_tokens, 1]), scalar_dims, ttnn.float32)

    grid = mesh_device.compute_with_storage_grid_size()
    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
        0,
    )

    entries_before = mesh_device.num_program_cache_entries()
    for site in range(sites):
        fused = ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
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
        got = ttnn.to_torch(fused[0], mesh_composer=composer)
        want = _oracle(partial[site : site + 1], running_sum, shift[site : site + 1], mass[site : site + 1], query)
        _, vs_torch = assert_with_pcc(want, got.float(), PCC)
        logger.info(f"site {site} vs torch: {vs_torch}")

    grew_by = mesh_device.num_program_cache_entries() - entries_before
    assert grew_by == 1, f"{sites} sites built {grew_by} programs; the site must not key the cache"


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
@pytest.mark.parametrize(
    "bad, message",
    [
        ("rows_same_tile_bucket", "requires an unbatched running_sum matching partial's plane"),
        ("unaligned_width", "requires tile-aligned inner dims"),
    ],
    ids=["rows_same_tile_bucket", "unaligned_width"],
)
def test_rejects_shapes_that_only_agree_once_padded(mesh_device, device_params, expect_error, bad, message):
    """Padding is not guaranteed zero, and both arms below would read it as data.

    Row counts inside one tile bucket compare equal padded while the output is labelled
    with the logical count, so the shorter operand's padding lands in rows the caller
    reads back. An unaligned width is worse: the statistics reduce runs the whole padded
    row, so the padding sets the score for every row rather than for some of them."""
    torch.manual_seed(2026)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    num_tokens = 64 * sp_factor
    # 40 columns per chip, which pads to 64 — the only arm that needs an unaligned shard.
    hidden = (40 if bad == "unaligned_width" else 32) * tp_factor
    # 50 rows per chip against partial's 64: the same two tiles, a different row count.
    prefix_tokens = 50 * sp_factor if bad == "rows_same_tile_bucket" else num_tokens

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
    tt_partial = to_dev(torch.randn([1, 1, num_tokens, hidden], dtype=torch.bfloat16), stream_dims)
    tt_prefix = to_dev(torch.randn([1, 1, prefix_tokens, hidden], dtype=torch.bfloat16), stream_dims)
    tt_query = to_dev(torch.randn([1, 1, 1, hidden], dtype=torch.bfloat16), vector_dims)
    tt_shift = to_dev(torch.randn([1, 1, num_tokens, 1]), scalar_dims, ttnn.float32)
    tt_mass = to_dev(torch.rand([1, 1, num_tokens, 1]) + 1.0, scalar_dims, ttnn.float32)
    tt_stats = to_dev(torch.zeros([1, 2 * tp_factor, num_tokens, 1]), scalar_dims, ttnn.float32)

    grid = mesh_device.compute_with_storage_grid_size()
    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
        0,
    )

    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
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
        )


# The op sends only on the TP axis, so these are the two fabrics that put a ring under it.
TP_AXIS_WRAPPED = frozenset({ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY})


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
def test_the_route_to_a_peer_contradicts_its_rank_once_the_tp_axis_wraps(mesh_device, device_params):
    """The direction a peer's packet leaves on is the routing tables' to choose, not rank order's.

    The gather unicasts to each peer on the TP axis and has to open its outgoing fabric
    connection on the direction that peer's route actually takes. Ordering peers by rank
    happens to agree with that on a line, which is why an unwrapped fabric cannot tell the
    two rules apart. Wrap the axis and they come apart at the ends: the far peer is one hop
    the short way round and `cols - 1` hops the way its rank suggests. A connection opened
    on the wrong direction never delivers, and the gather's arrival semaphore never fills.

    The wrap is classified from one route on row 0 and then held against the request, so an
    arm cannot pass on a fabric it did not ask for -- auto-discovery answers a torus request
    it cannot cable by substituting a lesser one behind a log warning, and the mesh graph
    descriptor that would name the substitute is absent on exactly that path. Every
    remaining route is a separate claim: the line arm pins that rank order is genuinely
    indistinguishable there, and the wrapped arms pin that it is genuinely wrong.

    The sweep also pins that no same-row route leaves the TP axis. Routing is
    dimension-ordered, so a destination with zero displacement on the SP axis takes its
    first hop along TP whatever the SP axis's own topology is. That is what makes the
    Line/Ring fabric a faithful stand-in for Ring/Ring here: it hands the collective the
    same ring, and the wrap it leaves out is one this op never traverses.
    """
    rows, cols = tuple(mesh_device.shape)
    assert cols > 2, "a wrap is only distinguishable from rank order when the far peer is not adjacent"

    shapes = ttnn.get_physical_mesh_shapes()
    assert len(shapes) == 1, f"expected a single local mesh, got {shapes}"
    mesh_id = ttnn.MeshId(next(iter(shapes)))

    def node(row, col):
        return ttnn.FabricNodeId(mesh_id, row * cols + col)

    def route(row, src, dst):
        direction = ttnn.get_eth_forwarding_direction(node(row, src), node(row, dst))
        assert direction is not None, f"no route on row {row} from column {src} to {dst}"
        return direction

    far = cols - 1
    forward, backward = route(0, 0, 1), route(0, 1, 0)
    assert forward != backward, "row 0 routes both ways along TP over one direction"

    wrapped = route(0, 0, far) == backward
    assert wrapped == (ttnn.get_fabric_config() in TP_AXIS_WRAPPED), (
        f"{ttnn.get_fabric_config()} was requested but the TP axis came up "
        f"{'wrapped' if wrapped else 'unwrapped'}; an arm running on a fabric it did not ask "
        f"for proves nothing about the one it did -- check the wrap cabling"
    )

    off_axis = []
    contradictions = []

    for row in range(rows):
        assert (route(row, 0, 1), route(row, 1, 0)) == (
            forward,
            backward,
        ), f"row {row} names the TP axis differently from row 0"
        assert (route(row, 0, far) == backward) == wrapped, f"row {row} disagrees with row 0 about the wrap"

        for src in range(cols):
            for dst in range(cols):
                if src == dst:
                    continue
                direction = route(row, src, dst)
                if direction not in (forward, backward):
                    off_axis.append((row, src, dst, direction))
                if direction != (forward if dst > src else backward):
                    contradictions.append((row, src, dst, direction))

    assert not off_axis, f"a same-row route left the TP axis, so SP topology can perturb it: {off_axis[:4]}"

    total = rows * cols * (cols - 1)
    if wrapped:
        assert contradictions, "a wrapped TP axis that never contradicts rank order cannot gate this"
        logger.info(f"TP axis wrapped: rank order mispredicts {len(contradictions)} of {total} routes")
    else:
        assert not contradictions, f"rank order should be indistinguishable on an unwrapped axis: {contradictions[:4]}"
        logger.info(f"TP axis unwrapped: rank order matches all {total} routes")
