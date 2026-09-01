# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the neighborhood attention device op against the torch reference.

The op consumes tensors in BRICKED site order. Bricking is done here in torch, using the
index formula already pinned to `neighborhood_permute` by test_neighborhood_permute.py, so a
failure here is the OP rather than the ordering.
"""

import pytest
import torch

import ttnn
from models.tt_dit.layers.neighborhood_attention import _query_chunk_bricks
from models.tt_dit.layers.neighborhood_reference import neighborhood_attention_3d

SITES_PER_BRICK = 32


def bricked_index_table(volume, brick):
    """Natural site index for each bricked slot; ghost slots get -1."""
    volume_time, volume_height, volume_width = volume
    brick_time, brick_height, brick_width = brick
    bricks = [-(-volume_axis // brick_axis) for volume_axis, brick_axis in zip(volume, brick)]

    table = torch.full((bricks[0] * bricks[1] * bricks[2] * SITES_PER_BRICK,), -1, dtype=torch.long)
    for site_time in range(volume_time):
        for site_height in range(volume_height):
            for site_width in range(volume_width):
                brick_index = (
                    (site_time // brick_time) * (bricks[1] * bricks[2])
                    + (site_height // brick_height) * bricks[2]
                    + (site_width // brick_width)
                )
                site_index_in_brick = (
                    (site_time % brick_time) * (brick_height * brick_width)
                    + (site_height % brick_height) * brick_width
                    + (site_width % brick_width)
                )
                natural_index = (site_time * volume_height + site_height) * volume_width + site_width
                table[brick_index * SITES_PER_BRICK + site_index_in_brick] = natural_index
    return table


def to_bricked(natural, table):
    """[batch, sites_natural, heads, head_dim] -> [batch, sites_bricked, heads, head_dim]."""
    batch, _, head_count, head_dim = natural.shape
    bricked = torch.zeros(batch, table.numel(), head_count, head_dim, dtype=natural.dtype)
    present = table >= 0
    bricked[:, present] = natural[:, table[present]]
    return bricked


def pearson(left, right):
    left_flat = left.flatten().float()
    right_flat = right.flatten().float()
    return torch.corrcoef(torch.stack([left_flat, right_flat]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "volume, context_window, stride",
    [
        ((4, 8, 8), (3, 3, 3), (1, 1, 1)),
        ((4, 8, 8), (3, 5, 5), (2, 4, 4)),
        # 256 bricks x 2 heads = 512 work items over ~130 cores, so several per core. The two
        # cases above give exactly ONE item per core, which cannot expose a per-item circular
        # buffer leak -- the running max leaked a tile per item and jammed on the third.
        # 64 bricks x 2 heads = 128 items, about one per core -- but the gather is a genuine
        # SUB-volume here. In the two cases above the gather covers the whole volume, so the
        # reader's brick addressing is never exercised: any enumeration order works because the
        # union is complete and the mask does all the filtering.
        ((8, 16, 16), (3, 3, 3), (1, 1, 1)),
        # 256 bricks x 2 heads = 512 work items over ~130 cores, so several per core. The first
        # two cases give exactly ONE item per core, which cannot expose a per-item circular
        # buffer leak -- the running max leaked a tile per item and jammed on the third.
        ((8, 32, 32), (3, 3, 3), (1, 1, 1)),
        # A stride of several bricks, which is the only shape that gives a MULTI-BRICK query
        # chunk: 2x2x2 bricks = 256 queries sharing one gather and one broadcast mask. Every
        # case above derives a one-brick chunk, so none of them exercise the chunk path at all.
        ((8, 16, 16), (4, 8, 8), (4, 8, 8)),
        # The same, on a volume whose bricks do NOT divide by the chunk: 3 brick-rows in time
        # against a chunk 2 deep, so the last chunk hangs half outside the volume. Those rows
        # have no queries to read and no home to write to, and getting that wrong corrupts the
        # final time slice while leaving every other one correct.
        ((6, 16, 16), (4, 8, 8), (4, 8, 8)),
    ],
    ids=[
        "stride_one",
        "stride_equals_brick",
        "partial_gather",
        "many_items_per_core",
        "multi_brick_chunk",
        "chunk_overhangs_volume",
    ],
)
@pytest.mark.parametrize(
    # head_dim 32 is ONE tile per query row; 64 is two. Everything that walks a row -- the Q and
    # K/V reads, the QK^T inner dimension, the PV output width -- is indexed by head_dim_tiles,
    # and none of it was covered while every case used a single tile.
    "head_dim",
    [32, 64],
    ids=["one_tile_row", "two_tile_row"],
)
@pytest.mark.parametrize(
    # A chunk's score tiles are live in DST, so it can never exceed 8 tiles. "widest" is the
    # largest legal chunk (a single chunk when the gather is small enough, which isolates
    # QK -> mask -> softmax -> PV -> normalize from the running-max bookkeeping); "narrow"
    # forces several chunks so the online rescale runs.
    "chunking",
    ["widest_chunk", "narrow_chunk"],
)
def test_matches_torch_reference(mesh_device, volume, context_window, stride, chunking, head_dim):
    torch.manual_seed(0)
    brick = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    # Derived, never chosen: the chunk is one query group measured in bricks. Same call the
    # executor makes, so the test cannot drift from what the model runs.
    query_chunk_bricks = _query_chunk_bricks(stride, brick)
    plan = ttnn.transformer.neighborhood_plan(
        volume, context_window, stride, brick, query_chunk_bricks=query_chunk_bricks
    )

    head_count = 2
    site_count = volume[0] * volume[1] * volume[2]
    query, key, value = (torch.randn(1, site_count, head_count, head_dim) for _ in range(3))

    expected = neighborhood_attention_3d(
        query, key, value, volume=volume, context_window=context_window, stride=stride, brick=brick, scale=1.0
    )

    gather_brick_count = plan["gather_brick_count"]
    DST_CAPACITY_TILES = 8
    widest_chunk = min(gather_brick_count, DST_CAPACITY_TILES)
    tiles_per_kv_chunk = widest_chunk if chunking == "widest_chunk" else max(1, widest_chunk // 3)

    table = bricked_index_table(volume, brick)

    # Op layout: [batch, 1, sites_bricked, heads*head_dim] -- sites are the tile ROW axis, so one
    # tile row is one brick and heads never have to move.
    def upload(tensor):
        bricked = to_bricked(tensor, table).contiguous()
        bricked = bricked.reshape(1, 1, bricked.shape[1], head_count * head_dim)
        return ttnn.from_torch(bricked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
        1, 1, plan["chunk_count"], plan["gather_origin_columns"]
    )
    origin_on_device = ttnn.from_torch(
        origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    actual_device = ttnn.transformer.neighborhood_scaled_dot_product_attention(
        upload(query),
        upload(key),
        upload(value),
        origin_on_device,
        volume=volume,
        context_window=context_window,
        stride=stride,
        brick=brick,
        query_chunk_bricks=query_chunk_bricks,
        head_count=head_count,
        scale=1.0,
        tiles_per_kv_chunk=tiles_per_kv_chunk,
    )

    actual_bricked = ttnn.to_torch(actual_device).float()  # [batch, 1, sites_bricked, heads*head_dim]
    actual_bricked = actual_bricked.reshape(1, -1, head_count, head_dim)
    present = table >= 0
    actual = torch.zeros_like(expected)
    actual[:, table[present]] = actual_bricked[:, present]

    correlation = pearson(actual, expected)
    assert correlation > 0.99, f"PCC {correlation:.5f} against the torch reference"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_shards_match_the_whole_volume(mesh_device):
    """Two shards with DIFFERENT origins reproduce the unsharded answer, on one program.

    This is the property the mesh needs and the one a zero origin cannot show. Window placement
    is GLOBAL -- a query half a window from a shard seam must still see a full window, clamped
    only at the true volume boundary. If the shard origin were compile-time (it was), every
    device of a mesh would share one value, believe it sat at the origin, and clamp at its own
    seam: plausible output, wrong receptive field along every internal boundary.

    Both shards run through the same cached program here, which is exactly what a mesh does.
    """
    torch.manual_seed(0)
    volume, context_window, stride = (4, 8, 16), (3, 3, 3), (1, 1, 1)
    head_count, head_dim = 2, 64
    brick = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))

    site_count = volume[0] * volume[1] * volume[2]
    query, key, value = (torch.randn(1, site_count, head_count, head_dim) for _ in range(3))
    expected = neighborhood_attention_3d(
        query,
        key,
        value,
        volume=volume,
        context_window=context_window,
        stride=stride,
        brick=brick,
        scale=1.0,
    ).reshape(1, *volume, head_count, head_dim)

    volume_form = [tensor.reshape(1, *volume, head_count, head_dim) for tensor in (query, key, value)]

    # Each shard holds what it owns PLUS the halo its windows reach into. The halo is a whole
    # number of bricks so the shard bricks the same way the global volume would.
    owned_width, halo_width = 8, brick[2]
    shards = [
        {"origin": (0, 0, 0), "resident_width": owned_width + halo_width, "owned": (0, owned_width)},
        {
            "origin": (0, 0, owned_width - halo_width),
            "resident_width": owned_width + halo_width,
            "owned": (halo_width, halo_width + owned_width),
        },
    ]

    for shard in shards:
        origin_width = shard["origin"][2]
        resident = (volume[0], volume[1], shard["resident_width"])
        plan = ttnn.transformer.neighborhood_plan(
            volume, context_window, stride, brick, shard_extent=resident, shard_origin=shard["origin"]
        )
        table = bricked_index_table(resident, brick)

        def upload(tensor):
            window = tensor[:, :, :, origin_width : origin_width + shard["resident_width"]]
            flat = window.reshape(1, resident[0] * resident[1] * resident[2], head_count, head_dim)
            bricked = to_bricked(flat, table).contiguous()
            bricked = bricked.reshape(1, 1, bricked.shape[1], head_count * head_dim)
            return ttnn.from_torch(bricked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
            1, 1, plan["chunk_count"], plan["gather_origin_columns"]
        )
        actual_device = ttnn.transformer.neighborhood_scaled_dot_product_attention(
            *(upload(tensor) for tensor in volume_form),
            ttnn.from_torch(origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device),
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            shard_extent=resident,
            shard_origin=shard["origin"],
            head_count=head_count,
            scale=1.0,
            tiles_per_kv_chunk=min(plan["gather_brick_count"], 8),
        )

        bricked_out = ttnn.to_torch(actual_device).float().reshape(1, -1, head_count, head_dim)
        present = table >= 0
        resident_out = torch.zeros(1, resident[0] * resident[1] * resident[2], head_count, head_dim)
        resident_out[:, table[present]] = bricked_out[:, present]
        resident_out = resident_out.reshape(1, *resident, head_count, head_dim)

        owned_start, owned_end = shard["owned"]
        actual_owned = resident_out[:, :, :, owned_start:owned_end]
        global_start = origin_width + owned_start
        expected_owned = expected[:, :, :, global_start : global_start + (owned_end - owned_start)]

        correlation = pearson(actual_owned, expected_owned)
        assert correlation > 0.99, f"shard at width {origin_width}: PCC {correlation:.5f} against the unsharded volume"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_symmetric_halo_shards_match_the_whole_volume(mesh_device):
    """Three shards with a UNIFORM symmetric halo, including negative origins.

    A mesh runs one program, so every device must hold the same resident extent. A symmetric
    halo gives that, at the cost of the end devices' halo hanging off the volume: shard 0 sits at
    origin -halo. Those columns are storage the volume does not contain -- no query owns them and
    no window reaches them -- which is why the origin has to be able to say "below zero" rather
    than being clamped to it. Clamping would silently renumber every column of that device.
    """
    torch.manual_seed(0)
    volume, context_window, stride = (4, 8, 24), (3, 3, 3), (1, 1, 1)
    head_count, head_dim = 2, 64
    brick = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    halo_width, owned_width = brick[2], 8
    resident = (volume[0], volume[1], owned_width + 2 * halo_width)

    site_count = volume[0] * volume[1] * volume[2]
    query, key, value = (torch.randn(1, site_count, head_count, head_dim) for _ in range(3))
    expected = neighborhood_attention_3d(
        query,
        key,
        value,
        volume=volume,
        context_window=context_window,
        stride=stride,
        brick=brick,
        scale=1.0,
    ).reshape(1, *volume, head_count, head_dim)
    volume_form = [tensor.reshape(1, *volume, head_count, head_dim) for tensor in (query, key, value)]

    table = bricked_index_table(resident, brick)
    resident_sites = resident[0] * resident[1] * resident[2]

    for shard_index in range(volume[2] // owned_width):
        origin_width = shard_index * owned_width - halo_width  # NEGATIVE for shard 0
        plan = ttnn.transformer.neighborhood_plan(
            volume,
            context_window,
            stride,
            brick,
            shard_extent=resident,
            shard_origin=(0, 0, origin_width),
        )

        def upload(tensor):
            # Columns outside the volume are real storage holding nothing meaningful. Zeros
            # stand in for whatever the halo exchange would leave there; nothing reads them.
            window = torch.zeros(1, *resident, head_count, head_dim)
            for local in range(resident[2]):
                source = origin_width + local
                if 0 <= source < volume[2]:
                    window[:, :, :, local] = tensor[:, :, :, source]
            flat = window.reshape(1, resident_sites, head_count, head_dim)
            bricked = to_bricked(flat, table).contiguous()
            bricked = bricked.reshape(1, 1, bricked.shape[1], head_count * head_dim)
            return ttnn.from_torch(bricked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
            1, 1, plan["chunk_count"], plan["gather_origin_columns"]
        )
        actual_device = ttnn.transformer.neighborhood_scaled_dot_product_attention(
            *(upload(tensor) for tensor in volume_form),
            ttnn.from_torch(origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device),
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            shard_extent=resident,
            shard_origin=(0, 0, origin_width),
            head_count=head_count,
            scale=1.0,
            tiles_per_kv_chunk=min(plan["gather_brick_count"], 8),
        )

        bricked_out = ttnn.to_torch(actual_device).float().reshape(1, -1, head_count, head_dim)
        present = table >= 0
        resident_out = torch.zeros(1, resident_sites, head_count, head_dim)
        resident_out[:, table[present]] = bricked_out[:, present]
        resident_out = resident_out.reshape(1, *resident, head_count, head_dim)

        # Every device owns the same local band: the middle, past its own halo.
        actual_owned = resident_out[:, :, :, halo_width : halo_width + owned_width]
        global_start = shard_index * owned_width
        expected_owned = expected[:, :, :, global_start : global_start + owned_width]

        correlation = pearson(actual_owned, expected_owned)
        assert correlation > 0.99, f"shard {shard_index} at global origin {origin_width}: PCC {correlation:.5f}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    # A negative origin puts local column 0 BELOW the volume, which is what the production halo
    # does and what the interior gate has to survive: it must still admit the bricks that sit
    # comfortably inside the global volume, not reject the whole shard.
    "owned_width",
    [None, 12],
    ids=["unsharded", "w_sharded_negative_origin"],
)
@pytest.mark.parametrize(
    # (2,4,4) is what `neighborhood_choose_brick` returns for an 11^3 window; (8,2,2) is what
    # `_choose_sharded_brick` returns for the 1080p decode, because it gathers 147 bricks rather
    # than 175. Its W brick of 2 also puts the relative span at 3x7x7 instead of 7x5x5, so it is a
    # different table, a different slot linearisation and a different halo -- none of which the
    # (2,4,4) case exercises. It also does not divide the time axis, so it carries ghost bricks.
    "brick",
    [None, (8, 2, 2)],
    ids=["chosen_brick", "brick_822"],
)
def test_interior_table_matches_generated_masks(mesh_device, owned_width, brick):
    """The UPLOADED relative mask table, on a volume large enough to have interior bricks.

    Every other case here passes no ``interior_mask`` at all, on a volume small enough that every
    brick clamps at an edge -- so the uploaded table is never consulted and the kernel generates
    every tile. This is the path the 1080p decode spends ~75% of its bricks on, and the one where
    cb_mask holds a whole work item so those tiles are written once and then reused; getting the
    slot -> table page permutation wrong shows up here and nowhere else.

    Volume (16, 24, 24) with an 11^3 window is the smallest shape that has genuine interior
    bricks on all three axes -- at (12, 16, 16) the time axis has none, since a brick-aligned
    origin cannot land on the single unclamped time site.
    """
    from models.tt_dit.layers.neighborhood_attention import _build_relative_masks, halo_sites

    torch.manual_seed(0)
    volume, context_window, stride = (16, 24, 24), (11, 11, 11), (1, 1, 1)
    head_count, head_dim = 2, 64
    if brick is None:
        brick = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))

    site_count = volume[0] * volume[1] * volume[2]
    query, key, value = (torch.randn(1, site_count, head_count, head_dim) for _ in range(3))
    expected = neighborhood_attention_3d(
        query, key, value, volume=volume, context_window=context_window, stride=stride, brick=brick, scale=1.0
    ).reshape(1, *volume, head_count, head_dim)
    volume_form = [tensor.reshape(1, *volume, head_count, head_dim) for tensor in (query, key, value)]

    interior_mask = ttnn.from_torch(
        _build_relative_masks(context_window, brick),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    halo = 0 if owned_width is None else halo_sites(context_window[2], brick[2])
    owned = volume[2] if owned_width is None else owned_width
    resident = (volume[0], volume[1], owned + 2 * halo)
    resident_sites = resident[0] * resident[1] * resident[2]
    table = bricked_index_table(resident, brick)

    for shard_index in range(volume[2] // owned):
        origin_width = shard_index * owned - halo  # NEGATIVE for shard 0 when sharded
        plan = ttnn.transformer.neighborhood_plan(
            volume,
            context_window,
            stride,
            brick,
            # Must match what the op is handed below. The table carries one row per query CHUNK,
            # so letting the planner default this while the op gets an explicit (possibly forced,
            # via DIFFVAE_NA_CHUNK_BRICKS) chunk builds a table of the wrong length -- 288 rows
            # against the 72 the op expects at chunk (2,1,2) -- and fails op validation before any
            # kernel runs. The other call sites here pass it to both or to neither.
            query_chunk_bricks=_query_chunk_bricks(stride, brick),
            shard_extent=resident,
            shard_origin=(0, 0, origin_width),
        )

        def upload(tensor):
            window = torch.zeros(1, *resident, head_count, head_dim)
            for local in range(resident[2]):
                source = origin_width + local
                if 0 <= source < volume[2]:
                    window[:, :, :, local] = tensor[:, :, :, source]
            bricked = to_bricked(window.reshape(1, resident_sites, head_count, head_dim), table).contiguous()
            bricked = bricked.reshape(1, 1, bricked.shape[1], head_count * head_dim)
            return ttnn.from_torch(bricked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
            1, 1, plan["chunk_count"], plan["gather_origin_columns"]
        )
        actual_device = ttnn.transformer.neighborhood_scaled_dot_product_attention(
            *(upload(tensor) for tensor in volume_form),
            ttnn.from_torch(origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device),
            interior_mask=interior_mask,
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            query_chunk_bricks=_query_chunk_bricks(stride, brick),
            shard_extent=resident,
            shard_origin=(0, 0, origin_width),
            head_count=head_count,
            scale=1.0,
            tiles_per_kv_chunk=min(plan["gather_brick_count"], 8),
        )

        bricked_out = ttnn.to_torch(actual_device).float().reshape(1, -1, head_count, head_dim)
        present = table >= 0
        resident_out = torch.zeros(1, resident_sites, head_count, head_dim)
        resident_out[:, table[present]] = bricked_out[:, present]
        resident_out = resident_out.reshape(1, *resident, head_count, head_dim)

        # Compare the owned band: halo columns sit outside this shard's slice of the volume.
        actual_owned = resident_out[:, :, :, halo : halo + owned]
        expected_owned = expected[:, :, :, shard_index * owned : shard_index * owned + owned]
        correlation = pearson(actual_owned, expected_owned)
        assert correlation > 0.99, f"shard {shard_index} at origin {origin_width}: PCC {correlation:.5f}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "volume, context_window, width_local, shard_count, expected_brick, expected_gather",
    [
        # 1080p decode: (8,2,2) and (2,8,2) both gather 147 bricks with the same halo; the
        # deeper time brick measured 11% faster per query brick, and the -brick_time tiebreak
        # in the scoring tuple picks it.
        ((84, 272, 480), (11, 11, 11), 60, 8, (8, 2, 2), 147),
    ],
    ids=["1080p_decode"],
)
def test_choose_sharded_brick_regression(
    mesh_device, volume, context_window, width_local, shard_count, expected_brick, expected_gather
):
    """Pin the brick choice and gather count for known production geometries.

    ``_choose_sharded_brick`` searches all valid 32-site bricks and picks the one that minimises
    the gather brick count as measured by the real planner.  A change to the search, the scoring,
    or the planner's alignment handling can silently shift the choice and the downstream gather
    size -- which changes kernel runtime without any code in the kernel itself changing.
    """
    from models.tt_dit.layers.neighborhood_attention import (
        _BRICK_CHOICE_CACHE,
        _choose_sharded_brick,
        _query_chunk_bricks,
        halo_sites,
    )

    _BRICK_CHOICE_CACHE.clear()

    stride = (1, 1, 1)
    brick = _choose_sharded_brick(volume, context_window, stride, width_local, shard_count)
    assert brick == expected_brick, (
        f"expected brick {expected_brick} for volume={volume} window={context_window} "
        f"width_local={width_local} shard_count={shard_count}, got {brick}"
    )

    # Verify the gather count -- the metric the function optimises. A planner change that
    # increases it (even with the same brick) is a perf regression worth catching.
    halo = halo_sites(min(context_window[2], volume[2]), brick[2])
    resident = (volume[0], volume[1], width_local + 2 * halo)
    plan = ttnn.transformer.neighborhood_plan(
        volume,
        context_window,
        stride,
        brick,
        query_chunk_bricks=_query_chunk_bricks(stride, brick),
        shard_extent=resident,
        shard_origin=(0, 0, -halo),
    )
    assert (
        plan["gather_brick_count"] == expected_gather
    ), f"expected gather_brick_count={expected_gather} for brick {brick}, got {plan['gather_brick_count']}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_choose_sharded_brick_delegates_at_stride_gt_one(mesh_device):
    """stride > (1,1,1) delegates to ``neighborhood_choose_brick`` unconditionally."""
    from models.tt_dit.layers.neighborhood_attention import _choose_sharded_brick

    brick = _choose_sharded_brick(
        volume=(8, 16, 16),
        context_window=(3, 5, 5),
        stride=(2, 4, 4),
        width_local=8,
        shard_count=2,
    )
    expected = tuple(ttnn.transformer.neighborhood_choose_brick((3, 5, 5)))
    assert brick == expected, f"stride > 1 should return {expected}, got {brick}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_choose_sharded_brick_rejects_oversized_bricks(mesh_device):
    """A volume with few time frames must not select a brick deeper than the volume.

    Without the ``brick > volume`` filter, a 12-frame volume picks (16, 1, 2) -- a brick that
    exceeds the time axis.  That scores excellently (77 gathered bricks against the 147 a real
    brick needs) because the degenerate axis contributes a single slot to the gather, but the op
    wedges on the resulting ghost-heavy grid.
    """
    from models.tt_dit.layers.neighborhood_attention import _BRICK_CHOICE_CACHE, _choose_sharded_brick

    _BRICK_CHOICE_CACHE.clear()

    # 12 time frames: small enough that (16, 1, 2) exceeds it on the time axis.
    volume = (12, 272, 480)
    context_window = (11, 11, 11)
    stride = (1, 1, 1)
    width_local = 60
    shard_count = 8

    brick = _choose_sharded_brick(volume, context_window, stride, width_local, shard_count)

    assert all(b <= v for b, v in zip(brick, volume)), (
        f"brick {brick} exceeds volume {volume} on at least one axis -- " f"the oversized-brick filter is not working"
    )
