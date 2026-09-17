# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Correctness for moe_fanout_reach, the op that produces a chip's row of the multicast reach table.

The gate is exact equality against `_mc_reach`, the same torch reference `test_prefill_dispatch_fabric2d`
builds its multicast fixtures from -- not a threshold. A reach table that OVERSTATES makes an origin send
fewer pages than the relay downstream waits for, which stops the axis rather than producing wrong numbers,
so "close" is not a meaningful state for this tensor to be in.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_dispatch_fabric2d import (
    ROUTING_PROFILES,
    _draw_indices,
    _expert_dispatch_table,
    _mc_reach,
    _moe_grid_split,
    _sub_device_manager,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config, get_gate_outputs


def _decomposed_reach(indices_row, table_g, offs_row, capacity, extent, my_row, num_routed_experts, topk, num_cores):
    """The op's decomposition replayed on host: compaction, per-core histograms, the scan, the walk.

    This is the argument the op rests on, made executable. The rank of a pick is the number of earlier
    picks of the same expert; the cores take contiguous token ranges in order, so the picks before a
    core's range are exactly the sum of the earlier cores' histograms, and seeding each core's allocator
    with that sum reproduces the sequential allocator at every pick -- drops included, since the counter
    advances whether or not the page fits. The Hillis-Steele rounds and the closing one-core shift are
    written out as the kernel runs them rather than as a plain prefix sum, because the round count is a
    choice this op makes and getting it short would silently under-seed the later cores.
    """
    seq = indices_row.shape[0]
    m = extent // 2

    compact_of = [-1] * num_routed_experts
    compact_offsets = []
    for e in range(num_routed_experts):
        row = int(table_g[e])
        if row < 0 or row >= extent:
            continue
        compact_of[e] = len(compact_offsets)
        compact_offsets.append(int(offs_row[e]))
    n_compact = len(compact_offsets)

    bounds = [(c * seq) // num_cores for c in range(num_cores + 1)]

    hist = [[0] * n_compact for _ in range(num_cores)]
    for c in range(num_cores):
        for t in range(bounds[c], bounds[c + 1]):
            for k in range(topk):
                e = int(indices_row[t, k])
                if e >= num_routed_experts or compact_of[e] < 0:
                    continue
                hist[c][compact_of[e]] += 1

    rounds = 0
    while (1 << rounds) < num_cores:
        rounds += 1
    buf = [list(h) for h in hist]
    for r in range(rounds):
        step = 1 << r
        nxt = [list(b) for b in buf]
        for c in range(num_cores):
            if c >= step:
                nxt[c] = [buf[c][i] + buf[c - step][i] for i in range(n_compact)]
        buf = nxt
    base = []
    for c in range(num_cores):
        before = buf[c - 1] if c >= 1 else [0] * n_compact
        base.append([compact_offsets[i] + before[i] for i in range(n_compact)])

    classes = [[0] * (m + 2) for _ in range(2)]
    for c in range(num_cores):
        alloc = list(base[c])
        for t in range(bounds[c], bounds[c + 1]):
            far = [0, 0]
            for k in range(topk):
                e = int(indices_row[t, k])
                if e >= num_routed_experts:
                    continue
                ci = compact_of[e]
                if ci < 0:
                    continue
                page = alloc[ci]
                alloc[ci] = page + 1
                if page >= capacity:
                    continue
                row = int(table_g[e])
                if row == my_row:
                    continue
                cw = (row - my_row) % extent
                ccw = extent - cw
                if cw <= ccw:
                    far[0] = max(far[0], cw)
                else:
                    far[1] = max(far[1], ccw)
            for d in (0, 1):
                if far[d] > 0:
                    classes[d][far[d]] += 1

    reach = torch.zeros(2, m + 2, dtype=torch.int64)
    for d in (0, 1):
        run = 0
        for h in range(m, 0, -1):
            run += classes[d][h]
            reach[d, h] = run
    return reach


def _routing_draw(G, H, seq_len_per_chip, topk, num_routed_experts, routing, seed):
    """One draw and the offsets table it implies, matching what test_dispatch_fabric2d hands the op."""
    torch.manual_seed(seed)
    table = _expert_dispatch_table(num_routed_experts, H, G)
    experts_per_chip = num_routed_experts // G // H
    if routing is None:
        experts_per_group = num_routed_experts // G
        indices = torch.zeros(G, H, seq_len_per_chip, topk, dtype=torch.int64)
        for g in range(G):
            for h in range(H):
                for t in range(seq_len_per_chip):
                    indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]
    else:
        share, hot_weight = ROUTING_PROFILES[routing]
        indices = _draw_indices(G, H, seq_len_per_chip, topk, num_routed_experts, share, hot_weight)

    offs = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    for g in range(G):
        o, _, _, _ = get_gate_outputs(
            indices[g],
            H,
            num_routed_experts,
            experts_per_chip,
            seq_len_per_chip,
            topk,
            expert_dispatch_table=table[g : g + 1],
        )
        offs[g] = o[0].to(torch.int32)
    return table, indices, offs


@pytest.mark.parametrize("num_cores", [1, 2, 3, 5, 8, 64], ids=lambda c: f"{c}core")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
def test_moe_fanout_reach_decomposition(num_cores, capacity_div):
    """The split over cores reproduces the sequential allocator exactly, on host, with no device.

    64 tokens over 3 or 5 cores is the case that matters: the ranges are uneven and the scan's round
    count is not a power of two, which is where an off-by-one in the seeding would show up.
    """
    G, H, seq, topk, num_routed_experts = 2, 8, 64, 4, 128
    capacity = max(1, H * seq * topk // capacity_div)
    table, indices, offs = _routing_draw(G, H, seq, topk, num_routed_experts, routing=None, seed=23)
    reference = _mc_reach(indices, table, offs, capacity, G, H, seq, topk)

    for g in range(G):
        for origin in range(H):
            got = _decomposed_reach(
                indices[g, origin], table[g], offs[g, origin], capacity, H, origin, num_routed_experts, topk, num_cores
            )
            assert torch.equal(got, reference[g, origin]), (
                f"g={g} origin={origin} cores={num_cores}: decomposition {got.tolist()} "
                f"!= reference {reference[g, origin].tolist()}"
            )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
# A divisor of 1 is the roomy case no token can overflow; the large one puts capacity below the
# per-expert load so the allocator starts dropping. Tight is the case that matters: it is the only one
# where a drop moves a token's farthest hop, and where reach stops being a function of the indices alone.
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
@pytest.mark.parametrize("seq_len_per_chip", [32, 128], ids=lambda s: f"seq{s}")
# In-group routing gives every token somewhere to go. Production routes over all experts, so most picks
# resolve to -1 and the survivors concentrate on a few chips -- which is what makes far-hop classes of
# size 0 and 1 common and whole directions empty.
@pytest.mark.parametrize("routing", [None, "hottest"], ids=lambda r: r or "in-group")
@pytest.mark.timeout(600)
def test_moe_fanout_reach(mesh_device, device_params, capacity_div, seq_len_per_chip, routing):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op measures hops on the dispatch axis, which extract_mesh_config puts at 0"
    num_routed_experts = 256
    topk = 8
    m = H // 2
    capacity = max(1, H * seq_len_per_chip * topk // capacity_div)

    table, indices, offs = _routing_draw(G, H, seq_len_per_chip, topk, num_routed_experts, routing, seed=7)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # The same indices tensor dispatch_fabric2d takes, in the same layout, for the same reason: reach
    # describes what THAT routing sends.
    tt_idx = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)
    # This device's own row: dim 1 (the dispatch row) down the mesh rows, dim 0 (the group) across.
    tt_offs_row = shard(offs, (1, 0), ttnn.int32)

    reach = ttnn.experimental.deepseek_prefill.moe_fanout_reach(
        tt_idx,
        tt_table,
        tt_offs_row,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=topk,
        dispatch_group_size=H,
        max_dispatch_buffer_token_size=capacity,
        cluster_axis=sp_axis,
    )
    assert tuple(reach.shape) == (1, 2, m + 2), reach.shape

    reference = _mc_reach(indices, table, offs, capacity, G, H, seq_len_per_chip, topk)
    got = ttnn.get_device_tensors(reach)
    mesh_cols = tuple(mesh_device.shape)[1]

    bad = 0
    travelling = 0
    for dev in range(H * G):
        r, g = dev // mesh_cols, dev % mesh_cols
        row = ttnn.to_torch(got[dev]).to(torch.int64).reshape(2, m + 2)
        want = reference[g, r]
        travelling += int(row[:, 1].sum())
        if not torch.equal(row, want):
            logger.error(f"device {dev} (row {r}, group {g}): got {row.tolist()} want {want.tolist()}")
            bad += 1
    assert bad == 0, f"{bad} of {H * G} devices disagree with the torch reference"

    # A table of zeros matches a zero reference, so say that something actually travelled.
    assert travelling > 0, "no token crossed a cable in either direction; the comparison proved nothing"

    # reach[0] and reach[m + 1] are what make reach[h] - reach[h + 1] a class size for every h up to m;
    # a nonzero in either turns a chunk length negative at the far end of the ring.
    for dev in range(H * G):
        row = ttnn.to_torch(got[dev]).to(torch.int64).reshape(2, m + 2)
        assert int(row[:, 0].sum()) == 0 and int(row[:, m + 1].sum()) == 0, f"device {dev} row {row.tolist()}"

    if capacity_div > 1:
        # Without this the tight case could be silently identical to the roomy one and the drop rule --
        # the whole reason this cannot be folded into masked_bincount -- would go untested.
        roomy = _mc_reach(indices, table, offs, H * seq_len_per_chip * topk, G, H, seq_len_per_chip, topk)
        assert not torch.equal(roomy, reference), "the tight config dropped nothing that moved a farthest hop"

    logger.info(
        f"moe_fanout_reach: mesh={tuple(mesh_device.shape)} seq={seq_len_per_chip} capacity={capacity} "
        f"routing={routing or 'in-group'}: {H * G} rows exact, {travelling} token-directions travelling"
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_moe_fanout_reach_sequence_length_is_in_the_cache_key(mesh_device, device_params):
    """Three launches in one process at ONE capacity, two sequence lengths, then back to the first.

    The op splits the sequence over the grid and sizes its L1 carve from how long a range that leaves
    each core, so a cached program built for one sequence length reads the wrong number of tokens for
    another. The rest of the matrix never catches that: capacity there is derived from the sequence
    length, so the two always move together and the cache key is distinguished either way. Here they do
    not, and the third launch also says a re-dispatch picks up the new output buffer rather than writing
    into the first launch's.
    """
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    num_routed_experts, topk = 256, 8
    capacity = 96  # tight at both lengths, and unchanged between them

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    for seq in (128, 64, 128):
        table, indices, offs = _routing_draw(G, H, seq, topk, num_routed_experts, routing=None, seed=5)
        reach = ttnn.experimental.deepseek_prefill.moe_fanout_reach(
            shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16),
            shard(table.unsqueeze(1), (None, 0), ttnn.int32),
            shard(offs, (1, 0), ttnn.int32),
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=topk,
            dispatch_group_size=H,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=sp_axis,
        )
        reference = _mc_reach(indices, table, offs, capacity, G, H, seq, topk)
        got = ttnn.get_device_tensors(reach)
        for dev in range(H * G):
            r, g = dev // G, dev % G
            row = ttnn.to_torch(got[dev]).to(torch.int64).reshape(2, H // 2 + 2)
            assert torch.equal(row, reference[g, r]), (
                f"seq={seq} device {dev} (row {r}, group {g}): got {row.tolist()} " f"want {reference[g, r].tolist()}"
            )
        logger.info(f"moe_fanout_reach: seq={seq} at capacity {capacity} exact on all {H * G} devices")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_moe_fanout_reach_subdevice(mesh_device, device_params):
    """The model's carve: dispatch gets row 0 of the Tensix grid, the shared expert the rest.

    A narrower core set is not just fewer workers. The scan's round count, the reduction tree's depth
    and how many tokens each core walks all come from it, so this is the only device coverage of a core
    count that is not the full grid the rest of the file takes. Tight capacity on top, because a shorter
    scan chain is exactly where an under-seeded allocator would show up and drops are what make the
    seeding observable.
    """
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    num_routed_experts, topk, seq = 256, 8, 128
    capacity = max(1, H * seq * topk // 64)
    table, indices, offs = _routing_draw(G, H, seq, topk, num_routed_experts, routing="hottest", seed=13)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Allocated before the manager is loaded, as the sibling sub-device test does.
    tt_idx = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)
    tt_offs_row = shard(offs, (1, 0), ttnn.int32)

    dispatch_cores, shared_cores = _moe_grid_split(mesh_device)
    n_cores = sum((r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1) for r in dispatch_cores.ranges())
    logger.info(f"moe_fanout_reach on the dispatch sub-device: {n_cores} cores in row 0, {seq} tokens")

    with _sub_device_manager(mesh_device, [dispatch_cores, shared_cores]) as (dispatch_sd, _shared_sd):
        reach = ttnn.experimental.deepseek_prefill.moe_fanout_reach(
            tt_idx,
            tt_table,
            tt_offs_row,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=topk,
            dispatch_group_size=H,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=sp_axis,
            subdevice_id=dispatch_sd,
        )
        reference = _mc_reach(indices, table, offs, capacity, G, H, seq, topk)
        got = ttnn.get_device_tensors(reach)
        for dev in range(H * G):
            r, g = dev // G, dev % G
            row = ttnn.to_torch(got[dev]).to(torch.int64).reshape(2, H // 2 + 2)
            assert torch.equal(row, reference[g, r]), (
                f"device {dev} (row {r}, group {g}) on {n_cores} cores: got {row.tolist()} "
                f"want {reference[g, r].tolist()}"
            )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_moe_fanout_reach_all_gathers_into_the_transport_shape(mesh_device, device_params):
    """The op's row, all-gathered along the dispatch axis, is the table dispatch_fabric2d validates.

    The op emits one chip's row; the transport needs every origin's, because a relay sizes a chunk it
    neither wrote nor receives. Production closes that with a CCL, not a host round trip, so this is
    what says the output's shape and page size are usable as they come out -- a row is only
    extent / 2 + 2 INT32, which is a small page for an all-gather to move.
    """
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    num_routed_experts, topk, seq = 256, 8, 128
    capacity = max(1, H * seq * topk // 64)
    table, indices, offs = _routing_draw(G, H, seq, topk, num_routed_experts, routing="hottest", seed=19)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    reach = ttnn.experimental.deepseek_prefill.moe_fanout_reach(
        shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16),
        shard(table.unsqueeze(1), (None, 0), ttnn.int32),
        shard(offs, (1, 0), ttnn.int32),
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=topk,
        dispatch_group_size=H,
        max_dispatch_buffer_token_size=capacity,
        cluster_axis=sp_axis,
    )
    gathered = ttnn.all_gather(reach, dim=0, cluster_axis=sp_axis)
    hops = H // 2 + 2
    # The shape dispatch_fabric2d checks: [.., extent, 2, >= extent / 2 + 2], origin-major.
    assert tuple(gathered.shape)[-3:] == (H, 2, hops), gathered.shape

    reference = _mc_reach(indices, table, offs, capacity, G, H, seq, topk)
    got = ttnn.get_device_tensors(gathered)
    for dev in range(H * G):
        g = dev % G
        rows = ttnn.to_torch(got[dev]).to(torch.int64).reshape(H, 2, hops)
        # Replicated along the dispatch axis: every chip in group g holds every origin's row.
        assert torch.equal(rows, reference[g]), f"device {dev} (group {g}) gathered table differs"
