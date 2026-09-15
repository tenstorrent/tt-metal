# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bring-up and correctness for dispatch_fabric2d.

The op moves each token to the chips hosting the experts it was routed to, one fabric hop at a time,
relaying through a DRAM forwarding buffer rather than leaving multi-hop routing to the fabric. It is a
transport replacement for `dispatch` and must place every token on the same page that op would, so the
gate is byte-exact equality against it rather than a correlation threshold.

The routing metadata is derived by `get_gate_outputs` from the same indices the op is given, so the
control tensors and the routing agree by construction -- which is what the reader's own prologue check
relies on.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config, get_gate_outputs


def _reference_dispatch(indices, table, offs, x, capacity, G, H, seq, topk, emb):
    """What `dispatch` would place, per destination chip, from every source chip.

    Replays the same per-expert allocator the production op uses, including the rule that a token past
    capacity is dropped while its counter still advances -- every later token's page depends on it.

    Returns payload[g][dst_row], metadata[g][dst_row] and, per (g, dst_row, page), the source row that
    wrote it, so a caller can compare only the pages a given source contributed.
    """
    payload = torch.zeros(G, H, capacity, emb, dtype=torch.bfloat16)
    meta = torch.full((G, H, capacity, 3), -1, dtype=torch.int32)
    src_of = torch.full((G, H, capacity), -1, dtype=torch.int32)
    for g in range(G):
        for s in range(H):
            alloc = offs[g, s].clone().to(torch.int64)
            for t in range(seq):
                for k in range(topk):
                    e = int(indices[g, s, t, k])
                    row = int(table[g, e])
                    if row == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    page = int(alloc[e])
                    alloc[e] += 1
                    payload[g, row, page] = x[s, g, t]
                    meta[g, row, page] = torch.tensor([s * G + g, t, k], dtype=torch.int32)
                    src_of[g, row, page] = s
    return payload, meta, src_of


def _expert_dispatch_table(num_routed_experts: int, dispatch_group_size: int, num_dispatch_groups: int):
    """expert -> chip within its own dispatch group, -1 for experts of other groups.

    The trailing sentinel column is what makes a padded token's unguarded lookup resolve to -1.
    """
    experts_per_group = num_routed_experts // num_dispatch_groups
    experts_per_chip = experts_per_group // dispatch_group_size
    table = torch.full((num_dispatch_groups, num_routed_experts + 1), -1, dtype=torch.int32)
    for g in range(num_dispatch_groups):
        for e in range(experts_per_group):
            table[g, g * experts_per_group + e] = e // experts_per_chip
    return table


# Routing profiles, as (share of picks landing in this dispatch group, weight on the hot half of its
# chips). Both knobs matter and both were wrong before:
#
# - Drawing every pick from the group's own experts makes ~4x too many picks land in-group, which
#   makes collisions on one destination chip far more likely than production and OVERSTATES fan-out
#   by roughly 3x. Production spreads picks over all 256 experts across 4 groups.
# - Drawing uniformly across chips is fan-out's worst case: one copy per direction saves nothing when
#   no two of a token's destinations share a direction. It reports ~1.2x, which reads as noise.
#
# The interesting configurations are the hot ones, because that is where the layers the perf harness
# selects actually sit. Calibrate the exact shares on a perf-qualified machine; this host cannot
# measure and the profiles are only meant to span the range.
ROUTING_PROFILES = {
    "uniform": (0.250, 1.0),
    "hot": (0.372, 2.0),
    "hottest": (0.473, 3.0),
}


def _draw_indices(G, H, seq, topk, num_routed_experts, in_group_share, hot_weight):
    """topk distinct experts per token, drawn from ALL experts with a controllable in-group skew.

    How many of a token's picks land in its own dispatch group is drawn FIRST, then that many distinct
    in-group experts and the rest from the other groups. Weighting the whole expert list and drawing
    topk distinct picks from it does not work: sampling without replacement pulls the share well above
    the weight it was solved for (47.3% asked, 63.3% delivered), and a share that drifts up is the
    measurement trap this generator exists to avoid.

    Within the group, the chips in its hot half are weighted `hot_weight`, which is what concentrates
    traffic onto one directed link -- the thing multicast is measured on.
    """
    experts_per_group = num_routed_experts // G
    experts_per_chip = experts_per_group // H
    in_weights = torch.ones(experts_per_group, dtype=torch.float64)
    in_weights[experts_per_chip * (H // 2) :] = hot_weight

    indices = torch.zeros(G, H, seq, topk, dtype=torch.int64)
    for g in range(G):
        base = g * experts_per_group
        elsewhere = torch.cat([torch.arange(0, base), torch.arange(base + experts_per_group, num_routed_experts)])
        for h in range(H):
            for t in range(seq):
                n_in = int((torch.rand(topk) < in_group_share).sum())
                picks = torch.empty(0, dtype=torch.int64)
                if n_in > 0:
                    picks = base + torch.multinomial(in_weights, n_in, replacement=False)
                if n_in < topk:
                    picks = torch.cat([picks, elsewhere[torch.randperm(elsewhere.numel())[: topk - n_in]]])
                indices[g, h, t] = picks
    return indices


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4-2link",
        ),
        # One link halves stream_count, so the opposite chip's chunk is split two ways instead of
        # four: the split arithmetic and the region layout both change shape, not just size.
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
# A divisor of 1 is the roomy case no token can overflow; the large one puts capacity below the
# per-expert load so the allocator starts dropping, which is the case where the offsets table counts
# more tokens than any origin actually sends.
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
@pytest.mark.parametrize("seq_len_per_chip", [32, 128], ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("num_routed_experts", [256], ids=lambda n: f"exp{n}")
@pytest.mark.parametrize("emb_dim", [256], ids=lambda e: f"emb{e}")
# Both transports must land the same bytes on the same pages, so they share one gate. Multicast sends
# one page per token per direction and lets every chip en route keep what is addressed to it; unicast
# sends one per (token, expert). Nothing about the output distinguishes them, which is the point.
@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
# In-group routing gives every token somewhere to go and is what the byte-exactness gate was built on.
# Production routes over all experts, so most picks resolve to -1 and the surviving ones concentrate on
# a few chips -- a distribution this op had never been run against.
@pytest.mark.parametrize("routing", [None, "hottest"], ids=lambda r: r or "in-group")
def test_dispatch_fabric2d(
    mesh_device,
    device_params,
    num_links,
    seq_len_per_chip,
    capacity_div,
    num_routed_experts,
    emb_dim,
    fanout,
    routing,
):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"
    num_experts_per_tok = 8
    experts_per_chip = num_routed_experts // G // H
    # Capacity the production op would use: every source chip's tokens for one expert, tile-aligned.
    max_dispatch_buffer_token_size = max(1, H * seq_len_per_chip * num_experts_per_tok // capacity_div)

    logger.info(
        f"dispatch_fabric2d: mesh={tuple(mesh_device.shape)} H={H} G={G} experts_per_chip={experts_per_chip} "
        f"seq={seq_len_per_chip} topk={num_experts_per_tok} capacity={max_dispatch_buffer_token_size}"
    )

    torch.manual_seed(7)
    table = _expert_dispatch_table(num_routed_experts, H, G)

    # Per group, route only into that group's own experts so every token has somewhere to go.
    experts_per_group = num_routed_experts // G
    indices = torch.zeros(G, H, seq_len_per_chip, num_experts_per_tok, dtype=torch.int64)
    if routing is not None:
        share, hot_weight = ROUTING_PROFILES[routing]
        indices = _draw_indices(G, H, seq_len_per_chip, num_experts_per_tok, num_routed_experts, share, hot_weight)
    else:
        for g in range(G):
            base = g * experts_per_group
            for h in range(H):
                for t in range(seq_len_per_chip):
                    pick = torch.randperm(experts_per_group)[:num_experts_per_tok]
                    indices[g, h, t] = base + pick

    offs = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    counts = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    region = torch.zeros(G, H, num_routed_experts, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g],
            H,
            num_routed_experts,
            experts_per_chip,
            seq_len_per_chip,
            num_experts_per_tok,
            expert_dispatch_table=table[g : g + 1],
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = torch.randn(H, G, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    tt_x = shard(x, (0, 1), ttnn.bfloat16)
    tt_idx = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    # expert_offsets is the ALL-ROWS table: replicated along the dispatch axis, since a relaying chip
    # sizes a run it neither wrote nor receives.
    tt_offs = shard(offs, (None, 0), ttnn.int32)
    tt_counts = shard(counts[:, 0:1, :], (None, 0), ttnn.int32)
    tt_region = shard(region[:, 0:1, :], (None, 0), ttnn.int32)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)

    # Multicast sizes a chunk as "tokens from this origin still travelling this way", which no
    # per-expert count can express. Production has to grow that table in masked_bincount; until then
    # the test supplies it, derived from the same indices and the same drop rule the op replays.
    tt_reach = None
    if fanout:
        reach = _mc_reach(
            indices, table, offs, max_dispatch_buffer_token_size, G, H, seq_len_per_chip, num_experts_per_tok
        )
        tt_reach = shard(reach.to(torch.int32), (None, 0), ttnn.int32)

    payload, metadata = ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
        tt_x,
        tt_idx,
        tt_offs,
        tt_table,
        tt_counts,
        tt_region,
        fanout_reach=tt_reach,
        fanout=fanout,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=3,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert tuple(payload.shape)[-2:] == (max_dispatch_buffer_token_size, emb_dim), payload.shape
    assert tuple(metadata.shape)[-2:] == (max_dispatch_buffer_token_size, 3), metadata.shape

    ref_payload, ref_meta, src_of = _reference_dispatch(
        indices, table, offs, x, max_dispatch_buffer_token_size, G, H, seq_len_per_chip, num_experts_per_tok, emb_dim
    )

    # Every page any chip sourced should now be in place: the neighbour's by a single hop, farther
    # ones relayed through the forwarding regions, and this chip's own by the local phase. `src_of`
    # marks the pages no token lands on, which stay untouched.
    got_payload = ttnn.get_device_tensors(payload)
    got_meta = ttnn.get_device_tensors(metadata)
    mesh_cols = tuple(mesh_device.shape)[1]

    checked = 0
    checked_local = 0
    bad = 0
    for dev in range(H * G):
        r, g = dev // mesh_cols, dev % mesh_cols
        pages = [p for p in range(max_dispatch_buffer_token_size) if int(src_of[g, r, p]) >= 0]
        local = sum(1 for p in pages if int(src_of[g, r, p]) == r)
        if not pages:
            continue
        pay = ttnn.to_torch(got_payload[dev]).reshape(max_dispatch_buffer_token_size, emb_dim)
        met = ttnn.to_torch(got_meta[dev]).to(torch.int32).reshape(max_dispatch_buffer_token_size, 3)
        idx = torch.tensor(pages)
        checked += len(pages)
        checked_local += local
        if not torch.equal(pay[idx], ref_payload[g, r][idx]):
            n = (pay[idx] != ref_payload[g, r][idx]).any(-1).sum().item()
            logger.error(f"device {dev} (row {r}, group {g}): {n}/{len(pages)} payload pages differ")
            bad += 1
        if not torch.equal(met[idx], ref_meta[g, r][idx]):
            n = (met[idx] != ref_meta[g, r][idx]).any(-1).sum().item()
            logger.error(f"device {dev} (row {r}, group {g}): {n}/{len(pages)} metadata pages differ")
            logger.error(f"  first got={met[idx][0].tolist()} want={ref_meta[g, r][idx][0].tolist()}")
            bad += 1

    logger.info(f"pages compared byte-exact: {checked} across {H * G} devices " f"({checked_local} of them same-chip)")
    assert checked > 0, "no dispatched pages found; the reference or the routing is wrong"

    # A page is only placed for a token the allocator did not drop, so this is how many it dropped.
    routed = int(sum((table[g, indices[g]] != -1).sum() for g in range(G)))
    dropped = routed - int((src_of >= 0).sum())
    logger.info(f"tokens routed: {routed}, dropped for capacity: {dropped}")
    if capacity_div > 1:
        assert dropped > 0, "the tight config dropped nothing, so the clamp is still untested"
    # Without this the local phase could regress to writing nothing and the comparison would still pass.
    assert checked_local > 0, "no same-chip pages found; the local phase would go untested"
    assert bad == 0, f"{bad} device/tensor comparisons differ from the dispatch reference"


# --------------------------------------------------------------------------------------------------
# Host-side agreement test. No device: the whole dense-chunk protocol rests on the writer chip and the
# reader chip deriving identical chunk lengths and positions from the replicated offsets table, and
# that is pure arithmetic. On device the same invariant is only an ASSERT, which is compiled out unless
# TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS is set -- and watcher cannot run this op at all, because it
# inflates the fabric eth kernels past their config buffer. So this is where the invariant is actually
# checked, and it reaches extents the 8x4 galaxy cannot.
# --------------------------------------------------------------------------------------------------


def _slice_begin(n, idx, count):
    return (n * idx) // count


def _fwd_ref_frame(m):
    """(origin, dst) hop offsets of every chunk a chip forwards, in the order upstream writes them."""
    return [(origin, dst) for origin in range(-1, -m, -1) for dst in range(origin + m, 0, -1)]


def _forwarding_chunks(stream, my_row, extent, num_links):
    travel = 1 if stream % 2 == 0 else -1
    link, m = stream // 2, extent // 2
    out = []
    for origin, dst in _fwd_ref_frame(m):
        opposite = (dst - origin) == m
        out.append(
            (
                (my_row + travel * origin) % extent,
                (my_row + travel * dst) % extent,
                stream if opposite else link,
                2 * num_links if opposite else num_links,
            )
        )
    return out


def _outgoing_chunks(stream, my_row, extent, num_links):
    travel = 1 if stream % 2 == 0 else -1
    link, m = stream // 2, extent // 2
    nbr = (my_row + travel) % extent
    out = [
        (my_row, (my_row + travel * d) % extent, stream if d == m else link, 2 * num_links if d == m else num_links)
        for d in range(m, 1, -1)
    ]
    return out + [c for c in _forwarding_chunks(stream, my_row, extent, num_links) if c[1] != nbr]


@pytest.mark.parametrize("extent", [4, 6, 8, 12], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
def test_dispatch_fabric2d_chunk_agreement(extent, num_links, capacity_div):
    """What one chip writes into its neighbour's region is what that neighbour reads, position by position."""
    G, seq, topk = 2, 16, 4
    num_routed_experts = extent * topk * G  # so experts_per_group divides evenly by extent
    experts_per_chip = num_routed_experts // G // extent
    capacity = max(1, extent * seq * topk // capacity_div)

    torch.manual_seed(11)
    table = _expert_dispatch_table(num_routed_experts, extent, G)
    experts_per_group = num_routed_experts // G
    indices = torch.zeros(G, extent, seq, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(extent):
            for t in range(seq):
                indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]

    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g], extent, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs, counts, region = o[0].long(), c[0, 0].long(), r[0, 0].long()
        chip_experts = [[e for e in range(num_routed_experts) if table[g, e] == row] for row in range(extent)]
        for row in range(extent):
            assert len(chip_experts[row]) == experts_per_chip

        def run_len(origin, e):
            at = int(offs[origin, e])
            routed = int(offs[origin + 1, e]) - at if origin + 1 < extent else int(counts[e]) + int(region[e]) - at
            # Matches the kernel: tokens past the expert's capacity are dropped by the origin while its
            # counter still advances, so the table counts more than it ever sends.
            return min(routed, max(0, capacity - at))

        # The prologue's own count of what this chip owes each expert must equal what every other chip
        # derives for it from the table alone -- otherwise the two ends size the same chunk differently.
        for my in range(extent):
            alloc = offs[my].clone()
            bucket = torch.zeros(extent, experts_per_chip, dtype=torch.int64)
            for t in range(seq):
                for k in range(topk):
                    e = int(indices[g, my, t, k])
                    row = int(table[g, e])
                    if row == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    alloc[e] += 1
                    bucket[row, chip_experts[row].index(e)] += 1
            for row in range(extent):
                for j in range(experts_per_chip):
                    e = chip_experts[row][j]
                    assert int(bucket[row, j]) == run_len(my, e), (
                        f"g={g} my_row={my} dst_row={row} expert={e}: prologue counted "
                        f"{int(bucket[row, j])} but the table says {run_len(my, e)}"
                    )

        def starts(chunks):
            at, out = 0, []
            for origin, dst, idx, cnt in chunks:
                for j in range(experts_per_chip):
                    n = run_len(origin, chip_experts[dst][j])
                    out.append(at)
                    at += _slice_begin(n, idx + 1, cnt) - _slice_begin(n, idx, cnt)
            return out, at

        # What the host reserves per stream, without knowing any chunk length. The kernel only ever
        # checks this with an ASSERT, which is compiled out unless lightweight kernel asserts are on --
        # so an overflow silently writes into the next stream's slice of a shared DRAM tensor.
        m = extent // 2
        per_pair = seq * min(topk, experts_per_chip)
        div_up = lambda a, b: (a + b - 1) // b
        bound = sum((m - dd - 1) * div_up(per_pair, num_links) + div_up(per_pair, 2 * num_links) for dd in range(1, m))
        bound += (m * (m - 1) // 2) * experts_per_chip

        for stream in range(2 * num_links):
            travel = 1 if stream % 2 == 0 else -1
            for row in range(extent):
                nbr = (row + travel) % extent
                wrote, wrote_total = starts(_outgoing_chunks(stream, row, extent, num_links))
                reads, reads_total = starts(_forwarding_chunks(stream, nbr, extent, num_links))
                assert reads_total <= bound, (
                    f"g={g} stream={stream} row={nbr}: region needs {reads_total} pages but the host "
                    f"bound is {bound} -- this stream would write into the next stream's region"
                )
                assert wrote == reads and wrote_total == reads_total, (
                    f"g={g} stream={stream}: row {row} writes {wrote_total} pages at {wrote[:6]}... "
                    f"but row {nbr} reads {reads_total} at {reads[:6]}..."
                )


@pytest.mark.parametrize("seq_len_per_chip", [32, 128], ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
# Production routes over ALL experts, so only about a quarter of a token's picks land in this dispatch
# group and the rest resolve to -1. Every other test here routes entirely in-group, which is the
# simplification that hides whatever the -1 path does to the chunk arithmetic.
@pytest.mark.parametrize("cross_group", [False, True], ids=lambda c: "cross-group" if c else "in-group")
def test_dispatch_fabric2d_region_bound_production_shape(seq_len_per_chip, num_links, capacity_div, cross_group):
    """The device matrix's own geometry, checked on host: does any stream's region exceed the host bound?

    Same question the kernel's ASSERT(at <= fwd_pages_per_stream) asks, but reachable without a galaxy
    and without lightweight kernel asserts.
    """
    extent, G, num_routed_experts, topk = 8, 4, 256, 8
    experts_per_chip = num_routed_experts // G // extent
    capacity = max(1, extent * seq_len_per_chip * topk // capacity_div)

    torch.manual_seed(7)
    table = _expert_dispatch_table(num_routed_experts, extent, G)
    experts_per_group = num_routed_experts // G
    indices = torch.zeros(G, extent, seq_len_per_chip, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(extent):
            for t in range(seq_len_per_chip):
                if cross_group:
                    indices[g, h, t] = torch.randperm(num_routed_experts)[:topk]
                else:
                    indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]

    m = extent // 2
    per_pair = seq_len_per_chip * min(topk, experts_per_chip)
    div_up = lambda a, b: (a + b - 1) // b
    bound = sum((m - dd - 1) * div_up(per_pair, num_links) + div_up(per_pair, 2 * num_links) for dd in range(1, m))
    bound += (m * (m - 1) // 2) * experts_per_chip

    worst = 0
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g],
            extent,
            num_routed_experts,
            experts_per_chip,
            seq_len_per_chip,
            topk,
            expert_dispatch_table=table[g : g + 1],
        )
        offs, counts, region = o[0].long(), c[0, 0].long(), r[0, 0].long()
        chip_experts = [[e for e in range(num_routed_experts) if table[g, e] == row] for row in range(extent)]

        def run_len(origin, e):
            at = int(offs[origin, e])
            routed = int(offs[origin + 1, e]) - at if origin + 1 < extent else int(counts[e]) + int(region[e]) - at
            return min(routed, max(0, capacity - at))

        # check_buckets' invariant at the real geometry: the prologue's own tally must equal what the
        # table says, for every destination row including this chip's own.
        for my in range(extent):
            alloc = offs[my].clone()
            bucket = torch.zeros(extent, experts_per_chip, dtype=torch.int64)
            for t in range(seq_len_per_chip):
                for k in range(topk):
                    e = int(indices[g, my, t, k])
                    row = int(table[g, e])
                    if row == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    alloc[e] += 1
                    bucket[row, chip_experts[row].index(e)] += 1
            for row in range(extent):
                for j in range(experts_per_chip):
                    e = chip_experts[row][j]
                    assert int(bucket[row, j]) == run_len(my, e), (
                        f"g={g} my_row={my} dst_row={row} expert={e}: prologue counted "
                        f"{int(bucket[row, j])}, table says {run_len(my, e)}"
                    )

        for stream in range(2 * num_links):
            for row in range(extent):
                at = 0
                for origin, dst, idx, cnt in _forwarding_chunks(stream, row, extent, num_links):
                    for j in range(experts_per_chip):
                        n = run_len(origin, chip_experts[dst][j])
                        at += _slice_begin(n, idx + 1, cnt) - _slice_begin(n, idx, cnt)
                worst = max(worst, at)
                assert at <= bound, f"g={g} stream={stream} row={row}: region needs {at} pages, host bound is {bound}"
    logger.info(f"worst region occupancy {worst} of {bound} pages ({100 * worst / bound:.0f}%)")


# --------------------------------------------------------------------------------------------------
# Per-chip dedup chunk arithmetic, on host. One send per (origin, destination chip) rather than per
# (origin, destination, expert): a token picking several experts on one chip crosses the cable once,
# carrying a page list.
#
# The op does NOT implement this -- it implements multicast (see the next block), which subsumes it.
# Kept deliberately, as the A/B that says how much of the win each half is worth. Measured on the
# busiest directed link at production geometry with cross-group routing:
#
#     routing      today      dedup only       multicast
#     uniform      6,441      6,086 (1.06x)    5,262 (1.22x)
#     hot          9,625      8,589 (1.12x)    6,582 (1.46x)
#     hottest     12,832     10,786 (1.19x)    7,482 (1.72x)
#
# Two thirds of the win is multicast. Without this comparison the natural assumption is that dedup --
# much the simpler change -- is most of it, and it is not.
# --------------------------------------------------------------------------------------------------


def _fanout_presence(indices, table, offs, capacity, G, H, seq, topk, num_routed_experts):
    """presence[g][origin][dst] = tokens from `origin` with at least one SURVIVING expert on `dst`.

    Post-drop on purpose. A token whose every page on `dst` was dropped for capacity must not be
    counted, or the origin sends fewer pages than the relay waits for.
    """
    presence = torch.zeros(G, H, H, dtype=torch.int64)
    for g in range(G):
        for origin in range(H):
            alloc = offs[g, origin].clone().to(torch.int64)
            for t in range(seq):
                hit = set()
                for k in range(topk):
                    e = int(indices[g, origin, t, k])
                    row = int(table[g, e])
                    if row == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    alloc[e] += 1
                    hit.add(row)
                for row in hit:
                    presence[g, origin, row] += 1
    return presence


@pytest.mark.parametrize("extent", [4, 8], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
def test_dispatch_fabric2d_fanout_chunk_agreement(extent, num_links, capacity_div):
    """Region agreement under per-chip dedup. Retained for the dedup-vs-multicast A/B, not shipped."""
    G, seq, topk = 2, 16, 4
    num_routed_experts = extent * topk * G
    experts_per_chip = num_routed_experts // G // extent
    capacity = max(1, extent * seq * topk // capacity_div)

    torch.manual_seed(13)
    table = _expert_dispatch_table(num_routed_experts, extent, G)
    experts_per_group = num_routed_experts // G
    indices = torch.zeros(G, extent, seq, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(extent):
            for t in range(seq):
                indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]

    offs = torch.zeros(G, extent, num_routed_experts, dtype=torch.int64)
    for g in range(G):
        o, _, _, _ = get_gate_outputs(
            indices[g], extent, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs[g] = o[0].long()

    presence = _fanout_presence(indices, table, offs, capacity, G, extent, seq, topk, num_routed_experts)

    # Fan-out can only ever help: one send per destination chip instead of one per expert.
    sends_per_expert = 0
    for g in range(G):
        for origin in range(extent):
            alloc = offs[g, origin].clone().to(torch.int64)
            for t in range(seq):
                for k in range(topk):
                    e = int(indices[g, origin, t, k])
                    if int(table[g, e]) == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    alloc[e] += 1
                    sends_per_expert += 1
    assert int(presence.sum()) <= sends_per_expert, "fan-out sends more pages than per-expert does"

    for g in range(G):

        def run_len(origin, dst):
            return int(presence[g, origin, dst])

        def starts(chunks):
            at, out = 0, []
            for origin, dst, idx, cnt in chunks:
                n = run_len(origin, dst)
                out.append(at)
                at += _slice_begin(n, idx + 1, cnt) - _slice_begin(n, idx, cnt)
            return out, at

        for stream in range(2 * num_links):
            travel = 1 if stream % 2 == 0 else -1
            for row in range(extent):
                nbr = (row + travel) % extent
                wrote, wrote_total = starts(_outgoing_chunks(stream, row, extent, num_links))
                reads, reads_total = starts(_forwarding_chunks(stream, nbr, extent, num_links))
                assert wrote == reads and wrote_total == reads_total, (
                    f"g={g} stream={stream}: row {row} writes {wrote_total} pages at {wrote[:6]}... "
                    f"but row {nbr} reads {reads_total} at {reads[:6]}..."
                )


# --------------------------------------------------------------------------------------------------
# Multicast (drop-off) chunk arithmetic, on host.
#
# A token crosses a cable once per DIRECTION, not once per expert or even once per destination chip:
# the copy travels to the farthest destination that way and every chip en route that wants it keeps
# one and passes it on. The relay already forwards hop by hop, so this costs a consume, not a new
# transport -- but a chunk is now (origin, hop) and its length comes from a reach table, since
# per-expert counts cannot express "how many tokens travel at least h hops this way".
# --------------------------------------------------------------------------------------------------


def _mc_direction(origin, dst, extent):
    """Short way round; a tie at exactly half the ring goes clockwise so reach stays well defined."""
    d = (dst - origin) % extent
    return (1, d) if d <= extent - d else (-1, extent - d)


def _mc_far_lists(indices, table, offs, capacity, G, extent, seq, topk):
    """far[g][origin][dir_idx] = farthest hop of each travelling token, in token order.

    Post-drop: a token whose every surviving page lies elsewhere must not hold a hop open, or the
    origin sends fewer pages than the relay waits for. A token with no destination that way is not
    in the list at all.
    """
    lists = [[[[] for _ in range(2)] for _ in range(extent)] for _ in range(G)]
    for g in range(G):
        for origin in range(extent):
            alloc = offs[g, origin].clone().to(torch.int64)
            for t in range(seq):
                far = {1: 0, -1: 0}
                for k in range(topk):
                    e = int(indices[g, origin, t, k])
                    row = int(table[g, e])
                    if row == -1:
                        continue
                    if alloc[e] >= capacity:
                        alloc[e] += 1
                        continue
                    alloc[e] += 1
                    if row == origin:
                        continue
                    s, d = _mc_direction(origin, row, extent)
                    far[s] = max(far[s], d)
                for s, di in ((1, 0), (-1, 1)):
                    if far[s] > 0:
                        lists[g][origin][di].append(far[s])
    return lists


def _mc_reach(indices, table, offs, capacity, G, extent, seq, topk):
    """reach[g][origin][dir_idx][h] = tokens from origin whose farthest hop that way is >= h.

    dir_idx 0 is clockwise. Terminated by a zero at m + 1 so that reach[h] - reach[h + 1] is the
    number of tokens whose farthest hop is exactly h, for every h including m.
    """
    m = extent // 2
    reach = torch.zeros(G, extent, 2, m + 2, dtype=torch.int64)
    lists = _mc_far_lists(indices, table, offs, capacity, G, extent, seq, topk)
    for g in range(G):
        for origin in range(extent):
            for di in range(2):
                for far in lists[g][origin][di]:
                    for h in range(1, far + 1):
                        reach[g, origin, di, h] += 1
    return reach


def _mc_chunk(reach_row, h, link, num_links, m):
    """Pages one link carries of an origin's traffic at hop h.

    NOT a slice of reach[h]. A multicast chunk shrinks as chips consume from it, so a link's
    contiguous share of the hop-h list is not the share the hop-(h+1) list would hand it, and the
    two sides of a region end up expecting different counts -- a deadlock, and only when
    num_links > 1, which is why slicing reach[h] survived a one-link proof.

    Splitting each farthest-hop CLASS instead is stable under that shrinkage: a token keeps its link
    for the whole journey, so a link's pages at hop h are its shares of the classes with far >= h.
    Class sizes are reach[f] - reach[f + 1], already in the table. With one link it telescopes back
    to reach[h].
    """
    total = 0
    for f in range(h, m + 1):
        n = int(reach_row[f]) - int(reach_row[f + 1])
        total += _slice_begin(n, link + 1, num_links) - _slice_begin(n, link, num_links)
    return total


def _mc_region_hop(hop):
    """Reach index a region chunk is sized at, mirroring mc_region_hop in the reader.

    A page only enters a forwarding region when it still has work beyond the chip about to hold it.
    The origin writes its one-hop destinations straight into their output pages, so a chunk landing
    one hop out carries the hop-2 population rather than the hop-1 one.

    Sizing at reach[hop + 1] instead, so every relay delivers one hop ahead, cuts DRAM transfers about
    25% but raises link pages about 32%: a token that both delivers to the neighbour and forwards puts
    its payload on that link twice, and low link load is the whole of multicast's advantage.

    This has to track the kernel. It is the number both sides of a region derive independently, and
    a disagreement is a deadlock rather than wrong data.
    """
    return max(hop, 2)


def _mc_link_of(rank, class_size, num_links):
    """Which link a token rides, from its rank within its farthest-hop class."""
    for link in range(num_links - 1):
        if rank < _slice_begin(class_size, link + 1, num_links):
            return link
    return num_links - 1


@pytest.mark.parametrize("extent", [4, 8], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
def test_dispatch_fabric2d_multicast_chunk_agreement(extent, num_links, capacity_div):
    G, seq, topk = 2, 16, 4
    num_routed_experts = extent * topk * G
    experts_per_chip = num_routed_experts // G // extent
    capacity = max(1, extent * seq * topk // capacity_div)
    m = extent // 2

    torch.manual_seed(17)
    table = _expert_dispatch_table(num_routed_experts, extent, G)
    epg = num_routed_experts // G
    indices = torch.zeros(G, extent, seq, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(extent):
            for t in range(seq):
                indices[g, h, t] = g * epg + torch.randperm(epg)[:topk]

    offs = torch.zeros(G, extent, num_routed_experts, dtype=torch.int64)
    for g in range(G):
        o, _, _, _ = get_gate_outputs(
            indices[g], extent, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs[g] = o[0].long()

    reach = _mc_reach(indices, table, offs, capacity, G, extent, seq, topk)
    far_lists = _mc_far_lists(indices, table, offs, capacity, G, extent, seq, topk)
    # Monotone by construction; if this ever breaks the chunk lengths below go negative.
    for g in range(G):
        for o_ in range(extent):
            for di in range(2):
                for h in range(1, m + 1):
                    assert reach[g, o_, di, h] >= reach[g, o_, di, h + 1]

    # Flow conservation, which region agreement alone does NOT give: the two sides of a region can
    # agree on a length that neither the origin nor any relay can actually produce. Walk the tokens,
    # give each the link its farthest-hop class rank earns it, and count what survives to each hop.
    # This is the check that fails for the obvious rule -- slice reach[h] -- at num_links > 1.
    for g in range(G):
        for origin in range(extent):
            for di in range(2):
                seen = {}
                held = [[0] * (m + 2) for _ in range(num_links)]
                for far in far_lists[g][origin][di]:
                    size = int(reach[g, origin, di, far]) - int(reach[g, origin, di, far + 1])
                    rank = seen.get(far, 0)
                    seen[far] = rank + 1
                    link = _mc_link_of(rank, size, num_links)
                    for h in range(1, far + 1):
                        held[link][h] += 1
                for link in range(num_links):
                    for h in range(1, m + 1):
                        want = _mc_chunk(reach[g, origin, di], h, link, num_links, m)
                        assert held[link][h] == want, (
                            f"g={g} origin={origin} dir={di} link={link} hop={h}: the link holds "
                            f"{held[link][h]} pages but every chip sizes that chunk at {want}"
                        )

    # What the host reserves per stream without knowing a single chunk length. Overrun is silent --
    # a stream writes into the next stream's region -- and the kernel's only guard is a compiled-out
    # ASSERT, so the bound is checked here.
    bound = m * (-(-seq // num_links) + m)

    for g in range(G):

        def chunk(origin, di, h, link):
            return _mc_chunk(reach[g, origin, di], h, link, num_links, m)

        for stream in range(2 * num_links):
            travel = 1 if stream % 2 == 0 else -1
            di, link = (0 if travel == 1 else 1), stream // 2
            for row in range(extent):
                nbr = (row + travel) % extent
                # what row puts on the cable: its own tokens, then each upstream origin pushed one hop on
                out, at = [], 0
                for j in range(0, m):
                    origin = (row - j * travel) % extent
                    out.append(at)
                    at += chunk(origin, di, _mc_region_hop(j + 1), link)
                # what nbr reads: the same origins, each one hop further along
                rd, at2 = [], 0
                for j in range(1, m + 1):
                    origin = (nbr - j * travel) % extent
                    rd.append(at2)
                    at2 += chunk(origin, di, _mc_region_hop(j), link)
                assert out == rd and at == at2, (
                    f"g={g} stream={stream}: row {row} writes {at} pages at {out} " f"but row {nbr} reads {at2} at {rd}"
                )
                assert at2 <= bound, (
                    f"g={g} stream={stream} row={nbr}: region needs {at2} pages but the host bound is "
                    f"{bound} -- this stream would write into the next stream's region"
                )
