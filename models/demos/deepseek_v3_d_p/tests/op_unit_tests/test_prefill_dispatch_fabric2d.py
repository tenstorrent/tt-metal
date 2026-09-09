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
def test_dispatch_fabric2d(
    mesh_device, device_params, num_links, seq_len_per_chip, capacity_div, num_routed_experts, emb_dim
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

    payload, metadata = ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
        tt_x,
        tt_idx,
        tt_offs,
        tt_table,
        tt_counts,
        tt_region,
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
def test_dispatch_fabric2d_region_bound_production_shape(seq_len_per_chip, num_links, capacity_div):
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
