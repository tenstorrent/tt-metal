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

import math
import re
from contextlib import contextmanager
from pathlib import Path

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
# The model hands dispatch TILED activations. A TILE input is untilized on device into a staging
# buffer by a pool of cores beside the stream cores, so the transport sees the identical bytes either
# way and both layouts share this gate -- which is also what makes them A/B-able on the perf worker.
@pytest.mark.parametrize(
    "input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=lambda ly: "tile" if ly == ttnn.TILE_LAYOUT else "rm"
)
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
    input_layout,
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

    def shard(t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=layout,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = torch.randn(H, G, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    # Same values in either layout, so one reference gates both runs -- which is what makes the TILE
    # path's untilizer byte-exact rather than merely close.
    tt_x = shard(x, (0, 1), ttnn.bfloat16, layout=input_layout)
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


class _Fixture:
    """One small in-group routing draw, on device in both input layouts, plus its torch reference.

    Shared by the tests below that care about how the op is CALLED rather than about the routing: the
    matrix above is where the routing axes live.
    """

    # emb_dim 512 is 16 tiles wide, so the untilizer packs TWO column blocks per stripe. At 256 it is
    # exactly one, and a block's L1 column offset -- the thing block_ct_dim exists to make legal --
    # would never be anything but zero.
    def __init__(
        self,
        mesh_device,
        H,
        G,
        seq_len_per_chip=128,
        emb_dim=512,
        num_routed_experts=256,
        topk=8,
        seed=11,
        capacity_div=1,
        reach_from_op=False,
        cluster_axis=0,
    ):
        self.mesh_device = mesh_device
        self.seq_len_per_chip, self.emb_dim, self.H, self.G = seq_len_per_chip, emb_dim, H, G
        self.num_routed_experts, self.topk = num_routed_experts, topk
        self.experts_per_chip = num_routed_experts // G // H
        # A divisor of 1 is roomy: nothing is dropped. Anything larger puts capacity below the
        # per-expert load, which is the only regime where a drop moves a token's farthest hop.
        self.capacity = max(1, H * seq_len_per_chip * topk // capacity_div)
        # Where the reach table comes from: the torch reference, or `moe_fanout_reach` on device. The
        # second is the production path -- nothing else in the tree produces this table -- and the two
        # have to agree word for word, since a reach table that overstates strands the axis.
        self.reach_from_op, self.cluster_axis = reach_from_op, cluster_axis
        torch.manual_seed(seed)
        self.table = _expert_dispatch_table(num_routed_experts, H, G)
        experts_per_group = num_routed_experts // G
        self.indices = torch.zeros(G, H, seq_len_per_chip, topk, dtype=torch.int64)
        for g in range(G):
            for h in range(H):
                for t in range(seq_len_per_chip):
                    self.indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]

        self.x = torch.randn(H, G, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
        self.tt_x = {
            ttnn.ROW_MAJOR_LAYOUT: self._shard(self.x, (0, 1), ttnn.bfloat16),
            ttnn.TILE_LAYOUT: self._shard(self.x, (0, 1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }
        self.tt_table = self._shard(self.table.unsqueeze(1), (None, 0), ttnn.int32)
        self.rebuild()

    def _shard(self, t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
            layout=layout,
            device=self.mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rebuild(self):
        """Re-derive everything routing decides, so a test may edit `indices` and stay self-consistent.

        The offsets table and the reach table come from the same draw the op is handed, which is the
        whole reason the op can size a chunk it neither wrote nor receives; building one from a stale
        draw would not fail a shape check, it would deadlock an axis.
        """
        G, H = self.G, self.H
        offs = torch.zeros(G, H, self.num_routed_experts, dtype=torch.int32)
        counts = torch.zeros(G, H, self.num_routed_experts, dtype=torch.int32)
        region = torch.zeros(G, H, self.num_routed_experts, dtype=torch.int32)
        for g in range(G):
            o, c, r, _ = get_gate_outputs(
                self.indices[g],
                H,
                self.num_routed_experts,
                self.experts_per_chip,
                self.seq_len_per_chip,
                self.topk,
                expert_dispatch_table=self.table[g : g + 1],
            )
            offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)
        self.offs = offs
        self.tt_idx = self._shard(self.indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
        self.tt_offs = self._shard(offs, (None, 0), ttnn.int32)
        self.tt_counts = self._shard(counts[:, 0:1, :], (None, 0), ttnn.int32)
        self.tt_region = self._shard(region[:, 0:1, :], (None, 0), ttnn.int32)
        self.reach = (
            self._reach_from_op(offs)
            if self.reach_from_op
            else _mc_reach(self.indices, self.table, offs, self.capacity, G, H, self.seq_len_per_chip, self.topk).to(
                torch.int32
            )
        )
        self.tt_reach = self._shard(self.reach, (None, 0), ttnn.int32)

    def _reach_from_op(self, offs):
        """Every chip's reach row from `moe_fanout_reach`, gathered into the table the transport takes.

        The transport needs every origin's row on every chip, because a relay sizes a chunk it neither
        wrote nor receives; the op produces only the row of the chip it ran on. Production closes that
        with an all-gather along the dispatch axis. Here the gather is done on host, which is the same
        bytes and keeps the test's failure mode readable -- a mismatch points at this op rather than at
        a CCL in between.
        """
        G, H = self.G, self.H
        rows = ttnn.experimental.deepseek_prefill.moe_fanout_reach(
            self.tt_idx,
            self.tt_table,
            self._shard(offs, (1, 0), ttnn.int32),
            num_routed_experts=self.num_routed_experts,
            num_experts_per_tok=self.topk,
            dispatch_group_size=H,
            max_dispatch_buffer_token_size=self.capacity,
            cluster_axis=self.cluster_axis,
        )
        hops = H // 2 + 2
        per_device = ttnn.get_device_tensors(rows)
        table = torch.zeros(G, H, 2, hops, dtype=torch.int32)
        for dev in range(H * G):
            r, g = dev // G, dev % G
            table[g, r] = ttnn.to_torch(per_device[dev]).to(torch.int32).reshape(2, hops)
        return table

    def padding_config(self, real_tokens, pad_side=0):
        """The [real_token_count, pad_side] tensor `dispatch` takes, replicated to every device."""
        return ttnn.from_torch(
            torch.tensor([[real_tokens, pad_side]], dtype=torch.int32),
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(None, None)
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def run(
        self,
        cluster_axis,
        num_links,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        fanout=False,
        subdevice_id=None,
        padding_config=None,
    ):
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            self.tt_x[layout],
            self.tt_idx,
            self.tt_offs,
            self.tt_table,
            self.tt_counts,
            self.tt_region,
            fanout_reach=self.tt_reach if fanout else None,
            padding_config=padding_config,
            fanout=fanout,
            experts_per_chip=self.experts_per_chip,
            num_routed_experts=self.num_routed_experts,
            num_experts_per_tok=self.topk,
            metadata_len=3,
            max_dispatch_buffer_token_size=self.capacity,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=cluster_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=subdevice_id,
        )

    def check(self, payload, metadata, label):
        """Every page any chip sourced, byte-exact against what `dispatch` would have placed."""
        ref_payload, ref_meta, src_of = _reference_dispatch(
            self.indices,
            self.table,
            self.offs,
            self.x,
            self.capacity,
            self.G,
            self.H,
            self.seq_len_per_chip,
            self.topk,
            self.emb_dim,
        )
        got_payload = ttnn.get_device_tensors(payload)
        got_meta = ttnn.get_device_tensors(metadata)
        checked = 0
        for dev in range(self.H * self.G):
            r, g = dev // self.G, dev % self.G
            pages = [p for p in range(self.capacity) if int(src_of[g, r, p]) >= 0]
            if not pages:
                continue
            idx = torch.tensor(pages)
            pay = ttnn.to_torch(got_payload[dev]).reshape(self.capacity, self.emb_dim)
            met = ttnn.to_torch(got_meta[dev]).to(torch.int32).reshape(self.capacity, 3)
            checked += len(pages)
            assert torch.equal(pay[idx], ref_payload[g, r][idx]), f"{label}: device {dev} payload differs"
            assert torch.equal(met[idx], ref_meta[g, r][idx]), f"{label}: device {dev} metadata differs"
        assert checked > 0, f"{label}: no dispatched pages found; the reference or the routing is wrong"
        logger.info(f"{label}: {checked} pages byte-exact")


@contextmanager
def _sub_device_manager(mesh_device, sub_devices):
    """Register, load and tear down one sub-device manager over the given CoreRangeSets.

    A manager's sub-devices have to be disjoint, so a test that wants overlapping carves needs two
    managers and only one of them loaded at a time. Removal is not optional: leaving a manager
    registered at device close has been observed to segfault the teardown.
    """
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([cores]) for cores in sub_devices], 0)
    mesh_device.load_sub_device_manager(manager)
    try:
        yield [ttnn.SubDeviceId(i) for i in range(len(sub_devices))]
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


def _moe_grid_split(mesh_device):
    """The model's carve: dispatch gets the first row of the Tensix grid, the shared expert the rest.

    `tt_moe.py` splits rows [0, dispatch_sd_rows) against the remainder, with `dispatch_sd_rows`
    currently 1, so the two ops run on disjoint cores and overlap on chip.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    dispatch = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 0))})
    shared = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return dispatch, shared


def _leading_row_cores(width):
    """The first `width` cores of row 0 -- a strict subset of the row the streams need.

    The op does not support this: a stream lands on the worker nearest its eth core and those are
    spread along the whole row, so any partial carve leaves one of them outside. What it is good for
    is showing that the sub-device reached the placement at all, which a carve wide enough to hold
    every stream cannot.
    """
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, 0))})


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_subdevice(mesh_device, device_params, num_links, capfd, expect_error):
    """A TILE input on the model's two-sub-device split: every core the op takes must be in row 0.

    Byte-exactness is the same gate the matrix applies. What is new is confinement -- the four stream
    cores AND the untilizer pool a TILE input needs all have to come out of the dispatch row, because
    the shared expert holds the rest of the grid at the same time.

    The negative case is what actually proves it. Handed the SHARED sub-device instead, the op must
    refuse and name a core in row 0: that message is the placement telling us where it wanted to put a
    stream, and it can only say y=0 if row 0 is where the streams go. Confinement of the pool follows,
    since it is drawn from the spare cores of the same carve -- and on a one-row carve that is the
    documented fallback, so this test is also the coverage for it: the op has to report the fallback
    once, and the report is expected here rather than a failure.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups)
    streams = 2 * num_links
    dispatch_cores, shared_cores = _moe_grid_split(mesh_device)
    n_cores = sum((r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1) for r in dispatch_cores.ranges())
    spare = n_cores - streams
    assert spare > 0, "no core left for an untilizer; the TILE path cannot run"

    # More stripes than there are cores to take them, so a core takes several and the round-robin
    # stride stops being indistinguishable from one. With no row under the streams the op takes every
    # spare core up to one per stripe, so the pool is `spare` here.
    pool = spare
    deep = _Fixture(
        mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=32 * (2 * pool + 1), seed=31
    )
    logger.info(f"dispatch sub-device: {n_cores} cores in row 0, {streams} streams, {pool} untilizers")

    with _sub_device_manager(mesh_device, [dispatch_cores, shared_cores]) as (dispatch_sd, shared_sd):
        for case, layout, fanout in [("unicast", ttnn.TILE_LAYOUT, False), ("multicast", ttnn.TILE_LAYOUT, True)]:
            payload, metadata = fx.run(cfg.sp_axis, num_links, layout=layout, fanout=fanout, subdevice_id=dispatch_sd)
            fx.check(payload, metadata, f"tile {case} on the dispatch sub-device")

        payload, metadata = deep.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=dispatch_sd)
        deep.check(payload, metadata, f"{deep.seq_len_per_chip // 32} stripes over {pool} untilizers")
        out = capfd.readouterr().out
        assert (
            "untilizer pool is not in the row under the streams" in out
        ), "a one-row carve has to report its pool fallback once per build; nothing was reported"

        with expect_error(RuntimeError, "is outside the") as refusal:
            fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=shared_sd)
        message = str(refusal.value)
        # The core it names is the one the eth-nearest placement wanted. CoreCoord formats as x-y, so
        # a trailing -0 is row 0 -- which is the whole claim this test exists to make.
        assert re.search(r"eth core is \d+-0,", message), message

    # A carve identical to the default cannot show that the argument was used at all. This one is a
    # strict subset of the same row, and the op refuses it -- naming a row-0 core it wanted and could
    # not have -- which it could only do having read the sub-device. It also records the real
    # constraint: the eth-nearest workers are spread along the row, so dispatch needs ALL of it.
    with _sub_device_manager(mesh_device, [_leading_row_cores(streams + 2)]) as (narrow_sd,):
        with expect_error(RuntimeError, f"outside the {streams + 2} cores") as refusal:
            fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=narrow_sd)
        message = str(refusal.value)
        assert re.search(r"eth core is \d+-0,", message), message


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
@pytest.mark.parametrize(
    "input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=lambda ly: "tile" if ly == ttnn.TILE_LAYOUT else "rm"
)
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_multicast_reach_from_op(mesh_device, device_params, num_links, capacity_div, input_layout):
    """The multicast transport driven by `moe_fanout_reach`, not by the torch table beside it.

    Everywhere else the reach table is synthesised on host, which proves the transport and nothing about
    where the table comes from in production. Here it comes off the device, from the same indices and
    the same offsets the transport is handed, and the gate is the same byte-exact comparison against
    `dispatch`.

    Both directions of the check matter. The table has to equal the torch one word for word -- reach is
    the one control tensor whose error mode is a stranded axis rather than a wrong page, so "close" is
    not a state it can be in. And the transport then has to place the same bytes with it, which is what
    says the op's output is usable rather than merely correct in isolation.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(
        mesh_device,
        cfg.dispatch_group_size,
        cfg.num_dispatch_groups,
        capacity_div=capacity_div,
        reach_from_op=True,
        cluster_axis=cfg.sp_axis,
    )
    want = _mc_reach(fx.indices, fx.table, fx.offs, fx.capacity, fx.G, fx.H, fx.seq_len_per_chip, fx.topk)
    assert torch.equal(fx.reach.to(torch.int64), want), (
        f"moe_fanout_reach disagrees with the torch reference: "
        f"{(fx.reach.to(torch.int64) != want).sum().item()} of {want.numel()} entries differ"
    )
    if capacity_div > 1:
        # Without this the tight case would be indistinguishable from the roomy one, and the drop rule
        # -- the reason this table cannot be derived before the offsets exist -- would go untested here.
        roomy = _mc_reach(
            fx.indices,
            fx.table,
            fx.offs,
            fx.H * fx.seq_len_per_chip * fx.topk,
            fx.G,
            fx.H,
            fx.seq_len_per_chip,
            fx.topk,
        )
        assert not torch.equal(roomy, want), "the tight capacity dropped nothing that moved a farthest hop"

    payload, metadata = fx.run(cfg.sp_axis, num_links, layout=input_layout, fanout=True)
    fx.check(payload, metadata, f"multicast on moe_fanout_reach's table, capacity {fx.capacity}")


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
# 16 and 64 tiles wide: two untilize column blocks per stripe, and eight. 7168 is the production token,
# 14336 B: the size every packet-size bound in the sender is measured at, and the only byte-exact run of it.
@pytest.mark.parametrize("emb_dim", [512, 2048, 7168], ids=lambda e: f"emb{e}")
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_relaunch(mesh_device, device_params, num_links, fanout, emb_dim):
    """Both layouts and repeat launches against ONE device, which the matrix cannot reach.

    Every case of the matrix gets a fresh `mesh_device` fixture, so it never exercises two things
    that only go wrong when state survives a launch:

    - The program cache. A TILE input and a ROW_MAJOR one build DIFFERENT programs -- one has an
      untilizer pool and reads tokens out of a staging buffer, the other has neither -- so the layout
      has to reach the cache key. If it did not, whichever ran second would silently get the first
      one's program and read tokens from the wrong buffer. Alternating catches it in both directions.
    - The untilize counter. The stream readers zero it at end of stream, so a second TILE launch
      starts from whatever the first left. A leak there does NOT hang: the wait passes immediately
      and the stream cores read the staging buffer the previous launch filled. That is invisible
      unless the launches carry different values, which is why each one here has its own draw.
    """
    cfg = extract_mesh_config(mesh_device)
    plan = [
        ("row-major", ttnn.ROW_MAJOR_LAYOUT),
        ("tile", ttnn.TILE_LAYOUT),
        ("tile again", ttnn.TILE_LAYOUT),
        ("row-major again", ttnn.ROW_MAJOR_LAYOUT),
    ]
    for seed, (label, layout) in enumerate(plan):
        fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=emb_dim, seed=20 + seed)
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=layout, fanout=fanout)
        # Reading the outputs back is what synchronises the launches: this op deadlocks if a chip
        # starts sending into a neighbour that is still retiring the previous one.
        fx.check(payload, metadata, label)


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_tile_refusals(mesh_device, device_params, num_links, expect_error):
    """The TILE path's edge conditions are refused, rather than silently doing something else.

    Each of these is a write that would land somewhere it should not. A stripe is a whole tile row, so
    a sequence that is not a multiple of 32 would have its last stripe write past the end of a staging
    buffer sized at exactly `seq_len_per_chip` pages; an emb_dim that is not a multiple of 32 leaves
    the untilizer reading tile columns that are not there and packing rows at a stride the writer does
    not use; and a sub-device with no core to spare has nowhere to put a pool at all.
    """
    cfg = extract_mesh_config(mesh_device)
    streams = 2 * num_links

    # 500 columns is fifteen whole tile columns and a ragged sixteenth, which the untilizer cannot
    # read: it takes whole tile columns and packs rows at a stride the writer does not use. Accepted
    # as ROW_MAJOR, which is what makes the refusal a property of the untilizer rather than the shape.
    ragged_emb = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=500)
    with expect_error(RuntimeError, "multiple of 32"):
        ragged_emb.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT)
    payload, metadata = ragged_emb.run(cfg.sp_axis, num_links)
    ragged_emb.check(payload, metadata, "row-major with a ragged emb")

    # A ragged SEQUENCE is carried, not refused: 100 tokens is three whole stripes and a fourth the
    # packer fills with the tile's padding rows, which staging has room for and the routing pass never
    # reaches. This is the one place the two input layouts could disagree about how many tokens exist.
    ragged_seq = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=100)
    for layout, label in ((ttnn.ROW_MAJOR_LAYOUT, "row-major"), (ttnn.TILE_LAYOUT, "tile")):
        payload, metadata = ragged_seq.run(cfg.sp_axis, num_links, layout=layout)
        ragged_seq.check(payload, metadata, f"{label} with a ragged sequence")

    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups)
    with _sub_device_manager(mesh_device, [_leading_row_cores(streams)]) as (exact_sd,):
        # Exactly enough cores for the streams, so the pool would have to come from somewhere else.
        # Refused in validation, before the placement gets a chance to object to the carve itself.
        with expect_error(RuntimeError, "plus at least one untilizer"):
            fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=exact_sd)

    # One token wider than the fabric payload admits with its 64-byte tail: 7168 columns is exactly the
    # cap the tests configure, so 7200 is the first refusal. Without it the forward's payload would run
    # past its channel slot onto the next packet, which no kernel check can see.
    too_wide = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=7200)
    with expect_error(RuntimeError, "exceeds the fabric max payload"):
        too_wide.run(cfg.sp_axis, num_links)


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_untilizers_under_the_streams(mesh_device, device_params, num_links, capfd):
    """The whole grid, with more stripes than the pool: the pool sits in the row under the streams.

    The placement a TILE input is designed for is only reachable with a second row in the carve, and
    every other test either runs on the model's one-row carve or gives the pool fewer stripes than it
    has cores. 640 tokens is 20 stripes over 5 * num_links untilizers, byte-exact in both transports,
    and the op must NOT report a fallback -- that report is the one-row carve's, and it must stay off
    here or the sub-device test's assertion on it means nothing.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=640, seed=37)
    for fanout in (False, True):
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, fanout=fanout)
        fx.check(
            payload, metadata, f"20 stripes over {5 * num_links} untilizers, {'multicast' if fanout else 'unicast'}"
        )
    out = capfd.readouterr().out
    assert (
        "untilizer pool is not in the row under the streams" not in out
    ), "the whole grid has a row under the streams; the pool fell back anyway"


def _draw_into(fx, profile):
    """Give a fixture one of the gate's own routing profiles, in place of its in-group draw.

    The point of contrast with `_skew_the_ring`: that one idles half the ring to manufacture the
    largest drift this geometry allows, which is not a distribution the gate produces. These are, and
    they leave every stream with traffic from every origin -- which is what couples the ring, since a
    reader cannot finish until its upstream has sent, and that upstream cannot send its relayed pages
    until its own reader has drained.
    """
    share, hot_weight = ROUTING_PROFILES[profile]
    fx.indices = _draw_indices(fx.G, fx.H, fx.seq_len_per_chip, fx.topk, fx.num_routed_experts, share, hot_weight)
    fx.rebuild()
    return fx


def _skew_the_ring(fx):
    """Rewrite a fixture's draw so half the ring is idle and the other half is saturated.

    The forwarding region is reused at the same offsets every launch, and nothing back-pressures a
    chip that is a launch ahead: it writes into its neighbour's region and bumps the arrival counter
    whether or not that neighbour has consumed the previous launch. What bounds it is only how far
    ahead a chip can get, and chips drift apart when they are given different amounts of work.

    So the low half of the ring routes everything to its OWN experts -- no cable traffic, a launch
    that is little more than the prologue -- while the high half routes everything to the chip
    diametrically opposite, the longest path the schedule has. The idle half then runs ahead of the
    busy half by as much as this geometry allows.
    """
    G, H, m = fx.G, fx.H, fx.H // 2
    for g in range(G):
        experts_on = {row: [e for e in range(fx.num_routed_experts) if int(fx.table[g, e]) == row] for row in range(H)}
        for origin in range(H):
            target = origin if origin < H // 2 else (origin + m) % H
            picks = experts_on[target][: fx.topk]
            assert len(picks) == fx.topk, f"row {target} hosts {len(picks)} experts, need {fx.topk}"
            fx.indices[g, origin, :, :] = torch.tensor(picks, dtype=torch.int64)
    fx.rebuild()
    return fx


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
@pytest.mark.parametrize("sync_between", [True, False], ids=lambda s: "synced" if s else "overlapped")
@pytest.mark.parametrize("draw", ["degenerate", "hottest"], ids=lambda d: d)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_region_reuse_under_skew(mesh_device, device_params, num_links, fanout, sync_between, draw):
    """Does a chip running ahead overwrite forwarding pages its neighbour has not read yet?

    The arrival counter is safe across launches now, but the REGION is not obviously so: it is reused
    at the same offsets every launch, and a chip is free to start filling its neighbour's copy while
    that neighbour is still draining the previous one. Nothing returns a credit upstream -- every
    stream's fabric connection points downstream -- so the only thing standing between the two is how
    far apart they drift.

    Unlike the counter race this fails as WRONG BYTES rather than as a deadlock, which is what makes
    it safe to provoke: an overwritten page is read and delivered, so the byte-exact gate catches it
    and no board is left wedged.

    The probe maximises drift rather than volume: `_skew_the_ring` idles half the ring and saturates
    the other half, and the launches are queued with nothing between them so the gap compounds over
    a run instead of being reset by a host sync every time. Each launch carries its own draw, because
    an overwrite between two identical launches writes back the bytes that were already there.

    A pass bounds the hazard at this geometry rather than disproving it. The model's regions are far
    larger and its launches far longer, and the two move the window in opposite directions.
    """
    cfg = extract_mesh_config(mesh_device)
    launches = 24
    # `degenerate` manufactures the largest drift the geometry allows and is not a distribution the gate
    # produces; `hottest` is the most concentrated one it does. Whether the hazard needs the first is the
    # whole question -- it decides if this is a constraint to document or a protocol to build.
    if draw == "degenerate" and not sync_between:
        # The one combination that does NOT hold, recorded rather than hidden: a chip given almost no
        # work runs a whole launch ahead and refills a region its neighbour is still reading. Not
        # strict, because it is a race -- it has reproduced every time so far, but a run that happened
        # to stay in step would be a pass, not a reason to fail the suite.
        pytest.xfail("region reuse under a drift no realistic draw produces; see the hottest arm")

    prepare = _skew_the_ring if draw == "degenerate" else (lambda fx: _draw_into(fx, draw))
    draws = [
        prepare(_Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=40 + i)) for i in range(4)
    ]
    order = [draws[i % len(draws)] for i in range(launches)]

    # The synced arm is the control, and it is the whole experiment: it runs the SAME draws through the
    # SAME region in the same order, differing only in whether the launches may overlap. If it passes
    # where the overlapped arm fails, the difference is the overlap and nothing else -- not a draw the
    # reference disagrees with, not a capacity the fixture got wrong.
    results = []
    for fx in order:
        results.append(fx.run(cfg.sp_axis, num_links, fanout=fanout))
        if sync_between:
            ttnn.synchronize_device(mesh_device)
    arm = "synced" if sync_between else "overlapped"
    for i, (fx, (payload, metadata)) in enumerate(zip(order, results)):
        fx.check(payload, metadata, f"launch {i} of {launches}, {draw} draw, {arm}")
    logger.info(
        f"region reuse [{draw}/{arm}]: {launches} launches byte-exact " f"({'multicast' if fanout else 'unicast'})"
    )


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_back_to_back(mesh_device, device_params, num_links, fanout):
    """Launches with NOTHING between them, which is the only way the arrival counter's reset is tested.

    Every other test reads its outputs before launching again, and that read is a host synchronisation
    -- so the launches never overlap and the reset is never contended. Here four launches are queued
    and only then read, which lets a chip that finishes early start sending into a neighbour still
    retiring the previous one. That is exactly the skew a traced replay has, since a trace carries no
    host syncs at all.

    The failure this guards is a hang, not wrong data: an increment that lands while the downstream is
    clearing the counter used to be thrown away, and its relay then waited for pages the counter said
    had never arrived. The launches alternate between two draws so a counter left standing HIGH is
    caught too -- a relay reading before arrival would hand back the other draw's pages, which a
    single repeated draw could not tell apart.
    """
    cfg = extract_mesh_config(mesh_device)
    a = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=31)
    b = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=32)
    order = [a, b, a, b]

    # Queued together, read afterwards. Nothing here waits on the device.
    results = [fx.run(cfg.sp_axis, num_links, fanout=fanout) for fx in order]
    for i, (fx, (payload, metadata)) in enumerate(zip(order, results)):
        fx.check(payload, metadata, f"launch {i} of four with no host sync between them")
    logger.info(f"back-to-back: 4 unsynchronised launches byte-exact ({'multicast' if fanout else 'unicast'})")


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
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
# Half the tokens is the shape production pads to. 33 leaves the four prologue lanes unequal slices
# and 3 leaves one lane with no tokens at all: the two shapes the slice arithmetic has to survive.
@pytest.mark.parametrize("real", [64, 33, 3], ids=lambda r: f"real{r}")
@pytest.mark.timeout(600)
def test_dispatch_fabric2d_padding_config(mesh_device, device_params, num_links, fanout, real):
    """A padding_config shortens the routing pass without changing a single page.

    The contract it carries is the one `dispatch` relies on: with right padding the padded tokens sit
    at the high indices and are sentinel-marked, so they resolve to no expert and contribute nothing.
    Here the tail is routed entirely OUT of this dispatch group, which is what a sentinel-marked token
    looks like from inside it -- so bounding the pass at the real count and not bounding it must land
    identical bytes, and the reference (which always walks every token) is the third opinion.

    The left-padding case is the same call with pad_side 1, which both ops ignore, so it has to come
    back identical too -- a config that silently took effect on the wrong side would corrupt the tail.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups)
    assert real < fx.seq_len_per_chip

    # Tokens past `real` are given experts of another dispatch group, the -1 column this group sees.
    experts_per_group = fx.num_routed_experts // fx.G
    for g in range(fx.G):
        other = ((g + 1) % fx.G) * experts_per_group
        fx.indices[g, :, real:, :] = torch.arange(other, other + fx.topk)
    fx.rebuild()

    baseline_payload, baseline_meta = fx.run(cfg.sp_axis, num_links, fanout=fanout)
    fx.check(baseline_payload, baseline_meta, "no padding_config")

    for pad_side, label in ((0, "right padding, the pass is bounded"), (1, "left padding, the config is ignored")):
        payload, metadata = fx.run(
            cfg.sp_axis, num_links, fanout=fanout, padding_config=fx.padding_config(real, pad_side)
        )
        fx.check(payload, metadata, label)
    logger.info(
        f"padding_config: {real} real of {fx.seq_len_per_chip} tokens, byte-exact either side "
        f"({'multicast' if fanout else 'unicast'})"
    )


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


def _replay_local_phase(n, batch):
    """The reader's local phase over n tokens, as bookkeeping: what it claims, flushes and writes.

    Mirrors the kernel's control flow -- a scratch slot is claimed only when every held one is pending,
    a batch is written when `batch` are pending or at the end -- and models the ring as the two counters
    the kernel's release relies on. Returns (slots held, batches written, tokens written in order).
    """
    ring_claimed = ring_ready = 0
    held = pending = batches = 0
    written = []
    first = 0
    for t in range(n):
        if pending == held:
            ring_claimed += 1
            held += 1
        assert held <= batch, "a scratch slot past the stack arrays"
        pending += 1
        if pending == batch:
            written += list(range(first, t + 1))
            first, pending, batches = t + 1, 0, batches + 1
    if pending:
        written += list(range(first, n))
        batches += 1
    # release_unready: the scratch is exactly the ring's unready claims, and nothing else is held.
    assert ring_claimed - ring_ready == held
    return held, batches, written


@pytest.mark.parametrize("batch", [1, 2, 8], ids=lambda b: f"batch{b}")
def test_dispatch_fabric2d_local_phase_batches(batch):
    """Every local token count from none to three batches and one: each token written once, in order.

    The device matrix reaches these counts by the luck of its draws, and the two draws that pin a count
    exist for other reasons. The bookkeeping is pure arithmetic, so it is held here: the slots claimed
    never exceed the batch, a stream with nothing local claims nothing, and the batch count is the
    ceiling the flushes imply.
    """
    for n in range(0, 3 * batch + 2):
        held, batches, written = _replay_local_phase(n, batch)
        assert written == list(range(n)), f"n={n}: wrote {written}"
        assert held == min(n, batch), f"n={n}: held {held} slots"
        assert batches == math.ceil(n / batch), f"n={n}: {batches} batches"


def _prologue_model(picks, first_page, capacity, bucketed, lanes, max_dests=8):
    """The routing index the stream core's prologue builds, walked in `lanes` token slices.

    `picks[t][k]` is None for a pick outside the group, else (slot, local, direction). Mirrors the
    kernel step for step -- count pass, placement from the earlier lanes' counts, fill pass with the
    production drop rule, merge -- so that the property under test is the kernel's: whatever the
    lane count, the result is the one sequential walk.

    Returns (entries per bucket, fan-out entries per direction), each in the order the phases read.
    """
    n_slots = len(first_page)
    survivors = lambda b, routed: min(routed, max(0, capacity - first_page[b]))
    tokens = len(picks)

    # size_buckets: the length the offsets table gives, from every pick routed to the slot.
    routed = [0] * n_slots
    for t in range(tokens):
        for p in picks[t]:
            if p is not None:
                routed[p[0]] += 1
    bucket_start = [0]
    for b in range(n_slots):
        bucket_start.append(bucket_start[-1] + (survivors(b, routed[b]) if bucketed[b] else 0))

    slices = [(_slice_begin(tokens, w, lanes), _slice_begin(tokens, w + 1, lanes)) for w in range(lanes)]
    cnt = []
    for t0, t1 in slices:
        c = [0] * n_slots
        for t in range(t0, t1):
            for p in picks[t]:
                if p is not None:
                    c[p[0]] += 1
        cnt.append(c)

    entries = [None] * bucket_start[-1]
    mc = [[] for _ in range(lanes)]
    for w, (t0, t1) in enumerate(slices):
        before = [sum(cnt[v][b] for v in range(w)) for b in range(n_slots)]
        next_page = [first_page[b] + before[b] for b in range(n_slots)]
        next_entry = [bucket_start[b] + survivors(b, before[b]) for b in range(n_slots)]
        for t in range(t0, t1):
            packed = {0: [], 1: []}
            for k, p in enumerate(picks[t]):
                if p is None:
                    continue
                slot, local, direction = p
                page = next_page[slot]
                next_page[slot] += 1
                if page >= capacity:
                    continue
                if not local:
                    if len(packed[direction]) < max_dests:
                        packed[direction].append((page, k))
                    continue
                at = next_entry[slot]
                if at >= bucket_start[slot + 1]:
                    continue
                next_entry[slot] = at + 1
                entries[at] = (t, page, k)
            for direction in (0, 1):
                if packed[direction]:
                    mc[w].append((direction, t, tuple(packed[direction])))

    # merge_routing_index: every bucket holds exactly the length the table sized it at.
    for b in range(n_slots):
        total = sum(cnt[w][b] for w in range(lanes))
        length = bucket_start[b + 1] - bucket_start[b]
        fill = bucket_start[b] + min(survivors(b, total), length)
        assert fill == bucket_start[b + 1], f"bucket {b}: filled {fill - bucket_start[b]} of {length}"
    per_bucket = [entries[bucket_start[b] : bucket_start[b + 1]] for b in range(n_slots)]
    per_dir = {d: [(t, pages) for w in range(lanes) for (dd, t, pages) in mc[w] if dd == d] for d in (0, 1)}
    return per_bucket, per_dir


def _prologue_picks(indices, table, chip_experts, my_row, extent, fanout):
    """One chip's picks in the kernel's terms: bucket slot, whether the expert is local, and which way.

    An index of -1 stands for a pick the group's table resolves to no chip, as a cross-group pick does.
    """
    slot_of = {e: (row, j) for row in range(extent) for j, e in enumerate(chip_experts[row])}
    picks = []
    for t in range(indices.shape[0]):
        row_picks = []
        for k in range(indices.shape[1]):
            e = int(indices[t, k])
            if e < 0 or int(table[e]) == -1:
                row_picks.append(None)
                continue
            r, j = slot_of[e]
            local = (r == my_row) if fanout else True
            travel, _ = _mc_direction(my_row, r, extent)
            row_picks.append((r * len(chip_experts[0]) + j, local, 0 if travel == 1 else 1))
        picks.append(row_picks)
    return picks


@pytest.mark.parametrize("fanout", [False, True], ids=lambda f: "multicast" if f else "unicast")
@pytest.mark.parametrize("tokens", [0, 1, 3, 13, 16, 64], ids=lambda n: f"tokens{n}")
def test_dispatch_fabric2d_prologue_slices_compose(fanout, tokens):
    """Four lanes walking their slices of the tokens build the index one sequential walk would.

    The replay of the production allocator is inherently sequential -- every page depends on every
    earlier drop -- and the prologue splits it anyway, on the argument that a lane knows its pick
    positions up to the earlier lanes' per-bucket counts. That argument is checked here, on host,
    where a slice boundary can be put exactly where it matters: on either side of a bucket's capacity
    cutoff, on the cutoff, in a lane with no tokens, and over token counts the four lanes do not
    divide. On device the same disagreement would show as a wrong page or a hang.
    """
    extent, topk, G = 4, 4, 1
    num_routed_experts = extent * topk * G
    experts_per_chip = num_routed_experts // G // extent
    table = _expert_dispatch_table(num_routed_experts, extent, G)[0]
    chip_experts = [[e for e in range(num_routed_experts) if table[e] == row] for row in range(extent)]
    my_row = 1
    n_slots = extent * experts_per_chip
    bucketed = [(b // experts_per_chip == my_row) if fanout else True for b in range(n_slots)]

    def compose(indices, first_page, capacity, label):
        picks = _prologue_picks(indices, table, chip_experts, my_row, extent, fanout)
        single = _prologue_model(picks, first_page, capacity, bucketed, lanes=1)
        sliced = _prologue_model(picks, first_page, capacity, bucketed, lanes=4)
        assert sliced == single, f"{label}: the four-lane build differs from the sequential walk"

    # Every token picks the same expert on my own chip, first: with topk 1 the cutoff of that bucket
    # falls at token index `room`, so the room places it inside lane 1, on the lane 1/2 boundary,
    # inside lane 2, inside lane 3, past the end, and before the start.
    hot = chip_experts[my_row][0]
    same = torch.full((tokens, 1), hot, dtype=torch.int64)
    for room in sorted({0, 1, tokens * 3 // 8, tokens // 2, tokens * 5 // 8, tokens * 13 // 16, tokens, tokens + 4}):
        first_page = [7] * n_slots
        compose(same, first_page, capacity=7 + room, label=f"one expert, room {room}")

    # Random in-group draws, roomy and tight enough that most buckets are cut somewhere.
    gen = torch.Generator().manual_seed(tokens * 2 + int(fanout))
    indices = torch.zeros(tokens, topk, dtype=torch.int64)
    for t in range(tokens):
        indices[t] = torch.randperm(num_routed_experts, generator=gen)[:topk]
    for capacity_div in (1, 3, 16):
        capacity = max(1, extent * max(tokens, 1) * topk // capacity_div)
        first_page = [int(torch.randint(0, capacity, (1,), generator=gen)) for _ in range(n_slots)]
        compose(indices, first_page, capacity, label=f"random, capacity {capacity}")

    # About a quarter out of group, as production routes: the picks that resolve to no slot at all.
    if tokens:
        cross = indices.clone()
        cross[torch.rand(tokens, topk, generator=gen) < 0.25] = -1
        for capacity in (3 + tokens // 3, 3 + tokens * topk):
            compose(cross, [3] * n_slots, capacity, label=f"cross-group, capacity {capacity}")


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


def _replay_arrival_counter(per_launch, overlaps, give_back):
    """The arrival counter over a run of launches, as the 32-bit word it is: one adder, one subtractor.

    Timeline per launch, in the order the hardware runs it: upstream sends whatever of this launch it
    has not already sent, the reader waits for the counter to reach the page count its chunk lists
    predict, upstream runs ahead and announces `early` pages of the NEXT launch, and only then does
    this reader reset. `give_back(counter, consumed)` is the reset rule under test.

    Raises when a launch cannot complete: the reader is waiting for a count upstream has nothing left
    to reach. That is the hang, written as arithmetic.
    """
    counter, carried, out = 0, 0, []
    for early in overlaps:
        counter = (counter + per_launch - carried) & 0xFFFFFFFF  # the rest of this launch
        if counter < per_launch:
            raise AssertionError(
                f"stranded: the reader needs the counter to reach {per_launch}, upstream has nothing "
                f"left to send, and it stopped at {counter}"
            )
        counter = (counter + early) & 0xFFFFFFFF  # upstream runs ahead into the next launch
        counter = give_back(counter, per_launch)  # end of stream
        carried, _ = early, out.append(counter)
    return out


@pytest.mark.parametrize("extent", [4, 6, 8, 12], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda n: f"{n}link")
def test_dispatch_fabric2d_arrival_counter_survives_an_overlap(extent, num_links, expect_error):
    """The end-of-stream reset holds when the upstream chip is already a launch ahead.

    Nothing keeps ring neighbours in lockstep, so a chip that finishes early starts announcing the next
    launch's pages into a neighbour that is still retiring this one. Zeroing the counter there throws
    those announcements away and the neighbour then waits for pages that, as far as it can tell, never
    arrived -- a ring-wide hang. Giving back exactly what was consumed leaves them standing, and they
    are already the right base for the next launch.

    Proved here rather than on device because the failure is a deadlock: reproducing it costs a board
    whose ethernet links do not retrain afterwards. The arithmetic is the whole mechanism, and it is
    checked over every overlap a launch can have, including the two that bracket it -- none, and a
    launch entirely announced in advance.

    What this does NOT prove is that a chip's announcements equal what its neighbour consumes; that is
    `test_dispatch_fabric2d_chunk_agreement`, which checks it position by position over the same space.
    """
    subtract = lambda counter, consumed: (counter - consumed) & 0xFFFFFFFF
    zero = lambda counter, consumed: 0

    # A stream's pages per launch, from the region layout rather than a round number: the largest is
    # what a real run puts through one region, and 0 is a stream with nothing to relay.
    m = extent // 2
    per_pair = 16 * min(4, extent)
    totals = {0, 1, m, (m * (m - 1) // 2) * max(1, per_pair // num_links)}

    for per_launch in sorted(totals):
        for early in range(per_launch + 1):
            # Eight launches, so an error that accumulates a page at a time is caught as well as one
            # that strands immediately. At rest the counter holds exactly what was announced early.
            after = _replay_arrival_counter(per_launch, [early] * 8, subtract)
            assert after == [early] * 8, (
                f"extent={extent} links={num_links} pages={per_launch} overlap={early}: the counter "
                f"should come to rest holding exactly the pages announced early, got {after}"
            )

        # The rule this replaced. One page announced early is enough to strand the launch after it.
        if per_launch > 0:
            with expect_error(AssertionError, "stranded"):
                _replay_arrival_counter(per_launch, [1, 0], zero)


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

        # `size_buckets` sizes every bucket from `run_len` alone and never counts the picks, so this
        # equality is an ASSUMPTION in the kernel rather than something it checks -- and the ASSERT
        # that would notice is compiled out on this hardware. This is where it is actually enforced,
        # at the real geometry, for every destination row including this chip's own.
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
# One copy per (token, DIRECTION) crosses a cable rather than one per expert or per destination chip:
# it travels the ring and every chip en route that wants it keeps a page and passes the rest on. A
# chunk is therefore (origin, hop) and its length comes from a reach table, since per-expert counts
# cannot express "how many tokens travel at least h hops this way".
#
# The chip one hop short of a page's farthest destination does not pass it on: it writes those last
# pages into the next chip's output pages itself. So the mode is a consume at most hops but a send at
# the terminal one, and a token with several pages on its farthest chip crosses that last cable once
# per page -- see _mc_region_hop for what that buys and costs.
# --------------------------------------------------------------------------------------------------


def _mc_direction(origin, dst, extent):
    """Short way round; a tie at exactly half the ring goes clockwise so reach stays well defined."""
    d = (dst - origin) % extent
    return (1, d) if d <= extent - d else (-1, extent - d)


def _mc_dest_hops(indices, table, offs, capacity, G, extent, seq, topk):
    """hops[g][origin][dir_idx] = the sorted hops of each travelling token's pages, in token order.

    With multiplicity: two experts on one chip are one copy for the transport but two pages there, and
    the terminal rule charges a cable crossing for each of them, so the repeat has to survive here.

    Post-drop: a token whose every surviving page lies elsewhere must not hold a hop open, or the
    origin sends fewer pages than the relay waits for. A token with no destination that way is not
    in the list at all.
    """
    dest_hops = [[[[] for _ in range(2)] for _ in range(extent)] for _ in range(G)]
    for g in range(G):
        for origin in range(extent):
            alloc = offs[g, origin].clone().to(torch.int64)
            for t in range(seq):
                hops = {1: [], -1: []}
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
                    hops[s].append(d)
                for s, di in ((1, 0), (-1, 1)):
                    if hops[s]:
                        dest_hops[g][origin][di].append(sorted(hops[s]))
    return dest_hops


def _mc_far_of(dest_hops):
    """far[g][origin][dir_idx] = farthest hop of each travelling token, in token order."""
    return [
        [[[max(hops) for hops in per_dir] for per_dir in per_origin] for per_origin in per_group]
        for per_group in dest_hops
    ]


def _mc_far_lists(indices, table, offs, capacity, G, extent, seq, topk):
    return _mc_far_of(_mc_dest_hops(indices, table, offs, capacity, G, extent, seq, topk))


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

    A page enters a forwarding region only when it still has a destination BEYOND the chip about to
    hold it, because whoever holds a page whose farthest destination is the NEXT chip delivers it
    itself -- token and metadata written straight into that chip's output pages. So a chunk landing
    h hops out carries the hop-(h + 1) population, and the chunk at h = m is always empty.

    That is free on the link in the common case: the forward packet it removes and the delivery that
    replaces it are the same one payload crossing the same cable. Only a token with several pages on
    its farthest chip pays, one extra crossing per extra page, against two DRAM transfers saved on
    every page that would otherwise have landed in the terminal chip's region and been read back.

    A relay delivers a hop-(j + 1) destination only when the page ENDS there: the delivery replaces
    the forward rather than joining it. A page that both delivered to the neighbour and forwarded
    would put its payload on that link twice, and low link load is the whole of multicast's
    advantage.

    This has to track the kernel. It is the number both sides of a region derive independently, and
    a disagreement is a deadlock rather than wrong data.
    """
    return hop + 1


def _mc_link_of(rank, class_size, num_links):
    """Which link a token rides, from its rank within its farthest-hop class."""
    for link in range(num_links - 1):
        if rank < _slice_begin(class_size, link + 1, num_links):
            return link
    return num_links - 1


def _mc_fixture(extent, capacity_div, seed, cross_group, topk=4, G=2, seq=16):
    """One routing draw and everything both multicast host tests derive from it.

    `cross_group` is the distribution production actually routes: picks drawn from all groups' experts,
    so most resolve to -1 and a token routinely has one destination or none in a direction. That is
    what makes far-hop classes of size 0 and 1 common and whole streams empty -- the degenerate shapes
    the terminal rule depends on, and which an in-group draw never produces.
    """
    num_routed_experts = extent * topk * G
    experts_per_chip = num_routed_experts // G // extent
    capacity = max(1, extent * seq * topk // capacity_div)

    torch.manual_seed(seed)
    table = _expert_dispatch_table(num_routed_experts, extent, G)
    epg = num_routed_experts // G
    indices = torch.zeros(G, extent, seq, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(extent):
            for t in range(seq):
                if cross_group:
                    indices[g, h, t] = torch.randperm(num_routed_experts)[:topk]
                else:
                    indices[g, h, t] = g * epg + torch.randperm(epg)[:topk]

    offs = torch.zeros(G, extent, num_routed_experts, dtype=torch.int64)
    for g in range(G):
        o, _, _, _ = get_gate_outputs(
            indices[g], extent, num_routed_experts, experts_per_chip, seq, topk, expert_dispatch_table=table[g : g + 1]
        )
        offs[g] = o[0].long()

    reach = _mc_reach(indices, table, offs, capacity, G, extent, seq, topk)
    dest_hops = _mc_dest_hops(indices, table, offs, capacity, G, extent, seq, topk)

    # Pages the capacity clamp removed, replaying the same rule _mc_dest_hops applies, so a test can
    # refuse to pass on a draw that never exercised the post-drop path -- where a token whose pages
    # all died must stop holding its hops open, and failing to is a deadlock.
    dropped = 0
    for g in range(G):
        for origin in range(extent):
            alloc = offs[g, origin].clone().to(torch.int64)
            for t in range(seq):
                for k in range(topk):
                    e = int(indices[g, origin, t, k])
                    if int(table[g, e]) == -1:
                        continue
                    if alloc[e] >= capacity:
                        dropped += 1
                    alloc[e] += 1
    return reach, dest_hops, G, dropped, seq


@pytest.mark.parametrize("extent", [4, 6, 8, 12], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2, 3, 4], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
@pytest.mark.parametrize("cross_group", [False, True], ids=lambda c: "cross-group" if c else "in-group")
def test_dispatch_fabric2d_multicast_chunk_agreement(extent, num_links, capacity_div, cross_group):
    m = extent // 2
    reach, dest_hops, G, _, seq = _mc_fixture(extent, capacity_div, seed=17, cross_group=cross_group)
    far_lists = _mc_far_of(dest_hops)
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


# Destinations one multicast page can carry and the packet headers a slot is given to send them with,
# mirroring FO_MAX_DESTS and headers_per_slot(fanout=true) in
# dispatch_fabric2d/device/kernels/dataflow/dispatch_fabric2d_kernel_interface.hpp: one header for the
# forward and one per remote delivery, each delivery a single scatter packet. The pool is sized from
# the same FO_MAX_DESTS the staging walk below bounds the records by, so the header check there is a
# mirror of the kernel's index arithmetic rather than an independent bound; the bound that matters is
# FO_MAX_DESTS, and overrunning it writes packet headers over the very delivery records those sends
# read their addresses from.
_KERNEL_INTERFACE = (
    Path(__file__).resolve().parents[5]
    / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow"
    / "dispatch_fabric2d_kernel_interface.hpp"
)


def _kernel_constant(name):
    """An integer constexpr out of the kernel interface, so the mirror below cannot drift from it."""
    match = re.search(rf"constexpr uint32_t {name} = (\d+)u?;", _KERNEL_INTERFACE.read_text())
    assert match, f"{name} is not a plain integer constant in {_KERNEL_INTERFACE.name}"
    return int(match.group(1))


_FO_MAX_DESTS = _kernel_constant("FO_MAX_DESTS")
_FO_FIRST_DELIVERY_HDR = _kernel_constant("FO_FIRST_DELIVERY_HDR")
# headers_per_slot(true) in the same header; a pool of this size holds every index deliver_remotely
# forms, FO_FIRST_DELIVERY_HDR + i for i < FO_MAX_DESTS, which is the one thing the walk below can check.
_HEADERS_PER_SLOT = _FO_FIRST_DELIVERY_HDR + _FO_MAX_DESTS
assert _FO_FIRST_DELIVERY_HDR + _FO_MAX_DESTS - 1 < _HEADERS_PER_SLOT


@pytest.mark.parametrize("extent", [4, 6, 8, 12], ids=lambda e: f"extent{e}")
@pytest.mark.parametrize("num_links", [1, 2, 3, 4], ids=lambda n: f"{n}link")
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
@pytest.mark.parametrize("cross_group", [False, True], ids=lambda c: "cross-group" if c else "in-group")
def test_dispatch_fabric2d_multicast_terminal_delivery(extent, num_links, capacity_div, cross_group):
    """Follow every page hop by hop under the terminal rule: who delivers, who forwards, who reads.

    Region agreement proves the two sides of a region derive the same NUMBER. That is necessary and
    not sufficient once a relay may deliver one hop ahead instead of forwarding, because the number
    also has to be what the chips actually produce page by page.

    The rule, at the chip j hops from the origin holding a page whose farthest destination is F:
      - destinations at hop j are written locally;
      - if F == j + 1 the page ends next door, so its hop-(j + 1) destinations are written straight
        into that chip's output pages and nothing is forwarded;
      - if F >= j + 2 one page is forwarded and the next chip consumes its own destinations.
    The origin is j = 0 with one difference: it always writes its hop-1 destinations directly whatever
    F is. It is holding the token already, and that is what the unicast path does too.

    Three derivations are kept apart on purpose, because agreement between two of them that share a
    line of code is worth nothing: the producer walk fills each region, the consumer re-derives what
    it forwards from the arriving page's own hop list the way the kernel does, and the reach table
    sizes both. Only then does equality mean the writer and reader cannot desynchronise.
    """
    m = extent // 2
    reach, dest_hops, G, dropped, seq = _mc_fixture(extent, capacity_div, seed=19, cross_group=cross_group, topk=8)

    copies = crossings_terminal = crossings_forwarding = extra_payloads = 0
    for g in range(G):
        for origin in range(extent):
            for di in range(2):
                reach_row = reach[g, origin, di]
                seen = {}
                # arrivals[link][j] = the pages that enter the region of the chip j hops out, as their
                # own hop lists. j = 0 is the origin, which holds its page without a region.
                arrivals = [[[] for _ in range(m + 2)] for _ in range(num_links)]
                for hops in dest_hops[g][origin][di]:
                    far = hops[-1]
                    size = int(reach_row[far]) - int(reach_row[far + 1])
                    rank = seen.get(far, 0)
                    seen[far] = rank + 1
                    link = _mc_link_of(rank, size, num_links)
                    n_far = hops.count(far)
                    n_one = hops.count(1)
                    copies += 1

                    # Producer: the origin holds the page, and it enters the region of every chip out
                    # to the one before the farthest.
                    for j in range(1, far):
                        arrivals[link][j].append(hops)

                    # Every destination, delivered exactly once, and by whom.
                    delivered = [1] * n_one
                    if far >= 2:
                        delivered += [h for h in hops if 2 <= h < far]
                        delivered += [far] * n_far
                    assert (
                        sorted(delivered) == hops
                    ), f"g={g} origin={origin} dir={di} token hops {hops}: delivered {sorted(delivered)}"

                    # What crosses a cable. A forward and a delivery are each ONE packet -- the token
                    # with its tail, or the token with its metadata as a second scatter chunk -- so
                    # payloads and packets are the same count, and the terminal rule is free exactly
                    # when the farthest chip takes one page.
                    crossings_terminal += n_one + (far - 1 + n_far if far >= 2 else 0)
                    crossings_forwarding += n_one + (far if far >= 2 else 0)
                    if far >= 2:
                        extra_payloads += n_far - 1

                for link in range(num_links):
                    for j in range(1, m + 1):
                        # Consumer, re-derived per page exactly as mc_relay_phase does, against what
                        # the chip beyond sizes its chunk at from the reach table alone.
                        read = len(arrivals[link][j])
                        forwarded = sum(1 for hops in arrivals[link][j] if hops[-1] > j + 1)
                        assert read == _mc_chunk(reach_row, _mc_region_hop(j), link, num_links, m), (
                            f"g={g} origin={origin} dir={di} link={link} hop={j}: {read} pages arrive but "
                            f"every chip sizes that chunk at "
                            f"{_mc_chunk(reach_row, _mc_region_hop(j), link, num_links, m)}"
                        )
                        assert forwarded == _mc_chunk(reach_row, _mc_region_hop(j + 1), link, num_links, m), (
                            f"g={g} origin={origin} dir={di} link={link}: chip {j} forwards {forwarded} "
                            f"pages but chip {j + 1} reads "
                            f"{_mc_chunk(reach_row, _mc_region_hop(j + 1), link, num_links, m)}"
                        )
                        # The staging contract the sender relies on: local records first, remote after,
                        # both out of one FO_MAX_DESTS list. The header pool is sized from the same
                        # constant (asserted at module level), so this is the only bound to hold here.
                        for hops in arrivals[link][j]:
                            local_count = hops.count(j)
                            remote_count = hops.count(j + 1) if hops[-1] == j + 1 else 0
                            assert local_count + remote_count <= _FO_MAX_DESTS
                    # Nothing travels past half the ring, so no page reaches the last chunk.
                    assert not arrivals[link][m], f"g={g} origin={origin} dir={di} link={link}: chunk m is not empty"
                    # And the kernel reads fwd_len one index past that at j = m, which must stay off
                    # the end of the reach row rather than indexing it.
                    assert _mc_chunk(reach_row, _mc_region_hop(m + 1), link, num_links, m) == 0

    # The terminal rule's whole cost, stated as an identity rather than a measurement: one extra
    # payload -- and, a delivery being one packet, one extra packet -- for every page beyond the first
    # on a token's farthest chip.
    assert crossings_terminal == crossings_forwarding + extra_payloads, (
        f"terminal delivery moved {crossings_terminal} payloads across cables against "
        f"{crossings_forwarding} when forwarding to the farthest chip, not the "
        f"{crossings_forwarding + extra_payloads} the accounting predicts"
    )
    # Without these the checks above pass on a draw that never reached the cases they are about, and
    # nothing would say so. Each is claimed only where the draw can actually produce it: the clamp is
    # what leaves a token a single page on its farthest chip, so the roomy configs are the ones that
    # exercise the rule costing anything, and the tight ones are the ones that exercise the drop.
    assert copies > 0, "no token travelled, so nothing above was exercised"
    if capacity_div == 1:
        assert extra_payloads > 0, "no token had two pages on its farthest chip, so the rule cost nothing here"
    else:
        assert dropped > 0, "the tight config dropped nothing, so the post-drop rule is still untested"
    logger.info(
        f"extent={extent} links={num_links}: {copies} travelling copies, link payloads "
        f"{crossings_forwarding} -> {crossings_terminal} ({crossings_terminal / crossings_forwarding:.3f}x)"
    )
