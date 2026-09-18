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

Cases default to the production chunk: 5120 tokens over an 8-chip dispatch group, so
seq_len_per_chip is 640. Two tests vary it on purpose -- the ragged-sequence case in
`test_dispatch_fabric2d_refusals`, and `test_dispatch_fabric2d_padding_config`.

The mesh axis comes from the shared table in `tests/pcc/mesh_configs.py`, so CI's hardware-class
selection through `requires_mesh_topology` is consistent with `test_prefill_dispatch.py`.
"""

import re
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config, get_gate_outputs

# One 5120-token chunk spread over the 8-chip dispatch group.
CHUNK = 5 * 1024
DISPATCH_GROUP_SIZE = 8
SEQ_LEN_PER_CHIP = CHUNK // DISPATCH_GROUP_SIZE

# The production Galaxy rows, both link counts. This op relays single hops around a ring, so it needs
# an axis whose closing link is cabled and an extent of at least 4 -- which rules out the 2x1 and mesh
# rows but not the TorusY Nx1 proxies `test_prefill_dispatch.py` also selects; those are left out only
# because nothing has run this op on them yet.
# One link halves stream_count, so the opposite chip's chunk is split two ways instead of four: the
# split arithmetic and the region layout change shape, not just size.
_MESH_IDS = ("fabric2d-torus-xy-8x4-1link", "fabric2d-torus-xy-8x4-2link")
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id in _MESH_IDS]
assert len(_MESH_CONFIGS) == len(_MESH_IDS), "dispatch_fabric2d mesh configs missing from ALL_MESH_CONFIGS"

# The production 8x4 row on its own, for the tests whose subject is how the op is CALLED rather than
# the ring geometry.
_PRODUCTION_MESH = [param for param in _MESH_CONFIGS if param.id == "fabric2d-torus-xy-8x4-2link"]


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


# (share of a token's picks landing in its own dispatch group, weight on the hot half of that group's
# chips). 0.25 is the uniform share: production spreads picks over every group's experts, so about
# 1/num_dispatch_groups of them land in group and the rest resolve to -1. The hot-half weight is the
# only knob concentrating the survivors onto a few destination chips; it is not calibrated against a
# captured layer, so treat it as a skew that exercises the path rather than a production figure.
PRODUCTION_ROUTING = (0.25, 3.0)


def _draw_indices(G, H, seq, topk, num_routed_experts, in_group_share, hot_weight):
    """topk distinct experts per token, drawn from ALL experts with a controllable in-group skew.

    How many of a token's picks land in its own dispatch group is drawn FIRST, then that many distinct
    in-group experts and the rest from the other groups. Weighting the whole expert list and drawing
    topk distinct picks from it instead pulls the realized share well above the target, because
    sampling without replacement favours the weighted entries -- and a share that drifts changes the
    routing the test believes it is exercising.
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
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# A divisor of 1 is the roomy case no token can overflow; the large one puts capacity below the
# per-expert load so the allocator starts dropping, which is the case where the offsets table counts
# more tokens than any origin actually sends.
@pytest.mark.parametrize("capacity_div", [1, 64], ids=lambda d: "roomy" if d == 1 else "tight")
# In-group routing gives every token somewhere to go and is what the byte-exactness gate was built on;
# production routes over all experts, so most picks resolve to -1 and the survivors concentrate on a
# few chips.
@pytest.mark.parametrize("routing", [None, "production"], ids=lambda r: r or "in-group")
# The model hands dispatch TILED activations. A TILE input is untilized on device into a staging
# buffer by a pool of cores beside the stream cores, so the transport sees the identical bytes either
# way and both layouts share this gate.
@pytest.mark.parametrize(
    "input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=lambda ly: "tile" if ly == ttnn.TILE_LAYOUT else "rm"
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d(mesh_device, device_params, num_links, capacity_div, routing, input_layout):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"
    seq_len_per_chip = SEQ_LEN_PER_CHIP
    num_routed_experts, num_experts_per_tok, emb_dim = 256, 8, 256
    experts_per_chip = num_routed_experts // G // H
    # Capacity the production op would use: every source chip's tokens for one expert, tile-aligned.
    max_dispatch_buffer_token_size = max(1, H * seq_len_per_chip * num_experts_per_tok // capacity_div)

    logger.info(
        f"dispatch_fabric2d: mesh={tuple(mesh_device.shape)} H={H} G={G} experts_per_chip={experts_per_chip} "
        f"seq={seq_len_per_chip} topk={num_experts_per_tok} capacity={max_dispatch_buffer_token_size}"
    )

    torch.manual_seed(7)
    table = _expert_dispatch_table(num_routed_experts, H, G)

    experts_per_group = num_routed_experts // G
    if routing is None:
        # Per group, route only into that group's own experts so every token has somewhere to go.
        indices = torch.zeros(G, H, seq_len_per_chip, num_experts_per_tok, dtype=torch.int64)
        for g in range(G):
            base = g * experts_per_group
            for h in range(H):
                for t in range(seq_len_per_chip):
                    indices[g, h, t] = base + torch.randperm(experts_per_group)[:num_experts_per_tok]
    else:
        indices = _draw_indices(G, H, seq_len_per_chip, num_experts_per_tok, num_routed_experts, *PRODUCTION_ROUTING)

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

    logger.info(f"pages compared byte-exact: {checked} across {H * G} devices ({checked_local} of them same-chip)")
    assert checked > 0, "no dispatched pages found; the reference or the routing is wrong"

    # A page is only placed for a token the allocator did not drop, so this is how many it dropped.
    routed = int(sum((table[g, indices[g]] != -1).sum() for g in range(G)))
    dropped = routed - int((src_of >= 0).sum())
    logger.info(f"tokens routed: {routed}, dropped for capacity: {dropped}")
    if capacity_div > 1:
        assert dropped > 0, "the tight config dropped nothing, so the clamp is still untested"
    else:
        assert dropped == 0, "the roomy config dropped a token, so it is not the no-overflow case"
    # Without this the local phase could regress to writing nothing and the comparison would still pass.
    assert checked_local > 0, "no same-chip pages found; the local phase would go untested"
    assert bad == 0, f"{bad} device/tensor comparisons differ from the dispatch reference"


class _Fixture:
    """One in-group routing draw at the production chunk, on device in both input layouts, plus its
    torch reference.

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
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        emb_dim=512,
        num_routed_experts=256,
        topk=8,
        seed=11,
        capacity_div=1,
    ):
        self.mesh_device = mesh_device
        self.seq_len_per_chip, self.emb_dim, self.H, self.G = seq_len_per_chip, emb_dim, H, G
        self.num_routed_experts, self.topk = num_routed_experts, topk
        self.experts_per_chip = num_routed_experts // G // H
        self.capacity = max(1, H * seq_len_per_chip * topk // capacity_div)
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

        The offsets table comes from the same draw the op is handed, which is the whole reason the op
        can size a chunk it neither wrote nor receives; building one from a stale draw would not fail a
        shape check, it would deadlock an axis.
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
        self._reference = None

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

    def run(self, cluster_axis, num_links, layout=ttnn.ROW_MAJOR_LAYOUT, subdevice_id=None, padding_config=None):
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            self.tt_x[layout],
            self.tt_idx,
            self.tt_offs,
            self.tt_table,
            self.tt_counts,
            self.tt_region,
            padding_config=padding_config,
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

    def reference(self):
        """Cached, because the multi-launch tests check several launches of one draw and the replay is
        a sequential walk over every (token, pick) pair.

        `rebuild()` is the ONLY invalidation point. It covers everything a test may edit -- `indices`,
        and the offsets it re-derives from them. `x`, `capacity` and `emb_dim` are fixed for a
        fixture's lifetime, which is what `tt_x` living in `__init__` already assumes.
        """
        if self._reference is None:
            self._reference = _reference_dispatch(
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
        return self._reference

    def check(self, payload, metadata, label):
        """Every page any chip sourced, byte-exact against what `dispatch` would have placed."""
        ref_payload, ref_meta, src_of = self.reference()
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
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_subdevice(mesh_device, device_params, num_links, capfd, expect_error):
    """A TILE input on the model's two-sub-device split: every core the op takes must be in row 0.

    Byte-exactness is the same gate the matrix applies. What is new is confinement -- the stream cores
    AND the untilizer pool a TILE input needs all have to come out of the dispatch row, because the
    shared expert holds the rest of the grid at the same time.

    The negative case is what actually proves it. Handed the SHARED sub-device instead, the op must
    refuse and name a core in row 0: that message is the placement telling us where it wanted to put a
    stream, and it can only say y=0 if row 0 is where the streams go. Confinement of the pool follows,
    since it is drawn from the spare cores of the same carve -- and on a one-row carve that is the
    documented fallback, so this test is also the coverage for it: the op has to report the fallback
    once, and the report is expected here rather than a failure.
    """
    cfg = extract_mesh_config(mesh_device)
    # Tight capacity: this test's subject is core confinement, not the allocator, and a roomy buffer at
    # this token width costs a gigabyte of host reference for nothing.
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, capacity_div=8)
    streams = 2 * num_links
    dispatch_cores, shared_cores = _moe_grid_split(mesh_device)
    n_cores = sum((r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1) for r in dispatch_cores.ranges())
    pool = n_cores - streams
    assert pool > 0, "no core left for an untilizer; the TILE path cannot run"
    # More stripes than there are cores to take them, so a core takes several and the round-robin
    # stride stops being indistinguishable from one.
    stripes = fx.seq_len_per_chip // 32
    assert stripes > pool, f"{stripes} stripes over {pool} cores does not exercise the round robin"
    logger.info(f"dispatch sub-device: {n_cores} cores in row 0, {streams} streams, {pool} untilizers")

    with _sub_device_manager(mesh_device, [dispatch_cores, shared_cores]) as (dispatch_sd, shared_sd):
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=dispatch_sd)
        fx.check(payload, metadata, f"tile on the dispatch sub-device, {stripes} stripes over {pool} untilizers")
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
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
# 90 and 224 tiles wide. 90 is not a multiple of 8, so it is the width that exercises
# `untilize_block_ct_dim`'s divisor search (block 6, fifteen blocks per stripe) rather than taking the
# 8 every power-of-two width takes; gpt_oss_120b's 2880 is a deployed emb_dim of exactly this shape.
# 7168 is the production token, 14336 B: the size every packet-size bound in the sender is measured at.
@pytest.mark.parametrize("emb_dim", [2880, 7168], ids=lambda e: f"emb{e}")
@pytest.mark.timeout(1800)
def test_dispatch_fabric2d_relaunch(mesh_device, device_params, num_links, emb_dim):
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
    entries_after_first = None
    for seed, (label, layout) in enumerate(plan):
        # Tight capacity: at the production token width a roomy buffer is a gigabyte of pages per chip
        # for a test whose subject is state carried between launches, not the allocator.
        fx = _Fixture(
            mesh_device,
            cfg.dispatch_group_size,
            cfg.num_dispatch_groups,
            emb_dim=emb_dim,
            seed=20 + seed,
            capacity_div=8,
        )
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=layout)
        # Reading the outputs back is what synchronises the launches: this op deadlocks if a chip
        # starts sending into a neighbour that is still retiring the previous one.
        fx.check(payload, metadata, f"{label}, emb {emb_dim}")
        if entries_after_first is None:
            entries_after_first = mesh_device.num_program_cache_entries()
    # Two programs for four launches, as a DELTA over whatever the cache already held: the row-major
    # launch built the first, the TILE one adds exactly one more, and the two repeats add none. Wrong
    # bytes would catch a layout missing from the cache key; only the counter catches the opposite -- a
    # key so specific that every launch rebuilds, which a correctness gate cannot see and which is what
    # kills prefill perf.
    assert mesh_device.num_program_cache_entries() == entries_after_first + 1, (
        f"four launches over two layouts should add one program to the {entries_after_first} the first "
        f"built, got {mesh_device.num_program_cache_entries()}"
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_refusals(mesh_device, device_params, num_links, expect_error):
    """The op's edge conditions are refused, rather than silently doing something else.

    Each refusal stands in for a write that would land somewhere it should not, or a production
    `dispatch` feature this transport cannot carry: an emb_dim that is not a multiple of 32 leaves the
    untilizer reading tile columns that are not there; a sub-device with no core to spare has nowhere
    to put a pool; a token wider than the fabric payload admits would run past its channel slot; and
    an fp8 input or a longer metadata tail would need scales this op has no room for on the wire.

    One edge condition is CARRIED rather than refused -- a ragged sequence -- and it is checked here
    beside the refusals because that is the boundary it sits on.
    """
    cfg = extract_mesh_config(mesh_device)
    streams = 2 * num_links

    # 500 columns is fifteen whole tile columns and a ragged sixteenth, which the untilizer cannot
    # read: it takes whole tile columns and packs rows at a stride the writer does not use. Accepted
    # as ROW_MAJOR, which is what makes the refusal a property of the untilizer rather than the shape.
    ragged_emb = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=500, capacity_div=8)
    with expect_error(RuntimeError, "multiple of 32"):
        ragged_emb.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT)
    payload, metadata = ragged_emb.run(cfg.sp_axis, num_links)
    ragged_emb.check(payload, metadata, "row-major with a ragged emb")

    # A ragged SEQUENCE is carried, not refused: 660 tokens is twenty whole stripes and a twenty-first
    # the packer fills with the tile's padding rows, which staging has room for and the routing pass
    # never reaches. This is the one place the two input layouts could disagree about how many tokens
    # exist.
    ragged_seq = _Fixture(
        mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=660, capacity_div=8
    )
    for layout, label in ((ttnn.ROW_MAJOR_LAYOUT, "row-major"), (ttnn.TILE_LAYOUT, "tile")):
        payload, metadata = ragged_seq.run(cfg.sp_axis, num_links, layout=layout)
        ragged_seq.check(payload, metadata, f"{label} with a ragged sequence")

    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, capacity_div=8)
    with _sub_device_manager(mesh_device, [_leading_row_cores(streams)]) as (exact_sd,):
        # Exactly enough cores for the streams, so the pool would have to come from somewhere else.
        # Refused in validation, before the placement gets a chance to object to the carve itself.
        with expect_error(RuntimeError, "plus at least one untilizer"):
            fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=exact_sd)

    # One token wider than the fabric payload admits with its 64-byte tail: 7168 columns is exactly the
    # cap the tests configure, so 7200 is the first refusal. Without it the forward's payload would run
    # past its channel slot onto the next packet, which no kernel check can see.
    too_wide = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=7200, capacity_div=64)
    with expect_error(RuntimeError, "exceeds the fabric max payload"):
        too_wide.run(cfg.sp_axis, num_links)

    # The two production `dispatch` paths this transport does not carry, called directly because the
    # fixture only ever builds the supported shape. Both are documented limitations of the op; these
    # are the tests that will fail the day someone implements either.
    def raw(input_tensor, metadata_len):
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            input_tensor,
            fx.tt_idx,
            fx.tt_offs,
            fx.tt_table,
            fx.tt_counts,
            fx.tt_region,
            experts_per_chip=fx.experts_per_chip,
            num_routed_experts=fx.num_routed_experts,
            num_experts_per_tok=fx.topk,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=fx.capacity,
            seq_len_per_chip=fx.seq_len_per_chip,
            cluster_axis=cfg.sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # A non-BFLOAT16 input is the fp8-scaled path's shape, whose per-block scales would have to ride
    # the 64-byte routing tail on every hop.
    with expect_error(RuntimeError, "must be BFLOAT16"):
        raw(fx._shard(fx.x.to(torch.float32), (0, 1), ttnn.float32), 3)
    # A longer tail is that same fp8-scaled layout, by the other name.
    with expect_error(RuntimeError, "metadata_len must be 3"):
        raw(fx.tt_x[ttnn.ROW_MAJOR_LAYOUT], 4)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_untilizers_under_the_streams(mesh_device, device_params, num_links, capfd):
    """The whole grid, with more stripes than the pool: the pool sits in the row under the streams.

    The placement a TILE input is designed for is only reachable with a second row in the carve, and
    the sub-device test runs on the model's one-row carve. 640 tokens is 20 stripes over 5 * num_links
    untilizers, and the op must NOT report a fallback -- that report is the one-row carve's, and it
    must stay off here or the sub-device test's assertion on it means nothing.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=37, capacity_div=8)
    payload, metadata = fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT)
    fx.check(payload, metadata, f"{fx.seq_len_per_chip // 32} stripes over {5 * num_links} untilizers")
    out = capfd.readouterr().out
    assert (
        "untilizer pool is not in the row under the streams" not in out
    ), "the whole grid has a row under the streams; the pool fell back anyway"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_back_to_back(mesh_device, device_params, num_links):
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
    a = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=31, capacity_div=8)
    b = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=32, capacity_div=8)
    order = [a, b, a, b]

    # Queued together, read afterwards. Nothing here waits on the device.
    results = [fx.run(cfg.sp_axis, num_links) for fx in order]
    for i, (fx, (payload, metadata)) in enumerate(zip(order, results)):
        fx.check(payload, metadata, f"launch {i} of four with no host sync between them")
    logger.info("back-to-back: 4 unsynchronised launches byte-exact")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
# Half the tokens is the shape production pads to. 3 leaves one of the four prologue lanes with no
# tokens at all, which is the degenerate shape the slice arithmetic has to survive.
@pytest.mark.parametrize("real", [320, 3], ids=lambda r: f"real{r}")
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_padding_config(mesh_device, device_params, num_links, real):
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
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, capacity_div=8)
    assert real < fx.seq_len_per_chip

    # Tokens past `real` are given experts of another dispatch group, the -1 column this group sees.
    experts_per_group = fx.num_routed_experts // fx.G
    for g in range(fx.G):
        other = ((g + 1) % fx.G) * experts_per_group
        fx.indices[g, :, real:, :] = torch.arange(other, other + fx.topk)
    fx.rebuild()

    baseline_payload, baseline_meta = fx.run(cfg.sp_axis, num_links)
    fx.check(baseline_payload, baseline_meta, "no padding_config")

    for pad_side, label in ((0, "right padding, the pass is bounded"), (1, "left padding, the config is ignored")):
        payload, metadata = fx.run(cfg.sp_axis, num_links, padding_config=fx.padding_config(real, pad_side))
        fx.check(payload, metadata, label)
    logger.info(f"padding_config: {real} real of {fx.seq_len_per_chip} tokens, byte-exact either side")
