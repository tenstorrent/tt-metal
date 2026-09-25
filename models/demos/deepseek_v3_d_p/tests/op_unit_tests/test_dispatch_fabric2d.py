# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bring-up and correctness for dispatch_fabric2d.

The op moves each token to the chips hosting the experts it was routed to, one fabric hop at a time.
Each chip on the way stores the token in a DRAM forwarding buffer and forwards it. Where each token
lands is fully determined by the routing, so the gate is byte-exact equality against a torch
reference.

The routing metadata is derived by `get_gate_outputs` from the same indices the op is given, so the
control tensors and the routing agree by construction. The bucket-fill ASSERT in merge_routing_index
(watcher builds only) relies on that.

Cases default to the production chunk: 5120 tokens over an 8-chip dispatch group, so
seq_len_per_chip is 640. Two tests vary it on purpose:
`test_dispatch_fabric2d_partial_last_tile` and `test_dispatch_fabric2d_padding_config`.

The mesh axis comes from the shared table in `tests/pcc/mesh_configs.py`, so CI's hardware-class
selection through `requires_mesh_topology` matches the model's other prefill tests.
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

# The production Galaxy row and the 8x1 TorusY LoudBox proxy. This op forwards single hops around a
# ring, so it needs an axis whose closing link is cabled and an extent of at least 4, which rules
# out the 2x1 and mesh rows. The proxy is one dispatch group on a single-axis ring, so it covers
# neither cross-group routing nor the 2D torus.
_MESH_IDS = (
    "fabric2d-torus-xy-8x4-2link",
    "fabric2d-torus-y-8x1-2link",
)
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id in _MESH_IDS]
assert len(_MESH_CONFIGS) == len(_MESH_IDS), "dispatch_fabric2d mesh configs missing from ALL_MESH_CONFIGS"

# The production 8x4 row on its own, for the tests whose subject is how the op is CALLED rather than
# the ring geometry.
_PRODUCTION_MESH = [param for param in _MESH_CONFIGS if param.id == "fabric2d-torus-xy-8x4-2link"]


def _reference_dispatch(indices, table, offs, x, capacity, G, H, seq, topk, emb):
    """The pages the op must place, per destination chip, from every source chip.

    Replays the per-expert page allocator, including the rule that a token past capacity is dropped
    while its counter still advances: page numbers must match the offsets table, which counts every
    routed token.

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
# chips). Calibrated against captured MoE layers: the ones the perf harness ranks first sit at a 37%
# in-group share (see the capture table in perf/test_dispatch_combine_perf.py), against 25% for an
# evenly spread layer. The hot-half weight is what concentrates the kept tokens onto a few destination
# chips, which is where the link load the transport is measured on comes from.
#
# The two knobs move together and must stay a calibrated pair: an in-group share from one layer with
# the skew from another describes no real routing, and the device time it produces matches nothing.
PRODUCTION_ROUTING = (0.372, 2.0)


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
# production routes over all experts, so most picks resolve to -1 and the kept tokens concentrate on a
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
    if routing == "production" and G == 1:
        pytest.skip("production routing sends most picks to other dispatch groups; this mesh has only one")
    seq_len_per_chip = SEQ_LEN_PER_CHIP
    num_routed_experts, num_experts_per_tok, emb_dim = 256, 8, 256
    experts_per_chip = num_routed_experts // G // H
    # Roomy capacity holds every source chip's tokens for one expert; the divisor shrinks it.
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
    # expert_offsets is the ALL-ROWS table: replicated along the dispatch axis, since a forwarding chip
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

    # Every page any chip sourced should now be in place: the downstream chip's by a single hop, farther
    # ones forwarded through the fwd_sections, and this chip's own by the local phase. `src_of`
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

    Shared by the tests below that care about how the op is CALLED rather than about the routing.
    Routing coverage (in-group vs production draws, roomy vs tight capacity) lives in
    test_dispatch_fabric2d's parametrization.
    """

    # emb_dim 512 is 16 tiles wide, so the untilizer packs two column blocks per tile row. At 256 it is
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
        """Bring every routing input back in line with `indices`. Call it after editing `indices`.

        Recomputes the expert offsets, counts and region tables from `indices`, uploads them together
        with `indices` itself, and clears the cached `reference()`.

        The op trusts the offsets table to match the indices: it is how every chip on the axis, forwarding
        chips included, sizes the chunks it waits for and forwards. A table from a stale draw has the right
        shape, so nothing rejects it, and the op then places the wrong pages.
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
        """The [real_token_count, pad_side] padding_config tensor, replicated to every device."""
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
        """The pages the op must place for this draw: (payload, metadata, and each page's
        source row, -1 for a page no token lands on).

        Computed once and reused. The replay walks every (token, pick) pair in Python, and the
        multi-launch tests compare several launches against the same draw.

        The cache is cleared only by `rebuild()`, so after editing `indices` a test must call it or it
        is checked against the old draw. Nothing else needs clearing: `x`, `capacity` and `emb_dim`
        never change after `__init__`.
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
        """Every page any chip sourced, byte-exact against `reference()`."""
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


def _moe_grid_split(mesh_device, dispatch_rows=1):
    """The model's split: dispatch gets the first `dispatch_rows` rows of the grid, the shared expert
    the rest.

    `tt_moe.py` splits rows [0, dispatch_sd_rows) against the remainder, with `dispatch_sd_rows`
    currently 1, so the two ops run on disjoint cores and overlap on chip.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    last = dispatch_rows - 1
    dispatch = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last))})
    shared = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
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
    """The op only uses cores of the sub-device it is given.

    The grid is split as the model splits it: dispatch gets the top rows, the shared expert gets the
    rest. The op is run four times, each time given a different sub-device:

    1. Rows 0 and 1 with a TILE input, so the untilizers need row 1 too: must succeed, byte-exact,
       with every untilizer in row 1 -- no warning that some spilled elsewhere.
    2. Row 0 only, the model's split today, with a ROW_MAJOR input: must succeed, byte-exact.
    3. Everything but row 0: must refuse, and the error must name a row-0 core. That shows the op
       wants row 0 for its streams, so runs 1 and 2 were not a fluke.
    4. Part of row 0: must refuse. An op that ignored its sub-device would take the whole grid and
       succeed, so the refusal shows the argument is actually read.
    """
    cfg = extract_mesh_config(mesh_device)
    # Tight capacity: this test's subject is core confinement, not the allocator, and a roomy buffer at
    # this token width costs a gigabyte of host reference for nothing.
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, capacity_div=8)
    streams = 2 * num_links
    untilizers = 5 * num_links
    grid_x = mesh_device.compute_with_storage_grid_size().x
    assert grid_x >= untilizers, f"row 1 has {grid_x} cores, fewer than the {untilizers} untilizers want"

    with _sub_device_manager(mesh_device, list(_moe_grid_split(mesh_device, dispatch_rows=2))) as (dispatch_sd, _):
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT, subdevice_id=dispatch_sd)
        fx.check(payload, metadata, f"tile on rows 0-1, {untilizers} untilizers")
        out = capfd.readouterr().out
        assert "too few spare cores for the untilizer pool" not in out, "row 1 has room; the pool spilled anyway"

    with _sub_device_manager(mesh_device, list(_moe_grid_split(mesh_device))) as (dispatch_sd, shared_sd):
        payload, metadata = fx.run(cfg.sp_axis, num_links, subdevice_id=dispatch_sd)
        fx.check(payload, metadata, "row-major on row 0")

        with expect_error(RuntimeError, "is outside the") as refusal:
            fx.run(cfg.sp_axis, num_links, subdevice_id=shared_sd)
        message = str(refusal.value)
        # The core it names is the one the eth-nearest placement wanted. CoreCoord formats as x-y, so
        # a trailing -0 is row 0, which is what this test checks.
        assert re.search(r"eth core is \d+-0,", message), message

    # The eth-nearest workers are spread along row 0, so dispatch needs all of it; the error naming
    # the row-0 core it wanted is only possible if the op read this sub-device.
    with _sub_device_manager(mesh_device, [_leading_row_cores(streams + 2)]) as (narrow_sd,):
        with expect_error(RuntimeError, f"outside the {streams + 2} cores") as refusal:
            fx.run(cfg.sp_axis, num_links, subdevice_id=narrow_sd)
        message = str(refusal.value)
        assert re.search(r"eth core is \d+-0,", message), message


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
# 90 and 224 tiles wide. 90 is not a multiple of 8, so it is the width that exercises
# `untilize_block_ct_dim`'s divisor search (block 6, fifteen blocks per tile row) rather than taking the
# 8 every power-of-two width takes; gpt_oss_120b's 2880 is a deployed emb_dim of exactly this shape.
# 7168 is the production token, 14336 B: the size every packet-size bound in the sender is measured at.
@pytest.mark.parametrize("emb_dim", [2880, 7168], ids=lambda e: f"emb{e}")
@pytest.mark.timeout(1800)
def test_dispatch_fabric2d_relaunch(mesh_device, device_params, num_links, emb_dim):
    """Four launches on one device, alternating layouts, to catch state leaking from one launch into
    the next.

    Each case of test_dispatch_fabric2d gets a fresh device, so it cannot catch either of these:

    - Program cache. TILE and ROW_MAJOR inputs build different programs: only TILE has untilizer
      cores and reads tokens from a staging buffer. If the layout were missing from the cache key,
      the second layout would reuse the first one's program and read tokens from the wrong place.
      Running row-major, tile, tile, row-major catches that in both directions.
    - Untilize counter. Nothing resets it between launches except the stream readers zeroing it at
      the end of each launch. If they fail to, the next TILE launch does not hang: its wait passes
      at once and it reads the previous launch's staging buffer. Each launch therefore uses a new
      random draw, so stale data shows up as wrong pages.
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
        # starts sending into a downstream chip that is still retiring the previous one.
        fx.check(payload, metadata, f"{label}, emb {emb_dim}")
        if entries_after_first is None:
            entries_after_first = mesh_device.num_program_cache_entries()
    # Only the first TILE launch may add a program; the repeats must hit the cache. A key that is too
    # specific rebuilds on every launch and still passes the byte-exact checks, but costs prefill perf.
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
def test_dispatch_fabric2d_partial_last_tile(mesh_device, device_params, num_links):
    """A seq_len_per_chip that is not a multiple of 32 dispatches byte-exact in both layouts.

    660 tokens is twenty full 32-row tiles plus a last tile holding only 20 real rows. The TILE path
    untilizes that last tile whole, padding rows included; staging has room for them and the routing
    pass stops at seq_len_per_chip, so they must never become pages. ROW_MAJOR has no padding rows,
    which makes this the one case where the two layouts could disagree on how many tokens exist.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=660, capacity_div=8)
    for layout, label in ((ttnn.ROW_MAJOR_LAYOUT, "row-major"), (ttnn.TILE_LAYOUT, "tile")):
        payload, metadata = fx.run(cfg.sp_axis, num_links, layout=layout)
        fx.check(payload, metadata, f"{label}, 660 tokens per chip")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_unaligned_emb_dim(mesh_device, device_params, num_links):
    """An emb_dim that is not a multiple of 32 dispatches byte-exact from a ROW_MAJOR input.

    Every other case uses a tile-aligned emb_dim. TILE inputs cannot have this shape (the untilizer
    reads whole tile columns), but ROW_MAJOR tokens are plain pages and should not care.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=500, capacity_div=8)
    payload, metadata = fx.run(cfg.sp_axis, num_links)
    fx.check(payload, metadata, "row-major, emb_dim 500")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_back_to_back(mesh_device, device_params, num_links):
    """Four launches queued with no host sync between them, as in a traced replay.

    Every other test reads outputs between launches, so launches never overlap. Here a chip that
    finishes early can start sending into a downstream chip still finishing the previous launch, racing the
    reset of the arrival counter:

    - An increment lost to the reset hangs the forward waiting for it.
    - A counter left too high lets the forward read pages before they arrive. Alternating two draws
      makes those stale pages differ from the expected ones.
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
# Half the tokens is the shape production pads to. 3 leaves one of the four routing index RISCs with no
# tokens at all, which is the degenerate shape the slice arithmetic has to survive.
@pytest.mark.parametrize("real", [320, 3], ids=lambda r: f"real{r}")
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_padding_config(mesh_device, device_params, num_links, real):
    """A padding_config makes the routing pass stop early without changing any output page.

    Tokens past `real` are routed to another dispatch group, so this group sees them as padding: they
    produce no pages. The op is run three times and each must match the reference, which walks every
    token:

    - No padding_config.
    - Right padding (pad_side 0): the pass stops at `real`.
    - Left padding (pad_side 1): ignored, so the whole sequence is walked.
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
