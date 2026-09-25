# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bring-up and correctness for combine_fabric2d.

The op sends each expert-processed token back to the chip it came from, into the top-k slot it was
dispatched from, one fabric hop at a time. Each chip on the way stores the token in a DRAM
forwarding buffer and forwards it. Where each token lands is fully determined by the routing, so the
gate is byte-exact equality against a torch reference.

The input is laid out as dispatch_fabric2d leaves it: per local expert, tokens grouped by the chip
they came from, starting at the page `expert_offsets` gives. Every page holds its own random token,
so a token delivered to the wrong slot cannot match by accident.

Combine reads each origin chip's run length from `expert_offsets` and does not check capacity, so a
buffer dispatch dropped tokens from is not valid input. Every case sizes the buffer to fit every
routed token.

Cases default to the production chunk: 5120 tokens over an 8-chip dispatch group, so
seq_len_per_chip is 640. `test_combine_fabric2d_partial_last_tile` varies it on purpose.

H is the dispatch group size (mesh rows), G the number of groups (mesh columns). Routing tensors are
indexed (G, H, ...), device tensors (H, G, ...) to match the mesh.
"""

import functools
from typing import NamedTuple

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

# The production Galaxy row and its 8x1 TorusY LoudBox proxy. The proxy is one dispatch group on a
# single-axis ring, so it covers neither cross-group routing nor the 2D torus.
_MESH_IDS = (
    "fabric2d-torus-xy-8x4-2link",
    "fabric2d-torus-y-8x1-2link",
)
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id in _MESH_IDS]
assert len(_MESH_CONFIGS) == len(_MESH_IDS), "combine_fabric2d mesh configs missing from ALL_MESH_CONFIGS"

# The production 8x4 mesh on its own, for the tests about how the op is called, not the ring geometry.
_PRODUCTION_MESH = [param for param in _MESH_CONFIGS if param.id == "fabric2d-torus-xy-8x4-2link"]


def _expert_dispatch_table(num_routed_experts: int, dispatch_group_size: int, num_dispatch_groups: int):
    """expert -> chip within its own dispatch group, -1 for experts of other groups.

    The trailing sentinel column is always -1, as in the table dispatch is given.
    """
    experts_per_group = num_routed_experts // num_dispatch_groups
    experts_per_chip = experts_per_group // dispatch_group_size
    table = torch.full((num_dispatch_groups, num_routed_experts + 1), -1, dtype=torch.int32)
    for g in range(num_dispatch_groups):
        for e in range(experts_per_group):
            table[g, g * experts_per_group + e] = e // experts_per_chip
    return table


# (share of a token's picks landing in its own dispatch group, weight on the hot half of that group's
# chips), calibrated on the captured layers perf/test_dispatch_combine_perf.py ranks first. Keep the
# two as a pair: one layer's share with another's skew describes no real routing. Same values as
# dispatch_fabric2d's tests.
PRODUCTION_ROUTING = (0.372, 2.0)


def _draw_indices(G, H, seq, topk, num_routed_experts, in_group_share, hot_weight):
    """topk distinct experts per token, drawn from all experts with a controllable in-group skew.

    The number of picks in the token's own dispatch group is drawn first, then that many in-group experts
    and the rest from other groups. Weighting the whole expert list instead overshoots the target share,
    because sampling without replacement favours the weighted entries.
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


def _in_group_indices(G, H, seq, topk, num_routed_experts):
    """Picks only among the token's own group's experts, so every pick comes back."""
    experts_per_group = num_routed_experts // G
    indices = torch.zeros(G, H, seq, topk, dtype=torch.int64)
    for g in range(G):
        for h in range(H):
            for t in range(seq):
                indices[g, h, t] = g * experts_per_group + torch.randperm(experts_per_group)[:topk]
    return indices


def _dispatch_layout(indices, table, offs):
    """(holder_pos, page), each (G, H, seq, topk): the position of the chip holding each pick and its
    page there, both -1 for a pick outside the group.

    An origin's picks for one expert fill consecutive pages from offs[g, s, e] in (token, top-k)
    order, as dispatch allocates them.
    """
    G, H, seq, topk = indices.shape
    num_routed_experts = offs.shape[-1]
    holder_pos = torch.full((G, H, seq, topk), -1, dtype=torch.int64)
    page = torch.full((G, H, seq, topk), -1, dtype=torch.int64)
    for g in range(G):
        for s in range(H):
            flat = indices[g, s].reshape(-1)
            # How many earlier picks of this origin went to the same expert.
            rank = (torch.nn.functional.one_hot(flat, num_routed_experts).cumsum(0) - 1).gather(1, flat.unsqueeze(1))
            pos = table[g, flat].to(torch.int64)
            kept = pos != -1
            holder_pos[g, s] = torch.where(kept, pos, -1).reshape(seq, topk)
            page[g, s] = torch.where(kept, offs[g, s, flat].to(torch.int64) + rank.squeeze(1), -1).reshape(seq, topk)
    return holder_pos, page


def _roomy_capacity(seq_len_per_chip, topk, experts_per_chip):
    """Pages per chip: 1.25x the in-group average (seq * topk), the heaviest draw here, plus one tile per
    expert for tile-aligned regions. Depends only on geometry, so different draws share one program."""
    tiles = -(-(seq_len_per_chip * topk * 5) // (4 * ttnn.TILE_SIZE))
    return ttnn.TILE_SIZE * (tiles + experts_per_chip)


class _Routing(NamedTuple):
    indices: torch.Tensor
    table: torch.Tensor
    offs: torch.Tensor
    counts: torch.Tensor
    region: torch.Tensor
    holder_pos: torch.Tensor
    page: torch.Tensor


def _route(indices, H, num_routed_experts, seq_len_per_chip, topk):
    """The control tables and layout for one draw. All come from one get_gate_outputs call, so they
    agree: the op trusts `offs` to size every run it sends or forwards."""
    G = indices.shape[0]
    experts_per_chip = num_routed_experts // G // H
    table = _expert_dispatch_table(num_routed_experts, H, G)
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
            topk,
            expert_dispatch_table=table[g : g + 1],
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)
    return _Routing(indices, table, offs, counts, region, *_dispatch_layout(indices, table, offs))


# Built once per session for each geometry, seed and routing, so the layout variants of a case share
# it. The tensors are shared: never modify them.
@functools.cache
def _drawn_routing(G, H, seq_len_per_chip, topk, num_routed_experts, seed, routing):
    torch.manual_seed(seed)
    if routing is None:
        indices = _in_group_indices(G, H, seq_len_per_chip, topk, num_routed_experts)
    else:
        assert routing == "production", f"unknown routing {routing!r}"
        indices = _draw_indices(G, H, seq_len_per_chip, topk, num_routed_experts, *PRODUCTION_ROUTING)
    return _route(indices, H, num_routed_experts, seq_len_per_chip, topk)


# One random payload per buffer shape, drawn once: at the production width it is 3 GB and takes
# seconds. Fixtures rotate it by page, so each gets different tokens.
@functools.lru_cache(maxsize=1)
def _payload_pool(H, G, capacity, emb_dim):
    generator = torch.Generator().manual_seed(0)
    # float32 then cast: bfloat16 randn is several times slower on CPU.
    return torch.randn(H, G, capacity, emb_dim, generator=generator).to(torch.bfloat16)


class _Fixture:
    """One routing draw laid out as dispatch leaves it, on device, plus its torch reference.

    `routing` is None for in-group picks, "production" for PRODUCTION_ROUTING, or a (G, H, seq, topk)
    index tensor. `capacity`
    is None for `_roomy_capacity` or "exact" for the smallest buffer that holds the draw.

    The payload is the shared pool rotated by seed + payload_shift pages, so fixtures with different
    rotations differ at every page and stale data from another launch cannot match.
    """

    def __init__(
        self,
        mesh_device,
        H,
        G,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        # 16 tiles: two untilize blocks per tile row. At 256 there is one, so the block offset stays 0.
        emb_dim=512,
        num_routed_experts=256,
        topk=8,
        seed=11,
        routing=None,
        capacity=None,
        payload_shift=0,
    ):
        self.mesh_device = mesh_device
        self.seq_len_per_chip, self.emb_dim, self.H, self.G = seq_len_per_chip, emb_dim, H, G
        self.num_routed_experts, self.topk = num_routed_experts, topk
        self.experts_per_chip = num_routed_experts // G // H
        if routing is None or isinstance(routing, str):
            self.routing = _drawn_routing(G, H, seq_len_per_chip, topk, num_routed_experts, seed, routing)
        else:
            self.routing = _route(routing, H, num_routed_experts, seq_len_per_chip, topk)
        r = self.routing
        assert tuple(r.indices.shape) == (G, H, seq_len_per_chip, topk), r.indices.shape
        # A kept pick's expert is in this dispatch group, so combine brings it back.
        self.kept = r.holder_pos >= 0

        used = int(r.page.max()) + 1
        if capacity is None:
            capacity = _roomy_capacity(seq_len_per_chip, topk, self.experts_per_chip)
        elif capacity == "exact":
            capacity = used
        assert used <= capacity, f"the draw needs {used} pages per chip, the buffer holds {capacity}"
        self.capacity = capacity

        # Pages no pick lands on stay random too: the op must never read them into an output slot.
        self.payload = torch.roll(_payload_pool(H, G, capacity, emb_dim), shifts=seed + payload_shift, dims=2)
        # A metadata row is (origin chip's linear mesh index, token, top-k slot), as dispatch writes it.
        meta = torch.full((H, G, capacity, 3), -1, dtype=torch.int32)
        g_idx, s_idx, t_idx, k_idx = self.kept.nonzero(as_tuple=True)
        meta[r.holder_pos[self.kept], g_idx, r.page[self.kept]] = torch.stack(
            [s_idx * G + g_idx, t_idx, k_idx], dim=-1
        ).to(torch.int32)

        self._tt_payload = {}
        self.tt_meta = self._shard(meta, (0, 1), ttnn.int32)
        # expert_offsets holds every origin chip's row, replicated along the dispatch axis: a forwarding
        # chip needs them to size what it forwards. Counts and region offsets are the same on every
        # chip of a group, so each device takes one row.
        self.tt_offs = self._shard(r.offs, (None, 0), ttnn.int32)
        self.tt_counts = self._shard(r.counts[:, 0:1, :], (None, 0), ttnn.int32)
        self.tt_region = self._shard(r.region[:, 0:1, :], (None, 0), ttnn.int32)

    def _shard(self, t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
            layout=layout,
            device=self.mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def tt_payload(self, layout):
        """The payload on device in `layout`, uploaded once. A TILE upload runs a tilize program."""
        if layout not in self._tt_payload:
            self._tt_payload[layout] = self._shard(self.payload, (0, 1), ttnn.bfloat16, layout=layout)
        return self._tt_payload[layout]

    def run(self, cluster_axis, num_links, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            self.tt_payload(layout),
            self.tt_meta,
            self.tt_counts,
            self.tt_region,
            self.tt_offs,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.topk,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=cluster_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def picks_by_distance(self):
        """Kept picks per ring distance d = (holder - origin) mod H. d = 0 is the local copy, d = H/2 is
        split over both directions, any other d is d hops one way."""
        s = torch.arange(self.H).view(1, self.H, 1, 1)
        distance = (self.routing.holder_pos - s) % self.H
        return [int((self.kept & (distance == d)).sum()) for d in range(self.H)]

    def check(self, output, label):
        """Every output slot a kept pick maps to, byte-exact against the page dispatch put it at.

        Slots of picks routed to another group are not written and not compared: the op does not
        initialise its output.
        """
        r = self.routing
        got_all = ttnn.get_device_tensors(output)  # row-major over the (H, G) mesh
        assert len(got_all) == self.H * self.G, f"{label}: {len(got_all)} device tensors for {self.H * self.G} chips"
        checked = 0
        bad = 0
        for dev in range(self.H * self.G):
            s, g = dev // self.G, dev % self.G
            mask = self.kept[g, s]
            if not mask.any():
                continue
            got = ttnn.to_torch(got_all[dev]).reshape(self.seq_len_per_chip, self.topk, self.emb_dim)[mask]
            want = self.payload[r.holder_pos[g, s][mask], g, r.page[g, s][mask]]
            checked += int(mask.sum())
            if not torch.equal(got, want):
                wrong = (got != want).any(-1)
                t, k = mask.nonzero()[wrong.nonzero()[0, 0]].tolist()
                logger.error(
                    f"{label}: device {dev} (pos {s}, group {g}): {int(wrong.sum())}/{int(mask.sum())} slots differ, "
                    f"first at token {t} top-k {k}, sent from pos {int(r.holder_pos[g, s, t, k])} "
                    f"page {int(r.page[g, s, t, k])}"
                )
                bad += 1
        assert checked > 0, f"{label}: no kept picks; the reference or the routing is wrong"
        assert bad == 0, f"{label}: {bad} of {self.H * self.G} devices differ from the combine reference"
        logger.info(f"{label}: {checked} slots byte-exact")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# In-group gives every pick a chip to come back from. Production routes over all experts, so most picks
# belong to other groups and the kept ones concentrate on a few chips.
@pytest.mark.parametrize("routing", [None, "production"], ids=lambda r: r or "in-group")
# The routed expert hands combine TILE tokens. Untilizer cores next to the senders turn them into rows,
# so both layouts send the same bytes and share one reference.
@pytest.mark.parametrize(
    "input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=lambda ly: "tile" if ly == ttnn.TILE_LAYOUT else "rm"
)
@pytest.mark.timeout(900)
def test_combine_fabric2d(mesh_device, device_params, num_links, routing, input_layout):
    cfg = extract_mesh_config(mesh_device)
    assert cfg.sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"
    H, G = cfg.dispatch_group_size, cfg.num_dispatch_groups
    if routing == "production" and G == 1:
        pytest.skip("production routing sends most picks to other dispatch groups; this mesh has only one")
    # The layouts share the draw but not the payload: the op does not initialise its output, so the
    # other layout's leftover output could otherwise match.
    fx = _Fixture(
        mesh_device,
        H,
        G,
        seed=7,
        routing=routing,
        payload_shift=int(input_layout == ttnn.TILE_LAYOUT),
    )
    logger.info(
        f"combine_fabric2d: mesh={tuple(mesh_device.shape)} H={H} G={G} experts_per_chip={fx.experts_per_chip} "
        f"seq={fx.seq_len_per_chip} topk={fx.topk} capacity={fx.capacity}"
    )

    # This checks the draw, not the op: a route with no picks goes untested.
    by_distance = fx.picks_by_distance()
    logger.info(f"kept picks by ring distance: {by_distance}")
    missing = [d for d, n in enumerate(by_distance) if n == 0]
    assert not missing, f"no picks at ring distance {missing} (0 is the local copy); those routes go untested"

    output = fx.run(cfg.sp_axis, num_links, layout=input_layout)
    assert tuple(output.shape)[-3:] == (fx.seq_len_per_chip, fx.topk, fx.emb_dim), output.shape
    fx.check(output, "combine")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
# 2880 (gpt_oss_120b) is 90 tiles, not a multiple of 8, so the untilizer falls back to 6-tile blocks.
# 7168 is the production token (14336 B, 224 tiles), the size the fabric payload limit is checked against.
@pytest.mark.parametrize("emb_dim", [2880, 7168], ids=lambda e: f"emb{e}")
@pytest.mark.timeout(1800)
def test_combine_fabric2d_relaunch(mesh_device, device_params, num_links, emb_dim):
    """Four launches on one device (row-major, tile, tile, row-major) to catch state leaking from one
    launch into the next.

    - Program cache: only TILE has untilizer cores, so a cache key missing the layout would reuse the
      wrong program.
    - Handshake counters: they outlive the launch and only the kernels zero them. A stale one makes the
      next launch read or overwrite slots early; a new draw per launch turns that into wrong slots.
    """
    cfg = extract_mesh_config(mesh_device)
    plan = [
        ("row-major", ttnn.ROW_MAJOR_LAYOUT),
        ("tile", ttnn.TILE_LAYOUT),
        ("tile again", ttnn.TILE_LAYOUT),
        ("row-major again", ttnn.ROW_MAJOR_LAYOUT),
    ]
    added = []
    for seed, (label, layout) in enumerate(plan):
        fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=emb_dim, seed=20 + seed)
        fx.tt_payload(layout)  # uploaded first, so its tilize program is not counted below
        before = mesh_device.num_program_cache_entries()
        output = fx.run(cfg.sp_axis, num_links, layout=layout)
        added.append(mesh_device.num_program_cache_entries() - before)
        # Reading the output back keeps the launches apart; test_combine_fabric2d_back_to_back overlaps them.
        fx.check(output, f"{label}, emb {emb_dim}")
    # Each layout's first launch builds one program and the repeats must hit the cache. A key that is too
    # specific rebuilds every launch and still passes the byte-exact checks, but costs prefill perf.
    assert added == [1, 1, 0, 0], f"programs the four launches added: {added}, expected [1, 1, 0, 0]"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_combine_fabric2d_partial_last_tile(mesh_device, device_params, num_links):
    """A buffer sized exactly to the draw, so it ends part way into a tile, combines byte-exact in both
    layouts.

    TILE untilizes that last tile whole, padding rows included; only real pages may reach the output.
    660 tokens per chip also makes seq_len_per_chip a non-multiple of 32.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seq_len_per_chip=660, capacity="exact")
    assert fx.capacity % ttnn.TILE_SIZE != 0, f"capacity {fx.capacity} is tile-aligned, so no tile is partial"
    for layout, label in ((ttnn.ROW_MAJOR_LAYOUT, "row-major"), (ttnn.TILE_LAYOUT, "tile")):
        output = fx.run(cfg.sp_axis, num_links, layout=layout)
        fx.check(output, f"{label}, {fx.capacity} pages, 660 tokens per chip")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_combine_fabric2d_unaligned_emb_dim(mesh_device, device_params, num_links, expect_error):
    """A token that is not a multiple of 64 B is refused before launch.

    A forwarded token and its 64 B routing tail share one DRAM page. 496 bf16 is 992 B: it passes the
    16 B NoC check and fails that one. TILE, because a ROW_MAJOR row this size trips the page-size check
    first wherever DRAM aligns to 64 B.
    """
    cfg = extract_mesh_config(mesh_device)
    fx = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, emb_dim=496)
    fx.tt_payload(ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "must be 64-byte aligned for DRAM"):
        fx.run(cfg.sp_axis, num_links, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _PRODUCTION_MESH,
    indirect=["mesh_device", "device_params"],
)
# TILE adds the untilizer handshakes to the race.
@pytest.mark.parametrize(
    "input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=lambda ly: "tile" if ly == ttnn.TILE_LAYOUT else "rm"
)
@pytest.mark.timeout(900)
# The reader zeroes fwd_arrived at end of stream, dropping a bump from an upstream chip already in the
# next launch, which hangs the ring: https://github.com/tenstorrent/tt-metal/issues/57948
@pytest.mark.skip(reason="overlapped launches can hang: https://github.com/tenstorrent/tt-metal/issues/57948")
def test_combine_fabric2d_back_to_back(mesh_device, device_params, num_links, input_layout):
    """Four launches queued with no host sync between them, as in a traced replay.

    A chip that finishes early can send into a neighbour still finishing the previous launch, racing the
    counter reset that launch ends with: a lost increment hangs, a stale one reads early. Alternating two
    draws makes stale slots visible.
    """
    cfg = extract_mesh_config(mesh_device)
    a = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=31)
    b = _Fixture(mesh_device, cfg.dispatch_group_size, cfg.num_dispatch_groups, seed=32)
    order = [a, b, a, b]
    # Uploaded first, so no upload sits between the queued launches.
    for fx in (a, b):
        fx.tt_payload(input_layout)

    results = [fx.run(cfg.sp_axis, num_links, layout=input_layout) for fx in order]
    for i, (fx, output) in enumerate(zip(order, results)):
        fx.check(output, f"launch {i} of four with no host sync between them")
    logger.info("back-to-back: 4 unsynchronised launches byte-exact")
