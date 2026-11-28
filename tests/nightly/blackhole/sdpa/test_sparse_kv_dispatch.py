# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_kv_dispatch: the KV rows every chip's top-k needs, sent over store-and-forward trees into the full-size
gather buffer at their owner-major pages (KV_DISPATCH_2D_PLAN.md).

Exactness: on every chip, every page a needed row maps to equals the cache's row (the owner-major layout is the chips'
cache shards concatenated in row-major chip order, which is what high_bw_all_gather writes)."""

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.blackhole.sdpa.kvd2d_routing import Routing

TOPK, STRIPE = 2048, 160

_ROUTER = ttnn.FabricRouterConfig()
_ROUTER.max_packet_payload_size_bytes = int(
    os.environ.get("KVD_PAYLOAD", 12352)
)  # GLM-5.3 mesh (moe_fabric_payload_size)
_TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 2048,
    "trace_region_size": 64 * 1024 * 1024,
    "fabric_router_config": _ROUTER,
}

# every 2D fabric the dispatch supports, with the axes it wraps (wrap_x: along a mesh row, wrap_y: along a column)
_FABRICS = {
    "torus_xy": (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, True, True),
    "torus_x": (ttnn.FabricConfig.FABRIC_2D_TORUS_X, True, False),
    "torus_y": (ttnn.FabricConfig.FABRIC_2D_TORUS_Y, False, True),
    "mesh_2d": (ttnn.FabricConfig.FABRIC_2D, False, False),
}


# (KVD_MESH=4,4: the 4x4 carve, run through its mesh graph descriptor)
_MESH = tuple(int(x) for x in os.environ.get("KVD_MESH", "8,4").split(","))


def _fabric_params(name):
    return {**_TORUS_XY, "fabric_config": _FABRICS[name][0]}


def _grid(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])


class Dispatch:
    """Owns the dispatch's buffers and semaphores for one mesh, cache size and row format."""

    def __init__(self, mesh_device, max_rows, width, dtype, planners=None, links=2, lanes_per_link=None):
        # defaults: the op's (48 planners, 2 lanes a link), overridable by KVD_PLANNERS / KVD_LANES
        self.mesh_device, self.max_rows, self.links = mesh_device, max_rows, links
        self.planners = planners or int(os.environ.get("KVD_PLANNERS", "48"))
        self.lanes = lanes_per_link or int(os.environ.get("KVD_LANES", "2"))
        n = mesh_device.get_num_devices()
        common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        zeros = ttnn.from_torch(
            torch.zeros(1, 1, max_rows, width, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
            **common,
        )
        self.output = zeros if dtype == ttnn.bfloat16 else ttnn.typecast(zeros, dtype)
        slot_bytes = ttnn._ttnn.operations.experimental.sparse_kv_push_inbox_slot_bytes(max_rows, STRIPE, n)
        # the inbox (every chip's plan slice for this owner) in L1, interleaved (KVD_INBOX_L1=0: DRAM): the
        # planners read ~32 x 64 B lines a word from it
        inbox_mc = ttnn.DRAM_MEMORY_CONFIG if os.environ.get("KVD_INBOX_L1", "1") == "0" else ttnn.L1_MEMORY_CONFIG
        self.inbox = ttnn.from_torch(
            torch.zeros(1, 1, n, slot_bytes // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=inbox_mc,
        )
        spec = ttnn.experimental.sparse_kv_dispatch_scratch_spec(
            mesh_device, max_rows, STRIPE, self.planners, links, self.lanes
        )
        self.scratch = ttnn.allocate_tensor_on_device(spec, mesh_device)
        grid = _grid(mesh_device)
        self.sems = [ttnn.create_global_semaphore(mesh_device, grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(4)]
        ttnn.synchronize_device(mesh_device)

    def plan(self, indices, chunk_start):
        table, _ = ttnn.experimental.sparse_push_plan(
            indices,
            stripe_chunk=STRIPE,
            max_rows=self.max_rows,
            page_bytes=self.output.buffer_aligned_page_size(),
            chunk_start=chunk_start,
            owner_major=True,
        )
        return table

    def variant(self, planners, tag=None):
        """The kwargs of a call with another planner count: its own scratch and persistent words (one set a program
        variant, as the model's dispatch_resources)."""
        if not hasattr(self, "_variants"):
            self._variants = {}
        key = (planners, tag)  # (tag: another placement of that count, e.g. single_program: its own words)
        if key not in self._variants:
            mesh_device = self.output.device()
            spec = ttnn.experimental.sparse_kv_dispatch_scratch_spec(
                mesh_device, self.max_rows, STRIPE, planners, self.links, self.lanes
            )
            sems = [
                ttnn.create_global_semaphore(mesh_device, _grid(mesh_device), 0, ttnn.BufferType.L1_SMALL)
                for _ in range(4)
            ]
            self._variants[key] = dict(
                planners=planners,
                scratch=ttnn.allocate_tensor_on_device(spec, mesh_device),
                fence_semaphore=sems[0],
                arrived_semaphore=sems[1],
                credit_semaphore=sems[2],
                inbox_semaphore=sems[3],
            )
        return self._variants[key]

    def replay_buffer(self):
        if not hasattr(self, "_replay"):
            mesh_device = self.output.device()
            spec = ttnn.experimental.sparse_kv_dispatch_replay_spec(mesh_device, self.max_rows, STRIPE)
            self._replay = ttnn.allocate_tensor_on_device(spec, mesh_device)
        return self._replay

    def __call__(self, cache, table, slot=0, **kw):
        args = dict(
            output_tensor=self.output,
            table=table,
            inbox=self.inbox,
            scratch=self.scratch,
            stripe_chunk=STRIPE,
            max_rows=self.max_rows,
            fence_semaphore=self.sems[0],
            arrived_semaphore=self.sems[1],
            credit_semaphore=self.sems[2],
            inbox_semaphore=self.sems[3],
            num_links=self.links,
            planners=self.planners,
            lanes_per_link=self.lanes,
            batch_index=slot,
        )
        args.update(kw)
        return ttnn.experimental.sparse_kv_dispatch(cache, **args)


def _cache(mesh_device, slots, rows, kv_fmt, seed):
    gen = torch.Generator().manual_seed(seed)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    if kv_fmt == "bf16":
        host = torch.randn(slots, 1, rows, 576, generator=gen).to(torch.bfloat16)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, mesh_mapper=shard, **common), 576, ttnn.bfloat16
    latent = torch.randn(slots, 1, rows, 512, generator=gen).to(torch.bfloat16)
    rope = torch.randn(slots, 1, rows, 64, generator=gen).to(torch.bfloat16)
    fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        ttnn.from_torch(latent, dtype=ttnn.bfloat16, mesh_mapper=shard, **common), round_scale_to_power_of_two=True
    )
    cache = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(
        fp8, scales, ttnn.from_torch(rope, dtype=ttnn.bfloat16, mesh_mapper=shard, **common)
    )
    return cache, 656, ttnn.fp8_e4m3


def _touched(mesh_shape, natural, max_rows, wrap_x=True, wrap_y=True):
    """Per chip, the pages the dispatch may write: rows it needs, and rows its trees relay through it."""
    R, C = mesh_shape
    n = R * C
    rt = Routing(R, C, wrap_x and C >= 3, wrap_y and R >= 3)
    masks = {}
    for i in range(n):
        u = np.unique(natural[i].ravel())
        for g in u[(u >= 0) & (u < max_rows)].tolist():
            masks[g] = masks.get(g, 0) | (1 << i)
    touched = [set() for _ in range(n)]
    for g, mk in masks.items():
        o = (g // STRIPE) % n
        page = o * (max_rows // n) + (g // STRIPE // n) * STRIPE + g % STRIPE
        for i in range(n):
            if mk >> i & 1:
                touched[i].add(page)
        m = mk & ~(1 << o)
        if m:
            edges, _ = rt.tree(o, m, rt.order_of(o, m, g), g)
            for _, (r, c) in edges:
                touched[r * C + c].add(page)
    return touched


def _as_bits(t):
    """Raw bits of a host tensor read back from device (exact compare, NaN payloads included)."""
    t = t.contiguous()
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t.view(torch.int32)


def _check(mesh_device, d, cache, slot, natural):
    n = mesh_device.get_num_devices()
    # fp8 has no single-device to_torch: compare both sides after the same on-device typecast to bf16
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_all = ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[slot, 0]
    outs = ttnn.get_device_tensors(as_bf16(d.output))
    bad = 0
    for i in range(n):
        need = np.unique(natural[i].ravel())
        need = need[(need >= 0) & (need < d.max_rows)]
        pages = (need // STRIPE % n) * (d.max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
        got = ttnn.to_torch(outs[i])[0, 0]
        p = torch.from_numpy(pages.astype(np.int64))
        g, e = _as_bits(got[p].to(torch.float32)), _as_bits(cache_all[p].to(torch.float32))
        miss = (g != e).any(dim=1)
        bad += int(miss.sum())
        if miss.any():
            zero = (got[p][miss].abs().sum(dim=1) == 0).sum()
            if os.environ.get("KVD_DEBUG") and i < 2:
                ca = _as_bits(cache_all.to(torch.float32))
                for r in need[miss.numpy()][:4]:
                    pg = (r // STRIPE % n) * (d.max_rows // n) + (r // STRIPE // n) * STRIPE + r % STRIPE
                    row_bits = _as_bits(got[pg : pg + 1].to(torch.float32))
                    hit = (ca == row_bits).all(dim=1).nonzero().flatten().tolist()
                    logger.error(f"  chip {i} row {r} page {pg}: content equals cache pages {hit[:4]}")
            own = int((need // STRIPE % n == i).sum())
            logger.error(
                f"chip {i}: {int(miss.sum())} of {len(pages)} needed rows wrong ({int(zero)} still zero; {own} own), "
                f"e.g. rows {need[miss.numpy()][:5]}"
            )
    return bad


@run_for_blackhole()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params, fabric",
    [(_fabric_params(f), f) for f in _FABRICS],
    ids=list(_FABRICS),
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("kv_fmt", ["fp8", "bf16"])
@pytest.mark.parametrize("slabs, picks", [(4, 300), (8, 2048)], ids=["small", "dense_picks"])
@pytest.mark.parametrize("lanes", [1, 2], ids=["lanes1", "lanes2"])
def test_dispatch_exact(mesh_device, fabric, kv_fmt, slabs, picks, lanes):
    """Random top-k over three calls (program cache hits, persistent words back at zero): uniform, per-chip extents, empty."""
    n = mesh_device.get_num_devices()
    max_rows = slabs * n * STRIPE
    cache, width, dtype = _cache(mesh_device, 2, max_rows, kv_fmt, seed=1)
    d = Dispatch(mesh_device, max_rows, width, dtype, lanes_per_link=lanes)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rng = np.random.default_rng(3)
    # call 0: every chip picks over the whole cache; call 1: chip i's picks stop at a different point (its plan covers
    # fewer blocks, after a longer call left words in the inbox); call 2: nothing needed
    poison_host = torch.full((1, 1, max_rows, width), 3.0, dtype=torch.bfloat16)
    poison = ttnn.from_torch(
        poison_host,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
    poison = poison if dtype == ttnn.bfloat16 else ttnn.typecast(poison, dtype)
    poison_bits = _as_bits(
        ttnn.to_torch(
            ttnn.get_device_tensors(poison if dtype == ttnn.bfloat16 else ttnn.typecast(poison, ttnn.bfloat16))[0]
        )[0, 0].to(torch.float32)
    )
    for call, slot in ((0, 0), (1, 1), (2, 0)):
        ttnn.copy(poison, d.output)
        natural = np.stack([rng.integers(0, max_rows, size=(STRIPE, TOPK)) for _ in range(n)])
        if call == 1:
            for i in range(n):
                natural[i] %= max(STRIPE, max_rows * (i + 1) // n)
        natural[:, :, picks:] = max_rows  # sentinel past the cache: ignored
        if call == 2:
            natural[:] = max_rows
        idx = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard,
            **common,
        )
        table = d.plan(idx, chunk_start=max_rows - n * STRIPE)
        d(cache, table, slot=slot)
        ttnn.synchronize_device(mesh_device)
        bad = _check(mesh_device, d, cache, slot, natural)
        assert bad == 0, f"call {call}: {bad} needed rows wrong"
        # nothing else changed: every page outside the chip's needed and relayed rows still holds the poison
        touched = _touched(tuple(mesh_device.shape), natural, max_rows, *_FABRICS[fabric][1:])
        outs = ttnn.get_device_tensors(d.output if dtype == ttnn.bfloat16 else ttnn.typecast(d.output, ttnn.bfloat16))
        for i in sorted({0, n // 2, n - 1}):
            got = _as_bits(ttnn.to_torch(outs[i])[0, 0].to(torch.float32))
            keep = torch.ones(max_rows, dtype=torch.bool)
            keep[list(touched[i])] = False
            stray = int((got[keep] != poison_bits[keep]).any(dim=1).sum())
            assert stray == 0, f"call {call} chip {i}: {stray} pages written that no tree puts there"
        logger.info(f"call {call}: every needed row landed, nothing else written")


@run_for_blackhole()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params, fabric",
    [(_fabric_params(f), f) for f in _FABRICS],
    ids=list(_FABRICS),
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
@pytest.mark.parametrize("split_slabs", [0, 3, 7], ids=["split0", "split_mid", "split_last"])
@pytest.mark.parametrize(
    "replay", ["no_replay", "replay", "replay_mixed", "mixed", "replay_fallback", "replay_fewer", "replay_stale"]
)
@pytest.mark.parametrize("slabs", [8, 4], ids=["slabs8", "slabs4"])
def test_dispatch_parts(mesh_device, fabric, split_slabs, replay, slabs):
    """part=PREFIX sends exactly the needed rows of blocks before the plan's split block, REST the others: each alone
    into a poisoned buffer (its rows exact, the other part's rows still poison), then PREFIX + REST == ALL.
    replay: the first call records the planners' routing, every later one replays it (replay_mixed: recorded with 16
    planners, as the model's pre-send beside attention, replayed with the default count; replay_fallback: replays from a
    record never written, so every planner routes; replay_fewer: ALL recorded on the whole chip with the
    default count, the parts replayed with 4 planners on GLM's pre-send grid, the bottom 8 rows of the last 4 columns).
    replay_stale: replays with another plan build's generation, so every planner routes). Each replaying call's
    planners report replayed or routed."""
    # (slabs 4, 48 planners: a record stream stride not a multiple of 64 B before rounding; replay cases only)
    # (only `replay` records with 48 planners; replay_mixed's 16-planner stride was aligned anyway)
    if slabs != 8 and (replay != "replay" or split_slabs >= slabs):
        pytest.skip("slabs 4: the 48-planner replay case at splits below 4")
    if replay != "no_replay" and os.environ.get("KVD_DESIGN") != "3":
        pytest.skip("replay needs KVD_DESIGN=3")
    n = mesh_device.get_num_devices()
    max_rows = slabs * n * STRIPE
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=7)
    d = Dispatch(mesh_device, max_rows, width, dtype, planners=48 if slabs != 8 else None)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rng = np.random.default_rng(11)
    natural = np.stack([rng.integers(0, max_rows, size=(STRIPE, TOPK)) for _ in range(n)])
    natural[:, :, 1024:] = max_rows
    idx = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        **common,
    )
    chunk_start = split_slabs * n * STRIPE
    table = d.plan(idx, chunk_start=chunk_start)
    split_block = chunk_start // STRIPE
    poison = ttnn.from_torch(
        torch.full((1, 1, max_rows, width), 3.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
    poison = poison if dtype == ttnn.bfloat16 else ttnn.typecast(poison, dtype)
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_all = ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[0, 0]
    cache_bits = _as_bits(cache_all.to(torch.float32))

    def page(g):
        return (g // STRIPE % n) * (max_rows // n) + (g // STRIPE // n) * STRIPE + g % STRIPE

    def check(sent, label):
        outs = ttnn.get_device_tensors(as_bf16(d.output))
        for i in range(n):
            need = np.unique(natural[i].ravel())
            need = need[(need >= 0) & (need < max_rows)]
            prefix = need // STRIPE < split_block
            got = _as_bits(ttnn.to_torch(outs[i])[0, 0].to(torch.float32))
            for sel, name in ((prefix, "PREFIX"), (~prefix, "REST")):
                pages = torch.from_numpy(page(need[sel]).astype(np.int64))
                if pages.numel() == 0:
                    continue
                exact = (got[pages] == cache_bits[pages]).all(dim=1)
                if name in sent:
                    assert bool(exact.all()), f"{label} chip {i}: {int((~exact).sum())} {name} rows wrong"
                else:
                    assert not bool(exact.any()), f"{label} chip {i}: {int(exact.sum())} {name} rows sent by mistake"

    blank = None
    if replay == "replay_fallback":  # a record never written (zeros: no magic)
        spec = ttnn.experimental.sparse_kv_dispatch_replay_spec(mesh_device, d.max_rows, STRIPE)
        blank = ttnn.zeros(
            spec.shape,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    small = None
    if replay == "replay_fewer":
        if d.planners <= 4:
            pytest.skip("replay_fewer: records with more planners than the 4 it replays with")
        g = mesh_device.compute_with_storage_grid_size()
        small = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(g.x - 4, g.y - 8), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    # every call after the first reuses the plan's exchanged slices (skip_exchange)
    exchanged = False
    order = (("PREFIX",), ("REST",), ("PREFIX", "REST"), ("ALL",))
    if replay == "replay_fewer":  # (GLM: the first call of a plan build is a whole-chip ALL)
        order = (("ALL",), ("PREFIX",), ("REST",), ("PREFIX", "REST"))
    for li, parts in enumerate(order):
        ttnn.copy(poison, d.output)
        for name in parts:
            kw = {}
            if replay in ("replay", "replay_mixed", "replay_fewer", "replay_stale"):
                kw = dict(replay=2 if exchanged else 1, replay_buffer=d.replay_buffer(), replay_generation=7)
            # (a record of plan build 7 replayed as build 8: PREFIX and REST first; their second calls, cache hits,
            # replay as 7: the generation is patched on a hit)
            stale = replay == "replay_stale" and exchanged and li < 2
            if stale:
                kw["replay_generation"] = 8
            if replay == "replay_fewer" and exchanged:  # (as GLM: REST one program, PREFIX per-chip placement)
                rest = name == "REST"
                kw.update(d.variant(4, "single" if rest else None), core_grid=small, single_program=rest)
            if replay == "replay_fallback" and exchanged:
                kw = dict(replay=2, replay_buffer=blank)
            if replay in ("replay_mixed", "mixed") and not exchanged:
                kw.update(d.variant(16))
            d(cache, table, part={"ALL": 0, "PREFIX": 1, "REST": 2}[name], skip_exchange=exchanged, **kw)
            exchanged = True
            if kw.get("replay") == 2:  # each replaying call: every planner of every chip 1 replayed (2: routed)
                got = _replay_status(kw["replay_buffer"], kw.get("planners", d.planners))
                want = 2 if replay == "replay_fallback" or stale else 1
                assert got == {want}, f"{fabric} split {split_slabs} {replay} {name} of {'+'.join(parts)}: status {got}"
        ttnn.synchronize_device(mesh_device)
        sent = {"PREFIX", "REST"} if parts in (("ALL",), ("PREFIX", "REST")) else set(parts)
        check(sent, "+".join(parts))
        logger.info(f"{fabric} split {split_slabs} {replay}: {'+'.join(parts)} exact")


def _replay_status(buf, planners):
    """The replay outcome each planner of each chip wrote on its last replaying call (word 2 of its header line)."""
    out = set()
    for t in ttnn.get_device_tensors(buf):
        page0 = ttnn.to_torch(t)[0, 0, 0].to(torch.int64) & 0xFFFFFFFF
        out |= {int(page0[16 * (1 + p) + 2]) for p in range(planners)}
    return out


@run_for_blackhole()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params, fabric",
    [(_fabric_params(f), f) for f in _FABRICS],
    ids=list(_FABRICS),
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
@pytest.mark.parametrize("split_slabs", [3, 7], ids=["split_mid", "split_last"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp0", "sp1"])
@pytest.mark.parametrize("replay", [False, True], ids=["no_replay", "replay"])
def test_dispatch_compact(mesh_device, fabric, split_slabs, sp_axis, replay):
    """Compact landing: the plan's rank table (rank_table=True) and the dispatch landing rows in the push's compact
    buffer. Every top-k index of every chip, rewritten by the plan (compact indices), addresses that row: ALL, and
    PREFIX then REST, each into a poisoned buffer."""
    n = mesh_device.get_num_devices()
    max_rows = 8 * n * STRIPE
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=13)
    d = Dispatch(mesh_device, max_rows, width, dtype)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rng = np.random.default_rng(17)
    natural = np.stack([rng.integers(0, max_rows, size=(STRIPE, TOPK)) for _ in range(n)])
    natural[:, :, 1536:] = max_rows  # sentinel
    idx = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        **common,
    )
    chunk_start = split_slabs * n * STRIPE
    page_bytes = cache.buffer_aligned_page_size()
    table, compact_idx, rank = ttnn.experimental.sparse_push_plan(
        idx,
        stripe_chunk=STRIPE,
        max_rows=max_rows,
        page_bytes=page_bytes,
        chunk_start=chunk_start,
        owner_major=True,
        rank_table=True,
    )
    compact = ttnn.experimental.sparse_kv_push_create_output_buffer(
        mesh_device, distinct_rows=STRIPE * TOPK, max_rows=max_rows, stripe_chunk=STRIPE, row_width=width, dtype=dtype
    )
    rows = compact.shape[2]
    poison = ttnn.from_torch(
        torch.full((1, 1, rows, width), 3.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
    poison = poison if dtype == ttnn.bfloat16 else ttnn.typecast(poison, dtype)
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_bits = _as_bits(
        ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[0, 0].to(torch.float32)
    )
    pages_of = [ttnn.to_torch(t).reshape(STRIPE, TOPK).to(torch.int64) for t in ttnn.get_device_tensors(compact_idx)]

    def call(part, skip):
        return ttnn.experimental.sparse_kv_dispatch(
            cache,
            output_tensor=compact,
            table=table,
            inbox=d.inbox,
            scratch=d.scratch,
            stripe_chunk=STRIPE,
            max_rows=max_rows,
            fence_semaphore=d.sems[0],
            arrived_semaphore=d.sems[1],
            credit_semaphore=d.sems[2],
            inbox_semaphore=d.sems[3],
            num_links=d.links,
            planners=d.planners,
            lanes_per_link=d.lanes,
            part=part,
            skip_exchange=skip,
            rank_table=rank,
            sp_axis=sp_axis,
            **(dict(replay=2 if skip else 1, replay_buffer=d.replay_buffer()) if replay else {}),
        )

    rows_n, cols_n = tuple(mesh_device.shape)
    # owner o of block b (b % n) is on chip owner_chip(o): row-major o for sp_axis 0, (o % rows, o / rows) for 1
    owner_chip = lambda o: o if sp_axis == 0 else (o % rows_n) * cols_n + o // rows_n
    exchanged = False
    for parts in ((0,), (1, 2)):
        ttnn.copy(poison, compact)
        for part in parts:
            call(part, exchanged)
            if replay and exchanged:  # each replaying call: every planner of every chip replayed
                got = _replay_status(d.replay_buffer(), d.planners)
                assert got == {1}, f"{fabric} split {split_slabs} sp{sp_axis} part {part}: replay status {got}"
            exchanged = True
        ttnn.synchronize_device(mesh_device)
        outs = ttnn.get_device_tensors(as_bf16(compact))
        for i in range(n):
            got = _as_bits(ttnn.to_torch(outs[i])[0, 0].to(torch.float32))
            nat = torch.from_numpy(natural[i]).to(torch.int64)
            valid = nat < max_rows
            pages, rws = pages_of[i][valid], nat[valid]
            # (the concatenated cache is in chip order: row g is on its owner's chip, at its local row)
            owners = rws // STRIPE % n
            chips = owners if sp_axis == 0 else (owners % rows_n) * cols_n + owners // rows_n
            rws = chips * (max_rows // n) + (rws // STRIPE // n) * STRIPE + rws % STRIPE
            wrong = int((got[pages] != cache_bits[rws]).any(dim=1).sum())
            if wrong and i == 0 and os.environ.get("KVD_DEBUG"):
                rt = ttnn.to_torch(ttnn.get_device_tensors(rank)[0])[0, 0].to(torch.int64)
                logger.error(f"rank page 0 (run bases): {rt[0, :n].tolist()}")
                bad = (got[pages] != cache_bits[rws]).any(dim=1).nonzero().flatten()[:6]
                poisoned = int((got == _as_bits(torch.full((1, got.shape[1]), 3.0))).all(dim=1).sum())
                logger.error(f"compact rows {got.shape[0]}, still poison {poisoned}")
                for j in bad.tolist():
                    r, pg = int(rws[j]), int(pages[j])
                    hit = (got == cache_bits[r]).all(dim=1).nonzero().flatten()[:4].tolist()
                    blk = r // STRIPE
                    logger.error(
                        f"row {r} (block {blk}, owner {blk % n}): plan page {pg}, row found at pages {hit}; rank page {rt[1 + blk, :12].tolist()}"
                    )
            assert wrong == 0, f"{fabric} parts {parts} chip {i}: {wrong} of {len(rws)} indices read a wrong row"
        logger.info(f"{fabric} split {split_slabs} parts {parts}: every compact index reads its row")


_GLM_TRACE = "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120/dsa"
_GLM_START = 1039360  # the trace's chunk start (a whole number of slabs)
# per-layer tree roofline (tools/roofline_real_topk_8x4.jsonl), us
_GLM_TREE_US = {6: 103, 14: 59, 30: 55, 46: 256, 62: 274, 74: 191}


def _traced_us(mesh_device, fn, n=10):
    import time

    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / n * 1e6
    ttnn.release_trace(mesh_device, tid)
    return us


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("layer", [int(x) for x in os.environ.get("KVD_LAYERS", "14,62").split(",")])
def test_dispatch_real_topk(mesh_device, layer):
    """Staged GLM-5.3 1M top-k: exact on two chips, and the traced time per call against the layer's tree roofline."""
    from safetensors.numpy import load_file

    f = f"{_GLM_TRACE}/dsa_topk_indices_layer_{layer}/rows_00000000_00005120.safetensors"
    if not os.path.exists(f):
        pytest.skip(f"not staged: {f}")
    n = mesh_device.get_num_devices()
    max_rows = _GLM_START + n * STRIPE
    natural = next(iter(load_file(f).values())).astype(np.int64).reshape(n, STRIPE, TOPK)
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=5)
    kw = {}
    if os.environ.get("KVD_B"):
        kw["rows_per_packet"] = int(os.environ["KVD_B"])
    if os.environ.get("KVD_SLOTS"):
        kw["landing_slots"] = int(os.environ["KVD_SLOTS"])
    d = Dispatch(mesh_device, max_rows, width, dtype)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    idx = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        **common,
    )
    table = d.plan(idx, chunk_start=_GLM_START)
    d(cache, table, **kw)
    ttnn.synchronize_device(mesh_device)
    if not os.environ.get("KVD_SKIP_CHECK"):
        # exactness on two chips (a full 1M readback per chip is ~1.4 GB)
        as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
        cache_all = ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[0, 0]
        outs = ttnn.get_device_tensors(as_bf16(d.output))
        for i in (0, n - 1):
            need = np.unique(natural[i].ravel())
            need = need[(need >= 0) & (need < max_rows)]
            pages = torch.from_numpy(
                ((need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE).astype(
                    np.int64
                )
            )
            got = ttnn.to_torch(outs[i])[0, 0]
            bad = int(
                (
                    (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32))).any(dim=1)
                ).sum()
            )
            assert bad == 0, f"layer {layer} chip {i}: {bad} of {len(pages)} needed rows wrong"
        logger.info(f"layer {layer}: chips 0 and n - 1 exact")
    if os.environ.get("KVD_PROFILE"):
        for _ in range(3):
            d(cache, table, **kw)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        return
    us = _traced_us(mesh_device, lambda: d(cache, table, **kw))
    roof = _GLM_TREE_US.get(layer)
    vs = f"tree roofline {roof} us, x{us / roof:.2f}" if roof else "no roofline staged"
    logger.warning(f"KVD layer {layer} 1M: {us:.0f} us per call ({vs}); HBAG measured ~4250 us")


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dispatch_ablate(mesh_device):
    """Ablations on staged GLM 1M top-k in one bring-up. KVD_ABLATE is a JSON list of configs, each
    {"name": ..., "env": {...}, "planners": 48, "lanes": 2, "kw": {...}, "check": false}; KVD_LAYERS picks layers.
    Every config clears the program cache (env-driven defines and kernel edits recompile) and is timed traced."""
    import json

    from safetensors.numpy import load_file

    configs = json.loads(os.environ.get("KVD_ABLATE", '[{"name": "default"}]'))
    layers = [int(x) for x in os.environ.get("KVD_LAYERS", "14,62").split(",")]
    n = mesh_device.get_num_devices()
    max_rows = _GLM_START + n * STRIPE
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=5)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    data = {}
    for layer in layers:
        f = f"{_GLM_TRACE}/dsa_topk_indices_layer_{layer}/rows_00000000_00005120.safetensors"
        if not os.path.exists(f):
            pytest.skip(f"not staged: {f}")
        natural = next(iter(load_file(f).values())).astype(np.int64).reshape(n, STRIPE, TOPK)
        data[layer] = (
            natural,
            ttnn.from_torch(
                torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                **common,
            ),
        )
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_all = None
    dispatchers = {}
    rows = []
    for cfg in configs:
        saved = {k: os.environ.get(k) for k in cfg.get("env", {})}
        os.environ.update({k: str(v) for k, v in cfg.get("env", {}).items()})
        try:
            mesh_device.disable_and_clear_program_cache()
            mesh_device.enable_program_cache()
            key = (cfg.get("planners", 48), cfg.get("lanes", 2))
            if key not in dispatchers:
                dispatchers[key] = Dispatch(mesh_device, max_rows, width, dtype, planners=key[0], lanes_per_link=key[1])
            d = dispatchers[key]
            kw = dict(cfg.get("kw", {}))
            if "grid_cols" in cfg:  # [first, last] columns, every row: e.g. the columns beside GLM's attention
                g = mesh_device.compute_with_storage_grid_size()
                c0, c1 = cfg["grid_cols"]
                kw["core_grid"] = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(c0, 0), ttnn.CoreCoord(min(c1, g.x - 1), g.y - 1))]
                )
                logger.warning(f"KVDA {cfg.get('name')}: grid {g.x}x{g.y}, cores {kw['core_grid'].num_cores()}")
            for layer in layers:
                natural, idx = data[layer]
                table = d.plan(idx, chunk_start=_GLM_START)
                if cfg.get("check"):  # poison first: an earlier config's rows must not pass for this one's
                    zeros = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
                    poison = zeros if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(zeros, d.output.dtype)
                    ttnn.copy(poison, d.output)
                    ttnn.deallocate(poison)
                d(cache, table, **kw)
                ttnn.synchronize_device(mesh_device)
                exact = None
                if cfg.get("check"):
                    if cache_all is None:
                        cache_all = ttnn.to_torch(
                            as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)
                        )[0, 0]
                    outs = ttnn.get_device_tensors(as_bf16(d.output))
                    bad = 0
                    for i in (0, n - 1):
                        need = np.unique(natural[i].ravel())
                        need = need[(need >= 0) & (need < max_rows)]
                        pages = torch.from_numpy(
                            (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                        )
                        got = ttnn.to_torch(outs[i])[0, 0]
                        bad += int(
                            (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32)))
                            .any(dim=1)
                            .sum()
                        )
                    exact = bad == 0
                    assert exact, f"{cfg.get('name')} L{layer}: {bad} needed rows wrong on chips 0 and n - 1"
                us = _traced_us(mesh_device, lambda: d(cache, table, **kw))
                plan_us = (
                    _traced_us(mesh_device, lambda: d.plan(idx, chunk_start=_GLM_START)) if cfg.get("plan") else None
                )
                rows.append((cfg.get("name"), layer, us, plan_us, exact))
                logger.warning(
                    f"KVDA {cfg.get('name')} L{layer}: {us:.0f} us"
                    + (f" plan {plan_us:.0f} us" if plan_us else "")
                    + (f" exact={exact}" if exact is not None else "")
                )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    for name, layer, us, plan_us, exact in rows:
        roof = _GLM_TREE_US.get(layer)
        logger.warning(
            f"KVDA | {name:24s} | L{layer} | {us:7.0f} us | x{us / roof:5.2f} roofline"
            + (f" | plan {plan_us:.0f} us" if plan_us else "")
            + (f" | exact={exact}" if exact is not None else "")
        )


# K3.1's DSA (full-attention) layers (KimiK3Config.mla_layer_ids): the dumps' occurrence i is model layer [i]
_K31_DSA_LAYERS = list(range(3, 92, 4)) + [92]
# K3.1 tree roofline on the real prompt (KV_DISPATCH_FUSED_PLAN.md section 11), us: (KV slabs, layer) -> us
_K31_TREE_US = {(0, 0): 273, (0, 1): 331, (0, 2): 503, (0, 3): 175, (0, 4): 441}
_K31_TREE_US.update({(61, 0): 81, (61, 1): 154, (61, 2): 275, (61, 3): 149, (61, 4): 237})


def _bench_setup(mesh_device, cases):
    """The bench's state: cache, Dispatch and, per case (KV slabs, layer), its plan table, the pages chips 0 and n - 1
    need, the tree roofline and a label. K3.1 (KVD_K31_DIR) layers are DSA occurrences (_K31_DSA_LAYERS). The
    Dispatch (planners, lanes) is built here, once: configs vary kernel knobs (KVD3, KVD_ABL), not KVD_PLANNERS."""
    from safetensors.numpy import load_file

    k31 = os.environ.get("KVD_K31_DIR")
    n = mesh_device.get_num_devices()
    slab = n * STRIPE
    max_rows = _GLM_START + slab
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=5)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_all = ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[0, 0]
    d = Dispatch(mesh_device, max_rows, width, dtype)
    data = {}
    for k, layer in cases:
        kv = max_rows if k == 0 else k * slab
        start = kv - slab
        if k31:
            f = f"{k31}/topk_start{start}_occ{layer}.pt"
            if not os.path.exists(f):
                pytest.skip(f"not staged: {f}")
            natural = torch.load(f).numpy().astype(np.int64).reshape(n, STRIPE, TOPK)
            roof = _K31_TREE_US.get((k, layer))
        else:
            f = f"{_GLM_TRACE}/dsa_topk_indices_layer_{layer}/rows_00000000_00005120.safetensors"
            if not os.path.exists(f):
                pytest.skip(f"not staged: {f}")
            natural = next(iter(load_file(f).values())).astype(np.int64).reshape(n, STRIPE, TOPK)
            if k:
                natural = _rescale_topk(natural, max_rows, kv)
            roof = _GLM_TREE_US.get(layer) if k == 0 else None
        idx = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
            **common,
        )
        pages = []
        for i in (0, n - 1):
            need = np.unique(natural[i].ravel())
            need = need[(need >= 0) & (need < max_rows)]
            pages.append(
                torch.from_numpy(
                    (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                )
            )
        name = f"DSA{layer}=L{_K31_DSA_LAYERS[layer]}" if k31 else f"L{layer}"
        data[(k, layer)] = (d.plan(idx, chunk_start=start), pages, roof, f"{kv // 1000}k {name}")
    ttnn.synchronize_device(mesh_device)
    return dict(mesh=mesh_device, cache=cache, cache_all=cache_all, d=d, data=data, common=common, as_bf16=as_bf16)


def _replay_once(mesh_device, fn, before_last):
    """fn captured in a trace, replayed twice; before_last() runs just before the second replay (stats marks)."""
    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    before_last()
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, tid)


def _bench_run(st, configs, cases, timed=True, after=None, traced_once=False, mark=None):
    """Every config (name, env, check): program cache cleared, exactness, then (timed) the dispatch traced, best of
    2 x 20 calls. check: "all" (chips 0 and n - 1, every case), true (the default: chip 0, the first case: reading a 1M
    output back costs ~4 s a chip), false. after(name, case): called after the checked call (e.g. to collect stats). Returns
    ({(name, case): us}, summary lines)."""
    mesh_device, cache, d, data, common = st["mesh"], st["cache"], st["d"], st["data"], st["common"]
    mesh_n = mesh_device.get_num_devices()
    res = {}
    for cfg in configs:
        name = cfg.get("name", "?")
        saved = {key: os.environ.get(key) for key in cfg.get("env", {})}
        os.environ.update({key: str(v) for key, v in cfg.get("env", {}).items()})
        try:
            mesh_device.disable_and_clear_program_cache()
            mesh_device.enable_program_cache()
            for ci, case in enumerate(cases):
                table, pages, roof, label = data[case]
                check = cfg.get("check", True)
                chips = ((0, mesh_n - 1) if check == "all" else (0,)) if check and (check == "all" or ci == 0) else ()
                if chips or not timed:
                    zeros = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
                    poison = zeros if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(zeros, d.output.dtype)
                    ttnn.copy(poison, d.output)
                    ttnn.deallocate(poison)
                    if traced_once:  # (stats as the timed runs see them: the last of two traced replays)

                        def before_last():
                            # poison again: the check sees only the last replay's rows
                            p = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
                            q = p if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(p, d.output.dtype)
                            ttnn.copy(q, d.output)
                            ttnn.synchronize_device(mesh_device)
                            if mark:
                                mark()

                        _replay_once(mesh_device, lambda: d(cache, table), before_last)
                    else:
                        d(cache, table)
                    ttnn.synchronize_device(mesh_device)
                    if after:
                        after(name, case)
                if chips:
                    outs = ttnn.get_device_tensors(st["as_bf16"](d.output))
                    bad = 0
                    for i, pg in zip(chips, pages):
                        got = ttnn.to_torch(outs[i])[0, 0]
                        bad += int(
                            (_as_bits(got[pg].to(torch.float32)) != _as_bits(st["cache_all"][pg].to(torch.float32)))
                            .any(dim=1)
                            .sum()
                        )
                    assert bad == 0, f"{name} {label}: {bad} needed rows wrong on chips {chips}"
                if timed:
                    us = min(_traced_us(mesh_device, lambda: d(cache, table), n=20) for _ in range(2))
                    res[(name, case)] = us
                    logger.warning(f"KVDB {name} {label}: {us:.0f} us")
        finally:
            for key, v in saved.items():
                if v is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = v
    lines = []
    if timed:
        base = configs[0].get("name", "?")
        for cfg in configs:
            name = cfg.get("name", "?")
            cells = []
            for case in cases:
                us, roof, label = res[(name, case)], data[case][2], data[case][3]
                cells.append(
                    f"{label} {us:6.0f}"
                    + (f" x{us / roof:4.2f}" if roof else "")
                    + (f" {100 * (us / res[(base, case)] - 1):+5.1f}%" if name != base else "")
                )
            lines.append(f"KVDB | {name:20s} | " + " | ".join(cells))
    return res, lines


def _bench_cases(spec=None):
    k31 = os.environ.get("KVD_K31_DIR")
    spec = spec or os.environ.get("KVD_BENCH", "0:2,0:3,61:0" if k31 else "0:62,0:14")
    return [tuple(int(x) for x in c.split(":")) for c in spec.split(",")]


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dispatch_bench(mesh_device):
    """A few representative cases, every config in one bring-up (one payload: run once per KVD_PAYLOAD).
    KVD_BENCH: cases "slabs:layer,..." (slabs 0 = 1M; K3.1 layer = DSA occurrence); default K3.1 (KVD_K31_DIR)
    0:2,0:3,61:0, GLM 0:62,0:14. KVD_ABLATE: JSON list of configs {"name", "env", "check" (default true)}."""
    import json

    cases = _bench_cases()
    configs = json.loads(os.environ.get("KVD_ABLATE", '[{"name": "base"}]'))
    st = _bench_setup(mesh_device, cases)
    _, lines = _bench_run(st, configs, cases)
    for line in lines:
        logger.warning(line)


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dispatch_server(mesh_device):
    """Experiment server: one bring-up, then requests from KVD_SERVER_DIR until a stop request.
    req_<id>.json: {"configs": [...] (as KVD_ABLATE), "cases": "0:2,..." (a subset of the started set), "stats": bool,
    "stop": bool}. Result res_<id>.txt: the KVDB summary lines, or the error; with stats, the dispatch runs once per
    case with KVD_STATS (untraced, or "traced": the last of two replays) and the DPRINT lines it printed (TT_METAL_DPRINT_FILE) go to res_<id>.dprint.
    Kernel sources are recompiled when they change: KVD_ABL gets SALT=<hash of the kernel sources>."""
    import glob
    import json
    import time
    import traceback

    q = os.environ.get("KVD_SERVER_DIR")
    if not q:
        pytest.skip("experiment server: set KVD_SERVER_DIR (scratch tooling, runs until a stop request)")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    kdirs = [
        f"{root}/ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/dispatch/device/kernels",
        f"{root}/ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels",
        f"{root}/ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels",
    ]
    dfile = os.environ.get("TT_METAL_DPRINT_FILE")
    all_cases = _bench_cases()
    st = _bench_setup(mesh_device, all_cases)
    with open(f"{q}/ready", "w") as f:
        f.write(" ".join(st["data"][c][3] for c in all_cases) + "\n")
    logger.warning(f"KVD server ready: {q}")
    while True:
        reqs = sorted(glob.glob(f"{q}/req_*.json"), key=os.path.getmtime)
        if not reqs:
            time.sleep(0.2)
            continue
        path = reqs[0]
        rid = os.path.basename(path)[4:-5]
        try:
            req = json.load(open(path))
        except json.JSONDecodeError:  # still being written, or bad: set aside after 5 s (it would block the queue)
            if time.time() - os.path.getmtime(path) > 5:
                os.rename(path, f"{q}/bad_{rid}.json")
                open(f"{q}/res_{rid}.txt", "w").write("ERROR: request is not valid JSON\n")
            time.sleep(0.2)
            continue
        os.rename(path, f"{q}/run_{rid}.json")
        if req.get("stop"):
            open(f"{q}/res_{rid}.txt", "w").write("stopped\n")
            break
        t0 = time.time()
        out = []
        try:
            import hashlib

            h = hashlib.sha1()
            for p in sorted(p for k in kdirs for p in glob.glob(f"{k}/*")):
                h.update(open(p, "rb").read())
            salt = int(h.hexdigest()[:7], 16)  # (kernel sources' contents: an edit recompiles)
            configs = req.get("configs", [{"name": "base"}])
            for cfg in configs:
                env = cfg.setdefault("env", {})
                abl = ",".join(x for x in (env.get("KVD_ABL", os.environ.get("KVD_ABL", "")), f"SALT={salt}") if x)
                env["KVD_ABL"] = abl
            cases = _bench_cases(req["cases"]) if req.get("cases") else all_cases
            if req.get("stats"):
                for cfg in configs:
                    cfg["env"]["KVD_STATS"] = "1"
                marks = []
                size = lambda: os.path.getsize(dfile) if dfile and os.path.exists(dfile) else 0

                def after(name, case):
                    time.sleep(2.0)  # (the DPRINT server drains asynchronously)
                    marks.append((name, st["data"][case][3], size()))

                start = [size()]

                def mark():  # traced: only the last replay's prints (the warm-up call and replay print too)
                    time.sleep(2.0)
                    start[0] = size()

                _bench_run(st, configs, cases, timed=False, after=after, traced_once=bool(req.get("traced")), mark=mark)
                with open(dfile, "rb") as f:
                    f.seek(start[0])
                    blob = f.read().decode(errors="replace")
                with open(f"{q}/res_{rid}.dprint", "w") as f:
                    f.write(blob)
                out.append(f"stats: {len(blob.splitlines())} DPRINT lines ({[(m[0], m[1]) for m in marks]})")
            else:
                _, lines = _bench_run(st, configs, cases)
                out += lines
        except Exception:
            out.append("ERROR\n" + traceback.format_exc())
        out.append(f"({time.time() - t0:.1f} s)")
        with open(f"{q}/res_{rid}.tmp", "w") as f:
            f.write("\n".join(out) + "\n")
        os.rename(f"{q}/res_{rid}.tmp", f"{q}/res_{rid}.txt")


def _rescale_topk(natural, src_rows, kv):
    """Synthetic top-k at KV length kv from real top-k over src_rows: every index scaled to [0, kv), then pushed
    apart so each query keeps its picks distinct (sorted, no index above kv - 1). Pads (< 0) stay pads."""
    flat = natural.reshape(-1, natural.shape[-1])
    valid = (flat >= 0) & (flat < src_rows)  # (pads: negative or past the cache)
    cnt = valid.sum(axis=1, keepdims=True)
    assert int(cnt.max()) <= kv, f"{int(cnt.max())} picks do not fit in {kv} rows"
    big = np.iinfo(np.int64).max // 4
    v = np.sort(np.where(valid, flat * kv // src_rows, big), axis=1)
    j = np.arange(flat.shape[1])[None, :]
    v = np.maximum.accumulate(v - j, axis=1) + j  # strictly increasing
    v = np.minimum(v, kv - cnt + j)  # room for the rest below kv
    v = np.where(j < cnt, v, -1)
    return v.reshape(natural.shape)


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dispatch_kv_sweep(mesh_device):
    """Plan + dispatch against high_bw_all_gather at a range of KV lengths (KVD_KV, in slabs of n x STRIPE rows;
    0 = the real 1M chunk). Below 1M the top-k is the real 1M top-k rescaled to the KV length (synthetic). Buffers
    are sized for 1M as in the model; HBAG gathers the populated prefix (gathered_dim_size), as mla.py does."""
    from safetensors.numpy import load_file

    k31 = os.environ.get("KVD_K31_DIR")
    layers = [int(x) for x in os.environ.get("KVD_LAYERS", "0,1,2,3,4" if k31 else "14,62").split(",")]
    n = mesh_device.get_num_devices()
    slab = n * STRIPE
    max_rows = _GLM_START + slab
    # (K3.1: slabs 61 and 101 are the real 312k and 512k dumps)
    kvs = [int(x) for x in os.environ.get("KVD_KV", "1,10,25,61,101,0" if k31 else "1,10,25,51,102,0").split(",")]
    cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=5)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    d = Dispatch(mesh_device, max_rows, width, dtype)
    hb_out = ttnn.allocate_tensor_on_device(d.output.spec, mesh_device)
    # the existing push (v2: owner-major plan table, table exchange in its own inbox, all rows), as mla.py runs it
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart, SparseKvPushPlanner

    push_planner = SparseKvPushPlanner(mesh_device, 2)
    push_rows = ttnn.experimental.sparse_kv_push_output_spec(
        mesh_device, distinct_rows=STRIPE * TOPK, max_rows=max_rows, stripe_chunk=STRIPE, row_width=width, dtype=dtype
    ).shape[2]
    push_out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec([1, 1, push_rows, width], dtype, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG), mesh_device
    )
    push_inbox = ttnn.from_torch(
        torch.zeros(1, 1, n, d.inbox.shape[3], dtype=torch.int32),
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
    push = lambda table: ttnn.experimental.sparse_kv_push(
        cache,
        output_tensor=push_out,
        table=table,
        inbox=push_inbox,
        part=SparseKvPushPart.ALL,
        stripe_chunk=STRIPE,
        max_rows=max_rows,
        num_links=2,
        ready_semaphore=push_planner.ready_semaphore,
        done_semaphore=push_planner.done_semaphore,
        batch_index=0,
    )
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    cache_all = ttnn.to_torch(as_bf16(cache), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))[0, 0]
    # KVD_K31_DIR: K3.1 top-k dumps (topk_start{start}_occ{layer}.pt, real at 312k and 1M; KVD_LAYERS = DSA layers)
    real, real_at = {}, {}  # real[layer]: 1M top-k; real_at[(kv, layer)]: other real KV points
    for layer in layers:
        if k31:
            if not os.path.exists(f"{k31}/topk_start{_GLM_START}_occ{layer}.pt"):
                pytest.skip(f"not staged: {k31}/topk_start{_GLM_START}_occ{layer}.pt")
            load = lambda start: (
                torch.load(f"{k31}/topk_start{start}_occ{layer}.pt").numpy().astype(np.int64).reshape(n, STRIPE, TOPK)
            )
            real[layer] = load(_GLM_START)
            for k in kvs:  # every other KV point with a real dump (its chunk start = kv - slab)
                if k and os.path.exists(f"{k31}/topk_start{k * slab - slab}_occ{layer}.pt"):
                    real_at[(k * slab, layer)] = load(k * slab - slab)
            continue
        f = f"{_GLM_TRACE}/dsa_topk_indices_layer_{layer}/rows_00000000_00005120.safetensors"
        if not os.path.exists(f):
            pytest.skip(f"not staged: {f}")
        real[layer] = next(iter(load_file(f).values())).astype(np.int64).reshape(n, STRIPE, TOPK)
    rows = []
    for k in kvs:
        kv = max_rows if k == 0 else k * slab
        start = kv - slab
        hb = lambda: ttnn.experimental.high_bw_all_gather(
            cache, dim=2, output_tensor=hb_out, cluster_axis=None, num_links=2, gathered_dim_size=kv
        )
        hb_us = _traced_us(mesh_device, hb)
        for layer in layers:
            natural = (
                real[layer]
                if k == 0
                else real_at.get((kv, layer), None)
                if (kv, layer) in real_at
                else _rescale_topk(real[layer], max_rows, kv)
            )
            is_real = k == 0 or (kv, layer) in real_at
            idx = ttnn.from_torch(
                torch.from_numpy(natural.reshape(1, 1, n * STRIPE, TOPK).astype(np.int32)),
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                **common,
            )
            zeros = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
            poison = zeros if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(zeros, d.output.dtype)
            ttnn.copy(poison, d.output)
            ttnn.deallocate(poison)
            table = d.plan(idx, chunk_start=start)
            d(cache, table)
            ttnn.synchronize_device(mesh_device)
            outs = ttnn.get_device_tensors(as_bf16(d.output))
            bad, need_max = 0, 0
            for i in (0, n - 1):
                need = np.unique(natural[i].ravel())
                need = need[(need >= 0) & (need < max_rows)]
                need_max = max(need_max, len(need))
                pages = torch.from_numpy(
                    (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                )
                got = ttnn.to_torch(outs[i])[0, 0]
                bad += int(
                    (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32)))
                    .any(dim=1)
                    .sum()
                )
            assert bad == 0, f"KV {kv} L{layer}: {bad} needed rows wrong on chips 0 and n - 1"
            us = _traced_us(mesh_device, lambda: d(cache, table))
            plan_us = _traced_us(mesh_device, lambda: d.plan(idx, chunk_start=start))
            # both back to back in one trace (what the model replays); vs plan_us + us: the effect of tracing the two
            # programs together (both still write and read the table)
            pd_us = _traced_us(mesh_device, lambda: d(cache, d.plan(idx, chunk_start=start)))
            fused_us = float("nan")
            if os.environ.get("KVD_FUSED"):  # one op: the plan fused into the dispatch (kvd4), exact first
                zeros = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
                poison = zeros if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(zeros, d.output.dtype)
                ttnn.copy(poison, d.output)
                ttnn.deallocate(poison)
                # its own zeroed table (a plan that wrote nothing would fail the check) and a copy of the indices
                # at another address (later calls hit the program cache with new addresses); the FUSED_NOPLAN probe
                # skips the plan, so it dispatches from the prebuilt table
                noplan = any(k.split("=")[0] == "FUSED_NOPLAN" for k in os.environ.get("KVD_ABL", "").split(","))
                zero_table = lambda: ttnn.from_torch(
                    torch.zeros(tuple(table.shape), dtype=torch.int32),
                    dtype=ttnn.uint32,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    **common,
                )
                ftable = table if noplan else zero_table()
                fidx = ttnn.clone(idx)
                fused = lambda: d(cache, ftable, plan_indices=fidx, plan_chunk_start=start)
                fused()
                ttnn.synchronize_device(mesh_device)
                outs = ttnn.get_device_tensors(as_bf16(d.output))
                fbad = 0
                for i in (0, n - 1):
                    need = np.unique(natural[i].ravel())
                    need = need[(need >= 0) & (need < max_rows)]
                    pages = torch.from_numpy(
                        (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                    )
                    got = ttnn.to_torch(outs[i])[0, 0]
                    fbad += int(
                        (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32)))
                        .any(dim=1)
                        .sum()
                    )
                assert fbad == 0, f"fused KV {kv} L{layer}: {fbad} needed rows wrong on chips 0 and n - 1"
                fused_us = _traced_us(mesh_device, fused)
                # the cached program again, with a second table and indices allocated while the first pair lives
                # (other addresses: the override must patch them), checked the same way
                ftable2 = table if noplan else zero_table()
                fidx2 = ttnn.clone(idx)
                assert fidx2.buffer_address() != fidx.buffer_address()
                zeros = ttnn.zeros(d.output.shape, dtype=ttnn.bfloat16, **common)
                poison = zeros if d.output.dtype == ttnn.bfloat16 else ttnn.typecast(zeros, d.output.dtype)
                ttnn.copy(poison, d.output)
                ttnn.deallocate(poison)
                d(cache, ftable2, plan_indices=fidx2, plan_chunk_start=start)
                ttnn.synchronize_device(mesh_device)
                outs = ttnn.get_device_tensors(as_bf16(d.output))
                for i in (0, n - 1):
                    need = np.unique(natural[i].ravel())
                    need = need[(need >= 0) & (need < max_rows)]
                    pages = torch.from_numpy(
                        (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                    )
                    got = ttnn.to_torch(outs[i])[0, 0]
                    bad2 = int(
                        (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32)))
                        .any(dim=1)
                        .sum()
                    )
                    assert bad2 == 0, f"fused (new addresses) KV {kv} L{layer} chip {i}: {bad2} needed rows wrong"
                for x in (fidx, fidx2) if noplan else (fidx, ftable, fidx2, ftable2):
                    ttnn.deallocate(x)
            push_us = float("nan")
            # (no push with PLAN_ABL=DISPATCH_ONLY: the table has no per-owner counts for it)
            if os.environ.get("KVD_SWEEP_PUSH", "0") != "0" and "DISPATCH_ONLY" not in os.environ.get("PLAN_ABL", ""):
                push(table)
                ttnn.synchronize_device(mesh_device)
                push_us = _traced_us(mesh_device, lambda: push(table))
            ttnn.deallocate(idx)
            rows.append((kv, layer, need_max, plan_us, us, push_us, hb_us, is_real, pd_us, fused_us))
            logger.warning(
                f"KVDS KV {kv} L{layer}: plan {plan_us:.0f} dispatch {us:.0f} push {push_us:.0f} hbag {hb_us:.0f} us"
            )
    push_planner.release()
    for kv, layer, need_max, plan_us, us, push_us, hb_us, is_real, pd_us, fused_us in rows:
        logger.warning(
            f"KVDS | KV {kv:8d} | L{layer} | needed rows {need_max:7d} | plan {plan_us:5.0f} | dispatch {us:5.0f} | "
            f"plan+dispatch {plan_us + us:5.0f} (one trace {pd_us:5.0f}) | fused {fused_us:5.0f} | push {push_us:5.0f} | plan+push {plan_us + push_us:5.0f} | "
            f"hbag {hb_us:5.0f} us | {'real' if is_real else 'synthetic'}"
        )


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize(
    "mesh_device", [tuple(int(x) for x in os.environ.get("KVD_MESH", "8,4").split(","))], indirect=True
)
def test_zz_push_vs_dispatch_real_scratch(mesh_device):
    """SCRATCH: the existing push (sparse_push_plan + all-gather + sparse_kv_push ALL into the compact buffer) against
    sparse_kv_dispatch, on GLM-5.3 real top-k dumped from test_sparse_mla_real_data_perf (KVD_TOPK_DIR, files
    topk_L<layer>_c<cache>.pt). Traced us per call; each op on the full grid and on the cores beside GLM attention."""
    import glob

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart, SparseKvPushPlanner

    files = sorted(glob.glob(f"{os.environ['KVD_TOPK_DIR']}/topk_*_c*.pt"))
    assert files, "no top-k dumps"
    n = mesh_device.get_num_devices()
    g = mesh_device.compute_with_storage_grid_size()
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    planner = SparseKvPushPlanner(mesh_device, links=2)
    _, push_grid = planner.grids(STRIPE)
    beside = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    logger.warning(
        f"PVD grid {g.x}x{g.y}: push beside {push_grid.num_cores()} cores, dispatch beside {beside.num_cores()}"
    )
    rows = []
    for f in files:
        dump = torch.load(f)
        tag = os.path.basename(f)[5:-3]
        start, chunk = dump["chunk_start"], dump["chunk"]
        natural = dump["indices"].reshape(n, STRIPE, TOPK).to(torch.int64)
        max_rows = -(-(start + chunk) // (n * STRIPE)) * (n * STRIPE)
        valid = natural[(natural >= 0) & (natural < max_rows)]
        distinct = [
            int(torch.unique(natural[i][(natural[i] >= 0) & (natural[i] < max_rows)]).numel()) for i in range(n)
        ]
        idx = ttnn.from_torch(
            natural.reshape(1, 1, n * STRIPE, TOPK).to(torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
            **common,
        )
        cache, width, dtype = _cache(mesh_device, 1, max_rows, "fp8", seed=5)
        res = {}
        # existing: plan (+ table all-gather) and push of every needed row into the compact buffer
        page_bytes = cache.buffer_aligned_page_size()
        planner.invalidate()
        plan = planner.plan(idx, stripe_chunk=STRIPE, page_bytes=page_bytes, max_rows=max_rows, chunk_start=start)
        compact = planner.push_buffer(0, cache, distinct_rows=STRIPE * TOPK, max_rows=max_rows, stripe_chunk=STRIPE)

        def push(grid=None, part=SparseKvPushPart.ALL):
            return ttnn.experimental.sparse_kv_push(
                cache,
                output_tensor=compact,
                table=plan.table,
                stripe_chunk=STRIPE,
                max_rows=max_rows,
                part=part,
                num_links=2,
                sp_axis=0,
                batch_index=0,
                ready_semaphore=planner.ready_semaphore,
                done_semaphore=planner.done_semaphore,
                **({"core_grid": grid} if grid is not None else {}),
            )

        def push_plan():
            planner.invalidate()
            planner.plan(idx, stripe_chunk=STRIPE, page_bytes=page_bytes, max_rows=max_rows, chunk_start=start)

        res["push"] = _traced_us(mesh_device, push)
        res["push_beside"] = _traced_us(mesh_device, lambda: push(push_grid))
        res["push_rest"] = _traced_us(mesh_device, lambda: push(part=SparseKvPushPart.REST))
        res["push_plan"] = _traced_us(mesh_device, push_plan)
        # proposed: owner-major plan, then the dispatch (48 planners full grid; 16 planners K=2 beside attention)
        for name, planners, env, grid in (
            ("kvd", 48, {}, None),
            ("kvd_beside", 16, {"KVD3": os.environ.get("KVD_BESIDE_KVD3", "K=2")}, beside),
        ):
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                mesh_device.disable_and_clear_program_cache()
                mesh_device.enable_program_cache()
                d = Dispatch(mesh_device, max_rows, width, dtype, planners=planners)
                table = d.plan(idx, chunk_start=start)
                kw = {"core_grid": grid} if grid is not None else {}
                if os.environ.get("KVD_B"):  # rows per packet (smaller slots, less L1)
                    kw["rows_per_packet"] = int(os.environ["KVD_B"])
                d(cache, table, **kw)
                ttnn.synchronize_device(mesh_device)
                if name == "kvd":  # exactness on chips 0 and n - 1, as test_dispatch_ablate
                    cache_all = ttnn.to_torch(
                        ttnn.typecast(cache, ttnn.bfloat16), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)
                    )[0, 0]
                    outs = ttnn.get_device_tensors(ttnn.typecast(d.output, ttnn.bfloat16))
                    bad = 0
                    for i in (0, n - 1):
                        need = torch.unique(natural[i][(natural[i] >= 0) & (natural[i] < max_rows)])
                        pages = (need // STRIPE % n) * (max_rows // n) + (need // STRIPE // n) * STRIPE + need % STRIPE
                        got = ttnn.to_torch(outs[i])[0, 0]
                        bad += int(
                            (_as_bits(got[pages].to(torch.float32)) != _as_bits(cache_all[pages].to(torch.float32)))
                            .any(dim=1)
                            .sum()
                        )
                    assert bad == 0, f"{tag}: dispatch wrote {bad} wrong rows"
                res[name] = _traced_us(mesh_device, lambda: d(cache, table, **kw))
                if name == "kvd":  # REST on the exchanged plan, as a shared layer after a pre-send
                    res["kvd_rest"] = _traced_us(mesh_device, lambda: d(cache, table, part=2, skip_exchange=True, **kw))
                # plan once, dispatch many: record once (eager), then replay ALL and REST
                rb = dict(replay_buffer=d.replay_buffer())
                d(cache, table, skip_exchange=True, replay=1, **rb, **kw)
                ttnn.synchronize_device(mesh_device)
                res[name + "_replay"] = _traced_us(
                    mesh_device, lambda: d(cache, table, skip_exchange=True, replay=2, **rb, **kw)
                )
                if name == "kvd":
                    res["kvd_rest_replay"] = _traced_us(
                        mesh_device, lambda: d(cache, table, part=2, skip_exchange=True, replay=2, **rb, **kw)
                    )
                if name == "kvd":
                    res["kvd_plan"] = _traced_us(mesh_device, lambda: d.plan(idx, chunk_start=start))
                # compact landing (the push's compact buffer, the plan's rank table), same grid and planners
                _, _, rank = ttnn.experimental.sparse_push_plan(
                    idx,
                    stripe_chunk=STRIPE,
                    max_rows=max_rows,
                    page_bytes=page_bytes,
                    chunk_start=start,
                    owner_major=True,
                    rank_table=True,
                )
                ctab = ttnn.experimental.sparse_push_plan(
                    idx,
                    stripe_chunk=STRIPE,
                    max_rows=max_rows,
                    page_bytes=page_bytes,
                    chunk_start=start,
                    owner_major=True,
                    rank_table=True,
                )[0]
                cbuf = ttnn.experimental.sparse_kv_push_create_output_buffer(
                    mesh_device,
                    distinct_rows=STRIPE * TOPK,
                    max_rows=max_rows,
                    stripe_chunk=STRIPE,
                    row_width=width,
                    dtype=dtype,
                )
                ckw = dict(kw, rank_table=rank)
                ckw.pop("output_tensor", None)
                compact_call = lambda: ttnn.experimental.sparse_kv_dispatch(
                    cache,
                    output_tensor=cbuf,
                    table=ctab,
                    inbox=d.inbox,
                    scratch=d.scratch,
                    stripe_chunk=STRIPE,
                    max_rows=max_rows,
                    fence_semaphore=d.sems[0],
                    arrived_semaphore=d.sems[1],
                    credit_semaphore=d.sems[2],
                    inbox_semaphore=d.sems[3],
                    num_links=d.links,
                    planners=d.planners,
                    lanes_per_link=d.lanes,
                    **ckw,
                )
                compact_call()
                ttnn.synchronize_device(mesh_device)
                res[name + "_compact"] = _traced_us(mesh_device, compact_call)
                ckw.update(rb, skip_exchange=True, replay=1)
                compact_call()
                ttnn.synchronize_device(mesh_device)
                ckw["replay"] = 2
                res[name + "_compact_replay"] = _traced_us(mesh_device, compact_call)
                ttnn.deallocate(d.output)
                ttnn.deallocate(d.scratch)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        mesh_device.disable_and_clear_program_cache()
        mesh_device.enable_program_cache()
        row = (tag, start, int(valid.numel()), max(distinct), sum(distinct) // n, res)
        rows.append(row)
        logger.warning(f"PVD {tag}: " + " ".join(f"{k}={v:.0f}" for k, v in res.items()))
        ttnn.deallocate(cache)
        ttnn.deallocate(idx)
    for tag, start, _, dmax, dmean, res in rows:
        logger.warning(
            f"PVD | {tag:12s} | distinct/chip mean {dmean:6d} max {dmax:6d} | "
            + " | ".join(f"{k} {v:6.0f}" for k, v in res.items())
        )
