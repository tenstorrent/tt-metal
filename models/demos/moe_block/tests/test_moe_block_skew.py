# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Can a late chip's combine packets reach a chip that already finished the combine?

The combine's home chips forward each source chip's ring layouts to the source's port once per call, whatever rows the
source sends them (combine_port_writer.cpp send_layouts). A source port with no rows for a home finishes as soon as its
own work is done, so a late home's layout packet could land in the source's arena after its program ended - in L1 the
next op owns (the arena is per call). The fix: a port ends only after every layout packet addressed to it has landed.

LoudBox 2x4 (FABRIC_2D), GLM-5.3-Flash at chunk 1280, combine_snap 2^-14 (the combine's sum is order independent, so
outputs are bit-comparable across arrival orders). Two neighbouring chips are made cold (their experts get no rows: a
large negative router bias) and idle (a padded chunk whose real tokens end before theirs: they send no rows either), so
their programs have nothing to wait for but the combine's start barrier and the ports between them nothing to inject
or return; one of them is made late (a spin kernel on it right before the experts + combine program), so its layouts
come last while the other one's program is about to end. Each skewed forward:
  * the combine's output (y_own) and the block's output are bit-identical to the unskewed forward's;
  * a sentinel on every worker core of every other chip, right after its combine, fills the combine's layout words
    ([arena + LAYR, + 4 N AP)) with a pattern, waits, and counts the words a late packet overwrote: zero.
Positive control: an intentionally unfenced fabric writer (kernels/poke.cpp) on one chip sprays that range of another
chip's cores while its sentinel watches: the sentinel must count the writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

R, C = 2, 4
CHUNK = 1280
ACC = 9  # (the combine plans 9 accumulators on the LoudBox's 11x10 grid)
E, K, H, I, Is = 288, 8, 4096, 2048, 2048  # GLM-5.3-Flash
SNAP = 2.0**-14
LATE, IDLE = (1, 3), (1, 2)  # (the cold, idle chips: the last two, the last one late)
SPAN = (0, CHUNK - 2 * CHUNK // (R * C))  # (a padded chunk: every chip's tokens real but the last two chips')
ITERS = 10
US = 1350  # wall-clock cycles per us (AICLK ~1.35 GHz)
WATCH_US = 2000  # (how long the sentinel watches after the combine)
PATTERN = 0xA5A55A5A
REPO = Path(__file__).resolve().parents[4]
KDIR = "models/demos/moe_block/tests/kernels/"


def _fw():
    path = REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py"
    spec = importlib.util.spec_from_file_location("flat_routed_expert_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hash(*parts) -> int:
    return hash(parts) & 0xFFFFFFFFFFFF


def _one_core():
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _delay_op(mesh, dummy, cycles_by_coord: dict):
    """Spin `cycles` on core (0, 0) of each listed chip."""
    desc = ttnn.MeshProgramDescriptor()
    for (r, c), cycles in cycles_by_coord.items():
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [int(cycles)]
        k = ttnn.KernelDescriptor(
            kernel_source=KDIR + "delay.cpp",
            core_ranges=_one_core(),
            compile_time_args=[],
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
        prog.custom_program_hash = _hash("delay", r, c, int(cycles))
        coord = ttnn.MeshCoordinate(r, c)
        desc[ttnn.MeshCoordinateRange(coord, coord)] = prog
    ttnn.generic_op([dummy, dummy], desc)


def _worker_cores(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return [(x, y) for y in range(grid.y) for x in range(grid.x)]


def _sentinel_prog(mesh, out, target, words, cycles, coord):
    cores = _worker_cores(mesh)
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [int(target), int(words), PATTERN, int(cycles), int(out.buffer_address()), i]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "sentinel.cpp",
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
    prog.custom_program_hash = _hash("sentinel", coord, int(target), int(words), int(cycles))
    return prog


def _poke_prog(mesh, src, dst, target, nbytes, pre_delay):
    """Unfenced: src sprays [target, target + nbytes) of every worker core of dst (kernels/poke.cpp)."""
    cores = []
    for x, y in _worker_cores(mesh):
        v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        cores.append(v.x | (v.y << 16))
    chunk = 64
    dst_id = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*dst))
    src_id = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*src))
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [
        int(dst_id.mesh_id),
        int(dst_id.chip_id),
        0,
        int(target),
        int(nbytes),
        chunk,
        1,
        int(pre_delay),
        0xFFFFFFFF,
        len(cores),
        *cores,
    ]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "poke.cpp",
        core_ranges=_one_core(),
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=chunk,
        core_ranges=_one_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=chunk)],
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.kernels[0].runtime_args[0][0].extend(
        ttnn.setup_fabric_connection(src_id, dst_id, 0, prog, ttnn.CoreCoord(0, 0))
    )
    prog.custom_program_hash = _hash("poke", src, dst, int(target), int(nbytes), int(pre_delay))
    return prog


def _counts(out, skip=()) -> list[int]:
    """per chip (row-major): the words the sentinel found overwritten, summed over its cores (skip: chips without one)"""
    res = []
    for i, d in enumerate(ttnn.get_device_tensors(out)):
        if i in skip:
            res.append(0)
            continue
        t = ttnn.to_torch(d).reshape(-1, 16).to(torch.int64)
        assert bool((t[:, 2] == 0x5E47).all()), f"chip {i}: a sentinel core did not report"
        res.append(int(t[:, 0].sum()))
    return res


class _Skew:
    """block.mb with the skew and the sentinel around the experts + combine program, and its output kept."""

    def __init__(self, inner, mesh, dummy, out, layout_words):
        self._inner, self.mesh, self.dummy, self.out = inner, mesh, dummy, out
        self.layout_words = layout_words  # (LAYR offset, words)
        self.block = None  # (its arena_address: this forward's arena)
        self.delay = {}  # {coord: cycles} before the combine
        self.watch = False
        self.y_own = None

    def __getattr__(self, k):
        f = getattr(self._inner, k)
        if k == "routed_experts_combine":

            def combine(*a, **kw):
                if self.delay:
                    _delay_op(self.mesh, self.dummy, self.delay)
                return f(*a, **kw)

            return combine
        if k == "fused_tail":

            def tail(y_own, *a, **kw):
                self.y_own = y_own
                if self.watch:  # (right after the combine on every chip: its layout words, watched)
                    off, words = self.layout_words
                    desc = ttnn.MeshProgramDescriptor()
                    for r in range(R):
                        for c in range(C):
                            coord = ttnn.MeshCoordinate(r, c)
                            desc[ttnn.MeshCoordinateRange(coord, coord)] = _sentinel_prog(
                                self.mesh, self.out, self.block.arena_address + off, words, WATCH_US * US, (r, c)
                            )
                    ttnn.generic_op([self.dummy, self.out], desc)
                return f(y_own, *a, **kw)

            return tail
        return f


@pytest.fixture(scope="module")
def mesh():
    if ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} LoudBox")
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = (2 * H + 32 + 63) // 64 * 64
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=rc)
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=5888, trace_region_size=64 << 20)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def setup(mesh):
    mb = ttnn.experimental.moe_block
    fw = _fw()
    act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=10.0)
    dims = mb.MoEModelDims(
        num_experts=E, top_k=K, emb_dim=H, routed_expert_hidden_dim=I, routed_expert_activation=act,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=2.5),
        shared_expert_hidden_dim=Is, shared_expert_activation=act,
    )  # fmt: skip
    N, EPC = R * C, E // (R * C)
    dtypes = mb.MoEDtypes()
    cfg = mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, persist_l1_small=True)
    cfg.combine_snap = SNAP
    res = mb.MoEBlockResources(mesh, dims, cfg, dtypes)
    block = MoEBlock(mesh, res, dtypes=dtypes, accumulators=ACC, activation=fw.ACTS["silu"])
    g = torch.Generator().manual_seed(5)
    rnd = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
    f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
    bias = rnd(E, sc=0.01)
    for r, c in (LATE, IDLE):  # (their experts are never picked)
        bias[(r * C + c) * EPC : (r * C + c + 1) * EPC] -= 100.0
    head_w = mb.prepare_fused_head_weights(
        f32(rnd(Is, H, sc=H**-0.5)), f32(rnd(Is, H, sc=H**-0.5)), f32(rnd(H, Is, sc=Is**-0.5)), None,
        f32(rnd(E, H, sc=H**-0.5)), f32(bias), resources=res, dtypes=dtypes, accumulators=ACC,
    )  # fmt: skip
    distinct = [(rnd(H, I, sc=H**-0.5), rnd(H, I, sc=H**-0.5), rnd(I, H, sc=I**-0.5)) for _ in range(4)]
    gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
    experts = fw.FlatRoutedExpert(
        mesh, [[distinct[i % 4] for i in range(EPC)]] * N, m=block.expert_kw["max_tokens_per_expert"],
        H=dims.routed_width, I=I, gids=gids, n_global=E, act="silu", pin=1, layout=block.expert_kw["layout"],
    )  # fmt: skip
    weights = MoEBlockWeights(head=head_w, expert_ids=experts.gidx, gate_up=experts.w_gu, down=experts.w_d)
    x = ttnn.from_torch(
        rnd(1, 1, CHUNK, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )  # fmt: skip
    rep = dict(device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    dummy = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **rep)
    ncores = len(_worker_cores(mesh))
    out = ttnn.from_torch(
        torch.zeros(1, 1, ncores, 16, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **rep
    )
    st = mb.combine_state_layout(N, ACC)  # (CRED, USED, RET, LAYR, LAYO, U0, CNT, MISC, HDRS, RINGS)
    AP = (ACC + 3) // 4 * 4
    skew = _Skew(block.mb, mesh, dummy, out, (st[3], N * AP))
    block.mb = skew
    skew.block = block
    yield dict(mesh=mesh, block=block, weights=weights, x=x, skew=skew, out=out, dummy=dummy, layr=st[3], N=N, AP=AP)
    experts.close()
    res.release()


def _owner_rows(t):
    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1]) for d in ttnn.get_device_tensors(t)], dim=0)


def _forward(s):
    mesh, block, skew = s["mesh"], s["block"], s["skew"]
    y = block(s["x"], s["weights"], actual_start=SPAN[0], actual_end=SPAN[1])
    ttnn.synchronize_device(mesh)
    out = dict(
        y_own=_owner_rows(skew.y_own).contiguous().view(torch.int16), out=_owner_rows(y).contiguous().view(torch.int16)
    )
    ttnn.deallocate(y)
    return out


def test_skewed_combine(setup):
    s = setup
    skew = s["skew"]
    _forward(s)  # (warm: every program built, the arena address known)
    ref = _forward(s)
    assert torch.equal(ref["y_own"], _forward(s)["y_own"]), "not deterministic unskewed"
    skews = {
        "late 5 us": {LATE: 5 * US},
        "late 200 us": {LATE: 200 * US},
        "late 1 ms": {LATE: 1000 * US},
        "all but late 300 us": {(r, c): 300 * US for r in range(R) for c in range(C) if (r, c) != LATE},
    }
    bad = []
    for name, delay in skews.items():
        skew.delay, skew.watch = delay, True
        worst_out, worst_late = 0, [0] * (R * C)
        try:
            for _ in range(ITERS):
                o = _forward(s)
                worst_out = max(worst_out, int((o["y_own"] != ref["y_own"]).sum()), int((o["out"] != ref["out"]).sum()))
                worst_late = [max(a, b) for a, b in zip(worst_late, _counts(s["out"]))]
        finally:
            skew.delay, skew.watch = {}, False
        logger.info(f"skew {name:22s} x{ITERS}: output mismatches {worst_out}, late layout words per chip {worst_late}")
        if worst_out or any(worst_late):
            bad.append((name, worst_out, worst_late))
    assert not bad, f"the combine under skew: {bad}"


def test_sentinel_positive_control(setup):
    """An unfenced writer on chip (0, 0) sprays chip (0, 1)'s layout words while every chip's sentinel watches: only
    (0, 1)'s sentinel counts writes."""
    s = setup
    mesh, skew = s["mesh"], s["skew"]
    if getattr(s["block"], "arena_address", None) is None:
        _forward(s)
    src, dst = (0, 0), (0, 1)
    target, words = s["block"].arena_address + s["layr"], s["N"] * s["AP"]
    desc = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(C):
            coord = ttnn.MeshCoordinate(r, c)
            if (r, c) == src:
                prog = _poke_prog(mesh, src, dst, target, words * 4, pre_delay=200 * US)
            else:
                prog = _sentinel_prog(mesh, s["out"], target, words, WATCH_US * US, (r, c))
            desc[ttnn.MeshCoordinateRange(coord, coord)] = prog
    ttnn.generic_op([s["dummy"], s["out"]], desc)
    ttnn.synchronize_device(mesh)
    si, di = src[0] * C + src[1], dst[0] * C + dst[1]
    got = _counts(s["out"], skip=(si,))
    logger.info(f"positive control: late words per chip {got}")
    assert got[di] > 0, "the sentinel missed the unfenced writes"
    assert all(v == 0 for i, v in enumerate(got) if i != di), got
