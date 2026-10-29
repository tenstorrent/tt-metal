# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Functional correctness of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

The block mask and the pooled K / V are built in torch, so the cases control density and structure exactly.
Each case runs the sparse op and the dense `ring_joint_scaled_dot_product_attention` on the same inputs, and
checks the sparse output against:

  dense      the dense ring op. A keep-all mask makes the sparse op dense attention, so it must match closely;
             at lower densities it must stay close.
  reference  fp32 torch block-sparse attention with the same mask: kept 64-token blocks exact, every other valid
             block one pooled entry (mean K, mean V) weighted by its token count. This is what the op computes,
             so it separates kernel errors from the approximation. The kept-only tests omit pooled K / V (the
             VSA fine-branch mode); their reference drops the pooled entries and normalizes over the kept blocks.

Runs on a single ring (1xN) or the 4x8 Galaxy (TP=4 heads x SP=8 ring). The 5 s MiniMax-H3 geometry is skipped
on CI.
"""

import math
import os
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)

BLOCK = 64  # tokens per (square) mask block, passed as block_size; also the q chunk. The op supports 64 only for now
HEADS_PER_DEVICE = 14  # MiniMax-H3 at TP=4
D = 128
KEEP_ALL = 1.0
SLOW = pytest.mark.skipif(os.environ.get("CI") == "true", reason="production-size shapes - skip on CI")


def _pad32(n):
    return -(-n // 32) * 32


@dataclass(frozen=True)
class Geometry:
    rows: int  # padded rows per device (ring shard)
    logical_n: int  # real tokens in the whole sequence
    k_chunk: int  # the sparse op's K / V piece size
    dense_chunks: tuple = (256, 512)  # the dense op's (q_chunk, k_chunk)

    def nb(self):
        return -(-self.rows // BLOCK)

    def nb_pad(self):
        return _pad32(self.nb())


def _pcc(a, b):
    """PCC over the entries where both are defined (a sliced reference leaves the rest NaN)."""
    a, b = a.flatten().double(), b.flatten().double()
    keep = ~(torch.isnan(a) | torch.isnan(b))
    a, b = a[keep], b[keep]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _block_lengths(geo, S):
    """[S, nb_pad] valid tokens per block (64, a partial last block, the logical_n tail, or 0 for padding)."""
    lens = torch.zeros(S, geo.nb_pad(), dtype=torch.int64)
    for s in range(S):
        for b in range(geo.nb()):
            start = s * geo.rows + b * BLOCK
            end = min(start + BLOCK, (s + 1) * geo.rows, geo.logical_n)
            lens[s, b] = max(0, end - start)
    return lens


def make_mask(geo, S, H, *, density, band=1, seed=0):
    """[S, H, nb_pad, S * nb_pad] 0/1 block mask: shard s's local Q block qb keeps global block (s', b).

    Kept: block 0 of every shard (the op needs one exact block per shard), the diagonal band |i - j| <= band in
    global block order, and other full blocks at random with probability `density`. Only full 64-token blocks may
    be exact; partial and padding blocks are always pooled or ignored.
    """
    g = torch.Generator().manual_seed(seed)
    nb, nb_pad = geo.nb(), geo.nb_pad()
    lens = _block_lengths(geo, S)  # [S, nb_pad]
    full = (lens == BLOCK).reshape(-1)  # [S * nb_pad]
    mask = (torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < density).to(torch.int64)
    k_global = torch.arange(S * nb_pad)
    k_index = (k_global // nb_pad) * nb + (k_global % nb_pad)  # global block number in sequence order
    for s in range(S):
        q_index = s * nb + torch.arange(nb_pad)
        mask[s, :, (q_index[:, None] - k_index[None, :]).abs() <= band] = 1
    mask[..., torch.arange(S) * nb_pad] = 1  # block 0 of every shard
    mask = mask * full.to(torch.int64)  # full blocks only
    mask[:, :, nb:] = 0  # padding Q rows
    return mask


def pool(x, geo, S):
    """[1, H, S * rows, D] -> [1, H, S * nb_pad, D] per-block mean over the valid tokens (zero for padding)."""
    lens = _block_lengths(geo, S)
    _, H, _, Dh = x.shape
    out = torch.zeros(1, H, S * geo.nb_pad(), Dh)
    for s in range(S):
        for b in range(geo.nb()):
            n = int(lens[s, b])
            if n:
                start = s * geo.rows + b * BLOCK
                out[0, :, s * geo.nb_pad() + b] = x[0, :, start : start + n].float().mean(1)
    return out


def sparse_reference(q, k, v, mask, pooled_k, pooled_v, geo, S, scale, *, pooled=True, heads=None, shards=None):
    """fp64 block-sparse attention: kept blocks exact, other valid blocks one pooled entry of weight len.

    pooled=False: kept blocks only (no pooled entries); the softmax normalizes over the kept blocks.
    heads / shards: compute only these (the rest of the output stays NaN), in 512-row chunks, so a slice of a
    production-size call is checked exactly in seconds; _pcc compares the computed entries only.
    """
    _, H, _, _ = q.shape
    nb, nb_pad = geo.nb(), geo.nb_pad()
    lens = _block_lengths(geo, S).reshape(-1)  # [S * nb_pad]
    valid = lens > 0
    log_len = torch.where(valid, lens.clamp(min=1).double().log(), torch.zeros_like(lens, dtype=torch.float64))
    # token -> padded global block id, and which tokens are real
    tok = torch.arange(S * geo.rows)
    tok_block = (tok // geo.rows) * nb_pad + (tok % geo.rows) // BLOCK
    tok_valid = tok < geo.logical_n
    out = torch.full((1, H, S * geo.rows, q.shape[3]), float("nan"))
    for h in range(H) if heads is None else heads:
        kh, vh = k[0, h].double(), v[0, h].double()
        pk, pv = pooled_k[0, h].double(), pooled_v[0, h].double()
        for s, r0 in ((s, r0) for s in (range(S) if shards is None else shards) for r0 in range(0, geo.rows, 512)):
            r1 = min(r0 + 512, geo.rows)
            rows = slice(s * geo.rows + r0, s * geo.rows + r1)
            qh = q[0, h, rows].double()  # [rows, D]
            keep = mask[s, h, (torch.arange(r0, r1) // BLOCK)].bool()  # [rows, S * nb_pad]
            exact = keep[:, tok_block] & tok_valid[None, :]  # [rows, S * rows]
            le = (qh @ kh.T) * scale
            le = le.masked_fill(~exact, float("-inf"))
            lp = (qh @ pk.T) * scale + log_len[None, :]
            lp = lp.masked_fill(keep | ~valid[None, :] | (not pooled), float("-inf"))
            w = torch.softmax(torch.cat([le, lp], dim=1), dim=1)
            out[0, h, rows] = (w[:, : le.shape[1]] @ vh + w[:, le.shape[1] :] @ pv).float()
    return out


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


class Case:
    """Inputs, device buffers and both ops for one geometry on the module's mesh."""

    def __init__(self, rt, geo):
        self.rt, self.geo = rt, geo
        self.S, self.TP = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
        self.H = HEADS_PER_DEVICE * self.TP  # heads in total; HEADS_PER_DEVICE per device
        self.scale = 1.0 / math.sqrt(D)
        dev = rt.mesh_device
        shape = tuple(dev.shape)

        def dims(sp, tp):
            d = [None, None]
            d[rt.sp_axis], d[rt.tp_axis] = sp, (tp if self.TP > 1 else None)
            return d

        self.seq_dims = dims(2, 1)  # q / k / v / output: sequence on SP, heads on TP
        self.mapper = lambda d: ttnn.ShardTensor2dMesh(dev, mesh_shape=shape, dims=d)
        self.composer = ttnn.ConcatMesh2dToTensor(
            dev, mesh_shape=shape, dims=[x if x is not None else 0 for x in self.seq_dims]
        )
        n = self.S * geo.rows
        self.buf_k, self.buf_v = (
            ttnn.from_torch(
                torch.zeros(1, self.H, n, D),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=self.mapper(dims(None, 1)),
            )
            for _ in range(2)
        )
        grid = rt.sdpa_compute_grid
        self.sparse_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=BLOCK, k_chunk_size=geo.k_chunk, exp_approx_mode=False
        )
        self.dense_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=geo.dense_chunks[0],
            k_chunk_size=geo.dense_chunks[1],
            exp_approx_mode=False,
        )
        self.dims = dims

    def _upload(self, x, d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(x, dtype=dtype, layout=layout, device=self.rt.mesh_device, mesh_mapper=self.mapper(d))

    def inputs(self, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (1, self.H, self.S * self.geo.rows, D)
        # bf16-rounded on host so the reference sees exactly what the device reads
        return tuple(torch.randn(shape, generator=g).bfloat16().float() for _ in range(3))

    def blocks_per_core(self):
        """(most, fewest) Q blocks a working core holds: the op splits the flat (head, Q block) units evenly."""
        cols, rows = self.rt.sdpa_compute_grid
        cores = cols * rows
        total = HEADS_PER_DEVICE * self.geo.nb()
        return -(-total // cores), max(1, total // cores)

    def run(self, q, k, v, mask, *, want_dense=True, want_ref=True, q_blocks_per_pass=None, pooled=True):
        """pooled=False omits pooled K / V: kept blocks only (the VSA fine-branch mode)."""
        rt, geo = self.rt, self.geo
        tq, tk, tv = (self._upload(x, self.seq_dims) for x in (q, k, v))
        pooled_k, pooled_v = (pool(x, geo, self.S).bfloat16().float() for x in (k, v))
        t_mask = self._upload(mask, self.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        t_pk, t_pv = (self._upload(x, self.dims(None, 1)) for x in (pooled_k, pooled_v)) if pooled else (None, None)
        out = ttnn.transformer.sparse_ring_sdpa(
            tq,
            tk,
            tv,
            t_mask,
            t_pk,
            t_pv,
            persistent_output_buffer_k=self.buf_k,
            persistent_output_buffer_v=self.buf_v,
            logical_n=geo.logical_n,
            program_config=self.sparse_config,
            compute_kernel_config=rt.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=rt.mesh_device,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            block_size=BLOCK,
            q_blocks_per_pass=q_blocks_per_pass,
        )
        sparse = ttnn.to_torch(out, mesh_composer=self.composer).float()
        dense = None
        if want_dense:
            tt_dense = call_sdpa(
                tq,
                tk,
                tv,
                geo.logical_n,
                False,
                False,
                self.buf_k,
                self.buf_v,
                self.dense_config,
                rt.compute_kernel_config,
                rt.ccl_semaphore_handles,
                rt.num_links,
                rt.sp_axis,
                rt.mesh_device,
                rt.topology,
                rt.worker_sub_device_id,
                rt.ccl_column,
            )
            dense = ttnn.to_torch(tt_dense, mesh_composer=self.composer).float()
        # Reference coverage (fp64 on the host dominates the test time):
        #   True / "sample": heads first, second, middle, last on every shard (every device, the head boundaries);
        #   "slice": first and last head of the first and last shard (production sizes); "full": everything.
        if want_ref == "full":
            sl = {}
        elif want_ref == "slice":
            sl = dict(heads=[0, self.H - 1], shards=[0, self.S - 1])
        else:
            sl = dict(heads=sorted({0, 1, self.H // 2, self.H - 1}))
        ref = (
            sparse_reference(q, k, v, mask, pooled_k, pooled_v, geo, self.S, self.scale, pooled=pooled, **sl)
            if want_ref
            else None
        )
        n = geo.logical_n  # rows past logical_n are padding the ops leave undefined
        return {
            "sparse": sparse[:, :, :n],
            "dense": None if dense is None else dense[:, :, :n],
            "ref": None if ref is None else ref[:, :, :n],
        }


def _density(mask, geo, S):
    full = (_block_lengths(geo, S) == BLOCK).reshape(-1)
    return float(mask[:, :, : geo.nb()][..., full].float().mean())


def _check(tag, r, density, *, dense_min, ref_min):
    p_dense = _pcc(r["dense"], r["sparse"]) if r["dense"] is not None else float("nan")
    p_ref = _pcc(r["ref"], r["sparse"]) if r["ref"] is not None else float("nan")
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs dense {p_dense:.5f}, vs reference {p_ref:.5f}")
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    if dense_min is not None:
        assert p_dense > dense_min, f"{tag}: pcc vs dense {p_dense:.5f}"
    if ref_min is not None:
        assert p_ref > ref_min, f"{tag}: pcc vs reference {p_ref:.5f}"


# rows 512: few pieces per head; 1024: about two Q blocks per core, cores straddling heads; k 256 / 512: 4 / 8
# blocks per piece; a logical_n below S * rows: a short last shard.
ALIGNED = {
    "r512k512": Geometry(512, 512 * MESH_CONFIG.sp_size, 512),
    "r1024k256": Geometry(1024, 1024 * MESH_CONFIG.sp_size, 256),
    "r1024k512": Geometry(1024, 1024 * MESH_CONFIG.sp_size, 512),
    "r1024k256short": Geometry(1024, 1024 * MESH_CONFIG.sp_size - 512, 256),
}
# Shards ending in a partial block (32 tokens), and a logical_n off a block boundary: those blocks are pooled by
# design even with every full block kept, so these are judged against the reference.
RAGGED = {
    "r1184k256": Geometry(1184, 1184 * MESH_CONFIG.sp_size, 256),
    "r1184k512": Geometry(1184, 1184 * MESH_CONFIG.sp_size, 512),
    "r1184ln_off": Geometry(1184, 1184 * MESH_CONFIG.sp_size - 472, 256),
}


@pytest.mark.parametrize("geom", list(ALIGNED))
def test_keep_all_matches_dense(runtime, geom):
    """Every full block kept and no partial blocks: the sparse op is dense attention."""
    case = Case(runtime, ALIGNED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=KEEP_ALL)
    r = case.run(*case.inputs(0), mask, want_ref="full" if geom == "r512k512" else True)
    _check(f"keepall {geom}", r, _density(mask, case.geo, case.S), dense_min=0.9995, ref_min=0.9995)


@pytest.mark.parametrize("density", [0.5, 0.2, 0.05])
@pytest.mark.parametrize("geom", list(ALIGNED))
def test_sparse_vs_dense(runtime, geom, density):
    """Random kept blocks over the forced band: exact against the reference. Dense is logged, not gated: on randn
    inputs dense attention is nearly uniform and its output close to zero, so how close the block-sparse
    approximation comes to it depends on the data, not on the kernel."""
    case = Case(runtime, ALIGNED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=density, seed=1)
    r = case.run(*case.inputs(1), mask)
    _check(f"d{density} {geom}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("density", [KEEP_ALL, 0.2])
@pytest.mark.parametrize("geom", list(RAGGED))
def test_ragged_shards(runtime, geom, density):
    """Partial last blocks of each shard and a logical_n tail are pooled, never exact."""
    case = Case(runtime, RAGGED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=density, seed=2)
    r = case.run(*case.inputs(2), mask)
    # With every full block kept, only the partial blocks are pooled: close to dense (measured 0.990-0.991).
    dense_min = 0.98 if density == KEEP_ALL else None
    _check(f"ragged d{density} {geom}", r, _density(mask, case.geo, case.S), dense_min=dense_min, ref_min=0.999)


def test_minimal_mask(runtime):
    """Only block 0 of every shard and the diagonal: most ring iterations carry one exact block per Q block and
    the pooled entries carry almost all of the softmax mass."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_mask(case.geo, case.S, case.H, density=0.0, band=0, seed=3)
    r = case.run(*case.inputs(3), mask)
    _check("minimal", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)
    assert torch.isfinite(r["sparse"]).all()


def test_heads_differ(runtime):
    """Each head has its own mask: a dense first head and a minimal last head in the same call."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=4)
    mask[:, 0] = make_mask(case.geo, case.S, 1, density=KEEP_ALL)[:, 0]
    mask[:, -1] = make_mask(case.geo, case.S, 1, density=0.0, band=0)[:, 0]
    r = case.run(*case.inputs(4), mask)
    _check("heads differ", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


def test_determinism(runtime):
    """Same inputs twice give the same bits: a race in the K / V delivery or the plan shows up here first."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=5)
    q, k, v = case.inputs(5)
    a = case.run(q, k, v, mask, want_dense=False, want_ref=False)["sparse"]
    b = case.run(q, k, v, mask, want_dense=False, want_ref=False)["sparse"]
    assert torch.equal(a, b), f"max |diff| {(a - b).abs().max().item():.3e}"


@pytest.mark.parametrize("geom", ["r1024k256", "r1024k512"])
def test_fresh_inputs_per_call(runtime, geom):
    """Four calls with the program cached and new inputs and masks each: a call that skipped an all-gather wait
    would compute on the previous call's K / V and fail its own reference."""
    case = Case(runtime, ALIGNED[geom])
    for seed in (11, 12, 13, 14):
        mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=seed)
        r = case.run(*case.inputs(seed), mask, want_dense=False)
        _check(f"call {seed} {geom}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@SLOW
@pytest.mark.parametrize("density", [KEEP_ALL, 0.15])
@pytest.mark.parametrize("k_chunk", [256, 384])
def test_production_5s(runtime, k_chunk, density):
    """MiniMax-H3 at 5 s: 4736 rows per device, the dense op at its measured chunks (320, 384). Gated on a sliced fp64
    reference (first / last head of the first / last shard); dense is logged: at 37,888 random keys dense attention
    is nearly flat, so bf16 rounding in either op dominates its PCC."""
    geo = Geometry(4736, 4736 * MESH_CONFIG.sp_size, k_chunk, dense_chunks=(320, 384))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=6)
    r = case.run(*case.inputs(6), mask, want_ref="slice")
    _check(f"5s k{k_chunk} d{density}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("passes", [2, 3])
@pytest.mark.parametrize("geom", ["r1024k256", "r1184k256"])
def test_forced_passes(runtime, geom, passes):
    """Each core's Q blocks split into several passes over K / V gathered once: the same output as one pass.

    Not bit-identical in general: a step's lazy-rescale decision covers the Q blocks stepped together, and the
    grouping depends on how many blocks are resident.
    """
    geo = ALIGNED.get(geom) or RAGGED[geom]
    case = Case(runtime, geo)
    most, fewest = case.blocks_per_core()
    q_blocks_per_pass = -(-most // passes)
    if -(-most // q_blocks_per_pass) != passes or passes > fewest:
        pytest.skip(f"{most} / {fewest} blocks per core cannot be split into {passes} passes")
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=7)
    q, k, v = case.inputs(7)
    one = case.run(q, k, v, mask, want_dense=False, want_ref=False)
    r = case.run(q, k, v, mask, want_dense=False, q_blocks_per_pass=q_blocks_per_pass)
    p_one = _pcc(one["sparse"], r["sparse"])
    max_diff = (one["sparse"] - r["sparse"]).abs().max().item()
    logger.info(
        f"[passes {geom}] {passes} passes of <= {q_blocks_per_pass} blocks ({most} per core): pcc vs one pass "
        f"{p_one:.6f}, max |diff| {max_diff:.3e}"
    )
    _check(f"passes {passes} {geom}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    assert p_one > 0.9999, f"{passes} passes vs one pass: pcc {p_one:.6f}"


def _cb_bytes(geo, S, slots, kv_pages):
    """The sparse op's slot-dependent CB bytes (lay_out_cbs in sparse_ring_sdpa_program_factory.cpp): bf16 tiles,
    D 128, the pooled route on. Keep in step with the factory."""
    sq, sk, dt = BLOCK // 32, geo.k_chunk // 32, D // 32
    tiles = (
        sq * dt * slots  # q
        + 2 * sk * dt * kv_pages  # k, v
        + 2  # identity scale, column identity
        + max(sq * sk, min(slots, 4) * sq * sk)  # qk_im
        + 4 * sq * slots  # max, exp max diff, stats, sum
        + sq * dt  # out
        + sq * dt * slots  # prev out
        + 1  # recip scratch
        + sk  # pooled bias
    )
    return tiles * 2048 + 64 + S * geo.nb_pad() * 4  # + the plan page


def _pass_split(geo, S, most, usable):
    """(kv pages, passes) the factory picks with `usable` bytes of L1 below the lowest live buffer, group delivery
    on: 3 pages unless 2 need fewer passes."""

    def passes(pages):
        slots = next((s for s in range(min(32, most), 0, -1) if _cb_bytes(geo, S, s, pages) <= usable), 0)
        return -(-most // slots) if slots else math.inf

    p3, p2 = passes(3), passes(2)
    return (2, p2) if p2 < p3 else (3, p3)


def test_two_kv_pages_with_group_mcast(runtime):
    """Group delivery over a 2-page K/V CB in several passes: the factory picks 2 pages when they need fewer passes
    than 3, which happens with the pipeline's L1 (CCL buffers pinned) but not with this runtime's free L1. An L1
    filler buffer lowers the live L1 top into a window where 3 pages need more passes than 2 (from the factory's CB
    formula, mirrored in _cb_bytes). Assumes interleaved L1 buffers are allocated top-down in the allocator the
    factory queries (lowest_occupied_compute_l1_address), so the filler is the lowest live buffer; the factory's
    log line ("... K/V pages ...") shows the choice."""
    geo = Geometry(2048, 2048 * MESH_CONFIG.sp_size, 256)
    case = Case(runtime, geo)
    most, fewest = case.blocks_per_core()
    # Every window [lo, hi) of usable L1 over which the choice is constant; keep those where 2 pages win and still
    # take several passes, the largest first (the smallest filler).
    edges = sorted({_cb_bytes(geo, case.S, s, p) for s in range(1, most + 1) for p in (2, 3)})
    target = None
    for lo, hi in reversed(list(zip(edges, edges[1:]))):
        pages, n_passes = _pass_split(geo, case.S, most, lo)
        if pages == 2 and 2 <= n_passes <= fewest:
            target = (lo + hi) // 2 // 32 * 32
            break
    if target is None:
        pytest.skip(f"{most} / {fewest} blocks per core: no L1 size makes 2 pages win with several passes")

    dev = runtime.mesh_device
    base = ttnn._ttnn.device.GetAllocatorBaseAddress(dev, ttnn.BufferType.L1)

    def l1_buffer(n_bytes):
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, n_bytes // 2),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )

    mask = make_mask(geo, case.S, case.H, density=0.2, seed=12)
    q, k, v = case.inputs(12)
    unfilled = case.run(q, k, v, mask, want_dense=False, want_ref=False)
    probe = l1_buffer(32)
    filler_bytes = (probe.buffer_address() - base - target) // 32 * 32
    if filler_bytes <= 0:
        probe.deallocate()
        pytest.skip(f"only {probe.buffer_address() - base} B of L1 free, the 2-page window needs {target}")
    filler = l1_buffer(filler_bytes)
    try:
        usable = filler.buffer_address() - base
        pages, n_passes = _pass_split(geo, case.S, most, usable)
        assert pages == 2 and n_passes >= 2, f"filler left {usable} B (expected {target}): {pages} pages"
        logger.info(f"[2 pages] {usable} B of L1 usable: {n_passes} passes over 2 K/V pages ({most} blocks per core)")
        # The live L1 top is not in the program hash: rebuild instead of reusing the unfilled run's program.
        dev.clear_program_cache()
        r = case.run(q, k, v, mask, want_dense=False)
    finally:
        filler.deallocate()
        probe.deallocate()
        dev.clear_program_cache()
    p_unfilled = _pcc(unfilled["sparse"], r["sparse"])
    _check("2 kv pages", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    assert p_unfilled > 0.9999, f"2 pages vs the unfilled run: pcc {p_unfilled:.6f}"


@SLOW
@pytest.mark.parametrize("density", [KEEP_ALL, 0.15])
def test_production_10s(runtime, density):
    """MiniMax-H3 at 10 s: 9184 rows per device, more Q blocks per core than fit L1 at once, so the op picks
    several passes (logged by the program factory). Gated on a sliced fp64 reference; dense is logged (each shard's
    32-token last block is pooled even with every full block kept)."""
    geo = Geometry(9184, 9184 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    most, _ = case.blocks_per_core()
    logger.info(f"[10s] {most} Q blocks per core at most")
    mask = make_mask(geo, case.S, case.H, density=density, seed=8)
    r = case.run(*case.inputs(8), mask, want_ref="slice")
    _check(f"10s d{density}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


# Kept blocks only (no pooled K / V): the mode the VSA fine branch needs. The softmax normalizes over the kept blocks,
# so only keep-all matches dense; sparser masks are judged against the kept-only reference.


@pytest.mark.parametrize("geom", list(ALIGNED))
def test_kept_only_keep_all_matches_dense(runtime, geom):
    """No pooled route and every full block kept: dense attention."""
    case = Case(runtime, ALIGNED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=KEEP_ALL)
    r = case.run(*case.inputs(9), mask, pooled=False, want_ref="full" if geom == "r512k512" else True)
    _check(f"kept-only keepall {geom}", r, _density(mask, case.geo, case.S), dense_min=0.9995, ref_min=0.9995)


@pytest.mark.parametrize("density", [0.2, 0.05])
@pytest.mark.parametrize("geom", list(ALIGNED))
def test_kept_only(runtime, geom, density):
    """No pooled route: exactly the softmax over the kept blocks (dense is logged, not gated)."""
    case = Case(runtime, ALIGNED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=density, seed=10)
    r = case.run(*case.inputs(10), mask, pooled=False)
    _check(f"kept-only d{density} {geom}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("geom", ["r1024k256", "r1184k256"])
def test_kept_only_forced_passes(runtime, geom):
    """No pooled route, two passes: the same output as one pass and the kept-only reference."""
    geo = ALIGNED.get(geom) or RAGGED[geom]
    case = Case(runtime, geo)
    most, fewest = case.blocks_per_core()
    q_blocks_per_pass = -(-most // 2)
    if -(-most // q_blocks_per_pass) != 2 or fewest < 2:
        pytest.skip(f"{most} / {fewest} blocks per core cannot be split into 2 passes")
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=11)
    q, k, v = case.inputs(11)
    one = case.run(q, k, v, mask, want_dense=False, want_ref=False, pooled=False)
    r = case.run(q, k, v, mask, want_dense=False, q_blocks_per_pass=q_blocks_per_pass, pooled=False)
    p_one = _pcc(one["sparse"], r["sparse"])
    _check(f"kept-only passes {geom}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    assert p_one > 0.9999, f"2 passes vs one pass: pcc {p_one:.6f}"
