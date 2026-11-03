# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Functional correctness of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

The block mask and the pooled K / V are built in torch, so the cases control density and structure exactly. Each case
runs the sparse op (and, where it is compared, the dense `ring_joint_scaled_dot_product_attention`) on the same
inputs and checks the sparse output against:

  reference  fp64 torch block-sparse attention with the same mask: kept 64-token blocks exact, every other valid
             block one pooled entry (mean K, mean V) weighted by its token count (Sol). The kept-only tests omit
             pooled K / V (VSA's fine branch); their reference drops the pooled entries and normalizes over the kept
             blocks. This is what the op computes, so it separates kernel errors from the approximation.
  dense      the dense ring op. A keep-all mask without partial blocks makes the sparse op dense attention, so it
             must match closely; otherwise dense is logged.

Small geometries gate on 0.999 PCC to the reference. Production lengths (MiniMax-H3 5 / 10 / 15 s) gate against the
dense op's own accuracy on the same inputs (_check_vs_dense_floor). The cases cover: keep-all, random densities,
partial last blocks, a logical_n tail and a shard with less than one block of live tokens, mask structures (random,
band, clustered, uneven rows), minimal masks, masks without block 0, Q blocks that keep nothing (zero output
kept-only), kept partial blocks, padding inside the sequence (the block_counts input: ragged and empty blocks at
random and VSA's ragged 4x4x4 layout), block-float inputs, frame-sparse masks (the old frame allow table), per-head
masks, determinism, new inputs on a cached program, several Q passes, pair steps (r1024k256: three blocks of one head
per core) and the kept-only mode. Runs on a single ring (1xN) or the 4x8 Galaxy (TP=4 heads x SP=8 ring); production
lengths are skipped on CI.
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


def _block_lengths(geo, S, counts=None):
    """[S, nb_pad] real tokens per block: the geometric length (64, a partial last block, the logical_n tail, or 0
    for padding), or the block_counts input `counts` ([S * nb_pad]) capped by it, as the op counts them."""
    lens = torch.zeros(S, geo.nb_pad(), dtype=torch.int64)
    for s in range(S):
        for b in range(geo.nb()):
            start = s * geo.rows + b * BLOCK
            end = min(start + BLOCK, (s + 1) * geo.rows, geo.logical_n)
            lens[s, b] = max(0, end - start)
    if counts is not None:
        lens = torch.minimum(lens, counts.reshape(S, geo.nb_pad()).to(torch.int64))
    return lens


def random_counts(geo, S, seed=0):
    """[S * nb_pad] block_counts with padding anywhere in the sequence: each block full (70 %), ragged 1-63 tokens
    (20 %), exactly 32 (5 %: trimmed to one tile, no key bias) or empty (5 %)."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(S, geo.nb_pad(), generator=g)
    ragged = torch.randint(1, BLOCK, (S, geo.nb_pad()), generator=g)
    c = torch.where(u < 0.7, BLOCK, torch.where(u < 0.9, ragged, torch.where(u < 0.95, 32, 0)))
    return torch.minimum(c, _block_lengths(geo, S)).reshape(-1)


def vsa_counts(geo, S, grid, prefix):
    """[S * nb_pad] block_counts of VSA's layout (models/tt_dit/pipelines/minimax_h3/vsa_geometry.py on
    sadesoye/vsa-ring-sparse): the prefix segments chopped into 64-token blocks (a ragged tail each), then the video
    grid (t, h, w) in 4 x 4 x 4 tiles, cube-major, one tile per block (ragged at the t / h / w edges), then padding
    blocks. None if the layout does not fit the sequence."""
    blocks = []
    for n in prefix:
        blocks += [BLOCK] * (n // BLOCK) + ([n % BLOCK] if n % BLOCK else [])
    T, Hg, W = grid
    for t0 in range(0, T, 4):
        for h0 in range(0, Hg, 4):
            for w0 in range(0, W, 4):
                blocks.append(min(4, T - t0) * min(4, Hg - h0) * min(4, W - w0))
    nb, nb_pad = geo.nb(), geo.nb_pad()
    if len(blocks) > S * nb:
        return None
    counts = torch.zeros(S * nb_pad, dtype=torch.int64)
    for n, c in enumerate(blocks):
        counts[(n // nb) * nb_pad + n % nb] = c
    return counts


def make_mask(geo, S, H, *, density, band=1, seed=0, keep_block0=True, counts=None):
    """[S, H, nb_pad, S * nb_pad] 0/1 block mask: shard s's local Q block qb keeps global block (s', b).

    Kept: block 0 of every shard (as Sol's pre-pass selects it; keep_block0=False leaves it to chance), the diagonal
    band |i - j| <= band in global block order (none for band < 0), and other blocks at random with probability
    `density`. Only full 64-token blocks are kept, as Sol's pre-pass keeps them; with `counts` (a block_counts
    tensor, or the geometric lengths) every block with a real token may be.
    """
    g = torch.Generator().manual_seed(seed)
    nb, nb_pad = geo.nb(), geo.nb_pad()
    lens = _block_lengths(geo, S, counts)  # [S, nb_pad]
    full = ((lens == BLOCK) if counts is None else (lens > 0)).reshape(-1)  # [S * nb_pad]
    mask = (torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < density).to(torch.int64)
    k_global = torch.arange(S * nb_pad)
    k_index = (k_global // nb_pad) * nb + (k_global % nb_pad)  # global block number in sequence order
    for s in range(S):
        q_index = s * nb + torch.arange(nb_pad)
        mask[s, :, (q_index[:, None] - k_index[None, :]).abs() <= band] = 1
    if keep_block0:
        mask[..., torch.arange(S) * nb_pad] = 1
    mask = mask * full.to(torch.int64)  # keepable blocks only
    mask[:, :, nb:] = 0  # padding Q rows
    return mask


MASK_STRUCTURES = ["random", "band", "clustered", "uneven"]
CLUSTER = 8  # blocks per side of a clustered tile: two 256-token pieces


def make_structured_mask(geo, S, H, *, structure, density, seed=0, keep_block0=True):
    """make_mask with kept blocks laid out by `structure`, at about `density` before the always-kept blocks:

      random     each block independently with probability `density`
      band       the diagonal band of global half-width density * N / 2 (N blocks in the sequence), every head
      clustered  CLUSTER x CLUSTER tiles of (Q, K) blocks kept whole, each with probability `density`, per head
      uneven     each Q block its own probability, exponential with mean `density` (some near empty, some dense)

    The diagonal and (keep_block0) block 0 of every shard are always kept; only full blocks may be kept.
    """
    g = torch.Generator().manual_seed(seed)
    nb, nb_pad = geo.nb(), geo.nb_pad()
    full = (_block_lengths(geo, S) == BLOCK).reshape(-1)  # [S * nb_pad]
    k_global = torch.arange(S * nb_pad)
    k_index = (k_global // nb_pad) * nb + (k_global % nb_pad)  # [S * nb_pad] global block number in sequence order
    q_index = torch.arange(S)[:, None] * nb + torch.arange(nb_pad)[None, :]  # [S, nb_pad]
    dist = q_index[:, :, None] - k_index[None, None, :]  # [S, nb_pad, S * nb_pad]
    if structure == "random":
        keep = torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < density
    elif structure == "band":
        half = round(density * S * nb / 2)
        keep = (dist.abs() <= half)[:, None].expand(S, H, nb_pad, S * nb_pad).clone()
    elif structure == "clustered":
        n_tiles = -(-S * nb_pad // CLUSTER)
        tiles = torch.rand(H, n_tiles, n_tiles, generator=g) < density
        qt = (q_index // CLUSTER).clamp(max=n_tiles - 1)  # [S, nb_pad]
        kt = (k_index // CLUSTER).clamp(max=n_tiles - 1)  # [S * nb_pad]
        keep = tiles[:, qt][..., kt].permute(1, 0, 2, 3).contiguous()  # [H, S, nb_pad, K] -> [S, H, nb_pad, K]
    elif structure == "uneven":
        p = -density * torch.rand(S, H, nb_pad, 1, generator=g).clamp(min=1e-6).log()
        keep = torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < p.clamp(max=1.0)
    else:
        raise ValueError(f"unknown mask structure {structure}")
    keep |= (dist == 0)[:, None]  # the diagonal
    if keep_block0:
        keep[..., torch.arange(S) * nb_pad] = True
    mask = keep.to(torch.int64) * full.to(torch.int64)  # full blocks only
    mask[:, :, nb:] = 0  # padding Q rows
    return mask


def window_allow(num_frames, num_frames_padded, window, add_last_frame):
    """[num_frames_padded, num_frames_padded] frame allow table of the frame-sparse ring_joint path
    (sosborne/sparse-ring-sdpa): query frame i attends frames [i - window // 2, i + window - window // 2) and, with
    add_last_frame, the last real frame. Padding frames attend and are attended by nothing."""
    hl, hr = window // 2, window - window // 2
    allow = torch.zeros(num_frames_padded, num_frames_padded, dtype=torch.int64)
    for i in range(num_frames):
        allow[i, max(0, i - hl) : min(num_frames, i + hr)] = 1
        if add_last_frame:
            allow[i, num_frames - 1] = 1
    return allow


def frame_block_mask(allow, tokens_per_frame, geo, S, H):
    """The sparse op's block mask for a frame allow table (allow[qf, kf] != 0: query frame qf attends key frame kf),
    the same for every head: Q block qb of shard s keeps global block (s', b) iff their frames are allowed. Frames
    are tokens_per_frame consecutive tokens of the whole sequence and may straddle shards; each block lies in one
    frame when tokens_per_frame and the shard rows are multiples of the block."""
    assert tokens_per_frame % BLOCK == 0 and geo.rows % BLOCK == 0, "a block must lie inside one frame"
    nb, nb_pad = geo.nb(), geo.nb_pad()
    nf = allow.shape[0]
    first_tok = torch.arange(S)[:, None] * geo.rows + torch.arange(nb_pad)[None, :] * BLOCK  # [S, nb_pad]
    frame = (first_tok // tokens_per_frame).clamp(max=nf - 1)
    live = (torch.arange(nb_pad) < nb).expand(S, nb_pad)  # padding blocks past each shard's nb
    mask = allow[frame][:, :, frame.reshape(-1)]  # [S, nb_pad, S * nb_pad]
    mask = mask * live[:, :, None] * live.reshape(-1)[None, None, :]
    return (mask != 0).to(torch.int64)[:, None].expand(S, H, nb_pad, S * nb_pad).contiguous()


def frame_reference(q, k, v, allow, tokens_per_frame, logical_n, scale, heads):
    """fp64 frame-sparse attention straight from the allow table (independent of frame_block_mask): every real
    query token attends the real tokens of its frame's allowed frames; nothing allowed gives zero. NaN for the
    heads not computed and for rows past logical_n."""
    out = torch.full(q.shape, float("nan"))
    tok_frame = torch.arange(q.shape[2]) // tokens_per_frame
    real = torch.arange(q.shape[2]) < logical_n
    for qf in range(allow.shape[0]):
        rows = torch.nonzero((tok_frame == qf) & real).flatten()
        if rows.numel() == 0:
            continue
        keys = torch.nonzero(allow[qf][tok_frame.clamp(max=allow.shape[0] - 1)].bool() & real).flatten()
        for h in heads:
            if keys.numel() == 0:
                out[0, h, rows] = 0.0
                continue
            w = torch.softmax((q[0, h, rows].double() @ k[0, h, keys].double().T) * scale, dim=1)
            out[0, h, rows] = (w @ v[0, h, keys].double()).float()
    return out


def pool(x, geo, S, counts=None):
    """[1, H, S * rows, D] -> [1, H, S * nb_pad, D] per-block mean over the real tokens (zero for padding)."""
    lens = _block_lengths(geo, S, counts)
    _, H, _, Dh = x.shape
    out = torch.zeros(1, H, S * geo.nb_pad(), Dh)
    for s in range(S):
        for b in range(geo.nb()):
            n = int(lens[s, b])
            if n:
                start = s * geo.rows + b * BLOCK
                out[0, :, s * geo.nb_pad() + b] = x[0, :, start : start + n].float().mean(1)
    return out


def sparse_reference(
    q, k, v, mask, pooled_k, pooled_v, geo, S, scale, *, pooled=True, heads=None, shards=None, counts=None
):
    """fp64 block-sparse attention: kept blocks exact over their real tokens, every other block with a real token
    one pooled entry of weight len (its real tokens: the geometric length, or `counts`, the block_counts input).

    pooled=False: kept blocks only (no pooled entries); the softmax normalizes over the kept blocks, and a row with
    nothing to attend is zero (the op's contract).
    heads / shards: compute only these (the rest of the output stays NaN), in 512-row chunks, so a slice of a
    production-size call is checked exactly in seconds; _pcc compares the computed entries only.
    """
    _, H, _, _ = q.shape
    nb, nb_pad = geo.nb(), geo.nb_pad()
    lens = _block_lengths(geo, S, counts).reshape(-1)  # [S * nb_pad]
    valid = lens > 0
    log_len = torch.where(valid, lens.clamp(min=1).double().log(), torch.zeros_like(lens, dtype=torch.float64))
    # token -> padded global block id, and which tokens are real (the first lens of each block)
    tok = torch.arange(S * geo.rows)
    tok_block = (tok // geo.rows) * nb_pad + (tok % geo.rows) // BLOCK
    tok_valid = (tok % geo.rows) % BLOCK < lens[tok_block]
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
            w = torch.softmax(torch.cat([le, lp], dim=1), dim=1).nan_to_num(0.0)  # an empty row: NaN -> 0
            out[0, h, rows] = (w[:, : le.shape[1]] @ vh + w[:, le.shape[1] :] @ pv).float()
    return out


def dense_reference(q, k, v, logical_n, scale, heads, shards, rows):
    """fp64 exact attention over the logical_n real tokens for the given heads / shards (NaN elsewhere), in 512-row
    chunks: the dense op's own accuracy floor on the same inputs."""
    out = torch.full(q.shape, float("nan"))
    for h in heads:
        kh, vh = k[0, h, :logical_n].double(), v[0, h, :logical_n].double()
        for s, r0 in ((s, r0) for s in shards for r0 in range(0, rows, 512)):
            sl = slice(s * rows + r0, s * rows + min(r0 + 512, rows))
            w = torch.softmax((q[0, h, sl].double() @ kh.T) * scale, dim=1)
            out[0, h, sl] = (w @ vh).float()
    return out


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


class Case:
    """Inputs, device buffers and both ops for one geometry on the module's mesh. dtype: of Q, K, V and the pooled
    K / V (bf16, or block float)."""

    def __init__(self, rt, geo, dtype=ttnn.bfloat16):
        self.rt, self.geo, self.dtype = rt, geo, dtype
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
                dtype=dtype,
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

    def quantize(self, x):
        """x rounded to the case's dtype on the host, so the reference sees exactly what the device reads."""
        if self.dtype == ttnn.bfloat16:
            return x.bfloat16().float()
        return ttnn.to_torch(ttnn.from_torch(x, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)).float()

    def inputs(self, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (1, self.H, self.S * self.geo.rows, D)
        return tuple(self.quantize(torch.randn(shape, generator=g)) for _ in range(3))

    def blocks_per_core(self):
        """(most, fewest) Q blocks a working core holds: the op splits the flat (head, Q block) units evenly."""
        cols, rows = self.rt.sdpa_compute_grid
        cores = cols * rows
        total = HEADS_PER_DEVICE * self.geo.nb()
        return -(-total // cores), max(1, total // cores)

    def run(
        self,
        q,
        k,
        v,
        mask,
        *,
        want_dense=True,
        want_ref=True,
        q_blocks_per_pass=None,
        pooled=True,
        dense_floor=False,
        counts=None,
    ):
        """pooled=False omits pooled K / V: kept blocks only (the VSA fine-branch mode). dense_floor=True (with
        want_ref="slice") also computes the exact dense reference on the same slice (r["dense_ref"]). counts: the
        block_counts input ([S * nb_pad] real tokens per global block), also used by the pooling and the reference."""
        rt, geo = self.rt, self.geo
        tq, tk, tv = (self._upload(x, self.seq_dims, dtype=self.dtype) for x in (q, k, v))
        pooled_k, pooled_v = (self.quantize(pool(x, geo, self.S, counts)) for x in (k, v))
        t_mask = self._upload(mask, self.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        t_pk, t_pv = (
            (self._upload(x, self.dims(None, 1), dtype=self.dtype) for x in (pooled_k, pooled_v))
            if pooled
            else (None, None)
        )
        t_counts = (
            None
            if counts is None
            else self._upload(
                counts.reshape(1, 1, 1, -1), self.dims(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        )
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
            block_counts=t_counts,
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
        dense_ref = None
        if want_ref == "full":
            sl = {}
        elif want_ref == "slice":
            sl = dict(heads=[0, self.H - 1], shards=[0, self.S - 1])
            if dense_floor:
                dense_ref = dense_reference(q, k, v, geo.logical_n, self.scale, rows=geo.rows, **sl)
        else:
            sl = dict(heads=sorted({0, 1, self.H // 2, self.H - 1}))
        ref = (
            sparse_reference(
                q, k, v, mask, pooled_k, pooled_v, geo, self.S, self.scale, pooled=pooled, counts=counts, **sl
            )
            if want_ref
            else None
        )
        n = geo.logical_n  # rows past logical_n are padding the ops leave undefined
        return {
            "sparse": sparse[:, :, :n],
            "dense": None if dense is None else dense[:, :, :n],
            "ref": None if ref is None else ref[:, :, :n],
            "dense_ref": None if dense_ref is None else dense_ref[:, :, :n],
        }


def _density(mask, geo, S, counts=None):
    """Kept fraction of the keepable blocks (full ones; with counts, every block with a real token)."""
    lens = _block_lengths(geo, S, counts).reshape(-1)
    keepable = (lens == BLOCK) if counts is None else (lens > 0)
    return float(mask[:, :, : geo.nb()][..., keepable].float().mean())


def _check(tag, r, density, *, dense_min, ref_min):
    p_dense = _pcc(r["dense"], r["sparse"]) if r["dense"] is not None else float("nan")
    p_ref = _pcc(r["ref"], r["sparse"]) if r["ref"] is not None else float("nan")
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs dense {p_dense:.5f}, vs reference {p_ref:.5f}")
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    if dense_min is not None:
        assert p_dense > dense_min, f"{tag}: pcc vs dense {p_dense:.5f}"
    if ref_min is not None:
        assert p_ref > ref_min, f"{tag}: pcc vs reference {p_ref:.5f}"


# Production lengths are gated against the dense op's accuracy on the same inputs: bf16 inputs and accumulators bound
# how close either op gets to an fp64 reference, and the bound drops with the number of keys. The sparse op may not
# lose more than DENSE_FLOOR_MARGIN of PCC to its exact (block-sparse) reference than the dense op loses to exact
# attention.
DENSE_FLOOR_MARGIN = 5e-4


def _block_pccs(ref, out):
    """PCC of every computed (head, 64-row Q block) of out against ref (blocks the reference left NaN are skipped),
    worst first: whether an error is spread evenly (accumulation) or concentrated (a head, a core's blocks, a shard)."""
    res = []
    for h in range(ref.shape[1]):
        for b0 in range(0, ref.shape[2], BLOCK):
            rb = ref[0, h, b0 : b0 + BLOCK]
            if torch.isnan(rb).all():
                continue
            res.append((_pcc(rb, out[0, h, b0 : b0 + BLOCK]), h, b0 // BLOCK))
    return sorted(res)


def _check_vs_dense_floor(tag, r, density):
    p_ref = _pcc(r["ref"], r["sparse"])
    p_floor = _pcc(r["dense_ref"], r["dense"])
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs reference {p_ref:.5f}, dense vs exact {p_floor:.5f}")
    blocks, dense_blocks = _block_pccs(r["ref"], r["sparse"]), _block_pccs(r["dense_ref"], r["dense"])
    if blocks:
        med = blocks[len(blocks) // 2][0]
        worst = ", ".join(f"h{h} qb{qb} {p:.5f}" for p, h, qb in blocks[:3])
        logger.info(
            f"[{tag}] per Q block (global block index): median {med:.5f}, worst {worst}; dense median "
            f"{dense_blocks[len(dense_blocks) // 2][0]:.5f}, worst {dense_blocks[0][0]:.5f}"
        )
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    assert p_ref > p_floor - DENSE_FLOOR_MARGIN, f"{tag}: pcc vs reference {p_ref:.5f}, dense floor {p_floor:.5f}"


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


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_sub_block_shard_tail(runtime, pooled):
    """logical_n leaves the last shard 32 live tokens: no full block, so no exact block anywhere in that shard; its
    ring iteration streams no piece (kept-only: those keys are never attended; pooled: one pooled entry)."""
    geo = Geometry(1024, 1024 * (MESH_CONFIG.sp_size - 1) + 32, 256)
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=20)
    r = case.run(*case.inputs(20), mask, want_dense=False, pooled=pooled)
    mode = "pooled" if pooled else "kept-only"
    _check(f"sub-block tail {mode}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("geom", list(RAGGED))
def test_kept_partial_blocks(runtime, geom, pooled):
    """Mask bits on a shard's partial last block and on the logical_n tail block: computed exactly over their real
    tokens (only the tiles holding them; a key bias inside the tile logical_n ends in)."""
    geo = RAGGED[geom]
    case = Case(runtime, geo)
    live = _block_lengths(geo, case.S).reshape(-1)
    mask = make_mask(geo, case.S, case.H, density=0.3, seed=22, counts=live)
    r = case.run(*case.inputs(22), mask, want_dense=False, pooled=pooled)
    mode = "pooled" if pooled else "kept-only"
    _check(f"kept partial {mode} {geom}", r, _density(mask, geo, case.S, live), dense_min=None, ref_min=0.999)


# VSA-like padding inside the sequence (the block_counts input): a small VSA layout (two prefix segments, a video grid
# in ragged 4 x 4 x 4 tiles, padding blocks after them) and ragged / empty blocks at random.
VSA_SMALL = dict(grid=(17, 12, 22), prefix=(100, 70))


def _counts_for(structure, geo, S, seed):
    if structure == "random":
        return random_counts(geo, S, seed=seed)
    counts = vsa_counts(geo, S, **VSA_SMALL)
    if counts is None:
        pytest.skip(f"the VSA layout does not fit {S} shards of {geo.rows} rows")
    return counts


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("structure", ["random", "vsa"])
@pytest.mark.parametrize("geom", ["r1024k256", "r1024k512"])
def test_block_counts(runtime, geom, structure, pooled):
    """Blocks with fewer real tokens than 64 anywhere in the sequence, and empty blocks: tokens past a block's
    count are never attended (its tiles trimmed, a key bias in the tile its count ends in), an empty block is
    never computed or pooled, and the pooled entries weigh the real tokens. The same mask without the counts
    attends the padding tokens, so it must be far from the counts reference (the counts reach the kernels)."""
    geo = ALIGNED[geom]
    case = Case(runtime, geo)
    counts = _counts_for(structure, geo, case.S, seed=21)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=21, keep_block0=False, counts=counts)
    q, k, v = case.inputs(21)
    r = case.run(q, k, v, mask, want_dense=False, pooled=pooled, counts=counts)
    mode = "pooled" if pooled else "kept-only"
    tag = f"counts {structure} {mode} {geom}"
    _check(tag, r, _density(mask, geo, case.S, counts), dense_min=None, ref_min=0.999)
    without = case.run(q, k, v, mask, want_dense=False, want_ref=False, pooled=pooled)["sparse"]
    p_without = _pcc(r["ref"], without)
    logger.info(f"[{tag}] without the counts: pcc vs reference {p_without:.5f}")
    assert p_without < 0.999, f"{tag}: the counts made no difference (pcc without them {p_without:.5f})"


def test_block_counts_passes(runtime):
    """block_counts with the Q blocks split into two passes: the counts table and the key bias outlive a pass."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo)
    most, fewest = case.blocks_per_core()
    q_blocks_per_pass = -(-most // 2)
    if -(-most // q_blocks_per_pass) != 2 or fewest < 2:
        pytest.skip(f"{most} / {fewest} blocks per core cannot be split into 2 passes")
    counts = random_counts(geo, case.S, seed=23)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=23, keep_block0=False, counts=counts)
    r = case.run(
        *case.inputs(23), mask, want_dense=False, pooled=False, counts=counts, q_blocks_per_pass=q_blocks_per_pass
    )
    _check("counts passes 2", r, _density(mask, geo, case.S, counts), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("density", [0.1, 0.35])
@pytest.mark.parametrize("structure", MASK_STRUCTURES)
def test_mask_structures(runtime, structure, density):
    """Kept blocks laid out as independent blocks, a band, whole tiles, or rows of very different density: the plan
    and the step grouping see contiguous runs, scattered blocks, empty and nearly full Q blocks."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_structured_mask(case.geo, case.S, case.H, structure=structure, density=density, seed=15)
    r = case.run(*case.inputs(15), mask, want_dense=False)
    _check(f"{structure} d{density}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


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


def _empty_rows(mask, geo, S):
    """[H, S * rows] True on the query rows whose Q block keeps no block in any shard."""
    empty = mask.sum(-1) == 0  # [S, H, nb_pad]
    return empty[:, :, torch.arange(geo.rows) // BLOCK].permute(1, 0, 2).reshape(mask.shape[1], S * geo.rows)


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("structure", ["random", "band"])
@pytest.mark.parametrize("geom", ["r1024k256", "r1024k512", "r1184k256"])
def test_without_block0(runtime, geom, structure, pooled):
    """Masks that leave block 0 of the shards to chance (VSA's selection, a windowed band): the op attends exactly
    the mask. Many (Q block, shard) pairs keep nothing, so a Q block's softmax state is created at whichever piece and
    ring iteration its first kept block arrives, next to blocks stepped together with it that already have state."""
    geo = ALIGNED.get(geom) or RAGGED[geom]
    case = Case(runtime, geo)
    if structure == "random":
        mask = make_mask(geo, case.S, case.H, density=0.05, band=-1, seed=18, keep_block0=False)
    else:
        mask = make_structured_mask(geo, case.S, case.H, structure="band", density=0.1, seed=18, keep_block0=False)
    assert not mask[:, :, : geo.nb(), torch.arange(case.S) * geo.nb_pad()].all(), "every row keeps block 0"
    r = case.run(*case.inputs(18), mask, want_dense=False, pooled=pooled)
    mode = "pooled" if pooled else "kept-only"
    _check(f"no block 0 {structure} {mode} {geom}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_empty_q_blocks(runtime, pooled, passes):
    """Q blocks that keep no block anywhere (VSA's padding blocks): a whole head, every third Q block of the first
    shard, and the last head on the last shard. Kept-only, nothing enters their softmax and the op writes zeros
    (finite: the rows flow into later layers); with the pooled route they attend the pooled entries only."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo)
    q_blocks_per_pass = None
    if passes > 1:
        most, fewest = case.blocks_per_core()
        q_blocks_per_pass = -(-most // passes)
        if -(-most // q_blocks_per_pass) != passes or passes > fewest:
            pytest.skip(f"{most} / {fewest} blocks per core cannot be split into {passes} passes")
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=19, keep_block0=False)
    mask[:, 1] = 0
    mask[0, :, ::3] = 0
    mask[case.S - 1, case.H - 1] = 0
    r = case.run(*case.inputs(19), mask, want_dense=False, pooled=pooled, q_blocks_per_pass=q_blocks_per_pass)
    mode = "pooled" if pooled else "kept-only"
    _check(f"empty q blocks {mode} passes {passes}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    if not pooled:
        empty = _empty_rows(mask, geo, case.S)[:, : geo.logical_n]
        out = r["sparse"][0]
        assert (out[empty] == 0).all(), f"empty Q blocks: max |out| {out[empty].abs().max().item():.3e}"


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf8", "bf4"])
def test_block_float_inputs(runtime, dtype, pooled):
    """Block-float Q, K, V and pooled K / V: the K / V CBs take the inputs' tile format and size, and the pooled
    entries are read into them in the same format. The reference sees the same rounded values."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo, dtype=dtype)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=25)
    r = case.run(*case.inputs(25), mask, want_dense=False, pooled=pooled)
    mode = "pooled" if pooled else "kept-only"
    _check(f"{dtype} {mode}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


# Frame-sparse attention (the frame allow table of the closed frame-sparse ring_joint path, sosborne/sparse-ring-sdpa),
# expressed as a block mask (frame_block_mask), kept blocks only. Frames of 2048 tokens straddle shards (two shards
# per frame), 256-token frames give 32 frames; one padding frame each, as the frame counts were padded to the ring.
@pytest.mark.parametrize("allow_all", [False, True], ids=["window", "allow_all"])
@pytest.mark.parametrize("tokens_per_frame", [256, 512, 2048])
def test_frame_mask(runtime, tokens_per_frame, allow_all):
    """A windowed frame mask (window 5 and the last frame, as the old tests) or every real frame allowed, against
    an fp64 frame-level reference built from the allow table alone (checks the conversion too)."""
    S = MESH_CONFIG.sp_size
    rows = 1024
    if (S * rows) % tokens_per_frame:
        pytest.skip(f"{S} shards of {rows} rows are not whole {tokens_per_frame}-token frames")
    nf_pad = S * rows // tokens_per_frame
    nf = nf_pad - 1
    geo = Geometry(rows, nf * tokens_per_frame, 256)
    case = Case(runtime, geo)
    if allow_all:
        allow = torch.zeros(nf_pad, nf_pad, dtype=torch.int64)
        allow[:nf, :nf] = 1
    else:
        allow = window_allow(nf, nf_pad, window=5, add_last_frame=True)
    mask = frame_block_mask(allow, tokens_per_frame, geo, case.S, case.H)
    q, k, v = case.inputs(26)
    r = case.run(q, k, v, mask, want_dense=allow_all, pooled=False)
    heads = sorted({0, 1, case.H // 2, case.H - 1})
    ref = frame_reference(q, k, v, allow, tokens_per_frame, geo.logical_n, case.scale, heads)[:, :, : geo.logical_n]
    tag = f"frames {tokens_per_frame} {'allow-all' if allow_all else 'window'}"
    p_frames = _pcc(ref, r["sparse"])
    logger.info(f"[{tag}] {nf} of {nf_pad} frames real: pcc vs frame reference {p_frames:.5f}")
    assert _pcc(ref, r["ref"]) > 0.99999, f"{tag}: the block mask does not express the frame table"
    _check(tag, r, _density(mask, geo, case.S), dense_min=0.9995 if allow_all else None, ref_min=0.999)
    assert p_frames > 0.999, f"{tag}: pcc vs frame reference {p_frames:.5f}"


@SLOW
@pytest.mark.parametrize("tokens_per_frame", [3840, 1920], ids=["720p", "480p"])
def test_frame_mask_video(runtime, tokens_per_frame):
    """The old frame-sparse video shapes: 21 frames padded to 24, window 5 and the last frame. Gated against the
    dense op's accuracy floor on the same inputs."""
    S = MESH_CONFIG.sp_size
    nf, nf_pad = 21, -(-21 // S) * S
    if (nf_pad * tokens_per_frame) % (S * BLOCK):
        pytest.skip(f"{nf_pad} frames of {tokens_per_frame} tokens do not split into {S} shards of whole blocks")
    rows = nf_pad * tokens_per_frame // S
    geo = Geometry(rows, nf * tokens_per_frame, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    allow = window_allow(nf, nf_pad, window=5, add_last_frame=True)
    mask = frame_block_mask(allow, tokens_per_frame, geo, case.S, case.H)
    r = case.run(*case.inputs(27), mask, want_ref="slice", dense_floor=True, pooled=False)
    _check_vs_dense_floor(f"frames video {tokens_per_frame}", r, _density(mask, geo, case.S))


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
    """MiniMax-H3 at 5 s: 4736 rows per device (one pass, pair steps), the dense op at its measured chunks (320, 384).
    Gated on a sliced fp64 reference (first / last head of the first / last shard) against the dense op's accuracy
    floor on the same inputs."""
    geo = Geometry(4736, 4736 * MESH_CONFIG.sp_size, k_chunk, dense_chunks=(320, 384))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=6)
    r = case.run(*case.inputs(6), mask, want_ref="slice", dense_floor=True)
    _check_vs_dense_floor(f"5s k{k_chunk} d{density}", r, _density(mask, geo, case.S))


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


@SLOW
@pytest.mark.parametrize("density", [KEEP_ALL, 0.15])
def test_production_10s(runtime, density):
    """MiniMax-H3 at 10 s: 9184 rows per device, more Q blocks per core than fit L1 at once, so the op picks two
    passes with pair steps (logged by the program factory). Gated against the dense op's accuracy floor (each shard's
    32-token last block is pooled even with every full block kept)."""
    geo = Geometry(9184, 9184 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    most, _ = case.blocks_per_core()
    logger.info(f"[10s] {most} Q blocks per core at most")
    mask = make_mask(geo, case.S, case.H, density=density, seed=8)
    r = case.run(*case.inputs(8), mask, want_ref="slice", dense_floor=True)
    _check_vs_dense_floor(f"10s d{density}", r, _density(mask, geo, case.S))


@SLOW
@pytest.mark.parametrize("density", [0.17])
def test_production_15s(runtime, density):
    """MiniMax-H3 at 15 s: 13664 rows per device, two passes of 14 Q blocks per core and no pair steps: the
    configuration with the most resident blocks per core (the compute threads' stack sizing case). Gated against the
    dense op's accuracy floor."""
    geo = Geometry(13664, 13664 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=16)
    r = case.run(*case.inputs(16), mask, want_ref="slice", dense_floor=True)
    _check_vs_dense_floor(f"15s d{density}", r, _density(mask, geo, case.S))


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


@SLOW
@pytest.mark.parametrize("density", [0.1])
def test_production_kept_only_10s(runtime, density):
    """No pooled route (VSA's fine branch) at MiniMax-H3 10 s: kept blocks only, two passes. Gated on the sliced
    kept-only reference against the dense op's accuracy floor."""
    geo = Geometry(9184, 9184 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=17)
    r = case.run(*case.inputs(17), mask, want_ref="slice", dense_floor=True, pooled=False)
    _check_vs_dense_floor(f"kept-only 10s d{density}", r, _density(mask, geo, case.S))


@SLOW
@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_production_vsa_15s(runtime, pooled):
    """VSA's 15 s 768p layout (sadesoye/vsa-ring-sparse: video grid 107 x 24 x 42, prefix segments of 512, 1008 and
    1200 tokens): 1,825 blocks with ragged tiles at every w edge and the t edge, padded to 14656 rows per device.
    Kept blocks at density 0.1. Gated against the dense op's accuracy floor on the same inputs."""
    geo = Geometry(14656, 14656 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    counts = vsa_counts(geo, case.S, grid=(107, 24, 42), prefix=(512, 1008, 1200))
    if counts is None:
        pytest.skip(f"the VSA layout does not fit {case.S} shards of {geo.rows} rows")
    mask = make_mask(geo, case.S, case.H, density=0.1, seed=24, keep_block0=False, counts=counts)
    r = case.run(*case.inputs(24), mask, want_ref="slice", dense_floor=True, pooled=pooled, counts=counts)
    mode = "pooled" if pooled else "kept-only"
    _check_vs_dense_floor(f"vsa 15s {mode}", r, _density(mask, geo, case.S, counts))
