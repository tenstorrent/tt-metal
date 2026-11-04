# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Functional correctness of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

Masks, block counts and pooled K / V are built in torch. The output is checked against an fp64 block-sparse reference
with the same mask (kept blocks exact, other blocks one pooled entry, or none in kept-only mode), against the dense
ring op where a keep-all mask makes the two equal, and at production lengths against the dense op's own accuracy.
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

BLOCK = 64  # tokens per (square) mask block, passed as block_size; also the q chunk
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
    """[S * nb_pad] block_counts of VSA's layout: the prefix segments in 64-token blocks (a ragged tail each), then
    the video grid (t, h, w) in 4 x 4 x 4 tiles, cube-major, one tile per block, then padding blocks. None if the
    layout does not fit the sequence."""
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

    Kept: block 0 of every shard (unless keep_block0=False), the band |i - j| <= band in global block order (none for
    band < 0), and other blocks with probability `density`. Only full blocks are kept; with `counts` (block_counts,
    or the geometric lengths) any block with a real token may be.
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
    """[num_frames_padded, num_frames_padded] frame allow table: query frame i attends frames
    [i - window // 2, i + window - window // 2) and, with add_last_frame, the last real frame. Padding frames attend
    and are attended by nothing."""
    hl, hr = window // 2, window - window // 2
    allow = torch.zeros(num_frames_padded, num_frames_padded, dtype=torch.int64)
    for i in range(num_frames):
        allow[i, max(0, i - hl) : min(num_frames, i + hr)] = 1
        if add_last_frame:
            allow[i, num_frames - 1] = 1
    return allow


def frame_block_mask(allow, tokens_per_frame, geo, S, H):
    """Block mask (every head) for a frame allow table: a Q block keeps a K block iff allow[its frame, their frame].
    Frames are tokens_per_frame consecutive tokens and may straddle shards; a block must lie in one frame."""
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


def round_significand(x, bits):
    """x rounded to `bits` significant bits, to nearest even, on its fp32 pattern: the LoFi recipe's input rounding
    (ttnn.transformer.prepare_sdpa_input), so LoFi's narrower matmul operands lose nothing further."""
    shift = 24 - bits
    raw = x.float().contiguous().view(torch.int32)
    raw = (raw + ((1 << (shift - 1)) - 1) + ((raw >> shift) & 1)) & ~((1 << shift) - 1)
    return raw.view(torch.float32)


class Case:
    """Inputs, device buffers and both ops for one geometry on the module's mesh. dtype: of K, V and the pooled K / V
    (bf16, or block float), and of Q unless lofi. lofi: LoFi matmuls, on Q rounded to 7 significant bits (bf16) and
    K / V to 5."""

    def __init__(self, rt, geo, dtype=ttnn.bfloat16, lofi=False):
        self.rt, self.geo, self.dtype, self.lofi = rt, geo, dtype, lofi
        self.q_dtype = ttnn.bfloat16 if lofi else dtype
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
            compute_with_storage_grid_size=grid,
            q_chunk_size=BLOCK,
            k_chunk_size=geo.k_chunk,
            exp_approx_mode=False,
            matmul_math_fidelity=ttnn.MathFidelity.LoFi if lofi else None,
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

    def quantize(self, x, dtype=None, bits=None):
        """x as the device reads it: rounded to `bits` significant bits (LoFi inputs), then stored in `dtype` (default
        the K / V dtype), so the reference sees exactly the device's values."""
        x = x if bits is None else round_significand(x, bits)
        dtype = self.dtype if dtype is None else dtype
        if dtype == ttnn.bfloat16:
            return x.bfloat16().float()
        return ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()

    def kv_bits(self):
        return 5 if self.lofi else None

    def prepare(self, q, k, v):
        """Q, K, V as the op's inputs (see quantize)."""
        q = self.quantize(q, self.q_dtype, 7 if self.lofi else None)
        return (q, *(self.quantize(x, bits=self.kv_bits()) for x in (k, v)))

    def raw_inputs(self, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (1, self.H, self.S * self.geo.rows, D)
        return tuple(torch.randn(shape, generator=g) for _ in range(3))

    def inputs(self, seed):
        return self.prepare(*self.raw_inputs(seed))

    def blocks_per_core(self):
        """(most, fewest) Q blocks a working core holds: the op splits the flat (head, Q block) units evenly."""
        cols, rows = self.rt.sdpa_compute_grid
        cores = cols * rows
        total = HEADS_PER_DEVICE * self.geo.nb()
        return -(-total // cores), max(1, total // cores)

    def q_blocks_per_pass_for(self, passes):
        """q_blocks_per_pass that splits every core's Q blocks into `passes` passes (None for one); skips if none."""
        if passes == 1:
            return None
        most, fewest = self.blocks_per_core()
        q_blocks_per_pass = -(-most // passes)
        if -(-most // q_blocks_per_pass) != passes or passes > fewest:
            pytest.skip(f"{most} / {fewest} blocks per core cannot be split into {passes} passes")
        return q_blocks_per_pass

    def call(self, q, k, v, mask, pooled_k=None, pooled_v=None, **overrides):
        """The sparse op on device tensors with this case's arguments; `overrides` replace or add op kwargs."""
        rt = self.rt
        kwargs = dict(
            persistent_output_buffer_k=self.buf_k,
            persistent_output_buffer_v=self.buf_v,
            logical_n=self.geo.logical_n,
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
        )
        return ttnn.transformer.sparse_ring_sdpa(q, k, v, mask, pooled_k, pooled_v, **{**kwargs, **overrides})

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
        """pooled=False omits pooled K / V (kept blocks only). dense_floor=True (with want_ref="slice") also computes
        the exact dense reference on the slice. counts: the block_counts input ([S * nb_pad]), also used by the
        pooling and the reference."""
        rt, geo = self.rt, self.geo
        tq = self._upload(q, self.seq_dims, dtype=self.q_dtype)
        tk, tv = (self._upload(x, self.seq_dims, dtype=self.dtype) for x in (k, v))
        pooled_k, pooled_v = (self.quantize(pool(x, geo, self.S, counts), bits=self.kv_bits()) for x in (k, v))
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
        out = self.call(tq, tk, tv, t_mask, t_pk, t_pv, q_blocks_per_pass=q_blocks_per_pass, block_counts=t_counts)
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
        # Reference coverage: True: heads first, second, middle, last on every shard; "slice": first and last head of
        # the first and last shard (production sizes); "full": everything.
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


def _mode(pooled):
    return "pooled" if pooled else "kept-only"


def _check(tag, r, density, *, dense_min, ref_min):
    p_dense = _pcc(r["dense"], r["sparse"]) if r["dense"] is not None else float("nan")
    p_ref = _pcc(r["ref"], r["sparse"]) if r["ref"] is not None else float("nan")
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs dense {p_dense:.5f}, vs reference {p_ref:.5f}")
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    if dense_min is not None:
        assert p_dense > dense_min, f"{tag}: pcc vs dense {p_dense:.5f}"
    if ref_min is not None:
        assert p_ref > ref_min, f"{tag}: pcc vs reference {p_ref:.5f}"


# Production lengths: the sparse op may lose at most DENSE_FLOOR_MARGIN more PCC to its fp64 reference than the dense
# op loses to exact attention on the same inputs (bf16 bounds both, more so with more keys).
DENSE_FLOOR_MARGIN = 5e-4


def _block_pccs(ref, out):
    """(PCC, head, Q block) of every Q block the reference computed, worst first."""
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
GEOMETRIES = {**ALIGNED, **RAGGED}


@pytest.mark.parametrize("geom", list(ALIGNED))
def test_keep_all_matches_dense(runtime, geom):
    """Every full block kept and no partial blocks: the sparse op is dense attention."""
    case = Case(runtime, ALIGNED[geom])
    mask = make_mask(case.geo, case.S, case.H, density=KEEP_ALL)
    r = case.run(*case.inputs(0), mask, want_ref="full" if geom == "r512k512" else True)
    _check(f"keepall {geom}", r, _density(mask, case.geo, case.S), dense_min=0.9995, ref_min=0.9995)


@pytest.mark.parametrize("density", [0.5, 0.2, 0.05])
@pytest.mark.parametrize("geom", list(ALIGNED))
def test_random_density(runtime, geom, density):
    """Random kept blocks over the band, against the reference. Dense is logged only (that gap depends on the data)."""
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
    # With every full block kept, only the partial blocks are pooled: close to dense.
    dense_min = 0.98 if density == KEEP_ALL else None
    _check(f"ragged d{density} {geom}", r, _density(mask, case.geo, case.S), dense_min=dense_min, ref_min=0.999)


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_sub_block_shard_tail(runtime, pooled):
    """The last shard holds 32 live tokens (no full block): its ring iteration streams no piece."""
    geo = Geometry(1024, 1024 * (MESH_CONFIG.sp_size - 1) + 32, 256)
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=20)
    r = case.run(*case.inputs(20), mask, want_dense=False, pooled=pooled)
    _check(f"sub-block tail {_mode(pooled)}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("geom", list(RAGGED))
def test_kept_partial_blocks(runtime, geom, pooled):
    """Kept partial last blocks and logical_n tail blocks: computed exactly over their real tokens."""
    geo = RAGGED[geom]
    case = Case(runtime, geo)
    live = _block_lengths(geo, case.S).reshape(-1)
    mask = make_mask(geo, case.S, case.H, density=0.3, seed=22, counts=live)
    r = case.run(*case.inputs(22), mask, want_dense=False, pooled=pooled)
    _check(f"kept partial {_mode(pooled)} {geom}", r, _density(mask, geo, case.S, live), dense_min=None, ref_min=0.999)


# Padding inside the sequence (block_counts): a small VSA layout, or ragged / empty blocks at random.
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
    """Ragged and empty blocks anywhere: only real tokens are attended or pooled. Without the counts the output must
    differ (the counts reach the kernels)."""
    geo = ALIGNED[geom]
    case = Case(runtime, geo)
    counts = _counts_for(structure, geo, case.S, seed=21)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=21, keep_block0=False, counts=counts)
    q, k, v = case.inputs(21)
    r = case.run(q, k, v, mask, want_dense=False, pooled=pooled, counts=counts)
    tag = f"counts {structure} {_mode(pooled)} {geom}"
    _check(tag, r, _density(mask, geo, case.S, counts), dense_min=None, ref_min=0.999)
    without = case.run(q, k, v, mask, want_dense=False, want_ref=False, pooled=pooled)["sparse"]
    p_without = _pcc(r["ref"], without)
    logger.info(f"[{tag}] without the counts: pcc vs reference {p_without:.5f}")
    assert p_without < 0.999, f"{tag}: the counts made no difference (pcc without them {p_without:.5f})"


def test_block_counts_passes(runtime):
    """block_counts with two Q passes: the counts table and the key bias outlive a pass."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo)
    q_blocks_per_pass = case.q_blocks_per_pass_for(2)
    counts = random_counts(geo, case.S, seed=23)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=23, keep_block0=False, counts=counts)
    r = case.run(
        *case.inputs(23), mask, want_dense=False, pooled=False, counts=counts, q_blocks_per_pass=q_blocks_per_pass
    )
    _check("counts passes 2", r, _density(mask, geo, case.S, counts), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("density", [0.1, 0.35])
@pytest.mark.parametrize("structure", MASK_STRUCTURES)
def test_mask_structures(runtime, structure, density):
    """Kept blocks as independent blocks, a band, whole tiles, or rows of very different density."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_structured_mask(case.geo, case.S, case.H, structure=structure, density=density, seed=15)
    r = case.run(*case.inputs(15), mask, want_dense=False)
    _check(f"{structure} d{density}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


def test_minimal_mask(runtime):
    """Only block 0 of every shard and the diagonal: the pooled entries carry almost all of the softmax mass."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_mask(case.geo, case.S, case.H, density=0.0, band=0, seed=3)
    r = case.run(*case.inputs(3), mask)
    _check("minimal", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


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
    """Block 0 left to chance: a Q block's softmax state starts at whichever piece its first kept block is in."""
    geo = GEOMETRIES[geom]
    case = Case(runtime, geo)
    if structure == "random":
        mask = make_mask(geo, case.S, case.H, density=0.05, band=-1, seed=18, keep_block0=False)
    else:
        mask = make_structured_mask(geo, case.S, case.H, structure="band", density=0.1, seed=18, keep_block0=False)
    assert not mask[:, :, : geo.nb(), torch.arange(case.S) * geo.nb_pad()].all(), "every row keeps block 0"
    r = case.run(*case.inputs(18), mask, want_dense=False, pooled=pooled)
    tag = f"no block 0 {structure} {_mode(pooled)} {geom}"
    _check(tag, r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_empty_q_blocks(runtime, pooled, passes):
    """Q blocks that keep nothing: zeros kept-only, the pooled entries only with the pooled route."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo)
    q_blocks_per_pass = case.q_blocks_per_pass_for(passes)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=19, keep_block0=False)
    mask[:, 1] = 0
    mask[0, :, ::3] = 0
    mask[case.S - 1, case.H - 1] = 0
    r = case.run(*case.inputs(19), mask, want_dense=False, pooled=pooled, q_blocks_per_pass=q_blocks_per_pass)
    tag = f"empty q blocks {_mode(pooled)} passes {passes}"
    _check(tag, r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    if not pooled:
        empty = _empty_rows(mask, geo, case.S)[:, : geo.logical_n]
        out = r["sparse"][0]
        assert (out[empty] == 0).all(), f"empty Q blocks: max |out| {out[empty].abs().max().item():.3e}"


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf8", "bf4"])
def test_block_float_inputs(runtime, dtype, pooled):
    """Block-float Q, K, V and pooled K / V; the reference sees the same rounded values."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo, dtype=dtype)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=25)
    r = case.run(*case.inputs(25), mask, want_dense=False, pooled=pooled)
    _check(f"{dtype} {_mode(pooled)}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


# LoFi matmuls on inputs rounded as the LoFi recipe rounds them (Case(lofi=True)): exact against the reference on the
# rounded inputs (LoFi drops no bits of pre-rounded operands), and within the recipe's relative L2 bounds of
# attention on the unrounded inputs (tech_reports/FlashAttention/SDPAPrecisionRecipes.md, FAST).
LOFI_REL_L2_MAX = {ttnn.bfloat16: 0.042, ttnn.bfloat8_b: 0.044}


def _rel_l2(ref, out):
    keep = ~torch.isnan(ref)
    return ((out[keep].double() - ref[keep].double()).norm() / ref[keep].double().norm()).item()


def _check_lofi(tag, case, raw, mask, r, pooled, **ref_slice):
    """The recipe's bound against fp64 attention on the unrounded inputs `raw` (same reference slice as r)."""
    geo = case.geo
    pk, pv = (pool(x, geo, case.S) for x in raw[1:])
    exact = sparse_reference(*raw, mask, pk, pv, geo, case.S, case.scale, pooled=pooled, **ref_slice)
    err = _rel_l2(exact[:, :, : geo.logical_n], r["sparse"])
    logger.info(f"[{tag}] relative L2 vs the unrounded inputs {err:.4f}")
    assert err < LOFI_REL_L2_MAX[case.dtype], f"{tag}: relative L2 {err:.4f}"


@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
@pytest.mark.parametrize("dtype", list(LOFI_REL_L2_MAX), ids=["bf16", "bf8"])
@pytest.mark.parametrize("geom", ["r1024k256", "r1184k512"])
def test_lofi(runtime, geom, dtype, pooled):
    """LoFi with K / V in `dtype` and bf16 Q."""
    geo = GEOMETRIES[geom]
    case = Case(runtime, geo, dtype=dtype, lofi=True)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=28)
    raw = case.raw_inputs(28)
    r = case.run(*case.prepare(*raw), mask, want_dense=False, pooled=pooled)
    tag = f"lofi {dtype} {_mode(pooled)} {geom}"
    _check(tag, r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    _check_lofi(tag, case, raw, mask, r, pooled, heads=sorted({0, 1, case.H // 2, case.H - 1}))


# Frame-sparse attention as a block mask (frame_block_mask), kept blocks only, one padding frame. 2048-token frames
# straddle shards.
@pytest.mark.parametrize("allow_all", [False, True], ids=["window", "allow_all"])
@pytest.mark.parametrize("tokens_per_frame", [256, 512, 2048])
def test_frame_mask(runtime, tokens_per_frame, allow_all):
    """Window 5 plus the last frame, or every real frame, against a frame-level fp64 reference from the allow table."""
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
    """Video frame shapes: 21 frames padded to 24, window 5 and the last frame; gated on the dense accuracy floor."""
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
    """Same inputs twice give the same bits."""
    case = Case(runtime, ALIGNED["r1024k256"])
    mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=5)
    q, k, v = case.inputs(5)
    a = case.run(q, k, v, mask, want_dense=False, want_ref=False)["sparse"]
    b = case.run(q, k, v, mask, want_dense=False, want_ref=False)["sparse"]
    assert torch.equal(a, b), f"max |diff| {(a - b).abs().max().item():.3e}"


@pytest.mark.parametrize("geom", ["r1024k256", "r1024k512"])
def test_fresh_inputs_per_call(runtime, geom):
    """Four calls on the cached program, new inputs and masks each: catches stale K / V from a previous call."""
    case = Case(runtime, ALIGNED[geom])
    for seed in (11, 12, 13, 14):
        mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=seed)
        r = case.run(*case.inputs(seed), mask, want_dense=False)
        _check(f"call {seed} {geom}", r, _density(mask, case.geo, case.S), dense_min=None, ref_min=0.999)


@SLOW
@pytest.mark.parametrize("density", [KEEP_ALL, 0.15])
@pytest.mark.parametrize("k_chunk", [256, 384])
def test_production_5s(runtime, k_chunk, density):
    """MiniMax-H3 at 5 s (4736 rows per device), gated on the dense accuracy floor."""
    geo = Geometry(4736, 4736 * MESH_CONFIG.sp_size, k_chunk, dense_chunks=(320, 384))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=6)
    r = case.run(*case.inputs(6), mask, want_ref="slice", dense_floor=True)
    _check_vs_dense_floor(f"5s k{k_chunk} d{density}", r, _density(mask, geo, case.S))


@pytest.mark.parametrize("passes", [2, 3])
@pytest.mark.parametrize("geom", ["r1024k256", "r1184k256"])
def test_forced_passes(runtime, geom, passes):
    """Several Q passes match one pass (not bit-identical: lazy rescaling depends on which blocks step together)."""
    geo = GEOMETRIES[geom]
    case = Case(runtime, geo)
    q_blocks_per_pass = case.q_blocks_per_pass_for(passes)
    most, _ = case.blocks_per_core()
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
    """MiniMax-H3 at 10 s (9184 rows per device, two passes), gated on the dense accuracy floor."""
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
    """MiniMax-H3 at 15 s (13664 rows per device, the most resident blocks per core), gated on the dense floor."""
    geo = Geometry(13664, 13664 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=16)
    r = case.run(*case.inputs(16), mask, want_ref="slice", dense_floor=True)
    _check_vs_dense_floor(f"15s d{density}", r, _density(mask, geo, case.S))


@SLOW
@pytest.mark.parametrize("pooled", [True, False], ids=["pooled", "kept_only"])
@pytest.mark.parametrize("rows, density", [(4736, 0.15), (9184, 0.15), (13664, 0.17)], ids=["5s", "10s", "15s"])
def test_production_lofi(runtime, rows, density, pooled):
    """LoFi with bf8 K / V and bf16 Q at the MiniMax-H3 lengths, on the sliced references."""
    geo = Geometry(rows, rows * MESH_CONFIG.sp_size, 256)
    case = Case(runtime, geo, dtype=ttnn.bfloat8_b, lofi=True)
    mask = make_mask(geo, case.S, case.H, density=density, seed=29)
    raw = case.raw_inputs(29)
    r = case.run(*case.prepare(*raw), mask, want_dense=False, want_ref="slice", pooled=pooled)
    tag = f"lofi bf8 {_mode(pooled)} {rows} rows d{density}"
    _check(tag, r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    _check_lofi(tag, case, raw, mask, r, pooled, heads=[0, case.H - 1], shards=[0, case.S - 1])


# Kept blocks only (no pooled K / V): only keep-all matches dense.


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
    geo = GEOMETRIES[geom]
    case = Case(runtime, geo)
    q_blocks_per_pass = case.q_blocks_per_pass_for(2)
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
    """Kept blocks only at MiniMax-H3 10 s, gated on the dense accuracy floor."""
    geo = Geometry(9184, 9184 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=17)
    r = case.run(*case.inputs(17), mask, want_ref="slice", dense_floor=True, pooled=False)
    _check_vs_dense_floor(f"kept-only 10s d{density}", r, _density(mask, geo, case.S))


@SLOW
@pytest.mark.parametrize("pooled", [False, True], ids=["kept_only", "pooled"])
def test_production_vsa_15s(runtime, pooled):
    """VSA's 15 s 768p layout (grid 107 x 24 x 42, prefixes 512 / 1008 / 1200, 14656 rows per device), density 0.1,
    gated on the dense accuracy floor."""
    geo = Geometry(14656, 14656 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
    counts = vsa_counts(geo, case.S, grid=(107, 24, 42), prefix=(512, 1008, 1200))
    if counts is None:
        pytest.skip(f"the VSA layout does not fit {case.S} shards of {geo.rows} rows")
    mask = make_mask(geo, case.S, case.H, density=0.1, seed=24, keep_block0=False, counts=counts)
    r = case.run(*case.inputs(24), mask, want_ref="slice", dense_floor=True, pooled=pooled, counts=counts)
    _check_vs_dense_floor(f"vsa 15s {_mode(pooled)}", r, _density(mask, geo, case.S, counts))


# One invalid argument per call: the op rejects it before running (TT_FATAL raises RuntimeError).
INVALID = {  # id: (the one invalid argument, the TT_FATAL text it must raise)
    "block_size_32": (dict(block_size=32), "block_size must be"),
    "pooled_k_only": (dict(pooled="k_only"), "pooled_k and pooled_v must be passed together"),
    "pooled_bf8": (dict(pooled=ttnn.bfloat8_b), "pooled_k / pooled_v must be tile in the dtype of k / v"),
    "q_bf8_kv_bf16": (dict(q_dtype=ttnn.bfloat8_b), "q must be bf16 or the dtype of k / v"),
    "mask_int32": (dict(mask_dtype=(ttnn.int32, torch.int32)), "block_mask must be uint32 row-major"),
    "mask_bf16": (dict(mask_dtype=(ttnn.bfloat16, torch.bfloat16)), "block_mask must be uint32 row-major"),
    "mask_width": (dict(mask_extra=16), "block_mask width"),
    "counts_shape": (dict(counts_rows=2), "block_counts must be"),
    "gqa": (dict(kv_heads=HEADS_PER_DEVICE * MESH_CONFIG.tp_size // 2), "supports MHA only"),
    "scale_0": (dict(scale=0.0), "scale must be finite and positive"),
    "scale_neg": (dict(scale=-1.0), "scale must be finite and positive"),
    "q_blocks_per_pass_0": (dict(q_blocks_per_pass=0), "q_blocks_per_pass must be in"),
    "q_blocks_per_pass_33": (dict(q_blocks_per_pass=33), "q_blocks_per_pass must be in"),
    "batch_2": (dict(batch=2), "batch 1 only"),
}


@pytest.mark.parametrize("bad", list(INVALID))
def test_invalid_args(runtime, expect_error, bad):
    """The op rejects one invalid argument (on an otherwise valid kept-only call) with its own check's error."""
    spec, message = INVALID[bad]
    spec = dict(spec)
    case = Case(runtime, ALIGNED["r512k512"])
    geo, S, H = case.geo, case.S, case.H
    batch, kv_heads, n = spec.pop("batch", 1), spec.pop("kv_heads", H), S * geo.rows
    q = case._upload(torch.zeros(batch, H, n, D), case.seq_dims, dtype=spec.pop("q_dtype", ttnn.bfloat16))
    k, v = (case._upload(torch.zeros(batch, kv_heads, n, D), case.seq_dims) for _ in range(2))
    bufs = {
        f"persistent_output_buffer_{x}": case._upload(torch.zeros(batch, kv_heads, n, D), case.dims(None, 1))
        for x in "kv"
    }
    mask = torch.nn.functional.pad(make_mask(geo, S, H, density=KEEP_ALL), (0, spec.pop("mask_extra", 0)))
    t_dtype, torch_dtype = spec.pop("mask_dtype", (ttnn.uint32, mask.dtype))
    t_mask = case._upload(mask.to(torch_dtype), case.dims(0, 1), dtype=t_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    pooled = spec.pop("pooled", None)
    pk = pv = None
    if pooled is not None:
        dtype = ttnn.bfloat16 if pooled == "k_only" else pooled
        pk = case._upload(torch.zeros(1, H, S * geo.nb_pad(), D), case.dims(None, 1), dtype=dtype)
        pv = None if pooled == "k_only" else pk
    if "counts_rows" in spec:
        counts = torch.full((1, 1, spec.pop("counts_rows"), S * geo.nb_pad()), BLOCK, dtype=torch.int32)
        spec["block_counts"] = case._upload(
            counts, case.dims(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    with expect_error(RuntimeError, message):
        case.call(q, k, v, t_mask, pk, pv, **bufs, **spec)
