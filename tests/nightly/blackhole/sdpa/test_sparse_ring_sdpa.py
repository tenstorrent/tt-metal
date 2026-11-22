# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Functional correctness of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

The block mask and the pooled K / V are built in torch, so the cases control density and structure exactly. Each case
runs the sparse op (and, where it is compared, the dense `ring_joint_scaled_dot_product_attention`) on the same
inputs and checks the sparse output against:

  reference  fp64 torch block-sparse attention with the same mask: kept 64-token blocks exact, every other valid
             block one pooled entry (mean K, mean V) weighted by its token count. This is what the op computes, so it
             separates kernel errors from the approximation.
  dense      the dense ring op. A keep-all mask without partial blocks makes the sparse op dense attention, so it
             must match closely; otherwise dense is logged.

Small geometries gate on 0.999 PCC to the reference. Production lengths (MiniMax-H3 5 / 10 / 15 s) gate against the
dense op's own accuracy on the same inputs (_check_vs_dense_floor). The cases cover: keep-all, random densities,
ragged shards with partial last blocks, Q blocks that keep nothing, BFP8 inputs, determinism, new inputs on a cached
program, several Q passes, the runtime logical length (logical_n_tensor: one program, and one trace, for every
length of a padded shape), and the block residual (residual_bias: one residual per Q block from its mean query over
the blocks it leaves out, merged by log-sum-exp, against an fp64 reference of that algorithm). Runs on a single ring (1xN) or the 4x8 Galaxy (TP=4 heads x SP=8 ring); production
lengths are skipped on CI.
"""

import math
import os
from dataclasses import dataclass, replace

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

BLOCK = 64
HEADS_PER_DEVICE = 14
D = 128
KEEP_ALL = 1.0
SLOW = pytest.mark.skipif(os.environ.get("CI") == "true", reason="production-size shapes - skip on CI")
# One captured sparse op (test_runtime_len_trace_replay) with ample headroom.
TRACE_REGION_SIZE = 8 * 1024 * 1024


def _pad32(n):
    return -(-n // 32) * 32


@dataclass(frozen=True)
class Geometry:
    rows: int
    logical_n: int
    k_chunk: int
    dense_chunks: tuple = (256, 512)

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


def make_mask(geo, S, H, *, density, band=1, seed=0, keep_block0=True, counts=None):
    """[S, H, nb_pad, S * nb_pad] 0/1 block mask: shard s's local Q block qb keeps global block (s', b).

    Kept: block 0 of every shard (as Sol's pre-pass selects it; keep_block0=False leaves it to chance), the diagonal
    band |i - j| <= band in global block order (none for band < 0), and other blocks at random with probability
    `density`. Only full 64-token blocks are kept, as Sol's pre-pass keeps them; with `counts` (a block_counts
    tensor, or the geometric lengths) every block with a real token may be.
    """
    g = torch.Generator().manual_seed(seed)
    nb, nb_pad = geo.nb(), geo.nb_pad()
    lens = _block_lengths(geo, S, counts)
    full = ((lens == BLOCK) if counts is None else (lens > 0)).reshape(-1)
    mask = (torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < density).to(torch.int64)
    k_global = torch.arange(S * nb_pad)
    k_index = (k_global // nb_pad) * nb + (k_global % nb_pad)
    for s in range(S):
        q_index = s * nb + torch.arange(nb_pad)
        mask[s, :, (q_index[:, None] - k_index[None, :]).abs() <= band] = 1
    if keep_block0:
        mask[..., torch.arange(S) * nb_pad] = 1
    mask = mask * full.to(torch.int64)
    mask[:, :, nb:] = 0
    return mask


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
    nb_pad = geo.nb_pad()
    lens = _block_lengths(geo, S, counts).reshape(-1)
    valid = lens > 0
    log_len = torch.where(valid, lens.clamp(min=1).double().log(), torch.zeros_like(lens, dtype=torch.float64))
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
            qh = q[0, h, rows].double()
            keep = mask[s, h, (torch.arange(r0, r1) // BLOCK)].bool()
            exact = keep[:, tok_block] & tok_valid[None, :]
            le = (qh @ kh.T) * scale
            le = le.masked_fill(~exact, float("-inf"))
            lp = (qh @ pk.T) * scale + log_len[None, :]
            lp = lp.masked_fill(keep | ~valid[None, :] | (not pooled), float("-inf"))
            w = torch.softmax(torch.cat([le, lp], dim=1), dim=1).nan_to_num(0.0)
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


RESIDUAL_COLS = 2
RESIDUAL_NEG = -30000.0


def residual_bias_split(w, scale, cols=RESIDUAL_COLS):
    """Raw log weights w (the residual weighs exp(w * scale)) -> [..., cols] bf16-exact column biases whose weights sum
    to it: each column the remaining log weight truncated to bf16, the next the rest (the model's pre-pass rule)."""
    out = []
    w = w.float()
    for c in range(cols):
        b = (w.contiguous().view(torch.int32) & -65536).view(torch.float32)
        out.append(b)
        w = w + torch.log((1.0 - torch.exp((b - w) * scale)).clamp(min=1e-30)) / scale
    return torch.stack(out, -1)


def block_residual_inputs(q, k, v, mask, geo, S, scale, quantize, cols=RESIDUAL_COLS):
    """The block residual of every (shard, head, Q block) in fp64: over the valid blocks the mask leaves out,
    p_j = exp(scale q_bar . kc_j + ln len_j - m), res = sum p_j vc_j / sum p_j, lse = m + ln sum p_j (q_bar, kc, vc the
    block means over real tokens). Returns the op's inputs (pooled_k zeros and pooled_v = res, [S, H, cols * nb_pad, D],
    entry qb * cols + c; residual_bias [S, H, nb_pad, 32]) and, for the references, res as the device reads it and the
    raw log weights of the op's biases (w_op) and of the exact lse (w_exact), [S, H, nb_pad]."""
    nb_pad = geo.nb_pad()
    lens = _block_lengths(geo, S).reshape(-1)
    valid = lens > 0
    log_len = torch.where(valid, lens.clamp(min=1).double().log(), torch.full_like(lens, -1e30, dtype=torch.float64))
    qbar = pool(q, geo, S)[0].double().view(-1, S, nb_pad, q.shape[3]).permute(1, 0, 2, 3)
    kc, vc = pool(k, geo, S)[0].double(), pool(v, geo, S)[0].double()
    lg = scale * (qbar @ kc.transpose(-2, -1)[None]) + log_len
    lg = lg.masked_fill(mask.bool() | ~valid, float("-inf"))
    lse = torch.logsumexp(lg, -1)
    p = torch.exp(lg - lse.clamp(min=-1e30)[..., None]).nan_to_num(0.0)
    res = quantize((p @ vc[None]).float())
    bias = residual_bias_split((lse / scale).clamp(min=RESIDUAL_NEG), scale, cols)
    w_op = torch.logsumexp(bias.double() * scale, -1) / scale
    rows = torch.zeros(*bias.shape[:3], 32)
    rows[..., :cols] = bias
    pv = res.repeat_interleave(cols, dim=2)
    return {
        "pooled_k": torch.zeros_like(pv),
        "pooled_v": pv,
        "bias_rows": rows,
        "res": res,
        "w_op": w_op,
        "w_exact": lse / scale,
    }


def residual_reference(q, k, v, mask, res, w_log, geo, S, scale, *, heads=None, shards=None):
    """fp64: kept blocks exact over their real tokens, merged per Q block with its residual res [S, H, nb_pad, D] of
    raw log weight w_log [S, H, nb_pad] (weight exp(w_log * scale)) by log-sum-exp. heads / shards as in
    sparse_reference."""
    _, H, _, _ = q.shape
    nb_pad = geo.nb_pad()
    lens = _block_lengths(geo, S).reshape(-1)
    tok = torch.arange(S * geo.rows)
    tok_block = (tok // geo.rows) * nb_pad + (tok % geo.rows) // BLOCK
    tok_valid = (tok % geo.rows) % BLOCK < lens[tok_block]
    out = torch.full((1, H, S * geo.rows, q.shape[3]), float("nan"))
    for h in range(H) if heads is None else heads:
        kh, vh = k[0, h].double(), v[0, h].double()
        for s, r0 in ((s, r0) for s in (range(S) if shards is None else shards) for r0 in range(0, geo.rows, 512)):
            r1 = min(r0 + 512, geo.rows)
            rows = slice(s * geo.rows + r0, s * geo.rows + r1)
            qb = torch.arange(r0, r1) // BLOCK
            exact = mask[s, h, qb].bool()[:, tok_block] & tok_valid[None, :]
            le = ((q[0, h, rows].double() @ kh.T) * scale).masked_fill(~exact, float("-inf"))
            lr = (w_log[s, h, qb].double() * scale)[:, None]
            w = torch.softmax(torch.cat([le, lr], dim=1), dim=1).nan_to_num(0.0)
            out[0, h, rows] = (w[:, :-1] @ vh + w[:, -1:] * res[s, h, qb].double()).float()
    return out


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG, trace_region_size=TRACE_REGION_SIZE)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


class Case:
    """Inputs, device buffers and both ops for one geometry on the module's mesh. dtype: of Q, K, V and the pooled
    K / V (bf16, or block float)."""

    def __init__(self, rt, geo, dtype=ttnn.bfloat16):
        self.rt, self.geo, self.dtype = rt, geo, dtype
        self.S, self.TP = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
        self.H = HEADS_PER_DEVICE * self.TP
        self.scale = 1.0 / math.sqrt(D)
        dev = rt.mesh_device
        shape = tuple(dev.shape)

        def dims(sp, tp):
            d = [None, None]
            d[rt.sp_axis], d[rt.tp_axis] = sp, (tp if self.TP > 1 else None)
            return d

        self.seq_dims = dims(2, 1)
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

    def _upload(self, x, d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, on_device=True):
        device = self.rt.mesh_device if on_device else None
        return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, mesh_mapper=self.mapper(d))

    def length_tensor(self, n, on_device=True):
        """The logical_n_tensor input: [1, 1, 1, 1] uint32, replicated (on_device=False: the host twin that
        ttnn.copy_host_to_device_tensor rewrites it with)."""
        x = torch.tensor([n], dtype=torch.int64).reshape(1, 1, 1, 1)
        return self._upload(
            x, self.dims(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, on_device=on_device
        )

    def dense_table(self, blocks, rows=64, on_device=True):
        """The dense_q_blocks_tensor input: [1, 1, 1, rows] uint32, the global dense Q blocks, then 0xFFFFFFFF (unused),
        replicated (on_device=False: the host twin for ttnn.copy_host_to_device_tensor)."""
        x = torch.full((1, 1, 1, rows), 0xFFFFFFFF, dtype=torch.int64)
        x[0, 0, 0, : len(blocks)] = torch.tensor(list(blocks), dtype=torch.int64)
        return self._upload(
            x, self.dims(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, on_device=on_device
        )

    def head_weights(self, seed):
        """A head_weights input (per (head, shard) kept-block counts, any values: they only steer the dynamic split)."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(0, 64, (1, 1, HEADS_PER_DEVICE * self.S, 8), generator=g)
        return self._upload(x, self.dims(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

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
        logical_n_tensor=None,
        host_logical_n=None,
        head_weights=None,
        dense_q_blocks=(),
        dense_q_blocks_tensor=None,
        dense_q_blocks_max_per_shard=8,
        residual=None,
    ):
        """pooled=False omits pooled K / V: kept blocks only. dense_floor=True (with
        want_ref="slice") also computes the exact dense reference on the same slice (r["dense_ref"]). counts: the
        block_counts input ([S * nb_pad] real tokens per global block), also used by the pooling and the reference.
        logical_n_tensor: the op's runtime length input (holding geo.logical_n); host_logical_n: the logical_n
        argument instead of geo.logical_n (a placeholder with the tensor). head_weights / dense_q_blocks: the op's.
        residual: block_residual_inputs' dict: the block-residual inputs replace the pooled ones, and the reference is
        residual_reference with the op's biases (r["ref_exact_lse"]: with the exact lse)."""
        rt, geo = self.rt, self.geo
        tq, tk, tv = (self._upload(x, self.seq_dims, dtype=self.dtype) for x in (q, k, v))
        pooled_k, pooled_v = (self.quantize(pool(x, geo, self.S, counts)) for x in (k, v))
        t_mask = self._upload(mask, self.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        t_pk, t_pv = (
            (self._upload(x, self.dims(None, 1), dtype=self.dtype) for x in (pooled_k, pooled_v))
            if pooled and residual is None
            else (None, None)
        )
        t_rb = None
        if residual is not None:
            t_pk, t_pv = (
                self._upload(residual[x], self.dims(0, 1), dtype=self.dtype) for x in ("pooled_k", "pooled_v")
            )
            t_rb = self._upload(
                residual["bias_rows"], self.dims(0, 1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
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
            logical_n=geo.logical_n if host_logical_n is None else host_logical_n,
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
            head_weights=head_weights,
            dense_q_blocks=list(dense_q_blocks),
            logical_n_tensor=logical_n_tensor,
            dense_q_blocks_tensor=dense_q_blocks_tensor,
            dense_q_blocks_max_per_shard=dense_q_blocks_max_per_shard,
            **({} if t_rb is None else {"residual_bias": t_rb}),
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
        dense_ref = None
        if want_ref == "full":
            sl = {}
        elif want_ref == "slice":
            sl = dict(heads=[0, self.H - 1], shards=[0, self.S - 1])
            if dense_floor:
                dense_ref = dense_reference(q, k, v, geo.logical_n, self.scale, rows=geo.rows, **sl)
        else:
            sl = dict(heads=sorted({0, 1, self.H // 2, self.H - 1}))
        ref_exact = None
        if residual is not None and want_ref:
            ref, ref_exact = (
                residual_reference(q, k, v, mask, residual["res"], residual[w], geo, self.S, self.scale, **sl)
                for w in ("w_op", "w_exact")
            )
        else:
            ref = (
                sparse_reference(
                    q, k, v, mask, pooled_k, pooled_v, geo, self.S, self.scale, pooled=pooled, counts=counts, **sl
                )
                if want_ref
                else None
            )
        n = geo.logical_n
        return {
            "ref_exact_lse": None if ref_exact is None else ref_exact[:, :, :n],
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


def _rel_l2(ref, got):
    """Relative L2 error over the entries where both are defined."""
    a, b = ref.flatten().double(), got.flatten().double()
    keep = ~(torch.isnan(a) | torch.isnan(b))
    return ((b[keep] - a[keep]).norm() / a[keep].norm()).item()


def _check(tag, r, density, *, dense_min, ref_min):
    p_dense = _pcc(r["dense"], r["sparse"]) if r["dense"] is not None else float("nan")
    p_ref = _pcc(r["ref"], r["sparse"]) if r["ref"] is not None else float("nan")
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs dense {p_dense:.5f}, vs reference {p_ref:.5f}")
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    if dense_min is not None:
        assert p_dense > dense_min, f"{tag}: pcc vs dense {p_dense:.5f}"
    if ref_min is not None:
        assert p_ref > ref_min, f"{tag}: pcc vs reference {p_ref:.5f}"


DENSE_FLOOR_MARGIN = 5e-4


def _check_vs_dense_floor(tag, r, density):
    p_ref = _pcc(r["ref"], r["sparse"])
    p_floor = _pcc(r["dense_ref"], r["dense"])
    logger.info(f"[{tag}] block density {density:.3f}: pcc vs reference {p_ref:.5f}, dense vs exact {p_floor:.5f}")
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    assert p_ref > p_floor - DENSE_FLOOR_MARGIN, f"{tag}: pcc vs reference {p_ref:.5f}, dense floor {p_floor:.5f}"


ALIGNED = {
    "r512k512": Geometry(512, 512 * MESH_CONFIG.sp_size, 512),
    "r1024k256": Geometry(1024, 1024 * MESH_CONFIG.sp_size, 256),
    "r1024k512": Geometry(1024, 1024 * MESH_CONFIG.sp_size, 512),
    "r1024k256short": Geometry(1024, 1024 * MESH_CONFIG.sp_size - 512, 256),
}
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
    dense_min = 0.98 if density == KEEP_ALL else None
    _check(f"ragged d{density} {geom}", r, _density(mask, case.geo, case.S), dense_min=dense_min, ref_min=0.999)


@pytest.mark.parametrize("passes", [1, 2])
def test_empty_q_blocks(runtime, passes):
    """Q blocks that keep no block anywhere: a whole head, every third Q block of the first shard, and the last head
    on the last shard. They attend the pooled entries only."""
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
    r = case.run(*case.inputs(19), mask, want_dense=False, q_blocks_per_pass=q_blocks_per_pass)
    _check(f"empty q blocks passes {passes}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


def test_bfp8_inputs(runtime):
    """BFP8 Q, K, V and pooled K / V: the K / V CBs take the inputs' tile format and size, and the pooled entries are
    read into them in the same format. The reference sees the same rounded values."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo, dtype=ttnn.bfloat8_b)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=25)
    r = case.run(*case.inputs(25), mask, want_dense=False)
    _check("bfp8", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)


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


@SLOW
@pytest.mark.parametrize("density", [KEEP_ALL, 0.15])
def test_production_10s(runtime, density):
    """MiniMax-H3 at 10 s: 9184 rows per device, more Q blocks per core than fit L1 at once, so the op picks two
    passes with pair steps. Gated against the dense op's accuracy floor (each shard's
    32-token last block is pooled even with every full block kept)."""
    geo = Geometry(9184, 9184 * MESH_CONFIG.sp_size, 256, dense_chunks=(256, 512))
    case = Case(runtime, geo)
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


# Runtime logical length (logical_n_tensor): per geometry, three lengths of the same padded shape - the full length, one
# ending mid-tile, and one leaving the last shard empty (also mid-tile).
RUNTIME_LEN_GEOMETRIES = {"r1024k256": ALIGNED["r1024k256"], "r1184k512": RAGGED["r1184k512"]}


def _runtime_lengths(geo, S):
    full = S * geo.rows
    return [full, full - 472, (S - 1) * geo.rows - 40]


def _dense_rows(mask, geo, S, blocks):
    """The global Q blocks `blocks` (shard * nb + local block) keep every full block, as Sol's exact prefix rows."""
    full = (_block_lengths(geo, S) == BLOCK).reshape(-1).to(torch.int64)
    for g in blocks:
        mask[g // geo.nb(), :, g % geo.nb()] = full
    return mask


@pytest.mark.parametrize("dyn", [False, True], ids=["even_split", "dyn_split"])
@pytest.mark.parametrize("geom", list(RUNTIME_LEN_GEOMETRIES))
def test_runtime_len_one_program(runtime, geom, dyn, monkeypatch):
    """logical_n_tensor: three lengths of one padded shape run on ONE cached program (the host logical_n a fixed
    placeholder), each bit-identical to the per-length program without the tensor. dyn_split also runs the dynamic Q
    split with dense Q blocks, whose cost (a dense row's kept blocks) the reader recounts from the live length."""
    base = RUNTIME_LEN_GEOMETRIES[geom]
    case = Case(runtime, base)
    S = case.S
    lengths = _runtime_lengths(base, S)
    extra = {}
    dense = []
    if dyn:
        monkeypatch.setenv("TT_SPARSE_RING_DYN_SPLIT", "on")
        dense = [1, base.nb() + 3]
        extra = dict(head_weights=case.head_weights(41), dense_q_blocks=dense)
    q, k, v = case.inputs(41)
    masks = []
    for n in lengths:
        geo = replace(base, logical_n=n)
        masks.append(_dense_rows(make_mask(geo, S, case.H, density=0.2, seed=41), geo, S, dense))

    dev = runtime.mesh_device
    refs = []
    start = dev.num_program_cache_entries()
    for n, mask in zip(lengths, masks):
        case.geo = replace(base, logical_n=n)
        refs.append(case.run(q, k, v, mask, want_dense=False, want_ref=False, **extra)["sparse"])
    tag = f"runtime_len {geom} {'dyn' if dyn else 'even'}"
    new_programs = dev.num_program_cache_entries() - start
    logger.info(f"[{tag}] without the tensor: {new_programs} new programs for {len(lengths)} lengths")

    t_len = case.length_tensor(lengths[0])
    added = []
    for i, (n, mask) in enumerate(zip(lengths, masks)):
        case.geo = replace(base, logical_n=n)
        ttnn.copy_host_to_device_tensor(case.length_tensor(n, on_device=False), t_len)
        before = dev.num_program_cache_entries()
        got = case.run(
            q, k, v, mask, want_dense=False, want_ref=False, logical_n_tensor=t_len, host_logical_n=lengths[0], **extra
        )["sparse"]
        added.append(dev.num_program_cache_entries() - before)
        same = torch.equal(got, refs[i])
        diff = (got - refs[i]).abs().max().item()
        logger.info(f"[{tag}] logical_n {n}: new programs {added[-1]}, bit-identical {same}, max |diff| {diff:.3e}")
        assert same, f"logical_n {n}: differs from the per-length program, max |diff| {diff:.3e}"
    assert added == [1, 0, 0], f"program cache entries added per call {added}: expected one program for every length"


def test_runtime_len_trace_replay(runtime):
    """One trace captured at one length, replayed at others with only the inputs and logical_n_tensor rewritten in
    place: each replay bit-identical to the per-length program without the tensor."""
    base = RUNTIME_LEN_GEOMETRIES["r1184k512"]
    case = Case(runtime, base)
    rt, S = runtime, case.S
    dev = rt.mesh_device
    lengths = _runtime_lengths(base, S)
    q, k, v = case.inputs(43)
    per_len = []
    refs = []
    for n in lengths:
        case.geo = geo = replace(base, logical_n=n)
        mask = make_mask(geo, S, case.H, density=0.2, seed=43)
        pooled = [case.quantize(pool(x, geo, S)) for x in (k, v)]
        per_len.append((mask, pooled))
        refs.append(case.run(q, k, v, mask, want_dense=False, want_ref=False)["sparse"])

    tq, tk, tv = (case._upload(x, case.seq_dims) for x in (q, k, v))
    mask_args = (case.dims(0, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    t_mask = case._upload(per_len[0][0], *mask_args)
    t_pk, t_pv = (case._upload(x, case.dims(None, 1)) for x in per_len[0][1])
    t_len = case.length_tensor(lengths[0])

    def call():
        return ttnn.transformer.sparse_ring_sdpa(
            tq,
            tk,
            tv,
            t_mask,
            t_pk,
            t_pv,
            persistent_output_buffer_k=case.buf_k,
            persistent_output_buffer_v=case.buf_v,
            logical_n=lengths[0],
            program_config=case.sparse_config,
            compute_kernel_config=rt.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=dev,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            block_size=BLOCK,
            logical_n_tensor=t_len,
        )

    ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
    out = call()
    ttnn.end_trace_capture(dev, trace_id, cq_id=0)
    ttnn.synchronize_device(dev)
    try:
        for i in [1, 2, 0, 2, 1]:
            n = lengths[i]
            mask, (pk, pv) = per_len[i]
            ttnn.copy_host_to_device_tensor(case._upload(mask, *mask_args, on_device=False), t_mask)
            ttnn.copy_host_to_device_tensor(case._upload(pk, case.dims(None, 1), on_device=False), t_pk)
            ttnn.copy_host_to_device_tensor(case._upload(pv, case.dims(None, 1), on_device=False), t_pv)
            ttnn.copy_host_to_device_tensor(case.length_tensor(n, on_device=False), t_len)
            ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(out, mesh_composer=case.composer).float()[:, :, :n]
            same = torch.equal(got, refs[i])
            diff = (got - refs[i]).abs().max().item()
            logger.info(
                f"[runtime_len trace] replay at logical_n {n} (captured at {lengths[0]}): bit-identical {same}, "
                f"max |diff| {diff:.3e}"
            )
            assert same, f"replay at logical_n {n}: max |diff| {diff:.3e}"
    finally:
        ttnn.release_trace(dev, trace_id)


# Dense Q block table (dense_q_blocks_tensor): per length a different set of dense prefix Q blocks, at most two on a
# shard (the program is sized by dense_q_blocks_max_per_shard=2, as the list path sizes itself), the production
# switches.
_PROD_SWITCHES = {
    "TT_SPARSE_RING_Q_PERM": "1",
    "TT_SPARSE_RING_FAST_PBIAS": "2",
    "TT_SPARSE_RING_DYN_SPLIT": "on",
    "TT_SPARSE_RING_DYN_EXTRA_PASS": "1",
    "TT_SPARSE_RING_KV_PAGES": "max",
    "TT_SPARSE_RING_MEMBER_RESCALE": "1",
    "TT_SPARSE_RING_MULTIPASS_PAIRS": "1",
    "TT_SPARSE_RING_BANKED_READS": "1",
}


def _dense_sets(nb):
    return [[1, 2, nb + 3], [1, nb + 1, nb + 2, 2 * nb + 2], [2, 3, nb + 4, 3 * nb + 5]]


@pytest.mark.parametrize("geom", list(RUNTIME_LEN_GEOMETRIES))
def test_runtime_len_dense_table(runtime, geom, monkeypatch):
    """dense_q_blocks_tensor with logical_n_tensor: three (length, dense Q blocks) pairs of one padded shape run on ONE
    cached program, each bit-identical to the per-length program with the dense_q_blocks list."""
    for name, value in _PROD_SWITCHES.items():
        monkeypatch.setenv(name, value)
    base = RUNTIME_LEN_GEOMETRIES[geom]
    case = Case(runtime, base)
    S, dev = case.S, runtime.mesh_device
    lengths = _runtime_lengths(base, S)
    dense_sets = _dense_sets(base.nb())
    hw = case.head_weights(47)
    q, k, v = case.inputs(47)
    masks = []
    for n, dense in zip(lengths, dense_sets):
        geo = replace(base, logical_n=n)
        masks.append(_dense_rows(make_mask(geo, S, case.H, density=0.2, seed=47), geo, S, dense))

    refs = []
    for n, mask, dense in zip(lengths, masks, dense_sets):
        case.geo = replace(base, logical_n=n)
        refs.append(
            case.run(q, k, v, mask, want_dense=False, want_ref=False, head_weights=hw, dense_q_blocks=dense)["sparse"]
        )

    t_len = case.length_tensor(lengths[0])
    t_dense = case.dense_table(dense_sets[0])
    added = []
    for i, (n, mask, dense) in enumerate(zip(lengths, masks, dense_sets)):
        case.geo = replace(base, logical_n=n)
        ttnn.copy_host_to_device_tensor(case.length_tensor(n, on_device=False), t_len)
        ttnn.copy_host_to_device_tensor(case.dense_table(dense, on_device=False), t_dense)
        before = dev.num_program_cache_entries()
        got = case.run(
            q,
            k,
            v,
            mask,
            want_dense=False,
            want_ref=False,
            logical_n_tensor=t_len,
            host_logical_n=lengths[0],
            head_weights=hw,
            dense_q_blocks_tensor=t_dense,
            dense_q_blocks_max_per_shard=2,
        )["sparse"]
        added.append(dev.num_program_cache_entries() - before)
        same = torch.equal(got, refs[i])
        diff = (got - refs[i]).abs().max().item()
        logger.info(
            f"[runtime_len dense_table {geom}] logical_n {n} dense {dense}: new programs {added[-1]}, bit-identical "
            f"{same}, max |diff| {diff:.3e}"
        )
        assert same, f"logical_n {n} dense {dense}: differs from the per-length list program, max |diff| {diff:.3e}"
    assert added == [1, 0, 0], f"program cache entries added per call {added}: expected one program for every pair"


# Block residual (residual_bias): each Q block attends its kept blocks exactly and summarizes the blocks it leaves out
# by one residual from its mean query, merged by log-sum-exp (the reference algorithm the MiniMax-H3 pre-pass computes
# in its sol_residual kernel). The inputs come from block_residual_inputs, the reference from
# residual_reference with the op's bf16 column biases; the exact-lse reference is logged (the biases' own error). The
# rel-L2 gate is the op's own rounding floor: the default (pooled) path against its reference on the same inputs.
RESIDUAL_FLOOR_MARGIN = 1.25


def _residual_run(case, mask, seed, **kw):
    q, k, v = case.inputs(seed)
    res = block_residual_inputs(q, k, v, mask, case.geo, case.S, case.scale, case.quantize)
    want_ref = kw.pop("want_ref", True)
    r = case.run(q, k, v, mask, want_dense=False, want_ref=want_ref, residual=res, **kw)
    base = case.run(q, k, v, mask, want_dense=False, want_ref=want_ref, **kw)
    r["floor"] = _rel_l2(base["ref"], base["sparse"])
    return r


def _check_residual(tag, r, density):
    p_ref = _pcc(r["ref"], r["sparse"])
    rel = _rel_l2(r["ref"], r["sparse"])
    rel_exact = _rel_l2(r["ref_exact_lse"], r["sparse"])
    logger.info(
        f"[{tag}] block density {density:.3f}: vs residual reference pcc {p_ref:.6f} rel-L2 {rel:.3e} "
        f"(exact-lse reference {rel_exact:.3e}; default path vs its reference {r['floor']:.3e})"
    )
    assert torch.isfinite(r["sparse"]).all(), f"{tag}: non-finite output"
    bar = RESIDUAL_FLOOR_MARGIN * r["floor"]
    assert p_ref > 0.999 and rel < bar, f"{tag}: pcc {p_ref:.6f}, rel-L2 {rel:.3e} (bar {bar:.3e})"


@pytest.mark.parametrize("prod", [False, True], ids=["defaults", "prod_switches"])
@pytest.mark.parametrize("geom", ["r1024k256", "r1184k512", "r1184ln_off"])
def test_block_residual(runtime, geom, prod, monkeypatch):
    """Random kept blocks without the forced block 0, aligned and ragged shards, with the op's defaults and with the
    production switches plus the runtime length."""
    geo = {**ALIGNED, **RAGGED}[geom]
    extra = {}
    case = Case(runtime, geo)
    if prod:
        for name, value in _PROD_SWITCHES.items():
            monkeypatch.setenv(name, value)
        extra = dict(logical_n_tensor=case.length_tensor(geo.logical_n), head_weights=case.head_weights(31))
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=31, keep_block0=False)
    r = _residual_run(case, mask, 31, **extra)
    _check_residual(f"residual {geom} {'prod' if prod else 'defaults'}", r, _density(mask, geo, case.S))


def test_block_residual_bfp8(runtime):
    """BFP8 inputs: the residual entries are read in the K / V CBs' block float format."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo, dtype=ttnn.bfloat8_b)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=33, keep_block0=False)
    _check_residual("residual bfp8", _residual_run(case, mask, 33), _density(mask, geo, case.S))


def test_block_residual_dense_and_empty_rows(runtime):
    """Q blocks that keep every full block (a residual over the partial blocks only, or none: the -30000 bias) and Q
    blocks that keep nothing (the residual only), on a ragged geometry."""
    geo = RAGGED["r1184ln_off"]
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=0.2, seed=35, keep_block0=False)
    mask = _dense_rows(mask, geo, case.S, [0, 1, geo.nb() + 2])
    mask[:, 1, 5:9] = 0
    mask[case.S - 1, case.H - 1] = 0
    _check_residual("residual dense / empty rows", _residual_run(case, mask, 35), _density(mask, geo, case.S))


@SLOW
@pytest.mark.parametrize(("rows", "k_chunk"), [(4736, 256), (9184, 256)], ids=["5s", "10s"])
def test_block_residual_production(runtime, rows, k_chunk, monkeypatch):
    """MiniMax-H3 5 s / 10 s with the production switches, the runtime length, the dense Q block table and head
    weights: the program fits and matches the residual reference on a slice."""
    for name, value in _PROD_SWITCHES.items():
        monkeypatch.setenv(name, value)
    geo = Geometry(rows, rows * MESH_CONFIG.sp_size - 300, k_chunk)
    case = Case(runtime, geo)
    dense = [0, 1, 2 * geo.nb() + 1]
    mask = _dense_rows(make_mask(geo, case.S, case.H, density=0.15, seed=37, keep_block0=False), geo, case.S, dense)
    r = _residual_run(
        case,
        mask,
        37,
        want_ref="slice",
        logical_n_tensor=case.length_tensor(geo.logical_n),
        dense_q_blocks_tensor=case.dense_table(dense),
        dense_q_blocks_max_per_shard=2,
        head_weights=case.head_weights(37),
    )
    _check_residual(f"residual {rows} rows", r, _density(mask, geo, case.S))
