# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn routing pre-pass for the ring-joint SDPA path.

Runs as plain ttnn ops between the fused QK-norm/RoPE and the ring SDPA. Per device it produces,
from the local Q/K/V shards [1, H_local, R, D]:

* ``block_mask``  uint32 row-major [1, H, n_q_chunks_pad, n_blocks_pad]: 1 = this (local Q chunk, global
  64-token block) pair is attended exactly (the kernel gathers kept blocks into dense K/V chunks), 0 = the
  block goes through the pooled correction. Only full 64-token blocks inside logical_n are ever exact;
  block 0 of every shard is forced exact (one gathered chunk per ring iteration).
* ``pooled_k`` / ``pooled_v``  bf16 tile [1, H, n_blocks_pad, D]: per-64-token block mean K / mean V
  for the WHOLE sequence (all-gathered along SP), per-shard blocking.
* ``bias``  OPTIONAL (``emit_bias``; default from ``TT_SOL_EMIT_BIAS``) bf16 tile
  [1, H, n_q_chunks_pad * 32, n_blocks_pad]: per Q chunk, one 32-row tile row holding the additive
  column bias for the pooled logits: ln(block token count) / scale for blocks the Q chunk approximates
  (the kernel evaluates exp((QK + bias - max) * scale), so this weights the pooled column by its token
  count), a large negative value for blocks it computes exactly (no double counting) and for padding
  blocks. When not emitted (``bias`` is None) the ring op's reader generates the identical tile rows on
  device from the mask row and the block geometry (sol_bias_gen.hpp), which removes the bias upload and
  two pre-pass ops. Interface contract: SOL_ATTN_INTERFACE.md next to this file.

Geometry: blocks are 64 tokens inside each SP shard (the last block of a shard may be 32 tokens);
K chunks are ``k_chunk`` tokens inside each shard; global K chunk index = shard * k_chunks_per_shard
+ local index. Every tensor dimension is padded to a tile multiple.

``torch_reference(...)`` is the same arithmetic in fp32 torch on host (for unit tests).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

import ttnn

LOG2E = 1.4426950408889634
VAR_EPS = 1.0e-6
NEG = -30000.0  # finite "minus infinity" for bf16 bias/mask arithmetic (exp2 underflows to exactly 0)
BLOCK = 64
TILE = 32


def _pad32(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class SolGeometry:
    shards: int
    rows: int  # rows per shard (padded sequence / shards)
    logical_n: int
    q_chunk: int
    k_chunk: int
    block: int = BLOCK

    @property
    def nb(self) -> int:  # blocks per shard
        return -(-self.rows // self.block)

    @property
    def nb_pad(self) -> int:
        return _pad32(self.nb)

    @property
    def n_blocks_pad(self) -> int:  # gathered pooled length (keeps per-shard padding)
        return self.shards * self.nb_pad

    @property
    def nqc(self) -> int:
        return -(-self.rows // self.q_chunk)

    @property
    def nqc_pad(self) -> int:
        return _pad32(self.nqc)

    @property
    def nkc(self) -> int:  # K chunks per shard
        return -(-self.rows // self.k_chunk)

    @property
    def nkc_all(self) -> int:
        return self.shards * self.nkc

    @property
    def nkc_pad(self) -> int:
        return _pad32(self.nkc_all)

    def block_lengths(self) -> torch.Tensor:
        """[shards, nb_pad] valid tokens per (padded) block, clipped to logical_n."""
        out = torch.zeros(self.shards, self.nb_pad)
        for s in range(self.shards):
            for b in range(self.nb):
                start = s * self.rows + b * self.block
                end = min(start + self.block, s * self.rows + self.rows, self.logical_n)
                out[s, b] = max(0, end - start)
        return out


def build_constants(
    geo: SolGeometry,
    *,
    band: int,
    sink_tokens: int,
    force_first_chunk: bool,
    scale: float,
    dense_prefix_queries: bool = True,
) -> dict:
    """Host-built constant matrices (torch fp32) for the device pre-pass; all tile-padded.

    The bias is in pre-scale logit units: the SDPA compute kernel evaluates exp((QK + bias - max) * scale),
    so a column bias of ln(len)/scale weights the pooled column by exactly `len`.
    """
    S, nb, nbp, nbg = geo.shards, geo.nb, geo.nb_pad, geo.n_blocks_pad
    lengths = geo.block_lengths()  # [S, nbp]
    # Pooling matrix transposed, per shard: [S, 1, rows, nb_pad], entry 1/len for a block's valid rows.
    pool_t = torch.zeros(S, 1, geo.rows, nbp)
    for s in range(S):
        for b in range(nb):
            length = int(lengths[s, b])
            if length:
                pool_t[s, 0, b * geo.block : b * geo.block + length, b] = 1.0 / length
    # Global block index of each padded block: g = s * nb_pad + b. Band on real indices (s*nb + b) so the
    # neighbourhood crosses shard boundaries like the reference's contiguous blocking.
    gidx = torch.full((nbg,), -1)
    for s in range(S):
        gidx[s * nbp : s * nbp + nb] = torch.arange(nb) + s * nb
    valid_row = (lengths.reshape(-1) > 0).float()  # [nbg]
    full_row = (lengths.reshape(-1) >= geo.block).float()  # exact-eligible: full blocks only
    # Band per shard: [S, 1, nb_pad(local q blocks), nbg]
    band_m = torch.zeros(S, 1, nbp, nbg)
    for s in range(S):
        qg = torch.full((nbp,), -(10**9))
        qg[:nb] = torch.arange(nb) + s * nb
        band_m[s, 0] = ((qg[:, None] - gidx[None, :]).abs() <= band).float() * valid_row[None, :]
        if dense_prefix_queries:
            # Prefix (text + condition) QUERY rows attend densely, as the reference implementation recomputes them
            # outside the sparse kernel: every Q block whose first token lies before sink_tokens keeps all valid K
            # blocks. Folded
            # into the band constant so it costs no extra op. (The ragged last block stays pooled: the kernel
            # only gathers full blocks.)
            for b in range(nb):
                if s * geo.rows + b * geo.block < sink_tokens:
                    band_m[s, 0, b] = valid_row
    # Sink columns: blocks whose first token lies before sink_tokens.
    sink_row = torch.zeros(1, 1, 1, nbg)
    for s in range(S):
        for b in range(nb):
            if s * geo.rows + b * geo.block < sink_tokens and lengths[s, b] > 0:
                sink_row[0, 0, 0, s * nbp + b] = 1.0
    # Block -> global K chunk pooling: [1, 1, nbg, nkc_pad]
    kc_pool = torch.zeros(1, 1, nbg, geo.nkc_pad)
    for s in range(S):
        for b in range(nb):
            if lengths[s, b] > 0:
                kc_pool[0, 0, s * nbp + b, s * geo.nkc + (b * geo.block) // geo.k_chunk] = 1.0
    # Local Q block -> Q chunk pooling (transposed): [S, 1, nb_pad, nqc_pad]  (same for every shard)
    qc_pool_t = torch.zeros(1, 1, nbp, geo.nqc_pad)
    for b in range(nb):
        qc_pool_t[0, 0, b, (b * geo.block) // geo.q_chunk] = 1.0
    # Forced columns: block 0 of every shard (one gathered chunk per ring iteration)
    force_row = torch.zeros(1, 1, 1, nbg)
    if force_first_chunk:
        for s in range(S):
            force_row[0, 0, 0, s * nbp] = 1.0
    # Valid K chunks (hold at least one real token)
    valid_chunk_row = torch.zeros(1, 1, 1, geo.nkc_pad)
    for s in range(S):
        for c in range(geo.nkc):
            if s * geo.rows + c * geo.k_chunk < geo.logical_n:
                valid_chunk_row[0, 0, 0, s * geo.nkc + c] = 1.0
    len_bias_row = torch.full((1, 1, 1, nbg), NEG)
    pos = lengths.reshape(-1) > 0
    len_bias_row[0, 0, 0, pos] = torch.log(lengths.reshape(-1)[pos]) / scale
    return {
        "pool_t": pool_t,
        "band": band_m,
        "sink_row": sink_row,
        "valid_row": valid_row.view(1, 1, 1, nbg),
        "full_row": full_row.view(1, 1, 1, nbg),
        "kc_pool": kc_pool,
        "kc_pool_t": kc_pool.transpose(-2, -1).contiguous(),
        "qc_pool_t": qc_pool_t,
        "force_row": force_row,
        "valid_chunk_row": valid_chunk_row,
        "len_bias_row": len_bias_row,
        "n_valid_blocks": float(pos.sum()),
    }


# ----------------------------------------------------------------------------- static row max (resident mode)


def row_max_constants(head_dim: int, scale: float) -> dict:
    """Host constants for `compute_row_max` (fp32 torch, replicated on upload).

    ``e0_col`` [1, 1, D, 32]: column 0 all ones, the rest zero. A squared Q/K row times it is the row's squared
    norm in tile column 0 with zeros in columns 1..31, i.e. already the one-tile-column layout the ring op takes.
    ``row_max_bias`` [1, 1, 1, 32]: ln(64) / scale in column 0 only, the largest pooled-column bias the ring op
    generates (ln(block length) / scale, block length <= 64), in the same raw logit units.
    """
    e0_col = torch.zeros(1, 1, head_dim, TILE)
    e0_col[0, 0, :, 0] = 1.0
    row_max_bias = torch.zeros(1, 1, 1, TILE)
    row_max_bias[0, 0, 0, 0] = math.log(BLOCK) / scale
    return {"e0_col": e0_col, "row_max_bias": row_max_bias}


def compute_row_max(
    q: ttnn.Tensor, k: ttnn.Tensor, *, consts: dict, ccl_manager, sp_axis: int, compute_kernel_config
) -> ttnn.Tensor:
    """Resident mode's static row max: bf16 tile [1, H, R, 32] from the local Q/K shards [1, H, R, D].

    Column 0 of row r holds ``||q_r|| * max_over_all_keys ||k|| + ln(64) / scale`` in RAW (unscaled) logit units
    (the ring op applies the softmax scale inside its exp); columns 1..31 are zero. By Cauchy-Schwarz it bounds
    every raw logit of the row plus the largest pooled-column bias, so flash attention with this fixed max is exact
    up to rounding and the kernel drops its running-max bookkeeping.

    The key-norm max is per head over the WHOLE sequence: the local max is all-gathered along SP on a tile-shaped
    [1, H, 32, 32] tensor (whole tiles per device, so the gather concatenates cleanly) and max-reduced, so every
    device of the ring holds the same value. R keeps the Q shard's row count; TILE layout pads it like Q.
    """
    qq = ttnn.multiply(q, q)
    q_norm = ttnn.sqrt(ttnn.matmul(qq, consts["e0_col"], compute_kernel_config=compute_kernel_config))  # [1,H,R,32]
    ttnn.deallocate(qq)
    kk = ttnn.multiply(k, k)
    k_sq = ttnn.matmul(kk, consts["e0_col"], compute_kernel_config=compute_kernel_config)  # [1,H,R,32]
    ttnn.deallocate(kk)
    k_sq_max = ttnn.max(k_sq, dim=2, keepdim=True)  # [1,H,1,32]: column 0 = local max ||k||^2, the rest 0
    ttnn.deallocate(k_sq)
    k_sq_tile = ttnn.repeat_interleave(k_sq_max, TILE, dim=2)  # [1,H,32,32], tile-shaped for the gather
    ttnn.deallocate(k_sq_max)
    k_sq_all = ccl_manager.all_gather(k_sq_tile, dim=2, mesh_axis=sp_axis, use_hyperparams=False)  # [1,H,32*S,32]
    ttnn.deallocate(k_sq_tile)
    # Column 0 = global max ||k||, times a 2 % margin so the bf16 rounding of the norms and of the product can never
    # put the bound below a row's true max (an overestimate only scales the bf16 exp terms, which costs nothing).
    k_norm_max = ttnn.multiply(ttnn.sqrt(ttnn.max(k_sq_all, dim=2, keepdim=True)), 1.02)  # [1,H,1,32]
    ttnn.deallocate(k_sq_all)
    # H-broadcast of the per-head max over the Q rows, then the bias row (same broadcast as len_bias_row).
    row_max = ttnn.add(ttnn.multiply(q_norm, k_norm_max), consts["row_max_bias"])  # [1,H,R,32]
    ttnn.deallocate(q_norm)
    ttnn.deallocate(k_norm_max)
    return row_max


# ----------------------------------------------------------------------------- torch reference


def torch_reference(q, k, v, geo: SolGeometry, consts: dict, *, tau: float, scale: float, thresh_type: str = "diag"):
    """Host fp32 mirror of the device pre-pass. q/k/v: [S, H, rows, D] (one shard per leading index).

    Returns chunk_mask [H, nqc_pad(local), nkc_pad] per shard as [S, H, nqc_pad, nkc_pad], pooled k/v
    [H, nbg, D], bias [S, H, nqc_pad, nbg] (one row per Q chunk; the device version repeats it 32x).
    """
    S, H, R, D = q.shape
    log2_scale = scale * LOG2E
    pool_t = consts["pool_t"]  # [S,1,R,nbp]
    q_bar = torch.stack([q[s].float().transpose(1, 2) @ pool_t[s, 0] for s in range(S)])  # [S,H,D,nbp]
    q_bar = q_bar.transpose(2, 3)  # [S,H,nbp,D]
    kc = torch.cat(
        [(k[s].float().transpose(1, 2) @ pool_t[s, 0]).transpose(1, 2) for s in range(S)], dim=1
    )  # [H,nbg,D]
    vc = torch.cat([(v[s].float().transpose(1, 2) @ pool_t[s, 0]).transpose(1, 2) for s in range(S)], dim=1)
    nv = consts["n_valid_blocks"]
    mean = kc.sum(dim=1, keepdim=True) / nv  # [H,1,D]
    if thresh_type == "diag":
        var = ((kc * kc).sum(dim=1, keepdim=True) / nv - mean * mean).clamp(min=0)
        q_raw_mean = (q_bar * mean[None]).sum(-1, keepdim=True)  # [S,H,nbp,1]
        q_var = (q_bar * q_bar * var[None]).sum(-1, keepdim=True)
    else:
        second = kc.transpose(1, 2) @ kc / nv  # [H,D,D]
        q_raw_mean = (q_bar * mean[None]).sum(-1, keepdim=True)
        q_var = (((q_bar @ second[None]) * q_bar).sum(-1, keepdim=True) - q_raw_mean**2).clamp(min=0)
    thr = q_raw_mean * log2_scale + tau * torch.sqrt(q_var * log2_scale * log2_scale + VAR_EPS)
    scores = (q_bar @ kc[None].transpose(2, 3)) * log2_scale  # [S,H,nbp,nbg]
    exact = (scores > thr).float()
    exact = exact + consts["band"] + consts["sink_row"]
    exact = (exact > 0.5).float() * consts["full_row"]
    m2 = (exact.transpose(2, 3) @ consts["qc_pool_t"]).transpose(2, 3)  # [S,H,nqc_pad,nbg]: any Q block of the chunk
    block_mask = ((m2 + consts["force_row"]) > 0.5).float() * consts["full_row"]
    bias = consts["len_bias_row"] + block_mask * NEG
    return {"block_mask": block_mask, "pooled_k": kc, "pooled_v": vc, "bias": bias, "q_bar": q_bar, "threshold": thr}


# ----------------------------------------------------------------------------- device pre-pass


class SolPrepass:
    """Device-side routing pre-pass. Constants are uploaded once per geometry (bucket rung)."""

    def __init__(
        self,
        *,
        mesh_device,
        ccl_manager,
        sp_axis: int,
        tau: float,
        thresh_type: str = "diag",
        band: int = 1,
        force_first_chunk: bool = True,
        dense_prefix_queries: bool = True,
        mode: str = "gather",
        emit_bias: bool | None = None,
        row_max: bool = False,
    ):
        # "gather": block mask [1,H,nqc_pad,n_blocks_pad], the reader gathers kept 64-token blocks (chains off).
        # "chunk": K-chunk mask [1,H,nqc_pad,nkc_pad] (any kept block in the chunk keeps the chunk); the reader
        #          streams densely with chains and compute skips dropped chunks; blocks of dropped chunks are
        #          approximated by the pooled chunks.
        # "compact": the gather-form block mask (full blocks only; block 0 of every shard forced kept so every
        #          (Q chunk, shard) computes at least one block), streamed through the chains like chunk mode; the
        #          reader compacts each core's kept blocks in L1. Pass sol_mode="compact".
        # "pack" is compact plus the ring op's cross-chunk packing; the pre-pass output is identical to compact.
        assert mode in ("gather", "chunk", "compact", "pack"), mode
        self.mode = mode
        # emit_bias=False: the ring op generates the pooled-column bias on device. TT_SOL_EMIT_BIAS sets the default
        # (1 = emit the bias tensor, 0 = let the op generate it).
        self.emit_bias = (os.environ.get("TT_SOL_EMIT_BIAS", "1") == "1") if emit_bias is None else bool(emit_bias)
        # row_max=True: also return "row_max", the static per-query-row logit bound (compute_row_max) the ring op's
        # resident mode accepts as the deprecated sol_row_max input. The op ignores it; the model leaves it off.
        self.row_max = bool(row_max)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.tau = float(tau)
        self.thresh_type = thresh_type
        self.band = band
        self.force_first_chunk = force_first_chunk
        self.dense_prefix_queries = dense_prefix_queries
        self._consts: dict | None = None
        # Own global semaphores for the two pooled gathers, alternated per call (as SolPrepassFused, SOL_AG_OWN_SEMS):
        # the CCL manager's all-gather pool is shared with the ring attention op's fused K/V gather, whose reader
        # zeroes its semaphore at exit, so a lagging chip could lose a running chip's credits.
        self._ag_sems: list | None = None
        self._ag_idx = 0
        self._key = None
        self.last_density: float | None = None
        # Density of the (q_chunk x 64-key) block mask before any K-chunk union (64x64 blocks when q_chunk is 64,
        # i.e. resident mode). In chunk mode `last_density` is the coarser (q_chunk x k_chunk) density the op skips at.
        self.last_block_density: float | None = None
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.hifi2 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def _upload(self, t: torch.Tensor, *, shard: bool, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        shape = tuple(self.mesh_device.shape)
        if shard:
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=[None, 0] if self.sp_axis == 1 else [0, None], mesh_shape=shape
            )
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper)

    # Uploaded constants shared by every SolPrepass on the same mesh (all layers use identical geometry, sink and
    # scale): one build + upload per process instead of one per layer.
    _shared_consts: dict = {}

    def prepare(self, geo: SolGeometry, sink_tokens: int, scale: float) -> None:
        key = (geo, sink_tokens, scale)
        if key == self._key:
            return
        shared_key = (
            id(self.mesh_device),
            self.sp_axis,
            self.mode,
            self.band,
            self.force_first_chunk,
            self.dense_prefix_queries,
            key,
        )
        cached = SolPrepass._shared_consts.get(shared_key)
        if cached is not None:
            self._consts = cached
            self.geo = geo
            self._key = key
            return
        c = build_constants(
            geo,
            band=self.band,
            sink_tokens=sink_tokens,
            force_first_chunk=self.force_first_chunk,
            scale=scale,
            dense_prefix_queries=self.dense_prefix_queries,
        )
        self._consts = {
            "pool_t": self._upload(c["pool_t"], shard=True),
            "band": self._upload(c["band"], shard=True),
            "sink_row": self._upload(c["sink_row"], shard=False),
            "full_row": self._upload(c["full_row"], shard=False),
            "qc_pool_t": self._upload(c["qc_pool_t"], shard=False),
            "force_row": self._upload(c["force_row"], shard=False),
            "len_bias_row": self._upload(c["len_bias_row"], shard=False),
            "kc_pool": self._upload(c["kc_pool"], shard=False) if self.mode == "chunk" else None,
            "kc_pool_t": self._upload(c["kc_pool_t"], shard=False) if self.mode == "chunk" else None,
            "valid_row": self._upload(c["valid_row"], shard=False),
            "n_valid_blocks": c["n_valid_blocks"],
        }
        SolPrepass._shared_consts[shared_key] = self._consts
        self.geo = geo
        self._key = key

    def _host_density(self, mask01: ttnn.Tensor, cols: torch.Tensor) -> float:
        """Mean of a [1,H,nqc_pad,W] 0/1 mask over the real Q chunks and the given real columns (host readback)."""
        host = ttnn.to_torch(
            mask01,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=[0, 1] if self.sp_axis == 0 else [1, 0],
                mesh_shape=tuple(self.mesh_device.shape),
            ),
        )
        # host: [S, 4H (TP replicas), nqc_pad, W]; count real Q chunks and the real columns only
        sub = host[:, : host.shape[1] // tuple(self.mesh_device.shape)[1 - self.sp_axis], : self.geo.nqc].float()
        return float(sub[..., cols].mean())

    def _block_cols(self) -> torch.Tensor:
        geo = self.geo
        return torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(geo.shards)])

    def _finish_chunk_mode(self, block01: ttnn.Tensor, report_density: bool) -> dict:
        c = self._consts
        geo = self.geo
        if report_density:
            # The block-level density (q_chunk x 64 keys) before the K-chunk union: the sparsity the model sees.
            self.last_block_density = self._host_density(block01, self._block_cols())
        # Any kept block in a K chunk keeps the chunk: [1,H,nqc_pad,nbg] @ [nbg,nkc_pad] -> [1,H,nqc_pad,nkc_pad]
        chunk01 = ttnn.gtz(ttnn.subtract(ttnn.matmul(block01, c["kc_pool"], compute_kernel_config=self.hifi2), 0.5))
        ttnn.deallocate(block01)
        if report_density:
            cols = torch.cat([torch.arange(s * geo.nkc, s * geo.nkc + geo.nkc) for s in range(geo.shards)])
            self.last_density = self._host_density(chunk01, cols)
        bias = None
        if self.emit_bias:
            # Blocks of kept chunks are exact (every valid token of the chunk is computed): bias NEG there.
            kept_blk = ttnn.multiply(
                ttnn.gtz(ttnn.subtract(ttnn.matmul(chunk01, c["kc_pool_t"], compute_kernel_config=self.hifi2), 0.5)),
                c["valid_row"],
            )
            bias_row = ttnn.add(ttnn.multiply(kept_blk, NEG), c["len_bias_row"])
            ttnn.deallocate(kept_blk)
            bias = ttnn.repeat_interleave(bias_row, TILE, dim=2)  # [1,H,nqc_pad*32,nbg]
            ttnn.deallocate(bias_row)
        mask_u32 = ttnn.to_layout(ttnn.typecast(chunk01, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(chunk01)
        return {"block_mask": mask_u32, "pooled_k": self._pooled_k, "pooled_v": self._pooled_v, "bias": bias}

    # Row-max constants depend only on (device, head dim, scale), not on the geometry: shared like _shared_consts.
    _shared_row_max_consts: dict = {}

    def _row_max(self, q: ttnn.Tensor, k: ttnn.Tensor, scale: float) -> ttnn.Tensor:
        head_dim = int(q.shape[3])
        key = (id(self.mesh_device), head_dim, scale)
        consts = SolPrepass._shared_row_max_consts.get(key)
        if consts is None:
            consts = {name: self._upload(t, shard=False) for name, t in row_max_constants(head_dim, scale).items()}
            SolPrepass._shared_row_max_consts[key] = consts
        return compute_row_max(
            q, k, consts=consts, ccl_manager=self.ccl_manager, sp_axis=self.sp_axis, compute_kernel_config=self.hifi4
        )

    def _gather(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather a pooled [1, H, nb_pad, D] tensor along dim 2 over SP: CCLManager.all_gather with this pre-pass's
        own semaphores."""
        ccl = self.ccl_manager
        if self._ag_sems is None:
            self._ag_sems = [
                [ttnn.create_global_semaphore(self.mesh_device, ccl.ccl_cores, 0) for _ in range(2)] for _ in range(2)
            ]
        sems = self._ag_sems[self._ag_idx]
        self._ag_idx ^= 1
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            barrier_semaphore=ccl.get_barrier_semaphore(self.sp_axis),
            dim=2,
            multi_device_global_semaphore=sems,
            num_links=ccl.num_links,
            topology=ccl.topology,
            cluster_axis=self.sp_axis,
        )

    def _pool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1,H,R,D] -> [1,H,nb_pad,D] block means (local shard)."""
        xt = ttnn.transpose(x, -2, -1)
        pt = ttnn.matmul(xt, self._consts["pool_t"], compute_kernel_config=self.hifi4)
        ttnn.deallocate(xt)
        p = ttnn.transpose(pt, -2, -1)
        ttnn.deallocate(pt)
        return p

    def __call__(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, *, scale: float, report_density: bool = False
    ) -> dict:
        c = self._consts
        geo = self.geo
        log2_scale = scale * LOG2E
        row_max = self._row_max(q, k, scale) if self.row_max else None  # [1,H,R,32] (resident mode)
        q_bar = self._pool(q)  # [1,H,nbp,D]
        kc_local = self._pool(k)
        vc_local = self._pool(v)
        kc = self._gather(kc_local)
        vc = self._gather(vc_local)
        ttnn.deallocate(kc_local)
        ttnn.deallocate(vc_local)

        nv = c["n_valid_blocks"]
        mean = ttnn.multiply(ttnn.sum(kc, dim=2, keepdim=True), 1.0 / nv)  # [1,H,1,D]
        kc2 = ttnn.multiply(kc, kc)
        e2 = ttnn.multiply(ttnn.sum(kc2, dim=2, keepdim=True), 1.0 / nv)
        ttnn.deallocate(kc2)
        var = ttnn.clamp(ttnn.subtract(e2, ttnn.multiply(mean, mean)), min=0.0)
        q_raw_mean = ttnn.sum(ttnn.multiply(q_bar, mean), dim=3, keepdim=True)  # [1,H,nbp,1]
        q_var = ttnn.sum(ttnn.multiply(ttnn.multiply(q_bar, q_bar), var), dim=3, keepdim=True)
        thr = ttnn.add(
            ttnn.multiply(q_raw_mean, log2_scale),
            ttnn.multiply(ttnn.sqrt(ttnn.add(ttnn.multiply(q_var, log2_scale * log2_scale), VAR_EPS)), self.tau),
        )
        kct = ttnn.transpose(kc, -2, -1)  # [1,H,D,nbg]
        scores = ttnn.multiply(ttnn.matmul(q_bar, kct, compute_kernel_config=self.hifi2), log2_scale)  # [1,H,nbp,nbg]
        ttnn.deallocate(kct)
        exact = ttnn.gtz(ttnn.subtract(scores, thr))  # W-broadcast of thr
        ttnn.deallocate(scores)
        exact = ttnn.add(ttnn.add(exact, c["band"]), c["sink_row"])
        exact = ttnn.multiply(ttnn.gtz(ttnn.subtract(exact, 0.5)), c["full_row"])
        # Union over the Q blocks of each Q chunk: (exact^T @ Qc)^T  -> [1,H,nqc_pad,nbg]
        exact_t = ttnn.transpose(exact, -2, -1)
        ttnn.deallocate(exact)
        m2t = ttnn.matmul(exact_t, c["qc_pool_t"], compute_kernel_config=self.hifi2)  # [1,H,nbg,nqc_pad]
        ttnn.deallocate(exact_t)
        m2 = ttnn.transpose(m2t, -2, -1)
        ttnn.deallocate(m2t)
        block01 = ttnn.multiply(ttnn.gtz(ttnn.subtract(ttnn.add(m2, c["force_row"]), 0.5)), c["full_row"])
        ttnn.deallocate(m2)
        if self.mode == "chunk":
            self._pooled_k, self._pooled_v = kc, vc
            out = self._finish_chunk_mode(block01, report_density)
            out["row_max"] = row_max
            return out
        # gather / compact: block-form mask
        if report_density:
            self.last_density = self._host_density(block01, self._block_cols())
            self.last_block_density = self.last_density
        bias = None
        if self.emit_bias:
            bias_row = ttnn.add(ttnn.multiply(block01, NEG), c["len_bias_row"])
            bias = ttnn.repeat_interleave(bias_row, TILE, dim=2)  # [1,H,nqc_pad*32,nbg]
            ttnn.deallocate(bias_row)
        mask_u32 = ttnn.to_layout(ttnn.typecast(block01, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(block01)
        return {
            "block_mask": mask_u32,
            "pooled_k": kc,
            "pooled_v": vc,
            "bias": bias,
            "q_bar": q_bar,
            "row_max": row_max,
        }
