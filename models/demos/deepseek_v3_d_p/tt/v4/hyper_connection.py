# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connections in ttnn: ``TtHyperConnection`` (both mixing sites of every V4 layer) and ``TtHyperHead``.

The op sequence is exactly ``tt/v4/mhc_math.py`` (proven against the reference device-free); read that module's
docstring for the algebra. Everything runs in fp32 -- the reference keeps the whole module in fp32
(``_keep_in_fp32_modules_strict``) and the ``fn`` contraction spans H*D = 16384 elements.

Residual streams are a LIST of ``H = 4`` tensors ``[1, 1, S_l, D_l]`` (bf16, TP-sharded on D, SP-sharded on S),
never one ``[1, 4, S_l, D_l]`` tensor: per-stream matmuls need no slicing and the PP pack is a plain concat.

TP: the norm-folded logit row is a per-token ``[S_l, 32]`` partial sum over the local D slice; ONE all-gather on
dim 1 + ``fast_reduce_nc`` finishes it (the pattern of ``TtIndexer._tp_all_reduce_via_gather``). Nothing else in
the module communicates.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.v4 import mhc_math as M


class _TtHyperBase(LightweightModule):
    def __init__(self, device, *, hidden: int, rms_eps: float, hc_eps: float, sp_axis: int, tp_axis: int):
        super().__init__()
        self.device = device
        self.hidden = int(hidden)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.ccl_num_links = 2 if is_blackhole() else 1
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.fp32 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._gather_bufs: dict = {}
        # STACKED mode (PREFILL_MHC_STACKED=1): the four streams are concatenated into ONE [1, 4, S_l, D_l] tensor inside
        # the module, so the per-stream loops become single batched ops (matmul over the batch dim, broadcast multiply +
        # fast_reduce_nc). Same algebra, ~5x fewer programs per site: the block is device-bound on program count once
        # the attention is traced (DS4F-0246: 574-896 programs per layer, ~280 of them these fp32 mHC ops).
        self.stacked = os.environ.get("PREFILL_MHC_STACKED", "0") == "1"
        self._sel_consts: dict = {}

    # ---- host -> device --------------------------------------------------------------------------------------
    def _mesh_mapper(self, sp_dim=None, tp_dim=None):
        if not self.is_mesh:
            return None
        dims = [None, None]
        if sp_dim is not None and self.sp_factor > 1:
            dims[self.sp_axis] = sp_dim
        if tp_dim is not None and self.tp_factor > 1:
            dims[self.tp_axis] = tp_dim
        if dims == [None, None]:
            return ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)

    def _const(self, x: torch.Tensor, *, tp_dim=None):
        """An fp32 TILE constant, replicated unless ``tp_dim`` names the axis to split over TP."""
        return ttnn.from_torch(
            x.float().contiguous(),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=self._mesh_mapper(tp_dim=tp_dim),
        )

    def _upload_row_constants(self, W: dict):
        # FN / ONES: [H, D, 32] -> per stream [1, 1, D_l, 32], TP-split on D
        self.FN = [self._const(W["FN"][h].unsqueeze(0).unsqueeze(0), tp_dim=2) for h in range(M.HC)]
        self.FN_STACK = self._const(W["FN"].view(1, M.HC, W["FN"].shape[1], M.ROW), tp_dim=2)  # [1, H, D_l, 32]
        self.ONES = self._const(W["ONES"][0].unsqueeze(0).unsqueeze(0), tp_dim=2)
        self.SV = self._const(W["SV"].view(1, 1, 1, M.ROW))
        self.BV = self._const(W["BV"].view(1, 1, 1, M.ROW))
        self.MS = self._const(W["MS"].view(1, 1, M.ROW, M.ROW))

    # ---- the fp32 helpers --------------------------------------------------------------------------------------
    def _mm(self, a, b):
        return ttnn.matmul(a, b, dtype=ttnn.float32, memory_config=self.memory_config, compute_kernel_config=self.fp32)

    def _tp_all_reduce(self, part):
        """[1, 1, S_l, 32] fp32 partial -> the TP sum, via all-gather on dim 1 + local reduce."""
        if self.tp_factor == 1:
            return part
        key = tuple(part.shape)
        buf = self._gather_bufs.get(key)
        if buf is None:
            buf = ttnn.from_torch(
                torch.zeros(1, self.tp_factor, key[2], key[3]),
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
                mesh_mapper=self._mesh_mapper(),
            )
            self._gather_bufs[key] = buf
        g = ttnn.experimental.high_bw_all_gather(
            part, dim=1, output_tensor=buf, num_links=self.ccl_num_links, cluster_axis=self.tp_axis
        )
        return ttnn.experimental.fast_reduce_nc(g, dims=[1], output=None, compute_kernel_config=self.fp32)

    def _mix_row(self, xf: list):
        """Steps 2-4 of mhc_math: the norm-folded, TP-reduced ``[1, 1, S_l, 32]`` logit row from the fp32 streams."""
        part = None
        for h, x in enumerate(xf):
            p = self._mm(x, self.FN[h])
            s = self._mm(ttnn.multiply(x, x), self.ONES)
            part = ttnn.add(p, s) if part is None else ttnn.add(part, ttnn.add(p, s))
        part = self._tp_all_reduce(part)
        ms = self._mm(part, self.MS)
        rstd = ttnn.rsqrt(ttnn.add(ms, self.rms_eps))
        return ttnn.multiply(part, rstd)

    # ---- stacked helpers (PREFILL_MHC_STACKED) -----------------------------------------------------------------
    def _stack(self, streams: list):
        """[1, H, S_l, D_l] fp32 from the H bf16 streams (2 programs)."""
        return ttnn.typecast(ttnn.concat(streams, dim=1), ttnn.float32)

    def _mix_row_stacked(self, Xf):
        """``_mix_row`` on the stacked streams: batched fn matmul + squares in 5 programs instead of 5 per stream."""
        p = self._mm(Xf, self.FN_STACK)  # [1, H, S_l, 32], per-stream fn
        s_ = self._mm(ttnn.multiply(Xf, Xf), self.ONES)  # ONES broadcast over the stream dim
        part = ttnn.experimental.fast_reduce_nc(
            ttnn.add(p, s_), dims=[1], output=None, compute_kernel_config=self.fp32
        )  # [1, 1, S_l, 32]
        part = self._tp_all_reduce(part)
        ms = self._mm(part, self.MS)
        rstd = ttnn.rsqrt(ttnn.add(ms, self.rms_eps))
        return ttnn.multiply(part, rstd)

    def _select_cols(self, row, cols: list):
        """``row[:, cols]`` as a [1, 1, S_l, len(cols)] tensor via one constant selection matmul (any column order)."""
        key = (int(row.shape[3]), tuple(cols))
        sel = self._sel_consts.get(key)
        if sel is None:
            m = torch.zeros(1, 1, int(row.shape[3]), len(cols))
            for j, c in enumerate(cols):
                m[0, 0, c, j] = 1.0
            sel = self._sel_consts[key] = self._const(m)
        return self._mm(row, sel)

    def _weighted_sum_stacked(self, weights_cols, Xf):
        """sum_h weights[:, h] * Xf[h]: weights [1, 1, S_l, H] -> [1, H, S_l, 1] (per-stream column), broadcast
        multiply over D, reduce over the stream dim -> [1, 1, S_l, D_l] fp32 (3 programs instead of 3 per stream)."""
        w = ttnn.permute(weights_cols, (0, 3, 2, 1))  # [1, H, S_l, 1]
        prod = ttnn.multiply(Xf, w)
        return ttnn.experimental.fast_reduce_nc(prod, dims=[1], output=None, compute_kernel_config=self.fp32)

    @staticmethod
    def _col(row, c: int):
        """Column ``c`` of a ``[1, 1, S_l, 32]`` row as ``[1, 1, S_l, 1]`` (broadcast operand)."""
        return ttnn.slice(row, [0, 0, 0, c], [1, 1, row.shape[2], c + 1])

    def _weighted_sum(self, weights_row, first_col: int, xf: list):
        """sum_h weights_row[:, first_col + h] * xf[h]  (fp32)."""
        acc = None
        for h, x in enumerate(xf):
            t = ttnn.multiply(x, self._col(weights_row, first_col + h))
            acc = t if acc is None else ttnn.add(acc, t)
        return acc


class TtHyperConnection(_TtHyperBase):
    """One mHC site. ``forward(streams) -> (post_row, comb_row, collapsed)``; ``mix(streams, y, post_row, comb_row)
    -> new streams``. ``post_row``/``comb_row`` are the fp32 ``[1, 1, S_l, 32]`` rows of mhc_math (columns 4..7 and
    8..23 meaningful); ``collapsed`` and the mixed streams are bf16 ``[1, 1, S_l, D_l]``."""

    def __init__(
        self,
        device,
        *,
        hidden: int,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        rms_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        super().__init__(device, hidden=hidden, rms_eps=rms_eps, hc_eps=hc_eps, sp_axis=sp_axis, tp_axis=tp_axis)
        self.sinkhorn_iters = int(sinkhorn_iters)
        W = M.prep_hyper_connection(fn, base, scale, self.hidden)
        K = M.sinkhorn_constants(self.hc_eps)
        self._upload_row_constants(W)
        self.COMB_MASK = self._const(W["COMB_MASK"].view(1, 1, 1, M.ROW))
        self.ONE_COL = self._const(W["ONE_COL"].view(1, 1, 1, M.ROW))
        self.R_SOFT = self._const(W["R_SOFT"].view(1, 1, M.ROW, M.ROW))
        self.R_AUG = self._const(K["R_AUG"].view(1, 1, M.ROW, M.ROW))
        self.C_AUG = self._const(K["C_AUG"].view(1, 1, M.ROW, M.ROW))
        self.EPS_COMB = self._const(K["EPS_COMB"].view(1, 1, 1, M.ROW))

    @classmethod
    def from_reference(cls, device, ref, config, **kw):
        return cls(
            device,
            hidden=config.hidden_size,
            fn=ref.fn.detach(),
            base=ref.base.detach(),
            scale=ref.scale.detach(),
            rms_eps=config.rms_norm_eps,
            hc_eps=config.hc_eps,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            **kw,
        )

    def forward(self, streams: list):
        assert len(streams) == M.HC
        if self.stacked:
            xf = None
            Xf = self._stack(streams)
            mix = self._mix_row_stacked(Xf)
        else:
            xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
            mix = self._mix_row(xf)
        aff = ttnn.add(ttnn.multiply(mix, self.SV), self.BV)
        sig = ttnn.sigmoid(aff)
        pre = ttnn.add(sig, self.hc_eps)
        post = ttnn.multiply(sig, 2.0)
        # comb: exact row softmax on the 16 comb columns, then Sinkhorn with the eps folded into the sum matrices
        E = ttnn.add(ttnn.multiply(ttnn.exp(ttnn.clamp(aff, -M.EXP_CLAMP, M.EXP_CLAMP)), self.COMB_MASK), self.ONE_COL)
        X = ttnn.div(E, self._mm(E, self.R_SOFT))
        X = ttnn.add(X, self.EPS_COMB)
        X = ttnn.div(X, self._mm(X, self.C_AUG))
        for _ in range(self.sinkhorn_iters - 1):
            X = ttnn.div(X, self._mm(X, self.R_AUG))
            X = ttnn.div(X, self._mm(X, self.C_AUG))
        if self.stacked:
            w = self._select_cols(pre, list(range(M.PRE0, M.PRE0 + M.HC)))
            collapsed = ttnn.typecast(self._weighted_sum_stacked(w, Xf), streams[0].dtype)
            ttnn.deallocate(Xf)
            return post, X, collapsed
        collapsed = ttnn.typecast(self._weighted_sum(pre, M.PRE0, xf), streams[0].dtype)
        for x in xf:
            ttnn.deallocate(x)
        return post, X, collapsed

    def mix(self, streams: list, y, post_row, comb_row) -> list:
        """out_k = post[:, 4+k] * y + sum_j comb[j, k] * x_j   (comb consumed transposed, as the reference)."""
        assert len(streams) == M.HC
        if self.stacked:
            return self._mix_stacked(streams, y, post_row, comb_row)
        xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
        yf = ttnn.typecast(y, ttnn.float32)
        out = []
        for k in range(M.HC):
            acc = ttnn.multiply(yf, self._col(post_row, M.POST0 + k))
            for j, x in enumerate(xf):
                acc = ttnn.add(acc, ttnn.multiply(x, self._col(comb_row, M.comb_index(j, k))))
            out.append(ttnn.typecast(acc, streams[0].dtype))
            ttnn.deallocate(acc)
        for x in xf:
            ttnn.deallocate(x)
        ttnn.deallocate(yf)
        return out

    def _mix_stacked(self, streams: list, y, post_row, comb_row) -> list:
        """The stacked ``mix``: y joins the streams as a 5th slab, the (comb | post) rows are concatenated once and the
        20 weights re-ordered k-major by one selection matmul; each output stream is then ONE broadcast multiply + ONE
        stream-dim reduce (24 programs for the site instead of ~65)."""
        X5 = self._stack(streams + [y])  # [1, H+1, S_l, D_l] fp32
        rows = ttnn.concat([comb_row, post_row], dim=3)  # [1, 1, S_l, 64]: comb cols 0..31, post cols 32..63
        cols = []
        for k in range(M.HC):
            cols += [M.comb_index(j, k) for j in range(M.HC)] + [M.ROW + M.POST0 + k]
        W = self._select_cols(rows, cols)  # [1, 1, S_l, (H+1)*H], k-major
        out = []
        for k in range(M.HC):
            wk = ttnn.slice(W, [0, 0, 0, k * (M.HC + 1)], [1, 1, W.shape[2], (k + 1) * (M.HC + 1)])
            acc = self._weighted_sum_stacked(wk, X5)
            out.append(ttnn.typecast(acc, streams[0].dtype))
            ttnn.deallocate(acc)
        ttnn.deallocate(X5)
        return out


class TtHyperHead(_TtHyperBase):
    """The final stream collapse before the model norm: the pre-branch of a HyperConnection alone."""

    def __init__(
        self,
        device,
        *,
        hidden: int,
        hc_fn: torch.Tensor,
        hc_base: torch.Tensor,
        hc_scale: torch.Tensor,
        rms_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        super().__init__(device, hidden=hidden, rms_eps=rms_eps, hc_eps=hc_eps, sp_axis=sp_axis, tp_axis=tp_axis)
        self._upload_row_constants(M.prep_hyper_head(hc_fn, hc_base, hc_scale, self.hidden))

    @classmethod
    def from_reference(cls, device, ref, config, **kw):
        return cls(
            device,
            hidden=config.hidden_size,
            hc_fn=ref.hc_fn.detach(),
            hc_base=ref.hc_base.detach(),
            hc_scale=ref.hc_scale.detach(),
            rms_eps=config.rms_norm_eps,
            hc_eps=config.hc_eps,
            **kw,
        )

    def forward(self, streams: list):
        if self.stacked:
            Xf = self._stack(streams)
            mix = self._mix_row_stacked(Xf)
            pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mix, self.SV), self.BV)), self.hc_eps)
            out = ttnn.typecast(
                self._weighted_sum_stacked(self._select_cols(pre, list(range(M.HC))), Xf), streams[0].dtype
            )
            ttnn.deallocate(Xf)
            return out
        xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
        mix = self._mix_row(xf)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mix, self.SV), self.BV)), self.hc_eps)
        out = ttnn.typecast(self._weighted_sum(pre, 0, xf), streams[0].dtype)
        for x in xf:
            ttnn.deallocate(x)
        return out
