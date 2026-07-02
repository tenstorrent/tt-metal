# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN single-graph consolidation — Stage 4 (perception path).

Stages 3.4–3.7 made every weight-bearing op run on TTNN, but each drop-in
converted torch→device→torch at its own boundary and the thin tensor glue in
``DiffusionDriveModel.forward`` (the keyval cat/permute, the ``concat_cross_bev``
``F.interpolate`` and the cross-feature cat) stayed on host.  In particular the
bilinear ``F.interpolate`` at ``reference/model.py:974`` was the **last non-glue
PyTorch compute op** in the whole forward path.

``TtnnPerceptionForward`` consolidates ``DiffusionDriveModel.forward`` lines
955–984 into one on-device sequence::

    bev_downscale (1×1 conv 512→256) ─┐
    status_encoding (linear 8→256)   ─┴─► keyval = cat + embedding-add     (device)
    keyval[:-1] ─► reshape NHWC ─► ttnn.upsample(bilinear ×8)  ◄─ kills :974 (device)
    cat([upsampled, bev_upscale]) ─► bev_proj (linear+relu+ln)            (device)
    query_embedding ─► tf_decoder (3 layers) ─► split                     (device)

It reuses the weight tensors already prepared by the Stage-3.4 drop-ins (no
re-conversion).  Only the backbone→perception input boundary (one H2D) and the
perception→trajectory-head output boundary (one D2H) cross the host; the entire
perception block is a single device graph with ``concat_cross_bev`` upsampled
on device.

Valid only at the production resolution (bev_feature 8×8, bev_upscale 64×64 →
integer ×8 upsample), matching the Stage-3.6 fusion constraint; asserts
divisibility otherwise.  All math in bfloat16; PCC ≥ 0.99 vs the reference.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.diffusion_drive.tt.ttnn_backbone import _to_host_tile, _to_ttnn_tile
from models.demos.diffusion_drive.tt.ttnn_resnet34 import _ttnn_conv2d

# ---------------------------------------------------------------------------
# Layout helpers (avg_pool/upsample/conv auto-shard; reshapes want ROW_MAJOR)
# ---------------------------------------------------------------------------


def _interleave(x):
    if x.is_sharded():
        return ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    return x


def _clean_tile(x):
    """Interleaved DRAM + TILE — safe for matmul/linear/layer_norm/add."""
    return ttnn.to_layout(_interleave(x), ttnn.TILE_LAYOUT)


def _to_rm(x):
    """Interleaved DRAM + ROW_MAJOR — safe for rank-changing reshape/slice."""
    return ttnn.to_layout(_interleave(x), ttnn.ROW_MAJOR_LAYOUT)


def _to_dev_rm(x: torch.Tensor, device):
    """torch (B,C,H,W) → ttnn ROW_MAJOR (1,1,B*H*W,C)."""
    return ttnn.from_torch(_nhwc_flat(x), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _to_host_rm(x: torch.Tensor):
    """Host twin of ``_to_dev_rm`` (no ``device=``) for trace-input refill via
    ``ttnn.copy_host_to_device_tensor``."""
    return ttnn.from_torch(_nhwc_flat(x), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _nhwc_flat(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).contiguous().reshape(1, 1, B * H * W, C).to(torch.bfloat16)


class TtnnPerceptionForward:
    """Consolidated on-device perception path (DiffusionDriveModel.forward 955–984).

    Parameters
    ----------
    model : DiffusionDriveModel
        Model whose perception submodules have already been swapped to the
        Stage-3.4 TTNN drop-ins (``build_stage3_4``).  Their prepared weight
        tensors are reused directly — no re-conversion.
    device : ttnn.Device
    """

    def __init__(self, model, device) -> None:
        self._d = device

        bd = model._bev_downscale  # TtnnConv1x1
        self._bd_w, self._bd_b, self._bd_cout = bd._w, bd._b, bd._cout
        se = model._status_encoding  # TtnnLinear
        self._se_w, self._se_b = se._w, se._b
        bp = model.bev_proj  # TtnnBevProj
        self._bp_lw, self._bp_lb = bp._lw, bp._lb
        self._bp_g, self._bp_beta, self._bp_eps = bp._g, bp._beta, bp._eps
        self._tf_layers = model._tf_decoder._layers  # list[_TtnnDecoderLayer], device-in/out

        emb = model._keyval_embedding  # nn.Embedding (not swapped)
        self._keyval_emb = ttnn.from_torch(
            emb.weight.detach().reshape(1, emb.num_embeddings, emb.embedding_dim).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._query_w = model._query_embedding.weight.detach().to(torch.bfloat16)  # (Nq, D) torch
        self._query_splits = list(model._query_splits)

        # Stage 8: perception-forward trace state (a second trace after the backbone).
        # The query embedding is a per-forward constant; pre-lift it to device once so
        # the traced region contains no H2D (illegal between begin/end_trace_capture).
        self._query_dev = None  # (B, Nq, D) TILE device tensor, created lazily per B
        self._shapes = None  # cached shape scalars (production res is fixed)
        self._perc_trace_id = None
        self._perc_in = None  # (bev_feat_t, bev_up_t, status_t) fixed-address device inputs
        self._perc_out = None  # (traj_q, agents_q, cbf_nhwc, status_enc) fixed device outputs

    def __call__(self, bev_upscale: torch.Tensor, bev_feature: torch.Tensor, status_feat: torch.Tensor):
        """Eager path: lift inputs (H2D) → device core → read outputs (D2H).

        Returns (traj_query, agents_query, cross_bev_feature, status_enc) as torch."""
        self._shapes = self._compute_shapes(bev_upscale, bev_feature, status_feat)
        self._ensure_query_dev(self._shapes["B"])
        bev_feat_t, bev_up_t, status_t = self._lift(bev_upscale, bev_feature, status_feat)
        q_rm, cbf_nhwc, status_enc = self._forward_dev(bev_feat_t, bev_up_t, status_t)
        return self._read(q_rm, cbf_nhwc, status_enc)

    # ------------------------------------------------------------------
    # Boundaries (host↔device) — kept out of the traceable device core
    # ------------------------------------------------------------------

    def _compute_shapes(self, bev_upscale, bev_feature, status_feat) -> dict:
        Cf, hb, wb = int(bev_feature.shape[1]), int(bev_feature.shape[2]), int(bev_feature.shape[3])
        Cu, Hb, Wb = int(bev_upscale.shape[1]), int(bev_upscale.shape[2]), int(bev_upscale.shape[3])
        assert Hb % hb == 0 and Wb % wb == 0, (
            f"consolidated perception needs an integer upsample ratio (got {hb}×{wb} → {Hb}×{Wb}); "
            "Stage-4 is only valid at production resolution (bev 8×8 → 64×64)."
        )
        return dict(
            B=int(status_feat.shape[0]),
            Cf=Cf,
            hb=hb,
            wb=wb,
            Cu=Cu,
            Hb=Hb,
            Wb=Wb,
            D=self._bd_cout,
            n_tok=hb * wb,
            sh=Hb // hb,
            sw=Wb // wb,
            Nq=int(self._query_w.shape[0]),
        )

    def _ensure_query_dev(self, B: int) -> None:
        """Pre-lift the (constant) query embedding to device once (no per-forward H2D
        — required so the traced core contains no host→device write)."""
        if self._query_dev is None:
            qw = self._query_w.unsqueeze(0).expand(B, -1, -1).contiguous()
            self._query_dev = ttnn.from_torch(qw, layout=ttnn.TILE_LAYOUT, device=self._d)

    def _lift(self, bev_upscale, bev_feature, status_feat):
        s = self._shapes
        bev_feat_t = _to_ttnn_tile(bev_feature, s["B"], s["hb"], s["wb"], s["Cf"], self._d)
        bev_up_t = _to_dev_rm(bev_upscale, self._d)  # (1,1,B*Hb*Wb,Cu) ROW_MAJOR
        status_t = ttnn.from_torch(status_feat.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        return bev_feat_t, bev_up_t, status_t

    def _read(self, q_rm, cbf_nhwc, status_enc):
        s = self._shapes
        B, D, Hb, Wb, Nq, n_traj = s["B"], s["D"], s["Hb"], s["Wb"], s["Nq"], self._query_splits[0]
        q = ttnn.to_torch(q_rm).reshape(B, Nq, D).float()
        traj_query = q[:, :n_traj].contiguous()
        agents_query = q[:, n_traj:].contiguous()
        cross_bev_feature = ttnn.to_torch(cbf_nhwc).reshape(B, Hb, Wb, D).permute(0, 3, 1, 2).float()
        status_torch = ttnn.to_torch(status_enc).reshape(B, 1, D).float()
        return traj_query, agents_query, cross_bev_feature, status_torch

    # ------------------------------------------------------------------
    # Device core — a single static graph, trace-legal (no host hops inside)
    # ------------------------------------------------------------------

    def _forward_dev(self, bev_feat_t, bev_up_t, status_t):
        """(bev_feat_t TILE, bev_up_t ROW_MAJOR, status_t TILE) → (q_rm, cbf_nhwc, status_enc)
        device tensors. Same op sequence as the original __call__; the input H2D and
        output D2H are hoisted into _lift/_read so this is one traceable graph."""
        d = self._d
        s = self._shapes
        B, Cf, hb, wb, Cu, Hb, Wb, D = s["B"], s["Cf"], s["hb"], s["wb"], s["Cu"], s["Hb"], s["Wb"], s["D"]
        n_tok, sh, sw, Nq = s["n_tok"], s["sh"], s["sw"], s["Nq"]

        # 1. bev_downscale: 1×1 conv 512→D → tokens (B, n_tok, D) (NHWC token order).
        bev_d, _, _, self._bd_w, self._bd_b = _ttnn_conv2d(
            d, bev_feat_t, self._bd_w, self._bd_b, B, hb, wb, Cf, D, 1, 1, 0
        )
        bev_flat = _clean_tile(ttnn.reshape(_to_rm(bev_d), (B, n_tok, D)))

        # 2. status_encoding: linear 8→D → (B,1,D)
        status_enc = ttnn.reshape(ttnn.linear(status_t, self._se_w, bias=self._se_b), (B, 1, D))

        # 3. keyval = cat([bev_flat, status_enc]) + keyval_embedding.
        keyval = _clean_tile(ttnn.concat([_to_rm(bev_flat), _to_rm(status_enc)], dim=1))
        keyval = ttnn.add(keyval, self._keyval_emb)

        # 4. concat_cross_bev: drop status token, NHWC, bilinear upsample ×(sh,sw) on device.
        cb = ttnn.slice(_to_rm(keyval), [0, 0, 0], [B, n_tok, D])
        cb = ttnn.reshape(cb, (B, hb, wb, D))
        cb_up = ttnn.upsample(cb, [sh, sw], mode="bilinear")

        # 5. cross_bev_feature = bev_proj(cat([concat_cross_bev, bev_upscale], channels)).
        cb_tok = ttnn.reshape(_to_rm(cb_up), (B, Hb * Wb, D))
        bu_tok = ttnn.reshape(bev_up_t, (B, Hb * Wb, Cu))
        tok = _clean_tile(ttnn.concat([cb_tok, bu_tok], dim=2))
        y = ttnn.linear(tok, self._bp_lw, bias=self._bp_lb, activation="relu")  # fused ReLU (bev_proj)
        cbf = ttnn.layer_norm(y, weight=self._bp_g, bias=self._bp_beta, epsilon=self._bp_eps)
        cbf_nhwc = ttnn.reshape(_to_rm(cbf), (B, Hb, Wb, D))

        # 6. perception decoder: query_embedding → 3× TransformerDecoderLayer.
        q = self._query_dev
        Skv = n_tok + 1
        for layer in self._tf_layers:
            q = layer(q, keyval, B, Nq, Skv)
        return _to_rm(q), cbf_nhwc, status_enc

    # ------------------------------------------------------------------
    # Stage 8: perception-forward trace capture / replay
    # ------------------------------------------------------------------

    def capture_trace(self, bev_upscale, bev_feature, status_feat) -> None:
        """Capture the perception device core as a replayable trace.

        Pre-allocates the three fixed-address device inputs (bev_feature/bev_upscale/
        status), double-warms ``_forward_dev`` so every kernel variant is JIT-built,
        then records ``clone(inputs) → _forward_dev → outputs``. The clone inside the
        captured region lets ``run_trace`` refill the persistent inputs (a legal H2D
        outside capture) before each replay. Mirrors the backbone trace."""
        self._shapes = self._compute_shapes(bev_upscale, bev_feature, status_feat)
        self._ensure_query_dev(self._shapes["B"])
        self._perc_in = self._lift(bev_upscale, bev_feature, status_feat)

        def _traced():
            ins = tuple(ttnn.clone(t) for t in self._perc_in)
            return self._forward_dev(*ins)

        for _ in range(2):  # double warm-up populates the program cache for all kernels
            outs = _traced()
            for t in outs:
                ttnn.deallocate(t)
        ttnn.synchronize_device(self._d)

        try:
            self._perc_trace_id = ttnn.begin_trace_capture(self._d, cq_id=0)
            self._perc_out = _traced()
            ttnn.end_trace_capture(self._d, self._perc_trace_id, cq_id=0)
        except Exception:
            if self._perc_trace_id is not None:
                try:
                    ttnn.release_trace(self._d, self._perc_trace_id)
                except Exception:
                    pass
            self._perc_trace_id = None
            raise

    def run_trace(self, bev_upscale, bev_feature, status_feat):
        """Refill the fixed inputs, replay the perception trace, read the outputs."""
        if self._perc_trace_id is None:
            raise RuntimeError("run_trace called before capture_trace")
        s = self._shapes
        ttnn.copy_host_to_device_tensor(_to_host_tile(bev_feature, s["B"], s["hb"], s["wb"], s["Cf"]), self._perc_in[0])
        ttnn.copy_host_to_device_tensor(_to_host_rm(bev_upscale), self._perc_in[1])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(status_feat.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT), self._perc_in[2]
        )
        ttnn.execute_trace(self._d, self._perc_trace_id, cq_id=0, blocking=True)
        return self._read(*self._perc_out)

    def release_trace(self) -> None:
        if self._perc_trace_id is not None:
            try:
                ttnn.release_trace(self._d, self._perc_trace_id)
            finally:
                self._perc_trace_id = None
