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
from models.demos.diffusion_drive.tt.ttnn_backbone import _to_ttnn_tile
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
    B, C, H, W = x.shape
    nhwc = x.permute(0, 2, 3, 1).contiguous().reshape(1, 1, B * H * W, C).to(torch.bfloat16)
    return ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


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

    def __call__(self, bev_upscale: torch.Tensor, bev_feature: torch.Tensor, status_feat: torch.Tensor):
        """Returns (traj_query, agents_query, cross_bev_feature, status_enc) as torch tensors."""
        d = self._d
        B = status_feat.shape[0]
        Cf, hb, wb = int(bev_feature.shape[1]), int(bev_feature.shape[2]), int(bev_feature.shape[3])
        Cu, Hb, Wb = int(bev_upscale.shape[1]), int(bev_upscale.shape[2]), int(bev_upscale.shape[3])
        D = self._bd_cout
        n_tok = hb * wb
        assert Hb % hb == 0 and Wb % wb == 0, (
            f"consolidated perception needs an integer upsample ratio (got {hb}×{wb} → {Hb}×{Wb}); "
            "Stage-4 is only valid at production resolution (bev 8×8 → 64×64)."
        )
        sh, sw = Hb // hb, Wb // wb

        # 1. bev_downscale: 1×1 conv 512→D on (B,Cf,hb,wb) → tokens (B, n_tok, D).
        #    The conv output is row-major NHWC (token = h*wb+w), which already
        #    equals the reference's flatten(-2,-1).permute(0,2,1) ordering.
        xt = _to_ttnn_tile(bev_feature, B, hb, wb, Cf, d)
        bev_d, _, _, self._bd_w, self._bd_b = _ttnn_conv2d(d, xt, self._bd_w, self._bd_b, B, hb, wb, Cf, D, 1, 1, 0)
        bev_flat = _clean_tile(ttnn.reshape(_to_rm(bev_d), (B, n_tok, D)))  # (B, n_tok, D)

        # 2. status_encoding: linear 8→D → (B,1,D)
        st = ttnn.from_torch(status_feat.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
        status_enc = ttnn.reshape(ttnn.linear(st, self._se_w, bias=self._se_b), (B, 1, D))

        # 3. keyval = cat([bev_flat, status_enc]) + keyval_embedding.
        #    Token-dim cat with a size-1 tail → do it ROW_MAJOR (avoids the
        #    32-tile-alignment constraint on the concatenated dim), then TILE.
        keyval = _clean_tile(ttnn.concat([_to_rm(bev_flat), _to_rm(status_enc)], dim=1))  # (B, n_tok+1, D)
        keyval = ttnn.add(keyval, self._keyval_emb)  # broadcast (1, n_tok+1, D)

        # 4. concat_cross_bev: drop the status token, view as NHWC, bilinear upsample
        #    ×(sh,sw) on device — this replaces reference/model.py:974 F.interpolate.
        cb = ttnn.slice(_to_rm(keyval), [0, 0, 0], [B, n_tok, D])  # (B, n_tok, D)
        cb = ttnn.reshape(cb, (B, hb, wb, D))  # NHWC
        cb_up = ttnn.upsample(cb, [sh, sw], mode="bilinear")  # (B, Hb, Wb, D)

        # 5. cross_bev_feature = bev_proj(cat([concat_cross_bev, bev_upscale], channels)).
        #    Build per-pixel tokens (B, Hb*Wb, D+Cu) and run linear+relu+layer_norm.
        cb_tok = ttnn.reshape(_to_rm(cb_up), (B, Hb * Wb, D))  # (B, HW, D)
        bu_tok = ttnn.reshape(_to_dev_rm(bev_upscale, d), (B, Hb * Wb, Cu))  # (B, HW, Cu)
        tok = _clean_tile(ttnn.concat([cb_tok, bu_tok], dim=2))  # (B, HW, D+Cu)
        y = ttnn.linear(tok, self._bp_lw, bias=self._bp_lb, activation="relu")  # fused ReLU (bev_proj)
        cbf = ttnn.layer_norm(y, weight=self._bp_g, bias=self._bp_beta, epsilon=self._bp_eps)  # (B, HW, D)
        cbf_nhwc = ttnn.reshape(_to_rm(cbf), (B, Hb, Wb, D))
        cross_bev_feature = ttnn.to_torch(cbf_nhwc).reshape(B, Hb, Wb, D).permute(0, 3, 1, 2).float()

        # 6. perception decoder: query_embedding → 3× TransformerDecoderLayer → split
        Nq = self._query_w.shape[0]
        q = ttnn.from_torch(
            self._query_w.unsqueeze(0).expand(B, -1, -1).contiguous(), layout=ttnn.TILE_LAYOUT, device=d
        )
        Skv = n_tok + 1
        for layer in self._tf_layers:
            q = layer(q, keyval, B, Nq, Skv)
        q_rm = _to_rm(q)
        n_traj = self._query_splits[0]
        traj_query = ttnn.to_torch(ttnn.slice(q_rm, [0, 0, 0], [B, n_traj, D])).reshape(B, n_traj, D).float()
        agents_query = ttnn.to_torch(ttnn.slice(q_rm, [0, n_traj, 0], [B, Nq, D])).reshape(B, Nq - n_traj, D).float()

        status_torch = ttnn.to_torch(status_enc).reshape(B, 1, D).float()
        return traj_query, agents_query, cross_bev_feature, status_torch
