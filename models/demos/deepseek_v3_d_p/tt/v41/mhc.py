# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 hyper-connection residual (graph nodes B1, B2, B17, B18, B23, F1).

V4.1 lags the collapse by one sublayer (``inference/model.py`` ``Block.forward``): attention collapses the
streams with the ``pre`` the previous block's FFN produced (a one-hot on copy 0 before the first block), the
FFN collapses with the ``pre`` split before attention, and the block hands the ``pre`` split before its FFN
to the next block. After the last block the model collapses with that last ``pre``; there is no head mix.

Streams are ``[1, 1, S/sp, hc_mult * hidden/tp]`` fp32 (per chip: its hidden slice of every stream);
``pre_mix`` is ``[1, 1, S/sp, hc_mult]`` fp32.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

SUBLAYER_DTYPE = ttnn.bfloat16


def mhc_config(config) -> MHCConfig:
    return MHCConfig(
        dim=config.EMB_SIZE,
        n=config.HC_MULT,
        sinkhorn_iters=config.HC_SINKHORN_ITERS,
        eps=config.HC_EPS,
        norm_eps=config.RMS_NORM_EPS,
    )


def initial_pre_mix(mesh_device, config, tokens: int) -> ttnn.Tensor:
    """One-hot on copy 0 for the first block's attention, token-sharded over SP."""
    pre = torch.zeros(1, 1, tokens, config.HC_MULT)
    pre[..., 0] = 1.0
    return ttnn.from_torch(
        pre,
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(2, None)),
    )


class TtV41HyperConnections(LightweightModule):
    """The two hyper-connection sites of one block, applied around caller-provided sublayers."""

    def __init__(self, mesh_device, config, hc_attn, hc_ffn):
        """``hc_attn`` / ``hc_ffn``: the checkpoint's ``(fn [24, 4*dim], base [24], scale [3])`` per site."""
        cfg = mhc_config(config)
        topology = per_axis_topology()[1]  # the TP axis (tp_axis=1) of the opened fabric
        self.attn_site = TtMHCWrap(mesh_device, cfg, *hc_attn, tp_axis=1, topology=topology)
        self.ffn_site = TtMHCWrap(mesh_device, cfg, *hc_ffn, tp_axis=1, topology=topology)

    @staticmethod
    def _sublayer(fn, h):
        """Run a bf16 sublayer on the fp32 collapsed stream."""
        out = fn(ttnn.typecast(h, SUBLAYER_DTYPE))
        return ttnn.typecast(out, ttnn.float32)

    def forward(self, x, pre_mix, attention, ffn):
        """(streams, incoming pre_mix) -> (streams, pre_mix for the next block).

        ``attention`` and ``ffn`` map a collapsed ``[1, 1, S/sp, hidden/tp]`` bf16 stream to the same shape;
        the norms are theirs."""
        attn_pre, attn_post, attn_comb = self.attn_site.split(x)
        h = self._sublayer(attention, self.attn_site.collapse(x, pre_mix))
        x = self.attn_site.hc_post(h, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_site.split(x)
        h = self._sublayer(ffn, self.ffn_site.collapse(x, attn_pre))
        x = self.ffn_site.hc_post(h, x, ffn_post, ffn_comb)
        return x, ffn_pre

    def final_collapse(self, x, pre_mix):
        """After the last block: collapse the streams with its FFN ``pre`` -> ``[1, 1, S/sp, hidden/tp]`` fp32."""
        return self.ffn_site.collapse(x, pre_mix)
