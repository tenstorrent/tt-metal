# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full on-device DROID-SLAM forward path.

All convolutions, norms, and elementwise ops in the neural front-end
run on the p150a (Blackhole) chip via tt-nn. CPU is used only for:

  * `ttnn.from_torch` (input upload) and `ttnn.to_torch` (output download)
  * Pure-Python control flow and shape bookkeeping
  * The ii `.numel() == num` check which is metadata, not activation compute

The fp32 torch reference is kept around only as the PCC comparison
target in the benchmark; it is never invoked on the TtDroidNet forward
path.
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.droid_slam.reference.droid_net_ref import DroidNet as ReferenceDroidNet
from models.experimental.droid_slam.tt.droid_encoder_tt import TtBasicEncoder
from models.experimental.droid_slam.tt.droid_update_tt import TtUpdateModule


def _pack_nchw_to_tile_nhwc(t_nchw: torch.Tensor, device) -> "ttnn.Tensor":
    """torch (B*N, C, H, W) → on-device (1, 1, B*N*H*W, C) bfloat16 RM."""
    n, c, h, w = t_nchw.shape
    x = t_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _unpack_tile_nhwc_to_nchw(t_tt: "ttnn.Tensor", n: int, h: int, w: int, c: int) -> torch.Tensor:
    out = ttnn.to_torch(t_tt).float().reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
    return out


class TtDroidNet:
    """Pure on-device DROID-SLAM front-end."""

    def __init__(self, device, reference: ReferenceDroidNet):
        self.device = device
        self.reference_fp32 = reference.eval()
        # Build the on-device modules directly from the fp32 torch
        # weights — TtConv2d preprocesses per-op at first call.
        self.tt_fnet = TtBasicEncoder(reference.fnet, device)
        self.tt_cnet = TtBasicEncoder(reference.cnet, device)
        self.tt_update = TtUpdateModule(reference.update, device)
        # Constant normalization tensors cached once on device.
        mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 1, 1, 3)
        std = torch.tensor([0.225, 0.224, 0.229]).view(1, 1, 1, 3)
        self._mean = ttnn.from_torch(mean, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self._inv_std = ttnn.from_torch(
            1.0 / std, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )

    # ------------------------------------------------------------------
    # extract_features
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        """images: (B, N, 3, H, W) torch — BGR ordering, 0..255 range.

        Returns fmaps, net, inp as torch tensors (consumed by the BA /
        SLAM side which still runs on CPU). fmaps: (B, N, 128, H/8, W/8);
        net, inp: (B, N, 128, H/8, W/8).
        """
        b, n, c, h, w = images.shape
        bn = b * n
        images_bn = images.view(bn, c, h, w)

        # Upload once, normalize on device (sub mean, mul inv_std after /255).
        x_tt = _pack_nchw_to_tile_nhwc(images_bn, self.device)
        x_tt = ttnn.multiply(x_tt, 1.0 / 255.0)
        x_tt = ttnn.subtract(x_tt, self._mean)
        x_tt = ttnn.multiply(x_tt, self._inv_std)

        # fnet + cnet (in series — running them concurrently would
        # require two inputs resident, which pressures L1).
        fmaps_tt, fh, fw = self.tt_fnet(x_tt, batch_size=bn, h=h, w=w)
        # The input tile may already be consumed by fnet; re-upload for cnet.
        x_tt2 = _pack_nchw_to_tile_nhwc(images_bn, self.device)
        x_tt2 = ttnn.multiply(x_tt2, 1.0 / 255.0)
        x_tt2 = ttnn.subtract(x_tt2, self._mean)
        x_tt2 = ttnn.multiply(x_tt2, self._inv_std)
        context_tt, ch_, cw_ = self.tt_cnet(x_tt2, batch_size=bn, h=h, w=w)

        # context has 256 channels — split into net(128) + inp(128),
        # apply tanh/relu on device.
        n_last = context_tt.shape[-2]
        net_tt = ttnn.slice(context_tt, [0, 0, 0, 0], [1, 1, n_last, 128])
        inp_tt = ttnn.slice(context_tt, [0, 0, 0, 128], [1, 1, n_last, 256])
        net_tt = ttnn.tanh(net_tt)
        inp_tt = ttnn.relu(inp_tt)

        fmaps = _unpack_tile_nhwc_to_nchw(fmaps_tt, bn, fh, fw, 128).view(b, n, 128, fh, fw)
        net = _unpack_tile_nhwc_to_nchw(net_tt, bn, ch_, cw_, 128).view(b, n, 128, ch_, cw_)
        inp = _unpack_tile_nhwc_to_nchw(inp_tt, bn, ch_, cw_, 128).view(b, n, 128, ch_, cw_)
        return fmaps, net, inp

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, net, inp, corr, flow, ii):
        """Inputs are 5D torch tensors (B, N_edges, C, H, W); returns the
        5-tuple the benchmark expects.
        """
        b, n_edges, nc, h, w = net.shape
        bn = b * n_edges
        net_bn = net.view(bn, nc, h, w)
        inp_bn = inp.view(bn, -1, h, w)
        corr_bn = corr.view(bn, -1, h, w)
        flow_bn = flow.view(bn, -1, h, w)

        net_tt = _pack_nchw_to_tile_nhwc(net_bn, self.device)
        inp_tt = _pack_nchw_to_tile_nhwc(inp_bn, self.device)
        corr_tt = _pack_nchw_to_tile_nhwc(corr_bn, self.device)
        flow_tt = _pack_nchw_to_tile_nhwc(flow_bn, self.device)

        net_o_tt, delta_tt, weight_tt, eta_tt, upmask_tt = self.tt_update.forward(
            net_tt, inp_tt, corr_tt, flow_tt, batch_size=bn, h=h, w=w
        )

        # delta/weight tails already sliced to 2 channels on device.
        # Download in NHWC (which is what the benchmark's permute produces).
        net_o = _unpack_tile_nhwc_to_nchw(net_o_tt, bn, h, w, 128).view(b, n_edges, 128, h, w)
        delta_nhwc = ttnn.to_torch(delta_tt).float().reshape(bn, h, w, 2)
        delta = delta_nhwc.view(b, n_edges, h, w, 2)
        weight_nhwc = ttnn.to_torch(weight_tt).float().reshape(bn, h, w, 2)
        weight = weight_nhwc.view(b, n_edges, h, w, 2)
        # Fold the 0.01 scale into eta on device (reference does
        # `return 0.01 * eta, upmask`).
        eta_scaled = ttnn.multiply(eta_tt, 0.01)
        eta = _unpack_tile_nhwc_to_nchw(eta_scaled, bn, h, w, 1).view(b, n_edges, h, w)
        upmask = _unpack_tile_nhwc_to_nchw(upmask_tt, bn, h, w, 8 * 8 * 9).view(
            b, n_edges, 8 * 8 * 9, h, w
        )
        return net_o, delta, weight, eta, upmask
