# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_fadnetpp.reference.dispnetc import DispNetC
from models.experimental.functional_fadnetpp.reference.dispnetres import DispNetRes

import torch
import torch.nn as nn


class FadNetPP(nn.Module):
    def warp_right_to_left(self, x, disp, warp_grid=None):
        B, C, H, W = x.shape
        # mesh grid
        if warp_grid is not None:
            xx0, yy = warp_grid
            xx = xx0 + disp
            xx = 2.0 * xx / max(W - 1, 1) - 1.0
        else:
            xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
            yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
            xx = xx.view(1, -1).repeat(H, 1)
            yy = yy.view(-1, 1).repeat(1, W)

            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

            # apply disparity to x-axis
            xx = xx + disp
            xx = 2.0 * xx / max(W - 1, 1) - 1.0
            yy = 2.0 * yy / max(H - 1, 1) - 1.0

        grid = torch.cat((xx, yy), 1)

        vgrid = grid

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)

        return output

    def channel_length(self, x):
        return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + 1e-8)

    def __init__(self, in_planes):
        super(FadNetPP, self).__init__()
        self.input_channel = 3
        self.dispnetc = DispNetC()
        self.dispnetres = DispNetRes(in_planes)

    def forward(self, input: torch.Tensor, enabled_tensorrt=False):
        dispnetc_flows = self.dispnetc(input)
        dispnetc_final_flow = dispnetc_flows[0]
        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = self.warp_right_to_left(input[:, self.input_channel :, :, :], -dispnetc_final_flow)
        diff_img0 = input[:, : self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = self.channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_net2 = torch.cat((input, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim=1)

        # dispnetres
        if enabled_tensorrt:
            dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_final_flow)
        else:
            dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_flows)

        index = 0
        dispnetres_final_flow = dispnetres_flows[index]

        return dispnetc_final_flow, dispnetres_final_flow
