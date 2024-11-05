# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_fadnetpp.tt.ttnn_dispnetc import TtDispNetC
from models.experimental.functional_fadnetpp.tt.tt_dispnetres import TtDispNetRes
import ttnn
import torch
import torch.nn as nn


class TtFadNetPP(nn.Module):
    def output_preprocessing(self, output_tensor, height, width, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch.float32)
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                height,
                width,
            ],
        )
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = torch.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        return input_tensor

    def warp_right_to_left(device, x, disp, warp_grid=None):
        B, C, H, W = x.size()
        # mesh grid
        if warp_grid is not None:
            xx0, yy = warp_grid
            xx = xx0 + disp
            xx = 2.0 * xx / max(W - 1, 1) - 1.0
        else:
            xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
            yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
            # if x.is_cuda:
            #    xx = xx.cuda()
            #    yy = yy.cuda()
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

    def channel_length(device, x):
        return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + 1e-8)

    def __init__(self, parameters, device, in_planes, model):
        super(TtFadNetPP, self).__init__()
        self.input_channel = 3
        self.dispnetc = TtDispNetC(parameters.dispnetc, model)
        self.dispnetres = TtDispNetRes(device, parameters.dispnetres, in_planes)

    def forward(self, device, input, input1, input2, enabled_tensorrt=False):
        dispnetc_flows = self.dispnetc(device, input, input1, input2)

        dispnetc_final_flow = dispnetc_flows[0]
        dispnetc_final_flow = self.output_preprocessing(dispnetc_final_flow, 960, 576, device)
        input = self.output_preprocessing(input, 960, 576, device)

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = self.warp_right_to_left(input[:, self.input_channel :, :, :], -dispnetc_final_flow)
        diff_img0 = input[:, : self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = self.channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_net2 = torch.cat((input, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim=1)

        dispnetc_final_flow = self.input_preprocessing(dispnetc_final_flow, device)
        inputs_net2 = self.input_preprocessing(inputs_net2, device)

        # dispnetres
        if enabled_tensorrt:
            dispnetres_flows = self.dispnetres(device, inputs_net2, dispnetc_final_flow)
        else:
            dispnetres_flows = self.dispnetres(device, inputs_net2, dispnetc_flows)

        index = 0
        dispnetres_final_flow = dispnetres_flows[index]

        return dispnetc_final_flow, dispnetres_final_flow
