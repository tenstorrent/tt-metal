# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from models.experimental.functional_fadnetpp.tt.ttnn_extractnet import ttExtractNet
from models.experimental.functional_fadnetpp.tt.ttnn_cunet import ttCUNet
import ttnn
import tt_lib


class ttDispNetC:
    def output_preprocessing(self, output_tensor, height, width, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
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

    def __init__(self, parameters, model, resBlock=True) -> None:
        super().__init__()
        self.maxdisp = 192
        self.extractnet = ttExtractNet(parameters.extractnet)
        self.cunet = ttCUNet(parameters.cunet, model.cunet)

    def __call__(self, device, input, input1, input2):
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extractnet(device, input1, input2)
        conv3a_l = self.output_preprocessing(conv3a_l, 120, 72, device)
        conv3a_r = self.output_preprocessing(conv3a_r, 120, 72, device)

        def build_corr(img_left, img_right, max_disp=40, zero_volume=None):
            B, C, H, W = img_left.shape
            if zero_volume is not None:
                tmp_zero_volume = zero_volume  # * 0.0
                volume = tmp_zero_volume
            else:
                volume = img_left.new_zeros([B, max_disp, H, W])
            for i in range(max_disp):
                if (i > 0) & (i < W):
                    volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, : W - i]).mean(dim=1)
                else:
                    volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

            volume = volume.contiguous()
            return volume

        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp // 8 + 16)
        conv3a_l = self.input_preprocessing(conv3a_l, device)
        out_corr = self.input_preprocessing(out_corr, device)
        dispnetc_flows = self.cunet(input1, device, conv1_l, conv2_l, conv3a_l, out_corr)

        return dispnetc_flows
