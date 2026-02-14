# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.diffusion_drive.resnet import TtnnResNet34
from models.experimental.diffusion_drive.ttnn_diffusion_head import TtnnDiffusionHead

class TtnnDiffusionDrive:
    def __init__(self, conv_args, conv_pth, device):
        self.device = device
        # Backbone (ResNet-34)
        # We assume conv_args and conv_pth are structured for the backbone
        # conv_args could be the whole model args, or a subsection.
        # Based on ufld_v2, we pass the whole structure if keys match.
        self.backbone = TtnnResNet34(conv_args, conv_pth, device)
        
        # Head
        # In a real scenario, we'd pass specific args for the head.
        self.head = TtnnDiffusionHead(in_channels=512, out_channels=2, device=device) # 2 for trajectory x,y?

    def __call__(self, input, batch_size=1):
        # Backbone forward
        # Expected input shape: [N, C, H, W] -> transformed inside backbone
        # Backbone returns features (likely layer4 output)
        
        # Note: TtnnResnet34.call handles input layout conversion if raw tensor is passed,
        # but ufld_v2 example passes already formatted input. 
        # We'll assume input handling matches backbone requirement.
        
        x = self.backbone(input, batch_size=batch_size)
        
        # Head forward
        if self.head:
            x = self.head(x)
            
        return x
