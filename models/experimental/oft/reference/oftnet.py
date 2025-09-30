import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.oft.reference import resnet
from models.experimental.oft.reference.oft import OFT


class OftNet(nn.Module):
    def __init__(self, num_classes=1, frontend="resnet18", topdown_layers=8, grid_res=0.5, grid_height=6.0):
        super().__init__()

        assert frontend in ["resnet18", "resnet34"], "unrecognised frontend"
        self.frontend = getattr(resnet, frontend)(pretrained=False)

        self.lat8 = nn.Conv2d(128, 256, 1)
        self.lat16 = nn.Conv2d(256, 256, 1)
        self.lat32 = nn.Conv2d(512, 256, 1)
        self.bn8 = nn.GroupNorm(16, 256)
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn32 = nn.GroupNorm(16, 256)

        self.oft8 = OFT(256, grid_res, grid_height, 1 / 8.0)
        self.oft16 = OFT(256, grid_res, grid_height, 1 / 16.0)
        self.oft32 = OFT(256, grid_res, grid_height, 1 / 32.0)

        self.topdown = nn.Sequential(*[resnet.BasicBlock(256, 256) for _ in range(topdown_layers)])

        self.head = nn.Conv2d(256, num_classes * 9, kernel_size=3, padding=1)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, image, calib, grid):
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        feats8, feats16, feats32 = self.frontend(image)

        lat8 = F.relu(self.bn8(self.lat8(feats8)))
        lat16 = F.relu(self.bn16(self.lat16(feats16)))
        lat32 = F.relu(self.bn32(self.lat32(feats32)))
        # return lat8, lat16, lat32
        ortho8 = self.oft8(lat8, calib, grid)
        ortho16 = self.oft16(lat16, calib, grid)
        ortho32 = self.oft32(lat32, calib, grid)
        ortho = ortho8 + ortho16 + ortho32

        topdown = self.topdown(ortho)

        batch, _, depth, width = topdown.size()
        outputs = self.head(topdown).view(batch, -1, 9, depth, width)
        scores, pos_offsets, dim_offsets, ang_offsets = torch.split(outputs, [1, 3, 3, 2], dim=2)

        return scores.squeeze(2), pos_offsets, dim_offsets, ang_offsets
