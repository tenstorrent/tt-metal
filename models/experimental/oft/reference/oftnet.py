import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.oft.reference import resnet
from models.experimental.oft.reference.oft import OFT

from loguru import logger


class OftNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=6.0,
        dtype=torch.float32,
        scale_features=False,
    ):
        super().__init__()

        # Construct frontend network
        assert frontend in ["resnet18", "resnet34"], "unrecognised frontend"
        self.frontend = getattr(resnet, frontend)(pretrained=False, dtype=dtype)

        # Lateral layers convert resnet outputs to a common feature size
        self.lat8 = nn.Conv2d(128, 256, 1, dtype=dtype)
        self.lat16 = nn.Conv2d(256, 256, 1, dtype=dtype)
        self.lat32 = nn.Conv2d(512, 256, 1, dtype=dtype)
        self.bn8 = nn.GroupNorm(16, 256)  # GroupNorm doesn't have dtype parameter
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn32 = nn.GroupNorm(16, 256)

        # Orthographic feature transforms
        self.oft8 = OFT(256, grid_res, grid_height, 1 / 8.0, dtype=dtype)
        self.oft16 = OFT(256, grid_res, grid_height, 1 / 16.0, dtype=dtype)
        self.oft32 = OFT(256, grid_res, grid_height, 1 / 32.0, dtype=dtype)

        # Topdown network
        self.topdown = nn.Sequential(*[resnet.BasicBlock(256, 256, dtype=dtype) for _ in range(topdown_layers)])

        # Detection head
        self.head = nn.Conv2d(256, num_classes * 9, kernel_size=3, padding=1, dtype=dtype)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=dtype))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=dtype))

        self.dtype = dtype
        self.scale_features = scale_features

        # Convert all parameters and buffers to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def forward(self, image, calib, grid):
        if image.dtype != self.dtype:
            logger.warning(f"Input image dtype {image.dtype} does not match model dtype {self.dtype}, converting")
            image = image.to(self.dtype)
        if calib.dtype != self.dtype:
            logger.warning(f"Input calib dtype {calib.dtype} does not match model dtype {self.dtype}, converting")
            calib = calib.to(self.dtype)
        if grid.dtype != self.dtype:
            logger.warning(f"Input grid dtype {grid.dtype} does not match model dtype {self.dtype}, converting")
            grid = grid.to(self.dtype)

        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Run frontend network
        feats8, feats16, feats32 = self.frontend(image)

        # Apply lateral layers to convert image features to common feature size
        lat8 = F.relu(self.bn8(self.lat8(feats8)))
        lat16 = F.relu(self.bn16(self.lat16(feats16)))
        lat32 = F.relu(self.bn32(self.lat32(feats32)))

        ######
        # this is not part of the original implementation.
        if self.scale_features:
            lat8_pre = (
                f"{float(lat8.min()):.6f}, {float(lat8.max()):.6f}, {float(lat8.mean()):.6f}, {float(lat8.std()):.6f}"
            )
            lat16_pre = f"{float(lat16.min()):.6f}, {float(lat16.max()):.6f}, {float(lat16.mean()):.6f}, {float(lat16.std()):.6f}"
            lat32_pre = f"{float(lat32.min()):.6f}, {float(lat32.max()):.6f}, {float(lat32.mean()):.6f}, {float(lat32.std()):.6f}"
            lat8 = lat8 / (lat8.shape[-1] * lat8.shape[-2] * 8)
            lat16 = lat16 / (lat16.shape[-1] * lat16.shape[-2] * 8)
            lat32 = lat32 / (lat32.shape[-1] * lat32.shape[-2] * 8)
            logger.info(
                f" lat8 min,max,mean,std {lat8_pre} => {float(lat8.min()):.6f}, {float(lat8.max()):.6f}, {float(lat8.mean()):.6f}, {float(lat8.std()):.6f}"
            )
            logger.info(
                f"lat16 min,max,mean,std {lat16_pre} => {float(lat16.min()):.6f}, {float(lat16.max()):.6f}, {float(lat16.mean()):.6f}, {float(lat16.std()):.6f}"
            )
            logger.info(
                f"lat32 min,max,mean,std {lat32_pre} => {float(lat32.min()):.6f}, {float(lat32.max()):.6f}, {float(lat32.mean()):.6f}, {float(lat32.std()):.6f}"
            )
        ######

        # Apply OFT and sum
        # return ortho_feats, integral_img, bbox_corners[..., [0, 1]], bbox_corners[..., [2, 3]], bbox_corners[..., [2, 1]], bbox_corners[..., [0, 3]]
        # from models.experimental.oft.reference.oft import integral_image
        # int_img8 = integral_image(lat8)
        # int_img16 = integral_image(lat16)
        # int_img32 = integral_image(lat32)
        ortho8, integral_img8, bbox_top_left8, bbox_btm_right8, bbox_top_right8, bbox_btm_left8 = self.oft8(
            lat8, calib, grid
        )
        ortho16, integral_img16, bbox_top_left16, bbox_btm_right16, bbox_top_right16, bbox_btm_left16 = self.oft16(
            lat16, calib, grid
        )
        ortho32, integral_img32, bbox_top_left32, bbox_btm_right32, bbox_top_right32, bbox_btm_left32 = self.oft32(
            lat32, calib, grid
        )

        ortho = ortho8 + ortho16 + ortho32

        # Apply topdown network
        topdown = self.topdown(ortho)

        # Predict encoded outputs
        batch, _, depth, width = topdown.size()
        outputs = self.head(topdown).view(batch, -1, 9, depth, width)
        scores, pos_offsets, dim_offsets, ang_offsets = torch.split(outputs, [1, 3, 3, 2], dim=2)

        # return scores.squeeze(2), pos_offsets, dim_offsets, ang_offsets
        return (
            [
                feats8,
                feats16,
                feats32,
                lat8,
                lat16,
                lat32,
                integral_img8,
                integral_img16,
                integral_img32,
                bbox_top_left8,
                bbox_btm_right8,
                bbox_top_right8,
                bbox_btm_left8,
                bbox_top_left16,
                bbox_btm_right16,
                bbox_top_right16,
                bbox_btm_left16,
                bbox_top_left32,
                bbox_btm_right32,
                bbox_top_right32,
                bbox_btm_left32,
                ortho8,
                ortho16,
                ortho32,
                ortho,
                calib,
                grid,
                topdown,
            ],
            scores,
            pos_offsets,
            dim_offsets,
            ang_offsets,
        )
