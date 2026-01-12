# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of MonoDiffusion
Used for accuracy validation (PCC comparison)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding"""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.get_sinusoidal_embedding(timesteps)
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb

    def get_sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning"""

    def __init__(self, channels: int, time_embed_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.time_proj = nn.Linear(time_embed_dim, channels)

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        if time_emb is not None:
            time_proj = self.time_proj(time_emb)[:, :, None, None]
            x = x + time_proj

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x, inplace=True)

        return x


class Encoder(nn.Module):
    """Feature encoder (ResNet-like)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        features.append(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return x, features


class DiffusionUNet(nn.Module):
    """Diffusion U-Net for depth denoising"""

    def __init__(self, time_embed_dim: int = 256):
        super().__init__()

        # Down blocks
        self.down1 = ResidualBlock(512, time_embed_dim)
        self.down_conv1 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        # Middle block
        self.mid1 = ResidualBlock(512, time_embed_dim)
        self.mid2 = ResidualBlock(512, time_embed_dim)

        # Up blocks
        self.up_conv1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.up1 = ResidualBlock(1024, time_embed_dim)  # 512 + 512 from skip
        self.up_reduce = nn.Conv2d(1024, 512, 1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        encoder_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # Down
        skip1 = self.down1(x, time_emb)
        x = self.down_conv1(skip1)

        # Middle
        x = self.mid1(x, time_emb)
        x = self.mid2(x, time_emb)

        # Up
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_emb)
        x = self.up_reduce(x)

        return x


class Decoder(nn.Module):
    """Multi-scale decoder for depth refinement"""

    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        x = self.up1(x)
        x = F.relu(self.conv1(x), inplace=True)

        x = self.up2(x)
        x = F.relu(self.conv2(x), inplace=True)

        x = self.up3(x)
        x = F.relu(self.conv3(x), inplace=True)

        x = self.up4(x)
        x = F.relu(self.conv4(x), inplace=True)

        depth = torch.sigmoid(self.final(x))

        return depth


class UncertaintyHead(nn.Module):
    """Uncertainty estimation head"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, 1)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(depth), inplace=True)
        uncertainty = F.softplus(self.conv2(x))
        return uncertainty


class MonoDiffusionPyTorch(nn.Module):
    """Complete MonoDiffusion model in PyTorch"""

    def __init__(self, num_inference_steps: int = 20):
        super().__init__()

        self.encoder = Encoder()
        self.timestep_embedding = TimestepEmbedding(256)
        self.diffusion_unet = DiffusionUNet(256)
        self.decoder = Decoder()
        self.uncertainty_head = UncertaintyHead()

        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Encode
        encoded, features = self.encoder(x)

        # Diffusion process (simplified)
        depth_map = encoded
        for t in range(self.num_inference_steps - 1, -1, -1):
            timestep = torch.tensor([t], device=x.device, dtype=torch.long)
            time_emb = self.timestep_embedding(timestep)

            # Denoise
            noise_pred = self.diffusion_unet(depth_map, time_emb, features)
            alpha_t = 1.0 - (t / 1000.0)
            depth_map = depth_map - noise_pred * (1.0 - alpha_t)

        # Decode
        depth = self.decoder(depth_map, features)

        # Uncertainty
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(depth)

        return depth, uncertainty


def create_reference_model(num_inference_steps: int = 20) -> MonoDiffusionPyTorch:
    """Create PyTorch reference model"""
    model = MonoDiffusionPyTorch(num_inference_steps)
    model.eval()
    return model
