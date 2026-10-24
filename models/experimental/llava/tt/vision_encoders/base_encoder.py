# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from models.common.lightweightmodule import LightweightModule


class BaseVisionEncoder(LightweightModule, ABC):
    """
    Abstract base class for all vision encoders in LLaVA

    Design pattern: Factory + Strategy
    - Factory: builder.py creates concrete encoders
    - Strategy: Each encoder implements this interface

    Supported encoders:
    - CLIPVisionEncoder: CLIP ViT-L/14 @ 336px (LLaVA 1.5)
    - SigLIPVisionEncoder: SigLIP encoder (LLaVA-MORE)
    - Future: DINOVisionEncoder, IJepaVisionEncoder, etc.
    """

    @abstractmethod
    def forward(self, images):
        """
        Encode images to vision token embeddings

        Args:
            images: Input images
                - torch.Tensor: [batch, 3, H, W] (will be converted to ttnn)
                - ttnn.Tensor: [batch, 3, H, W] (already on device)

        Returns:
            vision_tokens: ttnn.Tensor [batch, num_patches, hidden_dim]
                Example: [1, 576, 1024] for CLIP ViT-L/14

        Note: CLS token should be dropped if present (LLaVA uses only patch tokens)
        """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        Output hidden dimension of the encoder

        Returns:
            int: Hidden dimension
                - CLIP: 1024
                - SigLIP: Variable (check config)
                - DINO: 1024
        """

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """
        Number of output tokens per image

        Returns:
            int: Number of patches
                - CLIP @ 336px: 576 (24×24 grid)
                - SigLIP @ 384px: 729 (27×27 grid)
        """

    @property
    def patch_size(self) -> int:
        """
        Patch size in pixels (optional, for reference)

        Returns:
            int: Patch size (default 14 for CLIP/SigLIP)
        """
        return 14

    @property
    def resolution(self) -> int:
        """
        Input image resolution (optional, for reference)

        Returns:
            int: Resolution (default 336 for CLIP, 384 for SigLIP)
        """
        return 336
