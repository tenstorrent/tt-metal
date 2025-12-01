# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.tt_dit.models.transformers.sd35_med.patch_embed_sd35_medium import PatchEmbed
from models.common.utility_functions import comp_pcc, comp_allclose
from typing import Optional


class PatchEmbedRef(torch.nn.Module):
    """Reference PyTorch MLP matching MM-DiT implementation"""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.patch_size = (patch_size, patch_size)
        self.flatten = flatten

        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = (
                self.img_size[0] // patch_size,
                self.img_size[1] // patch_size,
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = torch.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=torch.bfloat16,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # → [B, embed_dim, H/P, W/P]

        if self.flatten:
            x = x.flatten(2)  # → [B, embed_dim, N]
            x = x.transpose(1, 2)  # → [B, N, embed_dim]

        return x


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "img_size, patch_size, in_chans, embed_dim, batch_size",
    [
        (128, 16, 16, 512, 1),
        (256, 16, 16, 512, 1),
        (512, 32, 3, 768, 1),
        (1024, 32, 3, 768, 1),
    ],
    ids=["128p", "256p", "512p", "1024p"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_patch_embed(device, dtype, img_size, patch_size, in_chans, embed_dim, batch_size, reset_seeds):
    """
    Test SD3.5 Medium PatchEmbed Layer.
    Validates the TTNN implementation with Pytorch reference
    """
    torch.manual_seed(1234)

    # Create reference model
    ref = PatchEmbedRef(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    ref.eval()

    # Create TTNN model
    tt = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_chans,
        embed_dim=embed_dim,
        mesh_device=device,
    )

    # Load weights from reference model
    state = ref.state_dict()

    tt.load_torch_state_dict(state)

    # Create input tensor
    torch_input = torch.randn(batch_size, in_chans, img_size, img_size, dtype=torch.bfloat16)

    # Torch forward
    with torch.no_grad():
        ref_out = ref(torch_input)

    # ------------------------------------------------------------
    # TTNN expects NHWC layout
    # Torch format = [B, C, H, W]
    # Convert → [B, H, W, C]
    # ------------------------------------------------------------
    nhwc_input = torch_input.permute(0, 2, 3, 1).contiguous()

    tt_input = ttnn.from_torch(
        nhwc_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # TTNN forward
    tt_out = tt(tt_input)

    # Convert back to torch
    tt_out_torch = ttnn.to_torch(tt_out)

    # Comparing the outputs of ttnn vs pytorch
    passing, pcc_msg = comp_pcc(ref_out, tt_out_torch, 0.99)

    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"PCC={pcc_msg}")

    assert passing, f"PCC FAILED: {pcc_msg}"
    logger.info("PatchEmbed Passed successfully!")
