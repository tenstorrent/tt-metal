# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from ..tt.utils import assert_quality

if TYPE_CHECKING:
    from ..reference.patch_embedding import PatchEmbed


@pytest.mark.parametrize(
    "batch_size",
    [
        2,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_patch_embedding(
    *,
    device: ttnn.Device,
    batch_size: int,
) -> None:
    dtype = torch.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=dtype
    )
    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    parameters = TtPatchEmbedParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtPatchEmbed(parameters)

    torch_input_tensor = torch.randn((batch_size, 16, 64, 64), dtype=dtype)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    assert_quality(torch_output, tt_output, pcc=0.999_990)
