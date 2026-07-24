# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host-side input instantiation for I2I (re-exports ref PyTorch path).
#
# Conditional VAE tokens are static across denoise steps. We build them once on
# host with PyTorch scatter (mirrors upstream ``instantiate_vae_image_tokens``),
# then upload ``inputs_embeds`` to device. Arbitrary-index scatter at
# ``vae_image_mask`` positions is not yet a TTNN op — same pattern as demo wte.

from models.experimental.hunyuan_image_3_0.ref.image_gen.input_instantiate import (
    instantiate_continuous_tokens,
    instantiate_vae_image_tokens,
    instantiate_vit_image_tokens,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import build_i2i_inputs_embeds

__all__ = [
    "instantiate_continuous_tokens",
    "instantiate_vae_image_tokens",
    "instantiate_vit_image_tokens",
    "build_i2i_inputs_embeds",
]
