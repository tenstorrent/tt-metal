# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.utility_functions import is_wormhole_b0, is_blackhole
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs.model_configs_1024x1024 import (
    VAEModelOptimisations,
)
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs.model_configs_1024x1024_BH import (
    VAEModelOptimisationsBH,
)


def load_vae_model_optimisations(
    image_resolution,
    conv_act_dtype=None,
    conv_w_dtype=None,
    attention_weights_dtype=None,
    ff_weights_dtype=None,
):
    """
    Load the appropriate VAEModelOptimisation object based on the provided image resolution and hardware type.

    Args:
        image_resolution (tuple): A tuple of (height, width) representing the image resolution.
            Supported resolutions are (512, 512) and (1024, 1024).
        conv_act_dtype: Optional dtype for convolution activations. Defaults to ttnn.bfloat16.
        conv_w_dtype: Optional dtype for convolution weights. Defaults to ttnn.bfloat16.
        attention_weights_dtype: Optional dtype for attention weights. Defaults to ttnn.bfloat8_b.
        ff_weights_dtype: Optional dtype for feedforward weights. Defaults to ttnn.bfloat8_b.

    Returns:
        VAEModelOptimisations or VAEModelOptimisationsBH:
            The appropriate VAEModelOptimisation object based on the image resolution and hardware type.
            For 1024x1024 resolution, automatically selects VAEModelOptimisationsBH for Blackhole
            hardware or VAEModelOptimisations for Wormhole hardware.
            For 512x512 resolution, returns VAEModelOptimisations regardless of hardware type.

    Raises:
        ValueError: If the image_resolution is not supported.

    Example:
        >>> model_opt = load_vae_model_optimisations((1024, 1024))  # Auto-selects based on hardware
    """
    if not isinstance(image_resolution, (tuple, list)) or len(image_resolution) != 2:
        raise ValueError(f"image_resolution must be a tuple or list of length 2, got {image_resolution}")

    height, width = image_resolution

    # Prepare kwargs for initialization
    init_kwargs = {}
    if conv_act_dtype is not None:
        init_kwargs["conv_act_dtype"] = conv_act_dtype
    if conv_w_dtype is not None:
        init_kwargs["conv_w_dtype"] = conv_w_dtype
    if attention_weights_dtype is not None:
        init_kwargs["attention_weights_dtype"] = attention_weights_dtype
    if ff_weights_dtype is not None:
        init_kwargs["ff_weights_dtype"] = ff_weights_dtype

    # 512x512 and 1024x1024 share the same optimisations because they only differ in the number of DRAM
    # slices which is determined by the tensor NHW dimension.
    if (height, width) == (512, 512):
        return VAEModelOptimisations(**init_kwargs)
    elif (height, width) == (1024, 1024):
        # Check hardware type and return appropriate config
        if is_wormhole_b0():
            return VAEModelOptimisations(**init_kwargs)
        elif is_blackhole():
            return VAEModelOptimisationsBH(**init_kwargs)
        else:
            raise ValueError(
                "Unsupported hardware type for 1024x1024 resolution. Only Blackhole and Wormhole_B0 are supported."
            )
    else:
        raise ValueError(
            f"Unsupported image_resolution: {image_resolution}. "
            "Supported resolutions are (512, 512) and (1024, 1024)."
        )
