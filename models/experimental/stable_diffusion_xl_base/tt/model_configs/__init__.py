# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.stable_diffusion_xl_base.tt.model_configs.model_configs_512x512 import (
    ModelOptimisations512x512,
)
from models.experimental.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024 import (
    ModelOptimisations1024x1024,
)


def load_model_optimisations(
    image_resolution,
    conv_act_dtype=None,
    conv_w_dtype=None,
    attention_weights_dtype=None,
    ff_weights_dtype=None,
    force_full_grid=False,
):
    """
    Load the appropriate ModelOptimisation object based on the provided image resolution.

    Args:
        image_resolution (tuple): A tuple of (height, width) representing the image resolution.
            Supported resolutions are (512, 512) and (1024, 1024).
        conv_act_dtype: Optional dtype for convolution activations. Defaults to ttnn.bfloat16.
        conv_w_dtype: Optional dtype for convolution weights. Defaults to ttnn.bfloat16.
        attention_weights_dtype: Optional dtype for attention weights. Defaults to ttnn.bfloat8_b.
        ff_weights_dtype: Optional dtype for feedforward weights. Defaults to ttnn.bfloat8_b.
        force_full_grid (bool): Optional flag to force full grid. Defaults to False.

    Returns:
        ModelOptimisations512x512 or ModelOptimisations1024x1024: The appropriate ModelOptimisation
            object based on the image resolution.

    Raises:
        ValueError: If the image_resolution is not supported.

    Example:
        >>> model_opt = load_model_optimisations((512, 512))
        >>> model_opt = load_model_optimisations((1024, 1024))
    """
    if not isinstance(image_resolution, (tuple, list)) or len(image_resolution) != 2:
        raise ValueError(f"image_resolution must be a tuple or list of length 2, got {image_resolution}")

    height, width = image_resolution

    # Prepare kwargs for initialization
    init_kwargs = {"force_full_grid": force_full_grid}
    if conv_act_dtype is not None:
        init_kwargs["conv_act_dtype"] = conv_act_dtype
    if conv_w_dtype is not None:
        init_kwargs["conv_w_dtype"] = conv_w_dtype
    if attention_weights_dtype is not None:
        init_kwargs["attention_weights_dtype"] = attention_weights_dtype
    if ff_weights_dtype is not None:
        init_kwargs["ff_weights_dtype"] = ff_weights_dtype

    if (height, width) == (512, 512):
        return ModelOptimisations512x512(**init_kwargs)
    elif (height, width) == (1024, 1024):
        return ModelOptimisations1024x1024(**init_kwargs)
    else:
        raise ValueError(
            f"Unsupported image_resolution: {image_resolution}. " "Only (512, 512) and (1024, 1024) are supported."
        )
