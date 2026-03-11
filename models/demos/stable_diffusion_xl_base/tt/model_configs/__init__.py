# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.utility_functions import is_wormhole_b0, is_blackhole
from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_512x512 import (
    ModelOptimisations512x512,
)
from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024 import (
    ModelOptimisations1024x1024,
)
from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024BH import (
    ModelOptimisations1024x1024BH,
)


def get_image_resolution_from_model_config(model_config):
    """
    Get the image resolution from a ModelOptimisations instance.

    Args:
        model_config: A ModelOptimisations instance (either ModelOptimisations512x512 or
            ModelOptimisations1024x1024).

    Returns:
        tuple: A tuple of (height, width) representing the image resolution.

    Raises:
        ValueError: If the model_config type is not recognized.

    Example:
        >>> model_opt = ModelOptimisations512x512()
        >>> resolution = get_image_resolution_from_model_config(model_opt)
        >>> print(resolution)  # (512, 512)
    """
    if isinstance(model_config, ModelOptimisations512x512):
        return (512, 512)
    elif isinstance(model_config, ModelOptimisations1024x1024):
        return (1024, 1024)
    else:
        raise ValueError(
            f"Unsupported model_config type: {type(model_config).__name__}. "
            "Expected ModelOptimisations512x512 or ModelOptimisations1024x1024."
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
    Load the appropriate ModelOptimisation object based on the provided image resolution and hardware type.

    Args:
        image_resolution (tuple): A tuple of (height, width) representing the image resolution.
            Supported resolutions are (512, 512) and (1024, 1024).
        conv_act_dtype: Optional dtype for convolution activations. Defaults to ttnn.bfloat16.
        conv_w_dtype: Optional dtype for convolution weights. Defaults to ttnn.bfloat16.
        attention_weights_dtype: Optional dtype for attention weights. Defaults to ttnn.bfloat8_b.
        ff_weights_dtype: Optional dtype for feedforward weights. Defaults to ttnn.bfloat8_b.
        force_full_grid (bool): Optional flag to force full grid. Defaults to False.

    Returns:
        ModelOptimisations512x512, ModelOptimisations1024x1024, or ModelOptimisations1024x1024BH:
            The appropriate ModelOptimisation object based on the image resolution and hardware type.
            For 1024x1024 resolution, automatically selects ModelOptimisations1024x1024BH for Blackhole
            hardware or ModelOptimisations1024x1024 for Wormhole hardware.
            For 512x512 resolution, returns ModelOptimisations512x512 regardless of hardware type.

    Raises:
        ValueError: If the image_resolution is not supported.

    Example:
        >>> model_opt = load_model_optimisations((512, 512))
        >>> model_opt = load_model_optimisations((1024, 1024))  # Auto-selects based on hardware
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
        # For now, 512x512 image resolution uses the same configs regardless of hardware type
        return ModelOptimisations512x512(**init_kwargs)
    elif (height, width) == (1024, 1024):
        # Check hardware type and return appropriate config
        if is_wormhole_b0():
            return ModelOptimisations1024x1024(**init_kwargs)
        elif is_blackhole():
            return ModelOptimisations1024x1024BH(**init_kwargs)
        else:
            raise ValueError(
                "Unsupported hardware type for 1024x1024 resolution. Only Blackhole and Wormhole_B0 are supported."
            )
    else:
        raise ValueError(
            f"Unsupported image_resolution: {image_resolution}. "
            "Supported resolutions are (512, 512) and (1024, 1024)."
        )
