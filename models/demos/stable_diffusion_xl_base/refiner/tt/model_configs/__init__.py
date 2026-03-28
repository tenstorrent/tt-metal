# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


class RefinerModelOptimisationsBase:
    """
    Base class for all refiner model optimisations.
    This allows isinstance checks to work across all refiner configurations.
    """

    pass


from models.demos.stable_diffusion_xl_base.refiner.tt.model_configs.model_configs_512x512 import (
    RefinerModelOptimisations512x512,
)
from models.demos.stable_diffusion_xl_base.refiner.tt.model_configs.model_configs_1024x1024 import (
    RefinerModelOptimisations1024x1024,
)
from models.demos.stable_diffusion_xl_base.refiner.tt.model_configs.model_configs_1024x1024BH import (
    RefinerModelOptimisations1024x1024BH,
)
from models.common.utility_functions import is_wormhole_b0, is_blackhole


def load_refiner_model_optimisations(
    image_resolution,
    conv_act_dtype=None,
    conv_w_dtype=None,
    attention_weights_dtype=None,
    ff_weights_dtype=None,
):
    """
    Load the appropriate RefinerModelOptimisation object based on the provided image resolution and hardware type.

    Args:
        image_resolution (tuple): A tuple of (height, width) representing the image resolution.
            Supported resolutions are (512, 512) and (1024, 1024).
        conv_act_dtype: Optional dtype for convolution activations. Defaults to ttnn.bfloat16.
        conv_w_dtype: Optional dtype for convolution weights. Defaults to ttnn.bfloat16.
        attention_weights_dtype: Optional dtype for attention weights. Defaults to ttnn.bfloat8_b.
        ff_weights_dtype: Optional dtype for feedforward weights. Defaults to ttnn.bfloat8_b.

    Returns:
        RefinerModelOptimisations512x512, RefinerModelOptimisations1024x1024, or RefinerModelOptimisations1024x1024BH:
            The appropriate RefinerModelOptimisation object based on the image resolution and hardware type.
            For 1024x1024 resolution, automatically selects RefinerModelOptimisations1024x1024BH for Blackhole
            hardware or RefinerModelOptimisations1024x1024 for Wormhole hardware.
            For 512x512 resolution, returns RefinerModelOptimisations512x512 regardless of hardware type.

    Raises:
        ValueError: If the image_resolution is not supported.

    Example:
        >>> model_opt = load_refiner_model_optimisations((1024, 1024))  # Auto-selects based on hardware
        >>> model_opt = load_refiner_model_optimisations((512, 512))
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

    if (height, width) == (512, 512):
        return RefinerModelOptimisations512x512(**init_kwargs)
    elif (height, width) == (1024, 1024):
        # Check hardware type and return appropriate config
        if is_wormhole_b0():
            return RefinerModelOptimisations1024x1024(**init_kwargs)
        elif is_blackhole():
            return RefinerModelOptimisations1024x1024BH(**init_kwargs)
        else:
            raise ValueError(
                "Unsupported hardware type for 1024x1024 resolution. Only Blackhole and Wormhole_B0 are supported."
            )
    else:
        raise ValueError(
            f"Unsupported image_resolution: {image_resolution}. "
            "Supported resolutions are (512, 512) and (1024, 1024)."
        )
