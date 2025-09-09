import ttnn
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_conv2d_wrapper import Conv2d


def custom_preprocessor(model, name):
    """
    Custom preprocessor for all Panoptic-DeepLab model components including ResNet backbone.

    This function handles weight preprocessing for:
    - Conv2d wrapper layers (with optional normalization)
    - Pure BatchNorm/SyncBatchNorm layers
    - Standard PyTorch Conv2d layers (for ResNet backbone)
    - BottleneckBlock and StemBlock (ResNet components)
    """
    parameters = {}

    # Handle Conv2d wrapper layers (used in heads)
    if isinstance(model, Conv2d):
        # Convolutional part
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

        # Handle normalization if present
        if hasattr(model, "norm") and model.norm is not None:
            norm_params = {}
            if isinstance(model.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
                norm_params["weight"] = ttnn.from_torch(
                    model.norm.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["bias"] = ttnn.from_torch(
                    model.norm.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_mean"] = ttnn.from_torch(
                    model.norm.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_var"] = ttnn.from_torch(
                    model.norm.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            elif isinstance(model.norm, nn.LayerNorm):
                norm_params["weight"] = ttnn.from_torch(model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                norm_params["bias"] = ttnn.from_torch(model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            parameters["norm"] = norm_params

    # Handle pure normalization layers
    elif isinstance(model, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
        parameters["weight"] = ttnn.from_torch(
            model.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["bias"] = ttnn.from_torch(
            model.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["running_mean"] = ttnn.from_torch(
            model.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["running_var"] = ttnn.from_torch(
            model.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    # Handle standard PyTorch Conv2d layers (used in ResNet backbone)
    elif isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

        # Handle attached normalization (ResNet style: conv.norm)
        if hasattr(model, "norm") and model.norm is not None:
            norm_params = {}
            if isinstance(model.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
                norm_params["weight"] = ttnn.from_torch(
                    model.norm.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["bias"] = ttnn.from_torch(
                    model.norm.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_mean"] = ttnn.from_torch(
                    model.norm.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_var"] = ttnn.from_torch(
                    model.norm.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            parameters["norm"] = norm_params

    return parameters


def create_panoptic_deeplab_parameters(model: PytorchPanopticDeepLab, device):
    """
    Create preprocessed parameters for the complete Panoptic-DeepLab model.

    This function uses ttnn.preprocess_model_parameters to automatically traverse
    the entire PyTorch model (including ResNet backbone) and convert all weights
    to TTNN tensors using the custom preprocessor.

    Args:
        model: PyTorch Panoptic-DeepLab model with loaded weights
        device: TTNN device for tensor placement

    Returns:
        Preprocessed parameters suitable for TTNN model initialization
    """
    from loguru import logger

    logger.info("Starting unified weight preprocessing for complete Panoptic-DeepLab model")

    def model_initializer():
        return model

    # Use preprocess_model_parameters to handle the entire model uniformly
    parameters = preprocess_model_parameters(
        initialize_model=model_initializer,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    logger.info("Unified weight preprocessing completed successfully")
    logger.debug(f"Generated parameter structure: {list(parameters.keys())}")

    return parameters


# The main API function is create_panoptic_deeplab_parameters() above.
# Pass a PyTorch model with loaded weights to get preprocessed TTNN parameters.
