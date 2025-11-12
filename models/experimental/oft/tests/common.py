# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from loguru import logger

GRID_RES = 0.5
GRID_SIZE = (80.0, 80.0)
GRID_HEIGHT = 4.0
Y_OFFSET = 1.74

H_PADDED = 384
W_PADDED = 1280

NMS_THRESH = 0.2
CHECKPOINTS_PATH_ENV = "CHECKPOINTS_PATH"


def test_weights_downloaded(model_location_generator, cache_weights_locally):
    checkpoints_path = prepare_checkpoint_path(model_location_generator, cache_weights_locally)
    print(f"Checkpoints path: {checkpoints_path}")
    assert checkpoints_path is not None


def prepare_checkpoint_path(model_location_generator=None, cache_weights_locally=None):
    """
    Prepare path to OFT checkpoint file.

    Args:
        cache_weights: Cache weights fixture for unified download/cache handling
        model_location_generator: Fallback model location generator

    Returns:
        Path: Path to checkpoint-0600.pth file
    """
    model_version = "vision-models/oft"
    checkpoint_filename = "checkpoint-0600.pth"
    is_local_env = False
    checkpoints_path = None
    # first try to download to local weights
    if cache_weights_locally is not None:
        weights_path, is_local_env = cache_weights_locally(model_version, model_subdir="")
        if is_local_env and weights_path is not None:
            checkpoints_path = os.path.join(weights_path, checkpoint_filename)
            logger.info(f"Custom model weights function returned path: {checkpoints_path}")
        else:
            logger.warning("Custom model weights function did not return a valid path!")
    else:
        logger.warning("Custom model weights function is None!")

    if not is_local_env and model_location_generator is not None:
        cached_weights_path = "/tmp/ttnn_model_cache/model_weights/vision-models/oft/checkpoint-0600.pth"
        if os.path.exists(cached_weights_path):
            checkpoints_path = cached_weights_path
        else:
            checkpoints_path = model_location_generator(
                model_version,
                checkpoint_filename=checkpoint_filename,
                ci_v2_timeout_in_s=300,
                endpoint_prefix="tt-metal-models",
                model_subdir="",
            )
            logger.info(f"Model location generator returned path: {checkpoints_path}")
    else:
        logger.warning("Model location generator is None or running in local environment!")
    return checkpoints_path


def load_checkpoint(ref_model, model_location_generator=None, cache_weights_locally=None):
    checkpoints_path = prepare_checkpoint_path(
        model_location_generator=model_location_generator, cache_weights_locally=cache_weights_locally
    )
    if checkpoints_path is None:
        logger.error("Checkpoint path is not set!")
        raise RuntimeError("Checkpoint path is not set!")
    if checkpoints_path is not None and os.path.isfile(checkpoints_path):
        logger.info(f"Loading model weights from {checkpoints_path}")
        checkpoint = torch.load(checkpoints_path, map_location="cpu")

        # Load state dict as is
        ref_model.load_state_dict(checkpoint["model"], strict=True)

        # Ensure all weights are converted to the specified dtype after loading
        ref_model.to(ref_model.dtype)
        logger.info(f"Converted all model weights to {ref_model.dtype}")
    else:
        logger.error(f"Checkpoint path {checkpoints_path} does not exist, using random weights")
        assert False, f"Checkpoint path {checkpoints_path} does not exist"

    return ref_model


def visualize_tensor_distributions(tensor1, tensor2, title1="Tensor 1", title2="Tensor 2"):
    """
    Visualizes the distribution of values in two tensors.

    Args:
        tensor1: First tensor to visualize
        tensor2: Second tensor to visualize
        title1: Title for the first tensor's histogram
        title2: Title for the second tensor's histogram

    Returns:
        matplotlib.axes.Axes: Axes object containing the plots
    """
    import ttnn
    import matplotlib.pyplot as plt

    if isinstance(tensor1, ttnn.Tensor):
        tensor1 = ttnn.to_torch(tensor1)
    if isinstance(tensor2, ttnn.Tensor):
        tensor2 = ttnn.to_torch(tensor2)

    # Flatten tensors to 1D
    t1_flat = tensor1.float().flatten().detach().cpu().numpy()
    t2_flat = tensor2.float().flatten().detach().cpu().numpy()

    # Calculate statistics
    t1_mean, t1_std = t1_flat.mean(), t1_flat.std()
    t2_mean, t2_std = t2_flat.mean(), t2_flat.std()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram for tensor1
    ax1.hist(t1_flat, bins=50, alpha=0.7)
    ax1.axvline(t1_mean, color="r", linestyle="--", label=f"Mean: {t1_mean:.4f}")
    ax1.axvline(t1_mean + t1_std, color="g", linestyle=":", label=f"Std: {t1_std:.4f}")
    ax1.axvline(t1_mean - t1_std, color="g", linestyle=":")
    ax1.set_title(f"{title1}\nMean: {t1_mean:.4f}, Std: {t1_std:.4f}")
    ax1.legend()

    # Plot histogram for tensor2
    ax2.hist(t2_flat, bins=50, alpha=0.7)
    ax2.axvline(t2_mean, color="r", linestyle="--", label=f"Mean: {t2_mean:.4f}")
    ax2.axvline(t2_mean + t2_std, color="g", linestyle=":", label=f"Std: {t2_std:.4f}")
    ax2.axvline(t2_mean - t2_std, color="g", linestyle=":")
    ax2.set_title(f"{title2}\nMean: {t2_mean:.4f}, Std: {t2_std:.4f}")
    ax2.legend()

    plt.tight_layout()
    return fig
