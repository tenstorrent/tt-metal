# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.demos.unet_3d.demo.config import load_config
from models.demos.unet_3d.demo.dataset import get_test_loaders
from models.demos.unet_3d.demo.predictor import Predictor
from models.demos.unet_3d.demo.utils import configure_logging
from models.demos.unet_3d.runner.performant_runner import UNet3DRunner
from models.demos.unet_3d.torch_impl.model import UNet3DTch
from models.demos.unet_3d.ttnn_impl.model import UNet3D

logger = configure_logging()


def get_default_device_name() -> str:
    """Get the default device name based on CUDA availability.

    Returns:
        The default device name ("cuda" if available, otherwise "cpu").
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_from_config(config: dict) -> str:
    """Get the device name from the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        The device name ("cuda" if available, otherwise "cpu").
    """
    device_str = config.get("device", get_default_device_name())
    return torch.device(device_str)


def get_predictor(model: UNet3D | UNet3DTch, cfg: dict) -> Predictor:
    """Create and return a predictor instance based on the configuration.

    Args:
        model: The trained model to use for prediction.
        cfg (dict): Configuration dictionary for dataset, model, and predictor.

    Returns:
        A predictor instance.
    """
    if cfg["dataset"]["output_dir"] is not None:
        os.makedirs(cfg["dataset"]["output_dir"], exist_ok=True)

    return Predictor(
        model,
        cfg["dataset"]["output_dir"],
        cfg["output_dataset"],
        cfg["model"]["out_channels"],
        performance_metric=cfg["predictor"]["performance_metric"],
        label_internal_path=cfg["predictor"]["label_internal_path"],
    )


def print_metrics(metrics) -> None:
    for metric_dict in metrics:
        for metric_name, value in metric_dict.items():
            logger.info(f"Average {metric_name}: class-wise {value}, mean {value.mean()}")


def get_device():
    num_devices = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, num_devices),
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=2048,
        trace_region_size=679936,
        num_command_queues=2,
    )


def main():
    """Main entry point for prediction with 3D U-Net models.

    Creates the model, loads trained weights, runs predictions on a single H5 volume,
    and computes evaluation metrics if specified.
    """
    config = load_config()

    model = UNet3DRunner()
    device = get_device()
    num_devices = device.get_num_devices()
    model.initialize_inference(device, config)
    predictor = get_predictor(model, config)
    metrics = []
    for test_loader in get_test_loaders(config["dataset"], num_devices):
        metric = predictor(test_loader)
        if metric is not None:
            metrics.append(metric)

    if metrics:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
