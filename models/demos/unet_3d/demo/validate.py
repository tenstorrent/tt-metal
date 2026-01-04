# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from safetensors.torch import load_file

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


def dispatch_model_backend(config) -> UNet3D | UNet3DTch:
    state_dict = load_file(config["model_path"])
    if config["backend"] == "ttnn":
        device = ttnn.open_device(
            device_id=0,
            dispatch_core_config=ttnn.device.DispatchCoreConfig(),
            l1_small_size=8192 * 4,
        )
        model = UNet3D(
            device,
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_channels=config["model"]["base_channels"],
            num_levels=config["model"]["num_levels"],
            num_groups=config["model"]["num_groups"],
            scale_factor=config["model"]["scale_factor"],
        )
    elif config["backend"] == "torch":
        device = get_device_from_config(config)
        model = UNet3DTch(
            device,
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_channels=config["model"]["base_channels"],
            num_levels=config["model"]["num_levels"],
            num_groups=config["model"]["num_groups"],
            scale_factor=config["model"]["scale_factor"],
        )
        model.to(device)
        model.eval()

    else:
        raise ValueError(f"Unknown backend: {config['backend']}")
    model.load_state_dict(state_dict)

    return model


def get_predictor(model: UNet3D | UNet3DTch, cfg: dict) -> Predictor:
    """Create and return a predictor instance based on the configuration.

    Args:
        model: The trained model to use for prediction.
        args: Parsed CLI arguments.

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


def main():
    """Main entry point for prediction with 3D U-Net models.

    Creates the model, loads trained weights, runs predictions on a single H5 volume,
    and computes evaluation metrics if specified.
    """
    config = load_config()

    # model = dispatch_model_backend(config)
    model = UNet3DRunner()
    # device = ttnn.open_device(
    #     device_id=0,
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=2048,
        trace_region_size=679936,
        num_command_queues=2,
    )
    model.initialize_inference(device, config)
    predictor = get_predictor(model, config)
    metrics = []
    for test_loader in get_test_loaders(config["dataset"]):
        metric = predictor(test_loader)
        if metric is not None:
            metrics.append(metric)

    if metrics:
        logger.info(f"Per-class metric: {metrics}")
        print_metrics(metrics)


if __name__ == "__main__":
    main()
