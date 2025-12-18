# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import argparse
import numpy as np
from loguru import logger

from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import build_dataset
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.common.utility_functions import comp_pcc
from models.experimental.detr3d.reference.utils.ap_calculator import APCalculator


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.parameters = None
        self.device = None


def load_model_weights(model, weights_path):
    """Load model weights from .pth file"""
    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")["model"]
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(f"Weights file not found at {weights_path}, using random weights")
    return model


def run_detr3d_inference(
    dataset_root_dir=None,
    weights_path=None,
    output_dir="models/experimental/detr3d/demo/outputs/",
    encoder_only=False,
    seed=0,
    batch_size=1,
    num_workers=0,  # Set to 0 for reproducibility
):
    """Run DETR3D inference with AP calculation"""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup model configuration
    args = Detr3dArgs()
    args.dataset_name = "sunrgbd"
    args.dataset_root_dir = dataset_root_dir
    args.meta_data_dir = None
    args.batchsize_per_gpu = batch_size
    args.dataset_num_workers = num_workers
    args.use_color = False

    # Build dataset to get ground truth labels
    datasets, dataset_config = build_dataset(args)

    # Create DataLoader for test split
    test_dataset = datasets["test"]
    sampler = torch.utils.data.SequentialSampler(test_dataset)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        test_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=lambda x: None,
    )

    # Build and load PyTorch reference model
    ref_module, _ = build_3detr(args, dataset_config)
    if weights_path:
        ref_module = load_model_weights(ref_module, weights_path)
    ref_module.eval()

    # Initialize APCalculator for PyTorch model
    ref_ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    # Open TTNN device
    ttnn_device = ttnn.open_device(device_id=0, l1_small_size=16384)

    try:
        # Preprocess model parameters
        logger.info("Preprocessing model parameters...")
        ref_module_parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_module,
            custom_preprocessor=create_custom_mesh_preprocessor(None),
            device=ttnn_device,
        )

        # Build TTNN model
        logger.info("Building TTNN model...")
        ttnn_args = Tt3DetrArgs()
        ttnn_args.parameters = ref_module_parameters
        ttnn_args.device = ttnn_device

        ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

        # Initialize APCalculator for TTNN model
        ttnn_ap_calculator = APCalculator(
            dataset_config=dataset_config,
            ap_iou_thresh=[0.25, 0.5],
            class2type_map=dataset_config.class2type,
            exact_eval=True,
        )

        # Iterate over batches
        for batch_idx, batch_data_label in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx}")

            # Move to device
            torch_device = next(ref_module.parameters()).device
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(torch_device)

            # Create input dict
            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
                "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            }

            # Run PyTorch reference inference
            with torch.no_grad():
                ref_outputs = ref_module(inputs=inputs, encoder_only=encoder_only)

            # Accumulate AP for PyTorch model
            ref_ap_calculator.step_meter(ref_outputs, batch_data_label)

            # Run TTNN model inference
            tt_outputs = ttnn_module(inputs=inputs, encoder_only=encoder_only)

            # Convert ALL TTNN outputs to float32 for AP calculation
            logger.info("Converting TTNN outputs to float32 for AP calculation...")
            for key in tt_outputs["outputs"]:
                tensor = tt_outputs["outputs"][key]

                # Convert TTNN tensor to PyTorch tensor if needed
                if not isinstance(tensor, torch.Tensor):
                    tensor = ttnn.to_torch(tensor)

                # Convert BFloat16 to Float32
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()

                tt_outputs["outputs"][key] = tensor
            # Accumulate AP for TTNN model
            ttnn_ap_calculator.step_meter(tt_outputs, batch_data_label)

            # Compare outputs (PCC)
            if not encoder_only:
                logger.info(f"Comparing outputs for batch {batch_idx}...")
                SKIP_KEYS = ["angle_continuousk"]

                for key in ref_outputs["outputs"]:
                    if key in SKIP_KEYS:
                        continue

                    passing, pcc_message = comp_pcc(ref_outputs["outputs"][key], tt_outputs["outputs"][key], 0.97)
                    logger.info(f"Batch {batch_idx} Output Key '{key}' PCC: {pcc_message}")

        # Compute final AP metrics
        logger.info("\n" + "=" * 50)
        logger.info("Computing Average Precision metrics...")
        logger.info("=" * 50)

        ref_metrics = ref_ap_calculator.compute_metrics()
        ttnn_metrics = ttnn_ap_calculator.compute_metrics()

        ref_metric_str = ref_ap_calculator.metrics_to_str(ref_metrics)
        ttnn_metric_str = ttnn_ap_calculator.metrics_to_str(ttnn_metrics)

        logger.info("\nPyTorch Model AP Metrics:")
        logger.info(ref_metric_str)

        logger.info("\nTTNN Model AP Metrics:")
        logger.info(ttnn_metric_str)

        logger.info("\nDETR3D inference with AP calculation completed!")

    finally:
        ttnn.close_device(ttnn_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR3D 3D Object Detection Inference")
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        required=True,
        help="Path to point cloud file (.npz format, e.g., 000001_pc.npz)",
    )
    parser.add_argument(
        "--test-ckpt",
        type=str,
        default=None,
        help="Path to test checkpoint (.pth file)",
    )
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    run_detr3d_inference(
        args.dataset_root_dir,
        args.test_ckpt,
        seed=args.seed,
    )
