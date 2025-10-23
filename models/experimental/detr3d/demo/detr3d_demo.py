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
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.common.utility_functions import comp_pcc
from torch.utils.data import DataLoader


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.modules = None
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
    batch_size=1,
    num_workers=4,
    seed=0,
):
    """Run DETR3D inference using DataLoader like the main training script"""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup model configuration
    args = Detr3dArgs()
    args.dataset_name = "sunrgbd"
    args.dataset_root_dir = dataset_root_dir
    args.batchsize_per_gpu = batch_size
    args.dataset_num_workers = num_workers
    args.use_color = False

    # Build dataset
    from models.experimental.detr3d.source.detr3d.datasets import build_dataset

    datasets, dataset_config = build_dataset(args)

    # Create DataLoader for test split
    test_dataset = datasets["test"]
    sampler = torch.utils.data.SequentialSampler(test_dataset)

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

    # Open TTNN device BEFORE the loop
    ttnn_device = ttnn.open_device(device_id=0, l1_small_size=16384)

    try:
        # Preprocess model parameters ONCE
        logger.info("Preprocessing model parameters...")
        ref_module_parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_module,
            custom_preprocessor=create_custom_mesh_preprocessor(None),
            device=ttnn_device,
        )

        # Build TTNN model ONCE
        logger.info("Building TTNN model...")
        ttnn_args = Tt3DetrArgs()
        ttnn_args.modules = ref_module
        ttnn_args.parameters = ref_module_parameters
        ttnn_args.device = ttnn_device

        ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

        # Iterate over batches - run BOTH models on same input
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
            logger.info(f"Running PyTorch model on batch {batch_idx}...")
            with torch.no_grad():
                ref_outputs = ref_module(inputs=inputs, encoder_only=encoder_only)

            # Run TTNN model inference on SAME input
            logger.info(f"Running TTNN model on batch {batch_idx}...")
            tt_outputs = ttnn_module(inputs=inputs, encoder_only=encoder_only)

            # Compare outputs for this batch
            if encoder_only:
                logger.info(f"Comparing encoder outputs for batch {batch_idx}...")
                for idx, (tt_out, torch_out) in enumerate(zip(tt_outputs, ref_outputs)):
                    if not isinstance(tt_out, torch.Tensor):
                        tt_out = ttnn.to_torch(tt_out)
                        tt_out = torch.reshape(tt_out, torch_out.shape)

                    passing, pcc_message = comp_pcc(torch_out, tt_out, 0.97)
                    logger.info(f"Batch {batch_idx} Encoder Output {idx} PCC: {pcc_message}")

                    if passing:
                        logger.info(f"Batch {batch_idx} Encoder Output {idx} Test Passed!")
                    else:
                        logger.warning(f"Batch {batch_idx} Encoder Output {idx} Test Failed!")
            else:
                # Compare decoder outputs
                logger.info(f"Comparing model outputs for batch {batch_idx}...")
                SKIP_KEYS = []

                for key in ref_outputs["outputs"]:
                    if key in SKIP_KEYS:
                        logger.info(f"Output Key '{key}' - Skipped")
                        # import pdb; pdb.set_trace()
                        continue

                    passing, pcc_message = comp_pcc(ref_outputs["outputs"][key], tt_outputs["outputs"][key], 0.97)
                    logger.info(f"Batch {batch_idx} Output Key '{key}' PCC: {pcc_message}")

                    if passing:
                        logger.info(f"Batch {batch_idx} Output Key '{key}' Test Passed!")
                    else:
                        logger.warning(f"Batch {batch_idx} Output Key '{key}' Test Failed!")

        logger.info("DETR3D inference completed!")

    finally:
        ttnn.close_device(ttnn_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR3D 3D Object Detection Inference")
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        required=True,
        help="Root directory containing the dataset files",
    )
    parser.add_argument(
        "--test-ckpt",
        type=str,
        default=None,
        help="Path to test checkpoint (.pth file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    run_detr3d_inference(
        args.dataset_root_dir,
        args.test_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
