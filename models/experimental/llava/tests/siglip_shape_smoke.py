#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone SigLIP-shape smoke test (no pytest-timeout interference).

The pytest-timeout=300s global in pytest.ini was killing the SigLIP test
because the n300 ETH topology discovery intermittently takes 5+ minutes
on its own, leaving no time for the actual encoder forward. This script
runs the same logic with no implicit pytest timeout so we can see whether
the SigLIP code itself succeeds once the device is responsive.

Usage:
    source <path to llava_env.sh>
    MESH_DEVICE=N300 python models/demos/llava/tests/siglip_shape_smoke.py
"""

import sys
import time

import torch
from loguru import logger

import ttnn
from models.demos.llava.reference.llava_reference import create_siglip_encoder as create_siglip_encoder_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.vision_encoders.siglip_encoder import create_siglip_encoder as create_siglip_encoder_ttnn


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=== SigLIP shape smoke test ===")
    t0 = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    logger.info(f"Device opened in {time.time()-t0:.1f}s, {mesh.get_num_devices()} chips")

    try:
        model_name = "google/siglip-so400m-patch14-384"
        batch_size = 1

        args = VisionModelArgs(
            mesh_device=mesh,
            max_batch_size=batch_size,
            vision_encoder_type="siglip",
            vision_hidden_size=1152,
            vision_num_patches=729,
            vision_resolution=384,
        )

        logger.info("Loading HF SigLIP reference... (uses cached weights)")
        t1 = time.time()
        pytorch_encoder = create_siglip_encoder_pytorch(model_name=model_name, device="cpu")
        logger.info(f"  HF reference loaded in {time.time()-t1:.1f}s")

        state_dict = pytorch_encoder.vision_model.state_dict()
        logger.info(f"  state_dict has {len(state_dict)} keys")

        logger.info("Constructing TTNN SigLIP encoder...")
        t1 = time.time()
        ttnn_encoder = create_siglip_encoder_ttnn(args=args, state_dict=state_dict)
        logger.info(f"  TTNN encoder constructed in {time.time()-t1:.1f}s")

        test_input = torch.randn(batch_size, 3, 384, 384)

        logger.info("Running TTNN forward...")
        t1 = time.time()
        ttnn_output = ttnn_encoder(test_input)
        elapsed = time.time() - t1
        logger.info(f"  TTNN forward done in {elapsed:.2f}s")

        ttnn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_output)[0])
        logger.info(f"  Output shape: {tuple(ttnn_output_torch.shape)}")
        expected_shape = (batch_size, 729, 1152)
        actual_shape = tuple(ttnn_output_torch.shape)
        assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"
        logger.info(f"PASS: shape matches expected {expected_shape}")
    finally:
        ttnn.close_mesh_device(mesh)
        logger.info("Device closed.")


if __name__ == "__main__":
    main()
