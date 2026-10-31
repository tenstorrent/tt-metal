#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone SigLIP validation suite (no pytest-timeout interference).

Combines three validations into a single device-open session to avoid
repeated UMD topology discovery overhead on the flaky n300:
  1. PCC vs HF reference (threshold 0.92)
  2. Determinism (same input, two runs, max_diff < 1e-6)
  3. Output shape ([1, 729, 1152])

Usage:
    MESH_DEVICE=N300 python models/demos/llava/tests/siglip_validation_smoke.py
    MESH_DEVICE=N150 python models/demos/llava/tests/siglip_validation_smoke.py
"""

import os
import sys
import time

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llava.reference.llava_reference import create_siglip_encoder as create_siglip_encoder_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.vision_encoders.siglip_encoder import create_siglip_encoder as create_siglip_encoder_ttnn

PCC_THRESHOLD = 0.92


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    mesh_str = os.environ.get("MESH_DEVICE", "N300")
    mesh_shape = (1, 1) if mesh_str == "N150" else (1, 2)
    model_name = "google/siglip-so400m-patch14-384"
    batch_size = 1

    logger.info("=" * 80)
    logger.info(f"SigLIP Validation Suite — device={mesh_str}, model={model_name}")
    logger.info("=" * 80)

    t0 = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    logger.info(f"Device opened in {time.time()-t0:.1f}s — {mesh.get_num_devices()} chips")

    results = {}
    try:
        args = VisionModelArgs(
            mesh_device=mesh,
            max_batch_size=batch_size,
            vision_encoder_type="siglip",
            vision_hidden_size=1152,
            vision_num_patches=729,
            vision_resolution=384,
        )

        # Load HF reference + state_dict
        logger.info("Loading HF SigLIP reference...")
        t1 = time.time()
        pytorch_encoder = create_siglip_encoder_pytorch(model_name=model_name, device="cpu")
        state_dict = pytorch_encoder.vision_model.state_dict()
        logger.info(f"  HF reference loaded in {time.time()-t1:.1f}s")

        # Build TTNN encoder
        logger.info("Constructing TTNN SigLIP encoder...")
        t1 = time.time()
        ttnn_encoder = create_siglip_encoder_ttnn(args=args, state_dict=state_dict)
        logger.info(f"  TTNN encoder constructed in {time.time()-t1:.1f}s")

        torch.manual_seed(42)
        test_input = torch.randn(batch_size, 3, 384, 384)

        # --- Test 1: Shape ---
        logger.info("--- Test 1/3: Output Shape ---")
        t1 = time.time()
        ttnn_output = ttnn_encoder(test_input)
        elapsed_forward = time.time() - t1
        ttnn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_output)[0])
        actual_shape = tuple(ttnn_output_torch.shape)
        expected_shape = (batch_size, 729, 1152)
        shape_pass = actual_shape == expected_shape
        logger.info(f"  Expected: {expected_shape}, Actual: {actual_shape} — {'PASS' if shape_pass else 'FAIL'}")
        results["shape"] = shape_pass
        results["forward_time_s"] = elapsed_forward

        # --- Test 2: PCC vs HF reference ---
        logger.info("--- Test 2/3: PCC vs HF Reference ---")
        with torch.no_grad():
            pytorch_output = pytorch_encoder(test_input)
        passing_pcc, pcc_value = comp_pcc(pytorch_output, ttnn_output_torch, pcc=PCC_THRESHOLD)
        diff = (pytorch_output - ttnn_output_torch).abs()
        logger.info(f"  PCC: {pcc_value:.6f} (threshold {PCC_THRESHOLD}) — {'PASS' if passing_pcc else 'FAIL'}")
        logger.info(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
        results["pcc_value"] = pcc_value
        results["pcc_pass"] = passing_pcc

        # --- Test 3: Determinism ---
        logger.info("--- Test 3/3: Determinism ---")
        torch.manual_seed(99)
        test_input2 = torch.randn(batch_size, 3, 384, 384)

        output1 = ttnn.to_torch(ttnn.get_device_tensors(ttnn_encoder(test_input))[0])
        output2 = ttnn.to_torch(ttnn.get_device_tensors(ttnn_encoder(test_input))[0])
        max_diff = (output1 - output2).abs().max().item()
        det_pass = max_diff < 1e-6
        logger.info(f"  Max diff between 2 runs: {max_diff:.10f} — {'PASS' if det_pass else 'FAIL'}")
        results["determinism_max_diff"] = max_diff
        results["determinism_pass"] = det_pass

        # Summary
        all_pass = results["shape"] and results["pcc_pass"] and results["determinism_pass"]
        logger.info("=" * 80)
        logger.info("RESULTS SUMMARY:")
        logger.info(f"  Shape:       {'PASS' if results['shape'] else 'FAIL'} ({actual_shape})")
        logger.info(
            f"  PCC:         {results['pcc_value']:.6f} — {'PASS' if results['pcc_pass'] else 'FAIL'} (threshold {PCC_THRESHOLD})"
        )
        logger.info(
            f"  Determinism: {results['determinism_max_diff']:.10f} — {'PASS' if results['determinism_pass'] else 'FAIL'}"
        )
        logger.info(f"  Forward time: {results['forward_time_s']:.2f}s")
        logger.info(f"  Device: {mesh_str} ({mesh.get_num_devices()} chips)")
        logger.info(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        logger.info("=" * 80)

    finally:
        ttnn.close_mesh_device(mesh)
        logger.info("Device closed.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
