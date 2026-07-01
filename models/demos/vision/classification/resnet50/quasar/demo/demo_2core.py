# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# 2-CORE (reduced-grid) demo for the Quasar resnet50 model.
# -----------------------------------------------------------------------------
# This is a SEPARATE COPY of test_demo_sample from demo.py, configured for a
# tiny 2x1 / 2-worker compute grid (e.g. the craq-sim emulator whose
# compute_with_storage_grid_size() reports only 1-2 cores).
#
# It wires the 2-core variants of the model / test-infra / runner:
#   demo_runner_2core -> resnet50_test_infra_2core -> ttnn_functional_resnet50_2core
# and uses batch_size=2 so the per-core activation shards fit in the sim's
# ~4 MB/bank L1 (batch 16 would need a ~7 MB tensor = ~3.5 MB/bank, which OOMs
# on 2 banks). The number of cores itself comes from the device descriptor
# (compute_with_storage_grid_size()), not from this file; all grids in the
# model/infra clamp themselves down to the device's real core count.
#
# The stock demo.py (batch 16, full grid) is left untouched.
#
# Run (Quasar / craq-sim, slow-dispatch, force JIT):
#   TT_METAL_SIMULATOR=<sim> TT_METAL_SLOW_DISPATCH_MODE=1 \
#   TT_METAL_FORCE_JIT_COMPILE=1 ARCH_NAME=quasar \
#   pytest models/demos/vision/classification/resnet50/quasar/demo/demo_2core.py::test_demo_sample_2core
# =============================================================================

import pytest

from models.demos.vision.classification.resnet50.quasar.demo.demo_runner_2core import run_resnet_inference


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    # 2-core bring-up: batch 2 keeps the per-core activation footprint inside the sim's L1.
    ((2, "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"),),
)
def test_demo_sample_2core(mesh_device, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    run_resnet_inference(batch_size, input_loc, imagenet_label_dict, mesh_device, model_location_generator)
