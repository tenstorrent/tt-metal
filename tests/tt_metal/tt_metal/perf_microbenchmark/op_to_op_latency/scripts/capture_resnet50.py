#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE REAL ResNet-50 inference forward under ttnn graph capture, dump JSON for
raw_hazard_analyzer.py. Single Wormhole chip (1x1 mesh / N150).

This is the REAL ttnn ResNet-50 (models/demos/vision/classification/resnet50) run through its own
test infra -- exactly the setup in
models/demos/vision/classification/resnet50/wormhole/tests/test_resnet50_functional.py, reduced to a
single captured forward. CNN => no in-place ops, so the RAW dependency structure should be fully
resolvable.

Weights: use_pretrained_weight=False builds a RANDOM-weight torchvision.models.resnet50() (NO
download; the device-program dependency/hazard structure is independent of weight VALUES -- the conv
shapes/config are what matter). Batch 16 (batch 1/2/8 are skipped by the infra on WH).
"""
import json
import sys

import torch
from loguru import logger

import ttnn
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
    create_test_infra,
)

OUT = "/tmp/resnet_capture.json"
BATCH_SIZE = 16
L1_SMALL_SIZE = 24576


def main():
    torch.manual_seed(0)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE)
    try:
        test_infra = create_test_infra(
            mesh_device,
            BATCH_SIZE,
            act_dtype=ttnn.bfloat8_b,
            weight_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            use_pretrained_weight=False,  # random weights -> fully offline, no download
        )
        logger.info(f"ResNet-50 test infra built (batch={BATCH_SIZE}, random weights)")

        tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(mesh_device)

        # Warm up program cache / JIT-configure the convs OUTSIDE capture so the graph reflects
        # steady-state dispatch (the first invocation compiles/allocs; we want a cached invocation).
        test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
        test_infra.run()
        ttnn.synchronize_device(mesh_device)
        test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
        test_infra.run()
        ttnn.synchronize_device(mesh_device)

        # Fresh input for the captured forward.
        test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)

        logger.info("Capturing ONE ResNet-50 forward...")
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = test_infra.run()
        ttnn.synchronize_device(mesh_device)
        captured = ttnn.graph.end_graph_capture()

        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    sys.exit(main())
