# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device profile of forward_device on a reduced batch (profiler buffers overflow at 1024 when chunked)."""

import sys

import torch
import ttnn
from tracy import signpost

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    _load_reference,
)
from models.experimental.chronos_forecast.tt.model import TtChronos
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

batch = int(sys.argv[1])
precision_name = sys.argv[2]
chunk = None if sys.argv[3] == "none" else int(sys.argv[3])

dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
dev.enable_program_cache()
try:
    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    reference, _ = _load_reference()
    model = TtChronos.from_torch_model(
        dev, reference, precision, l1_chunk_tokens=None if chunk is None else chunk * 160
    )
    torch.manual_seed(0)
    prepared = model.prepare_inputs(context=torch.randn(batch, CONTEXT), num_output_patches=NUM_OUTPUT_PATCHES)
    inputs = model.upload_inputs(prepared)
    for _ in range(2):
        out = model.forward_device(inputs)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(dev)
    signpost("chronos_device_forward_start")
    out = model.forward_device(inputs)
    ttnn.synchronize_device(dev)
    signpost("chronos_device_forward_stop")
    ttnn.ReadDeviceProfiler(dev)
    ttnn.deallocate(out)
    model.deallocate_inputs(inputs)
finally:
    ttnn.close_mesh_device(dev)
