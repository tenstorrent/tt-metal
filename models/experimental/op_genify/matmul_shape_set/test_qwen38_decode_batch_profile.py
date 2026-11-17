# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile stock Qwen3.8-27B decode at batch 1, 16 and 32 (#65148: decode M <= 32).

Same generator, fabric, mesh and reduced stack (layers 0 and 3) as test_qwen38_profile.py. Each step is
the generator's own `_model_step()` (model.decode on its cache, page table, positions and rope indices),
run eagerly; the decode trace replays the same ops.

Run (needs MODEL_WEIGHTS_DIR and a Tracy build):
    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_qwen38_decode_batch_profile.py
"""

from pathlib import Path

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric

BATCHES = (1, 16, 32)


def test_qwen38_decode_batch_profile():
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(
            Path(__file__).resolve().parents[3] / "demos" / "qwen38_27b_qb2",
            mesh,
            layer_indices=[0, 3],
            sampling_strategy="split",
            head_strategy="dram",
        )
        ttnn.ReadDeviceProfiler(mesh)
        for batch in BATCHES:
            try:
                gen._ensure_cache(batch, 256)
                gen._model_step()  # warm-up: compile
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                signpost(f"DECODE_B{batch}")
                gen._model_step()
                ttnn.synchronize_device(mesh)
                signpost(f"DECODE_B{batch}_END")
                ttnn.ReadDeviceProfiler(mesh)
            except Exception as e:  # report and keep the batches that ran
                if batch != BATCHES[-1]:
                    raise
                logger.warning(f"decode at batch {batch} failed: {e}")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
