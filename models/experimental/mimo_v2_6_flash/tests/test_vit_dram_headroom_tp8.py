# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How much free DRAM per chip a device ViT pass needs on one row (tt/vision), per row bucket: the multimodal LLM must
leave that much after its KV cache and traces (MiMoV2OmniForCausalLM's KV reservation and startup check,
generator_vllm.VIT_PASS_DRAM_MB).

For each bucket the test fills the row's DRAM down to a given headroom (64 MB filler tensors, interleaved like the
ViT's own) and runs one warmed pass of that bucket; a bisection finds the least headroom that passes. It logs the
table and checks it against VIT_PASS_DRAM_MB (the table the serving code reserves from)."""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes, model_path
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import VIT_PASS_DRAM_MB
from models.experimental.mimo_v2_6_flash.tt.vision.encoder import MiMoVisionEncoder
from models.experimental.mimo_v2_6_flash.tt.vision.vit import PassGeometry

BUCKETS = (4096, 8192, 16384, 32768)
STEP_MB = 16


def free_mb(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return view.total_bytes_free_per_bank * view.num_banks / 2**20, view.largest_contiguous_bytes_free_per_bank


def fill_to(mesh, headroom_mb):
    """Filler tensors until about ``headroom_mb`` MB per chip stay free."""
    fillers = []
    while True:
        free, _ = free_mb(mesh)
        take = min(64.0, free - headroom_mb)
        if take < 1:
            return fillers
        rows = max(32, int(take * 2**20 / (1024 * 2)) // 32 * 32)  # bf16 [rows, 1024]
        fillers.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        )


def runs_with(vit, mesh, rows, headroom_mb):
    fillers = fill_to(mesh, headroom_mb)
    ok = True
    try:
        out = vit.forward_pass(
            torch.zeros(rows, vit.cfg.patch_dim), PassGeometry.build([(1, 32, rows // 32)], vit.buckets)
        )
        del out
        ttnn.synchronize_device(mesh)
    except Exception as e:  # noqa: BLE001 - an allocation failure is the answer
        ok = False
        logger.debug(f"{rows} rows with {headroom_mb} MB free: {type(e).__name__}: {str(e)[:120]}")
    for t in fillers:
        t.deallocate(True)
    gc.collect()
    return ok


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vit_dram_headroom_tp8(mesh_device):
    row = dp_submeshes(mesh_device)[0]
    encoder = MiMoVisionEncoder(row, model_path())
    encoder.warm(BUCKETS)
    vit = encoder.vit
    table = {}
    for rows in BUCKETS:
        lo, hi = 0, 2048  # MB: fails at lo (assumed), passes at hi
        assert runs_with(vit, row, rows, hi), f"{rows} rows fail even with {hi} MB free"
        while hi - lo > STEP_MB:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if runs_with(vit, row, rows, mid) else (mid, hi)
        table[rows] = hi
        logger.info(f"ViT pass of {rows} patch rows: needs {hi} MB of free DRAM per chip (fails at {lo} MB)")
    logger.info(f"ViT pass DRAM headroom per chip (MB) by bucket: {table}; serving reserves {VIT_PASS_DRAM_MB}")
    for rows, need in table.items():
        assert VIT_PASS_DRAM_MB[rows] >= need, (rows, need, VIT_PASS_DRAM_MB[rows])
