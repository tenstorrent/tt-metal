# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy capture of ONE warmed eager audio decode on a 4x8 Galaxy at T-shard factor 8, for a squeeze recipe.

Profiling harness, not a gate. ``SQZ_RECIPE`` names a recipe of ``test_audio_decode_squeeze.py`` (default ``full``);
one forward inside the signposted window. Run with the device dump disabled and read the profiler's C++ report:

    python -m tracy -p -v --op-support-count 8000 --disable-device-data-dump-to-files \\
      --disable-device-data-push-to-tracy -m pytest \\
      models/tt_dit/tests/models/minimax_h3/tools/tracy_audio_decode_sqz_harness.py -k 600lat_b1 -s
    # -> generated/profiler/.logs/cpp_device_perf_report.csv (per op per chip)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.models.minimax_h3.test_audio_decode_squeeze import (
    PACK_KEY,
    RECIPES,
    _load,
    apply_recipe,
)
from models.tt_dit.utils.test import line_params_8k

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**line_params_8k, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8_8k",
    )
]
HOP_LENGTH = 800


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    ("num_latent_frames", "batch"),
    [(207, 1), (207, 2), (600, 1), (600, 2)],
    ids=["207lat_b1", "207lat_b2", "600lat_b1", "600lat_b2"],  # single tokens: `python -m tracy` re-splits -k args
)
def test_tracy_audio_decode_sqz(mesh_device, num_latent_frames, batch):
    from tracy import signpost

    name = os.environ.get("SQZ_RECIPE", "full")
    recipe = RECIPES[name]
    decoder, config = _load(mesh_device, pack_bands=recipe.get(PACK_KEY))
    counts = apply_recipe(decoder, recipe)
    torch.manual_seed(2)
    latents = torch.randn(batch, config["latent_channels"], num_latent_frames)
    _ = decoder(latents)  # warm the program cache outside the window
    ttnn.synchronize_device(mesh_device)
    logger.info(f"tracy: recipe {name} {counts}, {num_latent_frames} latents x batch {batch}")
    signpost("start")
    _ = decoder(latents)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
