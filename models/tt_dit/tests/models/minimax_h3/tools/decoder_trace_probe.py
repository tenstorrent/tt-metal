# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Does replaying the ViT decoder forward as a captured trace save device time? One chip, real weights, the Galaxy's 11x10
matmul grid, the shipping unit (7 latent frames x 16 x 16): eager min-of-10 vs trace replay min-of-10, outputs compared
bit for bit on two different inputs (the second replay refreshes the captured input buffer). Kill if the replay saves < 3 ms
of the ~92 ms wave (workflow idea 7 / S-8).
    pytest models/tt_dit/tests/models/minimax_h3/tools/decoder_trace_probe.py -s
"""

import glob
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ..common import DECODE_LATENT_FRAMES, LATENT_TILE, build_visual_decoder, load_config, weights_subdir
from .....models.vae.minimax_h3.vae_minimax_h3 import prepare_decoder_state
from .....utils.tracing import traced_function

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536, "trace_region_size": 400_000_000}, id="single_device_trace")]


def _timed(mesh_device, fn, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decoder_trace_replay(mesh_device):
    from safetensors import safe_open

    from models.tt_dit.models.vae.minimax_h3 import decoder_minimax_h3 as decoder_module
    from models.tt_dit.models.vae.minimax_h3.blockings_minimax_h3_vae import register_h3_vae_decoder_blockings

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    torch.manual_seed(1)
    decoder_module.get_matmul_core_grid = lambda _device: ttnn.CoreCoord(11, 10)
    register_h3_vae_decoder_blockings()
    decoder = build_visual_decoder(config, mesh_device)
    state = {}
    for path in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as shard:
            for key in shard.keys():
                if key.startswith(("decoder.", "post_quant_conv.")):
                    state[key] = shard.get_tensor(key)
    decoder.load_torch_state_dict(prepare_decoder_state(state))

    num_patches = DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE
    latents = torch.randn(2, num_patches, config["latent_channels"])
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    t0, t1 = dev(latents[0:1]), dev(latents[1:2])

    out_e0, ms_eager = _timed(mesh_device, lambda: decoder(t0))
    ref0 = ttnn.to_torch(out_e0)
    ref1 = ttnn.to_torch(decoder(t1))

    @traced_function(inject_mesh_device=True, clone_prep_inputs=False)
    def run(tokens):
        return decoder(tokens)

    out_t0, ms_first = _timed(mesh_device, lambda: run(t0, mesh_device=mesh_device, traced=True), n=1)
    _, ms_replay = _timed(mesh_device, lambda: run(t0, mesh_device=mesh_device, traced=True))
    got0 = ttnn.to_torch(run(t0, mesh_device=mesh_device, traced=True))
    got1 = ttnn.to_torch(run(t1, mesh_device=mesh_device, traced=True))
    eq0, eq1 = bool(torch.equal(got0, ref0)), bool(torch.equal(got1, ref1))
    logger.info(
        f"DECTRACE one chip, 11x10 matmul grid: eager {ms_eager:.2f} ms, trace replay {ms_replay:.2f} ms "
        f"(saves {ms_eager - ms_replay:.2f} ms/wave, {100 * (1 - ms_replay / ms_eager):.1f} %); replay bit-identical to eager: "
        f"same input {eq0}, refreshed input {eq1}; kill if the saving is < 3 ms"
    )
