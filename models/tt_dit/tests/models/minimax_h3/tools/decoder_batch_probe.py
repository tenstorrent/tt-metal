# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Two decode tiles per device per wave: the ViT decoder forward at B=1 vs B=2 on one chip (random weights, the Galaxy's
11x10 matmul grid forced, M=3648 blockings registered), timed and checked slice by slice against the B=1 outputs.
Kill criterion from the plan: B=2 taking >= 1.90x the B=1 wave means < 5 % per tile and the idea is dropped.
    pytest models/tt_dit/tests/models/minimax_h3/tools/decoder_batch_probe.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

from ..common import DECODE_LATENT_FRAMES, LATENT_TILE, build_visual_decoder, load_config, weights_subdir
from .....models.vae.minimax_h3.vae_minimax_h3 import prepare_decoder_state

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
SEQ = 1824  # padded decoder sequence (1792 patches + suffix)
DIM = 2048


def _timed(mesh_device, fn, n=5):
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
def test_decoder_batch_two(mesh_device):
    from models.tt_dit.models.vae.minimax_h3 import decoder_minimax_h3 as decoder_module
    from models.tt_dit.models.vae.minimax_h3.blockings_minimax_h3_vae import register_h3_vae_decoder_blockings
    from models.tt_dit.utils.matmul import register_matmul_configs

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    torch.manual_seed(1)

    # The pipeline's matmuls run on the Galaxy's 11x10 grid; a one-chip mesh would get 12x10 and other blockings.
    decoder_module.get_matmul_core_grid = lambda _device: ttnn.CoreCoord(11, 10)
    register_h3_vae_decoder_blockings()
    # B=2 doubles M; start from the swept M=1824 tuples with the M block that 11x10 actually runs (6 = 60 tiles / 10 rows).
    register_matmul_configs(
        {
            "11x10": {
                (2 * SEQ, DIM, 3 * DIM): (6, 2, 10, (2, 2)),
                (2 * SEQ, DIM, DIM): (6, 2, 6, (2, 2)),
                (2 * SEQ, DIM, 8 * DIM): (6, 4, 12, (2, 2)),
                (2 * SEQ, 4 * DIM, DIM): (6, 4, 6, (2, 2)),
            }
        }
    )

    import glob
    import os

    from safetensors import safe_open

    decoder = build_visual_decoder(config, mesh_device)
    # The real checkpoint, decoder keys only, from however many shards it has (the reference module's random state
    # needs the pinned diffusers, absent here).
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
    t0, t1, t01 = dev(latents[0:1]), dev(latents[1:2]), dev(latents)

    out0, ms0 = _timed(mesh_device, lambda: decoder(t0))
    out1, ms1 = _timed(mesh_device, lambda: decoder(t1))
    try:
        out01, ms01 = _timed(mesh_device, lambda: decoder(t01))
    except Exception as exc:  # noqa: BLE001
        logger.info(f"DECBATCH B=2 FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        raise
    ref0, ref1 = ttnn.to_torch(out0), ttnn.to_torch(out1)
    got = ttnn.to_torch(out01)
    eq0, eq1 = bool(torch.equal(got[0:1], ref0)), bool(torch.equal(got[1:2], ref1))
    d0 = (got[0:1].float() - ref0.float()).abs().max().item()
    d1 = (got[1:2].float() - ref1.float()).abs().max().item()
    ratio = ms01 / max(ms0, 1e-9)
    logger.info(
        f"DECBATCH one chip, 11x10 matmul grid: B=1 {ms0:.2f} ms (second tile {ms1:.2f}), B=2 {ms01:.2f} ms = {ratio:.3f}x "
        f"-> per tile {ms01 / 2:.2f} ms ({100 * (1 - ms01 / (2 * ms0)):.1f} % saved); slices bit-identical {eq0}/{eq1} "
        f"(max |diff| {d0:.2e}/{d1:.2e}); kill if >= 1.90x"
    )
