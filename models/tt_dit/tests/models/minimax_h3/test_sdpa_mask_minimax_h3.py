# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The decoder without its dense attention mask against the decoder with it, real weights, one wave.

`MINIMAX_H3_SDPA_MASK=view` presents q/k/v to SDPA with the logical sequence length of the valid
tokens, so the op masks the tile-pad keys itself; `dense` is the tip's (1,1,S,S) -inf mask. The valid
rows must agree bit for bit and every row, pad rows included, must stay finite -- the pad rows feed
the next layer's K/V through the same view and a NaN there would poison a softmax.

Runs on the 4x8 ring like the served decode and loads the decoder out of the TT_DIT weight cache, so it
needs a served decode to have populated it (MINIMAX_H3_MODEL_PATH for the config only).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

from .common import weights_subdir

MESH_4X8_RING = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="4x8ring",
    )
]
MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)


def _cached_vae(mesh_device):
    from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
    from ....parallel.config import ParallelFactor, VAEParallelConfig
    from ....parallel.manager import CCLManager
    from ....pipelines.minimax_h3.pipeline_minimax_h3 import MODEL_NAME
    from ....utils import cache
    from ....utils.conv3d import conv3d_blocking_hash

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(mesh_axis=0, factor=1))

    def load_cached(module, subfolder, state):
        blocking = conv3d_blocking_hash(module)
        cache.load_model(
            module,
            model_name=MODEL_NAME,
            subfolder=f"{subfolder}_{blocking}" if blocking else subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            dtype="fp32",
            get_torch_state_dict=lambda: (_ for _ in ()).throw(
                RuntimeError(f"cache miss for {subfolder}; run a served decode first to populate it")
            ),
        )

    vae = MiniMaxH3Vae(
        config,
        task="t2va",
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring),
        device_stitch=True,
        weight_loader=load_cached,
        pixel_denorm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD),
    )
    load_cached(vae.decoder, vae._decoder_subfolder(), {})
    return vae, config


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8_RING, indirect=["mesh_device", "device_params"])
def test_sdpa_mask_view_matches_dense(mesh_device):
    vae, config = _cached_vae(mesh_device)
    decoder = vae.decoder
    num_frames, height, width = decoder.latent_shape
    patches = num_frames * height * width
    valid = decoder.num_patches + decoder.num_suffix_tokens
    torch.manual_seed(4)
    tokens = ttnn.from_torch(
        torch.randn(mesh_device.get_num_devices(), patches, config.latent_channels),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    outputs, timings = {}, {}
    for mode in ("dense", "view"):
        decoder.sdpa_mask = mode
        out = decoder(tokens)  # program cache
        ttnn.synchronize_device(mesh_device)
        best = float("inf")
        for _ in range(5):
            ttnn.synchronize_device(mesh_device)
            mark = time.perf_counter()
            out = decoder(tokens)
            ttnn.synchronize_device(mesh_device)
            best = min(best, time.perf_counter() - mark)
        timings[mode] = best * 1e3
        outputs[mode] = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
        logger.info(
            f"sdpa_mask={mode}: decoder wave min-of-5 wall {best * 1e3:.1f} ms, out {tuple(outputs[mode].shape)}"
        )

    dense, view = outputs["dense"], outputs["view"]
    assert dense.shape == view.shape, f"{tuple(dense.shape)} != {tuple(view.shape)}"
    assert torch.isfinite(view).all(), f"{(~torch.isfinite(view)).sum().item()} non-finite values with the view"
    assert torch.isfinite(dense).all()
    differing = (dense[:, :valid] != view[:, :valid]).sum().item()
    logger.info(
        f"valid rows {valid} of {dense.shape[1]}: {differing} values differ; pad rows max |diff| "
        f"{(dense[:, valid:] - view[:, valid:]).abs().max().item():.3e}; wall dense {timings['dense']:.1f} ms, "
        f"view {timings['view']:.1f} ms"
    )
    assert differing == 0, f"{differing} of {dense[:, :valid].numel()} valid-row values differ between dense and view"
