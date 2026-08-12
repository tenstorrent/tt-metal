"""Does snake fusion survive on a multi-device mesh? One unsharded decode on the 4x8 mesh.

This is the exact case that killed `test_audio_decode_t_parallel`'s t_factor=1 baseline:
`_snake_conv_params` read alpha/beta with a bare `ttnn.to_torch`, but those parameters are replicated
across the mesh, so it hit `buffers.size() == 1` (pytensor.cpp:299). Because the baseline died, the
test's `assert baseline_ran` fired and the whole T-parallel path read as broken.

Cheaper than rerunning the 12-minute t_parallel test: build once, decode once, print the checksum.
Passing here means fusion and a mesh are now a reachable combination.

Run with the fix on PYTHONPATH ahead of TT_METAL_HOME:
  PYTHONPATH=<worktree>:$PYTHONPATH MINIMAX_H3_AUDIO_FUSE_BAND=1 MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV=1 \
    python fusion_on_mesh.py
"""

import json
import os
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.layers import audio_resample
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

print(f"audio_resample loaded from: {audio_resample.__file__}")
print(f"fusion enabled: {audio_resample.fuse_snake_into_conv_enabled()}")

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    print(f"mesh devices: {d.get_num_devices()}")
    dec = MiniMaxH3AudioDecoder(
        latent_channels=cfg["latent_channels"],
        latent_dim=cfg["latent_dim"],
        decoder_dim=cfg["decoder_dim"],
        decoder_rates=tuple(cfg["decoder_rates"]),
        decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(x) for x in cfg["resblock_dilation_sizes"]),
        mesh_device=d,
        parallel_config=None,  # unsharded on a 32-chip mesh: the case that crashed
        ccl_manager=None,
    )
    dec.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors"))),
        strict=False,
    )
    torch.manual_seed(2)
    lat = torch.randn(2, cfg["latent_channels"], 207) * 0.1
    t0 = time.perf_counter()
    out = dec(lat)
    ttnn.synchronize_device(d)
    dt = time.perf_counter() - t0
    o = torch.as_tensor(out).float()
    print(f"RESULT fusion-on-mesh OK: shape {tuple(o.shape)} absmax {o.abs().max():.6f} first decode {dt:.4f}s")
    assert o.abs().max() > 1e-6, "all-zero output would make this vacuous"
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
