# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The VAE decoder on real weights against the pinned diffusers decoder run in float64 on the CPU (its post_quant_conv
is folded into our proj_in). Needs MINIMAX_H3_MODEL_PATH, and MINIMAX_H3_REF_SRC when diffusers lacks the module."""

import glob
import importlib.util
import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn

from ..common import DECODE_LATENT_FRAMES, LATENT_TILE, build_visual_decoder, load_config, weights_subdir
from .....models.vae.minimax_h3.decoder_minimax_h3 import unpatchify
from .....models.vae.minimax_h3.vae_minimax_h3 import prepare_decoder_state

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]


def _reference_module():
    try:
        from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

        return ref
    except ImportError:
        pass
    src = os.environ.get("MINIMAX_H3_REF_SRC")
    if not src or not os.path.isfile(src):
        pytest.skip(
            "pinned diffusers reference not installed; set MINIMAX_H3_REF_SRC to its autoencoder_kl_minimax_h3.py"
        )
    import diffusers.models.autoencoders as pkg

    name = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(pkg, "autoencoder_kl_minimax_h3", module)
    return module


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, str]:
    a = actual.double().flatten()
    e = expected.double().flatten()
    pcc = float(torch.corrcoef(torch.stack([a, e]))[0, 1])
    rel = float(torch.linalg.norm(a - e) / torch.linalg.norm(e))
    return pcc, f"pcc {pcc:.6f}, rel-RMSE {rel:.3e}, max |diff| {float((a - e).abs().max()):.3e}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decoder_against_reference(mesh_device):
    from safetensors import safe_open

    from models.tt_dit.models.vae.minimax_h3.blockings_minimax_h3_vae import register_h3_vae_decoder_blockings

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    ref = _reference_module()
    config = load_config(weights_dir)
    state = {}
    for path in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as shard:
            for key in shard.keys():
                if key.startswith(("decoder.", "post_quant_conv.")):
                    state[key] = shard.get_tensor(key)

    channels = config["latent_channels"]
    reference = ref.MiniMaxH3VideoViTDecoder3d(
        in_channels=channels,
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"],
        num_attention_heads=config["decoder_num_attention_heads"],
        attention_head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        ffn_mult=config["decoder_ffn_mult"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        norm_eps=config["decoder_norm_eps"],
    )
    reference.load_state_dict({k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")})
    post = torch.nn.Conv3d(channels, channels, kernel_size=1)
    post.load_state_dict({"weight": state["post_quant_conv.weight"], "bias": state["post_quant_conv.bias"]})
    reference, post = reference.double().eval(), post.double().eval()

    torch.manual_seed(4)
    z = torch.randn(1, channels, DECODE_LATENT_FRAMES, LATENT_TILE, LATENT_TILE)
    mark = time.perf_counter()
    with torch.no_grad():
        expected = reference(post(z.double()))
    logger.info(f"DECREF float64 reference decode {time.perf_counter() - mark:.1f} s, output {tuple(expected.shape)}")

    register_h3_vae_decoder_blockings()
    tokens = z.permute(0, 2, 3, 4, 1).reshape(1, -1, channels)
    dev = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    decoder = build_visual_decoder(config, mesh_device)
    decoder.load_torch_state_dict(prepare_decoder_state(state))
    out = decoder(dev)
    ttnn.synchronize_device(mesh_device)
    mark = time.perf_counter()
    out = decoder(dev)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - mark) * 1e3
    pixels = unpatchify(
        ttnn.to_torch(out).float(),
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        out_channels=config["out_channels"],
    )
    pcc, text = _metrics(pixels, expected)
    logger.info(f"DECREF decoder vs float64 reference: {text}; decoder {ms:.1f} ms")
    # The fused q/k RMS path measures pcc 0.999404 / rel-RMSE 3.019e-2; the rms_norm pair it replaced 0.999368 / 3.105e-2.
    assert pcc > 0.999, f"decoder drifted from the reference: pcc {pcc:.6f}"
