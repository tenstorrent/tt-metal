# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Torch reference implementations for RVC Flow Decoder and HiFi-GAN Generator.

Uses the actual synthesizer module classes (ResidualCouplingBlock, GeneratorNSF)
from torch_impl/vc/synthesizer.py with checkpoint weights loaded. This ensures
exact numerical parity with the torch implementation.

Used by:
  - test_runtime.py (PCC validation of TTNN output)
  - demo.py (generating reference WAV for comparison)
"""

import math
import torch

from models.demos.rvc.torch_impl.vc.synthesizer import (
    ResidualCouplingBlock,
    GeneratorNSF,
)


def load_flow_torch_modules(sd):
    """Build a torch ResidualCouplingBlock with checkpoint weights.

    Args:
        sd: checkpoint state dict (from safetensors.torch.load_file)

    Returns:
        dict with 'flow' key containing the loaded torch module.
    """
    flow = ResidualCouplingBlock(
        channels=192, hidden_channels=192,
        kernel_size=5, dilation_rate=1, num_layers=3,
        gin_channels=256,
    )
    # Load only flow.* keys
    flow_sd = {k.replace("flow.", "", 1): v for k, v in sd.items() if k.startswith("flow.")}
    flow.load_state_dict(flow_sd)
    flow.eval()
    return {"flow": flow}


def torch_flow_forward(z_p, g, flow_mods):
    """Run torch flow decoder.

    Args:
        z_p:       [1, 192, T] latent prior
        g:         [1, 256, 1] speaker embedding
        flow_mods: dict from load_flow_torch_modules()

    Returns:
        z: [1, 192, T] decoded latent
    """
    with torch.no_grad():
        return flow_mods["flow"](z_p, g=g)


def build_torch_generator(sd):
    """Build a torch GeneratorNSF with checkpoint weights.

    Args:
        sd: checkpoint state dict

    Returns:
        dict with 'gen' key containing the loaded torch module.
    """
    gen = GeneratorNSF(
        initial_channel=192,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[12, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[24, 20, 4, 4],
        gin_channels=256,
        sr=48000,
        validation=True,
    )
    # Load only dec.* keys
    dec_sd = {k.replace("dec.", "", 1): v for k, v in sd.items() if k.startswith("dec.")}
    gen.load_state_dict(dec_sd, strict=False)
    gen.eval()
    return {"gen": gen}


def torch_generator_forward(z, har_source, g, gen_mods):
    """Run torch generator forward pass.

    Args:
        z:          [1, 192, T] latent from flow decoder
        har_source: [1, 1, T*480] harmonic excitation (transposed: [B, C, T])
        g:          [1, 256, 1] speaker embedding
        gen_mods:   dict from build_torch_generator()

    Returns:
        audio: [1, 1, T*480] generated waveform in [-1, 1]
    """
    gen = gen_mods["gen"]
    # GeneratorNSF.forward expects (x, f0, g) but internally
    # generates har_source from f0. For PCC testing we need to
    # bypass that and directly pass har_source.
    # So we replicate the forward manually:
    import torch.nn.functional as F
    from models.demos.rvc.torch_impl.vc.synthesizer import linear_channel_first

    with torch.no_grad():
        x = gen.conv_pre(z)
        if g is not None:
            x = x + linear_channel_first(g, gen.cond_linear)

        for i, (ups, noise_convs) in enumerate(zip(gen.ups, gen.noise_convs)):
            x = F.leaky_relu(x, gen.lrelu_slope)
            x = ups(x)
            if isinstance(noise_convs, torch.nn.Linear):
                x_source = linear_channel_first(har_source, noise_convs)
            else:
                x_source = noise_convs(har_source)
            x = x + x_source[:, :, :x.shape[2]]

            xs = gen.resblocks[i * gen.num_kernels](x)
            for j in range(i * gen.num_kernels + 1, (i + 1) * gen.num_kernels):
                xs += gen.resblocks[j](x)
            x = xs / gen.num_kernels

        x = F.leaky_relu(x)
        x = gen.conv_post(x)
        x = torch.tanh(x)
        return x
