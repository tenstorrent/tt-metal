# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast probe: does a candidate bf16 VAE conv blocking table FIT L1 and hold PCC vs HF?

Applies a candidate table (edited inline below or via _VAE_BF16_BLACKHOLE), builds the bf16 VAE,
runs one decode at T=120, and prints PASS/FAIL + PCC. Catches L1 OOM (TT_THROW) per-shape so we can
see which blocking is too big BEFORE the slow Tracy profile. Not a benchmark — a fit/PCC gate.

    python models/experimental/acestep/perf/try_vae_blocking.py
"""

import torch
import ttnn

from diffusers import AutoencoderOobleck
from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.weight_utils import vae_dir
from models.experimental.acestep.tt.vae_decoder import OobleckDecoder, OobleckVAEConfig
from models.experimental.acestep.tt.model_config import _effective_vae_decoder_state
from models.experimental.acestep.tt.vae_conv_config import apply_vae_conv3d_config


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()
        cfg = OobleckVAEConfig.from_diffusers(vae.config)
        torch.manual_seed(0)
        lat = torch.randn(1, cfg.decoder_input_channels, 120)
        with torch.no_grad():
            ref = vae.decoder(lat)

        n_reg = apply_vae_conv3d_config(dev, ttnn.bfloat16)
        print(f"registered {n_reg} bf16 blockings")
        dec = OobleckDecoder(cfg, mesh_device=dev, dtype=ttnn.bfloat16)
        dec.load_torch_state_dict(_effective_vae_decoder_state(vae.decoder))
        lt = ttnn.from_torch(
            lat.transpose(1, 2).contiguous(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        out = ttnn.to_torch(dec.forward(lt)).float()[..., : cfg.audio_channels]
        out = out.reshape(1, -1, cfg.audio_channels).transpose(1, 2)
        n = min(ref.shape[-1], out.shape[-1])
        ok, msg = comp_pcc(ref[..., :n], out[..., :n], 0.97)
        print(f"RESULT: PCC {msg} -> {'PASS' if ok else 'FAIL'} (gate 0.97)")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
