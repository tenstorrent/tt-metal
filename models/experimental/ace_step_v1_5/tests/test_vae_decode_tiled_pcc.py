# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: production ``decode_tiled`` vs PyTorch ``AutoencoderOobleck.decode`` @ 25 Hz latent frames."""

from __future__ import annotations

import os

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.tests._prod_test_helpers import vae_hf_dir

_PCC = float(os.environ.get("ACE_STEP_VAE_TILED_PCC", "0.98"))


@pytest.mark.parametrize("latent_frames,label", [(375, "15s"), (750, "30s")])
def test_vae_decode_tiled_pcc_vs_torch(device, torch_seed, latent_frames: int, label: str):
    vae_dir = vae_hf_dir()
    if vae_dir is None:
        pytest.skip("HF VAE dir not found under ACE_STEP_CHECKPOINT_DIR/vae")

    pytest.importorskip("diffusers")
    from diffusers.models import AutoencoderOobleck

    import ttnn
    from models.experimental.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

    torch.manual_seed(int(torch_seed))
    vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(torch.bfloat16)
    c_lat = int(vae.config.decoder_input_channels)

    lat_btc = (torch.randn(1, int(latent_frames), c_lat, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    lat_bct = lat_btc.transpose(1, 2).contiguous()

    with torch.inference_mode():
        ref_bct = vae.decode(lat_bct).sample.float()
    ref_btc = ref_bct.transpose(1, 2).contiguous()

    tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(str(vae_dir), device=device, latent_frames=int(latent_frames))
    lat_tt = ttnn.from_torch(
        lat_btc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    chunk = 32
    overlap = 4
    if int(latent_frames) >= 750:
        overlap = max(overlap, 14)
    out_tt = tt_vae.decode_tiled(lat_tt, chunk_size=chunk, overlap=overlap, use_trace=False)
    got_btc = ttnn.to_torch(out_tt).float()
    if got_btc.ndim == 4:
        got_btc = got_btc.squeeze(1)

    t = min(int(ref_btc.shape[1]), int(got_btc.shape[1]))
    c = min(int(ref_btc.shape[2]), int(got_btc.shape[2]))
    ref_s = ref_btc[:, :t, :c]
    got_s = got_btc[:, :t, :c]

    print(
        f"\n[vae_tiled_pcc][{label}] latent_frames={latent_frames} c_lat={c_lat} "
        f"ref={tuple(ref_btc.shape)} got={tuple(got_btc.shape)} chunk={chunk} overlap={overlap}",
        flush=True,
    )
    score = assert_pcc_print(f"vae_decode_tiled_{label}", ref_s, got_s, pcc=_PCC)
    print(
        f"[ace_step_v1_5][PCC] vae_decode_tiled_{label}_summary: pcc={score:.6f}",
        flush=True,
    )
