"""Profile script for tracy. Run with:
    python3 -m tracy -p -v -r --op-support-count 30000 models/demos/inworld_tts/tt/profile_decoder.py
"""

import os
import sys

import torch

# Add train_venv for vector_quantize_pytorch
for p in sorted(
    [
        os.path.join("models/demos/inworld_tts/train_venv/lib", d, "site-packages")
        for d in os.listdir("models/demos/inworld_tts/train_venv/lib")
        if d.startswith("python")
    ],
    reverse=True,
):
    if os.path.isdir(p):
        sys.path.append(p)

from vector_quantize_pytorch import ResidualFSQ

import ttnn
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder

# Load weights
ckpt = torch.load(
    "models/demos/inworld_tts/training/vectorized_data_full/.cache/models--HKUSTAudio--xcodec2/"
    "snapshots/06071873ab345f44488d235dae3cb10b5901fd90/ckpt/epoch=4-step=1400000.ckpt",
    map_location="cpu",
    weights_only=False,
)
sd = ckpt["state_dict"]
quantizer = ResidualFSQ(dim=2048, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1)
quantizer.load_state_dict(
    {k.replace("generator.quantizer.", ""): v for k, v in sd.items() if k.startswith("generator.quantizer.")},
    strict=False,
)
quantizer.eval()
decoder_sd = {}
for k, v in sd.items():
    if k.startswith("generator.backbone."):
        decoder_sd[k.replace("generator.", "")] = v
    elif k.startswith("generator.head."):
        decoder_sd[k.replace("generator.", "")] = v
    elif k.startswith("fc_post_a."):
        decoder_sd[k] = v

device = ttnn.open_device(device_id=0, l1_small_size=16384)

tt_dec = TtCodecDecoder(
    device=device,
    state_dict=decoder_sd,
    quantizer=quantizer,
    backbone_prefix="backbone.",
    head_prefix="head.",
)

torch.manual_seed(42)
vq_codes = torch.randint(0, 65536, (1, 100))

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = tt_dec(vq_codes)

# Profiled run
with torch.no_grad():
    _ = tt_dec(vq_codes)

ttnn.close_device(device)
