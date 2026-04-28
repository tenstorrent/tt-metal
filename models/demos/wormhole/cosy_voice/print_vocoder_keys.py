import sys

import torch

# Add ref dir to path to find cosyvoice modules
sys.path.append("/root/tt-metal/models/demos/wormhole/cosy_voice/ref/CosyVoice")

import os

model_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/ref/CosyVoice/pretrained_models/CosyVoice-300M"
checkpoint_path = os.path.join(model_dir, "llm.pt")  # wait, vocoder is in hift.pt!
hift_path = os.path.join(model_dir, "hift.pt")

if not os.path.exists(hift_path):
    print(f"File not found: {hift_path}")
    sys.exit(1)

state_dict = torch.load(hift_path, map_location="cpu")
for k in state_dict.keys():
    if "resblocks.0.convs1.0" in k:
        print(f"{k}: {state_dict[k].shape}")
