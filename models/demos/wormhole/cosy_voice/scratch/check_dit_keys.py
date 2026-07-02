import torch

sd = torch.load(
    "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B/flow.pt",
    map_location="cpu",
    weights_only=True,
)
prefix = "decoder.estimator.transformer_blocks.0"
for k in sorted(sd.keys()):
    if k.startswith(prefix):
        print(f"  {k[len(prefix)+1:]}  shape={sd[k].shape}")
