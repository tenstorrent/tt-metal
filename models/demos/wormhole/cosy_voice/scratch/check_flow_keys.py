import torch

sd = torch.load(
    "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B/flow.pt",
    map_location="cpu",
    weights_only=True,
)
for k in sorted(sd.keys()):
    if "decoder.estimator.time_embed" in k:
        print(k)
