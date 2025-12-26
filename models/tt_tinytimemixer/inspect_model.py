#!/usr/bin/env python3

import torch
from transformers import TinyTimeMixerForPrediction as HFTinyTimeMixerForPrediction

# Load the model
model = HFTinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r1")

print("Model loaded")
print(f"Config: {model.config}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# Print state dict keys
state_dict = model.state_dict()
for key in state_dict.keys():
    print(f"{key}: {state_dict[key].shape}")