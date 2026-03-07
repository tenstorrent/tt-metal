#!/usr/bin/env python3

import torch
from .configuration_tinytimemixer import TinyTimeMixerConfig
from .modeling_tinytimemixer import TinyTimeMixerForPrediction

# Create config
config = TinyTimeMixerConfig(
    context_length=512,
    patch_length=16,
    num_input_channels=7,
    patch_stride=8,
    d_model=64,
    prediction_length=96,
    num_layers=6,
)

print(f"Num patches: {config.num_patches}")

# Create model
model = TinyTimeMixerForPrediction(config)

# Create dummy input
batch_size = 1
past_values = torch.randn(batch_size, config.context_length, config.num_input_channels)

# Run model
with torch.no_grad():
    output = model(past_values)

print(f"Input shape: {past_values.shape}")
print(f"Output shape: {output.shape}")
print("Model runs successfully")