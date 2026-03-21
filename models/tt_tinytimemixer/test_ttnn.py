#!/usr/bin/env python3

import torch
import ttnn
import pytest
#!/usr/bin/env python3

import torch
import ttnn
import pytest
from transformers import TinyTimeMixerForPrediction
from models.tt_tinytimemixer.modeling_tinytimemixer_ttnn import TinyTimeMixerForPredictionTTNN

def test_tinytimemixer_ttnn_from_hf():
    # Model name
    model_name = "ibm-granite/granite-timeseries-ttm-r1"

    # PyTorch model
    pytorch_model = TinyTimeMixerForPrediction.from_pretrained(model_name)
    pytorch_model.eval()
    config = pytorch_model.config
    state_dict = pytorch_model.state_dict()

    # Device
    device = ttnn.open_device(0)

    # TTNN model
    ttnn_model = TinyTimeMixerForPredictionTTNN(config, device, state_dict)

    # Input
    past_values = torch.randn(1, config.context_length, config.num_input_channels)

    # PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(past_values).prediction_outputs
        
    # Squeeze to match the ttnn_output shape
    pytorch_output = pytorch_output.squeeze(0)

    # TTNN output
    ttnn_output = ttnn_model(past_values)
    
    # Squeeze to match the pytorch_output shape
    ttnn_output = ttnn_output.squeeze(0)

    # Close device
    ttnn.close_device(device)

    # Compare
    all_close = torch.allclose(pytorch_output, ttnn_output, atol=1e-2, rtol=1e-2)
    assert all_close, f"Outputs do not match! Max diff: {torch.max(torch.abs(pytorch_output - ttnn_output))}"
