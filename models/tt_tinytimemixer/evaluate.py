#!/usr/bin/env python3

import torch
import ttnn
from transformers import TinyTimeMixerForPrediction
from models.tt_tinytimemixer.modeling_tinytimemixer_ttnn import TinyTimeMixerForPredictionTTNN
# Assuming the dataset can be loaded from a local path
# from datasets import load_dataset 

def get_ett_data(split="test"):
    # This is a placeholder function.
    # In a real scenario, you would load the ETT dataset from a file or a library.
    # For example:
    # dataset = load_dataset("monash_tsf", "ett_h1")
    # data = dataset[split]
    
    # For now, return random data with the correct dimensions
    # ETT has 7 features (channels)
    num_samples = 10
    context_length = 512
    prediction_length = 96
    num_channels = 7
    
    past_values = torch.randn(num_samples, context_length, num_channels)
    future_values = torch.randn(num_samples, prediction_length, num_channels)
    
    return past_values, future_values


def evaluate():
    # Model name
    model_name = "ibm-granite/granite-timeseries-ttm-r1"

    # Load PyTorch model for comparison
    pytorch_model = TinyTimeMixerForPrediction.from_pretrained(model_name)
    pytorch_model.eval()
    config = pytorch_model.config

    # Setup TTNN device and model
    device = ttnn.open_device(0)
    ttnn_model = TinyTimeMixerForPredictionTTNN(config, device, pytorch_model.state_dict())

    # Load data
    past_values_full, future_values_full = get_ett_data("test")
    
    all_mse = []
    all_mae = []

    for i in range(past_values_full.shape[0]):
        past_values = past_values_full[i].unsqueeze(0)
        future_values = future_values_full[i]

        # TTNN output
        ttnn_output = ttnn_model(past_values).squeeze(0)

        # Calculate metrics
        mse = torch.mean((ttnn_output - future_values) ** 2).item()
        mae = torch.mean(torch.abs(ttnn_output - future_values)).item()
        
        all_mse.append(mse)
        all_mae.append(mae)

    # Close device
    ttnn.close_device(device)
    
    avg_mse = sum(all_mse) / len(all_mse)
    avg_mae = sum(all_mae) / len(all_mae)

    print(f"Evaluation on ETT (placeholder data):")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  MAE: {avg_mae:.4f}")
    
    # The bounty requires MSE/MAE to be within 5% of PyTorch reference.
    # A full implementation would also run the PyTorch model and compare the metrics.
    print("\nNote: This script uses placeholder data. For a real evaluation, replace get_ett_data() with a proper data loader.")


if __name__ == "__main__":
    evaluate()
