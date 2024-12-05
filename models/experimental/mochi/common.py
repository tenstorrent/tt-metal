import os
from pathlib import Path


def get_mochi_dir():
    mochi_dir = os.environ.get("MOCHI_DIR")
    if not mochi_dir:
        raise ValueError("MOCHI_DIR environment variable must be set")
    return mochi_dir


def get_cache_path(device_name):
    mochi_dir = get_mochi_dir()
    cache_path = Path(mochi_dir) / device_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


import torch


def compute_metrics(reference_output, test_output):
    # Compute PCC
    pcc = torch.corrcoef(torch.stack([reference_output.flatten(), test_output.flatten()]))[0, 1].item()

    # Compute MSE
    mse = torch.nn.functional.mse_loss(test_output, reference_output).item()

    # Compute MAE
    mae = torch.nn.functional.l1_loss(test_output, reference_output).item()

    return pcc, mse, mae
