# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import logging
from datetime import datetime
from pathlib import Path
import shutil

# -------------------------------
# Setup logging and output folder
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTRUMENT_DIR = "module_io_data"
try:
    os.makedirs(INSTRUMENT_DIR, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create instrumentation directory: {e}")
    INSTRUMENT_DIR = "."


def config_to_dict(config):
    """Convert a config object to a serializable dictionary."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif hasattr(config, "__dict__"):
        return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    return str(config)


def save_tensor_data(tensor, save_dir, name):
    """Save a tensor to a .pt file."""
    if isinstance(tensor, torch.Tensor):
        path = os.path.join(save_dir, f"{name}.pt")
        torch.save(tensor.detach().cpu(), path)
        return {"type": "tensor", "shape": list(tensor.shape), "dtype": str(tensor.dtype), "path": path}
    elif isinstance(tensor, (list, tuple)):
        return [save_tensor_data(t, save_dir, f"{name}_{i}") for i, t in enumerate(tensor)]
    return str(tensor)


def instrument(module_name):
    """
    Decorator for instrumenting a module's forward method.
    Records the module's settings, inputs, and outputs in a structured directory format:
    module_io_data/
        YYYY_MM_DD_HHMMSS_modulename/
            metadata.json      # Contains settings and non-tensor data
            inputs/           # Directory for input tensors
                input_0.pt    # Individual tensor files
                input_1.pt
            outputs/          # Directory for output tensors
                output_0.pt
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create a new directory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(INSTRUMENT_DIR, f"{module_name}_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            # Create subdirectories for inputs and outputs
            inputs_dir = os.path.join(run_dir, "inputs")
            outputs_dir = os.path.join(run_dir, "outputs")
            os.makedirs(inputs_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)

            # Record settings
            settings = {}
            if hasattr(self, "config"):
                try:
                    settings = config_to_dict(self.config)
                except Exception as e:
                    settings = str(e)
            else:
                for attr in dir(self):
                    if not attr.startswith("_") and not callable(getattr(self, attr)):
                        try:
                            val = getattr(self, attr)
                            if isinstance(val, torch.Tensor):
                                settings[attr] = save_tensor_data(val, run_dir, f"setting_{attr}")
                            else:
                                settings[attr] = str(val)
                        except Exception as e:
                            settings[attr] = str(e)

            # Save inputs
            inputs = {
                "args": [save_tensor_data(a, inputs_dir, f"arg_{i}") for i, a in enumerate(args)],
                "kwargs": {k: save_tensor_data(v, inputs_dir, f"kwarg_{k}") for k, v in kwargs.items()},
            }

            # Call the original forward
            output = func(self, *args, **kwargs)

            # Save outputs
            if isinstance(output, torch.Tensor):
                output_data = save_tensor_data(output, outputs_dir, "output")
            elif isinstance(output, (list, tuple)):
                output_data = [save_tensor_data(o, outputs_dir, f"output_{i}") for i, o in enumerate(output)]
            else:
                output_data = str(output)

            # Save metadata
            metadata = {
                "module": module_name,
                "class": self.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "settings": settings,
                "inputs": inputs,
                "outputs": output_data,
            }

            with open(os.path.join(run_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return output

        return wrapper

    return decorator
