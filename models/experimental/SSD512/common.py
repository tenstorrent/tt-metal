# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import gc
from models.experimental.SSD512.reference.ssd import build_ssd
from models.experimental.SSD512.reference.layers.functions.prior_box import PriorBox
from models.experimental.SSD512.reference.configs.config import voc


def setup_seeds_and_deterministic(reset_seeds=True, seed=0):
    """
    Setup random seeds and deterministic algorithms for reproducibility.
    """
    if reset_seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_and_init_torch_model(phase="test", size=512, num_classes=21):
    """
    Build PyTorch SSD model and initialize with random weights.
    """
    torch_model = build_ssd(phase, size=size, num_classes=num_classes)
    torch_model.eval()

    # Initialize with random weights (xavier uniform)
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    return torch_model


def build_and_load_ttnn_model(torch_model, device, num_classes=21, weight_device=None):
    """
    Build TTNN SSD512 model and load weights from PyTorch model.
    """
    from models.experimental.SSD512.tt.tt_ssd import build_ssd512

    ttnn_model = build_ssd512(num_classes=num_classes, device=device)
    ttnn_model.load_weights_from_torch(torch_model, weight_device=weight_device)

    # Synchronize device after weight loading
    if device is not None:
        ttnn.synchronize_device(device)
        gc.collect()
        ttnn.synchronize_device(device)

    return ttnn_model


def generate_prior_boxes(cfg=None, device=None):
    """
    Generate prior boxes for SSD512.
    """
    if cfg is None:
        cfg = voc["SSD512"]

    prior_box = PriorBox(cfg)
    priors = prior_box.forward()

    # Ensure it's a torch tensor
    if not isinstance(priors, torch.Tensor):
        priors = torch.tensor(priors)

    return priors


def synchronize_device(device):
    """
    Synchronize device and perform garbage collection.
    """
    if device is not None:
        ttnn.synchronize_device(device)
        gc.collect()
        ttnn.synchronize_device(device)


def cleanup_device_memory(device):
    """
    Cleanup device memory by synchronizing and attempting to deallocate buffers.
    """
    if device is not None:
        ttnn.synchronize_device(device)
        gc.collect()

        # Try to deallocate all buffers to free L1 memory
        try:
            if hasattr(ttnn, "deallocate_buffers"):
                ttnn.deallocate_buffers(device)
        except Exception:
            pass

        # Additional synchronization to ensure all operations complete
        ttnn.synchronize_device(device)
        gc.collect()
        ttnn.synchronize_device(device)


def create_conv2d_weights_and_bias(
    in_channels, out_channels, kernel_size, device=None, dtype=ttnn.bfloat16, init_method="kaiming_normal"
):
    """
    Create and initialize conv2d weights and bias for TTNN layers.
    """
    # Initialize weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
    weight = torch.empty(weight_shape)

    if init_method == "kaiming_normal":
        torch.nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
    elif init_method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(weight)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    # Initialize bias: (out_channels,)
    bias = torch.zeros(out_channels)

    # Convert to TTNN format
    if device is not None:
        weight_ttnn = ttnn.from_torch(
            weight,
            device=device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        bias_reshaped = bias.reshape((1, 1, 1, -1))
        bias_ttnn = ttnn.from_torch(
            bias_reshaped,
            device=device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    else:
        weight_ttnn = weight
        bias_ttnn = bias

    return weight_ttnn, bias_ttnn
