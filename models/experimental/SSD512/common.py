# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.SSD512.reference.ssd import build_ssd


SSD512_L1_SMALL_SIZE = 98304
SSD512_NUM_CLASSES = 21


def load_torch_model(phase="test", size=512, num_classes=21):
    """Load the SSD512 torch model."""
    torch_model = build_ssd(phase, size=size, num_classes=num_classes)
    torch_model.eval()
    return torch_model


def reshape_prediction_tensors(pred_tensors, memory_config):
    """Reshape and concatenate prediction tensors from multiple feature scales."""
    reshaped_tensors = []
    for pred in pred_tensors:
        if pred.is_sharded():
            pred = ttnn.sharded_to_interleaved(pred, memory_config)
        pred = ttnn.to_memory_config(pred, memory_config)
        if pred.layout != ttnn.ROW_MAJOR_LAYOUT:
            pred = ttnn.to_layout(pred, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        batch_size = pred.shape[0]
        total_elements = pred.shape[1] * pred.shape[2] * pred.shape[3]
        pred_reshaped = ttnn.experimental.view(pred, (batch_size, total_elements))
        pred_reshaped = ttnn.to_memory_config(pred_reshaped, memory_config)
        reshaped_tensors.append(pred_reshaped)

    if len(reshaped_tensors) > 1:
        return ttnn.concat(reshaped_tensors, dim=1, memory_config=memory_config)
    return reshaped_tensors[0]


def create_ssd512_input_tensors(batch=1, input_height=512, input_width=512, mesh_mapper=None):
    """Create SSD512 input tensors."""
    torch_input = torch.randn(batch, 3, input_height, input_width)
    ttnn_input = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper)
    return torch_input, ttnn_input
