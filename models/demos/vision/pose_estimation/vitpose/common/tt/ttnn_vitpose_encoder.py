# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_layer import vitpose_layer


def vitpose_encoder(hidden_states, *, layer_parameters, num_heads=12, compute_kernel_config=None):
    for params in layer_parameters:
        hidden_states = vitpose_layer(hidden_states, parameters=params, num_heads=num_heads, compute_kernel_config=compute_kernel_config)
    return hidden_states
