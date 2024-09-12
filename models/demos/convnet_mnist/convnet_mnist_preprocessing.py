# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def custom_preprocessor(parameters, device):
    parameters.conv1.bias = ttnn.to_device(parameters.conv1.bias, device)
    parameters.conv1.bias = ttnn.to_device(parameters.conv1.bias, device)

    parameters.fc1.weight = ttnn.to_device(parameters.fc1.weight, device)
    parameters.fc1.bias = ttnn.to_device(parameters.fc1.bias, device)
    parameters.fc2.weight = ttnn.to_device(parameters.fc2.weight, device)
    parameters.fc2.bias = ttnn.to_device(parameters.fc2.bias, device)

    return parameters
