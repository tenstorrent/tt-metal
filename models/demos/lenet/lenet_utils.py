# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
import torchvision.transforms as transforms
from models.experimental.lenet.reference.lenet import LeNet5
import ttnn


def get_test_data(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
        ]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
    )

    batch = []
    images = []
    outputs = []

    for i in range(batch_size):
        img, output = test_dataset[i]
        tensor = transform(img).unsqueeze(0)
        batch.append(tensor)
        images.append(img)
        outputs.append(output)

    batch = torch.cat(batch)
    return batch, images, outputs


def load_torch_lenet(weka_path, num_classes):
    model2 = LeNet5(num_classes).to("cpu")
    checkpoint = torch.load(weka_path, map_location=torch.device("cpu"))
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()
    return model2, checkpoint["model_state_dict"]


def custom_preprocessor(model, device):
    parameters = {}

    layers_to_process = ["layer1", "layer2", "fc", "fc1", "fc2"]

    for layer in layers_to_process:
        if layer.startswith("layer"):
            conv_layer = getattr(model, layer)[0]
            bn_layer = getattr(model, layer)[1]

            weight = conv_layer.weight
            bias = conv_layer.bias

            running_mean = bn_layer.running_mean
            running_var = bn_layer.running_var
            eps = 1e-05

            scale = bn_layer.weight
            shift = bn_layer.bias

            weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]

            if bias is not None:
                bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
            else:
                bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

            weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
            bias = ttnn.reshape(bias, (1, 1, 1, -1))

        else:  # Handling linear layers
            linear_layer = getattr(model, layer)
            weight = linear_layer.weight
            weight = torch.permute(weight, (1, 0))
            bias = linear_layer.bias
            weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        parameters[layer] = {"weight": weight, "bias": bias}

    return parameters


def custom_preprocessor_device(parameters, device):
    parameters.fc.weight = ttnn.to_device(parameters.fc.weight, device)
    parameters.fc.bias = ttnn.to_device(parameters.fc.bias, device)
    parameters.fc1.weight = ttnn.to_device(parameters.fc1.weight, device)
    parameters.fc1.bias = ttnn.to_device(parameters.fc1.bias, device)
    parameters.fc2.weight = ttnn.to_device(parameters.fc2.weight, device)
    parameters.fc2.bias = ttnn.to_device(parameters.fc2.bias, device)

    return parameters
